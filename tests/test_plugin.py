from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from auxiliary_brain import plugin
from auxiliary_brain.config import BrainConfig
from auxiliary_brain.llama_server import LlamaServerError, LlamaServerStatus
from auxiliary_brain.local_api import EndpointProbe
from auxiliary_brain.runtime import BrainRuntimeError, RunResult


def result(
    task_key: str,
    output: dict[str, Any],
    *,
    prediction_id: str | None = None,
) -> RunResult:
    return RunResult(
        task_key=task_key,
        output=output,
        raw_output=json.dumps(output),
        model="tiny-model",
        base_url="http://127.0.0.1:1234/v1",
        latency_ms=2.5,
        prediction_id=prediction_id,
    )


class FakeRuntime:
    def __init__(self, mode: str, results: list[RunResult] | None = None) -> None:
        self.mode = mode
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []

    def config(self) -> BrainConfig:
        return BrainConfig(mode=self.mode)

    def run(self, task_key: str, text: str, **kwargs: Any) -> RunResult:
        self.calls.append({"task_key": task_key, "text": text, **kwargs})
        if not self.results:
            raise AssertionError("unexpected local inference")
        return self.results.pop(0)


def server_status(
    tmp_path: Path,
    *,
    running: bool = True,
    ready: bool = True,
    model: str = "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M",
) -> LlamaServerStatus:
    return LlamaServerStatus(
        running=running,
        ready=ready,
        identity_verified=running,
        pid=4242 if running else None,
        host="127.0.0.1",
        port=8080,
        model=model,
        executable=str(tmp_path / "llama-server") if running else None,
        started_at="2026-07-16T12:00:00+00:00" if running else None,
        log_path=tmp_path / "llama-server.log",
        state_path=tmp_path / "llama-server.json",
    )


def test_server_parser_exposes_managed_lifecycle() -> None:
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert parser.parse_args(["server", "install"]).server_command == "install"
    start = parser.parse_args(["server", "start"])
    assert start.server_command == "start"
    assert start.model == "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M"
    assert start.host == "127.0.0.1"
    assert start.port == 8080
    assert start.wait_seconds == 600.0
    assert parser.parse_args(["server", "status"]).server_command == "status"
    assert parser.parse_args(["server", "logs"]).server_command == "logs"
    assert parser.parse_args(["server", "stop"]).server_command == "stop"


def test_explicit_help_prints_catalog_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["help"])) == 0

    output = capsys.readouterr().out
    assert "server" in output
    assert "status" in output
    assert "examples:" in output
    assert "hermes brain server start" in output


def test_status_and_doctor_json_use_shared_reports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ReportRuntime:
        def status(self) -> dict[str, Any]:
            return {"schema_version": 1, "secret_safe": True}

        def doctor(self) -> dict[str, Any]:
            return {"schema_version": 1, "ok": False, "checks": []}

    monkeypatch.setattr(plugin, "RUNTIME", ReportRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "secret_safe": True,
    }
    assert plugin.brain_command(parser.parse_args(["doctor", "--json"])) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_doctor_human_output_prints_named_fixes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ReportRuntime:
        def doctor(self) -> dict[str, Any]:
            return {
                "ok": False,
                "checks": [
                    {
                        "name": "endpoint",
                        "status": "FAIL",
                        "message": "unavailable",
                        "fix": "Start it.",
                    }
                ],
            }

    monkeypatch.setattr(plugin, "RUNTIME", ReportRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["doctor"])) == 1
    output = capsys.readouterr().out
    assert "[FAIL] endpoint: unavailable" in output
    assert "Fix: Start it." in output


def test_server_start_waits_verifies_exact_model_then_configures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = server_status(tmp_path)
    observed: dict[str, Any] = {}

    class ManagedRuntime:
        def config(self) -> BrainConfig:
            return BrainConfig(
                mode="assist",
                capture=False,
                timeout_seconds=11,
                discovery_timeout_seconds=1.25,
                max_input_chars=4096,
            )

        def save_configuration(self, **kwargs: Any) -> BrainConfig:
            observed["saved"] = kwargs
            return BrainConfig(**kwargs)

    def fake_start(**kwargs: Any) -> LlamaServerStatus:
        observed["start"] = kwargs
        return status

    monkeypatch.setattr(plugin, "RUNTIME", ManagedRuntime())
    monkeypatch.setattr(plugin, "start_llama_server", fake_start)
    monkeypatch.setattr(
        plugin,
        "probe_endpoint",
        lambda *_args, **_kwargs: EndpointProbe(
            status.base_url,
            reachable=True,
            models=(status.model,),
        ),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    exit_code = plugin.brain_command(parser.parse_args(["server", "start"]))

    assert exit_code == 0
    assert observed["start"] == {
        "executable": None,
        "install_if_missing": True,
        "model": status.model,
        "host": "127.0.0.1",
        "port": 8080,
        "wait_ready_seconds": 600.0,
    }
    assert observed["saved"] == {
        "base_url": status.base_url,
        "model": status.model,
        "mode": "assist",
        "capture": False,
        "auto_discover": False,
        "timeout_seconds": 11,
        "discovery_timeout_seconds": 1.25,
        "max_input_chars": 4096,
    }
    assert "Managed auxiliary brain is ready" in capsys.readouterr().out


def test_server_start_model_mismatch_leaves_configuration_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = server_status(tmp_path)

    class UnchangedRuntime:
        def save_configuration(self, **_kwargs: Any) -> None:
            pytest.fail("configuration must not change before exact model verification")

    monkeypatch.setattr(plugin, "RUNTIME", UnchangedRuntime())
    monkeypatch.setattr(plugin, "start_llama_server", lambda **_kwargs: status)
    monkeypatch.setattr(
        plugin,
        "probe_endpoint",
        lambda *_args, **_kwargs: EndpointProbe(
            status.base_url,
            reachable=True,
            models=("some-other-model",),
        ),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    exit_code = plugin.brain_command(parser.parse_args(["server", "start"]))

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "model verification failed" in output
    assert str(status.log_path) in output


def test_server_start_keeps_stable_model_alias_for_promoted_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = server_status(tmp_path)
    base = tmp_path / "base.gguf"
    adapter = tmp_path / "adapter.gguf"
    deployment = {
        "base_model_path": str(base),
        "base_model_sha256": "1" * 64,
        "adapter_path": str(adapter),
        "adapter_sha256": "2" * 64,
    }
    observed: dict[str, Any] = {}

    class ManagedRuntime:
        def config(self) -> BrainConfig:
            return BrainConfig()

        def save_configuration(self, **kwargs: Any) -> BrainConfig:
            observed["saved"] = kwargs
            return BrainConfig(**kwargs)

    monkeypatch.setattr(plugin, "RUNTIME", ManagedRuntime())
    monkeypatch.setattr(plugin, "active_deployment_artifacts", lambda: deployment)
    monkeypatch.setattr(
        plugin,
        "install_llama_cpp",
        lambda: SimpleNamespace(path="pinned-llama-server"),
    )
    monkeypatch.setattr(plugin, "verify_loaded_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        plugin,
        "start_llama_server",
        lambda **kwargs: observed.setdefault("start", kwargs) and status,
    )
    monkeypatch.setattr(
        plugin,
        "probe_endpoint",
        lambda *_args, **_kwargs: EndpointProbe(
            status.base_url,
            reachable=True,
            models=(status.model,),
        ),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["server", "start"])) == 0
    assert observed["start"]["executable"] == "pinned-llama-server"
    assert observed["start"]["install_if_missing"] is False
    assert observed["start"]["model_path"] == str(base)
    assert observed["start"]["lora_adapter_path"] == str(adapter)
    assert observed["saved"]["model"] == status.model


def test_server_start_allows_custom_model_after_rollback_to_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_model = "local/custom-model"
    status = server_status(tmp_path, model=custom_model)
    deployment = {
        "base_model_path": str(tmp_path / "base.gguf"),
        "base_model_sha256": "1" * 64,
        "adapter_path": None,
        "adapter_sha256": None,
    }
    observed: dict[str, Any] = {}

    class ManagedRuntime:
        def config(self) -> BrainConfig:
            return BrainConfig()

        def save_configuration(self, **kwargs: Any) -> BrainConfig:
            observed["saved"] = kwargs
            return BrainConfig(**kwargs)

    monkeypatch.setattr(plugin, "RUNTIME", ManagedRuntime())
    monkeypatch.setattr(plugin, "active_deployment_artifacts", lambda: deployment)
    monkeypatch.setattr(
        plugin,
        "install_llama_cpp",
        lambda: pytest.fail("base-only rollback must not force the pinned runtime"),
    )
    monkeypatch.setattr(
        plugin,
        "start_llama_server",
        lambda **kwargs: observed.setdefault("start", kwargs) and status,
    )
    monkeypatch.setattr(
        plugin,
        "probe_endpoint",
        lambda *_args, **_kwargs: EndpointProbe(
            status.base_url,
            reachable=True,
            models=(custom_model,),
        ),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert (
        plugin.brain_command(parser.parse_args(["server", "start", "--model", custom_model])) == 0
    )
    assert observed["start"] == {
        "executable": None,
        "install_if_missing": True,
        "model": custom_model,
        "host": "127.0.0.1",
        "port": 8080,
        "wait_ready_seconds": 600.0,
    }
    assert observed["saved"]["model"] == custom_model


def test_server_manager_errors_are_safe_cli_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        plugin,
        "start_llama_server",
        lambda **_kwargs: (_ for _ in ()).throw(LlamaServerError("tiny engine unavailable")),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["server", "start"])) == 1
    assert "Auxiliary brain: tiny engine unavailable" in capsys.readouterr().out


def test_server_status_and_stop_use_managed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stopped = server_status(tmp_path, running=False, ready=False)
    monkeypatch.setattr(plugin, "get_llama_server_status", lambda: stopped)
    monkeypatch.setattr(plugin, "stop_llama_server", lambda **_kwargs: stopped)
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["server", "status"])) == 1
    assert "state      : stopped" in capsys.readouterr().out
    assert plugin.brain_command(parser.parse_args(["server", "stop"])) == 0
    assert "state      : stopped" in capsys.readouterr().out


def test_server_logs_prints_requested_bounded_tail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[int] = []

    def fake_logs(*, lines: int) -> str:
        observed.append(lines)
        return "line two\nline three"

    monkeypatch.setattr(plugin, "read_llama_server_logs", fake_logs)
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["server", "logs", "--lines", "2"])) == 0
    assert observed == [2]
    assert capsys.readouterr().out == "line two\nline three\n"


def test_cli_mode_changes_behavior_without_inference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: list[str] = []

    class ModeRuntime:
        def set_mode(self, mode: str) -> BrainConfig:
            observed.append(mode)
            return BrainConfig(mode=mode, capture=False)

    monkeypatch.setattr(plugin, "RUNTIME", ModeRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    args = parser.parse_args(["mode", "off"])

    exit_code = plugin.brain_command(args)

    assert exit_code == 0
    assert observed == ["off"]
    assert "Auxiliary brain mode: off (capture=off)" in capsys.readouterr().out


def test_setup_preserves_existing_mode_and_capture_when_flags_are_omitted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved: dict[str, Any] = {}

    class SetupRuntime:
        def config(self) -> BrainConfig:
            return BrainConfig(mode="assist", capture=False)

        def save_configuration(self, **kwargs: Any) -> BrainConfig:
            saved.update(kwargs)
            return BrainConfig(**kwargs)

    monkeypatch.setattr(plugin, "RUNTIME", SetupRuntime())
    monkeypatch.setattr(
        plugin,
        "probe_endpoint",
        lambda *_args, **_kwargs: EndpointProbe(
            "http://127.0.0.1:1234/v1",
            reachable=True,
            models=("tiny-model",),
        ),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    args = parser.parse_args(
        [
            "setup",
            "--base-url",
            "http://127.0.0.1:1234/v1",
            "--model",
            "tiny-model",
        ]
    )

    assert plugin.brain_command(args) == 0
    assert saved["mode"] == "assist"
    assert saved["capture"] is False
    assert saved["auto_discover"] is False
    output = capsys.readouterr().out
    assert "mode     : assist" in output
    assert "capture  : off" in output


def test_setup_requires_explicit_url_when_api_key_is_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(plugin.API_KEY_ENV, "local-secret")
    monkeypatch.setattr(
        plugin,
        "discover_endpoint",
        lambda *_args, **_kwargs: pytest.fail("authenticated setup must not scan ports"),
    )
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    exit_code = plugin.brain_command(parser.parse_args(["setup", "--auto"]))

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "pass --base-url" in output
    assert "sent only to the endpoint you selected" in output


def test_pre_llm_explicit_mode_never_runs_local_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRuntime("explicit")
    monkeypatch.setattr(plugin, "RUNTIME", fake)

    assert plugin.pre_llm_call(user_message="ordinary cloud turn") is None
    assert fake.calls == []


def test_pre_llm_shadow_records_route_but_injects_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRuntime(
        "shadow",
        [
            result(
                "route",
                {
                    "target": "local",
                    "task": "generic_extract",
                    "reason": "bounded extraction",
                    "confidence": 0.9,
                },
            )
        ],
    )
    monkeypatch.setattr(plugin, "RUNTIME", fake)

    hook_result = plugin.pre_llm_call(
        user_message="extract this",
        session_id="session-1",
        platform="telegram",
        sender_id="telegram-user-1",
        turn_id="turn-1",
    )

    assert hook_result is None
    assert fake.calls == [
        {
            "task_key": "route",
            "text": "extract this",
            "source": "pre_llm_call",
            "session_id": "session-1",
            "metadata": {
                "platform": "telegram",
                "sender_id": "telegram-user-1",
                "turn_id": "turn-1",
            },
        }
    ]


def test_pre_llm_assist_injects_compact_untrusted_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRuntime(
        "assist",
        [
            result(
                "route",
                {
                    "target": "local",
                    "task": "generic_extract",
                    "reason": "bounded extraction",
                    "confidence": 0.9,
                },
            ),
            result(
                "generic_extract",
                {"summary": "Local hint", "confidence": 0.7},
            ),
        ],
    )
    monkeypatch.setattr(plugin, "RUNTIME", fake)

    hook_result = plugin.pre_llm_call(
        user_message="extract this",
        session_id=42,
        platform="cli",
        turn_id="turn-2",
    )

    assert hook_result is not None
    context = hook_result["context"]
    assert context.startswith("<auxiliary_brain_context>")
    assert "Untrusted local extraction; verify it" in context
    assert 'task=generic_extract result={"summary":"Local hint"' in context
    assert context.endswith("</auxiliary_brain_context>")
    assert [call["task_key"] for call in fake.calls] == [
        "route",
        "generic_extract",
    ]
    assert fake.calls[1] == {
        "task_key": "generic_extract",
        "text": "extract this",
        "source": "pre_llm_call_assist",
        "session_id": "42",
        "metadata": {"platform": "cli", "sender_id": None, "turn_id": "turn-2"},
    }


@pytest.mark.parametrize("failure_site", ["config", "run"])
def test_pre_llm_hook_fails_open(failure_site: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenRuntime(FakeRuntime):
        def config(self) -> BrainConfig:
            if failure_site == "config":
                raise RuntimeError("tiny brain misplaced its helmet")
            return BrainConfig(mode="shadow")

        def run(self, task_key: str, text: str, **kwargs: Any) -> RunResult:
            raise BrainRuntimeError("local server asleep")

    monkeypatch.setattr(plugin, "RUNTIME", BrokenRuntime("shadow"))

    assert plugin.pre_llm_call(user_message="keep the cloud turn alive") is None


def test_cli_correction_reads_complete_json_from_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    correction = {
        "summary": "Reviewed",
        "category": "note",
        "entities": [],
        "action_items": [],
        "fields": {},
        "confidence": 1.0,
    }
    path = tmp_path / "correction.json"
    path.write_text(json.dumps(correction), encoding="utf-8")
    observed: dict[str, Any] = {}

    class CorrectionRuntime:
        def correct(
            self,
            prediction_id: str,
            corrected: dict[str, Any],
            *,
            note: str | None = None,
        ) -> SimpleNamespace:
            observed.update(
                prediction_id=prediction_id,
                corrected=corrected,
                note=note,
            )
            return SimpleNamespace(id=prediction_id, task_key="generic_extract")

    monkeypatch.setattr(plugin, "RUNTIME", CorrectionRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    args = parser.parse_args(["correct", "pred_42", "--file", str(path), "--note", "looks right"])

    exit_code = plugin.brain_command(args)

    assert exit_code == 0
    assert observed == {
        "prediction_id": "pred_42",
        "corrected": correction,
        "note": "looks right",
    }
    assert "Correction stored for pred_42 (generic_extract)." in capsys.readouterr().out


def test_cli_correction_rejects_non_object_json_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "correction.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    class CorrectionRuntime:
        def correct(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("invalid correction must fail before runtime mutation")

    monkeypatch.setattr(plugin, "RUNTIME", CorrectionRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    args = parser.parse_args(["correct", "pred_42", "--file", str(path)])

    assert plugin.brain_command(args) == 1
    assert "correction must be one complete JSON object" in capsys.readouterr().out


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cli_correction_rejects_non_finite_json_file(
    constant: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "correction.json"
    path.write_text(f'{{"confidence":{constant}}}', encoding="utf-8")

    class CorrectionRuntime:
        def correct(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("non-finite correction must fail before runtime mutation")

    monkeypatch.setattr(plugin, "RUNTIME", CorrectionRuntime())
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    args = parser.parse_args(["correct", "pred_42", "--file", str(path)])

    assert plugin.brain_command(args) == 1
    assert "non-finite JSON number is not allowed" in capsys.readouterr().out
