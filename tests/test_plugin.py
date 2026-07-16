from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import pytest

from auxiliary_brain import plugin
from auxiliary_brain.config import BrainConfig
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
        turn_id="turn-1",
    )

    assert hook_result is None
    assert fake.calls == [
        {
            "task_key": "route",
            "text": "extract this",
            "source": "pre_llm_call",
            "session_id": "session-1",
            "metadata": {"platform": "telegram", "turn_id": "turn-1"},
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
        "metadata": {"platform": "cli", "turn_id": "turn-2"},
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
