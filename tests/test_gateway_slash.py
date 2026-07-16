from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import threading
from typing import Any

import pytest

from auxiliary_brain import plugin
from auxiliary_brain.config import BrainConfig
from auxiliary_brain.runtime import BrainRuntimeError, RunResult


class RecordingContext:
    """Small public-API stand-in for Hermes' PluginContext."""

    def __init__(self) -> None:
        self.commands: dict[str, dict[str, Any]] = {}

    def register_auxiliary_task(self, **_kwargs: Any) -> None:
        pass

    def register_cli_command(self, **_kwargs: Any) -> None:
        pass

    def register_hook(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def register_command(
        self,
        name: str,
        handler: Any,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }


class GatewayConfigRuntime:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.changes: list[bool] = []

    def config(self) -> BrainConfig:
        return BrainConfig(gateway_slash_enabled=self.enabled)

    def gateway_slash_enabled(self) -> bool:
        return self.enabled

    def set_gateway_slash_enabled(self, enabled: bool) -> BrainConfig:
        self.changes.append(enabled)
        self.enabled = enabled
        return self.config()


class SlashRuntime(GatewayConfigRuntime):
    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.calls: list[dict[str, Any]] = []
        self.run_thread_ids: list[int] = []
        self.status_thread_ids: list[int] = []

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        assert refresh is True
        self.status_thread_ids.append(threading.get_ident())
        return {
            "config": {
                "valid": True,
                "effective": {
                    "mode": "explicit",
                    "model": "tiny-model",
                    "api_key": "status-secret",
                },
            },
            "endpoint": {
                "reachable": True,
                "model": "tiny-model",
                "latency_ms": 2.5,
                "base_url": "http://127.0.0.1:8080/v1",
            },
            "auth": {"present": True, "source": "env", "token": "status-secret"},
            "storage": {"data_root": "C:/private/brain"},
        }

    def run(self, task_key: str, text: str, **kwargs: Any) -> RunResult:
        self.run_thread_ids.append(threading.get_ident())
        self.calls.append({"task_key": task_key, "text": text, **kwargs})
        return RunResult(
            task_key=task_key,
            output={"summary": f"handled {task_key}"},
            raw_output=json.dumps({"summary": f"handled {task_key}"}),
            model="tiny-model",
            base_url="http://127.0.0.1:8080/v1",
            latency_ms=2.5,
            prediction_id="prediction-secret",
        )


def _registered_handler(
    monkeypatch: pytest.MonkeyPatch,
    runtime: GatewayConfigRuntime,
) -> Any:
    monkeypatch.setattr(plugin, "RUNTIME", runtime)
    context = RecordingContext()
    plugin.register(context)
    assert set(context.commands) == {"brain"}
    handler = context.commands["brain"]["handler"]
    assert inspect.iscoroutinefunction(handler)
    return handler


def test_gateway_slash_config_is_opt_in_and_round_trips() -> None:
    default = BrainConfig()
    enabled = BrainConfig.from_mapping({"gateway_slash_enabled": "yes"})

    assert default.gateway_slash_enabled is False
    assert enabled.gateway_slash_enabled is True
    assert enabled.as_dict()["gateway_slash_enabled"] is True


def test_gateway_cli_parser_exposes_status_enable_and_disable() -> None:
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert parser.parse_args(["gateway", "status"]).gateway_command == "status"
    enable = parser.parse_args(["gateway", "enable", "--acknowledge-busy-risk"])
    assert enable.gateway_command == "enable"
    assert enable.acknowledge_busy_risk is True
    assert parser.parse_args(["gateway", "disable"]).gateway_command == "disable"


def test_gateway_cli_requires_explicit_risk_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = GatewayConfigRuntime()
    monkeypatch.setattr(plugin, "RUNTIME", runtime)
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["gateway", "enable"])) == 1
    assert runtime.changes == []
    output = capsys.readouterr().out.lower()
    assert "acknowledge-busy-risk" in output
    assert "busy" in output


def test_gateway_cli_status_enable_and_disable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = GatewayConfigRuntime()
    monkeypatch.setattr(plugin, "RUNTIME", runtime)
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)

    assert plugin.brain_command(parser.parse_args(["gateway", "status"])) == 0
    assert "disabled" in capsys.readouterr().out.lower()

    args = parser.parse_args(["gateway", "enable", "--acknowledge-busy-risk"])
    assert plugin.brain_command(args) == 0
    assert runtime.changes == [True]
    assert "enabled" in capsys.readouterr().out.lower()

    assert plugin.brain_command(parser.parse_args(["gateway", "disable"])) == 0
    assert runtime.changes == [True, False]
    assert "disabled" in capsys.readouterr().out.lower()


def test_brain_slash_is_registered_once_and_uses_optional_argument_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin, "RUNTIME", GatewayConfigRuntime(False))
    context = RecordingContext()

    plugin.register(context)

    assert set(context.commands) == {"brain"}
    assert context.commands["brain"]["args_hint"].startswith("[")


def test_brain_slash_registration_supports_older_hermes_command_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OlderContext(RecordingContext):
        def register_command(
            self,
            name: str,
            handler: Any,
            description: str = "",
        ) -> None:
            self.commands[name] = {
                "handler": handler,
                "description": description,
            }

    monkeypatch.setattr(plugin, "RUNTIME", GatewayConfigRuntime(False))
    context = OlderContext()

    plugin.register(context)

    assert set(context.commands) == {"brain"}


@pytest.mark.parametrize(
    "failure",
    [BrainRuntimeError("private config path"), RuntimeError("private config path")],
)
def test_brain_slash_invocation_fails_closed_when_profile_gate_is_unavailable(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenConfigRuntime:
        def gateway_slash_enabled(self) -> bool:
            raise failure

    monkeypatch.setattr(plugin, "RUNTIME", BrokenConfigRuntime())
    context = RecordingContext()

    plugin.register(context)
    output = asyncio.run(context.commands["brain"]["handler"]("help"))

    assert "unavailable" in output.lower()
    assert "private config path" not in output


@pytest.mark.parametrize("command", ["help", "status", "checkin private update"])
def test_brain_slash_disabled_profile_is_inert(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime(enabled=False)
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler(command))

    assert "disabled for this profile" in output.lower()
    assert runtime.calls == []
    assert runtime.status_thread_ids == []


def test_brain_slash_rejects_multiplex_gateway_before_profile_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime(enabled=True)
    handler = _registered_handler(monkeypatch, runtime)
    monkeypatch.setattr(plugin, "_multiplex_gateway_active", lambda: True)

    output = asyncio.run(handler("checkin private update"))

    assert "multiplex-profile" in output
    assert runtime.calls == []
    assert runtime.status_thread_ids == []


def test_brain_slash_help_is_explicit_and_non_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler("help"))

    for command in ("status", "checkin", "followup", "note", "extract"):
        assert command in output
    for local_admin_command in ("setup", "server", "correct", "train"):
        assert f"/brain {local_admin_command}" not in output
    assert runtime.calls == []


def test_brain_slash_status_is_sanitized_and_runs_off_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    async def invoke() -> tuple[str, int]:
        event_loop_thread = threading.get_ident()
        return await handler("status"), event_loop_thread

    output, event_loop_thread = asyncio.run(invoke())

    assert "tiny-model" in output
    assert "reachable" in output.lower() or "ready" in output.lower()
    assert "status-secret" not in output
    assert "127.0.0.1" not in output
    assert "C:/private/brain" not in output
    assert runtime.status_thread_ids
    assert runtime.status_thread_ids[0] != event_loop_thread


def test_brain_slash_status_redacts_endpoint_credential_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "endpoint-status-secret"

    class EchoStatusRuntime(SlashRuntime):
        def status(self, *, refresh: bool = False) -> dict[str, Any]:
            report = super().status(refresh=refresh)
            report["endpoint"]["model"] = secret
            return report

    monkeypatch.setenv(plugin.API_KEY_ENV, secret)
    handler = _registered_handler(monkeypatch, EchoStatusRuntime())

    output = asyncio.run(handler("status"))

    assert secret not in output
    assert "[redacted]" in output


@pytest.mark.parametrize(
    "failure",
    [
        BrainRuntimeError("endpoint=http://127.0.0.1 token=status-secret"),
        RuntimeError("unexpected status-secret"),
    ],
)
def test_brain_slash_status_returns_generic_safe_errors(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenStatusRuntime(SlashRuntime):
        def status(self, *, refresh: bool = False) -> dict[str, Any]:
            raise failure

    handler = _registered_handler(monkeypatch, BrokenStatusRuntime())

    output = asyncio.run(handler("status"))

    assert "unavailable" in output.lower() or "failed" in output.lower()
    assert "status-secret" not in output
    assert "127.0.0.1" not in output


@pytest.mark.parametrize(
    ("command", "task_key"),
    [
        ("checkin", "progress_checkin"),
        ("followup", "follow_up"),
        ("note", "research_note"),
        ("extract", "generic_extract"),
    ],
)
def test_brain_slash_uses_only_fixed_task_mappings(
    command: str,
    task_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler(f"{command} remember this"))

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["task_key"] == task_key
    assert runtime.calls[0]["text"] == "remember this"
    assert f"handled {task_key}" in output
    assert "prediction-secret" not in output
    assert "127.0.0.1" not in output


def test_brain_slash_redacts_endpoint_credential_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "endpoint-result-secret"

    class EchoRuntime(SlashRuntime):
        def run(self, task_key: str, text: str, **kwargs: Any) -> RunResult:
            result = super().run(task_key, text, **kwargs)
            return RunResult(
                task_key=result.task_key,
                output={"summary": f"malicious echo: {secret}"},
                raw_output=result.raw_output,
                model=result.model,
                base_url=result.base_url,
                latency_ms=result.latency_ms,
            )

    monkeypatch.setenv(plugin.API_KEY_ENV, secret)
    handler = _registered_handler(monkeypatch, EchoRuntime())

    output = asyncio.run(handler("checkin hello"))

    assert secret not in output
    assert "[redacted]" in output


def test_brain_slash_rejects_unknown_task_without_local_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler("run arbitrary_task private text"))

    assert "unknown" in output.lower() or "help" in output.lower()
    assert runtime.calls == []


def test_brain_slash_does_not_reflect_large_unknown_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler("x" * 80_001))

    assert len(output) < 1_000
    assert "x" * 100 not in output
    assert "8,000" in output
    assert runtime.calls == []


def test_brain_slash_accepts_normal_whitespace_between_action_and_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    asyncio.run(handler("checkin\n\t remember this"))

    assert runtime.calls[0]["task_key"] == "progress_checkin"
    assert runtime.calls[0]["text"] == "remember this"


def test_brain_slash_rejects_input_over_8000_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    output = asyncio.run(handler("checkin " + ("x" * 8_001)))

    assert "8000" in output or "8,000" in output
    assert runtime.calls == []


def test_brain_slash_runs_inference_off_the_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SlashRuntime()
    handler = _registered_handler(monkeypatch, runtime)

    async def invoke() -> int:
        event_loop_thread = threading.get_ident()
        await handler("checkin " + ("x" * 8_000))
        return event_loop_thread

    event_loop_thread = asyncio.run(invoke())

    assert runtime.run_thread_ids
    assert runtime.run_thread_ids[0] != event_loop_thread
    assert len(runtime.calls[0]["text"]) == 8_000


@pytest.mark.parametrize(
    "failure",
    [
        BrainRuntimeError("endpoint=http://127.0.0.1 token=runtime-secret"),
        RuntimeError("unexpected runtime-secret"),
    ],
)
def test_brain_slash_returns_generic_safe_runtime_errors(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRuntime(SlashRuntime):
        def run(self, task_key: str, text: str, **kwargs: Any) -> RunResult:
            raise failure

    handler = _registered_handler(monkeypatch, BrokenRuntime())

    output = asyncio.run(handler("checkin hello"))

    assert "unavailable" in output.lower() or "failed" in output.lower()
    assert "runtime-secret" not in output
    assert "127.0.0.1" not in output
