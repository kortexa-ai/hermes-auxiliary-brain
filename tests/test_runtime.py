from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from auxiliary_brain import runtime as runtime_module
from auxiliary_brain.config import BrainConfig
from auxiliary_brain.local_api import EndpointProbe
from auxiliary_brain.runtime import BrainRuntime, BrainRuntimeError, RunResult
from auxiliary_brain.version import __version__

GENERIC_OUTPUT = {
    "summary": "A compact result",
    "category": "note",
    "entities": ["Hermes"],
    "action_items": ["Review it"],
    "fields": {"duration": "30 minutes"},
    "confidence": 0.8,
}


class FakeClient:
    def __init__(self, responses: Iterator[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return next(self._responses)


@pytest.fixture
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch) -> BrainRuntime:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    instance = BrainRuntime()
    config = BrainConfig(
        mode="explicit",
        base_url="http://127.0.0.1:1234/v1",
        model="tiny-model",
        capture=True,
    )
    monkeypatch.setattr(instance, "config", lambda: config)
    monkeypatch.setattr(
        instance,
        "probe",
        lambda **_kwargs: (
            EndpointProbe(
                "http://127.0.0.1:1234/v1",
                reachable=True,
                models=("tiny-model",),
                latency_ms=1.5,
            ),
            "tiny-model",
        ),
    )
    return instance


def install_fake_client(monkeypatch: pytest.MonkeyPatch, *responses: str) -> FakeClient:
    client = FakeClient(iter(responses))
    monkeypatch.setattr(
        runtime_module,
        "OpenAICompatibleClient",
        lambda *_args, **_kwargs: client,
    )
    return client


def install_host_config(monkeypatch: pytest.MonkeyPatch, root: dict[str, Any]) -> None:
    package = ModuleType("hermes_cli")
    package.__path__ = []  # type: ignore[attr-defined]
    config_module = ModuleType("hermes_cli.config")
    config_module.load_config = lambda: root  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def install_host_config_io(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Any,
    root: dict[str, Any],
) -> None:
    config_path.write_text(json.dumps(root), encoding="utf-8")
    package = ModuleType("hermes_cli")
    package.__path__ = []  # type: ignore[attr-defined]
    config_module = ModuleType("hermes_cli.config")
    config_module.get_config_path = lambda: config_path  # type: ignore[attr-defined]
    config_module.fast_safe_load = json.load  # type: ignore[attr-defined]

    def atomic_config_write(path: Any, value: dict[str, Any], **_kwargs: Any) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    config_module.atomic_config_write = atomic_config_write  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def test_run_returns_result_and_captures_reviewable_records(
    runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = install_fake_client(monkeypatch, json.dumps(GENERIC_OUTPUT))

    result = runtime.run(
        "generic_extract",
        "  Summarize this note  ",
        source="test",
        session_id="session-1",
        metadata={"turn_id": "turn-1"},
    )

    assert isinstance(result, RunResult)
    assert result.task_key == "generic_extract"
    assert result.output == GENERIC_OUTPUT
    assert result.model == "tiny-model"
    assert result.base_url == "http://127.0.0.1:1234/v1"
    assert result.event_id is not None
    assert result.prediction_id is not None
    assert result.repaired is False
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"]["type"] == "json_schema"

    event = runtime.store().get_event(result.event_id)
    prediction = runtime.store().get_prediction(result.prediction_id)
    assert event is not None
    assert event.input_text == "Summarize this note"
    assert event.metadata["source"] == "test"
    assert event.metadata["plugin_version"] == __version__
    assert event.metadata["repaired"] is False
    assert event.metadata["turn_id"] == "turn-1"
    assert len(event.metadata["task_contract_hash"]) == 64
    int(event.metadata["task_contract_hash"], 16)
    assert prediction is not None
    assert prediction.output == GENERIC_OUTPUT
    assert prediction.confidence == 0.8


def test_run_can_disable_capture(runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_client(monkeypatch, json.dumps(GENERIC_OUTPUT))

    result = runtime.run(
        "generic_extract",
        "Do not store this",
        source="evaluation",
        capture=False,
    )

    assert result.event_id is None
    assert result.prediction_id is None
    assert runtime.store().stats()["events"] == 0


def test_run_repairs_one_invalid_model_response(
    runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = install_fake_client(
        monkeypatch,
        "this is not JSON",
        json.dumps(GENERIC_OUTPUT),
    )

    result = runtime.run(
        "generic_extract",
        "Repair me",
        source="test",
        capture=False,
    )

    assert result.repaired is True
    assert result.output == GENERIC_OUTPUT
    assert len(client.calls) == 2
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert "previous output did not satisfy" in client.calls[1]["messages"][-1]["content"]


def test_run_redacts_endpoint_secret_from_return_and_persisted_prediction(
    runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "endpoint-secret-must-not-escape"
    config = BrainConfig(
        mode="explicit",
        base_url="http://127.0.0.1:1234/v1",
        api_key=secret,
        capture=True,
    )
    monkeypatch.setattr(runtime, "config", lambda: config)
    monkeypatch.setattr(
        runtime,
        "probe",
        lambda **_kwargs: (
            EndpointProbe(
                "http://127.0.0.1:1234/v1",
                reachable=True,
                models=(secret,),
            ),
            secret,
        ),
    )
    echoed = {
        **GENERIC_OUTPUT,
        "summary": f"malicious echo: {secret}",
        "fields": {secret: secret},
    }
    install_fake_client(monkeypatch, json.dumps(echoed))

    result = runtime.run("generic_extract", "Keep credentials private", source="test")
    prediction = runtime.store().get_prediction(result.prediction_id or "missing")

    assert secret not in json.dumps(result.output)
    assert secret not in result.raw_output
    assert secret not in result.model
    assert prediction is not None
    assert secret not in json.dumps(prediction.output)
    assert secret not in (prediction.raw_output or "")
    assert secret not in (prediction.model or "")
    assert "[redacted]" in json.dumps(result.output)


def test_correction_export_and_training_record(
    runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    install_fake_client(monkeypatch, json.dumps(GENERIC_OUTPUT))
    result = runtime.run("generic_extract", "Original", source="test")
    corrected = {
        **GENERIC_OUTPUT,
        "summary": "Human-corrected result",
        "confidence": 1.0,
    }

    prediction = runtime.correct(
        result.prediction_id or "missing",
        corrected,
        note="reviewed by a human",
    )
    destination, count = runtime.export(tmp_path / "training.jsonl")

    assert prediction.id == result.prediction_id
    assert count == 1
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["output"] == corrected
    assert exported["corrected"] is True
    assert exported["note"] == "reviewed by a human"


def test_evaluate_ignores_confidence_and_does_not_capture(
    runtime: BrainRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_client(monkeypatch, json.dumps(GENERIC_OUTPUT))
    result = runtime.run("generic_extract", "Example", source="test")
    expected = {**GENERIC_OUTPUT, "confidence": 1.0}
    runtime.correct(result.prediction_id or "missing", expected)
    calls: list[dict[str, Any]] = []

    def fake_run(task_key: str, text: str, **kwargs: Any) -> RunResult:
        calls.append({"task_key": task_key, "text": text, **kwargs})
        return RunResult(
            task_key=task_key,
            output={**expected, "confidence": 0.1},
            raw_output="{}",
            model="tiny-model",
            base_url="http://127.0.0.1:1234/v1",
            latency_ms=1.0,
        )

    monkeypatch.setattr(runtime, "run", fake_run)

    report = runtime.evaluate(limit=10)

    assert report == {
        "examples": 1,
        "completed": 1,
        "exact_matches": 1,
        "exact_accuracy": 1.0,
        "field_accuracy": 1.0,
        "failures": [],
    }
    assert calls == [
        {
            "task_key": "generic_extract",
            "text": "Example",
            "source": "evaluation",
            "capture": False,
        }
    ]


def test_non_loopback_configuration_fails_before_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    instance = BrainRuntime()
    monkeypatch.setattr(
        instance,
        "config",
        lambda: BrainConfig(
            mode="explicit",
            base_url="https://models.example.com/v1",
            model="tiny-model",
        ),
    )
    inference_constructed = False

    def forbidden_client(*_args: Any, **_kwargs: Any) -> None:
        nonlocal inference_constructed
        inference_constructed = True
        pytest.fail("inference client must not be constructed for a remote URL")

    monkeypatch.setattr(runtime_module, "OpenAICompatibleClient", forbidden_client)

    with pytest.raises(BrainRuntimeError, match="loopback IP address"):
        instance.run("generic_extract", "Private text", source="test")

    assert inference_constructed is False


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (
            {"provider": "auto"},
            "provider must be custom",
        ),
        (
            {"provider": "openrouter"},
            "provider must be custom",
        ),
        (
            {
                "provider": "custom",
                "api_key": "do-not-store-secrets-here",
            },
            "put the credential in AUXILIARY_BRAIN_API_KEY instead",
        ),
    ],
)
def test_config_rejects_unsafe_auxiliary_routing(
    route: dict[str, Any], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_host_config(
        monkeypatch,
        {"auxiliary": {"auxiliary_brain_reflex": route}},
    )

    with pytest.raises(BrainRuntimeError, match=message):
        BrainRuntime().config()


def test_set_mode_works_offline_and_preserves_capture_and_routing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    original = {
        "plugins": {
            "enabled": ["auxiliary-brain"],
            "entries": {"auxiliary-brain": {"config": {"mode": "assist", "capture": False}}},
        },
        "auxiliary": {
            "auxiliary_brain_reflex": {
                "provider": "custom",
                "base_url": "http://127.0.0.1:65535/v1",
                "model": "sleeping-model",
            }
        },
    }
    install_host_config_io(monkeypatch, config_path, original)
    monkeypatch.setattr(
        runtime_module,
        "probe_endpoint",
        lambda *_args, **_kwargs: pytest.fail("mode changes must not probe a server"),
    )
    monkeypatch.setattr(
        runtime_module,
        "discover_endpoint",
        lambda *_args, **_kwargs: pytest.fail("mode changes must not discover a server"),
    )

    updated = BrainRuntime().set_mode("off")
    persisted = json.loads(config_path.read_text(encoding="utf-8"))

    assert updated.mode == "off"
    assert updated.capture is False
    assert persisted["plugins"]["entries"]["auxiliary-brain"]["config"] == {
        "mode": "off",
        "capture": False,
    }
    assert persisted["auxiliary"] == original["auxiliary"]


def test_gateway_slash_setting_is_offline_and_preserves_existing_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    original = {
        "plugins": {
            "enabled": ["auxiliary-brain"],
            "entries": {
                "auxiliary-brain": {
                    "config": {"mode": "explicit", "capture": False},
                    "custom": "keep-me",
                }
            },
        },
        "auxiliary": {
            "auxiliary_brain_reflex": {
                "provider": "custom",
                "base_url": "http://127.0.0.1:65535/v1",
                "model": "sleeping-model",
            }
        },
    }
    install_host_config_io(monkeypatch, config_path, original)
    monkeypatch.setattr(
        runtime_module,
        "probe_endpoint",
        lambda *_args, **_kwargs: pytest.fail("gateway config changes must not probe a server"),
    )
    monkeypatch.setattr(
        runtime_module,
        "discover_endpoint",
        lambda *_args, **_kwargs: pytest.fail("gateway config changes must not discover a server"),
    )

    runtime = BrainRuntime()
    assert runtime.set_gateway_slash_enabled(True) is True
    enabled = json.loads(config_path.read_text(encoding="utf-8"))
    assert enabled["plugins"]["entries"]["auxiliary-brain"] == {
        "config": {
            "mode": "explicit",
            "capture": False,
            "gateway_slash_enabled": True,
        },
        "custom": "keep-me",
    }
    assert enabled["auxiliary"] == original["auxiliary"]

    configured = runtime.save_configuration(
        base_url="http://127.0.0.1:65535/v1",
        model="sleeping-model",
        mode="explicit",
        capture=False,
        auto_discover=False,
    )
    after_setup = json.loads(config_path.read_text(encoding="utf-8"))
    assert configured.gateway_slash_enabled is True
    assert (
        after_setup["plugins"]["entries"]["auxiliary-brain"]["config"]["gateway_slash_enabled"]
        is True
    )
    assert after_setup["plugins"]["entries"]["auxiliary-brain"]["custom"] == "keep-me"

    assert runtime.set_gateway_slash_enabled(False) is False
    disabled = json.loads(config_path.read_text(encoding="utf-8"))
    assert (
        disabled["plugins"]["entries"]["auxiliary-brain"]["config"]["gateway_slash_enabled"]
        is False
    )


def test_probe_does_not_auto_discover_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    install_host_config(
        monkeypatch,
        {"plugins": {"entries": {"auxiliary-brain": {"config": {"auto_discover": True}}}}},
    )
    monkeypatch.setenv("AUXILIARY_BRAIN_API_KEY", "local-secret")
    monkeypatch.setattr(
        runtime_module,
        "discover_endpoint",
        lambda *_args, **_kwargs: pytest.fail("authenticated runtime must not scan ports"),
    )

    with pytest.raises(BrainRuntimeError, match="no base URL is configured"):
        BrainRuntime().probe()
