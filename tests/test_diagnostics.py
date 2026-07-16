from __future__ import annotations

import json
from pathlib import Path

import pytest

from auxiliary_brain import diagnostics
from auxiliary_brain.config import BrainConfig
from auxiliary_brain.llama_server import LlamaServerStatus
from auxiliary_brain.local_api import EndpointProbe
from auxiliary_brain.runtime import BrainRuntimeError


class FakeStore:
    def stats(self) -> dict[str, int]:
        return {"events": 3, "predictions": 2, "corrections": 1}


class FakeRuntime:
    def __init__(
        self,
        root: Path,
        *,
        config: BrainConfig | Exception | None = None,
        endpoint: EndpointProbe | Exception | None = None,
    ) -> None:
        self.root = root
        self._config = config or BrainConfig(
            base_url="http://127.0.0.1:8080/v1",
            model="tiny-model",
            api_key="super-secret",
            auto_discover=False,
        )
        self._endpoint = endpoint or EndpointProbe(
            "http://127.0.0.1:8080/v1",
            reachable=True,
            models=("tiny-model", "echo-super-secret"),
            latency_ms=1.5,
        )

    def data_root(self) -> Path:
        return self.root / "auxiliary-brain"

    def config(self) -> BrainConfig:
        if isinstance(self._config, Exception):
            raise self._config
        return self._config

    def probe(self, *, refresh: bool = False) -> tuple[EndpointProbe, str]:
        if isinstance(self._endpoint, Exception):
            raise self._endpoint
        model = self._endpoint.choose_model(self.config().model, strict=True)
        if model is None:
            raise BrainRuntimeError("configured model is not exposed")
        return self._endpoint, model

    def store(self) -> FakeStore:
        return FakeStore()


def managed_status(tmp_path: Path, *, ready: bool = True) -> LlamaServerStatus:
    executable = tmp_path / "llama-server"
    executable.write_text("binary", encoding="utf-8")
    return LlamaServerStatus(
        running=True,
        ready=ready,
        identity_verified=True,
        pid=4242,
        host="127.0.0.1",
        port=8080,
        model="tiny-model",
        executable=str(executable),
        started_at="2026-07-16T12:00:00+00:00",
        log_path=tmp_path / "llama-server.log",
        state_path=tmp_path / "llama-server.json",
    )


def test_status_is_structured_secret_safe_and_identifies_managed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_llama_server_status",
        lambda **_kwargs: managed_status(tmp_path),
    )

    report = diagnostics.build_status_report(FakeRuntime(tmp_path), refresh=True)

    assert report["schema_version"] == 1
    assert report["plugin"]["version"]
    assert report["config"]["auth"] == {
        "configured": True,
        "source": "AUXILIARY_BRAIN_API_KEY",
    }
    assert "api_key" not in report["config"]
    assert "super-secret" not in json.dumps(report)
    assert report["endpoint"]["exact_model_match"] is True
    assert report["server"]["configured_endpoint_ownership"] == "managed"
    assert report["server"]["pid"] == 4242
    assert report["stats"] == {"events": 3, "predictions": 2, "corrections": 1}


def test_status_survives_invalid_config_and_still_reports_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_llama_server_status",
        lambda **_kwargs: managed_status(tmp_path),
    )
    runtime = FakeRuntime(tmp_path, config=BrainRuntimeError("broken config"))

    report = diagnostics.build_status_report(runtime)

    assert report["config"] == {
        "valid": False,
        "error": "broken config",
        "auth": {"configured": False, "source": None},
    }
    assert report["endpoint"]["reachable"] is False
    assert report["storage"]["stats"]["events"] == 3
    json.dumps(report)


def test_status_redacts_a_credential_echoed_by_the_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_llama_server_status",
        lambda **_kwargs: managed_status(tmp_path),
    )
    runtime = FakeRuntime(
        tmp_path,
        endpoint=BrainRuntimeError("endpoint echoed Bearer super-secret"),
    )

    report = diagnostics.build_status_report(runtime)
    encoded = json.dumps(report)

    assert "super-secret" not in encoded
    assert "Bearer [redacted]" in encoded


def test_doctor_allows_healthy_external_server_with_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = LlamaServerStatus(
        False,
        False,
        False,
        None,
        "127.0.0.1",
        8080,
        "tiny-model",
        None,
        None,
        tmp_path / "llama-server.log",
        tmp_path / "llama-server.json",
    )
    monkeypatch.setattr(diagnostics, "get_llama_server_status", lambda **_kwargs: stopped)
    runtime = FakeRuntime(
        tmp_path,
        config=BrainConfig(
            base_url="http://127.0.0.1:8080/v1",
            model="tiny-model",
            auto_discover=False,
        ),
        endpoint=EndpointProbe(
            "http://127.0.0.1:8080/v1",
            reachable=True,
            models=("tiny-model",),
            latency_ms=2,
        ),
    )

    report = diagnostics.build_doctor_report(runtime)

    assert report["ok"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["endpoint"]["status"] == "PASS"
    assert checks["managed_server"]["status"] == "WARN"
    assert checks["server_binary"]["status"] == "WARN"
    assert checks["storage"]["status"] == "PASS"


def test_doctor_model_mismatch_and_unwritable_storage_are_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_llama_server_status",
        lambda **_kwargs: managed_status(tmp_path),
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_writable",
        lambda _path: (False, "permission denied"),
    )
    runtime = FakeRuntime(
        tmp_path,
        endpoint=BrainRuntimeError("configured model is not exposed"),
    )

    report = diagnostics.build_doctor_report(runtime)

    assert report["ok"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["endpoint"]["status"] == "FAIL"
    assert checks["model_identity"]["status"] == "FAIL"
    assert checks["storage"] == {
        "name": "storage",
        "status": "FAIL",
        "message": "permission denied",
        "fix": f"Repair permissions for `{tmp_path / 'auxiliary-brain'}`.",
    }


def test_doctor_reports_managed_state_corruption_without_hiding_other_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_llama_server_status",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad state")),
    )

    report = diagnostics.build_doctor_report(FakeRuntime(tmp_path))

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["managed_server"]["status"] == "FAIL"
    assert "bad state" in checks["managed_server"]["message"]
    assert checks["storage"]["status"] == "PASS"
