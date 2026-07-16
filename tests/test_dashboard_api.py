from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - exercised in the stdlib-only test environment
    FastAPI = None  # type: ignore[assignment,misc]
    TestClient = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(FastAPI is None, reason="FastAPI is provided by Hermes")


def _api_module():
    from dashboard import plugin_api

    return plugin_api


def _client():
    api = _api_module()
    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/auxiliary-brain")
    return api, TestClient(app)


def test_dashboard_manifest_is_hidden_and_declares_api() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "dashboard" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "auxiliary-brain"
    assert manifest["tab"]["hidden"] is True
    assert manifest["api"] == "plugin_api.py"
    assert manifest["entry"] == "dist/index.js"
    assert (manifest_path.parent / manifest["entry"]).is_file()


def test_status_passes_refresh_and_returns_runtime_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, client = _client()
    refresh_values: list[bool] = []

    class FakeRuntime:
        def status(self, *, refresh: bool = False) -> dict[str, Any]:
            refresh_values.append(refresh)
            return {"ok": True, "refresh": refresh}

    monkeypatch.setattr(api, "RUNTIME", FakeRuntime())

    assert client.get("/api/plugins/auxiliary-brain/status").json() == {
        "ok": True,
        "refresh": False,
    }
    assert client.get("/api/plugins/auxiliary-brain/status?refresh=true").json() == {
        "ok": True,
        "refresh": True,
    }
    assert refresh_values == [False, True]


def test_status_maps_runtime_failure_to_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, client = _client()

    class BrokenRuntime:
        def status(self, *, refresh: bool = False) -> dict[str, Any]:
            raise api.BrainRuntimeError("tiny brain is napping")

    monkeypatch.setattr(api, "RUNTIME", BrokenRuntime())

    response = client.get("/api/plugins/auxiliary-brain/status")

    assert response.status_code == 503
    assert response.json() == {"detail": "tiny brain is napping"}


def test_checkin_runs_only_the_fixed_task_and_returns_sanitized_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, client = _client()
    calls: list[dict[str, Any]] = []

    class FakeRuntime:
        def run(self, task: str, text: str, **kwargs: Any) -> SimpleNamespace:
            calls.append({"task": task, "text": text, **kwargs})
            return SimpleNamespace(
                output={"next_action": "practice"},
                model="tiny-model",
                latency_ms=3.5,
                prediction_id="pred_42",
                raw_output="must not leak",
                base_url="must not leak",
                event_id="must not leak",
            )

    monkeypatch.setattr(api, "RUNTIME", FakeRuntime())

    response = client.post(
        "/api/plugins/auxiliary-brain/checkin",
        json={"text": "  Completed today's practice.  "},
    )

    assert response.status_code == 200
    assert calls == [
        {
            "task": "progress_checkin",
            "text": "Completed today's practice.",
            "source": "dashboard-api",
        }
    ]
    assert response.json() == {
        "output": {"next_action": "practice"},
        "model": "tiny-model",
        "latency_ms": 3.5,
        "prediction_id": "pred_42",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "   "},
        {"text": "x" * 8_001},
        {"text": "valid", "task": "generic_extract"},
        {"text": "valid", "model": "override"},
        {"text": "valid", "base_url": "http://127.0.0.1:9999/v1"},
        {},
    ],
)
def test_checkin_rejects_blank_oversized_missing_and_override_inputs(
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, client = _client()

    class ForbiddenRuntime:
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("invalid input must fail before local inference")

    monkeypatch.setattr(api, "RUNTIME", ForbiddenRuntime())

    response = client.post("/api/plugins/auxiliary-brain/checkin", json=payload)

    assert response.status_code == 422


def test_checkin_maps_runtime_failure_to_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, client = _client()

    class BrokenRuntime:
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            raise api.BrainRuntimeError("local model unavailable")

    monkeypatch.setattr(api, "RUNTIME", BrokenRuntime())

    response = client.post(
        "/api/plugins/auxiliary-brain/checkin",
        json={"text": "Completed a practice session."},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "local model unavailable"}
