from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from auxiliary_brain import local_api
from auxiliary_brain.local_api import (
    EndpointProbe,
    LocalAPIError,
    OpenAICompatibleClient,
    discover_endpoint,
    list_models,
    normalize_base_url,
    probe_endpoint,
)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
        ("http://localhost:8080/", "http://localhost:8080/v1"),
        ("https://127.42.0.1/api/v1/", "https://127.42.0.1/api/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ],
)
def test_normalize_base_url(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "localhost:1234",
        "ftp://localhost/model",
        "http://localhost/v1?q=x",
        "https://example.test/api/v1",
        "http://192.168.1.20:8000/v1",
        "http://user:password@localhost:8000/v1",
    ],
)
def test_normalize_base_url_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(value)


def test_urlopen_disables_proxies_and_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    class FakeOpener:
        def open(self, request: Any, *, timeout: float) -> FakeResponse:
            observed["request"] = request
            observed["timeout"] = timeout
            return FakeResponse({"ok": True})

    def fake_build_opener(*handlers: object) -> FakeOpener:
        observed["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(local_api, "build_opener", fake_build_opener)
    request = local_api.Request("http://127.0.0.1:1234/v1/models")

    with local_api.urlopen(request, timeout=0.5):
        pass

    handlers = observed["handlers"]
    assert isinstance(handlers[0], local_api.ProxyHandler)
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], local_api._NoRedirectHandler)
    assert observed["timeout"] == 0.5


def test_list_models_deduplicates_ids_and_sends_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["authorization"] = request.headers.get("Authorization")
        observed["timeout"] = timeout
        return FakeResponse({"data": [{"id": "tiny"}, {"id": "tiny"}, {"id": "other"}, {}]})

    monkeypatch.setattr(local_api, "urlopen", fake_urlopen)

    models = list_models("http://localhost:1234", api_key="secret", timeout=0.25)

    assert models == ("tiny", "other")
    assert observed == {
        "url": "http://localhost:1234/v1/models",
        "authorization": "Bearer secret",
        "timeout": 0.25,
    }


def test_http_error_redacts_echoed_endpoint_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "endpoint-secret-that-must-not-escape"

    def rejected(request: Any, *, timeout: float) -> FakeResponse:
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(f"rejected Bearer {secret}".encode()),
        )

    monkeypatch.setattr(local_api, "urlopen", rejected)

    with pytest.raises(LocalAPIError) as caught:
        list_models("http://localhost:1234", api_key=secret, timeout=0.25)

    assert secret not in str(caught.value)
    assert "Bearer [redacted]" in str(caught.value)


def test_probe_endpoint_converts_transport_error_to_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise URLError("no tiny brain today")

    monkeypatch.setattr(local_api, "urlopen", unavailable)

    result = probe_endpoint("http://localhost:1234")

    assert result.reachable is False
    assert result.base_url == "http://localhost:1234/v1"
    assert "no tiny brain today" in (result.error or "")
    assert result.latency_ms is not None


def test_discover_endpoint_stops_at_first_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_probe(candidate: str, **_kwargs: object) -> EndpointProbe:
        seen.append(candidate)
        return EndpointProbe(candidate, reachable=candidate.endswith("second"))

    monkeypatch.setattr(local_api, "probe_endpoint", fake_probe)

    result = discover_endpoint(["first", "second", "third"])

    assert result == EndpointProbe("second", reachable=True)
    assert seen == ["first", "second"]


def test_endpoint_model_choice_prefers_requested_then_lfm() -> None:
    endpoint = EndpointProbe(
        "http://localhost/v1",
        reachable=True,
        models=("generic", "LiquidAI/LFM2.5-230M"),
    )

    assert endpoint.choose_model("generic") == "generic"
    assert endpoint.choose_model("lfm2.5") == "LiquidAI/LFM2.5-230M"
    assert endpoint.choose_model() == "LiquidAI/LFM2.5-230M"
    assert endpoint.choose_model("missing-adapter", strict=True) is None


def test_completion_posts_openai_shape_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["body"] = json.loads(request.data.decode("utf-8"))
        observed["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr(local_api, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient("http://localhost:8080", timeout=3)

    result = client.complete(
        [{"role": "user", "content": "hello"}],
        model="tiny",
        max_tokens=64,
        response_format={"type": "json_object"},
        extra_body={"seed": 7},
    )

    assert result == '{"ok":true}'
    assert observed["url"] == "http://localhost:8080/v1/chat/completions"
    assert observed["timeout"] == 3
    assert observed["body"] == {
        "seed": 7,
        "model": "tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def test_completion_rejects_malformed_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local_api,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({"choices": []}),
    )

    with pytest.raises(LocalAPIError, match="did not contain choices"):
        OpenAICompatibleClient("http://localhost:8080").complete(
            [{"role": "user", "content": "hello"}], model="tiny"
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_completion_rejects_non_finite_request_before_http(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        local_api,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid JSON must not reach HTTP"),
    )

    with pytest.raises(LocalAPIError, match="request body is not valid JSON"):
        OpenAICompatibleClient("http://localhost:8080").complete(
            [{"role": "user", "content": "hello"}],
            model="tiny",
            extra_body={"non_finite": value},
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_models_response_rejects_non_finite_json_constants(
    constant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = FakeResponse({})
    response._raw = f'{{"data":{constant}}}'.encode()
    monkeypatch.setattr(
        local_api,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(LocalAPIError, match="invalid JSON"):
        list_models("http://localhost:8080")
