"""Tiny OpenAI-compatible client and localhost endpoint discovery.

The plugin should work with LM Studio, llama.cpp, Ollama's OpenAI endpoint,
vLLM, and similar servers without importing an SDK.  Keeping this layer on the
standard library also makes the install pleasantly uneventful.
"""

from __future__ import annotations

import ipaddress
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .version import __version__

DEFAULT_ENDPOINTS: tuple[str, ...] = (
    "http://127.0.0.1:1234/v1",  # LM Studio
    "http://127.0.0.1:8080/v1",  # llama.cpp
    "http://127.0.0.1:11434/v1",  # Ollama
    "http://127.0.0.1:8000/v1",  # vLLM and many dev servers
)


class LocalAPIError(RuntimeError):
    """A local inference server could not satisfy a request."""


@dataclass(frozen=True, slots=True)
class EndpointProbe:
    base_url: str
    reachable: bool
    models: tuple[str, ...] = ()
    latency_ms: float | None = None
    error: str | None = None

    def choose_model(
        self,
        requested: str | None = None,
        *,
        strict: bool = False,
    ) -> str | None:
        """Choose a model, optionally requiring an exact configured identity."""

        if requested:
            if requested in self.models:
                return requested
            if strict:
                return None
            if not self.models:
                return requested
            requested_lower = requested.lower()
            partial = next(
                (model for model in self.models if requested_lower in model.lower()),
                None,
            )
            if partial:
                return partial
        for needle in ("lfm2.5-230m", "lfm2-230m", "lfm-230m"):
            match = next((model for model in self.models if needle in model.lower()), None)
            if match:
                return match
        return self.models[0] if self.models else requested


def normalize_base_url(value: str) -> str:
    """Normalize a loopback OpenAI-compatible base URL to include ``/v1``.

    Explicit brain commands promise not to leave the machine.  Accepting an
    arbitrary URL here would turn "local" into a decorative adjective, which
    is how tiny privacy disasters hatch.
    """

    value = value.strip()
    if not value:
        raise ValueError("base URL cannot be empty")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("base URL must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base URL cannot contain user information")
    hostname = (parts.hostname or "").rstrip(".").lower()
    if not _is_loopback_host(hostname):
        raise ValueError("base URL host must be localhost or a loopback IP address")
    if parts.query or parts.fragment:
        raise ValueError("base URL cannot contain a query string or fragment")
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def list_models(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 1.0,
) -> tuple[str, ...]:
    """Return model IDs exposed by an OpenAI-compatible ``/models`` route."""

    payload = _request_json(
        "GET",
        f"{normalize_base_url(base_url)}/models",
        api_key=api_key,
        timeout=timeout,
    )
    data = payload.get("data")
    if not isinstance(data, list):
        raise LocalAPIError("model response did not contain a data list")
    models: list[str] = []
    for item in data:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            model_id = item["id"].strip()
            if model_id and model_id not in models:
                models.append(model_id)
    return tuple(models)


def probe_endpoint(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 0.75,
) -> EndpointProbe:
    """Check one endpoint without raising, returning diagnostic details."""

    normalized = base_url.strip()
    started = time.perf_counter()
    try:
        normalized = normalize_base_url(base_url)
        models = list_models(normalized, api_key=api_key, timeout=timeout)
    except (LocalAPIError, ValueError) as exc:
        elapsed = (time.perf_counter() - started) * 1_000
        return EndpointProbe(
            base_url=normalized,
            reachable=False,
            latency_ms=round(elapsed, 1),
            error=str(exc),
        )
    elapsed = (time.perf_counter() - started) * 1_000
    return EndpointProbe(
        base_url=normalized,
        reachable=True,
        models=models,
        latency_ms=round(elapsed, 1),
    )


def discover_endpoint(
    candidates: Iterable[str] = DEFAULT_ENDPOINTS,
    *,
    api_key: str | None = None,
    timeout: float = 0.75,
) -> EndpointProbe | None:
    """Return the first reachable local endpoint, or ``None``."""

    for candidate in candidates:
        probe = probe_endpoint(candidate, api_key=api_key, timeout=timeout)
        if probe.reachable:
            return probe
    return None


class OpenAICompatibleClient:
    """Minimal synchronous client for a local chat-completions server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def models(self) -> tuple[str, ...]:
        return list_models(self.base_url, api_key=self.api_key, timeout=self.timeout)

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> str:
        """Run one non-streaming chat completion and return its text."""

        body: dict[str, Any] = dict(extra_body or {})
        body.update(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
        )
        if response_format is not None:
            body["response_format"] = dict(response_format)
        payload = _request_json(
            "POST",
            f"{self.base_url}/chat/completions",
            body=body,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalAPIError("completion response did not contain choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LocalAPIError("completion choice was not an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise LocalAPIError("completion choice did not contain a message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [
                item.get("text", "")
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            if chunks:
                return "".join(str(chunk) for chunk in chunks)
        raise LocalAPIError("completion message did not contain text")


def _request_json(
    method: str,
    url: str,
    *,
    body: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": f"hermes-auxiliary-brain/{__version__}",
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        try:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LocalAPIError(f"request body is not valid JSON: {exc}") from exc
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read(2_048).decode("utf-8", errors="replace").strip()
        detail = redact_secret(detail, api_key)
        suffix = f": {detail}" if detail else ""
        raise LocalAPIError(f"local server returned HTTP {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LocalAPIError(f"cannot reach local server: {reason}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LocalAPIError("local server returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LocalAPIError("local server returned a non-object JSON response")
    return value


def redact_secret(value: str, secret: str | None) -> str:
    """Remove an endpoint credential from server-controlled diagnostics."""

    return value.replace(secret, "[redacted]") if secret else value


def redact_tree(value: Any, secret: str | None) -> Any:
    """Remove an endpoint credential from nested JSON-compatible data."""

    if isinstance(value, str):
        return redact_secret(value, secret)
    if isinstance(value, dict):
        return {
            redact_secret(key, secret) if isinstance(key, str) else key: redact_tree(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_tree(item, secret) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_tree(item, secret) for item in value)
    return value


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def urlopen(request: Request, *, timeout: float):  # noqa: ANN201
    """Open without environment proxies or redirects.

    ``normalize_base_url`` validates the original destination. Disabling both
    mechanisms prevents a proxy or redirect from quietly changing it later.
    The small wrapper also remains easy to replace in deterministic tests.
    """

    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _is_loopback_host(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")
