"""Runtime configuration for the auxiliary brain.

This module deliberately knows nothing about Hermes' config loader.  The plugin
adapter can pass it any mapping, while tests and standalone callers can use the
same small value object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

VALID_MODES = frozenset({"off", "explicit", "shadow", "assist"})


class ConfigError(ValueError):
    """Raised when auxiliary-brain configuration is invalid."""


@dataclass(frozen=True, slots=True)
class BrainConfig:
    """Validated settings used by the local runtime.

    ``explicit`` only runs when a user invokes a brain command. ``shadow``
    records local predictions without changing the main agent's behavior.
    ``assist`` may return compact local context to the main agent.  The plugin
    integration is responsible for enforcing those modes.
    """

    mode: str = "explicit"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    auto_discover: bool = True
    capture: bool = True
    timeout_seconds: float = 8.0
    discovery_timeout_seconds: float = 0.75
    max_input_chars: int = 8_000
    gateway_slash_enabled: bool = False

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in VALID_MODES:
            allowed = ", ".join(sorted(VALID_MODES))
            raise ConfigError(f"mode must be one of: {allowed}")
        object.__setattr__(self, "mode", mode)

        if self.base_url is not None:
            base_url = self.base_url.strip()
            object.__setattr__(self, "base_url", base_url or None)
        if self.model is not None:
            model = self.model.strip()
            object.__setattr__(self, "model", model or None)
        if self.api_key is not None:
            api_key = self.api_key.strip()
            object.__setattr__(self, "api_key", api_key or None)

        if not 0.05 <= self.timeout_seconds <= 300:
            raise ConfigError("timeout_seconds must be between 0.05 and 300")
        if not 0.05 <= self.discovery_timeout_seconds <= 30:
            raise ConfigError("discovery_timeout_seconds must be between 0.05 and 30")
        if not 256 <= self.max_input_chars <= 1_000_000:
            raise ConfigError("max_input_chars must be between 256 and 1000000")

    @property
    def is_active(self) -> bool:
        return self.mode != "off"

    def clip_input(self, text: str) -> str:
        """Bound data sent to a small model, keeping the most recent text."""

        if len(text) <= self.max_input_chars:
            return text
        omitted = len(text) - self.max_input_chars
        marker = f"[... {omitted} earlier characters omitted ...]\n"
        keep = max(0, self.max_input_chars - len(marker))
        return marker + text[-keep:]

    def as_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "mode": self.mode,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "auto_discover": self.auto_discover,
            "capture": self.capture,
            "timeout_seconds": self.timeout_seconds,
            "discovery_timeout_seconds": self.discovery_timeout_seconds,
            "max_input_chars": self.max_input_chars,
            "gateway_slash_enabled": self.gateway_slash_enabled,
        }
        if redact_secrets and value["api_key"]:
            value["api_key"] = "***"
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> BrainConfig:
        """Build a config from plugin YAML data while rejecting typos.

        Quietly accepting misspelled safety settings is impolite, even for a
        brain with only 230 million neurons.
        """

        if raw is None:
            return cls()
        known = {
            "mode",
            "base_url",
            "model",
            "api_key",
            "auto_discover",
            "capture",
            "timeout_seconds",
            "discovery_timeout_seconds",
            "max_input_chars",
            "gateway_slash_enabled",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")

        values = dict(raw)
        for key in ("auto_discover", "capture", "gateway_slash_enabled"):
            if key in values:
                values[key] = _coerce_bool(values[key], key)
        for key in ("timeout_seconds", "discovery_timeout_seconds"):
            if key in values:
                try:
                    values[key] = float(values[key])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"{key} must be a number") from exc
        if "max_input_chars" in values:
            try:
                values["max_input_chars"] = int(values["max_input_chars"])
            except (TypeError, ValueError) as exc:
                raise ConfigError("max_input_chars must be an integer") from exc
        return cls(**values)


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{name} must be true or false")
