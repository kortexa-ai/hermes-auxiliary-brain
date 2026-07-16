from __future__ import annotations

import pytest

from auxiliary_brain.config import BrainConfig, ConfigError


def test_defaults_are_safe_and_explicit() -> None:
    config = BrainConfig()

    assert config.mode == "explicit"
    assert config.is_active is True
    assert config.capture is True
    assert config.base_url is None
    assert config.model is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"auto_discover": "yes", "capture": "0"}, (True, False)),
        ({"auto_discover": False, "capture": True}, (False, True)),
    ],
)
def test_from_mapping_coerces_boolean_values(
    raw: dict[str, object], expected: tuple[bool, bool]
) -> None:
    config = BrainConfig.from_mapping(raw)

    assert (config.auto_discover, config.capture) == expected


def test_from_mapping_normalizes_strings_and_numbers() -> None:
    config = BrainConfig.from_mapping(
        {
            "mode": " SHADOW ",
            "base_url": " http://localhost:1234/v1 ",
            "model": " tiny-brain ",
            "api_key": " local-only ",
            "timeout_seconds": "2.5",
            "max_input_chars": "4096",
        }
    )

    assert config.mode == "shadow"
    assert config.base_url == "http://localhost:1234/v1"
    assert config.model == "tiny-brain"
    assert config.api_key == "local-only"
    assert config.timeout_seconds == 2.5
    assert config.max_input_chars == 4096


def test_config_rejects_unknown_settings() -> None:
    with pytest.raises(ConfigError, match="unknown setting.*mystery_knob"):
        BrainConfig.from_mapping({"mystery_knob": True})


@pytest.mark.parametrize(
    "raw",
    [
        {"mode": "telepathy"},
        {"capture": "perhaps"},
        {"timeout_seconds": 0},
        {"discovery_timeout_seconds": 31},
        {"max_input_chars": 255},
    ],
)
def test_invalid_config_is_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        BrainConfig.from_mapping(raw)


def test_clip_input_keeps_recent_text_with_bounded_marker() -> None:
    config = BrainConfig(max_input_chars=256)
    text = "old:" + ("x" * 400) + ":new"

    clipped = config.clip_input(text)

    assert len(clipped) == 256
    assert clipped.startswith("[... ")
    assert clipped.endswith(":new")
    assert "old:" not in clipped


def test_as_dict_redacts_key_by_default() -> None:
    config = BrainConfig(api_key="secret")

    assert config.as_dict()["api_key"] == "***"
    assert config.as_dict(redact_secrets=False)["api_key"] == "secret"
