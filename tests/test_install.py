from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import install as installer


def make_source(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text("name: auxiliary-brain\n", encoding="utf-8")
    (root / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    (root / "LICENSE").write_text("Test license\n", encoding="utf-8")
    package = root / "auxiliary_brain"
    package.mkdir()
    (package / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "runtime.pyc").write_bytes(b"not really bytecode")
    dashboard = root / "dashboard"
    dashboard.mkdir()
    (dashboard / "manifest.json").write_text('{"name":"auxiliary-brain"}\n')
    (root / "tests").mkdir()
    (root / "tests" / "should_not_copy.py").write_text("nope = True\n")
    docs = root / "docs"
    docs.mkdir()
    (docs / "training.md").write_text("# Training\n", encoding="utf-8")
    return root


def test_default_hermes_home_prefers_environment(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(target))

    assert installer.default_hermes_home() == target


def test_runtime_sources_rejects_incomplete_checkout(tmp_path) -> None:
    with pytest.raises(installer.InstallError, match="plugin.yaml.*__init__.py"):
        installer.runtime_sources(tmp_path)


def test_copy_runtime_dry_run_does_not_mutate_destination(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "home" / "plugins" / "auxiliary-brain"

    installer.copy_runtime(source, destination, force=False, dry_run=True)

    assert not destination.exists()
    assert "copy plugin.yaml" in capsys.readouterr().out


def test_copy_runtime_installs_only_runtime_paths(tmp_path) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "home" / "plugins" / "auxiliary-brain"

    installer.copy_runtime(source, destination, force=False, dry_run=False)

    assert (destination / "plugin.yaml").is_file()
    assert (destination / "__init__.py").is_file()
    assert (destination / "LICENSE").read_text(encoding="utf-8") == "Test license\n"
    assert (destination / "auxiliary_brain" / "runtime.py").is_file()
    assert (destination / "dashboard" / "manifest.json").is_file()
    assert (destination / "docs" / "training.md").is_file()
    assert not (destination / "auxiliary_brain" / "__pycache__").exists()
    assert not (destination / "tests").exists()


def test_copy_runtime_requires_force_then_replaces_existing_plugin(tmp_path) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "home" / "plugins" / "auxiliary-brain"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(installer.InstallError, match="--force"):
        installer.copy_runtime(source, destination, force=False, dry_run=False)

    installer.copy_runtime(source, destination, force=True, dry_run=False)

    assert not (destination / "stale.txt").exists()
    assert (destination / "plugin.yaml").is_file()


def test_enable_plugin_uses_target_home_and_safe_override_flag(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(installer, "hermes_command", lambda: ["hermes-test"])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer.subprocess, "run", fake_run)

    assert installer.enable_plugin(tmp_path, dry_run=False) is True
    assert observed["command"] == [
        "hermes-test",
        "plugins",
        "enable",
        "auxiliary-brain",
        "--no-allow-tool-override",
    ]
    assert observed["env"]["HERMES_HOME"] == str(tmp_path)  # type: ignore[index]


def test_enable_plugin_without_hermes_is_recoverable(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(installer, "hermes_command", lambda: None)

    assert installer.enable_plugin(tmp_path, dry_run=False) is False
    assert "copied but not enabled" in capsys.readouterr().err
