from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from auxiliary_brain.runtime import PLUGIN_VERSION
from auxiliary_brain.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_version_is_semver_and_matches_manifest() -> None:
    manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s]+)\s*$", manifest, re.MULTILINE)

    assert match is not None
    assert match.group(1) == __version__ == PLUGIN_VERSION
    assert re.fullmatch(
        r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)",
        __version__,
    )
    dashboard_manifest = json.loads(
        (ROOT / "dashboard" / "manifest.json").read_text(encoding="utf-8")
    )
    assert dashboard_manifest["version"] == __version__


def test_python_package_reads_the_canonical_version_module() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "auxiliary_brain.version.__version__"
    }
