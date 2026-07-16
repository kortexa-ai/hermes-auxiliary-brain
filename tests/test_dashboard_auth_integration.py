"""Optional E2E proof that Hermes, not the plugin, authenticates API routes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERMES_AGENT_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", ""))
HAS_HERMES_CHECKOUT = bool(
    os.environ.get("HERMES_AGENT_ROOT")
    and (HERMES_AGENT_ROOT / "hermes_cli" / "web_server.py").is_file()
)


@pytest.mark.skipif(
    not HAS_HERMES_CHECKOUT,
    reason="set HERMES_AGENT_ROOT to run against a Hermes Agent checkout",
)
def test_real_hermes_dashboard_requires_auth_for_plugin_status(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    hermes_home = tmp_path / "home"
    plugin_dir = hermes_home / "plugins" / "auxiliary-brain"
    plugin_dir.parent.mkdir(parents=True)
    for name in ("plugin.yaml", "__init__.py", "auxiliary_brain", "dashboard"):
        source = source_root / name
        if source.is_dir():
            shutil.copytree(source, plugin_dir / name)
        else:
            plugin_dir.mkdir(exist_ok=True)
            shutil.copy2(source, plugin_dir / name)
    (hermes_home / "config.yaml").write_text(
        """plugins:
  enabled:
    - auxiliary-brain
  entries:
    auxiliary-brain:
      config:
        auto_discover: false
auxiliary:
  auxiliary_brain_reflex:
    provider: custom
    model: ''
    base_url: ''
    timeout: 8
""",
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "empty-bundled"
    empty_bundled.mkdir()
    script = f"""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from hermes_cli import plugins as hp

hp.get_bundled_plugins_dir = lambda: Path({str(empty_bundled)!r})
manager = hp.PluginManager()
manager._scan_entry_points = lambda: []
manager.discover_and_load()
assert manager._plugins['auxiliary-brain'].error is None

from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

with TestClient(app) as client:
    unauth = client.get('/api/plugins/auxiliary-brain/status')
    auth = client.get(
        '/api/plugins/auxiliary-brain/status',
        headers={{_SESSION_HEADER_NAME: _SESSION_TOKEN}},
    )
print('AUTH_RESULT=' + json.dumps({{
    'unauth_status': unauth.status_code,
    'auth_status': auth.status_code,
    'plugin': auth.json().get('plugin'),
}}))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = os.pathsep.join([str(HERMES_AGENT_ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(
        value.removeprefix("AUTH_RESULT=")
        for value in result.stdout.splitlines()
        if value.startswith("AUTH_RESULT=")
    )
    observed = json.loads(line)
    assert observed["unauth_status"] == 401
    assert observed["auth_status"] == 200
    assert observed["plugin"]["id"] == "auxiliary-brain"
