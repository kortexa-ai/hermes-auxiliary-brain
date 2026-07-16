"""Optional live-Hermes integration tests.

Windows local run::

    $env:HERMES_AGENT_ROOT='C:\\src\\hermes-agent'
    C:\\src\\hermes-agent\\venv\\Scripts\\python.exe -m pytest tests/test_hermes_integration.py

The tests use only an ephemeral loopback HTTP server and temporary HERMES_HOME.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

HERMES_AGENT_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", ""))
HAS_HERMES_CHECKOUT = bool(
    os.environ.get("HERMES_AGENT_ROOT")
    and (HERMES_AGENT_ROOT / "hermes_cli" / "plugins.py").is_file()
)


def _load_real_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_yaml: str,
) -> tuple[Any, Path, Any]:
    """Install a fresh copy and load it through Hermes' real directory loader."""

    source_root = Path(__file__).resolve().parents[1]
    hermes_home = tmp_path / "home"
    plugin_dir = hermes_home / "plugins" / "auxiliary-brain"
    plugin_dir.parent.mkdir(parents=True)
    for name in (
        "plugin.yaml",
        "__init__.py",
        "auxiliary_brain",
        "SKILL.md",
        "after-install.md",
    ):
        source = source_root / name
        if source.is_dir():
            shutil.copytree(source, plugin_dir / name)
        elif source.is_file():
            plugin_dir.mkdir(exist_ok=True)
            shutil.copy2(source, plugin_dir / name)

    assert (plugin_dir / "plugin.yaml").is_file(), "plugin entry point is incomplete"
    assert (plugin_dir / "__init__.py").is_file(), "plugin entry point is incomplete"

    (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    empty_bundled = tmp_path / "empty-bundled"
    empty_bundled.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.syspath_prepend(str(HERMES_AGENT_ROOT))

    # PluginManager uses a stable module name. Remove a previous test's loaded
    # copy so this run gets a new global runtime bound to its temp profile.
    for module_name in list(sys.modules):
        if module_name == "hermes_plugins.auxiliary_brain" or module_name.startswith(
            "hermes_plugins.auxiliary_brain."
        ):
            monkeypatch.delitem(sys.modules, module_name)

    from hermes_cli import plugins as hermes_plugins

    monkeypatch.setattr(hermes_plugins, "get_bundled_plugins_dir", lambda: empty_bundled)
    manager = hermes_plugins.PluginManager()
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    manager.discover_and_load()
    return manager, hermes_home, hermes_plugins


def _start_local_model_server(
    *,
    models: list[str],
    completion: dict[str, Any],
) -> tuple[ThreadingHTTPServer, threading.Thread, list[dict[str, Any]]]:
    observed: list[dict[str, Any]] = []

    class LocalModelHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _record(self, body: dict[str, Any] | None = None) -> None:
            observed.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "peer": self.client_address[0],
                    "body": body,
                }
            )

        def _send(self, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._record()
            self._send({"object": "list", "data": [{"id": model} for model in models]})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self._record(body)
            self._send(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(completion),
                            }
                        }
                    ]
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalModelHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="aux-brain-e2e-server",
        daemon=True,
    )
    server_thread.start()
    return server, server_thread, observed


def _stop_local_model_server(server: ThreadingHTTPServer, server_thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=5)


def _run_loaded_cli(manager: Any, argv: list[str]) -> int:
    entry = manager._cli_commands["brain"]
    parser = argparse.ArgumentParser()
    entry["setup_fn"](parser)
    args = parser.parse_args(argv)
    return entry["handler_fn"](args)


@pytest.mark.skipif(
    not HAS_HERMES_CHECKOUT,
    reason="set HERMES_AGENT_ROOT to run against a Hermes Agent checkout",
)
def test_real_hermes_discovers_and_registers_the_plugin(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real directory loader without loading bundled plugins."""

    manager, _, _ = _load_real_plugin(
        tmp_path,
        monkeypatch,
        config_yaml="plugins:\n  enabled:\n    - auxiliary-brain\n",
    )

    loaded = manager._plugins["auxiliary-brain"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert manager._plugin_commands == {}, "the plugin must not expose unsafe slash commands"
    assert "brain" in manager._cli_commands
    assert set(manager._aux_tasks) == {"auxiliary_brain_reflex"}
    assert manager.has_hook("pre_llm_call")
    assert manager._plugin_tool_names == set(), "the sidecar must not grow core tools"
    parser = argparse.ArgumentParser()
    manager._cli_commands["brain"]["setup_fn"](parser)
    server_args = parser.parse_args(["server", "start"])
    assert server_args.brain_command == "server"
    assert server_args.server_command == "start"


@pytest.mark.skipif(
    not HAS_HERMES_CHECKOUT,
    reason="set HERMES_AGENT_ROOT to run against a Hermes Agent checkout",
)
def test_loaded_brain_command_uses_real_loopback_http_and_captures_sqlite(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cross the actual Hermes loader, CLI handler, HTTP client, and store."""

    completion = {
        "summary": "Captured by the tiny loopback brain",
        "category": "note",
        "entities": ["Hermes"],
        "action_items": ["Review the capture"],
        "fields": {"transport": "loopback"},
        "confidence": 0.91,
    }
    server, server_thread, observed = _start_local_model_server(
        models=["tiny-e2e"],
        completion=completion,
    )
    port = server.server_address[1]
    config_yaml = f"""plugins:
  enabled:
    - auxiliary-brain
  entries:
    auxiliary-brain:
      config:
        mode: explicit
        capture: true
        auto_discover: false
auxiliary:
  auxiliary_brain_reflex:
    provider: custom
    model: tiny-e2e
    base_url: http://127.0.0.1:{port}/v1
    timeout: 3
"""

    try:
        manager, hermes_home, _ = _load_real_plugin(
            tmp_path,
            monkeypatch,
            config_yaml=config_yaml,
        )
        exit_code = _run_loaded_cli(
            manager,
            [
                "run",
                "generic_extract",
                "Keep",
                "this",
                "entirely",
                "on",
                "loopback",
            ],
        )
        response = capsys.readouterr().out
    finally:
        _stop_local_model_server(server, server_thread)

    assert exit_code == 0
    assert "task=generic_extract model=tiny-e2e" in response
    assert "prediction_id=pred_" in response
    assert '"summary": "Captured by the tiny loopback brain"' in response
    assert [(item["method"], item["path"]) for item in observed] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]
    assert {item["peer"] for item in observed} == {"127.0.0.1"}

    request_body = observed[1]["body"]
    assert request_body["model"] == "tiny-e2e"
    assert request_body["stream"] is False
    assert request_body["response_format"]["type"] == "json_schema"
    assert "Keep this entirely on loopback" in request_body["messages"][-1]["content"]

    database = hermes_home / "auxiliary-brain" / "brain.db"
    assert database.is_file()
    with sqlite3.connect(database) as connection:
        event = connection.execute(
            "SELECT input_text, task_key, metadata_json FROM events"
        ).fetchone()
        prediction = connection.execute(
            "SELECT task_key, output_json, model, base_url FROM predictions"
        ).fetchone()

    assert event is not None
    assert event[0] == "Keep this entirely on loopback"
    assert event[1] == "generic_extract"
    assert json.loads(event[2])["source"] == "cli"
    assert prediction is not None
    assert prediction[0] == "generic_extract"
    assert json.loads(prediction[1]) == completion
    assert prediction[2] == "tiny-e2e"
    assert prediction[3] == f"http://127.0.0.1:{port}/v1"


@pytest.mark.skipif(
    not HAS_HERMES_CHECKOUT,
    reason="set HERMES_AGENT_ROOT to run against a Hermes Agent checkout",
)
def test_configured_model_mismatch_stops_before_completion_and_capture(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reachable server may not silently substitute a different model."""

    server, server_thread, observed = _start_local_model_server(
        models=["different-model"],
        completion={
            "summary": "This response must never be requested",
            "category": "failure",
            "entities": [],
            "action_items": [],
            "fields": {},
            "confidence": 1.0,
        },
    )
    port = server.server_address[1]
    config_yaml = f"""plugins:
  enabled:
    - auxiliary-brain
  entries:
    auxiliary-brain:
      config:
        mode: explicit
        capture: true
        auto_discover: false
auxiliary:
  auxiliary_brain_reflex:
    provider: custom
    model: tiny-e2e
    base_url: http://127.0.0.1:{port}/v1
    timeout: 3
"""

    try:
        manager, hermes_home, _ = _load_real_plugin(
            tmp_path,
            monkeypatch,
            config_yaml=config_yaml,
        )
        exit_code = _run_loaded_cli(
            manager,
            [
                "run",
                "generic_extract",
                "Do",
                "not",
                "send",
                "this",
                "to",
                "a",
                "substitute",
                "model",
            ],
        )
        response = capsys.readouterr().out
    finally:
        _stop_local_model_server(server, server_thread)

    assert exit_code == 1
    assert "configured model 'tiny-e2e' is not exposed" in response
    assert "available: different-model" in response
    assert [(item["method"], item["path"]) for item in observed] == [("GET", "/v1/models")]
    assert {item["peer"] for item in observed} == {"127.0.0.1"}
    assert not (hermes_home / "auxiliary-brain" / "brain.db").exists()
