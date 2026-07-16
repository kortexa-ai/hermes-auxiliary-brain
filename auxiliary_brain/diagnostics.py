"""Shared, secret-safe status and doctor reports for CLI and HTTP surfaces."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .llama_server import (
    LLAMA_CPP_RELEASE,
    LlamaServerError,
    get_llama_server_status,
)
from .llama_server import (
    resolve_data_root as resolve_llama_data_root,
)
from .local_api import normalize_base_url, redact_secret, redact_tree
from .tasks import list_tasks
from .version import __version__

if TYPE_CHECKING:
    from .runtime import BrainRuntime

REPORT_SCHEMA_VERSION = 1


def build_status_report(runtime: BrainRuntime, *, refresh: bool = False) -> dict[str, Any]:
    """Build one JSON-serializable report without leaking credentials."""

    from .runtime import API_KEY_ENV, PLUGIN_ID, BrainRuntimeError, resolve_api_key

    data_root = runtime.data_root()
    profile_home = data_root.parent
    config: dict[str, Any]
    cfg = None
    try:
        cfg = runtime.config()
        config = {
            "valid": True,
            "mode": cfg.mode,
            "capture": cfg.capture,
            "auto_discover": cfg.auto_discover,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "timeout_seconds": cfg.timeout_seconds,
            "discovery_timeout_seconds": cfg.discovery_timeout_seconds,
            "max_input_chars": cfg.max_input_chars,
            "gateway_slash_enabled": cfg.gateway_slash_enabled,
            "auth": {
                "configured": bool(cfg.api_key),
                "source": API_KEY_ENV if cfg.api_key else None,
            },
        }
    except BrainRuntimeError as exc:
        try:
            secret = resolve_api_key()
        except BrainRuntimeError:
            secret = None
        config = {
            "valid": False,
            "error": redact_secret(str(exc), secret),
            "auth": {
                "configured": bool(secret),
                "source": API_KEY_ENV if secret else None,
            },
        }

    endpoint: dict[str, Any]
    if cfg is None:
        endpoint = {
            "reachable": False,
            "exact_model_match": False,
            "error": "endpoint not checked because configuration is invalid",
        }
    else:
        try:
            probe, model = runtime.probe(refresh=refresh)
            endpoint = {
                "reachable": True,
                "base_url": probe.base_url,
                "model": model,
                "models": list(probe.models),
                "latency_ms": probe.latency_ms,
                "exact_model_match": not cfg.model or model == cfg.model,
            }
        except BrainRuntimeError as exc:
            endpoint = {
                "reachable": False,
                "exact_model_match": False,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "error": redact_secret(str(exc), cfg.api_key),
            }

    llama_root = resolve_llama_data_root(profile_home)
    try:
        managed_status = get_llama_server_status(hermes_home=profile_home)
        managed = managed_status.as_dict()
        managed["state_error"] = None
    except (LlamaServerError, OSError, ValueError) as exc:
        managed = {
            "running": False,
            "ready": False,
            "identity_verified": False,
            "state_error": str(exc),
            "state_path": str(llama_root / "llama-server.json"),
            "log_path": str(llama_root / "llama-server.log"),
        }
    managed.update(
        {
            "build": LLAMA_CPP_RELEASE,
            "root": str(llama_root),
            "cache_path": str(llama_root / "cache"),
        }
    )

    configured_url = config.get("base_url")
    managed_url = managed.get("base_url")
    has_managed_state = any(
        managed.get(field) for field in ("pid", "command", "executable", "started_at")
    )
    if (
        has_managed_state
        and configured_url
        and managed_url
        and _same_endpoint(configured_url, managed_url)
    ):
        ownership = "managed"
    elif configured_url:
        ownership = "external"
    elif endpoint.get("reachable"):
        ownership = "auto-discovered"
    else:
        ownership = "none"
    managed["configured_endpoint_ownership"] = ownership

    stats: dict[str, Any]
    storage_error = None
    try:
        stats = runtime.store().stats()
    except Exception as exc:  # report every subsystem even if SQLite is unhealthy
        stats = {}
        storage_error = str(exc)
    storage = {
        "data_root": str(data_root),
        "database": str(data_root / "brain.db"),
        "writable": _appears_writable(data_root),
        "stats": stats,
        "error": storage_error,
    }

    mode = config.get("mode")
    capture = config.get("capture")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plugin": {"id": PLUGIN_ID, "version": __version__},
        "profile": {"name": _profile_name(), "home": str(profile_home)},
        "config": config,
        "endpoint": endpoint,
        "server": managed,
        "storage": storage,
        "tasks": [task.key for task in list_tasks()],
        # Backward-compatible keys for callers of BrainRuntime.status().
        "mode": mode,
        "capture": capture,
        "data_root": str(data_root),
        "stats": stats,
    }
    if cfg is not None:
        secret = cfg.api_key
    else:
        try:
            secret = resolve_api_key()
        except BrainRuntimeError:
            secret = None
    return redact_tree(report, secret)


def build_doctor_report(runtime: BrainRuntime) -> dict[str, Any]:
    """Run deeper named checks and return fixes without raising early."""

    report = build_status_report(runtime, refresh=True)
    config = report["config"]
    endpoint = report["endpoint"]
    server = report["server"]
    storage = report["storage"]
    checks: list[dict[str, Any]] = []

    if config.get("valid"):
        _add_check(checks, "config", "PASS", "Hermes configuration parsed successfully")
    else:
        _add_check(
            checks,
            "config",
            "FAIL",
            str(config.get("error") or "configuration is invalid"),
            "Run `hermes brain setup --auto` or `hermes brain server start`.",
        )

    base_url = config.get("base_url")
    if base_url:
        try:
            normalized = normalize_base_url(str(base_url))
            _add_check(checks, "loopback_policy", "PASS", f"Endpoint is local: {normalized}")
        except ValueError as exc:
            _add_check(
                checks,
                "loopback_policy",
                "FAIL",
                str(exc),
                "Configure a localhost or loopback endpoint only.",
            )
    elif config.get("auto_discover"):
        _add_check(
            checks,
            "loopback_policy",
            "WARN",
            "Endpoint uses loopback-only auto-discovery",
            "Run `hermes brain setup --base-url <loopback-url>` for a fixed endpoint.",
        )
    else:
        _add_check(
            checks,
            "loopback_policy",
            "FAIL",
            "No endpoint is configured",
            "Run `hermes brain server start` or `hermes brain setup --auto`.",
        )

    if endpoint.get("reachable"):
        _add_check(
            checks,
            "endpoint",
            "PASS",
            f"{endpoint.get('base_url')} answered in {endpoint.get('latency_ms')}ms",
        )
    else:
        _add_check(
            checks,
            "endpoint",
            "FAIL",
            str(endpoint.get("error") or "local endpoint is unavailable"),
            "Start the server; for managed llama.cpp run `hermes brain server logs --lines 100`.",
        )

    if endpoint.get("exact_model_match"):
        _add_check(
            checks,
            "model_identity",
            "PASS",
            f"Configured model is exposed exactly: {endpoint.get('model')}",
        )
    else:
        _add_check(
            checks,
            "model_identity",
            "FAIL",
            "The configured model was not verified at the endpoint",
            "Load the configured model or rerun setup with its exact model id.",
        )

    ownership = server.get("configured_endpoint_ownership")
    if server.get("state_error"):
        _add_check(
            checks,
            "managed_server",
            "FAIL",
            str(server["state_error"]),
            f"Inspect or remove stale state only after verifying `{server.get('state_path')}`.",
        )
    elif ownership == "managed":
        if server.get("running") and server.get("ready") and server.get("identity_verified"):
            _add_check(
                checks,
                "managed_server",
                "PASS",
                f"PID {server.get('pid')} matches the managed binary and is ready at "
                f"{server.get('base_url')}",
            )
        else:
            _add_check(
                checks,
                "managed_server",
                "FAIL",
                str(server.get("error") or "managed server is not ready"),
                "Run `hermes brain server start`, then "
                "`hermes brain server logs --lines 100` if it is not ready.",
            )
    else:
        _add_check(
            checks,
            "managed_server",
            "WARN",
            f"Configured endpoint ownership: {ownership}",
            "No fix is needed for a healthy external server.",
        )

    executable = server.get("executable")
    if executable and Path(str(executable)).is_file():
        _add_check(checks, "server_binary", "PASS", str(executable))
    elif ownership == "managed":
        _add_check(
            checks,
            "server_binary",
            "FAIL",
            "Managed llama.cpp executable is missing",
            "Run `hermes brain server install --force`.",
        )
    else:
        _add_check(
            checks,
            "server_binary",
            "WARN",
            "No managed llama.cpp executable is active",
            "This is expected when using an external server.",
        )

    server_root = Path(str(server.get("root")))
    path_message = (
        f"cache={server.get('cache_path')}; log={server.get('log_path')}; "
        f"state={server.get('state_path')}"
    )
    _add_check(
        checks,
        "server_paths",
        "PASS" if server_root.exists() else "WARN",
        path_message,
        None if server_root.exists() else "Paths are created on the first managed server start.",
    )

    writable, write_error = _verify_writable(Path(storage["data_root"]))
    storage["writable"] = writable
    if writable:
        _add_check(checks, "storage", "PASS", f"Writable: {storage['data_root']}")
    else:
        _add_check(
            checks,
            "storage",
            "FAIL",
            write_error or "storage is not writable",
            f"Repair permissions for `{storage['data_root']}`.",
        )

    auth = config.get("auth") or {}
    if auth.get("configured"):
        _add_check(
            checks,
            "endpoint_auth",
            "PASS",
            f"Credential is present via {auth.get('source')} (value hidden)",
        )
    else:
        _add_check(checks, "endpoint_auth", "PASS", "Endpoint is configured keyless")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ok": not any(check["status"] == "FAIL" for check in checks),
        "checks": checks,
        "status": report,
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    message: str,
    fix: str | None = None,
) -> None:
    checks.append({"name": name, "status": status, "message": message, "fix": fix})


def _profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return os.environ.get("HERMES_PROFILE") or "default"


def _same_endpoint(left: str, right: str) -> bool:
    try:
        return normalize_base_url(left) == normalize_base_url(right)
    except ValueError:
        return False


def _appears_writable(path: Path) -> bool:
    candidate = path if path.exists() else path.parent
    return candidate.exists() and os.access(candidate, os.W_OK)


def _verify_writable(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(prefix=".doctor-", dir=path)
        os.close(descriptor)
        Path(raw_path).unlink(missing_ok=True)
        return True, None
    except OSError as exc:
        return False, str(exc)
