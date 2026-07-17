"""Install and manage a private llama.cpp sidecar for the auxiliary brain.

The process manager is deliberately independent of Hermes' plugin adapter.  It
uses only the Python standard library, binds the server to loopback, and keeps
its replaceable binaries separate from its small durable state file and log.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import ipaddress
import json
import os
import platform
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .version import __version__

DEFAULT_MODEL = "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

LLAMA_CPP_RELEASE = "b10046"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}"
STATE_FILENAME = "llama-server.json"
LOG_FILENAME = "llama-server.log"
LOCK_FILENAME = ".llama-server.lock"
INSTALL_DIRECTORY = "runtime"
STATE_FORMAT_VERSION = 1
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_LOG_TAIL_BYTES = 1024 * 1024
MAX_STATE_BYTES = 64 * 1024
MAX_INSTALL_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_INSTALL_FILES = 10_000


class LlamaServerError(RuntimeError):
    """Base class for safe, user-facing llama.cpp manager failures."""


class LlamaExecutableNotFound(LlamaServerError):
    """No supported llama.cpp server executable could be found."""


class LlamaServerStateError(LlamaServerError):
    """Durable server state is invalid, busy, or unsafe to act upon."""


class LlamaConfigurationError(LlamaServerError, ValueError):
    """A server option would be invalid or weaken the local-only boundary."""


class LlamaInstallError(LlamaServerError):
    """The pinned llama.cpp release could not be installed safely."""


class LlamaUnsupportedPlatform(LlamaInstallError):
    """No pinned llama.cpp binary is available for this platform."""


@dataclass(frozen=True, slots=True)
class LlamaReleaseAsset:
    """One pinned official llama.cpp release artifact."""

    name: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{RELEASE_URL}/{self.name}"


@dataclass(frozen=True, slots=True)
class LlamaExecutable:
    """A resolved llama.cpp executable and its command-line shape."""

    path: str
    style: str

    def __post_init__(self) -> None:
        if self.style not in {"llama", "llama-server"}:
            raise LlamaConfigurationError("style must be 'llama' or 'llama-server'")


@dataclass(frozen=True, slots=True)
class LlamaServerStatus:
    """A snapshot of the managed server process and endpoint."""

    running: bool
    ready: bool
    identity_verified: bool
    pid: int | None
    host: str
    port: int
    model: str
    executable: str | None
    started_at: str | None
    log_path: Path
    state_path: Path
    command: tuple[str, ...] = ()
    error: str | None = None
    lora_adapter_path: str | None = None
    lora_adapter_sha256: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/v1"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["log_path"] = str(self.log_path)
        value["state_path"] = str(self.state_path)
        value["command"] = list(self.command)
        value["base_url"] = self.base_url
        return value


# Digests and byte sizes are copied from the official GitHub release metadata.
# Pinning both makes an upstream asset replacement noisy instead of magical.
_RELEASE_ASSETS: dict[tuple[str, str], LlamaReleaseAsset] = {
    ("windows", "x64"): LlamaReleaseAsset(
        "llama-b10046-bin-win-cpu-x64.zip",
        "e5be2cf92f3232a2888b3d42983228565f35facdf66eb16076a95d7a756d06df",
        18_418_449,
    ),
    ("windows", "arm64"): LlamaReleaseAsset(
        "llama-b10046-bin-win-cpu-arm64.zip",
        "7f9958be4bdfc110c4eaa7bb49eeb115573d44d80a8b9400063bc49336923f1c",
        12_303_807,
    ),
    ("linux", "x64"): LlamaReleaseAsset(
        "llama-b10046-bin-ubuntu-x64.tar.gz",
        "f4b4a3cfc7b52e903417ca5fa6eb592cf42f8dcff266b62af382eb992fe5e7f6",
        16_022_392,
    ),
    ("linux", "arm64"): LlamaReleaseAsset(
        "llama-b10046-bin-ubuntu-arm64.tar.gz",
        "9362941732fdd04bfa32701eacc199306125f04e1755e5b732b9fe2225ebadaf",
        12_939_381,
    ),
    ("darwin", "x64"): LlamaReleaseAsset(
        "llama-b10046-bin-macos-x64.tar.gz",
        "4f69d6ac6ce327d34bc0e2bee49c3d13501ef83dba86fd81fde267a414303089",
        11_174_676,
    ),
    ("darwin", "arm64"): LlamaReleaseAsset(
        "llama-b10046-bin-macos-arm64.tar.gz",
        "d8ef7fa2179e79ef83d074e6f2eab947562e3e80301cac2fd39c561615c67a4f",
        10_895_913,
    ),
}

_PROTECTED_SERVER_OPTIONS = frozenset(
    {
        "-hf",
        "-hfr",
        "--hf-repo",
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "--host",
        "--port",
        "-p",
        "--alias",
        "-a",
        "--lora",
        "--lora-scaled",
        "--lora-init-without-apply",
    }
)
_PROTECTED_ENVIRONMENT = frozenset(
    {
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_HOST",
        "LLAMA_ARG_PORT",
        "LLAMA_ARG_ALIAS",
        "LLAMA_ARG_LORA",
        "LLAMA_ARG_LORA_SCALED",
        "LLAMA_ARG_LORA_INIT_WITHOUT_APPLY",
    }
)
_SERVER_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)


def resolve_data_root(hermes_home: str | Path | None = None) -> Path:
    """Return the active profile's durable auxiliary-brain directory."""

    if hermes_home is not None:
        home = Path(hermes_home)
    else:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:  # pragma: no cover - standalone fallback
            configured = os.environ.get("HERMES_HOME", "").strip()
            home = Path(configured) if configured else Path.home() / ".hermes"
    return home.expanduser().resolve() / "auxiliary-brain" / "llama.cpp"


def get_release_asset(
    *, system: str | None = None, machine: str | None = None
) -> LlamaReleaseAsset:
    """Select the pinned CPU build for the current OS and architecture."""

    system_key = (system or platform.system()).strip().lower()
    if system_key.startswith("win"):
        system_key = "windows"
    elif system_key in {"mac", "macos"}:
        system_key = "darwin"
    machine_key = (machine or platform.machine()).strip().lower()
    if machine_key in {"amd64", "x86_64", "x86-64"}:
        machine_key = "x64"
    elif machine_key in {"aarch64", "arm64"}:
        machine_key = "arm64"
    asset = _RELEASE_ASSETS.get((system_key, machine_key))
    if asset is None:
        raise LlamaUnsupportedPlatform(
            f"llama.cpp {LLAMA_CPP_RELEASE} has no pinned CPU build for "
            f"{system_key}/{machine_key}; install llama.cpp explicitly and ensure "
            "`llama` or `llama-server` is on PATH"
        )
    return asset


def llama_install_hint() -> str:
    """Explain the two deterministic ways to provide llama.cpp."""

    return (
        "install the pinned llama.cpp build with `hermes brain server install`, "
        "or install llama.cpp yourself and put `llama` or `llama-server` on PATH"
    )


def find_llama_executable(
    executable: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> LlamaExecutable:
    """Resolve an explicit, PATH, or profile-installed llama.cpp executable."""

    if executable is not None:
        candidate = _resolve_executable(executable)
        if candidate is None:
            raise LlamaExecutableNotFound(f"llama.cpp executable not found: {executable}")
        return _describe_executable(candidate)

    for name in ("llama", "llama-server"):
        candidate = shutil.which(name)
        if candidate:
            return _describe_executable(Path(candidate))

    install_root = resolve_data_root(hermes_home) / INSTALL_DIRECTORY / LLAMA_CPP_RELEASE
    asset = get_release_asset()
    candidate = _verified_installed_executable(install_root / _platform_key(), asset)
    if candidate is not None:
        return _describe_executable(candidate)

    raise LlamaExecutableNotFound(
        "no llama.cpp server executable was found; " + llama_install_hint()
    )


def find_profile_llama_executable(
    *,
    hermes_home: str | Path | None = None,
) -> LlamaExecutable:
    """Resolve only the checksum-recorded pinned runtime for this profile."""

    asset = get_release_asset()
    platform_root = (
        resolve_data_root(hermes_home) / INSTALL_DIRECTORY / LLAMA_CPP_RELEASE / _platform_key()
    )
    candidate = _verified_installed_executable(platform_root, asset)
    if candidate is None:
        raise LlamaExecutableNotFound(
            "the profile-pinned llama.cpp runtime is missing or changed; "
            "run `hermes brain server install --force`"
        )
    return _describe_executable(candidate)


def install_llama_cpp(
    *,
    hermes_home: str | Path | None = None,
    force: bool = False,
) -> LlamaExecutable:
    """Install a checksum-pinned official llama.cpp CPU release for this profile."""

    asset = get_release_asset()
    data_root = resolve_data_root(hermes_home)
    install_root = data_root / INSTALL_DIRECTORY / LLAMA_CPP_RELEASE
    platform_root = install_root / _platform_key()
    data_root.mkdir(parents=True, exist_ok=True)

    with _operation_lock(data_root):
        existing = _verified_installed_executable(platform_root, asset)
        if existing is not None and not force:
            return _describe_executable(existing)

        archive = data_root / f".{asset.name}.{uuid.uuid4().hex}.part"
        staging = install_root / f".{_platform_key()}.{uuid.uuid4().hex}.staging"
        backup = install_root / f".{_platform_key()}.{uuid.uuid4().hex}.backup"
        install_root.mkdir(parents=True, exist_ok=True)
        try:
            _download_release_asset(asset, archive)
            staging.mkdir(parents=True, exist_ok=False)
            _safe_extract_archive(archive, asset.name, staging)
            installed = _find_installed_executable(staging)
            if installed is None:
                raise LlamaInstallError(
                    f"official asset {asset.name} contained neither llama nor llama-server"
                )
            _make_executable(installed)
            relative_executable = installed.relative_to(staging).as_posix()
            _write_json_atomic(
                staging / "install.json",
                {
                    "release": LLAMA_CPP_RELEASE,
                    "asset": asset.name,
                    "sha256": asset.sha256,
                    "executable": relative_executable,
                    "executable_sha256": _sha256_file(installed),
                    "files": _artifact_tree_manifest(staging),
                    "installed_at": _utc_now(),
                },
            )

            if platform_root.exists():
                platform_root.replace(backup)
            staging.replace(platform_root)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if backup.exists() and not platform_root.exists():
                backup.replace(platform_root)
            raise
        finally:
            archive.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists():
                shutil.rmtree(backup)

    installed = _find_installed_executable(platform_root)
    if installed is None:  # pragma: no cover - guarded before atomic move
        raise LlamaInstallError("llama.cpp install completed without an executable")
    return _describe_executable(installed)


def build_server_command(
    executable: LlamaExecutable | str | Path,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra_args: Sequence[str] = (),
    model_path: str | Path | None = None,
    model_sha256: str | None = None,
    lora_adapter_path: str | Path | None = None,
    lora_adapter_sha256: str | None = None,
    lora_init_without_apply: bool = False,
) -> list[str]:
    """Build an argv-only, loopback-bound llama.cpp server command."""

    resolved = (
        executable
        if isinstance(executable, LlamaExecutable)
        else _describe_executable(_require_explicit_executable(executable))
    )
    normalized_host = _normalize_loopback_host(host)
    normalized_port = _validate_port(port)
    normalized_model = _validate_model(model)
    extras = _validate_extra_args(extra_args)
    local_model_path, _local_model_sha256 = _validate_gguf_artifact(
        model_path,
        model_sha256,
        path_option="model_path",
        sha256_option="model_sha256",
        label="local model",
    )
    adapter_path, _adapter_sha256 = _validate_lora_adapter(
        lora_adapter_path,
        lora_adapter_sha256,
        init_without_apply=lora_init_without_apply,
    )
    command = [resolved.path]
    if resolved.style == "llama":
        command.append("serve")
    if local_model_path is None:
        command.extend(["-hf", normalized_model])
    else:
        command.extend(["-m", local_model_path, "--alias", normalized_model])
    command.extend(["--host", normalized_host, "--port", str(normalized_port)])
    if adapter_path is not None:
        command.extend(["--lora", adapter_path])
        if lora_init_without_apply:
            command.append("--lora-init-without-apply")
    command.extend(extras)
    return command


def start_llama_server(
    *,
    hermes_home: str | Path | None = None,
    executable: str | Path | None = None,
    install_if_missing: bool = False,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra_args: Sequence[str] = (),
    model_path: str | Path | None = None,
    model_sha256: str | None = None,
    lora_adapter_path: str | Path | None = None,
    lora_adapter_sha256: str | None = None,
    lora_init_without_apply: bool = False,
    wait_ready_seconds: float = 0.0,
) -> LlamaServerStatus:
    """Start a detached managed server; ``-hf`` downloads the model as needed."""

    data_root = resolve_data_root(hermes_home)
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        resolved = find_llama_executable(executable, hermes_home=hermes_home)
    except LlamaExecutableNotFound:
        if not install_if_missing or executable is not None:
            raise
        resolved = install_llama_cpp(hermes_home=hermes_home)
    command = build_server_command(
        resolved,
        model=model,
        host=host,
        port=port,
        extra_args=extra_args,
        model_path=model_path,
        model_sha256=model_sha256,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_sha256=lora_adapter_sha256,
        lora_init_without_apply=lora_init_without_apply,
    )
    normalized_host = _normalize_loopback_host(host)
    normalized_port = _validate_port(port)
    normalized_model = _validate_model(model)
    normalized_model_path = None
    normalized_model_sha256 = None
    if model_path is not None:
        normalized_model_path = command[command.index("-m") + 1]
        assert model_sha256 is not None  # validated by build_server_command
        normalized_model_sha256 = model_sha256.strip().lower()
    normalized_adapter_path = None
    normalized_adapter_sha256 = None
    if lora_adapter_path is not None:
        normalized_adapter_path = command[command.index("--lora") + 1]
        assert lora_adapter_sha256 is not None  # validated by build_server_command
        normalized_adapter_sha256 = lora_adapter_sha256.strip().lower()
    wait_ready_seconds = _validate_timeout(wait_ready_seconds, "wait_ready_seconds", maximum=3600)
    state_path = data_root / STATE_FILENAME
    log_path = data_root / LOG_FILENAME

    with _operation_lock(data_root):
        existing_state = _read_state(state_path)
        if existing_state is not None:
            existing = _status_from_state(existing_state, state_path, log_path)
            if existing.running:
                raise LlamaServerStateError(
                    f"managed llama.cpp server is already running with PID {existing.pid}"
                )
            if existing.pid and _pid_alive(existing.pid):
                raise LlamaServerStateError(existing.error or "cannot verify managed server PID")
            state_path.unlink(missing_ok=True)
        if _port_is_open(normalized_host, normalized_port):
            raise LlamaServerStateError(
                f"loopback port {normalized_host}:{normalized_port} is already in use"
            )

        cache_root = data_root / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        environment = _server_environment(cache_root)
        popen_options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "cwd": str(Path(resolved.path).parent),
            "env": environment,
            "shell": False,
            "close_fds": True,
        }
        if os.name == "nt":
            popen_options["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            popen_options["start_new_session"] = True

        started_at = _utc_now()
        with log_path.open("ab", buffering=0) as log:
            log.write(
                (
                    f"\n[{started_at}] starting llama.cpp {LLAMA_CPP_RELEASE}: "
                    f"{json.dumps(command)}\n"
                ).encode()
            )
            popen_options["stdout"] = log
            try:
                process = subprocess.Popen(command, **popen_options)
            except OSError as exc:
                raise LlamaServerError(f"could not start llama.cpp: {exc}") from exc

        state = {
            "format_version": STATE_FORMAT_VERSION,
            "pid": process.pid,
            "host": normalized_host,
            "port": normalized_port,
            "model": normalized_model,
            "model_path": normalized_model_path,
            "model_sha256": normalized_model_sha256,
            "executable": str(Path(resolved.path).resolve()),
            "started_at": started_at,
            "log_path": str(log_path),
            "command": command,
            "lora_adapter_path": normalized_adapter_path,
            "lora_adapter_sha256": normalized_adapter_sha256,
        }
        try:
            _write_json_atomic(state_path, state)
        except Exception:
            _terminate_spawned_process(process)
            raise

        time.sleep(0.1)
        return_code = process.poll()
        if return_code is not None:
            state_path.unlink(missing_ok=True)
            tail = _read_log_tail(log_path)
            detail = f"; log tail:\n{tail}" if tail else ""
            raise LlamaServerError(
                f"llama.cpp exited during startup with code {return_code}{detail}"
            )

    if wait_ready_seconds:
        try:
            return wait_for_llama_server(
                hermes_home=hermes_home,
                timeout_seconds=wait_ready_seconds,
            )
        except BaseException:
            with contextlib.suppress(LlamaServerError):
                current = get_llama_server_status(hermes_home=hermes_home)
                if (
                    current.pid == process.pid
                    and current.executable is not None
                    and Path(current.executable).resolve() == Path(resolved.path).resolve()
                ):
                    _terminate_spawned_process(process)
                    state_path.unlink(missing_ok=True)
            raise
    return get_llama_server_status(hermes_home=hermes_home)


def get_llama_server_status(*, hermes_home: str | Path | None = None) -> LlamaServerStatus:
    """Read durable metadata and verify both PID identity and endpoint health."""

    data_root = resolve_data_root(hermes_home)
    state_path = data_root / STATE_FILENAME
    log_path = data_root / LOG_FILENAME
    state = _read_state(state_path)
    if state is None:
        return _empty_status(state_path, log_path)
    return _status_from_state(state, state_path, log_path)


def wait_for_llama_server(
    *,
    hermes_home: str | Path | None = None,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.25,
) -> LlamaServerStatus:
    """Wait until the managed server's loopback health endpoint reports ready."""

    timeout_seconds = _validate_timeout(timeout_seconds, "timeout_seconds", maximum=3600)
    poll_interval_seconds = _validate_timeout(
        poll_interval_seconds, "poll_interval_seconds", minimum=0.05, maximum=10
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = get_llama_server_status(hermes_home=hermes_home)
        if not status.running:
            raise LlamaServerError(status.error or "managed llama.cpp server is not running")
        if status.ready:
            return status
        if time.monotonic() >= deadline:
            raise LlamaServerError(
                f"llama.cpp is still loading after {timeout_seconds:g}s; see {status.log_path}"
            )
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def stop_llama_server(
    *,
    hermes_home: str | Path | None = None,
    timeout_seconds: float = 5.0,
) -> LlamaServerStatus:
    """Stop only the process whose live executable matches durable metadata."""

    timeout_seconds = _validate_timeout(timeout_seconds, "timeout_seconds", maximum=300)
    data_root = resolve_data_root(hermes_home)
    state_path = data_root / STATE_FILENAME
    log_path = data_root / LOG_FILENAME
    if not data_root.exists():
        return _empty_status(state_path, log_path)

    with _operation_lock(data_root):
        state = _read_state(state_path)
        if state is None:
            return _empty_status(state_path, log_path)
        status = _status_from_state(state, state_path, log_path)
        if not status.running:
            if status.pid and _pid_alive(status.pid):
                raise LlamaServerStateError(
                    status.error
                    or f"PID {status.pid} is alive but its identity could not be verified"
                )
            state_path.unlink(missing_ok=True)
            return _empty_status(state_path, log_path, error="removed stale server metadata")

        assert status.pid is not None  # validated state + running invariant
        if not _process_identity_matches(status.pid, status.executable or ""):
            raise LlamaServerStateError(
                f"PID {status.pid} no longer matches {status.executable}; refusing to signal it"
            )
        if os.name == "nt":
            _stop_windows_process(status.pid, status.executable or "", timeout_seconds)
        else:
            _stop_posix_process(status.pid, status.executable or "", timeout_seconds)
        state_path.unlink(missing_ok=True)
    return _empty_status(state_path, log_path)


def read_llama_server_logs(
    *,
    hermes_home: str | Path | None = None,
    lines: int = 100,
) -> str:
    """Return a bounded tail of the profile-local managed server log."""

    if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= 10_000:
        raise LlamaConfigurationError("lines must be an integer between 1 and 10000")
    path = resolve_data_root(hermes_home) / LOG_FILENAME
    if not path.is_file():
        raise LlamaServerError(
            f"managed server log does not exist yet: {path}; start the server first"
        )
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            offset = max(0, size - MAX_LOG_TAIL_BYTES)
            handle.seek(offset)
            payload = handle.read(MAX_LOG_TAIL_BYTES)
    except OSError as exc:
        raise LlamaServerError(f"cannot read managed server log {path}: {exc}") from exc
    text = payload.decode("utf-8", errors="replace")
    selected = text.splitlines()[-lines:]
    if offset:
        selected.insert(0, f"[... log tail limited to {MAX_LOG_TAIL_BYTES} bytes ...]")
    return "\n".join(selected)


def _status_from_state(
    state: dict[str, Any], state_path: Path, default_log_path: Path
) -> LlamaServerStatus:
    pid = _state_integer(state, "pid", minimum=1, maximum=2**31 - 1)
    host = _normalize_loopback_host(_state_string(state, "host"))
    port = _validate_port(_state_integer(state, "port", minimum=1, maximum=65535))
    model = _validate_model(_state_string(state, "model"))
    executable = _state_string(state, "executable")
    started_at = _state_string(state, "started_at")
    model_path, model_sha256 = _state_gguf_artifact(
        state,
        state_path,
        path_key="model_path",
        sha256_key="model_sha256",
        label="model",
    )
    adapter_path, adapter_sha256 = _state_lora_adapter(state, state_path)
    command_value = state.get("command")
    if not isinstance(command_value, list) or not all(
        isinstance(item, str) for item in command_value
    ):
        raise LlamaServerStateError(f"invalid command in {state_path}")
    log_value = state.get("log_path")
    log_path = Path(log_value) if isinstance(log_value, str) and log_value else default_log_path
    if not _pid_alive(pid):
        return LlamaServerStatus(
            False,
            False,
            False,
            pid,
            host,
            port,
            model,
            executable,
            started_at,
            log_path,
            state_path,
            tuple(command_value),
            f"managed llama.cpp PID {pid} is not running",
            lora_adapter_path=adapter_path,
            lora_adapter_sha256=adapter_sha256,
            model_path=model_path,
            model_sha256=model_sha256,
        )
    if not _process_identity_matches(pid, executable):
        return LlamaServerStatus(
            False,
            False,
            False,
            pid,
            host,
            port,
            model,
            executable,
            started_at,
            log_path,
            state_path,
            tuple(command_value),
            f"PID {pid} does not match stored executable {executable}",
            lora_adapter_path=adapter_path,
            lora_adapter_sha256=adapter_sha256,
            model_path=model_path,
            model_sha256=model_sha256,
        )
    ready = _endpoint_ready(host, port)
    return LlamaServerStatus(
        True,
        ready,
        True,
        pid,
        host,
        port,
        model,
        executable,
        started_at,
        log_path,
        state_path,
        tuple(command_value),
        lora_adapter_path=adapter_path,
        lora_adapter_sha256=adapter_sha256,
        model_path=model_path,
        model_sha256=model_sha256,
    )


def _empty_status(
    state_path: Path, log_path: Path, *, error: str | None = None
) -> LlamaServerStatus:
    return LlamaServerStatus(
        False,
        False,
        False,
        None,
        DEFAULT_HOST,
        DEFAULT_PORT,
        DEFAULT_MODEL,
        None,
        None,
        log_path,
        state_path,
        error=error,
    )


def _download_release_asset(asset: LlamaReleaseAsset, destination: Path) -> None:
    if asset.size > MAX_ARCHIVE_BYTES:
        raise LlamaInstallError(f"pinned asset is unexpectedly large: {asset.size} bytes")
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": f"hermes-auxiliary-brain/{__version__}"},
    )
    opener = urllib.request.build_opener(_PinnedHTTPSRedirectHandler())
    digest = hashlib.sha256()
    count = 0
    try:
        with opener.open(request, timeout=60) as response, destination.open("xb") as output:
            _validate_download_url(response.geturl())
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_ARCHIVE_BYTES or count > asset.size:
                    raise LlamaInstallError("llama.cpp download exceeded its pinned size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except (OSError, urllib.error.URLError) as exc:
        raise LlamaInstallError(f"could not download {asset.url}: {exc}") from exc
    if count != asset.size:
        raise LlamaInstallError(
            f"llama.cpp download size mismatch: expected {asset.size}, got {count}"
        )
    actual = digest.hexdigest()
    if actual != asset.sha256:
        raise LlamaInstallError(f"llama.cpp SHA256 mismatch: expected {asset.sha256}, got {actual}")


class _PinnedHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _ALLOWED_DOWNLOAD_HOSTS:
        raise LlamaInstallError(f"refusing unsafe llama.cpp download redirect: {url}")


def _safe_extract_archive(archive: Path, archive_name: str, destination: Path) -> None:
    if archive_name.endswith(".zip"):
        _safe_extract_zip(archive, destination)
        return
    if archive_name.endswith((".tar.gz", ".tgz")):
        _safe_extract_tar(archive, destination)
        return
    raise LlamaInstallError(f"unsupported llama.cpp archive format: {archive_name}")


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    total = 0
    try:
        source = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise LlamaInstallError(f"invalid llama.cpp ZIP archive: {exc}") from exc
    with source:
        for member in source.infolist():
            target = _safe_archive_target(destination, member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise LlamaInstallError(f"archive symlink is not allowed: {member.filename}")
            if member.flag_bits & 0x1:
                raise LlamaInstallError(
                    f"encrypted archive entry is not allowed: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise LlamaInstallError("llama.cpp archive expands beyond the safety limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("xb") as output:
                shutil.copyfileobj(input_file, output)
            if mode:
                target.chmod(mode & 0o777)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    total = 0
    try:
        source = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise LlamaInstallError(f"invalid llama.cpp tar archive: {exc}") from exc
    with source:
        for member in source:
            target = _safe_archive_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym():
                link_target = _safe_symlink_target(destination, target, member.linkname)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link_target)
                continue
            if not member.isfile():
                raise LlamaInstallError(f"non-file archive entry is not allowed: {member.name}")
            total += member.size
            if total > MAX_EXTRACTED_BYTES:
                raise LlamaInstallError("llama.cpp archive expands beyond the safety limit")
            extracted = source.extractfile(member)
            if extracted is None:
                raise LlamaInstallError(f"could not read archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.closing(extracted), target.open("xb") as output:
                shutil.copyfileobj(extracted, output)
            target.chmod(member.mode & 0o777)


def _safe_archive_target(destination: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any("\x00" in part for part in path.parts)
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise LlamaInstallError(f"unsafe archive path: {name}")
    target = destination.joinpath(*path.parts).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise LlamaInstallError(f"archive path escapes installation root: {name}") from exc
    return target


def _safe_symlink_target(destination: Path, link_path: Path, link_name: str) -> str:
    normalized = link_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or "\x00" in normalized
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise LlamaInstallError(f"unsafe archive symlink target: {link_name}")
    resolved_target = link_path.parent.joinpath(*path.parts).resolve()
    try:
        resolved_target.relative_to(destination.resolve())
    except ValueError as exc:
        raise LlamaInstallError(f"archive symlink escapes installation root: {link_name}") from exc
    return normalized


def _find_installed_executable(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    names = ("llama.exe", "llama") if os.name == "nt" else ("llama",)
    names += ("llama-server.exe", "llama-server") if os.name == "nt" else ("llama-server",)
    for name in names:
        candidates = sorted(path for path in root.rglob(name) if path.is_file())
        if candidates:
            return candidates[0]
    return None


def _verified_installed_executable(
    platform_root: Path,
    asset: LlamaReleaseAsset,
) -> Path | None:
    manifest_path = platform_root / "install.json"
    try:
        if manifest_path.stat().st_size > MAX_INSTALL_MANIFEST_BYTES:
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or (
            manifest.get("release") != LLAMA_CPP_RELEASE
            or manifest.get("asset") != asset.name
            or manifest.get("sha256") != asset.sha256
        ):
            return None
        relative = manifest.get("executable")
        expected_sha256 = manifest.get("executable_sha256")
        files = manifest.get("files")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or not isinstance(files, dict)
            or not 1 <= len(files) <= MAX_INSTALL_FILES
        ):
            return None
        actual_files = _artifact_tree_manifest(platform_root)
        if files != actual_files:
            return None
        executable = (platform_root / relative).resolve()
        if not executable.is_relative_to(platform_root.resolve()) or not executable.is_file():
            return None
        if _sha256_file(executable) != expected_sha256:
            return None
        return executable
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, RuntimeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.name == "install.json" and path.parent == root:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raw_target = os.readlink(path)
            normalized_target = raw_target.replace("\\", "/")
            target = PurePosixPath(normalized_target)
            if (
                not normalized_target
                or target.is_absolute()
                or "\x00" in normalized_target
                or (target.parts and target.parts[0].endswith(":"))
            ):
                raise OSError(f"runtime tree contains an unsafe symlink: {path}")
            resolved_target = path.resolve(strict=True)
            if not resolved_target.is_relative_to(resolved_root) or not resolved_target.is_file():
                raise OSError(f"runtime symlink leaves the installation root: {path}")
            files[relative] = {"type": "symlink", "target": normalized_target}
            if len(files) > MAX_INSTALL_FILES:
                raise OSError("runtime tree contains too many files")
            continue
        if not path.is_file():
            continue
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if len(files) > MAX_INSTALL_FILES:
            raise OSError("runtime tree contains too many files")
    return files


def _make_executable(path: Path) -> None:
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _platform_key() -> str:
    asset = get_release_asset()
    return (
        asset.name.removeprefix(f"llama-{LLAMA_CPP_RELEASE}-bin-")
        .removesuffix(".tar.gz")
        .removesuffix(".zip")
    )


def _resolve_executable(value: str | Path) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    has_separator = any(separator in text for separator in (os.sep, os.altsep) if separator)
    candidate = Path(text).expanduser() if has_separator or Path(text).is_absolute() else None
    if candidate is None:
        located = shutil.which(text)
        candidate = Path(located) if located else None
    if candidate is None or not candidate.is_file():
        return None
    if os.name != "nt" and not os.access(candidate, os.X_OK):
        return None
    return candidate.resolve()


def _require_explicit_executable(value: str | Path) -> Path:
    candidate = _resolve_executable(value)
    if candidate is None:
        raise LlamaExecutableNotFound(f"llama.cpp executable not found: {value}")
    return candidate


def _describe_executable(path: Path) -> LlamaExecutable:
    stem = path.stem.lower()
    if stem == "llama":
        style = "llama"
    elif stem == "llama-server":
        style = "llama-server"
    else:
        raise LlamaExecutableNotFound(
            f"unsupported llama.cpp executable name {path.name!r}; expected llama or llama-server"
        )
    return LlamaExecutable(str(path.resolve()), style)


def _normalize_loopback_host(host: str) -> str:
    value = str(host).strip().strip("[]")
    if value.lower() == "localhost":
        return DEFAULT_HOST
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise LlamaConfigurationError("llama.cpp host must be a literal loopback address") from exc
    if not address.is_loopback:
        raise LlamaConfigurationError("llama.cpp host must be loopback; public binds are refused")
    return address.compressed


def _validate_port(port: int) -> int:
    if isinstance(port, bool):
        raise LlamaConfigurationError("port must be an integer between 1 and 65535")
    try:
        value = int(port)
    except (TypeError, ValueError) as exc:
        raise LlamaConfigurationError("port must be an integer between 1 and 65535") from exc
    if value != port or not 1 <= value <= 65535:
        raise LlamaConfigurationError("port must be an integer between 1 and 65535")
    return value


def _validate_model(model: str) -> str:
    value = str(model).strip()
    if (
        not value
        or len(value) > 300
        or value.startswith("-")
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise LlamaConfigurationError(
            "model must be one Hugging Face repository[:quant] identifier"
        )
    return value


def _validate_extra_args(extra_args: Sequence[str]) -> list[str]:
    result: list[str] = []
    for raw in extra_args:
        value = str(raw)
        option = value.split("=", 1)[0]
        if option in _PROTECTED_SERVER_OPTIONS or option.startswith("--lora-"):
            raise LlamaConfigurationError(f"extra_args cannot override protected option {option}")
        if "\x00" in value:
            raise LlamaConfigurationError("extra_args cannot contain NUL bytes")
        result.append(value)
    return result


def _validate_lora_adapter(
    adapter_path: str | Path | None,
    expected_sha256: str | None,
    *,
    init_without_apply: bool,
) -> tuple[str | None, str | None]:
    if not isinstance(init_without_apply, bool):
        raise LlamaConfigurationError("lora_init_without_apply must be a boolean")
    if (adapter_path is None) != (expected_sha256 is None):
        raise LlamaConfigurationError(
            "lora_adapter_path and lora_adapter_sha256 must be provided together"
        )
    if adapter_path is None:
        if init_without_apply:
            raise LlamaConfigurationError("lora_init_without_apply requires a LoRA adapter")
        return None, None

    return _validate_gguf_artifact(
        adapter_path,
        expected_sha256,
        path_option="lora_adapter_path",
        sha256_option="lora_adapter_sha256",
        label="LoRA adapter",
    )


def _validate_gguf_artifact(
    artifact_path: str | Path | None,
    expected_sha256: str | None,
    *,
    path_option: str,
    sha256_option: str,
    label: str,
) -> tuple[str | None, str | None]:
    if (artifact_path is None) != (expected_sha256 is None):
        raise LlamaConfigurationError(
            f"{path_option} and {sha256_option} must be provided together"
        )
    if artifact_path is None:
        return None, None

    if not isinstance(expected_sha256, str):  # paired above; keeps type narrowing explicit
        raise LlamaConfigurationError(f"{sha256_option} must be a 64-character SHA256")
    normalized_sha256 = expected_sha256.strip().lower()
    if len(normalized_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha256
    ):
        raise LlamaConfigurationError(f"{sha256_option} must be a 64-character SHA256")

    try:
        raw_path = os.fspath(artifact_path)
    except TypeError as exc:
        raise LlamaConfigurationError(f"{path_option} must be an absolute .gguf path") from exc
    if not isinstance(raw_path, str) or "\x00" in raw_path:
        raise LlamaConfigurationError(f"{path_option} must be an absolute .gguf path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise LlamaConfigurationError(f"{path_option} must be absolute")
    if candidate.suffix.lower() != ".gguf":
        raise LlamaConfigurationError(f"{path_option} must name a .gguf file")
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise LlamaConfigurationError(f"{label} does not exist: {candidate}") from exc
    if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise LlamaConfigurationError(f"{path_option} must be a regular file")
    try:
        resolved = candidate.resolve(strict=True)
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LlamaConfigurationError(f"cannot read {label} {candidate}: {exc}") from exc
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != normalized_sha256:
        raise LlamaConfigurationError(
            f"{label} SHA256 mismatch for {resolved}: "
            f"expected {normalized_sha256}, got {actual_sha256}"
        )
    return str(resolved), normalized_sha256


def _validate_timeout(value: float, name: str, *, minimum: float = 0.0, maximum: float) -> float:
    if isinstance(value, bool):
        raise LlamaConfigurationError(f"{name} must be between {minimum:g} and {maximum:g}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise LlamaConfigurationError(f"{name} must be a number") from exc
    if not minimum <= normalized <= maximum:
        raise LlamaConfigurationError(f"{name} must be between {minimum:g} and {maximum:g}")
    return normalized


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        if path.stat().st_size > MAX_STATE_BYTES:
            raise LlamaServerStateError(f"server state exceeds {MAX_STATE_BYTES} bytes: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
    except LlamaServerStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LlamaServerStateError(f"cannot read server state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise LlamaServerStateError(f"server state must be a JSON object: {path}")
    if state.get("format_version") != STATE_FORMAT_VERSION:
        raise LlamaServerStateError(f"unsupported server state format in {path}")
    return state


def _state_string(state: dict[str, Any], key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value:
        raise LlamaServerStateError(f"invalid {key!r} in server state")
    return value


def _state_lora_adapter(state: dict[str, Any], state_path: Path) -> tuple[str | None, str | None]:
    return _state_gguf_artifact(
        state,
        state_path,
        path_key="lora_adapter_path",
        sha256_key="lora_adapter_sha256",
        label="LoRA adapter",
    )


def _state_gguf_artifact(
    state: dict[str, Any],
    state_path: Path,
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> tuple[str | None, str | None]:
    artifact_path = state.get(path_key)
    artifact_sha256 = state.get(sha256_key)
    if artifact_path is None and artifact_sha256 is None:
        return None, None
    if not isinstance(artifact_path, str) or not isinstance(artifact_sha256, str):
        raise LlamaServerStateError(f"invalid {label} metadata in {state_path}")
    if not Path(artifact_path).is_absolute() or Path(artifact_path).suffix.lower() != ".gguf":
        raise LlamaServerStateError(f"invalid {label} path in {state_path}")
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise LlamaServerStateError(f"invalid {label} SHA256 in {state_path}")
    return artifact_path, artifact_sha256


def _state_integer(state: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LlamaServerStateError(f"invalid {key!r} in server state")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _operation_lock(data_root: Path) -> Iterator[None]:
    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / LOCK_FILENAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        owner_pid = _read_lock_pid(lock_path)
        if owner_pid is not None and _pid_alive(owner_pid):
            raise LlamaServerStateError(
                f"another llama.cpp operation holds {lock_path}; try again shortly"
            ) from None
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0
        if age <= 300:
            raise LlamaServerStateError(
                f"another llama.cpp operation holds {lock_path}; try again shortly"
            ) from None
        lock_path.unlink(missing_ok=True)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise LlamaServerStateError(
                f"another llama.cpp operation holds {lock_path}; try again shortly"
            ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode())
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_lock_pid(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(256)
    except OSError:
        return None
    prefix = b"pid="
    if not raw.startswith(prefix):
        return None
    token = raw[len(prefix) :].split(maxsplit=1)[0]
    try:
        pid = int(token)
    except ValueError:
        return None
    return pid if 0 < pid <= 2**31 - 1 else None


def _server_environment(cache_root: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _SERVER_ENVIRONMENT_ALLOWLIST
    }
    for name in _PROTECTED_ENVIRONMENT:
        environment.pop(name, None)
    environment["LLAMA_CACHE"] = str(cache_root)
    return environment


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.exists():
        try:
            if "\nState:\tZ" in proc_status.read_text(encoding="utf-8", errors="replace"):
                return False
        except OSError:
            pass
    return True


def _windows_pid_alive(pid: int) -> bool:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _process_identity_matches(pid: int, expected: str) -> bool:
    actual = _process_image_path(pid)
    if actual is None:
        return False
    return os.path.normcase(os.path.realpath(actual)) == os.path.normcase(
        os.path.realpath(expected)
    )


def _process_image_path(pid: int) -> str | None:
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            size = ctypes.c_ulong(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return None
            return buffer.value
        finally:
            kernel32.CloseHandle(handle)
    proc_exe = Path(f"/proc/{pid}/exe")
    if proc_exe.exists():
        try:
            return os.readlink(proc_exe)
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            buffer = ctypes.create_string_buffer(4096)
            if libproc.proc_pidpath(pid, buffer, len(buffer)) <= 0:
                return None
            return os.fsdecode(buffer.value)
        except (OSError, AttributeError):
            return None
    return None


def _stop_posix_process(pid: int, executable: str, timeout_seconds: float) -> None:
    _signal_posix(pid, signal.SIGTERM)
    if _wait_for_exit(pid, timeout_seconds):
        return
    if not _process_identity_matches(pid, executable):
        if _pid_alive(pid):
            raise LlamaServerStateError(
                f"PID {pid} changed identity while stopping; refusing SIGKILL"
            )
        return
    _signal_posix(pid, signal.SIGKILL)
    if not _wait_for_exit(pid, min(timeout_seconds, 2.0)):
        raise LlamaServerError(f"llama.cpp PID {pid} did not stop")


def _signal_posix(pid: int, signum: int) -> None:
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        return


def _stop_windows_process(pid: int, executable: str, timeout_seconds: float) -> None:
    options: dict[str, Any] = {"capture_output": True, "text": True, "shell": False}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.run(["taskkill", "/PID", str(pid), "/T"], check=False, **options)
    if _wait_for_exit(pid, timeout_seconds):
        return
    if not _process_identity_matches(pid, executable):
        if _pid_alive(pid):
            raise LlamaServerStateError(
                f"PID {pid} changed identity while stopping; refusing forced taskkill"
            )
        return
    result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, **options)
    if not _wait_for_exit(pid, min(timeout_seconds, 2.0)):
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise LlamaServerError(f"llama.cpp PID {pid} did not stop{suffix}")


def _wait_for_exit(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return True


def _terminate_spawned_process(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()


def _endpoint_ready(host: str, port: int) -> bool:
    url_host = f"[{host}]" if ":" in host else host
    request = urllib.request.Request(f"http://{url_host}:{port}/health")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=0.25) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _read_log_tail(path: Path, limit: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
