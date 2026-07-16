from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import stat
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from auxiliary_brain import llama_server
from auxiliary_brain.llama_server import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    LlamaExecutable,
    LlamaExecutableNotFound,
    LlamaInstallError,
    LlamaReleaseAsset,
    LlamaServerStateError,
    build_server_command,
    get_release_asset,
    install_llama_cpp,
    resolve_data_root,
    start_llama_server,
    stop_llama_server,
)


def test_defaults_are_small_local_and_loopback_only() -> None:
    assert DEFAULT_MODEL == "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M"
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8080


def test_data_root_is_profile_scoped(tmp_path: Path) -> None:
    assert resolve_data_root(tmp_path) == (tmp_path / "auxiliary-brain" / "llama.cpp").resolve()


@pytest.mark.parametrize(
    ("executable", "expected"),
    [
        (
            LlamaExecutable("/opt/llama/bin/llama", "llama"),
            [
                "/opt/llama/bin/llama",
                "serve",
                "-hf",
                DEFAULT_MODEL,
                "--host",
                DEFAULT_HOST,
                "--port",
                str(DEFAULT_PORT),
            ],
        ),
        (
            LlamaExecutable(r"C:\Program Files\llama.cpp\llama-server.exe", "llama-server"),
            [
                r"C:\Program Files\llama.cpp\llama-server.exe",
                "-hf",
                DEFAULT_MODEL,
                "--host",
                DEFAULT_HOST,
                "--port",
                str(DEFAULT_PORT),
            ],
        ),
    ],
)
def test_build_server_command_uses_argument_lists_for_both_executable_styles(
    executable: LlamaExecutable,
    expected: list[str],
) -> None:
    assert build_server_command(executable) == expected


def test_build_server_command_keeps_extra_arguments_as_separate_values() -> None:
    executable = LlamaExecutable("/opt/llama/bin/llama", "llama")

    command = build_server_command(
        executable,
        model="repo/model:Q8_0",
        port=8123,
        extra_args=("--threads", "4", "--ctx-size", "2048"),
    )

    assert command == [
        "/opt/llama/bin/llama",
        "serve",
        "-hf",
        "repo/model:Q8_0",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--threads",
        "4",
        "--ctx-size",
        "2048",
    ]


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--host", "0.0.0.0"),
        ("--host=0.0.0.0",),
        ("--port", "9999"),
        ("--port=9999",),
        ("-hf", "some/other-model"),
    ],
)
def test_build_server_command_rejects_extra_arguments_that_override_safety_settings(
    extra_args: tuple[str, ...],
) -> None:
    executable = LlamaExecutable("llama-server", "llama-server")

    with pytest.raises(ValueError):
        build_server_command(executable, extra_args=extra_args)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "example.com"])
def test_build_server_command_rejects_non_loopback_bindings(host: str) -> None:
    executable = LlamaExecutable("llama-server", "llama-server")

    with pytest.raises(ValueError):
        build_server_command(executable, host=host)


@pytest.mark.parametrize(
    ("system", "machine", "name_fragment"),
    [
        ("Windows", "AMD64", "win-cpu-x64.zip"),
        ("Windows", "ARM64", "win-cpu-arm64.zip"),
        ("Linux", "x86_64", "ubuntu-x64.tar.gz"),
        ("Linux", "aarch64", "ubuntu-arm64.tar.gz"),
        ("Darwin", "x86_64", "macos-x64.tar.gz"),
        ("macOS", "arm64", "macos-arm64.tar.gz"),
    ],
)
def test_release_assets_cover_supported_desktop_platforms(
    system: str,
    machine: str,
    name_fragment: str,
) -> None:
    asset = get_release_asset(system=system, machine=machine)

    assert name_fragment in asset.name
    assert len(asset.sha256) == 64
    assert asset.size > 0
    assert asset.url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/")


def test_install_uses_a_fake_archive_and_reuses_the_profile_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = LlamaReleaseAsset("llama-test-bin.zip", "0" * 64, 42)
    executable_name = "llama.exe" if os.name == "nt" else "llama"
    downloads: list[Path] = []

    def fake_download(_asset: LlamaReleaseAsset, destination: Path) -> None:
        downloads.append(destination)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(f"bin/{executable_name}", b"pretend llama binary")

    monkeypatch.setattr(llama_server, "get_release_asset", lambda: asset)
    monkeypatch.setattr(llama_server, "_platform_key", lambda: "test-platform")
    monkeypatch.setattr(llama_server, "_download_release_asset", fake_download)

    first = install_llama_cpp(hermes_home=tmp_path)
    second = install_llama_cpp(hermes_home=tmp_path)

    assert first == second
    assert first.style == "llama"
    assert Path(first.path).is_file()
    assert Path(first.path).is_relative_to(
        tmp_path
        / "auxiliary-brain"
        / "llama.cpp"
        / llama_server.INSTALL_DIRECTORY
        / llama_server.LLAMA_CPP_RELEASE
    )
    assert len(downloads) == 1
    metadata = json.loads(
        (Path(first.path).parents[1] / "install.json").read_text(encoding="utf-8")
    )
    assert metadata["release"] == llama_server.LLAMA_CPP_RELEASE
    assert metadata["asset"] == asset.name
    assert not list(resolve_data_root(tmp_path).glob("*.part"))


def test_install_cleans_partial_downloads_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = LlamaReleaseAsset("llama-test-bin.zip", "0" * 64, 42)

    def broken_download(_asset: LlamaReleaseAsset, destination: Path) -> None:
        destination.write_bytes(b"untrusted partial archive")
        raise LlamaInstallError("checksum mismatch")

    monkeypatch.setattr(llama_server, "get_release_asset", lambda: asset)
    monkeypatch.setattr(llama_server, "_platform_key", lambda: "test-platform")
    monkeypatch.setattr(llama_server, "_download_release_asset", broken_download)

    with pytest.raises(LlamaInstallError, match="checksum mismatch"):
        install_llama_cpp(hermes_home=tmp_path)

    assert not list(resolve_data_root(tmp_path).glob("*.part"))
    assert not list(
        (
            resolve_data_root(tmp_path)
            / llama_server.INSTALL_DIRECTORY
            / llama_server.LLAMA_CPP_RELEASE
        ).glob("*.staging")
    )


@pytest.mark.parametrize("valid_digest", [True, False])
def test_download_enforces_the_pinned_size_and_checksum(
    valid_digest: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned official archive bytes"
    digest = hashlib.sha256(payload).hexdigest() if valid_digest else "0" * 64
    asset = LlamaReleaseAsset("llama-test-bin.zip", digest, len(payload))

    class FakeResponse(io.BytesIO):
        def geturl(self) -> str:
            return asset.url

    class FakeOpener:
        def open(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse(payload)

    monkeypatch.setattr(
        llama_server.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )
    destination = tmp_path / "download.part"

    if valid_digest:
        llama_server._download_release_asset(asset, destination)
        assert destination.read_bytes() == payload
    else:
        with pytest.raises(LlamaInstallError, match="SHA256 mismatch"):
            llama_server._download_release_asset(asset, destination)


@pytest.mark.parametrize("member", ["../outside", "/absolute", r"..\outside"])
def test_archive_extraction_rejects_paths_outside_the_install_root(
    member: str,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "llama-test.zip"
    destination = tmp_path / "install"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"nope")

    with pytest.raises(LlamaInstallError, match="unsafe archive path"):
        llama_server._safe_extract_archive(archive_path, archive_path.name, destination)

    assert not (tmp_path / "outside").exists()


def test_zip_archive_extraction_rejects_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "llama-test.zip"
    destination = tmp_path / "install"
    destination.mkdir()
    link = zipfile.ZipInfo("bin/llama")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../outside")

    with pytest.raises(LlamaInstallError, match="archive symlink is not allowed"):
        llama_server._safe_extract_archive(archive_path, archive_path.name, destination)


def _write_tar_with_link(
    path: Path,
    *,
    link_name: str,
    link_type: bytes = tarfile.SYMTYPE,
) -> None:
    payload = b"shared library"
    with tarfile.open(path, "w:gz") as archive:
        target = tarfile.TarInfo("lib/libllama.so.1")
        target.size = len(payload)
        target.mode = 0o755
        archive.addfile(target, io.BytesIO(payload))
        link = tarfile.TarInfo("lib/libllama.so")
        link.type = link_type
        link.linkname = link_name
        archive.addfile(link)


def test_tar_archive_allows_relative_symlink_that_stays_inside_install_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "llama-test.tar.gz"
    destination = tmp_path / "install"
    destination.mkdir()
    _write_tar_with_link(archive_path, link_name="libllama.so.1")
    links: list[tuple[Path, str]] = []

    def fake_symlink_to(path: Path, target: str, *_args: Any, **_kwargs: Any) -> None:
        links.append((path, target))

    # Avoid requiring Windows Developer Mode while still exercising extraction
    # and the path-containment check.
    monkeypatch.setattr(type(destination), "symlink_to", fake_symlink_to)

    llama_server._safe_extract_archive(archive_path, archive_path.name, destination)

    assert (destination / "lib" / "libllama.so.1").read_bytes() == b"shared library"
    assert links == [(destination / "lib" / "libllama.so", "libllama.so.1")]


@pytest.mark.parametrize("link_name", ["/etc/passwd", "../../outside", r"..\..\outside"])
def test_tar_archive_rejects_symlinks_that_escape_install_root(
    link_name: str,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "llama-test.tar.gz"
    destination = tmp_path / "install"
    destination.mkdir()
    _write_tar_with_link(archive_path, link_name=link_name)

    with pytest.raises(LlamaInstallError, match="archive symlink"):
        llama_server._safe_extract_archive(archive_path, archive_path.name, destination)


def test_tar_archive_rejects_hardlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "llama-test.tar.gz"
    destination = tmp_path / "install"
    destination.mkdir()
    _write_tar_with_link(
        archive_path,
        link_name="libllama.so.1",
        link_type=tarfile.LNKTYPE,
    )

    with pytest.raises(LlamaInstallError, match="non-file archive entry"):
        llama_server._safe_extract_archive(archive_path, archive_path.name, destination)


def test_start_auto_installs_then_spawns_an_argv_only_detached_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_path = tmp_path / ("llama.exe" if os.name == "nt" else "llama")
    executable_path.write_bytes(b"fake executable")
    executable = LlamaExecutable(str(executable_path.resolve()), "llama")
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 4242

        def poll(self) -> None:
            return None

    def missing(*_args: Any, **_kwargs: Any) -> LlamaExecutable:
        raise LlamaExecutableNotFound("missing")

    def fake_install(**kwargs: Any) -> LlamaExecutable:
        observed["install"] = kwargs
        return executable

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        observed["command"] = command
        observed["popen"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(llama_server, "find_llama_executable", missing)
    monkeypatch.setattr(llama_server, "install_llama_cpp", fake_install)
    monkeypatch.setattr(llama_server, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(llama_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llama_server, "_pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(llama_server, "_process_image_path", lambda _pid: str(executable_path))
    monkeypatch.setattr(llama_server, "_endpoint_ready", lambda *_args: True)
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_ARG_PORT", "9999")
    monkeypatch.setenv("LLAMA_ARG_MODEL", "evil/model")

    status = start_llama_server(
        hermes_home=tmp_path,
        install_if_missing=True,
        extra_args=("--threads", "2"),
    )

    assert observed["install"] == {"hermes_home": tmp_path}
    assert observed["command"] == [
        str(executable_path.resolve()),
        "serve",
        "-hf",
        DEFAULT_MODEL,
        "--host",
        DEFAULT_HOST,
        "--port",
        str(DEFAULT_PORT),
        "--threads",
        "2",
    ]
    options = observed["popen"]
    assert options["shell"] is False
    assert options["stdin"] is llama_server.subprocess.DEVNULL
    assert options["stderr"] is llama_server.subprocess.STDOUT
    assert options["cwd"] == str(executable_path.parent)
    assert "LLAMA_ARG_HOST" not in options["env"]
    assert "LLAMA_ARG_PORT" not in options["env"]
    assert "LLAMA_ARG_MODEL" not in options["env"]
    assert options["env"]["LLAMA_CACHE"] == str(resolve_data_root(tmp_path) / "cache")
    if os.name == "nt":
        expected_flags = (
            llama_server.subprocess.DETACHED_PROCESS
            | llama_server.subprocess.CREATE_NEW_PROCESS_GROUP
        )
        assert options["creationflags"] == expected_flags
        assert "start_new_session" not in options
    else:
        assert options["start_new_session"] is True
        assert "creationflags" not in options
    assert status.running is True
    assert status.ready is True
    assert status.identity_verified is True
    assert status.base_url == "http://127.0.0.1:8080/v1"


def test_start_refuses_an_occupied_port_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_path = tmp_path / ("llama-server.exe" if os.name == "nt" else "llama-server")
    executable_path.write_bytes(b"fake executable")
    executable = LlamaExecutable(str(executable_path.resolve()), "llama-server")
    monkeypatch.setattr(llama_server, "find_llama_executable", lambda *_args, **_kwargs: executable)
    monkeypatch.setattr(llama_server, "_port_is_open", lambda *_args: True)
    monkeypatch.setattr(
        llama_server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("an occupied port must prevent process creation"),
    )

    with pytest.raises(LlamaServerStateError, match="already in use"):
        start_llama_server(hermes_home=tmp_path)


def _write_server_state(home: Path, executable: Path, *, pid: int = 4242) -> Path:
    data_root = resolve_data_root(home)
    data_root.mkdir(parents=True, exist_ok=True)
    state_path = data_root / llama_server.STATE_FILENAME
    state_path.write_text(
        json.dumps(
            {
                "format_version": llama_server.STATE_FORMAT_VERSION,
                "pid": pid,
                "host": DEFAULT_HOST,
                "port": DEFAULT_PORT,
                "model": DEFAULT_MODEL,
                "executable": str(executable.resolve()),
                "started_at": "2026-07-16T12:00:00+00:00",
                "log_path": str(data_root / llama_server.LOG_FILENAME),
                "command": [str(executable.resolve()), "-hf", DEFAULT_MODEL],
            }
        ),
        encoding="utf-8",
    )
    return state_path


def test_stop_refuses_to_signal_a_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "llama-server"
    expected.write_bytes(b"old process image")
    state_path = _write_server_state(tmp_path, expected)
    monkeypatch.setattr(llama_server, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        llama_server,
        "_process_image_path",
        lambda _pid: str(tmp_path / "completely-unrelated-process"),
    )
    monkeypatch.setattr(
        llama_server,
        "_stop_posix_process",
        lambda *_args: pytest.fail("a reused PID must never receive a signal"),
    )
    monkeypatch.setattr(
        llama_server,
        "_stop_windows_process",
        lambda *_args: pytest.fail("a reused PID must never receive taskkill"),
    )

    with pytest.raises(LlamaServerStateError, match="does not match stored executable"):
        stop_llama_server(hermes_home=tmp_path)

    assert state_path.exists(), "unsafe metadata must remain for operator inspection"


def test_stop_removes_stale_metadata_without_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"old process image")
    state_path = _write_server_state(tmp_path, executable)
    monkeypatch.setattr(llama_server, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        llama_server,
        "_stop_posix_process",
        lambda *_args: pytest.fail("a dead PID must not receive a signal"),
    )
    monkeypatch.setattr(
        llama_server,
        "_stop_windows_process",
        lambda *_args: pytest.fail("a dead PID must not receive taskkill"),
    )

    status = stop_llama_server(hermes_home=tmp_path)

    assert status.running is False
    assert status.error == "removed stale server metadata"
    assert not state_path.exists()


def test_posix_stop_escalates_only_after_rechecking_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    waits = iter([False, True])
    identity_checks: list[tuple[int, str]] = []
    monkeypatch.setattr(llama_server.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        llama_server,
        "_signal_posix",
        lambda pid, signum: signals.append((pid, signum)),
    )
    monkeypatch.setattr(llama_server, "_wait_for_exit", lambda *_args: next(waits))
    monkeypatch.setattr(
        llama_server,
        "_process_identity_matches",
        lambda pid, executable: identity_checks.append((pid, executable)) or True,
    )

    llama_server._stop_posix_process(4242, "/opt/llama-server", 1.0)

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert identity_checks == [(4242, "/opt/llama-server")]


def test_windows_stop_uses_argv_taskkill_and_rechecks_before_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[list[str], dict[str, Any]]] = []
    waits = iter([False, True])
    identity_checks: list[tuple[int, str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(llama_server.subprocess, "run", fake_run)
    monkeypatch.setattr(llama_server, "_wait_for_exit", lambda *_args: next(waits))
    monkeypatch.setattr(
        llama_server,
        "_process_identity_matches",
        lambda pid, executable: identity_checks.append((pid, executable)) or True,
    )

    llama_server._stop_windows_process(4242, r"C:\llama\llama-server.exe", 1.0)

    assert [command for command, _options in commands] == [
        ["taskkill", "/PID", "4242", "/T"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]
    assert all(options["shell"] is False for _command, options in commands)
    assert identity_checks == [(4242, r"C:\llama\llama-server.exe")]
