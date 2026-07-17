from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from auxiliary_brain import training
from auxiliary_brain.training import TrainingError


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def training_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    root = training.training_root()
    assert root == hermes_home.resolve() / "auxiliary-brain" / "training"
    return root


def _make_bundle(
    root: Path,
    *,
    experimental: bool = False,
    all_tasks: bool = False,
) -> Path:
    bundle = root / "bundles" / "bundle-staging"
    train = bundle / "train.jsonl"
    holdout = bundle / "holdout.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text("", encoding="utf-8")
    holdout.write_text("", encoding="utf-8")
    _write_json(
        bundle / "manifest.json",
        {
            "format_version": 1,
            "model": {
                "id": training.DEFAULT_NATIVE_MODEL,
                "revision": training.DEFAULT_NATIVE_REVISION,
            },
            "task_contract_hashes": {
                task.key: training.task_contract_hash(task)
                for task in (
                    training.list_tasks() if all_tasks else (training.get_task("generic_extract"),)
                )
            },
            "files": {
                "train.jsonl": {"sha256": _sha256(train), "examples": 1},
                "holdout.jsonl": {"sha256": _sha256(holdout), "examples": 2},
            },
            "promotion": {
                "experimental": experimental,
                "promotable": not experimental,
            },
        },
    )
    manifest_sha256 = _sha256(bundle / "manifest.json")
    final = bundle.parent / f"bundle-{manifest_sha256[:16]}"
    if final.exists():
        shutil.rmtree(bundle)
        return final
    bundle.rename(final)
    return final


def _run_record(run_dir: Path, **updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format_version": training.RUN_FORMAT_VERSION,
        "run_id": run_dir.name,
        "status": "trained",
        "logs": {},
        "artifacts": {},
    }
    record.update(updates)
    _write_json(run_dir / "run.json", record)
    return record


def _make_convertible_run(root: Path, name: str = "run-convert") -> Path:
    run_dir = root / "runs" / name
    adapter_dir = run_dir / "adapter"
    base_dir = run_dir / "base-config"
    adapter_dir.mkdir(parents=True)
    base_dir.mkdir()
    weights = adapter_dir / "adapter_model.safetensors"
    weights.write_bytes(b"tiny adapter weights")
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_config.write_text("{}", encoding="utf-8")
    base_config = base_dir / "config.json"
    base_config.write_text('{"model_type":"lfm2"}', encoding="utf-8")
    _run_record(
        run_dir,
        artifacts={
            "peft_adapter": {
                "path": str(adapter_dir),
                "weights": str(weights),
                "sha256": _sha256(weights),
                "config": str(adapter_config),
                "config_sha256": _sha256(adapter_config),
            },
            "base_config": {
                "path": str(base_dir),
                "sha256": _sha256(base_config),
            },
        },
        logs={"trainer": str(run_dir / "trainer.log")},
    )
    return run_dir


def _make_deployment_entry(root: Path, run_id: str, base: Path) -> dict[str, Any]:
    run_dir = _make_promotable_run(root, run_id)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    adapter = record["artifacts"]["gguf_adapter"]
    evaluation = record["evaluation"]
    bundle_sha256 = record["bundle_manifest_sha256"]
    contracts = json.loads((Path(record["bundle"]) / "manifest.json").read_text(encoding="utf-8"))[
        "task_contract_hashes"
    ]
    return {
        "run_id": run_id,
        "promoted_at": "2026-07-17T00:00:00+00:00",
        "adapter_path": adapter["path"],
        "adapter_sha256": adapter["sha256"],
        "base_model_path": str(base),
        "base_model_sha256": _sha256(base),
        "base_model_repository": training.DEFAULT_GGUF_REPOSITORY,
        "base_model_revision": training.DEFAULT_GGUF_REVISION,
        "evaluation_path": evaluation["path"],
        "evaluation_sha256": evaluation["sha256"],
        "bundle_manifest_sha256": bundle_sha256,
        "task_contract_hashes": contracts,
    }


def _make_promotable_run(root: Path, run_id: str) -> Path:
    bundle = _make_bundle(root, all_tasks=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    bundle_sha256 = _sha256(bundle / "manifest.json")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    adapter = run_dir / "adapter-f16.gguf"
    adapter.write_bytes(b"promotable adapter")
    evaluation_path = run_dir / "evaluation.json"
    evaluation_report = {
        "format_version": training.EVALUATION_FORMAT_VERSION,
        "evaluation_contract_version": training.EVALUATION_CONTRACT_VERSION,
        "plugin_version": training.__version__,
        "run_id": run_id,
        "bundle_manifest_sha256": bundle_sha256,
        "task_contract_hashes": manifest["task_contract_hashes"],
        "quality_passed": True,
        "promotion_eligible": True,
        "experimental": False,
        "quality_gate": {
            "candidate_schema_valid_for_all": True,
            "no_overall_exact_regression": True,
            "no_overall_field_regression": True,
            "no_per_task_regression": True,
            "all_tasks_covered": True,
        },
        "adapter": {"sha256": _sha256(adapter)},
        "base_model": {"sha256": training.DEFAULT_GGUF_SHA256},
        "llama_cpp": {
            "release": training.LLAMA_CPP_RELEASE,
            "commit": training.LLAMA_CPP_COMMIT,
        },
        "holdout": {
            "sha256": manifest["files"]["holdout.jsonl"]["sha256"],
            "examples": 2,
            "manifest_examples": 2,
            "sample_limit": training.MAX_EVALUATION_EXAMPLES,
        },
    }
    _write_json(evaluation_path, evaluation_report)
    _run_record(
        run_dir,
        status="evaluated",
        experimental=False,
        bundle=str(bundle),
        bundle_manifest_sha256=bundle_sha256,
        artifacts={
            "gguf_adapter": {
                "path": str(adapter),
                "sha256": _sha256(adapter),
            }
        },
        evaluation={
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
            "quality_passed": True,
            "promotion_eligible": True,
        },
    )
    return run_dir


def test_status_is_profile_local_and_reports_blank_readiness(training_root: Path) -> None:
    report = training.training_status()

    assert report["root"] == str(training_root)
    assert report["readiness"]["ready"] is False
    assert report["readiness"]["counts"]["eligible"] == 0
    assert report["environments"]["trainer"]["ready"] is False
    assert report["environments"]["converter"]["ready"] is False
    assert report["base_model"]["gguf_ready"] is False
    assert report["latest_bundle"] is None
    assert report["latest_run"] is None
    assert report["deployment"] is None


def test_environment_status_and_install_use_isolated_pinned_pip_command(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_python = training_root.parent / "base-python.exe"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"python")
    commands: list[list[str]] = []

    def fake_logged(
        command: list[str],
        _log_path: Path,
        **_kwargs: Any,
    ) -> int:
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = training._environment_python(Path(command[3]))
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
        return 0

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        if "-c" in command:
            assert command[0] == str(base_python.resolve())
            return SimpleNamespace(stdout="3.11\n")
        commands.append(command)
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        return SimpleNamespace(stdout="pip==25.0\n")

    assert training._environment_status(training_root, "trainer")["ready"] is False
    monkeypatch.setattr(training, "_run_logged", fake_logged)
    monkeypatch.setattr(training, "_nvidia_available", lambda: True)
    monkeypatch.setattr(
        training,
        "_trainer_accelerator",
        lambda _python: {"torch": "2.13.0+cu130", "cuda_available": True},
    )
    monkeypatch.setattr(training.subprocess, "run", fake_run)

    result = training.install_training_environment(
        "trainer",
        python_executable=base_python,
    )["trainer"]

    assert result["ready"] is True
    assert commands[0] == [str(base_python.resolve()), "-m", "venv", commands[0][3]]
    assert commands[1][1:8] == [
        "-m",
        "pip",
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--prefer-binary",
    ]
    assert commands[1][8:] == [
        "--index-url",
        training.TRAINER_CUDA_INDEX,
        training.TRAINER_TORCH_REQUIREMENT,
    ]
    assert commands[2][1:8] == commands[1][1:8]
    assert commands[2][8:] == [
        "--index-url",
        training.PYPI_INDEX,
        *training.TRAINER_REQUIREMENTS,
    ]
    assert commands[3][1:] == ["-m", "pip", "--isolated", "freeze", "--all"]
    assert Path(result["python"]).is_file()
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["base_python"] == str(base_python.resolve())


def test_environment_status_rejects_a_corrupt_manifest(training_root: Path) -> None:
    destination = training_root / "envs" / "trainer"
    python = training._environment_python(destination)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (destination / "environment.json").write_text("not-json", encoding="utf-8")

    result = training._environment_status(training_root, "trainer")

    assert result["ready"] is False
    assert "cannot read" in result["error"]


def test_environment_status_rejects_stale_requirement_pins(training_root: Path) -> None:
    destination = training_root / "envs" / "trainer"
    python = training._environment_python(destination)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _write_json(
        destination / "environment.json",
        {
            "format_version": 1,
            "component": "trainer",
            "requirements_sha256": "0" * 64,
            "freeze_sha256": "1" * 64,
        },
    )

    result = training._environment_status(training_root, "trainer")

    assert result["ready"] is False
    assert "pins changed" in result["error"]


def test_subprocess_environment_is_allowlisted_and_disables_telemetry(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GH_TOKEN", "must-not-leak")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "must-not-leak")
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = training._subprocess_environment(training_root)

    assert "GH_TOKEN" not in environment
    assert "SLACK_BOT_TOKEN" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:3128"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"
    assert environment["HF_HOME"] == str(training_root / "cache" / "huggingface")


class _StreamingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.reads = 0

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int) -> bytes:
        self.reads += 1
        return self.chunks.pop(0) if self.chunks else b""


def test_download_streams_and_atomically_accepts_matching_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [b"small ", b"streamed ", b"artifact"]
    payload = b"".join(chunks)
    response = _StreamingResponse(chunks)
    requests: list[Any] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _StreamingResponse:
        requests.append(request)
        assert timeout == 60.0
        return response

    monkeypatch.setattr(training.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "artifact.bin"

    training._download_verified(
        "https://example.test/artifact.bin",
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    assert destination.read_bytes() == payload
    assert response.reads == len(chunks) + 1
    assert requests[0].full_url == "https://example.test/artifact.bin"
    assert not list(tmp_path.glob("*.part"))


def test_download_removes_staging_file_on_checksum_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"corrupt"
    monkeypatch.setattr(
        training.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _StreamingResponse([payload]),
    )
    destination = tmp_path / "artifact.bin"

    with pytest.raises(TrainingError, match="SHA256"):
        training._download_verified(
            "https://example.test/artifact.bin",
            destination,
            expected_sha256="0" * 64,
            expected_size=len(payload),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_run_training_persists_success_and_tiny_smoke_settings(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root)
    commands: list[list[str]] = []
    monkeypatch.setattr(training, "_new_run_id", lambda _name: "run-success")
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "isolated-python", "ready": True},
    )

    def fake_trainer(command: list[str], _log_path: Path, **_kwargs: Any) -> int:
        commands.append(command)
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        run_dir = Path(request["output_dir"])
        adapter = run_dir / "adapter"
        base = run_dir / "base-config"
        adapter.mkdir()
        base.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (base / "config.json").write_text("{}", encoding="utf-8")
        _write_json(
            Path(request["result_path"]),
            {
                "status": "complete",
                "adapter_dir": str(adapter),
                "base_config_dir": str(base),
                "metrics": {"train_loss": 0.5},
            },
        )
        return 0

    monkeypatch.setattr(training, "_run_logged", fake_trainer)

    record = training.run_training(bundle, smoke=True, allow_cpu=True)

    assert record["status"] == "trained"
    assert record["experimental"] is True
    assert record["hyperparameters"]["allow_cpu"] is True
    assert record["hyperparameters"]["max_steps"] == 2
    assert record["hyperparameters"]["max_length"] == 512
    assert record["hyperparameters"]["gradient_accumulation_steps"] == 1
    assert record["artifacts"]["peft_adapter"]["sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert commands[0][:3] == [
        "isolated-python",
        str(Path(training.__file__).with_name("trainer_backend.py")),
        "--request",
    ]
    persisted = json.loads(
        (training_root / "runs" / "run-success" / "run.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "trained"


@pytest.mark.parametrize(
    ("outcome", "expected_status", "message"),
    [
        ("failure", "training_failed", "exited with code 7"),
        ("timeout", "training_timeout", "timed out"),
    ],
)
def test_run_training_persists_failure_states(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_status: str,
    message: str,
) -> None:
    bundle = _make_bundle(training_root)
    monkeypatch.setattr(training, "_new_run_id", lambda _name: f"run-{outcome}")
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "isolated-python", "ready": True},
    )

    def fake_trainer(command: list[str], _log_path: Path, **_kwargs: Any) -> int:
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return 7

    monkeypatch.setattr(training, "_run_logged", fake_trainer)

    with pytest.raises(TrainingError, match=message):
        training.run_training(bundle, timeout_seconds=1)

    record = json.loads(
        (training_root / "runs" / f"run-{outcome}" / "run.json").read_text(encoding="utf-8")
    )
    assert record["status"] == expected_status
    if outcome == "failure":
        assert record["return_code"] == 7


def test_run_training_persists_spawn_failure_state(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root)
    monkeypatch.setattr(training, "_new_run_id", lambda _name: "run-spawn-failure")
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "isolated-python", "ready": True},
    )
    monkeypatch.setattr(
        training,
        "_run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot spawn trainer")),
    )

    with pytest.raises(TrainingError, match="trainer could not start"):
        training.run_training(bundle)

    record = json.loads(
        (training_root / "runs" / "run-spawn-failure" / "run.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "training_failed"


def test_run_training_surfaces_sanitized_backend_failure(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root)
    monkeypatch.setattr(training, "_new_run_id", lambda _name: "run-safe-failure")
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "isolated-python", "ready": True},
    )

    def fake_trainer(command: list[str], _log_path: Path, **_kwargs: Any) -> int:
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        _write_json(
            Path(request["result_path"]),
            {
                "format_version": 1,
                "status": "failed",
                "phase": "preflight",
                "error": {
                    "code": "assistant_tokens_truncated",
                    "message": "assistant tokens were removed at the configured max length",
                },
            },
        )
        return 1

    monkeypatch.setattr(training, "_run_logged", fake_trainer)

    with pytest.raises(TrainingError, match="assistant_tokens_truncated"):
        training.run_training(bundle)

    record = json.loads(
        (training_root / "runs" / "run-safe-failure" / "run.json").read_text(encoding="utf-8")
    )
    assert record["trainer_failure"] == {
        "phase": "preflight",
        "code": "assistant_tokens_truncated",
        "message": "assistant tokens were removed at the configured max length",
    }


def test_run_training_rejects_result_path_traversal(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root)
    outside = training_root.parent / "stolen-adapter"
    outside.mkdir(parents=True)
    monkeypatch.setattr(training, "_new_run_id", lambda _name: "run-traversal")
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "isolated-python", "ready": True},
    )

    def fake_trainer(command: list[str], _log_path: Path, **_kwargs: Any) -> int:
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        _write_json(
            Path(request["result_path"]),
            {
                "status": "complete",
                "adapter_dir": str(outside),
                "base_config_dir": str(outside),
            },
        )
        return 0

    monkeypatch.setattr(training, "_run_logged", fake_trainer)

    with pytest.raises(TrainingError, match="adapter_dir is outside"):
        training.run_training(bundle)

    record = json.loads(
        (training_root / "runs" / "run-traversal" / "run.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "training_failed"


def test_conversion_persists_valid_gguf_and_exact_subprocess_argv(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_convertible_run(training_root)
    source = training_root / "tools" / "llama-source"
    source.mkdir(parents=True)
    converter = source / "convert_lora_to_gguf.py"
    converter.write_text("# fake", encoding="utf-8")
    commands: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "converter-python", "ready": True},
    )
    monkeypatch.setattr(training, "ensure_llama_source", lambda: source)

    def fake_converter(
        command: list[str],
        _log_path: Path,
        *,
        cwd: Path | None = None,
        **_kwargs: Any,
    ) -> int:
        commands.append((command, cwd))
        output = Path(command[command.index("--outfile") + 1])
        output.write_bytes(b"GGUF" + b"tiny-converted-adapter")
        return 0

    monkeypatch.setattr(training, "_run_logged", fake_converter)

    record = training.convert_training_run(run_dir)

    command, cwd = commands[0]
    assert record["status"] == "converted"
    assert record["artifacts"]["gguf_adapter"]["bytes"] > 16
    assert command == [
        "converter-python",
        str(converter),
        "--base",
        str(run_dir / "base-config"),
        "--outfile",
        str(run_dir / "adapter-f16.gguf"),
        "--outtype",
        "f16",
        str(run_dir / "adapter"),
    ]
    assert cwd == source


@pytest.mark.parametrize(("return_code", "writes_output"), [(8, False), (0, True)])
def test_conversion_persists_subprocess_or_invalid_output_failure(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    writes_output: bool,
) -> None:
    run_dir = _make_convertible_run(training_root, f"run-failure-{return_code}")
    source = training_root / "source"
    source.mkdir(parents=True)
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "converter-python", "ready": True},
    )
    monkeypatch.setattr(training, "ensure_llama_source", lambda: source)

    def fake_converter(command: list[str], *_args: Any, **_kwargs: Any) -> int:
        if writes_output:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"not-a-gguf-file!!")
        return return_code

    monkeypatch.setattr(training, "_run_logged", fake_converter)

    with pytest.raises(TrainingError, match="exited with code|valid GGUF"):
        training.convert_training_run(run_dir)

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "conversion_failed"


def test_conversion_persists_spawn_failure_state(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_convertible_run(training_root, "run-spawn-failure")
    source = training_root / "source"
    source.mkdir(parents=True)
    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "converter-python", "ready": True},
    )
    monkeypatch.setattr(training, "ensure_llama_source", lambda: source)
    monkeypatch.setattr(
        training,
        "_run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot spawn converter")),
    )

    with pytest.raises(TrainingError, match="adapter converter could not start"):
        training.convert_training_run(run_dir)

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "conversion_failed"


def test_conversion_refuses_corrupted_adapter_before_subprocess(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_convertible_run(training_root, "run-corrupt")
    (run_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    called = False

    def should_not_run(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "converter-python", "ready": True},
    )
    monkeypatch.setattr(training, "ensure_llama_source", lambda: training_root / "source")
    monkeypatch.setattr(training, "_run_logged", should_not_run)

    with pytest.raises(TrainingError, match="PEFT adapter SHA256 mismatch"):
        training.convert_training_run(run_dir)

    assert called is False


def test_conversion_refuses_corrupted_adapter_config_before_subprocess(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_convertible_run(training_root, "run-corrupt-config")
    (run_dir / "adapter" / "adapter_config.json").write_text(
        '{"lora_alpha":999}',
        encoding="utf-8",
    )
    called = False

    def should_not_run(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        training,
        "_require_environment",
        lambda _root, _component: {"python": "converter-python", "ready": True},
    )
    monkeypatch.setattr(training, "ensure_llama_source", lambda: training_root / "source")
    monkeypatch.setattr(training, "_run_logged", should_not_run)

    with pytest.raises(TrainingError, match="PEFT adapter config SHA256 mismatch"):
        training.convert_training_run(run_dir)

    assert called is False


def test_logged_worker_attaches_and_closes_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        def wait(self, *, timeout: float | None) -> int:
            events.append(f"wait:{timeout}")
            return 0

    process = Process()
    monkeypatch.setattr(
        training.subprocess,
        "Popen",
        lambda *_args, **_kwargs: events.append("spawn") or process,
    )
    monkeypatch.setattr(
        training,
        "_attach_windows_kill_job",
        lambda candidate: events.append("attach") or "job-handle" if candidate is process else None,
    )
    monkeypatch.setattr(
        training,
        "_close_windows_job",
        lambda handle: events.append(f"close:{handle}"),
    )

    result = training._run_logged(
        ["worker", "--safe"],
        tmp_path / "worker.log",
        env={},
        timeout_seconds=12,
    )

    assert result == 0
    assert events == ["spawn", "attach", "wait:12", "close:job-handle"]


def test_logged_worker_timeout_stops_before_closing_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        def wait(self, *, timeout: float | None) -> int:
            events.append(f"wait:{timeout}")
            raise subprocess.TimeoutExpired("worker", timeout)

    process = Process()
    monkeypatch.setattr(training.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(training, "_attach_windows_kill_job", lambda _process: "job-handle")
    monkeypatch.setattr(
        training,
        "_stop_spawned_process",
        lambda candidate: events.append("stop") if candidate is process else None,
    )
    monkeypatch.setattr(
        training,
        "_close_windows_job",
        lambda handle: events.append(f"close:{handle}"),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        training._run_logged(
            ["worker"],
            tmp_path / "worker.log",
            env={},
            timeout_seconds=1,
        )

    assert events == ["wait:1", "stop", "close:job-handle"]


def test_logged_worker_interrupt_stops_before_closing_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        def wait(self, *, timeout: float | None) -> int:
            events.append(f"wait:{timeout}")
            raise KeyboardInterrupt

    process = Process()
    monkeypatch.setattr(training.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(training, "_attach_windows_kill_job", lambda _process: "job-handle")
    monkeypatch.setattr(
        training,
        "_stop_spawned_process",
        lambda candidate: events.append("stop") if candidate is process else None,
    )
    monkeypatch.setattr(
        training,
        "_close_windows_job",
        lambda handle: events.append(f"close:{handle}"),
    )

    with pytest.raises(KeyboardInterrupt):
        training._run_logged(
            ["worker"],
            tmp_path / "worker.log",
            env={},
            timeout_seconds=None,
        )

    assert events == ["wait:None", "stop", "close:job-handle"]


def test_logged_worker_attach_failure_is_propagated_before_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited = False

    class Process:
        def wait(self, *, timeout: float | None) -> int:
            nonlocal waited
            waited = True
            return 0

    monkeypatch.setattr(training.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        training,
        "_attach_windows_kill_job",
        lambda _process: (_ for _ in ()).throw(TrainingError("cannot supervise worker")),
    )

    with pytest.raises(TrainingError, match="cannot supervise worker"):
        training._run_logged(
            ["worker"],
            tmp_path / "worker.log",
            env={},
            timeout_seconds=1,
        )

    assert waited is False


def _evaluation_rows() -> list[dict[str, Any]]:
    return [
        {
            "task": "generic_extract",
            "prediction_id": "pred-generic",
            "messages": [{"role": "user", "content": "GENERIC"}],
            "expected": {
                "summary": "expected",
                "category": "note",
                "entities": [],
                "action_items": [],
                "fields": {},
                "confidence": 1.0,
            },
        },
        {
            "task": "progress_checkin",
            "prediction_id": "pred-progress",
            "messages": [{"role": "user", "content": "PROGRESS"}],
            "expected": {
                "category": "training",
                "outcome": "completed",
                "quantity": 1,
                "unit": "session",
                "occurred_at": None,
                "note": "done",
                "next_action": None,
                "confidence": 1.0,
            },
        },
    ]


def test_holdout_reader_binds_full_count_and_samples_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    generic = _evaluation_rows()[0]["expected"]
    progress = _evaluation_rows()[1]["expected"]
    rows: list[dict[str, Any]] = []
    generic_ids = [f"generic-{index:03d}" for index in range(120)]
    largest_generic_rank = max(
        hashlib.sha256(f"generic_extract\0{prediction_id}".encode()).hexdigest()
        for prediction_id in generic_ids
    )
    progress_id = next(
        f"progress-{index}"
        for index in range(10_000)
        if hashlib.sha256(f"progress_checkin\0progress-{index}".encode()).hexdigest()
        > largest_generic_rank
    )
    for task, prediction_id, expected in [
        *[("generic_extract", prediction_id, generic) for prediction_id in generic_ids],
        ("progress_checkin", progress_id, progress),
    ]:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": prediction_id},
                    {"role": "assistant", "content": json.dumps(expected)},
                ],
                "metadata": {"task": task, "prediction_id": prediction_id},
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    first, first_total = training._read_holdout_rows(path)
    second, second_total = training._read_holdout_rows(path)

    assert first_total == second_total == 121
    assert len(first) == training.MAX_EVALUATION_EXAMPLES
    assert first == second
    assert any(row["prediction_id"] == progress_id for row in first)


class _EvaluationClient:
    def __init__(self, *, trade_regressions: bool) -> None:
        self.trade_regressions = trade_regressions

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        candidate = kwargs["extra_body"]["lora"][0]["scale"] == 1.0
        generic = "GENERIC" in messages[-1]["content"]
        if generic:
            value = {
                "summary": "expected",
                "category": "note",
                "entities": [],
                "action_items": [],
                "fields": {},
                "confidence": 0.9,
            }
            if self.trade_regressions and not candidate:
                value["category"] = "other"
        else:
            value = {
                "category": "training",
                "outcome": "completed",
                "quantity": 1,
                "unit": "session",
                "occurred_at": None,
                "note": "done",
                "next_action": None,
                "confidence": 0.9,
            }
            if self.trade_regressions and candidate:
                value["note"] = "different"
        return json.dumps(value)


def test_evaluation_passes_when_candidate_is_complete_and_not_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _EvaluationClient(trade_regressions=False)
    monkeypatch.setattr(training, "OpenAICompatibleClient", lambda *_args, **_kwargs: client)

    report = training._evaluate_rows(_evaluation_rows(), "http://local/v1", "tiny")

    assert report["quality_passed"] is True
    assert all(report["quality_gate"].values())
    assert report["candidate"]["completed"] == 2
    assert report["candidate"]["exact_accuracy"] == 1.0


def test_evaluation_per_task_gate_catches_regression_hidden_by_equal_overall_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _EvaluationClient(trade_regressions=True)
    monkeypatch.setattr(training, "OpenAICompatibleClient", lambda *_args, **_kwargs: client)

    report = training._evaluate_rows(_evaluation_rows(), "http://local/v1", "tiny")

    assert report["baseline"]["exact_accuracy"] == report["candidate"]["exact_accuracy"]
    assert report["baseline"]["field_accuracy"] == report["candidate"]["field_accuracy"]
    assert report["quality_gate"]["no_overall_exact_regression"] is True
    assert report["quality_gate"]["no_overall_field_regression"] is True
    assert report["quality_gate"]["no_per_task_regression"] is False
    assert report["quality_passed"] is False


def test_evaluation_refuses_an_already_open_port_before_spawn(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root, all_tasks=True)
    run_dir = training_root / "runs" / "run-port-busy"
    run_dir.mkdir(parents=True)
    adapter = run_dir / "adapter-f16.gguf"
    adapter.write_bytes(b"candidate adapter")
    _run_record(
        run_dir,
        status="converted",
        bundle=str(bundle),
        bundle_manifest_sha256=_sha256(bundle / "manifest.json"),
        artifacts={
            "gguf_adapter": {
                "path": str(adapter),
                "sha256": _sha256(adapter),
            }
        },
    )
    monkeypatch.setattr(
        training,
        "_read_holdout_rows",
        lambda _path: (_evaluation_rows(), len(_evaluation_rows())),
    )
    monkeypatch.setattr(training, "ensure_gguf_base", lambda: training_root / "base.gguf")
    monkeypatch.setattr(training, "_ensure_pinned_llama", lambda: "pinned-llama-server")
    monkeypatch.setattr(training, "build_server_command", lambda *_args, **_kwargs: ["llama"])
    probes: list[tuple[str, int]] = []
    monkeypatch.setattr(
        training,
        "_server_port_is_open",
        lambda host, port: probes.append((host, port)) or True,
    )

    with pytest.raises(TrainingError, match="evaluation port 18081 is already in use"):
        training.evaluate_training_run(run_dir, port=18081)

    assert probes == [(training.DEFAULT_HOST, 18081)]
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "converted"


@pytest.mark.parametrize("outcome", ["success", "timeout", "attach_failure"])
def test_evaluation_process_supervision_closes_on_every_exit(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    bundle = _make_bundle(training_root, all_tasks=True)
    run_dir = training_root / "runs" / f"run-supervision-{outcome}"
    run_dir.mkdir(parents=True)
    adapter = run_dir / "adapter-f16.gguf"
    adapter.write_bytes(b"candidate adapter")
    _run_record(
        run_dir,
        status="converted",
        bundle=str(bundle),
        bundle_manifest_sha256=_sha256(bundle / "manifest.json"),
        experimental=False,
        artifacts={
            "gguf_adapter": {
                "path": str(adapter),
                "sha256": _sha256(adapter),
            }
        },
    )
    process = object()
    events: list[str] = []
    monkeypatch.setattr(
        training,
        "_read_holdout_rows",
        lambda _path: (_evaluation_rows(), len(_evaluation_rows())),
    )
    monkeypatch.setattr(training, "ensure_gguf_base", lambda: training_root / "base.gguf")
    monkeypatch.setattr(training, "_ensure_pinned_llama", lambda: "pinned-llama-server")
    monkeypatch.setattr(training, "build_server_command", lambda *_args, **_kwargs: ["llama"])
    monkeypatch.setattr(training, "_server_port_is_open", lambda *_args: False)
    monkeypatch.setattr(
        training.subprocess,
        "Popen",
        lambda *_args, **_kwargs: events.append("spawn") or process,
    )

    def attach(candidate: object) -> str:
        assert candidate is process
        events.append("attach")
        if outcome == "attach_failure":
            raise TrainingError("cannot supervise evaluation server")
        return "job-handle"

    monkeypatch.setattr(training, "_attach_windows_kill_job", attach)
    monkeypatch.setattr(
        training,
        "_stop_spawned_process",
        lambda candidate: events.append("stop") if candidate is process else None,
    )
    monkeypatch.setattr(
        training,
        "_close_windows_job",
        lambda handle: events.append(f"close:{handle}"),
    )
    monkeypatch.setattr(
        training,
        "_wait_for_ephemeral_server",
        lambda *_args, **_kwargs: (
            events.append("ready")
            or SimpleNamespace(
                base_url="http://127.0.0.1:18082/v1",
                choose_model=lambda _value: "tiny",
            )
        ),
    )
    monkeypatch.setattr(training, "verify_loaded_adapter", lambda *_args, **_kwargs: None)

    def evaluate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("evaluate")
        if outcome == "timeout":
            raise TrainingError("evaluation exceeded its bounded runtime")
        return {
            "baseline": {},
            "candidate": {},
            "quality_gate": {},
            "quality_passed": True,
        }

    monkeypatch.setattr(training, "_evaluate_rows", evaluate)

    if outcome == "success":
        record = training.evaluate_training_run(run_dir, port=18082)
        assert record["status"] == "evaluated"
        assert events == [
            "spawn",
            "attach",
            "ready",
            "evaluate",
            "stop",
            "close:job-handle",
        ]
    else:
        expected = "bounded runtime" if outcome == "timeout" else "cannot supervise"
        with pytest.raises(TrainingError, match=expected):
            training.evaluate_training_run(run_dir, port=18082)
        expected_events = ["spawn", "attach"]
        if outcome == "timeout":
            expected_events.extend(["ready", "evaluate"])
        close_event = "close:job-handle" if outcome == "timeout" else "close:None"
        expected_events.extend(["stop", close_event])
        assert events == expected_events
        persisted = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "evaluation_failed"


@pytest.mark.parametrize(
    ("experimental", "quality_passed"),
    [(True, True), (False, False)],
)
def test_promotion_rejects_experimental_or_failing_evaluation(
    training_root: Path,
    experimental: bool,
    quality_passed: bool,
) -> None:
    run_dir = (
        training_root / "runs" / ("run-experimental" if experimental else "run-quality-failed")
    )
    _run_record(
        run_dir,
        status="evaluated",
        experimental=experimental,
        evaluation={
            "promotion_eligible": quality_passed and not experimental,
            "quality_passed": quality_passed,
        },
    )

    with pytest.raises(TrainingError, match="not promotion eligible"):
        training.promote_training_run(run_dir)


def test_promotion_binds_bundle_evaluation_adapter_and_history(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    prior = _make_deployment_entry(training_root, "run-prior", base)
    _write_json(
        training_root / "deployment.json",
        {
            "format_version": training.DEPLOYMENT_FORMAT_VERSION,
            "active": prior,
            "history": [],
            "updated_at": "before",
        },
    )
    run_dir = _make_promotable_run(training_root, "run-candidate")
    target = {
        "original_executable": "llama-server",
        "deployment_executable": "llama-server",
        "host": "127.0.0.1",
        "port": 8080,
    }
    restarts: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    monkeypatch.setattr(training, "ensure_gguf_base", lambda: base)
    monkeypatch.setattr(training, "_managed_server_restart_target", lambda: target)
    monkeypatch.setattr(
        training,
        "_restart_managed_server",
        lambda active, restart_target: restarts.append((active, restart_target)) or True,
    )

    result = training.promote_training_run(run_dir)

    assert result["active"]["run_id"] == "run-candidate"
    assert result["history"] == [prior]
    assert result["managed_server_restarted"] is True
    assert restarts == [(result["active"], target)]
    persisted = json.loads((training_root / "deployment.json").read_text(encoding="utf-8"))
    assert persisted["active"] == result["active"]
    assert persisted["history"] == [prior]


def test_promotion_restores_pointer_and_server_target_after_start_failure(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    prior = _make_deployment_entry(training_root, "run-prior", base)
    before = {
        "format_version": training.DEPLOYMENT_FORMAT_VERSION,
        "active": prior,
        "history": [],
        "updated_at": "before",
    }
    _write_json(training_root / "deployment.json", before)
    run_dir = _make_promotable_run(training_root, "run-candidate")
    target = {
        "original_executable": "llama-server",
        "deployment_executable": "llama-server",
        "host": "127.0.0.1",
        "port": 8080,
    }
    calls: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []

    def flaky_restart(
        active: dict[str, Any] | None,
        restart_target: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> bool:
        calls.append((active, restart_target))
        if len(calls) == 1:
            raise RuntimeError("new adapter did not start")
        return True

    monkeypatch.setattr(training, "ensure_gguf_base", lambda: base)
    monkeypatch.setattr(training, "_managed_server_restart_target", lambda: target)
    monkeypatch.setattr(training, "_restart_managed_server", flaky_restart)

    with pytest.raises(TrainingError, match="prior deployment pointer was restored"):
        training.promote_training_run(run_dir)

    assert json.loads((training_root / "deployment.json").read_text(encoding="utf-8")) == before
    assert calls[0][1] == target
    assert calls[1] == (prior, target)


def test_active_deployment_validates_hashes_and_confines_paths(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    base_sha = _sha256(base)
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", base_sha)
    entry = _make_deployment_entry(training_root, "run-active", base)
    _write_json(
        training_root / "deployment.json",
        {
            "format_version": training.DEPLOYMENT_FORMAT_VERSION,
            "active": entry,
            "history": [],
        },
    )

    assert training.active_deployment_artifacts() == entry

    adapter_path = Path(entry["adapter_path"])
    original_adapter = adapter_path.read_bytes()
    adapter_path.write_bytes(b"tampered")
    with pytest.raises(TrainingError, match="GGUF adapter SHA256 mismatch"):
        training.active_deployment_artifacts()

    adapter_path.write_bytes(original_adapter)
    outside = training_root.parent / "outside.gguf"
    outside.write_bytes(b"outside")
    entry["adapter_path"] = str(outside)
    entry["adapter_sha256"] = _sha256(outside)
    _write_json(
        training_root / "deployment.json",
        {
            "format_version": training.DEPLOYMENT_FORMAT_VERSION,
            "active": entry,
            "history": [],
        },
    )
    with pytest.raises(TrainingError, match="deployment does not match its evaluated training run"):
        training.active_deployment_artifacts()


def test_rollback_selects_and_verifies_previous_deployment(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    current = _make_deployment_entry(training_root, "run-current", base)
    previous = _make_deployment_entry(training_root, "run-previous", base)
    _write_json(
        training_root / "deployment.json",
        {
            "format_version": training.DEPLOYMENT_FORMAT_VERSION,
            "active": current,
            "history": [previous],
            "updated_at": "before",
        },
    )
    restarted_with: list[dict[str, Any] | None] = []
    monkeypatch.setattr(
        training,
        "_restart_managed_server",
        lambda active, _target: restarted_with.append(active) or True,
    )

    result = training.rollback_training_deployment()

    assert result["active"] == previous
    assert result["history"] == []
    assert result["managed_server_restarted"] is True
    assert restarted_with == [previous]
    persisted = json.loads((training_root / "deployment.json").read_text(encoding="utf-8"))
    assert persisted["active"] == previous


def test_rollback_recovers_when_current_adapter_is_corrupt(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    current = _make_deployment_entry(training_root, "run-current", base)
    previous = _make_deployment_entry(training_root, "run-previous", base)
    Path(current["adapter_path"]).write_bytes(b"corrupt")
    _write_json(
        training_root / "deployment.json",
        {
            "format_version": training.DEPLOYMENT_FORMAT_VERSION,
            "active": current,
            "history": [previous],
            "updated_at": "before",
        },
    )
    monkeypatch.setattr(training, "_managed_server_restart_target", lambda: None)

    result = training.rollback_training_deployment()

    assert result["active"] == previous
    assert result["skipped_invalid_history"] == 0


def test_rollback_restores_pointer_when_server_restart_fails(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    current = _make_deployment_entry(training_root, "run-current", base)
    previous = _make_deployment_entry(training_root, "run-previous", base)
    before = {
        "format_version": training.DEPLOYMENT_FORMAT_VERSION,
        "active": current,
        "history": [previous],
        "updated_at": "before",
    }
    _write_json(training_root / "deployment.json", before)
    calls = 0

    def flaky_restart(
        _active: dict[str, Any] | None,
        _target: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("restart failed")
        return True

    monkeypatch.setattr(training, "_restart_managed_server", flaky_restart)

    with pytest.raises(TrainingError, match="previous deployment pointer was restored"):
        training.rollback_training_deployment()

    assert json.loads((training_root / "deployment.json").read_text(encoding="utf-8")) == before
    assert calls == 2


def test_restart_uses_captured_target_after_failed_start_left_server_stopped(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    monkeypatch.setattr(training, "ensure_gguf_base", lambda: base)
    monkeypatch.setattr(
        training,
        "get_llama_server_status",
        lambda: SimpleNamespace(running=False),
    )
    starts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        training,
        "start_llama_server",
        lambda **kwargs: starts.append(kwargs),
    )
    monkeypatch.setattr(
        training,
        "wait_for_llama_server",
        lambda **_kwargs: SimpleNamespace(ready=True),
    )
    target = {
        "original_executable": "llama-server",
        "deployment_executable": "llama-server",
        "host": "127.0.0.1",
        "port": 8080,
    }

    assert training._restart_managed_server(None, target, restore_original=True) is True
    assert starts == [
        {
            "executable": "llama-server",
            "install_if_missing": False,
            "model": training.DEFAULT_MODEL,
            "host": "127.0.0.1",
            "port": 8080,
            "model_path": base,
            "model_sha256": _sha256(base),
            "lora_adapter_path": None,
            "lora_adapter_sha256": None,
            "wait_ready_seconds": 0.0,
        }
    ]


def test_restart_stops_exact_candidate_before_propagating_readiness_timeout(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = training_root / "models" / "base.gguf"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"tiny base")
    monkeypatch.setattr(training, "DEFAULT_GGUF_SHA256", _sha256(base))
    monkeypatch.setattr(training, "ensure_gguf_base", lambda: base)
    target = {
        "original_executable": "original-llama-server",
        "deployment_executable": "pinned-llama-server",
        "host": "127.0.0.1",
        "port": 8080,
        "pid": 101,
        "started_at": "original-start",
        "model": training.DEFAULT_MODEL,
        "command": ["original-llama-server", "--port", "8080"],
        "model_path": str(base),
        "model_sha256": _sha256(base),
        "lora_adapter_path": None,
        "lora_adapter_sha256": None,
    }
    original = SimpleNamespace(
        running=True,
        identity_verified=True,
        executable=target["original_executable"],
        host=target["host"],
        port=target["port"],
        pid=target["pid"],
        started_at=target["started_at"],
        model=target["model"],
        command=tuple(target["command"]),
        model_path=target["model_path"],
        model_sha256=target["model_sha256"],
        lora_adapter_path=None,
        lora_adapter_sha256=None,
    )
    candidate = SimpleNamespace(
        running=True,
        identity_verified=True,
        executable=target["deployment_executable"],
        pid=202,
        started_at="candidate-start",
        command=(target["deployment_executable"], "--port", "8080"),
    )
    statuses = iter((original, candidate))
    events: list[str] = []
    monkeypatch.setattr(training, "get_llama_server_status", lambda: next(statuses))

    def stop_server(**_kwargs: Any) -> None:
        stopped = "original" if events.count("stop-original") == 0 else "candidate"
        events.append(f"stop-{stopped}")

    def start_server(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["wait_ready_seconds"] == 0.0
        events.append("start-candidate")
        return candidate

    def wait_until_ready(**_kwargs: Any) -> None:
        events.append("wait-candidate")
        raise TimeoutError("candidate did not become ready")

    monkeypatch.setattr(training, "stop_llama_server", stop_server)
    monkeypatch.setattr(training, "start_llama_server", start_server)
    monkeypatch.setattr(training, "wait_for_llama_server", wait_until_ready)

    with pytest.raises(TimeoutError, match="did not become ready"):
        try:
            training._restart_managed_server(None, target)
        except TimeoutError:
            events.append("outer-restore-can-proceed")
            raise

    assert events == [
        "stop-original",
        "start-candidate",
        "wait-candidate",
        "stop-candidate",
        "outer-restore-can-proceed",
    ]


def test_logs_are_bounded_tailed_and_confined_to_run(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = training_root / "runs" / "run-logs"
    log = run_dir / "trainer.log"
    log.parent.mkdir(parents=True)
    log.write_text("old-1\nold-2\nkeep-1\nkeep-2\n", encoding="utf-8")
    _run_record(run_dir, logs={"trainer": str(log)})
    monkeypatch.setattr(training, "MAX_LOG_BYTES", 16)

    assert training.read_training_logs(run_dir, lines=2) == "keep-1\nkeep-2"

    with pytest.raises(TrainingError, match="log lines"):
        training.read_training_logs(run_dir, lines=0)
    with pytest.raises(TrainingError, match="log stage"):
        training.read_training_logs(run_dir, stage="secrets")

    outside = training_root.parent / "outside.log"
    outside.write_text("do not read", encoding="utf-8")
    _run_record(run_dir, logs={"trainer": str(outside)})
    with pytest.raises(TrainingError, match="trainer log is outside"):
        training.read_training_logs(run_dir)


def test_bundle_resolution_rejects_traversal_and_corrupted_artifacts(
    training_root: Path,
) -> None:
    outside = training_root.parent / "outside-bundle"
    outside.mkdir(parents=True)
    with pytest.raises(TrainingError, match="outside the active profile"):
        training._resolve_bundle(training_root, outside)

    bundle = _make_bundle(training_root)
    train_file = bundle / "train.jsonl"
    train_file.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(TrainingError, match="train.jsonl SHA256 mismatch"):
        training._resolve_bundle(training_root, bundle)


def test_bundle_resolution_rejects_oversized_manifest(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(training_root)
    monkeypatch.setattr(training, "MAX_MANIFEST_BYTES", 1)

    with pytest.raises(TrainingError, match="manifest exceeds the size limit"):
        training._resolve_bundle(training_root, bundle)


def test_stale_lock_check_uses_non_destructive_process_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / ".lifecycle.lock"
    _write_json(lock, {"pid": 4242})
    probes: list[int] = []
    monkeypatch.setattr(
        training,
        "_process_is_alive",
        lambda pid: probes.append(pid) or True,
    )

    assert training._remove_stale_lock(lock) is False
    assert probes == [4242]
    assert lock.exists()
