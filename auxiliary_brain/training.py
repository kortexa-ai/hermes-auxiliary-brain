"""Profile-local LoRA training, conversion, evaluation, and deployment.

The normal Hermes/plugin environment stays standard-library-only.  Heavy ML
packages live in isolated virtual environments and run in subprocesses, so a
failed experiment cannot turn the agent installation into dependency soup.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .llama_server import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    LLAMA_CPP_RELEASE,
    LlamaServerError,
    _safe_extract_zip,
    build_server_command,
    get_llama_server_status,
    install_llama_cpp,
    start_llama_server,
    stop_llama_server,
    wait_for_llama_server,
)
from .llama_server import (
    _pid_alive as _process_is_alive,
)
from .llama_server import (
    _port_is_open as _server_port_is_open,
)
from .local_api import LocalAPIError, OpenAICompatibleClient, probe_endpoint
from .local_api import urlopen as local_urlopen
from .runtime import BrainRuntime
from .tasks import TaskParseError, get_task, list_tasks
from .training_data import (
    DEFAULT_HOLDOUT_PERCENT,
    DEFAULT_MIN_HOLDOUT_EXAMPLES,
    DEFAULT_MIN_TRAIN_EXAMPLES,
    DEFAULT_MIN_UNIQUE_EXAMPLES,
    DEFAULT_SEED,
    MAX_MANIFEST_BYTES,
    TRAINING_BUNDLE_FORMAT_VERSION,
    inspect_readiness,
    prepare_bundle,
    task_contract_hash,
)
from .version import __version__

TRAINING_STATE_FORMAT_VERSION = 1
RUN_FORMAT_VERSION = 1
EVALUATION_FORMAT_VERSION = 1
EVALUATION_CONTRACT_VERSION = 1
DEPLOYMENT_FORMAT_VERSION = 1
EXPECTED_QUALITY_GATE_KEYS = frozenset(
    {
        "all_tasks_covered",
        "candidate_schema_valid_for_all",
        "no_overall_exact_regression",
        "no_overall_field_regression",
        "no_per_task_regression",
    }
)

DEFAULT_NATIVE_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_NATIVE_REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
DEFAULT_GGUF_REPOSITORY = "LiquidAI/LFM2.5-230M-GGUF"
DEFAULT_GGUF_REVISION = "fa224d4cb60cffe61eb58726712ef255bb64d0b7"
DEFAULT_GGUF_FILENAME = "LFM2.5-230M-Q4_K_M.gguf"
DEFAULT_GGUF_SHA256 = "7bbd90384d3deffe4c646ec9643b212802d32d4ce417c90a1ec9282100650062"
DEFAULT_GGUF_SIZE = 153_406_304

LLAMA_CPP_COMMIT = "32e789fdfd598e9a1872da55ac941e4d94f030bd"
LLAMA_SOURCE_URL = (
    f"https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/{LLAMA_CPP_RELEASE}"
)
LLAMA_SOURCE_SHA256 = "0c6608b4382c8056f4c398b57a801abe090a056d4160e7c4f90af9536b0c5745"
LLAMA_SOURCE_SIZE = 37_487_250
LLAMA_CONVERTER_TREE_SHA256 = "b23f2651e10694083afceab84b1e0866d9eeda249f1afd200827e6ede6ac7cca"

TRAINER_TORCH_REQUIREMENT = "torch==2.13.0"
TRAINER_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
PYPI_INDEX = "https://pypi.org/simple"
TRAINER_REQUIREMENTS = (
    "transformers==5.2.0",
    "trl==1.8.0",
    "peft==0.19.1",
    "accelerate==1.14.0",
    "datasets==5.0.0",
    "safetensors==0.8.0",
)
TRAINER_TARGET_MODULES = ("q_proj", "k_proj", "v_proj")
COMPONENTS = frozenset({"trainer", "converter"})
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
MAX_LOG_BYTES = 4 * 1024 * 1024
MAX_TRAINER_RESULT_BYTES = 64 * 1024
MAX_JSON_BYTES = 1024 * 1024
DEFAULT_EVALUATION_PORT = 8081
DEFAULT_EVALUATION_TIMEOUT = 600.0
DEFAULT_EVALUATION_RUN_TIMEOUT = 1800.0
EVALUATION_REQUEST_TIMEOUT = 30.0
MAX_EVALUATION_FAILURES = 100
MAX_EVALUATION_EXAMPLES = 100
DEFAULT_TRAINING_MAX_LENGTH = 512
_LINUX_PARENT_DEATH_WRAPPER = """\
import ctypes, os, signal, sys
parent_pid = int(sys.argv[1])
command = sys.argv[2:]
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
    raise SystemExit(127)
if os.getppid() != parent_pid:
    os.kill(os.getpid(), signal.SIGTERM)
os.execvpe(command[0], command, os.environ)
"""

_SUBPROCESS_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "CUDA_CACHE_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CURL_CA_BUNDLE",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "LOGNAME",
        "NO_PROXY",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "USERPROFILE",
        "WINDIR",
    }
)


class TrainingError(RuntimeError):
    """A safe, operator-facing training lifecycle operation failed."""


def training_root() -> Path:
    """Return the active profile's durable training directory."""

    return BrainRuntime.data_root() / "training"


def training_status() -> dict[str, Any]:
    """Return a cheap status report without importing any ML dependency."""

    root = training_root()
    readiness = inspect_readiness(BrainRuntime().store())
    latest_run = _latest_record(root / "runs", "run.json")
    latest_bundle = _latest_record(root / "bundles", "manifest.json")
    deployment = _read_deployment(root, required=False)
    return {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "plugin_version": __version__,
        "root": str(root),
        "readiness": readiness,
        "environments": {
            component: _environment_status(root, component) for component in sorted(COMPONENTS)
        },
        "llama_cpp": {
            "release": LLAMA_CPP_RELEASE,
            "commit": LLAMA_CPP_COMMIT,
            "source_installed": _llama_source_path(root).is_dir(),
        },
        "base_model": {
            "id": DEFAULT_NATIVE_MODEL,
            "revision": DEFAULT_NATIVE_REVISION,
            "gguf_repository": DEFAULT_GGUF_REPOSITORY,
            "gguf_revision": DEFAULT_GGUF_REVISION,
            "gguf_path": str(_gguf_model_path(root)),
            "gguf_ready": _artifact_matches(
                _gguf_model_path(root), DEFAULT_GGUF_SHA256, DEFAULT_GGUF_SIZE
            ),
        },
        "latest_bundle": latest_bundle,
        "latest_run": latest_run,
        "deployment": deployment,
    }


def prepare_training(
    *,
    task_key: str | None = None,
    seed: int = DEFAULT_SEED,
    holdout_percent: int = DEFAULT_HOLDOUT_PERCENT,
    min_unique_examples: int = DEFAULT_MIN_UNIQUE_EXAMPLES,
    min_train_examples: int = DEFAULT_MIN_TRAIN_EXAMPLES,
    min_holdout_examples: int = DEFAULT_MIN_HOLDOUT_EXAMPLES,
    acknowledge_unattributed_gateway: bool = False,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Create one immutable, deterministic training bundle."""

    root = training_root()
    with _operation_lock(root, "workload"):
        return prepare_bundle(
            BrainRuntime().store(),
            root / "bundles",
            model=DEFAULT_NATIVE_MODEL,
            revision=DEFAULT_NATIVE_REVISION,
            task_key=task_key,
            seed=seed,
            holdout_percent=holdout_percent,
            min_unique_examples=min_unique_examples,
            min_train_examples=min_train_examples,
            min_holdout_examples=min_holdout_examples,
            acknowledge_unattributed_gateway=acknowledge_unattributed_gateway,
            allow_small=allow_small,
        )


def install_training_environment(
    component: str = "all",
    *,
    force: bool = False,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Create isolated trainer/converter environments under the active profile."""

    selected = sorted(COMPONENTS) if component == "all" else [_validate_component(component)]
    root = training_root()
    _ensure_private_directory(root)
    results: dict[str, Any] = {}
    with _operation_lock(root, "workload"):
        with _operation_lock(root, "install"):
            if "converter" in selected:
                ensure_llama_source(force=force)
            for name in selected:
                results[name] = _install_environment(
                    root,
                    name,
                    force=force,
                    python_executable=python_executable,
                )
    return results


def ensure_llama_source(*, force: bool = False) -> Path:
    """Install the checksum-pinned llama.cpp source used by the converter."""

    root = training_root()
    destination = _llama_source_path(root)
    if destination.is_dir() and not force:
        if _converter_source_sha256(destination) == LLAMA_CONVERTER_TREE_SHA256:
            return destination
        raise TrainingError(
            f"llama.cpp converter source changed at {destination}; "
            "rerun train install converter --force"
        )

    tools_root = destination.parent
    _ensure_private_directory(tools_root)
    stage = tools_root / f".source-{uuid.uuid4().hex}"
    archive = tools_root / f".llama-{uuid.uuid4().hex}.zip"
    try:
        _download_verified(
            LLAMA_SOURCE_URL,
            archive,
            expected_sha256=LLAMA_SOURCE_SHA256,
            expected_size=LLAMA_SOURCE_SIZE,
        )
        _ensure_private_directory(stage)
        _safe_extract_zip(archive, stage)
        children = [item for item in stage.iterdir() if item.is_dir()]
        if len(children) != 1:
            raise TrainingError("llama.cpp source archive did not contain one root directory")
        source = children[0]
        if _converter_source_sha256(source) != LLAMA_CONVERTER_TREE_SHA256:
            raise TrainingError("llama.cpp source archive does not match the pinned converter")
        if destination.exists():
            _safe_remove_tree(destination, tools_root)
        os.replace(source, destination)
        return destination
    except (OSError, LlamaServerError) as exc:
        raise TrainingError(f"cannot install pinned llama.cpp source: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            archive.unlink()
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def ensure_gguf_base(*, force: bool = False) -> Path:
    """Download the immutable Q4 base used for adapter evaluation/deployment."""

    root = training_root()
    destination = _gguf_model_path(root)
    if not force and _artifact_matches(destination, DEFAULT_GGUF_SHA256, DEFAULT_GGUF_SIZE):
        return destination
    _ensure_private_directory(destination.parent)
    url = (
        f"https://huggingface.co/{DEFAULT_GGUF_REPOSITORY}/resolve/"
        f"{DEFAULT_GGUF_REVISION}/{DEFAULT_GGUF_FILENAME}"
    )
    _download_verified(
        url,
        destination,
        expected_sha256=DEFAULT_GGUF_SHA256,
        expected_size=DEFAULT_GGUF_SIZE,
    )
    return destination


def run_training(
    bundle: str | Path | None = None,
    *,
    smoke: bool = False,
    allow_cpu: bool = False,
    seed: int = DEFAULT_SEED,
    max_length: int = DEFAULT_TRAINING_MAX_LENGTH,
    epochs: float = 3.0,
    max_steps: int | None = None,
    learning_rate: float = 0.0001,
    gradient_accumulation_steps: int = 4,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the isolated trainer and register its PEFT adapter artifact."""

    root = training_root()
    with _operation_lock(root, "workload"):
        return _run_training_locked(
            bundle,
            smoke=smoke,
            allow_cpu=allow_cpu,
            seed=seed,
            max_length=max_length,
            epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            timeout_seconds=timeout_seconds,
        )


def _run_training_locked(
    bundle: str | Path | None = None,
    *,
    smoke: bool = False,
    allow_cpu: bool = False,
    seed: int = DEFAULT_SEED,
    max_length: int = DEFAULT_TRAINING_MAX_LENGTH,
    epochs: float = 3.0,
    max_steps: int | None = None,
    learning_rate: float = 0.0001,
    gradient_accumulation_steps: int = 4,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run training while the profile-wide workload lock is held."""

    root = training_root()
    bundle_path, bundle_manifest = _resolve_bundle(root, bundle)
    experimental = bool(bundle_manifest.get("promotion", {}).get("experimental"))
    if experimental and not smoke:
        raise TrainingError("experimental small bundles may only run with --smoke")
    if smoke:
        max_steps = 2 if max_steps is None else min(max_steps, 2)
        max_length = min(max_length, DEFAULT_TRAINING_MAX_LENGTH)
        gradient_accumulation_steps = 1
    _validate_training_options(
        seed=seed,
        max_length=max_length,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        timeout_seconds=timeout_seconds,
    )
    environment = _require_environment(root, "trainer")
    run_id = _new_run_id(bundle_path.name)
    run_dir = root / "runs" / run_id
    _ensure_private_directory(root)
    _ensure_private_directory(run_dir)
    result_path = run_dir / "trainer-result.json"
    request_path = run_dir / "trainer-request.json"
    log_path = run_dir / "trainer.log"
    request = {
        "format_version": 1,
        "bundle_dir": str(bundle_path),
        "output_dir": str(run_dir),
        "result_path": str(result_path),
        "base_model": DEFAULT_NATIVE_MODEL,
        "base_model_revision": DEFAULT_NATIVE_REVISION,
        "allow_cpu": allow_cpu,
        "seed": seed,
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "target_modules": list(TRAINER_TARGET_MODULES),
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
    }
    if max_steps is not None:
        request["max_steps"] = max_steps
    _write_json_atomic(request_path, request)
    record = {
        "format_version": RUN_FORMAT_VERSION,
        "run_id": run_id,
        "status": "training",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "bundle": str(bundle_path),
        "bundle_manifest_sha256": _sha256_file(bundle_path / "manifest.json"),
        "experimental": experimental or smoke,
        "trainer_environment": environment,
        "hyperparameters": {
            key: value
            for key, value in request.items()
            if key not in {"bundle_dir", "output_dir", "result_path"}
        },
        "logs": {"trainer": str(log_path)},
    }
    _write_json_atomic(run_dir / "run.json", record)

    backend = Path(__file__).with_name("trainer_backend.py")
    command = [environment["python"], str(backend), "--request", str(request_path)]
    env = _subprocess_environment(root)
    try:
        return_code = _run_logged(
            command,
            log_path,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _update_run(run_dir, status="training_timeout")
        raise TrainingError(f"training timed out; see {log_path}") from exc
    except OSError as exc:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="training_failed")
        raise TrainingError(f"trainer could not start; see {log_path}") from exc
    except BaseException:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="training_failed")
        raise
    if return_code != 0:
        failure = _read_trainer_failure(result_path)
        changes: dict[str, Any] = {
            "status": "training_failed",
            "return_code": return_code,
        }
        if failure is not None:
            changes["trainer_failure"] = failure
        _update_run(run_dir, **changes)
        detail = f" ({failure['code']}: {failure['message']})" if failure is not None else ""
        raise TrainingError(f"trainer exited with code {return_code}{detail}; see {log_path}")

    try:
        return _complete_training_run(run_dir, result_path, log_path)
    except OSError as exc:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="training_failed")
        raise TrainingError(f"trainer artifacts could not be validated; see {log_path}") from exc
    except BaseException:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="training_failed")
        raise


def _complete_training_run(
    run_dir: Path,
    result_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    result = _read_json(
        result_path,
        label="trainer result",
        max_bytes=MAX_TRAINER_RESULT_BYTES,
    )
    if result.get("status") != "complete":
        raise TrainingError(f"trainer did not publish a complete result; see {log_path}")
    adapter_dir = _path_under(run_dir, result.get("adapter_dir"), "adapter_dir")
    base_config_dir = _path_under(run_dir, result.get("base_config_dir"), "base_config_dir")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    required_artifacts = (
        adapter_file,
        adapter_config,
        base_config_dir / "config.json",
    )
    for path in required_artifacts:
        if not path.is_file():
            raise TrainingError(f"trainer result is missing {path.name}; see {log_path}")
    artifacts = {
        "peft_adapter": {
            "path": str(adapter_dir),
            "weights": str(adapter_file),
            "sha256": _sha256_file(adapter_file),
            "config": str(adapter_config),
            "config_sha256": _sha256_file(adapter_config),
        },
        "base_config": {
            "path": str(base_config_dir),
            "sha256": _sha256_file(base_config_dir / "config.json"),
        },
    }
    return _update_run(
        run_dir,
        status="trained",
        completed_at=_utc_now(),
        trainer_result=result,
        artifacts=artifacts,
    )


def convert_training_run(
    run: str | Path | None = None,
    *,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Serialize conversion attempts for one run."""

    root = training_root()
    run_dir, _record = _resolve_run(root, run)
    with _operation_lock(root, "workload"):
        with _operation_lock(run_dir, "lifecycle"):
            return _convert_training_run_locked(run_dir, timeout_seconds=timeout_seconds)


def _convert_training_run_locked(
    run: str | Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Convert a PEFT adapter with the matching pinned llama.cpp converter."""

    root = training_root()
    run_dir, record = _resolve_run(root, run)
    status = record.get("status")
    if status in {"converted", "evaluated"}:
        existing_artifacts = _mapping(record.get("artifacts"), "run artifacts")
        existing_gguf = _mapping(
            existing_artifacts.get("gguf_adapter"),
            "GGUF adapter artifact",
        )
        existing_path = _path_under(
            run_dir,
            existing_gguf.get("path"),
            "GGUF adapter path",
        )
        _verify_artifact(
            existing_path,
            str(existing_gguf.get("sha256") or ""),
            label="GGUF adapter",
        )
        return record
    if status not in {
        "trained",
        "converting",
        "conversion_failed",
        "conversion_timeout",
    }:
        raise TrainingError(f"run {run_dir.name} is not ready for conversion")
    environment = _require_environment(root, "converter")
    source = ensure_llama_source()
    artifacts = _mapping(record.get("artifacts"), "run artifacts")
    peft = _mapping(artifacts.get("peft_adapter"), "PEFT adapter artifact")
    adapter_dir = _path_under(run_dir, peft.get("path"), "PEFT adapter path")
    weights = _path_under(run_dir, peft.get("weights"), "PEFT weights path")
    _verify_artifact(weights, str(peft.get("sha256") or ""), label="PEFT adapter")
    adapter_config = _path_under(
        run_dir,
        peft.get("config"),
        "PEFT adapter config path",
    )
    _verify_artifact(
        adapter_config,
        str(peft.get("config_sha256") or ""),
        label="PEFT adapter config",
    )
    base = _mapping(artifacts.get("base_config"), "base config artifact")
    base_dir = _path_under(run_dir, base.get("path"), "base config path")
    _verify_artifact(base_dir / "config.json", str(base.get("sha256") or ""), label="base config")

    output = run_dir / "adapter-f16.gguf"
    try:
        output.unlink(missing_ok=True)
    except OSError as exc:
        raise TrainingError(f"cannot clear an incomplete GGUF adapter: {exc}") from exc
    log_path = run_dir / "converter.log"
    command = [
        environment["python"],
        str(source / "convert_lora_to_gguf.py"),
        "--base",
        str(base_dir),
        "--outfile",
        str(output),
        "--outtype",
        "f16",
        str(adapter_dir),
    ]
    _update_run(run_dir, status="converting")
    try:
        return_code = _run_logged(
            command,
            log_path,
            env=_subprocess_environment(root),
            timeout_seconds=timeout_seconds,
            cwd=source,
        )
    except subprocess.TimeoutExpired as exc:
        _update_run(run_dir, status="conversion_timeout")
        raise TrainingError(f"adapter conversion timed out; see {log_path}") from exc
    except OSError as exc:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="conversion_failed")
        raise TrainingError(f"adapter converter could not start; see {log_path}") from exc
    except BaseException:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="conversion_failed")
        raise
    if return_code != 0:
        _update_run(run_dir, status="conversion_failed", return_code=return_code)
        raise TrainingError(f"adapter conversion exited with code {return_code}; see {log_path}")
    try:
        return _complete_conversion_run(
            run_dir,
            record,
            artifacts,
            output,
            log_path,
            environment,
        )
    except OSError as exc:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="conversion_failed")
        raise TrainingError(f"converted adapter could not be validated; see {log_path}") from exc
    except BaseException:
        with contextlib.suppress(BaseException):
            _update_run(run_dir, status="conversion_failed")
        raise


def _complete_conversion_run(
    run_dir: Path,
    record: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    output: Path,
    log_path: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    magic = b""
    if output.is_file():
        with output.open("rb") as handle:
            magic = handle.read(4)
    if not output.is_file() or output.stat().st_size < 16 or magic != b"GGUF":
        raise TrainingError(f"converter did not produce a valid GGUF adapter; see {log_path}")
    converted = {
        "path": str(output),
        "sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
        "format": "gguf-f16-lora",
        "llama_cpp_release": LLAMA_CPP_RELEASE,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "converter_environment": dict(environment),
    }
    updated_artifacts = dict(artifacts)
    updated_artifacts["gguf_adapter"] = converted
    logs = dict(_mapping(record.get("logs"), "run logs"))
    logs["converter"] = str(log_path)
    return _update_run(
        run_dir,
        status="converted",
        converted_at=_utc_now(),
        artifacts=updated_artifacts,
        logs=logs,
    )


def evaluate_training_run(
    run: str | Path | None = None,
    *,
    port: int = DEFAULT_EVALUATION_PORT,
    startup_timeout: float = DEFAULT_EVALUATION_TIMEOUT,
) -> dict[str, Any]:
    """Serialize evaluation attempts for one run."""

    root = training_root()
    run_dir, _record = _resolve_run(root, run)
    with _operation_lock(root, "workload"):
        with _operation_lock(run_dir, "lifecycle"):
            return _evaluate_training_run_locked(
                run_dir,
                port=port,
                startup_timeout=startup_timeout,
            )


def _evaluate_training_run_locked(
    run: str | Path,
    *,
    port: int,
    startup_timeout: float,
) -> dict[str, Any]:
    """Compare the exact Q4 baseline and candidate adapter on frozen holdout rows."""

    if not 1 <= port <= 65535:
        raise TrainingError("evaluation port must be between 1 and 65535")
    if not 1 <= startup_timeout <= 3600:
        raise TrainingError("evaluation startup timeout must be between 1 and 3600 seconds")
    root = training_root()
    run_dir, record = _resolve_run(root, run)
    status = record.get("status")
    if status == "evaluated":
        existing_evaluation = _mapping(record.get("evaluation"), "run evaluation")
        existing_path = _path_under(
            run_dir,
            existing_evaluation.get("path"),
            "evaluation path",
        )
        _verify_artifact(
            existing_path,
            str(existing_evaluation.get("sha256") or ""),
            label="evaluation report",
        )
        return record
    if status not in {
        "converted",
        "evaluating",
        "evaluation_failed",
    }:
        raise TrainingError(f"run {run_dir.name} is not ready for evaluation")
    bundle_dir, bundle_manifest = _resolve_bundle(root, record.get("bundle"))
    bundle_manifest_sha256 = _sha256_file(bundle_dir / "manifest.json")
    if record.get("bundle_manifest_sha256") != bundle_manifest_sha256:
        raise TrainingError("training run bundle manifest changed after training")
    holdout_path = bundle_dir / "holdout.jsonl"
    file_info = _mapping(
        _mapping(bundle_manifest.get("files"), "bundle files").get("holdout.jsonl"),
        "holdout file",
    )
    _verify_artifact(holdout_path, str(file_info.get("sha256") or ""), label="holdout")
    manifest_examples = _positive_int(file_info.get("examples"), "holdout manifest examples")
    rows, scanned_examples = _read_holdout_rows(holdout_path)
    if not rows:
        raise TrainingError("the selected bundle has no holdout rows")
    if scanned_examples != manifest_examples:
        raise TrainingError("holdout manifest example count is inconsistent")

    artifacts = _mapping(record.get("artifacts"), "run artifacts")
    gguf = _mapping(artifacts.get("gguf_adapter"), "GGUF adapter artifact")
    adapter_path = _path_under(run_dir, gguf.get("path"), "GGUF adapter path")
    adapter_sha256 = str(gguf.get("sha256") or "")
    _verify_artifact(adapter_path, adapter_sha256, label="GGUF adapter")
    base_path = ensure_gguf_base()
    executable = _ensure_pinned_llama()
    command = build_server_command(
        executable,
        model=DEFAULT_MODEL,
        host=DEFAULT_HOST,
        port=port,
        model_path=base_path,
        model_sha256=DEFAULT_GGUF_SHA256,
        lora_adapter_path=adapter_path,
        lora_adapter_sha256=adapter_sha256,
        lora_init_without_apply=True,
    )
    log_path = run_dir / "evaluation-server.log"
    env = _subprocess_environment(root)
    process: subprocess.Popen[bytes] | None = None
    process_job: Any = None
    if _server_port_is_open(DEFAULT_HOST, port):
        raise TrainingError(f"evaluation port {port} is already in use")
    _update_run(run_dir, status="evaluating")
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            options: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(_supervised_command(command), **options)
            process_job = _attach_windows_kill_job(process)
            probe = _wait_for_ephemeral_server(process, port, startup_timeout)
            verify_loaded_adapter(port, adapter_path, expected_scale=0.0)
            model = probe.choose_model(None)
            if model is None:
                raise TrainingError("evaluation server exposed no model")
            report = _evaluate_rows(
                rows,
                probe.base_url,
                model,
                expected_tasks={task.key for task in list_tasks()},
            )
    except (OSError, LlamaServerError, LocalAPIError, TrainingError) as exc:
        _update_run(run_dir, status="evaluation_failed")
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(f"candidate evaluation failed; see {log_path}") from exc
    finally:
        if process is not None:
            _stop_spawned_process(process)
        _close_windows_job(process_job)

    experimental = bool(record.get("experimental")) or not bool(
        _mapping(bundle_manifest.get("promotion"), "bundle promotion").get("promotable")
    )
    report.update(
        {
            "format_version": EVALUATION_FORMAT_VERSION,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "plugin_version": __version__,
            "run_id": run_dir.name,
            "evaluated_at": _utc_now(),
            "base_model": {
                "repository": DEFAULT_GGUF_REPOSITORY,
                "revision": DEFAULT_GGUF_REVISION,
                "filename": DEFAULT_GGUF_FILENAME,
                "sha256": DEFAULT_GGUF_SHA256,
            },
            "adapter": {"sha256": adapter_sha256},
            "llama_cpp": {
                "release": LLAMA_CPP_RELEASE,
                "commit": LLAMA_CPP_COMMIT,
                "executable": executable,
            },
            "holdout": {
                "sha256": str(file_info["sha256"]),
                "manifest_examples": manifest_examples,
                "scanned_examples": scanned_examples,
                "sampled_examples": len(rows),
                "sample_limit": MAX_EVALUATION_EXAMPLES,
            },
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "task_contract_hashes": dict(
                _mapping(
                    bundle_manifest.get("task_contract_hashes"),
                    "bundle task contracts",
                )
            ),
            "experimental": experimental,
            "promotion_eligible": report["quality_passed"] and not experimental,
        }
    )
    evaluation_path = run_dir / "evaluation.json"
    report_bytes = len(
        json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    )
    if report_bytes > MAX_JSON_BYTES:
        _update_run(run_dir, status="evaluation_failed")
        raise TrainingError("evaluation report exceeded the safe size limit")
    _write_json_atomic(evaluation_path, report)
    logs = dict(_mapping(record.get("logs"), "run logs"))
    logs["evaluation_server"] = str(log_path)
    return _update_run(
        run_dir,
        status="evaluated",
        evaluated_at=report["evaluated_at"],
        evaluation={
            "path": str(evaluation_path),
            "sha256": _sha256_file(evaluation_path),
            "quality_passed": report["quality_passed"],
            "promotion_eligible": report["promotion_eligible"],
        },
        logs=logs,
    )


def promote_training_run(run: str | Path | None = None) -> dict[str, Any]:
    """Serialize deployment-pointer transactions."""

    root = training_root()
    run_dir, _record = _resolve_run(root, run)
    with _operation_lock(root, "workload"):
        with _operation_lock(run_dir, "lifecycle"):
            with _operation_lock(root, "deployment"):
                return _promote_training_run_locked(run_dir)


def _promote_training_run_locked(run: str | Path | None = None) -> dict[str, Any]:
    """Atomically select one passing candidate and restart a managed server if needed."""

    root = training_root()
    run_dir, record = _resolve_run(root, run)
    validated = _validate_promotable_run(root, run_dir, record)
    evaluation_path = validated["evaluation_path"]
    adapter_path = validated["adapter_path"]
    adapter_sha256 = validated["adapter_sha256"]
    bundle_manifest_sha256 = validated["bundle_manifest_sha256"]
    bundle_contracts = validated["task_contract_hashes"]
    base_path = ensure_gguf_base()
    entry = {
        "run_id": run_dir.name,
        "promoted_at": _utc_now(),
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha256,
        "base_model_path": str(base_path),
        "base_model_sha256": DEFAULT_GGUF_SHA256,
        "base_model_repository": DEFAULT_GGUF_REPOSITORY,
        "base_model_revision": DEFAULT_GGUF_REVISION,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": validated["evaluation_sha256"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "task_contract_hashes": bundle_contracts,
    }
    before = _read_deployment(root, required=False) or _empty_deployment()
    before_active = before.get("active")
    if isinstance(before_active, Mapping) and before_active.get("run_id") == run_dir.name:
        raise TrainingError(f"run {run_dir.name} is already active")
    prior_active = before_active
    history = list(before.get("history") or [])
    if prior_active is not None:
        history.insert(0, prior_active)
    after = {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "active": entry,
        "history": history[:20],
        "updated_at": _utc_now(),
    }
    server_target = _managed_server_restart_target()
    _write_deployment(root, after)
    try:
        restarted = _restart_managed_server(entry, server_target)
    except BaseException as exc:
        _write_deployment(root, before)
        with contextlib.suppress(BaseException):
            _restart_managed_server(
                prior_active,
                server_target,
                restore_original=True,
            )
        raise TrainingError("promotion failed; the prior deployment pointer was restored") from exc
    _update_run(run_dir, promoted_at=entry["promoted_at"])
    return {**after, "managed_server_restarted": restarted}


def rollback_training_deployment() -> dict[str, Any]:
    """Serialize deployment-pointer transactions."""

    root = training_root()
    with _operation_lock(root, "workload"):
        with _operation_lock(root, "deployment"):
            return _rollback_training_deployment_locked()


def _rollback_training_deployment_locked() -> dict[str, Any]:
    """Restore the previous verified deployment, or the unchanged base model."""

    root = training_root()
    before = _read_deployment(root, required=True, validate_active=False)
    history = list(before.get("history") or [])
    next_active = None
    skipped_invalid_history = 0
    while history:
        candidate = history.pop(0)
        try:
            _validate_deployment_entry(root, candidate)
        except TrainingError:
            skipped_invalid_history += 1
            continue
        next_active = candidate
        break
    after = {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "active": next_active,
        "history": history,
        "updated_at": _utc_now(),
    }
    server_target = _managed_server_restart_target()
    _write_deployment(root, after)
    try:
        restarted = _restart_managed_server(next_active, server_target)
    except BaseException as exc:
        _write_deployment(root, before)
        with contextlib.suppress(BaseException):
            _restart_managed_server(
                before.get("active"),
                server_target,
                restore_original=True,
            )
        raise TrainingError(
            "rollback failed; the previous deployment pointer was restored"
        ) from exc
    return {
        **after,
        "managed_server_restarted": restarted,
        "skipped_invalid_history": skipped_invalid_history,
    }


def active_deployment_artifacts() -> dict[str, Any] | None:
    """Return the verified active adapter/base pair for managed server startup."""

    root = training_root()
    deployment = _read_deployment(root, required=False)
    if not deployment:
        return None
    if deployment.get("active") is None:
        base = ensure_gguf_base()
        return {
            "base_model_path": str(base),
            "base_model_sha256": DEFAULT_GGUF_SHA256,
            "adapter_path": None,
            "adapter_sha256": None,
        }
    active = _mapping(deployment["active"], "active deployment")
    _validate_deployment_entry(root, active)
    return dict(active)


def read_training_logs(
    run: str | Path | None = None,
    *,
    stage: str = "trainer",
    lines: int = 100,
) -> str:
    """Return a bounded tail from one known run log."""

    if stage not in {"trainer", "converter", "evaluation_server"}:
        raise TrainingError("log stage must be trainer, converter, or evaluation_server")
    if not 1 <= lines <= 10_000:
        raise TrainingError("log lines must be between 1 and 10000")
    root = training_root()
    run_dir, record = _resolve_run(root, run)
    logs = _mapping(record.get("logs"), "run logs")
    value = logs.get(stage)
    if not value:
        raise TrainingError(f"run {run_dir.name} has no {stage} log")
    path = _path_under(run_dir, value, f"{stage} log")
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - MAX_LOG_BYTES))
            text = handle.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
    except OSError as exc:
        raise TrainingError(f"cannot read {stage} log: {exc}") from exc
    return "\n".join(text.splitlines()[-lines:])


def _evaluate_rows(
    rows: list[dict[str, Any]],
    base_url: str,
    model: str,
    *,
    expected_tasks: set[str] | None = None,
    timeout_seconds: float = DEFAULT_EVALUATION_RUN_TIMEOUT,
) -> dict[str, Any]:
    client = OpenAICompatibleClient(base_url, timeout=EVALUATION_REQUEST_TIMEOUT)
    deadline = time.monotonic() + timeout_seconds
    scores = {"baseline": _empty_score(), "candidate": _empty_score()}
    variants = {
        "baseline": [{"id": 0, "scale": 0.0}],
        "candidate": [{"id": 0, "scale": 1.0}],
    }
    for row in rows:
        task = get_task(row["task"])
        expected = _without_confidence(row["expected"])
        for name, lora in variants.items():
            if time.monotonic() >= deadline:
                raise TrainingError("evaluation exceeded its bounded runtime")
            score = scores[name]
            task_score = score["by_task"].setdefault(task.key, _empty_score(include_tasks=False))
            score["examples"] += 1
            task_score["examples"] += 1
            try:
                raw = _task_completion(
                    client,
                    task,
                    row["messages"],
                    model=model,
                    extra_body={"lora": lora},
                )
                actual = _without_confidence(task.parse(raw))
            except (LocalAPIError, TaskParseError, ValueError):
                failure = {"prediction_id": row["prediction_id"], "task": task.key}
                _record_evaluation_failure(score, failure)
                _record_evaluation_failure(task_score, failure)
                continue
            _score_result(score, expected, actual)
            _score_result(task_score, expected, actual)
    finalized = {name: _finalize_score(score) for name, score in scores.items()}
    baseline = finalized["baseline"]
    candidate = finalized["candidate"]
    all_tasks_covered = expected_tasks is None or set(candidate["by_task"]) == expected_tasks
    per_task_pass = all(
        _score_not_worse(
            baseline["by_task"][task],
            candidate["by_task"][task],
            require_complete=True,
        )
        for task in sorted(baseline["by_task"])
    )
    quality_passed = (
        _score_not_worse(baseline, candidate, require_complete=True)
        and per_task_pass
        and all_tasks_covered
    )
    return {
        **finalized,
        "quality_gate": {
            "candidate_schema_valid_for_all": candidate["completed"] == candidate["examples"],
            "no_overall_exact_regression": (
                candidate["exact_accuracy"] >= baseline["exact_accuracy"]
            ),
            "no_overall_field_regression": (
                candidate["field_accuracy"] >= baseline["field_accuracy"]
            ),
            "no_per_task_regression": per_task_pass,
            "all_tasks_covered": all_tasks_covered,
        },
        "quality_passed": quality_passed,
    }


def _task_completion(
    client: OpenAICompatibleClient,
    task: Any,
    messages: list[dict[str, str]],
    *,
    model: str,
    extra_body: Mapping[str, Any],
) -> str:
    try:
        raw = client.complete(
            messages,
            model=model,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            response_format=task.response_format,
            extra_body=extra_body,
        )
    except LocalAPIError as exc:
        if not any(code in str(exc) for code in ("HTTP 400", "HTTP 404", "HTTP 422")):
            raise
        raw = client.complete(
            messages,
            model=model,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            extra_body=extra_body,
        )
    try:
        task.parse(raw)
        return raw
    except TaskParseError as first_error:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw[:4_000]},
            {
                "role": "user",
                "content": (
                    "The previous output did not satisfy the JSON schema "
                    f"({first_error}). Return one corrected JSON object only."
                ),
            },
        ]
        return client.complete(
            repair_messages,
            model=model,
            temperature=0.0,
            max_tokens=task.max_tokens,
            extra_body=extra_body,
        )


def _empty_score(*, include_tasks: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "examples": 0,
        "completed": 0,
        "exact_matches": 0,
        "matching_fields": 0,
        "fields": 0,
        "failures": [],
        "failure_count": 0,
    }
    if include_tasks:
        value["by_task"] = {}
    return value


def _score_result(score: dict[str, Any], expected: dict[str, Any], actual: dict[str, Any]) -> None:
    score["completed"] += 1
    if actual == expected:
        score["exact_matches"] += 1
    for key, value in expected.items():
        score["fields"] += 1
        if actual.get(key) == value:
            score["matching_fields"] += 1


def _record_evaluation_failure(score: dict[str, Any], failure: dict[str, str]) -> None:
    score["failure_count"] += 1
    if len(score["failures"]) < MAX_EVALUATION_FAILURES:
        score["failures"].append(failure)


def _finalize_score(score: dict[str, Any]) -> dict[str, Any]:
    examples = score["examples"]
    fields = score["fields"]
    value = {
        "examples": examples,
        "completed": score["completed"],
        "exact_matches": score["exact_matches"],
        "exact_accuracy": round(score["exact_matches"] / examples, 6) if examples else 0.0,
        "field_accuracy": round(score["matching_fields"] / fields, 6) if fields else 0.0,
        "failures": score["failures"],
        "failure_count": score["failure_count"],
        "omitted_failures": score["failure_count"] - len(score["failures"]),
    }
    if "by_task" in score:
        value["by_task"] = {
            task: _finalize_score(task_score)
            for task, task_score in sorted(score["by_task"].items())
        }
    return value


def _score_not_worse(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    require_complete: bool,
) -> bool:
    return (
        (not require_complete or candidate["completed"] == candidate["examples"])
        and candidate["completed"] >= baseline["completed"]
        and candidate["exact_accuracy"] >= baseline["exact_accuracy"]
        and candidate["field_accuracy"] >= baseline["field_accuracy"]
    )


def _read_holdout_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    ranked: list[tuple[str, dict[str, Any]]] = []
    per_task: dict[str, tuple[str, dict[str, Any]]] = {}
    total = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                total = number
                if len(line) > 1_000_000:
                    raise TrainingError(f"holdout row {number} exceeds 1000000 characters")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TrainingError(f"holdout row {number} is not an object")
                messages = value.get("messages")
                metadata = value.get("metadata")
                if (
                    not isinstance(messages, list)
                    or len(messages) < 3
                    or not isinstance(metadata, dict)
                ):
                    raise TrainingError(f"holdout row {number} has an invalid shape")
                assistant = messages[-1]
                if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
                    raise TrainingError(f"holdout row {number} lacks an assistant target")
                task_key = str(metadata.get("task") or "")
                prediction_id = str(metadata.get("prediction_id") or "")
                task = get_task(task_key)
                expected = task.parse(str(assistant.get("content") or ""))
                prompt: list[dict[str, str]] = []
                for message in messages[:-1]:
                    if not isinstance(message, dict):
                        raise TrainingError(f"holdout row {number} contains an invalid message")
                    role = message.get("role")
                    content = message.get("content")
                    if role not in {"system", "user"} or not isinstance(content, str):
                        raise TrainingError(f"holdout row {number} contains an invalid message")
                    prompt.append({"role": role, "content": content})
                if not prediction_id:
                    raise TrainingError(f"holdout row {number} has no prediction id")
                row = {
                    "task": task_key,
                    "prediction_id": prediction_id,
                    "messages": prompt,
                    "expected": expected,
                }
                rank = hashlib.sha256(f"{task_key}\0{prediction_id}".encode()).hexdigest()
                candidate = (rank, row)
                ranked.append(candidate)
                ranked.sort(key=lambda item: item[0])
                del ranked[MAX_EVALUATION_EXAMPLES:]
                current = per_task.get(task_key)
                if current is None or rank < current[0]:
                    per_task[task_key] = candidate
    except (OSError, json.JSONDecodeError, KeyError, TaskParseError) as exc:
        raise TrainingError(f"cannot read holdout bundle: {exc}") from exc
    required = sorted(per_task.values(), key=lambda item: item[0])
    required_ids = {(item[1]["task"], item[1]["prediction_id"]) for item in required}
    remaining = [
        item for item in ranked if (item[1]["task"], item[1]["prediction_id"]) not in required_ids
    ]
    selected = (required + remaining)[:MAX_EVALUATION_EXAMPLES]
    return [row for _rank, row in sorted(selected, key=lambda item: item[0])], total


def _wait_for_ephemeral_server(
    process: subprocess.Popen[bytes],
    port: int,
    timeout_seconds: float,
) -> Any:
    base_url = f"http://{DEFAULT_HOST}:{port}/v1"
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise TrainingError(f"evaluation server exited with code {code}")
        probe = probe_endpoint(base_url, timeout=1.0)
        if probe.reachable and probe.models:
            return probe
        last_error = probe.error or last_error
        time.sleep(0.25)
    raise TrainingError(f"evaluation server did not become ready: {last_error}")


def verify_loaded_adapter(
    port: int,
    adapter_path: Path,
    *,
    expected_scale: float,
) -> None:
    """Verify the exact local adapter path and global scale without proxies."""

    url = f"http://{DEFAULT_HOST}:{port}/lora-adapters"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with local_urlopen(request, timeout=3.0) as response:
            value = json.loads(response.read(64 * 1024).decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise TrainingError("cannot verify the adapter loaded by llama.cpp") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise TrainingError("llama.cpp did not report exactly one loaded adapter")
    reported = value[0].get("path")
    if not isinstance(reported, str) or Path(reported).resolve() != adapter_path.resolve():
        raise TrainingError("llama.cpp reported a different loaded adapter path")
    scale = value[0].get("scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or abs(float(scale) - expected_scale) > 1e-6
    ):
        raise TrainingError(
            f"llama.cpp reported adapter scale {scale!r}; expected {expected_scale:g}"
        )


def _stop_spawned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _supervised_command(command: Sequence[str]) -> list[str]:
    argv = list(command)
    if sys.platform.startswith("linux"):
        return [
            sys.executable,
            "-c",
            _LINUX_PARENT_DEATH_WRAPPER,
            str(os.getpid()),
            *argv,
        ]
    return argv


def _attach_windows_kill_job(process: subprocess.Popen[bytes]) -> Any:
    if os.name != "nt" or not hasattr(process, "_handle"):
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        _stop_spawned_process(process)
        raise TrainingError("cannot create a kill-on-parent Windows job")
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    configured = kernel32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        handle,
        wintypes.HANDLE(process._handle),
    )
    if not assigned:
        kernel32.CloseHandle(handle)
        _stop_spawned_process(process)
        raise TrainingError("cannot supervise the Windows child process")
    return handle


def _close_windows_job(handle: Any) -> None:
    if os.name == "nt" and handle:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)


def _ensure_pinned_llama() -> str:
    """Use only the profile-installed pinned runtime for promotion evaluation."""

    return install_llama_cpp().path


def _managed_server_restart_target() -> dict[str, Any] | None:
    status = get_llama_server_status()
    if not status.running:
        if status.pid is not None and not status.identity_verified:
            raise TrainingError("cannot restart an unverified managed llama.cpp process")
        return None
    if not status.identity_verified or status.executable is None:
        raise TrainingError("cannot restart an unverified managed llama.cpp process")
    if status.model != DEFAULT_MODEL:
        raise TrainingError(
            "the managed server must use the default LFM model before deployment changes"
        )
    return {
        "original_executable": status.executable,
        "deployment_executable": _ensure_pinned_llama(),
        "host": status.host,
        "port": status.port,
        "pid": status.pid,
        "started_at": status.started_at,
        "model": status.model,
        "command": list(status.command),
        "model_path": status.model_path,
        "model_sha256": status.model_sha256,
        "lora_adapter_path": status.lora_adapter_path,
        "lora_adapter_sha256": status.lora_adapter_sha256,
    }


def _restart_managed_server(
    active: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    *,
    restore_original: bool = False,
) -> bool:
    if target is None:
        return False
    base_path = ensure_gguf_base()
    adapter_path = None
    adapter_sha256 = None
    if active is not None:
        _validate_deployment_entry(training_root(), active)
        adapter_path = str(active["adapter_path"])
        adapter_sha256 = str(active["adapter_sha256"])
    executable_key = "original_executable" if restore_original else "deployment_executable"
    executable = str(target[executable_key])
    host = str(target["host"])
    port = int(target["port"])
    status = get_llama_server_status()
    if status.running:
        if not _server_matches_restart_target(status, target):
            raise TrainingError("managed llama.cpp changed during the deployment transaction")
        stop_llama_server(timeout_seconds=10.0)
    elif not restore_original:
        raise TrainingError("managed llama.cpp stopped during the deployment transaction")
    started = start_llama_server(
        executable=executable,
        install_if_missing=False,
        model=DEFAULT_MODEL,
        host=host,
        port=port,
        model_path=base_path,
        model_sha256=DEFAULT_GGUF_SHA256,
        lora_adapter_path=adapter_path,
        lora_adapter_sha256=adapter_sha256,
        wait_ready_seconds=0.0,
    )
    try:
        wait_for_llama_server(timeout_seconds=DEFAULT_EVALUATION_TIMEOUT)
        if adapter_path is not None:
            verify_loaded_adapter(port, Path(adapter_path), expected_scale=1.0)
    except BaseException:
        current = get_llama_server_status()
        if _same_managed_server(current, started):
            with contextlib.suppress(BaseException):
                stop_llama_server(timeout_seconds=10.0)
        raise
    return True


def _server_matches_restart_target(status: Any, target: Mapping[str, Any]) -> bool:
    executable = target.get("original_executable")
    return bool(
        status.identity_verified
        and status.executable is not None
        and executable is not None
        and Path(status.executable).resolve() == Path(str(executable)).resolve()
        and status.host == target.get("host")
        and status.port == target.get("port")
        and status.pid == target.get("pid")
        and status.started_at == target.get("started_at")
        and status.model == target.get("model")
        and tuple(status.command) == tuple(target.get("command") or ())
        and status.model_path == target.get("model_path")
        and status.model_sha256 == target.get("model_sha256")
        and status.lora_adapter_path == target.get("lora_adapter_path")
        and status.lora_adapter_sha256 == target.get("lora_adapter_sha256")
    )


def _same_managed_server(left: Any, right: Any) -> bool:
    return bool(
        left.running
        and right is not None
        and left.identity_verified
        and right.identity_verified
        and left.pid == right.pid
        and left.started_at == right.started_at
        and left.executable == right.executable
        and tuple(left.command) == tuple(right.command)
    )


def _empty_deployment() -> dict[str, Any]:
    return {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "active": None,
        "history": [],
        "updated_at": _utc_now(),
    }


def _read_deployment(
    root: Path,
    *,
    required: bool,
    validate_active: bool = True,
) -> dict[str, Any] | None:
    path = root / "deployment.json"
    if not path.is_file():
        if required:
            raise TrainingError("there is no deployment to roll back")
        return None
    value = _read_json(path, label="deployment")
    if value.get("format_version") != DEPLOYMENT_FORMAT_VERSION:
        raise TrainingError("unsupported training deployment format")
    active = value.get("active")
    if active is not None:
        active = _mapping(active, "active deployment")
        if validate_active:
            _validate_deployment_entry(root, active)
    history = value.get("history")
    if not isinstance(history, list) or len(history) > 20:
        raise TrainingError("deployment history is invalid")
    for entry in history:
        if not isinstance(entry, Mapping):
            raise TrainingError("deployment history contains an invalid entry")
    return value


def _write_deployment(root: Path, value: Mapping[str, Any]) -> None:
    _ensure_private_directory(root)
    _write_json_atomic(root / "deployment.json", dict(value))


def _validate_promotable_run(
    root: Path,
    run_dir: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if record.get("status") != "evaluated":
        raise TrainingError(f"run {run_dir.name} has not completed evaluation")
    evaluation = _mapping(record.get("evaluation"), "run evaluation")
    if (
        evaluation.get("promotion_eligible") is not True
        or evaluation.get("quality_passed") is not True
        or record.get("experimental") is not False
    ):
        raise TrainingError("candidate is not promotion eligible")
    bundle_dir, bundle_manifest = _resolve_bundle(root, record.get("bundle"))
    bundle_sha256 = _sha256_file(bundle_dir / "manifest.json")
    if record.get("bundle_manifest_sha256") != bundle_sha256:
        raise TrainingError("training run bundle manifest changed after training")
    contracts = _validate_task_contract_hashes(
        bundle_manifest.get("task_contract_hashes"),
        label="training bundle task contracts",
    )
    if set(contracts) != {task.key for task in list_tasks()}:
        raise TrainingError("promotion requires current holdout coverage for every built-in task")
    promotion = _mapping(bundle_manifest.get("promotion"), "bundle promotion")
    if promotion.get("promotable") is not True or promotion.get("experimental") is not False:
        raise TrainingError("training bundle is not promotion eligible")
    evaluation_path = _path_under(run_dir, evaluation.get("path"), "evaluation path")
    evaluation_sha256 = str(evaluation.get("sha256") or "")
    _verify_artifact(evaluation_path, evaluation_sha256, label="evaluation report")
    report = _read_json(evaluation_path, label="evaluation report")
    artifacts = _mapping(record.get("artifacts"), "run artifacts")
    adapter = _mapping(artifacts.get("gguf_adapter"), "GGUF adapter artifact")
    adapter_path = _path_under(run_dir, adapter.get("path"), "GGUF adapter path")
    adapter_sha256 = str(adapter.get("sha256") or "")
    _verify_artifact(adapter_path, adapter_sha256, label="GGUF adapter")
    holdout = _mapping(
        _mapping(bundle_manifest.get("files"), "bundle files").get("holdout.jsonl"),
        "holdout file",
    )
    quality_gate = _mapping(report.get("quality_gate"), "evaluation quality gate")
    bindings_valid = (
        report.get("format_version") == EVALUATION_FORMAT_VERSION
        and report.get("evaluation_contract_version") == EVALUATION_CONTRACT_VERSION
        and isinstance(report.get("plugin_version"), str)
        and report.get("run_id") == run_dir.name
        and report.get("bundle_manifest_sha256") == bundle_sha256
        and report.get("task_contract_hashes") == contracts
        and report.get("quality_passed") is True
        and report.get("promotion_eligible") is True
        and report.get("experimental") is False
        and set(quality_gate) == EXPECTED_QUALITY_GATE_KEYS
        and all(value is True for value in quality_gate.values())
        and _mapping(report.get("adapter"), "evaluation adapter").get("sha256") == adapter_sha256
        and _mapping(report.get("base_model"), "evaluation base model").get("sha256")
        == DEFAULT_GGUF_SHA256
        and _mapping(report.get("llama_cpp"), "evaluation llama.cpp").get("release")
        == LLAMA_CPP_RELEASE
        and _mapping(report.get("llama_cpp"), "evaluation llama.cpp").get("commit")
        == LLAMA_CPP_COMMIT
        and _mapping(report.get("holdout"), "evaluation holdout").get("sha256")
        == holdout.get("sha256")
    )
    if not bindings_valid:
        raise TrainingError("evaluation report does not authorize this adapter and bundle")
    return {
        "adapter_path": adapter_path,
        "adapter_sha256": adapter_sha256,
        "evaluation_path": evaluation_path,
        "evaluation_sha256": evaluation_sha256,
        "bundle_manifest_sha256": bundle_sha256,
        "task_contract_hashes": contracts,
    }


def _validate_deployment_entry(root: Path, entry: Mapping[str, Any]) -> None:
    run_id = str(entry.get("run_id") or "")
    allowed_run_id = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not run_id or any(character not in allowed_run_id for character in run_id):
        raise TrainingError("deployment run id is invalid")
    run_dir = (root / "runs" / run_id).resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to((root / "runs").resolve()):
        raise TrainingError(f"deployment run does not exist: {run_id}")
    run_record = _read_json(run_dir / "run.json", label="deployment training run")
    validated = _validate_promotable_run(root, run_dir, run_record)
    if (
        entry.get("adapter_sha256") != validated["adapter_sha256"]
        or entry.get("evaluation_sha256") != validated["evaluation_sha256"]
        or entry.get("bundle_manifest_sha256") != validated["bundle_manifest_sha256"]
        or entry.get("task_contract_hashes") != validated["task_contract_hashes"]
        or Path(str(entry.get("adapter_path"))).resolve() != validated["adapter_path"]
        or Path(str(entry.get("evaluation_path"))).resolve() != validated["evaluation_path"]
    ):
        raise TrainingError("deployment does not match its evaluated training run")
    base = _path_under(root, entry.get("base_model_path"), "deployed base model")
    _verify_artifact(base, str(entry.get("base_model_sha256") or ""), label="deployed base")
    if (
        str(entry.get("base_model_sha256") or "").lower() != DEFAULT_GGUF_SHA256
        or entry.get("base_model_repository") != DEFAULT_GGUF_REPOSITORY
        or entry.get("base_model_revision") != DEFAULT_GGUF_REVISION
    ):
        raise TrainingError("deployment base model hash does not match this plugin version")


def _install_environment(
    root: Path,
    component: str,
    *,
    force: bool,
    python_executable: str | Path | None,
) -> dict[str, Any]:
    component = _validate_component(component)
    destination = root / "envs" / component
    existing = _environment_status(root, component)
    if existing["ready"] and not force:
        return existing
    base_python = Path(python_executable or sys.executable).expanduser().resolve()
    if not base_python.is_file():
        raise TrainingError(f"Python executable does not exist: {base_python}")
    _validate_base_python(base_python, root)
    _ensure_private_directory(destination.parent)
    stage = destination.parent / f".{component}-{uuid.uuid4().hex}"
    log_path = root / "logs" / f"install-{component}.log"
    _ensure_private_directory(log_path.parent)
    try:
        create_code = _run_logged(
            [str(base_python), "-m", "venv", str(stage)],
            log_path,
            env=_subprocess_environment(root),
            timeout_seconds=600,
        )
        if create_code != 0:
            raise TrainingError(
                f"cannot create {component} environment (venv exit {create_code}); see {log_path}"
            )
        python = _environment_python(stage)
        if not python.is_file():
            raise TrainingError(f"Python did not create a usable environment; see {log_path}")
        if component == "trainer":
            nvidia = _nvidia_available()
            torch_command = [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--prefer-binary",
            ]
            if nvidia:
                torch_command.extend(["--index-url", TRAINER_CUDA_INDEX])
            else:
                torch_command.extend(["--index-url", PYPI_INDEX])
            torch_command.append(TRAINER_TORCH_REQUIREMENT)
            torch_code = _run_logged(
                torch_command,
                log_path,
                env=_subprocess_environment(root),
                timeout_seconds=3600,
            )
            if torch_code != 0:
                raise TrainingError(
                    f"cannot install trainer environment (torch exit {torch_code}); see {log_path}"
                )
            command = [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--prefer-binary",
                "--index-url",
                PYPI_INDEX,
                *TRAINER_REQUIREMENTS,
            ]
            lock_text = _trainer_environment_lock(nvidia)
        else:
            requirements = (
                _llama_source_path(root) / "requirements" / "requirements-convert_lora_to_gguf.txt"
            )
            if not requirements.is_file():
                raise TrainingError("pinned llama.cpp converter requirements are missing")
            command = [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--prefer-binary",
                "--index-url",
                PYPI_INDEX,
                "-r",
                str(requirements),
            ]
            lock_text = _converter_environment_lock(root)
        code = _run_logged(
            command,
            log_path,
            env=_subprocess_environment(root),
            timeout_seconds=3600,
        )
        if code != 0:
            raise TrainingError(
                f"cannot install {component} environment (pip exit {code}); see {log_path}"
            )
        accelerator = None
        if component == "trainer":
            accelerator = _trainer_accelerator(python)
            if nvidia and accelerator.get("cuda_available") is not True:
                raise TrainingError(
                    "the NVIDIA GPU was detected but the isolated Torch build cannot use CUDA; "
                    f"see {log_path}"
                )
        freeze = subprocess.run(
            [str(python), "-m", "pip", "--isolated", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            env=_subprocess_environment(root),
            timeout=120,
        ).stdout
        manifest = {
            "format_version": 1,
            "component": component,
            "created_at": _utc_now(),
            "base_python": str(base_python),
            "python": str(_environment_python(destination)),
            "requirements_sha256": hashlib.sha256(lock_text.encode()).hexdigest(),
            "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "freeze": freeze.splitlines(),
        }
        if accelerator is not None:
            manifest["accelerator"] = accelerator
        _write_json_atomic(stage / "environment.json", manifest)
        if destination.exists():
            _safe_remove_tree(destination, destination.parent)
        os.replace(stage, destination)
        return _environment_status(root, component)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrainingError(f"cannot create {component} environment: {exc}") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _environment_status(root: Path, component: str) -> dict[str, Any]:
    destination = root / "envs" / component
    python = _environment_python(destination)
    manifest_path = destination / "environment.json"
    result: dict[str, Any] = {
        "ready": False,
        "path": str(destination),
        "python": str(python),
        "manifest": str(manifest_path),
    }
    if not python.is_file() or not manifest_path.is_file():
        return result
    try:
        manifest = _read_json(manifest_path, label=f"{component} environment")
    except TrainingError as exc:
        result["error"] = str(exc)
        return result
    if manifest.get("component") != component or manifest.get("format_version") != 1:
        result["error"] = "environment manifest does not match"
        return result
    requirements_sha256 = manifest.get("requirements_sha256")
    if requirements_sha256 not in _current_environment_requirement_hashes(root, component):
        result["error"] = "environment pins changed; reinstall this training component"
        return result
    result.update(
        {
            "ready": True,
            "requirements_sha256": requirements_sha256,
            "freeze_sha256": manifest.get("freeze_sha256"),
            "created_at": manifest.get("created_at"),
            "accelerator": manifest.get("accelerator"),
        }
    )
    return result


def _require_environment(root: Path, component: str) -> dict[str, Any]:
    status = _environment_status(root, component)
    if not status["ready"]:
        raise TrainingError(
            f"{component} environment is not installed; "
            f"run `hermes brain train install {component}`"
        )
    return status


def _trainer_environment_lock(nvidia: bool) -> str:
    return (
        "\n".join(
            (
                TRAINER_TORCH_REQUIREMENT,
                f"torch_index={TRAINER_CUDA_INDEX if nvidia else 'pypi'}",
                *TRAINER_REQUIREMENTS,
            )
        )
        + "\n"
    )


def _current_environment_requirement_hashes(root: Path, component: str) -> set[str]:
    if component == "trainer":
        return {
            hashlib.sha256(_trainer_environment_lock(nvidia).encode()).hexdigest()
            for nvidia in (False, True)
        }
    try:
        text = _converter_environment_lock(root)
    except OSError:
        return set()
    return {hashlib.sha256(text.encode()).hexdigest()}


def _converter_environment_lock(root: Path) -> str:
    requirements_root = _llama_source_path(root) / "requirements"
    files = sorted(requirements_root.glob("requirements-convert*.txt"))
    if not files:
        raise OSError("pinned llama.cpp converter requirements are missing")
    return "".join(f"[{path.name}]\n{path.read_text(encoding='utf-8')}\n" for path in files)


def _environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _validate_base_python(python: Path, root: Path) -> None:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_subprocess_environment(root),
        )
        major, minor = (int(part) for part in result.stdout.strip().split(".", 1))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise TrainingError(f"cannot verify Python executable {python}") from exc
    if (major, minor) < (3, 11):
        raise TrainingError("training environments require Python 3.11 or newer")


def _validate_component(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in COMPONENTS:
        raise TrainingError("training component must be trainer or converter")
    return normalized


def _nvidia_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=_subprocess_environment(training_root()),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _trainer_accelerator(python: Path) -> dict[str, Any]:
    script = (
        "import json,torch; "
        "mps=getattr(getattr(torch,'backends',None),'mps',None); "
        "print(json.dumps({'torch':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),"
        "'cuda_version':torch.version.cuda,"
        "'mps_available':bool(mps and mps.is_available())}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env={**_subprocess_environment(training_root()), "PYTHONNOUSERSITE": "1"},
        )
        value = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot verify the isolated Torch accelerator") from exc
    if not isinstance(value, dict):
        raise TrainingError("isolated Torch accelerator probe returned invalid data")
    return value


def _resolve_bundle(root: Path, value: str | Path | None) -> tuple[Path, dict[str, Any]]:
    bundles_root = (root / "bundles").resolve()
    if value is None:
        candidates = sorted(
            (item for item in bundles_root.glob("bundle-*") if item.is_dir()),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        )
        if not candidates:
            raise TrainingError("no training bundle exists; run `hermes brain train prepare`")
        path = candidates[-1].resolve()
    else:
        raw = Path(value).expanduser()
        path = (bundles_root / raw if len(raw.parts) == 1 else raw).resolve()
    if not path.is_dir() or not path.is_relative_to(bundles_root):
        raise TrainingError(f"training bundle is outside the active profile: {path}")
    manifest_path = path / "manifest.json"
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise TrainingError("training bundle manifest exceeds the size limit")
    except OSError as exc:
        raise TrainingError(
            f"cannot inspect training bundle manifest {manifest_path}: {exc}"
        ) from exc
    manifest = _read_json(manifest_path, label="training bundle manifest")
    manifest_sha256 = _sha256_file(manifest_path)
    if path.name != f"bundle-{manifest_sha256[:16]}":
        raise TrainingError("training bundle directory does not match its manifest hash")
    if manifest.get("format_version") != TRAINING_BUNDLE_FORMAT_VERSION:
        raise TrainingError("training bundle uses an unsupported format")
    if manifest.get("model") != {
        "id": DEFAULT_NATIVE_MODEL,
        "revision": DEFAULT_NATIVE_REVISION,
    }:
        raise TrainingError("training bundle uses a different native model revision")
    _validate_task_contract_hashes(
        manifest.get("task_contract_hashes"),
        label="training bundle task contracts",
    )
    files = _mapping(manifest.get("files"), "bundle files")
    if set(files) != {"train.jsonl", "holdout.jsonl"}:
        raise TrainingError("training bundle must contain train.jsonl and holdout.jsonl")
    for name, details in files.items():
        info = _mapping(details, f"bundle file {name}")
        _verify_artifact(path / str(name), str(info.get("sha256") or ""), label=str(name))
    return path, manifest


def _resolve_run(root: Path, value: str | Path | None) -> tuple[Path, dict[str, Any]]:
    runs_root = (root / "runs").resolve()
    if value is None:
        candidates = sorted(
            (item for item in runs_root.glob("run-*") if item.is_dir()),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        )
        if not candidates:
            raise TrainingError("no training run exists")
        path = candidates[-1].resolve()
    else:
        raw = Path(value).expanduser()
        path = (runs_root / raw if len(raw.parts) == 1 else raw).resolve()
    if not path.is_dir() or not path.is_relative_to(runs_root):
        raise TrainingError(f"training run is outside the active profile: {path}")
    record = _read_json(path / "run.json", label="training run")
    if record.get("format_version") != RUN_FORMAT_VERSION or record.get("run_id") != path.name:
        raise TrainingError(f"invalid training run record: {path}")
    return path, record


def _update_run(run_dir: Path, **changes: Any) -> dict[str, Any]:
    path = run_dir / "run.json"
    record = _read_json(path, label="training run")
    record.update(changes)
    record["updated_at"] = _utc_now()
    _write_json_atomic(path, record)
    return record


def _latest_record(root: Path, filename: str) -> dict[str, Any] | None:
    if not root.is_dir():
        return None
    candidates: list[tuple[int, str, Path]] = []
    for directory in root.iterdir():
        path = directory / filename
        if not directory.is_dir() or not path.is_file():
            continue
        try:
            candidates.append((path.stat().st_mtime_ns, directory.name, path))
        except OSError:
            continue
    for _mtime, _name, path in sorted(candidates, reverse=True):
        try:
            return _read_json(path, label=filename)
        except TrainingError:
            continue
    return None


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    env: Mapping[str, str],
    timeout_seconds: float | None,
    cwd: Path | None = None,
) -> int:
    if not command or not all(isinstance(item, str) and "\x00" not in item for item in command):
        raise TrainingError("subprocess command is invalid")
    _ensure_private_directory(log_path.parent)
    descriptor = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    if os.name != "nt":
        os.chmod(log_path, 0o600)
    process: subprocess.Popen[bytes] | None = None
    process_job: Any = None
    with os.fdopen(descriptor, "ab", buffering=0) as log:
        options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "shell": False,
            "env": dict(env),
            "cwd": cwd,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(_supervised_command(command), **options)
        process_job = _attach_windows_kill_job(process)
        try:
            return process.wait(timeout=timeout_seconds)
        except BaseException:
            _stop_spawned_process(process)
            raise
        finally:
            _close_windows_job(process_job)


def _subprocess_environment(root: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _SUBPROCESS_ENVIRONMENT_ALLOWLIST
    }
    environment.update(
        {
            "DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "HF_HOME": str(root / "cache" / "huggingface"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PIP_CACHE_DIR": str(root / "cache" / "pip"),
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        }
    )
    return environment


@contextlib.contextmanager
def _operation_lock(root: Path, name: str) -> Iterator[None]:
    _ensure_private_directory(root)
    path = root / f".{name}.lock"
    payload = json.dumps({"pid": os.getpid(), "created_at": _utc_now()}).encode()
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            break
        except FileExistsError:
            if attempt or not _remove_stale_lock(path):
                raise TrainingError(f"another training {name} operation is already running")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def _remove_stale_lock(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = value.get("pid")
    except (OSError, ValueError):
        return False
    if isinstance(pid, int) and pid > 0 and _process_is_alive(pid):
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if not url.startswith("https://"):
        raise TrainingError("training artifacts must use HTTPS")
    _ensure_private_directory(destination.parent)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"hermes-auxiliary-brain/{__version__}"},
    )
    try:
        descriptor = os.open(stage, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            with urllib.request.urlopen(request, timeout=60.0) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > min(MAX_DOWNLOAD_BYTES, expected_size + 1):
                        raise TrainingError("download exceeded its pinned size")
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        if size != expected_size:
            raise TrainingError(
                f"download size mismatch: expected {expected_size}, received {size}"
            )
        if digest.hexdigest() != expected_sha256:
            raise TrainingError("download SHA256 does not match the pinned artifact")
        os.replace(stage, destination)
    except (OSError, urllib.error.URLError) as exc:
        raise TrainingError(f"cannot download pinned training artifact: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            stage.unlink()


def _artifact_matches(path: Path, expected_sha256: str, expected_size: int | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return _sha256_file(path) == expected_sha256
    except OSError:
        return False


def _verify_artifact(path: Path, expected_sha256: str, *, label: str) -> None:
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise TrainingError(f"{label} has an invalid SHA256")
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f"{label} is missing or unsafe: {path}")
    if _sha256_file(path) != normalized:
        raise TrainingError(f"{label} SHA256 mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TrainingError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise TrainingError(f"{label} has an invalid size: {path}")
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except TrainingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TrainingError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"{label} must be a JSON object: {path}")
    return value


def _read_trainer_failure(path: Path) -> dict[str, str] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_TRAINER_RESULT_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("status") != "failed":
        return None
    error = value.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    message = error.get("message")
    phase = value.get("phase")
    safe_code_characters = "abcdefghijklmnopqrstuvwxyz0123456789_"
    if (
        not isinstance(code, str)
        or not 1 <= len(code) <= 64
        or any(character not in safe_code_characters for character in code)
        or not isinstance(message, str)
        or not 1 <= len(message) <= 512
        or any(ord(character) < 32 for character in message)
        or not isinstance(phase, str)
        or not 1 <= len(phase) <= 64
        or any(character not in safe_code_characters for character in phase)
    ):
        return None
    return {"phase": phase, "code": code, "message": message}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(stage, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
    except OSError as exc:
        raise TrainingError(f"cannot write {path}: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            stage.unlink()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path, 0o700)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingError(f"{label} must be a positive integer")
    return value


def _validate_task_contract_hashes(value: Any, *, label: str) -> dict[str, str]:
    contracts = _mapping(value, label)
    if not contracts:
        raise TrainingError(f"{label} is empty")
    validated: dict[str, str] = {}
    for task_key, recorded_hash in contracts.items():
        if not isinstance(task_key, str) or not isinstance(recorded_hash, str):
            raise TrainingError(f"{label} is invalid")
        try:
            expected_hash = task_contract_hash(get_task(task_key))
        except KeyError as exc:
            raise TrainingError(f"{label} uses unknown task {task_key!r}") from exc
        if recorded_hash != expected_hash:
            raise TrainingError(
                f"task contract changed for {task_key}; prepare and evaluate a new bundle"
            )
        validated[task_key] = recorded_hash
    return validated


def _path_under(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TrainingError(f"{label} path is invalid")
    path = Path(value).expanduser().resolve()
    resolved_root = root.resolve()
    if path == resolved_root or not path.is_relative_to(resolved_root):
        raise TrainingError(f"{label} is outside {resolved_root}")
    return path


def _safe_remove_tree(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or not resolved.is_relative_to(parent):
        raise TrainingError(f"refusing to remove unsafe path: {resolved}")
    shutil.rmtree(resolved)


def _validate_training_options(
    *,
    seed: int,
    max_length: int,
    epochs: float,
    max_steps: int | None,
    learning_rate: float,
    gradient_accumulation_steps: int,
    timeout_seconds: float | None,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise TrainingError("seed must be an integer between 0 and 2147483647")
    if isinstance(max_length, bool) or not 64 <= max_length <= 4096:
        raise TrainingError("max_length must be between 64 and 4096")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, float))
        or not 0.01 <= epochs <= 100
    ):
        raise TrainingError("epochs must be between 0.01 and 100")
    if max_steps is not None and (isinstance(max_steps, bool) or not 1 <= max_steps <= 1_000_000):
        raise TrainingError("max_steps must be between 1 and 1000000")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0 < learning_rate <= 0.1
    ):
        raise TrainingError("learning_rate must be greater than 0 and at most 0.1")
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not 1 <= gradient_accumulation_steps <= 1024
    ):
        raise TrainingError("gradient_accumulation_steps must be between 1 and 1024")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= timeout_seconds <= 7 * 24 * 3600
    ):
        raise TrainingError("timeout_seconds must be between 1 second and 7 days")


def _new_run_id(bundle_name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{bundle_name}\0{uuid.uuid4().hex}".encode()).hexdigest()[:8]
    return f"run-{stamp.lower()}-{suffix}"


def _llama_source_path(root: Path) -> Path:
    return root / "tools" / f"llama.cpp-{LLAMA_CPP_RELEASE}"


def _converter_source_sha256(root: Path) -> str | None:
    paths = {root / "convert_lora_to_gguf.py"}
    paths.update((root / "conversion").rglob("*.py"))
    paths.update((root / "gguf-py" / "gguf").rglob("*.py"))
    paths.update((root / "requirements").glob("requirements-convert*.txt"))
    files = sorted(
        (path for path in paths if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if (
        root / "convert_lora_to_gguf.py" not in files
        or root / "conversion" / "lfm2.py" not in files
        or not any(path.is_relative_to(root / "gguf-py" / "gguf") for path in files)
        or not any(path.parent == root / "requirements" for path in files)
    ):
        return None
    digest = hashlib.sha256()
    try:
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            digest.update(b"\0")
    except OSError:
        return None
    return digest.hexdigest()


def _gguf_model_path(root: Path) -> Path:
    return root / "models" / DEFAULT_GGUF_FILENAME


def _without_confidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "confidence"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
