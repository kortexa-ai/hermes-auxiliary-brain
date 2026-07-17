"""Isolated, low-memory LoRA training backend.

This module deliberately imports no machine-learning packages at import time.
The normal Hermes process can validate and launch a request without loading
PyTorch (or its several-hundred-megabyte entourage) into memory.

The subprocess contract is a JSON request passed with ``--request``.  A run
reads ``<bundle_dir>/train.jsonl``, stages a PEFT adapter and the base-model
configuration required by llama.cpp's LoRA converter, then atomically publishes
both the artifact directory and a small result JSON file.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUEST_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1
METADATA_FORMAT_VERSION = 1

DEFAULT_BASE_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 512
MAX_MAX_LENGTH = 4_096
DEFAULT_EPOCHS = 3.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj")
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05

MAX_REQUEST_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TRAIN_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
MAX_EXAMPLES = 100_000
MIN_TRANSFORMERS_VERSION = (5, 2, 0)

_REQUEST_KEYS = {
    "format_version",
    "bundle_dir",
    "output_dir",
    "result_path",
    "base_model",
    "base_model_revision",
    "allow_cpu",
    "seed",
    "max_length",
    "epochs",
    "max_steps",
    "learning_rate",
    "gradient_accumulation_steps",
    "lora",
    "target_modules",
    "rank",
    "alpha",
    "dropout",
}
_LORA_KEYS = {"target_modules", "rank", "alpha", "dropout"}
_MODULE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")
_GENERATION_TAG = re.compile(r"\{%[-+]?\s*generation\s*[-+]?%\}")
_END_GENERATION_TAG = re.compile(r"\{%[-+]?\s*endgeneration\s*[-+]?%\}")


class TrainerBackendError(RuntimeError):
    """Expected failure with a message safe to persist outside the worker."""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class RequestValidationError(TrainerBackendError):
    """The subprocess request does not satisfy the versioned contract."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid_request", message)


class BundleValidationError(TrainerBackendError):
    """The deterministic training JSONL is missing or malformed."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid_bundle", message)


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """Small LFM2 LoRA defaults, optionally overridden by an explicit request."""

    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Validated worker request with fully resolved local paths."""

    bundle_dir: Path
    output_dir: Path
    result_path: Path
    base_model: str = DEFAULT_BASE_MODEL
    base_model_revision: str | None = None
    allow_cpu: bool = False
    seed: int = DEFAULT_SEED
    max_length: int = DEFAULT_MAX_LENGTH
    epochs: float = DEFAULT_EPOCHS
    max_steps: int = -1
    learning_rate: float = DEFAULT_LEARNING_RATE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    lora: LoraSettings = LoraSettings()


@dataclass(frozen=True, slots=True)
class BundleInfo:
    """Non-sensitive facts established while streaming over ``train.jsonl``."""

    train_path: Path
    examples: int
    bytes: int
    sha256: str
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    model_id: str | None = None
    model_revision: str | None = None


@dataclass(frozen=True, slots=True)
class DevicePlan:
    """Selected single-device precision policy."""

    kind: str
    dtype_attribute: str
    precision: str
    bf16: bool
    fp16: bool


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Result returned to the small CLI wrapper."""

    result_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MLDependencies:
    torch: Any
    transformers: Any
    trl: Any
    peft: Any
    datasets: Any
    versions: dict[str, str]


def default_result_path(request_path: str | Path) -> Path:
    """Return the predictable result location used even for malformed requests."""

    path = Path(request_path).expanduser().resolve()
    return path.with_name(f"{path.stem}.result.json")


def load_training_request(request_path: str | Path) -> TrainingRequest:
    """Read and validate a request without importing any ML dependencies."""

    path = Path(request_path).expanduser().resolve()
    try:
        if path.stat().st_size > MAX_REQUEST_BYTES:
            raise RequestValidationError("request JSON exceeds the size limit")
        raw = path.read_text(encoding="utf-8")
    except RequestValidationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RequestValidationError("request JSON could not be read") from exc
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise RequestValidationError("request JSON is invalid") from exc
    request = parse_training_request(
        value,
        base_dir=path.parent,
        fallback_result_path=default_result_path(path),
    )
    if request.result_path == path:
        raise RequestValidationError("result_path must not overwrite the request JSON")
    return request


def parse_training_request(
    value: Any,
    *,
    base_dir: str | Path,
    fallback_result_path: str | Path | None = None,
) -> TrainingRequest:
    """Validate the pure-data request contract and resolve its paths."""

    if not isinstance(value, Mapping):
        raise RequestValidationError("request must be a JSON object")
    unknown = sorted(set(value) - _REQUEST_KEYS)
    if unknown:
        raise RequestValidationError("request contains unknown fields")
    if _integer(value.get("format_version"), "format_version", minimum=1, maximum=1) != 1:
        raise RequestValidationError("unsupported request format_version")

    root = Path(base_dir).expanduser().resolve()
    bundle_dir = _request_path(value, "bundle_dir", root)
    output_dir = _request_path(value, "output_dir", root)
    if "result_path" in value:
        result_path = _request_path(value, "result_path", root)
    elif fallback_result_path is not None:
        result_path = Path(fallback_result_path).expanduser().resolve()
    else:
        result_path = (root / "training-result.json").resolve()

    if result_path == output_dir:
        raise RequestValidationError("result_path must not be output_dir")
    if result_path == bundle_dir or _is_relative_to(result_path, bundle_dir):
        raise RequestValidationError("result_path must be outside bundle_dir")
    if output_dir == bundle_dir or _is_relative_to(output_dir, bundle_dir):
        raise RequestValidationError("output_dir must be outside bundle_dir")

    base_model = _string(
        value.get("base_model", DEFAULT_BASE_MODEL),
        "base_model",
        maximum=1_024,
    )
    revision_value = value.get("base_model_revision")
    base_model_revision = None
    if revision_value is not None:
        base_model_revision = _string(revision_value, "base_model_revision", maximum=256)

    allow_cpu = _boolean(value.get("allow_cpu", False), "allow_cpu")
    seed = _integer(value.get("seed", DEFAULT_SEED), "seed", minimum=0, maximum=2**31 - 1)
    max_length = _integer(
        value.get("max_length", DEFAULT_MAX_LENGTH),
        "max_length",
        minimum=64,
        maximum=MAX_MAX_LENGTH,
    )
    epochs = _number(value.get("epochs", DEFAULT_EPOCHS), "epochs", minimum=0.01, maximum=100)
    max_steps = value.get("max_steps", -1)
    if max_steps != -1:
        max_steps = _integer(max_steps, "max_steps", minimum=1, maximum=1_000_000)
    learning_rate = _number(
        value.get("learning_rate", DEFAULT_LEARNING_RATE),
        "learning_rate",
        minimum=1e-7,
        maximum=1.0,
    )
    accumulation = _integer(
        value.get("gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION_STEPS),
        "gradient_accumulation_steps",
        minimum=1,
        maximum=1_024,
    )
    flat_lora_keys = _LORA_KEYS & set(value)
    if "lora" in value and flat_lora_keys:
        raise RequestValidationError("LoRA settings must be nested or flat, not both")
    lora_value = (
        value.get("lora", {}) if not flat_lora_keys else {key: value[key] for key in flat_lora_keys}
    )
    lora = _parse_lora(lora_value)

    return TrainingRequest(
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        result_path=result_path,
        base_model=base_model,
        base_model_revision=base_model_revision,
        allow_cpu=allow_cpu,
        seed=seed,
        max_length=max_length,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=accumulation,
        lora=lora,
    )


def inspect_training_bundle(bundle_dir: str | Path) -> BundleInfo:
    """Stream-validate conversational examples without retaining their text."""

    directory = Path(bundle_dir).expanduser().resolve()
    train_path = directory / "train.jsonl"
    try:
        stat = train_path.stat()
    except OSError as exc:
        raise BundleValidationError("bundle train.jsonl is missing or unreadable") from exc
    if not train_path.is_file() or train_path.is_symlink():
        raise BundleValidationError("bundle train.jsonl is not a regular file")
    if stat.st_size == 0:
        raise BundleValidationError("bundle train.jsonl is empty")
    if stat.st_size > MAX_TRAIN_BYTES:
        raise BundleValidationError("bundle train.jsonl exceeds the size limit")

    digest = hashlib.sha256()
    examples = 0
    try:
        with train_path.open("rb") as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if len(raw_line) > MAX_ROW_BYTES:
                    raise BundleValidationError(
                        f"bundle train.jsonl row {row_number} exceeds the size limit"
                    )
                if not raw_line.strip():
                    raise BundleValidationError(f"bundle train.jsonl row {row_number} is blank")
                try:
                    row = json.loads(raw_line)
                except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                    raise BundleValidationError(
                        f"bundle train.jsonl row {row_number} is invalid JSON"
                    ) from exc
                _validate_conversation_row(row, row_number)
                examples += 1
                if examples > MAX_EXAMPLES:
                    raise BundleValidationError("bundle train.jsonl has too many examples")
    except BundleValidationError:
        raise
    except OSError as exc:
        raise BundleValidationError("bundle train.jsonl could not be read") from exc

    if examples == 0:
        raise BundleValidationError("bundle train.jsonl contains no examples")
    manifest_path, manifest_sha256, model_id, model_revision = _inspect_bundle_manifest(
        directory,
        examples=examples,
        train_bytes=stat.st_size,
        train_sha256=digest.hexdigest(),
    )
    return BundleInfo(
        train_path=train_path,
        examples=examples,
        bytes=stat.st_size,
        sha256=digest.hexdigest(),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        model_id=model_id,
        model_revision=model_revision,
    )


def select_device(torch_module: Any, *, allow_cpu: bool) -> DevicePlan:
    """Select one conservative device without automatic offload or quantization."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        try:
            supports_bf16 = bool(cuda.is_bf16_supported())
        except (AttributeError, RuntimeError):
            supports_bf16 = False
        if supports_bf16:
            return DevicePlan("cuda", "bfloat16", "bf16", bf16=True, fp16=False)
        return DevicePlan("cuda", "float16", "fp16", bf16=False, fp16=True)

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return DevicePlan("mps", "float16", "fp16", bf16=False, fp16=True)

    if allow_cpu:
        return DevicePlan("cpu", "float32", "fp32", bf16=False, fp16=False)
    raise TrainerBackendError(
        "device_unavailable",
        "no supported GPU is available; CPU training requires allow_cpu=true",
    )


def run_request(
    request_path: str | Path,
    *,
    train_fn: Callable[[TrainingRequest, BundleInfo], Mapping[str, Any]] | None = None,
) -> RunOutcome:
    """Execute one request and always attempt to write an explicit final state."""

    path = Path(request_path).expanduser().resolve()
    result_path = default_result_path(path)
    phase = "request_validation"
    started_at = _utc_now()
    request: TrainingRequest | None = None
    try:
        request = load_training_request(path)
        result_path = request.result_path
        _write_json_atomic(
            result_path,
            {
                "format_version": RESULT_FORMAT_VERSION,
                "status": "running",
                "started_at": started_at,
            },
        )
        phase = "bundle_validation"
        bundle = inspect_training_bundle(request.bundle_dir)
        phase = "training"
        artifacts = dict((train_fn or train_lora)(request, bundle))
        payload = {
            "format_version": RESULT_FORMAT_VERSION,
            "status": "complete",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "artifacts": artifacts,
            **artifacts,
        }
        exit_code = 0
    except TrainerBackendError as exc:
        payload = _failure_payload(started_at, phase, exc.code, exc.safe_message)
        exit_code = 1
    except Exception:
        # External ML exceptions can contain a rendered sample.  Never persist or
        # print their text; the phase and stable code are enough for orchestration.
        payload = _failure_payload(
            started_at,
            phase,
            "training_failed" if request is not None else "backend_failed",
            "training failed inside the isolated backend"
            if request is not None
            else "the isolated training backend failed",
        )
        exit_code = 1

    payload["exit_code"] = exit_code
    try:
        _write_json_atomic(result_path, payload)
    except OSError:
        # The CLI still exits non-zero.  Avoid a traceback that could include a
        # user-controlled path or data from an underlying library exception.
        exit_code = 1
        payload = _failure_payload(
            started_at,
            "result_publish",
            "result_write_failed",
            "the isolated backend could not publish its final result",
        )
        payload["exit_code"] = exit_code
    return RunOutcome(result_path=result_path, payload=payload)


def train_lora(request: TrainingRequest, bundle: BundleInfo) -> dict[str, Any]:
    """Train and atomically publish one PEFT LoRA adapter."""

    if request.output_dir.is_symlink() or (
        request.output_dir.exists() and not request.output_dir.is_dir()
    ):
        raise TrainerBackendError(
            "output_invalid",
            "output_dir is not a safe directory",
        )
    workspace_exists = request.output_dir.exists()
    if workspace_exists:
        artifact_dir = request.output_dir / "trainer-artifacts"
        staging_parent = request.output_dir
    else:
        artifact_dir = request.output_dir
        staging_parent = request.output_dir.parent
        staging_parent.mkdir(parents=True, exist_ok=True)
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise TrainerBackendError(
            "output_exists",
            "training artifacts already exist; completed artifacts are immutable",
        )

    dependencies = _load_ml_dependencies()
    plan = select_device(dependencies.torch, allow_cpu=request.allow_cpu)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{request.output_dir.name}.training-",
            dir=staging_parent,
        )
    )
    dataset_cache = Path(
        tempfile.mkdtemp(
            prefix=f".{request.output_dir.name}.dataset-",
            dir=staging_parent,
        )
    )
    if os.name != "nt":
        staging.chmod(0o700)
        dataset_cache.chmod(0o700)
    published = False
    trainer: Any = None
    dataset: Any = None
    model: Any = None
    try:
        dependencies.transformers.set_seed(request.seed, deterministic=True)
        device = _device_metadata(dependencies.torch, plan)
        base_model, base_model_revision = _resolve_bundle_model(request, bundle)
        dataset_snapshot = dataset_cache / "train.jsonl"
        _copy_verified_train(bundle, dataset_snapshot)

        dataset = dependencies.datasets.load_dataset(
            "json",
            data_files={"train": str(dataset_snapshot)},
            split="train",
            cache_dir=str(dataset_cache / "hf-cache"),
            keep_in_memory=False,
        )
        if len(dataset) != bundle.examples or "messages" not in dataset.column_names:
            raise TrainerBackendError(
                "dataset_load_failed",
                "the dataset loader did not preserve the validated training examples",
            )
        dataset = dataset.select_columns(["messages"])

        tokenizer_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {
            "dtype": getattr(dependencies.torch, plan.dtype_attribute),
            "device_map": {"": plan.kind},
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        if base_model_revision is not None:
            tokenizer_kwargs["revision"] = base_model_revision
            model_kwargs["revision"] = base_model_revision

        tokenizer = dependencies.transformers.AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=False,
            **tokenizer_kwargs,
        )
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not (
            _GENERATION_TAG.search(template) and _END_GENERATION_TAG.search(template)
        ):
            raise TrainerBackendError(
                "assistant_mask_unavailable",
                "the base model chat template does not expose an assistant token mask",
            )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise TrainerBackendError(
                    "tokenizer_invalid",
                    "the base tokenizer has neither a padding token nor an end token",
                )
            tokenizer.pad_token = tokenizer.eos_token

        model = dependencies.transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )
        model.config.use_cache = False
        effective_max_length = _effective_max_length(request, tokenizer, model.config)
        assistant_mask_stats = _preflight_assistant_masks(
            dataset,
            tokenizer,
            max_length=effective_max_length,
        )

        base_config_dir = staging / "base_config"
        adapter_dir = staging / "adapter"
        model.config.save_pretrained(base_config_dir)
        tokenizer.save_pretrained(base_config_dir)

        lora_config = dependencies.peft.LoraConfig(
            r=request.lora.rank,
            lora_alpha=request.lora.alpha,
            lora_dropout=request.lora.dropout,
            target_modules=list(request.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        training_args = dependencies.trl.SFTConfig(
            output_dir=str(staging / "trainer"),
            num_train_epochs=request.epochs,
            max_steps=request.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            learning_rate=request.learning_rate,
            max_length=effective_max_length,
            packing=False,
            assistant_only_loss=True,
            bf16=plan.bf16,
            fp16=plan.fp16,
            use_cpu=plan.kind == "cpu",
            optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            dataloader_pin_memory=plan.kind == "cuda",
            save_strategy="no",
            logging_strategy="no",
            eval_strategy="no",
            report_to="none",
            disable_tqdm=True,
            push_to_hub=False,
            seed=request.seed,
            data_seed=request.seed,
            full_determinism=True,
        )
        trainer = dependencies.trl.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        train_output = trainer.train()
        trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
        if (
            not (adapter_dir / "adapter_config.json").is_file()
            or not (adapter_dir / "adapter_model.safetensors").is_file()
        ):
            raise TrainerBackendError(
                "adapter_save_failed",
                "PEFT did not produce the expected safe adapter files",
            )

        trainable_parameters, total_parameters = _parameter_counts(trainer.model)
        metrics = _numeric_metrics(getattr(train_output, "metrics", {}))
        resolved_revision = getattr(model.config, "_commit_hash", None)
        metadata = {
            "format_version": METADATA_FORMAT_VERSION,
            "base_model": {
                "id": base_model,
                "requested_revision": base_model_revision,
                "resolved_revision": resolved_revision,
                "config_dir": "base_config",
            },
            "dataset": {
                "examples": bundle.examples,
                "bytes": bundle.bytes,
                "sha256": bundle.sha256,
                "manifest_sha256": bundle.manifest_sha256,
            },
            "dependencies": dependencies.versions,
            "device": device,
            "lora": {
                "target_modules": list(request.lora.target_modules),
                "rank": request.lora.rank,
                "alpha": request.lora.alpha,
                "dropout": request.lora.dropout,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            },
            "training": {
                "seed": request.seed,
                "max_length": effective_max_length,
                "configured_max_length": request.max_length,
                "epochs": request.epochs,
                "max_steps": request.max_steps,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "learning_rate": request.learning_rate,
                "packing": False,
                "assistant_only_loss": True,
                "attention_implementation": "eager",
                "quantization": None,
                "report_to": "none",
                "assistant_mask_preflight": assistant_mask_stats,
                "metrics": metrics,
            },
        }
        _write_json_atomic(staging / "training_metadata.json", metadata)
        with contextlib.suppress(OSError):
            (staging / "trainer").rmdir()

        os.replace(staging, artifact_dir)
        published = True
        return {
            "artifact_dir": str(artifact_dir),
            "adapter_dir": str(artifact_dir / "adapter"),
            "base_config_dir": str(artifact_dir / "base_config"),
            "metadata_path": str(artifact_dir / "training_metadata.json"),
            "base_model": base_model,
            "resolved_revision": resolved_revision,
            "dataset_sha256": bundle.sha256,
            "examples": bundle.examples,
            "dependencies": dependencies.versions,
            "device": device,
            "metrics": metrics,
        }
    except TrainerBackendError:
        raise
    except Exception as exc:
        raise TrainerBackendError(
            "training_failed",
            "LoRA training failed inside the isolated backend",
        ) from exc
    finally:
        trainer = None
        dataset = None
        model = None
        gc.collect()
        cuda = getattr(dependencies.torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            with contextlib.suppress(RuntimeError):
                cuda.empty_cache()
        shutil.rmtree(dataset_cache, ignore_errors=True)
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _load_ml_dependencies() -> _MLDependencies:
    modules: dict[str, Any] = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            raise TrainerBackendError(
                "dependency_unavailable",
                f"required training dependency {name} is unavailable",
            ) from exc

    version = _version_tuple(str(getattr(modules["transformers"], "__version__", "")))
    if version < MIN_TRANSFORMERS_VERSION:
        raise TrainerBackendError(
            "dependency_incompatible",
            "Transformers 5.2.0 or newer is required for LFM2.5 training",
        )
    for module_name, symbol in (
        ("transformers", "AutoModelForCausalLM"),
        ("transformers", "AutoTokenizer"),
        ("trl", "SFTConfig"),
        ("trl", "SFTTrainer"),
        ("peft", "LoraConfig"),
        ("datasets", "load_dataset"),
    ):
        if not hasattr(modules[module_name], symbol):
            raise TrainerBackendError(
                "dependency_incompatible",
                "installed training dependencies do not expose the required APIs",
            )

    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TrainerBackendError(
                "dependency_unavailable",
                f"required training dependency {name} is unavailable",
            ) from exc
    return _MLDependencies(versions=versions, **modules)


def _parse_lora(value: Any) -> LoraSettings:
    if not isinstance(value, Mapping):
        raise RequestValidationError("lora must be a JSON object")
    if set(value) - _LORA_KEYS:
        raise RequestValidationError("lora contains unknown fields")
    targets_value = value.get("target_modules", list(DEFAULT_LORA_TARGET_MODULES))
    if not isinstance(targets_value, list) or not 1 <= len(targets_value) <= 16:
        raise RequestValidationError("lora.target_modules must be a non-empty JSON array")
    targets: list[str] = []
    for item in targets_value:
        if not isinstance(item, str) or _MODULE_NAME.fullmatch(item) is None:
            raise RequestValidationError("lora.target_modules contains an invalid name")
        if item in targets:
            raise RequestValidationError("lora.target_modules contains a duplicate")
        targets.append(item)
    rank = _integer(value.get("rank", DEFAULT_LORA_RANK), "lora.rank", minimum=1, maximum=256)
    alpha = _integer(
        value.get("alpha", DEFAULT_LORA_ALPHA),
        "lora.alpha",
        minimum=1,
        maximum=1_024,
    )
    dropout = _number(
        value.get("dropout", DEFAULT_LORA_DROPOUT),
        "lora.dropout",
        minimum=0.0,
        maximum=0.5,
    )
    return LoraSettings(tuple(targets), rank, alpha, dropout)


def _validate_conversation_row(row: Any, row_number: int) -> None:
    if not isinstance(row, Mapping):
        raise BundleValidationError(f"bundle train.jsonl row {row_number} must be an object")
    messages = row.get("messages")
    if not isinstance(messages, list) or not 2 <= len(messages) <= 64:
        raise BundleValidationError(
            f"bundle train.jsonl row {row_number} must contain conversational messages"
        )
    roles: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise BundleValidationError(
                f"bundle train.jsonl row {row_number} contains an invalid message"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise BundleValidationError(
                f"bundle train.jsonl row {row_number} contains an invalid message role"
            )
        if not isinstance(content, str) or not content:
            raise BundleValidationError(
                f"bundle train.jsonl row {row_number} contains empty message content"
            )
        roles.append(role)
    if roles[0] == "system":
        roles = roles[1:]
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        raise BundleValidationError(
            f"bundle train.jsonl row {row_number} must end with an assistant response"
        )
    if "system" in roles:
        raise BundleValidationError(
            f"bundle train.jsonl row {row_number} contains a misplaced system message"
        )
    if any(left == right for left, right in zip(roles, roles[1:], strict=False)):
        raise BundleValidationError(
            f"bundle train.jsonl row {row_number} has consecutive messages with the same role"
        )


def _inspect_bundle_manifest(
    directory: Path,
    *,
    examples: int,
    train_bytes: int,
    train_sha256: str,
) -> tuple[Path | None, str | None, str | None, str | None]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None, None, None, None
    try:
        stat = manifest_path.stat()
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise BundleValidationError("bundle manifest.json is not a regular file")
        if stat.st_size == 0 or stat.st_size > MAX_MANIFEST_BYTES:
            raise BundleValidationError("bundle manifest.json has an invalid size")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except BundleValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BundleValidationError("bundle manifest.json is invalid or unreadable") from exc
    if not isinstance(manifest, Mapping) or manifest.get("format_version") != 1:
        raise BundleValidationError("bundle manifest.json has an unsupported format")

    files = manifest.get("files")
    train_details = files.get("train.jsonl") if isinstance(files, Mapping) else None
    if not isinstance(train_details, Mapping):
        raise BundleValidationError("bundle manifest.json does not describe train.jsonl")
    if (
        train_details.get("sha256") != train_sha256
        or train_details.get("bytes") != train_bytes
        or train_details.get("examples") != examples
    ):
        raise BundleValidationError("bundle train.jsonl does not match manifest.json")

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise BundleValidationError("bundle manifest.json does not pin a base model")
    model_id = model.get("id")
    revision = model.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise BundleValidationError("bundle manifest.json has an invalid base model id")
    if not isinstance(revision, str) or not revision.strip():
        raise BundleValidationError("bundle manifest.json has an invalid base model revision")
    return (
        manifest_path,
        hashlib.sha256(raw).hexdigest(),
        model_id.strip(),
        revision.strip(),
    )


def _resolve_bundle_model(
    request: TrainingRequest,
    bundle: BundleInfo,
) -> tuple[str, str | None]:
    if bundle.model_id is not None and bundle.model_id != request.base_model:
        raise TrainerBackendError(
            "model_mismatch",
            "the requested base model does not match the deterministic bundle",
        )
    if (
        bundle.model_revision is not None
        and request.base_model_revision is not None
        and bundle.model_revision != request.base_model_revision
    ):
        raise TrainerBackendError(
            "model_mismatch",
            "the requested base model revision does not match the deterministic bundle",
        )
    return (
        bundle.model_id or request.base_model,
        bundle.model_revision or request.base_model_revision,
    )


def _copy_verified_train(bundle: BundleInfo, destination: Path) -> None:
    """Snapshot the already-inspected JSONL and reject any intervening change."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        if bundle.train_path.is_symlink():
            raise BundleValidationError("bundle train.jsonl changed after validation")
        with bundle.train_path.open("rb") as source, destination.open("xb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
                size += len(block)
            target.flush()
            os.fsync(target.fileno())
        if os.name != "nt":
            destination.chmod(0o600)
    except BundleValidationError:
        raise
    except OSError as exc:
        raise BundleValidationError("bundle train.jsonl could not be snapshotted") from exc
    if size != bundle.bytes or digest.hexdigest() != bundle.sha256:
        destination.unlink(missing_ok=True)
        raise BundleValidationError("bundle train.jsonl changed after validation")


def _preflight_assistant_masks(
    dataset: Any,
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, int]:
    """Ensure right truncation preserves every supervised assistant token.

    TRL versions before its post-truncation mask check can otherwise train on
    all-ignored labels.  Iterate one row at a time so the validation adds no
    dataset-sized Python allocation.
    """

    examples = 0
    truncated_examples = 0
    minimum: int | None = None
    try:
        for row in dataset:
            messages = row.get("messages") if isinstance(row, Mapping) else None
            common = {
                "tokenize": True,
                "add_generation_prompt": False,
                "padding": False,
                "return_dict": True,
                "return_assistant_tokens_mask": True,
            }
            full_encoded = tokenizer.apply_chat_template(
                messages,
                truncation=False,
                **common,
            )
            truncated_encoded = tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=max_length,
                **common,
            )
            full_mask = (
                full_encoded.get("assistant_masks") if isinstance(full_encoded, Mapping) else None
            )
            truncated_mask = (
                truncated_encoded.get("assistant_masks")
                if isinstance(truncated_encoded, Mapping)
                else None
            )
            full_supervised = _count_mask_tokens(full_mask)
            supervised = _count_mask_tokens(truncated_mask)
            examples += 1
            if full_supervised == 0:
                raise TrainerBackendError(
                    "assistant_mask_preflight_failed",
                    "the tokenizer produced no assistant-only loss tokens",
                )
            if supervised != full_supervised:
                truncated_examples += 1
            minimum = supervised if minimum is None else min(minimum, supervised)
    except TrainerBackendError:
        raise
    except Exception as exc:
        raise TrainerBackendError(
            "assistant_mask_preflight_failed",
            "the tokenizer could not validate assistant-only loss masks",
        ) from exc
    if examples == 0:
        raise TrainerBackendError(
            "assistant_mask_preflight_failed",
            "the training dataset contains no examples",
        )
    if truncated_examples:
        raise TrainerBackendError(
            "assistant_tokens_truncated",
            f"{truncated_examples} training examples lose assistant target tokens at max_length; "
            "increase max_length or shorten the input",
        )
    return {"examples": examples, "minimum_assistant_tokens": minimum or 0}


def _count_mask_tokens(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return sum(_count_mask_tokens(item) for item in value)
    try:
        return int(value == 1)
    except (TypeError, ValueError, RuntimeError):
        return 0


def _effective_max_length(request: TrainingRequest, tokenizer: Any, config: Any) -> int:
    limits = [request.max_length]
    for value in (
        getattr(tokenizer, "model_max_length", None),
        getattr(config, "max_position_embeddings", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
            limits.append(value)
    return min(limits)


def _device_metadata(torch_module: Any, plan: DevicePlan) -> dict[str, Any]:
    name: str | None = None
    if plan.kind == "cuda":
        with contextlib.suppress(AttributeError, RuntimeError):
            name = str(torch_module.cuda.get_device_name(0))[:256]
    elif plan.kind == "mps":
        name = "Apple MPS"
    elif plan.kind == "cpu":
        name = "CPU"
    return {
        "type": plan.kind,
        "name": name,
        "precision": plan.precision,
        "flash_attention": False,
        "bitsandbytes": False,
    }


def _parameter_counts(model: Any) -> tuple[int | None, int | None]:
    method = getattr(model, "get_nb_trainable_parameters", None)
    if not callable(method):
        return None, None
    try:
        trainable, total = method()
        return int(trainable), int(total)
    except (TypeError, ValueError, RuntimeError):
        return None, None


def _numeric_metrics(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,127}", key) is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        if isinstance(item, float) and not math.isfinite(item):
            continue
        result[key] = item
    return result


def _failure_payload(
    started_at: str,
    phase: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "failed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "phase": phase,
        "error": {"code": code, "message": message},
    }


def _request_path(value: Mapping[str, Any], key: str, root: Path) -> Path:
    text = _string(value.get(key), key, maximum=4_096)
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _string(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise RequestValidationError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise RequestValidationError(f"{field} contains control characters")
    return value.strip()


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise RequestValidationError(f"{field} must be a boolean")
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RequestValidationError(f"{field} must be an integer in the allowed range")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(f"{field} must be a finite number in the allowed range")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RequestValidationError(f"{field} must be a finite number in the allowed range")
    return number


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return (0, 0, 0)
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated auxiliary-brain LoRA job")
    parser.add_argument("--request", required=True, type=Path, help="path to the JSON request")
    args = parser.parse_args(argv)
    outcome = run_request(args.request)
    print(
        json.dumps(
            {"status": outcome.payload["status"], "result_path": str(outcome.result_path)},
            sort_keys=True,
        )
    )
    return int(outcome.payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
