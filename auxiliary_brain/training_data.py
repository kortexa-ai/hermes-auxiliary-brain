"""Deterministic, privacy-gated training bundle preparation.

This module deliberately stops at the dataset boundary.  It uses no ML SDK and
does not load a model, which keeps readiness checks cheap enough for small
machines and makes the resulting bundle useful to more than one trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .store import DATASET_FORMAT_VERSION, BrainStore
from .tasks import BASE_INSTRUCTION, TaskSpec, get_task, list_tasks, validate_json
from .version import __version__

TRAINING_BUNDLE_FORMAT_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_HOLDOUT_PERCENT = 20
DEFAULT_MIN_UNIQUE_EXAMPLES = 20
DEFAULT_MIN_TRAIN_EXAMPLES = 16
DEFAULT_MIN_HOLDOUT_EXAMPLES = 4

# Keep these dependency-free preparation limits aligned with trainer_backend.
# They are repeated here intentionally so readiness never imports the ML stack.
MAX_TRAIN_BYTES = 64 * 1024 * 1024
MAX_HOLDOUT_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
MAX_EXAMPLES = 100_000
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FINDING_IDS = 100

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600

_ATTRIBUTION_KEYS = (
    "author_id",
    "gateway_user_id",
    "platform_user_id",
    "sender_id",
    "user_id",
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:sk|hf|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?token|auth[ _-]?token|password|passwd|secret)"
        r"\b\s*(?:is|=|:)\s*[\"']?[A-Za-z0-9._~+/-]{8,}={0,2}",
        re.IGNORECASE,
    ),
)


class TrainingDataError(RuntimeError):
    """A safe training bundle could not be prepared."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True, slots=True)
class _Candidate:
    prediction_id: str
    task: TaskSpec
    input_text: str
    assistant_json: str
    contract_hash: str
    group_id: str
    split: str

    def row(self) -> dict[str, Any]:
        return {
            "messages": [
                *self.task.build_messages(self.input_text),
                {"role": "assistant", "content": self.assistant_json},
            ],
            "metadata": {
                "prediction_id": self.prediction_id,
                "task": self.task.key,
                "task_contract_hash": self.contract_hash,
            },
        }


@dataclass(frozen=True, slots=True)
class _Analysis:
    report: dict[str, Any]
    train: tuple[_Candidate, ...]
    holdout: tuple[_Candidate, ...]


def task_contract_hash(task: TaskSpec) -> str:
    """Return the capture-time hash for a task's complete inference contract."""

    contract = {
        "base_instruction": BASE_INSTRUCTION,
        "key": task.key,
        "instruction": task.instruction,
        "schema": task.schema,
        "max_tokens": task.max_tokens,
        "temperature": task.temperature,
    }
    encoded = _canonical_json(contract)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_duplicate_input(value: str) -> str:
    """Normalize text only for duplicate grouping, never for model input."""

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def inspect_readiness(
    store: BrainStore,
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
    """Inspect corrected rows without writing files or loading model weights."""

    return _analyze(
        store,
        task_key=task_key,
        seed=seed,
        holdout_percent=holdout_percent,
        min_unique_examples=min_unique_examples,
        min_train_examples=min_train_examples,
        min_holdout_examples=min_holdout_examples,
        acknowledge_unattributed_gateway=acknowledge_unattributed_gateway,
        allow_small=allow_small,
    ).report


def prepare_bundle(
    store: BrainStore,
    root: str | Path,
    *,
    model: str,
    revision: str,
    task_key: str | None = None,
    seed: int = DEFAULT_SEED,
    holdout_percent: int = DEFAULT_HOLDOUT_PERCENT,
    min_unique_examples: int = DEFAULT_MIN_UNIQUE_EXAMPLES,
    min_train_examples: int = DEFAULT_MIN_TRAIN_EXAMPLES,
    min_holdout_examples: int = DEFAULT_MIN_HOLDOUT_EXAMPLES,
    acknowledge_unattributed_gateway: bool = False,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Create an immutable train/holdout bundle with an auditable manifest.

    The final directory name is derived from the manifest.  Files are first
    written under a sibling staging directory and the completed directory is
    renamed into place, so an interrupted run cannot look complete.
    """

    model = _required_text(model, "model")
    revision = _required_text(revision, "revision")
    analysis = _analyze(
        store,
        task_key=task_key,
        seed=seed,
        holdout_percent=holdout_percent,
        min_unique_examples=min_unique_examples,
        min_train_examples=min_train_examples,
        min_holdout_examples=min_holdout_examples,
        acknowledge_unattributed_gateway=acknowledge_unattributed_gateway,
        allow_small=allow_small,
    )
    if not analysis.report["ready"]:
        codes = ", ".join(item["code"] for item in analysis.report["errors"])
        raise TrainingDataError(
            f"training data is not ready ({codes or 'readiness gate failed'})",
            report=analysis.report,
        )

    destination_root = Path(root).expanduser().resolve()
    destination_root_created = not destination_root.exists()
    destination_root.mkdir(
        mode=_PRIVATE_DIRECTORY_MODE,
        parents=True,
        exist_ok=True,
    )
    if destination_root_created:
        _restrict_mode(destination_root, _PRIVATE_DIRECTORY_MODE)
    staging = destination_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
    _restrict_mode(staging, _PRIVATE_DIRECTORY_MODE)
    try:
        train_file = _write_jsonl(staging / "train.jsonl", analysis.train)
        holdout_file = _write_jsonl(staging / "holdout.jsonl", analysis.holdout)
        manifest = {
            "format_version": TRAINING_BUNDLE_FORMAT_VERSION,
            "plugin_version": __version__,
            "model": {"id": model, "revision": revision},
            "seed": seed,
            "split": {
                "algorithm": "sha256-normalized-input-threshold-v1",
                "holdout_percent": holdout_percent,
            },
            "task_contract_hashes": analysis.report["task_contract_hashes"],
            "promotion": {
                "experimental": analysis.report["experimental"],
                "promotable": analysis.report["promotable"],
            },
            "preparation": {
                "acknowledge_unattributed_gateway": acknowledge_unattributed_gateway,
                "allow_small": allow_small,
                "task_key": task_key,
                "thresholds": analysis.report["thresholds"],
            },
            "counts": analysis.report["counts"],
            "files": {
                "holdout.jsonl": holdout_file,
                "train.jsonl": train_file,
            },
        }
        manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")
        if len(manifest_bytes) > MAX_MANIFEST_BYTES:
            raise TrainingDataError("training bundle manifest exceeds the size limit")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _write_bytes(staging / "manifest.json", manifest_bytes)
        final = destination_root / f"bundle-{manifest_sha256[:16]}"

        if final.exists():
            _verify_existing_bundle(final, manifest_bytes, manifest)
            shutil.rmtree(staging)
            return _bundle_result(final, manifest, analysis.report, False, manifest_sha256)

        try:
            os.replace(staging, final)
        except FileExistsError:
            _verify_existing_bundle(final, manifest_bytes, manifest)
            shutil.rmtree(staging)
            return _bundle_result(final, manifest, analysis.report, False, manifest_sha256)
        return _bundle_result(final, manifest, analysis.report, True, manifest_sha256)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _analyze(
    store: BrainStore,
    *,
    task_key: str | None,
    seed: int,
    holdout_percent: int,
    min_unique_examples: int,
    min_train_examples: int,
    min_holdout_examples: int,
    acknowledge_unattributed_gateway: bool,
    allow_small: bool,
) -> _Analysis:
    _validate_options(
        seed=seed,
        holdout_percent=holdout_percent,
        min_unique_examples=min_unique_examples,
        min_train_examples=min_train_examples,
        min_holdout_examples=min_holdout_examples,
    )
    source_usage, examples = store.bounded_training_examples(
        task_key=task_key,
        corrected_only=True,
        limit=MAX_EXAMPLES + 1,
        max_examples=MAX_EXAMPLES,
        max_bytes=MAX_SOURCE_BYTES,
        max_row_bytes=MAX_ROW_BYTES,
    )
    source_errors: list[dict[str, Any]] = []
    if source_usage["examples"] > MAX_EXAMPLES:
        source_errors.append(
            _maximum_finding("max_source_examples", source_usage["examples"], MAX_EXAMPLES)
        )
    if source_usage["bytes"] > MAX_SOURCE_BYTES:
        source_errors.append(
            _maximum_finding("max_source_bytes", source_usage["bytes"], MAX_SOURCE_BYTES)
        )
    if source_usage["max_row_bytes"] > MAX_ROW_BYTES:
        source_errors.append(
            _maximum_finding(
                "max_source_row_bytes",
                source_usage["max_row_bytes"],
                MAX_ROW_BYTES,
            )
        )
    if source_errors:
        experimental = bool(
            allow_small
            or min_unique_examples < DEFAULT_MIN_UNIQUE_EXAMPLES
            or min_train_examples < DEFAULT_MIN_TRAIN_EXAMPLES
            or min_holdout_examples < DEFAULT_MIN_HOLDOUT_EXAMPLES
        )
        return _Analysis(
            {
                "ready": False,
                "promotable": False,
                "experimental": experimental,
                "counts": {
                    "corrected": source_usage["examples"],
                    "eligible": 0,
                    "excluded": source_usage["examples"],
                    "unique_examples": 0,
                    "train": 0,
                    "train_unique": 0,
                    "holdout": 0,
                    "holdout_unique": 0,
                },
                "thresholds": {
                    "min_unique_examples": min_unique_examples,
                    "min_train_examples": min_train_examples,
                    "min_holdout_examples": min_holdout_examples,
                },
                "trainer_limits": _trainer_limits(),
                "trainer_usage": {
                    "source_examples": source_usage["examples"],
                    "source_bytes": source_usage["bytes"],
                    "source_max_row_bytes": source_usage["max_row_bytes"],
                    "train_examples": 0,
                    "train_bytes": 0,
                    "holdout_bytes": 0,
                    "largest_row_bytes": 0,
                },
                "split": {
                    "algorithm": "sha256-normalized-input-threshold-v1",
                    "holdout_percent": holdout_percent,
                    "seed": seed,
                },
                "task_contract_hashes": {},
                "errors": sorted(source_errors, key=lambda item: item["code"]),
                "warnings": [],
            },
            (),
            (),
        )
    excluded: dict[str, list[str]] = {}
    eligible: list[_Candidate] = []
    gateway_ids: list[str] = []
    secret_ids: list[str] = []
    observed_examples = 0

    for example in examples:
        observed_examples += 1
        prediction_id = str(example.get("prediction_id") or "").strip()
        finding_id = prediction_id or "<missing>"
        if example.get("corrected") is not True:
            _add_id(excluded, "not_corrected", finding_id)
            continue
        if example.get("dataset_format_version") != DATASET_FORMAT_VERSION:
            _add_id(excluded, "dataset_format_mismatch", finding_id)
            continue
        task_name = str(example.get("task") or "")
        try:
            task = get_task(task_name)
        except KeyError:
            _add_id(excluded, "unknown_task", finding_id)
            continue
        contract_hash = task_contract_hash(task)
        metadata = example.get("metadata")
        if not isinstance(metadata, Mapping):
            _add_id(excluded, "invalid_metadata", finding_id)
            continue
        if metadata.get("task_contract_hash") != contract_hash:
            _add_id(excluded, "task_contract_mismatch", finding_id)
            continue
        output = example.get("output")
        schema_errors = validate_json(output, task.schema)
        if schema_errors:
            _add_id(excluded, "invalid_corrected_output", finding_id)
            continue
        input_text = example.get("input")
        if not isinstance(input_text, str) or not input_text.strip():
            _add_id(excluded, "invalid_input", finding_id)
            continue
        if not prediction_id:
            _add_id(excluded, "missing_prediction_id", finding_id)
            continue

        assistant_json = _canonical_json(output)
        normalized = normalize_duplicate_input(input_text)
        group_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        split = _split_for_group(group_id, seed, holdout_percent)
        eligible.append(
            _Candidate(
                prediction_id=prediction_id,
                task=task,
                input_text=input_text,
                assistant_json=assistant_json,
                contract_hash=contract_hash,
                group_id=group_id,
                split=split,
            )
        )
        if _is_unattributed_gateway(metadata):
            gateway_ids.append(prediction_id)
        if _contains_likely_secret(input_text) or _contains_likely_secret(assistant_json):
            secret_ids.append(prediction_id)

    eligible.sort(
        key=lambda item: (
            item.group_id,
            item.task.key,
            item.input_text,
            item.assistant_json,
            item.prediction_id,
        )
    )
    train = tuple(item for item in eligible if item.split == "train")
    holdout = tuple(item for item in eligible if item.split == "holdout")
    unique_groups = {item.group_id for item in eligible}
    train_groups = {item.group_id for item in train}
    holdout_groups = {item.group_id for item in holdout}
    required_tasks = {task.key for task in list_tasks()}
    train_tasks = {item.task.key for item in train}
    holdout_tasks = {item.task.key for item in holdout}
    train_bytes = 0
    holdout_bytes = 0
    largest_row_bytes = 0
    oversized_row_ids: list[str] = []
    for candidate in eligible:
        row_bytes = _serialized_row_size(candidate)
        largest_row_bytes = max(largest_row_bytes, row_bytes)
        if candidate.split == "train":
            train_bytes += row_bytes
        else:
            holdout_bytes += row_bytes
        if row_bytes > MAX_ROW_BYTES:
            oversized_row_ids.append(candidate.prediction_id)
    counts = {
        "corrected": observed_examples,
        "eligible": len(eligible),
        "excluded": observed_examples - len(eligible),
        "unique_examples": len(unique_groups),
        "train": len(train),
        "train_unique": len(train_groups),
        "holdout": len(holdout),
        "holdout_unique": len(holdout_groups),
    }
    thresholds = {
        "min_unique_examples": min_unique_examples,
        "min_train_examples": min_train_examples,
        "min_holdout_examples": min_holdout_examples,
    }

    errors: list[dict[str, Any]] = []
    warnings = [
        _finding(code, _excluded_message(code), ids) for code, ids in sorted(excluded.items())
    ]
    if gateway_ids:
        finding = _finding(
            "unattributed_gateway_slash",
            "gateway capture rows have no sender attribution; review ownership before training",
            gateway_ids,
        )
        (warnings if acknowledge_unattributed_gateway else errors).append(finding)
    if secret_ids:
        errors.append(
            _finding(
                "likely_secret",
                "likely credential material appears in training input or corrected output",
                secret_ids,
            )
        )

    if len(train) > MAX_EXAMPLES:
        errors.append(_maximum_finding("max_train_examples", len(train), MAX_EXAMPLES))
    if observed_examples > MAX_EXAMPLES:
        errors.append(_maximum_finding("max_source_examples", observed_examples, MAX_EXAMPLES))
    if train_bytes > MAX_TRAIN_BYTES:
        errors.append(_maximum_finding("max_train_bytes", train_bytes, MAX_TRAIN_BYTES))
    if holdout_bytes > MAX_HOLDOUT_BYTES:
        errors.append(_maximum_finding("max_holdout_bytes", holdout_bytes, MAX_HOLDOUT_BYTES))
    if oversized_row_ids:
        finding = _finding(
            "max_row_bytes",
            f"{largest_row_bytes} present; at most {MAX_ROW_BYTES} allowed",
            oversized_row_ids,
        )
        finding.update({"actual": largest_row_bytes, "maximum": MAX_ROW_BYTES})
        errors.append(finding)

    if not eligible:
        errors.append(
            {
                "code": "no_eligible_examples",
                "message": "no corrected current-contract examples are eligible",
            }
        )
    elif not train:
        errors.append(
            {
                "code": "no_train_examples",
                "message": "the deterministic split contains no training examples",
            }
        )

    gate_findings = []
    if counts["unique_examples"] < min_unique_examples:
        gate_findings.append(
            _gate_finding("min_unique_examples", counts["unique_examples"], min_unique_examples)
        )
    if counts["train_unique"] < min_train_examples:
        gate_findings.append(
            _gate_finding("min_train_examples", counts["train_unique"], min_train_examples)
        )
    if counts["holdout_unique"] < min_holdout_examples:
        gate_findings.append(
            _gate_finding("min_holdout_examples", counts["holdout_unique"], min_holdout_examples)
        )
    missing_train_tasks = required_tasks - train_tasks
    if missing_train_tasks:
        gate_findings.append(
            _task_split_finding("task_missing_train", "train", missing_train_tasks)
        )
    missing_holdout_tasks = required_tasks - holdout_tasks
    if missing_holdout_tasks:
        gate_findings.append(
            _task_split_finding("task_missing_holdout", "holdout", missing_holdout_tasks)
        )
    (warnings if allow_small else errors).extend(gate_findings)

    errors.sort(key=lambda item: item["code"])
    warnings.sort(key=lambda item: item["code"])
    task_hashes = {
        item.task.key: item.contract_hash
        for item in sorted(eligible, key=lambda candidate: candidate.task.key)
    }
    experimental = bool(
        allow_small
        or min_unique_examples < DEFAULT_MIN_UNIQUE_EXAMPLES
        or min_train_examples < DEFAULT_MIN_TRAIN_EXAMPLES
        or min_holdout_examples < DEFAULT_MIN_HOLDOUT_EXAMPLES
    )
    ready = not errors
    report = {
        "ready": ready,
        "promotable": ready and not experimental,
        "experimental": experimental,
        "counts": counts,
        "thresholds": thresholds,
        "trainer_limits": _trainer_limits(),
        "trainer_usage": {
            "source_examples": observed_examples,
            "source_bytes": source_usage["bytes"],
            "source_max_row_bytes": source_usage["max_row_bytes"],
            "train_examples": len(train),
            "train_bytes": train_bytes,
            "holdout_bytes": holdout_bytes,
            "largest_row_bytes": largest_row_bytes,
        },
        "split": {
            "algorithm": "sha256-normalized-input-threshold-v1",
            "holdout_percent": holdout_percent,
            "seed": seed,
        },
        "task_contract_hashes": task_hashes,
        "errors": errors,
        "warnings": warnings,
    }
    return _Analysis(report, train, holdout)


def _trainer_limits() -> dict[str, int]:
    return {
        "max_source_examples": MAX_EXAMPLES,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_train_examples": MAX_EXAMPLES,
        "max_train_bytes": MAX_TRAIN_BYTES,
        "max_holdout_bytes": MAX_HOLDOUT_BYTES,
        "max_row_bytes": MAX_ROW_BYTES,
    }


def _split_for_group(group_id: str, seed: int, holdout_percent: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "holdout" if bucket < holdout_percent * 100 else "train"


def _is_unattributed_gateway(metadata: Mapping[str, Any]) -> bool:
    source = str(metadata.get("source") or "").strip().casefold()
    platform = str(metadata.get("platform") or "").strip().casefold()
    gateway_capture = source == "gateway-slash" or (
        source in {"pre_llm_call", "pre_llm_call_assist"}
        and platform not in {"acp", "cli", "desktop", "tui"}
    )
    if not gateway_capture:
        return False
    return not any(str(metadata.get(key) or "").strip() for key in _ATTRIBUTION_KEYS)


def _contains_likely_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _write_jsonl(
    path: Path,
    candidates: tuple[_Candidate, ...],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with _open_private_binary(path) as handle:
        for candidate in candidates:
            line = (_canonical_json(candidate.row()) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            size += len(line)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "examples": len(candidates),
    }


def _write_bytes(path: Path, value: bytes) -> None:
    with _open_private_binary(path) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_existing_bundle(
    path: Path,
    manifest_bytes: bytes,
    manifest: Mapping[str, Any],
) -> None:
    try:
        if (path / "manifest.json").read_bytes() != manifest_bytes:
            raise TrainingDataError(f"bundle path collision at {path}")
        for name, details in manifest["files"].items():
            if _file_sha256(path / name) != details["sha256"]:
                raise TrainingDataError(f"existing bundle is corrupt: {path / name}")
    except OSError as exc:
        raise TrainingDataError(f"cannot verify existing bundle {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_result(
    path: Path,
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    created: bool,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "manifest_path": path / "manifest.json",
        "created": created,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
        "readiness": readiness,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _serialized_row_size(candidate: _Candidate) -> int:
    """Return the exact JSONL byte count without retaining another dataset copy."""

    return len((_canonical_json(candidate.row()) + "\n").encode("utf-8"))


def _open_private_binary(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _restrict_mode(path: Path, mode: int) -> None:
    if os.name != "nt":
        path.chmod(mode)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _validate_options(
    *,
    seed: int,
    holdout_percent: int,
    min_unique_examples: int,
    min_train_examples: int,
    min_holdout_examples: int,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(holdout_percent, bool) or not isinstance(holdout_percent, int):
        raise ValueError("holdout_percent must be an integer")
    if not 1 <= holdout_percent <= 99:
        raise ValueError("holdout_percent must be between 1 and 99")
    for name, value in (
        ("min_unique_examples", min_unique_examples),
        ("min_train_examples", min_train_examples),
        ("min_holdout_examples", min_holdout_examples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _add_id(findings: dict[str, list[str]], code: str, prediction_id: str) -> None:
    findings.setdefault(code, []).append(prediction_id)


def _finding(code: str, message: str, prediction_ids: list[str]) -> dict[str, Any]:
    unique_ids = sorted(set(prediction_ids))
    shown = unique_ids[:MAX_FINDING_IDS]
    return {
        "code": code,
        "message": message,
        "prediction_ids": shown,
        "prediction_id_count": len(unique_ids),
        "omitted_prediction_ids": len(unique_ids) - len(shown),
    }


def _gate_finding(code: str, actual: int, required: int) -> dict[str, Any]:
    return {
        "code": code,
        "message": f"{actual} available; at least {required} required",
        "actual": actual,
        "required": required,
    }


def _maximum_finding(code: str, actual: int, maximum: int) -> dict[str, Any]:
    return {
        "code": code,
        "message": f"{actual} present; at most {maximum} allowed",
        "actual": actual,
        "maximum": maximum,
    }


def _task_split_finding(code: str, split: str, tasks: set[str]) -> dict[str, Any]:
    return {
        "code": code,
        "message": f"built-in tasks have no unique {split} examples",
        "tasks": sorted(tasks),
    }


def _excluded_message(code: str) -> str:
    return {
        "dataset_format_mismatch": "rows use an unsupported captured dataset format",
        "invalid_corrected_output": "corrected outputs do not match the current task schema",
        "invalid_input": "rows have empty or invalid input text",
        "invalid_metadata": "rows do not contain metadata objects",
        "missing_prediction_id": "rows have no prediction provenance identifier",
        "not_corrected": "rows are not backed by a human correction",
        "task_contract_mismatch": "rows were captured under a different task contract",
        "unknown_task": "rows reference tasks unavailable in the current plugin",
    }[code]
