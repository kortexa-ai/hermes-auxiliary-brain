from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from auxiliary_brain import trainer_backend, training_data
from auxiliary_brain.runtime import _task_contract_hash as runtime_task_contract_hash
from auxiliary_brain.store import BrainStore
from auxiliary_brain.tasks import get_task, list_tasks
from auxiliary_brain.training_data import (
    TrainingDataError,
    inspect_readiness,
    normalize_duplicate_input,
    prepare_bundle,
    task_contract_hash,
)
from auxiliary_brain.version import __version__


def _output(index: int) -> dict[str, object]:
    return {
        "summary": f"Summary {index}",
        "category": "note",
        "entities": [],
        "action_items": [],
        "fields": {},
        "confidence": 1.0,
    }


def _task_output(task_key: str, index: int) -> dict[str, object]:
    if task_key == "route":
        return {
            "target": "local",
            "task": "generic_extract",
            "reason": f"Narrow extraction {index}",
            "confidence": 1.0,
        }
    if task_key == "progress_checkin":
        return {
            "category": "practice",
            "outcome": "completed",
            "quantity": 1,
            "unit": "session",
            "occurred_at": None,
            "note": f"Completed session {index}",
            "next_action": None,
            "confidence": 1.0,
        }
    if task_key == "follow_up":
        return {
            "title": f"Follow up {index}",
            "status": "todo",
            "contact": None,
            "due_at": None,
            "next_action": None,
            "tags": [],
            "confidence": 1.0,
        }
    if task_key == "research_note":
        return {
            "topic": f"Topic {index}",
            "entities": [],
            "source": None,
            "claims": [],
            "questions": [],
            "next_action": None,
            "due_at": None,
            "needs_verification": False,
            "requires_high_stakes_judgment": False,
            "confidence": 1.0,
        }
    return _output(index)


def _add_example(
    store: BrainStore,
    index: int,
    *,
    task_key: str = "generic_extract",
    text: str | None = None,
    source: str = "cli",
    metadata: dict[str, object] | None = None,
    output: dict[str, object] | None = None,
    contract_hash: str | None = None,
) -> str:
    task = get_task(task_key)
    captured_output = output or _task_output(task_key, index)
    prediction_id = f"pred_{index:03d}"
    event = store.record_event(
        kind="local_task",
        task_key=task.key,
        input_text=text or f"Training example {index}",
        metadata={
            "source": source,
            "task_contract_hash": contract_hash or task_contract_hash(task),
            **(metadata or {}),
        },
        event_id=f"evt_{index:03d}",
        created_at=f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
    )
    store.record_prediction(
        event_id=event.id,
        task_key=task.key,
        output=captured_output,
        prediction_id=prediction_id,
        created_at=f"2026-01-01T01:{index // 60:02d}:{index % 60:02d}+00:00",
    )
    store.record_correction(
        prediction_id=prediction_id,
        corrected=captured_output,
        correction_id=f"corr_{index:03d}",
        created_at=f"2026-01-01T02:{index // 60:02d}:{index % 60:02d}+00:00",
    )
    return prediction_id


def _store_with_examples(tmp_path: Path, count: int = 60) -> BrainStore:
    store = BrainStore(tmp_path / "brain.db")
    other_tasks = [task.key for task in list_tasks() if task.key != "generic_extract"]
    generic_count = count - 2 * len(other_tasks)
    if generic_count < 2:
        raise ValueError("count is too small for full built-in task coverage")
    for index in range(generic_count):
        text = None
        if index < 2:
            split = ("train", "holdout")[index]
            text = _text_for_split(f"generic {split}", split)
        _add_example(store, index, text=text)
    index = generic_count
    for task_key in other_tasks:
        for split in ("train", "holdout"):
            _add_example(
                store,
                index,
                task_key=task_key,
                text=_text_for_split(f"{task_key} {split}", split),
            )
            index += 1
    return store


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _finding_codes(report: dict[str, object], kind: str) -> set[str]:
    return {item["code"] for item in report[kind]}  # type: ignore[index, union-attr]


def _text_for_split(prefix: str, split: str) -> str:
    for index in range(10_000):
        text = f"{prefix} {index}"
        group_id = hashlib.sha256(normalize_duplicate_input(text).encode("utf-8")).hexdigest()
        if training_data._split_for_group(group_id, 42, 20) == split:
            return text
    raise AssertionError(f"could not find deterministic {split} text")


def test_readiness_uses_current_corrected_contract_and_stable_split(tmp_path: Path) -> None:
    store = _store_with_examples(tmp_path)

    first = inspect_readiness(store)
    second = inspect_readiness(store)

    assert first == second
    assert first["ready"] is True
    assert first["promotable"] is True
    assert first["counts"]["eligible"] == 60
    assert first["counts"]["train"] + first["counts"]["holdout"] == 60
    assert first["task_contract_hashes"] == {
        task.key: task_contract_hash(task) for task in list_tasks()
    }
    assert task_contract_hash(get_task("generic_extract")) == runtime_task_contract_hash(
        get_task("generic_extract")
    )


def test_prepare_writes_exact_messages_canonical_json_and_manifest(tmp_path: Path) -> None:
    store = _store_with_examples(tmp_path)
    root = tmp_path / "bundles"

    result = prepare_bundle(
        store,
        root,
        model="LiquidAI/LFM2.5-230M",
        revision="0123456789abcdef",
    )
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    rows = _rows(result["path"] / "train.jsonl") + _rows(result["path"] / "holdout.jsonl")
    row = next(item for item in rows if item["metadata"]["prediction_id"] == "pred_007")

    assert result["created"] is True
    assert row["messages"][:2] == get_task("generic_extract").build_messages("Training example 7")
    assert row["messages"][2] == {
        "role": "assistant",
        "content": json.dumps(
            _output(7), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }
    assert manifest["format_version"] == 1
    assert manifest["plugin_version"] == __version__
    assert manifest["model"] == {
        "id": "LiquidAI/LFM2.5-230M",
        "revision": "0123456789abcdef",
    }
    assert manifest["preparation"] == {
        "acknowledge_unattributed_gateway": False,
        "allow_small": False,
        "task_key": None,
        "thresholds": {
            "min_holdout_examples": 4,
            "min_train_examples": 16,
            "min_unique_examples": 20,
        },
    }
    assert (
        result["readiness"]["trainer_usage"]["train_examples"]
        == manifest["files"]["train.jsonl"]["examples"]
    )
    assert (
        result["readiness"]["trainer_usage"]["train_bytes"]
        == manifest["files"]["train.jsonl"]["bytes"]
    )
    for name, details in manifest["files"].items():
        assert hashlib.sha256((result["path"] / name).read_bytes()).hexdigest() == details["sha256"]
    all_ids = [row["metadata"]["prediction_id"] for row in rows]
    assert sorted(all_ids) == [f"pred_{index:03d}" for index in range(60)]
    backend_bundle = trainer_backend.inspect_training_bundle(result["path"])
    assert backend_bundle.examples == manifest["files"]["train.jsonl"]["examples"]
    assert backend_bundle.manifest_sha256 == result["manifest_sha256"]
    assert not list(root.glob(".staging-*"))

    repeated = prepare_bundle(
        store,
        root,
        model="LiquidAI/LFM2.5-230M",
        revision="0123456789abcdef",
    )
    assert repeated["created"] is False
    assert repeated["path"] == result["path"]
    assert repeated["manifest_sha256"] == result["manifest_sha256"]


def test_nfkc_whitespace_casefold_duplicates_never_cross_splits(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    variants = ["Ｆｏｏ   BAR", "  foo\nbar  ", "Foo Bar"]
    for index, text in enumerate(variants):
        _add_example(store, index, text=text)

    assert {normalize_duplicate_input(value) for value in variants} == {"foo bar"}
    result = prepare_bundle(
        store,
        tmp_path / "bundles",
        model="model",
        revision="revision",
        allow_small=True,
    )
    sources = {
        name.removesuffix(".jsonl"): [
            row["metadata"]["prediction_id"] for row in _rows(result["path"] / name)
        ]
        for name in ("train.jsonl", "holdout.jsonl")
    }
    populated_splits = [name for name in ("train", "holdout") if sources[name]]
    assert len(populated_splits) == 1
    assert sorted(sources[populated_splits[0]]) == ["pred_000", "pred_001", "pred_002"]
    assert result["manifest"]["counts"]["unique_examples"] == 1
    assert result["manifest"]["promotion"] == {
        "experimental": True,
        "promotable": False,
    }


def test_bundle_bytes_do_not_depend_on_database_insertion_order(tmp_path: Path) -> None:
    first_store = BrainStore(tmp_path / "first.db")
    second_store = BrainStore(tmp_path / "second.db")
    for index in range(12):
        _add_example(first_store, index)
    for index in reversed(range(12)):
        _add_example(second_store, index)

    first = prepare_bundle(
        first_store,
        tmp_path / "first",
        model="model",
        revision="revision",
        allow_small=True,
    )
    second = prepare_bundle(
        second_store,
        tmp_path / "second",
        model="model",
        revision="revision",
        allow_small=True,
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    for name in ("manifest.json", "train.jsonl", "holdout.jsonl"):
        assert (first["path"] / name).read_bytes() == (second["path"] / name).read_bytes()


def test_failed_final_rename_removes_incomplete_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0)
    root = tmp_path / "bundles"

    def fail_rename(source: Path, destination: Path) -> None:
        raise OSError(f"pretend rename failure for {source.name} to {destination.name}")

    monkeypatch.setattr(training_data.os, "replace", fail_rename)
    with pytest.raises(OSError, match="pretend rename failure"):
        prepare_bundle(
            store,
            root,
            model="model",
            revision="revision",
            allow_small=True,
        )

    assert not list(root.iterdir())


def test_stale_contract_and_invalid_schema_are_excluded(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, contract_hash="0" * 64)
    _add_example(store, 1, output={"summary": "missing required fields"})
    _add_example(store, 2)

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is True
    assert report["counts"] == {
        "corrected": 3,
        "eligible": 1,
        "excluded": 2,
        "unique_examples": 1,
        "train": report["counts"]["train"],
        "train_unique": report["counts"]["train_unique"],
        "holdout": report["counts"]["holdout"],
        "holdout_unique": report["counts"]["holdout_unique"],
    }
    assert {"task_contract_mismatch", "invalid_corrected_output"} <= _finding_codes(
        report, "warnings"
    )


def test_unattributed_gateway_rows_block_until_explicit_acknowledgement(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, source="gateway-slash")

    blocked = inspect_readiness(store, allow_small=True)
    acknowledged = inspect_readiness(
        store,
        allow_small=True,
        acknowledge_unattributed_gateway=True,
    )

    assert blocked["ready"] is False
    assert "unattributed_gateway_slash" in _finding_codes(blocked, "errors")
    assert acknowledged["ready"] is True
    assert "unattributed_gateway_slash" in _finding_codes(acknowledged, "warnings")


def test_attributed_gateway_row_does_not_need_acknowledgement(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(
        store,
        0,
        source="gateway-slash",
        metadata={"sender_id": "local-owner"},
    )

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is True
    assert "unattributed_gateway_slash" not in _finding_codes(report, "errors")


@pytest.mark.parametrize("source", ["pre_llm_call", "pre_llm_call_assist"])
def test_gateway_hook_capture_requires_sender_attribution(
    tmp_path: Path,
    source: str,
) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, source=source, metadata={"platform": "telegram"})

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is False
    assert "unattributed_gateway_slash" in _finding_codes(report, "errors")


def test_gateway_hook_capture_with_sender_is_attributed(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(
        store,
        0,
        source="pre_llm_call",
        metadata={"platform": "discord", "sender_id": "discord-user-1"},
    )

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is True
    assert "unattributed_gateway_slash" not in _finding_codes(report, "errors")


@pytest.mark.parametrize(
    "text",
    [
        "Use api_key=supersecretcredential123 for the request",
        "OpenAI key sk-abcdefghijklmnopqrstuvwxyz",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        "Token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
    ],
)
def test_likely_secret_lint_blocks_without_echoing_secret(tmp_path: Path, text: str) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, text=text)

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is False
    assert "likely_secret" in _finding_codes(report, "errors")
    assert "supersecretcredential123" not in json.dumps(report)
    with pytest.raises(TrainingDataError) as raised:
        prepare_bundle(
            store,
            tmp_path / "bundles",
            model="model",
            revision="revision",
            allow_small=True,
        )
    assert raised.value.report == report


def test_minimum_gates_require_explicit_non_promotable_allow_small(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0)

    strict = inspect_readiness(store)
    experimental = inspect_readiness(store, allow_small=True)

    assert strict["ready"] is False
    assert {
        "min_unique_examples",
        "min_train_examples",
        "min_holdout_examples",
    } <= _finding_codes(strict, "errors")
    assert experimental["ready"] is True
    assert experimental["experimental"] is True
    assert experimental["promotable"] is False
    assert {
        "min_unique_examples",
        "min_train_examples",
        "min_holdout_examples",
    } <= _finding_codes(experimental, "warnings")


def test_lowered_minimums_are_always_experimental(tmp_path: Path) -> None:
    store = _store_with_examples(tmp_path)

    report = inspect_readiness(
        store,
        min_unique_examples=1,
        min_train_examples=1,
        min_holdout_examples=1,
    )

    assert report["ready"] is True
    assert report["experimental"] is True
    assert report["promotable"] is False


@pytest.mark.parametrize(
    ("route_split", "expected_code"),
    [
        ("train", "task_missing_holdout"),
        ("holdout", "task_missing_train"),
    ],
)
def test_each_task_requires_unique_train_and_holdout_coverage(
    tmp_path: Path,
    route_split: str,
    expected_code: str,
) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, text=_text_for_split("generic train", "train"))
    _add_example(store, 1, text=_text_for_split("generic holdout", "holdout"))
    _add_example(
        store,
        2,
        task_key="route",
        text=_text_for_split(f"route {route_split}", route_split),
        output={
            "target": "local",
            "task": "generic_extract",
            "reason": "Narrow extraction",
            "confidence": 1.0,
        },
    )
    options = {
        "min_unique_examples": 1,
        "min_train_examples": 1,
        "min_holdout_examples": 1,
    }

    strict = inspect_readiness(store, **options)
    experimental = inspect_readiness(store, allow_small=True, **options)

    assert strict["ready"] is False
    assert expected_code in _finding_codes(strict, "errors")
    missing = next(item for item in strict["errors"] if item["code"] == expected_code)
    assert "route" in missing["tasks"]
    assert experimental["ready"] is True
    assert expected_code in _finding_codes(experimental, "warnings")
    assert experimental["experimental"] is True
    assert experimental["promotable"] is False


def test_selected_task_dataset_requires_allow_small_when_other_builtins_are_missing(
    tmp_path: Path,
) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0, text=_text_for_split("generic train", "train"))
    _add_example(store, 1, text=_text_for_split("generic holdout", "holdout"))
    options = {
        "task_key": "generic_extract",
        "min_unique_examples": 1,
        "min_train_examples": 1,
        "min_holdout_examples": 1,
    }

    strict = inspect_readiness(store, **options)
    experimental = inspect_readiness(store, allow_small=True, **options)
    missing_builtins = {task.key for task in list_tasks()} - {"generic_extract"}

    assert strict["ready"] is False
    for code in ("task_missing_train", "task_missing_holdout"):
        finding = next(item for item in strict["errors"] if item["code"] == code)
        assert set(finding["tasks"]) == missing_builtins
        assert code in _finding_codes(experimental, "warnings")
    assert "route" in missing_builtins
    assert experimental["ready"] is True
    assert experimental["experimental"] is True
    assert experimental["promotable"] is False


def test_readiness_reports_dependency_free_trainer_limits(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0)

    report = inspect_readiness(store, allow_small=True)

    assert report["trainer_limits"] == {
        "max_source_examples": 100_000,
        "max_source_bytes": 64 * 1024 * 1024,
        "max_train_examples": 100_000,
        "max_train_bytes": 64 * 1024 * 1024,
        "max_holdout_bytes": 32 * 1024 * 1024,
        "max_row_bytes": 1024 * 1024,
    }
    assert report["trainer_usage"]["train_examples"] == report["counts"]["train"]


@pytest.mark.parametrize(
    ("constant", "maximum", "expected_code"),
    [
        ("MAX_EXAMPLES", 1, "max_source_examples"),
        ("MAX_TRAIN_BYTES", 1, "max_train_bytes"),
        ("MAX_ROW_BYTES", 1, "max_source_row_bytes"),
    ],
)
def test_trainer_limits_block_before_bundle_files_are_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    maximum: int,
    expected_code: str,
) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0)
    _add_example(store, 1)
    root = tmp_path / "bundles"
    monkeypatch.setattr(training_data, constant, maximum)
    monkeypatch.setattr(training_data, "_split_for_group", lambda *_args: "train")

    report = inspect_readiness(store, allow_small=True)

    assert report["ready"] is False
    assert expected_code in _finding_codes(report, "errors")
    with pytest.raises(TrainingDataError) as raised:
        prepare_bundle(
            store,
            root,
            model="model",
            revision="revision",
            allow_small=True,
        )
    assert expected_code in _finding_codes(raised.value.report, "errors")
    assert not root.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not available on Windows")
def test_bundle_directories_and_sensitive_files_are_private(tmp_path: Path) -> None:
    store = BrainStore(tmp_path / "brain.db")
    _add_example(store, 0)

    result = prepare_bundle(
        store,
        tmp_path / "bundles",
        model="model",
        revision="revision",
        allow_small=True,
    )

    assert stat.S_IMODE((tmp_path / "bundles").stat().st_mode) == 0o700
    assert stat.S_IMODE(result["path"].stat().st_mode) == 0o700
    for name in ("manifest.json", "train.jsonl", "holdout.jsonl"):
        assert stat.S_IMODE((result["path"] / name).stat().st_mode) == 0o600
