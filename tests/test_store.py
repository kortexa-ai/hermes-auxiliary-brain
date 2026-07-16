from __future__ import annotations

import json
import sqlite3

import pytest

from auxiliary_brain.store import BrainStore


def populated_store(tmp_path) -> tuple[BrainStore, str, str]:
    store = BrainStore(tmp_path / "brain.sqlite3")
    event = store.record_event(
        kind="explicit",
        task_key="generic_extract",
        input_text="Capture this task",
        session_id="session-1",
        metadata={"source": "test"},
        event_id="evt_1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    prediction = store.record_prediction(
        event_id=event.id,
        task_key="generic_extract",
        output={"summary": "Initial", "confidence": 0.4},
        raw_output='{"summary":"Initial","confidence":0.4}',
        model="tiny",
        base_url="http://localhost:1234/v1",
        latency_ms=12.5,
        prediction_id="pred_1",
        created_at="2026-01-01T00:00:01+00:00",
    )
    return store, event.id, prediction.id


def test_round_trip_event_prediction_and_correction(tmp_path) -> None:
    store, event_id, prediction_id = populated_store(tmp_path)
    correction = store.record_correction(
        prediction_id=prediction_id,
        corrected={"summary": "Corrected", "confidence": 1.0},
        note="human reviewed",
        correction_id="corr_1",
        created_at="2026-01-01T00:00:02+00:00",
    )

    event = store.get_event(event_id)
    prediction = store.get_prediction(prediction_id)

    assert event is not None
    assert event.input_text == "Capture this task"
    assert event.metadata == {"source": "test"}
    assert prediction is not None
    assert prediction.confidence == 0.4
    assert prediction.raw_output is not None
    assert store.corrections_for(prediction_id) == [correction]


def test_prediction_infers_confidence_from_output(tmp_path) -> None:
    store = BrainStore(tmp_path / "brain.sqlite3")
    event = store.record_event(kind="test", input_text="x")

    prediction = store.record_prediction(
        event_id=event.id,
        task_key="route",
        output={"confidence": 0.75},
    )

    assert prediction.confidence == 0.75


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_prediction_rejects_out_of_range_confidence(tmp_path, confidence: float) -> None:
    store = BrainStore(tmp_path / "brain.sqlite3")
    event = store.record_event(kind="test", input_text="x")

    with pytest.raises(ValueError, match="between 0 and 1"):
        store.record_prediction(
            event_id=event.id,
            task_key="route",
            output={},
            confidence=confidence,
        )


def test_foreign_keys_reject_orphan_records(tmp_path) -> None:
    store = BrainStore(tmp_path / "brain.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        store.record_prediction(event_id="missing", task_key="route", output={})
    with pytest.raises(sqlite3.IntegrityError):
        store.record_correction(prediction_id="missing", corrected={})


def test_training_examples_require_correction_by_default(tmp_path) -> None:
    store, _, prediction_id = populated_store(tmp_path)

    assert store.training_examples() == []

    store.record_correction(
        prediction_id=prediction_id,
        corrected={"summary": "Human answer", "confidence": 1.0},
        note="approved",
    )
    examples = store.training_examples()

    assert examples == [
        {
            "dataset_format_version": 1,
            "task": "generic_extract",
            "input": "Capture this task",
            "output": {"summary": "Human answer", "confidence": 1.0},
            "metadata": {"source": "test"},
            "corrected": True,
            "note": "approved",
            "prediction_id": "pred_1",
            "model": "tiny",
        }
    ]


def test_latest_correction_wins_deterministically(tmp_path) -> None:
    store, _, prediction_id = populated_store(tmp_path)
    store.record_correction(
        prediction_id=prediction_id,
        corrected={"version": 1},
        correction_id="corr_old",
        created_at="2026-01-01T00:00:02+00:00",
    )
    store.record_correction(
        prediction_id=prediction_id,
        corrected={"version": 2},
        correction_id="corr_new",
        created_at="2026-01-01T00:00:03+00:00",
    )

    assert store.training_examples()[0]["output"] == {"version": 2}


def test_uncorrected_examples_can_be_requested(tmp_path) -> None:
    store, _, _ = populated_store(tmp_path)

    examples = store.training_examples(corrected_only=False)

    assert len(examples) == 1
    assert examples[0]["corrected"] is False
    assert examples[0]["output"] == {"summary": "Initial", "confidence": 0.4}


def test_recent_predictions_filters_by_task_and_validates_limit(tmp_path) -> None:
    store, _, _ = populated_store(tmp_path)
    other_event = store.record_event(kind="test", input_text="other")
    store.record_prediction(
        event_id=other_event.id,
        task_key="route",
        output={},
        prediction_id="pred_2",
        created_at="2026-01-02T00:00:00+00:00",
    )

    assert [item.id for item in store.recent_predictions(limit=1)] == ["pred_2"]
    assert [item.id for item in store.recent_predictions(task_key="generic_extract")] == ["pred_1"]
    with pytest.raises(ValueError, match="limit"):
        store.recent_predictions(limit=0)


def test_stats_and_jsonl_export(tmp_path) -> None:
    store, _, prediction_id = populated_store(tmp_path)
    store.record_correction(prediction_id=prediction_id, corrected={"summary": "Final"})
    destination = tmp_path / "exports" / "examples.jsonl"

    count = store.export_jsonl(destination)
    lines = destination.read_text(encoding="utf-8").splitlines()

    assert count == 1
    assert len(lines) == 1
    assert json.loads(lines[0])["output"] == {"summary": "Final"}
    assert store.stats() == {
        "events": 1,
        "predictions": 1,
        "corrections": 1,
        "corrected_predictions": 1,
        "by_task": {"generic_extract": 1},
    }


def test_reopening_database_preserves_records(tmp_path) -> None:
    path = tmp_path / "brain.sqlite3"
    first = BrainStore(path)
    event = first.record_event(kind="test", input_text="persistent")

    second = BrainStore(path)

    assert second.get_event(event.id) == event


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_store_rejects_non_finite_metadata_and_predictions(tmp_path, value: float) -> None:
    store = BrainStore(tmp_path / "brain.sqlite3")

    with pytest.raises(ValueError, match="Out of range float values"):
        store.record_event(kind="test", input_text="x", metadata={"value": value})

    event = store.record_event(kind="test", input_text="valid")
    with pytest.raises(ValueError, match="Out of range float values"):
        store.record_prediction(
            event_id=event.id,
            task_key="route",
            output={"value": value},
        )
    assert store.stats()["predictions"] == 0


def test_export_rejects_non_finite_json_already_in_database(tmp_path) -> None:
    store, _, prediction_id = populated_store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE predictions SET output_json = ? WHERE id = ?",
            ('{"value":NaN}', prediction_id),
        )
    destination = tmp_path / "should-not-exist.jsonl"

    with pytest.raises(ValueError, match="non-finite JSON number"):
        store.export_jsonl(destination, corrected_only=False)

    assert not destination.exists()
