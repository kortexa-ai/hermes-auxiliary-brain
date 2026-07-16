"""Local SQLite record of auxiliary-brain inputs, outputs, and corrections."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DATASET_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class EventRecord:
    id: str
    session_id: str | None
    kind: str
    task_key: str | None
    input_text: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    id: str
    event_id: str
    task_key: str
    output: dict[str, Any]
    raw_output: str | None
    model: str | None
    base_url: str | None
    latency_ms: float | None
    confidence: float | None
    created_at: str


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    id: str
    prediction_id: str
    corrected: dict[str, Any]
    note: str | None
    created_at: str


class BrainStore:
    """Thread-safe-by-connection SQLite persistence.

    A connection is opened per operation.  This is a little less clever than a
    connection pool and a lot harder to wedge when gateway threads get lively.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record_event(
        self,
        *,
        kind: str,
        input_text: str,
        session_id: str | None = None,
        task_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> EventRecord:
        kind = kind.strip()
        if not kind:
            raise ValueError("event kind cannot be empty")
        record = EventRecord(
            id=event_id or _new_id("evt"),
            session_id=session_id,
            kind=kind,
            task_key=task_key,
            input_text=input_text,
            metadata=dict(metadata or {}),
            created_at=created_at or _utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events
                    (id, session_id, kind, task_key, input_text, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.kind,
                    record.task_key,
                    record.input_text,
                    _dump(record.metadata),
                    record.created_at,
                ),
            )
        return record

    def record_prediction(
        self,
        *,
        event_id: str,
        task_key: str,
        output: Mapping[str, Any],
        raw_output: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        latency_ms: float | None = None,
        confidence: float | None = None,
        prediction_id: str | None = None,
        created_at: str | None = None,
    ) -> PredictionRecord:
        task_key = task_key.strip()
        if not task_key:
            raise ValueError("prediction task_key cannot be empty")
        output_dict = dict(output)
        if confidence is None:
            candidate = output_dict.get("confidence")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                confidence = float(candidate)
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        record = PredictionRecord(
            id=prediction_id or _new_id("pred"),
            event_id=event_id,
            task_key=task_key,
            output=output_dict,
            raw_output=raw_output,
            model=model,
            base_url=base_url,
            latency_ms=latency_ms,
            confidence=confidence,
            created_at=created_at or _utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions
                    (id, event_id, task_key, output_json, raw_output, model,
                     base_url, latency_ms, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.event_id,
                    record.task_key,
                    _dump(record.output),
                    record.raw_output,
                    record.model,
                    record.base_url,
                    record.latency_ms,
                    record.confidence,
                    record.created_at,
                ),
            )
        return record

    def record_correction(
        self,
        *,
        prediction_id: str,
        corrected: Mapping[str, Any],
        note: str | None = None,
        correction_id: str | None = None,
        created_at: str | None = None,
    ) -> CorrectionRecord:
        record = CorrectionRecord(
            id=correction_id or _new_id("corr"),
            prediction_id=prediction_id,
            corrected=dict(corrected),
            note=note,
            created_at=created_at or _utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO corrections
                    (id, prediction_id, corrected_json, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.prediction_id,
                    _dump(record.corrected),
                    record.note,
                    record.created_at,
                ),
            )
        return record

    def get_event(self, event_id: str) -> EventRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _event_from_row(row) if row else None

    def get_prediction(self, prediction_id: str) -> PredictionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        return _prediction_from_row(row) if row else None

    def corrections_for(self, prediction_id: str) -> list[CorrectionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM corrections
                WHERE prediction_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (prediction_id,),
            ).fetchall()
        return [_correction_from_row(row) for row in rows]

    def recent_predictions(
        self,
        *,
        limit: int = 50,
        task_key: str | None = None,
    ) -> list[PredictionRecord]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        query = "SELECT * FROM predictions"
        params: list[Any] = []
        if task_key:
            query += " WHERE task_key = ?"
            params.append(task_key)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_prediction_from_row(row) for row in rows]

    def training_examples(
        self,
        *,
        task_key: str | None = None,
        corrected_only: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return deterministic examples, preferring the latest correction."""

        if limit is not None and not 1 <= limit <= 1_000_000:
            raise ValueError("limit must be between 1 and 1000000")
        query = """
            SELECT
                e.id AS event_id,
                e.input_text,
                e.metadata_json,
                p.id AS prediction_id,
                p.task_key,
                p.output_json,
                p.model,
                p.created_at,
                c.id AS correction_id,
                c.corrected_json,
                c.note AS correction_note
            FROM predictions AS p
            JOIN events AS e ON e.id = p.event_id
            LEFT JOIN corrections AS c ON c.id = (
                SELECT c2.id
                FROM corrections AS c2
                WHERE c2.prediction_id = p.id
                ORDER BY c2.created_at DESC, c2.rowid DESC
                LIMIT 1
            )
            WHERE (? IS NULL OR p.task_key = ?)
              AND (? = 0 OR c.id IS NOT NULL)
            ORDER BY p.created_at ASC, p.rowid ASC
        """
        params: list[Any] = [task_key, task_key, int(corrected_only)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        examples: list[dict[str, Any]] = []
        for row in rows:
            corrected = row["corrected_json"] is not None
            output_json = row["corrected_json"] if corrected else row["output_json"]
            examples.append(
                {
                    "dataset_format_version": DATASET_FORMAT_VERSION,
                    "task": row["task_key"],
                    "input": row["input_text"],
                    "output": _load_object(output_json),
                    "metadata": _load_object(row["metadata_json"]),
                    "corrected": corrected,
                    "note": row["correction_note"],
                    "prediction_id": row["prediction_id"],
                    "model": row["model"],
                }
            )
        return examples

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM events) AS events,
                    (SELECT COUNT(*) FROM predictions) AS predictions,
                    (SELECT COUNT(*) FROM corrections) AS corrections,
                    (SELECT COUNT(DISTINCT prediction_id) FROM corrections)
                        AS corrected_predictions
                """
            ).fetchone()
            tasks = connection.execute(
                """
                SELECT task_key, COUNT(*) AS count
                FROM predictions
                GROUP BY task_key
                ORDER BY task_key
                """
            ).fetchall()
        return {
            "events": row["events"],
            "predictions": row["predictions"],
            "corrections": row["corrections"],
            "corrected_predictions": row["corrected_predictions"],
            "by_task": {item["task_key"]: item["count"] for item in tasks},
        }

    def export_jsonl(
        self,
        path: str | Path,
        *,
        task_key: str | None = None,
        corrected_only: bool = True,
    ) -> int:
        """Export fine-tuning/evaluation examples without SDK assumptions."""

        examples = self.training_examples(task_key=task_key, corrected_only=corrected_only)
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        return len(examples)

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, SCHEMA_VERSION}:
                raise RuntimeError(f"unsupported auxiliary-brain database version: {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    kind TEXT NOT NULL,
                    task_key TEXT,
                    input_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    task_key TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    raw_output TEXT,
                    model TEXT,
                    base_url TEXT,
                    latency_ms REAL,
                    confidence REAL CHECK(confidence IS NULL OR
                        (confidence >= 0 AND confidence <= 1)),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL
                        REFERENCES predictions(id) ON DELETE CASCADE,
                    corrected_json TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON events(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_task
                    ON predictions(task_key, created_at);
                CREATE INDEX IF NOT EXISTS idx_corrections_prediction
                    ON corrections(prediction_id, created_at);
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _dump(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value, parse_constant=_reject_json_constant)
    if not isinstance(decoded, dict):
        raise RuntimeError("stored JSON value is not an object")
    return decoded


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _event_from_row(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        id=row["id"],
        session_id=row["session_id"],
        kind=row["kind"],
        task_key=row["task_key"],
        input_text=row["input_text"],
        metadata=_load_object(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _prediction_from_row(row: sqlite3.Row) -> PredictionRecord:
    return PredictionRecord(
        id=row["id"],
        event_id=row["event_id"],
        task_key=row["task_key"],
        output=_load_object(row["output_json"]),
        raw_output=row["raw_output"],
        model=row["model"],
        base_url=row["base_url"],
        latency_ms=row["latency_ms"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


def _correction_from_row(row: sqlite3.Row) -> CorrectionRecord:
    return CorrectionRecord(
        id=row["id"],
        prediction_id=row["prediction_id"],
        corrected=_load_object(row["corrected_json"]),
        note=row["note"],
        created_at=row["created_at"],
    )
