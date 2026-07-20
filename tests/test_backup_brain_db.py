from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.backup_brain_db import backup_database


def test_backup_database_copies_live_wal_database(tmp_path: Path) -> None:
    source = tmp_path / "brain.db"
    destination = tmp_path / "transfer" / "brain.db"
    with sqlite3.connect(source) as database:
        database.execute("PRAGMA journal_mode = WAL")
        database.execute("CREATE TABLE examples (value TEXT NOT NULL)")
        database.execute("INSERT INTO examples VALUES ('portable goblin')")
        database.commit()
        report = backup_database(source, destination)

    with sqlite3.connect(destination) as database:
        assert database.execute("SELECT value FROM examples").fetchone() == ("portable goblin",)
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert report["path"] == str(destination)
    assert report["bytes"] == destination.stat().st_size
    assert len(str(report["sha256"])) == 64


def test_backup_database_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "brain.db"
    destination = tmp_path / "backup.db"
    with sqlite3.connect(source) as database:
        database.execute("CREATE TABLE examples (value TEXT)")
    destination.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        backup_database(source, destination)

    assert destination.read_bytes() == b"keep me"


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_backup_database_refuses_source_sidecars(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "brain.db"
    with sqlite3.connect(source) as database:
        database.execute("CREATE TABLE examples (value TEXT)")
    sidecar = Path(f"{source}{suffix}")
    sidecar.write_bytes(b"keep me")

    with pytest.raises(ValueError, match="sidecar"):
        backup_database(source, sidecar, force=True)

    assert sidecar.read_bytes() == b"keep me"
