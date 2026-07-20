"""Create a consistent, private backup of an auxiliary-brain SQLite database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def backup_database(source: Path, destination: Path, *, force: bool = False) -> dict[str, object]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")
    if source == destination:
        raise ValueError("source and destination must be different files")
    sidecars = {Path(f"{source}{suffix}").resolve() for suffix in ("-wal", "-shm", "-journal")}
    if destination in sidecars:
        raise ValueError("destination must not replace a SQLite sidecar of the source")
    if destination.exists() and not force:
        raise FileExistsError(f"destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        source_uri = f"{source.as_uri()}?mode=ro"
        with closing(sqlite3.connect(source_uri, uri=True)) as source_db:
            with closing(sqlite3.connect(temporary)) as backup_db:
                source_db.backup(backup_db)
                result = backup_db.execute("PRAGMA integrity_check").fetchone()
                if result is None or result[0] != "ok":
                    raise RuntimeError("backup failed SQLite integrity_check")
                backup_db.commit()
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    digest = _sha256_file(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="active brain.db path")
    parser.add_argument("destination", type=Path, help="new backup file")
    parser.add_argument("--force", action="store_true", help="replace an existing destination")
    args = parser.parse_args()
    try:
        report = backup_database(args.source, args.destination, force=args.force)
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
