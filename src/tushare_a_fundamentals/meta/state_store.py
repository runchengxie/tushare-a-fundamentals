"""State store backed by SQLite.

This module manages ingestion progress for each dataset and partition year.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from typing import Optional

DATASET_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_state (
  dataset TEXT NOT NULL,
  part_year INTEGER NOT NULL,
  min_key TEXT,
  max_key TEXT,
  last_checked_at TEXT,
  last_success_at TEXT,
  dirty INTEGER DEFAULT 0,
  PRIMARY KEY (dataset, part_year)
);
"""

WATERMARKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
  dataset TEXT PRIMARY KEY,
  low TEXT,
  high TEXT
);
"""

KV_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_state (
  dataset TEXT NOT NULL,
  state_key TEXT NOT NULL,
  state_value TEXT,
  PRIMARY KEY (dataset, state_key)
);
"""


def init_state_store(path: str | Path) -> Connection:
    """Initialise a SQLite-backed state store.

    Parameters
    ----------
    path: str | Path
        File path to the SQLite database. Parent directories will be created
        automatically.
    """
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p.as_posix())
    cur = conn.cursor()
    cur.execute(DATASET_STATE_SCHEMA)
    cur.execute(WATERMARKS_SCHEMA)
    cur.execute(KV_STATE_SCHEMA)
    conn.commit()
    return conn


def upsert_kv_state(
    conn: Connection, dataset: str, key: str, value: str
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO kv_state(dataset, state_key, state_value)
        VALUES (?, ?, ?)
        ON CONFLICT(dataset, state_key) DO UPDATE SET
            state_value=excluded.state_value
        """,
        (dataset, key, value),
    )
    conn.commit()


def delete_kv_state(
    conn: Connection, dataset: str, key: str | None = None
) -> None:
    cur = conn.cursor()
    if key is None:
        cur.execute("DELETE FROM kv_state WHERE dataset=?", (dataset,))
    else:
        cur.execute(
            "DELETE FROM kv_state WHERE dataset=? AND state_key=?",
            (dataset, key),
        )
    conn.commit()


def fetch_all_kv_state(
    conn: Connection, dataset: str | None = None
) -> list[tuple[str, str, str]]:
    cur = conn.cursor()
    if dataset is None:
        cur.execute(
            "SELECT dataset, state_key, state_value FROM kv_state"
            " ORDER BY dataset, state_key"
        )
    else:
        cur.execute(
            "SELECT dataset, state_key, state_value FROM kv_state"
            " WHERE dataset=? ORDER BY state_key",
            (dataset,),
        )
    return cur.fetchall()


@dataclass(eq=True, frozen=True)
class DatasetState:
    dataset: str
    part_year: int
    min_key: Optional[str] = None
    max_key: Optional[str] = None
    last_checked_at: Optional[str] = None
    last_success_at: Optional[str] = None
    dirty: int = 0


def upsert_dataset_state(conn: Connection, state: DatasetState) -> None:
    """Insert or update a dataset_state row."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dataset_state(dataset, part_year, min_key, max_key,
                                 last_checked_at, last_success_at, dirty)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset, part_year) DO UPDATE SET
            min_key=excluded.min_key,
            max_key=excluded.max_key,
            last_checked_at=excluded.last_checked_at,
            last_success_at=excluded.last_success_at,
            dirty=excluded.dirty
        """,
        (
            state.dataset,
            state.part_year,
            state.min_key,
            state.max_key,
            state.last_checked_at,
            state.last_success_at,
            state.dirty,
        ),
    )
    conn.commit()


def get_dataset_state(
    conn: Connection, dataset: str, part_year: int
) -> Optional[DatasetState]:
    """Fetch a DatasetState row if present."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT dataset, part_year, min_key, max_key, last_checked_at,
               last_success_at, dirty
        FROM dataset_state
        WHERE dataset=? AND part_year=?
        """,
        (dataset, part_year),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return DatasetState(*row)
