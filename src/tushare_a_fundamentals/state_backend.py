from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .meta.state_store import fetch_all_kv_state, init_state_store, upsert_kv_state


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


class StateBackend(Protocol):
    """抽象状态后端接口。"""

    def get(self, dataset: str, key: str, default: str) -> str:
        ...

    def set(self, dataset: str, key: str, value: str) -> None:
        ...

    def delete(
        self, dataset: str, key: str | None = None, *, year: Optional[int] = None
    ) -> None:
        ...

    def snapshot(self, dataset: str | None = None) -> Dict[str, Any]:
        ...


class JsonStateBackend:
    """JSON 文件后端，保持原有行为。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            try:
                self.data: Dict[str, Dict[str, str]] = json.loads(
                    path.read_text("utf-8")
                )
            except json.JSONDecodeError:
                self.data = {}
        else:
            self.data = {}

    def get(self, dataset: str, key: str, default: str) -> str:
        bucket = self.data.get(dataset, {})
        return str(bucket.get(key, default))

    def set(self, dataset: str, key: str, value: str) -> None:
        bucket = self.data.setdefault(dataset, {})
        bucket[key] = value
        self._flush()

    def delete(
        self, dataset: str, key: str | None = None, *, year: Optional[int] = None
    ) -> None:
        if year is not None:
            # JSON 后端不支持基于年份的拆分，忽略该参数。
            pass
        if key is not None:
            bucket = self.data.get(dataset)
            if not bucket or key not in bucket:
                return
            del bucket[key]
            if not bucket:
                self.data.pop(dataset, None)
        else:
            self.data.pop(dataset, None)
        self._flush()

    def snapshot(self, dataset: str | None = None) -> Dict[str, Any]:
        if dataset:
            return dict(self.data.get(dataset, {}))
        return {ds: dict(values) for ds, values in self.data.items()}

    def _flush(self) -> None:
        _ensure_parent(self.path)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8"
        )


class SQLiteStateBackend:
    """SQLite 后端，统一管理 kv 状态与扩展信息。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        conn = init_state_store(path)
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.as_posix())

    def get(self, dataset: str, key: str, default: str) -> str:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT state_value FROM kv_state WHERE dataset=? AND state_key=?",
                (dataset, key),
            )
            row = cur.fetchone()
            if row is None:
                return default
            return str(row[0]) if row[0] is not None else default
        finally:
            conn.close()

    def set(self, dataset: str, key: str, value: str) -> None:
        conn = self._connect()
        try:
            upsert_kv_state(conn, dataset, key, value)
        finally:
            conn.close()

    def delete(
        self, dataset: str, key: str | None = None, *, year: Optional[int] = None
    ) -> None:
        conn = self._connect()
        try:
            if key is not None:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM kv_state WHERE dataset=? AND state_key=?",
                    (dataset, key),
                )
                conn.commit()
            else:
                cur = conn.cursor()
                if year is not None:
                    cur.execute(
                        "DELETE FROM dataset_state WHERE dataset=? AND part_year=?",
                        (dataset, year),
                    )
                    cur.execute("DELETE FROM watermarks WHERE dataset=?", (dataset,))
                else:
                    cur.execute("DELETE FROM dataset_state WHERE dataset=?", (dataset,))
                    cur.execute("DELETE FROM watermarks WHERE dataset=?", (dataset,))
                cur.execute("DELETE FROM kv_state WHERE dataset=?", (dataset,))
                conn.commit()
        finally:
            conn.close()

    def snapshot(self, dataset: str | None = None) -> Dict[str, Any]:
        conn = self._connect()
        try:
            payload: Dict[str, Any] = {}
            rows = fetch_all_kv_state(conn, dataset)
            kv: Dict[str, Dict[str, str]] = {}
            for ds, key, value in rows:
                bucket = kv.setdefault(ds, {})
                bucket[key] = value
            if dataset is None:
                payload["kv_state"] = {ds: dict(values) for ds, values in kv.items()}
            else:
                payload["kv_state"] = dict(kv.get(dataset, {}))

            cur = conn.cursor()
            if dataset is None:
                cur.execute(
                    "SELECT dataset, low, high FROM watermarks ORDER BY dataset"
                )
                payload["watermarks"] = [
                    {"dataset": ds, "low": low, "high": high}
                    for ds, low, high in cur.fetchall()
                ]
                cur.execute(
                    "SELECT dataset, part_year, min_key, max_key, last_checked_at,"
                    " last_success_at, dirty"
                    " FROM dataset_state ORDER BY dataset, part_year"
                )
                payload["dataset_state"] = [
                    {
                        "dataset": ds,
                        "part_year": part_year,
                        "min_key": min_key,
                        "max_key": max_key,
                        "last_checked_at": last_checked,
                        "last_success_at": last_success,
                        "dirty": dirty,
                    }
                    for (
                        ds,
                        part_year,
                        min_key,
                        max_key,
                        last_checked,
                        last_success,
                        dirty,
                    ) in cur.fetchall()
                ]
            else:
                cur.execute(
                    "SELECT dataset, low, high FROM watermarks WHERE dataset=?",
                    (dataset,),
                )
                payload["watermarks"] = [
                    {"dataset": ds, "low": low, "high": high}
                    for ds, low, high in cur.fetchall()
                ]
                cur.execute(
                    "SELECT dataset, part_year, min_key, max_key, last_checked_at,"
                    " last_success_at, dirty"
                    " FROM dataset_state WHERE dataset=? ORDER BY part_year",
                    (dataset,),
                )
                payload["dataset_state"] = [
                    {
                        "dataset": ds,
                        "part_year": part_year,
                        "min_key": min_key,
                        "max_key": max_key,
                        "last_checked_at": last_checked,
                        "last_success_at": last_success,
                        "dirty": dirty,
                    }
                    for (
                        ds,
                        part_year,
                        min_key,
                        max_key,
                        last_checked,
                        last_success,
                        dirty,
                    ) in cur.fetchall()
                ]
            return payload
        finally:
            conn.close()


__all__ = [
    "StateBackend",
    "JsonStateBackend",
    "SQLiteStateBackend",
]
