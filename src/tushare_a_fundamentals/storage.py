"""Storage utilities for writing dataset outputs and failure reports."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as _pyarrow_parquet

from .transforms.deduplicate import mark_latest


class _ParquetProxy:
    def read_table(self, *args, **kwargs):
        return _pyarrow_parquet.read_table(*args, **kwargs)

    def write_table(self, *args, **kwargs):
        return _pyarrow_parquet.write_table(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(_pyarrow_parquet, name)


pq = _ParquetProxy()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def merge_and_deduplicate(
    frames: Sequence[pd.DataFrame],
    *,
    group_keys: Sequence[str] | None = None,
) -> Optional[pd.DataFrame]:
    """Combine a list of frames and drop duplicates by ``group_keys``.

    The helper mirrors the previous ``_concat_and_dedup`` implementation but lives
    outside ``MarketDatasetDownloader`` so it can be reused and tested in
    isolation.
    """

    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return None
    prepared: list[pd.DataFrame] = []
    timestamp = pd.Timestamp.utcnow()
    for df in valid:
        frame = df.copy()
        if "retrieved_at" not in frame.columns:
            frame["retrieved_at"] = timestamp
        prepared.append(frame)
    combined = pd.concat(prepared, ignore_index=True)
    for col in ("ts_code", "end_date", "ann_date"):
        if col in combined.columns:
            combined[col] = combined[col].astype(str)
    dedup_keys = [c for c in (group_keys or ()) if c in combined.columns]
    if "retrieved_at" in combined.columns:
        combined["retrieved_at"] = pd.to_datetime(
            combined["retrieved_at"], errors="coerce"
        )
    extra_sort_keys: list[str] = []
    if "retrieved_at" in combined.columns:
        extra_sort_keys.append("retrieved_at")
    if dedup_keys:
        flagged = mark_latest(
            combined,
            group_keys=dedup_keys,
            extra_sort_keys=extra_sort_keys,
        )
        if "is_latest" in flagged.columns:
            combined = flagged[flagged["is_latest"] == 1].drop(columns=["is_latest"])
        else:
            combined = flagged
        combined = combined.drop_duplicates(subset=dedup_keys)
    else:
        combined = combined.drop_duplicates()
    combined = combined.sort_index(ignore_index=True)
    return combined


def write_parquet_dataset(  # noqa: C901
    df: pd.DataFrame,
    root: str,
    dataset: str,
    year_col: str,
    *,
    group_keys: Sequence[str] | None = None,
) -> bool:
    if df.empty:
        return True
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    if year_col not in frame.columns:
        frame[year_col] = None
    frame[year_col] = frame[year_col].astype(str)
    years = frame[year_col].str[:4]
    frame["year"] = years.fillna("unknown").replace(
        to_replace=r"(?i)nan|nat|none", value="unknown", regex=True
    )
    dataset_root = Path(root) / dataset
    ensure_dir(dataset_root.as_posix())
    updated_any = False
    for year in sorted({y for y in frame["year"].dropna()}):
        partition_new = frame[frame["year"] == year].copy()
        if partition_new.empty:
            continue
        target_dir = dataset_root / f"year={year}"
        existing = pd.DataFrame()
        if target_dir.exists():
            try:
                tables = [
                    pq.read_table(p.as_posix()) for p in target_dir.glob("*.parquet")
                ]
                if tables:
                    existing = pa.concat_tables(tables).to_pandas()
            except Exception as exc:  # pragma: no cover - I/O errors
                print(f"警告：读取 {target_dir} 失败：{exc}")
        if not existing.empty:
            existing.columns = [c.lower() for c in existing.columns]
            if "year" not in existing.columns:
                existing["year"] = year
        all_cols = sorted({*partition_new.columns, *existing.columns})
        partition_new = partition_new.reindex(columns=all_cols)
        if not existing.empty:
            existing = existing.reindex(columns=all_cols)
            combined = pd.concat([existing, partition_new], ignore_index=True)
            deduped = merge_and_deduplicate(
                [combined], group_keys=group_keys or ()
            )
            if deduped is None:
                continue
            combined = deduped
        else:
            combined = partition_new
        combined = combined.drop(columns=["year"], errors="ignore")
        with TemporaryDirectory(dir=dataset_root.as_posix()) as tmpdir:
            tmp_year_dir = Path(tmpdir) / f"year={year}"
            ensure_dir(tmp_year_dir.as_posix())
            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(table, (tmp_year_dir / "data.parquet").as_posix())
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(tmp_year_dir.as_posix(), target_dir.as_posix())
        updated_any = True
    return updated_any


def write_failure_report(
    base_dir: str,
    dataset: str,
    kind: str,
    entries: Iterable[Dict[str, object]],
) -> None:
    failure_root = Path(base_dir) / "_state" / "failures"
    failure_path = failure_root / f"{dataset}_{kind}.json"
    entries = list(entries)
    if not entries:
        if failure_path.exists():
            try:
                failure_path.unlink()
            except OSError:
                pass
        return
    ensure_dir(failure_root.as_posix())
    payload = {
        "dataset": dataset,
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": entries,
    }
    failure_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        "utf-8",
    )


__all__ = [
    "ensure_dir",
    "merge_and_deduplicate",
    "pq",
    "write_failure_report",
    "write_parquet_dataset",
]

