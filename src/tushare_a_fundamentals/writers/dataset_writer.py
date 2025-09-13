from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import uuid
import time
import pandas as pd

from tushare_a_fundamentals.transforms.deduplicate import mark_latest


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_year(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("-", "", regex=False)
    return s.str.slice(0, 4)


def _compute_partition_cols(df: pd.DataFrame, partition_by: str) -> pd.DataFrame:
    """Compute partition columns on a copy of df.

    Supports spec like ``"year:end_date"``.
    """
    out = df.copy()
    try:
        gran, col = partition_by.split(":", 1)
    except ValueError:
        raise ValueError(f"不支持的分区规范：{partition_by}")
    if col not in out.columns:
        raise KeyError(f"分区字段不存在：{col}")
    if gran == "year":
        out["year"] = _extract_year(out[col])
    else:
        raise ValueError(f"不支持的分区粒度：{gran}")
    return out


def write_partitioned_dataset(
    df: pd.DataFrame,
    root: str | Path,
    dataset: str,
    partition_by: str,
    primary_key: Sequence[str] | None = None,
    version_by: Sequence[str] | None = None,
    only_latest: bool = True,
) -> list[Path]:
    """Write a partitioned Parquet dataset.

    - Adds/uses ``is_latest`` column via preferred version rule.
    - Partitions into ``root/dataset=<dataset>/year=<YYYY>/``.
    - Writes one file per partition per invocation.
    """
    if df.empty:
        return []

    # mark is_latest if not present
    flagged = df if "is_latest" in df.columns else mark_latest(df)
    if only_latest:
        flagged = flagged[flagged["is_latest"] == 1].copy()

    flagged = _compute_partition_cols(flagged, partition_by)

    rootp = Path(root)
    written: list[Path] = []
    for year, part in flagged.groupby("year"):
        base = rootp / f"dataset={dataset}" / f"year={year}"
        _ensure_dir(base)
        ts = time.strftime("%Y%m%d%H%M%S")
        fname = f"part-{ts}-{uuid.uuid4().hex[:8]}.parquet"
        fpath = base / fname
        # drop helper partition column from content; directories capture it
        content = part.drop(columns=["year"]) if "year" in part.columns else part
        content.to_parquet(fpath, index=False)
        written.append(fpath)
    return written

