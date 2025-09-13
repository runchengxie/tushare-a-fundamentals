from __future__ import annotations

from typing import Iterable, Sequence
import pandas as pd


def _prepare_sort_keys(df: pd.DataFrame) -> list[str]:
    sort_cols: list[str] = []
    if "report_type" in df.columns:
        df["_rt_pref"] = (df["report_type"] == 1).astype(int)
        sort_cols.append("_rt_pref")
    if "update_flag" in df.columns:
        df["_upd"] = (df["update_flag"].astype(str).str.upper() == "Y").astype(int)
        sort_cols.append("_upd")
    # prefer f_ann_date over ann_date if both exist
    for col in ("f_ann_date", "ann_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            sort_cols.append(col)
    return sort_cols


def mark_latest(
    df: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
    extra_sort_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add an ``is_latest`` column based on preferred version ordering.

    The default grouping is by (``ts_code``, ``end_date``). Within each group
    the latest row is selected using the following priority when present:
    ``report_type==1`` > ``update_flag=='Y'`` > ``f_ann_date`` desc > ``ann_date`` desc.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    # normalise end_date to sortable string if present
    if "end_date" in out.columns:
        out["end_date"] = out["end_date"].astype(str).str.replace("-", "", regex=False)

    sort_cols = _prepare_sort_keys(out)
    if extra_sort_keys:
        sort_cols.extend([c for c in extra_sort_keys if c in out.columns])
    if not sort_cols:
        # fall back to stable order
        sort_cols = list(out.columns[:1])

    gkeys = list(group_keys or ("ts_code", "end_date"))
    gkeys = [k for k in gkeys if k in out.columns]
    # stable sort to make cumcount deterministic across equal keys
    out = out.sort_values(
        sort_cols, ascending=[False] * len(sort_cols), kind="mergesort"
    )
    out["_rank"] = out.groupby(gkeys).cumcount()
    out["is_latest"] = (out["_rank"] == 0).astype(int)
    out = out.drop(
        columns=[c for c in ("_rt_pref", "_upd", "_rank") if c in out.columns]
    )
    # restore original row order
    return out.sort_index()


def select_latest(
    df: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
    extra_sort_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return only the latest row per group using ``mark_latest`` policy."""
    if df.empty:
        return df.copy()
    flagged = mark_latest(df, group_keys=group_keys, extra_sort_keys=extra_sort_keys)
    return (
        flagged[flagged["is_latest"] == 1].drop(columns=["is_latest"])
        if "is_latest" in flagged.columns
        else flagged
    )
