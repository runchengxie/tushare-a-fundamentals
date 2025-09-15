import argparse
import sys
from pathlib import Path

import pandas as pd

from ..common import _load_dataset, eprint


def cmd_coverage(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root)
    inv_path = root / "dataset=inventory_income" / "periods.parquet"
    try:
        inv = pd.read_parquet(inv_path)
    except Exception as exc:
        eprint(f"错误：读取 {inv_path} 失败：{exc}")
        sys.exit(2)
    periods = sorted(inv["end_date"].astype(str).tolist())
    single = _load_dataset(str(root), "fact_income_cum")
    if "is_latest" in single.columns:
        single = single[single["is_latest"] == 1]
    codes = sorted(single["ts_code"].unique())
    full = pd.MultiIndex.from_product(
        [codes, periods], names=["ts_code", "end_date"]
    ).to_frame(index=False)
    present = single[["ts_code", "end_date"]].drop_duplicates()
    present["is_present"] = 1
    cov = full.merge(present, on=["ts_code", "end_date"], how="left").fillna(
        {"is_present": 0}
    )
    if args.by in ("ticker", "ts_code"):
        pivot = cov.pivot(index="ts_code", columns="end_date", values="is_present")
        pivot.index.name = "ticker"
    else:
        pivot = cov.pivot(index="end_date", columns="ts_code", values="is_present")
        pivot.columns.name = "ticker"
    pivot = pivot.sort_index().fillna(0).astype(int)
    print(pivot.to_string())
