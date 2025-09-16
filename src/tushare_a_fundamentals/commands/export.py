import argparse
import sys
from typing import Dict

import pandas as pd

from ..common import (
    FLOW_FIELDS,
    _diff_to_single,
    _export_tables,
    _load_dataset,
    ensure_ts_code,
    eprint,
)


def cmd_export(args: argparse.Namespace) -> None:
    root = args.dataset_root
    years = getattr(args, "years", 10)
    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]
    out_fmt = args.out_format
    out_dir = args.out_dir
    prefix = args.prefix

    cum = ensure_ts_code(_load_dataset(root, "fact_income_cum"), context="export")
    if "is_latest" in cum.columns:
        cum = cum[cum["is_latest"] == 1]
    periods = sorted(cum["end_date"].astype(str).unique())
    if years is not None:
        keep = set(periods[-years * 4 :])
        cum = cum[cum["end_date"].astype(str).isin(keep)]

    built: Dict[str, pd.DataFrame] = {}

    if "cumulative" in kinds:
        built["cumulative"] = cum.sort_values([
            "ts_code",
            "end_date",
        ]).reset_index(drop=True)

    single = _diff_to_single(cum)
    if "single" in kinds:
        built["single"] = single.sort_values([
            "ts_code",
            "end_date",
        ]).reset_index(drop=True)

    if "annual" in kinds:
        if args.annual_strategy == "cumulative":
            annual = cum[cum["end_date"].astype(str).str.endswith("1231")].copy()
        else:
            sdf = single.copy()
            sdf["year"] = sdf["end_date"].astype(str).str.slice(0, 4)
            aggs = {c: "sum" for c in FLOW_FIELDS if c in sdf.columns}
            annual = sdf.groupby(["ts_code", "year"], as_index=False).agg({**aggs})
            annual["end_date"] = annual["year"].astype(str) + "1231"
            if set(["ann_date", "f_ann_date"]).issubset(cum.columns):
                last_ann = (
                    cum[cum["end_date"].astype(str).str.endswith("1231")]
                    .sort_values(["ts_code", "f_ann_date", "ann_date"])
                    .groupby("ts_code", as_index=False)
                    .tail(1)[["ts_code", "ann_date", "f_ann_date"]]
                )
                annual = annual.merge(last_ann, on="ts_code", how="left")
            annual = annual.drop(
                columns=[
                    c
                    for c in annual.columns
                    if c
                    not in set(
                        ["ts_code", "end_date", *FLOW_FIELDS, "ann_date", "f_ann_date"]
                    )
                ]
            )
        built["annual"] = annual
        built["annual"] = built["annual"].sort_values(
            ["ts_code", "end_date"]
        ).reset_index(drop=True)

    if not built:
        eprint("错误：未选择任何导出口径")
        sys.exit(2)

    _export_tables(built, out_dir, prefix, out_fmt)
