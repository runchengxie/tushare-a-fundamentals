import argparse
import sys
from pathlib import Path

import pandas as pd

from ..common import _load_dataset, ensure_ts_code, eprint


def cmd_coverage(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root)
    years = getattr(args, "years", 10)
    inv_path = root / "dataset=inventory_income" / "periods.parquet"
    try:
        inv = pd.read_parquet(inv_path)
    except Exception as exc:
        eprint(f"错误：读取 {inv_path} 失败：{exc}")
        sys.exit(2)
    periods = sorted(inv["end_date"].astype(str).tolist())
    if years is not None:
        periods = periods[-years * 4 :]
    fact = ensure_ts_code(
        _load_dataset(str(root), "fact_income_cum"), context="coverage"
    )
    if "is_latest" in fact.columns:
        fact = fact[fact["is_latest"] == 1]
    fact["end_date"] = fact["end_date"].astype(str)
    fact = fact[fact["end_date"].isin(periods)]
    present = fact[["ts_code", "end_date"]].drop_duplicates()
    if present.empty:
        eprint("警告：指定时间窗口内没有可用数据")
        return
    codes = sorted(present["ts_code"].unique())
    idx = pd.MultiIndex.from_product(
        [codes, periods], names=["ts_code", "end_date"]
    )
    target = idx.to_frame(index=False)
    earliest = present.groupby("ts_code")["end_date"].min()
    target["earliest"] = target["ts_code"].map(earliest)
    present_index = pd.MultiIndex.from_frame(present)
    target_index = pd.MultiIndex.from_frame(target[["ts_code", "end_date"]])
    status = pd.Series("missing", index=target.index)
    status[target_index.isin(present_index)] = "present"
    exempt_mask = target["end_date"] < target["earliest"]
    status[exempt_mask] = "exempt"
    target = target.drop(columns=["earliest"])
    target["status"] = status.values

    counts = target["status"].value_counts()
    present_count = int(counts.get("present", 0))
    missing_count = int(counts.get("missing", 0))
    exempt_count = int(counts.get("exempt", 0))
    effective_total = present_count + missing_count
    coverage_rate = (
        present_count / effective_total if effective_total else 1.0
    )
    print(
        f"覆盖股票 {len(codes)} 个，期末日 {len(periods)} 个；有效组合 {effective_total}"
    )
    print(
        f"已覆盖 {present_count}，缺口 {missing_count}，覆盖率 {coverage_rate:.2%}"
    )
    if exempt_count:
        print(f"自然缺失组合（上市前等）：{exempt_count}")

    if getattr(args, "csv", None):
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        missing = target[target["status"] == "missing"][["ts_code", "end_date"]]
        if missing.empty:
            summary = pd.DataFrame(columns=["ts_code", "missing_periods", "missing_count"])
        else:
            summary = (
                missing.groupby("ts_code")["end_date"]
                .agg(
                    missing_periods=lambda values: ";".join(
                        sorted(values.astype(str))
                    ),
                    missing_count="count",
                )
                .reset_index()
            )
        report = pd.DataFrame({"ts_code": codes})
        report = report.merge(summary, on="ts_code", how="left")
        report["missing_periods"] = report["missing_periods"].fillna("")
        report["missing_count"] = report["missing_count"].fillna(0).astype(int)
        report.to_csv(out_path, index=False)
        if missing.empty:
            print(f"缺口清单为空，已导出汇总模板：{out_path}")
        else:
            print(f"缺口清单已写入：{out_path}")

    if getattr(args, "by", None):
        value_map = {"present": 1, "missing": 0, "exempt": -1}
        target["value"] = target["status"].map(value_map)
        if args.by in ("ticker", "ts_code"):
            pivot = target.pivot(index="ts_code", columns="end_date", values="value")
            pivot.index.name = "ts_code"
        else:
            pivot = target.pivot(index="end_date", columns="ts_code", values="value")
            pivot.columns.name = "ts_code"
        pivot = pivot.sort_index().fillna(0).astype(int)
        print("标记说明：1=覆盖，0=缺口，-1=豁免")
        print(pivot.to_string())
