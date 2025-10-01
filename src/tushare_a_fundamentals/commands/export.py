import argparse
import sys

from ..common import (
    _export_tables,
    _load_dataset,
    build_income_export_tables,
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
    periods = sorted(cum["end_date"].astype(str).unique())
    total_periods = len(periods)
    if years is not None:
        requested_periods = years * 4
        if total_periods == 0:
            eprint(
                f"提示：导出窗口 {years} 年超出现有缓存范围，当前目录 {root} 下没有可导出的季度数据。"
            )
        elif requested_periods > total_periods:
            eprint(
                "提示：导出窗口 {years} 年超出现有缓存范围（仅有 {count} 个季度，"
                "{earliest} 至 {latest}），将导出全部可用数据。".format(
                    years=years,
                    count=total_periods,
                    earliest=periods[0],
                    latest=periods[-1],
                )
            )

    built = build_income_export_tables(
        cum,
        years=years,
        kinds=kinds,
        annual_strategy=args.annual_strategy,
    )
    if not built:
        eprint("错误：未选择任何导出口径或数据为空")
        sys.exit(2)

    _export_tables(built, out_dir, prefix, out_fmt)
