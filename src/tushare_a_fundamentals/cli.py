import argparse

from .common import (
    DEFAULT_FIELDS,
    _check_parquet_dependency,
    _run_bulk_mode,
    eprint,
    init_pro_api,
    load_yaml,
    merge_config,
    parse_report_types,
)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量下载A股基本面数据")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--years", type=int)
    p.add_argument("--quarters", type=int)
    vip_group = p.add_mutually_exclusive_group()
    vip_group.add_argument(
        "--vip", action="store_true", help="高级：显式启用 VIP（默认启用）"
    )
    vip_group.add_argument(
        "--no-vip", action="store_true", help="高级：禁用 VIP（已废弃）"
    )
    p.add_argument("--fields", type=str)
    p.add_argument("--outdir", type=str)
    p.add_argument("--prefix", type=str)
    p.add_argument("--format", choices=["csv", "parquet"])
    p.add_argument(
        "--report-types",
        type=str,
        help="逗号分隔的 report_type 列表（默认 1）",
    )
    p.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    p.add_argument("--skip-existing", action="store_true", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--token", type=str)

    sub = p.add_subparsers(dest="cmd")

    sp_exp = sub.add_parser(
        "export", help="由本地事实表构建 annual/single/cumulative 导出"
    )
    sp_exp.add_argument("--dataset-root", type=str, required=True)
    sp_exp.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    sp_exp.add_argument(
        "--kinds",
        type=str,
        default="annual,single",
        help="逗号分隔：annual,single,cumulative",
    )
    sp_exp.add_argument(
        "--annual-strategy",
        choices=["cumulative", "sum4"],
        default="cumulative",
        help="年度口径：累计或四季相加",
    )
    sp_exp.add_argument("--out-format", choices=["csv", "parquet"], default="csv")
    sp_exp.add_argument("--out-dir", type=str, default="out")
    sp_exp.add_argument("--prefix", type=str, default="income")

    sp_cov = sub.add_parser("coverage", help="盘点已覆盖的股票×期末日")
    sp_cov.add_argument("--dataset-root", type=str, required=True)
    sp_cov.add_argument(
        "--by",
        choices=["ticker", "ts_code", "period"],
        default="ticker",
        help="输出维度：ticker/ts_code 或 period",
    )

    sp_dl = sub.add_parser("download", help="下载数据（默认增量补全；--force 覆盖）")
    sp_dl.add_argument("--config", type=str, default=None)
    sp_dl.add_argument("--years", "--year", dest="years", type=int)
    sp_dl.add_argument("--quarters", type=int)
    sp_dl.add_argument(
        "--since", type=str, help="起始日期 YYYY-MM-DD（优先于 --years/--quarters）"
    )
    sp_dl.add_argument("--until", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    sp_dl.add_argument("--fields", type=str)
    sp_dl.add_argument("--outdir", type=str)
    sp_dl.add_argument("--prefix", type=str)
    sp_dl.add_argument("--format", choices=["csv", "parquet"])
    sp_dl.add_argument(
        "--report-types",
        type=str,
        help="逗号分隔的 report_type 列表（默认 1）",
    )
    sp_dl.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    sp_dl.add_argument("--token", type=str)
    sp_dl.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖已有文件（忽略增量跳过）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    if getattr(args, "cmd", None) == "download":
        from .commands.download import cmd_download

        return cmd_download(args)
    if getattr(args, "cmd", None) == "export":
        from .commands.export import cmd_export

        return cmd_export(args)
    if getattr(args, "cmd", None) == "coverage":
        from .commands.coverage import cmd_coverage

        return cmd_coverage(args)
    cfg_file = load_yaml(args.config)
    defaults = {
        "years": 10,
        "quarters": None,
        "since": None,
        "until": None,
        "fields": ",".join(DEFAULT_FIELDS),
        "outdir": "out",
        "prefix": "income",
        "format": "parquet",
        "skip_existing": True,
        "token": None,
        "export_colname": "ticker",
        "report_types": [1],
    }
    cli_overrides = {
        "years": args.years,
        "quarters": args.quarters,
        "since": getattr(args, "since", None) if hasattr(args, "since") else None,
        "until": getattr(args, "until", None) if hasattr(args, "until") else None,
        "fields": args.fields,
        "outdir": args.outdir,
        "prefix": args.prefix,
        "format": args.format,
        "skip_existing": args.skip_existing,
        "token": args.token,
        "export_colname": args.export_colname,
        "report_types": getattr(args, "report_types", None),
    }
    cfg = merge_config(cli_overrides, cfg_file, defaults)
    cfg["report_types"] = parse_report_types(cfg.get("report_types"))
    if getattr(args, "force", False):
        cfg["skip_existing"] = False
    pro = init_pro_api(cfg.get("token"))
    fields = cfg["fields"]
    fmt = cfg["format"]
    if fmt == "parquet" and not _check_parquet_dependency():
        eprint("警告：缺少 pyarrow 或 fastparquet，已回退到 csv 格式")
        fmt = "csv"
    outdir = cfg["outdir"]
    prefix = cfg["prefix"]
    _run_bulk_mode(pro, cfg, fields, fmt, outdir, prefix)


if __name__ == "__main__":
    main()
