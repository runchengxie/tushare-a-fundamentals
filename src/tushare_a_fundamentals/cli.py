import argparse

from .common import (
    DEFAULT_FIELDS,
    _check_parquet_dependency,
    _run_bulk_mode,
    eprint,
    init_pro_api,
    load_yaml,
    merge_config,
    normalize_fields,
    parse_report_types,
)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量下载A股基本面数据")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--years", type=int)
    p.add_argument("--quarters", type=int)
    p.add_argument(
        "--recent-quarters",
        type=int,
        help="近N季滚动刷新（默认 8）",
    )
    p.add_argument("--max-retries", type=int, help="接口重试次数上限（默认 3）")
    vip_group = p.add_mutually_exclusive_group()
    vip_group.add_argument(
        "--vip", action="store_true", help="高级：显式启用 VIP（默认启用）"
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
        "--skip-existing",
        action="store_true",
        default=None,
        help="仅补缺，不执行滚动刷新",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--token", type=str)

    sub = p.add_subparsers(dest="cmd")

    sp_exp = sub.add_parser(
        "export", help="由本地事实表构建 annual/single/cumulative 导出"
    )
    sp_exp.add_argument(
        "--dataset-root", type=str, default="out", help="数据集根目录（默认 out）"
    )
    sp_exp.add_argument("--years", type=int, default=10, help="近几年（默认 10）")
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
    sp_cov.add_argument(
        "--dataset-root", type=str, default="out", help="数据集根目录（默认 out）"
    )
    sp_cov.add_argument("--years", type=int, default=10, help="近几年（默认 10）")
    sp_cov.add_argument(
        "--by",
        choices=["ticker", "ts_code", "period"],
        default="ticker",
        help="输出维度：ticker/ts_code 或 period",
    )
    sp_cov.add_argument("--csv", type=str, help="缺口清单另存为 CSV")

    sp_state = sub.add_parser("state", help="查看与维护增量状态信息")
    sp_state.add_argument(
        "action",
        choices=["show", "clear", "set", "ls-failures"],
        help="操作类型",
    )
    sp_state.add_argument(
        "--backend",
        choices=["auto", "json", "sqlite"],
        default="auto",
        help="状态后端：auto/json/sqlite（默认 auto）",
    )
    sp_state.add_argument("--state-path", help="状态文件或数据库路径")
    sp_state.add_argument(
        "--data-dir",
        default="data",
        help="多数据集数据目录（默认 data）",
    )
    sp_state.add_argument("--dataset", help="指定数据集")
    sp_state.add_argument("--year", type=int, help="SQLite 状态时可指定年份")
    sp_state.add_argument("--key", help="JSON 状态键名")
    sp_state.add_argument("--value", help="JSON 状态值")
    sp_state.set_defaults(cmd="state")

    sp_dl = sub.add_parser("download", help="下载数据（默认增量补全；--force 覆盖）")
    sp_dl.add_argument("--config", type=str, default=None)
    sp_dl.add_argument("--years", "--year", dest="years", type=int)
    sp_dl.add_argument("--quarters", type=int)
    sp_dl.add_argument(
        "--recent-quarters",
        type=int,
        help="近N季滚动刷新（默认 8）",
    )
    sp_dl.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        help="接口重试次数上限（默认 3）",
    )
    sp_dl.add_argument(
        "--since", type=str, help="起始日期 YYYY-MM-DD（优先于 --years/--quarters）"
    )
    sp_dl.add_argument("--until", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    sp_dl.add_argument("--fields", type=str)
    sp_dl.add_argument("--outdir", type=str)
    sp_dl.add_argument("--prefix", type=str)
    sp_dl.add_argument("--format", choices=["csv", "parquet"])
    sp_dl.add_argument(
        "--datasets",
        nargs="+",
        help="启用多数据集批量下载（传入数据集名称列表）",
    )
    sp_dl.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        help="多数据集输出目录（默认 data）",
    )
    sp_dl.add_argument(
        "--with-audit",
        action="store_true",
        help="在默认数据集中追加 fina_audit",
    )
    sp_dl.add_argument(
        "--audit-only",
        action="store_true",
        help="仅下载 fina_audit（忽略其他数据集）",
    )
    sp_dl.add_argument(
        "--all",
        action="store_true",
        help="按配置列出的全部数据集（包含 fina_audit）",
    )
    sp_dl.add_argument(
        "--skip-existing",
        action="store_true",
        help="仅补缺，不执行滚动刷新",
    )
    sp_dl.add_argument(
        "--report-types",
        type=str,
        help="逗号分隔的 report_type 列表（默认 1）",
    )
    sp_dl.add_argument("--token", type=str)
    sp_dl.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖已有文件（忽略增量跳过）",
    )
    sp_dl.add_argument(
        "--max-per-minute",
        dest="max_per_minute",
        type=int,
        help="接口每分钟最大调用次数（默认 90）",
    )
    vip_toggle = sp_dl.add_mutually_exclusive_group()
    vip_toggle.add_argument(
        "--use-vip",
        dest="use_vip",
        action="store_true",
        help="优先使用 VIP 接口",
    )
    sp_dl.add_argument(
        "--state-path",
        dest="state_path",
        type=str,
        help="覆盖默认增量状态文件位置",
    )
    sp_dl.set_defaults(use_vip=None)
    flag_group = sp_dl.add_mutually_exclusive_group()
    flag_group.add_argument(
        "--raw-only",
        action="store_true",
        help="仅下载 raw，不构建数仓",
    )
    flag_group.add_argument(
        "--build-only",
        action="store_true",
        help="跳过下载，仅由 raw 构建数仓",
    )
    sp_dl.add_argument(
        "--allow-future",
        action="store_true",
        help="允许请求尚未披露的未来季度",
    )
    sp_dl.add_argument(
        "--no-export",
        action="store_true",
        dest="no_export",
        help="仅写 raw/parquet 数仓，不导出派生 CSV",
    )
    sp_dl.add_argument("--export", action="store_true", dest="export_enabled")
    sp_dl.set_defaults(export_enabled=None)
    sp_dl.add_argument(
        "--export-out-dir",
        dest="export_out_dir",
        type=str,
        help="导出目录（默认与 outdir 下格式目录一致）",
    )
    sp_dl.add_argument(
        "--export-format",
        dest="export_out_format",
        choices=["csv", "parquet"],
        help="导出格式（默认 csv）",
    )
    sp_dl.add_argument(
        "--export-kinds",
        dest="export_kinds",
        type=str,
        help="导出口径（默认 annual,single,cumulative）",
    )
    sp_dl.add_argument(
        "--export-annual-strategy",
        dest="export_annual_strategy",
        choices=["cumulative", "sum4"],
        help="年度导出策略（默认 cumulative）",
    )
    sp_dl.add_argument(
        "--export-years",
        dest="export_years",
        type=int,
        help="导出最近 N 年（默认沿用下载窗口，未指定则导出全部）",
    )
    sp_dl.add_argument(
        "--strict-export",
        dest="export_strict",
        action="store_true",
        help="导出失败时视为错误退出（默认仅警告）",
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
    if getattr(args, "cmd", None) == "state":
        from .commands.state import cmd_state

        return cmd_state(args)
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
        "skip_existing": False,
        "recent_quarters": 4,
        "token": None,
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
        "recent_quarters": getattr(args, "recent_quarters", None),
        "token": args.token,
        "report_types": getattr(args, "report_types", None),
    }
    cfg = merge_config(cli_overrides, cfg_file, defaults)
    cfg["report_types"] = parse_report_types(cfg.get("report_types"))
    cfg["fields"] = normalize_fields(cfg.get("fields"))
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
