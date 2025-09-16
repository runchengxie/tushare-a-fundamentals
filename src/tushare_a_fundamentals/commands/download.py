import argparse
import sys

from ..common import (
    DEFAULT_FIELDS,
    _check_parquet_dependency,
    _run_bulk_mode,
    build_datasets_from_raw,
    eprint,
    init_pro_api,
    load_yaml,
    merge_config,
    parse_report_types,
)


def cmd_download(args: argparse.Namespace) -> None:
    cfg_file = load_yaml(getattr(args, "config", None))
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
        "token": None,
        "report_types": [1],
        "allow_future": False,
        "recent_quarters": 8,
    }
    cli_overrides = {
        "years": getattr(args, "years", None),
        "quarters": getattr(args, "quarters", None),
        "since": getattr(args, "since", None),
        "until": getattr(args, "until", None),
        "fields": getattr(args, "fields", None),
        "outdir": getattr(args, "outdir", None),
        "prefix": getattr(args, "prefix", None),
        "format": getattr(args, "format", None),
        "token": getattr(args, "token", None),
        "report_types": getattr(args, "report_types", None),
        "allow_future": getattr(args, "allow_future", None),
        "recent_quarters": getattr(args, "recent_quarters", None),
    }
    cfg = merge_config(cli_overrides, cfg_file, defaults)
    cfg["report_types"] = parse_report_types(cfg.get("report_types"))
    raw_only = getattr(args, "raw_only", False)
    build_only = getattr(args, "build_only", False)
    if raw_only and build_only:
        eprint("错误：--raw-only 与 --build-only 互斥")
        sys.exit(2)
    if getattr(args, "force", False):
        cfg["skip_existing"] = False
    outdir = cfg["outdir"]
    prefix = cfg["prefix"]
    if not build_only:
        pro = init_pro_api(cfg.get("token"))
        fields = cfg["fields"]
        fmt = cfg["format"]
        if fmt == "parquet" and not _check_parquet_dependency():
            eprint("警告：缺少 pyarrow 或 fastparquet，已回退到 csv 格式")
            fmt = "csv"
        _run_bulk_mode(pro, cfg, fields, fmt, outdir, prefix)
    if not raw_only:
        build_datasets_from_raw(outdir, prefix)
