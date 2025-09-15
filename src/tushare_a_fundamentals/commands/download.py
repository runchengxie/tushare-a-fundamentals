import argparse

from ..common import (
    DEFAULT_FIELDS,
    _check_parquet_dependency,
    _run_bulk_mode,
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
        "skip_existing": True,  # download 默认增量
        "token": None,
        "export_colname": "ticker",
        "report_types": [1],
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
        "export_colname": getattr(args, "export_colname", None),
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
