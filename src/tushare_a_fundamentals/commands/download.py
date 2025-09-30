import argparse
import os
import sys
from argparse import Namespace

from ..common import (
    DEFAULT_FIELDS,
    _check_parquet_dependency,
    _periods_from_cfg,
    _run_bulk_mode,
    build_datasets_from_raw,
    eprint,
    init_pro_api,
    load_yaml,
    merge_config,
    normalize_fields,
    parse_report_types,
)
from ..downloader import (
    MarketDatasetDownloader,
    parse_dataset_requests,
    parse_yyyymmdd,
)
from .export import cmd_export


def _download_defaults() -> dict:
    return {
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
        "datasets": None,
        "data_dir": "data",
        "use_vip": True,
        "max_per_minute": 90,
        "state_path": None,
        "export_enabled": True,
        "export_out_dir": None,
        "export_out_format": "csv",
        "export_kinds": "annual,single,cumulative",
        "export_annual_strategy": "cumulative",
        "export_years": None,
        "export_strict": False,
        "max_retries": 3,
    }


def _collect_cli_overrides(args: argparse.Namespace) -> dict:
    overrides = {
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
        "datasets": getattr(args, "datasets", None),
        "data_dir": getattr(args, "data_dir", None),
        "use_vip": getattr(args, "use_vip", None),
        "max_per_minute": getattr(args, "max_per_minute", None),
        "state_path": getattr(args, "state_path", None),
        "export_out_dir": getattr(args, "export_out_dir", None),
        "export_out_format": getattr(args, "export_out_format", None),
        "export_kinds": getattr(args, "export_kinds", None),
        "export_annual_strategy": getattr(args, "export_annual_strategy", None),
        "export_years": getattr(args, "export_years", None),
        "export_strict": getattr(args, "export_strict", None),
        "max_retries": getattr(args, "max_retries", None),
    }
    if getattr(args, "export_enabled", None) is not None:
        overrides["export_enabled"] = getattr(args, "export_enabled")
    if getattr(args, "no_export", False):
        overrides["export_enabled"] = False
    return overrides


def _resolve_raw_format(cfg: dict) -> str:
    fmt = cfg.get("format", "parquet")
    if fmt == "parquet" and not _check_parquet_dependency():
        eprint("警告：缺少 pyarrow 或 fastparquet，已回退到 csv 格式")
        return "csv"
    return fmt


def _build_export_args(cfg: dict, outdir: str, prefix: str) -> Namespace | None:
    if not cfg.get("export_enabled", True):
        return None
    export_out_format = (cfg.get("export_out_format") or "csv").lower()
    export_out_dir = cfg.get("export_out_dir") or os.path.join(
        outdir, export_out_format
    )
    export_kinds_cfg = cfg.get("export_kinds")
    if isinstance(export_kinds_cfg, (list, tuple, set)):
        export_kinds = ",".join(
            str(k).strip() for k in export_kinds_cfg if str(k).strip()
        )
    elif export_kinds_cfg is None:
        export_kinds = "annual,single,cumulative"
    else:
        export_kinds = str(export_kinds_cfg)
    export_years = cfg.get("export_years")
    if export_years is None:
        export_years = cfg.get("years")
    return Namespace(
        dataset_root=outdir,
        years=export_years,
        kinds=export_kinds,
        annual_strategy=cfg.get("export_annual_strategy", "cumulative"),
        out_format=export_out_format,
        out_dir=export_out_dir,
        prefix=prefix,
    )


def _run_export(export_args: Namespace, strict: bool | None) -> None:
    try:
        cmd_export(export_args)
    except SystemExit as exc:
        if strict:
            raise
        eprint(f"警告：导出失败（已保留 parquet）：{exc}")
    except Exception as exc:  # pragma: no cover - defensive guard
        eprint(f"警告：导出失败（已保留 parquet）：{exc}")
        if strict:
            raise


def cmd_download(args: argparse.Namespace) -> None:
    cfg_file = load_yaml(getattr(args, "config", None))
    defaults = _download_defaults()
    cli_overrides = _collect_cli_overrides(args)
    cfg = merge_config(cli_overrides, cfg_file, defaults)
    cfg["report_types"] = parse_report_types(cfg.get("report_types"))
    cfg["fields"] = normalize_fields(cfg.get("fields"))
    try:
        max_retries = int(cfg.get("max_retries", 3))
    except (TypeError, ValueError):
        max_retries = 3
    if max_retries < 0:
        max_retries = 0
    cfg["max_retries"] = max_retries
    dataset_requests = parse_dataset_requests(cfg.get("datasets"))
    if dataset_requests:
        if getattr(args, "force", False):
            eprint("警告：多数据集模式暂不支持 --force，将忽略该参数")
        if getattr(args, "raw_only", False) or getattr(args, "build_only", False):
            eprint("错误：多数据集模式不支持 --raw-only 或 --build-only")
            sys.exit(2)
        pro = init_pro_api(cfg.get("token"))
        data_dir = cfg.get("data_dir") or "data"
        use_vip = cfg.get("use_vip")
        if use_vip is None:
            use_vip = True
        max_per_minute = cfg.get("max_per_minute")
        if max_per_minute is None:
            max_per_minute = 90
        start_raw = cfg.get("since")
        end_raw = cfg.get("until")
        if not start_raw or not end_raw:
            periods_window = _periods_from_cfg(cfg)
            if periods_window:
                if not start_raw:
                    start_raw = periods_window[0]
                if not end_raw:
                    end_raw = periods_window[-1]
        downloader = MarketDatasetDownloader(
            pro,
            data_dir,
            use_vip=use_vip,
            max_per_minute=max_per_minute,
            state_path=cfg.get("state_path"),
            allow_future=bool(cfg.get("allow_future")),
            max_retries=max_retries,
        )
        downloader.run(
            dataset_requests,
            start=parse_yyyymmdd(start_raw),
            end=parse_yyyymmdd(end_raw),
            refresh_periods=int(cfg.get("recent_quarters") or 0),
        )
        return
    raw_only = getattr(args, "raw_only", False)
    build_only = getattr(args, "build_only", False)
    if raw_only and build_only:
        eprint("错误：--raw-only 与 --build-only 互斥")
        sys.exit(2)
    if getattr(args, "force", False):
        cfg["skip_existing"] = False
    outdir = cfg["outdir"]
    prefix = cfg["prefix"]
    raw_fmt = cfg.get("format", "parquet")
    if not build_only:
        pro = init_pro_api(cfg.get("token"))
        fields = cfg["fields"]
        raw_fmt = _resolve_raw_format(cfg)
        _run_bulk_mode(pro, cfg, fields, raw_fmt, outdir, prefix)
    built = False
    if not raw_only:
        built = build_datasets_from_raw(outdir, prefix, raw_format=raw_fmt)
    export_args = _build_export_args(cfg, outdir, prefix) if built else None
    if export_args is not None:
        _run_export(export_args, strict=cfg.get("export_strict"))
