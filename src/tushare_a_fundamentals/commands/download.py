import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

from ..common import (
    DEFAULT_FIELDS,
    _concat_non_empty,
    _export_tables,
    _periods_from_cfg,
    build_income_export_tables,
    ensure_enough_credits,
    ensure_ts_code,
    eprint,
    init_pro_api,
    load_yaml,
    merge_config,
    normalize_fields,
    parse_report_types,
)
from ..downloader import (
    DatasetRequest,
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
        "outdir": None,
        "prefix": "income",
        "format": "parquet",
        "token": None,
        "report_types": [1],
        "allow_future": False,
        "recent_quarters": 4,
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


def _build_export_args(cfg: dict, outdir: str, prefix: str) -> Namespace | None:
    if not cfg.get("export_enabled", True):
        return None
    export_out_format = (cfg.get("export_out_format") or "csv").lower()
    export_out_dir_cfg = cfg.get("export_out_dir")
    if export_out_dir_cfg:
        export_out_dir = export_out_dir_cfg
    else:
        export_out_dir = os.path.normpath(outdir)
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


def _load_dataset_from_data_dir(data_dir: str, dataset: str) -> pd.DataFrame:
    base = Path(data_dir) / dataset
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception as exc:  # pragma: no cover - defensive logging
            eprint(f"警告：读取 {file} 失败：{exc}")
    if not frames:
        return pd.DataFrame()
    combined = _concat_non_empty(frames)
    if combined.empty:
        return combined
    return ensure_ts_code(combined, context=dataset)


def _export_income_from_multi(cfg: dict, data_dir: str, requests) -> None:
    if not cfg.get("export_enabled", True):
        return
    dataset_names = {req.name for req in requests}
    if "income" not in dataset_names:
        return
    data_dir = cfg.get("data_dir") or "data"
    outdir = cfg.get("outdir") or data_dir
    prefix = cfg.get("prefix") or "income"
    export_args = _build_export_args(cfg, outdir, prefix)
    if export_args is None:
        return
    income_df = _load_dataset_from_data_dir(data_dir, "income")
    if income_df.empty:
        eprint("提示：income 数据为空，跳过自动导出")
        return
    kinds = [s.strip() for s in export_args.kinds.split(",") if s.strip()]
    built = build_income_export_tables(
        income_df,
        years=export_args.years,
        kinds=kinds,
        annual_strategy=export_args.annual_strategy,
    )
    if not built:
        eprint("提示：未生成可导出的 income 数据，跳过自动导出")
        return
    _export_tables(
        built, export_args.out_dir, export_args.prefix, export_args.out_format
    )


def cmd_download(args: argparse.Namespace) -> None:
    cfg_file = load_yaml(getattr(args, "config", None))
    defaults = _download_defaults()
    cfg_missing = not bool(cfg_file)
    if cfg_missing:
        defaults["datasets"] = DEFAULT_DATASET_CONFIG
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
    use_vip = cfg.get("use_vip")
    if use_vip is None:
        use_vip = True
    try:
        dataset_requests, info_msgs, warn_msgs = _build_dataset_plan(
            cfg,
            args,
            use_vip=use_vip,
            cfg_missing=cfg_missing,
        )
    except ValueError as exc:
        eprint(f"错误：{exc}")
        sys.exit(2)

    for msg in warn_msgs:
        eprint(msg)
    for msg in info_msgs:
        print(msg)

    _run_multi_dataset_flow(
        cfg,
        dataset_requests,
        use_vip=use_vip,
    )


AUDIT_DATASET_NAME = "fina_audit"
VIP_ONLY_DATASETS = {"forecast", "fina_indicator", "fina_mainbz"}

DEFAULT_DATASET_CONFIG = [
    {"name": "income", "report_types": [1]},
    {"name": "balancesheet", "report_types": [1]},
    {"name": "cashflow", "report_types": [1]},
    {"name": "forecast"},
    {"name": "express"},
    {"name": "dividend"},
    {"name": "fina_indicator"},
    {"name": "fina_audit"},
    {"name": "fina_mainbz", "type": ["P", "D", "I"]},
    {"name": "disclosure_date"},
]

_DEFAULT_DATASET_LOOKUP = {
    item["name"]: {k: v for k, v in item.items() if k != "name"}
    for item in DEFAULT_DATASET_CONFIG
}


def _default_request_for(name: str) -> DatasetRequest:
    options = _DEFAULT_DATASET_LOOKUP.get(name)
    if options:
        return DatasetRequest(name=name, options=dict(options))
    return DatasetRequest(name=name)


def _validate_dataset_flags(
    *,
    explicit: bool,
    add_audit: bool,
    audit_only: bool,
    include_all: bool,
) -> None:
    if explicit and (add_audit or audit_only or include_all):
        raise ValueError("--datasets 不可与 --with-audit/--audit-only/--all 同时使用")
    if audit_only and (add_audit or include_all):
        raise ValueError("--audit-only 不可与 --with-audit 或 --all 同时使用")
    if add_audit and include_all:
        raise ValueError("--with-audit 与 --all 不可同时使用")


def _apply_audit_selection(
    dataset_requests: list[DatasetRequest],
    *,
    explicit: bool,
    add_audit: bool,
    audit_only: bool,
    include_all: bool,
) -> tuple[list[DatasetRequest], bool]:
    if not dataset_requests and audit_only:
        return [_default_request_for(AUDIT_DATASET_NAME)], False
    if not dataset_requests or explicit:
        return list(dataset_requests), False

    non_audit: list[DatasetRequest] = []
    audit_list: list[DatasetRequest] = []
    for req in dataset_requests:
        if req.name == AUDIT_DATASET_NAME:
            audit_list.append(req)
        else:
            non_audit.append(req)

    if audit_only:
        return audit_list or [_default_request_for(AUDIT_DATASET_NAME)], False
    if include_all:
        if audit_list:
            return non_audit + audit_list, False
        return list(dataset_requests) + [
            _default_request_for(AUDIT_DATASET_NAME)
        ], False
    if add_audit:
        if audit_list:
            return non_audit + audit_list, False
        return non_audit + [_default_request_for(AUDIT_DATASET_NAME)], False

    return non_audit, bool(audit_list)


def _apply_vip_filter(
    dataset_requests: list[DatasetRequest],
    *,
    use_vip: bool,
) -> tuple[list[DatasetRequest], list[str]]:
    if use_vip:
        return dataset_requests, []
    skipped = [req for req in dataset_requests if req.name in VIP_ONLY_DATASETS]
    if not skipped:
        return dataset_requests, []
    skipped_names = ", ".join(sorted({req.name for req in skipped}))
    warn = (
        "警告：use_vip=false，已跳过仅支持 VIP 批量的接口："
        f"{skipped_names}（项目未实现个股枚举 fallback）"
    )
    kept = [req for req in dataset_requests if req.name not in VIP_ONLY_DATASETS]
    return kept, [warn]


def _info_messages(
    *,
    cfg_missing: bool,
    explicit: bool,
    skip_audit_info: bool,
    add_audit: bool,
    audit_only: bool,
    include_all: bool,
) -> list[str]:
    infos: list[str] = []
    if skip_audit_info and not (add_audit or audit_only or include_all):
        infos.append(
            "提示：默认跳过 fina_audit（需按股票循环）。"
            "使用 --with-audit / --audit-only / --all 可显式启用。"
        )
    if cfg_missing and not explicit:
        infos.append(
            "提示：未找到配置文件，已按示例配置启用默认数据集（默认跳过 fina_audit）。"
        )
    return infos


def _build_dataset_plan(
    cfg: dict,
    args: argparse.Namespace,
    *,
    use_vip: bool,
    cfg_missing: bool,
) -> tuple[list[DatasetRequest], list[str], list[str]]:
    dataset_requests = parse_dataset_requests(cfg.get("datasets"))
    explicit = getattr(args, "datasets", None) is not None
    add_audit = bool(getattr(args, "with_audit", False))
    audit_only = bool(getattr(args, "audit_only", False))
    include_all = bool(getattr(args, "all", False))

    _validate_dataset_flags(
        explicit=explicit,
        add_audit=add_audit,
        audit_only=audit_only,
        include_all=include_all,
    )

    dataset_requests, skip_audit_info = _apply_audit_selection(
        list(dataset_requests),
        explicit=explicit,
        add_audit=add_audit,
        audit_only=audit_only,
        include_all=include_all,
    )

    if not dataset_requests:
        raise ValueError("未解析到任何数据集，请检查配置或命令行参数")

    dataset_requests, warn_msgs = _apply_vip_filter(
        dataset_requests,
        use_vip=use_vip,
    )
    if not dataset_requests:
        raise ValueError("use_vip=false 时所有数据集均被跳过，无任务可执行")

    info_msgs = _info_messages(
        cfg_missing=cfg_missing,
        explicit=explicit,
        skip_audit_info=skip_audit_info,
        add_audit=add_audit,
        audit_only=audit_only,
        include_all=include_all,
    )

    return dataset_requests, info_msgs, warn_msgs


def _run_multi_dataset_flow(
    cfg: dict,
    dataset_requests: list[DatasetRequest],
    *,
    use_vip: bool,
) -> None:
    pro = init_pro_api(cfg.get("token"))
    if use_vip:
        ensure_enough_credits(pro)
    data_dir = cfg.get("data_dir") or "data"
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
        max_retries=int(cfg.get("max_retries", 3)),
    )

    if os.getenv("TUSHARE_TOKEN_2"):
        print("提示：检测到 TUSHARE_TOKEN_2，已启用多 token 轮询。")

    downloader.run(
        dataset_requests,
        start=parse_yyyymmdd(start_raw),
        end=parse_yyyymmdd(end_raw),
        refresh_periods=int(cfg.get("recent_quarters") or 0),
    )
    _export_income_from_multi(cfg, data_dir, dataset_requests)
