import argparse
import os
import sys
from argparse import Namespace

from ..common import (
    DEFAULT_FIELDS,
    _periods_from_cfg,
    ensure_enough_credits,
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
        "audit_quarters": None,
        "audit_years": None,
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
        "export_enabled": False,
        "export_out_dir": None,
        "export_out_format": "csv",
        "export_kinds": "",
        "export_annual_strategy": "cumulative",
        "export_years": None,
        "export_strict": False,
        "max_retries": 3,
        "progress": "auto",
    }


def _collect_cli_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "years": getattr(args, "years", None),
        "quarters": getattr(args, "quarters", None),
        "since": getattr(args, "since", None),
        "until": getattr(args, "until", None),
        "audit_quarters": getattr(args, "audit_quarters", None),
        "audit_years": getattr(args, "audit_years", None),
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
        "progress": getattr(args, "progress", None),
    }
    if getattr(args, "export_enabled", None) is not None:
        overrides["export_enabled"] = getattr(args, "export_enabled")
    if getattr(args, "no_export", False):
        overrides["export_enabled"] = False
    return overrides


def _build_export_args(cfg: dict) -> Namespace | None:
    if not cfg.get("export_enabled", False):
        return None
    data_dir = cfg.get("data_dir") or "data"
    export_out_dir_cfg = cfg.get("export_out_dir")
    if export_out_dir_cfg:
        export_out_dir = export_out_dir_cfg
    else:
        export_out_dir = os.path.normpath(cfg.get("outdir") or data_dir)

    export_kinds_cfg = cfg.get("export_kinds")
    if isinstance(export_kinds_cfg, (list, tuple, set)):
        export_kinds = ",".join(
            str(k).strip() for k in export_kinds_cfg if str(k).strip()
        )
    elif export_kinds_cfg is None:
        export_kinds = ""
    else:
        export_kinds = str(export_kinds_cfg)

    export_years = cfg.get("export_years")
    if export_years is None:
        export_years = cfg.get("years")

    return Namespace(
        dataset_root=data_dir,
        years=export_years,
        kinds=export_kinds,
        annual_strategy=cfg.get("export_annual_strategy", "cumulative"),
        out_format=(cfg.get("export_out_format") or "csv").lower(),
        out_dir=export_out_dir,
        prefix=cfg.get("prefix") or "income",
        flat_datasets=cfg.get("export_flat_datasets", "auto"),
        flat_exclude=cfg.get("export_flat_exclude", ""),
        split_by=cfg.get("export_split_by", "none"),
        gzip=bool(cfg.get("export_gzip", False)),
        no_income=bool(cfg.get("export_no_income", False)),
        no_flat=bool(cfg.get("export_no_flat", False)),
        progress=cfg.get("progress", "auto"),
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
    audit_only = bool(getattr(args, "audit_only", False))
    cli_max_retries_provided = getattr(args, "max_retries", None) is not None
    cfg_max_retries_provided = False
    if isinstance(cfg_file, dict):
        cfg_max_retries_value = cfg_file.get("max_retries")
        cfg_max_retries_provided = cfg_max_retries_value is not None
    if audit_only and not cli_max_retries_provided and not cfg_max_retries_provided:
        defaults["max_retries"] = 5
    audit_window_missing = False
    if audit_only:
        period_flag_names = ("since", "until", "quarters", "years")
        audit_flag_names = ("audit_quarters", "audit_years")

        def _provided(value: object) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            return True

        cli_has_window = any(
            _provided(getattr(args, name, None)) for name in period_flag_names
        )
        cfg_has_window = False
        audit_cfg_has_window = False
        if isinstance(cfg_file, dict):
            for name in period_flag_names:
                if _provided(cfg_file.get(name)):
                    cfg_has_window = True
                    break
            for name in audit_flag_names:
                if _provided(cfg_file.get(name)):
                    audit_cfg_has_window = True
                    break
        audit_cli_has_window = any(
            _provided(getattr(args, name, None)) for name in audit_flag_names
        )
        audit_window_missing = not (
            cli_has_window or cfg_has_window or audit_cli_has_window or audit_cfg_has_window
        )
    cfg_missing = not bool(cfg_file)
    recent_quarters_from_cli = getattr(args, "recent_quarters", None) is not None
    recent_quarters_from_cfg = False
    if isinstance(cfg_file, dict):
        recent_quarters_from_cfg = cfg_file.get("recent_quarters") is not None
    if cfg_missing:
        defaults["datasets"] = DEFAULT_DATASET_CONFIG
    cli_overrides = _collect_cli_overrides(args)
    cfg = merge_config(cli_overrides, cfg_file, defaults)
    if audit_only:
        period_flag_names = ("since", "until", "quarters", "years")

        def _provided(value: object) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            return True

        audit_quarters = cfg.get("audit_quarters")
        audit_years = cfg.get("audit_years")
        if _provided(audit_quarters):
            cfg["quarters"] = audit_quarters
            cfg["years"] = None
        elif _provided(audit_years):
            cfg["years"] = audit_years
            cfg["quarters"] = None
        elif audit_window_missing:
            cfg["quarters"] = 1
            cfg["years"] = None
        if not recent_quarters_from_cli and not recent_quarters_from_cfg:
            window_quarters_raw = cfg.get("quarters")
            try:
                window_quarters = int(window_quarters_raw) if window_quarters_raw else 0
            except (TypeError, ValueError):
                window_quarters = 0
            if window_quarters > 0:
                recent_raw = cfg.get("recent_quarters")
                try:
                    recent_value = int(recent_raw) if recent_raw is not None else 0
                except (TypeError, ValueError):
                    recent_value = 0
                if recent_value > window_quarters:
                    cfg["recent_quarters"] = window_quarters
    cfg["report_types"] = parse_report_types(cfg.get("report_types"))
    cfg["fields"] = normalize_fields(cfg.get("fields"))
    raw_progress = str(cfg.get("progress", "auto") or "auto").strip().lower()
    if raw_progress not in {"auto", "rich", "plain", "none"}:
        raw_progress = "auto"
    cfg["progress"] = raw_progress
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
    ctx = init_pro_api(cfg.get("token"))
    if use_vip:
        if not ctx.vip_tokens:
            eprint(
                "错误：未检测到满足 VIP 门槛（≥5000 积分）的 token。"
                "如需批量抓取，请为至少一个 token 提供 5000 积分或设置 --use-vip=false。"
            )
            sys.exit(2)
        ensure_enough_credits(ctx.vip_or_default())
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
        ctx.any_client,
        data_dir,
        vip_pro=ctx.vip_client,
        use_vip=use_vip,
        max_per_minute=max_per_minute,
        state_path=cfg.get("state_path"),
        allow_future=bool(cfg.get("allow_future")),
        max_retries=int(cfg.get("max_retries", 3)),
        progress_mode=cfg.get("progress", "auto"),
    )

    downloader.run(
        dataset_requests,
        start=parse_yyyymmdd(start_raw),
        end=parse_yyyymmdd(end_raw),
        refresh_periods=int(cfg.get("recent_quarters") or 0),
    )
    export_args = _build_export_args(cfg)
    if export_args is not None:
        _run_export(export_args, cfg.get("export_strict"))
