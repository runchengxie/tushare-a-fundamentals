import argparse
import sys
from argparse import Namespace

from ..common import (
    DEFAULT_FIELDS,
    _periods_from_cfg as _periods_from_cfg_fn,
    ensure_enough_credits as _ensure_enough_credits,
    eprint,
    init_pro_api as _init_pro_api,
    load_yaml,
    merge_config,
    normalize_fields,
    parse_report_types,
)
from ..downloader import (
    DatasetRequest,
    MarketDatasetDownloader as _MarketDatasetDownloader,
    parse_dataset_requests,
)
from .export import cmd_export

# Backwards compatibility for tests that patch the downloader in this module.
MarketDatasetDownloader = _MarketDatasetDownloader
from ..workflows.download import (
    DownloadExecutionError,
    run_download_pipeline,
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


# Keep frequently patched helpers available for tests.
init_pro_api = _init_pro_api
_periods_from_cfg = _periods_from_cfg_fn
ensure_enough_credits = _ensure_enough_credits


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

    try:
        run_download_pipeline(
            cfg,
            dataset_requests,
            use_vip=use_vip,
            downloader_cls=MarketDatasetDownloader,
            init_pro=init_pro_api,
            ensure_credits=ensure_enough_credits,
            periods_from_cfg=_periods_from_cfg,
            export_callback=_run_export,
        )
    except DownloadExecutionError as exc:
        eprint(f"错误：{exc.message}")
        sys.exit(exc.exit_code)


AUDIT_DATASET_NAME = "fina_audit"
DIVIDEND_DATASET_NAME = "dividend"
VIP_ONLY_DATASETS = {"forecast", "fina_indicator", "fina_mainbz"}

DEFAULT_DATASET_CONFIG = [
    {"name": "income", "report_types": [1]},
    {"name": "balancesheet", "report_types": [1]},
    {"name": "cashflow", "report_types": [1]},
    {"name": "forecast"},
    {"name": "express"},
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
    dividend_only: bool,
) -> None:
    if explicit and (add_audit or audit_only or include_all):
        raise ValueError("--datasets 不可与 --with-audit/--audit-only/--all 同时使用")
    if audit_only and (add_audit or include_all):
        raise ValueError("--audit-only 不可与 --with-audit 或 --all 同时使用")
    if add_audit and include_all:
        raise ValueError("--with-audit 与 --all 不可同时使用")
    if dividend_only and (explicit or add_audit or audit_only or include_all):
        raise ValueError(
            "--dividend-only 不可与 --datasets/--with-audit/--audit-only/--all 同时使用"
        )


def _apply_dividend_selection(
    dataset_requests: list[DatasetRequest],
    *,
    explicit: bool,
    include_all: bool,
    dividend_only: bool,
) -> tuple[list[DatasetRequest], bool]:
    if dividend_only:
        return [_default_request_for(DIVIDEND_DATASET_NAME)], False
    if explicit:
        return list(dataset_requests), False

    non_dividend: list[DatasetRequest] = []
    dividend_list: list[DatasetRequest] = []
    for req in dataset_requests:
        if req.name == DIVIDEND_DATASET_NAME:
            dividend_list.append(req)
        else:
            non_dividend.append(req)

    if include_all:
        if dividend_list:
            return non_dividend + dividend_list, False
        return non_dividend + [
            _default_request_for(DIVIDEND_DATASET_NAME)
        ], False

    if dividend_list:
        return non_dividend + dividend_list, False

    return non_dividend, True


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
    skip_dividend_info: bool,
    add_audit: bool,
    audit_only: bool,
    include_all: bool,
    dividend_only: bool,
) -> list[str]:
    infos: list[str] = []
    if skip_audit_info and not (add_audit or audit_only or include_all):
        infos.append(
            "提示：默认跳过 fina_audit（需按股票循环）。"
            "使用 --with-audit / --audit-only / --all 可显式启用。"
        )
    if skip_dividend_info and not dividend_only:
        infos.append(
            "提示：默认跳过 dividend（逐日抓取，耗时较长）。"
            "使用 --dividend-only 可独立运行。"
        )
    if cfg_missing and not explicit and not dividend_only:
        infos.append(
            "提示：未找到配置文件，已按示例配置启用默认数据集（默认跳过 fina_audit 与 dividend）。"
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
    dividend_only = bool(getattr(args, "dividend_only", False))

    _validate_dataset_flags(
        explicit=explicit,
        add_audit=add_audit,
        audit_only=audit_only,
        include_all=include_all,
        dividend_only=dividend_only,
    )

    dataset_requests, skip_dividend_info = _apply_dividend_selection(
        list(dataset_requests),
        explicit=explicit,
        include_all=include_all,
        dividend_only=dividend_only,
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
        skip_dividend_info=skip_dividend_info,
        add_audit=add_audit,
        audit_only=audit_only,
        include_all=include_all,
        dividend_only=dividend_only,
    )

    return dataset_requests, info_msgs, warn_msgs
