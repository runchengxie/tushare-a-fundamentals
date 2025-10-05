"""High-level workflows around dataset downloads and optional export."""

from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from ..common import (
    _periods_from_cfg,
    ensure_enough_credits,
    eprint,
    init_pro_api,
)
from ..downloader import DatasetRequest, MarketDatasetDownloader, parse_yyyymmdd
from ..commands.export import cmd_export


@dataclass
class DownloadExecutionError(Exception):
    message: str
    exit_code: int = 2

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.message


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


def run_export(export_args: Namespace, strict: bool | None) -> None:
    try:
        cmd_export(export_args)
    except SystemExit as exc:
        if strict:
            raise DownloadExecutionError(str(exc), exit_code=exc.code or 1) from exc
        eprint(f"警告：导出失败（已保留 parquet）：{exc}")
    except Exception as exc:  # pragma: no cover - defensive guard
        eprint(f"警告：导出失败（已保留 parquet）：{exc}")
        if strict:
            raise DownloadExecutionError(str(exc)) from exc


def run_download_pipeline(
    cfg: dict,
    dataset_requests: Sequence[DatasetRequest],
    *,
    use_vip: bool,
    downloader_cls: type[MarketDatasetDownloader] | None = None,
    init_pro: Callable[[Optional[str]], object] | None = None,
    ensure_credits: Callable[[object], None] | None = None,
    periods_from_cfg: Callable[[dict], Sequence[str]] | None = None,
    export_callback: Callable[[Namespace, Optional[bool]], None] | None = None,
) -> None:
    init_fn = init_pro or init_pro_api
    ensure_fn = ensure_credits or ensure_enough_credits
    periods_fn = periods_from_cfg or _periods_from_cfg
    exporter = export_callback or run_export

    ctx = init_fn(cfg.get("token"))
    if use_vip:
        if not ctx.vip_tokens:
            raise DownloadExecutionError(
                "未检测到满足 VIP 门槛（≥5000 积分）的 token。"
                "如需批量抓取，请为至少一个 token 提供 5000 积分或设置 --use-vip=false。"
            )
        ensure_fn(ctx.vip_or_default())
    data_dir = cfg.get("data_dir") or "data"
    max_per_minute = cfg.get("max_per_minute")
    if max_per_minute is None:
        max_per_minute = 90
    start_raw = cfg.get("since")
    end_raw = cfg.get("until")
    if not start_raw or not end_raw:
        periods_window = periods_fn(cfg)
        if periods_window:
            if not start_raw:
                start_raw = periods_window[0]
            if not end_raw:
                end_raw = periods_window[-1]

    downloader_type = downloader_cls or MarketDatasetDownloader
    downloader = downloader_type(
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
        list(dataset_requests),
        start=parse_yyyymmdd(start_raw),
        end=parse_yyyymmdd(end_raw),
        refresh_periods=int(cfg.get("recent_quarters") or 0),
    )
    export_args = _build_export_args(cfg)
    if export_args is not None:
        exporter(export_args, cfg.get("export_strict"))


__all__ = [
    "DownloadExecutionError",
    "run_export",
    "run_download_pipeline",
]
