from __future__ import annotations

import calendar
import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import (
    RetryExhaustedError,
    RetryPolicy,
    _concat_non_empty,
    call_with_retry,
    ensure_ts_code,
    last_publishable_period,
)
from .state_backend import JsonStateBackend, StateBackend
from .transforms.deduplicate import mark_latest

DATE_FMT = "%Y%m%d"
MAX_PAGES = 200


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    api: str
    vip_api: Optional[str] = None
    period_field: Optional[str] = None
    date_field: Optional[str] = None
    date_start_param: str = "start_date"
    date_end_param: str = "end_date"
    primary_keys: Sequence[str] = field(default_factory=tuple)
    dedup_group_keys: Sequence[str] = field(default_factory=tuple)
    default_year_column: str = "ann_date"
    default_start: str = "20000101"
    fields: Optional[str] = None
    report_types: Sequence[int] = field(default_factory=tuple)
    type_param: Optional[str] = None
    type_values: Sequence[str] = field(default_factory=tuple)
    vip_supports_pagination: bool = False
    api_supports_pagination: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    requires_ts_code: bool = False
    code_param: str = "ts_code"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "income": DatasetSpec(
        name="income",
        api="income",
        vip_api="income_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        report_types=(1,),
        fields=None,
        vip_supports_pagination=True,
    ),
    "balancesheet": DatasetSpec(
        name="balancesheet",
        api="balancesheet",
        vip_api="balancesheet_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        vip_supports_pagination=True,
    ),
    "cashflow": DatasetSpec(
        name="cashflow",
        api="cashflow",
        vip_api="cashflow_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        vip_supports_pagination=True,
    ),
    "forecast": DatasetSpec(
        name="forecast",
        api="forecast",
        vip_api="forecast_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "type"),
        dedup_group_keys=("ts_code", "end_date", "type"),
        default_year_column="end_date",
        fields=None,
        vip_supports_pagination=True,
    ),
    "express": DatasetSpec(
        name="express",
        api="express",
        vip_api="express_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        vip_supports_pagination=True,
    ),
    "dividend": DatasetSpec(
        name="dividend",
        api="dividend",
        vip_api=None,
        period_field=None,
        date_field="ann_date",
        primary_keys=(
            "ts_code",
            "ann_date",
            "record_date",
            "ex_date",
            "imp_ann_date",
        ),
        dedup_group_keys=(
            "ts_code",
            "ann_date",
            "record_date",
            "ex_date",
            "imp_ann_date",
        ),
        default_year_column="ann_date",
        fields=None,
    ),
    "fina_indicator": DatasetSpec(
        name="fina_indicator",
        api="fina_indicator",
        vip_api="fina_indicator_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        vip_supports_pagination=True,
    ),
    "fina_audit": DatasetSpec(
        name="fina_audit",
        api="fina_audit",
        vip_api=None,
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        api_supports_pagination=True,
        requires_ts_code=True,
        code_param="ts_code",
    ),
    "fina_mainbz": DatasetSpec(
        name="fina_mainbz",
        api="fina_mainbz",
        vip_api="fina_mainbz_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "bz_item", "type"),
        dedup_group_keys=("ts_code", "end_date", "bz_item", "type"),
        default_year_column="end_date",
        type_param="type",
        type_values=("P", "D"),
        fields=None,
        vip_supports_pagination=True,
    ),
    "disclosure_date": DatasetSpec(
        name="disclosure_date",
        api="disclosure_date",
        vip_api=None,
        period_field="end_date",
        date_field=None,
        primary_keys=(
            "ts_code",
            "end_date",
            "ann_date",
            "pre_date",
            "actual_date",
        ),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        api_supports_pagination=True,
    ),
}


def today_yyyymmdd() -> str:
    return datetime.utcnow().strftime(DATE_FMT)


def parse_yyyymmdd(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if len(trimmed) == 10 and trimmed[4] == "-" and trimmed[7] == "-":
        trimmed = trimmed.replace("-", "")
    try:
        datetime.strptime(trimmed, DATE_FMT)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"无效日期：{value}") from exc
    return trimmed


def month_windows(start: str, end: str) -> List[Tuple[str, str]]:
    s = datetime.strptime(start, DATE_FMT).date().replace(day=1)
    e = datetime.strptime(end, DATE_FMT).date()
    windows: List[Tuple[str, str]] = []
    cur = s
    while cur <= e:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        win_start = cur
        win_end = date(cur.year, cur.month, last_day)
        if win_end > e:
            win_end = e
        windows.append((win_start.strftime(DATE_FMT), win_end.strftime(DATE_FMT)))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return windows


def quarter_end_for(d: date) -> date:
    q = (d.month - 1) // 3 + 1
    month = q * 3
    last_day = calendar.monthrange(d.year, month)[1]
    return date(d.year, month, last_day)


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def quarter_periods(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, DATE_FMT).date()
    e = datetime.strptime(end, DATE_FMT).date()
    aligned = quarter_end_for(s)
    periods: List[str] = []
    cur = aligned
    while cur <= e:
        periods.append(cur.strftime(DATE_FMT))
        cur = add_months(cur, 3)
        cur = quarter_end_for(cur)
    return periods


def move_quarters(period: str, delta: int) -> str:
    base = datetime.strptime(period, DATE_FMT).date()
    shifted = add_months(base, delta * 3)
    shifted = quarter_end_for(shifted)
    return shifted.strftime(DATE_FMT)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_parquet_dataset(  # noqa: C901
    df: pd.DataFrame,
    root: str,
    dataset: str,
    year_col: str,
    *,
    group_keys: Sequence[str] | None = None,
) -> bool:
    if df.empty:
        return True
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    if year_col not in frame.columns:
        frame[year_col] = None
    frame[year_col] = frame[year_col].astype(str)
    years = frame[year_col].str[:4]
    frame["year"] = years.fillna("unknown").replace(
        to_replace=r"(?i)nan|nat|none", value="unknown", regex=True
    )
    dataset_root = Path(root) / dataset
    ensure_dir(dataset_root.as_posix())
    updated_any = False
    for year in sorted({y for y in frame["year"].dropna()}):
        partition_new = frame[frame["year"] == year].copy()
        if partition_new.empty:
            continue
        target_dir = dataset_root / f"year={year}"
        existing = pd.DataFrame()
        if target_dir.exists():
            try:
                tables = [
                    pq.read_table(p.as_posix()) for p in target_dir.glob("*.parquet")
                ]
                if tables:
                    existing = pa.concat_tables(tables).to_pandas()
            except Exception as exc:  # pragma: no cover - I/O errors
                print(f"警告：读取 {target_dir} 失败：{exc}")
        if not existing.empty:
            existing.columns = [c.lower() for c in existing.columns]
            if "year" not in existing.columns:
                existing["year"] = year
        all_cols = sorted({*partition_new.columns, *existing.columns})
        partition_new = partition_new.reindex(columns=all_cols)
        if not existing.empty:
            existing = existing.reindex(columns=all_cols)
        frames_to_concat = [partition_new]
        if not existing.empty:
            frames_to_concat.insert(0, existing)
        combined = _concat_non_empty(frames_to_concat)
        if "retrieved_at" in combined.columns:
            combined["retrieved_at"] = pd.to_datetime(
                combined["retrieved_at"], errors="coerce"
            )
        dedup_keys = [c for c in (group_keys or []) if c in combined.columns]
        extra_sort: List[str] = []
        if "retrieved_at" in combined.columns:
            extra_sort.append("retrieved_at")
        if dedup_keys:
            flagged = mark_latest(
                combined,
                group_keys=dedup_keys,
                extra_sort_keys=extra_sort,
            )
            if "is_latest" in flagged.columns:
                combined = flagged[flagged["is_latest"] == 1].drop(
                    columns=["is_latest"]
                )
            else:
                combined = flagged
            combined = combined.drop_duplicates(subset=dedup_keys)
        else:
            combined = combined.drop_duplicates()
        combined = combined.sort_index()
        with TemporaryDirectory(dir=dataset_root.as_posix()) as tmpdir:
            tmp_year_dir = Path(tmpdir) / f"year={year}"
            ensure_dir(tmp_year_dir.as_posix())
            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(table, (tmp_year_dir / "data.parquet").as_posix())
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(tmp_year_dir.as_posix(), target_dir.as_posix())
        updated_any = True
    return updated_any


class RateLimiter:
    def __init__(self, max_per_minute: int = 90) -> None:
        self.max_per_minute = max_per_minute
        self.calls: List[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.max_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                window_start = now - 60
                self.calls = [t for t in self.calls if t >= window_start]
                if len(self.calls) < self.max_per_minute:
                    self.calls.append(now)
                    return
                sleep_for = 60 - (now - self.calls[0]) + 0.1
            if sleep_for <= 0:
                # Prevent busy wait if timestamps are virtually identical.
                time.sleep(0.05)
            else:
                time.sleep(sleep_for)


@dataclass(frozen=True)
class PeriodCombination:
    report_type: Optional[int] = None
    type_value: Optional[str] = None

    def state_key(self, base: str, spec: DatasetSpec) -> str:
        parts = [base]
        if self.report_type is not None:
            parts.append(f"rt={self.report_type}")
        if spec.type_param and self.type_value is not None:
            parts.append(f"{spec.type_param}={self.type_value}")
        return ":".join(parts)

    def as_params(self, spec: DatasetSpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.report_type is not None:
            params["report_type"] = self.report_type
        if spec.type_param and self.type_value is not None:
            params[spec.type_param] = self.type_value
        return params

    def describe(self, spec: DatasetSpec) -> str:
        parts: List[str] = []
        if self.report_type is not None:
            parts.append(f"report_type={self.report_type}")
        if spec.type_param and self.type_value is not None:
            parts.append(f"{spec.type_param}={self.type_value}")
        if not parts:
            return "默认组合"
        return ", ".join(parts)


@dataclass
class PeriodFetchOutcome:
    frames: List[pd.DataFrame] = field(default_factory=list)
    last_successful_period: Optional[str] = None
    had_failure: bool = False
    failed_periods: List[str] = field(default_factory=list)


@dataclass
class DownloadAccumulator:
    frames: List[pd.DataFrame] = field(default_factory=list)
    updates: List[Tuple[str, str, str]] = field(default_factory=list)
    failures: List[Any] = field(default_factory=list)

    def merge(self, other: "DownloadAccumulator") -> None:
        self.frames.extend(other.frames)
        self.updates.extend(other.updates)
        self.failures.extend(other.failures)


@dataclass
class DatasetRequest:
    name: str
    options: Dict[str, Any] = field(default_factory=dict)


class MarketDatasetDownloader:
    def __init__(
        self,
        pro: Any,
        data_dir: str,
        *,
        use_vip: bool = True,
        max_per_minute: int = 90,
        state_path: Optional[str] = None,
        state_backend: StateBackend | None = None,
        allow_future: bool = False,
        max_retries: int = 3,
    ) -> None:
        self.pro = pro
        self.data_dir = data_dir
        self.use_vip = use_vip
        if getattr(pro, "__is_token_pool__", False):
            self.limiter = RateLimiter(max_per_minute=0)
            if hasattr(pro, "set_rate"):
                pro.set_rate(max_per_minute or 90)
        else:
            self.limiter = RateLimiter(max_per_minute=max_per_minute)
        self.allow_future = allow_future
        self.retry_policy = RetryPolicy(max_retries=max_retries)
        state_file = (
            Path(state_path) if state_path else Path(data_dir) / "_state" / "state.json"
        )
        self.state = state_backend or JsonStateBackend(state_file)

    def run(
        self,
        requests: Sequence[DatasetRequest],
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        refresh_periods: int = 0,
    ) -> None:
        if not requests:
            raise ValueError("datasets 列表不能为空")
        end_date = parse_yyyymmdd(end) or today_yyyymmdd()
        start_date = parse_yyyymmdd(start)
        for req in requests:
            spec = self._spec_for(req.name)
            print(f"[{datetime.now()}] >>> 抓取 {spec.name}")
            if spec.requires_ts_code:
                print(f"提示：{spec.name} 仅支持按股票循环，本次将枚举 ts_code")
            self._download_dataset(
                spec,
                req.options,
                start_date,
                end_date,
                refresh_periods,
            )
            print(f"[{datetime.now()}] <<< 完成 {spec.name}")

    def _spec_for(self, name: str) -> DatasetSpec:
        if name not in DATASET_SPECS:
            raise KeyError(f"未知数据集：{name}")
        return DATASET_SPECS[name]

    def _download_dataset(
        self,
        spec: DatasetSpec,
        options: Dict[str, Any],
        start_date: Optional[str],
        end_date: str,
        refresh_periods: int,
    ) -> None:
        if spec.requires_ts_code:
            self._download_per_stock(
                spec,
                options,
                start_date,
                end_date,
                refresh_periods,
            )
            return
        if spec.period_field is not None:
            self._download_periodic(
                spec,
                options,
                start_date,
                end_date,
                refresh_periods,
            )
        if spec.date_field is not None:
            self._download_calendar(spec, options, start_date, end_date)

    def _download_periodic(
        self,
        spec: DatasetSpec,
        options: Dict[str, Any],
        start_date: Optional[str],
        end_date: str,
        refresh_periods: int,
    ) -> None:
        report_types = self._resolve_report_types(spec, options)
        type_values = self._resolve_type_values(spec, options)
        combinations = self._build_period_combinations(report_types, type_values)
        method_name, paginate = self._resolve_method(spec)
        period_end = self._bounded_period_end(end_date)

        accumulator = DownloadAccumulator()
        for combo in combinations:
            combo_result = self._run_periodic_combo(
                spec,
                combo,
                start_date,
                period_end,
                refresh_periods,
                method_name,
                paginate,
            )
            accumulator.merge(combo_result)

        combined = self._concat_and_dedup(accumulator.frames, spec)
        write_ok = True
        if combined is not None:
            write_ok = write_parquet_dataset(
                combined,
                self.data_dir,
                spec.name,
                spec.default_year_column,
                group_keys=spec.dedup_group_keys or spec.primary_keys,
            )
        failure_entries = [
            {"combo": desc, "periods": periods}
            for desc, periods in accumulator.failures
        ]
        self._record_failures(spec, failure_entries, "periods")
        if write_ok:
            for dataset, key, value in accumulator.updates:
                self.state.set(dataset, key, value)
            for desc, periods in accumulator.failures:
                failed = ", ".join(periods)
                print(f"提示：{spec.name} {desc} 未成功的 period: {failed}")

    def _download_per_stock(
        self,
        spec: DatasetSpec,
        options: Dict[str, Any],
        start_date: Optional[str],
        end_date: str,
        refresh_periods: int,
    ) -> None:
        if spec.period_field is None:
            raise ValueError(f"{spec.name} 缺少 period_field，无法按股票抓取")
        stock_df = self._resolve_stock_universe(options)
        if stock_df.empty:
            print(f"警告：{spec.name} 未找到可用股票清单，已跳过")
            return
        report_types = self._resolve_report_types(spec, options)
        type_values = self._resolve_type_values(spec, options)
        combinations = self._build_period_combinations(report_types, type_values)
        method_name, paginate = self._resolve_method(spec)
        period_end = self._bounded_period_end(end_date)
        accumulator = DownloadAccumulator()
        for _, row in stock_df.iterrows():
            ts_code = str(row.get("ts_code", "")).strip()
            if not ts_code:
                continue
            earliest_period = self._normalize_period(row.get("earliest_period"))
            stock_result = self._run_stock_download(
                spec,
                combinations,
                ts_code,
                earliest_period,
                start_date,
                period_end,
                refresh_periods,
                method_name,
                paginate,
            )
            accumulator.merge(stock_result)

        combined = self._concat_and_dedup(accumulator.frames, spec)
        write_ok = True
        if combined is not None:
            write_ok = write_parquet_dataset(
                combined,
                self.data_dir,
                spec.name,
                spec.default_year_column,
                group_keys=spec.dedup_group_keys or spec.primary_keys,
            )
        self._record_failures(spec, accumulator.failures, "per_stock")
        if write_ok:
            for dataset, key, value in accumulator.updates:
                self.state.set(dataset, key, value)
            for entry in accumulator.failures:
                periods = ", ".join(entry["periods"])
                print(
                    f"提示：{spec.name} {entry['ts_code']} 未成功的 period: {periods}"
                )

    def _run_periodic_combo(
        self,
        spec: DatasetSpec,
        combo: PeriodCombination,
        start_date: Optional[str],
        period_end: str,
        refresh_periods: int,
        method_name: str,
        paginate: bool,
    ) -> DownloadAccumulator:
        result = DownloadAccumulator()
        state_key = combo.state_key("last_period", spec)
        last_period = self.state.get(spec.name, state_key, spec.default_start)
        effective_start = self._resolve_periodic_start(
            spec,
            start_date,
            last_period,
            refresh_periods,
        )
        if effective_start is None or effective_start > period_end:
            return result
        periods = quarter_periods(effective_start, period_end)
        if not periods:
            return result
        outcome = self._collect_periods(spec, periods, combo, method_name, paginate)
        result.frames.extend(outcome.frames)
        if outcome.last_successful_period is not None:
            result.updates.append(
                (spec.name, state_key, outcome.last_successful_period)
            )
        if outcome.failed_periods:
            result.failures.append((combo.describe(spec), outcome.failed_periods))
        return result

    def _run_stock_download(
        self,
        spec: DatasetSpec,
        combinations: Sequence[PeriodCombination],
        ts_code: str,
        earliest_period: Optional[str],
        start_date: Optional[str],
        period_end: str,
        refresh_periods: int,
        method_name: str,
        paginate: bool,
    ) -> DownloadAccumulator:
        result = DownloadAccumulator()
        for combo in combinations:
            combo_result = self._run_stock_combo(
                spec,
                combo,
                ts_code,
                earliest_period,
                start_date,
                period_end,
                refresh_periods,
                method_name,
                paginate,
            )
            result.merge(combo_result)
        return result

    def _run_stock_combo(
        self,
        spec: DatasetSpec,
        combo: PeriodCombination,
        ts_code: str,
        earliest_period: Optional[str],
        start_date: Optional[str],
        period_end: str,
        refresh_periods: int,
        method_name: str,
        paginate: bool,
    ) -> DownloadAccumulator:
        result = DownloadAccumulator()
        state_key = combo.state_key(f"last_period:ts={ts_code}", spec)
        default_start = earliest_period or spec.default_start
        last_period = self.state.get(spec.name, state_key, default_start)
        effective_start = self._resolve_periodic_start(
            spec,
            start_date,
            last_period,
            refresh_periods,
            lower_bound=earliest_period,
        )
        if effective_start is None or effective_start > period_end:
            return result
        periods = quarter_periods(effective_start, period_end)
        if not periods:
            return result

        combo_desc = combo.describe(spec)
        failed_periods: List[str] = []
        last_success: Optional[str] = None
        for period_value in periods:
            frame, success = self._call_stock_period(
                spec,
                method_name,
                paginate,
                combo,
                ts_code,
                period_value,
            )
            if not success:
                failed_periods.append(period_value)
                print(
                    "警告："
                    f"{spec.name} {combo_desc} 针对 {ts_code} "
                    f"在 {period_value} 抓取失败，请稍后手动排查"
                )
                continue
            last_success = period_value
            if frame is not None and not frame.empty:
                result.frames.append(frame)
        if last_success is not None:
            result.updates.append((spec.name, state_key, last_success))
        if failed_periods:
            result.failures.append(
                {
                    "ts_code": ts_code,
                    "combo": combo_desc,
                    "periods": failed_periods,
                }
            )
        return result

    def _call_stock_period(
        self,
        spec: DatasetSpec,
        method_name: str,
        paginate: bool,
        combo: PeriodCombination,
        ts_code: str,
        period_value: str,
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        params = dict(spec.extra_params)
        params[spec.period_field] = period_value
        params[spec.code_param] = ts_code
        params.update(combo.as_params(spec))
        df = self._call_api(
            method_name,
            params,
            spec.fields,
            paginate=paginate,
        )
        if df is None:
            return None, False
        if df.empty:
            return df, True
        frame = df.copy()
        if spec.code_param not in frame.columns:
            frame[spec.code_param] = ts_code
        if (
            spec.type_param
            and spec.type_param not in frame.columns
            and combo.type_value is not None
        ):
            frame[spec.type_param] = combo.type_value
        if combo.report_type is not None and "report_type" not in frame.columns:
            frame["report_type"] = combo.report_type
        return frame, True

    def _resolve_periodic_start(
        self,
        spec: DatasetSpec,
        start_override: Optional[str],
        last_period: Optional[str],
        refresh_periods: int,
        *,
        lower_bound: Optional[str] = None,
    ) -> Optional[str]:
        candidate = start_override or last_period or spec.default_start
        candidate = self._normalize_period(candidate)
        candidate = self._max_period(candidate, spec.default_start)
        if lower_bound:
            candidate = self._max_period(candidate, lower_bound)
        backfill: Optional[str] = None
        if refresh_periods and last_period:
            backfill = move_quarters(last_period, -max(refresh_periods, 0))
            backfill = self._normalize_period(backfill)
            backfill = self._max_period(backfill, spec.default_start)
            if lower_bound:
                backfill = self._max_period(backfill, lower_bound)
        if backfill is not None and (
            start_override is None or (candidate is not None and backfill < candidate)
        ):
            candidate = backfill
        return candidate

    def _download_calendar(
        self,
        spec: DatasetSpec,
        options: Dict[str, Any],
        start_date: Optional[str],
        end_date: str,
    ) -> None:
        state_key = "last_date"
        default_start = spec.default_start
        last_date = self.state.get(spec.name, state_key, default_start)
        effective_start = start_date or last_date or spec.default_start
        effective_start = max(effective_start, spec.default_start)
        windows = month_windows(effective_start, end_date)
        if not windows:
            return
        collected: List[pd.DataFrame] = []
        last_completed: Optional[str] = None
        failed_windows: List[str] = []
        for win_start, win_end in windows:
            df = self._fetch_window(spec, win_start, win_end)
            if df is None:
                failed_windows.append(f"{win_start}-{win_end}")
                continue
            if not df.empty:
                collected.append(df)
            last_completed = win_end
        combined = self._concat_and_dedup(collected, spec)
        write_ok = True
        if combined is not None:
            year_col = spec.default_year_column
            write_ok = write_parquet_dataset(
                combined,
                self.data_dir,
                spec.name,
                year_col,
                group_keys=spec.dedup_group_keys or spec.primary_keys,
            )
        failure_entries = [{"window": win} for win in failed_windows]
        self._record_failures(spec, failure_entries, "windows")
        if write_ok and last_completed is not None:
            self.state.set(spec.name, state_key, last_completed)
            if failed_windows:
                print(
                    f"提示：{spec.name} 部分窗口抓取失败：{', '.join(failed_windows)}"
                )

    def _resolve_report_types(
        self, spec: DatasetSpec, options: Dict[str, Any]
    ) -> Sequence[int]:
        if "report_types" in options:
            vals = options["report_types"]
            if isinstance(vals, str):
                parts = [p.strip() for p in vals.split(",") if p.strip()]
                return [int(p) for p in parts]
            if isinstance(vals, Sequence):
                return [int(v) for v in vals]
        if spec.report_types:
            return list(spec.report_types)
        return []

    def _resolve_type_values(
        self, spec: DatasetSpec, options: Dict[str, Any]
    ) -> Sequence[str]:
        if spec.type_param is None:
            return []
        if spec.type_param in options:
            vals = options[spec.type_param]
            if isinstance(vals, str):
                parts = [p.strip() for p in vals.split(",") if p.strip()]
                return parts
            if isinstance(vals, Sequence):
                return [str(v) for v in vals]
        if spec.type_values:
            return list(spec.type_values)
        return []

    def _resolve_stock_universe(self, options: Dict[str, Any]) -> pd.DataFrame:
        explicit = self._normalize_code_list(options.get("ts_codes"))
        if explicit:
            return pd.DataFrame({"ts_code": explicit})
        inferred = self._load_codes_from_fact()
        if inferred is not None:
            return inferred
        fallback = self._load_codes_from_stock_basic()
        if fallback is not None:
            return fallback
        return pd.DataFrame(columns=["ts_code", "earliest_period"])

    def _normalize_code_list(self, raw: Any) -> List[str]:
        if raw is None:
            return []
        items: List[str]
        if isinstance(raw, str):
            candidates = raw.replace("\n", ",").split(",")
            items = [c.strip() for c in candidates if c.strip()]
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            items = [str(c).strip() for c in raw if str(c).strip()]
        else:
            items = [str(raw).strip()]
        seen: Set[str] = set()
        unique: List[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _load_codes_from_fact(self) -> Optional[pd.DataFrame]:
        root = Path(self.data_dir) / "dataset=fact_income_cum"
        if not root.exists():
            return None
        frames: List[pd.DataFrame] = []
        for file in sorted(root.rglob("*.parquet")):
            try:
                frames.append(pd.read_parquet(file, columns=["ts_code", "end_date"]))
            except Exception as exc:  # pragma: no cover - defensive I/O
                print(f"警告：读取 {file} 失败：{exc}")
        if not frames:
            return None
        combined = _concat_non_empty(frames)
        if combined.empty:
            return None
        combined = ensure_ts_code(combined, context="fact_income_cum")
        combined["end_date"] = combined["end_date"].astype(str)
        grouped = combined.groupby("ts_code")["end_date"].min().reset_index()
        grouped = grouped.rename(columns={"end_date": "earliest_period"})
        return grouped

    def _load_codes_from_stock_basic(self) -> Optional[pd.DataFrame]:
        params = {"list_status": "L"}
        df = self._call_api(
            "stock_basic",
            params,
            fields="ts_code,list_date",
            paginate=True,
        )
        if df is None or df.empty:
            return None
        frame = df.copy()
        frame["ts_code"] = frame["ts_code"].astype(str)
        if "list_date" in frame.columns:
            frame["earliest_period"] = frame["list_date"].apply(
                self._list_date_to_period
            )
        else:
            frame["earliest_period"] = None
        return frame[["ts_code", "earliest_period"]]

    def _normalize_period(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            text = text.replace("-", "")
        if len(text) != 8:
            return None
        return text

    def _max_period(self, value: Optional[str], floor: Optional[str]) -> Optional[str]:
        if value is None:
            return floor
        if floor is None:
            return value
        return max(value, floor)

    def _list_date_to_period(self, raw: Any) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        normalized = self._normalize_period(text)
        if normalized is None:
            return None
        try:
            day = datetime.strptime(normalized, DATE_FMT).date()
        except ValueError:
            return None
        return quarter_end_for(day).strftime(DATE_FMT)

    def _build_period_combinations(
        self,
        report_types: Sequence[int],
        type_values: Sequence[str],
    ) -> List[PeriodCombination]:
        rt_values = list(report_types) if report_types else [None]
        type_opts = list(type_values) if type_values else [None]
        return [
            PeriodCombination(report_type=rt, type_value=tv)
            for rt in rt_values
            for tv in type_opts
        ]

    def _resolve_method(self, spec: DatasetSpec) -> Tuple[str, bool]:
        vip = spec.vip_api if self.use_vip else None
        method_name = vip or spec.api
        paginate = spec.vip_supports_pagination if vip else spec.api_supports_pagination
        return method_name, paginate

    def _bounded_period_end(self, end_date: str) -> str:
        if self.allow_future:
            return end_date
        limit = last_publishable_period(date.today())
        return min(end_date, limit)

    def _collect_periods(
        self,
        spec: DatasetSpec,
        periods: Sequence[str],
        combo: PeriodCombination,
        method_name: str,
        paginate: bool,
    ) -> PeriodFetchOutcome:
        outcome = PeriodFetchOutcome()
        for period_value in periods:
            df, success = self._fetch_period(
                spec,
                method_name,
                paginate,
                period_value,
                combo,
            )
            if not success:
                outcome.had_failure = True
                outcome.failed_periods.append(period_value)
                combo_desc = combo.describe(spec)
                print(
                    f"警告：{spec.name} {combo_desc} 在 {period_value} 抓取失败，"
                    "请稍后手动排查"
                )
                continue
            outcome.last_successful_period = period_value
            if df is not None and not df.empty:
                outcome.frames.append(df)
        return outcome

    def _fetch_period(
        self,
        spec: DatasetSpec,
        method_name: str,
        paginate: bool,
        period_value: str,
        combo: PeriodCombination,
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        params = dict(spec.extra_params)
        params[spec.period_field] = period_value
        params.update(combo.as_params(spec))
        df = self._call_api(
            method_name,
            params,
            spec.fields,
            paginate=paginate,
        )
        if df is None:
            return None, False
        if df.empty:
            return df, True
        frame = df.copy()
        if (
            spec.type_param
            and spec.type_param not in frame.columns
            and combo.type_value is not None
        ):
            frame[spec.type_param] = combo.type_value
        if combo.report_type is not None and "report_type" not in frame.columns:
            frame["report_type"] = combo.report_type
        return frame, True

    def _fetch_window(
        self, spec: DatasetSpec, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        params = dict(spec.extra_params)
        params[spec.date_start_param] = start
        params[spec.date_end_param] = end
        df = self._call_api(
            spec.api,
            params,
            spec.fields,
            paginate=spec.api_supports_pagination,
        )
        return df

    def _call_api(  # noqa: C901
        self,
        api_name: str,
        params: Dict[str, Any],
        fields: Optional[str],
        *,
        paginate: bool,
    ) -> Optional[pd.DataFrame]:
        func = getattr(self.pro, api_name, None)
        if func is None:

            def fallback_call(**kwargs: Any) -> pd.DataFrame:
                return self.pro.query(api_name, **kwargs)

            func = fallback_call
        policy = self.retry_policy

        def _log_retry(attempt: int, exc: Exception, wait_seconds: float) -> None:
            print(
                f"警告：调用 {api_name} 异常，{wait_seconds:.1f}s 后重试"
                f"（第 {attempt}/{policy.max_retries} 次）: {exc}"
            )

        def _invoke() -> pd.DataFrame:  # noqa: C901
            limit_val = params.get("limit", 10000) or 10000
            try:
                limit = int(limit_val)
            except (TypeError, ValueError):
                limit = 10000
            if limit <= 0:
                limit = 10000
            rows: List[pd.DataFrame] = []
            offset = 0
            pages = 0
            use_pagination = paginate
            page_limit_hit = False
            seen_signatures: Set[bytes] = set()
            while True:
                call_params = params.copy()
                if use_pagination or offset > 0:
                    call_params["limit"] = limit
                    call_params["offset"] = offset
                elif "limit" in call_params:
                    call_params.setdefault("limit", limit)
                if fields:
                    call_params.setdefault("fields", fields)
                self.limiter.wait()
                df = func(**call_params)
                if df is None or df.empty:
                    break
                signature = hashlib.sha1(
                    pd.util.hash_pandas_object(df, index=True).values.tobytes()
                ).digest()
                if signature in seen_signatures:
                    print(
                        "警告："
                        f"调用 {api_name} 分页出现重复结果（offset={offset}），已提前终止"
                    )
                    break
                seen_signatures.add(signature)
                rows.append(df)
                pages += 1
                if len(df) < limit or pages >= MAX_PAGES:
                    if pages >= MAX_PAGES:
                        page_limit_hit = True
                    break
                offset += limit
                use_pagination = True
            if not rows:
                return pd.DataFrame()
            if page_limit_hit:
                print(f"警告：调用 {api_name} 达到分页上限 {MAX_PAGES}，结果可能被截断")
            return _concat_non_empty(rows)

        try:
            return call_with_retry(
                _invoke,
                policy=policy,
                description=f"调用 {api_name}",
                on_retry=_log_retry,
            )
        except RetryExhaustedError as exc:
            print(f"警告：调用 {api_name} 失败：{exc.last_exception}")
            return None

    def _record_failures(
        self,
        spec: DatasetSpec,
        entries: Sequence[Dict[str, Any]],
        kind: str,
    ) -> None:
        failure_root = Path(self.data_dir) / "_state" / "failures"
        failure_path = failure_root / f"{spec.name}_{kind}.json"
        if not entries:
            if failure_path.exists():
                try:
                    failure_path.unlink()
                except OSError:
                    pass
            return
        ensure_dir(failure_root.as_posix())
        payload = {
            "dataset": spec.name,
            "kind": kind,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": entries,
        }
        failure_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            "utf-8",
        )

    def _concat_and_dedup(
        self,
        frames: Sequence[pd.DataFrame],
        spec: DatasetSpec,
    ) -> Optional[pd.DataFrame]:
        valid = [df for df in frames if df is not None and not df.empty]
        if not valid:
            return None
        prepared: List[pd.DataFrame] = []
        timestamp = pd.Timestamp.utcnow()
        for df in valid:
            frame = df.copy()
            if "retrieved_at" not in frame.columns:
                frame["retrieved_at"] = timestamp
            prepared.append(frame)
        combined = _concat_non_empty(prepared)
        for col in ("ts_code", "end_date", "ann_date"):
            if col in combined.columns:
                combined[col] = combined[col].astype(str)
        dedup_keys = [
            c
            for c in (spec.dedup_group_keys or spec.primary_keys)
            if c in combined.columns
        ]
        if "retrieved_at" in combined.columns:
            combined["retrieved_at"] = pd.to_datetime(
                combined["retrieved_at"], errors="coerce"
            )
        extra_sort_keys: List[str] = []
        if "retrieved_at" in combined.columns:
            extra_sort_keys.append("retrieved_at")
        if dedup_keys:
            flagged = mark_latest(
                combined,
                group_keys=dedup_keys,
                extra_sort_keys=extra_sort_keys,
            )
            if "is_latest" in flagged.columns:
                combined = flagged[flagged["is_latest"] == 1].drop(
                    columns=["is_latest"]
                )
            else:
                combined = flagged
            combined = combined.drop_duplicates(subset=dedup_keys)
        else:
            combined = combined.drop_duplicates()
        return combined


def parse_dataset_requests(raw: Any) -> List[DatasetRequest]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
        return [DatasetRequest(name=item) for item in items]
    if isinstance(raw, Sequence):
        out: List[DatasetRequest] = []
        for item in raw:
            if isinstance(item, str):
                out.append(DatasetRequest(name=item))
            elif isinstance(item, dict):
                name = item.get("name")
                if not name:
                    continue
                options = {k: v for k, v in item.items() if k != "name"}
                out.append(DatasetRequest(name=name, options=options))
        return out
    raise TypeError("datasets 配置格式不支持")


__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "DatasetRequest",
    "MarketDatasetDownloader",
    "month_windows",
    "quarter_periods",
    "move_quarters",
    "parse_yyyymmdd",
    "parse_dataset_requests",
]
