from __future__ import annotations

import calendar
import hashlib
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .common import (
    RetryExhaustedError,
    RetryPolicy,
    _concat_non_empty,
    call_with_retry,
    ensure_ts_code,
    last_publishable_period,
)
from .dataset_specs import DATASET_SPECS, DatasetSpec
from .storage import (
    ensure_dir,
    merge_and_deduplicate,
    write_failure_report,
    write_parquet_dataset,
    pq,
)
from .state_backend import JsonStateBackend, StateBackend
from .progress import ProgressManager

DATE_FMT = "%Y%m%d"
MAX_PAGES = 200
FRAME_FLUSH_THRESHOLD_ROWS = 200_000
SAFETY_EPS = 0.1


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


class RateLimiter:
    def __init__(self, max_per_minute: int = 90) -> None:
        self.max_per_minute = max(int(max_per_minute), 0)
        self.calls: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, now: Optional[float] = None) -> tuple[bool, float]:
        """Attempt to reserve a call slot without sleeping."""

        if self.max_per_minute <= 0:
            return True, 0.0
        timestamp = time.time() if now is None else now
        with self._lock:
            window_start = timestamp - 60.0
            while self.calls and self.calls[0] < window_start:
                self.calls.popleft()
            if len(self.calls) < self.max_per_minute:
                self.calls.append(timestamp)
                return True, 0.0
            if not self.calls:
                return False, 0.05
            wait_for = self.calls[0] + 60.0 - timestamp
            return False, max(wait_for, 0.0)

    def acquire(self) -> None:
        if self.max_per_minute <= 0:
            return
        while True:
            ok, wait_for = self.try_acquire()
            if ok:
                return
            # pad a tiny epsilon so we re-enter strictly after the 60-second window
            time.sleep(max(wait_for + SAFETY_EPS, 0.05))

    def wait(self) -> None:
        """Backward-compatible alias for ``acquire``."""

        self.acquire()


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
    last_contiguous_period: Optional[str] = None
    had_failure: bool = False
    failed_periods: List[str] = field(default_factory=list)
    truncated_periods: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DownloadAccumulator:
    frames: List[pd.DataFrame] = field(default_factory=list)
    updates: List[Tuple[str, str, str]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def merge(self, other: "DownloadAccumulator") -> None:
        self.frames.extend(other.frames)
        self.updates.extend(other.updates)
        self.failures.extend(other.failures)

    def total_rows(self) -> int:
        total = 0
        for frame in self.frames:
            if frame is not None:
                total += len(frame)
        return total

    def pop_frames(self) -> List[pd.DataFrame]:
        frames = self.frames
        self.frames = []
        return frames


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
        vip_pro: Any | None = None,
        use_vip: bool = True,
        max_per_minute: int = 90,
        state_path: Optional[str] = None,
        state_backend: StateBackend | None = None,
        allow_future: bool = False,
        max_retries: int = 3,
        flush_threshold_rows: int = FRAME_FLUSH_THRESHOLD_ROWS,
        progress_mode: str = "auto",
    ) -> None:
        self.pro = pro
        self.vip_pro = vip_pro
        self.data_dir = data_dir
        self.use_vip = use_vip
        self._max_per_minute = max(int(max_per_minute), 0)
        self._client_limiters: Dict[int, RateLimiter] = {}
        self._warned_vip_fallback = False
        self._warned_dividend_range_failure = False
        self._ensure_limiter(self.pro)
        if self.vip_pro is not None:
            self._ensure_limiter(self.vip_pro)
        self.allow_future = allow_future
        self.retry_policy = RetryPolicy(max_retries=max_retries)
        self.flush_threshold_rows = max(int(flush_threshold_rows), 0)
        self._field_cache: Dict[Tuple[str, str], Set[str]] = {}
        self._field_missing_logged: Set[Tuple[str, str]] = set()
        self._field_extra_logged: Set[Tuple[str, str]] = set()
        state_file = (
            Path(state_path) if state_path else Path(data_dir) / "_state" / "state.json"
        )
        self.state = state_backend or JsonStateBackend(state_file)
        self.progress = ProgressManager(progress_mode)

    def _log(self, message: str) -> None:
        self.progress.log(message)

    def _start_task(self, description: str, total: int) -> Any:
        return self.progress.add_task(description, total)

    def _combo_task_label(
        self,
        spec: DatasetSpec,
        combo: PeriodCombination,
        *,
        ts_code: Optional[str] = None,
    ) -> str:
        parts = [spec.name]
        combo_desc = combo.describe(spec)
        if combo_desc and combo_desc != "默认组合":
            parts.append(combo_desc)
        if ts_code:
            parts.append(ts_code)
        return " ".join(parts)

    def _ensure_limiter(self, client: Any) -> RateLimiter:
        key = id(client)
        limiter = self._client_limiters.get(key)
        if limiter is not None:
            return limiter
        if getattr(client, "__is_token_pool__", False):
            limiter = RateLimiter(max_per_minute=0)
            set_rate = getattr(client, "set_rate", None)
            if callable(set_rate):
                try:
                    set_rate(self._max_per_minute or 90)
                except Exception:
                    pass
        else:
            limiter = RateLimiter(max_per_minute=self._max_per_minute)
        self._client_limiters[key] = limiter
        return limiter

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
        with self.progress.live():
            dataset_task = self._start_task(
                f"数据集任务（共 {len(requests)} 个）", len(requests)
            )
            for req in requests:
                spec = self._spec_for(req.name)
                self._log(f"[{datetime.now()}] >>> 抓取 {spec.name}")
                if spec.requires_ts_code:
                    self._log(f"提示：{spec.name} 仅支持按股票循环，本次将枚举 ts_code")
                self._download_dataset(
                    spec,
                    req.options,
                    start_date,
                    end_date,
                    refresh_periods,
                )
                self._log(f"[{datetime.now()}] <<< 完成 {spec.name}")
                self.progress.advance(dataset_task, 1)

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

    def _should_flush(self, accumulator: DownloadAccumulator) -> bool:
        if self.flush_threshold_rows <= 0:
            return False
        return accumulator.total_rows() >= self.flush_threshold_rows

    def _flush_accumulator(
        self, accumulator: DownloadAccumulator, spec: DatasetSpec
    ) -> bool:
        if not accumulator.frames:
            return True
        frames = accumulator.pop_frames()
        combined = merge_and_deduplicate(
            frames,
            group_keys=spec.dedup_group_keys or spec.primary_keys,
        )
        if combined is None:
            return True
        return write_parquet_dataset(
            combined,
            self.data_dir,
            spec.name,
            spec.default_year_column,
            group_keys=spec.dedup_group_keys or spec.primary_keys,
        )

    # Backwards-compatible wrappers kept for tests and external callers. These
    # delegate to the shared storage helpers.
    def _record_failures(
        self,
        spec: DatasetSpec,
        entries: Sequence[Dict[str, Any]],
        kind: str,
    ) -> None:
        write_failure_report(self.data_dir, spec.name, kind, entries)

    def _concat_and_dedup(
        self,
        frames: Sequence[pd.DataFrame],
        spec: DatasetSpec,
    ) -> Optional[pd.DataFrame]:
        return merge_and_deduplicate(
            frames,
            group_keys=spec.dedup_group_keys or spec.primary_keys,
        )

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
        client, method_name, paginate = self._resolve_method(spec)
        period_end = self._bounded_period_end(end_date)

        accumulator = DownloadAccumulator()
        write_ok = True
        for combo in combinations:
            combo_result = self._run_periodic_combo(
                spec,
                combo,
                start_date,
                period_end,
                refresh_periods,
                client,
                method_name,
                paginate,
            )
            accumulator.merge(combo_result)
            if self._should_flush(accumulator):
                write_ok = write_ok and self._flush_accumulator(accumulator, spec)

        write_ok = write_ok and self._flush_accumulator(accumulator, spec)
        failure_entries = accumulator.failures
        write_failure_report(self.data_dir, spec.name, "periods", failure_entries)
        if write_ok:
            for dataset, key, value in accumulator.updates:
                self.state.set(dataset, key, value)
            self._print_failure_summary(spec, failure_entries)

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
            self._log(f"警告：{spec.name} 未找到可用股票清单，已跳过")
            return
        report_types = self._resolve_report_types(spec, options)
        type_values = self._resolve_type_values(spec, options)
        combinations = self._build_period_combinations(report_types, type_values)
        client, method_name, paginate = self._resolve_method(spec)
        period_end = self._bounded_period_end(end_date)
        accumulator = DownloadAccumulator()
        write_ok = True
        stock_task = self._start_task(
            f"{spec.name} 股票循环", len(stock_df.index)
        )
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
                client,
                method_name,
                paginate,
            )
            accumulator.merge(stock_result)
            if self._should_flush(accumulator):
                write_ok = write_ok and self._flush_accumulator(accumulator, spec)
            self.progress.advance(stock_task, 1)

        write_ok = write_ok and self._flush_accumulator(accumulator, spec)
        failure_entries = accumulator.failures
        write_failure_report(self.data_dir, spec.name, "per_stock", failure_entries)
        if write_ok:
            for dataset, key, value in accumulator.updates:
                self.state.set(dataset, key, value)
            self._print_failure_summary(spec, failure_entries)

    def _run_periodic_combo(
        self,
        spec: DatasetSpec,
        combo: PeriodCombination,
        start_date: Optional[str],
        period_end: str,
        refresh_periods: int,
        client: Any,
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
        combo_task = self._start_task(
            f"{self._combo_task_label(spec, combo)}", len(periods)
        )
        outcome = self._collect_periods(
            spec,
            periods,
            combo,
            client,
            method_name,
            paginate,
            progress_task=combo_task,
        )
        if outcome.frames:
            result.frames.extend(outcome.frames)
        if outcome.last_contiguous_period is not None:
            result.updates.append(
                (spec.name, state_key, outcome.last_contiguous_period)
            )
        if outcome.failed_periods or outcome.truncated_periods:
            params = dict(spec.extra_params)
            params.update(combo.as_params(spec))
            failure_record: Dict[str, Any] = {
                "combo": combo.describe(spec),
                "params": self._summarize_params(params),
            }
            if outcome.failed_periods:
                failure_record["failed_periods"] = list(outcome.failed_periods)
            if outcome.truncated_periods:
                failure_record["truncated"] = list(outcome.truncated_periods)
            result.failures.append(failure_record)
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
        client: Any,
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
                client,
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
        client: Any,
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
        truncated: List[Dict[str, Any]] = []
        failure_seen = False
        last_contiguous: Optional[str] = None
        combo_task = self._start_task(
            f"{self._combo_task_label(spec, combo, ts_code=ts_code)}",
            len(periods),
        )
        success_count = 0
        fail_count = 0
        for period_value in periods:
            frame, success = self._call_stock_period(
                spec,
                client,
                method_name,
                paginate,
                combo,
                ts_code,
                period_value,
            )
            if not success:
                failure_seen = True
                failed_periods.append(period_value)
                self._log(
                    f"警告：{spec.name} {combo_desc} 针对 {ts_code} 在 {period_value} 抓取失败，请稍后手动排查"
                )
                fail_count += 1
                self.progress.advance(
                    combo_task, 1, ok=success_count, fail=fail_count
                )
                continue
            success_count += 1
            if not failure_seen:
                last_contiguous = period_value
            if frame is not None:
                info = self._extract_truncation_metadata(
                    frame, period=period_value, ts_code=ts_code
                )
                if info:
                    truncated.append(info)
                if not frame.empty:
                    result.frames.append(frame)
            self.progress.advance(combo_task, 1, ok=success_count, fail=fail_count)
        if last_contiguous is not None:
            result.updates.append((spec.name, state_key, last_contiguous))
        if failed_periods or truncated:
            params = dict(spec.extra_params)
            params.update(combo.as_params(spec))
            failure_record: Dict[str, Any] = {
                "ts_code": ts_code,
                "combo": combo_desc,
                "params": self._summarize_params(params),
            }
            if failed_periods:
                failure_record["failed_periods"] = failed_periods
            if truncated:
                failure_record["truncated"] = truncated
            result.failures.append(failure_record)
        return result

    def _call_stock_period(
        self,
        spec: DatasetSpec,
        client: Any,
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
            client=client,
            paginate=paginate,
        )
        if df is None:
            return None, False
        if df.empty:
            return df, True
        frame = df.copy()
        frame.attrs = dict(df.attrs)
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
        failure_seen = False
        last_contiguous: Optional[str] = None
        window_records: Dict[str, Dict[str, Any]] = {}
        for win_start, win_end in windows:
            window_id = f"{win_start}-{win_end}"
            df = self._fetch_window(spec, win_start, win_end)
            params = dict(spec.extra_params)
            params[spec.date_start_param] = win_start
            params[spec.date_end_param] = win_end
            params_summary = self._summarize_params(params)
            if df is None:
                failure_seen = True
                window_records[window_id] = {
                    "window": window_id,
                    "status": "failed",
                    "params": params_summary,
                }
                continue
            info = self._extract_truncation_metadata(df, window=window_id)
            if info:
                entry = window_records.setdefault(
                    window_id, {"window": window_id, "params": params_summary}
                )
                entry["truncated"] = True
                if "pagination" in info:
                    entry["pagination"] = info["pagination"]
            if not df.empty:
                collected.append(df)
            if not failure_seen:
                last_contiguous = win_end
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
        failure_entries = [
            entry for entry in window_records.values() if len(entry) > 1
        ]
        write_failure_report(self.data_dir, spec.name, "windows", failure_entries)
        if write_ok and last_contiguous is not None:
            self.state.set(spec.name, state_key, last_contiguous)
            if failure_entries:
                self._print_calendar_failures(spec, failure_entries)

    def _print_failure_summary(
        self, spec: DatasetSpec, failures: Sequence[Dict[str, Any]]
    ) -> None:
        if not failures:
            return
        for entry in failures:
            combo_desc = entry.get("combo", "默认组合")
            ts_code = entry.get("ts_code")
            failed_periods = entry.get("failed_periods") or []
            truncated = entry.get("truncated") or []
            parts: List[str] = [spec.name]
            if ts_code:
                parts.append(str(ts_code))
            if combo_desc:
                parts.append(str(combo_desc))
            prefix = " ".join(parts)
            if failed_periods:
                failed = ", ".join(failed_periods)
                self._log(f"提示：{prefix} 未成功的 period: {failed}")
            for trunc in truncated:
                period = trunc.get("period")
                if period:
                    self._log(f"提示：{prefix} 在 {period} 分页达到上限，详见失败记录")
                else:
                    self._log(f"提示：{prefix} 分页达到上限，详见失败记录")

    def _print_calendar_failures(
        self, spec: DatasetSpec, failures: Sequence[Dict[str, Any]]
    ) -> None:
        for entry in failures:
            window = entry.get("window")
            status = entry.get("status")
            if status == "failed" and window:
                self._log(f"提示：{spec.name} 窗口 {window} 抓取失败")
            if entry.get("truncated") and window:
                self._log(f"提示：{spec.name} 窗口 {window} 分页达到上限，详见失败记录")

    def _extract_truncation_metadata(
        self,
        df: Optional[pd.DataFrame],
        *,
        period: Optional[str] = None,
        ts_code: Optional[str] = None,
        window: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if df is None or not hasattr(df, "attrs"):
            return None
        if not df.attrs.get("page_limit_hit"):
            return None
        metadata: Dict[str, Any] = {"page_limit_hit": True}
        if period is not None:
            metadata["period"] = period
        if ts_code is not None:
            metadata["ts_code"] = ts_code
        if window is not None:
            metadata["window"] = window
        pagination = df.attrs.get("pagination_info")
        if isinstance(pagination, dict):
            metadata["pagination"] = pagination
        else:
            metadata.setdefault("pagination", {})
        return metadata

    def _summarize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in sorted(params.items()):
            if "token" in key.lower():
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, (list, tuple, set)):
                summary[key] = ",".join(sorted(str(v) for v in value))
            else:
                summary[key] = str(value)
        return summary

    def _expected_fields(self, api_name: str, fields: str) -> Set[str]:
        key = (api_name, fields)
        cached = self._field_cache.get(key)
        if cached is not None:
            return cached
        items = [part.strip() for part in re.split(r"[\n,]", fields) if part.strip()]
        normalized = {item for item in items}
        self._field_cache[key] = normalized
        return normalized

    def _validate_fields(
        self, api_name: str, fields: Optional[str], df: pd.DataFrame
    ) -> None:
        if not fields:
            return
        expected = self._expected_fields(api_name, fields)
        actual = {str(col).strip() for col in df.columns}
        missing = sorted(field for field in expected - actual)
        extra = sorted(field for field in actual - expected)
        key = (api_name, fields)
        if missing and key not in self._field_missing_logged:
            self._log(f"警告：调用 {api_name} 返回缺少字段：{', '.join(missing)}")
            self._field_missing_logged.add(key)
        if extra and key not in self._field_extra_logged:
            self._log(f"提示：调用 {api_name} 返回新增字段：{', '.join(extra)}")
            self._field_extra_logged.add(key)

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
                self._log(f"警告：读取 {file} 失败：{exc}")
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
            client=self.pro,
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

    def _resolve_method(self, spec: DatasetSpec) -> Tuple[Any, str, bool]:
        if self.use_vip and spec.vip_api:
            if self.vip_pro is None:
                if not self._warned_vip_fallback:
                    self._log(
                        "警告：未检测到可用的 VIP token，已回落至普通接口，可能触发权限错误"
                    )
                    self._warned_vip_fallback = True
                client = self.pro
                method_name = spec.api
                paginate = spec.api_supports_pagination
            else:
                client = self.vip_pro
                method_name = spec.vip_api
                paginate = spec.vip_supports_pagination
        else:
            client = self.pro
            method_name = spec.api
            paginate = spec.api_supports_pagination
        return client, method_name, paginate

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
        client: Any,
        method_name: str,
        paginate: bool,
        *,
        progress_task: Any = None,
    ) -> PeriodFetchOutcome:
        outcome = PeriodFetchOutcome()
        failure_seen = False
        success_count = 0
        fail_count = 0
        for period_value in periods:
            df, success = self._fetch_period(
                spec,
                client,
                method_name,
                paginate,
                period_value,
                combo,
            )
            if not success:
                outcome.had_failure = True
                failure_seen = True
                outcome.failed_periods.append(period_value)
                combo_desc = combo.describe(spec)
                self._log(
                    f"警告：{spec.name} {combo_desc} 在 {period_value} 抓取失败，请稍后手动排查"
                )
                fail_count += 1
                self.progress.advance(
                    progress_task, 1, ok=success_count, fail=fail_count
                )
                continue
            success_count += 1
            outcome.last_successful_period = period_value
            if not failure_seen:
                outcome.last_contiguous_period = period_value
            if df is not None:
                info = self._extract_truncation_metadata(df, period=period_value)
                if info:
                    outcome.truncated_periods.append(info)
                if not df.empty:
                    outcome.frames.append(df)
            self.progress.advance(
                progress_task, 1, ok=success_count, fail=fail_count
            )
        return outcome

    def _fetch_period(
        self,
        spec: DatasetSpec,
        client: Any,
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
            client=client,
            paginate=paginate,
        )
        if df is None:
            return None, False
        if df.empty:
            return df, True
        frame = df.copy()
        frame.attrs = dict(df.attrs)
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
        mode = getattr(spec, "date_window_mode", "range")
        if mode == "ann_date":
            return self._fetch_dividend_window(spec, start, end)
        return self._fetch_window_range(spec, start, end)

    def _fetch_window_range(
        self, spec: DatasetSpec, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        params = dict(spec.extra_params)
        params[spec.date_start_param] = start
        params[spec.date_end_param] = end
        df = self._call_api(
            spec.api,
            params,
            spec.fields,
            client=self.pro,
            paginate=spec.api_supports_pagination,
        )
        return df

    def _fetch_dividend_window(
        self, spec: DatasetSpec, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        params = dict(spec.extra_params)
        params[spec.date_start_param] = start
        params[spec.date_end_param] = end
        params.setdefault("ann_date", start)
        if "where" not in params:
            params["where"] = f"ann_date>='{start}' and ann_date<='{end}'"
        df = self._call_api(
            spec.api,
            params,
            spec.fields,
            client=self.pro,
            paginate=spec.api_supports_pagination,
        )
        if df is not None and not df.empty:
            return df

        if not getattr(self, "_warned_dividend_range_failure", False):
            self._log(
                "提示：dividend 接口不支持按区间抓取 ann_date，将改用逐日抓取（耗时较长）。"
            )
            self._warned_dividend_range_failure = True

        frames: list[pd.DataFrame] = []
        covered_dates: set[str] = set()
        if df is not None and not df.empty:
            frames.append(df)
            covered_dates = {
                value
                for value in (
                    self._normalize_calendar_value(v) for v in df.get("ann_date", [])
                )
                if value
            }
        for day in self._iter_date_range(start, end):
            if day in covered_dates:
                continue
            day_params = dict(spec.extra_params)
            day_params["ann_date"] = day
            day_df = self._call_api(
                spec.api,
                day_params,
                spec.fields,
                client=self.pro,
                paginate=spec.api_supports_pagination,
            )
            if day_df is None or day_df.empty:
                continue
            frames.append(day_df)
            covered_dates.update(
                value
                for value in (
                    self._normalize_calendar_value(v) for v in day_df.get("ann_date", [])
                )
                if value
            )
        if not frames:
            return df
        if len(frames) == 1:
            return frames[0]
        return _concat_non_empty(frames)

    def _iter_date_range(self, start: str, end: str) -> list[str]:
        try:
            start_date = datetime.strptime(start, DATE_FMT).date()
            end_date = datetime.strptime(end, DATE_FMT).date()
        except ValueError:
            return []
        days: list[str] = []
        cur = start_date
        while cur <= end_date:
            days.append(cur.strftime(DATE_FMT))
            cur += timedelta(days=1)
        return days

    def _normalize_calendar_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "nat"}:
            return None
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            text = text.replace("-", "")
        if len(text) != 8 or not text.isdigit():
            return None
        return text

    def _call_api(  # noqa: C901
        self,
        api_name: str,
        params: Dict[str, Any],
        fields: Optional[str],
        *,
        client: Any,
        paginate: bool,
    ) -> Optional[pd.DataFrame]:
        func = getattr(client, api_name, None)
        if func is None:

            def fallback_call(**kwargs: Any) -> pd.DataFrame:
                return client.query(api_name, **kwargs)

            func = fallback_call
        policy = self.retry_policy
        limiter = self._ensure_limiter(client)

        def _log_retry(attempt: int, exc: Exception, wait_seconds: float) -> None:
            self._log(
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
            use_pagination = bool(paginate)
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
                limiter.acquire()
                df = func(**call_params)
                if df is None or df.empty:
                    break
                signature = hashlib.sha1(
                    pd.util.hash_pandas_object(df, index=True).values.tobytes()
                ).digest()
                if signature in seen_signatures:
                    self._log(
                        f"警告：调用 {api_name} 分页出现重复结果（offset={offset}），已提前终止"
                    )
                    break
                seen_signatures.add(signature)
                rows.append(df)
                pages += 1
                if use_pagination and (len(df) < limit or pages >= MAX_PAGES):
                    if pages >= MAX_PAGES:
                        page_limit_hit = True
                    break
                if use_pagination:
                    offset += limit
            if not rows:
                result = pd.DataFrame()
            else:
                result = _concat_non_empty(rows)
            if fields:
                self._validate_fields(api_name, fields, result)
            if page_limit_hit:
                self._log(
                    f"警告：调用 {api_name} 达到分页上限 {MAX_PAGES}，结果可能被截断"
                )
                result.attrs["page_limit_hit"] = True
                result.attrs["pagination_info"] = {
                    "pages": pages,
                    "limit": limit,
                    "max_pages": MAX_PAGES,
                    "params": self._summarize_params(params),
                }
            return result

        try:
            return call_with_retry(
                _invoke,
                policy=policy,
                description=f"调用 {api_name}",
                on_retry=_log_retry,
            )
        except RetryExhaustedError as exc:
            self._log(f"警告：调用 {api_name} 失败：{exc.last_exception}")
            return None

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
