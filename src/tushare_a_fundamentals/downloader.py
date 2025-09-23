from __future__ import annotations

import calendar
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATE_FMT = "%Y%m%d"


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
    default_year_column: str = "ann_date"
    default_start: str = "20000101"
    fields: Optional[str] = None
    report_types: Sequence[int] = field(default_factory=tuple)
    type_param: Optional[str] = None
    type_values: Sequence[str] = field(default_factory=tuple)
    vip_supports_pagination: bool = False
    api_supports_pagination: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "income": DatasetSpec(
        name="income",
        api="income",
        vip_api="income_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        default_year_column="end_date",
        report_types=(1,),
        fields=None,
    ),
    "balancesheet": DatasetSpec(
        name="balancesheet",
        api="balancesheet",
        vip_api="balancesheet_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        default_year_column="end_date",
        fields=None,
    ),
    "cashflow": DatasetSpec(
        name="cashflow",
        api="cashflow",
        vip_api="cashflow_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        default_year_column="end_date",
        fields=None,
    ),
    "forecast": DatasetSpec(
        name="forecast",
        api="forecast",
        vip_api="forecast_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "type"),
        default_year_column="end_date",
        fields=None,
    ),
    "express": DatasetSpec(
        name="express",
        api="express",
        vip_api="express_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
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
        default_year_column="end_date",
        fields=None,
    ),
    "fina_audit": DatasetSpec(
        name="fina_audit",
        api="fina_audit",
        vip_api=None,
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=None,
        api_supports_pagination=True,
    ),
    "fina_mainbz": DatasetSpec(
        name="fina_mainbz",
        api="fina_mainbz",
        vip_api="fina_mainbz_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "bz_item", "type"),
        default_year_column="end_date",
        type_param="type",
        type_values=("P", "D"),
        fields=None,
    ),
    "disclosure_date": DatasetSpec(
        name="disclosure_date",
        api="disclosure_date",
        vip_api=None,
        period_field=None,
        date_field="ann_date",
        date_start_param="start_date",
        date_end_param="end_date",
        primary_keys=(
            "ts_code",
            "end_date",
            "ann_date",
            "pre_date",
            "actual_date",
        ),
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


def write_parquet_dataset(
    df: pd.DataFrame,
    root: str,
    dataset: str,
    year_col: str,
) -> None:
    if df.empty:
        return
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    if year_col not in frame.columns:
        frame[year_col] = None
    years = frame[year_col].astype(str).str[:4]
    frame["year"] = years
    target_root = Path(root) / dataset
    ensure_dir(target_root.as_posix())
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=target_root.as_posix(),
        partition_cols=["year"],
    )


class RateLimiter:
    def __init__(self, max_per_minute: int = 90) -> None:
        self.max_per_minute = max_per_minute
        self.calls: List[float] = []

    def wait(self) -> None:
        if self.max_per_minute <= 0:
            return
        now = time.time()
        window_start = now - 60
        self.calls = [t for t in self.calls if t >= window_start]
        if len(self.calls) >= self.max_per_minute:
            sleep_for = 60 - (now - self.calls[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.calls.append(time.time())


class JsonState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text("utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def get(self, dataset: str, key: str, default: str) -> str:
        return str(self.data.get(dataset, {}).get(key, default))

    def update(self, dataset: str, key: str, value: str) -> None:
        bucket = self.data.setdefault(dataset, {})
        bucket[key] = value
        ensure_dir(self.path.parent.as_posix())
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            "utf-8",
        )


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
    ) -> None:
        self.pro = pro
        self.data_dir = data_dir
        self.use_vip = use_vip
        self.limiter = RateLimiter(max_per_minute=max_per_minute)
        state_file = (
            Path(state_path)
            if state_path
            else Path(data_dir) / "_state" / "state.json"
        )
        self.state = JsonState(state_file)

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
        state_key = "last_period"
        default_start = spec.default_start
        last_period = self.state.get(spec.name, state_key, default_start)
        effective_start = start_date or last_period
        effective_start = max(effective_start, spec.default_start)
        if refresh_periods and last_period:
            backfill_start = move_quarters(last_period, -max(refresh_periods, 0))
            if backfill_start < effective_start:
                effective_start = backfill_start
        periods = quarter_periods(effective_start, end_date)
        if not periods:
            return
        report_types = self._resolve_report_types(spec, options)
        type_values = self._resolve_type_values(spec, options)
        collected: List[pd.DataFrame] = []
        for period_value in periods:
            frames = self._fetch_period(spec, period_value, report_types, type_values)
            if frames:
                collected.extend(frames)
            self.state.update(spec.name, state_key, period_value)
        combined = self._concat_and_dedup(collected, spec.primary_keys)
        if combined is not None:
            year_col = spec.default_year_column
            write_parquet_dataset(combined, self.data_dir, spec.name, year_col)

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
        effective_start = start_date or last_date
        effective_start = max(effective_start, spec.default_start)
        windows = month_windows(effective_start, end_date)
        if not windows:
            return
        collected: List[pd.DataFrame] = []
        for win_start, win_end in windows:
            df = self._fetch_window(spec, win_start, win_end)
            if df is not None and not df.empty:
                collected.append(df)
            self.state.update(spec.name, state_key, win_end)
        combined = self._concat_and_dedup(collected, spec.primary_keys)
        if combined is not None:
            year_col = spec.default_year_column
            write_parquet_dataset(combined, self.data_dir, spec.name, year_col)

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

    def _fetch_period(
        self,
        spec: DatasetSpec,
        period_value: str,
        report_types: Sequence[int],
        type_values: Sequence[str],
    ) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        vip = spec.vip_api if self.use_vip else None
        method_name = vip or spec.api
        paginate = (
            spec.vip_supports_pagination if vip else spec.api_supports_pagination
        )
        combos: Iterable[Dict[str, Any]] = [{}]
        if report_types:
            combos = [{"report_type": rt} for rt in report_types]
        for combo in combos:
            type_iter: Iterable[Optional[str]]
            if type_values:
                type_iter = list(type_values)
            else:
                type_iter = [None]
            for type_value in type_iter:
                params = dict(spec.extra_params)
                params[spec.period_field] = period_value
                params.update(combo)
                if spec.type_param and type_value is not None:
                    params[spec.type_param] = type_value
                df = self._call_api(
                    method_name,
                    params,
                    spec.fields,
                    paginate=paginate,
                )
                if df is None or df.empty:
                    continue
                if (
                    spec.type_param
                    and spec.type_param not in df.columns
                    and type_value is not None
                ):
                    df = df.copy()
                    df[spec.type_param] = type_value
                frames.append(df)
        return frames

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

    def _call_api(
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
        self.limiter.wait()
        try:
            if not paginate:
                call_params = params.copy()
                if fields:
                    call_params.setdefault("fields", fields)
                return func(**call_params)
            offset = 0
            limit = params.get("limit", 10000)
            rows: List[pd.DataFrame] = []
            while True:
                call_params = params.copy()
                call_params["limit"] = limit
                call_params["offset"] = offset
                if fields:
                    call_params.setdefault("fields", fields)
                df = func(**call_params)
                if df is None or df.empty:
                    break
                rows.append(df)
                if len(df) < limit:
                    break
                offset += limit
            if not rows:
                return None
            return pd.concat(rows, ignore_index=True)
        except Exception as exc:  # pragma: no cover - network failure
            print(f"警告：调用 {api_name} 失败：{exc}")
            return None

    def _concat_and_dedup(
        self, frames: Sequence[pd.DataFrame], primary_keys: Sequence[str]
    ) -> Optional[pd.DataFrame]:
        valid = [df for df in frames if df is not None and not df.empty]
        if not valid:
            return None
        combined = pd.concat(valid, ignore_index=True)
        for col in ("ts_code", "end_date", "ann_date"):
            if col in combined.columns:
                combined[col] = combined[col].astype(str)
        if primary_keys:
            keep = [c for c in primary_keys if c in combined.columns]
            if keep:
                combined = combined.drop_duplicates(subset=keep, keep="last")
            else:
                combined = combined.drop_duplicates()
        else:
            combined = combined.drop_duplicates()
        combined["retrieved_at"] = pd.Timestamp.utcnow()
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
