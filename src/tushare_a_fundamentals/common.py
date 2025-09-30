import importlib
import math
import os
import random
from functools import partial
import sys
import time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypeVar

import pandas as pd
import yaml

from tushare_a_fundamentals.transforms.deduplicate import (
    mark_latest as _tx_mark_latest,
)
from tushare_a_fundamentals.transforms.deduplicate import (
    select_latest as _tx_select_latest,
)

# Do not auto-load .env on import to avoid polluting test environments.
# To load .env locally, use direnv or export variables in the shell.
if os.getenv("TUSHARE_API_KEY") and not os.getenv("TUSHARE_TOKEN"):
    os.environ["TUSHARE_TOKEN"] = os.getenv("TUSHARE_API_KEY")

FLOW_FIELDS = [
    "total_revenue",
    "revenue",
    "total_cogs",
    "operate_profit",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "ebit",
    "ebitda",
    "rd_exp",
]

IDENT_FIELDS = [
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "update_flag",
]

DEFAULT_FIELDS = IDENT_FIELDS + FLOW_FIELDS

PERIOD_NODES = ["0331", "0630", "0930", "1231"]


class Mode:
    ANNUAL = "annual"
    QUARTER = "quarter"
    # legacy aliases
    SEMIANNUAL = "semiannual"
    QUARTERLY = "quarterly"


@dataclass
class Plan:
    periodicity: Literal["annual", "semiannual", "quarterly"]


MODE_MAP = {
    Mode.ANNUAL: Plan("annual"),
    Mode.QUARTERLY: Plan("quarterly"),
    Mode.QUARTER: Plan("quarterly"),
}


# Cache the token used to initialize pro_api so we can reuse it
# for endpoints that may require explicit token passing (e.g., pro.user).
_GLOBAL_TOKEN: Optional[str] = None


def plan_from_mode(mode: str, periodicity: str | None = None) -> Plan:
    m = mode.lower()
    if m == Mode.QUARTER:
        eprint("警告：'quarter' 模式已弃用，请使用 'quarterly'")
    p = MODE_MAP[m]
    return Plan(periodicity or p.periodicity)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_yaml(path: Optional[str]) -> dict:
    candidate = path
    if not candidate:
        # 若未显式传入 --config，则尝试自动加载当前目录下的 config.yml（若存在）
        default_path = os.path.join(os.getcwd(), "config.yml")
        if os.path.exists(default_path):
            candidate = default_path
        else:
            return {}
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            print(f"已加载配置文件：{candidate}")
            return cfg
    except FileNotFoundError:
        eprint(f"错误：未找到配置文件 {candidate}")
        sys.exit(2)
    except Exception as exc:
        eprint(f"错误：读取配置文件失败：{exc}")
        sys.exit(2)


def merge_config(cli: dict, cfg: dict, defaults: dict) -> dict:
    merged = {**defaults, **cfg}
    for k, v in cli.items():
        if v is not None:
            merged[k] = v
    return merged


def parse_report_types(value) -> List[int]:
    """Parse ``report_types`` config into a list of ints.

    Accepts comma-separated strings, single ints, or lists; defaults to ``[1]``
    when unset.
    """
    if value is None:
        return [1]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        return [int(v) for v in value.split(",") if v.strip()]
    return [1]


def init_pro_api(token: Optional[str]):
    token_env = os.getenv("TUSHARE_TOKEN")
    tok = token or token_env
    if not tok:
        eprint(
            "错误：缺少 TuShare token。请通过环境变量 TUSHARE_TOKEN 或 --token 提供。"
        )
        sys.exit(2)
    try:
        import tushare as ts

        ts.set_token(tok)
        pro = ts.pro_api()
        global _GLOBAL_TOKEN
        _GLOBAL_TOKEN = tok
        return pro
    except Exception as exc:
        eprint(f"错误：初始化 TuShare 失败：{exc}")
        sys.exit(2)


def periods_for_mode_by_years(years: int, mode: str) -> List[str]:
    if years <= 0:
        return []
    from datetime import date

    cur_year = date.today().year
    nodes = (
        ["1231"]
        if mode == Mode.ANNUAL
        else (["0630", "1231"] if mode == Mode.SEMIANNUAL else PERIOD_NODES)
    )
    out: List[str] = []
    for y in range(cur_year - years + 1, cur_year + 1):
        for n in nodes:
            out.append(f"{y}{n}")
    return out


def periods_by_quarters(quarters: int) -> List[str]:
    if quarters <= 0:
        return []
    from datetime import date

    y, m = date.today().year, date.today().month
    if m <= 3:
        node = "1231"
        y -= 1
    elif m <= 6:
        node = "0331"
    elif m <= 9:
        node = "0630"
    else:
        node = "0930"
    order = ["0331", "0630", "0930", "1231"]
    idx = order.index(node)
    res: List[str] = []
    cy = y
    ci = idx
    for _ in range(quarters):
        res.append(f"{cy}{order[ci]}")
        ci -= 1
        if ci < 0:
            ci = 3
            cy -= 1
    return sorted(res)


def _backfill_periods(anchor: str, count: int) -> List[str]:
    """Return ``count`` quarterly periods ending at ``anchor`` (inclusive)."""

    if count <= 0:
        return []
    order = PERIOD_NODES
    if anchor[4:] not in order:
        raise ValueError(f"unsupported anchor period: {anchor}")
    year = int(anchor[:4])
    idx = order.index(anchor[4:])
    res: List[str] = []
    cy = year
    ci = idx
    for _ in range(count):
        res.append(f"{cy}{order[ci]}")
        ci -= 1
        if ci < 0:
            ci = len(order) - 1
            cy -= 1
    return sorted(res)


def last_publishable_period(today: date) -> str:
    """Return the last period expected to be publishable at ``today``."""
    y = today.year
    checkpoints = [
        (date(y, 4, 30), f"{y}0331"),
        (date(y, 8, 31), f"{y}0630"),
        (date(y, 10, 31), f"{y}0930"),
        (date(y + 1, 4, 30), f"{y}1231"),
    ]
    last = f"{y - 1}1231"
    for ddl, per in checkpoints:
        if today >= ddl:
            last = per
    return last


T = TypeVar("T")

_FATAL_EXCEPTION_TYPES: Tuple[type, ...] = (
    AttributeError,
    ValueError,
    TypeError,
)

_FATAL_MESSAGE_KEYWORDS: Tuple[str, ...] = (
    "permission",
    "权限",
    "无权限",
    "token",
    "denied",
    "unauthorized",
    "未授权",
    "认证失败",
    "参数错误",
    "parameter error",
    "invalid parameter",
    "invalid token",
    "missing token",
    "missing parameter",
)


class RetryExhaustedError(RuntimeError):
    """Raised when a retryable operation keeps failing beyond the limit."""

    def __init__(self, description: str, last_exception: Exception) -> None:
        super().__init__(description)
        self.last_exception = last_exception


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            self.max_retries = 0
        if self.base_delay <= 0:
            self.base_delay = 0.1
        if self.max_delay < self.base_delay:
            self.max_delay = self.base_delay
        if self.jitter < 0:
            self.jitter = 0.0

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, _FATAL_EXCEPTION_TYPES):
            return False
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                code = int(status_code)
            except (TypeError, ValueError):
                code = None
            if code is not None:
                if code >= 500 or code == 429:
                    return True
                if 400 <= code < 500:
                    return False
        message = str(exc)
        if not message:
            return True
        lowered = message.lower()
        for keyword in _FATAL_MESSAGE_KEYWORDS:
            if keyword in lowered:
                return False
        return True

    def next_delay(self, retry_index: int) -> float:
        base = min(self.base_delay * (2 ** retry_index), self.max_delay)
        if self.jitter <= 0:
            return base
        low = base * max(0.0, 1 - self.jitter)
        high = base * (1 + self.jitter)
        if low == high:
            return low
        return random.uniform(low, high)


def call_with_retry(
    func: Callable[[], T],
    *,
    policy: Optional[RetryPolicy] = None,
    description: Optional[str] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``func`` with retry and exponential backoff."""

    used_policy = policy or RetryPolicy()
    retries = 0
    while True:
        try:
            return func()
        except Exception as exc:  # pragma: no cover - error classification exercised in tests
            if not used_policy.is_retryable(exc):
                raise
            if retries >= used_policy.max_retries:
                raise RetryExhaustedError(description or str(exc), exc) from exc
            wait_seconds = used_policy.next_delay(retries)
            retries += 1
            if on_retry is not None:
                on_retry(retries, exc, wait_seconds)
            else:
                label = description or "调用"
                eprint(
                    "警告："
                    f"{label} 异常，{wait_seconds:.1f}s 后重试（第 {retries}/{used_policy.max_retries} 次）：{exc}"
                )
            sleep_func(wait_seconds)


def _available_credits(pro) -> float | None:  # noqa: C901
    """Return detected total credits (sum of expiring credits) or None if unknown.

    Be tolerant to schema differences: prefer Chinese column "到期积分",
    but also try any column that looks like points (e.g., contains
    "积分" or "point").
    """
    df = None
    try:
        df = pro.user()
    except Exception:
        df = None
    # Some environments require passing token explicitly to user()
    if df is None or hasattr(df, "empty") and df.empty:
        try:
            import tushare as ts

            tok = (
                _GLOBAL_TOKEN
                or os.getenv("TUSHARE_TOKEN")
                or os.getenv("TUSHARE_API_KEY")
            )
            if tok:
                pro2 = ts.pro_api(token=tok)
                df = pro2.user(token=tok)
        except Exception:
            df = None
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    target_cols: list[str] = []
    if "到期积分" in cols:
        target_cols = ["到期积分"]
    else:
        # Fallback: any column name containing 积分 or point (case-insensitive)
        target_cols = [
            c for c in cols if ("积分" in str(c)) or ("point" in str(c).lower())
        ]
    if not target_cols:
        return None
    total = 0.0
    for c in target_cols:
        try:
            s = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            if s.notna().any():
                total += float(s.sum(skipna=True))
        except Exception:
            continue
    return total if total > 0 else None


def _has_enough_credits(pro, required: int = 5000) -> bool:
    """Return True if total credits meet the threshold."""
    total = _available_credits(pro)
    if total is None:
        return False
    try:
        total_d = Decimal(str(total))
        required_d = Decimal(str(required))
    except (InvalidOperation, ValueError):
        total_f = float(total)
        required_f = float(required)
        if total_f >= required_f:
            return True
        return math.isclose(total_f, required_f, rel_tol=1e-6, abs_tol=1e-3)

    if total_d >= required_d:
        return True
    # Allow small rounding errors from the TuShare API (values like 4999.999).
    return 0 <= (required_d - total_d) <= Decimal("0.001")


def _concat_non_empty(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames after dropping empty or all-NA ones."""
    kept: List[pd.DataFrame] = []
    seen_order: List[str] = []
    seen_set: Set[str] = set()
    for df in dfs:
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        for col in df.columns:
            if col not in seen_set:
                seen_set.add(col)
                seen_order.append(col)
        if df.empty:
            continue
        if df.isna().to_numpy().all():
            continue
        kept.append(df)
    if not kept:
        if seen_order:
            return pd.DataFrame(columns=seen_order)
        return pd.DataFrame()
    combined = pd.concat(kept, ignore_index=True)
    if seen_order:
        combined = combined.reindex(columns=seen_order)
    return combined


def _check_parquet_dependency() -> bool:
    """Return True if a parquet engine (pyarrow/fastparquet) is available."""
    for name in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(name)
            return True
        except ModuleNotFoundError:
            continue
    return False


def _select_latest(
    df: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
    extra_sort_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """后向兼容：委托给 transforms.deduplicate.select_latest。"""
    gkeys = tuple(group_keys or ("ts_code", "end_date"))
    got = _tx_select_latest(df, group_keys=gkeys, extra_sort_keys=extra_sort_keys)
    if not got.empty:
        keep_cols = list(
            dict.fromkeys(
                [
                    *(
                        DEFAULT_FIELDS
                        if not set(DEFAULT_FIELDS).issubset(got.columns)
                        else got.columns.tolist()
                    )
                ]
            )
        )
        if set(keep_cols).issubset(got.columns):
            got = got[keep_cols]
    return got


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_ts_code(df: pd.DataFrame, *, context: str | None = None) -> pd.DataFrame:
    """Ensure the dataframe exposes ``ts_code`` as the security identifier."""

    if "ts_code" in df.columns:
        return df
    if "ticker" in df.columns:
        renamed = df.rename(columns={"ticker": "ts_code"})
        return renamed
    ctx = f"（{context}）" if context else ""
    raise KeyError(f"数据缺少 ts_code 列{ctx}")


def _diff_to_single(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _coerce_numeric(df, FLOW_FIELDS)
    df["year"] = df["end_date"].str.slice(0, 4)
    df["node"] = df["end_date"].str.slice(4, 8)
    df = df.sort_values(["ts_code", "year", "node"])  # ascending
    out = df.copy()
    for col in FLOW_FIELDS:
        if col in df.columns:
            out[col] = df.groupby(["ts_code", "year"], as_index=False)[col].diff()
            q1_mask = out["node"] == "0331"
            out.loc[q1_mask, col] = out.loc[q1_mask, col].fillna(df.loc[q1_mask, col])
    out = out.drop(columns=["year", "node"])
    return out


def _single_to_cumulative(single_df: pd.DataFrame) -> pd.DataFrame:
    if single_df.empty:
        return single_df
    df = single_df.copy()
    df = _coerce_numeric(df, FLOW_FIELDS)
    df["year"] = df["end_date"].str.slice(0, 4)
    df["node"] = df["end_date"].str.slice(4, 8)
    df = df.sort_values(["ts_code", "year", "node"])  # ascending
    for col in FLOW_FIELDS:
        if col in df.columns:
            df[col] = df.groupby(["ts_code", "year"], as_index=False)[col].cumsum()
    return df.drop(columns=["year", "node"])


def fetch_income_bulk(
    pro,
    periods: List[str],
    mode: str,
    fields: str,
    report_types: List[int] | None = None,
    period_report_pairs: Optional[Set[Tuple[str, int]]] = None,
    missing_detail: Optional[Dict[Tuple[str, int], Set[str]]] = None,
    refresh_pairs: Optional[Set[Tuple[str, int]]] = None,
    retry_policy: Optional[RetryPolicy] = None,
    initial_load: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple periods via ``income_vip`` for given report types."""
    tables: Dict[str, pd.DataFrame] = {}
    all_rows: List[pd.DataFrame] = []
    rts = [int(v) for v in (report_types or [1])]
    allowed_pairs: Optional[Set[Tuple[str, int]]] = None
    if period_report_pairs:
        allowed_pairs = {(str(p), int(rt)) for p, rt in period_report_pairs}
    refresh_lookup: Set[Tuple[str, int]] = set()
    if refresh_pairs:
        refresh_lookup = {(str(p), int(rt)) for p, rt in refresh_pairs}
    detail_lookup: Dict[Tuple[str, int], Set[str]] = {}
    if missing_detail:
        detail_lookup = {
            (str(p), int(rt)): codes for (p, rt), codes in missing_detail.items()
        }
    future_limit = last_publishable_period(date.today())
    policy = retry_policy or RetryPolicy()
    for per in periods:
        for rt in rts:
            if allowed_pairs is not None and (per, int(rt)) not in allowed_pairs:
                continue
            params = {"period": per, "report_type": rt}
            try:
                call = partial(pro.income_vip, fields=fields, **params)

                def _log_retry(attempt: int, exc: Exception, wait_seconds: float) -> None:
                    eprint(
                        "警告："
                        f"income_vip 期末 {per} report_type {rt} 异常，"
                        f"{wait_seconds:.1f}s 后重试（第 {attempt}/{policy.max_retries} 次）：{exc}"
                    )

                df = call_with_retry(
                    call,
                    policy=policy,
                    description=f"income_vip(period={per}, report_type={rt})",
                    on_retry=_log_retry,
                )
            except RetryExhaustedError as exc:
                last_exc = exc.last_exception
                eprint(
                    "警告："
                    f"期末 {per} report_type {rt} 多次重试仍失败：{last_exc}"
                )
                continue
            except Exception as exc:
                eprint(f"警告：期末 {per} report_type {rt} 拉取失败：{exc}")
                continue
            if df is None or len(df) == 0:
                pair = (per, int(rt))
                codes_missing = detail_lookup.get(pair)
                if per > future_limit:
                    reason = "未来期间未披露"
                elif codes_missing and len(codes_missing) > 0:
                    reason = (
                        f"报告口径缺失，涉及 {len(codes_missing)} 个 ts_code"
                    )
                elif pair in refresh_lookup:
                    reason = "滚动刷新：暂无新增"
                elif codes_missing is not None:
                    reason = "历史无返回（可能上市前或接口未开放）"
                elif initial_load:
                    reason = "初次下载暂未返回（可能上市前）"
                else:
                    reason = "接口返回为空"
                eprint(
                    f"警告：期末 {per} report_type {rt} 无返回：{reason}"
                )
                continue
            df = df.copy()
            df["retrieved_at"] = pd.Timestamp.utcnow()
            all_rows.append(df)
    if not all_rows:
        eprint("错误：未获取到任何数据")
        sys.exit(3)
    raw = _concat_non_empty(all_rows)
    if raw.empty:
        eprint("错误：未获取到任何数据")
        sys.exit(3)
    raw = _select_latest(
        raw,
        group_keys=("ts_code", "end_date", "report_type"),
        extra_sort_keys=("retrieved_at",),
    )
    raw = _coerce_numeric(raw, FLOW_FIELDS)
    tables["raw"] = raw
    return tables


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tables(
    tables: Dict[str, pd.DataFrame],
    outdir: str,
    base: str,
    fmt: str,
) -> None:
    _ensure_outdir(outdir)
    reports_dir = os.path.join(outdir, "reports")
    csv_dir = os.path.join(outdir, "csv")
    parquet_dir = os.path.join(outdir, "parquet")
    _ensure_outdir(reports_dir)
    _ensure_outdir(csv_dir)
    _ensure_outdir(parquet_dir)
    for kind, df in tables.items():
        fname = f"{base}_{kind}.{fmt}"
        if fmt == "csv":
            fpath = os.path.join(csv_dir, fname)
        else:
            fpath = os.path.join(parquet_dir, fname)
        try:
            if os.path.exists(fpath):
                print(f"已存在（覆盖）：{fpath}")
            out_df = df.copy()
            if fmt == "csv":
                out_df.to_csv(fpath, index=False)
            else:
                out_df.to_parquet(fpath, index=False)
            print(f"已保存：{fpath}")
        except Exception as exc:
            eprint(f"错误：保存失败 {fpath}：{exc}")
            sys.exit(4)


def _load_existing_raw(outdir: str, base: str, fmt: str) -> pd.DataFrame:
    fmt_dir = "csv" if fmt == "csv" else "parquet"
    raw_path = os.path.join(outdir, fmt_dir, f"{base}_raw.{fmt}")
    if not os.path.exists(raw_path):
        return pd.DataFrame()
    try:
        if fmt == "csv":
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_parquet(raw_path)
    except Exception as exc:
        eprint(f"警告：读取 {raw_path} 失败：{exc}，视为无历史数据")
        return pd.DataFrame()
    df = ensure_ts_code(df, context=raw_path)
    if "retrieved_at" in df.columns:
        df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], errors="coerce")
    else:
        df["retrieved_at"] = pd.NaT
    df["end_date"] = df["end_date"].astype(str)
    if "report_type" in df.columns:
        df["report_type"] = pd.to_numeric(df["report_type"], errors="coerce").astype(
            "Int64"
        )
    return df


def _plan_period_report_pairs(
    existing_raw: pd.DataFrame,
    periods: List[str],
    report_types: List[int],
    recent_quarters: int,
) -> Tuple[Set[Tuple[str, int]], Set[Tuple[str, int]], Dict[Tuple[str, int], Set[str]]]:
    sorted_periods = sorted({str(p) for p in periods})
    if not sorted_periods:
        return set(), set(), {}
    rts = [int(v) for v in (report_types or [1])]
    planned: Set[Tuple[str, int]] = set()
    missing_pairs: Set[Tuple[str, int]] = set()
    missing_detail: Dict[Tuple[str, int], Set[str]] = {}
    if existing_raw is None or existing_raw.empty:
        for per in sorted_periods:
            for rt in rts:
                planned.add((per, rt))
                missing_pairs.add((per, rt))
                missing_detail.setdefault((per, rt), set())
        return planned, missing_pairs, missing_detail

    df = existing_raw.copy()
    df = ensure_ts_code(df)
    df["end_date"] = df["end_date"].astype(str)
    if "report_type" not in df.columns:
        df["report_type"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["report_type"] = pd.to_numeric(df["report_type"], errors="coerce").astype(
            "Int64"
        )
    codes = sorted(df["ts_code"].dropna().astype(str).unique())
    earliest_map = (
        df.groupby("ts_code")["end_date"].min().dropna().astype(str).to_dict()
    )
    if not codes:
        for per in sorted_periods:
            for rt in rts:
                planned.add((per, rt))
                missing_pairs.add((per, rt))
                missing_detail.setdefault((per, rt), set())
    else:
        existing_clean = df.dropna(subset=["report_type"])
        existing_keys = {
            (code, end, int(rt))
            for code, end, rt in zip(
                existing_clean["ts_code"].astype(str),
                existing_clean["end_date"].astype(str),
                existing_clean["report_type"].astype(int),
            )
        }
        target_keys: Set[Tuple[str, str, int]] = set()
        for code in codes:
            first_period = earliest_map.get(code)
            for per in sorted_periods:
                if first_period and per < first_period:
                    continue
                for rt in rts:
                    target_keys.add((code, per, rt))
        missing_keys = target_keys - existing_keys
        for _code, per, rt in missing_keys:
            planned.add((per, rt))
            missing_pairs.add((per, rt))
            missing_detail.setdefault((per, rt), set()).add(_code)

    if recent_quarters and recent_quarters > 0:
        recent = sorted_periods[-recent_quarters:]
        for per in recent:
            for rt in rts:
                planned.add((per, rt))
    return planned, missing_pairs, missing_detail


def _merge_raw_tables(
    existing_raw: pd.DataFrame, new_raw: pd.DataFrame
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if existing_raw is not None and not existing_raw.empty:
        frames.append(existing_raw)
    if new_raw is not None and not new_raw.empty:
        frames.append(new_raw)
    if not frames:
        return pd.DataFrame()
    combined = _concat_non_empty(frames)
    if combined.empty:
        return combined
    combined = ensure_ts_code(combined)
    combined["end_date"] = combined["end_date"].astype(str)
    if "retrieved_at" in combined.columns:
        combined["retrieved_at"] = pd.to_datetime(
            combined["retrieved_at"], errors="coerce"
        )
    else:
        combined["retrieved_at"] = pd.NaT
    if "report_type" in combined.columns:
        combined["report_type"] = pd.to_numeric(
            combined["report_type"], errors="coerce"
        ).astype("Int64")
    merged = _select_latest(
        combined,
        group_keys=("ts_code", "end_date", "report_type"),
        extra_sort_keys=("retrieved_at",),
    )
    return merged.reset_index(drop=True)


def _periods_from_range(periods: str, since: str, until: Optional[str]) -> List[str]:
    from datetime import date, datetime

    def to_date(ymd: str) -> date:
        return datetime.strptime(ymd, "%Y-%m-%d").date()

    since_d = to_date(since)
    until_d = to_date(until) if until else date.today()
    if since_d > until_d:
        since_d, until_d = until_d, since_d
    nodes = {
        "annual": ["1231"],
        "semiannual": ["0630", "1231"],
        "quarterly": PERIOD_NODES,
    }[periods]
    res: list[str] = []
    for y in range(since_d.year, until_d.year + 1):
        for n in nodes:
            md = f"{n[:2]}-{n[2:]}"
            d = date.fromisoformat(f"{y}-{md}")
            if since_d <= d <= until_d:
                res.append(f"{y}{n}")
    return sorted(res)


def _periods_from_cfg(cfg: dict) -> List[str]:
    """根据 cfg 计算 periods 列表。

    优先级：since/until > quarters > years（默认10年）。
    - 若提供 since（可选 until），按季度粒度计算覆盖的 period 列表。
    - 否则若提供 quarters，按季度数量回溯。
    - 否则按 years 与 mode 计算（years 默认为 10）。
    """
    allow_future = bool(cfg.get("allow_future"))
    limit = None
    if not allow_future:
        limit = last_publishable_period(date.today())

    if cfg.get("since"):
        periods = _periods_from_range("quarterly", cfg["since"], cfg.get("until"))
    elif cfg.get("quarters") and cfg["quarters"] > 0:
        quarters = cfg["quarters"]
        if allow_future:
            periods = periods_by_quarters(quarters)
        else:
            periods = _backfill_periods(limit, quarters)
    else:
        request_years = int(cfg.get("years", 10))
        quarters = request_years * 4
        if allow_future:
            periods = periods_by_quarters(quarters)
        else:
            periods = _backfill_periods(limit, quarters)
    if not allow_future:
        periods = [p for p in periods if p <= limit]
    return periods


def _load_dataset(root: str, dataset: str) -> pd.DataFrame:
    base = os.path.join(root, f"dataset={dataset}")
    if not os.path.exists(base):
        eprint(f"错误：未找到数据集目录：{base}")
        sys.exit(2)
    files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith(".parquet"):
                files.append(os.path.join(dirpath, fn))
    if not files:
        eprint(f"错误：数据集为空：{base}")
        sys.exit(2)
    dfs: list[pd.DataFrame] = []
    for p in files:
        df = pd.read_parquet(p)
        dfs.append(ensure_ts_code(df, context=p))
    combined = _concat_non_empty(dfs)
    if combined.empty:
        return combined
    return ensure_ts_code(combined, context=f"dataset={dataset}")


def _load_raw_snapshot(
    outdir: str, prefix: str, raw_format: str = "parquet"
) -> tuple[pd.DataFrame | None, str | None]:
    fmt_preferences: List[str] = []
    if raw_format:
        fmt_preferences.append(raw_format)
    for candidate in ("parquet", "csv"):
        if candidate not in fmt_preferences:
            fmt_preferences.append(candidate)
    for fmt in fmt_preferences:
        fmt_dir = "csv" if fmt == "csv" else "parquet"
        candidate = os.path.join(outdir, fmt_dir, f"{prefix}_vip_quarterly_raw.{fmt}")
        if not os.path.exists(candidate):
            continue
        try:
            if fmt == "csv":
                return pd.read_csv(candidate), candidate
            return pd.read_parquet(candidate), candidate
        except Exception as exc:
            eprint(f"警告：读取 {candidate} 失败：{exc}")
    return None, None


def build_datasets_from_raw(outdir: str, prefix: str, raw_format: str = "parquet") -> bool:
    """Build inventory and fact datasets from the cached raw table.

    Returns ``True`` when datasets are successfully materialised; ``False`` when
    the raw snapshot is missing or unreadable.
    """

    raw, raw_path = _load_raw_snapshot(outdir, prefix, raw_format)
    if raw is None:
        eprint(f"警告：未找到 {prefix} 的 raw 数据文件，跳过数仓构建")
        return False
    if raw.empty:
        eprint(f"警告：原始数据为空，跳过数仓构建：{raw_path}")
        return False
    raw = ensure_ts_code(raw, context=raw_path)
    inv_dir = os.path.join(outdir, "dataset=inventory_income")
    os.makedirs(inv_dir, exist_ok=True)
    periods = (
        pd.Series(raw["end_date"].astype(str))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_frame(name="end_date")
    )
    periods.to_parquet(os.path.join(inv_dir, "periods.parquet"), index=False)
    fact_root = os.path.join(outdir, "dataset=fact_income_cum")
    flagged = _tx_mark_latest(raw, group_keys=("ts_code", "end_date"))
    latest = flagged[flagged["is_latest"] == 1].copy()
    latest["year"] = latest["end_date"].astype(str).str[:4]
    for y, dfy in latest.groupby("year"):
        year_dir = os.path.join(fact_root, f"year={y}")
        os.makedirs(year_dir, exist_ok=True)
        dfy.drop(columns=["year"]).to_parquet(
            os.path.join(year_dir, "part.parquet"), index=False
        )
    return True


def _export_tables(
    tables: Dict[str, pd.DataFrame],
    out_dir: str,
    prefix: str,
    fmt: str,
) -> None:
    base = f"{prefix}"
    out: Dict[str, pd.DataFrame] = {}
    for k, df in tables.items():
        out[k] = df
    save_tables(out, out_dir, base, fmt)


def _run_bulk_mode(
    pro, cfg: dict, fields: str, fmt: str, outdir: str, prefix: str
) -> None:
    if not _has_enough_credits(pro):
        total = _available_credits(pro)
        detected = "0" if total is None else repr(total)
        eprint(
            "错误：全市场批量需要至少 5000 积分。"
            f"（检测到总积分：{detected}）"
        )
        sys.exit(2)
    periods = _periods_from_cfg(cfg)
    base = f"{prefix}_vip_quarterly"
    report_types = cfg.get("report_types") or [1]
    recent_quarters = cfg.get("recent_quarters", 8) or 0
    max_retries_cfg = cfg.get("max_retries")
    try:
        max_retries = int(max_retries_cfg)
    except (TypeError, ValueError):
        max_retries = RetryPolicy().max_retries
    if max_retries < 0:
        max_retries = 0
    retry_policy = RetryPolicy(max_retries=max_retries)
    existing_raw = _load_existing_raw(outdir, base, fmt)
    planned_pairs, missing_pairs, missing_detail = _plan_period_report_pairs(
        existing_raw, periods, report_types, recent_quarters
    )
    if not planned_pairs and existing_raw.empty:
        eprint("错误：无下载计划且缺少历史数据，请调整参数后重试")
        sys.exit(2)
    refresh_pairs = planned_pairs - missing_pairs
    fetch_pairs = (
        missing_pairs if cfg.get("skip_existing") else planned_pairs
    )
    if fetch_pairs:
        period_list = sorted({per for per, _ in fetch_pairs})
        print(
            f"缺口组合 {len(missing_pairs)} 个，滚动刷新 {len(refresh_pairs)} 个；"
            f"本次实际抓取 {len(fetch_pairs)} 个 period×report_type 组合"
        )
        tables = fetch_income_bulk(
            pro,
            periods=period_list,
            mode="quarterly",
            fields=fields,
            report_types=report_types,
            period_report_pairs=fetch_pairs,
            missing_detail=missing_detail,
            refresh_pairs=refresh_pairs,
            retry_policy=retry_policy,
            initial_load=existing_raw.empty,
        )
        new_raw = tables.get("raw", pd.DataFrame())
    else:
        print("未发现缺口，且已跳过滚动刷新，本次不调用远程接口")
        new_raw = pd.DataFrame()
    merged_raw = _merge_raw_tables(existing_raw, new_raw)
    if merged_raw.empty:
        eprint("警告：合并后数据为空，未写出文件")
        return
    save_tables({"raw": merged_raw}, outdir, base, fmt)
