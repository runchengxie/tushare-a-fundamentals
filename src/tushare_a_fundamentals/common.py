import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml

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


def _retry_call(func, kwargs, max_tries=5, base_sleep=0.8):
    for i in range(max_tries):
        try:
            return func(**kwargs)
        except Exception:
            if i == max_tries - 1:
                raise
            time.sleep(base_sleep * (2**i))
    return None


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
    return total >= required


def _concat_non_empty(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames after dropping empty or all-NA ones."""
    non_empty = [df for df in dfs if not df.dropna(how="all").empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _check_parquet_dependency() -> bool:
    """Return True if a parquet engine (pyarrow/fastparquet) is available."""
    for name in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(name)
            return True
        except ModuleNotFoundError:
            continue
    return False


def _kinds_for_mode(mode: str) -> List[str]:
    return ["raw"]


def _already_downloaded(
    outdir: str, base: str, fmt: str, periods: List[str], kinds: List[str]
) -> bool:
    fmt_dir = "csv" if fmt == "csv" else "parquet"
    for kind in kinds:
        path = os.path.join(outdir, fmt_dir, f"{base}_{kind}.{fmt}")
        if not os.path.exists(path):
            return False
        try:
            if fmt == "csv":
                df = pd.read_csv(path, usecols=["end_date"])
            else:
                df = pd.read_parquet(path, columns=["end_date"])
        except Exception:
            return False
        existing = set(df["end_date"].astype(str))
        if not set(periods).issubset(existing):
            return False
    return True


def _select_latest(df: pd.DataFrame) -> pd.DataFrame:
    """后向兼容：委托给 transforms.deduplicate.select_latest。"""
    got = _tx_select_latest(df, group_keys=("ts_code", "end_date"))
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
    pro, periods: List[str], mode: str, fields: str
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple periods via income_vip and return cumulative tables."""
    tables: Dict[str, pd.DataFrame] = {}
    all_rows: List[pd.DataFrame] = []
    for per in periods:
        params = {"period": per}
        try:
            df = _retry_call(pro.income_vip, {"fields": fields, **params})
        except Exception as exc:
            eprint(f"警告：期末 {per} 拉取失败：{exc}")
            continue
        if df is None or len(df) == 0:
            eprint(f"警告：期末 {per} 接口返回为空")
            continue
        all_rows.append(df)
    if not all_rows:
        eprint("错误：未获取到任何数据")
        sys.exit(3)
    raw = _concat_non_empty(all_rows)
    if raw.empty:
        eprint("错误：未获取到任何数据")
        sys.exit(3)
    raw = _select_latest(raw)
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
    export_colname: str = "ticker",
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
            if export_colname == "ticker" and "ts_code" in out_df.columns:
                out_df = out_df.rename(columns={"ts_code": "ticker"})
            if fmt == "csv":
                out_df.to_csv(fpath, index=False)
            else:
                out_df.to_parquet(fpath, index=False)
            print(f"已保存：{fpath}")
        except Exception as exc:
            eprint(f"错误：保存失败 {fpath}：{exc}")
            sys.exit(4)


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
    if cfg.get("since"):
        return _periods_from_range("quarterly", cfg["since"], cfg.get("until"))
    if cfg.get("quarters") and cfg["quarters"] > 0:
        return periods_by_quarters(cfg["quarters"])
    return periods_for_mode_by_years(cfg.get("years", 10), Mode.QUARTERLY)


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
    dfs = [pd.read_parquet(p) for p in files]
    return _concat_non_empty(dfs)


def _export_tables(
    tables: Dict[str, pd.DataFrame],
    out_dir: str,
    prefix: str,
    fmt: str,
    export_colname: str,
) -> None:
    base = f"{prefix}"
    out: Dict[str, pd.DataFrame] = {}
    for k, df in tables.items():
        out[k] = df
    save_tables(out, out_dir, base, fmt, export_colname)


def _run_bulk_mode(
    pro, cfg: dict, fields: str, fmt: str, outdir: str, prefix: str
) -> None:
    if not _has_enough_credits(pro):
        total = _available_credits(pro) or 0
        eprint(f"错误：全市场批量需要至少 5000 积分。（检测到总积分：{int(total)}）")
        sys.exit(2)
    periods = _periods_from_cfg(cfg)
    base = f"{prefix}_vip_quarterly"
    kinds = _kinds_for_mode("quarterly")
    if cfg.get("skip_existing") and _already_downloaded(
        outdir, base, fmt, periods, kinds
    ):
        print("已存在所需数据，跳过下载")
        return
    tables = fetch_income_bulk(pro, periods=periods, mode="quarterly", fields=fields)
    save_tables(tables, outdir, base, fmt, cfg.get("export_colname", "ticker"))
