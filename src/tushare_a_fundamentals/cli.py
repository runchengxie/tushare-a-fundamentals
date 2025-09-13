import argparse
import os
import sys
import time
import yaml
import pandas as pd
from typing import List, Dict, Optional
import importlib
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
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
    TTM = "ttm"
    # legacy aliases
    SEMIANNUAL = "semiannual"
    QUARTERLY = "quarterly"


@dataclass
class Plan:
    periodicity: Literal["annual", "semiannual", "quarterly"]
    view: Literal["reported", "quarter", "ttm"]


MODE_MAP = {
    Mode.ANNUAL: Plan("annual", "reported"),
    Mode.QUARTER: Plan("quarterly", "quarter"),
    Mode.TTM: Plan("quarterly", "ttm"),
}


def plan_from_mode(
    mode: str, periodicity: str | None = None, view: str | None = None
) -> Plan:
    m = mode.lower()
    if m == Mode.QUARTERLY:
        eprint("警告：'quarterly' 模式已弃用，请使用 'quarter'")
        m = Mode.QUARTER
    p = MODE_MAP[m]
    return Plan(periodicity or p.periodicity, view or p.view)


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
        return pro
    except Exception as exc:
        eprint(f"错误：初始化 TuShare 失败：{exc}")
        sys.exit(2)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="income-downloader", description="批量下载A股利润表"
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--mode", choices=[Mode.ANNUAL, Mode.QUARTER, Mode.TTM, Mode.QUARTERLY]
    )
    p.add_argument("--years", type=int)
    p.add_argument("--quarters", type=int)
    p.add_argument("--ts-code", type=str)
    vip_group = p.add_mutually_exclusive_group()
    vip_group.add_argument("--vip", action="store_true")
    vip_group.add_argument("--no-vip", action="store_true")
    p.add_argument("--prefer-single-quarter", action="store_true")
    p.add_argument("--fields", type=str)
    p.add_argument("--outdir", type=str)
    p.add_argument("--prefix", type=str)
    p.add_argument("--format", choices=["csv", "parquet"])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--token", type=str)
    return p.parse_args()


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
    kinds = ["raw"]
    if mode in (Mode.QUARTER, Mode.TTM):
        kinds.append("single")
    if mode == Mode.TTM:
        kinds.append("ttm")
    return kinds


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
    if df.empty:
        return df
    df = df.copy()
    for col in ["ann_date", "f_ann_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = df["end_date"].astype(str).str.replace("-", "", regex=False)
    sort_cols = []
    if "report_type" in df.columns:
        df["_rt_pref"] = (df["report_type"] == 1).astype(int)
        sort_cols.append("_rt_pref")
    if "update_flag" in df.columns:
        df["_upd"] = (df["update_flag"].astype(str).str.upper() == "Y").astype(int)
        sort_cols.append("_upd")
    sort_cols += ["f_ann_date", "ann_date"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    keep_cols = list(
        dict.fromkeys(
            [
                *(
                    DEFAULT_FIELDS
                    if not set(DEFAULT_FIELDS).issubset(df.columns)
                    else df.columns.tolist()
                )
            ]
        )
    )
    grp_keys = ["ts_code", "end_date"]
    grp_keys = [k for k in grp_keys if k in df.columns]
    dedup = df.groupby(grp_keys, as_index=False).head(1)
    return dedup[keep_cols] if set(keep_cols).issubset(dedup.columns) else dedup


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


def _rolling_ttm(single_df: pd.DataFrame) -> pd.DataFrame:
    if single_df.empty:
        return single_df
    df = single_df.copy()
    df = df.sort_values(["ts_code", "end_date"])  # end_date already sortable
    for col in FLOW_FIELDS:
        if col in df.columns:
            df[col] = df.groupby("ts_code", as_index=False)[col].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )
    return df


def fetch_income_bulk(
    pro, periods: List[str], mode: str, fields: str, prefer_single_quarter: bool
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple periods via income_vip and return processed tables.

    Empty or all-NA responses are skipped before concatenation to avoid
    pandas FutureWarning.
    """
    tables: Dict[str, pd.DataFrame] = {}
    all_rows: List[pd.DataFrame] = []
    for per in periods:
        params = {"period": per}
        if prefer_single_quarter and mode in (Mode.QUARTER, Mode.TTM):
            params["report_type"] = 2
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
    if mode in (Mode.QUARTER, Mode.TTM):
        if prefer_single_quarter:
            single = raw.copy()
        else:
            single = _diff_to_single(raw)
        tables["single"] = single
    if mode == Mode.TTM:
        ttm = (
            _rolling_ttm(tables["single"])
            if "single" in tables
            else _rolling_ttm(_diff_to_single(raw))
        )
        tables["ttm"] = ttm
    return tables


def fetch_single_stock(
    pro,
    ts_code: str,
    years: Optional[int],
    quarters: Optional[int],
    mode: str,
    fields: str,
) -> Dict[str, pd.DataFrame]:
    """Fetch income statements for a single stock.

    Empty or all-NA responses are skipped before concatenation to avoid
    pandas FutureWarning.
    """
    if quarters and quarters > 0:
        periods = periods_by_quarters(quarters)
    else:
        if not years:
            eprint("错误：单票模式需要 --years 或 --quarters 之一")
            sys.exit(2)
        periods = periods_for_mode_by_years(years, mode)
    all_rows: List[pd.DataFrame] = []
    for per in periods:
        params = {"ts_code": ts_code, "period": per}
        try:
            df = _retry_call(pro.income, {"fields": fields, **params})
        except Exception as exc:
            eprint(f"警告：{ts_code} {per} 拉取失败：{exc}")
            continue
        if df is None or len(df) == 0:
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
    tables: Dict[str, pd.DataFrame] = {"raw": raw}
    if mode in (Mode.QUARTER, Mode.TTM):
        single = _diff_to_single(raw)
        tables["single"] = single
    if mode == Mode.TTM:
        ttm = (
            _rolling_ttm(tables["single"])
            if "single" in tables
            else _rolling_ttm(_diff_to_single(raw))
        )
        tables["ttm"] = ttm
    return tables


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tables(
    tables: Dict[str, pd.DataFrame], outdir: str, base: str, fmt: str
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
            if fmt == "csv":
                df.to_csv(fpath, index=False)
            else:
                df.to_parquet(fpath, index=False)
            print(f"已保存：{fpath}")
        except Exception as exc:
            eprint(f"错误：保存失败 {fpath}：{exc}")
            sys.exit(4)


def main():
    args = parse_cli()
    cfg_file = load_yaml(args.config)
    defaults = {
        "mode": Mode.QUARTER,
        "years": 10,
        "quarters": None,
        "ts_code": None,
        "vip": True,
        "prefer_single_quarter": True,
        "fields": ",".join(DEFAULT_FIELDS),
        "outdir": "out",
        "prefix": "income",
        "format": "parquet",
        "skip_existing": False,
        "token": None,
    }
    cli_overrides = {
        "mode": args.mode,
        "years": args.years,
        "quarters": args.quarters,
        "ts_code": (
            args["ts_code"]
            if isinstance(args, dict) and "ts_code" in args
            else args.ts_code
        ),
        "vip": (True if args.vip else (False if args.no_vip else None)),
        "prefer_single_quarter": args.prefer_single_quarter,
        "fields": args.fields,
        "outdir": args.outdir,
        "prefix": args.prefix,
        "format": args.format,
        "skip_existing": args.skip_existing,
        "token": args.token,
    }
    cfg = merge_config(cli_overrides, cfg_file, defaults)

    pro = init_pro_api(cfg.get("token"))

    mode = cfg["mode"]
    fields = cfg["fields"]
    fmt = cfg["format"]
    if fmt == "parquet" and not _check_parquet_dependency():
        eprint("警告：缺少 pyarrow 或 fastparquet，已回退到 csv 格式")
        fmt = "csv"
    outdir = cfg["outdir"]
    prefix = cfg["prefix"]

    if cfg.get("ts_code"):
        ts_code = cfg["ts_code"]
        if cfg.get("quarters") and cfg["quarters"] > 0:
            periods = periods_by_quarters(cfg["quarters"])
        else:
            periods = periods_for_mode_by_years(cfg["years"], mode)
        base = f"{prefix}_{ts_code}_{mode}"
        kinds = _kinds_for_mode(mode)
        if cfg.get("skip_existing") and _already_downloaded(
            outdir, base, fmt, periods, kinds
        ):
            print("已存在所需数据，跳过下载")
            return
        tables = fetch_single_stock(
            pro,
            ts_code=ts_code,
            years=cfg.get("years"),
            quarters=cfg.get("quarters"),
            mode=mode,
            fields=fields,
        )
        save_tables(tables, outdir, base, fmt)
    else:
        if cfg.get("vip") is False:
            eprint("错误：未提供 --ts-code 且未启用 --vip，无法进行全市场批量。")
            sys.exit(2)
        periods = (
            periods_by_quarters(cfg["quarters"])
            if cfg.get("quarters")
            else periods_for_mode_by_years(cfg["years"], mode)
        )
        base = f"{prefix}_vip_{mode}"
        kinds = _kinds_for_mode(mode)
        if cfg.get("skip_existing") and _already_downloaded(
            outdir, base, fmt, periods, kinds
        ):
            print("已存在所需数据，跳过下载")
            return
        tables = fetch_income_bulk(
            pro,
            periods=periods,
            mode=mode,
            fields=fields,
            prefer_single_quarter=cfg.get("prefer_single_quarter", True),
        )
        save_tables(tables, outdir, base, fmt)


if __name__ == "__main__":
    main()
