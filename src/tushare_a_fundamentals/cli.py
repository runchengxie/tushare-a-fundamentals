import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml

from tushare_a_fundamentals.transforms.deduplicate import (
    mark_latest as _tx_mark_latest,
)
from tushare_a_fundamentals.transforms.deduplicate import (
    select_latest as _tx_select_latest,
)
from tushare_a_fundamentals.writers.dataset_writer import write_partitioned_dataset

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


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量下载A股基本面数据")
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=[Mode.ANNUAL, Mode.QUARTER, Mode.QUARTERLY],
        help="高级：数据模式（默认 quarterly）",
    )
    p.add_argument("--years", type=int)
    p.add_argument("--quarters", type=int)
    vip_group = p.add_mutually_exclusive_group()
    vip_group.add_argument(
        "--vip", action="store_true", help="高级：显式启用 VIP（默认启用）"
    )
    vip_group.add_argument(
        "--no-vip", action="store_true", help="高级：禁用 VIP（已废弃）"
    )
    p.add_argument("--fields", type=str)
    p.add_argument("--outdir", type=str)
    p.add_argument("--prefix", type=str)
    p.add_argument("--format", choices=["csv", "parquet"])
    p.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    # 让默认值为 None，以便不覆盖默认“增量补全”
    p.add_argument("--skip-existing", action="store_true", default=None)
    # 顶层入口也支持 --force（强制覆盖）
    p.add_argument("--force", action="store_true")
    p.add_argument("--token", type=str)
    # 可选：分区化数据集写入相关参数
    p.add_argument(
        "--datasets-config",
        type=str,
        default=None,
        help="数据集配置文件，默认读取 configs/datasets.yaml",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="分区化数据集根目录，提供后将额外写入分区化数据集",
    )
    p.add_argument(
        "--no-only-latest",
        action="store_true",
        help="写入分区化数据集时，包含历史版本而不仅是最新快照",
    )

    # 子命令：download/build/coverage
    sub = p.add_subparsers(dest="cmd")
    # build 子命令
    sp_bld = sub.add_parser("build", help="由本地事实表构建 annual/quarterly 导出")
    sp_bld.add_argument("--dataset-root", type=str, required=True)
    sp_bld.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    sp_bld.add_argument(
        "--kinds",
        type=str,
        default="annual,quarterly",
        help="逗号分隔：annual,quarterly",
    )
    sp_bld.add_argument(
        "--annual-strategy",
        choices=["cumulative", "sum4"],
        default="cumulative",
        help="年度口径：累计或四季相加",
    )
    sp_bld.add_argument("--out-format", choices=["csv", "parquet"], default="csv")
    sp_bld.add_argument("--out-dir", type=str, default="out")
    sp_bld.add_argument("--prefix", type=str, default="income")

    # coverage 子命令
    sp_cov = sub.add_parser("coverage", help="盘点已覆盖的股票×期末日")
    sp_cov.add_argument("--dataset-root", type=str, required=True)
    sp_cov.add_argument(
        "--by",
        choices=["ticker", "ts_code", "period"],
        default="ticker",
        help="输出维度：ticker/ts_code 或 period",
    )

    # download 子命令（统一入口，默认增量补全，--force 强制覆盖）
    sp_dl = sub.add_parser("download", help="下载数据（默认增量补全；--force 覆盖）")
    sp_dl.add_argument("--config", type=str, default=None)
    sp_dl.add_argument(
        "--mode",
        choices=[Mode.ANNUAL, Mode.QUARTER, Mode.QUARTERLY],
        help="高级：数据模式（默认 quarterly）",
    )
    sp_dl.add_argument("--years", "--year", dest="years", type=int)
    sp_dl.add_argument("--quarters", type=int)
    sp_dl.add_argument(
        "--since", type=str, help="起始日期 YYYY-MM-DD（优先于 --years/--quarters）"
    )
    sp_dl.add_argument("--until", type=str, help="结束日期 YYYY-MM-DD（默认今天）")
    sp_dl.add_argument("--fields", type=str)
    sp_dl.add_argument("--outdir", type=str)
    sp_dl.add_argument("--prefix", type=str)
    sp_dl.add_argument("--format", choices=["csv", "parquet"])
    sp_dl.add_argument(
        "--export-colname",
        choices=["ticker", "ts_code"],
        default="ticker",
        help="导出列名：ticker 或 ts_code",
    )
    sp_dl.add_argument("--token", type=str)
    sp_dl.add_argument(
        "--datasets-config",
        type=str,
        default=None,
        help="数据集配置文件，默认读取 configs/datasets.yaml",
    )
    sp_dl.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="分区化数据集根目录，提供后将额外写入分区化数据集",
    )
    sp_dl.add_argument(
        "--no-only-latest",
        action="store_true",
        help="写入分区化数据集时，包含历史版本而不仅是最新快照",
    )
    sp_dl.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖已有文件（忽略增量跳过）",
    )
    args = p.parse_args()
    return args


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


def _load_datasets_config(path: Optional[str]) -> dict:
    if path is None:
        candidate = os.path.join(os.getcwd(), "configs", "datasets.yaml")
        if os.path.exists(candidate):
            path = candidate
        else:
            return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        eprint(f"警告：读取数据集配置失败（{path}）：{exc}")
        return {}


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


def _periods_from_cfg(cfg: dict, mode: str) -> List[str]:
    """根据 cfg 计算 periods 列表。

    优先级：since/until > quarters > years（默认10年）。
    - 若提供 since（可选 until），自动依据 mode 对应的粒度计算覆盖的 period 列表。
    - 否则若提供 quarters，按季度数量回溯。
    - 否则按 years 与 mode 计算（years 默认为 10）。
    """
    if cfg.get("since"):
        plan = plan_from_mode(mode)
        return _periods_from_range(plan.periodicity, cfg["since"], cfg.get("until"))
    if cfg.get("quarters") and cfg["quarters"] > 0:
        return periods_by_quarters(cfg["quarters"])
    return periods_for_mode_by_years(cfg.get("years", 10), mode)


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


def _write_datasets_from_tables(
    tables: Dict[str, pd.DataFrame], cfg: dict, periods: List[str]
) -> None:
    """将下载结果写入分区化数据集，兼容旧/新两套布局。

    - 新布局：dataset=income（包含 is_latest；only_latest 由 cfg.no_only_latest 控制）
    - 旧布局（为覆盖 build/coverage 等流程保留）：
        - dataset=fact_income_single（只写最新快照）
        - dataset=fact_income_cum（只写最新快照）
        - dataset=inventory_income/periods.parquet
    """
    root = cfg.get("dataset_root")
    if not root:
        return
    ds_cfg = _load_datasets_config(cfg.get("datasets_config"))

    raw_df = tables.get("raw")
    if raw_df is not None and not raw_df.empty:
        dataset = "income"
        ds = (ds_cfg.get("datasets", {}) or {}).get(dataset, {}) if ds_cfg else {}
        partition_by = ds.get("partition_by", "year:end_date")
        primary_key = ds.get("primary_key", ["ts_code", "end_date", "report_type"])
        version_by = ds.get("version_by", ["ann_date", "f_ann_date"])
        only_latest = not cfg.get("no_only_latest", False)
        raw_with_flag = _tx_mark_latest(raw_df)
        written = write_partitioned_dataset(
            raw_with_flag,
            root,
            dataset,
            partition_by,
            primary_key,
            version_by,
            only_latest=only_latest,
        )
        for p in written:
            print(f"已写入分区文件：{p}")

    # 旧布局：从 tables 推导 single/cum 并写入
    single_df = tables.get("single")
    if (
        (single_df is None or single_df.empty)
        and raw_df is not None
        and not raw_df.empty
    ):
        single_df = _diff_to_single(raw_df)
    cum_df = (
        _single_to_cumulative(single_df)
        if single_df is not None and not single_df.empty
        else pd.DataFrame()
    )

    if single_df is not None and not single_df.empty:
        from tushare_a_fundamentals.transforms.deduplicate import mark_latest as _mark

        single_df = _mark(single_df)
        write_partitioned_dataset(
            single_df,
            root,
            "fact_income_single",
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("partition_by")
                or "year:end_date"
            ),
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("primary_key")
                or ["ts_code", "end_date", "report_type"]
            ),
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("version_by")
                or ["ann_date", "f_ann_date"]
            ),
            only_latest=True,
        )
        print("已写入：fact_income_single")
    if cum_df is not None and not cum_df.empty:
        from tushare_a_fundamentals.transforms.deduplicate import mark_latest as _mark

        cum_df = _mark(cum_df)
        write_partitioned_dataset(
            cum_df,
            root,
            "fact_income_cum",
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("partition_by")
                or "year:end_date"
            ),
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("primary_key")
                or ["ts_code", "end_date", "report_type"]
            ),
            (
                ds_cfg.get("datasets", {}).get("income", {}).get("version_by")
                or ["ann_date", "f_ann_date"]
            ),
            only_latest=True,
        )
        print("已写入：fact_income_cum")

    # 写入 inventory_periods（覆盖清单）
    if periods:
        inv = pd.DataFrame({"end_date": periods})
        inv["is_present"] = 1
        inv = inv.sort_values("end_date")
        try:
            from pathlib import Path

            inv_dir = Path(root) / "dataset=inventory_income"
            inv_dir.mkdir(parents=True, exist_ok=True)
            inv_path = inv_dir / "periods.parquet"
            inv.to_parquet(inv_path, index=False)
            print(f"已写入：{inv_path}")
        except Exception as exc:
            eprint(f"警告：写入 inventory_periods 失败：{exc}")


def _run_bulk_mode(
    pro, cfg: dict, mode: str, fields: str, fmt: str, outdir: str, prefix: str
) -> None:
    if not _has_enough_credits(pro):
        total = _available_credits(pro) or 0
        eprint(f"错误：全市场批量需要至少 5000 积分。（检测到总积分：{int(total)}）")
        sys.exit(2)
    periods = _periods_from_cfg(cfg, mode)
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
    )
    save_tables(tables, outdir, base, fmt, cfg.get("export_colname", "ticker"))
    _write_datasets_from_tables(tables, cfg, periods)


def cmd_build(args: argparse.Namespace) -> None:
    root = args.dataset_root
    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]
    out_fmt = args.out_format
    out_dir = args.out_dir
    prefix = args.prefix

    single = _load_dataset(root, "fact_income_single")
    cum = _load_dataset(root, "fact_income_cum")
    # 只保留最新快照
    if "is_latest" in single.columns:
        single = single[single["is_latest"] == 1]
    if "is_latest" in cum.columns:
        cum = cum[cum["is_latest"] == 1]

    built: Dict[str, pd.DataFrame] = {}
    if "quarterly" in kinds:
        built["quarterly"] = single.copy()
    if "annual" in kinds:
        if args.annual_strategy == "cumulative":
            annual = cum.copy()
            annual = annual[annual["end_date"].astype(str).str.endswith("1231")]
        else:
            sdf = single.copy()
            sdf["year"] = sdf["end_date"].astype(str).str.slice(0, 4)
            aggs = {c: "sum" for c in FLOW_FIELDS if c in sdf.columns}
            annual = sdf.groupby(["ts_code", "year"], as_index=False).agg({**aggs})
            annual["end_date"] = annual["year"].astype(str) + "1231"
            # 组装标识列（取该年内最新公告）
            if set(["ann_date", "f_ann_date"]).issubset(sdf.columns):
                last_ann = (
                    sdf.sort_values(["ts_code", "year", "f_ann_date", "ann_date"])
                    .groupby(["ts_code", "year"], as_index=False)
                    .tail(1)[["ts_code", "year", "ann_date", "f_ann_date"]]
                )
                annual = annual.merge(last_ann, on=["ts_code", "year"], how="left")
            annual = annual.drop(
                columns=[
                    c
                    for c in annual.columns
                    if c
                    not in set(
                        ["ts_code", "end_date", *FLOW_FIELDS, "ann_date", "f_ann_date"]
                    )
                ]
            )
        built["annual"] = annual
    if not built:
        eprint("错误：未选择任何导出口径")
        sys.exit(2)
    _export_tables(built, out_dir, prefix, out_fmt, args.export_colname)


def cmd_coverage(args: argparse.Namespace) -> None:
    from pathlib import Path

    root = Path(args.dataset_root)
    inv_path = root / "dataset=inventory_income" / "periods.parquet"
    try:
        inv = pd.read_parquet(inv_path)
    except Exception as exc:
        eprint(f"错误：读取 {inv_path} 失败：{exc}")
        sys.exit(2)
    periods = sorted(inv["end_date"].astype(str).tolist())
    single = _load_dataset(str(root), "fact_income_single")
    if "is_latest" in single.columns:
        single = single[single["is_latest"] == 1]
    codes = sorted(single["ts_code"].unique())
    full = pd.MultiIndex.from_product(
        [codes, periods], names=["ts_code", "end_date"]
    ).to_frame(index=False)
    present = single[["ts_code", "end_date"]].drop_duplicates()
    present["is_present"] = 1
    cov = full.merge(present, on=["ts_code", "end_date"], how="left").fillna(
        {"is_present": 0}
    )
    if args.by in ("ticker", "ts_code"):
        pivot = cov.pivot(index="ts_code", columns="end_date", values="is_present")
        pivot.index.name = "ticker"
    else:
        pivot = cov.pivot(index="end_date", columns="ts_code", values="is_present")
        pivot.columns.name = "ticker"
    pivot = pivot.sort_index().fillna(0).astype(int)
    print(pivot.to_string())


def cmd_download(args: argparse.Namespace) -> None:
    """统一下载入口：默认增量补全，--force 则强制覆盖重下。

    行为与顶层旧参数一致，但默认改为增量补全（如果已有所需 period 则跳过）。
    """
    cfg_file = load_yaml(getattr(args, "config", None))
    # 默认开启增量跳过
    defaults = {
        "mode": Mode.QUARTERLY,
        "years": 10,
        "quarters": None,
        "since": None,
        "until": None,
        "fields": ",".join(DEFAULT_FIELDS),
        "outdir": "out",
        "prefix": "income",
        "format": "parquet",
        "skip_existing": True,  # download 默认增量
        "token": None,
        "datasets_config": None,
        "dataset_root": None,
        "no_only_latest": False,
        "export_colname": "ticker",
    }
    cli_overrides = {
        "mode": getattr(args, "mode", None),
        "years": getattr(args, "years", None),
        "quarters": getattr(args, "quarters", None),
        "since": getattr(args, "since", None),
        "until": getattr(args, "until", None),
        "fields": getattr(args, "fields", None),
        "outdir": getattr(args, "outdir", None),
        "prefix": getattr(args, "prefix", None),
        "format": getattr(args, "format", None),
        # 注意：skip_existing 默认 True，仅在 --force 时覆盖为 False
        "token": getattr(args, "token", None),
        "datasets_config": getattr(args, "datasets_config", None),
        "dataset_root": getattr(args, "dataset_root", None),
        "no_only_latest": getattr(args, "no_only_latest", None),
        "export_colname": getattr(args, "export_colname", None),
    }
    cfg = merge_config(cli_overrides, cfg_file, defaults)

    # 顶层也支持 --force：强制覆盖，忽略增量跳过
    if getattr(args, "force", False):
        cfg["skip_existing"] = False
    if getattr(args, "force", False):
        cfg["skip_existing"] = False

    pro = init_pro_api(cfg.get("token"))
    mode = cfg["mode"]
    fields = cfg["fields"]
    fmt = cfg["format"]
    if fmt == "parquet" and not _check_parquet_dependency():
        eprint("警告：缺少 pyarrow 或 fastparquet，已回退到 csv 格式")
        fmt = "csv"
    outdir = cfg["outdir"]
    prefix = cfg["prefix"]
    _run_bulk_mode(pro, cfg, mode, fields, fmt, outdir, prefix)


def main():
    args = parse_cli()
    # 子命令优先
    if getattr(args, "cmd", None) == "download":
        return cmd_download(args)
    if getattr(args, "cmd", None) == "build":
        return cmd_build(args)
    if getattr(args, "cmd", None) == "coverage":
        return cmd_coverage(args)
    cfg_file = load_yaml(args.config)
    defaults = {
        "mode": Mode.QUARTERLY,
        "years": 10,
        "quarters": None,
        "since": None,
        "until": None,
        "fields": ",".join(DEFAULT_FIELDS),
        "outdir": "out",
        "prefix": "income",
        "format": "parquet",
        "skip_existing": True,  # 顶层无子命令也采用“默认增量补全”
        "token": None,
        "datasets_config": None,
        "dataset_root": None,
        "no_only_latest": False,
        "export_colname": "ticker",
    }
    cli_overrides = {
        "mode": args.mode,
        "years": args.years,
        "quarters": args.quarters,
        "since": getattr(args, "since", None) if hasattr(args, "since") else None,
        "until": getattr(args, "until", None) if hasattr(args, "until") else None,
        "fields": args.fields,
        "outdir": args.outdir,
        "prefix": args.prefix,
        "format": args.format,
        "skip_existing": args.skip_existing,
        "token": args.token,
        "datasets_config": args.datasets_config,
        "dataset_root": args.dataset_root,
        "no_only_latest": args.no_only_latest,
        "export_colname": args.export_colname,
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
    _run_bulk_mode(pro, cfg, mode, fields, fmt, outdir, prefix)


if __name__ == "__main__":
    main()
