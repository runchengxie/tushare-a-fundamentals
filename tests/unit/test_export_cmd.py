import argparse
from pathlib import Path

import pandas as pd
import pytest

from tushare_a_fundamentals.commands import export as expmod

pytestmark = pytest.mark.unit


def _make_dataset(root: Path, name: str, year: str, frame: pd.DataFrame) -> None:
    target = root / name / f"year={year}"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "data.parquet", index=False)


def _default_args(dataset_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=str(dataset_root),
        years=None,
        kinds="annual,single,cumulative",
        annual_strategy="cumulative",
        out_format="csv",
        out_dir=str(dataset_root),
        prefix="income",
        flat_datasets="auto",
        flat_exclude="",
        split_by="none",
        gzip=False,
        no_income=False,
        no_flat=False,
    )


def test_cmd_export_full_flow(tmp_path):
    data_root = tmp_path / "data"
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "end_date": ["20230331", "20230630", "20231231"],
            "total_revenue": [10.0, 30.0, 90.0],
            "ann_date": ["20230410", "20230710", "20240110"],
            "f_ann_date": ["20230411", "20230711", "20240111"],
        }
    )
    balance = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20230331"],
            "total_assets": [100.0],
        }
    )
    _make_dataset(data_root, "income", "2023", income)
    _make_dataset(data_root, "balancesheet", "2023", balance)

    args = _default_args(data_root)
    expmod.cmd_export(args)

    csv_dir = data_root / "csv"
    assert (csv_dir / "income_cumulative.csv").exists()
    assert (csv_dir / "income_single.csv").exists()
    assert (csv_dir / "income_annual.csv").exists()
    balance_csv = csv_dir / "balancesheet.csv"
    assert balance_csv.exists()
    loaded = pd.read_csv(balance_csv)
    assert len(loaded) == 1
    assert loaded.loc[0, "total_assets"] == 100.0


def test_cmd_export_skip_income(tmp_path):
    data_root = tmp_path / "data"
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20230331"],
            "total_revenue": [10.0],
        }
    )
    extras = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20230331"],
            "cash": [5.0],
        }
    )
    _make_dataset(data_root, "income", "2023", income)
    _make_dataset(data_root, "cashflow", "2023", extras)

    args = _default_args(data_root)
    args.no_income = True
    expmod.cmd_export(args)

    csv_dir = data_root / "csv"
    assert not (csv_dir / "income_cumulative.csv").exists()
    assert (csv_dir / "cashflow.csv").exists()


def test_cmd_export_skip_flat(tmp_path):
    data_root = tmp_path / "data"
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20230331"],
            "total_revenue": [10.0],
        }
    )
    _make_dataset(data_root, "income", "2023", income)
    _make_dataset(data_root, "express", "2023", income)

    args = _default_args(data_root)
    args.no_flat = True
    expmod.cmd_export(args)

    csv_dir = data_root / "csv"
    assert (csv_dir / "income_cumulative.csv").exists()
    assert not (csv_dir / "express.csv").exists()


def test_cmd_export_conflicting_flags_raises(tmp_path):
    args = _default_args(tmp_path)
    args.no_income = True
    args.no_flat = True

    with pytest.raises(SystemExit) as excinfo:
        expmod.cmd_export(args)

    assert excinfo.value.code == 2
