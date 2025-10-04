from types import SimpleNamespace

import pandas as pd

from tushare_a_fundamentals.commands.coverage import cmd_coverage


def _prepare_dataset(root):
    inv_dir = root / "dataset=inventory_income"
    inv_dir.mkdir()
    pd.DataFrame({"end_date": ["20231231", "20230930"]}).to_parquet(
        inv_dir / "periods.parquet"
    )
    fact_dir = root / "dataset=fact_income_cum"
    fact_dir.mkdir()
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20230930"],
            "is_latest": [1, 1],
        }
    ).to_parquet(fact_dir / "data.parquet")
    return root


def test_cmd_coverage_by(tmp_path, capsys):
    root = _prepare_dataset(tmp_path)

    args = SimpleNamespace(dataset_root=str(root), by="ticker")
    cmd_coverage(args)
    out = capsys.readouterr().out
    assert "000001.SZ" in out

    args = SimpleNamespace(dataset_root=str(root), by="period")
    cmd_coverage(args)
    out = capsys.readouterr().out
    assert "20230930" in out


def test_cmd_coverage_years(tmp_path, capsys):
    inv_dir = tmp_path / "dataset=inventory_income"
    inv_dir.mkdir()
    periods = [
        "20211231",
        "20220331",
        "20220630",
        "20220930",
        "20221231",
    ]
    pd.DataFrame({"end_date": periods}).to_parquet(inv_dir / "periods.parquet")
    fact_dir = tmp_path / "dataset=fact_income_cum"
    fact_dir.mkdir()
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(periods),
            "end_date": periods,
            "is_latest": [1] * len(periods),
        }
    ).to_parquet(fact_dir / "data.parquet")
    args = SimpleNamespace(dataset_root=str(tmp_path), by="period", years=1)
    cmd_coverage(args)
    out = capsys.readouterr().out
    assert "20211231" not in out
    assert "20221231" in out


def test_cmd_coverage_csv_output(tmp_path):
    root = _prepare_dataset(tmp_path)
    csv_path = tmp_path / "coverage.csv"
    args = SimpleNamespace(dataset_root=str(root), by="ticker", csv=str(csv_path))

    cmd_coverage(args)

    content = csv_path.read_text("utf-8").strip().splitlines()
    assert len(content) >= 2
    header = content[0].split(",")
    assert {"ts_code", "missing_periods"}.issubset(set(header))
