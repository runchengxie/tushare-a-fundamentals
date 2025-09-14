import pandas as pd
from types import SimpleNamespace

from tushare_a_fundamentals.cli import cmd_coverage


def _prepare_dataset(root):
    inv_dir = root / "dataset=inventory_income"
    inv_dir.mkdir()
    pd.DataFrame({"end_date": ["20231231", "20230930"]}).to_parquet(
        inv_dir / "periods.parquet"
    )
    fact_dir = root / "dataset=fact_income_single"
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
