import pandas as pd
import pytest

from tushare_a_fundamentals.common import build_datasets_from_raw

pytestmark = pytest.mark.unit


def test_build_datasets_from_raw(tmp_path):
    raw_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231", "20230930"],
            "report_type": [1, 6, 1],
            "update_flag": ["N", "N", "N"],
        }
    )
    df.to_parquet(raw_dir / "income_vip_quarterly_raw.parquet", index=False)
    build_datasets_from_raw(str(tmp_path), "income")
    inv = pd.read_parquet(tmp_path / "dataset=inventory_income" / "periods.parquet")
    assert inv["end_date"].tolist() == ["20230930", "20231231"]
    fact_file = tmp_path / "dataset=fact_income_cum" / "year=2023" / "part.parquet"
    fact = pd.read_parquet(fact_file)
    assert set(fact["end_date"]) == {"20230930", "20231231"}
    assert (fact["is_latest"] == 1).all()
