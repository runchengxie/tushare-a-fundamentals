import pandas as pd
import pytest

from tushare_a_fundamentals.common import save_tables

pytestmark = pytest.mark.unit


def test_save_naming(tmp_path):
    tables = {
        "raw": pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20231231"]}),
    }
    outdir = tmp_path
    save_tables(tables, str(outdir), "income_vip_quarter", "csv")
    assert (outdir / "csv" / "income_vip_quarter_raw.csv").exists()
