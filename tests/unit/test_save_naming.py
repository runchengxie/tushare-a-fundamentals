import os
import sys
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
import app as appmod  # noqa: E402


def test_save_naming(tmp_path):
    tables = {
        "raw": pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20231231"]}),
        "single": pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20231231"]}),
        "ttm": pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20231231"]}),
    }
    outdir = tmp_path
    appmod.save_tables(tables, str(outdir), "income_vip_quarterly", "csv")
    assert (outdir / "income_vip_quarterly_raw.csv").exists()
    assert (outdir / "income_vip_quarterly_single.csv").exists()
    assert (outdir / "income_vip_quarterly_ttm.csv").exists()
