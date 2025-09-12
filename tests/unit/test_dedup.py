import os
import sys
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
import app as appmod  # noqa: E402


def test_select_latest_priority():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ"]*3,
        "end_date": ["20231231"]*3,
        "report_type": [2, 1, 2],
        "f_ann_date": ["20240201", "20240131", "20240210"],
        "ann_date": ["20240201", "20240131", "20240209"],
    })
    got = appmod._select_latest(df)
    assert len(got) == 1
    assert int(got.iloc[0]["report_type"]) == 1


def test_update_flag_break_tie():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "end_date": ["20231231", "20231231"],
        "report_type": [2, 2],
        "f_ann_date": ["20240210", "20240210"],
        "ann_date": ["20240209", "20240209"],
        "update_flag": ["N", "Y"],
    })
    got = appmod._select_latest(df)
    assert got.iloc[0]["update_flag"] == "Y"
