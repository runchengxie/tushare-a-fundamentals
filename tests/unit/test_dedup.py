import pandas as pd
import pytest

from tushare_a_fundamentals.common import _select_latest
from tushare_a_fundamentals.transforms.deduplicate import (
    mark_latest as _tx_mark_latest,
)

pytestmark = pytest.mark.unit


def test_select_latest_priority():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "end_date": ["20231231"] * 3,
            "report_type": [2, 1, 2],
            "f_ann_date": ["20240201", "20240131", "20240210"],
            "ann_date": ["20240201", "20240131", "20240209"],
        }
    )
    got = _select_latest(df)
    assert len(got) == 1
    assert int(got.iloc[0]["report_type"]) == 1


def test_update_flag_break_tie():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "report_type": [2, 2],
            "f_ann_date": ["20240210", "20240210"],
            "ann_date": ["20240209", "20240209"],
            "update_flag": ["N", "Y"],
        }
    )
    got = _select_latest(df)
    assert got.iloc[0]["update_flag"] == "Y"


def test_mark_latest_extra_sort_keys():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "ann_date": ["20240101", "20240101"],
            "priority": [1, 2],
        }
    )
    flagged = _tx_mark_latest(df, extra_sort_keys=["priority"])
    assert flagged.loc[flagged["is_latest"] == 1, "priority"].iloc[0] == 2


def test_mark_latest_custom_group_keys():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20230630"],
            "ann_date": ["20240101", "20230801"],
        }
    )
    flagged = _tx_mark_latest(df, group_keys=["ts_code"])
    assert flagged["is_latest"].sum() == 1
    assert flagged.loc[flagged["is_latest"] == 1, "end_date"].iloc[0] == "20231231"


def test_select_latest_with_report_type_grouping():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "report_type": [1, 6],
            "ann_date": ["20240101", "20240102"],
        }
    )
    got = _select_latest(df, group_keys=("ts_code", "end_date", "report_type"))
    assert set(got["report_type"]) == {1, 6}
