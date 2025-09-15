import pandas as pd
import pytest

from tushare_a_fundamentals.common import fetch_income_bulk

pytestmark = pytest.mark.unit


class DummyPro:
    def __init__(self):
        self.calls = []

    def income_vip(self, fields: str, period: str, report_type: int):
        self.calls.append(report_type)
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "end_date": [period],
                "report_type": [report_type],
                "ann_date": ["20240101"],
                "f_ann_date": ["20240101"],
            }
        )


def test_fetch_income_bulk_multiple_report_types():
    pro = DummyPro()
    periods = ["20231231"]
    fields = "ts_code,ann_date,f_ann_date,end_date,report_type"
    tables = fetch_income_bulk(
        pro, periods=periods, mode="quarterly", fields=fields, report_types=[1, 6]
    )
    assert pro.calls == [1, 6]
    assert set(tables["raw"]["report_type"]) == {1, 6}
