import pandas as pd
import pytest

from tushare_a_fundamentals.common import fetch_income_bulk, normalize_fields

pytestmark = pytest.mark.unit


class DummyPro:
    def __init__(self):
        self.calls = []
        self.kwargs = []

    def income_vip(self, **kwargs):
        self.calls.append(kwargs.get("report_type"))
        self.kwargs.append(kwargs)
        period = kwargs.get("period")
        report_type = kwargs.get("report_type")
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
    assert pro.kwargs and pro.kwargs[0]["fields"] == fields
    assert set(tables["raw"]["report_type"]) == {1, 6}


def test_fetch_income_bulk_skips_empty_fields():
    pro = DummyPro()
    periods = ["20231231"]
    tables = fetch_income_bulk(
        pro, periods=periods, mode="quarterly", fields=None, report_types=[1]
    )
    assert pro.calls == [1]
    assert pro.kwargs and "fields" not in pro.kwargs[0]
    assert not tables["raw"].empty


def test_normalize_fields_helpers():
    assert normalize_fields("") is None
    assert normalize_fields("   ") is None
    assert normalize_fields(["a", "", "b"]) == "a,b"
