import argparse

import pandas as pd
import pytest

from tushare_a_fundamentals.commands import export as expmod

pytestmark = pytest.mark.unit


def test_cmd_export_generates_single_and_cumulative(monkeypatch):
    cum = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20230331", "20230630"],
            "total_revenue": [10.0, 30.0],
        }
    )

    monkeypatch.setattr(expmod, "_load_dataset", lambda root, name: cum)
    saved = {}

    def fake_export_tables(built, out_dir, prefix, out_fmt, colname):
        saved.update(built)

    monkeypatch.setattr(expmod, "_export_tables", fake_export_tables)

    args = argparse.Namespace(
        dataset_root="root",
        kinds="single,cumulative",
        out_format="csv",
        out_dir="out",
        prefix="income",
        export_colname="ticker",
        annual_strategy="cumulative",
    )

    expmod.cmd_export(args)

    assert set(saved.keys()) == {"single", "cumulative"}
    single_df = saved["single"]
    q1, q2 = single_df["total_revenue"].tolist()
    assert q1 == 10.0
    assert q2 == 20.0


def test_cmd_export_years_filter(monkeypatch):
    periods = [
        "20211231",
        "20220331",
        "20220630",
        "20220930",
        "20221231",
    ]
    cum = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(periods),
            "end_date": periods,
            "total_revenue": list(range(len(periods))),
        }
    )

    monkeypatch.setattr(expmod, "_load_dataset", lambda root, name: cum)
    saved = {}

    def fake_export_tables(built, out_dir, prefix, out_fmt, colname):
        saved.update(built)

    monkeypatch.setattr(expmod, "_export_tables", fake_export_tables)

    args = argparse.Namespace(
        dataset_root="root",
        kinds="cumulative",
        out_format="csv",
        out_dir="out",
        prefix="income",
        export_colname="ticker",
        annual_strategy="cumulative",
        years=1,
    )

    expmod.cmd_export(args)

    cum_df = saved["cumulative"]
    assert set(cum_df["end_date"].astype(str)) == set(periods[-4:])
