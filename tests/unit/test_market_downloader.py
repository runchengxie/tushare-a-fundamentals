import json
from pathlib import Path

import pandas as pd
import pytest

from tushare_a_fundamentals.downloader import (
    DATASET_SPECS,
    DatasetRequest,
    MarketDatasetDownloader,
    parse_dataset_requests,
)

pytestmark = pytest.mark.unit


class DummyPro:
    def __init__(self):
        self.period_calls: list[str] = []
        self.window_calls: list[tuple[str, str]] = []
        self.audit_calls: list[tuple[str, str]] = []

    def income_vip(self, **kwargs):
        period = kwargs.get("period")
        self.period_calls.append(period)
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "end_date": [period],
                "report_type": [kwargs.get("report_type", 1)],
                "ann_date": ["20200101"],
            }
        )

    def dividend(self, **kwargs):
        win = (kwargs.get("start_date"), kwargs.get("end_date"))
        self.window_calls.append(win)
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": [kwargs.get("end_date")],
                "record_date": [None],
                "ex_date": [None],
                "imp_ann_date": [None],
            }
        )

    def fina_audit(self, **kwargs):
        ts_code = kwargs.get("ts_code")
        period = kwargs.get("period")
        self.audit_calls.append((ts_code, period))
        return pd.DataFrame(
            {
                "end_date": [period],
                "audit_opinion": ["标准无保留意见"],
            }
        )


def test_parse_dataset_requests():
    parsed = parse_dataset_requests(["income", {"name": "dividend", "foo": "bar"}])
    assert parsed == [
        DatasetRequest(name="income"),
        DatasetRequest(name="dividend", options={"foo": "bar"}),
    ]


def test_market_downloader_periodic(tmp_path, monkeypatch):
    pro = DummyPro()
    saved = []

    def fake_write(df, root, dataset, year_col, *, group_keys=None):
        saved.append((root, dataset, year_col, group_keys, df.copy()))
        return True

    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.write_parquet_dataset", fake_write
    )
    state_path = tmp_path / "state.json"
    dl = MarketDatasetDownloader(
        pro,
        data_dir=str(tmp_path),
        vip_pro=pro,
        use_vip=True,
        max_per_minute=0,
        state_path=str(state_path),
    )
    dl.run(
        [DatasetRequest(name="income", options={"report_types": [1]})],
        start="2020-01-01",
        end="2020-12-31",
    )
    assert pro.period_calls == [
        "20200331",
        "20200630",
        "20200930",
        "20201231",
    ]
    assert saved
    out_root, dataset, year_col, group_keys, df = saved[0]
    assert Path(out_root) == tmp_path
    assert dataset == "income"
    assert year_col == "end_date"
    assert group_keys == ("ts_code", "end_date")
    assert len(df) == 4
    state = json.loads(Path(state_path).read_text("utf-8"))
    assert state["income"]["last_period:rt=1"] == "20201231"


def test_market_downloader_prefers_vip_client(tmp_path, monkeypatch):
    class PassivePro:
        def income_vip(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("基础 token 不应承担 VIP 请求")

    vip = DummyPro()
    saved = []

    def fake_write(df, root, dataset, year_col, *, group_keys=None):
        saved.append(df.copy())
        return True

    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.write_parquet_dataset", fake_write
    )
    state_path = tmp_path / "state.json"
    dl = MarketDatasetDownloader(
        PassivePro(),
        data_dir=str(tmp_path),
        vip_pro=vip,
        use_vip=True,
        max_per_minute=0,
        state_path=str(state_path),
    )
    dl.run(
        [DatasetRequest(name="income", options={"report_types": [1]})],
        start="2020-01-01",
        end="2020-12-31",
    )
    assert vip.period_calls == [
        "20200331",
        "20200630",
        "20200930",
        "20201231",
    ]
    assert saved


def test_market_downloader_calendar(tmp_path, monkeypatch):
    pro = DummyPro()
    saved = []

    def fake_write(df, root, dataset, year_col, *, group_keys=None):
        saved.append((dataset, group_keys, df.copy()))
        return True

    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.write_parquet_dataset", fake_write
    )
    state_path = tmp_path / "state.json"
    dl = MarketDatasetDownloader(
        pro,
        data_dir=str(tmp_path),
        use_vip=False,
        max_per_minute=0,
        state_path=str(state_path),
    )
    req = DatasetRequest(name="dividend")
    dl.run([req], start="2020-01-01", end="2020-02-29")
    assert saved
    assert pro.window_calls == [("20200101", "20200131"), ("20200201", "20200229")]
    state = json.loads(Path(state_path).read_text("utf-8"))
    assert state["dividend"]["last_date"] == "20200229"


def test_market_downloader_per_stock(tmp_path, monkeypatch):
    pro = DummyPro()
    saved: list[tuple[str, pd.DataFrame]] = []

    def fake_write(df, root, dataset, year_col, *, group_keys=None):
        saved.append((dataset, df.copy()))
        return True

    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.write_parquet_dataset", fake_write
    )
    state_path = tmp_path / "state.json"
    dl = MarketDatasetDownloader(
        pro,
        data_dir=str(tmp_path),
        use_vip=False,
        max_per_minute=0,
        state_path=str(state_path),
    )
    req = DatasetRequest(
        name="fina_audit",
        options={"ts_codes": ["000001.SZ", "000002.SZ"]},
    )
    dl.run([req], start="2020-01-01", end="2020-12-31")
    assert pro.audit_calls == [
        ("000001.SZ", "20200331"),
        ("000001.SZ", "20200630"),
        ("000001.SZ", "20200930"),
        ("000001.SZ", "20201231"),
        ("000002.SZ", "20200331"),
        ("000002.SZ", "20200630"),
        ("000002.SZ", "20200930"),
        ("000002.SZ", "20201231"),
    ]
    assert saved
    dataset, df = saved[0]
    assert dataset == "fina_audit"
    assert set(df["ts_code"].unique()) == {"000001.SZ", "000002.SZ"}
    state = json.loads(Path(state_path).read_text("utf-8"))
    bucket = state["fina_audit"]
    assert "last_period" not in bucket
    assert bucket["last_period:ts=000001.SZ"] == "20201231"
    assert bucket["last_period:ts=000002.SZ"] == "20201231"

def test_failure_log_written_and_cleared(tmp_path):
    pro = DummyPro()
    dl = MarketDatasetDownloader(
        pro,
        data_dir=str(tmp_path),
        vip_pro=pro,
        use_vip=True,
        max_per_minute=0,
        state_path=str(tmp_path / "state.json"),
    )
    spec = DATASET_SPECS["income"]
    entries = [{"combo": "report_type=1", "periods": ["20200101"]}]
    dl._record_failures(spec, entries, "periods")
    failure_path = tmp_path / "_state" / "failures" / "income_periods.json"
    assert failure_path.exists()
    data = json.loads(failure_path.read_text("utf-8"))
    assert data["entries"] == entries
    dl._record_failures(spec, [], "periods")
    assert not failure_path.exists()
