from datetime import date

import pytest

from tushare_a_fundamentals.common import _periods_from_cfg, last_publishable_period

pytestmark = pytest.mark.unit


def test_last_publishable_period():
    assert last_publishable_period(date(2025, 9, 15)) == "20250630"
    assert last_publishable_period(date(2025, 11, 1)) == "20250930"


def test_periods_from_cfg_trim_future(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "20250630",
    )
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    periods = _periods_from_cfg(cfg)
    assert periods == ["20250331", "20250630"]
    cfg["allow_future"] = True
    periods = _periods_from_cfg(cfg)
    assert periods == ["20250331", "20250630", "20250930", "20251231"]


def test_periods_from_cfg_years_backfill(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "20250630",
    )
    periods = _periods_from_cfg({})
    assert len(periods) == 40
    assert periods[0] == "20150930"
    assert periods[-1] == "20250630"


def test_periods_from_cfg_quarters_backfill(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "20250630",
    )
    periods = _periods_from_cfg({"quarters": 4})
    assert periods == ["20240930", "20241231", "20250331", "20250630"]


def test_periods_from_cfg_years_allow_future(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "20250630",
    )
    captured: dict[str, int] = {}

    def fake_periods_by_quarters(count: int) -> list[str]:
        captured["count"] = count
        return [f"P{i}" for i in range(count)]

    monkeypatch.setattr(
        "tushare_a_fundamentals.common.periods_by_quarters",
        fake_periods_by_quarters,
    )
    periods = _periods_from_cfg({"years": 1, "allow_future": True})
    assert captured["count"] == 4
    assert periods == ["P0", "P1", "P2", "P3"]
