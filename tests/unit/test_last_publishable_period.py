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


def test_periods_from_cfg_prefers_explicit_range(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "99991231",
    )
    def fail_quarters(count: int) -> list[str]:
        raise AssertionError("should not use quarters")

    def fail_backfill(anchor: str, count: int) -> list[str]:
        raise AssertionError("should not backfill")

    monkeypatch.setattr(
        "tushare_a_fundamentals.common.periods_by_quarters", fail_quarters
    )
    monkeypatch.setattr(
        "tushare_a_fundamentals.common._backfill_periods", fail_backfill
    )

    called: dict[str, tuple[str, str | None]] = {}

    def fake_from_range(mode: str, since: str, until: str | None) -> list[str]:
        called["args"] = (mode, since, until)
        return ["RANGE"]

    monkeypatch.setattr(
        "tushare_a_fundamentals.common._periods_from_range", fake_from_range
    )

    periods = _periods_from_cfg(
        {
            "since": "2023-01-01",
            "until": "2023-12-31",
            "quarters": 4,
            "years": 3,
            "allow_future": True,
        }
    )

    assert periods == ["RANGE"]
    assert called["args"] == ("quarterly", "2023-01-01", "2023-12-31")


def test_periods_from_cfg_quarters_override_years(monkeypatch):
    monkeypatch.setattr(
        "tushare_a_fundamentals.common.last_publishable_period",
        lambda today: "20251231",
    )
    calls: dict[str, int] = {}

    def fake_quarters(count: int) -> list[str]:
        calls["count"] = count
        return [f"Q{i}" for i in range(count)]

    monkeypatch.setattr(
        "tushare_a_fundamentals.common.periods_by_quarters", fake_quarters
    )

    def fail_backfill(anchor: str, count: int) -> list[str]:
        raise AssertionError("should not backfill")

    monkeypatch.setattr(
        "tushare_a_fundamentals.common._backfill_periods", fail_backfill
    )

    periods = _periods_from_cfg({"quarters": 3, "years": 10, "allow_future": True})

    assert calls["count"] == 3
    assert periods == ["Q0", "Q1", "Q2"]
