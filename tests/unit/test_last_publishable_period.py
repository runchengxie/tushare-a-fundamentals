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
