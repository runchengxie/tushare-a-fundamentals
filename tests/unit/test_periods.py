import pytest

from tushare_a_fundamentals.common import (
    Mode,
    periods_by_quarters,
    periods_for_mode_by_years,
)

pytestmark = pytest.mark.unit


def test_periods_years_quarterly():
    periods = periods_for_mode_by_years(2, Mode.QUARTERLY)
    assert all(p.endswith(("0331", "0630", "0930", "1231")) for p in periods)
    assert len(periods) == 8


def test_periods_quarters_count():
    periods = periods_by_quarters(6)
    assert len(periods) == 6
    assert periods == sorted(periods)


@pytest.mark.parametrize("years", [0, -1])
def test_periods_years_non_positive(years):
    assert periods_for_mode_by_years(years, Mode.QUARTERLY) == []


@pytest.mark.parametrize("quarters", [0, -2])
def test_periods_quarters_non_positive(quarters):
    assert periods_by_quarters(quarters) == []
