import pytest

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.unit


def test_periods_years_annual():
    periods = appmod.periods_for_mode_by_years(2, appmod.Mode.ANNUAL)
    assert all(p.endswith("1231") for p in periods)
    assert len(periods) == 2


def test_periods_quarters_count():
    periods = appmod.periods_by_quarters(6)
    assert len(periods) == 6
    assert periods == sorted(periods)


@pytest.mark.parametrize("years", [0, -1])
def test_periods_years_non_positive(years):
    assert appmod.periods_for_mode_by_years(years, appmod.Mode.ANNUAL) == []


@pytest.mark.parametrize("quarters", [0, -2])
def test_periods_quarters_non_positive(quarters):
    assert appmod.periods_by_quarters(quarters) == []
