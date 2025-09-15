import pytest

from tushare_a_fundamentals.common import _periods_from_range

pytestmark = pytest.mark.unit


def test_periods_from_range_quarterly():
    periods = _periods_from_range("quarterly", "2019-01-01", "2019-12-31")
    assert periods == ["20190331", "20190630", "20190930", "20191231"]


def test_periods_from_range_annual():
    periods = _periods_from_range("annual", "2019-01-01", "2019-12-31")
    assert periods == ["20191231"]
