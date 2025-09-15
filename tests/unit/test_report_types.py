import pytest

from tushare_a_fundamentals.common import parse_report_types

pytestmark = pytest.mark.unit


def test_parse_report_types():
    assert parse_report_types(None) == [1]
    assert parse_report_types("1,6") == [1, 6]
    assert parse_report_types(3) == [3]
