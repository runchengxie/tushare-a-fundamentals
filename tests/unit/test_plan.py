import pytest

from tushare_a_fundamentals.common import plan_from_mode

pytestmark = pytest.mark.unit


def test_plan_mapping():
    p = plan_from_mode("annual")
    assert p.periodicity == "annual"


def test_plan_override():
    p = plan_from_mode("annual", periodicity="quarterly")
    assert p.periodicity == "quarterly"


def test_plan_rejects_quarter_alias():
    with pytest.raises(ValueError):
        plan_from_mode("quarter")
