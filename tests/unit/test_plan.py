import pytest
from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.unit


def test_plan_mapping():
    p = appmod.plan_from_mode("annual")
    assert p.periodicity == "annual"
    assert p.view == "reported"

    p = appmod.plan_from_mode("quarter")
    assert p.periodicity == "quarterly"
    assert p.view == "quarter"

    p = appmod.plan_from_mode("ttm")
    assert p.periodicity == "quarterly"
    assert p.view == "ttm"


def test_plan_override():
    p = appmod.plan_from_mode("annual", periodicity="quarterly")
    assert p.periodicity == "quarterly"
    assert p.view == "reported"


def test_plan_deprecated_alias(capfd):
    p = appmod.plan_from_mode("quarter")
    assert p.periodicity == "quarterly"
    assert p.view == "quarter"
    err = capfd.readouterr().err
    assert "已弃用" in err
