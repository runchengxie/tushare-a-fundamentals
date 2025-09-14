import pytest

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.unit


def test_plan_mapping():
    p = appmod.plan_from_mode("annual")
    assert p.periodicity == "annual"

    p = appmod.plan_from_mode("quarter")
    assert p.periodicity == "quarterly"


def test_plan_override():
    p = appmod.plan_from_mode("annual", periodicity="quarterly")
    assert p.periodicity == "quarterly"


def test_plan_deprecated_alias(capfd):
    p = appmod.plan_from_mode("quarter")
    assert p.periodicity == "quarterly"
    err = capfd.readouterr().err
    assert "已弃用" in err
