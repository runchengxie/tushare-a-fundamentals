import pandas as pd
import pytest

from tushare_a_fundamentals.cli import _has_enough_credits

pytestmark = pytest.mark.unit


class DummyPro:
    def __init__(self, df):
        self._df = df

    def user(self):
        return self._df


def test_has_enough_credits_true():
    pro = DummyPro(pd.DataFrame({"到期积分": [3000, 2500]}))
    assert _has_enough_credits(pro)


def test_has_enough_credits_false():
    pro = DummyPro(pd.DataFrame({"到期积分": [1000, 2000]}))
    assert not _has_enough_credits(pro)


def test_has_enough_credits_commas():
    pro = DummyPro(pd.DataFrame({"到期积分": ["3,000", "2,500"]}))
    assert _has_enough_credits(pro)
