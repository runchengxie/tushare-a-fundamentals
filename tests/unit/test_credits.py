import pandas as pd
import pytest

from tushare_a_fundamentals.common import (
    TokenInfo,
    _format_token_log,
    _has_enough_credits,
)

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


def test_has_enough_credits_boundary():
    pro = DummyPro(pd.DataFrame({"到期积分": [4999.999, 0.0]}))
    assert _has_enough_credits(pro)


def test_format_token_log_marks_detection_disabled():
    info = TokenInfo(token="abc", credits=None, is_vip=False)

    message = _format_token_log(1, info, [], detect_vip=False)

    assert "未启用自动检测" in message
    assert "token#1" in message


def test_format_token_log_reports_detection_failure():
    info = TokenInfo(
        token="xyz",
        credits=None,
        is_vip=False,
        source="自动识别",
        detection_failed=True,
    )

    message = _format_token_log(2, info, [], detect_vip=True)

    assert "积分检测失败" in message
    assert "来源：自动识别" in message
