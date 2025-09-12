import os
import sys
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
import app as appmod  # noqa: E402


def test_ttm_rolling_sum_min4():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ"]*4,
        "end_date": ["20230331", "20230630", "20230930", "20231231"],
        "total_revenue": [10.0, 15.0, 20.0, 25.0],
    })
    ttm = appmod._rolling_ttm(df)
    vals = ttm["total_revenue"].tolist()
    assert vals[:3] == [None, None, None]
    assert vals[3] == 70.0
