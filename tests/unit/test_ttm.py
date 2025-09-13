import os
import sys
import math
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from tushare_a_fundamentals import cli as appmod


def test_ttm_rolling_sum_min4():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ"]*4,
        "end_date": ["20230331", "20230630", "20230930", "20231231"],
        "total_revenue": [10.0, 15.0, 20.0, 25.0],
    })
    ttm = appmod._rolling_ttm(df)
    vals = ttm["total_revenue"].tolist()
    assert all(math.isnan(v) for v in vals[:3])
    assert vals[3] == 70.0
