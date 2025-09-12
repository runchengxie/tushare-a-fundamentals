import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import app as appmod


def test_diff_to_single_cumulative_to_quarterly():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ"]*3,
        "end_date": ["20230331", "20230630", "20230930"],
        "total_revenue": [10.0, 25.0, 45.0],
    })
    got = appmod._diff_to_single(df)
    q1, q2, q3 = got["total_revenue"].tolist()
    assert q1 == 10.0
    assert q2 == 15.0
    assert q3 == 20.0
