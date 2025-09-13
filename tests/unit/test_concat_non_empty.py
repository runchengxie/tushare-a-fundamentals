import warnings
import pandas as pd
from tushare_a_fundamentals.cli import _concat_non_empty


def test_concat_non_empty_filters_all_na():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [None]})
    df3 = pd.DataFrame({"a": [2]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _concat_non_empty([df1, df2, df3])
    assert out.shape[0] == 2
