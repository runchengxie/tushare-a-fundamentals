import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "tools"))
from convert_parquet_to_csv import convert_parquet_to_csv

pytestmark = pytest.mark.unit


def test_convert_parquet_to_csv(tmp_path):
    pytest.importorskip("pyarrow")
    src = tmp_path / "parquet"
    dest = tmp_path / "csv"
    src.mkdir()
    df = pd.DataFrame({"a": [1], "b": [2]})
    df.to_parquet(src / "foo.parquet")
    convert_parquet_to_csv(src, dest)
    out = dest / "foo.csv"
    assert out.exists()
    df2 = pd.read_csv(out)
    assert df.equals(df2)
