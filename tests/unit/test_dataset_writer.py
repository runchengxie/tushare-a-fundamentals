from pathlib import Path

import pandas as pd
import pytest

from tushare_a_fundamentals.writers.dataset_writer import write_partitioned_dataset

pytestmark = pytest.mark.unit


def _sample_df():
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "ann_date": ["20240101", "20240101"],
            "f_ann_date": ["20240102", "20240102"],
            "report_type": [1, 1],
            "revenue": [10, 20],
            "is_latest": [1, 0],
        }
    )


def test_write_only_latest(tmp_path):
    df = _sample_df()
    paths = write_partitioned_dataset(
        df,
        tmp_path,
        "income",
        "year:end_date",
        primary_key=["ts_code"],
        version_by=["ann_date"],
        only_latest=True,
    )
    assert len(paths) == 1
    written = pd.read_parquet(paths[0])
    assert written.shape[0] == 1
    assert (Path(paths[0]).parent.parent.name) == "dataset=income"
    assert Path(paths[0]).parent.name == "year=2023"


def test_write_invalid_partition(tmp_path):
    df = _sample_df()
    with pytest.raises(ValueError):
        write_partitioned_dataset(df, tmp_path, "income", "month:end_date")
    with pytest.raises(KeyError):
        write_partitioned_dataset(
            df.drop(columns=["end_date"]), tmp_path, "income", "year:end_date"
        )
