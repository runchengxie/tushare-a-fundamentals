from pathlib import Path

import pandas as pd

from tushare_a_fundamentals.downloader import write_parquet_dataset


def _read_partition(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def test_write_parquet_dataset_deduplicates(tmp_path):
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20231231", "20231231"],
            "retrieved_at": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-02-01T00:00:00Z"),
            ],
            "value": [1, 2],
        }
    )

    ok = write_parquet_dataset(
        df,
        root=tmp_path.as_posix(),
        dataset="dividend",
        year_col="ann_date",
        group_keys=("ts_code", "ann_date"),
    )

    assert ok is True
    partition_dir = tmp_path / "dividend" / "year=2023"
    stored = _read_partition(partition_dir)
    assert len(stored) == 1
    assert stored.loc[0, "value"] == 2

    df_update = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20231231", "20240131"],
            "retrieved_at": [
                pd.Timestamp("2024-03-01T00:00:00Z"),
                pd.Timestamp("2024-04-01T00:00:00Z"),
            ],
            "value": [3, 4],
        }
    )

    write_parquet_dataset(
        df_update,
        root=tmp_path.as_posix(),
        dataset="dividend",
        year_col="ann_date",
        group_keys=("ts_code", "ann_date"),
    )

    stored_current = _read_partition(partition_dir)
    assert len(stored_current) == 1
    assert stored_current.loc[0, "value"] == 3

    partition_dir_new = tmp_path / "dividend" / "year=2024"
    stored_new = _read_partition(partition_dir_new)
    assert len(stored_new) == 1
    assert stored_new.loc[0, "ts_code"] == "000002.SZ"
