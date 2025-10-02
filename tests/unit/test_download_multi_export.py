from pathlib import Path

import pandas as pd
import pytest

from tushare_a_fundamentals.commands import download as dlmod
from tushare_a_fundamentals.downloader import DatasetRequest

pytestmark = pytest.mark.unit


def test_export_income_from_multi(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20230331", "20230630"],
            "total_revenue": [10.0, 30.0],
            "ann_date": ["20230410", "20230710"],
            "f_ann_date": ["20230411", "20230711"],
        }
    )
    monkeypatch.setattr(
        dlmod,
        "_load_dataset_from_data_dir",
        lambda root, dataset: df.copy(),
    )
    cfg = dlmod._download_defaults()
    cfg.update(
        {
            "export_enabled": True,
            "outdir": str(tmp_path / "out"),
            "export_out_dir": None,
            "export_kinds": "cumulative,single",
        }
    )

    requests = [DatasetRequest(name="income")]

    dlmod._export_income_from_multi(cfg, str(data_dir), requests)

    csv_dir = Path(cfg["outdir"]) / "csv"
    assert (csv_dir / "income_cumulative.csv").exists()
    assert (csv_dir / "income_single.csv").exists()
