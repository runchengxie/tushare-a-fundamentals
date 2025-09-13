import sys
import pandas as pd
import pytest

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.integration


def test_cli_ingest_and_build(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "init_pro_api", lambda token: object())

    def fake_ingest_single(pro, ts_code, periods, fields):
        df = pd.DataFrame(
            {
                "ts_code": [ts_code],
                "end_date": [periods[0]],
                "ann_date": ["20240101"],
                "f_ann_date": ["20240102"],
                "report_type": [1],
                "revenue": [100],
            }
        )
        return df, df.copy(), df.copy()

    monkeypatch.setattr(appmod, "_ingest_single", fake_ingest_single)

    argv_ingest = [
        "funda",
        "ingest",
        "--since",
        "2023-01-01",
        "--until",
        "2023-12-31",
        "--periods",
        "quarterly",
        "--ts-code",
        "000001.SZ",
        "--dataset-root",
        str(tmp_path),
        "--token",
        "fake",
    ]
    monkeypatch.setattr(sys, "argv", argv_ingest)
    appmod.main()
    assert list((tmp_path / "dataset=fact_income_single").glob("**/*.parquet"))

    out_dir = tmp_path / "out"
    argv_build = [
        "funda",
        "build",
        "--dataset-root",
        str(tmp_path),
        "--kinds",
        "annual,quarterly",
        "--out-format",
        "csv",
        "--out-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv_build)
    appmod.main()
    assert (out_dir / "csv" / "income_annual.csv").exists()
