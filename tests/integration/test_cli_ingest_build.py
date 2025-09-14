import sys

import pandas as pd
import pytest

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.integration


def test_cli_download_build_and_coverage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(appmod, "init_pro_api", lambda token: object())
    monkeypatch.setattr(appmod, "_has_enough_credits", lambda pro: True)

    def fake_fetch_income_bulk(pro, periods, mode, fields):
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "end_date": ["20231231"],
                "ann_date": ["20240101"],
                "f_ann_date": ["20240102"],
                "report_type": [1],
                "revenue": [100],
            }
        )
        return {"raw": df.copy()}

    monkeypatch.setattr(appmod, "fetch_income_bulk", fake_fetch_income_bulk)

    argv_dl = [
        "funda",
        "download",
        "--since",
        "2023-01-01",
        "--until",
        "2023-12-31",
        "--dataset-root",
        str(tmp_path),
        "--token",
        "fake",
    ]
    monkeypatch.setattr(sys, "argv", argv_dl)
    appmod.main()
    assert list((tmp_path / "dataset=fact_income_single").glob("**/*.parquet"))

    capsys.readouterr()
    argv_cov = [
        "funda",
        "coverage",
        "--dataset-root",
        str(tmp_path),
        "--by",
        "ticker",
    ]
    monkeypatch.setattr(sys, "argv", argv_cov)
    appmod.main()
    cov_out = capsys.readouterr().out
    assert "000001.SZ" in cov_out

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
    annual_csv = out_dir / "csv" / "income_annual.csv"
    assert annual_csv.exists()
    df = pd.read_csv(annual_csv)
    assert "ticker" in df.columns
