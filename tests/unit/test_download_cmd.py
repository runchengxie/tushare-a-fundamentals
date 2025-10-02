from argparse import Namespace

import pytest

import tushare_a_fundamentals.commands.download as download_cmd

pytestmark = pytest.mark.unit


def test_cmd_download_multi_dataset_uses_configured_years(monkeypatch, tmp_path):
    captured = {}

    class DummyDownloader:
        def __init__(self, *args, **kwargs):
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests
            captured["start"] = start
            captured["end"] = end
            captured["refresh"] = refresh_periods

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: object())
    monkeypatch.setattr(download_cmd, "load_yaml", lambda path: {})

    args = Namespace(
        config=None,
        datasets=["income"],
        years=None,
        quarters=None,
        since=None,
        until=None,
        fields="",
        outdir=None,
        prefix=None,
        format=None,
        token=None,
        report_types=None,
        allow_future=False,
        recent_quarters=None,
        data_dir=str(tmp_path),
        use_vip=None,
        max_per_minute=None,
        state_path=None,
        export_out_dir=None,
        export_out_format=None,
        export_kinds=None,
        export_annual_strategy=None,
        export_years=None,
        export_strict=None,
        export_enabled=None,
        no_export=True,
        max_retries=None,
    )

    download_cmd.cmd_download(args)

    assert captured["requests"] and captured["requests"][0].name == "income"
    assert captured["start"] is not None
    assert captured["end"] is not None
    assert len(captured["start"]) == 8
    assert len(captured["end"]) == 8
    assert captured["start"] <= captured["end"]
    assert captured["refresh"] == 4
