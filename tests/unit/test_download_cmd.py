from argparse import Namespace

import pytest

import tushare_a_fundamentals.commands.download as download_cmd
from tushare_a_fundamentals.common import ProContext

pytestmark = pytest.mark.unit


def test_cmd_download_multi_dataset_uses_configured_years(monkeypatch, tmp_path):
    captured = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["pro"] = pro
            captured["vip_pro"] = vip_pro
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests
            captured["start"] = start
            captured["end"] = end
            captured["refresh"] = refresh_periods

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
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
        progress="plain",
    )

    download_cmd.cmd_download(args)

    assert captured["requests"] and captured["requests"][0].name == "income"
    assert captured["start"] is not None
    assert captured["end"] is not None
    assert len(captured["start"]) == 8
    assert len(captured["end"]) == 8
    assert captured["start"] <= captured["end"]
    assert captured["refresh"] == 4
    assert captured["init_kwargs"]["progress_mode"] == "plain"


def test_cmd_download_invalid_progress_falls_back(monkeypatch, tmp_path):
    captured = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["progress_mode"] = kwargs.get("progress_mode")

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            pass

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
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
        progress="???",
    )

    download_cmd.cmd_download(args)

    assert captured["progress_mode"] == "auto"
