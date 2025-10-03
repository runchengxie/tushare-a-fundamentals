from argparse import Namespace

import pytest

import tushare_a_fundamentals.commands.download as download_cmd
from tushare_a_fundamentals.common import ProContext

pytestmark = pytest.mark.unit


def test_cmd_download_runs_export_when_enabled(monkeypatch, tmp_path):
    captured = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["data_dir"] = data_dir

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)

    dummy_ctx = ProContext(
        any_client=object(),
        vip_client=object(),
        tokens=["tok"],
        vip_tokens=["tok"],
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(download_cmd, "load_yaml", lambda path: {})

    export_args = {}

    def fake_export(ns):
        export_args["args"] = ns

    monkeypatch.setattr(download_cmd, "cmd_export", fake_export)

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
        export_enabled=True,
        no_export=False,
        max_retries=None,
    )

    download_cmd.cmd_download(args)

    assert "args" in export_args
    ns = export_args["args"]
    assert ns.dataset_root == str(tmp_path)
    assert ns.out_format == "csv"
    assert ns.out_dir == str(tmp_path)
    assert ns.kinds == ""
    assert ns.flat_datasets == "auto"
    assert ns.flat_exclude == ""
    assert ns.split_by == "none"
    assert ns.gzip is False
    assert ns.no_income is False
    assert ns.no_flat is False
    assert ns.progress == "auto"
