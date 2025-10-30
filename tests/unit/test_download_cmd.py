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


def test_cmd_download_audit_only_prefers_audit_quarters(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["data_dir"] = data_dir
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests
            captured["start"] = start
            captured["end"] = end
            captured["refresh"] = refresh_periods

    def fake_periods_from_cfg(cfg):
        captured["quarters_cfg"] = cfg.get("quarters")
        captured["years_cfg"] = cfg.get("years")
        return ["20240101", "20240331"]

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    monkeypatch.setattr(download_cmd, "_periods_from_cfg", fake_periods_from_cfg)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(
        download_cmd,
        "load_yaml",
        lambda path: {"years": 10, "audit_quarters": 1},
    )

    args = Namespace(
        config=None,
        datasets=None,
        years=None,
        quarters=None,
        since=None,
        until=None,
        audit_quarters=None,
        audit_years=None,
        audit_only=True,
        with_audit=False,
        all=False,
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

    assert captured["quarters_cfg"] == 1
    assert captured["years_cfg"] is None
    assert captured["requests"] and captured["requests"][0].name == "fina_audit"
    assert captured["start"] == "20240101"
    assert captured["end"] == "20240331"
    assert captured["init_kwargs"]["max_retries"] == 5


def test_run_export_strict_propagates_system_exit(monkeypatch):
    args = Namespace()

    def boom(_: Namespace) -> None:
        raise SystemExit(2)

    monkeypatch.setattr(download_cmd, "cmd_export", boom)

    with pytest.raises(SystemExit) as excinfo:
        download_cmd._run_export(args, True)

    assert excinfo.value.code == 2


def test_cmd_download_audit_only_falls_back_to_one_quarter(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["data_dir"] = data_dir
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests
            captured["start"] = start
            captured["end"] = end
            captured["refresh"] = refresh_periods

    def fake_periods_from_cfg(cfg):
        captured["quarters_cfg"] = cfg.get("quarters")
        captured["years_cfg"] = cfg.get("years")
        return ["20240101", "20240331"]

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    monkeypatch.setattr(download_cmd, "_periods_from_cfg", fake_periods_from_cfg)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(download_cmd, "load_yaml", lambda path: None)

    args = Namespace(
        config=None,
        datasets=None,
        years=None,
        quarters=None,
        since=None,
        until=None,
        audit_quarters=None,
        audit_years=None,
        audit_only=True,
        with_audit=False,
        all=False,
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

    assert captured["quarters_cfg"] == 1
    assert captured["years_cfg"] is None
    assert captured["requests"] and captured["requests"][0].name == "fina_audit"
    assert captured["start"] == "20240101"
    assert captured["end"] == "20240331"
    assert captured["init_kwargs"]["max_retries"] == 5


def test_cmd_download_audit_only_respects_explicit_max_retries(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(
        download_cmd,
        "load_yaml",
        lambda path: {"max_retries": 2},
    )

    args = Namespace(
        config=None,
        datasets=None,
        years=None,
        quarters=None,
        since=None,
        until=None,
        audit_quarters=None,
        audit_years=None,
        audit_only=True,
        with_audit=False,
        all=False,
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

    assert captured["init_kwargs"]["max_retries"] == 2
    assert captured["requests"] and captured["requests"][0].name == "fina_audit"


def test_cmd_download_default_skips_dividend(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(download_cmd, "load_yaml", lambda path: None)

    args = Namespace(
        config=None,
        datasets=None,
        years=None,
        quarters=None,
        since=None,
        until=None,
        audit_quarters=None,
        audit_years=None,
        audit_only=False,
        with_audit=False,
        all=False,
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
        dividend_only=False,
    )

    download_cmd.cmd_download(args)

    requested = {req.name for req in captured.get("requests", [])}
    assert download_cmd.DIVIDEND_DATASET_NAME not in requested
    assert "income" in requested


def test_cmd_download_dividend_only(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyDownloader:
        def __init__(self, pro, data_dir, *, vip_pro=None, **kwargs):
            captured["init_kwargs"] = kwargs

        def run(self, requests, *, start=None, end=None, refresh_periods=0):
            captured["requests"] = requests

    monkeypatch.setattr(download_cmd, "MarketDatasetDownloader", DummyDownloader)
    dummy_ctx = ProContext(
        any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"]
    )
    monkeypatch.setattr(download_cmd, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(download_cmd, "ensure_enough_credits", lambda pro, required=5000: None)
    monkeypatch.setattr(download_cmd, "load_yaml", lambda path: None)

    args = Namespace(
        config=None,
        datasets=None,
        years=None,
        quarters=None,
        since=None,
        until=None,
        audit_quarters=None,
        audit_years=None,
        audit_only=False,
        with_audit=False,
        all=False,
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
        dividend_only=True,
    )

    download_cmd.cmd_download(args)

    requested = [req.name for req in captured.get("requests", [])]
    assert requested == [download_cmd.DIVIDEND_DATASET_NAME]
