import sys
from types import SimpleNamespace

import pytest

from tushare_a_fundamentals import cli as appmod
from tushare_a_fundamentals.common import ProContext

pytestmark = pytest.mark.integration


def test_cli_requires_subcommand(monkeypatch):
    dummy_ctx = ProContext(any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"])
    monkeypatch.setattr(appmod, "init_pro_api", lambda token: dummy_ctx)
    monkeypatch.setattr(sys, "argv", ["funda"])
    with pytest.raises(SystemExit) as excinfo:
        appmod.main()
    assert excinfo.value.code == 2


def test_cli_falls_back_to_download(monkeypatch):
    dummy_ctx = ProContext(any_client=object(), vip_client=object(), tokens=["tok"], vip_tokens=["tok"])
    monkeypatch.setattr(appmod, "init_pro_api", lambda token: dummy_ctx)
    args = SimpleNamespace(cmd=None, token=None)
    monkeypatch.setattr(appmod, "parse_cli", lambda: args)
    monkeypatch.setattr(sys, "argv", ["funda", "--token", "foo"])

    called: dict[str, SimpleNamespace] = {}

    def fake_download(received):
        called["args"] = received

    monkeypatch.setattr("tushare_a_fundamentals.commands.download.cmd_download", fake_download)

    appmod.main()

    assert called["args"] is args


def test_cli_unknown_command_reports_error(monkeypatch):
    args = SimpleNamespace(cmd="mystery")
    monkeypatch.setattr(appmod, "parse_cli", lambda: args)
    monkeypatch.setattr(sys, "argv", ["funda", "mystery"])

    captured: list[str] = []

    def fake_eprint(msg: str) -> None:
        captured.append(msg)

    monkeypatch.setattr(appmod, "eprint", fake_eprint)

    with pytest.raises(SystemExit) as excinfo:
        appmod.main()

    assert excinfo.value.code == 2
    assert captured and "错误：请使用子命令" in captured[0]
