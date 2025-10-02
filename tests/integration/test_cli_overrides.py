import sys

import pytest

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.integration


def test_cli_requires_subcommand(monkeypatch):
    monkeypatch.setattr(appmod, "init_pro_api", lambda token: object())
    monkeypatch.setattr(sys, "argv", ["funda"])
    with pytest.raises(SystemExit) as excinfo:
        appmod.main()
    assert excinfo.value.code == 2
