import sys

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
