import sys

import pytest
import yaml

from tushare_a_fundamentals import cli as appmod

pytestmark = pytest.mark.integration


def test_cli_overrides_config(tmp_path, monkeypatch):
    cfg = {"mode": "annual", "years": 1}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    called = {}

    def fake_run_bulk(pro, cfg, mode, fields, fmt, outdir, prefix):
        called["mode"] = mode
        called["quarters"] = cfg.get("quarters")
        called["years"] = cfg.get("years")

    monkeypatch.setattr(appmod, "init_pro_api", lambda token: object())
    monkeypatch.setattr(appmod, "_run_bulk_mode", fake_run_bulk)

    argv = [
        "funda",
        "--config",
        str(cfg_path),
        "--mode",
        "quarterly",
        "--quarters",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    appmod.main()
    assert called["mode"] == appmod.Mode.QUARTERLY
    assert called["quarters"] == 1
    assert called["years"] == 1
