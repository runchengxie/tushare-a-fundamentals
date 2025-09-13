import os
import sys
import yaml
import subprocess
import pytest

pytestmark = pytest.mark.integration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}


def test_cli_overrides_config(tmp_path):
    cfg = {"mode": "annual", "years": 1, "vip": True}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code = subprocess.run(
        [
            sys.executable,
            "-m",
            "tushare_a_fundamentals.cli",
            "--config",
            str(cfg_path),
            "--mode",
            "quarterly",
            "--quarters",
            "1",
            "--vip",
        ],
        capture_output=True,
        text=True,
        env=ENV,
    )
    assert code.returncode != 2
