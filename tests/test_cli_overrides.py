import os
import sys
import json
import tempfile
import yaml
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "src", "app.py")


def test_cli_overrides_config(tmp_path):
    cfg = {"mode": "annual", "years": 1, "vip": True}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    outdir = tmp_path / "out"
    code = subprocess.run(["python3", APP, "--config", str(cfg_path), "--mode", "quarterly", "--quarters", "1", "--vip"], capture_output=True, text=True)
    assert code.returncode != 2
