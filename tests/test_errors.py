import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "src", "app.py")


def test_error_no_token_env(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    p = subprocess.run(["python3", APP, "--mode", "annual", "--years", "1", "--vip"], capture_output=True, text=True)
    assert p.returncode == 2
    assert "缺少 TuShare token" in p.stderr


def test_error_no_vip_no_ts_code(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy")
    p = subprocess.run(["python3", APP, "--mode", "annual", "--years", "1", "--no-vip"], capture_output=True, text=True)
    assert p.returncode == 2
    assert "未提供 --ts-code 且未启用 --vip" in p.stderr
