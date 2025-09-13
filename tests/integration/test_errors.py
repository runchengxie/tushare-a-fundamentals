import os
import sys
import subprocess
import pytest

pytestmark = pytest.mark.integration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_cli(*args):
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    return subprocess.run(
        [sys.executable, "-m", "tushare_a_fundamentals.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_error_no_token_env(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    p = run_cli("--mode", "annual", "--years", "1", "--vip")
    assert p.returncode == 2
    assert "缺少 TuShare token" in p.stderr


def test_error_no_vip_no_ts_code(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy")
    p = run_cli("--mode", "annual", "--years", "1", "--no-vip")
    assert p.returncode == 2
    assert "未提供 --ts-code 且未启用 --vip" in p.stderr
