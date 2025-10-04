from __future__ import annotations

import pytest

pytest.importorskip("rich")

import tushare_a_fundamentals.progress as progress
from tushare_a_fundamentals.progress import PlainTicker, ProgressManager

pytestmark = pytest.mark.unit


def test_progress_manager_none_mode():
    pm = ProgressManager("none")
    assert pm.is_active is False
    task = pm.add_task("任务", 5)
    assert task is None
    with pm.live() as live_pm:
        assert live_pm is pm
    # advance should be a no-op
    pm.advance(task, ok=1, fail=0)


def test_progress_manager_plain_outputs(monkeypatch, capsys):
    pm = ProgressManager("plain")
    task = pm.add_task("下载", 3)
    assert isinstance(task, PlainTicker)

    monkeypatch.setattr(progress.time, "time", lambda: 2.0)

    pm.advance(task, ok=1)
    captured = capsys.readouterr().out
    assert "下载" in captured
    assert "1/3" in captured
    assert "✓1" in captured


def test_progress_manager_auto_prefers_rich_when_tty(monkeypatch):
    monkeypatch.setattr(progress, "_is_tty", lambda stream: True)
    pm = ProgressManager("auto")
    assert pm.progress is not None
    assert pm.console is not None


def test_progress_manager_auto_falls_back_to_plain(monkeypatch, capsys):
    monkeypatch.setattr(progress, "_is_tty", lambda stream: False)
    monkeypatch.setattr(progress.time, "time", lambda: 10.0)

    pm = ProgressManager("auto")

    task = pm.add_task("下载", 2)
    assert isinstance(task, PlainTicker)

    pm.advance(task, ok=2)

    captured = capsys.readouterr().out
    assert "下载" in captured
    assert "2/2" in captured
