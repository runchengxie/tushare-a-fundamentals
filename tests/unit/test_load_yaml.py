import os

import pytest

from tushare_a_fundamentals.common import load_yaml

pytestmark = pytest.mark.unit


def test_load_yaml_prefers_yml(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("foo: bar\n", encoding="utf-8")
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    data = load_yaml(None)

    assert data == {"foo": "bar"}


def test_load_yaml_supports_yaml(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("answer: 42\n", encoding="utf-8")
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    data = load_yaml(None)

    assert data == {"answer": 42}


def test_load_yaml_conflicting_files(monkeypatch, tmp_path):
    (tmp_path / "config.yml").write_text("foo: 1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("foo: 2\n", encoding="utf-8")
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    with pytest.raises(SystemExit):
        load_yaml(None)


def test_load_yaml_missing_files_returns_defaults(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    data = load_yaml(None)

    captured = capsys.readouterr()
    assert data == {}
    assert "提示：未检测到 config" in captured.out
