import json
from argparse import Namespace
from pathlib import Path

import pytest

from tushare_a_fundamentals.commands import state as state_cmd

pytestmark = pytest.mark.unit


def make_args(**kwargs):
    defaults = dict(
        action="show",
        backend="auto",
        state_path=None,
        data_dir="data",
        dataset=None,
        year=None,
        key=None,
        value=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_state_show_json(tmp_path, capsys):
    data_dir = tmp_path / "data"
    state_file = data_dir / "_state" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(json.dumps({"income": {"last_period": "20231231"}}), "utf-8")

    args = make_args(action="show", backend="json", state_path=None, data_dir=str(data_dir))
    state_cmd.cmd_state(args)

    captured = capsys.readouterr()
    assert "20231231" in captured.out


def test_state_clear_json(tmp_path):
    data_dir = tmp_path / "data"
    state_file = data_dir / "_state" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(json.dumps({"income": {"foo": "bar"}}), "utf-8")

    args = make_args(
        action="clear",
        backend="json",
        state_path=None,
        data_dir=str(data_dir),
        dataset="income",
        key="foo",
    )
    state_cmd.cmd_state(args)

    payload = json.loads(state_file.read_text("utf-8"))
    assert payload == {}


def test_state_ls_failures(tmp_path, capsys):
    data_dir = tmp_path / "data"
    failure_dir = data_dir / "_state" / "failures"
    failure_dir.mkdir(parents=True)
    sample = {
        "dataset": "income",
        "entries": [{"period": "20231231"}],
    }
    (failure_dir / "income_periods.json").write_text(json.dumps(sample), "utf-8")

    args = make_args(action="ls-failures", data_dir=str(data_dir))
    state_cmd.cmd_state(args)

    captured = capsys.readouterr()
    assert "income_periods.json" in captured.out
    assert "1 条记录" in captured.out


def test_state_set_sqlite(tmp_path, capsys):
    db_path = tmp_path / "state.db"
    args_set = make_args(
        action="set",
        backend="sqlite",
        state_path=str(db_path),
        dataset="income",
        key="last_period",
        value="20231231",
    )
    state_cmd.cmd_state(args_set)
    capsys.readouterr()

    args_show = make_args(
        action="show",
        backend="sqlite",
        state_path=str(db_path),
    )
    state_cmd.cmd_state(args_show)
    captured = capsys.readouterr()
    assert "kv_state" in captured.out
    assert "20231231" in captured.out


def test_state_backend_auto_prefers_sqlite(monkeypatch, tmp_path):
    repo_root = tmp_path
    (repo_root / "meta").mkdir()
    db_path = repo_root / "meta" / "state.db"
    db_path.write_bytes(b"")
    data_dir = repo_root / "data"
    data_dir.mkdir()

    args = make_args(data_dir=str(data_dir))
    monkeypatch.chdir(repo_root)

    backend, path = state_cmd._resolve_backend_and_path(args)

    assert backend == "sqlite"
    assert path == Path("meta") / "state.db"


def test_state_backend_auto_defaults_to_json(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    args = make_args(data_dir=str(data_dir))

    monkeypatch.chdir(tmp_path)

    backend, path = state_cmd._resolve_backend_and_path(args)

    assert backend == "json"
    assert path == data_dir / "_state" / "state.json"
