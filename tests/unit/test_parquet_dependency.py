import importlib
import pytest
from tushare_a_fundamentals.cli import _check_parquet_dependency


def test_parquet_dependency_present():
    pytest.importorskip("pyarrow")
    assert _check_parquet_dependency() is True


def test_parquet_dependency_missing(monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError

    monkeypatch.setattr(importlib, "import_module", lambda name: fake_import(name))
    assert _check_parquet_dependency() is False
