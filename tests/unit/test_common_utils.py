import importlib.util
import os
import sys

import pandas as pd
import pytest

import tushare_a_fundamentals.common as common
from tushare_a_fundamentals.common import ensure_ts_code, merge_config


def test_merge_config_applies_priorities():
    defaults = {"years": 10, "use_vip": True, "fields": None}
    cfg = {"years": 5, "use_vip": False}
    cli = {"use_vip": True}

    merged = merge_config(cli, cfg, defaults)

    assert merged["years"] == 5
    assert merged["use_vip"] is True


def test_merge_config_ignores_none_and_empty_strings():
    defaults = {"fields": "a,b", "token": None}
    cfg = {"fields": "a,b"}
    cli = {"fields": "", "token": None}

    merged = merge_config(cli, cfg, defaults)

    assert merged["fields"] == "a,b"
    assert merged.get("token") is None


def test_ensure_ts_code_passthrough():
    df = pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20231231"]})

    got = ensure_ts_code(df)

    assert list(got.columns)[0] == "ts_code"
    assert got.loc[0, "ts_code"] == "000001.SZ"


def test_ensure_ts_code_accepts_ticker_column():
    df = pd.DataFrame({"ticker": ["000002.SZ"], "end_date": ["20230630"]})

    got = ensure_ts_code(df)

    assert "ts_code" in got.columns
    assert got.loc[0, "ts_code"] == "000002.SZ"


def test_ensure_ts_code_missing_column():
    df = pd.DataFrame({"end_date": ["20231231"]})

    with pytest.raises(KeyError):
        ensure_ts_code(df, context="unit-test")


def test_legacy_api_key_env_mapping(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("TUSHARE_API_KEY", "legacy-token")

    spec = importlib.util.spec_from_file_location(
        "tushare_a_fundamentals.common_legacy", common.__file__
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert os.getenv("TUSHARE_TOKEN") == "legacy-token"
    finally:
        sys.modules.pop(spec.name, None)
