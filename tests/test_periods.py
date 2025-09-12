import os
import sys
import types
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import app as appmod


def test_periods_years_annual():
    periods = appmod.periods_for_mode_by_years(2, appmod.Mode.ANNUAL)
    assert all(p.endswith("1231") for p in periods)
    assert len(periods) == 2


def test_periods_quarters_count():
    periods = appmod.periods_by_quarters(6)
    assert len(periods) == 6
    assert periods == sorted(periods)
