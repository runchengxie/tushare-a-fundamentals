import argparse

import pandas as pd
import pytest

from tushare_a_fundamentals.commands import export as export_cmd


def test_cmd_export_exits_when_no_data(monkeypatch, tmp_path):
    empty = pd.DataFrame({"ts_code": [], "end_date": []})
    monkeypatch.setattr(export_cmd, "_load_dataset", lambda *args, **kwargs: empty)
    monkeypatch.setattr(export_cmd, "_export_tables", lambda *args, **kwargs: pytest.fail("should not export"))

    args = argparse.Namespace(
        dataset_root=str(tmp_path),
        kinds="",
        out_format="csv",
        out_dir=str(tmp_path),
        prefix="income",
        annual_strategy="cumulative",
        years=1,
    )

    with pytest.raises(SystemExit) as exc:
        export_cmd.cmd_export(args)

    assert exc.value.code == 2
