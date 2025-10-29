import pytest

from tushare_a_fundamentals.dataset_specs import DATASET_SPECS
from tushare_a_fundamentals.meta.doc_fields import DOC_FIELDS


pytestmark = pytest.mark.unit


def test_dataset_specs_use_doc_fields():
    for dataset, fields in DOC_FIELDS.items():
        assert dataset in DATASET_SPECS, f"{dataset} missing in DATASET_SPECS"
        spec = DATASET_SPECS[dataset]
        assert spec.fields == ",".join(fields)


def test_income_fields_include_continued_net_profit():
    income_fields = DOC_FIELDS["income"]
    assert "continued_net_profit" in income_fields

