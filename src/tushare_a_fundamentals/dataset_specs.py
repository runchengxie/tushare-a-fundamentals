"""Dataset specifications shared by downloader and related workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .meta.doc_fields import DOC_FIELDS


@dataclass(frozen=True)
class DatasetSpec:
    """Describe how a TuShare dataset should be fetched and stored."""

    name: str
    api: str
    vip_api: Optional[str] = None
    period_field: Optional[str] = None
    date_field: Optional[str] = None
    date_start_param: str = "start_date"
    date_end_param: str = "end_date"
    primary_keys: Sequence[str] = field(default_factory=tuple)
    dedup_group_keys: Sequence[str] = field(default_factory=tuple)
    default_year_column: str = "ann_date"
    default_start: str = "20000101"
    fields: Optional[str] = None
    report_types: Sequence[int] = field(default_factory=tuple)
    type_param: Optional[str] = None
    type_values: Sequence[str] = field(default_factory=tuple)
    vip_supports_pagination: bool = False
    api_supports_pagination: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    requires_ts_code: bool = False
    code_param: str = "ts_code"


def _fields_for(name: str) -> Optional[str]:
    fields = DOC_FIELDS.get(name)
    if not fields:
        return None
    return ",".join(fields)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "income": DatasetSpec(
        name="income",
        api="income",
        vip_api="income_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        report_types=(1,),
        fields=_fields_for("income"),
        vip_supports_pagination=True,
    ),
    "balancesheet": DatasetSpec(
        name="balancesheet",
        api="balancesheet",
        vip_api="balancesheet_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("balancesheet"),
        vip_supports_pagination=True,
    ),
    "cashflow": DatasetSpec(
        name="cashflow",
        api="cashflow",
        vip_api="cashflow_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "report_type"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("cashflow"),
        vip_supports_pagination=True,
    ),
    "forecast": DatasetSpec(
        name="forecast",
        api="forecast",
        vip_api="forecast_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "type"),
        dedup_group_keys=("ts_code", "end_date", "type"),
        default_year_column="end_date",
        fields=_fields_for("forecast"),
        vip_supports_pagination=True,
    ),
    "express": DatasetSpec(
        name="express",
        api="express",
        vip_api="express_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("express"),
        vip_supports_pagination=True,
    ),
    "dividend": DatasetSpec(
        name="dividend",
        api="dividend",
        vip_api=None,
        period_field=None,
        date_field="ann_date",
        primary_keys=(
            "ts_code",
            "ann_date",
            "record_date",
            "ex_date",
            "imp_ann_date",
        ),
        dedup_group_keys=(
            "ts_code",
            "ann_date",
            "record_date",
            "ex_date",
            "imp_ann_date",
        ),
        default_year_column="ann_date",
        fields=_fields_for("dividend"),
    ),
    "fina_indicator": DatasetSpec(
        name="fina_indicator",
        api="fina_indicator",
        vip_api="fina_indicator_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("fina_indicator"),
        vip_supports_pagination=True,
    ),
    "fina_audit": DatasetSpec(
        name="fina_audit",
        api="fina_audit",
        vip_api=None,
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date"),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("fina_audit"),
        api_supports_pagination=True,
        requires_ts_code=True,
        code_param="ts_code",
    ),
    "fina_mainbz": DatasetSpec(
        name="fina_mainbz",
        api="fina_mainbz",
        vip_api="fina_mainbz_vip",
        period_field="period",
        date_field=None,
        primary_keys=("ts_code", "end_date", "bz_item", "type"),
        dedup_group_keys=("ts_code", "end_date", "bz_item", "type"),
        default_year_column="end_date",
        type_param="type",
        type_values=("P", "D"),
        fields=_fields_for("fina_mainbz"),
        vip_supports_pagination=True,
    ),
    "disclosure_date": DatasetSpec(
        name="disclosure_date",
        api="disclosure_date",
        vip_api=None,
        period_field="end_date",
        date_field=None,
        primary_keys=(
            "ts_code",
            "end_date",
            "ann_date",
            "pre_date",
            "actual_date",
        ),
        dedup_group_keys=("ts_code", "end_date"),
        default_year_column="end_date",
        fields=_fields_for("disclosure_date"),
        api_supports_pagination=True,
    ),
}


__all__ = ["DatasetSpec", "DATASET_SPECS"]
