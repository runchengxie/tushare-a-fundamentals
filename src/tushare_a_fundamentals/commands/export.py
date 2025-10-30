from __future__ import annotations

import argparse
import gzip
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.types as patypes

from ..common import (
    _concat_non_empty,
    _export_tables,
    build_income_export_tables,
    ensure_ts_code,
    eprint,
)
from ..progress import ProgressManager


@dataclass
class ExportOptions:
    dataset_root: Path
    out_dir: Path
    out_format: str
    years: int | None
    kinds: list[str]
    annual_strategy: str
    include_income: bool
    include_flat: bool
    flat_datasets: list[str] | None
    flat_exclude: set[str]
    split_by: str
    gzip: bool
    prefix: str

    @property
    def out_dir_str(self) -> str:
        return str(self.out_dir)

    @property
    def flat_target_dir(self) -> Path:
        fmt = self.out_format.lower()
        base_name = self.out_dir.name.lower()
        if fmt == "csv":
            return self.out_dir if base_name == "csv" else self.out_dir / "csv"
        if fmt == "parquet":
            return self.out_dir if base_name == "parquet" else self.out_dir / "parquet"
        return self.out_dir


def cmd_export(args: argparse.Namespace) -> None:
    opts = _options_from_args(args)
    progress = ProgressManager(getattr(args, "progress", "auto"))
    if not opts.include_income and not opts.include_flat:
        eprint("错误：未选择任何导出目标（--no-income 与 --no-flat 同时启用）")
        sys.exit(2)
    if opts.gzip and opts.out_format != "csv":
        eprint("错误：--gzip 仅支持 csv 输出")
        sys.exit(2)

    exported_any = False
    with progress.live():
        if opts.include_income:
            exported_any = _export_income(opts) or exported_any
        if opts.include_flat:
            flat_count = _export_flat_datasets(opts, progress)
            exported_any = flat_count > 0 or exported_any

    if not exported_any:
        eprint("提示：未执行任何导出任务，请检查数据目录或参数配置。")


def _options_from_args(args: argparse.Namespace) -> ExportOptions:
    dataset_root = Path(args.dataset_root or "data")
    out_dir = Path(args.out_dir or "data")
    out_format = (getattr(args, "out_format", "csv") or "csv").lower()

    raw_kinds = getattr(args, "kinds", "") or ""
    kinds = [part.strip() for part in raw_kinds.split(",") if part.strip()]

    try:
        years_val = getattr(args, "years", None)
        years = None if years_val is None else int(years_val)
    except (TypeError, ValueError):
        years = None

    split_by = getattr(args, "split_by", "none") or "none"
    gzip_enabled = bool(getattr(args, "gzip", False))

    flat_raw = getattr(args, "flat_datasets", "auto") or "auto"
    flat_key = flat_raw.strip().lower()
    if flat_key in {"auto", "all"}:
        flat_datasets: list[str] | None = None
    elif flat_key in {"none", "skip"}:
        flat_datasets = []
    else:
        flat_datasets = [part.strip() for part in flat_raw.split(",") if part.strip()]

    raw_exclude = getattr(args, "flat_exclude", "") or ""
    flat_exclude = {part.strip().lower() for part in raw_exclude.split(",") if part.strip()}

    include_income = not bool(getattr(args, "no_income", False)) and bool(kinds)
    include_flat = not bool(getattr(args, "no_flat", False))
    if not include_flat:
        flat_datasets = flat_datasets if flat_datasets is not None else []

    return ExportOptions(
        dataset_root=dataset_root,
        out_dir=out_dir,
        out_format=out_format,
        years=years,
        kinds=kinds,
        annual_strategy=getattr(args, "annual_strategy", "cumulative"),
        include_income=include_income,
        include_flat=include_flat,
        flat_datasets=flat_datasets,
        flat_exclude=flat_exclude,
        split_by=split_by,
        gzip=gzip_enabled,
        prefix=getattr(args, "prefix", "income"),
    )


def _export_income(opts: ExportOptions) -> bool:
    income_df = _load_dataset_as_frame(opts.dataset_root, "income")
    if income_df.empty:
        income_df = _load_dataset_as_frame(opts.dataset_root, "dataset=fact_income_cum")
    if income_df.empty:
        eprint("提示：未找到 income 数据，跳过 income 导出")
        return False

    built = build_income_export_tables(
        income_df,
        years=opts.years,
        kinds=opts.kinds,
        annual_strategy=opts.annual_strategy,
    )
    if not built:
        eprint("提示：未生成可导出的 income 数据，跳过 income 导出")
        return False

    _export_tables(built, opts.out_dir_str, opts.prefix, opts.out_format)
    return True


def _load_dataset_as_frame(root: Path, name: str) -> pd.DataFrame:
    base = root / name
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception as exc:  # pragma: no cover - defensive I/O
            eprint(f"警告：读取 {file} 失败：{exc}")
    if not frames:
        return pd.DataFrame()
    combined = _concat_non_empty(frames)
    if combined.empty:
        return combined
    return ensure_ts_code(combined, context=name)


def _export_flat_datasets(
    opts: ExportOptions, progress: Optional[ProgressManager] = None
) -> int:
    datasets = opts.flat_datasets
    if datasets is None:
        datasets = _discover_datasets(opts.dataset_root)
    datasets = _unique_preserve_order(datasets)
    datasets = [name for name in datasets if name and name.lower() not in opts.flat_exclude]
    if not datasets:
        return 0

    total_exports = 0
    task = progress.add_task(f"导出平面数据集（共 {len(datasets)} 个）", len(datasets)) if progress else None
    for name in datasets:
        total_exports += _export_single_dataset(opts, name, progress)
        if progress is not None:
            progress.advance(task, 1)
    return total_exports


def _discover_datasets(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue
        if "=" in child.name:
            continue
        if not _contains_parquet(child):
            continue
        names.append(child.name)
    return names


def _contains_parquet(path: Path) -> bool:
    for _ in path.rglob("*.parquet"):
        return True
    return False


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _export_single_dataset(
    opts: ExportOptions,
    name: str,
    progress: Optional[ProgressManager] = None,
) -> int:
    source = opts.dataset_root / name
    if not source.exists():
        eprint(f"提示：跳过 {name}，目录不存在：{source}")
        return 0

    try:
        if opts.split_by == "year":
            partitions = sorted(p for p in source.glob("year=*") if p.is_dir())
            if not partitions:
                return int(_write_dataset(source, opts, name, progress))
            written = 0
            for part in partitions:
                year = part.name.split("=", 1)[1] if "=" in part.name else part.name
                base = f"{name}_{year or 'unknown'}"
                if _write_dataset(part, opts, base, progress):
                    written += 1
            if written == 0:
                eprint(f"提示：{name} 的年度分区无可导出数据，已跳过")
            return written

        return int(_write_dataset(source, opts, name, progress))
    except (OSError, ValueError) as exc:
        eprint(f"警告：读取 {source} 失败：{exc}")
        return 0


def _write_dataset(
    source: Path,
    opts: ExportOptions,
    base_name: str,
    progress: Optional[ProgressManager] = None,
) -> bool:
    try:
        dataset = _build_dataset(source)
    except (FileNotFoundError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError):
        return False

    scanner = ds.Scanner.from_dataset(dataset)
    batches = scanner.to_batches()
    try:
        first_batch = next(batches)
    except StopIteration:
        return False

    out_path = _build_output_path(opts.flat_target_dir, base_name, opts.out_format, opts.gzip)
    sink = None
    writer: pq.ParquetWriter | None = None
    first = True
    chain = itertools.chain([first_batch], batches)
    try:
        for batch in chain:
            table = pa.Table.from_batches([batch])
            table = _cast_table(table)
            if opts.out_format == "csv":
                if first:
                    sink = _open_csv_sink(out_path, opts.gzip)
                pcsv.write_csv(
                    table,
                    sink,
                    write_options=pcsv.WriteOptions(include_header=first),
                )
            else:
                if first:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(out_path.as_posix(), table.schema)
                assert writer is not None
                writer.write_table(table)
            first = False
    finally:
        if sink is not None:
            sink.close()
        if writer is not None:
            writer.close()

    _log_progress(progress, f"已保存：{out_path}")
    return True


def _log_progress(progress: Optional[ProgressManager], message: str) -> None:
    if progress is not None and progress.is_active:
        progress.log(message)
    else:
        print(message)


def _build_dataset(source: Path) -> ds.Dataset:
    partitioning = ds.partitioning(
        pa.schema([pa.field("year", pa.string())]), flavor="hive"
    )
    dataset = ds.dataset(
        source.as_posix(), format="parquet", partitioning=partitioning
    )
    schema = dataset.schema
    adjusted_schema = _resolve_null_fields(dataset, schema)
    if adjusted_schema is not schema:
        dataset = ds.dataset(
            source.as_posix(),
            format="parquet",
            partitioning=partitioning,
            schema=adjusted_schema,
        )
    return dataset


def _resolve_null_fields(dataset: ds.Dataset, schema: pa.Schema) -> pa.Schema:
    replacements: dict[str, pa.DataType] = {}
    null_field_names = [
        field.name for field in schema if patypes.is_null(field.type)
    ]
    if not null_field_names:
        return schema

    for fragment in dataset.get_fragments():
        fragment_schema = getattr(fragment, "physical_schema", None)
        if fragment_schema is None:
            fragment_schema = fragment.schema
        for name in list(null_field_names):
            idx = fragment_schema.get_field_index(name)
            if idx == -1:
                continue
            fragment_field = fragment_schema.field(idx)
            if patypes.is_null(fragment_field.type):
                continue
            replacements[name] = fragment_field.type
            null_field_names.remove(name)
        if not null_field_names:
            break

    if not replacements:
        return schema

    new_fields = [
        field.with_type(replacements.get(field.name, field.type))
        for field in schema
    ]
    return pa.schema(new_fields, metadata=schema.metadata)


def _build_output_path(target_dir: Path, base: str, fmt: str, gzip_enabled: bool) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = fmt.lower()
    filename = f"{base}.{suffix}"
    if fmt == "csv" and gzip_enabled:
        filename = f"{filename}.gz"
    return target_dir / filename


def _open_csv_sink(path: Path, gzip_enabled: bool):
    if gzip_enabled:
        return gzip.open(path, "wb")
    return path.open("wb")


_CANONICAL_CASTS: dict[str, pa.DataType] = {"year": pa.string()}


def _cast_table(table: pa.Table) -> pa.Table:
    schema = table.schema
    for name, target in _CANONICAL_CASTS.items():
        idx = schema.get_field_index(name)
        if idx == -1:
            continue
        column = table.column(idx)
        if column.type == target:
            continue
        casted = _cast_column(column, target)
        table = table.set_column(idx, pa.field(name, target), casted)
    return table


def _cast_column(column: pa.ChunkedArray | pa.Array, target: pa.DataType) -> pa.ChunkedArray | pa.Array:
    if isinstance(column, pa.ChunkedArray):
        cast_chunks = [_cast_single(chunk, target) for chunk in column.chunks]
        return pa.chunked_array(cast_chunks, type=target)
    return _cast_single(column, target)


def _cast_single(array: pa.Array, target: pa.DataType) -> pa.Array:
    try:
        return pc.cast(array, target_type=target, safe=False)
    except Exception:  # pragma: no cover - defensive fallback
        values = [None if value is None else str(value) for value in array.to_pylist()]
        return pa.array(values, type=pa.string())
