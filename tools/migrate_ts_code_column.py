#!/usr/bin/env python3
"""Rename legacy ticker columns in parquet outputs to ts_code."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def migrate_file(path: Path, dry_run: bool = False) -> bool:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"跳过 {path}: 读取失败（{exc}）")
        return False
    if "ticker" not in df.columns:
        return False
    if "ts_code" in df.columns:
        df = df.drop(columns=["ticker"])
        if dry_run:
            print(f"将删除冗余 ticker 列：{path}")
            return True
        df.to_parquet(path, index=False)
        print(f"已删除冗余 ticker 列：{path}")
        return True
    df = df.rename(columns={"ticker": "ts_code"})
    if dry_run:
        print(f"将重命名 ticker→ts_code：{path}")
        return True
    df.to_parquet(path, index=False)
    print(f"已重命名 ticker→ts_code：{path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="修正历史 parquet 中的 ticker 列")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("out/parquet"),
        help="parquet 根目录（默认 out/parquet）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将要修改的文件",
    )
    args = parser.parse_args()
    root: Path = args.root
    if not root.exists():
        print(f"目标目录不存在：{root}")
        return
    changed = 0
    for path in root.rglob("*.parquet"):
        if migrate_file(path, dry_run=args.dry_run):
            changed += 1
    action = "需修改" if args.dry_run else "已处理"
    print(f"{action}文件共 {changed} 个")


if __name__ == "__main__":
    main()
