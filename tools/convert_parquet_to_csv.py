#!/usr/bin/env python3
"""Convert existing parquet outputs to csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def convert_parquet_to_csv(src: str | Path, dest: str | Path) -> None:
    src_dir = Path(src)
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("*.parquet"):
        df = pd.read_parquet(p)
        out_file = dest_dir / (p.stem + ".csv")
        df.to_csv(out_file, index=False)
        print(f"已转换：{p} -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="parquet 转 csv")
    parser.add_argument("--src", default="out/parquet")
    parser.add_argument("--dest", default="out/csv")
    args = parser.parse_args()
    convert_parquet_to_csv(args.src, args.dest)


if __name__ == "__main__":
    main()
