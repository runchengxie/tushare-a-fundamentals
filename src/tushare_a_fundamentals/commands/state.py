from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..common import eprint
from ..state_backend import JsonStateBackend, SQLiteStateBackend, StateBackend

_DEFAULT_JSON_RELATIVE = Path("_state") / "state.json"
_DEFAULT_SQLITE_PATH = Path("meta") / "state.db"


def _default_json_path(data_dir: Path) -> Path:
    return data_dir / _DEFAULT_JSON_RELATIVE


def _resolve_backend_and_path(args: argparse.Namespace) -> tuple[str, Path]:
    backend = args.backend
    state_path_arg = Path(args.state_path) if args.state_path else None
    data_dir = Path(args.data_dir or "data")

    if backend == "auto":
        if state_path_arg:
            backend = "sqlite" if state_path_arg.suffix == ".db" else "json"
        else:
            sqlite_path = _DEFAULT_SQLITE_PATH
            if sqlite_path.exists():
                backend = "sqlite"
                state_path_arg = sqlite_path
            else:
                backend = "json"
                state_path_arg = _default_json_path(data_dir)
    elif backend == "sqlite" and state_path_arg is None:
        state_path_arg = _DEFAULT_SQLITE_PATH
    elif backend == "json" and state_path_arg is None:
        state_path_arg = _default_json_path(data_dir)

    if backend not in {"json", "sqlite"}:
        raise ValueError(f"未知 backend: {backend}")

    return backend, state_path_arg  # type: ignore[return-value]


def _ls_failures(data_dir: Path) -> None:
    root = data_dir / "_state" / "failures"
    if not root.exists():
        print("未发现失败记录目录")
        return
    files: Iterable[Path] = sorted(root.glob("*.json"))
    found = False
    for fp in files:
        found = True
        try:
            payload = json.loads(fp.read_text("utf-8"))
            entries = payload.get("entries", [])
            print(f"{fp}: {len(entries)} 条记录")
        except json.JSONDecodeError:
            print(f"{fp}: 无法解析（可能损坏）")
    if not found:
        print("未发现失败记录文件")


def cmd_state(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir or "data")
    backend, state_path = _resolve_backend_and_path(args)

    if args.action == "ls-failures":
        _ls_failures(data_dir)
        return

    backend_impl: StateBackend
    if backend == "json":
        backend_impl = JsonStateBackend(state_path)
    else:
        backend_impl = SQLiteStateBackend(state_path)

    if args.action == "show":
        result = backend_impl.snapshot(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.action == "clear":
        if not args.dataset:
            eprint("错误：清理状态时必须指定 --dataset")
            return
        backend_impl.delete(args.dataset, args.key, year=args.year)
        target = [f"dataset={args.dataset}"]
        if args.key:
            target.append(f"key={args.key}")
        if args.year is not None:
            target.append(f"year={args.year}")
        print(f"已清理状态：{', '.join(target)}")
        return

    if args.action == "set":
        if not args.dataset or not args.key or args.value is None:
            eprint("错误：设置状态时必须提供 --dataset、--key、--value")
            return
        backend_impl.set(args.dataset, args.key, args.value)
        print(f"已更新状态：{args.dataset}.{args.key} = {args.value}")
        return

    eprint(f"错误：未识别的操作 {args.action}")


def register_state_subparser(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser("state", help="查看与维护增量状态信息")
    sp.set_defaults(cmd="state")
    sp.add_argument(
        "action",
        choices=["show", "clear", "set", "ls-failures"],
        help="操作类型",
    )
    sp.add_argument("--backend", choices=["auto", "json", "sqlite"], default="auto")
    sp.add_argument("--state-path", help="状态文件或数据库路径")
    sp.add_argument("--data-dir", default="data", help="多数据集数据目录（默认 data）")
    sp.add_argument("--dataset", help="目标数据集名称")
    sp.add_argument("--year", type=int, help="针对 SQLite 状态时可指定年份分区")
    sp.add_argument("--key", help="JSON 状态键名")
    sp.add_argument("--value", help="JSON 状态值")
