from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import sqlite3

from ..common import eprint
from ..downloader import JsonState
from ..meta.state_store import init_state_store

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


def _load_json_state(path: Path) -> JsonState:
    return JsonState(path)


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _show_json_state(state: JsonState, dataset: str | None) -> Dict[str, Any]:
    if dataset:
        return state.data.get(dataset, {})
    return state.data


def _show_sqlite_state(path: Path, dataset: str | None) -> Dict[str, Any]:
    if not path.exists():
        return {}
    conn = sqlite3.connect(path.as_posix())
    try:
        cur = conn.cursor()
        payload: Dict[str, Any] = {}
        if dataset:
            cur.execute(
                "SELECT dataset, low, high FROM watermarks WHERE dataset=?", (dataset,)
            )
            payload["watermarks"] = [dict(zip(["dataset", "low", "high"], row)) for row in cur.fetchall()]
            cur.execute(
                "SELECT dataset, part_year, min_key, max_key, last_checked_at, last_success_at, dirty "
                "FROM dataset_state WHERE dataset=? ORDER BY part_year",
                (dataset,),
            )
            payload["dataset_state"] = [
                dict(
                    zip(
                        [
                            "dataset",
                            "part_year",
                            "min_key",
                            "max_key",
                            "last_checked_at",
                            "last_success_at",
                            "dirty",
                        ],
                        row,
                    )
                )
                for row in cur.fetchall()
            ]
        else:
            cur.execute("SELECT dataset, low, high FROM watermarks ORDER BY dataset")
            payload["watermarks"] = [dict(zip(["dataset", "low", "high"], row)) for row in cur.fetchall()]
            cur.execute(
                "SELECT dataset, part_year, min_key, max_key, last_checked_at, last_success_at, dirty "
                "FROM dataset_state ORDER BY dataset, part_year"
            )
            payload["dataset_state"] = [
                dict(
                    zip(
                        [
                            "dataset",
                            "part_year",
                            "min_key",
                            "max_key",
                            "last_checked_at",
                            "last_success_at",
                            "dirty",
                        ],
                        row,
                    )
                )
                for row in cur.fetchall()
            ]
        return payload
    finally:
        conn.close()


def _clear_json_state(state: JsonState, dataset: str | None, key: str | None) -> None:
    if dataset is None:
        eprint("错误：清理 JSON 状态时必须指定 --dataset")
        return
    bucket = state.data.get(dataset)
    if not bucket:
        eprint(f"提示：未找到数据集 {dataset} 的状态")
        return
    if key:
        if key in bucket:
            del bucket[key]
            if not bucket:
                del state.data[dataset]
        else:
            eprint(f"提示：数据集 {dataset} 下不存在键 {key}")
            return
    else:
        del state.data[dataset]
    _ensure_parent(state.path)
    state.path.write_text(json.dumps(state.data, ensure_ascii=False, indent=2), "utf-8")
    print(f"已清理 JSON 状态：dataset={dataset}{', key='+key if key else ''}")


def _set_json_state(state: JsonState, dataset: str | None, key: str | None, value: str | None) -> None:
    if not dataset or not key or value is None:
        eprint("错误：设置 JSON 状态时必须提供 --dataset、--key、--value")
        return
    bucket = state.data.setdefault(dataset, {})
    bucket[key] = value
    _ensure_parent(state.path)
    state.path.write_text(json.dumps(state.data, ensure_ascii=False, indent=2), "utf-8")
    print(f"已更新 JSON 状态：{dataset}.{key} = {value}")


def _clear_sqlite_state(path: Path, dataset: str | None, year: int | None) -> None:
    if dataset is None:
        eprint("错误：清理 SQLite 状态时必须指定 --dataset")
        return
    conn = init_state_store(path)
    try:
        cur = conn.cursor()
        if year is not None:
            cur.execute(
                "DELETE FROM dataset_state WHERE dataset=? AND part_year=?",
                (dataset, year),
            )
        else:
            cur.execute("DELETE FROM dataset_state WHERE dataset=?", (dataset,))
        cur.execute("DELETE FROM watermarks WHERE dataset=?", (dataset,))
        conn.commit()
        print(
            f"已清理 SQLite 状态：dataset={dataset}"
            f"{', year='+str(year) if year is not None else ''}"
        )
    finally:
        conn.close()


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

    if backend == "json":
        state = _load_json_state(state_path)
        if args.action == "show":
            result = _show_json_state(state, args.dataset)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action == "clear":
            _clear_json_state(state, args.dataset, args.key)
        elif args.action == "set":
            _set_json_state(state, args.dataset, args.key, args.value)
        else:
            eprint(f"错误：JSON 后端不支持操作 {args.action}")
    else:  # sqlite
        if args.action == "show":
            result = _show_sqlite_state(state_path, args.dataset)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action == "clear":
            _clear_sqlite_state(state_path, args.dataset, args.year)
        elif args.action == "set":
            eprint("错误：SQLite 后端暂不支持 set 操作")
        else:
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
