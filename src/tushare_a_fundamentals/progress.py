"""Lightweight progress reporting helpers for CLI workflows."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - defensive
        return False


@dataclass
class PlainTicker:
    """Fallback progress reporter for plain text environments."""

    desc: str
    total: int
    done: int = 0
    every_sec: float = 1.5
    last_print: float = 0.0
    ok: Optional[int] = None
    fail: Optional[int] = None

    def advance(
        self,
        step: int = 1,
        *,
        ok: Optional[int] = None,
        fail: Optional[int] = None,
    ) -> None:
        self.done += step
        if self.total >= 0:
            self.done = min(self.done, self.total)
        if ok is not None:
            self.ok = ok
        if fail is not None:
            self.fail = fail
        now = time.time()
        if (
            now - self.last_print >= self.every_sec
            or (self.total > 0 and self.done >= self.total)
        ):
            parts = [f"[..] {self.desc}: {self.done}/{self.total if self.total else '?'}"]
            stats: list[str] = []
            if self.ok is not None:
                stats.append(f"✓{self.ok}")
            if self.fail is not None and self.fail > 0:
                stats.append(f"✗{self.fail}")
            if stats:
                parts.append("(" + " ".join(stats) + ")")
            print(" ".join(parts), flush=True)
            self.last_print = now


class ProgressManager:
    """Encapsulates progress bar or plain ticker behaviour."""

    def __init__(self, mode: str = "auto") -> None:
        normalized = (mode or "auto").strip().lower()
        if normalized not in {"auto", "rich", "plain", "none"}:
            normalized = "auto"
        self.mode = normalized
        self._base_descriptions: Dict[Any, str] = {}

        self._use_rich = False
        self._use_plain = False

        if normalized == "none":
            pass
        elif normalized == "rich":
            self._use_rich = True
        elif normalized == "plain":
            self._use_plain = True
        else:  # auto
            if _is_tty(sys.stderr):
                self._use_rich = True
            else:
                self._use_plain = True

        self.console: Optional[Console] = None
        self.progress: Optional[Progress] = None
        if self._use_rich:
            self.console = Console(
                stderr=True,
                force_terminal=False,
                highlight=False,
                soft_wrap=True,
            )
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                refresh_per_second=5,
                console=self.console,
            )

    @property
    def is_active(self) -> bool:
        return self._use_rich or self._use_plain

    @contextmanager
    def live(self) -> Iterator["ProgressManager"]:
        if self.progress is not None:
            with self.progress:
                yield self
        else:
            yield self

    def add_task(self, description: str, total: int) -> Any:
        if not self.is_active or total <= 0:
            return None
        if self.progress is not None:
            task_id: TaskID = self.progress.add_task(description, total=total)
            self._base_descriptions[task_id] = description
            return task_id
        ticker = PlainTicker(description, total)
        self._base_descriptions[id(ticker)] = description
        return ticker

    def advance(
        self,
        task: Any,
        step: int = 1,
        *,
        ok: Optional[int] = None,
        fail: Optional[int] = None,
    ) -> None:
        if task is None:
            return
        if self.progress is not None and isinstance(task, int):
            base = self._base_descriptions.get(task)
            description = base
            stats: list[str] = []
            if ok is not None:
                stats.append(f"✓{ok}")
            if fail is not None and fail > 0:
                stats.append(f"✗{fail}")
            if stats and base:
                description = f"{base}  [{' '.join(stats)}]"
            update_kwargs: Dict[str, Any] = {"advance": step}
            if description is not None:
                update_kwargs["description"] = description
            self.progress.update(task, **update_kwargs)
            return
        if isinstance(task, PlainTicker):
            task.advance(step, ok=ok, fail=fail)

    def log(self, message: str) -> None:
        if self.console is not None:
            self.console.print(message)
        else:
            print(message)

