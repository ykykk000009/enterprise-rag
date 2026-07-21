"""Coordinate interactive answers with CPU-intensive background ingestion."""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class BackgroundWorkGate:
    """Pause cooperative ingestion checkpoints while an interactive request runs."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._interactive_requests = 0

    def begin_interactive(self) -> None:
        with self._condition:
            self._interactive_requests += 1

    def end_interactive(self) -> None:
        with self._condition:
            if self._interactive_requests == 0:
                raise RuntimeError("interactive request count cannot be negative")
            self._interactive_requests -= 1
            if self._interactive_requests == 0:
                self._condition.notify_all()

    def wait_for_background_work(self) -> None:
        with self._condition:
            while self._interactive_requests:
                self._condition.wait()


background_work_gate = BackgroundWorkGate()


@contextmanager
def lower_current_thread_priority() -> Iterator[None]:
    """Keep parser threads below normal priority so answers win CPU scheduling."""
    changed = _set_current_thread_priority(-1)
    try:
        yield
    finally:
        if changed:
            _set_current_thread_priority(0)


def _set_current_thread_priority(priority: int) -> bool:
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentThread()
        return bool(kernel32.SetThreadPriority(handle, priority))
    except (AttributeError, OSError):
        return False
