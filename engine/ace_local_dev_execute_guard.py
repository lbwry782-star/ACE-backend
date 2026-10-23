"""In-process guard against concurrent local cancel Execute API calls."""
from __future__ import annotations

import threading

_execute_lock = threading.Lock()
_execute_in_progress = False


class LocalDevExecuteBusyError(RuntimeError):
    pass


def try_begin_local_dev_execute() -> None:
    global _execute_in_progress
    with _execute_lock:
        if _execute_in_progress:
            raise LocalDevExecuteBusyError("ace_local_dev_execute_in_progress")
        _execute_in_progress = True


def end_local_dev_execute() -> None:
    global _execute_in_progress
    with _execute_lock:
        _execute_in_progress = False
