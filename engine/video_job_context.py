"""
Per-thread phase for the active video job (worker main thread).

Used by worker shutdown handlers to log which pipeline stage was interrupted.
"""

from __future__ import annotations

import threading

_local = threading.local()


def video_job_set_phase(phase: str) -> None:
    _local.phase = phase


def video_job_get_phase() -> str:
    p = getattr(_local, "phase", None)
    if p in ("planning", "runway", "postprocess"):
        return p
    return "unknown"


def video_job_clear_phase() -> None:
    if hasattr(_local, "phase"):
        delattr(_local, "phase")
