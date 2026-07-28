"""
Builder2 media finalization child-process lifecycle diagnostics.

Registered only from the preflight/recovery CLI entrypoint.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
from typing import Any


def _write_child_lifecycle_marker(marker: str) -> None:
    try:
        os.write(2, (marker + "\n").encode("utf-8", errors="replace"))
    except Exception:
        pass


def register_child_lifecycle_diagnostics() -> None:
    if getattr(register_child_lifecycle_diagnostics, "_registered", False):
        return
    register_child_lifecycle_diagnostics._registered = True  # type: ignore[attr-defined]

    atexit.register(lambda: _write_child_lifecycle_marker("BUILDER2_MEDIA_FINALIZATION_CHILD_ATEXIT"))

    def _handle_signal(signum: int, _frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        _write_child_lifecycle_marker(f"BUILDER2_MEDIA_FINALIZATION_CHILD_SIGNAL signal={sig_name}")
        raise SystemExit(128 + signum)

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass
