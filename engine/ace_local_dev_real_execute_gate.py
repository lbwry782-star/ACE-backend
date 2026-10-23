"""Mirror Invoke-OriCancelDev.ps1 REAL_EXECUTE guard for API fail-fast (no PS subprocess)."""
from __future__ import annotations

import os
from pathlib import Path

_REAL_DEV_CYCLE_MARKER_SESSION = "ori-20261023-010342-2d952e88"


def is_real_ace_cancel_execute_guarded() -> bool:
    """
    True when cancel Execute targets the default local ACE layout with the real dev cycle
    marker session — requires ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE unless isolated via PS env overrides.
    """
    if any(
        (os.environ.get(key) or "").strip()
        for key in (
            "ACE_ORI_PS_BACKEND_ROOT",
            "ACE_ORI_PS_FRONTEND_ROOT",
            "ACE_ORI_PS_REGISTRY_PATH",
            "ACE_ORI_PS_SESSIONS_ROOT",
            "ACE_SESSIONS_ROOT",
        )
    ):
        return False

    backend_root = Path(__file__).resolve().parent.parent
    sessions_root = backend_root.parent / "_backups" / "sessions"
    marker = sessions_root / _REAL_DEV_CYCLE_MARKER_SESSION
    return marker.is_dir()
