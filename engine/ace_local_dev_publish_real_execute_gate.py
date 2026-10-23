"""Fail-closed gate for real Publish Execute on the default local ACE layout."""
from __future__ import annotations

import os
from pathlib import Path


def _has_isolated_publish_test_layout() -> bool:
    return any(
        (os.environ.get(key) or "").strip()
        for key in (
            "ACE_ORI_PS_BACKEND_ROOT",
            "ACE_ORI_PS_FRONTEND_ROOT",
            "ACE_ORI_PS_REGISTRY_PATH",
            "ACE_ORI_PS_SESSIONS_ROOT",
            "ACE_SESSIONS_ROOT",
        )
    )


def is_real_ace_publish_execute_guarded() -> bool:
    """
    True when Publish Execute would target the default on-disk ACE layout (local dev).
    Real publish must not run from LOCAL; isolated tests override paths via env.
    """
    if _has_isolated_publish_test_layout():
        return False
    backend_root = Path(__file__).resolve().parent.parent
    sessions_root = backend_root.parent / "_backups" / "sessions"
    return sessions_root.is_dir()


def allow_real_publish_execute_from_env() -> bool:
    raw = (os.environ.get("ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def publish_real_execute_permitted() -> bool:
    if not allow_real_publish_execute_from_env():
        return False
    if is_real_ace_publish_execute_guarded():
        return False
    return True
