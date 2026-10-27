"""Shared Builder2 Flask test fixtures. Process-only; no ENV files or product bypass."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)


@contextmanager
def public_builder2_flask_security() -> Iterator[None]:
    """
    Public Ori Flask security: ACE_SECURITY_ENABLED=true and mocked paid-auth success.
    Does not edit ENV files or change product security gates.
    """
    import app as app_module

    saved = os.environ.get("ACE_SECURITY_ENABLED")
    reset_local_dev_config_for_tests()
    os.environ["ACE_SECURITY_ENABLED"] = "true"
    try:
        with patch.object(
            app_module,
            "_ace_security_guard_builder2_create",
            return_value=(None, None, None),
        ), patch.object(
            app_module,
            "_ace_security_guard_builder2_job",
            return_value=None,
        ), patch.object(
            app_module,
            "_ace_security_guard_builder2_allowance",
            return_value=None,
        ):
            yield
    finally:
        if saved is None:
            os.environ.pop("ACE_SECURITY_ENABLED", None)
        else:
            os.environ["ACE_SECURITY_ENABLED"] = saved


def attach_public_builder2_flask_security(test_case: Any) -> None:
    cm = public_builder2_flask_security()
    test_case._public_builder2_flask_security_cm = cm
    cm.__enter__()


def detach_public_builder2_flask_security(test_case: Any) -> None:
    cm = getattr(test_case, "_public_builder2_flask_security_cm", None)
    if cm is None:
        return
    test_case._public_builder2_flask_security_cm = None
    cm.__exit__(None, None, None)


@contextmanager
def local_builder2_flask_env() -> Iterator[Path]:
    """Verified LOCAL + TEMP AceRoot for Flask tests. Restored on exit."""
    saved_sessions = os.environ.get("ACE_SESSIONS_ROOT")
    saved_sec = os.environ.get("ACE_SECURITY_ENABLED")
    saved_tourney = os.environ.get("BUILDER2_TOURNAMENT_ENABLED")
    tmp = tempfile.mkdtemp(prefix="ace-b2-http-")
    reset_local_dev_config_for_tests()
    os.environ["ACE_SESSIONS_ROOT"] = tmp
    os.environ["ACE_SECURITY_ENABLED"] = "false"
    os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"
    enable_local_dev_security_bypass_for_tests()
    confirm_local_dev_loopback_bind("127.0.0.1")
    try:
        yield Path(tmp)
    finally:
        reset_local_dev_config_for_tests()
        if saved_sessions is None:
            os.environ.pop("ACE_SESSIONS_ROOT", None)
        else:
            os.environ["ACE_SESSIONS_ROOT"] = saved_sessions
        if saved_sec is None:
            os.environ.pop("ACE_SECURITY_ENABLED", None)
        else:
            os.environ["ACE_SECURITY_ENABLED"] = saved_sec
        if saved_tourney is None:
            os.environ.pop("BUILDER2_TOURNAMENT_ENABLED", None)
        else:
            os.environ["BUILDER2_TOURNAMENT_ENABLED"] = saved_tourney
        shutil.rmtree(tmp, ignore_errors=True)
