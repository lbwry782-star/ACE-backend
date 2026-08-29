"""
Builder1 production mode — require durable Redis and artifact storage.
"""
from __future__ import annotations

import os


def builder1_production_mode_enabled() -> bool:
    raw = (os.environ.get("BUILDER1_PRODUCTION_MODE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def builder1_ownership_required() -> bool:
    if builder1_production_mode_enabled():
        return True
    raw = (os.environ.get("BUILDER1_OWNERSHIP_REQUIRED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def builder1_redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def builder1_request_id_required() -> bool:
    if builder1_production_mode_enabled():
        return True
    raw = (os.environ.get("BUILDER1_REQUEST_ID_REQUIRED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def assert_builder1_production_ready() -> None:
    """
    Fail fast when production mode expects durable multi-user storage without Redis
    or without a non-temp artifact root.
    Tests and local dev may omit BUILDER1_PRODUCTION_MODE and use in-memory/temp fallback.
    """
    if builder1_production_mode_enabled() and not builder1_redis_configured():
        raise RuntimeError("builder1_production_requires_redis")
    if builder1_production_mode_enabled():
        from engine.builder1_artifact_storage import assert_builder1_durable_artifact_storage_ready

        assert_builder1_durable_artifact_storage_ready()
