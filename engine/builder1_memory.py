"""
Global Builder1 ad repetition guard (FIFO, Redis-backed when available).

Tracks normalized Object A and headline text (excluding product names is the
caller's responsibility when calling remember_headline_text).
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

_CAP = 200

_lock = threading.Lock()
_object_a_deque: Deque[str] = deque()
_headline_text_deque: Deque[str] = deque()
_memory_loaded = False
_redis_available = False

_OBJECT_A_KEY = "ACE:BUILDER1:OBJECT_A_MEMORY"
_HEADLINE_KEY = "ACE:BUILDER1:HEADLINE_MEMORY"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        n = _norm(str(item))
        if n:
            cleaned.append(n)
    if len(cleaned) > _CAP:
        cleaned = cleaned[-_CAP:]
    return cleaned


def _redis_get_json(key: str) -> object | None:
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            return None
        raw = get_redis().get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("BUILDER1_MEMORY_REDIS_UNAVAILABLE reason=%r", f"redis_get_failed:{exc}")
        return None


def _redis_set_json(key: str, value: object) -> bool:
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            return False
        get_redis().set(key, json.dumps(value, ensure_ascii=False))
        return True
    except Exception as exc:
        logger.warning("BUILDER1_MEMORY_REDIS_UNAVAILABLE reason=%r", f"redis_set_failed:{exc}")
        return False


def _load_from_redis_locked() -> bool:
    object_data = _redis_get_json(_OBJECT_A_KEY)
    headline_data = _redis_get_json(_HEADLINE_KEY)
    if object_data is None and headline_data is None:
        return False
    object_list = _normalize_list(object_data)
    headline_list = _normalize_list(headline_data)
    _object_a_deque.clear()
    _headline_text_deque.clear()
    _object_a_deque.extend(object_list)
    _headline_text_deque.extend(headline_list)
    logger.info(
        "BUILDER1_MEMORY_REDIS_LOADED object_a_count=%s headline_count=%s",
        len(_object_a_deque),
        len(_headline_text_deque),
    )
    return True


def _save_to_redis_locked() -> bool:
    payload_object = list(_object_a_deque)
    payload_headline = list(_headline_text_deque)
    ok_object = _redis_set_json(_OBJECT_A_KEY, payload_object)
    ok_headline = _redis_set_json(_HEADLINE_KEY, payload_headline)
    if ok_object and ok_headline:
        logger.info(
            "BUILDER1_MEMORY_REDIS_SAVED object_a_count=%s headline_count=%s",
            len(_object_a_deque),
            len(_headline_text_deque),
        )
        return True
    return False


def _ensure_loaded_locked() -> None:
    global _memory_loaded, _redis_available
    if _memory_loaded:
        return
    try:
        from engine.video_jobs_redis import redis_configured

        if redis_configured():
            _redis_available = _load_from_redis_locked()
            if not _redis_available:
                logger.warning("BUILDER1_MEMORY_REDIS_UNAVAILABLE reason=%r", "redis_keys_missing_or_unreadable")
        else:
            logger.warning("BUILDER1_MEMORY_REDIS_UNAVAILABLE reason=%r", "redis_not_configured")
    except Exception as exc:
        logger.warning("BUILDER1_MEMORY_REDIS_UNAVAILABLE reason=%r", f"redis_init_failed:{exc}")
    _memory_loaded = True
    if not _redis_available:
        logger.info(
            "BUILDER1_MEMORY_FALLBACK_IN_MEMORY object_a_count=%s headline_count=%s",
            len(_object_a_deque),
            len(_headline_text_deque),
        )


def remember_object_a(object_a: str) -> None:
    """Record Object A; FIFO-evict oldest when at capacity and this is a new value."""

    n = _norm(object_a)
    if not n:
        return
    evicted: str | None = None
    with _lock:
        _ensure_loaded_locked()
        if n in _object_a_deque:
            _object_a_deque.remove(n)
        else:
            if len(_object_a_deque) >= _CAP:
                evicted = _object_a_deque.popleft()
        _object_a_deque.append(n)
        if not _save_to_redis_locked():
            logger.info(
                "BUILDER1_MEMORY_FALLBACK_IN_MEMORY object_a_count=%s headline_count=%s",
                len(_object_a_deque),
                len(_headline_text_deque),
            )
    if evicted is not None:
        logger.info("BUILDER1_MEMORY_OBJECT_A_EVICTED object_a=%r", evicted)
    logger.info("BUILDER1_MEMORY_OBJECT_A_REMEMBERED object_a=%r", n)


def was_object_a_used(object_a: str) -> bool:
    n = _norm(object_a)
    if not n:
        return False
    with _lock:
        _ensure_loaded_locked()
        return n in _object_a_deque


def remember_headline_text(headline_text: str) -> None:
    """Record headline text (non–product-name portion); FIFO-evict oldest when needed."""

    n = _norm(headline_text)
    if not n:
        return
    evicted: str | None = None
    with _lock:
        _ensure_loaded_locked()
        if n in _headline_text_deque:
            _headline_text_deque.remove(n)
        else:
            if len(_headline_text_deque) >= _CAP:
                evicted = _headline_text_deque.popleft()
        _headline_text_deque.append(n)
        if not _save_to_redis_locked():
            logger.info(
                "BUILDER1_MEMORY_FALLBACK_IN_MEMORY object_a_count=%s headline_count=%s",
                len(_object_a_deque),
                len(_headline_text_deque),
            )
    if evicted is not None:
        logger.info("BUILDER1_MEMORY_HEADLINE_EVICTED headline_text=%r", evicted)
    logger.info("BUILDER1_MEMORY_HEADLINE_REMEMBERED headline_text=%r", n)


def was_headline_text_used(headline_text: str) -> bool:
    n = _norm(headline_text)
    if not n:
        return False
    with _lock:
        _ensure_loaded_locked()
        return n in _headline_text_deque


def get_builder1_memory_snapshot() -> dict:
    """Return current memory contents (oldest → newest per deque order)."""

    with _lock:
        _ensure_loaded_locked()
        return {
            "object_a": list(_object_a_deque),
            "headline_text": list(_headline_text_deque),
            "capacity_object_a": _CAP,
            "capacity_headline_text": _CAP,
        }
