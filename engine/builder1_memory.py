"""
Global in-memory Builder1 ad repetition guard (FIFO, not persisted).

Tracks normalized Object A and headline text (excluding product names is the
caller's responsibility when calling remember_headline_text).
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

_CAP = 200

_lock = threading.Lock()
_object_a_deque: Deque[str] = deque()
_headline_text_deque: Deque[str] = deque()


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def remember_object_a(object_a: str) -> None:
    """Record Object A; FIFO-evict oldest when at capacity and this is a new value."""

    n = _norm(object_a)
    if not n:
        return
    evicted: str | None = None
    with _lock:
        if n in _object_a_deque:
            _object_a_deque.remove(n)
        else:
            if len(_object_a_deque) >= _CAP:
                evicted = _object_a_deque.popleft()
        _object_a_deque.append(n)
    if evicted is not None:
        logger.info("BUILDER1_MEMORY_OBJECT_A_EVICTED object_a=%r", evicted)
    logger.info("BUILDER1_MEMORY_OBJECT_A_REMEMBERED object_a=%r", n)


def was_object_a_used(object_a: str) -> bool:
    n = _norm(object_a)
    if not n:
        return False
    with _lock:
        return n in _object_a_deque


def remember_headline_text(headline_text: str) -> None:
    """Record headline text (non–product-name portion); FIFO-evict oldest when needed."""

    n = _norm(headline_text)
    if not n:
        return
    evicted: str | None = None
    with _lock:
        if n in _headline_text_deque:
            _headline_text_deque.remove(n)
        else:
            if len(_headline_text_deque) >= _CAP:
                evicted = _headline_text_deque.popleft()
        _headline_text_deque.append(n)
    if evicted is not None:
        logger.info("BUILDER1_MEMORY_HEADLINE_EVICTED headline_text=%r", evicted)
    logger.info("BUILDER1_MEMORY_HEADLINE_REMEMBERED headline_text=%r", n)


def was_headline_text_used(headline_text: str) -> bool:
    n = _norm(headline_text)
    if not n:
        return False
    with _lock:
        return n in _headline_text_deque


def get_builder1_memory_snapshot() -> dict:
    """Return current memory contents (oldest → newest per deque order)."""

    with _lock:
        return {
            "object_a": list(_object_a_deque),
            "headline_text": list(_headline_text_deque),
            "capacity_object_a": _CAP,
            "capacity_headline_text": _CAP,
        }
