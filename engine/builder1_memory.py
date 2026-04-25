"""
Global Builder1 ad repetition guard (FIFO, persisted to JSON for process restarts).

Tracks normalized Object A and headline text (excluding product names is the
caller's responsibility when calling remember_headline_text).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

_CAP = 200

_lock = threading.Lock()
_object_a_deque: Deque[str] = deque()
_headline_text_deque: Deque[str] = deque()
_memory_loaded = False


def _memory_path() -> str:
    return os.environ.get("BUILDER1_MEMORY_FILE", "/tmp/builder1_memory.json")


def _ensure_loaded_locked() -> None:
    global _memory_loaded
    if _memory_loaded:
        return
    path = _memory_path()
    if not os.path.isfile(path):
        _memory_loaded = True
        logger.warning("BUILDER1_MEMORY_LOAD_FAILED path=%r error=%r", path, "not_found")
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("root_must_be_object")
        oa = data.get("object_a") or []
        ht = data.get("headline_text") or []
        if not isinstance(oa, list):
            raise ValueError("object_a_must_be_list")
        if not isinstance(ht, list):
            raise ValueError("headline_text_must_be_list")
        oa_s = [_norm(str(x)) for x in oa if _norm(str(x))]
        ht_s = [_norm(str(x)) for x in ht if _norm(str(x))]
        if len(oa_s) > _CAP:
            oa_s = oa_s[-_CAP:]
        if len(ht_s) > _CAP:
            ht_s = ht_s[-_CAP:]
        _object_a_deque.clear()
        _headline_text_deque.clear()
        _object_a_deque.extend(oa_s)
        _headline_text_deque.extend(ht_s)
        _memory_loaded = True
        logger.info(
            "BUILDER1_MEMORY_LOADED object_a_count=%s headline_count=%s path=%r",
            len(_object_a_deque),
            len(_headline_text_deque),
            path,
        )
    except Exception as exc:
        _object_a_deque.clear()
        _headline_text_deque.clear()
        _memory_loaded = True
        logger.warning("BUILDER1_MEMORY_LOAD_FAILED path=%r error=%r", path, str(exc))


def _save_locked() -> None:
    path = _memory_path()
    payload = {
        "object_a": list(_object_a_deque),
        "headline_text": list(_headline_text_deque),
    }
    dir_name = os.path.dirname(path) or "."
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        pass
    fd, tmp_path = tempfile.mkstemp(
        prefix=".builder1_memory_",
        suffix=".tmp",
        dir=dir_name,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(
            "BUILDER1_MEMORY_SAVED object_a_count=%s headline_count=%s path=%r",
            len(_object_a_deque),
            len(_headline_text_deque),
            path,
        )
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _norm(s: str) -> str:
    return (s or "").strip().lower()


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
        _save_locked()
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
        _save_locked()
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
