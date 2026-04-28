from __future__ import annotations

import logging
import os
import re
from typing import Any
from typing import Literal

logger = logging.getLogger(__name__)

EngineName = Literal["builder1", "builder2"]
MemoryKeyName = Literal["object_a", "headlines"]

_CAP = 200
_ENGINE_KEYS: dict[str, dict[str, str]] = {
    "builder1": {
        "object_a": "ace:builder1:used_object_a",
        "headlines": "ace:builder1:used_headlines",
    },
    "builder2": {
        "object_a": "ace:builder2:used_object_a",
        "headlines": "ace:builder2:used_headlines",
    },
}


def normalize_memory_value(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _resolve_key(engine_name: EngineName, key_name: MemoryKeyName) -> str:
    if engine_name not in _ENGINE_KEYS:
        raise ValueError("invalid_engine_name")
    return _ENGINE_KEYS[engine_name][key_name]


def _redis_client_or_none():
    redis_url = (os.environ.get("REDIS_URL") or "").strip()
    if not redis_url:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable")
        return None
    try:
        import redis as redis_lib

        return redis_lib.Redis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable err=%s", e)
        return None


def _read_list(engine_name: EngineName, key_name: MemoryKeyName) -> list[str]:
    key = _resolve_key(engine_name, key_name)
    r = _redis_client_or_none()
    if r is None:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable")
        return []
    try:
        raw = r.lrange(key, 0, -1) or []
        out: list[str] = []
        for item in raw:
            n = normalize_memory_value(item)
            if n:
                out.append(n)
        if len(out) > _CAP:
            out = out[-_CAP:]
        logger.info("ACE_MEMORY_READ engine=%s key=%s count=%s", engine_name, key_name, len(out))
        return out
    except Exception as e:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable err=%s", e)
        return []


def _write_unique_fifo(engine_name: EngineName, key_name: MemoryKeyName, value: str) -> None:
    v = normalize_memory_value(value)
    if not v:
        return
    key = _resolve_key(engine_name, key_name)
    r = _redis_client_or_none()
    if r is None:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable")
        return
    try:
        existing = r.lrange(key, 0, -1) or []
        existing_norm = {normalize_memory_value(x) for x in existing}
        if v in existing_norm:
            logger.info(
                "ACE_MEMORY_DUPLICATE_SKIP engine=%s key=%s value=%s",
                engine_name,
                key_name,
                v,
            )
            return
        r.rpush(key, v)
        r.ltrim(key, -_CAP, -1)
        size_after = int(r.llen(key) or 0)
        logger.info(
            "ACE_MEMORY_WRITE engine=%s key=%s value=%s size_after=%s",
            engine_name,
            key_name,
            v,
            size_after,
        )
    except Exception as e:
        logger.warning("ACE_MEMORY_SKIP reason=redis_unavailable err=%s", e)


def get_used_object_a(engine_name: EngineName) -> list[str]:
    return _read_list(engine_name, "object_a")


def get_used_headlines(engine_name: EngineName) -> list[str]:
    return _read_list(engine_name, "headlines")


def remember_object_a(engine_name: EngineName, value: str) -> None:
    _write_unique_fifo(engine_name, "object_a", value)


def remember_headline(engine_name: EngineName, value: str) -> None:
    _write_unique_fifo(engine_name, "headlines", value)


def remember_usage(
    engine_name: EngineName,
    *,
    object_a: str | None = None,
    headline: str | None = None,
) -> None:
    if object_a is not None:
        remember_object_a(engine_name, object_a)
    if headline is not None:
        remember_headline(engine_name, headline)
