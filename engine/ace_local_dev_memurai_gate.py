"""LOCAL Memurai availability gate — read-only state file; no service control."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from engine.ace_local_dev_config import is_local_dev_enabled
from engine.ace_local_dev_publish_candidate_registry import resolve_ace_root

logger = logging.getLogger(__name__)

AVAILABILITY_SCHEMA = 1
AVAILABILITY_FILENAME = "memurai-availability.json"

ERROR_MEMURAI_UNAVAILABLE = "ace_local_dev_memurai_unavailable"


def memurai_availability_path(*, ace_root: Optional[Path] = None) -> Path:
    root = ace_root if ace_root is not None else resolve_ace_root()
    return root / "_backups" / "local-dev" / AVAILABILITY_FILENAME


def load_memurai_availability(*, ace_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = memurai_availability_path(ace_root=ace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("MEMURAI_AVAILABILITY_READ_FAILED path=%s err=%s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def memurai_blocks_new_redis_work(*, ace_root: Optional[Path] = None) -> Optional[str]:
    """
    Return error code when new Redis-dependent LOCAL work must be blocked, else None.
    Missing availability file does not block (stack may predate the file).
    """
    if not is_local_dev_enabled():
        return None
    data = load_memurai_availability(ace_root=ace_root)
    if not data:
        return None
    if data.get("blocksNewRedisWork") is True:
        return ERROR_MEMURAI_UNAVAILABLE
    if data.get("pingOk") is False and data.get("blocksNewRedisWork") is not False:
        return ERROR_MEMURAI_UNAVAILABLE
    return None


def public_memurai_availability_payload(*, ace_root: Optional[Path] = None) -> Dict[str, Any]:
    data = load_memurai_availability(ace_root=ace_root)
    if not data:
        return {
            "ok": True,
            "available": None,
            "scenario": "unknown",
            "blocksNewRedisWork": False,
            "source": "missing",
        }
    return {
        "ok": True,
        "available": bool(data.get("pingOk")),
        "scenario": str(data.get("scenario") or ""),
        "blocksNewRedisWork": bool(data.get("blocksNewRedisWork")),
        "windowsServiceStatus": str(data.get("windowsServiceStatus") or ""),
        "updatedAt": str(data.get("updatedAt") or ""),
        "source": "file",
    }


def memurai_denied_json_response() -> Tuple[Dict[str, Any], int]:
    return (
        {
            "ok": False,
            "error": ERROR_MEMURAI_UNAVAILABLE,
            "message": "Memurai/Redis is not available. Start Memurai via services.msc, then retry.",
            "memuraiBlocksNewRedisWork": True,
        },
        503,
    )
