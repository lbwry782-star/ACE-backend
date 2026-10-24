"""Live assessment for recovery periodic save — no reliance on stale gate files alone."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_execute_guard import is_local_dev_execute_in_progress
from engine.ace_local_dev_publish_candidate_registry import resolve_ace_root
from engine.ace_local_dev_recovery_periodic_save_backend_probe import probe_local_backend_health
from engine.ace_local_dev_shutdown_gate import observe_active_local_dev_operations
from engine.ace_local_dev_startup_recovery_check import _observation_is_critical

logger = logging.getLogger(__name__)

_BLOCK_KINDS = frozenset({"CANCEL", "PUBLISH_EXECUTE", "PUBLISH_PREPARE", "NEW"})


def _observation_blocks_periodic_save(obs: Dict[str, Any]) -> bool:
    if _observation_is_critical(obs):
        return True
    kind = str(obs.get("kind") or "")
    status = str(obs.get("status") or "").upper()
    if kind in _BLOCK_KINDS and status in ("UNKNOWN", "PARTIAL"):
        return True
    if kind == "CANCEL" and status == "ACTIVE":
        return True
    return False


def assess_recovery_periodic_save_checkpoint(
    *,
    ace_root: Optional[Path] = None,
    backend_reachable: Optional[bool] = None,
    skip_backend_probe: bool = False,
) -> Dict[str, Any]:
    """
    Fresh scan of LOCAL dev operations. Does not trust cached shutdown-gate JSON without rescanning.
    When assessment cannot run safely, checkpoint must be skipped (allowed=False).
    """
    root = ace_root if ace_root is not None else resolve_ace_root()
    backend_probe: Dict[str, Any] = {}
    if skip_backend_probe:
        probe_ok = bool(backend_reachable) if backend_reachable is not None else False
        if not probe_ok:
            return {
                "allowed": False,
                "reason": "backend_probe_skipped_not_verified",
                "source": "backend_probe",
                "observationCount": 0,
            }
    else:
        backend_probe = probe_local_backend_health(ace_root=root)
        probe_ok = bool(backend_probe.get("ok"))
        if not probe_ok:
            return {
                "allowed": False,
                "reason": str(backend_probe.get("reason") or "backend_probe_failed"),
                "source": "backend_probe",
                "backendProbe": backend_probe,
                "observationCount": 0,
            }
    if is_local_dev_execute_in_progress():
        return {
            "allowed": False,
            "reason": "execute_guard_busy",
            "source": "live_execute_guard",
            "observationCount": 0,
        }
    sessions_root = root / "_backups" / "sessions"
    try:
        scan = observe_active_local_dev_operations(
            ace_root=root,
            sessions_root=sessions_root,
            backend_reachable=True,
        )
    except Exception as exc:
        logger.warning("RECOVERY_PERIODIC_SAVE_OBSERVE_FAILED err=%s", exc)
        return {
            "allowed": False,
            "reason": "observation_failed",
            "source": "unknown",
            "detail": str(exc)[:240],
            "observationCount": 0,
        }

    observations: List[Dict[str, Any]] = list(scan.get("observations") or [])
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        kind = str(obs.get("kind") or "")
        status = str(obs.get("status") or "").upper()
        if kind == "EXECUTE_GUARD" and status == "ACTIVE":
            return {
                "allowed": False,
                "reason": "execute_guard_busy",
                "source": "live_observation",
                "observationCount": len(observations),
            }
        if kind == "BACKEND" and status == "UNKNOWN":
            return {
                "allowed": False,
                "reason": "backend_unreachable_unknown",
                "source": "live_observation",
                "observationCount": len(observations),
            }
        if _observation_blocks_periodic_save(obs):
            kind = str(obs.get("kind") or "")
            return {
                "allowed": False,
                "reason": f"critical_operation:{kind.lower()}",
                "source": "live_observation",
                "observationCount": len(observations),
            }

    payload: Dict[str, Any] = {
        "allowed": True,
        "reason": "",
        "source": "live_observation",
        "observationCount": len(observations),
        "executeBusy": bool(scan.get("executeBusy")),
    }
    if backend_probe:
        payload["backendProbe"] = backend_probe
    return payload


def _cli_main(argv: Optional[List[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    ace_text = ""
    for idx, token in enumerate(args):
        if token == "--ace-root" and idx + 1 < len(args):
            ace_text = args[idx + 1]
    root = Path(ace_text).resolve() if ace_text else resolve_ace_root()
    payload = assess_recovery_periodic_save_checkpoint(ace_root=root)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0 if payload.get("allowed") else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
