"""Two-step confirm: bind Execute to a fresh Preview plan."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def compute_preview_fingerprint(raw_preview: Dict[str, Any]) -> str:
    plan = raw_preview.get("plan") if isinstance(raw_preview.get("plan"), dict) else {}
    cancel_cycle = plan.get("cancelCycle") if isinstance(plan.get("cancelCycle"), dict) else {}
    infra = plan.get("infraTargets") if isinstance(plan.get("infraTargets"), dict) else {}
    blockers: List[str] = sorted(str(b) for b in (plan.get("blockers") or []))
    drift: List[str] = sorted(str(d) for d in (plan.get("registryInfraDrift") or []))
    canonical = {
        "sessionId": str(plan.get("sessionId") or ""),
        "canExecute": bool(plan.get("canExecute")),
        "blockers": blockers,
        "registryInfraDrift": drift,
        "cancelCycleStatus": str(cancel_cycle.get("status") or ""),
        "backendInfraCommit": str(infra.get("backendCommit") or ""),
        "frontendInfraCommit": str(infra.get("frontendCommit") or ""),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
