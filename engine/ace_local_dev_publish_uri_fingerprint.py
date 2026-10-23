"""Fingerprint for Publish Uri Preview (two-step confirm for future Execute)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def compute_publish_uri_preview_fingerprint(raw_preview: Dict[str, Any]) -> str:
    plan = raw_preview.get("plan") if isinstance(raw_preview.get("plan"), dict) else {}
    production = plan.get("production") if isinstance(plan.get("production"), dict) else {}
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    blockers: List[str] = sorted(str(b) for b in (plan.get("blockers") or []))
    drift_infra: List[str] = sorted(str(d) for d in (plan.get("registryInfraDrift") or []))
    drift_wip: List[str] = sorted(str(d) for d in (plan.get("registryWipDrift") or []))
    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    be = proposed.get("backend") if isinstance(proposed.get("backend"), dict) else {}
    fe = proposed.get("frontend") if isinstance(proposed.get("frontend"), dict) else {}
    cancel_cycle = plan.get("cancelCycle") if isinstance(plan.get("cancelCycle"), dict) else {}

    canonical = {
        "sessionId": str(plan.get("sessionId") or ""),
        "blockers": blockers,
        "registryInfraDrift": drift_infra,
        "registryWipDrift": drift_wip,
        "cancelCycleStatus": str(cancel_cycle.get("status") or ""),
        "productionBackend": str(production.get("backendWeb") or ""),
        "productionFrontend": str(production.get("frontend") or ""),
        "proposedBackendHead": str(be.get("proposedHeadSha") or ""),
        "proposedFrontendHead": str(fe.get("proposedHeadSha") or ""),
        "canPrepareLocalPublish": bool(readiness.get("canPrepareLocalPublish")),
        "verifiedSafeForLiveProduction": bool(readiness.get("verifiedSafeForLiveProduction")),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
