"""Fingerprint for Publish Uri Preview and Candidate Dry-run confirm."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def _sorted_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(x) for x in value)


def compute_publish_uri_candidate_preview_fingerprint(raw_preview: Dict[str, Any]) -> str:
    plan = raw_preview.get("plan") if isinstance(raw_preview.get("plan"), dict) else {}
    production = plan.get("production") if isinstance(plan.get("production"), dict) else {}
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    be = proposed.get("backend") if isinstance(proposed.get("backend"), dict) else {}
    cancel_cycle = plan.get("cancelCycle") if isinstance(plan.get("cancelCycle"), dict) else {}

    candidate_blockers = _sorted_str_list(plan.get("candidateBlockers") or plan.get("blockers"))
    warnings = _sorted_str_list(plan.get("warnings"))
    drift_infra = _sorted_str_list(plan.get("registryInfraDrift"))
    drift_wip = _sorted_str_list(plan.get("registryWipDrift"))
    files = _sorted_str_list(be.get("files"))
    commit_shas = _sorted_str_list(plan.get("candidateCommitShas"))
    if not commit_shas:
        commits = be.get("commits") if isinstance(be.get("commits"), list) else []
        for row in commits:
            if isinstance(row, dict) and row.get("fullSha"):
                commit_shas.append(str(row["fullSha"]))
        commit_shas = sorted(set(commit_shas))

    canonical = {
        "mode": "PUBLISH_URI_CANDIDATE_PREVIEW",
        "sessionId": str(plan.get("sessionId") or ""),
        "productionBackend": str(production.get("backendWeb") or ""),
        "productionFrontend": str(production.get("frontend") or ""),
        "candidateBackendSha": str(plan.get("candidateBackendSha") or be.get("proposedHeadSha") or ""),
        "candidateBlockers": candidate_blockers,
        "warnings": warnings,
        "registryInfraDrift": drift_infra,
        "registryWipDrift": drift_wip,
        "cancelCycleStatus": str(cancel_cycle.get("status") or ""),
        "candidateFiles": files,
        "candidateCommitShas": commit_shas,
        "candidateVerifiedSafeForPublish": bool(
            readiness.get("candidateVerifiedSafeForPublish")
        ),
        "canPrepareLocalPublish": bool(readiness.get("canPrepareLocalPublish")),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
