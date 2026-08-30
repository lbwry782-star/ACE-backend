"""
Builder1 integrity-failure diagnostic persistence (job-scoped, TTL-bound).

Persists rejected-plan snapshots and deterministic detector evidence when campaign
integrity fails after paid planning — without creating a campaign session.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

INTEGRITY_DIAGNOSTIC_JOB_FIELD = "integrityFailureDiagnostic"


def record_integrity_evidence(
    evidence: Optional[MutableMapping[str, Any] | List[Dict[str, Any]]],
    *,
    code: str,
    detector: str,
    branch: str,
    reason: str,
    level: str = "plan",
    field: str = "",
    ad_index: Optional[int] = None,
    slogan_tokens: Optional[List[str]] = None,
    matched_terms: Optional[List[str]] = None,
    independent_visual_proof_absent: Optional[bool] = None,
    field_value_preview: str = "",
) -> None:
    if evidence is None:
        return
    entry: Dict[str, Any] = {
        "code": code,
        "detector": detector,
        "branch": branch,
        "level": level,
        "reason": reason,
    }
    if field:
        entry["field"] = field
    if ad_index is not None:
        entry["adIndex"] = int(ad_index)
    if slogan_tokens:
        entry["sloganTokens"] = list(slogan_tokens)
    if matched_terms:
        entry["matchedTerms"] = list(matched_terms)
    if independent_visual_proof_absent is not None:
        entry["independentVisualProofAbsent"] = bool(independent_visual_proof_absent)
    if field_value_preview:
        preview = str(field_value_preview).strip()
        if len(preview) > 240:
            preview = preview[:237] + "..."
        entry["fieldValuePreview"] = preview
    if isinstance(evidence, list):
        evidence.append(entry)
    elif isinstance(evidence, Mapping):
        bucket = evidence.setdefault(code, [])
        if isinstance(bucket, list):
            bucket.append(entry)


def build_rejected_plan_diagnostic_snapshot(plan_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalized plan representation as supplied to integrity validation."""
    snapshot = dict(plan_dict)
    internals = snapshot.get("planningInternals")
    if isinstance(internals, dict):
        snapshot["planningInternals"] = dict(internals)
    ads = snapshot.get("ads")
    if isinstance(ads, list):
        snapshot["ads"] = [dict(ad) if isinstance(ad, dict) else ad for ad in ads]
    return snapshot


def build_integrity_failure_diagnostic(
    *,
    reasons: List[str],
    details: List[Dict[str, Any]],
    rejected_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "capturedAt": time.time(),
        "reasons": list(reasons),
        "details": list(details),
        "rejectedPlanAvailable": True,
        "rejectedPlan": build_rejected_plan_diagnostic_snapshot(rejected_plan),
    }


def persist_integrity_failure_diagnostic(
    job_id: str,
    diagnostic: Mapping[str, Any],
    *,
    campaign_id: str = "",
) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    from engine.builder1_jobs_store import get_builder1_job, update_builder1_job

    if get_builder1_job(jid) is None:
        return False
    update_builder1_job(jid, **{INTEGRITY_DIAGNOSTIC_JOB_FIELD: dict(diagnostic)})
    branches = sorted(
        {
            str(item.get("branch") or "")
            for item in (diagnostic.get("details") or [])
            if isinstance(item, dict) and item.get("branch")
        }
    )
    logger.error(
        "BUILDER1_INTEGRITY_FAILED reasons=%s diagnosticPersisted=true jobId=%s campaignId=%s branches=%s",
        diagnostic.get("reasons"),
        jid,
        campaign_id or "",
        branches[:12],
    )
    return True


def get_integrity_failure_diagnostic(job_id: str) -> Optional[Dict[str, Any]]:
    from engine.builder1_jobs_store import get_builder1_job

    job = get_builder1_job((job_id or "").strip())
    if not job:
        return None
    raw = job.get(INTEGRITY_DIAGNOSTIC_JOB_FIELD)
    return dict(raw) if isinstance(raw, dict) else None
