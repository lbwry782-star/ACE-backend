"""
Builder1 explicit same-job planning resume — eligibility and dispatch helpers.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional

from engine.builder1_integrity_diagnostics import (
    INTEGRITY_DIAGNOSTIC_JOB_FIELD,
    get_integrity_failure_diagnostic,
)
from engine.builder1_job_planning_request import (
    planning_request_snapshot_from_job,
    snapshot_request_fingerprint,
)
from engine.builder1_jobs_store import get_builder1_job, update_builder1_job
from engine.builder1_planning_checkpoint import (
    build_planning_checkpoint_identity,
    load_planning_checkpoint_record,
)

logger = logging.getLogger(__name__)

_BLOCKING_PAID_STAGE_STATUSES = frozenset({"submitted", "in_flight", "outcome_unknown"})
_PLANNING_STAGE_FAILURE_MARKERS = frozenset(
    {
        "planning_failed",
        "product_name_generation_failed",
        "brand_physical_failed",
        "graphic_system_failed",
        "series_ads_failed",
        "conceptual_stage_failed",
        "strategy_slogan_stage_failed",
        "campaign_integrity_failed",
        "campaign_visibility_integrity_failed",
    }
)
_INTEGRITY_FAILURE_MARKERS = frozenset(
    {
        "campaign_integrity_failed",
        "campaign_visibility_integrity_failed",
    }
)
_PLANNING_ONLY_STAGES = frozenset(
    {
        "planning",
        "product_name_resolution",
        "strategy_slogan_stage",
        "conceptual_stage",
        "brand_physical",
        "graphic_system",
        "series_ads",
    }
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "yes"}


def _job_failure_text(job: Mapping[str, Any]) -> str:
    err = _clean(job.get("error"))
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    message = _clean(result.get("message") if isinstance(result, dict) else "")
    return f"{err} {message}".lower()


def _job_is_integrity_rejection(job: Mapping[str, Any]) -> bool:
    text = _job_failure_text(job)
    if any(marker in text for marker in _INTEGRITY_FAILURE_MARKERS):
        return True
    if get_integrity_failure_diagnostic(_clean(job.get("jobId") or "")) is not None:
        return True
    if isinstance(job.get(INTEGRITY_DIAGNOSTIC_JOB_FIELD), dict):
        return True
    return False


def _job_is_planning_stage_failure(job: Mapping[str, Any], *, job_id: str = "") -> bool:
    status = _clean(job.get("status")).lower()
    if status != "error":
        return False
    text = _job_failure_text(job)
    stage = _clean(job.get("stage")).lower()
    if stage in _PLANNING_ONLY_STAGES or stage == "planning":
        if "planning_failed" in text or any(
            marker in text
            for marker in _PLANNING_STAGE_FAILURE_MARKERS
            if marker not in _INTEGRITY_FAILURE_MARKERS
        ):
            return True
    if "planning_failed" in text:
        return True
    if _clean(job.get("error")) in {
        "builder1_generation_failed",
        "product_name_generation_failed",
    } and stage in {"", "planning", *(_PLANNING_ONLY_STAGES)}:
        return True
    checkpoint = load_planning_checkpoint_record(_clean(job_id) or _clean(job.get("jobId")))
    if checkpoint and dict(checkpoint.get("completedStages") or {}):
        return True
    return False


def _campaign_has_started_media(campaign_id: str) -> bool:
    if not campaign_id:
        return False
    from engine.builder1_campaign_store import get_campaign_session_raw

    raw = get_campaign_session_raw(campaign_id)
    if not raw:
        return False
    generated = raw.get("generated") or raw.get("generatedAds") or []
    if isinstance(generated, list) and generated:
        return True
    if int(raw.get("generatedCount") or 0) > 0:
        return True
    if raw.get("generatingIndex") is not None:
        return True
    paid_stage = _clean(raw.get("lastPaidStage"))
    if paid_stage and paid_stage not in _PLANNING_ONLY_STAGES:
        return True
    paid_status = _clean(raw.get("lastPaidStageStatus"))
    if paid_stage and paid_status in _BLOCKING_PAID_STAGE_STATUSES:
        return True
    return False


def _campaign_has_conflicting_lock(campaign_id: str, job_id: str) -> bool:
    if not campaign_id:
        return False
    from engine.builder1_campaign_store import get_campaign_session_raw

    raw = get_campaign_session_raw(campaign_id)
    if not raw:
        return False
    owner = _clean(raw.get("generatingLockOwnerJobId"))
    if not owner:
        return False
    return owner != _clean(job_id)


def _job_image_provider_blocks(job: Mapping[str, Any]) -> bool:
    paid_status = _clean(job.get("lastPaidStageStatus"))
    if paid_status in _BLOCKING_PAID_STAGE_STATUSES:
        return True
    paid_stage = _clean(job.get("lastPaidStage"))
    if paid_stage and paid_stage not in _PLANNING_ONLY_STAGES and paid_status in {
        "submitted",
        "in_flight",
        "outcome_unknown",
    }:
        return True
    return False


def _checkpoint_compatible_with_job(
    job: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    job_id: str = "",
) -> bool:
    snapshot = planning_request_snapshot_from_job(job)
    if snapshot is None:
        return False
    identity_raw = checkpoint.get("identity")
    if not isinstance(identity_raw, dict):
        return False
    jid = _clean(job_id) or _clean(identity_raw.get("jobId"))
    expected = build_planning_checkpoint_identity(
        job_id=jid,
        campaign_id=_clean(job.get("campaignId") or identity_raw.get("campaignId")),
        product_name=_clean(snapshot.get("productName")),
        product_description=_clean(snapshot.get("productDescription")),
        format_value=_clean(snapshot.get("format")) or "portrait",
        ad_count=int(snapshot.get("adCount") or job.get("targetAdCount") or 2),
        brand_guidelines=snapshot.get("brandGuidelines"),
    )
    for key in ("jobId", "campaignId", "requestFingerprint", "planningContractVersion"):
        if _clean(identity_raw.get(key)) != _clean(expected.to_dict().get(key)):
            return False
    if _clean(identity_raw.get("requestFingerprint")) != snapshot_request_fingerprint(snapshot):
        return False
    return bool(dict(checkpoint.get("completedStages") or {}))


def assess_builder1_planning_resume(job_id: str) -> Dict[str, Any]:
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid,
        "campaignId": "",
        "eligible": False,
        "rejectionReasons": [],
        "checkpointPresent": False,
        "completedStageCount": 0,
        "planningRequestSnapshotPresent": False,
    }
    job = get_builder1_job(jid)
    if job is None:
        report["rejectionReasons"].append("job_not_found")
        return report

    report["campaignId"] = _clean(job.get("campaignId"))
    if _clean(job.get("builder")) != "builder1":
        report["rejectionReasons"].append("not_builder1_job")

    status = _clean(job.get("status")).lower()
    if status == "cancelled" or _truthy(job.get("cancelRequested")):
        report["rejectionReasons"].append("job_cancelled")
    elif status == "running":
        report["rejectionReasons"].append("job_already_running")
    elif status == "done":
        report["rejectionReasons"].append("job_already_done")
    elif status != "error":
        report["rejectionReasons"].append("job_not_in_error_state")

    if _truthy(job.get("retryable")) is False and _clean(job.get("error")) == "stale_job_abandoned":
        report["rejectionReasons"].append("job_non_retryable")

    snapshot = planning_request_snapshot_from_job(job)
    report["planningRequestSnapshotPresent"] = snapshot is not None
    if snapshot is None:
        report["rejectionReasons"].append("planning_request_snapshot_missing")

    checkpoint = load_planning_checkpoint_record(jid)
    report["checkpointPresent"] = checkpoint is not None
    if checkpoint:
        report["completedStageCount"] = len(dict(checkpoint.get("completedStages") or {}))
    if not checkpoint or not dict(checkpoint.get("completedStages") or {}):
        report["rejectionReasons"].append("planning_checkpoint_missing")

    if _job_is_integrity_rejection(job):
        report["rejectionReasons"].append("integrity_rejection_not_planning_resume")

    if status == "error" and not _job_is_planning_stage_failure(job, job_id=jid):
        if "integrity_rejection_not_planning_resume" not in report["rejectionReasons"]:
            report["rejectionReasons"].append("not_planning_stage_failure")

    if _job_image_provider_blocks(job):
        report["rejectionReasons"].append("image_provider_inflight")

    cid = report["campaignId"]
    if _campaign_has_started_media(cid):
        report["rejectionReasons"].append("campaign_media_started")

    if _campaign_has_conflicting_lock(cid, jid):
        report["rejectionReasons"].append("campaign_generation_lock_conflict")

    if checkpoint and snapshot and not _checkpoint_compatible_with_job(job, checkpoint, job_id=jid):
        report["rejectionReasons"].append("checkpoint_identity_mismatch")

    if _clean(job.get("planningResumeRequested")):
        report["rejectionReasons"].append("planning_resume_already_requested")

    report["eligible"] = not report["rejectionReasons"]
    return report


def mark_builder1_planning_resume_requested(
    job_id: str,
    *,
    source: str = "api",
    request_id: str = "",
) -> Dict[str, Any]:
    jid = _clean(job_id)
    job = get_builder1_job(jid) or {}
    previous_error = _clean(job.get("error"))
    resume_count = int(job.get("planningResumeCount") or 0) + 1
    now = time.time()
    fields: Dict[str, Any] = {
        "status": "running",
        "stage": "planning",
        "error": "",
        "planningResumeRequested": True,
        "planningResumeCount": resume_count,
        "planningResumedAt": now,
        "planningResumeSource": _clean(source) or "api",
        "previousPlanningError": previous_error,
        "lastHeartbeatAt": now,
    }
    if request_id:
        fields["planningResumeRequestId"] = _clean(request_id)
    update_builder1_job(jid, **fields)
    logger.info(
        "BUILDER1_PLANNING_RESUME_ACCEPTED jobId=%s campaignId=%s planningResumeCount=%s previousPlanningError=%s",
        jid,
        _clean(job.get("campaignId")),
        resume_count,
        previous_error,
    )
    return fields


def log_builder1_planning_resume_rejected(job_id: str, reasons: List[str]) -> None:
    logger.info(
        "BUILDER1_PLANNING_RESUME_REJECTED jobId=%s reasons=%s",
        _clean(job_id),
        reasons,
    )


def log_builder1_planning_resume_started(job_id: str, *, campaign_id: str = "") -> None:
    logger.info(
        "BUILDER1_PLANNING_RESUME_STARTED jobId=%s campaignId=%s planningResume=true",
        _clean(job_id),
        _clean(campaign_id),
    )


def log_builder1_planning_resume_completed(
    job_id: str,
    *,
    campaign_id: str = "",
    ok: bool,
    error: str = "",
) -> None:
    logger.info(
        "BUILDER1_PLANNING_RESUME_COMPLETED jobId=%s campaignId=%s ok=%s error=%s planningResume=true",
        _clean(job_id),
        _clean(campaign_id),
        str(ok).lower(),
        _clean(error),
    )


def clear_builder1_planning_resume_requested(job_id: str) -> None:
    update_builder1_job(_clean(job_id), planningResumeRequested=False)
