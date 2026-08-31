"""
Builder1 integrity-failure recovery — revalidate and resume without replanning.

Reuses persisted integrityFailureDiagnostic.rejectedPlan after deterministic
integrity checks pass. Zero OpenAI planning calls.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from engine.builder1_campaign_integrity import validate_builder1_campaign_integrity
from engine.builder1_campaign_store import (
    CampaignStoreError,
    create_campaign_session,
    get_campaign_session_raw,
)
from engine.builder1_consolidated_stages import Builder1UpstreamSnapshot
from engine.builder1_integrity_diagnostics import (
    INTEGRITY_DIAGNOSTIC_JOB_FIELD,
    get_integrity_failure_diagnostic,
)
from engine.builder1_jobs_store import get_builder1_job, update_builder1_job
from engine.builder1_plan_spec import series_plan_from_store_dict

logger = logging.getLogger(__name__)

_BLOCKING_PAID_STAGE_STATUSES = frozenset({"submitted", "outcome_unknown"})
_PLANNING_FAILURE_MARKERS = frozenset(
    {
        "campaign_integrity_failed",
        "campaign_visibility_integrity_failed",
        "planning_failed",
    }
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _job_planning_failed(job: Mapping[str, Any]) -> bool:
    status = _clean(job.get("status")).lower()
    if status not in {"error", "failed"}:
        err = _clean(job.get("error"))
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        message = _clean(result.get("message") if isinstance(result, dict) else "")
        combined = f"{err} {message}".lower()
        if not any(marker in combined for marker in _PLANNING_FAILURE_MARKERS):
            return False
    err = _clean(job.get("error"))
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    message = _clean(result.get("message") if isinstance(result, dict) else "")
    combined = f"{err} {message}".lower()
    return any(marker in combined for marker in _PLANNING_FAILURE_MARKERS)


def _campaign_has_started_media(raw: Optional[Mapping[str, Any]]) -> bool:
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
    if paid_stage and paid_stage not in {
        "strategy_slogan_stage",
        "conceptual_stage",
        "brand_physical",
        "graphic_system",
        "series_ads",
        "product_name_resolution",
    }:
        return True
    paid_status = _clean(raw.get("lastPaidStageStatus"))
    if paid_stage and paid_status in _BLOCKING_PAID_STAGE_STATUSES:
        return True
    return False


def upstream_snapshot_from_series_plan(plan) -> Builder1UpstreamSnapshot:
    lineage = (plan.planning_internals or {}).get("conceptualLineage") or {}
    graphic = plan.graphic_generator
    layout = getattr(graphic, "layout_template", "") if graphic else ""
    device = getattr(graphic, "recurring_graphic_device", "") if graphic else ""
    return Builder1UpstreamSnapshot(
        product_name_resolved=plan.product_name_resolved,
        strategic_problem=plan.strategic_problem,
        relative_advantage=plan.relative_advantage,
        brand_slogan=plan.brand_slogan,
        implied_action=plan.slogan_action,
        selected_slogan_id=str(lineage.get("sourceSloganCandidateId") or "S01").upper(),
        conceptual_generator=plan.conceptual_generator,
        selected_conceptual_id=str(lineage.get("selectedConceptCandidateId") or "C01").upper(),
        physical_generator=plan.physical_generator,
        graphic_layout_template=layout,
        graphic_recurring_device=device,
    )


def revalidate_rejected_plan_dict(rejected_plan: Mapping[str, Any]) -> Dict[str, Any]:
    plan = series_plan_from_store_dict(dict(rejected_plan))
    upstream = upstream_snapshot_from_series_plan(plan)
    integrity = validate_builder1_campaign_integrity(
        plan,
        upstream=upstream,
        detected_language=plan.detected_language,
    )
    return {
        "ok": integrity.ok,
        "reasons": integrity.reasons,
        "integrityDetails": integrity.integrity_details or [],
    }


def assess_builder1_integrity_recovery(job_id: str) -> Dict[str, Any]:
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid,
        "campaignId": "",
        "originalFailureReasons": [],
        "rejectedPlanPresent": False,
        "planningCallsAlreadySpent": None,
        "revalidation": None,
        "remainingIntegrityReasons": [],
        "campaignSessionExists": False,
        "imageProviderState": "unknown",
        "recoveryEligible": False,
        "paidCallsPerformed": 0,
        "eligibilityFailures": [],
    }
    job = get_builder1_job(jid)
    if job is None:
        report["eligibilityFailures"].append("job_not_found")
        return report

    report["campaignId"] = _clean(job.get("campaignId"))
    diagnostic = get_integrity_failure_diagnostic(jid)
    if diagnostic is None:
        raw_diag = job.get(INTEGRITY_DIAGNOSTIC_JOB_FIELD)
        diagnostic = dict(raw_diag) if isinstance(raw_diag, dict) else None
    if diagnostic is None:
        report["eligibilityFailures"].append("integrity_failure_diagnostic_missing")
        return report

    report["originalFailureReasons"] = list(diagnostic.get("reasons") or [])
    rejected = diagnostic.get("rejectedPlan")
    if not isinstance(rejected, dict) or not rejected:
        report["eligibilityFailures"].append("rejected_plan_missing")
        return report
    report["rejectedPlanPresent"] = True

    if not _job_planning_failed(job):
        report["eligibilityFailures"].append("job_not_planning_failure")
    if _clean(job.get("lastPaidStageStatus")) != "succeeded":
        report["eligibilityFailures"].append("last_paid_stage_not_succeeded")
    if report["campaignId"] and _clean(job.get("campaignId")) != report["campaignId"]:
        report["eligibilityFailures"].append("campaign_id_mismatch")

    campaign_raw = get_campaign_session_raw(report["campaignId"]) if report["campaignId"] else None
    report["campaignSessionExists"] = campaign_raw is not None
    if _campaign_has_started_media(campaign_raw):
        report["eligibilityFailures"].append("campaign_media_already_started")
        report["imageProviderState"] = "started_or_inflight"

    revalidation = revalidate_rejected_plan_dict(rejected)
    report["revalidation"] = revalidation
    report["remainingIntegrityReasons"] = list(revalidation.get("reasons") or [])
    if not revalidation.get("ok"):
        report["eligibilityFailures"].append("revalidation_failed")

    report["recoveryEligible"] = not report["eligibilityFailures"]
    return report


def apply_builder1_integrity_recovery(
    job_id: str,
    *,
    enqueue_image_pipeline: bool = False,
) -> Dict[str, Any]:
    report = assess_builder1_integrity_recovery(job_id)
    report["applied"] = False
    report["sessionCreated"] = False
    report["imagePipelineEnqueued"] = False
    if not report.get("recoveryEligible"):
        return report

    diagnostic = get_integrity_failure_diagnostic(job_id)
    assert diagnostic is not None
    rejected = diagnostic.get("rejectedPlan")
    assert isinstance(rejected, dict)
    plan = series_plan_from_store_dict(dict(rejected))
    campaign_id = _clean(report.get("campaignId"))
    target_ad_count = int(plan.ad_count or report.get("targetAdCount") or 2)

    job = get_builder1_job(job_id) or {}
    ownership_fields = {
        k: str(v)
        for k, v in job.items()
        if k
        in {
            "ownerContextRef",
            "ownerContextVersion",
            "ownerContextPresent",
            "builder",
            "builder1ContractVersion",
            "originalRequestFingerprint",
        }
        and v is not None
    }

    if not report.get("campaignSessionExists"):
        create_campaign_session(
            campaign_id=campaign_id,
            plan=plan,
            target_ad_count=target_ad_count,
            ownership_fields=ownership_fields or None,
        )
        report["sessionCreated"] = True

    update_builder1_job(
        job_id,
        status="running",
        stage="building_prompts",
        error="",
        completedAds=0,
        totalAds=target_ad_count,
        targetAdCount=target_ad_count,
        planRevision=1,
        retryAdIndex=1,
    )
    report["applied"] = True

    if enqueue_image_pipeline:
        try:
            import app as ace_app

            ace_app._builder1_executor.submit(
                ace_app._builder1_run_resume_integrity_recovered_job,
                job_id,
                campaign_id,
                target_ad_count,
            )
            report["imagePipelineEnqueued"] = True
        except Exception as exc:
            report["imagePipelineEnqueueError"] = str(exc)
            logger.error(
                "BUILDER1_INTEGRITY_RECOVERY_ENQUEUE_FAILED jobId=%s err=%s",
                job_id,
                exc,
            )

    return report
