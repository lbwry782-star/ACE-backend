"""
Builder2 invalid final publication repair — bounded state correction, no media work.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_final_video_verification import verify_published_final_video_artifact
from engine.builder2_media_finalization_contract import (
    final_publication_metadata_valid,
    resolve_raw_runway_artifact_url,
)
from engine.builder2_tournament_store import _read_raw, save_tournament_state
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _first_url(*values: Any) -> str:
    for value in values:
        token = _clean(value)
        if token:
            return token
    return ""


def _already_recoverable_failed_state(state: Dict[str, Any]) -> bool:
    media = _media_bucket(state)
    return (
        _clean(state.get("status")) == "media_finalization_incomplete"
        and bool(state.get("mediaContinuationRequired"))
        and _clean(media.get("mediaResumeStatus")) == "finalization_failed"
        and _clean(media.get("advertisingClosureStatus")) == "failed"
        and not media.get("advertisingClosureRendered")
        and not _first_url(media.get("finalPublicUrl"), media.get("finalVideoWithClosureUrl"))
    )


def assess_invalid_final_publication_repair_eligibility(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
) -> tuple[bool, List[str]]:
    media = _media_bucket(state)
    reasons: List[str] = []
    final_public = _first_url(media.get("finalPublicUrl"), media.get("finalVideoWithClosureUrl"))
    raw_url = resolve_raw_runway_artifact_url(state)
    closure = state.get("advertisingClosure")
    claims_completed = _clean(state.get("status")) == "completed" or (
        bool(media.get("advertisingClosureRendered")) and _clean(media.get("advertisingClosureStatus")) == "completed"
    )
    if not claims_completed and not _already_recoverable_failed_state(state):
        reasons.append("persisted_completion_not_claimed")
    verification = verify_published_final_video_artifact(final_public) if final_public else None
    metadata_valid = final_publication_metadata_valid(media=media, closure_url=final_public)
    contract_satisfied = metadata_valid and verification is not None and verification.post_upload_verification_accepted
    if contract_satisfied:
        reasons.append("accessible_verified_final_present")
    if final_public and verification is not None and verification.post_upload_verification_accepted:
        reasons.append("final_url_accessible")
    if not raw_url:
        reasons.append("raw_runway_unavailable")
    if not is_valid_persisted_winner_development(state):
        reasons.append("accepted_winner_invalid")
    if not isinstance(closure, dict) or not _clean(closure.get("sloganText")):
        reasons.append("advertising_closure_missing")
    if _already_recoverable_failed_state(state):
        return True, ["already_recoverable_failed_state"]
    return not reasons, reasons


def repair_invalid_final_publication_state(job_id: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "repairCompleted": False,
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingCalls": 0,
        "ffmpegCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
        "leaseOperations": 0,
    }
    if not redis_configured():
        report["failureReason"] = "builder2_invalid_final_publication_repair_redis_unconfigured"
        return report

    raw_state = _read_raw(job_id)
    if raw_state is None:
        report["failureReason"] = "builder2_invalid_final_publication_repair_job_not_found"
        return report

    state = deepcopy(raw_state)
    media = _media_bucket(state)
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    job_raw = video_job_get_raw(job_id) or {}
    job_video_url = _first_url(job_raw.get("video_url"), job_raw.get("videoUrl"))
    final_public = _first_url(media.get("finalPublicUrl"), media.get("finalVideoWithClosureUrl"))

    if _already_recoverable_failed_state(state):
        report.update(
            {
                "ok": True,
                "repairCompleted": True,
                "repairIdempotent": True,
                "alreadyRecoverableFailedState": True,
                "recommendedNextAction": "run_finalization_preflight",
            }
        )
        return report

    eligible, blockers = assess_invalid_final_publication_repair_eligibility(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
    )
    report["repairEligibilityBlockers"] = blockers
    if not eligible:
        report["failureReason"] = "builder2_invalid_final_publication_repair_not_eligible"
        report["failureStage"] = "eligibility"
        return report

    broken_url = final_public
    broken_route = classify_url_route_family(broken_url) if broken_url else None
    if broken_url:
        media["brokenFinalPublicationUrl"] = broken_url
        media["invalidFinalPublicationRouteFamily"] = broken_route

    media.pop("finalPublicUrl", None)
    media.pop("finalVideoWithClosureUrl", None)
    media.pop("finalVideoPath", None)
    media["advertisingClosureRendered"] = False
    media["advertisingClosureStatus"] = "failed"
    media["actualFinalVideoDurationSeconds"] = None
    media["finalDurationAccepted"] = False
    media["finalPublicationVerificationAccepted"] = False
    media["finalPublicationDurableStorageConfirmed"] = False
    media["finalPublicationReferencePresent"] = False
    media.pop("finalPublicationBackendKind", None)
    media.pop("finalPublicationUploadedByteCount", None)
    media["mediaResumeStatus"] = "finalization_failed"
    media["finalizationFailureStage"] = "publication_verification"
    media["finalizationFailureCode"] = "final_publication_artifact_missing"
    media["finalizationFailureClass"] = "PublicationVerificationError"
    state["status"] = "media_finalization_incomplete"
    state["mediaContinuationRequired"] = True
    state["advertisingClosureStatus"] = "failed"

    save_tournament_state(job_id, state)
    report.update(
        {
            "ok": True,
            "repairCompleted": True,
            "repairIdempotent": False,
            "redisMutations": 1,
            "brokenFinalPublicationUrlPreserved": bool(broken_url),
            "recommendedNextAction": "run_finalization_preflight",
        }
    )
    return report


def print_invalid_final_publication_repair_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_INVALID_FINAL_PUBLICATION_REPAIR_JOB_ID"))
    if not job_id:
        print(
            json.dumps({"ok": False, "failureReason": "builder2_invalid_final_publication_repair_job_id_missing"}, indent=2),
            flush=True,
        )
        return 1
    logger.info("BUILDER2_INVALID_FINAL_PUBLICATION_REPAIR_START jobId=%s", job_id)
    report = repair_invalid_final_publication_state(job_id)
    print_invalid_final_publication_repair_report(report)
    logger.info(
        "BUILDER2_INVALID_FINAL_PUBLICATION_REPAIR_DONE jobId=%s ok=%s repairCompleted=%s",
        job_id,
        report.get("ok"),
        report.get("repairCompleted"),
    )
    return 0 if report.get("repairCompleted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
