"""
Builder2 final-video artifact inspector — read-only bounded URL verification.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_media_finalization_contract import (
    final_publication_metadata_valid,
    resolve_raw_runway_artifact_url,
    validate_builder2_media_completion_contract,
)
from engine.builder2_final_video_verification import verify_published_final_video_artifact
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_store import _read_raw
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

PRODUCTION_BROKEN_FINAL_TOKEN = "42228511edd94fa18eccedf4d39db8e0"


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


def _recommend_next_action(
    *,
    publication_contract_satisfied: bool,
    persisted_completion_contradicts_artifact: bool,
    raw_runway_recoverable: bool,
    final_url_accessible: bool,
) -> str:
    if publication_contract_satisfied and final_url_accessible:
        return "use_existing_verified_final"
    if persisted_completion_contradicts_artifact and raw_runway_recoverable:
        return "repair_invalid_final_publication_state"
    if not final_url_accessible and raw_runway_recoverable:
        return "repair_invalid_final_publication_state"
    if not raw_runway_recoverable:
        return "unrecoverable_without_regeneration"
    return "inspect_publication_storage"


def inspect_builder2_final_video_artifact(job_id: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "inspectionCompleted": False,
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
        report["failureReason"] = "builder2_final_video_artifact_inspect_redis_unconfigured"
        return report

    with read_only_builder2_inspection() as mutation_counter:
        raw_state = _read_raw(job_id)
        job_raw = video_job_get_raw(job_id) or {}
        report["redisMutations"] = mutation_counter.redis_mutations
        if raw_state is None:
            report["jobFound"] = False
            report["inspectionCompleted"] = True
            report["ok"] = True
            return report

        state = deepcopy(raw_state)
        media = _media_bucket(state)
        plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
        job_video_url = _first_url(job_raw.get("video_url"), job_raw.get("videoUrl"))
        final_public = _first_url(media.get("finalPublicUrl"), media.get("finalVideoWithClosureUrl"))
        raw_url = resolve_raw_runway_artifact_url(state)
        contract_ok, _, contract_failures = validate_builder2_media_completion_contract(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
            require_job_video_url_match=False,
        )
        metadata_valid = final_publication_metadata_valid(media=media, closure_url=final_public)
        verification = verify_published_final_video_artifact(
            final_public,
            expected_byte_count=media.get("finalPublicationUploadedByteCount"),
            durable_storage_confirmed=bool(media.get("finalPublicationDurableStorageConfirmed")),
        )
        publication_completed_persisted = bool(
            media.get("advertisingClosureRendered")
            and _clean(media.get("advertisingClosureStatus")) == "completed"
            and _clean(state.get("status")) == "completed"
        )
        publication_contract_satisfied = contract_ok and metadata_valid and verification.post_upload_verification_accepted
        persisted_completion_contradicts_artifact = publication_completed_persisted and not verification.post_upload_verification_accepted

        report.update(
            {
                "jobFound": True,
                "finalPublicUrl": final_public or None,
                "finalUrlRouteFamily": classify_url_route_family(final_public) if final_public else None,
                **verification.to_report_dict(),
                "publicationCompletedPersisted": publication_completed_persisted,
                "publicationContractActuallySatisfied": publication_contract_satisfied,
                "persistedCompletionContradictsArtifact": persisted_completion_contradicts_artifact,
                "completionContractSatisfied": contract_ok,
                "completionContractFailureReasons": contract_failures,
                "finalPublicationMetadataValid": metadata_valid,
                "rawRunwayRecoverable": bool(raw_url),
                "recommendedNextAction": _recommend_next_action(
                    publication_contract_satisfied=publication_contract_satisfied,
                    persisted_completion_contradicts_artifact=persisted_completion_contradicts_artifact,
                    raw_runway_recoverable=bool(raw_url),
                    final_url_accessible=verification.final_url_accessible,
                ),
                "inspectionCompleted": True,
                "ok": True,
            }
        )
        return report


def print_builder2_final_video_artifact_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_FINAL_VIDEO_ARTIFACT_INSPECT_JOB_ID"))
    if not job_id:
        print(
            json.dumps({"ok": False, "failureReason": "builder2_final_video_artifact_inspect_job_id_missing"}, indent=2),
            flush=True,
        )
        return 1
    logger.info("BUILDER2_FINAL_VIDEO_ARTIFACT_INSPECT_START jobId=%s", job_id)
    report = inspect_builder2_final_video_artifact(job_id)
    print_builder2_final_video_artifact_report(report)
    logger.info(
        "BUILDER2_FINAL_VIDEO_ARTIFACT_INSPECT_DONE jobId=%s accessible=%s contradicts=%s",
        job_id,
        report.get("finalUrlAccessible"),
        report.get("persistedCompletionContradictsArtifact"),
    )
    return 0 if report.get("inspectionCompleted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
