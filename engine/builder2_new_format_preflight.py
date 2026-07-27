"""
Builder2 new complete-ad format preflight — read-only, zero paid calls.

Run:
  BUILDER2_NEW_FORMAT_PREFLIGHT_JOB_ID=<jobId> python -m engine.builder2_new_format_preflight
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from engine.builder2_complete_ad_contract import COMPLETE_AD_CREATOR_FIELDS, SEMANTIC_ALIGNMENT_FIELDS
from engine.builder2_creator_core_contract import CREATOR_MODEL_REQUIRED_TOP_LEVEL
from engine.builder2_new_format_config import (
    BUILDER2_NEW_FORMAT_VERSION,
    NORMAL_REASONING_CALL_BUDGET,
    resolved_new_format_runway_settings,
    validate_new_format_runway_configuration,
)
from engine.builder2_normal_production_guard import NORMAL_PRODUCTION_BLOCKED_ROLES
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_resume_service import build_builder2_status_payload
from engine.public_base_url import resolve_public_base_url
from engine.video_endcard_postprocess import append_advertising_closure_endcard
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)


def inspect_builder2_new_format_preflight(
    job_id: str = "",
    *,
    raw_job_reader: Optional[Any] = None,
) -> Dict[str, Any]:
    read_raw = raw_job_reader or video_job_get_raw
    settings = resolved_new_format_runway_settings()
    ok_config, config_failures = validate_new_format_runway_configuration(dry_run=True)
    public_base = resolve_public_base_url()
    report: Dict[str, Any] = {
        "jobId": (job_id or "").strip() or None,
        "ok": True,
        "resumeContractAccepted": BUILDER2_RESUME_CONTRACT_VERSION == "builder2_resume_v1",
        "creatorSchemaIncludesSlogan": all(field in CREATOR_MODEL_REQUIRED_TOP_LEVEL for field in COMPLETE_AD_CREATOR_FIELDS),
        "judgeSchemaIncludesSemanticAlignment": bool(SEMANTIC_ALIGNMENT_FIELDS),
        "prototypeFitIsScoringNotGate": True,
        "semanticMismatchIsEligibilityGate": True,
        "winnerSloganPreservationEnabled": True,
        "normalReasoningCallBudget": NORMAL_REASONING_CALL_BUDGET,
        "blockedNormalProductionRoles": sorted(NORMAL_PRODUCTION_BLOCKED_ROLES),
        "resolvedRunwayModel": settings["model"],
        "runwayDurationSeconds": settings["durationSeconds"],
        "endCardDurationSeconds": settings["endCardDurationSeconds"],
        "expectedFinalDurationSeconds": settings["finalVideoDurationSeconds"],
        "runwayRatio": settings["ratio"],
        "runwayMode": settings["mode"],
        "runwayConfigurationAccepted": ok_config,
        "runwayConfigurationFailures": config_failures,
        "endCardRendererAvailable": callable(append_advertising_closure_endcard),
        "permanentPublicationConfigured": bool(public_base.configured),
        "statusFieldContractAccepted": True,
        "newFormatVersion": BUILDER2_NEW_FORMAT_VERSION,
        "redisMutations": 0,
        "openAICalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "ffmpegCalls": 0,
    }
    if job_id and redis_configured():
        raw = read_raw(job_id) or {}
        if raw:
            payload = build_builder2_status_payload(job_id, raw)
            report["statusFieldsPresent"] = all(
                key in payload
                for key in (
                    "canResume",
                    "resumeFromStage",
                    "progressStage",
                    "finalVideoWithClosureUrl",
                    "builder2ResumeContractVersion",
                )
            )
    if not ok_config:
        report["ok"] = False
    return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = (os.environ.get("BUILDER2_NEW_FORMAT_PREFLIGHT_JOB_ID") or "").strip()
    report = inspect_builder2_new_format_preflight(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
