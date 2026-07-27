"""
Builder2 complete-ad offline Creator revalidation — zero model calls.

Run:
  BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_JOB_ID=<jobId> \
  BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_CANDIDATE_ID=<candidateId> \
  python -m engine.builder2_complete_ad_creator_revalidate
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from engine.builder2_complete_ad_creator_recovery import (
    can_offline_revalidate_rejected_creator,
    load_rejected_creator_parsed_response,
    offline_revalidate_and_accept_rejected_creator,
)
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)


def run_offline_creator_revalidation(
    *,
    job_id: str,
    candidate_id: str,
    product_name: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "candidateId": candidate_id,
        "ok": False,
        "accepted": False,
        "openAICalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "ffmpegCalls": 0,
        "redisMutations": 0,
    }
    if not job_id or not candidate_id:
        report["failureReason"] = "builder2_complete_ad_creator_revalidate_missing_ids"
        return report
    if not redis_configured():
        report["failureReason"] = "builder2_complete_ad_creator_revalidate_redis_unconfigured"
        return report
    state = load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_complete_ad_creator_revalidate_job_not_found"
        return report
    payload = load_rejected_creator_parsed_response(state, candidate_id)
    if payload is None:
        report["failureReason"] = "builder2_complete_ad_creator_revalidate_parsed_response_missing"
        return report
    ok, reason = can_offline_revalidate_rejected_creator(
        state,
        candidate_id=candidate_id,
        product_name=product_name,
    )
    report["offlineRevalidatable"] = ok
    if not ok:
        report["failureReason"] = reason
        return report
    if dry_run:
        report["ok"] = True
        report["dryRun"] = True
        return report
    offline_revalidate_and_accept_rejected_creator(
        state,
        candidate_id=candidate_id,
        product_name=product_name,
    )
    save_tournament_state(job_id, state)
    report["ok"] = True
    report["accepted"] = True
    report["redisMutations"] = 1
    return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = (os.environ.get("BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_JOB_ID") or "").strip()
    candidate_id = (os.environ.get("BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_CANDIDATE_ID") or "").strip()
    dry_run = (os.environ.get("BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_DRY_RUN") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    report = run_offline_creator_revalidation(
        job_id=job_id,
        candidate_id=candidate_id,
        product_name=(os.environ.get("BUILDER2_COMPLETE_AD_CREATOR_REVALIDATE_PRODUCT_NAME") or "").strip(),
        dry_run=dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
