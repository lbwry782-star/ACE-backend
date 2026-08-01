"""
Builder2 offline Creator recovery — revalidate persisted responses under corrected validation.

Run:
  BUILDER2_CREATOR_OFFLINE_RECOVERY_JOB_ID=<jobId> python -m engine.builder2_creator_offline_recovery

Post-recovery read-only inspect:
  BUILDER2_CREATOR_REGRESSION_INSPECT_JOB_ID=<jobId> python -m engine.builder2_creator_regression_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from engine.builder2_complete_ad_creator_recovery import run_offline_creator_recovery_batch
from engine.builder2_creator_regression_inspect import inspect_builder2_creator_regression
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)


def run_builder2_creator_offline_recovery(
    *,
    job_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "dryRun": dry_run,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "judgeCalls": 0,
        "winnerCalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "openAICalls": 0,
        "stateMutated": False,
    }
    if not job_id:
        report["failureReason"] = "missing_job_id"
        return report
    if not redis_configured():
        report["failureReason"] = "redis_unconfigured"
        return report
    state = load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "job_not_found"
        return report
    if dry_run:
        inspect = inspect_builder2_creator_regression(state)
        report.update(
            {
                "ok": True,
                "wouldMutate": any(
                    item.get("offlineRecoveryPossible")
                    for item in inspect.get("creators") or []
                    if isinstance(item, dict)
                ),
                "inspect": inspect,
            }
        )
        return report
    batch = run_offline_creator_recovery_batch(
        state,
        compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
    )
    if batch.get("stateMutated"):
        save_tournament_state(job_id, state)
    post = inspect_builder2_creator_regression(state)
    report.update(batch)
    report["postRecoveryInspect"] = post
    report["ok"] = True
    return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = (os.environ.get("BUILDER2_CREATOR_OFFLINE_RECOVERY_JOB_ID") or "").strip()
    dry_run = (os.environ.get("BUILDER2_CREATOR_OFFLINE_RECOVERY_DRY_RUN") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    report = run_builder2_creator_offline_recovery(job_id=job_id, dry_run=dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
