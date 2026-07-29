"""
Builder2 incomplete-tournament resume — finish one missing Creator/Judge pair and continue.

Run:
  BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_JOB_ID=<jobId> python -m engine.builder2_incomplete_tournament_resume

Environment:
  BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_JOB_ID
  BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_MAX_CALLS (default 3)
  BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_RUN_MEDIA (default true)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_creator_recovery import find_rejected_creator_for_prototype
from engine.builder2_complete_ad_reasoning_resume import (
    run_controlled_complete_ad_reasoning_resume,
    validate_controlled_complete_ad_preconditions,
)
from engine.builder2_creator_slogan_repair_patch import populate_slogan_repair_call_report
from engine.builder2_tournament_completion_gate import (
    accepted_creator_count,
    accepted_judgment_count,
    missing_creator_prototype_ids,
    tournament_resolution_summary,
)
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

ENV_JOB_ID = "BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_JOB_ID"
ENV_MAX_CALLS = "BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_MAX_CALLS"
ENV_RUN_MEDIA = "BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_RUN_MEDIA"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = _clean(os.environ.get(name))
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _clean(os.environ.get(name))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _baseline_reasoning_metrics(state: Dict[str, Any]) -> Dict[str, int]:
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    return {
        "strategy": int(metrics.get("strategyCalls") or 0),
        "creator": int(metrics.get("creatorCalls") or 0)
        + int(metrics.get("creatorRepairCalls") or 0)
        + int(metrics.get("creatorRetryCalls") or 0),
        "judge": int(metrics.get("judgeCalls") or 0)
        + int(metrics.get("judgeRepairCalls") or 0)
        + int(metrics.get("judgeRetryCalls") or 0),
        "winner": int(metrics.get("winnerDevelopmentCalls") or 0),
    }


def _render_commands(*, job_id: str) -> Dict[str, str]:
    return {
        "incompleteTournamentResume": (
            f"BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_JOB_ID={job_id} "
            "python -m engine.builder2_incomplete_tournament_resume"
        ),
        "mediaResumeFallback": (
            f"BUILDER2_MEDIA_RESUME_JOB_ID={job_id} python -m engine.builder2_media_resume"
        ),
    }


def _initial_report(*, job_id: str) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "ok": False,
        "resumeEligible": False,
        "acceptedStrategyReused": False,
        "acceptedCreatorsReusedCount": 0,
        "acceptedJudgmentsReusedCount": 0,
        "missingPrototypeIds": [],
        "rejectedCandidateRepairAvailable": False,
        "thinkSmallNormalCreatorCalls": 0,
        "thinkSmallRepairCalls": 0,
        "thinkSmallJudgeCalls": 0,
        "invocationCreatorNormalCalls": 0,
        "invocationCreatorRepairCalls": 0,
        "persistedCreatorNormalCalls": 0,
        "persistedCreatorRepairCalls": 0,
        "totalCreatorNormalCalls": 0,
        "totalCreatorRepairCalls": 0,
        "additionalPaidRepairAllowed": True,
        "offlineSalvageAttempted": False,
        "offlineSalvageAccepted": False,
        "repeatedStrategyCalls": 0,
        "repeatedAcceptedCreatorCalls": 0,
        "repeatedAcceptedJudgeCalls": 0,
        "winnerSelected": False,
        "mediaPipelineStarted": False,
        "runwaySubmissionCalls": 0,
        "finalJobCompleted": False,
        "failureStage": None,
        "failureCode": None,
        "renderCommands": _render_commands(job_id=job_id),
    }


def run_incomplete_tournament_resume(
    *,
    job_id: str,
    llm_client: Optional[Any] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
    max_calls: int = 3,
    run_media: bool = True,
) -> Dict[str, Any]:
    report = _initial_report(job_id=job_id)
    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureStage"] = "startup"
        report["failureCode"] = "builder2_incomplete_tournament_resume_job_not_found"
        return report

    report["tournamentId"] = _clean(state.get("tournamentId"))
    if not redis_configured() and tournament_state is None:
        report["failureStage"] = "startup"
        report["failureCode"] = "builder2_incomplete_tournament_resume_redis_unconfigured"
        return report

    job_raw = {} if tournament_state is not None else (video_job_get_raw(job_id) or {})
    ok, pre_reason = validate_controlled_complete_ad_preconditions(state, job_raw)
    report["resumeEligible"] = ok
    if not ok:
        report["failureStage"] = "preconditions"
        report["failureCode"] = pre_reason
        summary = tournament_resolution_summary(state)
        report["missingPrototypeIds"] = list(summary.get("missingCreatorPrototypeIds") or [])
        report["acceptedCreatorsReusedCount"] = int(summary.get("acceptedCreatorCount") or 0)
        report["acceptedJudgmentsReusedCount"] = int(summary.get("acceptedJudgmentCount") or 0)
        return report

    missing = list(missing_creator_prototype_ids(state))
    report["missingPrototypeIds"] = missing
    report["acceptedCreatorsReusedCount"] = accepted_creator_count(state)
    report["acceptedJudgmentsReusedCount"] = accepted_judgment_count(state)
    report["acceptedStrategyReused"] = True

    missing_prototype = missing[0] if len(missing) == 1 else ""
    rejected_payload = find_rejected_creator_for_prototype(state, missing_prototype) if missing_prototype else None
    report["rejectedCandidateRepairAvailable"] = bool(
        isinstance(rejected_payload, dict)
        and isinstance(rejected_payload.get("parsed"), dict)
        and rejected_payload.get("parsed")
    )

    baseline_metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    baseline_creator_normal = int(baseline_metrics.get("creatorCalls") or 0)
    baseline_creator_repair = int(baseline_metrics.get("creatorRepairCalls") or 0)

    baseline = _baseline_reasoning_metrics(state)
    reasoning_report = run_controlled_complete_ad_reasoning_resume(
        job_id=job_id,
        llm_client=llm_client,
        tournament_state=state,
        max_calls=max_calls,
        stop_before_media=not run_media,
    )
    state = load_tournament_state(job_id) or state
    after = _baseline_reasoning_metrics(state)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}

    creator_normal_delta = max(0, int(metrics.get("creatorCalls") or 0) - baseline_creator_normal)
    creator_repair_delta = max(0, int(metrics.get("creatorRepairCalls") or 0) - baseline_creator_repair)
    creator_delta = creator_normal_delta + creator_repair_delta
    judge_delta = max(0, after["judge"] - baseline["judge"])
    strategy_delta = max(0, after["strategy"] - baseline["strategy"])
    winner_delta = max(0, after["winner"] - baseline["winner"])

    report["repeatedStrategyCalls"] = strategy_delta
    report["repeatedAcceptedCreatorCalls"] = max(0, creator_delta - (1 if missing_prototype == "think_small" else 0))
    report["repeatedAcceptedJudgeCalls"] = max(0, judge_delta - (1 if missing_prototype == "think_small" else 0))

    if missing_prototype == "think_small":
        populate_slogan_repair_call_report(
            state,
            report,
            prototype_id="think_small",
            invocation_creator_normal_calls=creator_normal_delta,
            invocation_creator_repair_calls=creator_repair_delta,
        )
        report["thinkSmallNormalCreatorCalls"] = report["totalCreatorNormalCalls"]
        report["thinkSmallRepairCalls"] = report["totalCreatorRepairCalls"]
        report["thinkSmallJudgeCalls"] = min(1, judge_delta)
    else:
        report["thinkSmallNormalCreatorCalls"] = int(baseline_metrics.get("creatorCalls") or 0)
        report["thinkSmallRepairCalls"] = int(baseline_metrics.get("creatorRepairCalls") or 0)
        report["thinkSmallJudgeCalls"] = 0

    report["winnerSelected"] = bool(reasoning_report.get("finalWinnerCandidateId") or state.get("winnerCandidateId"))
    report["acceptedCreatorsReusedCount"] = accepted_creator_count(state)
    report["acceptedJudgmentsReusedCount"] = accepted_judgment_count(state)

    if not reasoning_report.get("ok"):
        report["failureStage"] = reasoning_report.get("failureStage") or "reasoning_resume"
        report["failureCode"] = reasoning_report.get("failureReason")
        return report

    if run_media:
        from engine.builder2_media_resume import run_one_media_resume

        media_report = run_one_media_resume(job_id=job_id, dry_run=False, tournament_state=state)
        report["mediaPipelineStarted"] = bool(media_report.get("readyForMediaResume") or media_report.get("mediaReused"))
        report["runwaySubmissionCalls"] = int(media_report.get("runwaySubmissionCalls") or 0)
        report["finalJobCompleted"] = bool(media_report.get("jobCompleted"))
        if not media_report.get("ok"):
            report["failureStage"] = media_report.get("failureStage") or "media_pipeline"
            report["failureCode"] = media_report.get("failureReason")
            return report

    report["ok"] = True
    logger.info(
        "BUILDER2_INCOMPLETE_TOURNAMENT_RESUME_REUSED jobId=%s tournamentId=%s acceptedCreatorsReused=%s "
        "acceptedJudgmentsReused=%s missingPrototypeIds=%s winnerSelected=%s",
        job_id,
        report.get("tournamentId") or "(none)",
        report["acceptedCreatorsReusedCount"],
        report["acceptedJudgmentsReusedCount"],
        ",".join(report.get("missingPrototypeIds") or []),
        str(report["winnerSelected"]).lower(),
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get(ENV_JOB_ID))
    if not job_id:
        print(
            json.dumps(
                {"ok": False, "failureCode": "builder2_incomplete_tournament_resume_job_id_missing"},
                indent=2,
            )
        )
        return 1
    max_calls = _env_int(ENV_MAX_CALLS, 3)
    run_media = _env_bool(ENV_RUN_MEDIA, True)
    report = run_incomplete_tournament_resume(job_id=job_id, max_calls=max_calls, run_media=run_media)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
