"""
Builder2 Judge-only reasoning resume — dispatches missing Judges without Strategy/Creator/Winner.

Run:
  BUILDER2_JUDGE_ONLY_RESUME_JOB_ID=<jobId> python -m engine.builder2_judge_only_resume

Environment:
  BUILDER2_JUDGE_ONLY_RESUME_JOB_ID
  BUILDER2_JUDGE_ONLY_RESUME_MAX_CALLS (default 6)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from engine.builder2_complete_ad_reasoning_resume import (
    ControlledReasoningCallBudget,
    _execute_judge_generation_resume,
    _initial_report,
    _populate_report_accepted_counts,
    _populate_report_reasoning_calls,
)
from engine.builder2_complete_ad_resume_plan import (
    RESUME_STAGE_JUDGE_GENERATION,
    evaluate_complete_ad_reasoning_executor_preconditions,
    resolve_complete_ad_canonical_resume_plan,
)
from engine.builder2_execution_lease import acquire_job_lease, release_job_lease
from engine.builder2_tournament_recovery import new_worker_token
from engine.builder2_tournament_store import ensure_methodology_compatibility_decided, load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_ONLY_MAX_CALLS = 6


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _clean(os.environ.get(name))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def run_judge_only_reasoning_resume(
    *,
    job_id: str,
    llm_client: Optional[Any] = None,
    max_calls: int = DEFAULT_JUDGE_ONLY_MAX_CALLS,
    acquire_lease: bool = True,
) -> Dict[str, Any]:
    report = _initial_report(job_id=job_id)
    report["maximumAllowedReasoningCalls"] = max_calls
    report["judgeOnlyResume"] = True
    budget = ControlledReasoningCallBudget(max_calls=max_calls)
    worker_token = new_worker_token()
    lease_acquired = False
    state: Optional[Dict[str, Any]] = None

    try:
        if not redis_configured():
            report["failureReason"] = "builder2_judge_only_resume_redis_unconfigured"
            return report

        state = load_tournament_state(job_id)
        if state is None:
            report["failureReason"] = "builder2_judge_only_resume_job_not_found"
            return report

        job_raw = video_job_get_raw(job_id) or {}
        ok, pre_reason, plan = evaluate_complete_ad_reasoning_executor_preconditions(state, job_raw)
        report["tournamentId"] = _clean(state.get("tournamentId")) or None
        _populate_report_accepted_counts(report, state)
        report.update(
            {
                "resolvedResumeStage": plan.get("resolvedResumeStage"),
                "missingCreatorPrototypeIds": list(plan.get("missingCreatorPrototypeIds") or []),
                "missingJudgmentPrototypeIds": list(plan.get("missingJudgmentPrototypeIds") or []),
                "judgeCallsPlanned": int(plan.get("judgeCallsPlanned") or 0),
            }
        )
        if not ok:
            report["failureReason"] = pre_reason
            return report

        if plan.get("resolvedResumeStage") != RESUME_STAGE_JUDGE_GENERATION:
            report["failureReason"] = "builder2_judge_only_resume_not_judge_stage"
            return report
        if plan.get("missingCreatorPrototypeIds"):
            report["failureReason"] = "builder2_judge_only_resume_missing_creators_remain"
            return report

        if acquire_lease and not acquire_job_lease(job_id, worker_token):
            report["failureReason"] = "builder2_judge_only_resume_lease_unavailable"
            report["canResume"] = True
            return report
        lease_acquired = acquire_lease

        ensure_methodology_compatibility_decided(state, is_new_job=False)
        strategy = state["strategyFoundation"]
        report["strategyReused"] = True
        product_name = _clean(strategy.get("productNameResolved") or state.get("productNameResolved") or "Product")
        product_description = _clean(state.get("productDescription") or "Product description")
        language = _clean(state.get("contentLanguage") or state.get("language") or strategy.get("language") or "he")

        return _execute_judge_generation_resume(
            state=state,
            job_id=job_id,
            report=report,
            budget=budget,
            strategy=strategy,
            product_name=product_name,
            product_description=product_description,
            language=language,
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
            llm_client=llm_client,
            lease_acquired=lease_acquired,
            stop_after_judges=True,
        )
    finally:
        if lease_acquired:
            release_job_lease(job_id, worker_token)
        if state is not None:
            _populate_report_accepted_counts(report, state)
            _populate_report_reasoning_calls(report, budget)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_ONLY_RESUME_JOB_ID"))
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_judge_only_resume_job_id_missing"}, indent=2))
        return 1
    max_calls = _env_int("BUILDER2_JUDGE_ONLY_RESUME_MAX_CALLS", DEFAULT_JUDGE_ONLY_MAX_CALLS)
    report = run_judge_only_reasoning_resume(job_id=job_id, max_calls=max_calls)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
