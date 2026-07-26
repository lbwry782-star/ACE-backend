"""
Builder2 Winner-only resume — one Winner Development call after persisted judgments.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from engine.builder2_accepted_creator_store import (
    ACCEPTED_CREATOR_INDEX_KEY,
    backfill_accepted_creator_index,
)
from engine.builder2_accepted_judgment_store import (
    audit_reusable_accepted_judgment,
    backfill_accepted_judgment_index,
)
from engine.builder2_reasoning_resume import validate_reasoning_resume_state
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_metrics import ensure_metrics, MetricsTimer
from engine.builder2_tournament_store import (
    ensure_methodology_compatibility_decided,
    load_tournament_state,
    save_tournament_state,
)
from engine.builder2_winner_development import develop_builder2_winning_candidate
from engine.builder2_winner_development_diagnostics import PUBLIC_FAILURE_CODE, raise_public_winner_failure, STAGE_PERSISTENCE
from engine.builder2_winner_persistence import (
    is_valid_persisted_winner_development,
    persist_winner_development_atomically,
)
from engine.builder2_winner_resume_guard import RESUME_ISOLATION_ERROR, WinnerResumeIsolationGuard

logger = logging.getLogger(__name__)

DEFAULT_WINNER_RESUME_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _winner_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _metric_snapshot(metrics: Dict[str, Any]) -> Dict[str, int]:
    return {
        "strategyCalls": int(metrics.get("strategyCalls") or 0),
        "creatorCalls": int(metrics.get("creatorCalls") or 0),
        "judgeCalls": int(metrics.get("judgeCalls") or 0),
        "winnerNormalCalls": int(metrics.get("winnerNormalCalls") or 0),
        "winnerRepairCalls": int(metrics.get("winnerRepairCalls") or 0),
        "winnerRetryCalls": int(metrics.get("winnerRetryCalls") or 0),
        "winnerDevelopmentCalls": int(metrics.get("winnerDevelopmentCalls") or 0),
    }


def _metric_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: max(0, after.get(key, 0) - before.get(key, 0)) for key in before}


def _initial_report(*, job_id: str) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "acceptedCreatorCount": 0,
        "acceptedJudgmentCount": 0,
        "reusedJudgmentCount": 0,
        "winnerCandidateId": None,
        "winnerPrototypeId": None,
        "winnerScore": None,
        "winnerDevelopmentAccepted": False,
        "winnerReused": False,
        "winnerNormalCalls": 0,
        "winnerRepairCalls": 0,
        "winnerRetryCalls": 0,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "judgeCalls": 0,
        "startImageCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "mediaContinuationRequired": False,
        "failureStage": None,
        "failureReason": None,
        "missingPaths": [],
        "ok": False,
    }


def _resolve_winner_candidate_id(state: Dict[str, Any]) -> str:
    persisted = str(state.get("winnerCandidateId") or "").strip()
    if persisted:
        cand = (state.get("candidates") or {}).get(persisted)
        if isinstance(cand, dict) and cand.get("eligible") and cand.get("judgeStatus") == "accepted":
            return persisted
    return select_global_winner(state)


def _winner_score(state: Dict[str, Any], candidate_id: str) -> Optional[int]:
    cand = (state.get("candidates") or {}).get(candidate_id)
    if not isinstance(cand, dict):
        return None
    total = cand.get("totalScore")
    return int(total) if total is not None else None


def run_one_winner_resume(
    *,
    job_id: str,
    llm_client: Optional[Any] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    WinnerResumeIsolationGuard.begin()
    report = _initial_report(job_id=job_id)

    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_winner_resume_job_not_found"
        WinnerResumeIsolationGuard.end()
        return report

    valid, missing = validate_reasoning_resume_state(state)
    report["missingPaths"] = missing
    if not valid:
        report["failureReason"] = "builder2_winner_resume_state_incomplete"
        WinnerResumeIsolationGuard.end()
        return report

    backfill_accepted_creator_index(state)
    backfill_accepted_judgment_index(state)
    creator_index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    report["acceptedCreatorCount"] = len(creator_index)

    strategy = state.get("strategyFoundation") or {}
    ensure_metrics(state)
    metrics_before = _metric_snapshot(state.get("metrics") or {})
    ensure_methodology_compatibility_decided(state, is_new_job=False)
    compatibility_mode = bool(state.get("methodologyCompatibilityMode"))

    candidate_ids = sorted(str(key) for key in creator_index.keys())
    reused_judgments = 0
    for candidate_id in candidate_ids:
        snapshot = creator_index.get(candidate_id)
        if not isinstance(snapshot, dict):
            report["failureReason"] = "builder2_winner_resume_judgments_incomplete"
            report["missingPaths"] = [f"acceptedCreatorCandidates.{candidate_id}"]
            WinnerResumeIsolationGuard.end()
            return report
        reusable, _reason = audit_reusable_accepted_judgment(
            state,
            candidate_id=candidate_id,
            creator_snapshot=snapshot,
            strategy_foundation=strategy,
            compatibility_mode=compatibility_mode,
        )
        if reusable:
            reused_judgments += 1
        else:
            report["failureReason"] = "builder2_winner_resume_judgments_incomplete"
            report["missingPaths"] = [f"acceptedJudgments.{candidate_id}"]
            WinnerResumeIsolationGuard.end()
            return report

    report["acceptedJudgmentCount"] = reused_judgments
    report["reusedJudgmentCount"] = reused_judgments

    if is_valid_persisted_winner_development(state):
        winner_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or "")
        winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
        report["winnerCandidateId"] = winner_id
        report["winnerPrototypeId"] = state.get("winnerDevelopmentPrototypeId") or winner_rec.get("prototypeId")
        report["winnerScore"] = _winner_score(state, winner_id)
        report["winnerDevelopmentAccepted"] = True
        report["winnerReused"] = True
        report["mediaContinuationRequired"] = True
        report["ok"] = True
        WinnerResumeIsolationGuard.end()
        return report

    try:
        winner_id = _resolve_winner_candidate_id(state)
        winner_rec = state["candidates"][winner_id]
        prototype_id = str(winner_rec.get("prototypeId") or "")
        state["winnerCandidateId"] = winner_id
        report["winnerCandidateId"] = winner_id
        report["winnerPrototypeId"] = prototype_id
        report["winnerScore"] = _winner_score(state, winner_id)

        WinnerResumeIsolationGuard.enable_winner_development()
        WinnerResumeIsolationGuard.assert_safe_before_paid_call()

        product_name = str(state.get("productName") or state.get("productNameResolved") or "Resume Product")
        product_description = str(state.get("productDescription") or "Resume description")
        language = str(state.get("contentLanguage") or state.get("language") or "he")
        runway_mode = builder2_runway_generation_mode(resolve_builder2_runway_video_model())
        judgment_rec = (state.get("judgments") or {}).get(winner_rec.get("judgmentId") or "")
        winning_judgment = (judgment_rec or {}).get("judgment") or {}
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}

        state["status"] = "winner_developing"
        save_tournament_state(job_id, state)

        timer = MetricsTimer()
        winner_plan = develop_builder2_winning_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            prototype_id=prototype_id,
            runway_mode=runway_mode,
            llm_client=llm_client,
            compatibility_mode=compatibility_mode,
            state=state,
        )
        try:
            persist_winner_development_atomically(
                state,
                candidate_id=winner_id,
                prototype_id=prototype_id,
                winner_plan=winner_plan,
                winning_candidate=winning_candidate,
                preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
                compatibility_mode=compatibility_mode,
            )
        except Exception as exc:
            raise_public_winner_failure(
                exc,
                state=state,
                stage=STAGE_PERSISTENCE,
                top_level_keys=sorted(winner_plan.keys()),
            )
        state["status"] = "winner_plan_complete"
        state["mediaContinuationRequired"] = True
        save_tournament_state(job_id, state)

        report["winnerDevelopmentAccepted"] = True
        report["winnerReused"] = False
        report["mediaContinuationRequired"] = True
        report["ok"] = True
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else PUBLIC_FAILURE_CODE)
        report["failureReason"] = reason
        failure = state.get("winnerDevelopmentFailure") if state else None
        if isinstance(failure, dict):
            report["failureStage"] = failure.get("stage")
            if reason == PUBLIC_FAILURE_CODE and failure.get("preciseReason"):
                report["failureReason"] = PUBLIC_FAILURE_CODE
        save_tournament_state(job_id, state)
    finally:
        metrics_after = _metric_snapshot(state.get("metrics") or {}) if state else metrics_before
        delta = _metric_delta(metrics_before, metrics_after)
        report["winnerNormalCalls"] = delta["winnerNormalCalls"]
        report["winnerRepairCalls"] = delta["winnerRepairCalls"]
        report["winnerRetryCalls"] = delta["winnerRetryCalls"]
        report["strategyCalls"] = delta["strategyCalls"]
        report["creatorCalls"] = delta["creatorCalls"]
        report["judgeCalls"] = delta["judgeCalls"]
        report["startImageCalls"] = 0
        report["runwayCalls"] = 0
        report["ffmpegCalls"] = 0
        WinnerResumeIsolationGuard.end()

    return report


def print_winner_resume_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "jobId",
            "acceptedCreatorCount",
            "acceptedJudgmentCount",
            "reusedJudgmentCount",
            "winnerCandidateId",
            "winnerPrototypeId",
            "winnerScore",
            "winnerDevelopmentAccepted",
            "winnerReused",
            "winnerNormalCalls",
            "winnerRepairCalls",
            "winnerRetryCalls",
            "strategyCalls",
            "creatorCalls",
            "judgeCalls",
            "startImageCalls",
            "runwayCalls",
            "ffmpegCalls",
            "mediaContinuationRequired",
            "failureStage",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _winner_env("BUILDER2_WINNER_RESUME_JOB_ID", DEFAULT_WINNER_RESUME_JOB_ID)
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_winner_resume_job_id_missing"}, indent=2))
        return 1

    logger.info("BUILDER2_WINNER_RESUME_START jobId=%s", job_id)
    report = run_one_winner_resume(job_id=job_id)
    print_winner_resume_report(report)
    logger.info(
        "BUILDER2_WINNER_RESUME_DONE jobId=%s ok=%s winnerDevelopmentAccepted=%s winnerNormalCalls=%s failureStage=%s",
        job_id,
        report.get("ok"),
        report.get("winnerDevelopmentAccepted"),
        report.get("winnerNormalCalls"),
        report.get("failureStage"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
