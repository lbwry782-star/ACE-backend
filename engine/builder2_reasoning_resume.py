"""
Builder2 reasoning-only tournament resume — Judge missing candidates and Winner Development only.

Run: python -m engine.builder2_reasoning_resume

Environment:
  BUILDER2_REASONING_RESUME_JOB_ID

Does not regenerate Strategy, Creator, or media. Does not connect to the ordinary queue or recovery scan.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_accepted_creator_store import (
    ACCEPTED_CREATOR_INDEX_KEY,
    backfill_accepted_creator_index,
    load_accepted_creator_candidate,
)
from engine.builder2_accepted_judgment_store import (
    audit_reusable_accepted_judgment,
    backfill_accepted_judgment_index,
    persist_accepted_judgment,
)
from engine.builder2_judge import judge_candidate
from engine.builder2_judge_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE as JUDGE_SYSTEMIC_FAILURE_CODE,
    assert_judge_contract_available,
    is_judge_contract_circuit_breaker_tripped,
    record_judge_process_contract_failure,
)
from engine.builder2_prototypes import require_prototype
from engine.builder2_reasoning_resume_guard import (
    RESUME_ISOLATION_ERROR,
    ReasoningResumeIsolationGuard,
)
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import (
    ensure_methodology_compatibility_decided,
    load_tournament_state,
    record_process_failure_tag,
    save_tournament_state,
)
from engine.builder2_winner_development import develop_builder2_winning_candidate
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically

logger = logging.getLogger(__name__)

DEFAULT_RESUME_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _resume_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_snapshot(metrics: Dict[str, Any]) -> Dict[str, int]:
    return {
        "strategyCalls": int(metrics.get("strategyCalls") or 0),
        "creatorCalls": int(metrics.get("creatorCalls") or 0),
        "judgeCalls": int(metrics.get("judgeCalls") or 0),
        "judgeRepairCalls": int(metrics.get("judgeRepairCalls") or 0),
        "judgeRetryCalls": int(metrics.get("judgeRetryCalls") or 0),
        "winnerDevelopmentCalls": int(metrics.get("winnerDevelopmentCalls") or 0),
        "winnerNormalCalls": int(metrics.get("winnerNormalCalls") or 0),
        "winnerRepairCalls": int(metrics.get("winnerRepairCalls") or 0),
        "winnerRetryCalls": int(metrics.get("winnerRetryCalls") or 0),
    }


def _metric_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: max(0, after.get(key, 0) - before.get(key, 0)) for key in before}


def validate_reasoning_resume_state(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not isinstance(state, dict):
        return False, ["state"]

    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        missing.append("strategyFoundation")

    tournament_id = str(state.get("tournamentId") or "").strip()
    if not tournament_id:
        missing.append("tournamentId")

    job_id = str(state.get("jobId") or "").strip()
    if not job_id:
        missing.append("jobId")

    backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    if not isinstance(index, dict):
        missing.append("acceptedCreatorCandidates")
        return False, missing

    active_prototypes = list(
        state.get("initialActivePrototypeIds")
        or state.get("activePrototypeIds")
        or resolve_builder2_active_prototype_ids()
    )
    if not active_prototypes:
        missing.append("activePrototypeIds")

    candidate_ids = sorted(str(key) for key in index.keys())
    if len(candidate_ids) != len(set(candidate_ids)):
        missing.append("acceptedCreatorCandidates.duplicate_candidate_ids")

    if len(candidate_ids) != len(active_prototypes):
        missing.append(
            f"acceptedCreatorCandidates.count expected={len(active_prototypes)} actual={len(candidate_ids)}"
        )

    prototype_to_candidate: Dict[str, str] = {}
    for candidate_id in candidate_ids:
        snapshot = index.get(candidate_id)
        if not isinstance(snapshot, dict):
            missing.append(f"acceptedCreatorCandidates.{candidate_id}")
            continue
        if snapshot.get("validationStatus") != "accepted":
            missing.append(f"acceptedCreatorCandidates.{candidate_id}.validationStatus")
        if str(snapshot.get("candidateId") or "") != candidate_id:
            missing.append(f"acceptedCreatorCandidates.{candidate_id}.candidateId_mismatch")
        prototype_id = str(snapshot.get("prototypeId") or "")
        if not prototype_id:
            missing.append(f"acceptedCreatorCandidates.{candidate_id}.prototypeId")
            continue
        if prototype_id in prototype_to_candidate:
            missing.append(f"acceptedCreatorCandidates.duplicate_prototype:{prototype_id}")
        prototype_to_candidate[prototype_id] = candidate_id
        creator_output = snapshot.get("creatorOutput")
        if not isinstance(creator_output, dict) or not creator_output:
            missing.append(f"acceptedCreatorCandidates.{candidate_id}.creatorOutput")

    for prototype_id in active_prototypes:
        if prototype_id not in prototype_to_candidate:
            missing.append(f"acceptedCreatorCandidates.missing_prototype:{prototype_id}")

    return not missing, missing


def _initial_report(*, resume_id: str, job_id: str) -> Dict[str, Any]:
    return {
        "resumeId": resume_id,
        "jobId": job_id,
        "strategyLoaded": False,
        "acceptedCreatorCount": 0,
        "reusedJudgmentCount": 0,
        "judgeNormalCalls": 0,
        "judgeRepairCalls": 0,
        "judgeRetryCalls": 0,
        "acceptedJudgmentCount": 0,
        "eligibleCandidateCount": 0,
        "winnerSelected": False,
        "winnerDevelopmentAccepted": False,
        "winnerCalls": 0,
        "winnerNormalCalls": 0,
        "winnerRepairCalls": 0,
        "winnerRetryCalls": 0,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "startImageCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "reasoningResumeComplete": False,
        "mediaContinuationRequired": False,
        "failureReason": None,
        "missingPaths": [],
        "ok": False,
    }


def run_one_reasoning_resume(
    *,
    job_id: str,
    llm_client: Optional[Any] = None,
    resume_id: Optional[str] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ReasoningResumeIsolationGuard.begin()
    resume_id = resume_id or f"reasoning-resume-{uuid.uuid4().hex[:12]}"
    report = _initial_report(resume_id=resume_id, job_id=job_id)

    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_reasoning_resume_job_not_found"
        ReasoningResumeIsolationGuard.end()
        return report

    valid, missing = validate_reasoning_resume_state(state)
    report["missingPaths"] = missing
    if not valid:
        report["failureReason"] = "builder2_reasoning_resume_state_incomplete"
        ReasoningResumeIsolationGuard.end()
        return report

    report["strategyLoaded"] = True
    backfill_accepted_creator_index(state)
    backfill_accepted_judgment_index(state)
    creator_index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    report["acceptedCreatorCount"] = len(creator_index)

    ensure_metrics(state)
    metrics_before = _metric_snapshot(state.get("metrics") or {})
    ensure_methodology_compatibility_decided(state, is_new_job=False)
    compatibility_mode = bool(state.get("methodologyCompatibilityMode"))

    product_name = str(state.get("productName") or state.get("productNameResolved") or "Resume Product")
    product_description = str(state.get("productDescription") or "Resume description")
    language = str(state.get("contentLanguage") or state.get("language") or "he")
    strategy = state["strategyFoundation"]
    runway_mode = builder2_runway_generation_mode(resolve_builder2_runway_video_model())

    reused = 0
    accepted_judgments = 0
    paid_call_started = False

    try:
        candidate_ids = sorted(str(key) for key in creator_index.keys())
        for candidate_id in candidate_ids:
            snapshot = load_accepted_creator_candidate(
                job_id=job_id,
                candidate_id=candidate_id,
                tournament_state=state,
            )
            prototype_id = str(snapshot.get("prototypeId") or "")
            require_prototype(prototype_id)

            reusable, reuse_reason = audit_reusable_accepted_judgment(
                state,
                candidate_id=candidate_id,
                creator_snapshot=snapshot,
                strategy_foundation=strategy,
                compatibility_mode=compatibility_mode,
            )
            if reusable:
                reused += 1
                accepted_judgments += 1
                logger.info(
                    "BUILDER2_REASONING_RESUME_JUDGMENT_REUSED jobId=%s candidateId=%s",
                    job_id,
                    candidate_id,
                )
                continue

            if reuse_reason:
                logger.info(
                    "BUILDER2_REASONING_RESUME_JUDGMENT_NOT_REUSED jobId=%s candidateId=%s reason=%s",
                    job_id,
                    candidate_id,
                    reuse_reason,
                )

            if is_judge_contract_circuit_breaker_tripped(state):
                breaker = state.get("judgeContractCircuitBreaker") or {}
                paths = breaker.get("repeatedFieldPaths") or []
                trip_reason = breaker.get("trippedReason") or "contract_failure"
                raise Builder2TournamentError(
                    f"{JUDGE_SYSTEMIC_FAILURE_CODE}:{trip_reason}:{','.join(paths[:8])}"
                )

            if not paid_call_started:
                ReasoningResumeIsolationGuard.assert_safe_before_judge()
                paid_call_started = True

            assert_judge_contract_available(state)
            creator_output = snapshot.get("creatorOutput") or {}
            judgment_id = f"judge-{candidate_id}-{uuid.uuid4().hex[:8]}"
            try:
                judgment_id, judgment, total, scores = judge_candidate(
                    product_name=product_name,
                    product_description=product_description,
                    language=language,
                    strategy_foundation=strategy,
                    prototype_id=prototype_id,
                    candidate_id=candidate_id,
                    candidate=creator_output,
                    llm_client=llm_client,
                    state=state,
                    judgment_id=judgment_id,
                    compatibility_mode=compatibility_mode,
                )
            except Builder2TournamentError as exc:
                reason = str(exc.args[0] if exc.args else "builder2_judge_invalid_response")
                if reason.startswith(JUDGE_SYSTEMIC_FAILURE_CODE) or is_judge_contract_circuit_breaker_tripped(state):
                    if not reason.startswith(JUDGE_SYSTEMIC_FAILURE_CODE):
                        breaker = state.get("judgeContractCircuitBreaker") or {}
                        paths = breaker.get("repeatedFieldPaths") or []
                        trip_reason = breaker.get("trippedReason") or "contract_failure"
                        exc = Builder2TournamentError(
                            f"{JUDGE_SYSTEMIC_FAILURE_CODE}:{trip_reason}:{','.join(paths[:8])}"
                        )
                    record_judge_process_contract_failure(state, exc)
                    save_tournament_state(job_id, state)
                    raise exc
                record_process_failure_tag(state, reason)
                save_tournament_state(job_id, state)
                raise

            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=judgment_id,
                judgment=judgment,
                total=total,
                scores=scores,
            )
            accepted_judgments += 1
            save_tournament_state(job_id, state)

        report["reusedJudgmentCount"] = reused
        report["acceptedJudgmentCount"] = accepted_judgments

        eligible_count = sum(
            1
            for candidate_id in candidate_ids
            if bool((state.get("candidates") or {}).get(candidate_id, {}).get("eligible"))
        )
        report["eligibleCandidateCount"] = eligible_count

        if eligible_count == 0:
            judged = [
                cid
                for cid in candidate_ids
                if (state.get("candidates") or {}).get(cid, {}).get("judgeStatus") == "accepted"
            ]
            if len(judged) == len(candidate_ids):
                report["failureReason"] = "builder2_tournament_no_eligible_candidate"
                state["status"] = "no_eligible_candidate"
                state["completionReason"] = "builder2_tournament_no_eligible_candidate"
                save_tournament_state(job_id, state)
                ReasoningResumeIsolationGuard.end()
                return report
            report["failureReason"] = "builder2_reasoning_resume_judgments_incomplete"
            save_tournament_state(job_id, state)
            ReasoningResumeIsolationGuard.end()
            return report

        winner_id = str(state.get("winnerCandidateId") or "").strip()
        if not winner_id:
            winner_id = select_global_winner(state)
            state["winnerCandidateId"] = winner_id
            winner_rec = state["candidates"][winner_id]
            state["winnerSelection"] = {
                "candidateId": winner_id,
                "prototypeId": winner_rec.get("prototypeId"),
                "selectedAt": _utc_now_iso(),
                "eligibleCandidateCount": eligible_count,
            }
            save_tournament_state(job_id, state)
        report["winnerSelected"] = True

        if is_valid_persisted_winner_development(state) or state.get("winnerDevelopmentPlan"):
            report["winnerDevelopmentAccepted"] = True
            logger.info(
                "BUILDER2_REASONING_RESUME_WINNER_REUSED jobId=%s winnerCandidateId=%s",
                job_id,
                winner_id,
            )
        else:
            if not paid_call_started:
                ReasoningResumeIsolationGuard.assert_safe_before_winner_development()
            else:
                ReasoningResumeIsolationGuard.assert_safe_before_winner_development()

            winner_rec = state["candidates"][winner_id]
            judgment_rec = (state.get("judgments") or {}).get(winner_rec.get("judgmentId") or "")
            winning_judgment = (judgment_rec or {}).get("judgment") or {}
            winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
            state["status"] = "winner_developing"
            save_tournament_state(job_id, state)
            try:
                winner_plan = develop_builder2_winning_candidate(
                    product_name=product_name,
                    product_description=product_description,
                    language=language,
                    strategy_foundation=strategy,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    prototype_id=str(winner_rec.get("prototypeId") or ""),
                    runway_mode=runway_mode,
                    llm_client=llm_client,
                    compatibility_mode=compatibility_mode,
                    state=state,
                )
                persist_winner_development_atomically(
                    state,
                    candidate_id=winner_id,
                    prototype_id=str(winner_rec.get("prototypeId") or ""),
                    winner_plan=winner_plan,
                    winning_candidate=winning_candidate,
                    preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
                    compatibility_mode=compatibility_mode,
                )
                state["status"] = "winner_plan_complete"
                save_tournament_state(job_id, state)
                report["winnerDevelopmentAccepted"] = True
            except Builder2TournamentError as exc:
                failure = state.get("winnerDevelopmentFailure")
                if isinstance(failure, dict):
                    report["failureStage"] = failure.get("stage")
                report["failureReason"] = str(exc.args[0] if exc.args else "builder2_winner_development_failed")
                save_tournament_state(job_id, state)
                ReasoningResumeIsolationGuard.end()
                metrics_after = _metric_snapshot(state.get("metrics") or {})
                delta = _metric_delta(metrics_before, metrics_after)
                report["winnerNormalCalls"] = delta["winnerNormalCalls"]
                report["winnerRepairCalls"] = delta["winnerRepairCalls"]
                report["winnerRetryCalls"] = delta["winnerRetryCalls"]
                report["winnerCalls"] = delta["winnerNormalCalls"] + delta["winnerRepairCalls"] + delta["winnerRetryCalls"]
                report["strategyCalls"] = delta["strategyCalls"]
                report["creatorCalls"] = delta["creatorCalls"]
                report["judgeNormalCalls"] = delta["judgeCalls"]
                report["judgeRepairCalls"] = delta["judgeRepairCalls"]
                report["judgeRetryCalls"] = delta["judgeRetryCalls"]
                report["startImageCalls"] = 0
                report["runwayCalls"] = 0
                report["ffmpegCalls"] = 0
                return report

        state["reasoningResumeComplete"] = True
        state["mediaContinuationRequired"] = True
        state["reasoningResume"] = {
            "resumeId": resume_id,
            "completedAt": _utc_now_iso(),
            "winnerDevelopmentAccepted": bool(state.get("winnerDevelopmentPlan")),
            "reusedJudgmentCount": reused,
            "acceptedJudgmentCount": accepted_judgments,
        }
        save_tournament_state(job_id, state)

        report["reasoningResumeComplete"] = True
        report["mediaContinuationRequired"] = True
        report["ok"] = True
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_reasoning_resume_failed")
        report["failureReason"] = reason
        if reason.startswith(RESUME_ISOLATION_ERROR):
            report["ok"] = False
        elif reason.startswith(JUDGE_SYSTEMIC_FAILURE_CODE):
            report["ok"] = False
        else:
            report["ok"] = False
        if state is not None and job_id:
            save_tournament_state(job_id, state)
    finally:
        metrics_after = _metric_snapshot(state.get("metrics") or {}) if state else metrics_before
        delta = _metric_delta(metrics_before, metrics_after)
        report["judgeNormalCalls"] = delta["judgeCalls"]
        report["judgeRepairCalls"] = delta["judgeRepairCalls"]
        report["judgeRetryCalls"] = delta["judgeRetryCalls"]
        report["strategyCalls"] = delta["strategyCalls"]
        report["creatorCalls"] = delta["creatorCalls"]
        report["winnerCalls"] = delta["winnerNormalCalls"] + delta["winnerRepairCalls"] + delta["winnerRetryCalls"]
        report["winnerNormalCalls"] = delta["winnerNormalCalls"]
        report["winnerRepairCalls"] = delta["winnerRepairCalls"]
        report["winnerRetryCalls"] = delta["winnerRetryCalls"]
        report["startImageCalls"] = 0
        report["runwayCalls"] = 0
        report["ffmpegCalls"] = 0
        ReasoningResumeIsolationGuard.end()

    return report


def print_reasoning_resume_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "resumeId",
            "jobId",
            "strategyLoaded",
            "acceptedCreatorCount",
            "reusedJudgmentCount",
            "judgeNormalCalls",
            "judgeRepairCalls",
            "judgeRetryCalls",
            "acceptedJudgmentCount",
            "eligibleCandidateCount",
            "winnerSelected",
            "winnerDevelopmentAccepted",
            "winnerCalls",
            "strategyCalls",
            "creatorCalls",
            "startImageCalls",
            "runwayCalls",
            "ffmpegCalls",
            "reasoningResumeComplete",
            "mediaContinuationRequired",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _resume_env("BUILDER2_REASONING_RESUME_JOB_ID", DEFAULT_RESUME_JOB_ID)
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_reasoning_resume_job_id_missing"}, indent=2))
        return 1

    logger.info("BUILDER2_REASONING_RESUME_START jobId=%s", job_id)
    report = run_one_reasoning_resume(job_id=job_id)
    print_reasoning_resume_report(report)
    logger.info(
        "BUILDER2_REASONING_RESUME_DONE resumeId=%s ok=%s reasoningResumeComplete=%s mediaContinuationRequired=%s",
        report.get("resumeId"),
        report.get("ok"),
        report.get("reasoningResumeComplete"),
        report.get("mediaContinuationRequired"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
