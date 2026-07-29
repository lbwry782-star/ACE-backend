"""
Builder2 controlled complete-ad reasoning-only resume — missing slot completion under a strict call ceiling.

Run:
  BUILDER2_COMPLETE_AD_REASONING_RESUME_JOB_ID=<jobId> python -m engine.builder2_complete_ad_reasoning_resume

Environment:
  BUILDER2_COMPLETE_AD_REASONING_RESUME_JOB_ID
  BUILDER2_COMPLETE_AD_REASONING_RESUME_MAX_CALLS (default 3)
  BUILDER2_COMPLETE_AD_REASONING_RESUME_STOP_BEFORE_MEDIA (default true)
  BUILDER2_COMPLETE_AD_REASONING_RESUME_ALLOW_WINNER_HEADLINE_REPAIR (default false)
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
    persist_accepted_creator_candidate,
)
from engine.builder2_accepted_judgment_store import (
    audit_reusable_accepted_judgment,
    backfill_accepted_judgment_index,
    persist_accepted_judgment,
)
from engine.builder2_complete_ad_contract import copy_winner_advertising_closure_from_candidate
from engine.builder2_complete_ad_creator_recovery import (
    find_rejected_creator_for_prototype,
    try_offline_recover_rejected_creator_for_prototype,
)
from engine.builder2_complete_ad_resume_plan import parsed_winner_reusable_for_candidate
from engine.builder2_creator import generate_creator_candidate, is_slogan_word_limit_failure
from engine.builder2_creator_slogan_repair_patch import (
    additional_paid_slogan_repair_allowed,
    find_original_slogan_word_limit_rejection,
    populate_slogan_repair_call_report,
    try_offline_slogan_repair_salvage_for_prototype,
)
from engine.builder2_execution_lease import acquire_job_lease, release_job_lease
from engine.builder2_judge import judge_candidate
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_reasoning_failure_diagnostics import (
    log_reasoning_resume_failed,
    openai_http_status,
    parsing_failure_category,
    safe_exception_message,
)
from engine.builder2_reasoning_resume_guard import ReasoningResumeIsolationGuard
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_tournament_completion_gate import (
    accepted_creator_count,
    accepted_judgment_count,
    assigned_prototype_ids,
    mark_authoritative_winner_selection,
    missing_creator_prototype_ids,
    missing_judge_prototype_ids,
    tournament_resolution_summary,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_recovery import new_worker_token
from engine.builder2_tournament_store import (
    ensure_methodology_compatibility_decided,
    load_tournament_state,
    record_process_failure_tag,
    save_tournament_state,
)
from engine.builder2_winner_development import develop_builder2_winning_candidate
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    offline_revalidate_parsed_winner_response,
)
from engine.builder2_winner_headline_repair import attempt_winner_headline_repair_after_offline_failure
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

DEFAULT_MAX_CALLS = 3
GREENPEACE_PROTOTYPE = "greenpeace_essential_pairing"
CALL_BUDGET_EXHAUSTED = "builder2_complete_ad_reasoning_resume_call_budget_exhausted"
SLOGAN_WORD_LIMIT_FAILURE = "builder2_advertising_closure_invalid:sloganText.word_limit"


def _resolve_single_missing_creator_prototype(state: Dict[str, Any]) -> Optional[str]:
    missing = missing_creator_prototype_ids(state)
    return missing[0] if len(missing) == 1 else None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class ControlledReasoningCallBudget:
    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = max(1, int(max_calls))
        self.creator_calls_this_run = 0
        self.judge_calls_this_run = 0
        self.winner_calls_this_run = 0

    @property
    def total_this_run(self) -> int:
        return self.creator_calls_this_run + self.judge_calls_this_run + self.winner_calls_this_run

    def assert_can_call(self, role: str) -> None:
        if self.total_this_run >= self.max_calls:
            raise Builder2TournamentError(f"{CALL_BUDGET_EXHAUSTED}:{role}")

    def record(self, role: str) -> None:
        self.assert_can_call(role)
        if role == "builder2_creator":
            self.creator_calls_this_run += 1
        elif role == "builder2_judge":
            self.judge_calls_this_run += 1
        elif role == "builder2_winner":
            self.winner_calls_this_run += 1
        else:
            raise Builder2TournamentError(f"builder2_complete_ad_reasoning_resume_unknown_role:{role}")


def _reasoning_call_snapshot(state: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not state:
        return {"creator": 0, "judge": 0, "winner": 0}
    metrics = ensure_metrics(state)
    return {
        "creator": int(metrics.get("creatorCalls") or 0)
        + int(metrics.get("creatorRepairCalls") or 0)
        + int(metrics.get("creatorRetryCalls") or 0),
        "judge": int(metrics.get("judgeCalls") or 0)
        + int(metrics.get("judgeRepairCalls") or 0)
        + int(metrics.get("judgeRetryCalls") or 0),
        "winner": int(metrics.get("winnerDevelopmentCalls") or 0),
    }


def _sync_budget_from_metrics_delta(
    budget: ControlledReasoningCallBudget,
    *,
    baseline: Dict[str, int],
    state: Dict[str, Any],
) -> None:
    current = _reasoning_call_snapshot(state)
    budget.creator_calls_this_run = max(budget.creator_calls_this_run, current["creator"] - baseline["creator"])
    budget.judge_calls_this_run = max(budget.judge_calls_this_run, current["judge"] - baseline["judge"])
    budget.winner_calls_this_run = max(budget.winner_calls_this_run, current["winner"] - baseline["winner"])


def _populate_report_accepted_counts(report: Dict[str, Any], state: Optional[Dict[str, Any]]) -> None:
    if state is None:
        return
    report["acceptedCreatorCount"] = accepted_creator_count(state)
    report["acceptedJudgmentCount"] = accepted_judgment_count(state)


def _populate_report_reasoning_calls(report: Dict[str, Any], budget: ControlledReasoningCallBudget) -> None:
    report["creatorCallsThisRun"] = budget.creator_calls_this_run
    report["judgeCallsThisRun"] = budget.judge_calls_this_run
    report["winnerCallsThisRun"] = budget.winner_calls_this_run
    report["totalReasoningCallsThisRun"] = budget.total_this_run


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _initial_report(*, job_id: str) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "ok": False,
        "reasoningResumeCompleted": False,
        "stoppedBeforeMedia": False,
        "strategyReused": False,
        "creatorCallsThisRun": 0,
        "judgeCallsThisRun": 0,
        "winnerCallsThisRun": 0,
        "totalReasoningCallsThisRun": 0,
        "maximumAllowedReasoningCalls": DEFAULT_MAX_CALLS,
        "acceptedCreatorCount": 0,
        "acceptedJudgmentCount": 0,
        "finalWinnerCandidateId": None,
        "finalWinnerPrototypeId": None,
        "finalWinnerScore": None,
        "winnerChangedFromProvisional": False,
        "winnerDevelopmentReused": False,
        "winnerDevelopmentAccepted": False,
        "advertisingClosurePresent": False,
        "semanticAlignmentAccepted": False,
        "winnerSloganPreserved": False,
        "nextStage": None,
        "imageCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "failureReason": None,
        "failureStage": None,
        "canResume": True,
    }


def validate_controlled_complete_ad_preconditions(
    state: Dict[str, Any],
    job_raw: Optional[Dict[str, Any]] = None,
    *,
    expected_missing_prototype: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    job_id = _clean(state.get("jobId"))
    if not job_id:
        return False, "builder2_complete_ad_reasoning_resume_job_id_missing"
    tournament_id = _clean(state.get("tournamentId"))
    if not tournament_id:
        return False, "builder2_complete_ad_reasoning_resume_tournament_id_missing"

    resume_version = _clean(state.get("builder2ResumeContractVersion") or (job_raw or {}).get("builder2ResumeContractVersion"))
    if resume_version and resume_version != BUILDER2_RESUME_CONTRACT_VERSION:
        return False, "builder2_complete_ad_reasoning_resume_contract_mismatch"

    new_format = _clean(state.get("builder2NewFormatVersion") or (job_raw or {}).get("builder2NewFormatVersion"))
    if new_format and new_format != BUILDER2_NEW_FORMAT_VERSION:
        return False, "builder2_complete_ad_reasoning_resume_new_format_mismatch"

    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        return False, "builder2_complete_ad_reasoning_resume_strategy_missing"

    summary = tournament_resolution_summary(state)
    assigned = assigned_prototype_ids(state)
    if len(assigned) < 6:
        return False, "builder2_complete_ad_reasoning_resume_not_six_way"

    missing_creators = list(summary.get("missingCreatorPrototypeIds") or [])
    missing_judges = list(summary.get("missingJudgePrototypeIds") or [])

    if summary["acceptedCreatorCount"] > 6 or summary["acceptedJudgmentCount"] > 6:
        return False, "builder2_complete_ad_reasoning_resume_invalid_counts"

    if summary["acceptedCreatorCount"] == 6 and summary["acceptedJudgmentCount"] == 6:
        return True, None

    if summary["acceptedCreatorCount"] != 5 or summary["acceptedJudgmentCount"] != 5:
        return False, "builder2_complete_ad_reasoning_resume_unexpected_partial_state"

    if expected_missing_prototype is None:
        expected_missing_prototype = _resolve_single_missing_creator_prototype(state)
    if not expected_missing_prototype:
        return False, "builder2_complete_ad_reasoning_resume_unexpected_missing_creator"

    if missing_creators != [expected_missing_prototype]:
        return False, "builder2_complete_ad_reasoning_resume_unexpected_missing_creator"
    if expected_missing_prototype not in missing_judges:
        return False, "builder2_complete_ad_reasoning_resume_unexpected_missing_judge"

    media = _media_bucket(state)
    if _clean(media.get("startImageArtifact")) or _clean(media.get("runwayTaskId")):
        return False, "builder2_complete_ad_reasoning_resume_media_already_started"

    return True, None


def _candidate_id_for_prototype(state: Dict[str, Any], prototype_id: str) -> Optional[str]:
    backfill_accepted_creator_index(state)
    for candidate_id, rec in (state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}).items():
        if isinstance(rec, dict) and _clean(rec.get("prototypeId")) == prototype_id:
            return str(candidate_id)
    for candidate_id, rec in (state.get("candidates") or {}).items():
        if isinstance(rec, dict) and _clean(rec.get("prototypeId")) == prototype_id:
            if rec.get("validationStatus") == "accepted" or rec.get("creatorAcceptanceStatus") == "accepted":
                return str(candidate_id)
    return None


def _controlled_reasoning_already_complete(state: Dict[str, Any]) -> bool:
    summary = tournament_resolution_summary(state)
    if summary["acceptedCreatorCount"] != 6 or summary["acceptedJudgmentCount"] != 6:
        return False
    if not is_valid_persisted_winner_development(state):
        return False
    if state.get("mediaStarted"):
        return False
    if state.get("reasoningComplete") and _clean(state.get("progressStage")) == "media_prerequisite_validation":
        return True
    bucket = state.get("controlledCompleteAdReasoningResume")
    return isinstance(bucket, dict) and bool(bucket.get("stoppedBeforeMedia"))


def _populate_success_report_from_state(
    report: Dict[str, Any],
    state: Dict[str, Any],
    *,
    budget: ControlledReasoningCallBudget,
    winner_reused: bool,
    stop_before_media: bool,
) -> None:
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    judgment_rec = (state.get("judgments") or {}).get(winner_rec.get("judgmentId") or "")
    winning_judgment = (judgment_rec or {}).get("judgment") or {}
    closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
    report["acceptedCreatorCount"] = accepted_creator_count(state)
    report["acceptedJudgmentCount"] = accepted_judgment_count(state)
    report["finalWinnerCandidateId"] = winner_id or None
    report["finalWinnerPrototypeId"] = _clean(winner_rec.get("prototypeId")) or None
    report["finalWinnerScore"] = winner_rec.get("totalScore")
    report["winnerDevelopmentReused"] = winner_reused
    report["winnerDevelopmentAccepted"] = is_valid_persisted_winner_development(state)
    report["advertisingClosurePresent"] = bool(closure.get("sloganText"))
    report["semanticAlignmentAccepted"] = bool(
        (winning_judgment.get("semanticAlignmentAssessment") or {}).get("semanticAlignment")
    )
    report["winnerSloganPreserved"] = bool(state.get("winnerSelectedSloganHash"))
    report["creatorCallsThisRun"] = budget.creator_calls_this_run
    report["judgeCallsThisRun"] = budget.judge_calls_this_run
    report["winnerCallsThisRun"] = budget.winner_calls_this_run
    report["totalReasoningCallsThisRun"] = budget.total_this_run
    report["reasoningResumeCompleted"] = True
    report["stoppedBeforeMedia"] = stop_before_media
    report["nextStage"] = "media_prerequisite_validation"
    report["ok"] = True


def _clear_stale_winner_before_recompute(state: Dict[str, Any]) -> None:
    state.pop("provisionalWinnerCandidateId", None)
    state.pop("winnerCandidateId", None)
    state.pop("winnerDevelopmentPlan", None)
    state.pop("winnerDevelopmentCandidateId", None)
    state.pop("winnerDevelopmentPrototypeId", None)
    state.pop("winnerDevelopmentAcceptedAt", None)
    state.pop("winnerDevelopmentMetadata", None)
    state.pop("winnerDevelopmentAccepted", None)
    state.pop("advertisingClosure", None)
    state.pop("winnerSelectedSloganHash", None)


def _persist_resumable_failure(
    state: Dict[str, Any],
    *,
    job_id: str,
    failure_stage: str,
    failure_reason: str,
) -> None:
    state["status"] = "paused_for_reasoning_resume"
    state["failureStage"] = failure_stage
    state["failureReason"] = failure_reason
    state["canResume"] = True
    state["resumeFailure"] = {
        "failureStage": failure_stage,
        "failureReason": failure_reason,
        "storedAt": _utc_now_iso(),
    }
    save_tournament_state(job_id, state)


def _emit_resume_stage_failure(
    report: Dict[str, Any],
    state: Optional[Dict[str, Any]],
    *,
    job_id: str,
    failure_stage: str,
    failure_reason: str,
    budget: Optional[ControlledReasoningCallBudget] = None,
    reasoning_role: str = "",
    prototype_id: str = "",
    model: str = "",
    response_text_present: Optional[bool] = None,
    response_text_chars: Optional[int] = None,
    validation_rejection_code: str = "",
    redis_mutated: bool = False,
    lease_acquired: bool = False,
    exception_class: str = "",
    http_status: Optional[int] = None,
    with_traceback: bool = False,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    report["failureReason"] = failure_reason
    report["failureStage"] = failure_stage
    _populate_report_accepted_counts(report, state)
    if budget is not None:
        _populate_report_reasoning_calls(report, budget)
    tournament_id = _clean((state or {}).get("tournamentId"))
    log_reasoning_resume_failed(
        logger,
        job_id=job_id,
        tournament_id=tournament_id,
        prototype_id=prototype_id,
        reasoning_role=reasoning_role,
        model=model,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        event="BUILDER2_COMPLETE_AD_REASONING_RESUME_TERMINAL_FAILURE",
        exception_class=exception_class,
        http_status=http_status,
        response_text_present=response_text_present,
        response_text_chars=response_text_chars,
        parsing_failure_category_value=parsing_failure_category(failure_reason) or "",
        validation_rejection_code=validation_rejection_code or failure_reason,
        redis_mutated=redis_mutated,
        lease_released=lease_acquired,
        with_traceback=with_traceback,
        exc=exc,
    )
    return report


def _finalize_stop_before_media(state: Dict[str, Any], *, job_id: str) -> None:
    state["status"] = "paused_for_media_validation"
    state["lastCompletedStep"] = "reasoning_complete"
    state["reasoningComplete"] = True
    state["reasoningResumeCompleted"] = True
    state["mediaStarted"] = False
    state["mediaContinuationRequired"] = True
    state["canResume"] = True
    state["progressStage"] = "media_prerequisite_validation"
    state["failureStage"] = None
    state["failureReason"] = None
    state["controlledCompleteAdReasoningResume"] = {
        "completedAt": _utc_now_iso(),
        "stoppedBeforeMedia": True,
    }
    save_tournament_state(job_id, state)


def run_controlled_complete_ad_reasoning_resume(
    *,
    job_id: str,
    llm_client: Optional[Any] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
    max_calls: int = DEFAULT_MAX_CALLS,
    stop_before_media: bool = True,
    acquire_lease: bool = True,
) -> Dict[str, Any]:
    ReasoningResumeIsolationGuard.begin()
    report = _initial_report(job_id=job_id)
    report["maximumAllowedReasoningCalls"] = max_calls
    budget = ControlledReasoningCallBudget(max_calls=max_calls)
    worker_token = new_worker_token()
    lease_acquired = False
    state: Optional[Dict[str, Any]] = None
    current_stage = "startup"
    missing_prototype_id = ""

    try:
        current_stage = "preconditions"
        if not redis_configured() and tournament_state is None:
            report["failureReason"] = "builder2_complete_ad_reasoning_resume_redis_unconfigured"
            return report

        state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
        if state is None:
            report["failureReason"] = "builder2_complete_ad_reasoning_resume_job_not_found"
            return report

        job_raw = video_job_get_raw(job_id) or {}
        ok, pre_reason = validate_controlled_complete_ad_preconditions(state, job_raw)
        if not ok:
            report["failureReason"] = pre_reason
            return report

        if acquire_lease and not acquire_job_lease(job_id, worker_token):
            report["failureReason"] = "builder2_complete_ad_reasoning_resume_lease_unavailable"
            report["canResume"] = True
            return report
        lease_acquired = acquire_lease

        ensure_methodology_compatibility_decided(state, is_new_job=False)
        compatibility_mode = bool(state.get("methodologyCompatibilityMode"))
        strategy = state["strategyFoundation"]
        report["strategyReused"] = True

        if _controlled_reasoning_already_complete(state):
            winner_id = _clean(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId"))
            report["winnerChangedFromProvisional"] = False
            _populate_success_report_from_state(
                report,
                state,
                budget=budget,
                winner_reused=True,
                stop_before_media=stop_before_media,
            )
            if stop_before_media:
                _finalize_stop_before_media(state, job_id=job_id)
            return report

        product_name = _clean(strategy.get("productNameResolved") or state.get("productNameResolved") or "Product")
        product_description = _clean(state.get("productDescription") or "Product description")
        language = _clean(state.get("contentLanguage") or state.get("language") or strategy.get("language") or "he")
        runway_mode = builder2_runway_generation_mode(resolve_builder2_runway_video_model())

        provisional_winner = _clean(state.get("provisionalWinnerCandidateId"))

        missing_prototype_id = _resolve_single_missing_creator_prototype(state) or ""

        if missing_prototype_id and missing_prototype_id in missing_creator_prototype_ids(state):
            if not _candidate_id_for_prototype(state, missing_prototype_id):
                current_stage = "creator_generation"
                rejected_payload = find_rejected_creator_for_prototype(state, missing_prototype_id)
                recovered, recovered_id, offline_reason = try_offline_recover_rejected_creator_for_prototype(
                    state,
                    prototype_id=missing_prototype_id,
                    product_name=product_name,
                    compatibility_mode=compatibility_mode,
                )
                if recovered:
                    logger.info(
                        "BUILDER2_REJECTED_CREATOR_OFFLINE_RECOVERY_SUCCEEDED jobId=%s prototypeId=%s candidateId=%s",
                        job_id,
                        missing_prototype_id,
                        recovered_id,
                    )
                    save_tournament_state(job_id, state)
                elif rejected_payload:
                    logger.info(
                        "BUILDER2_REJECTED_CREATOR_OFFLINE_RECOVERY_IMPOSSIBLE jobId=%s prototypeId=%s reason=%s",
                        job_id,
                        missing_prototype_id,
                        offline_reason or "unknown",
                    )
                if not _candidate_id_for_prototype(state, missing_prototype_id):
                    rejected_failure = _clean((rejected_payload or {}).get("failureReason"))
                    slogan_word_limit_case = is_slogan_word_limit_failure(rejected_failure) or bool(
                        find_original_slogan_word_limit_rejection(state, missing_prototype_id)
                    )
                    if slogan_word_limit_case:
                        original_word_limit = find_original_slogan_word_limit_rejection(state, missing_prototype_id)
                        from engine.builder2_creator_slogan_repair_patch import find_slogan_repair_patch_source

                        repair_source = find_slogan_repair_patch_source(state, missing_prototype_id)
                        salvage_accepted, salvage_id, salvage_reason, salvage_paths = (
                            try_offline_slogan_repair_salvage_for_prototype(
                                state,
                                prototype_id=missing_prototype_id,
                                product_name=product_name,
                                compatibility_mode=compatibility_mode,
                                original_candidate_id=_clean((original_word_limit or {}).get("candidateId")),
                                patch_candidate_id=_clean((repair_source or {}).get("candidateId")),
                            )
                        )
                        report["offlineSalvageAttempted"] = True
                        report["offlineSalvageAccepted"] = salvage_accepted
                        if salvage_accepted:
                            logger.info(
                                "BUILDER2_SLOGAN_REPAIR_OFFLINE_SALVAGE_SUCCEEDED jobId=%s prototypeId=%s candidateId=%s",
                                job_id,
                                missing_prototype_id,
                                salvage_id,
                            )
                            save_tournament_state(job_id, state)
                        elif not additional_paid_slogan_repair_allowed(state, missing_prototype_id):
                            reason = salvage_reason or "builder2_slogan_repair_paid_retry_requires_approval"
                            if salvage_paths:
                                reason = f"{reason}:{','.join(salvage_paths[:8])}"
                            record_process_failure_tag(state, reason)
                            _persist_resumable_failure(
                                state,
                                job_id=job_id,
                                failure_stage="creator_generation",
                                failure_reason=reason,
                            )
                            populate_slogan_repair_call_report(state, report, prototype_id=missing_prototype_id)
                            return _emit_resume_stage_failure(
                                report,
                                state,
                                job_id=job_id,
                                failure_stage="creator_generation",
                                failure_reason=reason,
                                budget=budget,
                                reasoning_role="builder2_creator",
                                prototype_id=missing_prototype_id,
                                validation_rejection_code=reason,
                                redis_mutated=True,
                                lease_acquired=lease_acquired,
                            )
                        elif rejected_payload:
                            logger.warning(
                                "BUILDER2_SLOGAN_REPAIR_OFFLINE_SALVAGE_IMPOSSIBLE jobId=%s prototypeId=%s reason=%s paths=%s",
                                job_id,
                                missing_prototype_id,
                                salvage_reason or "unknown",
                                ",".join(salvage_paths[:8]) if salvage_paths else "(none)",
                            )
                if not _candidate_id_for_prototype(state, missing_prototype_id):
                    if rejected_payload:
                        logger.warning(
                            "BUILDER2_REJECTED_CREATOR_OFFLINE_RECOVERY_FALLBACK_OPENAI jobId=%s prototypeId=%s "
                            "offlineReason=%s",
                            job_id,
                            missing_prototype_id,
                            offline_reason or "unknown",
                        )
                    if (
                        is_slogan_word_limit_failure(_clean((rejected_payload or {}).get("failureReason")))
                        and not additional_paid_slogan_repair_allowed(state, missing_prototype_id)
                    ):
                        reason = "builder2_slogan_repair_paid_retry_requires_approval"
                        record_process_failure_tag(state, reason)
                        _persist_resumable_failure(
                            state,
                            job_id=job_id,
                            failure_stage="creator_generation",
                            failure_reason=reason,
                        )
                        populate_slogan_repair_call_report(state, report, prototype_id=missing_prototype_id)
                        return _emit_resume_stage_failure(
                            report,
                            state,
                            job_id=job_id,
                            failure_stage="creator_generation",
                            failure_reason=reason,
                            budget=budget,
                            reasoning_role="builder2_creator",
                            prototype_id=missing_prototype_id,
                            validation_rejection_code=reason,
                            redis_mutated=True,
                            lease_acquired=lease_acquired,
                        )
                    budget.assert_can_call("builder2_creator")
                    metrics_before = _reasoning_call_snapshot(state)
                    candidate_id = f"cand-1-{missing_prototype_id}-1-{uuid.uuid4().hex[:8]}"
                    creator_kwargs: Dict[str, Any] = {"single_attempt_only": True}
                    rejected_failure = _clean((rejected_payload or {}).get("failureReason"))
                    rejected_parsed = (rejected_payload or {}).get("parsed") if isinstance(rejected_payload, dict) else None
                    original_word_limit = find_original_slogan_word_limit_rejection(state, missing_prototype_id)
                    if isinstance(original_word_limit, dict) and isinstance(original_word_limit.get("parsed"), dict):
                        rejected_parsed = original_word_limit.get("parsed")
                        rejected_failure = _clean(original_word_limit.get("failureReason")) or SLOGAN_WORD_LIMIT_FAILURE
                    if (
                        isinstance(rejected_parsed, dict)
                        and rejected_parsed
                        and is_slogan_word_limit_failure(rejected_failure)
                    ):
                        creator_kwargs = {
                            "repair_only_from_parsed": rejected_parsed,
                            "repair_only_failure_reason": rejected_failure or SLOGAN_WORD_LIMIT_FAILURE,
                        }
                    elif rejected_payload:
                        creator_kwargs = {"single_attempt_only": False}
                    try:
                        candidate_id, candidate = generate_creator_candidate(
                            product_name=product_name,
                            product_description=product_description,
                            language=language,
                            strategy_foundation=strategy,
                            prototype_id=missing_prototype_id,
                            round_index=1,
                            attempt_number=1,
                            runway_mode=runway_mode,
                            llm_client=llm_client,
                            state=state,
                            candidate_id=candidate_id,
                            compatibility_mode=compatibility_mode,
                            **creator_kwargs,
                        )
                    except Builder2TournamentError as exc:
                        _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
                        reason = str(exc.args[0] if exc.args else "builder2_creator_validation_failed")
                        record_process_failure_tag(state, reason)
                        _persist_resumable_failure(
                            state,
                            job_id=job_id,
                            failure_stage="creator_generation",
                            failure_reason=reason,
                        )
                        return _emit_resume_stage_failure(
                            report,
                            state,
                            job_id=job_id,
                            failure_stage="creator_generation",
                            failure_reason=reason,
                            budget=budget,
                            reasoning_role="builder2_creator",
                            prototype_id=missing_prototype_id,
                            validation_rejection_code=reason,
                            redis_mutated=True,
                            lease_acquired=lease_acquired,
                        )
                    _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
                    if budget.creator_calls_this_run == metrics_before["creator"]:
                        budget.record("builder2_creator")
                    persist_accepted_creator_candidate(
                        state,
                        candidate_id=candidate_id,
                        prototype_id=missing_prototype_id,
                        round_index=1,
                        attempt_number=1,
                        creator_output=candidate,
                        strategy_foundation=strategy,
                    )
                    save_tournament_state(job_id, state)

        backfill_accepted_creator_index(state)
        missing_candidate_id = _candidate_id_for_prototype(state, missing_prototype_id) if missing_prototype_id else None
        if missing_prototype_id and not missing_candidate_id:
            reason = "builder2_complete_ad_reasoning_resume_missing_creator_unresolved"
            _persist_resumable_failure(
                state,
                job_id=job_id,
                failure_stage="creator_generation",
                failure_reason=reason,
            )
            return _emit_resume_stage_failure(
                report,
                state,
                job_id=job_id,
                failure_stage="creator_generation",
                failure_reason=reason,
                budget=budget,
                reasoning_role="builder2_creator",
                prototype_id=missing_prototype_id,
                redis_mutated=True,
                lease_acquired=lease_acquired,
            )

        backfill_accepted_judgment_index(state)
        if missing_prototype_id and missing_candidate_id:
            snapshot = (state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}).get(missing_candidate_id) or {}
            creator_output = (snapshot.get("creatorOutput") if isinstance(snapshot, dict) else None) or (
                (state.get("candidates") or {}).get(missing_candidate_id) or {}
            ).get("creatorOutput") or {}

            reusable, _reuse_reason = audit_reusable_accepted_judgment(
                state,
                candidate_id=missing_candidate_id,
                creator_snapshot={
                    "candidateId": missing_candidate_id,
                    "prototypeId": missing_prototype_id,
                    "creatorOutput": creator_output,
                    "validationStatus": "accepted",
                },
                strategy_foundation=strategy,
                compatibility_mode=compatibility_mode,
            )
            if not reusable and missing_prototype_id in missing_judge_prototype_ids(state):
                current_stage = "judge_generation"
                budget.assert_can_call("builder2_judge")
                ReasoningResumeIsolationGuard.assert_safe_before_judge()
                judgment_id = f"judge-{missing_candidate_id}-{uuid.uuid4().hex[:8]}"
                metrics_before = _reasoning_call_snapshot(state)
                try:
                    judgment_id, judgment, total, scores = judge_candidate(
                        product_name=product_name,
                        product_description=product_description,
                        language=language,
                        strategy_foundation=strategy,
                        prototype_id=missing_prototype_id,
                        candidate_id=missing_candidate_id,
                        candidate=creator_output,
                        llm_client=llm_client,
                        state=state,
                        judgment_id=judgment_id,
                        compatibility_mode=compatibility_mode,
                        single_attempt_only=True,
                    )
                except Builder2TournamentError as exc:
                    _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
                    reason = str(exc.args[0] if exc.args else "builder2_judge_invalid_response")
                    record_process_failure_tag(state, reason)
                    _persist_resumable_failure(
                        state,
                        job_id=job_id,
                        failure_stage="judge_generation",
                        failure_reason=reason,
                    )
                    return _emit_resume_stage_failure(
                        report,
                        state,
                        job_id=job_id,
                        failure_stage="judge_generation",
                        failure_reason=reason,
                        budget=budget,
                        reasoning_role="builder2_judge",
                        prototype_id=missing_prototype_id,
                        validation_rejection_code=reason,
                        redis_mutated=True,
                        lease_acquired=lease_acquired,
                    )
                _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
                if budget.judge_calls_this_run == metrics_before["judge"]:
                    budget.record("builder2_judge")
                persist_accepted_judgment(
                    state,
                    candidate_id=missing_candidate_id,
                    prototype_id=missing_prototype_id,
                    judgment_id=judgment_id,
                    judgment=judgment,
                    total=total,
                    scores=scores,
                )
                save_tournament_state(job_id, state)

        report["acceptedCreatorCount"] = accepted_creator_count(state)
        report["acceptedJudgmentCount"] = accepted_judgment_count(state)
        if report["acceptedCreatorCount"] != 6 or report["acceptedJudgmentCount"] != 6:
            reason = "builder2_complete_ad_reasoning_resume_six_way_incomplete"
            _persist_resumable_failure(
                state,
                job_id=job_id,
                failure_stage="judge_generation",
                failure_reason=reason,
            )
            return _emit_resume_stage_failure(
                report,
                state,
                job_id=job_id,
                failure_stage="judge_generation",
                failure_reason=reason,
                budget=budget,
                redis_mutated=True,
                lease_acquired=lease_acquired,
            )

        current_stage = "winner_selection"
        _clear_stale_winner_before_recompute(state)

        try:
            winner_id = select_global_winner(state)
        except Builder2TournamentError as exc:
            reason = str(exc.args[0] if exc.args else "builder2_tournament_no_valid_candidate")
            _persist_resumable_failure(
                state,
                job_id=job_id,
                failure_stage="winner_selection",
                failure_reason=reason,
            )
            return _emit_resume_stage_failure(
                report,
                state,
                job_id=job_id,
                failure_stage="winner_selection",
                failure_reason=reason,
                budget=budget,
                reasoning_role="builder2_winner",
                redis_mutated=True,
                lease_acquired=lease_acquired,
            )

        mark_authoritative_winner_selection(state, winner_id=winner_id)
        winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
        judgment_rec = (state.get("judgments") or {}).get(winner_rec.get("judgmentId") or "")
        winning_judgment = (judgment_rec or {}).get("judgment") or {}
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        copy_winner_advertising_closure_from_candidate(
            state,
            candidate_id=winner_id,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
        )
        save_tournament_state(job_id, state)

        report["finalWinnerCandidateId"] = winner_id
        report["finalWinnerPrototypeId"] = _clean(winner_rec.get("prototypeId"))
        report["finalWinnerScore"] = winner_rec.get("totalScore")
        report["winnerChangedFromProvisional"] = bool(provisional_winner and provisional_winner != winner_id)

        winner_reused = False
        persisted_winner_id = _clean(state.get("winnerDevelopmentCandidateId"))
        if is_valid_persisted_winner_development(state) and persisted_winner_id == winner_id:
            winner_reused = True
            report["winnerDevelopmentAccepted"] = True
        elif parsed_winner_reusable_for_candidate(state, winner_candidate_id=winner_id):
            source = build_server_owned_winner_source_reference(
                strategy_foundation=strategy,
                winning_candidate=winning_candidate,
                candidate_id=winner_id,
            )
            try:
                winner_plan = offline_revalidate_parsed_winner_response(
                    state,
                    source_reference=source,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    compatibility_mode=compatibility_mode,
                    job_id=job_id,
                    tournament_id=_clean(state.get("tournamentId")),
                )
                persist_winner_development_atomically(
                    state,
                    candidate_id=winner_id,
                    prototype_id=_clean(winner_rec.get("prototypeId")),
                    winner_plan=winner_plan,
                    winning_candidate=winning_candidate,
                    preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
                    compatibility_mode=compatibility_mode,
                )
                winner_reused = True
                report["winnerDevelopmentAccepted"] = True
            except Builder2TournamentError as exc:
                reason = str(exc.args[0] if exc.args else "builder2_winner_offline_revalidation_failed")
                allow_headline_repair = _env_bool(
                    "BUILDER2_COMPLETE_AD_REASONING_RESUME_ALLOW_WINNER_HEADLINE_REPAIR",
                    False,
                )
                metrics_before_winner = _reasoning_call_snapshot(state)
                repair_outcome = attempt_winner_headline_repair_after_offline_failure(
                    state,
                    job_id=job_id,
                    winner_candidate_id=winner_id,
                    prototype_id=_clean(winner_rec.get("prototypeId")),
                    product_name=product_name,
                    language=language,
                    strategy_foundation=strategy,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    offline_failure_reason=reason,
                    allow_repair=allow_headline_repair,
                    remaining_call_budget=max(0, budget.max_calls - budget.total_this_run),
                    compatibility_mode=compatibility_mode,
                    llm_client=llm_client,
                    tournament_id=_clean(state.get("tournamentId")),
                    on_eligible_before_call=lambda: budget.assert_can_call("builder2_winner"),
                )
                _sync_budget_from_metrics_delta(
                    budget,
                    baseline=metrics_before_winner,
                    state=state,
                )
                if repair_outcome.get("accepted"):
                    winner_reused = True
                    report["winnerDevelopmentAccepted"] = True
                    report["winnerHeadlineRepairAttempted"] = True
                    report["winnerHeadlineRepairAccepted"] = True
                else:
                    failure_reason = str(repair_outcome.get("failure_reason") or reason)
                    if repair_outcome.get("attempted"):
                        report["winnerHeadlineRepairAttempted"] = True
                        report["winnerHeadlineRepairAccepted"] = False
                    _persist_resumable_failure(
                        state,
                        job_id=job_id,
                        failure_stage="winner_development",
                        failure_reason=failure_reason,
                    )
                    return _emit_resume_stage_failure(
                        report,
                        state,
                        job_id=job_id,
                        failure_stage="winner_development",
                        failure_reason=failure_reason,
                        budget=budget,
                        reasoning_role="builder2_winner",
                        redis_mutated=True,
                        lease_acquired=lease_acquired,
                    )
        else:
            budget.assert_can_call("builder2_winner")
            ReasoningResumeIsolationGuard.assert_safe_before_winner_development()
            metrics_before = _reasoning_call_snapshot(state)
            try:
                winner_plan = develop_builder2_winning_candidate(
                    product_name=product_name,
                    product_description=product_description,
                    language=language,
                    strategy_foundation=strategy,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    prototype_id=_clean(winner_rec.get("prototypeId")),
                    runway_mode=runway_mode,
                    llm_client=llm_client,
                    compatibility_mode=compatibility_mode,
                    state=state,
                    candidate_id=winner_id,
                )
            except Builder2TournamentError as exc:
                _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
                reason = str(exc.args[0] if exc.args else "builder2_winner_development_failed")
                _persist_resumable_failure(
                    state,
                    job_id=job_id,
                    failure_stage="winner_development",
                    failure_reason=reason,
                )
                return _emit_resume_stage_failure(
                    report,
                    state,
                    job_id=job_id,
                    failure_stage="winner_development",
                    failure_reason=reason,
                    budget=budget,
                    reasoning_role="builder2_winner",
                    redis_mutated=True,
                    lease_acquired=lease_acquired,
                )
            _sync_budget_from_metrics_delta(budget, baseline=metrics_before, state=state)
            if budget.winner_calls_this_run == metrics_before["winner"]:
                budget.record("builder2_winner")
            persist_winner_development_atomically(
                state,
                candidate_id=winner_id,
                prototype_id=_clean(winner_rec.get("prototypeId")),
                winner_plan=winner_plan,
                winning_candidate=winning_candidate,
                preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
                compatibility_mode=compatibility_mode,
            )
            report["winnerDevelopmentAccepted"] = True

        report["winnerDevelopmentReused"] = winner_reused
        _populate_success_report_from_state(
            report,
            state,
            budget=budget,
            winner_reused=winner_reused,
            stop_before_media=stop_before_media,
        )
        report["winnerChangedFromProvisional"] = bool(provisional_winner and provisional_winner != winner_id)

        if stop_before_media:
            _finalize_stop_before_media(state, job_id=job_id)

        return report
    except Exception as exc:
        reason = f"builder2_complete_ad_reasoning_resume_unhandled:{type(exc).__name__}:{safe_exception_message(exc)}"
        if state is not None:
            try:
                record_process_failure_tag(state, reason)
                _persist_resumable_failure(
                    state,
                    job_id=job_id,
                    failure_stage=current_stage,
                    failure_reason=reason,
                )
                redis_mutated = True
            except Exception:
                logger.exception(
                    "BUILDER2_REASONING_RESUME_PERSIST_FAILURE jobId=%s failureStage=%s",
                    job_id,
                    current_stage,
                )
                redis_mutated = False
        else:
            redis_mutated = False
        return _emit_resume_stage_failure(
            report,
            state,
            job_id=job_id,
            failure_stage=current_stage,
            failure_reason=reason,
            budget=budget,
            reasoning_role="builder2_creator" if current_stage == "creator_generation" else "",
            prototype_id=missing_prototype_id if current_stage == "creator_generation" and missing_prototype_id else "",
            exception_class=type(exc).__name__,
            http_status=openai_http_status(exc),
            redis_mutated=redis_mutated,
            lease_acquired=lease_acquired,
            with_traceback=True,
            exc=exc,
        )
    finally:
        if lease_acquired:
            release_job_lease(job_id, worker_token)
        ReasoningResumeIsolationGuard.end()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_COMPLETE_AD_REASONING_RESUME_JOB_ID"))
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_complete_ad_reasoning_resume_job_id_missing"}, indent=2))
        log_reasoning_resume_failed(
            logger,
            job_id="(none)",
            failure_stage="startup",
            failure_reason="builder2_complete_ad_reasoning_resume_job_id_missing",
        )
        return 1
    max_calls = _env_int("BUILDER2_COMPLETE_AD_REASONING_RESUME_MAX_CALLS", DEFAULT_MAX_CALLS)
    stop_before_media = _env_bool("BUILDER2_COMPLETE_AD_REASONING_RESUME_STOP_BEFORE_MEDIA", True)
    report = run_controlled_complete_ad_reasoning_resume(
        job_id=job_id,
        max_calls=max_calls,
        stop_before_media=stop_before_media,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok"):
        log_reasoning_resume_failed(
            logger,
            job_id=job_id,
            tournament_id=_clean(report.get("tournamentId")),
            failure_stage=str(report.get("failureStage") or "unknown"),
            failure_reason=str(report.get("failureReason") or "unknown"),
            lease_released=True,
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
