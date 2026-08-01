"""
Builder2 complete-ad resume stage and role planning.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    can_offline_revalidate_rejected_creator,
    find_rejected_creator_for_prototype,
)
from engine.builder2_tournament_completion_gate import (
    assigned_prototype_ids,
    is_tournament_ready_for_winner_selection,
    missing_creator_prototype_ids,
    missing_judge_prototype_ids,
    tournament_resolution_summary,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import load_revalidatable_parsed_winner_response
from engine.builder2_judge_pending_repair import (
    pending_judge_repair_candidate_ids,
    pending_judge_repair_prototype_ids,
    resolve_pending_judge_repair,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def resolve_complete_ad_resume_stage(state: Dict[str, Any], *, read_only: bool = False) -> str:
    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        return "strategy"

    if missing_creator_prototype_ids(state, read_only=read_only):
        return "creator_generation"

    if missing_judge_prototype_ids(state, read_only=read_only):
        return "judge_generation"

    if not is_tournament_ready_for_winner_selection(state, read_only=read_only):
        if missing_creator_prototype_ids(state, read_only=read_only):
            return "creator_generation"
        return "judge_generation"

    if not _clean(state.get("winnerCandidateId")):
        return "winner_selection"

    if not is_valid_persisted_winner_development(state):
        return "winner_development"

    return "media_prerequisite_validation"


def plan_complete_ad_reasoning_roles(
    state: Dict[str, Any],
    *,
    read_only: bool = False,
) -> Dict[str, Any]:
    summary = tournament_resolution_summary(state, read_only=read_only)
    assigned_count = len(assigned_prototype_ids(state))
    missing_creators = list(summary.get("missingCreatorPrototypeIds") or [])
    missing_judges = list(summary.get("missingJudgePrototypeIds") or [])

    required: List[str] = []
    conditional: List[str] = []

    offline_creator_slots = 0
    for prototype_id in missing_creators:
        payload = find_rejected_creator_for_prototype(state, prototype_id)
        candidate_id = _clean((payload or {}).get("candidateId"))
        if payload and candidate_id and can_offline_revalidate_rejected_creator(state, candidate_id=candidate_id)[0]:
            offline_creator_slots += 1
            continue
        if "builder2_creator" not in required:
            required.append("builder2_creator")

    pending_judge_prototypes = set(missing_judges)
    pending_judge_prototypes.update(missing_creators)
    if pending_judge_prototypes and "builder2_judge" not in required:
        required.append("builder2_judge")

    parsed_winner = load_revalidatable_parsed_winner_response(state)
    parsed_winner_candidate_id = _clean((parsed_winner or {}).get("candidateId"))
    winner_dev_valid = is_valid_persisted_winner_development(state)
    winner_ready = bool(summary.get("readyForAuthoritativeWinnerSelection")) and bool(
        _clean(state.get("winnerCandidateId"))
    )

    if (
        summary.get("readyForAuthoritativeWinnerSelection")
        and not winner_ready
        and not winner_dev_valid
    ):
        conditional.append("builder2_winner")
    elif parsed_winner and not winner_dev_valid:
        conditional.append("builder2_winner")
    elif _clean(state.get("winnerCandidateId")) and not winner_dev_valid:
        conditional.append("builder2_winner")

    expected: List[str] = list(required)
    if conditional:
        expected.append("builder2_winner_if_winner_changes")

    paid_required = [role for role in required if role.startswith("builder2_")]
    minimum_calls = len(paid_required)
    maximum_calls = minimum_calls + (1 if conditional else 0)

    return {
        "requiredNextReasoningRoles": required,
        "conditionalNextReasoningRoles": conditional,
        "expectedNextReasoningRoles": expected,
        "minimumAdditionalReasoningCalls": minimum_calls,
        "maximumAdditionalReasoningCalls": maximum_calls,
        "offlineCreatorRevalidationSlots": offline_creator_slots,
        "assignedPrototypeCount": assigned_count,
        "summary": summary,
        "parsedWinnerCandidateId": parsed_winner_candidate_id or None,
    }


def assert_resume_ready_for_creator_generation(state: Dict[str, Any]) -> None:
    from engine.builder2_tournament_contracts import Builder2TournamentError

    assigned_count = len(assigned_prototype_ids(state))
    if assigned_count <= 0:
        raise Builder2TournamentError("builder2_complete_ad_resume_creator_gate:no_assigned_prototypes")
    missing = missing_creator_prototype_ids(state)
    if not missing:
        raise Builder2TournamentError("builder2_complete_ad_resume_creator_gate:no_missing_creators")


def assert_resume_ready_for_winner_selection(state: Dict[str, Any]) -> None:
    from engine.builder2_tournament_completion_gate import assert_tournament_ready_for_winner_selection

    assert_tournament_ready_for_winner_selection(state)


def assert_resume_ready_for_winner_development(state: Dict[str, Any]) -> None:
    from engine.builder2_tournament_contracts import Builder2TournamentError

    if not _clean(state.get("winnerCandidateId")):
        raise Builder2TournamentError("builder2_complete_ad_resume_winner_gate:authoritative_winner_missing")
    if not is_tournament_ready_for_winner_selection(state):
        raise Builder2TournamentError("builder2_complete_ad_resume_winner_gate:tournament_incomplete")


def parsed_winner_reusable_for_candidate(
    state: Dict[str, Any],
    *,
    winner_candidate_id: str,
) -> bool:
    parsed = load_revalidatable_parsed_winner_response(state)
    if not parsed:
        return False
    return _clean(parsed.get("candidateId")) == _clean(winner_candidate_id)


RESUME_STAGE_STRATEGY = "strategy"
RESUME_STAGE_CREATOR_GENERATION = "creator_generation"
RESUME_STAGE_JUDGE_GENERATION = "judge_generation"
RESUME_STAGE_MIXED_PARTIAL = "mixed_partial_reasoning"
RESUME_STAGE_WINNER_SELECTION = "winner_selection"
RESUME_STAGE_WINNER_DEVELOPMENT = "winner_development"
RESUME_STAGE_MEDIA_PREREQUISITE = "media_prerequisite_validation"
RESUME_STAGE_REASONING_COMPLETE = "reasoning_complete"
RESUME_STAGE_UNSUPPORTED = "unsupported"
PER_INVOCATION_REASONING_CALL_LIMIT = 3

_FAILED_JOB_STATUSES = frozenset({"failed", "error"})


def accepted_candidate_id_for_prototype(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    read_only: bool = False,
) -> str:
    from engine.builder2_accepted_creator_store import ACCEPTED_CREATOR_INDEX_KEY, backfill_accepted_creator_index

    if not read_only:
        backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    for candidate_id, rec in index.items():
        if isinstance(rec, dict) and _clean(rec.get("prototypeId")) == prototype_id:
            return str(candidate_id)
    for candidate_id, rec in (state.get("candidates") or {}).items():
        if not isinstance(rec, dict) or _clean(rec.get("prototypeId")) != prototype_id:
            continue
        if rec.get("validationStatus") == "accepted" or rec.get("creatorAcceptanceStatus") == "accepted":
            return str(candidate_id)
    return ""


def is_mixed_partial_resume_pattern(
    *,
    accepted_creator_count: int,
    accepted_judgment_count: int,
    missing_creator_prototype_ids: List[str],
    missing_judgment_prototype_ids: List[str],
    assigned_prototype_count: int,
) -> bool:
    if assigned_prototype_count < 6:
        return False
    if accepted_creator_count == 6 and accepted_judgment_count == 6:
        return False
    if accepted_creator_count == 6 and not missing_creator_prototype_ids and accepted_judgment_count < 6:
        return False
    if accepted_creator_count == 5 and accepted_judgment_count == 5 and len(missing_creator_prototype_ids) == 1:
        return False
    if accepted_creator_count == 6 and missing_creator_prototype_ids:
        return False
    if accepted_creator_count <= 0:
        return False
    if accepted_judgment_count > accepted_creator_count:
        return False
    if (
        accepted_creator_count == accepted_judgment_count
        and 0 < accepted_creator_count < 6
        and accepted_creator_count != 5
    ):
        return False
    if accepted_creator_count < 6 or accepted_judgment_count < 6:
        return True
    return False


def build_resume_plan_by_prototype(
    state: Dict[str, Any],
    *,
    read_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    missing_creators = set(missing_creator_prototype_ids(state, read_only=read_only))
    missing_judges = set(missing_judge_prototype_ids(state, read_only=read_only))
    for prototype_id in assigned_prototype_ids(state):
        candidate_id = accepted_candidate_id_for_prototype(state, prototype_id, read_only=read_only)
        pending = resolve_pending_judge_repair(state, candidate_id) if candidate_id else None
        has_creator = prototype_id not in missing_creators
        has_judgment = prototype_id not in missing_judges
        if has_judgment:
            creator_action = "reuse"
            judge_action = "reuse"
            normal_judge_calls = 0
            repair_judge_calls = 0
        elif pending:
            creator_action = "reuse"
            judge_action = "dispatch_repair"
            normal_judge_calls = 0
            repair_judge_calls = 1
        elif has_creator:
            creator_action = "reuse"
            judge_action = "dispatch"
            normal_judge_calls = 1
            repair_judge_calls = 0
        else:
            creator_action = "dispatch"
            judge_action = "dispatch_after_creator"
            normal_judge_calls = 1
            repair_judge_calls = 0
        entry: Dict[str, Any] = {
            "creatorAction": creator_action,
            "judgeAction": judge_action,
            "normalJudgeCallRequired": normal_judge_calls > 0,
            "repairJudgeCallRequired": repair_judge_calls > 0,
            "normalJudgeCalls": normal_judge_calls,
            "repairJudgeCalls": repair_judge_calls,
        }
        if pending:
            entry["sourceJudgmentId"] = pending.get("sourceJudgmentId")
            entry["sourceParsedResponseFingerprint"] = pending.get("sourceParsedResponseFingerprint")
            entry["sourceResponseFingerprint"] = pending.get("sourceResponseFingerprint")
            entry["pendingJudgeRepair"] = dict(pending)
        plan[prototype_id] = entry
    return plan


def compute_mixed_partial_call_plan(
    *,
    missing_creator_prototype_ids: List[str],
    missing_judgment_prototype_ids: List[str],
    required_judge_repair_calls: int = 0,
    per_invocation_call_limit: int = 3,
) -> Dict[str, Any]:
    remaining_creator = len(missing_creator_prototype_ids)
    remaining_judge_normal = max(0, len(missing_judgment_prototype_ids) - int(required_judge_repair_calls or 0))
    required_repairs = int(required_judge_repair_calls or 0)
    normal_before_winner = remaining_creator + remaining_judge_normal
    total_paid_before_winner = normal_before_winner + required_repairs
    return {
        "remainingCreatorNormalCalls": remaining_creator,
        "remainingJudgeNormalCalls": remaining_judge_normal,
        "requiredJudgeRepairCalls": required_repairs,
        "normalCallsBeforeWinner": normal_before_winner,
        "totalNormalCallsBeforeWinner": normal_before_winner,
        "totalRequiredRepairCallsBeforeWinner": required_repairs,
        "totalPaidCallsBeforeWinner": total_paid_before_winner,
        "conditionalWinnerNormalCalls": 1,
        "conditionalWinnerCalls": 1,
        "winnerNormalCallConditional": True,
        "possibleRepairCallsNotIncluded": True,
        "possibleFutureRepairCallsNotIncluded": True,
        "minimumAdditionalNormalReasoningCalls": normal_before_winner,
        "maximumAdditionalNormalReasoningCalls": normal_before_winner + 1,
        "minimumAdditionalReasoningCallsWithoutRepairs": normal_before_winner,
        "maximumAdditionalReasoningCallsWithoutRepairs": normal_before_winner + 1,
        "minimumAdditionalPaidReasoningCalls": total_paid_before_winner,
        "maximumAdditionalPaidReasoningCallsWithoutFutureRepairs": total_paid_before_winner + 1,
        "perInvocationCallLimit": max(1, int(per_invocation_call_limit or 3)),
        "totalCallsRemainingAcrossInvocations": total_paid_before_winner,
    }


def _guarded_failed_job_resume_allowed(state: Dict[str, Any], *, resume_eligible: bool) -> tuple[bool, str]:
    status = _clean(state.get("status")).lower()
    if status not in _FAILED_JOB_STATUSES:
        return True, ""
    if not resume_eligible:
        return False, "builder2_complete_ad_reasoning_resume_failed_status_inconsistent_state"
    if _media_started(state):
        return False, "builder2_complete_ad_reasoning_resume_failed_status_media_started"
    if _clean(state.get("winnerCandidateId")):
        return False, "builder2_complete_ad_reasoning_resume_failed_status_winner_present"
    if is_valid_persisted_winner_development(state):
        return False, "builder2_complete_ad_reasoning_resume_failed_status_winner_development_present"
    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        return False, "builder2_complete_ad_reasoning_resume_failed_status_strategy_missing"
    return True, ""


def _media_started(state: Dict[str, Any]) -> bool:
    if bool(state.get("mediaStarted")):
        return True
    media = state.get("mediaResume")
    if not isinstance(media, dict):
        return False
    return bool(_clean(media.get("startImageArtifact")) or _clean(media.get("runwayTaskId")))


def resolve_complete_ad_canonical_resume_plan(
    state: Dict[str, Any],
    *,
    read_only: bool = False,
    job_raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single side-effect-free resume plan shared by inspectors and executors.
    """
    from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
    from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION

    job_id = _clean(state.get("jobId"))
    tournament_id = _clean(state.get("tournamentId"))
    summary = tournament_resolution_summary(state, read_only=read_only)
    role_plan = plan_complete_ad_reasoning_roles(state, read_only=read_only)
    resolved_stage = resolve_complete_ad_resume_stage(state, read_only=read_only)
    missing_creators = list(summary.get("missingCreatorPrototypeIds") or [])
    missing_judges = list(summary.get("missingJudgePrototypeIds") or [])
    accepted_creators = int(summary.get("acceptedCreatorCount") or 0)
    accepted_judgments = int(summary.get("acceptedJudgmentCount") or 0)
    assigned = assigned_prototype_ids(state)

    strategy = state.get("strategyFoundation")
    strategy_present = isinstance(strategy, dict) and bool(strategy)

    resume_version = _clean(state.get("builder2ResumeContractVersion") or (job_raw or {}).get("builder2ResumeContractVersion"))
    new_format = _clean(state.get("builder2NewFormatVersion") or (job_raw or {}).get("builder2NewFormatVersion"))

    rejection_reason = ""
    resume_eligible = True

    if not job_id:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_job_id_missing"
    elif not tournament_id:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_tournament_id_missing"
    elif resume_version and resume_version != BUILDER2_RESUME_CONTRACT_VERSION:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_contract_mismatch"
    elif new_format and new_format != BUILDER2_NEW_FORMAT_VERSION:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_new_format_mismatch"
    elif not strategy_present:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_strategy_missing"
    elif len(assigned) < 6:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_not_six_way"
    elif accepted_creators > 6 or accepted_judgments > 6:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_invalid_counts"
    elif _media_started(state):
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_media_already_started"
    elif accepted_creators == 6 and accepted_judgments == 6:
        if not is_valid_persisted_winner_development(state):
            resolved_stage = RESUME_STAGE_WINNER_DEVELOPMENT if _clean(state.get("winnerCandidateId")) else RESUME_STAGE_WINNER_SELECTION
    elif accepted_creators == 6 and not missing_creators and missing_judges:
        resolved_stage = RESUME_STAGE_JUDGE_GENERATION
    elif accepted_creators == 5 and accepted_judgments == 5 and len(missing_creators) == 1:
        resolved_stage = RESUME_STAGE_CREATOR_GENERATION
    elif accepted_creators == 6 and accepted_judgments < 6 and missing_creators:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_unexpected_missing_creator"
    elif accepted_creators == 6 and accepted_judgments < 6 and not missing_judges:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_unexpected_partial_state"
    elif is_mixed_partial_resume_pattern(
        accepted_creator_count=accepted_creators,
        accepted_judgment_count=accepted_judgments,
        missing_creator_prototype_ids=missing_creators,
        missing_judgment_prototype_ids=missing_judges,
        assigned_prototype_count=len(assigned),
    ):
        resolved_stage = RESUME_STAGE_MIXED_PARTIAL
    elif accepted_creators != 5 or accepted_judgments != 5:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_unexpected_partial_state"

    if resume_eligible:
        failed_ok, failed_reason = _guarded_failed_job_resume_allowed(state, resume_eligible=resume_eligible)
        if not failed_ok:
            resume_eligible = False
            rejection_reason = failed_reason

    resume_plan_by_prototype = build_resume_plan_by_prototype(state, read_only=read_only)
    required_judge_repair_calls = sum(
        int((entry or {}).get("repairJudgeCalls") or 0) for entry in resume_plan_by_prototype.values()
    )
    pending_repair_candidate_ids = pending_judge_repair_candidate_ids(state)
    mixed_call_plan = compute_mixed_partial_call_plan(
        missing_creator_prototype_ids=missing_creators,
        missing_judgment_prototype_ids=missing_judges,
        required_judge_repair_calls=required_judge_repair_calls,
        per_invocation_call_limit=PER_INVOCATION_REASONING_CALL_LIMIT,
    )
    incomplete_prototypes = sorted(set(missing_creators) | set(missing_judges))

    from engine.builder2_strategy_evidence_grounding_contract import strategy_fingerprint

    strategy_fingerprint_value = strategy_fingerprint(strategy) if strategy_present and isinstance(strategy, dict) else ""

    judge_calls_planned = (
        mixed_call_plan["remainingJudgeNormalCalls"]
        if resolved_stage == RESUME_STAGE_MIXED_PARTIAL
        else (len(missing_judges) if resolved_stage == RESUME_STAGE_JUDGE_GENERATION else 0)
    )
    creator_calls_planned = (
        mixed_call_plan["remainingCreatorNormalCalls"]
        if resolved_stage == RESUME_STAGE_MIXED_PARTIAL
        else (1 if resolved_stage == RESUME_STAGE_CREATOR_GENERATION and missing_creators else 0)
    )

    ready_for_winner_development = (
        accepted_creators == 6
        and accepted_judgments == 6
        and bool(summary.get("readyForAuthoritativeWinnerSelection"))
    )

    return {
        "jobId": job_id or None,
        "tournamentId": tournament_id or None,
        "jobStatus": _clean(state.get("status")) or None,
        "pauseReason": _clean(state.get("failureReason")) or None,
        "failureStage": _clean(state.get("failureStage")) or None,
        "progressStage": _clean(state.get("progressStage")) or None,
        "strategyPresent": strategy_present,
        "strategyReusable": strategy_present,
        "strategyFingerprint": strategy_fingerprint_value or None,
        "strategyWouldDispatch": False if strategy_present else resolved_stage == RESUME_STAGE_STRATEGY,
        "acceptedCreatorCount": accepted_creators,
        "acceptedJudgmentCount": accepted_judgments,
        "missingCreatorPrototypeIds": missing_creators,
        "missingJudgmentPrototypeIds": missing_judges,
        "incompletePrototypeIds": incomplete_prototypes,
        "missingPrototypeIds": incomplete_prototypes,
        "resumePlanByPrototype": resume_plan_by_prototype,
        "resolvedResumeStage": resolved_stage,
        "resumeEligible": resume_eligible,
        "executorWouldAcceptState": resume_eligible,
        "executorRejectionReason": rejection_reason or None,
        "readyForJudges": accepted_creators == 6 and not missing_creators and bool(missing_judges),
        "readyForWinnerDevelopment": ready_for_winner_development,
        "winnerDevelopmentStarted": is_valid_persisted_winner_development(state),
        "reasoningComplete": bool(state.get("reasoningComplete")),
        "mediaStarted": _media_started(state),
        "requiredNextReasoningRoles": list(role_plan.get("requiredNextReasoningRoles") or []),
        "expectedNextReasoningRoles": list(role_plan.get("expectedNextReasoningRoles") or []),
        "judgeCallsPlanned": judge_calls_planned,
        "creatorCallsPlanned": creator_calls_planned,
        "remainingCreatorNormalCalls": mixed_call_plan["remainingCreatorNormalCalls"],
        "remainingJudgeNormalCalls": mixed_call_plan["remainingJudgeNormalCalls"],
        "normalCallsBeforeWinner": mixed_call_plan["normalCallsBeforeWinner"],
        "conditionalWinnerCalls": mixed_call_plan["conditionalWinnerCalls"],
        "winnerNormalCallConditional": mixed_call_plan["winnerNormalCallConditional"],
        "possibleRepairCallsNotIncluded": mixed_call_plan["possibleRepairCallsNotIncluded"],
        "minimumAdditionalNormalReasoningCalls": mixed_call_plan["minimumAdditionalNormalReasoningCalls"],
        "maximumAdditionalNormalReasoningCalls": mixed_call_plan["maximumAdditionalNormalReasoningCalls"],
        "minimumAdditionalReasoningCallsWithoutRepairs": mixed_call_plan["minimumAdditionalReasoningCallsWithoutRepairs"],
        "maximumAdditionalReasoningCallsWithoutRepairs": mixed_call_plan["maximumAdditionalReasoningCallsWithoutRepairs"],
        "requiredJudgeRepairCalls": mixed_call_plan["requiredJudgeRepairCalls"],
        "totalNormalCallsBeforeWinner": mixed_call_plan["totalNormalCallsBeforeWinner"],
        "totalRequiredRepairCallsBeforeWinner": mixed_call_plan["totalRequiredRepairCallsBeforeWinner"],
        "totalPaidCallsBeforeWinner": mixed_call_plan["totalPaidCallsBeforeWinner"],
        "conditionalWinnerNormalCalls": mixed_call_plan["conditionalWinnerNormalCalls"],
        "minimumAdditionalPaidReasoningCalls": mixed_call_plan["minimumAdditionalPaidReasoningCalls"],
        "maximumAdditionalPaidReasoningCallsWithoutFutureRepairs": mixed_call_plan[
            "maximumAdditionalPaidReasoningCallsWithoutFutureRepairs"
        ],
        "pendingJudgeRepairCandidateIds": pending_repair_candidate_ids,
        "pendingJudgeRepairCount": len(pending_repair_candidate_ids),
        "pendingJudgeRepairPrototypeIds": pending_judge_repair_prototype_ids(state),
        "perInvocationCallLimit": mixed_call_plan["perInvocationCallLimit"],
        "totalCallsRemainingAcrossInvocations": mixed_call_plan["totalCallsRemainingAcrossInvocations"],
        "creatorsWouldDispatch": creator_calls_planned > 0,
        "winnerWouldDispatch": resolved_stage in {
            RESUME_STAGE_WINNER_SELECTION,
            RESUME_STAGE_WINNER_DEVELOPMENT,
        },
        "mediaWouldDispatch": resolved_stage == RESUME_STAGE_MEDIA_PREREQUISITE,
        "summary": summary,
        "rolePlan": role_plan,
    }


def evaluate_complete_ad_reasoning_executor_preconditions(
    state: Dict[str, Any],
    job_raw: Optional[Dict[str, Any]] = None,
) -> tuple[bool, Optional[str], Dict[str, Any]]:
    plan = resolve_complete_ad_canonical_resume_plan(state, read_only=False, job_raw=job_raw)
    if not plan["executorWouldAcceptState"]:
        return False, plan["executorRejectionReason"], plan
    return True, None, plan
