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
RESUME_STAGE_WINNER_SELECTION = "winner_selection"
RESUME_STAGE_WINNER_DEVELOPMENT = "winner_development"
RESUME_STAGE_MEDIA_PREREQUISITE = "media_prerequisite_validation"
RESUME_STAGE_REASONING_COMPLETE = "reasoning_complete"
RESUME_STAGE_UNSUPPORTED = "unsupported"


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
    elif accepted_creators != 5 or accepted_judgments != 5:
        resume_eligible = False
        rejection_reason = "builder2_complete_ad_reasoning_resume_unexpected_partial_state"

    judge_calls_planned = len(missing_judges) if resolved_stage == RESUME_STAGE_JUDGE_GENERATION else 0
    creator_calls_planned = 1 if resolved_stage == RESUME_STAGE_CREATOR_GENERATION and missing_creators else 0

    strategy_would_dispatch = resolved_stage == RESUME_STAGE_STRATEGY
    creators_would_dispatch = creator_calls_planned > 0
    winner_would_dispatch = resolved_stage in {
        RESUME_STAGE_WINNER_SELECTION,
        RESUME_STAGE_WINNER_DEVELOPMENT,
    }
    media_would_dispatch = resolved_stage == RESUME_STAGE_MEDIA_PREREQUISITE

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
        "acceptedCreatorCount": accepted_creators,
        "acceptedJudgmentCount": accepted_judgments,
        "missingCreatorPrototypeIds": missing_creators,
        "missingJudgmentPrototypeIds": missing_judges,
        "missingPrototypeIds": sorted(set(missing_creators) | set(missing_judges)),
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
        "strategyWouldDispatch": strategy_would_dispatch,
        "creatorsWouldDispatch": creators_would_dispatch,
        "winnerWouldDispatch": winner_would_dispatch,
        "mediaWouldDispatch": media_would_dispatch,
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
