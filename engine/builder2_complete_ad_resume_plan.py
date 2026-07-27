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
