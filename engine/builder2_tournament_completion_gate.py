"""
Builder2 tournament completion gate — six prototype slots before authoritative winner selection.
"""
from __future__ import annotations

import hashlib
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.builder2_accepted_creator_store import (
    ACCEPTED_CREATOR_INDEX_KEY,
    backfill_accepted_creator_index,
    derive_accepted_creator_index,
)
from engine.builder2_accepted_judgment_store import (
    ACCEPTED_JUDGMENT_INDEX_KEY,
    backfill_accepted_judgment_index,
    derive_accepted_judgment_index,
)
from engine.builder2_judge_unavailable_resolution_contract import (
    unavailable_judgment_count,
    resolved_judgment_outcome_count,
)
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_tournament_contracts import Builder2TournamentError

TOURNAMENT_INCOMPLETE_BEFORE_WINNER = "builder2_tournament_incomplete_before_winner"
TOURNAMENT_NO_ELIGIBLE_WINNER = "builder2_tournament_no_eligible_winner"
STRICT_SIX_WAY_PROTOTYPE_COUNT = 6

PROTOTYPE_TERMINAL_OUTCOME_ACCEPTED_JUDGED = "accepted_creator_completed_judgment"
PROTOTYPE_TERMINAL_OUTCOME_CREATOR_REJECTED = "structurally_rejected_creator"
PROTOTYPE_TERMINAL_OUTCOME_JUDGE_UNAVAILABLE = "judge_unavailable"
PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED = "unresolved"

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def assigned_prototype_ids(state: Dict[str, Any]) -> List[str]:
    active = state.get("initialActivePrototypeIds") or state.get("activePrototypeIds")
    if isinstance(active, list) and active:
        return [str(pid).strip() for pid in active if str(pid).strip()]
    return list(resolve_builder2_active_prototype_ids())


def accepted_creator_index(state: Dict[str, Any], *, read_only: bool = False) -> Dict[str, Any]:
    if read_only:
        existing = state.get(ACCEPTED_CREATOR_INDEX_KEY)
        existing_count = len(existing) if isinstance(existing, dict) else 0
        merged = derive_accepted_creator_index(state)
        added = len(merged) - existing_count
        if added > 0:
            logger.info(
                "BUILDER2_ACCEPTED_CREATOR_INDEX_DERIVED_READ_ONLY jobId=%s tournamentId=%s count=%s",
                state.get("jobId"),
                state.get("tournamentId"),
                added,
            )
        return merged
    backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY)
    return index if isinstance(index, dict) else {}


def accepted_judgment_index(state: Dict[str, Any], *, read_only: bool = False) -> Dict[str, Any]:
    if read_only:
        existing = state.get(ACCEPTED_JUDGMENT_INDEX_KEY)
        existing_count = len(existing) if isinstance(existing, dict) else 0
        merged = derive_accepted_judgment_index(state)
        added = len(merged) - existing_count
        if added > 0:
            logger.info(
                "BUILDER2_ACCEPTED_JUDGMENT_INDEX_DERIVED_READ_ONLY jobId=%s tournamentId=%s count=%s",
                state.get("jobId"),
                state.get("tournamentId"),
                added,
            )
        return merged
    backfill_accepted_judgment_index(state)
    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY)
    return index if isinstance(index, dict) else {}


def accepted_creator_count(state: Dict[str, Any], *, read_only: bool = False) -> int:
    return len(accepted_creator_index(state, read_only=read_only))


def accepted_judgment_count(state: Dict[str, Any], *, read_only: bool = False) -> int:
    return len(accepted_judgment_index(state, read_only=read_only))


def prototype_ids_with_accepted_creators(state: Dict[str, Any], *, read_only: bool = False) -> Set[str]:
    accepted = {
        _clean(rec.get("prototypeId"))
        for rec in accepted_creator_index(state, read_only=read_only).values()
        if isinstance(rec, dict) and _clean(rec.get("prototypeId"))
    }
    assigned = set(assigned_prototype_ids(state))
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        prototype_id = _clean(rec.get("prototypeId"))
        if prototype_id not in assigned:
            continue
        if rec.get("validationStatus") == "accepted" or rec.get("creatorAcceptanceStatus") == "accepted":
            accepted.add(prototype_id)
    return accepted


def prototype_ids_with_accepted_judgments(state: Dict[str, Any], *, read_only: bool = False) -> Set[str]:
    judged = {
        _clean(rec.get("prototypeId"))
        for rec in accepted_judgment_index(state, read_only=read_only).values()
        if isinstance(rec, dict) and _clean(rec.get("prototypeId"))
    }
    assigned = set(assigned_prototype_ids(state))
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        prototype_id = _clean(rec.get("prototypeId"))
        if prototype_id not in assigned:
            continue
        if _clean(rec.get("judgmentId")) and rec.get("validationStatus") == "accepted":
            judged.add(prototype_id)
    return judged


def missing_creator_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    assigned = assigned_prototype_ids(state)
    accepted = prototype_ids_with_accepted_creators(state, read_only=read_only)
    rejected = set(structurally_rejected_creator_prototype_ids(state))
    return [pid for pid in assigned if pid not in accepted and pid not in rejected]


def missing_judge_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    assigned = assigned_prototype_ids(state)
    judged = prototype_ids_with_accepted_judgments(state, read_only=read_only)
    rejected = set(structurally_rejected_creator_prototype_ids(state))
    judge_terminal = terminal_judge_prototype_ids(state)
    accepted_creators = prototype_ids_with_accepted_creators(state, read_only=read_only)
    return [
        pid
        for pid in assigned
        if pid in accepted_creators and pid not in judged and pid not in judge_terminal
    ]


def missing_actionable_judge_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    from engine.builder2_judge_unavailable_resolution_contract import (
        has_operator_judgment_unavailable_resolution,
        prototype_id_for_candidate,
    )

    assigned = assigned_prototype_ids(state)
    judged = prototype_ids_with_accepted_judgments(state, read_only=read_only)
    terminal = terminal_judge_prototype_ids(state)
    excluded = set(terminal)
    accepted_creators = prototype_ids_with_accepted_creators(state, read_only=read_only)
    for candidate_id in (state.get("candidates") or {}).keys():
        if has_operator_judgment_unavailable_resolution(state, str(candidate_id)):
            prototype_id = prototype_id_for_candidate(state, str(candidate_id))
            if prototype_id:
                excluded.add(prototype_id)
    return [
        pid
        for pid in assigned
        if pid in accepted_creators and pid not in judged and pid not in excluded
    ]


def resolved_unavailable_judge_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    from engine.builder2_judge_unavailable_resolution_contract import (
        has_operator_judgment_unavailable_resolution,
        prototype_id_for_candidate,
    )

    resolved: List[str] = []
    for candidate_id in (state.get("candidates") or {}).keys():
        candidate_key = str(candidate_id)
        if has_operator_judgment_unavailable_resolution(state, candidate_key):
            prototype_id = prototype_id_for_candidate(state, candidate_key)
            if prototype_id:
                resolved.append(prototype_id)
    return sorted(set(resolved))


def structurally_rejected_creator_prototype_ids(state: Dict[str, Any]) -> List[str]:
    rejected: List[str] = []
    assigned = set(assigned_prototype_ids(state))
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        prototype_id = _clean(rec.get("prototypeId"))
        if prototype_id not in assigned:
            continue
        if rec.get("validationStatus") == "creator_rejected" or rec.get("status") == "creator_rejected":
            rejected.append(prototype_id)
    return sorted(set(rejected))


def terminal_judge_prototype_ids(state: Dict[str, Any]) -> Set[str]:
    terminal: Set[str] = set()
    assigned = set(assigned_prototype_ids(state))
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        prototype_id = _clean(rec.get("prototypeId"))
        if prototype_id not in assigned:
            continue
        if rec.get("validationStatus") == "judge_unavailable" or rec.get("status") == "judge_unavailable":
            terminal.add(prototype_id)
    return terminal


def uses_strict_six_way_winner_gate(state: Dict[str, Any]) -> bool:
    return len(assigned_prototype_ids(state)) >= STRICT_SIX_WAY_PROTOTYPE_COUNT


def resolve_prototype_terminal_outcome(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    read_only: bool = False,
) -> str:
    if prototype_id in structurally_rejected_creator_prototype_ids(state):
        return PROTOTYPE_TERMINAL_OUTCOME_CREATOR_REJECTED
    if prototype_id not in prototype_ids_with_accepted_creators(state, read_only=read_only):
        return PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED
    if prototype_id in prototype_ids_with_accepted_judgments(state, read_only=read_only):
        return PROTOTYPE_TERMINAL_OUTCOME_ACCEPTED_JUDGED
    if prototype_id in terminal_judge_prototype_ids(state):
        return PROTOTYPE_TERMINAL_OUTCOME_JUDGE_UNAVAILABLE
    return PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED


def collect_terminal_prototype_slots(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    return [
        prototype_id
        for prototype_id in assigned_prototype_ids(state)
        if is_prototype_slot_terminal(state, prototype_id, read_only=read_only)
    ]


def count_eligible_judged_candidates(state: Dict[str, Any]) -> int:
    return len(_eligible_winner_candidate_ids(state))


def _eligible_winner_candidate_ids(state: Dict[str, Any]) -> List[str]:
    from engine.builder2_metaphorical_embodiment_contract import judgment_rejects_literal_execution
    from engine.builder2_no_logo_contract import judgment_rejects_logo_policy
    from engine.builder2_tournament_manager import _creator_was_accepted, _has_valid_judgment, _resolve_judgment_for_candidate

    eligible_ids: List[str] = []
    for candidate_id, cand in (state.get("candidates") or {}).items():
        if not isinstance(cand, dict):
            continue
        if not _creator_was_accepted(cand, state=state):
            continue
        if not _has_valid_judgment(cand):
            continue
        if not cand.get("eligible"):
            continue
        judgment = _resolve_judgment_for_candidate(state, str(candidate_id))
        if judgment_rejects_literal_execution(judgment) if isinstance(judgment, dict) else False:
            continue
        if judgment_rejects_logo_policy(judgment) if isinstance(judgment, dict) else False:
            continue
        eligible_ids.append(str(candidate_id))
    return eligible_ids


def tournament_terminal_slot_diagnostics(state: Dict[str, Any], *, read_only: bool = False) -> Dict[str, Any]:
    assigned = assigned_prototype_ids(state)
    terminal_ids = collect_terminal_prototype_slots(state, read_only=read_only)
    rejected = structurally_rejected_creator_prototype_ids(state)
    accepted_creators = accepted_creator_count(state, read_only=read_only)
    accepted_judgments = accepted_judgment_count(state, read_only=read_only)
    eligible_ids = _eligible_winner_candidate_ids(state)
    assigned_count = len(assigned)
    terminal_count = len(terminal_ids)
    return {
        "assignedPrototypeCount": assigned_count,
        "terminalPrototypeCount": terminal_count,
        "acceptedCreatorCount": accepted_creators,
        "rejectedCreatorCount": len(rejected),
        "completedJudgmentCount": accepted_judgments,
        "eligibleCandidateCount": len(eligible_ids),
        "winnerSelectionReady": terminal_count == assigned_count and assigned_count > 0,
        "degradedTournament": terminal_count == assigned_count and accepted_creators < assigned_count,
        "terminalPrototypeIds": terminal_ids,
        "eligibleCandidateIds": eligible_ids,
    }


def log_tournament_terminal_slots(state: Dict[str, Any], *, read_only: bool = False) -> None:
    diag = tournament_terminal_slot_diagnostics(state, read_only=read_only)
    logger.info(
        "BUILDER2_TOURNAMENT_TERMINAL_SLOTS assigned=%s terminal=%s acceptedCreators=%s "
        "rejectedCreators=%s completedJudgments=%s eligibleCandidates=%s",
        diag["assignedPrototypeCount"],
        diag["terminalPrototypeCount"],
        diag["acceptedCreatorCount"],
        diag["rejectedCreatorCount"],
        diag["completedJudgmentCount"],
        diag["eligibleCandidateCount"],
    )
    if diag["winnerSelectionReady"] and diag["eligibleCandidateCount"]:
        logger.info(
            "BUILDER2_WINNER_POOL_READY eligibleCandidates=%s degradedTournament=%s",
            diag["eligibleCandidateCount"],
            diag["degradedTournament"],
        )


def is_prototype_slot_terminal(state: Dict[str, Any], prototype_id: str, *, read_only: bool = False) -> bool:
    return resolve_prototype_terminal_outcome(state, prototype_id, read_only=read_only) != PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED


def unresolved_creator_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    assigned = assigned_prototype_ids(state)
    rejected = set(structurally_rejected_creator_prototype_ids(state))
    accepted = prototype_ids_with_accepted_creators(state, read_only=read_only)
    return [pid for pid in assigned if pid not in accepted and pid not in rejected]


def unresolved_judge_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    unresolved: List[str] = []
    assigned = assigned_prototype_ids(state)
    rejected = set(structurally_rejected_creator_prototype_ids(state))
    judged = prototype_ids_with_accepted_judgments(state, read_only=read_only)
    judge_terminal = terminal_judge_prototype_ids(state)
    accepted_creators = prototype_ids_with_accepted_creators(state, read_only=read_only)
    for pid in assigned:
        if pid in rejected:
            continue
        if pid in judged:
            continue
        if pid in judge_terminal:
            continue
        if pid in accepted_creators:
            unresolved.append(pid)
    return unresolved


def tournament_resolution_summary(state: Dict[str, Any], *, read_only: bool = False) -> Dict[str, Any]:
    assigned = assigned_prototype_ids(state)
    diagnostics = tournament_terminal_slot_diagnostics(state, read_only=read_only)
    missing_creators = missing_creator_prototype_ids(state, read_only=read_only)
    missing_judges = missing_judge_prototype_ids(state, read_only=read_only)
    rejected = structurally_rejected_creator_prototype_ids(state)
    unresolved_creators = unresolved_creator_prototype_ids(state, read_only=read_only)
    unresolved_judges = unresolved_judge_prototype_ids(state, read_only=read_only)
    return {
        "assignedPrototypeCount": len(assigned),
        "terminalPrototypeCount": diagnostics["terminalPrototypeCount"],
        "acceptedCreatorCount": diagnostics["acceptedCreatorCount"],
        "rejectedCreatorCount": diagnostics["rejectedCreatorCount"],
        "acceptedJudgmentCount": diagnostics["completedJudgmentCount"],
        "eligibleCandidateCount": diagnostics["eligibleCandidateCount"],
        "winnerSelectionReady": diagnostics["winnerSelectionReady"],
        "degradedTournament": diagnostics["degradedTournament"],
        "unavailableJudgmentCount": unavailable_judgment_count(state, read_only=read_only),
        "resolvedJudgmentOutcomeCount": resolved_judgment_outcome_count(state, read_only=read_only),
        "missingCreatorPrototypeIds": missing_creators,
        "missingJudgePrototypeIds": missing_judges,
        "unresolvedCreatorPrototypeIds": unresolved_creators,
        "unresolvedJudgePrototypeIds": unresolved_judges,
        "structurallyRejectedCreatorPrototypeIds": rejected,
        "terminalPrototypeIds": diagnostics["terminalPrototypeIds"],
        "eligibleCandidateIds": diagnostics["eligibleCandidateIds"],
        "readyForAuthoritativeWinnerSelection": is_tournament_ready_for_winner_selection(state, read_only=read_only),
    }


def is_tournament_ready_for_winner_selection(state: Dict[str, Any], *, read_only: bool = False) -> bool:
    assigned = assigned_prototype_ids(state)
    if not assigned:
        return False
    if unresolved_creator_prototype_ids(state, read_only=read_only):
        return False
    if unresolved_judge_prototype_ids(state, read_only=read_only):
        return False
    return all(is_prototype_slot_terminal(state, prototype_id, read_only=read_only) for prototype_id in assigned)


def assert_tournament_ready_for_winner_selection(state: Dict[str, Any]) -> None:
    summary = tournament_resolution_summary(state)
    if summary["readyForAuthoritativeWinnerSelection"]:
        log_tournament_terminal_slots(state)
        return
    parts: List[str] = []
    if summary["unresolvedCreatorPrototypeIds"]:
        parts.append("unresolvedCreators=" + ",".join(summary["unresolvedCreatorPrototypeIds"]))
    if summary["unresolvedJudgePrototypeIds"]:
        parts.append("unresolvedJudges=" + ",".join(summary["unresolvedJudgePrototypeIds"]))
    if summary["missingCreatorPrototypeIds"]:
        parts.append("missingCreators=" + ",".join(summary["missingCreatorPrototypeIds"]))
    if summary["missingJudgePrototypeIds"]:
        parts.append("missingJudges=" + ",".join(summary["missingJudgePrototypeIds"]))
    parts.append(
        f"terminal={summary['terminalPrototypeCount']}/{summary['assignedPrototypeCount']}"
    )
    parts.append(
        f"acceptedCreators={summary['acceptedCreatorCount']}"
    )
    parts.append(
        f"rejectedCreators={summary['rejectedCreatorCount']}"
    )
    parts.append(
        f"judgments={summary['acceptedJudgmentCount']}/{summary['assignedPrototypeCount']}"
    )
    if summary["structurallyRejectedCreatorPrototypeIds"]:
        parts.append(
            "rejectedCreatorPrototypeIds="
            + ",".join(summary["structurallyRejectedCreatorPrototypeIds"])
        )
    raise Builder2TournamentError(f"{TOURNAMENT_INCOMPLETE_BEFORE_WINNER}:{';'.join(parts)}")


def invalidate_provisional_winner_if_incomplete(state: Dict[str, Any]) -> bool:
    winner_id = _clean(state.get("winnerCandidateId"))
    if not winner_id:
        return False
    if is_tournament_ready_for_winner_selection(state):
        state["winnerSelectionFinal"] = True
        return False
    state["provisionalWinnerCandidateId"] = winner_id
    state["provisionalWinnerScore"] = (state.get("candidates") or {}).get(winner_id, {}).get("totalScore")
    state.pop("winnerCandidateId", None)
    state.pop("winnerDevelopmentPlan", None)
    state.pop("winnerDevelopmentCandidateId", None)
    state.pop("winnerDevelopmentPrototypeId", None)
    state["winnerSelectionFinal"] = False
    return True


def mark_authoritative_winner_selection(state: Dict[str, Any], *, winner_id: str) -> None:
    state["winnerCandidateId"] = winner_id
    state["winnerSelectionFinal"] = True
    state.pop("provisionalWinnerCandidateId", None)
    state.pop("provisionalWinnerScore", None)


def compute_slogan_identity_hash(slogan_text: str) -> str:
    normalized = _clean(slogan_text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def persist_winner_slogan_identity(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    advertising_closure: Dict[str, Any],
) -> None:
    slogan = _clean(advertising_closure.get("sloganText"))
    state["winnerSelectedSloganText"] = slogan
    state["winnerSelectedProductNameText"] = _clean(advertising_closure.get("productNameText"))
    state["winnerSelectedSloganHash"] = compute_slogan_identity_hash(slogan)
    state["winnerSloganSource"] = "creator_candidate"
    state["winningCandidateId"] = candidate_id
    state["winningPrototypeId"] = prototype_id
