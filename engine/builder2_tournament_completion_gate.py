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
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_tournament_contracts import Builder2TournamentError

TOURNAMENT_INCOMPLETE_BEFORE_WINNER = "builder2_tournament_incomplete_before_winner"
STRICT_SIX_WAY_PROTOTYPE_COUNT = 6

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
    return [pid for pid in assigned if pid not in accepted]


def missing_judge_prototype_ids(state: Dict[str, Any], *, read_only: bool = False) -> List[str]:
    assigned = assigned_prototype_ids(state)
    judged = prototype_ids_with_accepted_judgments(state, read_only=read_only)
    return [pid for pid in assigned if pid not in judged]


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


def is_prototype_slot_terminal(state: Dict[str, Any], prototype_id: str, *, read_only: bool = False) -> bool:
    if prototype_id in structurally_rejected_creator_prototype_ids(state):
        return True
    if prototype_id not in prototype_ids_with_accepted_creators(state, read_only=read_only):
        return False
    if prototype_id in prototype_ids_with_accepted_judgments(state, read_only=read_only):
        return True
    return prototype_id in terminal_judge_prototype_ids(state)


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
    missing_creators = missing_creator_prototype_ids(state, read_only=read_only)
    missing_judges = missing_judge_prototype_ids(state, read_only=read_only)
    rejected = structurally_rejected_creator_prototype_ids(state)
    return {
        "assignedPrototypeCount": len(assigned),
        "acceptedCreatorCount": accepted_creator_count(state, read_only=read_only),
        "acceptedJudgmentCount": accepted_judgment_count(state, read_only=read_only),
        "missingCreatorPrototypeIds": missing_creators,
        "missingJudgePrototypeIds": missing_judges,
        "structurallyRejectedCreatorPrototypeIds": rejected,
        "readyForAuthoritativeWinnerSelection": is_tournament_ready_for_winner_selection(state, read_only=read_only),
    }


def is_tournament_ready_for_winner_selection(state: Dict[str, Any], *, read_only: bool = False) -> bool:
    assigned = assigned_prototype_ids(state)
    if not assigned:
        return False
    if uses_strict_six_way_winner_gate(state) and structurally_rejected_creator_prototype_ids(state):
        return False
    if unresolved_creator_prototype_ids(state, read_only=read_only):
        return False
    if unresolved_judge_prototype_ids(state, read_only=read_only):
        return False
    return all(is_prototype_slot_terminal(state, prototype_id, read_only=read_only) for prototype_id in assigned)


def assert_tournament_ready_for_winner_selection(state: Dict[str, Any]) -> None:
    summary = tournament_resolution_summary(state)
    if summary["readyForAuthoritativeWinnerSelection"]:
        logger.info(
            "BUILDER2_TOURNAMENT_ALL_PROTOTYPES_ACCEPTED acceptedCreators=%s acceptedJudgments=%s assigned=%s",
            summary["acceptedCreatorCount"],
            summary["acceptedJudgmentCount"],
            summary["assignedPrototypeCount"],
        )
        return
    parts: List[str] = []
    if summary["acceptedCreatorCount"] != summary["assignedPrototypeCount"]:
        parts.append(
            f"creators={summary['acceptedCreatorCount']}/{summary['assignedPrototypeCount']}"
        )
    if summary["acceptedJudgmentCount"] != summary["assignedPrototypeCount"]:
        parts.append(
            f"judgments={summary['acceptedJudgmentCount']}/{summary['assignedPrototypeCount']}"
        )
    if summary["missingCreatorPrototypeIds"]:
        parts.append("missingCreators=" + ",".join(summary["missingCreatorPrototypeIds"]))
    if summary["missingJudgePrototypeIds"]:
        parts.append("missingJudges=" + ",".join(summary["missingJudgePrototypeIds"]))
    if summary["structurallyRejectedCreatorPrototypeIds"]:
        parts.append("rejectedCreators=" + ",".join(summary["structurallyRejectedCreatorPrototypeIds"]))
    if unresolved_creator_prototype_ids(state):
        parts.append("unresolvedCreators=" + ",".join(unresolved_creator_prototype_ids(state)))
    if unresolved_judge_prototype_ids(state):
        parts.append("unresolvedJudges=" + ",".join(unresolved_judge_prototype_ids(state)))
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
