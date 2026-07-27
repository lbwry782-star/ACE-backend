"""
Builder2 complete-ad Creator recovery — persist and offline revalidate rejected parsed responses.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_accepted_creator_store import persist_accepted_creator_candidate
from engine.builder2_complete_ad_contract import validate_creator_complete_ad_fields
from engine.builder2_creator import validate_creator_candidate
from engine.builder2_tournament_contracts import Builder2TournamentError

REJECTED_CREATOR_PARSED_INDEX_KEY = "rejectedCreatorParsedResponses"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def persist_rejected_creator_parsed_response(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    parsed: Dict[str, Any],
    failure_reason: str,
    top_level_keys: Optional[List[str]] = None,
) -> None:
    if not isinstance(parsed, dict) or not parsed:
        return
    index = state.setdefault(REJECTED_CREATOR_PARSED_INDEX_KEY, {})
    if not isinstance(index, dict):
        index = {}
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = index
    index[candidate_id] = {
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "roundIndex": round_index,
        "attemptNumber": attempt_number,
        "parsed": deepcopy(parsed),
        "topLevelKeys": list(top_level_keys or sorted(parsed.keys())),
        "topLevelKeyCount": len(parsed),
        "failureReason": failure_reason,
        "storedAt": _utc_now_iso(),
    }


def load_rejected_creator_parsed_response(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        return None
    payload = index.get(candidate_id)
    if not isinstance(payload, dict):
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict) or not parsed:
        return None
    return deepcopy(payload)


def find_rejected_creator_for_prototype(state: Dict[str, Any], prototype_id: str) -> Optional[Dict[str, Any]]:
    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        return None
    for payload in index.values():
        if isinstance(payload, dict) and _clean(payload.get("prototypeId")) == prototype_id:
            return deepcopy(payload)
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        if _clean(rec.get("prototypeId")) != prototype_id:
            continue
        if rec.get("validationStatus") != "creator_rejected" and rec.get("status") != "creator_rejected":
            continue
        candidate_id = _clean(rec.get("candidateId"))
        if candidate_id:
            loaded = load_rejected_creator_parsed_response(state, candidate_id)
            if loaded:
                return loaded
    return None


def can_offline_revalidate_rejected_creator(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    product_name: str = "",
    compatibility_mode: bool = False,
) -> Tuple[bool, Optional[str]]:
    payload = load_rejected_creator_parsed_response(state, candidate_id)
    if payload is None:
        return False, "parsed_response_missing"
    parsed = dict(payload.get("parsed") or {})
    prototype_id = _clean(payload.get("prototypeId") or parsed.get("prototypeId"))
    if not prototype_id:
        return False, "prototype_id_missing"
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    try:
        candidate = validate_creator_candidate(
            parsed,
            assigned_prototype_id=prototype_id,
            prototype_display_name=prototype_id,
            strategy_foundation=strategy,
            compatibility_mode=compatibility_mode,
            job_id=_clean(state.get("jobId")),
            tournament_id=_clean(state.get("tournamentId")),
            candidate_id=candidate_id,
        )
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=strategy,
            assigned_prototype_id=prototype_id,
            product_name=product_name or _clean(strategy.get("productNameResolved")),
        )
    except Builder2TournamentError as exc:
        return False, str(exc.args[0] if exc.args else "revalidation_failed")
    return True, None


def offline_revalidate_and_accept_rejected_creator(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    product_name: str = "",
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    payload = load_rejected_creator_parsed_response(state, candidate_id)
    if payload is None:
        raise Builder2TournamentError("builder2_complete_ad_creator_revalidation_missing_parsed_response")
    parsed = dict(payload.get("parsed") or {})
    prototype_id = _clean(payload.get("prototypeId") or parsed.get("prototypeId"))
    round_index = int(payload.get("roundIndex") or parsed.get("roundIndex") or 1)
    attempt_number = int(payload.get("attemptNumber") or parsed.get("attemptNumber") or 1)
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    candidate = validate_creator_candidate(
        parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=prototype_id,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        candidate_id=candidate_id,
    )
    validate_creator_complete_ad_fields(
        candidate,
        strategy_foundation=strategy,
        assigned_prototype_id=prototype_id,
        product_name=product_name or _clean(strategy.get("productNameResolved")),
    )
    persist_accepted_creator_candidate(
        state,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        round_index=round_index,
        attempt_number=attempt_number,
        creator_output=candidate,
        strategy_foundation=strategy,
    )
    rec = state.setdefault("candidates", {}).setdefault(candidate_id, {})
    rec.update(
        {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": round_index,
            "attemptNumber": attempt_number,
            "creatorOutput": deepcopy(candidate),
            "creatorSnapshot": deepcopy(candidate),
            "validationStatus": "accepted",
            "creatorAcceptanceStatus": "accepted",
            "status": "accepted",
            "judgeStatus": "pending",
            "failureReason": None,
            "offlineRevalidatedAt": _utc_now_iso(),
        }
    )
    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if isinstance(index, dict):
        index.pop(candidate_id, None)
    return candidate
