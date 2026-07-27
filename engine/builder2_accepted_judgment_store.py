"""
Builder2 accepted Judge judgment persistence — immutable snapshots separate from Creator snapshots.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_judge import validate_judge_response
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import JUDGE_SCORE_RANGES, JUDGMENT_SCHEMA_VERSION
from engine.builder2_tournament_metrics import record_judge_valid
from engine.builder2_tournament_store import register_judgment

from engine.builder2_accepted_creator_store import (
    AcceptedCreatorCandidate,
    update_candidate_judge_state,
)

logger = logging.getLogger(__name__)

ACCEPTED_JUDGMENT_INDEX_KEY = "acceptedJudgments"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY)
    if not isinstance(index, dict):
        index = {}
        state[ACCEPTED_JUDGMENT_INDEX_KEY] = index
    return index


def _judgment_record_for_candidate(
    state: Dict[str, Any],
    candidate_id: str,
) -> Optional[Dict[str, Any]]:
    cand = (state.get("candidates") or {}).get(candidate_id)
    if not isinstance(cand, dict):
        return None
    judgment_id = str(cand.get("judgmentId") or "").strip()
    if not judgment_id:
        return None
    record = (state.get("judgments") or {}).get(judgment_id)
    if isinstance(record, dict):
        return record
    snapshot = cand.get("judgmentSnapshot")
    if isinstance(snapshot, dict):
        return {
            "judgmentId": judgment_id,
            "candidateId": candidate_id,
            "judgment": snapshot,
            "totalScore": cand.get("totalScore"),
            "scores": cand.get("tieScores") or {},
            "eligible": cand.get("eligible"),
        }
    return None


def _creator_snapshot_identity_matches(
    *,
    creator_snapshot: AcceptedCreatorCandidate,
    strategy_foundation: Dict[str, Any],
    prototype_id: str,
) -> bool:
    expected_strategy_id = expected_strategy_foundation_id(strategy_foundation)
    snapshot_strategy_id = str(
        creator_snapshot.get("strategyFoundationId")
        or (creator_snapshot.get("creatorOutput") or {}).get("strategyFoundationId")
        or ""
    )
    if expected_strategy_id and snapshot_strategy_id and expected_strategy_id != snapshot_strategy_id:
        return False
    if str(creator_snapshot.get("prototypeId") or "") != prototype_id:
        return False
    methodology = str(creator_snapshot.get("methodologyVersion") or "")
    if methodology and methodology != METHODOLOGY_VERSION:
        return False
    creator_output = creator_snapshot.get("creatorOutput") or {}
    output_methodology = str(creator_output.get("methodologyVersion") or "")
    if output_methodology and output_methodology != METHODOLOGY_VERSION:
        return False
    return True


def _judgment_has_complete_score_set(judgment: Dict[str, Any], scores: Dict[str, Any]) -> bool:
    payload_scores = judgment.get("scores") if isinstance(judgment.get("scores"), dict) else scores
    if not isinstance(payload_scores, dict):
        return False
    for name in JUDGE_SCORE_RANGES:
        if name not in payload_scores and name not in scores:
            return False
    return True


def audit_reusable_accepted_judgment(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    creator_snapshot: AcceptedCreatorCandidate,
    strategy_foundation: Dict[str, Any],
    compatibility_mode: bool = False,
) -> Tuple[bool, Optional[str]]:
    cand = (state.get("candidates") or {}).get(candidate_id)
    if not isinstance(cand, dict):
        return False, "candidate_record_missing"
    if cand.get("judgeStatus") != "accepted":
        return False, "judgeStatus_not_accepted"
    if cand.get("validationStatus") == "judge_unavailable":
        return False, "judge_unavailable"
    if cand.get("status") == "judge_unavailable":
        return False, "judge_unavailable"

    prototype_id = str(creator_snapshot.get("prototypeId") or cand.get("prototypeId") or "")
    if str(cand.get("prototypeId") or "") != prototype_id:
        return False, "prototypeId_mismatch"

    if not _creator_snapshot_identity_matches(
        creator_snapshot=creator_snapshot,
        strategy_foundation=strategy_foundation,
        prototype_id=prototype_id,
    ):
        return False, "creator_snapshot_identity_mismatch"

    judgment_record = _judgment_record_for_candidate(state, candidate_id)
    if judgment_record is None:
        return False, "judgment_record_missing"

    judgment = judgment_record.get("judgment")
    if not isinstance(judgment, dict) or not judgment:
        return False, "judgment_payload_missing"

    if str(judgment.get("candidateId") or "") != candidate_id:
        return False, "judgment_candidateId_mismatch"

    if judgment.get("schemaVersion") != JUDGMENT_SCHEMA_VERSION:
        return False, "judgment_schemaVersion_mismatch"

    methodology = str(judgment.get("methodologyVersion") or "")
    if methodology and methodology != METHODOLOGY_VERSION and not compatibility_mode:
        return False, "judgment_methodologyVersion_mismatch"

    scores = judgment_record.get("scores")
    if not isinstance(scores, dict):
        scores = cand.get("tieScores") if isinstance(cand.get("tieScores"), dict) else {}
    if not _judgment_has_complete_score_set(judgment, scores):
        return False, "judgment_scores_incomplete"

    creator_output = creator_snapshot.get("creatorOutput") or {}
    try:
        validate_judge_response(
            judgment,
            candidate_id=candidate_id,
            candidate=creator_output,
            compatibility_mode=compatibility_mode,
        )
    except Exception:
        return False, "judgment_contract_invalid"

    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY) or {}
    indexed = index.get(candidate_id)
    if isinstance(indexed, dict):
        indexed_judgment = indexed.get("judgment")
        if isinstance(indexed_judgment, dict) and indexed_judgment.get("schemaVersion") != JUDGMENT_SCHEMA_VERSION:
            return False, "acceptedJudgment_index_schema_invalid"

    return True, None


def backfill_accepted_judgment_index(state: Dict[str, Any], *, persist: bool = True) -> int:
    from engine.builder2_read_only_inspection import read_only_inspection_active

    derived_entries = _derive_missing_judgment_index_entries(state)
    if not derived_entries:
        return 0
    if not persist or read_only_inspection_active():
        logger.info(
            "BUILDER2_ACCEPTED_JUDGMENT_INDEX_DERIVED_READ_ONLY jobId=%s tournamentId=%s count=%s",
            state.get("jobId"),
            state.get("tournamentId"),
            len(derived_entries),
        )
        return len(derived_entries)
    index = _ensure_index(state)
    added = 0
    for candidate_id, entry in derived_entries.items():
        index[candidate_id] = entry
        added += 1
    if added:
        logger.info(
            "BUILDER2_ACCEPTED_JUDGMENT_INDEX_BACKFILLED jobId=%s tournamentId=%s count=%s",
            state.get("jobId"),
            state.get("tournamentId"),
            added,
        )
    return added


def derive_accepted_judgment_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY)
    merged: Dict[str, Dict[str, Any]] = deepcopy(index) if isinstance(index, dict) else {}
    merged.update(_derive_missing_judgment_index_entries(state))
    return merged


def _derive_missing_judgment_index_entries(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY)
    existing = index if isinstance(index, dict) else {}
    derived: Dict[str, Dict[str, Any]] = {}
    for candidate_id, cand in (state.get("candidates") or {}).items():
        if not isinstance(cand, dict) or candidate_id in existing:
            continue
        reusable, _reason = audit_reusable_accepted_judgment(
            state,
            candidate_id=str(candidate_id),
            creator_snapshot={
                "candidateId": str(candidate_id),
                "prototypeId": str(cand.get("prototypeId") or ""),
                "creatorOutput": deepcopy(cand.get("creatorSnapshot") or cand.get("creatorOutput") or {}),
                "strategyFoundationId": str(
                    (cand.get("creatorSnapshot") or cand.get("creatorOutput") or {}).get("strategyFoundationId") or ""
                ),
                "methodologyVersion": str(
                    (cand.get("creatorSnapshot") or cand.get("creatorOutput") or {}).get("methodologyVersion")
                    or METHODOLOGY_VERSION
                ),
                "validationStatus": "accepted",
            },
            strategy_foundation=state.get("strategyFoundation") or {},
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
        )
        if not reusable:
            continue
        record = _judgment_record_for_candidate(state, str(candidate_id))
        if record is None:
            continue
        derived[str(candidate_id)] = {
            "judgmentId": record.get("judgmentId"),
            "candidateId": str(candidate_id),
            "prototypeId": str(cand.get("prototypeId") or ""),
            "judgment": deepcopy(record.get("judgment")),
            "totalScore": record.get("totalScore"),
            "scores": deepcopy(record.get("scores") or cand.get("tieScores") or {}),
            "eligible": record.get("eligible"),
            "schemaVersion": JUDGMENT_SCHEMA_VERSION,
            "methodologyVersion": str((record.get("judgment") or {}).get("methodologyVersion") or METHODOLOGY_VERSION),
            "acceptedAt": str(cand.get("completedAt") or _utc_now_iso()),
        }
    return derived


def persist_accepted_judgment(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    judgment_id: str,
    judgment: Dict[str, Any],
    total: int,
    scores: Dict[str, int],
) -> None:
    completed_at = _utc_now_iso()
    register_judgment(
        state,
        {
            "judgmentId": judgment_id,
            "candidateId": candidate_id,
            "judgment": deepcopy(judgment),
            "totalScore": total,
            "scores": deepcopy(scores),
            "eligible": judgment.get("eligible"),
            "completedAt": completed_at,
        },
    )
    update_candidate_judge_state(
        state,
        candidate_id=candidate_id,
        judge_status="accepted",
        judgment_id=judgment_id,
        judgment_snapshot=judgment,
    )
    cand = state.setdefault("candidates", {}).setdefault(candidate_id, {})
    cand["prototypeId"] = prototype_id
    cand["eligible"] = bool(judgment.get("eligible"))
    cand["totalScore"] = total
    cand["tieScores"] = deepcopy(scores)
    cand["completedAt"] = completed_at

    index = _ensure_index(state)
    index[candidate_id] = {
        "judgmentId": judgment_id,
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "judgment": deepcopy(judgment),
        "totalScore": total,
        "scores": deepcopy(scores),
        "eligible": judgment.get("eligible"),
        "schemaVersion": JUDGMENT_SCHEMA_VERSION,
        "methodologyVersion": str(judgment.get("methodologyVersion") or METHODOLOGY_VERSION),
        "acceptedAt": completed_at,
    }
    record_judge_valid(state, eligible=bool(judgment.get("eligible")))
    logger.info(
        "BUILDER2_ACCEPTED_JUDGMENT_PERSISTED jobId=%s tournamentId=%s candidateId=%s judgmentId=%s eligible=%s",
        state.get("jobId"),
        state.get("tournamentId"),
        candidate_id,
        judgment_id,
        judgment.get("eligible"),
    )


def list_accepted_judgment_candidate_ids(state: Dict[str, Any]) -> List[str]:
    backfill_accepted_judgment_index(state)
    index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY) or {}
    return sorted(str(key) for key in index.keys())
