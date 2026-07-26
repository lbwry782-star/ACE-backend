"""
Builder2 Winner Development persistence — atomic accepted winner plan storage.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION
from engine.builder2_winner_development_diagnostics import (
    log_winner_development_persisted,
    STAGE_PERSISTENCE,
    raise_public_winner_failure,
)
from engine.builder2_winner_plan import validate_builder2_winner_plan

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_valid_persisted_winner_development(state: Dict[str, Any]) -> bool:
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict) or not plan:
        return False
    if plan.get("schemaVersion") != WINNER_PLAN_SCHEMA_VERSION:
        return False
    accepted_at = state.get("winnerDevelopmentAcceptedAt")
    if not accepted_at:
        meta = state.get("winnerDevelopmentMetadata")
        if isinstance(meta, dict):
            accepted_at = meta.get("acceptedAt")
    if not accepted_at:
        return False
    candidate_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or "").strip()
    if not candidate_id:
        return False
    return True


def has_failed_winner_attempt_after_paid_call(state: Dict[str, Any]) -> bool:
    failure = state.get("winnerDevelopmentFailure")
    if not isinstance(failure, dict):
        return False
    if failure.get("stage") in {None, STAGE_PERSISTENCE}:
        return False
    return bool(state.get("winnerDevelopmentPaidCallRecorded")) and not is_valid_persisted_winner_development(state)


def persist_winner_development_atomically(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    winner_plan: Dict[str, Any],
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    try:
        validated = validate_builder2_winner_plan(
            winner_plan,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
        )
    except Exception as exc:
        raise_public_winner_failure(
            exc,
            state=state,
            stage=STAGE_PERSISTENCE,
            top_level_keys=sorted(winner_plan.keys()) if isinstance(winner_plan, dict) else [],
        )

    accepted_at = _utc_now_iso()
    plan_copy = deepcopy(validated)
    state["winnerDevelopmentPlan"] = plan_copy
    state["winnerDevelopmentCandidateId"] = candidate_id
    state["winnerDevelopmentPrototypeId"] = prototype_id
    state["winnerDevelopmentAcceptedAt"] = accepted_at
    state["winnerDevelopmentSchemaVersion"] = str(plan_copy.get("schemaVersion") or WINNER_PLAN_SCHEMA_VERSION)
    state["winnerDevelopmentMethodologyVersion"] = str(
        plan_copy.get("methodologyVersion") or METHODOLOGY_VERSION
    )
    state["winnerCandidateId"] = candidate_id
    state["winnerDevelopmentMetadata"] = {
        "acceptedAt": accepted_at,
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "schemaVersion": state["winnerDevelopmentSchemaVersion"],
        "methodologyVersion": state["winnerDevelopmentMethodologyVersion"],
    }
    state.pop("winnerDevelopmentFailure", None)
    state["winnerDevelopmentAccepted"] = True
    log_winner_development_persisted(
        job_id=str(state.get("jobId") or ""),
        tournament_id=str(state.get("tournamentId") or ""),
        candidate_id=candidate_id,
        prototype_id=prototype_id,
    )
    return plan_copy
