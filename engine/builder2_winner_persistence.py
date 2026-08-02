"""
Builder2 Winner Development persistence — canonical accepted winner → media handoff.
"""
from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION, Builder2TournamentError
from engine.builder2_winner_development_diagnostics import (
    STAGE_PERSISTENCE,
    log_winner_development_persisted,
    raise_public_winner_failure,
)
from engine.builder2_winner_plan import validate_builder2_winner_plan
from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY

logger = logging.getLogger(__name__)

WINNER_DEVELOPMENT_SOURCE_NORMAL = "normal"
WINNER_DEVELOPMENT_SOURCE_OFFLINE_SALVAGE = "offline_salvage"
WINNER_DEVELOPMENT_SOURCE_OFFLINE_RECOVERY = "offline_recovery"
WINNER_DEVELOPMENT_SOURCE_REVALIDATE = "revalidate"
WINNER_DEVELOPMENT_SOURCE_HEADLINE_REPAIR = "headline_repair"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def derive_winner_prototype_id(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    winning_candidate: Optional[Dict[str, Any]] = None,
    explicit_prototype_id: str = "",
) -> str:
    prototype_id = _clean(explicit_prototype_id)
    if prototype_id:
        return prototype_id
    candidate = winning_candidate if isinstance(winning_candidate, dict) else {}
    prototype_id = _clean(candidate.get("prototypeId"))
    if prototype_id:
        return prototype_id
    winner_rec = (state.get("candidates") or {}).get(candidate_id) or {}
    return _clean(winner_rec.get("prototypeId"))


def compute_winner_development_plan_fingerprint(plan: Dict[str, Any]) -> str:
    from engine.builder2_headline_decision_contract import get_normalized_headline_decision

    payload = {
        "schemaVersion": _clean(plan.get("schemaVersion")),
        "methodologyVersion": _clean(plan.get("methodologyVersion")),
        "prototypeId": _clean(plan.get("prototypeId")),
        "copyContractVersion": _clean(plan.get("copyContractVersion")),
        "headlineDecision": get_normalized_headline_decision(plan),
        "headlineOverlaySkipped": plan.get("headlineOverlaySkipped") is True,
        "headlineCompatibilityAlias": plan.get("headlineCompatibilityAlias") is True,
        "canonicalCopySatisfiedBy": _clean(plan.get("canonicalCopySatisfiedBy")),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_valid_persisted_winner_development(state: Dict[str, Any]) -> bool:
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict) or not plan:
        return False
    if plan.get("schemaVersion") != WINNER_PLAN_SCHEMA_VERSION:
        return False
    if state.get("winnerDevelopmentAccepted") is not True:
        accepted_flag = state.get("winnerDevelopmentAcceptedAt")
        if not accepted_flag:
            meta = state.get("winnerDevelopmentMetadata")
            if not isinstance(meta, dict) or not meta.get("acceptedAt"):
                return False
    accepted_at = state.get("winnerDevelopmentAcceptedAt")
    if not accepted_at:
        meta = state.get("winnerDevelopmentMetadata")
        if isinstance(meta, dict):
            accepted_at = meta.get("acceptedAt")
    if not accepted_at:
        return False
    candidate_id = _clean(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId"))
    if not candidate_id:
        return False
    prototype_id = derive_winner_prototype_id(state, candidate_id=candidate_id)
    if not prototype_id:
        return False
    return True


def is_winner_media_continuation_ready(state: Dict[str, Any]) -> bool:
    return not collect_winner_media_continuation_missing_fields(state)


def collect_winner_media_continuation_missing_fields(state: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not is_valid_persisted_winner_development(state):
        missing.append("winnerDevelopmentPlan")
    if state.get("mediaContinuationRequired") is not True:
        missing.append("mediaContinuationRequired")
    candidate_id = _clean(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId"))
    if not candidate_id:
        missing.append("winnerDevelopmentCandidateId")
    prototype_id = _clean(state.get("winnerDevelopmentPrototypeId"))
    if not prototype_id:
        missing.append("winnerDevelopmentPrototypeId")
    if state.get("winnerDevelopmentAccepted") is not True:
        accepted_at = state.get("winnerDevelopmentAcceptedAt")
        meta = state.get("winnerDevelopmentMetadata")
        if not accepted_at and not (isinstance(meta, dict) and meta.get("acceptedAt")):
            missing.append("winnerDevelopmentAccepted")
    plan = state.get("winnerDevelopmentPlan")
    if isinstance(plan, dict):
        if SERVER_OWNED_WINNER_SOURCE_KEY not in plan:
            missing.append(f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}")
        if prototype_id and _clean(plan.get("prototypeId")) != prototype_id:
            missing.append("winnerDevelopmentPlan.prototypeId")
    winner_id = _clean(state.get("winnerCandidateId"))
    if winner_id and candidate_id and winner_id != candidate_id:
        missing.append("winnerCandidateId")
    return missing


def verify_winner_media_continuation_contract(
    state: Dict[str, Any],
    *,
    job_id: str = "",
    tournament_id: str = "",
) -> None:
    missing = collect_winner_media_continuation_missing_fields(state)
    if missing:
        raise Builder2TournamentError(f"builder2_winner_media_continuation_not_ready:{','.join(missing)}")
    expected_job = _clean(job_id)
    if expected_job and _clean(state.get("jobId")) != expected_job:
        raise Builder2TournamentError("builder2_winner_media_state_mismatch:jobId")
    expected_tournament = _clean(tournament_id)
    if expected_tournament and _clean(state.get("tournamentId")) != expected_tournament:
        raise Builder2TournamentError("builder2_winner_media_state_mismatch:tournamentId")
    stored_fp = _clean(state.get("winnerDevelopmentPlanFingerprint"))
    plan = state.get("winnerDevelopmentPlan")
    if isinstance(plan, dict) and stored_fp:
        current_fp = compute_winner_development_plan_fingerprint(plan)
        if current_fp != stored_fp:
            raise Builder2TournamentError("builder2_winner_media_state_stale:plan_fingerprint")


def reload_verified_winner_media_state(
    job_id: str,
    *,
    tournament_id: str = "",
) -> Dict[str, Any]:
    from engine.builder2_tournament_store import load_tournament_state

    state = load_tournament_state(job_id)
    if state is None:
        raise Builder2TournamentError("builder2_winner_media_state_missing:job")
    verify_winner_media_continuation_contract(
        state,
        job_id=job_id,
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
    )
    return state


def has_failed_winner_attempt_after_paid_call(state: Dict[str, Any]) -> bool:
    if is_valid_persisted_winner_development(state):
        return False
    if state.get("winnerDevelopmentFailureResolved") is True:
        return False
    failure = state.get("winnerDevelopmentFailure")
    if not isinstance(failure, dict):
        return False
    if failure.get("stage") in {None, STAGE_PERSISTENCE}:
        return False
    return bool(state.get("winnerDevelopmentPaidCallRecorded"))


def _resolve_historical_winner_failure(
    state: Dict[str, Any],
    *,
    accepted_at: str,
    resolved_by: str,
) -> None:
    failure = state.get("winnerDevelopmentFailure")
    if not isinstance(failure, dict) or not failure:
        state.pop("winnerDevelopmentFailureResolved", None)
        state.pop("winnerDevelopmentFailureResolvedBy", None)
        state.pop("winnerDevelopmentFailureResolvedAt", None)
        return
    history = state.get("winnerDevelopmentFailureHistory")
    if not isinstance(history, dict):
        state["winnerDevelopmentFailureHistory"] = deepcopy(failure)
    state["winnerDevelopmentFailureResolved"] = True
    state["winnerDevelopmentFailureResolvedBy"] = resolved_by
    state["winnerDevelopmentFailureResolvedAt"] = accepted_at


def _stamp_media_continuation_metadata(
    state: Dict[str, Any],
    *,
    plan: Dict[str, Any],
    candidate_id: str,
    prototype_id: str,
    accepted_at: str,
    source: str,
) -> None:
    from engine.builder2_single_slogan_contract import copy_contract_version, is_single_slogan_contract

    state["winnerDevelopmentPlan"] = plan
    state["winnerDevelopmentCandidateId"] = candidate_id
    state["winnerDevelopmentPrototypeId"] = prototype_id
    state["winnerDevelopmentAcceptedAt"] = accepted_at
    state["winnerDevelopmentAccepted"] = True
    state["winnerDevelopmentSource"] = source
    state["winnerDevelopmentSchemaVersion"] = _clean(plan.get("schemaVersion") or WINNER_PLAN_SCHEMA_VERSION)
    state["winnerDevelopmentMethodologyVersion"] = _clean(plan.get("methodologyVersion") or METHODOLOGY_VERSION)
    state["winnerCandidateId"] = candidate_id
    state["winnerDevelopmentMetadata"] = {
        "acceptedAt": accepted_at,
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "schemaVersion": state["winnerDevelopmentSchemaVersion"],
        "methodologyVersion": state["winnerDevelopmentMethodologyVersion"],
        "source": source,
    }
    state["winnerDevelopmentPlanFingerprint"] = compute_winner_development_plan_fingerprint(plan)
    state["mediaContinuationRequired"] = True
    state["winnerDevelopmentResponseReceived"] = True
    state["winnerDevelopmentParsed"] = True
    contract_version = copy_contract_version(state=state, plan=plan)
    if contract_version:
        state["copyContractVersion"] = contract_version
        if not _clean(plan.get("copyContractVersion")):
            plan["copyContractVersion"] = contract_version
    if is_single_slogan_contract(state=state, plan=plan):
        state["headlineOverlaySkipped"] = plan.get("headlineOverlaySkipped") is True
    dispatch_count = int((state.get("metrics") or {}).get("winnerDevelopmentCalls") or 0)
    if dispatch_count <= 0 and state.get("winnerDevelopmentPaidCallRecorded"):
        dispatch_count = 1
    state["winnerDevelopmentDispatchCount"] = max(0, min(dispatch_count, 1))


def persist_accepted_winner_development_for_media(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    winner_plan: Dict[str, Any],
    prototype_id: str = "",
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    source: str = WINNER_DEVELOPMENT_SOURCE_NORMAL,
    job_id: str = "",
    tournament_id: str = "",
    save: bool = False,
    resolve_historical_failure: bool = True,
) -> Dict[str, Any]:
    resolved_candidate = _clean(candidate_id)
    if not resolved_candidate:
        raise Builder2TournamentError("builder2_winner_media_continuation_not_ready:winnerDevelopmentCandidateId")
    resolved_prototype = derive_winner_prototype_id(
        state,
        candidate_id=resolved_candidate,
        winning_candidate=winning_candidate,
        explicit_prototype_id=prototype_id,
    )
    if not resolved_prototype:
        raise Builder2TournamentError("builder2_winner_media_continuation_not_ready:winnerDevelopmentPrototypeId")

    try:
        validated = validate_builder2_winner_plan(
            winner_plan,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
            tournament_state=state,
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
    if resolve_historical_failure:
        _resolve_historical_winner_failure(state, accepted_at=accepted_at, resolved_by=source)
    _stamp_media_continuation_metadata(
        state,
        plan=plan_copy,
        candidate_id=resolved_candidate,
        prototype_id=resolved_prototype,
        accepted_at=accepted_at,
        source=source,
    )
    log_winner_development_persisted(
        job_id=job_id or _clean(state.get("jobId")),
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
        candidate_id=resolved_candidate,
        prototype_id=resolved_prototype,
    )
    logger.info(
        "BUILDER2_WINNER_MEDIA_HANDOFF_PERSISTED jobId=%s candidateId=%s prototypeId=%s source=%s "
        "mediaContinuationRequired=true planFingerprint=%s",
        job_id or _clean(state.get("jobId")) or "(none)",
        resolved_candidate,
        resolved_prototype,
        source,
        _clean(state.get("winnerDevelopmentPlanFingerprint"))[:16],
    )

    resolved_job_id = _clean(job_id or state.get("jobId"))
    if save and resolved_job_id:
        from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION
        from engine.builder2_tournament_store import save_tournament_state

        state.setdefault("schemaVersion", TOURNAMENT_STATE_SCHEMA_VERSION)
        state.setdefault("jobId", resolved_job_id)
        save_tournament_state(resolved_job_id, state)

    return plan_copy


def persist_and_reload_accepted_winner_for_media(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    winner_plan: Dict[str, Any],
    prototype_id: str = "",
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    source: str = WINNER_DEVELOPMENT_SOURCE_NORMAL,
    job_id: str = "",
    tournament_id: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved_job_id = _clean(job_id or state.get("jobId"))
    if not resolved_job_id:
        raise Builder2TournamentError("builder2_winner_media_state_missing:jobId")
    from engine.builder2_tournament_store import load_tournament_state

    persist_accepted_winner_development_for_media(
        state,
        candidate_id=candidate_id,
        winner_plan=winner_plan,
        prototype_id=prototype_id,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        source=source,
        job_id=resolved_job_id,
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
        save=True,
    )
    reloaded = load_tournament_state(resolved_job_id)
    if reloaded is None:
        reloaded = deepcopy(state)
    verify_winner_media_continuation_contract(
        reloaded,
        job_id=resolved_job_id,
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
    )
    return deepcopy(reloaded.get("winnerDevelopmentPlan") or {}), reloaded


def backfill_winner_media_continuation_contract(state: Dict[str, Any]) -> None:
    if not is_valid_persisted_winner_development(state):
        return
    candidate_id = _clean(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId"))
    prototype_id = derive_winner_prototype_id(state, candidate_id=candidate_id)
    if prototype_id:
        state["winnerDevelopmentPrototypeId"] = prototype_id
    state["winnerDevelopmentAccepted"] = True
    state["mediaContinuationRequired"] = True
    plan = state.get("winnerDevelopmentPlan")
    if isinstance(plan, dict) and not _clean(state.get("winnerDevelopmentPlanFingerprint")):
        state["winnerDevelopmentPlanFingerprint"] = compute_winner_development_plan_fingerprint(plan)


def finalize_accepted_winner_reasoning_handoff(
    state: Dict[str, Any],
    *,
    job_id: str,
    stop_before_media: bool,
) -> None:
    backfill_winner_media_continuation_contract(state)
    verify_winner_media_continuation_contract(state, job_id=job_id)
    state["reasoningComplete"] = True
    state["mediaStarted"] = False
    state["canResume"] = True
    state["failureStage"] = None
    state["failureReason"] = None
    if stop_before_media:
        state["status"] = "paused_for_media_validation"
        state["lastCompletedStep"] = "reasoning_complete"
        state["progressStage"] = "media_prerequisite_validation"
        state["controlledCompleteAdReasoningResume"] = {
            "completedAt": _utc_now_iso(),
            "stoppedBeforeMedia": True,
        }
    else:
        state["status"] = "media_prerequisite_ready"
        state["lastCompletedStep"] = "winner_plan_complete"
        state["progressStage"] = "media_pipeline"
        state.pop("controlledCompleteAdReasoningResume", None)


def persist_winner_development_atomically(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    winner_plan: Dict[str, Any],
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    source: str = WINNER_DEVELOPMENT_SOURCE_NORMAL,
    save: bool = False,
) -> Dict[str, Any]:
    return persist_accepted_winner_development_for_media(
        state,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        winner_plan=winner_plan,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        source=source,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        save=save,
    )
