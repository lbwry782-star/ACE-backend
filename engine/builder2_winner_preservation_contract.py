"""
Builder2 Winner preservation contract — server-owned source reference and carry-forward.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_winner_plan import validate_builder2_winner_plan

logger = logging.getLogger(__name__)

SERVER_OWNED_WINNER_SOURCE_KEY = "serverOwnedWinnerSource"
SERVER_PRESERVATION_CHECK_KEY = "serverPreservationCheck"
MODEL_PRESERVATION_DIAGNOSTIC_KEY = "winnerPreservationCheckModelDiagnostic"
PARSED_WINNER_RESPONSE_KEY = "winnerDevelopmentParsedResponse"

PRESERVATION_BOOLEAN_FIELDS: tuple[str, ...] = (
    "problemPreserved",
    "relativeAdvantagePreserved",
    "mechanismPreserved",
    "prototypeMethodPreserved",
    "visualParallelPreserved",
    "structurePreserved",
    "editingOnlyStrengthens",
)


def _strategy_statement(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("statement") or "").strip()
    return str(block or "").strip()


def build_server_owned_winner_source_reference(
    *,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    candidate_id: str,
) -> Dict[str, Any]:
    silent = winning_candidate.get("silentVerification")
    if not isinstance(silent, dict):
        silent = {}
    prototype_method = winning_candidate.get("prototypeMethodApplication")
    if not isinstance(prototype_method, dict):
        prototype_method = winning_candidate.get("prototypeMethodContract")
    if not isinstance(prototype_method, dict):
        prototype_method = {}

    return {
        "sourceCandidateId": str(candidate_id or "").strip(),
        "sourcePrototypeId": str(winning_candidate.get("prototypeId") or "").strip(),
        "strategyFoundationId": expected_strategy_foundation_id(strategy_foundation),
        "methodologyVersion": str(winning_candidate.get("methodologyVersion") or METHODOLOGY_VERSION),
        "structureType": str(winning_candidate.get("structureType") or "").strip(),
        "visualParallelType": str(winning_candidate.get("visualParallelType") or "").strip(),
        "problemPerception": deepcopy(strategy_foundation.get("problemPerception") or {}),
        "relativeAdvantage": deepcopy(strategy_foundation.get("relativeAdvantage") or {}),
        "prototypeMethodContract": deepcopy(prototype_method),
        "coreCreativeMechanism": str(winning_candidate.get("coreCreativeMechanism") or "").strip(),
        "coreVisualIdea": str(
            winning_candidate.get("coreVisualIdea")
            or winning_candidate.get("conceptSummary")
            or ""
        ).strip(),
        "visualAnchor": deepcopy(winning_candidate.get("visualAnchor") or {}),
        "participationMechanism": deepcopy(winning_candidate.get("participationMechanism") or {}),
        "silentMovieContract": deepcopy(silent),
        "visualFamily": str(winning_candidate.get("visualFamily") or "").strip(),
    }


def build_winning_candidate_preservation_snapshot(
    *,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    candidate_id: str = "",
) -> Dict[str, Any]:
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    return {
        "strategyFoundationId": source["strategyFoundationId"],
        "prototypeId": source["sourcePrototypeId"],
        "structureType": source["structureType"],
        "visualParallelType": source["visualParallelType"],
        "coreCreativeMechanism": source["coreCreativeMechanism"],
        "visualMechanism": winning_candidate.get("visualMechanism"),
        "visualFamilyDefinition": (
            (winning_candidate.get("visualFamilyConsistency") or {}).get("familyDefinition")
            or source.get("visualFamily")
        ),
        "mainSubject": (winning_candidate.get("runwayFeasibility") or {}).get("mainSubject"),
        "mainAction": (winning_candidate.get("runwayFeasibility") or {}).get("mainAction"),
        "location": (winning_candidate.get("runwayFeasibility") or {}).get("location"),
        "sourceCandidateId": source["sourceCandidateId"],
    }


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def normalize_winner_response_compatibility_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(raw)
    model_check = out.get("winnerPreservationCheck")
    if model_check is not None and not isinstance(model_check, dict):
        out.pop("winnerPreservationCheck", None)
    elif isinstance(model_check, dict):
        normalized_check = dict(model_check)
        for key in PRESERVATION_BOOLEAN_FIELDS:
            if key in normalized_check:
                coerced = _coerce_optional_bool(normalized_check.get(key))
                normalized_check[key] = coerced
        out["winnerPreservationCheck"] = normalized_check
    return out


def capture_model_preservation_diagnostic(raw: Dict[str, Any]) -> Dict[str, Any]:
    model_check = raw.get("winnerPreservationCheck")
    diagnostic: Dict[str, Any] = {
        "fieldExisted": isinstance(model_check, dict),
        "fieldType": type(model_check).__name__ if model_check is not None else "missing",
    }
    if isinstance(model_check, dict):
        problem = model_check.get("problemPreserved")
        diagnostic["problemPreservedExisted"] = "problemPreserved" in model_check
        diagnostic["problemPreservedType"] = type(problem).__name__
        diagnostic["problemPreservedValue"] = problem if isinstance(problem, (bool, type(None))) else "non_boolean"
    else:
        diagnostic["problemPreservedExisted"] = False
        diagnostic["problemPreservedType"] = "missing"
        diagnostic["problemPreservedValue"] = None
    return diagnostic


def derive_server_preservation_check() -> Dict[str, Any]:
    return {
        "problemPreserved": True,
        "relativeAdvantagePreserved": True,
        "prototypeMethodPreserved": True,
        "coreMechanismPreserved": True,
        "source": "server_owned_contract",
    }


def validate_winner_preservation_contract_required(source_reference: Dict[str, Any]) -> None:
    if not source_reference.get("sourceCandidateId"):
        raise Builder2TournamentError("builder2_winner_preservation_contract_missing:sourceCandidateId")
    if not source_reference.get("sourcePrototypeId"):
        raise Builder2TournamentError("builder2_winner_preservation_contract_missing:sourcePrototypeId")
    if not source_reference.get("strategyFoundationId"):
        raise Builder2TournamentError("builder2_winner_preservation_contract_missing:strategyFoundationId")


def detect_winner_immutable_identity_violations(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
) -> None:
    validate_winner_preservation_contract_required(source_reference)
    expected_proto = str(source_reference.get("sourcePrototypeId") or "")
    expected_strategy = str(source_reference.get("strategyFoundationId") or "")
    expected_methodology = str(source_reference.get("methodologyVersion") or "")

    ref = raw.get("preservationReference")
    if isinstance(ref, dict):
        ref_proto = str(ref.get("prototypeId") or "").strip()
        if ref_proto and ref_proto != expected_proto:
            raise Builder2TournamentError("builder2_winner_source_identity_mismatch:preservationReference.prototypeId")
        ref_strategy = str(ref.get("strategyFoundationId") or "").strip()
        if ref_strategy and expected_strategy and ref_strategy != expected_strategy:
            raise Builder2TournamentError("builder2_winner_source_identity_mismatch:preservationReference.strategyFoundationId")

    model_methodology = str(raw.get("methodologyVersion") or "").strip()
    if model_methodology and expected_methodology and model_methodology != expected_methodology:
        raise Builder2TournamentError("builder2_winner_source_identity_mismatch:methodologyVersion")

    owned = raw.get(SERVER_OWNED_WINNER_SOURCE_KEY)
    if isinstance(owned, dict) and owned:
        owned_proto = str(owned.get("sourcePrototypeId") or "").strip()
        if owned_proto and owned_proto != expected_proto:
            raise Builder2TournamentError("builder2_winner_source_identity_mismatch:serverOwnedWinnerSource.sourcePrototypeId")
        owned_candidate = str(owned.get("sourceCandidateId") or "").strip()
        expected_candidate = str(source_reference.get("sourceCandidateId") or "").strip()
        if owned_candidate and expected_candidate and owned_candidate != expected_candidate:
            raise Builder2TournamentError("builder2_winner_source_identity_mismatch:serverOwnedWinnerSource.sourceCandidateId")


def validate_winner_source_identity(
    merged: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
) -> None:
    validate_winner_preservation_contract_required(source_reference)
    owned = merged.get(SERVER_OWNED_WINNER_SOURCE_KEY)
    if not isinstance(owned, dict) or not owned:
        raise Builder2TournamentError("builder2_winner_preservation_contract_missing:serverOwnedWinnerSource")

    expected_proto = str(source_reference.get("sourcePrototypeId") or "")
    expected_strategy = str(source_reference.get("strategyFoundationId") or "")
    expected_candidate = str(source_reference.get("sourceCandidateId") or "")

    if str(owned.get("sourcePrototypeId") or "") != expected_proto:
        raise Builder2TournamentError("builder2_winner_source_identity_mismatch:sourcePrototypeId")
    if str(owned.get("strategyFoundationId") or "") != expected_strategy:
        raise Builder2TournamentError("builder2_winner_source_identity_mismatch:strategyFoundationId")
    if str(owned.get("sourceCandidateId") or "") != expected_candidate:
        raise Builder2TournamentError("builder2_winner_source_identity_mismatch:sourceCandidateId")

    if str(merged.get("prototypeId") or "") != expected_proto:
        raise Builder2TournamentError("builder2_winner_immutable_field_override:prototypeId")
    ref = merged.get("preservationReference")
    if isinstance(ref, dict):
        if str(ref.get("prototypeId") or "") != expected_proto:
            raise Builder2TournamentError("builder2_winner_immutable_field_override:preservationReference.prototypeId")
        if str(ref.get("strategyFoundationId") or "") != expected_strategy:
            raise Builder2TournamentError("builder2_winner_immutable_field_override:preservationReference.strategyFoundationId")
        if str(ref.get("sourceCandidateId") or "") != expected_candidate:
            raise Builder2TournamentError("builder2_winner_immutable_field_override:preservationReference.sourceCandidateId")


def apply_server_owned_preservation(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(raw)
    owned = deepcopy(source_reference)
    out[SERVER_OWNED_WINNER_SOURCE_KEY] = owned

    out["prototypeId"] = owned["sourcePrototypeId"]
    out["structureType"] = owned.get("structureType") or out.get("structureType")
    out["visualParallelType"] = owned.get("visualParallelType") or out.get("visualParallelType")
    out["coreCreativeMechanism"] = owned["coreCreativeMechanism"]
    out["coreVisualIdea"] = owned["coreVisualIdea"] or out.get("coreVisualIdea")
    out["visualFamily"] = owned.get("visualFamily") or out.get("visualFamily")
    out["visualAnchor"] = deepcopy(owned.get("visualAnchor") or out.get("visualAnchor"))
    out["problemPerception"] = _strategy_statement(owned.get("problemPerception")) or out.get("problemPerception")
    out["relativeAdvantage"] = _strategy_statement(owned.get("relativeAdvantage")) or out.get("relativeAdvantage")

    out["preservationReference"] = {
        "strategyFoundationId": owned["strategyFoundationId"],
        "prototypeId": owned["sourcePrototypeId"],
        "structureType": owned.get("structureType"),
        "visualParallelType": owned.get("visualParallelType"),
        "coreCreativeMechanism": owned["coreCreativeMechanism"],
        "sourceCandidateId": owned["sourceCandidateId"],
    }
    model_diagnostic = capture_model_preservation_diagnostic(raw)
    if model_diagnostic.get("fieldExisted"):
        out[MODEL_PRESERVATION_DIAGNOSTIC_KEY] = model_diagnostic
    out[SERVER_PRESERVATION_CHECK_KEY] = derive_server_preservation_check()
    out.pop("winnerPreservationCheck", None)
    return out


def log_preservation_contract_applied(
    *,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    prototype_id: str,
    model_diagnostic: Dict[str, Any],
) -> None:
    logger.info(
        "BUILDER2_WINNER_PRESERVATION_CONTRACT_APPLIED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
        "modelFieldExisted=%s modelProblemPreservedType=%s modelProblemPreservedValue=%s "
        "serverPreservationSource=server_owned_contract",
        job_id,
        tournament_id,
        candidate_id,
        prototype_id,
        model_diagnostic.get("fieldExisted"),
        model_diagnostic.get("problemPreservedType"),
        model_diagnostic.get("problemPreservedValue"),
    )


def log_preservation_failure_diagnostics(
    *,
    job_id: str,
    tournament_id: str,
    model_diagnostic: Dict[str, Any],
    failure_field: Optional[str],
) -> None:
    logger.error(
        "BUILDER2_WINNER_PRESERVATION_FAILURE jobId=%s tournamentId=%s modelFieldExisted=%s "
        "modelProblemPreservedType=%s modelProblemPreservedValue=%s authoritativeSource=server_owned_contract "
        "failureField=%s",
        job_id,
        tournament_id,
        model_diagnostic.get("fieldExisted"),
        model_diagnostic.get("problemPreservedType"),
        model_diagnostic.get("problemPreservedValue"),
        failure_field,
    )


def process_winner_development_response(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return prepare_and_validate_persisted_winner_offline(
        raw,
        source_reference=source_reference,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        job_id=job_id,
        tournament_id=tournament_id,
        tournament_state=tournament_state,
    )


def prepare_and_validate_persisted_winner_offline(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from engine.builder2_single_slogan_contract import apply_persisted_winner_copy_contract_normalization
    from engine.builder2_winner_plan import validate_builder2_winner_plan

    working = deepcopy(raw)
    normalized = normalize_winner_response_compatibility_fields(working)
    model_diagnostic = capture_model_preservation_diagnostic(normalized)
    detect_winner_immutable_identity_violations(normalized, source_reference=source_reference)
    merged = apply_server_owned_preservation(normalized, source_reference=source_reference)
    if not compatibility_mode:
        apply_persisted_winner_copy_contract_normalization(
            merged,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            tournament_state=tournament_state,
        )
    validate_winner_source_identity(merged, source_reference=source_reference)
    log_preservation_contract_applied(
        job_id=job_id,
        tournament_id=tournament_id,
        candidate_id=str(source_reference.get("sourceCandidateId") or ""),
        prototype_id=str(source_reference.get("sourcePrototypeId") or ""),
        model_diagnostic=model_diagnostic,
    )
    snapshot = preservation_snapshot or build_winning_candidate_preservation_snapshot(
        strategy_foundation={"strategyFoundationId": source_reference.get("strategyFoundationId")},
        winning_candidate=winning_candidate,
        candidate_id=str(source_reference.get("sourceCandidateId") or ""),
    )
    try:
        from engine.builder2_complete_ad_contract import validate_winner_slogan_preservation

        if not compatibility_mode:
            validate_winner_slogan_preservation(merged, winning_candidate=winning_candidate)
        from engine.builder2_winner_scene_variations_normalization import (
            normalize_continuous_event_scene_variations_for_execution,
        )

        normalize_continuous_event_scene_variations_for_execution(
            merged,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=str(source_reference.get("sourceCandidateId") or ""),
            prototype_id=str(source_reference.get("sourcePrototypeId") or ""),
        )
        return validate_builder2_winner_plan(
            merged,
            winning_candidate=winning_candidate,
            preservation_snapshot=snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
            tournament_state=tournament_state,
        )
    except Builder2TournamentError as exc:
        field = str(exc.args[0]).split(":", 1)[-1] if exc.args else None
        log_preservation_failure_diagnostics(
            job_id=job_id,
            tournament_id=tournament_id,
            model_diagnostic=model_diagnostic,
            failure_field=field,
        )
        raise


def persist_parsed_winner_response(
    state: Dict[str, Any],
    *,
    parsed: Dict[str, Any],
    candidate_id: str,
    prototype_id: str,
    top_level_keys: Optional[List[str]] = None,
    response_char_count: int = 0,
) -> None:
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": deepcopy(parsed),
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "topLevelKeys": list(top_level_keys or sorted(parsed.keys())),
        "topLevelKeyCount": len(parsed),
        "responseCharCount": response_char_count,
    }


def load_revalidatable_parsed_winner_response(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = state.get(PARSED_WINNER_RESPONSE_KEY)
    if not isinstance(payload, dict):
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict) or not parsed:
        return None
    return deepcopy(payload)


def offline_revalidate_parsed_winner_response(
    state: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
) -> Dict[str, Any]:
    payload = load_revalidatable_parsed_winner_response(state)
    if payload is None:
        raise Builder2TournamentError("builder2_winner_offline_revalidation_missing_parsed_response")
    expected_candidate = str(source_reference.get("sourceCandidateId") or "").strip()
    if expected_candidate and str(payload.get("candidateId") or "").strip() != expected_candidate:
        raise Builder2TournamentError("builder2_winner_offline_revalidation_candidate_mismatch")
    parsed = dict(payload.get("parsed") or {})
    return process_winner_development_response(
        parsed,
        source_reference=source_reference,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        job_id=job_id,
        tournament_id=tournament_id,
        tournament_state=state,
    )
