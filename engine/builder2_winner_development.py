"""
Builder2 winner development — convert winning candidate to Runway-compatible plan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import resolve_builder2_winner_model
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_llm import call_builder2_role_json_with_text, parse_json_object
from engine.builder2_tournament_metrics import MetricsTimer, record_winner_call_elapsed, record_winner_paid_call_submitted
from engine.builder2_tournament_prompts import build_winner_development_prompt
from engine.builder2_winner_development_diagnostics import (
    STAGE_EXTRACTION,
    STAGE_METHODOLOGY_VALIDATION,
    STAGE_VALIDATION,
    clear_winner_failure_diagnostics,
    log_winner_development_extraction_ok,
    log_winner_development_response_received,
    log_winner_development_validation_ok,
    raise_public_winner_failure,
    safe_top_level_keys,
)
from engine.builder2_winner_plan import validate_and_normalize_builder2_winner_plan, validate_builder2_winner_plan
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    persist_parsed_winner_response,
    process_winner_development_response,
)

logger = logging.getLogger(__name__)


def validate_winner_plan(
    raw: Dict[str, Any],
    *,
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    source_reference: Optional[Dict[str, Any]] = None,
    job_id: str = "",
    tournament_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if source_reference is not None and winning_candidate is not None:
        return process_winner_development_response(
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
    return validate_builder2_winner_plan(
        raw,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
    )


def normalize_winner_plan_for_runway(
    winner_plan: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Dict[str, Any]:
    return validate_and_normalize_builder2_winner_plan(
        winner_plan,
        product_name=product_name,
        product_description=product_description,
        content_language=content_language,
    )


def develop_builder2_winning_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    prototype_id: str,
    runway_mode: str,
    llm_client: Optional[Any] = None,
    compatibility_mode: bool = False,
    state: Optional[Dict[str, Any]] = None,
    candidate_id: Optional[str] = None,
) -> Dict[str, Any]:
    prototype = require_prototype(prototype_id)
    resolved_candidate_id = str(
        candidate_id
        or (state or {}).get("winnerCandidateId")
        or winning_candidate.get("candidateId")
        or ""
    ).strip()
    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=resolved_candidate_id,
    )
    preservation_snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=resolved_candidate_id,
    )
    job_id = str((state or {}).get("jobId") or "")
    tournament_id = str((state or {}).get("tournamentId") or "")
    if state is not None:
        state["currentWinnerPrototypeId"] = prototype_id
        state["serverOwnedWinnerSource"] = source_reference

    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_START jobId=%s tournamentId=%s prototypeId=%s candidateId=%s",
        job_id,
        tournament_id,
        prototype_id,
        resolved_candidate_id,
    )
    prompt = build_winner_development_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        winning_judgment=winning_judgment,
        prototype=prototype,
        runway_mode=runway_mode,
        preservation_snapshot=preservation_snapshot,
        state=state,
    )

    timer = MetricsTimer()
    response_text = ""
    top_level_keys: list[str] = []
    raw: Dict[str, Any] = {}

    def _on_paid_request_submitted() -> None:
        if state is not None:
            record_winner_paid_call_submitted(state)
            state["winnerDevelopmentPaidCallRecorded"] = True

    try:
        from engine.builder2_job_cancellation import checkpoint_builder2_cancellation

        checkpoint_builder2_cancellation(job_id, stage="winner_development")
        raw, response_text = call_builder2_role_json_with_text(
            role="builder2_winner",
            model=resolve_builder2_winner_model(),
            prompt=prompt,
            llm_client=llm_client,
            on_paid_request_submitted=_on_paid_request_submitted,
        )
    except ValueError as exc:
        if state is not None:
            record_winner_call_elapsed(state, timer.elapsed_ms())
        raise_public_winner_failure(
            exc,
            state=state,
            stage=STAGE_EXTRACTION,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
        )
    except Exception as exc:
        if state is not None:
            record_winner_call_elapsed(state, timer.elapsed_ms())
        raise_public_winner_failure(
            exc,
            state=state,
            stage=STAGE_EXTRACTION,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
        )

    if state is not None:
        record_winner_call_elapsed(state, timer.elapsed_ms())

    log_winner_development_response_received(
        job_id=job_id,
        tournament_id=tournament_id,
        prototype_id=prototype_id,
        response_char_count=len(response_text),
    )

    if not isinstance(raw, dict):
        raise_public_winner_failure(
            ValueError("json_not_object"),
            state=state,
            stage=STAGE_EXTRACTION,
            response_char_count=len(response_text),
            top_level_keys=[],
        )

    top_level_keys = safe_top_level_keys(raw)
    log_winner_development_extraction_ok(
        job_id=job_id,
        tournament_id=tournament_id,
        prototype_id=prototype_id,
        top_level_keys=top_level_keys,
    )

    if state is not None:
        persist_parsed_winner_response(
            state,
            parsed=raw,
            candidate_id=resolved_candidate_id,
            prototype_id=prototype_id,
            top_level_keys=top_level_keys,
            response_char_count=len(response_text),
            response_text=response_text,
        )

    try:
        winner_plan = process_winner_development_response(
            raw,
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
            job_id=job_id,
            tournament_id=tournament_id,
            tournament_state=state,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "")
        stage = STAGE_METHODOLOGY_VALIDATION if reason.startswith(
            ("builder2_winner_validation_failed", "builder2_winner_source_identity_mismatch", "builder2_winner_preservation_contract_missing")
        ) else STAGE_VALIDATION
        if state is not None:
            from engine.builder2_winner_response_ledger import record_winner_validation_outcome

            record_winner_validation_outcome(
                state,
                candidate_id=resolved_candidate_id,
                accepted=False,
                failure_stage=stage,
                failure_field_path=str(exc.args[0]).split(":", 1)[-1] if ":" in reason else None,
                failure_reason=reason,
                exception_class=exc.__class__.__name__,
            )
        raise_public_winner_failure(
            exc,
            state=state,
            stage=stage,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
        )
    except Exception as exc:
        field_path = None
        stage = STAGE_VALIDATION
        if state is not None:
            from engine.builder2_winner_validation_replay import infer_typeerror_failure
            from engine.builder2_winner_response_ledger import record_winner_validation_outcome

            inferred_field, _, _ = infer_typeerror_failure(exc, plan=raw)
            field_path = inferred_field
            record_winner_validation_outcome(
                state,
                candidate_id=resolved_candidate_id,
                accepted=False,
                failure_stage=stage,
                failure_field_path=field_path,
                failure_reason=str(exc),
                exception_class=exc.__class__.__name__,
            )
        raise_public_winner_failure(
            exc,
            state=state,
            stage=stage,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
            failure_field_path=field_path,
        )

    log_winner_development_validation_ok(
        job_id=job_id,
        tournament_id=tournament_id,
        prototype_id=prototype_id,
    )
    winner_plan["winningCandidatePreservationSnapshot"] = preservation_snapshot
    if state is not None:
        clear_winner_failure_diagnostics(state)
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_OK jobId=%s tournamentId=%s prototypeId=%s candidateId=%s",
        job_id,
        tournament_id,
        prototype_id,
        resolved_candidate_id,
    )
    return winner_plan


def parse_winner_response_text(response_text: str) -> Dict[str, Any]:
    return parse_json_object(response_text)
