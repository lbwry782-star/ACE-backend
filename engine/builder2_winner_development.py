"""
Builder2 winner development — convert winning candidate to Runway-compatible plan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.builder2_methodology_validation import (
    build_winning_candidate_preservation_snapshot,
)
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import resolve_builder2_winner_model
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_llm import call_builder2_role_json_with_text, parse_json_object
from engine.builder2_tournament_metrics import MetricsTimer, record_winner_call_elapsed, record_winner_paid_call_submitted
from engine.builder2_tournament_prompts import build_winner_development_prompt
from engine.builder2_winner_development_diagnostics import (
    PUBLIC_FAILURE_CODE,
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

logger = logging.getLogger(__name__)


def validate_winner_plan(
    raw: Dict[str, Any],
    *,
    winning_candidate: Optional[Dict[str, Any]] = None,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    return validate_builder2_winner_plan(
        raw,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
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
) -> Dict[str, Any]:
    prototype = require_prototype(prototype_id)
    preservation_snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
    )
    job_id = str((state or {}).get("jobId") or "")
    tournament_id = str((state or {}).get("tournamentId") or "")
    if state is not None:
        state["currentWinnerPrototypeId"] = prototype_id

    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_START jobId=%s tournamentId=%s prototypeId=%s",
        job_id,
        tournament_id,
        prototype_id,
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
    )

    timer = MetricsTimer()
    response_text = ""
    top_level_keys: list[str] = []

    def _on_paid_request_submitted() -> None:
        if state is not None:
            record_winner_paid_call_submitted(state)
            state["winnerDevelopmentPaidCallRecorded"] = True

    try:
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

    try:
        winner_plan = validate_winner_plan(
            raw,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
        )
    except Builder2TournamentError as exc:
        stage = STAGE_METHODOLOGY_VALIDATION if str(exc.args[0] if exc.args else "").startswith(
            "builder2_winner_validation_failed"
        ) else STAGE_VALIDATION
        raise_public_winner_failure(
            exc,
            state=state,
            stage=stage,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
        )
    except Exception as exc:
        raise_public_winner_failure(
            exc,
            state=state,
            stage=STAGE_VALIDATION,
            response_char_count=len(response_text),
            top_level_keys=top_level_keys,
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
        "BUILDER2_WINNER_DEVELOPMENT_OK jobId=%s tournamentId=%s prototypeId=%s",
        job_id,
        tournament_id,
        prototype_id,
    )
    return winner_plan


def parse_winner_response_text(response_text: str) -> Dict[str, Any]:
    return parse_json_object(response_text)
