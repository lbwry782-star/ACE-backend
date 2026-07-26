"""
Builder2 Winner Development diagnostics — safe staged logging and failure classification.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

PUBLIC_FAILURE_CODE = "builder2_winner_development_failed"

STAGE_API_CALL = "api_call"
STAGE_RESPONSE_RECEIVED = "response_received"
STAGE_EXTRACTION = "extraction"
STAGE_VALIDATION = "validation"
STAGE_METHODOLOGY_VALIDATION = "methodology_validation"
STAGE_NORMALIZATION = "normalization"
STAGE_DOWNSTREAM_NORMALIZATION = "downstream_normalization"
STAGE_PERSISTENCE = "persistence"


def _failure_field(exc: BaseException) -> Optional[str]:
    message = str(exc.args[0] if exc.args else exc)
    if ":" in message:
        return message.split(":", 1)[1]
    return None


def _failure_code(exc: BaseException) -> str:
    message = str(exc.args[0] if exc.args else exc)
    if ":" in message:
        return message.split(":", 1)[0]
    return message


def classify_winner_failure(
    exc: BaseException,
    *,
    stage: str,
    response_char_count: int = 0,
    top_level_keys: Optional[List[str]] = None,
    repair_attempted: bool = False,
) -> Dict[str, Any]:
    code = _failure_code(exc)
    field_path = _failure_field(exc)
    category = "structural"
    precise_reason = code

    if stage == STAGE_EXTRACTION:
        category = "structural"
        precise_reason = str(exc.args[0] if exc.args else exc)
    elif code.startswith("builder2_winner_schema_invalid"):
        category = "structural"
        stage = STAGE_VALIDATION
        precise_reason = field_path or code
    elif code.startswith("builder2_winner_validation_failed"):
        category = "methodology"
        stage = STAGE_METHODOLOGY_VALIDATION
        precise_reason = field_path or code
    elif code.startswith("builder2_winner_downstream_invalid"):
        category = "structural"
        stage = STAGE_DOWNSTREAM_NORMALIZATION
        precise_reason = field_path or code
    elif stage == STAGE_VALIDATION:
        category = "structural"
        precise_reason = field_path or code
    elif stage == STAGE_PERSISTENCE:
        category = "persistence"
        precise_reason = code

    return {
        "stage": stage,
        "category": category,
        "exceptionClass": exc.__class__.__name__,
        "fieldPath": field_path,
        "preciseReason": precise_reason,
        "publicReason": PUBLIC_FAILURE_CODE,
        "responseCharCount": response_char_count,
        "topLevelKeys": list(top_level_keys or []),
        "repairAttempted": repair_attempted,
    }


def persist_winner_failure_diagnostics(state: Optional[Dict[str, Any]], diagnostics: Dict[str, Any]) -> None:
    if state is None:
        return
    state["winnerDevelopmentFailure"] = deepcopy(diagnostics)


def clear_winner_failure_diagnostics(state: Optional[Dict[str, Any]]) -> None:
    if state is None:
        return
    state.pop("winnerDevelopmentFailure", None)


def log_winner_development_response_received(
    *,
    job_id: str,
    tournament_id: str,
    prototype_id: str,
    response_char_count: int,
) -> None:
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_RESPONSE_RECEIVED jobId=%s tournamentId=%s prototypeId=%s responseCharCount=%s",
        job_id,
        tournament_id,
        prototype_id,
        response_char_count,
    )


def log_winner_development_extraction_ok(
    *,
    job_id: str,
    tournament_id: str,
    prototype_id: str,
    top_level_keys: List[str],
) -> None:
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_EXTRACTION_OK jobId=%s tournamentId=%s prototypeId=%s topLevelKeyCount=%s keys=%s",
        job_id,
        tournament_id,
        prototype_id,
        len(top_level_keys),
        ",".join(sorted(top_level_keys)[:12]),
    )


def log_winner_development_validation_ok(
    *,
    job_id: str,
    tournament_id: str,
    prototype_id: str,
) -> None:
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_VALIDATION_OK jobId=%s tournamentId=%s prototypeId=%s",
        job_id,
        tournament_id,
        prototype_id,
    )


def log_winner_development_normalization_ok(
    *,
    job_id: str,
    tournament_id: str,
    prototype_id: str,
) -> None:
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_NORMALIZATION_OK jobId=%s tournamentId=%s prototypeId=%s",
        job_id,
        tournament_id,
        prototype_id,
    )


def log_winner_development_persisted(
    *,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    prototype_id: str,
) -> None:
    logger.info(
        "BUILDER2_WINNER_DEVELOPMENT_PERSISTED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s",
        job_id,
        tournament_id,
        candidate_id,
        prototype_id,
    )


def log_winner_development_failed(diagnostics: Dict[str, Any], *, job_id: str, tournament_id: str, prototype_id: str) -> None:
    logger.error(
        "BUILDER2_WINNER_DEVELOPMENT_FAILED jobId=%s tournamentId=%s prototypeId=%s stage=%s category=%s "
        "exceptionClass=%s fieldPath=%s preciseReason=%s responseCharCount=%s topLevelKeyCount=%s repairAttempted=%s",
        job_id,
        tournament_id,
        prototype_id,
        diagnostics.get("stage"),
        diagnostics.get("category"),
        diagnostics.get("exceptionClass"),
        diagnostics.get("fieldPath"),
        diagnostics.get("preciseReason"),
        diagnostics.get("responseCharCount"),
        len(diagnostics.get("topLevelKeys") or []),
        diagnostics.get("repairAttempted"),
    )


def raise_public_winner_failure(
    exc: BaseException,
    *,
    state: Optional[Dict[str, Any]],
    stage: str,
    response_char_count: int = 0,
    top_level_keys: Optional[List[str]] = None,
    repair_attempted: bool = False,
) -> None:
    diagnostics = classify_winner_failure(
        exc,
        stage=stage,
        response_char_count=response_char_count,
        top_level_keys=top_level_keys,
        repair_attempted=repair_attempted,
    )
    persist_winner_failure_diagnostics(state, diagnostics)
    log_winner_development_failed(
        diagnostics,
        job_id=str((state or {}).get("jobId") or ""),
        tournament_id=str((state or {}).get("tournamentId") or ""),
        prototype_id=str((state or {}).get("winnerDevelopmentPrototypeId") or (state or {}).get("currentWinnerPrototypeId") or ""),
    )
    raise Builder2TournamentError(PUBLIC_FAILURE_CODE) from exc


def safe_top_level_keys(raw: Any) -> List[str]:
    if isinstance(raw, dict):
        return sorted(str(key) for key in raw.keys())
    return []
