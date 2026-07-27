"""
Builder2 reasoning resume — safe structured terminal failure diagnostics.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_exception_message(exc: BaseException, *, max_len: int = 300) -> str:
    return " ".join(str(exc).split())[:max_len]


def openai_http_status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status)
        except (TypeError, ValueError):
            return None
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status is not None:
            try:
                return int(response_status)
            except (TypeError, ValueError):
                return None
    return None


def parsing_failure_category(reason: str) -> Optional[str]:
    code = (reason or "").split(":", 1)[0]
    if code == "builder2_creator_empty_response":
        return "empty_response"
    if code == "builder2_creator_malformed_response":
        return "malformed_json"
    if code == "builder2_creator_schema_invalid":
        return "schema_invalid"
    return None


def log_reasoning_resume_failed(
    log: logging.Logger,
    *,
    job_id: str,
    failure_stage: str,
    failure_reason: str,
    event: str = "BUILDER2_REASONING_RESUME_FAILED",
    tournament_id: str = "",
    prototype_id: str = "",
    reasoning_role: str = "",
    model: str = "",
    exception_class: str = "",
    http_status: Optional[int] = None,
    response_text_present: Optional[bool] = None,
    response_text_chars: Optional[int] = None,
    parsing_failure_category_value: str = "",
    validation_rejection_code: str = "",
    redis_mutated: bool = False,
    lease_released: bool = False,
    with_traceback: bool = False,
    exc: Optional[BaseException] = None,
) -> None:
    message = (
        f"{event} jobId=%s tournamentId=%s prototypeId=%s "
        "reasoningRole=%s model=%s failureStage=%s failureReason=%s exceptionClass=%s "
        "httpStatus=%s responseTextPresent=%s responseTextChars=%s parsingFailureCategory=%s "
        "validationRejectionCode=%s redisMutated=%s leaseReleased=%s"
    )
    args = (
        job_id or "(none)",
        tournament_id or "(none)",
        prototype_id or "(none)",
        reasoning_role or "(none)",
        model or "(none)",
        failure_stage or "(none)",
        failure_reason or "(none)",
        exception_class or "(none)",
        http_status if http_status is not None else "(none)",
        str(response_text_present).lower() if response_text_present is not None else "(none)",
        response_text_chars if response_text_chars is not None else "(none)",
        parsing_failure_category_value or "(none)",
        validation_rejection_code or failure_reason or "(none)",
        str(redis_mutated).lower(),
        str(lease_released).lower(),
    )
    if with_traceback and exc is not None:
        log.exception(message, *args)
    else:
        log.error(message, *args)
