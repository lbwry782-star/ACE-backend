"""
Builder2 strategic foundation — parsing, validation, repair, and diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected
from engine.builder2_tournament_config import resolve_builder2_creator_model
from engine.builder2_tournament_contracts import (
    STRATEGY_SCHEMA_VERSION,
    VALID_GROUNDING_TYPES,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import extract_responses_output_text, parse_json_object
from engine.builder2_tournament_metrics import MetricsTimer, record_model_call
from engine.builder2_tournament_prompts import build_strategy_prompt, build_strategy_repair_prompt

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_GROUNDING_TYPE_ALIASES = {
    "user provided fact": "user_provided_fact",
    "observable practice": "observable_practice",
    "physical reality": "physical_reality",
    "common market behavior": "common_market_behavior",
    "market behavior": "common_market_behavior",
    "professional knowledge": "professional_knowledge",
}

_LANGUAGE_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "hebrew": "he",
    "iw": "he",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strategy_debug_log_response() -> bool:
    raw = (os.environ.get("BUILDER2_STRATEGY_DEBUG_LOG_RESPONSE") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _redact_excerpt(text: str, *, max_len: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _token_usage_from_response(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out: Dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)):
            out[key] = int(value)
    reasoning = getattr(usage, "reasoning_tokens", None)
    if isinstance(reasoning, (int, float)):
        out["reasoning_tokens"] = int(reasoning)
    cached = getattr(usage, "cached_input_tokens", None)
    if isinstance(cached, (int, float)):
        out["cached_input_tokens"] = int(cached)
    return out


def _normalize_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _LANGUAGE_ALIASES.get(text, text)


def _normalize_grounding_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    if text in _GROUNDING_TYPE_ALIASES:
        return _GROUNDING_TYPE_ALIASES[text]
    return text.replace(" ", "_")


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_strategy_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(raw)
    out["language"] = _normalize_language(out.get("language"))
    pp = out.get("problemPerception")
    if isinstance(pp, dict):
        pp = dict(pp)
        pp["groundingType"] = _normalize_grounding_type(pp.get("groundingType"))
        pp["groundingEvidence"] = _normalize_string_list(pp.get("groundingEvidence"))
        out["problemPerception"] = pp
    ms = out.get("mechanismScan")
    if isinstance(ms, dict):
        ms = dict(ms)
        ms["domainFacts"] = _normalize_string_list(ms.get("domainFacts"))
        out["mechanismScan"] = ms
    return out


def _raise_strategy_error(code: str, *, field: str | None = None) -> None:
    if field:
        raise Builder2TournamentError(f"{code}:{field}")
    raise Builder2TournamentError(code)


def validate_strategy_foundation(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a parsed strategy object. Raises Builder2TournamentError with a precise code.
    """
    normalized = normalize_strategy_raw(raw)
    planning_failure = normalized.get("planningFailure")
    if planning_failure:
        if planning_failure == "builder2_strategy_not_grounded":
            _raise_strategy_error("builder2_strategy_not_grounded")
        _raise_strategy_error("builder2_strategy_validation_failed", field="planningFailure")

    if normalized.get("schemaVersion") != STRATEGY_SCHEMA_VERSION:
        _raise_strategy_error("builder2_strategy_schema_invalid", field="schemaVersion")

    try:
        require_non_empty_str(normalized.get("productNameResolved"), field="productNameResolved")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    lang = require_non_empty_str(normalized.get("language"), field="language")
    if lang not in {"he", "en"}:
        _raise_strategy_error("builder2_strategy_schema_invalid", field="language")

    try:
        pp = require_dict(normalized.get("problemPerception"), field="problemPerception")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    try:
        require_non_empty_str(pp.get("statement"), field="problemPerception.statement")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    gt = require_non_empty_str(pp.get("groundingType"), field="problemPerception.groundingType")
    if gt not in VALID_GROUNDING_TYPES:
        _raise_strategy_error("builder2_strategy_schema_invalid", field="problemPerception.groundingType")

    evidence = pp.get("groundingEvidence")
    if not isinstance(evidence, list) or not evidence:
        _raise_strategy_error("builder2_strategy_validation_failed", field="problemPerception.groundingEvidence")

    try:
        require_non_empty_str(pp.get("whyItMatters"), field="problemPerception.whyItMatters")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    try:
        ra = require_dict(normalized.get("relativeAdvantage"), field="relativeAdvantage")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    try:
        require_non_empty_str(ra.get("statement"), field="relativeAdvantage.statement")
        require_non_empty_str(ra.get("derivationFromProblem"), field="relativeAdvantage.derivationFromProblem")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    try:
        ms = require_dict(normalized.get("mechanismScan"), field="mechanismScan")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    facts = ms.get("domainFacts")
    if not isinstance(facts, list) or not facts:
        _raise_strategy_error("builder2_strategy_validation_failed", field="mechanismScan.domainFacts")

    try:
        require_non_empty_str(ms.get("discoveredMechanism"), field="mechanismScan.discoveredMechanism")
        require_non_empty_str(ms.get("creativeOpportunity"), field="mechanismScan.creativeOpportunity")
    except Builder2TournamentError as exc:
        _raise_strategy_error("builder2_strategy_schema_invalid", field=_field_from_error(exc))

    return normalized


def _field_from_error(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "")
    if msg.startswith("builder2_tournament_invalid_field:"):
        return msg.split(":", 1)[1]
    return "unknown"


def _failure_code(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "builder2_strategy_validation_failed")
    return msg.split(":", 1)[0]


def _failure_field(exc: Builder2TournamentError) -> Optional[str]:
    msg = str(exc.args[0] if exc.args else "")
    if ":" in msg:
        return msg.split(":", 1)[1]
    return None


def _repairable_failure(code: str) -> bool:
    return code in {"builder2_strategy_schema_invalid", "builder2_strategy_validation_failed"}


def _write_strategy_diagnostics(
    state: Dict[str, Any],
    *,
    parse_status: str,
    schema_status: str,
    validation_field: Optional[str],
    validation_reason: str,
    response_length: int,
    top_level_keys: List[str],
    schema_version_received: Optional[str],
    token_usage: Optional[Dict[str, Any]] = None,
    repair_attempted: bool = False,
    response_excerpt: Optional[str] = None,
) -> None:
    record = {
        "receivedAt": _utc_now_iso(),
        "parseStatus": parse_status,
        "schemaStatus": schema_status,
        "validationFieldPaths": [validation_field] if validation_field else [],
        "validationReason": validation_reason,
        "responseLength": response_length,
        "topLevelKeys": top_level_keys,
        "schemaVersionReceived": schema_version_received,
        "tokenUsage": token_usage or {},
        "repairAttempted": repair_attempted,
    }
    if response_excerpt and _strategy_debug_log_response():
        record["debugResponseExcerpt"] = response_excerpt
    state["strategyDiagnostics"] = record


def _parse_strategy_payload(text: str) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    raw_text = (text or "").strip()
    if not raw_text:
        raise Builder2TournamentError("builder2_strategy_empty_response")
    try:
        parsed = parse_json_object(raw_text)
    except ValueError as exc:
        reason = str(exc)
        if reason == "empty_response":
            raise Builder2TournamentError("builder2_strategy_empty_response") from exc
        raise Builder2TournamentError("builder2_strategy_malformed_response") from exc
    keys = sorted(parsed.keys())
    schema_version = parsed.get("schemaVersion")
    schema_version_received = str(schema_version) if schema_version is not None else None
    return parsed, keys, schema_version_received


def _invoke_strategy_model(
    *,
    prompt: str,
    model: str,
    llm_client: Optional[Any],
    call_type: str,
) -> Tuple[str, Optional[Any], Dict[str, Any]]:
    log_builder2_model_selected(role="builder2_strategy", call_type=call_type)
    if llm_client is not None:
        raw = llm_client(role="builder2_strategy", model=model, prompt=prompt)
        if isinstance(raw, dict):
            text = json.dumps(raw, ensure_ascii=False)
            return text, None, {}
        return str(raw or ""), None, {}

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    timeout = float((os.environ.get("BUILDER2_TOURNAMENT_TIMEOUT_SECONDS") or "150").strip() or "150")
    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(timeout), max_retries=0)
    reasoning = build_builder2_reasoning_payload()
    response = openai_retry.openai_call_with_retry(
        lambda: client.responses.create(model=model, input=prompt, reasoning=reasoning),
        endpoint="responses",
    )
    text = extract_responses_output_text(response)
    return text, response, _token_usage_from_response(response)


def generate_strategy_foundation(
    *,
    product_name: str,
    product_description: str,
    language: str,
    llm_client: Optional[Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    job_id = state.get("jobId") or ""
    tournament_id = state.get("tournamentId") or ""
    model = resolve_builder2_creator_model()
    prompt = build_strategy_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
    )

    logger.info("BUILDER2_STRATEGY_GENERATION_START jobId=%s tournamentId=%s model=%s", job_id, tournament_id, model)
    timer = MetricsTimer()
    response_text, response_obj, token_usage = _invoke_strategy_model(
        prompt=prompt,
        model=model,
        llm_client=llm_client,
        call_type="normal",
    )
    elapsed = timer.elapsed_ms()
    record_model_call(state, role="builder2_strategy", elapsed_ms=elapsed, token_usage=token_usage or None)

    logger.info(
        "BUILDER2_STRATEGY_RESPONSE_RECEIVED jobId=%s tournamentId=%s model=%s elapsedMs=%.1f responseChars=%s",
        job_id,
        tournament_id,
        model,
        elapsed,
        len(response_text or ""),
    )
    if _strategy_debug_log_response() and response_text:
        logger.info(
            "BUILDER2_STRATEGY_RESPONSE_EXCERPT jobId=%s excerpt=%s",
            job_id,
            _redact_excerpt(response_text),
        )

    last_exc: Optional[Builder2TournamentError] = None
    repair_attempted = False
    for attempt in ("normal", "repair"):
        try:
            if attempt == "repair":
                if last_exc is None or not _repairable_failure(_failure_code(last_exc)):
                    break
                repair_attempted = True
                repair_prompt = build_strategy_repair_prompt(
                    product_name=product_name,
                    product_description=product_description,
                    language=language,
                    invalid_output=parse_json_object(response_text) if response_text else {},
                    validation_failures=[str(last_exc.args[0])],
                )
                timer = MetricsTimer()
                response_text, response_obj, token_usage = _invoke_strategy_model(
                    prompt=repair_prompt,
                    model=model,
                    llm_client=llm_client,
                    call_type="repair",
                )
                elapsed = timer.elapsed_ms()
                record_model_call(
                    state,
                    role="builder2_strategy",
                    elapsed_ms=elapsed,
                    repair=True,
                    token_usage=token_usage or None,
                )
                logger.info(
                    "BUILDER2_STRATEGY_REPAIR_RESPONSE_RECEIVED jobId=%s tournamentId=%s elapsedMs=%.1f responseChars=%s",
                    job_id,
                    tournament_id,
                    elapsed,
                    len(response_text or ""),
                )

            parsed, top_level_keys, schema_version_received = _parse_strategy_payload(response_text)
            _write_strategy_diagnostics(
                state,
                parse_status="ok",
                schema_status="pending",
                validation_field=None,
                validation_reason="",
                response_length=len(response_text or ""),
                top_level_keys=top_level_keys,
                schema_version_received=schema_version_received,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                response_excerpt=_redact_excerpt(response_text) if response_text else None,
            )
            foundation = validate_strategy_foundation(parsed)
            _write_strategy_diagnostics(
                state,
                parse_status="ok",
                schema_status="ok",
                validation_field=None,
                validation_reason="ok",
                response_length=len(response_text or ""),
                top_level_keys=top_level_keys,
                schema_version_received=schema_version_received,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                response_excerpt=_redact_excerpt(response_text) if response_text else None,
            )
            logger.info(
                "BUILDER2_STRATEGY_GENERATION_OK jobId=%s tournamentId=%s repairAttempted=%s",
                job_id,
                tournament_id,
                str(repair_attempted).lower(),
            )
            return foundation
        except Builder2TournamentError as exc:
            last_exc = exc
            code = _failure_code(exc)
            field = _failure_field(exc)
            if code == "builder2_strategy_empty_response":
                logger.error(
                    "BUILDER2_STRATEGY_PARSE_FAILED jobId=%s tournamentId=%s reason=%s exception=%s repairAttempted=%s",
                    job_id,
                    tournament_id,
                    code,
                    exc.__class__.__name__,
                    str(repair_attempted).lower(),
                )
            elif code == "builder2_strategy_malformed_response":
                logger.error(
                    "BUILDER2_STRATEGY_PARSE_FAILED jobId=%s tournamentId=%s reason=%s exception=%s repairAttempted=%s",
                    job_id,
                    tournament_id,
                    code,
                    exc.__class__.__name__,
                    str(repair_attempted).lower(),
                )
            elif code == "builder2_strategy_schema_invalid":
                logger.error(
                    "BUILDER2_STRATEGY_SCHEMA_FAILED jobId=%s tournamentId=%s field=%s reason=%s exception=%s repairAttempted=%s",
                    job_id,
                    tournament_id,
                    field or "",
                    code,
                    exc.__class__.__name__,
                    str(repair_attempted).lower(),
                )
            elif code == "builder2_strategy_not_grounded":
                logger.error(
                    "BUILDER2_STRATEGY_VALIDATION_FAILED jobId=%s tournamentId=%s field=%s reason=%s exception=%s repairAttempted=%s",
                    job_id,
                    tournament_id,
                    field or "",
                    code,
                    exc.__class__.__name__,
                    str(repair_attempted).lower(),
                )
            else:
                logger.error(
                    "BUILDER2_STRATEGY_VALIDATION_FAILED jobId=%s tournamentId=%s field=%s reason=%s exception=%s repairAttempted=%s",
                    job_id,
                    tournament_id,
                    field or "",
                    code,
                    exc.__class__.__name__,
                    str(repair_attempted).lower(),
                )

            keys_for_diag: List[str] = []
            schema_version_received: Optional[str] = None
            if response_text and code != "builder2_strategy_malformed_response":
                try:
                    parsed_for_diag, keys_for_diag, schema_version_received = _parse_strategy_payload(response_text)
                except Builder2TournamentError:
                    keys_for_diag = []
            _write_strategy_diagnostics(
                state,
                parse_status="ok"
                if code not in {"builder2_strategy_empty_response", "builder2_strategy_malformed_response"}
                else "failed",
                schema_status="failed" if code.startswith("builder2_strategy_schema") else "pending",
                validation_field=field,
                validation_reason=str(exc.args[0]),
                response_length=len(response_text or ""),
                top_level_keys=keys_for_diag,
                schema_version_received=schema_version_received,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                response_excerpt=_redact_excerpt(response_text) if response_text else None,
            )

            if attempt == "normal" and _repairable_failure(code):
                continue
            logger.error(
                "BUILDER2_STRATEGY_GENERATION_FAILED jobId=%s tournamentId=%s reason=%s repairAttempted=%s",
                job_id,
                tournament_id,
                str(exc.args[0]),
                str(repair_attempted).lower(),
            )
            raise

    assert last_exc is not None
    raise last_exc
