"""
Builder2 Judge role — scoring, validation, repair, retry, and diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_prototypes import Builder2Prototype, require_prototype
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected
from engine.builder2_tournament_config import resolve_builder2_judge_model
from engine.builder2_tournament_contracts import (
    JUDGE_PURITY_RULES,
    JUDGE_SCORE_MAX_TOTAL,
    JUDGE_SCORE_RANGES,
    JUDGMENT_SCHEMA_VERSION,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import extract_responses_output_text, parse_json_object
from engine.builder2_tournament_metrics import MetricsTimer, record_judge_unavailable, record_model_call
from engine.builder2_tournament_prompts import (
    build_judge_prompt,
    build_judge_repair_prompt,
    build_judge_retry_prompt,
)

logger = logging.getLogger(__name__)

JUDGE_CONFIDENCE_MIN = 0.0
JUDGE_CONFIDENCE_MAX = 1.0

_CREATIVE_RETRY_FIELDS = frozenset()


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _judge_debug_log_response() -> bool:
    raw = (os.environ.get("BUILDER2_JUDGE_DEBUG_LOG_RESPONSE") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _redact_excerpt(text: str, *, max_len: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _failure_code(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "builder2_judge_invalid_response")
    return msg.split(":", 1)[0]


def _failure_field(exc: Builder2TournamentError) -> Optional[str]:
    msg = str(exc.args[0] if exc.args else "")
    prefix = (
        "builder2_judge_schema_invalid:",
        "builder2_judge_score_invalid:",
        "builder2_judge_validation_failed:",
    )
    for p in prefix:
        if msg.startswith(p):
            return msg.split(":", 1)[1]
    return None


def _failure_rule(exc: Builder2TournamentError) -> Optional[str]:
    msg = str(exc.args[0] if exc.args else "")
    if msg.startswith("builder2_judge_purity_violation:"):
        return msg.split(":", 1)[1]
    return None


def _raise_judge_error(code: str, *, field: str | None = None, rule: str | None = None) -> None:
    if rule:
        raise Builder2TournamentError(f"{code}:{rule}")
    if field:
        raise Builder2TournamentError(f"{code}:{field}")
    raise Builder2TournamentError(code)


def _field_from_error(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "")
    if msg.startswith("builder2_tournament_invalid_field:"):
        return msg.split(":", 1)[1]
    return "unknown"


def _coerce_score_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        _raise_judge_error("builder2_judge_score_invalid", field=field)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        _raise_judge_error("builder2_judge_score_invalid", field=field)
    if isinstance(value, str):
        _raise_judge_error("builder2_judge_score_invalid", field=field)
    _raise_judge_error("builder2_judge_score_invalid", field=field)


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, bool):
        _raise_judge_error("builder2_judge_schema_invalid", field="confidence")
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        try:
            value = float(text)
        except ValueError:
            _raise_judge_error("builder2_judge_schema_invalid", field="confidence")
    if not isinstance(value, (int, float)):
        _raise_judge_error("builder2_judge_schema_invalid", field="confidence")
    number = float(value)
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < JUDGE_CONFIDENCE_MIN or number > JUDGE_CONFIDENCE_MAX:
        _raise_judge_error("builder2_judge_schema_invalid", field="confidence")
    return number


def normalize_judge_raw(raw: Dict[str, Any], *, candidate_id: str) -> Dict[str, Any]:
    out = dict(raw)
    out["candidateId"] = str(out.get("candidateId") or candidate_id).strip() or candidate_id
    scores = out.get("scores")
    if isinstance(scores, dict):
        cleaned = dict(scores)
        cleaned.pop("total", None)
        cleaned.pop("totalScore", None)
        out["scores"] = cleaned
    out.pop("totalScore", None)
    out.pop("total", None)
    return out


def _collect_purity_text(judgment: Dict[str, Any]) -> str:
    chunks: List[str] = [str(judgment.get("verdict") or "")]
    chunks.append(str(judgment.get("prototypeQualityComparison") or ""))
    for key in ("strengths", "weaknesses", "disqualifiers"):
        values = judgment.get(key)
        if isinstance(values, list):
            chunks.extend(str(v) for v in values)
    return "\n".join(chunks)


def validate_judge_purity(judgment: Dict[str, Any]) -> None:
    blob = _collect_purity_text(judgment).lower()
    for pattern, rule in JUDGE_PURITY_RULES:
        if pattern in blob:
            _raise_judge_error("builder2_judge_purity_violation", rule=rule)


def calculate_judge_total(scores: Dict[str, int]) -> int:
    total = 0
    for name, (low, high) in JUDGE_SCORE_RANGES.items():
        value = scores.get(name)
        if not isinstance(value, int):
            _raise_judge_error("builder2_judge_score_invalid", field=name)
        if value < low or value > high:
            _raise_judge_error("builder2_judge_score_invalid", field=name)
        total += value
    if total > JUDGE_SCORE_MAX_TOTAL:
        _raise_judge_error("builder2_judge_score_invalid", field="total")
    return total


def validate_judge_response(
    judgment: Dict[str, Any],
    *,
    candidate_id: str,
) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
    normalized = normalize_judge_raw(judgment, candidate_id=candidate_id)

    if normalized.get("schemaVersion") != JUDGMENT_SCHEMA_VERSION:
        _raise_judge_error("builder2_judge_schema_invalid", field="schemaVersion")

    cid = require_non_empty_str(normalized.get("candidateId"), field="candidateId")
    if cid != candidate_id:
        _raise_judge_error("builder2_judge_validation_failed", field="candidateId")

    eligible = normalized.get("eligible")
    if not isinstance(eligible, bool):
        _raise_judge_error("builder2_judge_schema_invalid", field="eligible")

    disqualifiers = normalized.get("disqualifiers")
    if not isinstance(disqualifiers, list):
        _raise_judge_error("builder2_judge_schema_invalid", field="disqualifiers")

    original_scores = judgment.get("scores") if isinstance(judgment.get("scores"), dict) else {}
    if (
        "total" in original_scores
        or "totalScore" in original_scores
        or "totalScore" in judgment
        or "total" in judgment
    ):
        _raise_judge_error("builder2_judge_schema_invalid", field="scores.total")

    try:
        scores_raw = require_dict(normalized.get("scores"), field="scores")
    except Builder2TournamentError as exc:
        _raise_judge_error("builder2_judge_schema_invalid", field=_field_from_error(exc))

    if "total" in scores_raw or "totalScore" in scores_raw:
        _raise_judge_error("builder2_judge_schema_invalid", field="scores.total")

    unknown = [key for key in scores_raw if key not in JUDGE_SCORE_RANGES]
    if unknown:
        _raise_judge_error("builder2_judge_schema_invalid", field="scores")

    scores: Dict[str, int] = {}
    for name in JUDGE_SCORE_RANGES:
        if name not in scores_raw:
            _raise_judge_error("builder2_judge_schema_invalid", field=f"scores.{name}")
        scores[name] = _coerce_score_value(scores_raw.get(name), field=name)

    total = calculate_judge_total(scores)

    try:
        require_non_empty_str(normalized.get("verdict"), field="verdict")
    except Builder2TournamentError as exc:
        _raise_judge_error("builder2_judge_validation_failed", field=_field_from_error(exc))

    strengths = normalized.get("strengths")
    weaknesses = normalized.get("weaknesses")
    if not isinstance(strengths, list) or not isinstance(weaknesses, list):
        _raise_judge_error("builder2_judge_schema_invalid", field="strengths" if not isinstance(strengths, list) else "weaknesses")

    try:
        require_non_empty_str(normalized.get("prototypeQualityComparison"), field="prototypeQualityComparison")
    except Builder2TournamentError as exc:
        _raise_judge_error("builder2_judge_validation_failed", field=_field_from_error(exc))

    confidence = _normalize_confidence(normalized.get("confidence"))

    out = dict(normalized)
    out["confidence"] = confidence
    if not eligible and not disqualifiers:
        out["disqualifiers"] = ["ineligible_without_reason"]

    validate_judge_purity(out)
    return out, total, scores


def _is_structural_repairable(code: str, field: Optional[str]) -> bool:
    if code in {
        "builder2_judge_schema_invalid",
        "builder2_judge_score_invalid",
        "builder2_judge_validation_failed",
    }:
        return field not in _CREATIVE_RETRY_FIELDS
    return False


def _is_clean_retryable(code: str, field: Optional[str]) -> bool:
    if code == "builder2_judge_purity_violation":
        return True
    return False


def _write_judge_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    response_received: bool,
    response_length: int,
    top_level_keys: List[str],
    schema_version_received: Optional[str],
    parse_status: str,
    schema_status: str,
    score_status: str,
    validation_status: str,
    purity_status: str,
    failure_field_paths: List[str],
    failure_reason: Optional[str],
    repair_attempted: bool,
    clean_retry_attempted: bool,
    response_excerpt: Optional[str] = None,
) -> None:
    diagnostics.clear()
    diagnostics.update(
        {
            "responseReceived": response_received,
            "responseLength": response_length,
            "topLevelKeys": top_level_keys,
            "schemaVersionReceived": schema_version_received,
            "parseStatus": parse_status,
            "schemaStatus": schema_status,
            "scoreStatus": score_status,
            "validationStatus": validation_status,
            "purityStatus": purity_status,
            "failureFieldPaths": failure_field_paths,
            "failureReason": failure_reason,
            "repairAttempted": repair_attempted,
            "cleanRetryAttempted": clean_retry_attempted,
        }
    )
    if response_excerpt and _judge_debug_log_response():
        diagnostics["debugResponseExcerpt"] = response_excerpt


def _invoke_judge_model(
    *,
    prompt: str,
    model: str,
    llm_client: Optional[Any],
    call_type: str,
) -> str:
    log_builder2_model_selected(role="builder2_judge", call_type=call_type)
    if llm_client is not None:
        raw = llm_client(role="builder2_judge", model=model, prompt=prompt)
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False)
        return str(raw or "")

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
    return extract_responses_output_text(response)


def _parse_judge_payload(text: str) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    raw_text = (text or "").strip()
    if not raw_text:
        _raise_judge_error("builder2_judge_empty_response")
    try:
        parsed = parse_json_object(raw_text)
    except ValueError as exc:
        if str(exc) == "empty_response":
            _raise_judge_error("builder2_judge_empty_response")
        _raise_judge_error("builder2_judge_malformed_response")
    keys = sorted(parsed.keys())
    schema_version = parsed.get("schemaVersion")
    return parsed, keys, str(schema_version) if schema_version is not None else None


def _log_judge_failure(
    *,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    judgment_id: str,
    prototype_id: str,
    exc: Builder2TournamentError,
    repair_attempted: bool,
    clean_retry_attempted: bool,
) -> None:
    code = _failure_code(exc)
    field = _failure_field(exc)
    rule = _failure_rule(exc)
    common = (
        f"jobId={job_id} tournamentId={tournament_id} candidateId={candidate_id} judgmentId={judgment_id} "
        f"prototypeId={prototype_id} repairAttempted={str(repair_attempted).lower()} "
        f"cleanRetryAttempted={str(clean_retry_attempted).lower()}"
    )
    if code in {"builder2_judge_empty_response", "builder2_judge_malformed_response"}:
        logger.error("BUILDER2_JUDGE_PARSE_FAILED %s reason=%s", common, exc.args[0])
    elif code == "builder2_judge_schema_invalid":
        logger.error("BUILDER2_JUDGE_SCHEMA_FAILED %s field=%s reason=%s", common, field or "", exc.args[0])
    elif code == "builder2_judge_score_invalid":
        logger.error("BUILDER2_JUDGE_SCORE_FAILED %s field=%s reason=%s", common, field or "", exc.args[0])
    elif code == "builder2_judge_purity_violation":
        logger.error("BUILDER2_JUDGE_PURITY_FAILED %s rule=%s reason=%s", common, rule or "", exc.args[0])
    else:
        logger.error("BUILDER2_JUDGE_VALIDATION_FAILED %s field=%s reason=%s", common, field or "", exc.args[0])


def judge_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype_id: str,
    candidate_id: str,
    candidate: Dict[str, Any],
    llm_client: Optional[Any] = None,
    state: Optional[Dict[str, Any]] = None,
    judgment_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
    prototype = require_prototype(prototype_id)
    judgment_id = judgment_id or f"judge-{candidate_id}-{uuid.uuid4().hex[:8]}"
    model = resolve_builder2_judge_model()
    job_id = (state or {}).get("jobId") or ""
    tournament_id = (state or {}).get("tournamentId") or ""
    diagnostics: Dict[str, Any] = {}
    repair_attempted = False
    clean_retry_attempted = False
    retry_rule: Optional[str] = None

    base_prompt = build_judge_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        prototype=prototype,
        candidate=candidate,
        candidate_id=candidate_id,
    )

    logger.info(
        "BUILDER2_JUDGE_START candidateId=%s judgmentId=%s prototypeId=%s",
        candidate_id,
        judgment_id,
        prototype_id,
    )

    last_exc: Optional[Builder2TournamentError] = None
    last_parsed: Dict[str, Any] = {}
    response_text = ""

    for phase in ("normal", "repair", "retry"):
        if phase == "repair":
            if last_exc is None or not _is_structural_repairable(_failure_code(last_exc), _failure_field(last_exc)):
                continue
            repair_attempted = True
            logger.info(
                "BUILDER2_JUDGE_REPAIR_START candidateId=%s judgmentId=%s reason=%s",
                candidate_id,
                judgment_id,
                last_exc.args[0],
            )
            prompt = build_judge_repair_prompt(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=strategy_foundation,
                prototype=prototype,
                candidate=candidate,
                candidate_id=candidate_id,
                invalid_output=last_parsed,
                validation_failures=[str(last_exc.args[0])],
            )
            call_type = "repair"
        elif phase == "retry":
            if clean_retry_attempted or last_exc is None:
                continue
            if not _is_clean_retryable(_failure_code(last_exc), _failure_field(last_exc)):
                continue
            clean_retry_attempted = True
            logger.info(
                "BUILDER2_JUDGE_RETRY_START candidateId=%s judgmentId=%s reason=%s",
                candidate_id,
                judgment_id,
                last_exc.args[0],
            )
            prompt = build_judge_retry_prompt(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=strategy_foundation,
                prototype=prototype,
                candidate=candidate,
                candidate_id=candidate_id,
                retry_rule=retry_rule or _failure_rule(last_exc) or str(last_exc.args[0]),
            )
            call_type = "retry"
        else:
            prompt = base_prompt
            call_type = "normal"

        timer = MetricsTimer()
        response_text = _invoke_judge_model(prompt=prompt, model=model, llm_client=llm_client, call_type=call_type)
        elapsed = timer.elapsed_ms()
        if state is not None:
            record_model_call(
                state,
                role="builder2_judge",
                elapsed_ms=elapsed,
                repair=(call_type == "repair"),
                retry=(call_type == "retry"),
            )

        logger.info(
            "BUILDER2_JUDGE_RESPONSE_RECEIVED jobId=%s tournamentId=%s candidateId=%s judgmentId=%s "
            "prototypeId=%s model=%s elapsedMs=%.1f responseChars=%s callType=%s",
            job_id,
            tournament_id,
            candidate_id,
            judgment_id,
            prototype_id,
            model,
            elapsed,
            len(response_text or ""),
            call_type,
        )
        if _judge_debug_log_response() and response_text:
            logger.info(
                "BUILDER2_JUDGE_RESPONSE_EXCERPT candidateId=%s excerpt=%s",
                candidate_id,
                _redact_excerpt(response_text),
            )

        try:
            parsed, top_level_keys, schema_version_received = _parse_judge_payload(response_text)
            last_parsed = parsed
            judgment, total, scores = validate_judge_response(parsed, candidate_id=candidate_id)
            _write_judge_diagnostics(
                diagnostics,
                response_received=True,
                response_length=len(response_text or ""),
                top_level_keys=top_level_keys,
                schema_version_received=schema_version_received,
                parse_status="ok",
                schema_status="ok",
                score_status="ok",
                validation_status="ok",
                purity_status="ok",
                failure_field_paths=[],
                failure_reason=None,
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
                response_excerpt=_redact_excerpt(response_text),
            )
            if repair_attempted and phase == "repair":
                logger.info("BUILDER2_JUDGE_REPAIR_OK candidateId=%s judgmentId=%s", candidate_id, judgment_id)
            if clean_retry_attempted and phase == "retry":
                logger.info("BUILDER2_JUDGE_RETRY_OK candidateId=%s judgmentId=%s", candidate_id, judgment_id)
            logger.info(
                "BUILDER2_JUDGE_OK candidateId=%s judgmentId=%s eligible=%s total=%s",
                candidate_id,
                judgment_id,
                str(judgment.get("eligible")).lower(),
                total,
            )
            if state is not None:
                state.setdefault("judgeDiagnosticsByCandidate", {})[candidate_id] = dict(diagnostics)
            return judgment_id, judgment, total, scores
        except Builder2TournamentError as exc:
            last_exc = exc
            retry_rule = _failure_rule(exc) or _failure_field(exc)
            _log_judge_failure(
                job_id=job_id,
                tournament_id=tournament_id,
                candidate_id=candidate_id,
                judgment_id=judgment_id,
                prototype_id=prototype_id,
                exc=exc,
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
            )
            code = _failure_code(exc)
            field = _failure_field(exc)
            _write_judge_diagnostics(
                diagnostics,
                response_received=bool(response_text),
                response_length=len(response_text or ""),
                top_level_keys=sorted(last_parsed.keys()) if last_parsed else [],
                schema_version_received=str(last_parsed.get("schemaVersion")) if last_parsed.get("schemaVersion") else None,
                parse_status="failed"
                if code in {"builder2_judge_empty_response", "builder2_judge_malformed_response"}
                else "ok",
                schema_status="failed" if code == "builder2_judge_schema_invalid" else "pending",
                score_status="failed" if code == "builder2_judge_score_invalid" else "pending",
                validation_status="failed"
                if code in {"builder2_judge_validation_failed", "builder2_judge_invalid_response"}
                else "pending",
                purity_status="failed" if code == "builder2_judge_purity_violation" else "pending",
                failure_field_paths=[field] if field else ([retry_rule] if retry_rule else []),
                failure_reason=str(exc.args[0]),
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
                response_excerpt=_redact_excerpt(response_text) if response_text else None,
            )
            if phase == "repair":
                logger.error(
                    "BUILDER2_JUDGE_REPAIR_FAILED candidateId=%s judgmentId=%s reason=%s",
                    candidate_id,
                    judgment_id,
                    exc.args[0],
                )
            if phase == "retry":
                logger.error(
                    "BUILDER2_JUDGE_RETRY_FAILED candidateId=%s judgmentId=%s reason=%s",
                    candidate_id,
                    judgment_id,
                    exc.args[0],
                )
            if phase == "normal" and _is_structural_repairable(code, field):
                continue
            if phase in {"normal", "repair"} and _is_clean_retryable(code, field):
                continue
            break

    assert last_exc is not None
    if state is not None:
        state.setdefault("judgeDiagnosticsByCandidate", {})[candidate_id] = dict(diagnostics)
        record_judge_unavailable(state)
    final_reason = str(last_exc.args[0])
    logger.error(
        "BUILDER2_JUDGMENT_UNAVAILABLE candidateId=%s judgmentId=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
        candidate_id,
        judgment_id,
        final_reason,
        str(repair_attempted).lower(),
        str(clean_retry_attempted).lower(),
    )
    raise Builder2TournamentError(final_reason)
