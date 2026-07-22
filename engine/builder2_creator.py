"""
Builder2 Creator role — generation, validation, repair, retry, and diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_prototypes import Builder2Prototype, require_prototype
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected
from engine.builder2_tournament_config import resolve_builder2_creator_model
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CREATOR_PURITY_RULES,
    VALID_CONTINUITY_RISK,
    VALID_STRUCTURE_TYPES,
    VALID_VISUAL_PARALLEL_TYPES,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import extract_responses_output_text, parse_json_object
from engine.builder2_tournament_metrics import MetricsTimer, record_creator_rejected, record_model_call
from engine.builder2_tournament_prompts import (
    build_creator_prompt,
    build_creator_repair_prompt,
    build_creator_retry_prompt,
)

logger = logging.getLogger(__name__)

_STRUCTURE_TYPE_ALIASES = {
    "continuous event": "continuous_event",
    "variation montage": "variation_montage",
}

_CONTINUITY_RISK_ALIASES = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_VISUAL_PARALLEL_ALIASES = {
    "side by side": "side_by_side",
    "motion similarity": "motion_similarity",
    "physical behavior": "physical_behavior",
    "graphic similarity": "graphic_similarity",
    "context collision": "context_collision",
    "context replacement": "context_replacement",
    "media replacement": "media_replacement",
    "medium as object": "medium_as_object",
    "essential pairing": "essential_pairing",
    "spatial proximity": "spatial_proximity",
    "consequence embodiment": "consequence_embodiment",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _creator_debug_log_response() -> bool:
    raw = (os.environ.get("BUILDER2_CREATOR_DEBUG_LOG_RESPONSE") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _redact_excerpt(text: str, *, max_len: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _failure_code(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "builder2_creator_invalid_candidate")
    return msg.split(":", 1)[0]


def _failure_field(exc: Builder2TournamentError) -> Optional[str]:
    msg = str(exc.args[0] if exc.args else "")
    if ":" in msg and not msg.startswith("builder2_creator_purity_violation:"):
        return msg.split(":", 1)[1]
    return None


def _failure_rule(exc: Builder2TournamentError) -> Optional[str]:
    msg = str(exc.args[0] if exc.args else "")
    if msg.startswith("builder2_creator_purity_violation:"):
        return msg.split(":", 1)[1]
    return None


def _raise_creator_error(code: str, *, field: str | None = None, rule: str | None = None) -> None:
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


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", text)


def _normalize_enum(value: Any, *, aliases: Dict[str, str], allowed: frozenset[str]) -> str:
    text = _normalize_token(value)
    if text in aliases:
        return aliases[text]
    snake = text.replace(" ", "_")
    return snake if snake in allowed else snake


def _coerce_bool_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    return bool(value)


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_creator_raw(
    raw: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
) -> Dict[str, Any]:
    out = dict(raw)
    pid = str(out.get("prototypeId") or "").strip()
    if pid.lower() == prototype_display_name.lower() or _normalize_token(pid) == _normalize_token(prototype_display_name):
        out["prototypeId"] = assigned_prototype_id
    elif not pid:
        out["prototypeId"] = assigned_prototype_id

    out["structureType"] = _normalize_enum(
        out.get("structureType"),
        aliases=_STRUCTURE_TYPE_ALIASES,
        allowed=VALID_STRUCTURE_TYPES,
    )
    out["visualParallelType"] = _normalize_enum(
        out.get("visualParallelType"),
        aliases=_VISUAL_PARALLEL_ALIASES,
        allowed=VALID_VISUAL_PARALLEL_TYPES,
    )

    runway = out.get("runwayFeasibility")
    if isinstance(runway, dict):
        runway = dict(runway)
        runway["continuityRisk"] = _normalize_enum(
            runway.get("continuityRisk"),
            aliases=_CONTINUITY_RISK_ALIASES,
            allowed=VALID_CONTINUITY_RISK,
        )
        runway["generationRisks"] = _normalize_string_list(runway.get("generationRisks"))
        out["runwayFeasibility"] = runway

    silent = out.get("silentVerification")
    if isinstance(silent, dict):
        silent = dict(silent)
        silent["understandableWithoutAudio"] = _coerce_bool_true(silent.get("understandableWithoutAudio"))
        out["silentVerification"] = silent

    report = out.get("creatorReport")
    if isinstance(report, dict):
        report = dict(report)
        gpu = str(report.get("goldPrototypeUsed") or "").strip()
        if gpu.lower() in {assigned_prototype_id.lower(), prototype_display_name.lower()}:
            report["goldPrototypeUsed"] = assigned_prototype_id
        report_vpt = report.get("visualParallelType")
        if report_vpt:
            report["visualParallelType"] = _normalize_enum(
                report_vpt,
                aliases=_VISUAL_PARALLEL_ALIASES,
                allowed=VALID_VISUAL_PARALLEL_TYPES,
            )
        elif out.get("visualParallelType"):
            report["visualParallelType"] = out["visualParallelType"]
        out["creatorReport"] = report

    return out


def _collect_purity_text(candidate: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("conceptSummary", "coreCreativeMechanism", "prototypeMethodApplied"):
        chunks.append(str(candidate.get(key) or ""))
    for section_key in ("sevenSecondStructure", "visualAnchor", "editingPlan", "creatorReport"):
        section = candidate.get(section_key)
        if isinstance(section, dict):
            chunks.extend(str(v) for v in section.values())
    runway = candidate.get("runwayFeasibility")
    if isinstance(runway, dict):
        for key in ("mainSubject", "mainAction", "location", "openingFrame", "whyRunwayShouldUnderstand"):
            chunks.append(str(runway.get(key) or ""))
    silent = candidate.get("silentVerification")
    if isinstance(silent, dict):
        chunks.append(str(silent.get("explanation") or ""))
    return "\n".join(chunks)


def validate_creator_purity(candidate: Dict[str, Any]) -> None:
    blob = _collect_purity_text(candidate).lower()
    for pattern, rule in CREATOR_PURITY_RULES:
        if pattern in blob:
            _raise_creator_error("builder2_creator_purity_violation", rule=rule)


def validate_creator_candidate(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
) -> Dict[str, Any]:
    normalized = normalize_creator_raw(
        candidate,
        assigned_prototype_id=assigned_prototype_id,
        prototype_display_name=prototype_display_name,
    )
    if normalized.get("planningFailure"):
        _raise_creator_error("builder2_creator_validation_failed", field="planningFailure")

    if normalized.get("schemaVersion") != CANDIDATE_SCHEMA_VERSION:
        _raise_creator_error("builder2_creator_schema_invalid", field="schemaVersion")

    try:
        if require_non_empty_str(normalized.get("prototypeId"), field="prototypeId") != assigned_prototype_id:
            _raise_creator_error("builder2_creator_schema_invalid", field="prototypeId")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))

    for field in (
        "prototypeMethodApplied",
        "coreCreativeMechanism",
        "conceptSummary",
        "visualFamily",
    ):
        try:
            require_non_empty_str(normalized.get(field), field=field)
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))

    vpt = require_non_empty_str(normalized.get("visualParallelType"), field="visualParallelType")
    if vpt not in VALID_VISUAL_PARALLEL_TYPES:
        _raise_creator_error("builder2_creator_schema_invalid", field="visualParallelType")

    structure = require_non_empty_str(normalized.get("structureType"), field="structureType")
    if structure not in VALID_STRUCTURE_TYPES:
        _raise_creator_error("builder2_creator_schema_invalid", field="structureType")

    try:
        seven = require_dict(normalized.get("sevenSecondStructure"), field="sevenSecondStructure")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    for key in ("beginning", "development", "resolution"):
        try:
            require_non_empty_str(seven.get(key), field=f"sevenSecondStructure.{key}")
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))

    try:
        anchor = require_dict(normalized.get("visualAnchor"), field="visualAnchor")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    for key in ("description", "whyEssential"):
        try:
            require_non_empty_str(anchor.get(key), field=f"visualAnchor.{key}")
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))

    try:
        silent = require_dict(normalized.get("silentVerification"), field="silentVerification")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    if silent.get("understandableWithoutAudio") is not True:
        _raise_creator_error("builder2_creator_schema_invalid", field="silentVerification.understandableWithoutAudio")
    try:
        require_non_empty_str(silent.get("explanation"), field="silentVerification.explanation")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))

    try:
        runway = require_dict(normalized.get("runwayFeasibility"), field="runwayFeasibility")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    for key in ("mainSubject", "mainAction", "location", "openingFrame", "whyRunwayShouldUnderstand"):
        try:
            require_non_empty_str(runway.get(key), field=f"runwayFeasibility.{key}")
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))
    risk = require_non_empty_str(runway.get("continuityRisk"), field="runwayFeasibility.continuityRisk")
    if risk not in VALID_CONTINUITY_RISK:
        _raise_creator_error("builder2_creator_schema_invalid", field="runwayFeasibility.continuityRisk")
    risks = runway.get("generationRisks")
    if risks is not None and not isinstance(risks, list):
        _raise_creator_error("builder2_creator_schema_invalid", field="runwayFeasibility.generationRisks")
    if isinstance(risks, list):
        runway["generationRisks"] = _normalize_string_list(risks)
    else:
        runway["generationRisks"] = []

    try:
        editing = require_dict(normalized.get("editingPlan"), field="editingPlan")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    for key in ("purpose", "reveal", "pacing"):
        try:
            require_non_empty_str(editing.get(key), field=f"editingPlan.{key}")
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))

    try:
        report = require_dict(normalized.get("creatorReport"), field="creatorReport")
    except Builder2TournamentError as exc:
        _raise_creator_error("builder2_creator_schema_invalid", field=_field_from_error(exc))
    for key in (
        "problemPerception",
        "relativeAdvantage",
        "mechanismScanSummary",
        "goldPrototypeUsed",
        "visualParallelType",
        "whyParallelExpressesAdvantage",
        "whyRunwayShouldUnderstand",
    ):
        try:
            require_non_empty_str(report.get(key), field=f"creatorReport.{key}")
        except Builder2TournamentError as exc:
            _raise_creator_error("builder2_creator_validation_failed", field=_field_from_error(exc))
    report_vpt = require_non_empty_str(report.get("visualParallelType"), field="creatorReport.visualParallelType")
    if report_vpt not in VALID_VISUAL_PARALLEL_TYPES:
        _raise_creator_error("builder2_creator_schema_invalid", field="creatorReport.visualParallelType")
    gpu = require_non_empty_str(report.get("goldPrototypeUsed"), field="creatorReport.goldPrototypeUsed")
    if gpu not in {assigned_prototype_id, prototype_display_name}:
        _raise_creator_error("builder2_creator_validation_failed", field="creatorReport.goldPrototypeUsed")

    forbidden_headline_keys = ("headline", "headlineText", "headlineCoreKeyword", "videoPrompt")
    for key in forbidden_headline_keys:
        if str(normalized.get(key) or "").strip():
            _raise_creator_error("builder2_creator_validation_failed", field=key)

    if assigned_prototype_id == "think_small":
        blob = _collect_purity_text(normalized).lower()
        if "weakness" not in blob and "weak" not in blob:
            _raise_creator_error("builder2_creator_validation_failed", field="think_small.weakness_required")

    if assigned_prototype_id == "greenpeace_essential_pairing":
        report_blob = json.dumps(report, ensure_ascii=False).lower()
        if "appearance" in report_blob and "only" in report_blob:
            _raise_creator_error("builder2_creator_validation_failed", field="essential_pairing.appearance_only")

    if vpt == "context_collision":
        report_blob = json.dumps(report, ensure_ascii=False).lower()
        if "bridge" not in report_blob and "connect" not in report_blob:
            _raise_creator_error("builder2_creator_validation_failed", field="context_collision.bridge_required")

    validate_creator_purity(normalized)
    return normalized


_CREATIVE_RETRY_FIELDS = {
    "think_small.weakness_required",
    "essential_pairing.appearance_only",
    "context_collision.bridge_required",
    "planningFailure",
}


def _is_structural_repairable(code: str, field: Optional[str] = None) -> bool:
    if code == "builder2_creator_schema_invalid":
        return True
    if code == "builder2_creator_validation_failed":
        return field not in _CREATIVE_RETRY_FIELDS
    return False


def _is_clean_retryable(code: str, field: Optional[str]) -> bool:
    if code == "builder2_creator_purity_violation":
        return True
    if code == "builder2_creator_validation_failed" and field in _CREATIVE_RETRY_FIELDS:
        return True
    return False


def _write_creator_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    response_received: bool,
    response_length: int,
    top_level_keys: List[str],
    schema_version_received: Optional[str],
    parse_status: str,
    schema_status: str,
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
            "validationStatus": validation_status,
            "purityStatus": purity_status,
            "failureFieldPaths": failure_field_paths,
            "failureReason": failure_reason,
            "repairAttempted": repair_attempted,
            "cleanRetryAttempted": clean_retry_attempted,
        }
    )
    if response_excerpt and _creator_debug_log_response():
        diagnostics["debugResponseExcerpt"] = response_excerpt


def _invoke_creator_model(
    *,
    prompt: str,
    model: str,
    llm_client: Optional[Any],
    call_type: str,
) -> Tuple[str, Optional[Any]]:
    log_builder2_model_selected(role="builder2_creator", call_type=call_type)
    if llm_client is not None:
        raw = llm_client(role="builder2_creator", model=model, prompt=prompt)
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False), None
        return str(raw or ""), None

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
    return extract_responses_output_text(response), response


def _parse_creator_payload(text: str) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    raw_text = (text or "").strip()
    if not raw_text:
        _raise_creator_error("builder2_creator_empty_response")
    try:
        parsed = parse_json_object(raw_text)
    except ValueError as exc:
        if str(exc) == "empty_response":
            _raise_creator_error("builder2_creator_empty_response")
        _raise_creator_error("builder2_creator_malformed_response")
    keys = sorted(parsed.keys())
    schema_version = parsed.get("schemaVersion")
    return parsed, keys, str(schema_version) if schema_version is not None else None


def _log_creator_failure(
    *,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    exc: Builder2TournamentError,
    repair_attempted: bool,
    clean_retry_attempted: bool,
) -> None:
    code = _failure_code(exc)
    field = _failure_field(exc)
    rule = _failure_rule(exc)
    if code == "builder2_creator_empty_response" or code == "builder2_creator_malformed_response":
        logger.error(
            "BUILDER2_CREATOR_PARSE_FAILED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
            "roundIndex=%s attempt=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
            job_id,
            tournament_id,
            candidate_id,
            prototype_id,
            round_index,
            attempt_number,
            exc.args[0],
            str(repair_attempted).lower(),
            str(clean_retry_attempted).lower(),
        )
    elif code == "builder2_creator_schema_invalid":
        logger.error(
            "BUILDER2_CREATOR_SCHEMA_FAILED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
            "roundIndex=%s attempt=%s field=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
            job_id,
            tournament_id,
            candidate_id,
            prototype_id,
            round_index,
            attempt_number,
            field or "",
            exc.args[0],
            str(repair_attempted).lower(),
            str(clean_retry_attempted).lower(),
        )
    elif code == "builder2_creator_purity_violation":
        logger.error(
            "BUILDER2_CREATOR_PURITY_FAILED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
            "roundIndex=%s attempt=%s rule=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
            job_id,
            tournament_id,
            candidate_id,
            prototype_id,
            round_index,
            attempt_number,
            rule or "",
            exc.args[0],
            str(repair_attempted).lower(),
            str(clean_retry_attempted).lower(),
        )
    else:
        logger.error(
            "BUILDER2_CREATOR_VALIDATION_FAILED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
            "roundIndex=%s attempt=%s field=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
            job_id,
            tournament_id,
            candidate_id,
            prototype_id,
            round_index,
            attempt_number,
            field or "",
            exc.args[0],
            str(repair_attempted).lower(),
            str(clean_retry_attempted).lower(),
        )


def generate_creator_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    runway_mode: str,
    llm_client: Optional[Any] = None,
    state: Optional[Dict[str, Any]] = None,
    candidate_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    prototype = require_prototype(prototype_id)
    candidate_id = candidate_id or f"cand-{round_index}-{prototype_id}-{attempt_number}-{uuid.uuid4().hex[:8]}"
    model = resolve_builder2_creator_model()
    job_id = (state or {}).get("jobId") or ""
    tournament_id = (state or {}).get("tournamentId") or ""
    diagnostics: Dict[str, Any] = {}
    repair_attempted = False
    clean_retry_attempted = False
    retry_rule: Optional[str] = None

    base_prompt = build_creator_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        prototype=prototype,
        candidate_id=candidate_id,
        attempt_number=attempt_number,
        runway_mode=runway_mode,
    )

    logger.info(
        "BUILDER2_CREATOR_START candidateId=%s prototypeId=%s roundIndex=%s attempt=%s",
        candidate_id,
        prototype_id,
        round_index,
        attempt_number,
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
                "BUILDER2_CREATOR_REPAIR_START candidateId=%s prototypeId=%s reason=%s",
                candidate_id,
                prototype_id,
                last_exc.args[0],
            )
            prompt = build_creator_repair_prompt(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=strategy_foundation,
                prototype=prototype,
                candidate_id=candidate_id,
                attempt_number=attempt_number,
                runway_mode=runway_mode,
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
                "BUILDER2_CREATOR_RETRY_START candidateId=%s prototypeId=%s reason=%s",
                candidate_id,
                prototype_id,
                last_exc.args[0],
            )
            prompt = build_creator_retry_prompt(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=strategy_foundation,
                prototype=prototype,
                candidate_id=candidate_id,
                attempt_number=attempt_number,
                runway_mode=runway_mode,
                retry_rule=retry_rule or _failure_field(last_exc) or str(last_exc.args[0]),
            )
            call_type = "retry"
        else:
            prompt = base_prompt
            call_type = "normal"

        timer = MetricsTimer()
        response_text, _response_obj = _invoke_creator_model(
            prompt=prompt,
            model=model,
            llm_client=llm_client,
            call_type=call_type,
        )
        elapsed = timer.elapsed_ms()
        if state is not None:
            record_model_call(
                state,
                role="builder2_creator",
                elapsed_ms=elapsed,
                repair=(call_type == "repair"),
                retry=(call_type == "retry"),
            )

        logger.info(
            "BUILDER2_CREATOR_RESPONSE_RECEIVED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
            "roundIndex=%s attempt=%s model=%s elapsedMs=%.1f responseChars=%s callType=%s",
            job_id,
            tournament_id,
            candidate_id,
            prototype_id,
            round_index,
            attempt_number,
            model,
            elapsed,
            len(response_text or ""),
            call_type,
        )
        if _creator_debug_log_response() and response_text:
            logger.info(
                "BUILDER2_CREATOR_RESPONSE_EXCERPT candidateId=%s excerpt=%s",
                candidate_id,
                _redact_excerpt(response_text),
            )

        try:
            parsed, top_level_keys, schema_version_received = _parse_creator_payload(response_text)
            last_parsed = parsed
            candidate = validate_creator_candidate(
                parsed,
                assigned_prototype_id=prototype_id,
                prototype_display_name=prototype.display_name,
            )
            _write_creator_diagnostics(
                diagnostics,
                response_received=True,
                response_length=len(response_text or ""),
                top_level_keys=top_level_keys,
                schema_version_received=schema_version_received,
                parse_status="ok",
                schema_status="ok",
                validation_status="ok",
                purity_status="ok",
                failure_field_paths=[],
                failure_reason=None,
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
                response_excerpt=_redact_excerpt(response_text),
            )
            if repair_attempted and phase == "repair":
                logger.info("BUILDER2_CREATOR_REPAIR_OK candidateId=%s prototypeId=%s", candidate_id, prototype_id)
            if clean_retry_attempted and phase == "retry":
                logger.info("BUILDER2_CREATOR_RETRY_OK candidateId=%s prototypeId=%s", candidate_id, prototype_id)
            logger.info("BUILDER2_CREATOR_OK candidateId=%s prototypeId=%s", candidate_id, prototype_id)
            if state is not None:
                state.setdefault("creatorDiagnosticsByCandidate", {})[candidate_id] = dict(diagnostics)
            return candidate_id, candidate
        except Builder2TournamentError as exc:
            last_exc = exc
            retry_rule = _failure_rule(exc) or _failure_field(exc)
            _log_creator_failure(
                job_id=job_id,
                tournament_id=tournament_id,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                round_index=round_index,
                attempt_number=attempt_number,
                exc=exc,
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
            )
            code = _failure_code(exc)
            field = _failure_field(exc)
            _write_creator_diagnostics(
                diagnostics,
                response_received=bool(response_text),
                response_length=len(response_text or ""),
                top_level_keys=sorted(last_parsed.keys()) if last_parsed else [],
                schema_version_received=str(last_parsed.get("schemaVersion")) if last_parsed.get("schemaVersion") else None,
                parse_status="failed"
                if code in {"builder2_creator_empty_response", "builder2_creator_malformed_response"}
                else "ok",
                schema_status="failed" if code == "builder2_creator_schema_invalid" else "pending",
                validation_status="failed"
                if code in {"builder2_creator_validation_failed", "builder2_creator_invalid_candidate"}
                else "pending",
                purity_status="failed" if code == "builder2_creator_purity_violation" else "pending",
                failure_field_paths=[field] if field else ([retry_rule] if retry_rule else []),
                failure_reason=str(exc.args[0]),
                repair_attempted=repair_attempted,
                clean_retry_attempted=clean_retry_attempted,
                response_excerpt=_redact_excerpt(response_text) if response_text else None,
            )
            if phase == "repair":
                logger.error(
                    "BUILDER2_CREATOR_REPAIR_FAILED candidateId=%s prototypeId=%s reason=%s",
                    candidate_id,
                    prototype_id,
                    exc.args[0],
                )
            if phase == "retry":
                logger.error(
                    "BUILDER2_CREATOR_RETRY_FAILED candidateId=%s prototypeId=%s reason=%s",
                    candidate_id,
                    prototype_id,
                    exc.args[0],
                )
            if phase == "normal" and _is_structural_repairable(code, field):
                continue
            if (phase in {"normal", "repair"}) and _is_clean_retryable(code, field):
                continue
            break

    assert last_exc is not None
    if state is not None:
        state.setdefault("creatorDiagnosticsByCandidate", {})[candidate_id] = dict(diagnostics)
        record_creator_rejected(state)
    final_reason = str(last_exc.args[0])
    logger.error(
        "BUILDER2_CREATOR_ATTEMPT_INELIGIBLE candidateId=%s prototypeId=%s reason=%s repairAttempted=%s cleanRetryAttempted=%s",
        candidate_id,
        prototype_id,
        final_reason,
        str(repair_attempted).lower(),
        str(clean_retry_attempted).lower(),
    )
    raise Builder2TournamentError(final_reason)
