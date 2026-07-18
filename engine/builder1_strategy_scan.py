"""
Builder1 strategy_scan candidate validation and focused replacement repair.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder1_client_boundary import (
    StrategyBoundaryFields,
    normalize_simple_strategic_action_value,
    parse_strategy_boundary_fields,
    validate_strategy_candidate_text_boundary,
)
from engine.builder1_plan_parser import (
    _check_unsupported_claims,
    _norm_key,
    _norm_text,
    check_unsupported_evidence,
)
from engine.builder1_plan_spec import RELATIVE_ADVANTAGE_SOURCES
from engine.builder1_staged_parsers import (
    STRATEGY_IDS,
    StageParseError,
    StrategyCandidate,
    coerce_json_dict,
    filter_eligible_strategy_candidates,
)

from engine.builder1_planning_contract import STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM

logger = logging.getLogger(__name__)

StrategyScanModelCaller = Callable[..., object]

CLAIM_RISKS = {"low", "medium", "high"}

STRATEGY_SCAN_REPLACEMENT_SYSTEM = STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM

_GLOBAL_SCAN_ERRORS = frozenset(
    {
        "strategy_scan_not_object",
        "strategy_scan_candidates_not_list",
        "strategy_scan_string_candidate",
        "strategy_scan_wrong_count",
        "strategy_scan_missing_ids",
        "strategy_scan_duplicate_id",
        "strategy_scan_invalid_id",
    }
)


def is_global_strategy_scan_failure(reasons: List[str]) -> bool:
    return any(reason in _GLOBAL_SCAN_ERRORS for reason in reasons)


def _candidate_grounding(item: Dict[str, Any]) -> str:
    for key in ("briefSupport", "briefGrounding"):
        value = _norm_text(item.get(key))
        if value:
            return value
    return ""


def _log_candidate_invalid(candidate_id: str, reason: str) -> None:
    field = "simpleStrategicAction"
    if "_briefSupport_" in reason or reason.endswith("_unsupported_grounding_claim"):
        field = "briefSupport"
    elif "_strategicProblem_" in reason:
        field = "strategicProblem"
    elif "_relativeAdvantage_" in reason:
        field = "relativeAdvantage"
    code = reason.split(f"strategy_scan_{candidate_id}_", 1)[-1] if f"strategy_scan_{candidate_id}_" in reason else reason
    logger.error(
        "BUILDER1_STRATEGY_CANDIDATE_INVALID candidateId=%s field=%s code=%s",
        candidate_id,
        field,
        code,
    )


def parse_strategy_scan_structure(raw_payload: object) -> List[Dict[str, Any]]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_scan", ["strategy_scan_not_object"]) from exc

    candidates_raw = obj.get("candidates")
    if not isinstance(candidates_raw, list):
        raise StageParseError("strategy_scan", ["strategy_scan_candidates_not_list"])

    if any(isinstance(c, str) for c in candidates_raw):
        reasons.append("strategy_scan_string_candidate")

    if len(candidates_raw) != 12:
        reasons.append("strategy_scan_wrong_count")

    normalized: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in candidates_raw:
        if not isinstance(item, dict):
            reasons.append("strategy_scan_candidate_not_object")
            continue
        cid = _norm_text(item.get("id")).upper()
        if cid not in STRATEGY_IDS:
            reasons.append("strategy_scan_invalid_id")
        if cid in seen_ids:
            reasons.append("strategy_scan_duplicate_id")
        seen_ids.add(cid)
        normalized.append(item)

    if len(seen_ids) != 12:
        reasons.append("strategy_scan_missing_ids")

    if reasons:
        raise StageParseError("strategy_scan", list(dict.fromkeys(reasons)))

    normalized.sort(key=lambda item: _norm_text(item.get("id")).upper())
    return normalized


@dataclass
class StrategyCandidateValidation:
    candidate: Optional[StrategyCandidate]
    raw_item: Dict[str, Any]
    reasons: List[str]


def validate_strategy_candidate_item(
    item: Dict[str, Any],
    *,
    candidate_id: str,
    product_description: str,
    exact_sigs: set[str],
) -> StrategyCandidateValidation:
    prefix = f"strategy_scan_{candidate_id}"
    reasons: List[str] = []
    raw_item = copy.deepcopy(item)

    lens = _norm_text(item.get("lens"))
    problem = _norm_text(item.get("strategicProblem"))
    advantage = _norm_text(item.get("relativeAdvantage"))
    support = _candidate_grounding(item)
    source = _norm_text(item.get("advantageSource"))
    risk = _norm_text(item.get("claimRisk")).lower()

    if not all([lens, problem, advantage, support, source, risk]):
        reasons.append(f"{prefix}_candidate_incomplete")

    if source not in RELATIVE_ADVANTAGE_SOURCES:
        reasons.append(f"{prefix}_invalid_advantage_source")
    if risk not in CLAIM_RISKS:
        reasons.append(f"{prefix}_invalid_claim_risk")

    sig = f"{_norm_key(problem)}|{_norm_key(advantage)}"
    if sig in exact_sigs and problem and advantage:
        reasons.append(f"{prefix}_exact_duplicate")
    if problem and advantage:
        exact_sigs.add(sig)

    if check_unsupported_evidence(problem, product_description):
        reasons.append(f"{prefix}_unsupported_grounding_claim")
    if check_unsupported_evidence(support, product_description):
        reasons.append(f"{prefix}_unsupported_grounding_claim")

    per_claim_reasons: List[str] = []
    _check_unsupported_claims(
        product_description=product_description,
        relative_advantage=advantage,
        relative_advantage_source=source,
        reasons=per_claim_reasons,
    )
    for _ in per_claim_reasons:
        reasons.append(f"{prefix}_unsupported_grounding_claim")

    boundary_item = dict(item)
    boundary_item["simpleStrategicAction"] = normalize_simple_strategic_action_value(
        item.get("simpleStrategicAction")
    )

    boundary, boundary_reasons = parse_strategy_boundary_fields(boundary_item, candidate_id=candidate_id)
    for reason in boundary_reasons:
        if reason.endswith("_simpleStrategicAction_without_optional_action"):
            reasons.append(f"{prefix}_invalid_simple_strategic_action")
        else:
            reasons.append(reason)

    if boundary is None:
        boundary = StrategyBoundaryFields(
            campaign_executable_now=False,
            requires_client_consultation=True,
            client_action_level="complex_required",
            implementation_cost_level="material",
            simple_strategic_action=None,
        )

    reasons.extend(
        validate_strategy_candidate_text_boundary(
            candidate_id=candidate_id,
            strategic_problem=problem,
            relative_advantage=advantage,
            brief_support=support,
            simple_strategic_action=boundary.simple_strategic_action,
            product_description=product_description,
        )
    )
    reasons = list(dict.fromkeys(reasons))

    if reasons:
        for reason in reasons:
            _log_candidate_invalid(candidate_id, reason)
        return StrategyCandidateValidation(candidate=None, raw_item=raw_item, reasons=reasons)

    candidate = StrategyCandidate(
        id=candidate_id,
        lens=lens,
        strategic_problem=problem,
        relative_advantage=advantage,
        brief_support=support,
        advantage_source=source,
        claim_risk=risk,
        campaign_executable_now=boundary.campaign_executable_now,
        requires_client_consultation=boundary.requires_client_consultation,
        client_action_level=boundary.client_action_level,
        implementation_cost_level=boundary.implementation_cost_level,
        simple_strategic_action=boundary.simple_strategic_action,
    )
    return StrategyCandidateValidation(candidate=candidate, raw_item=raw_item, reasons=[])


def validate_strategy_scan_set(
    candidates_raw: List[Dict[str, Any]],
    *,
    product_description: str,
) -> Tuple[Dict[str, List[str]], Dict[str, StrategyCandidate], Dict[str, Dict[str, Any]]]:
    invalid: Dict[str, List[str]] = {}
    valid_candidates: Dict[str, StrategyCandidate] = {}
    valid_raw: Dict[str, Dict[str, Any]] = {}
    exact_sigs: set[str] = set()
    lenses: set[str] = set()

    for item in candidates_raw:
        candidate_id = _norm_text(item.get("id")).upper()
        result = validate_strategy_candidate_item(
            item,
            candidate_id=candidate_id,
            product_description=product_description,
            exact_sigs=exact_sigs,
        )
        if result.reasons:
            invalid[candidate_id] = result.reasons
        else:
            assert result.candidate is not None
            valid_candidates[candidate_id] = result.candidate
            valid_raw[candidate_id] = result.raw_item
            lenses.add(_norm_key(result.candidate.lens))

    global_reasons: List[str] = []
    if len(valid_candidates) + len(invalid) == 12 and len(lenses) < 5:
        global_reasons.append("strategy_scan_insufficient_family_diversity")
    if valid_candidates and not filter_eligible_strategy_candidates(valid_candidates.values()):
        global_reasons.append("strategy_scan_no_eligible_candidates")

    if global_reasons and not invalid:
        for cid in STRATEGY_IDS:
            invalid[cid] = list(global_reasons)

    return invalid, valid_candidates, valid_raw


def build_strategy_scan_replacement_user_prompt(
    *,
    product_name: str,
    product_description: str,
    invalid_candidates: Dict[str, List[str]],
    candidate_items: Dict[str, Dict[str, Any]],
) -> str:
    requested = []
    for cid in sorted(invalid_candidates):
        requested.append(
            {
                "id": cid,
                "lens": candidate_items[cid].get("lens"),
                "rejectionReasons": invalid_candidates[cid],
            }
        )
    return (
        f"Product name: {product_name}\n"
        f"Brief:\n{product_description.strip()}\n\n"
        "Replace ONLY these invalid candidates:\n"
        f"{json.dumps(requested, ensure_ascii=False, indent=2)}\n\n"
        "Use only the submitted brief or a clearly labeled general category inference.\n"
        "Do not invent percentages, studies, surveys, dates, interview counts, reports, or factual capabilities.\n"
        "Return replacements with exactly the requested ids."
    )


def parse_strategy_candidate_replacements(raw_payload: object, *, allowed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    obj = coerce_json_dict(raw_payload)
    replacements_raw = obj.get("replacements")
    if not isinstance(replacements_raw, list):
        raise StageParseError("strategy_candidate_repair", ["strategy_candidate_repair_missing_replacements"])

    allowed = {cid.upper() for cid in allowed_ids}
    parsed: Dict[str, Dict[str, Any]] = {}
    for item in replacements_raw:
        if not isinstance(item, dict):
            raise StageParseError("strategy_candidate_repair", ["strategy_candidate_repair_entry_not_object"])
        cid = _norm_text(item.get("id")).upper()
        if cid not in allowed:
            raise StageParseError(
                "strategy_candidate_repair",
                [f"strategy_candidate_repair_unexpected_id:{cid}"],
            )
        parsed[cid] = item

    for cid in allowed_ids:
        upper = cid.upper()
        if upper not in parsed:
            raise StageParseError("strategy_candidate_repair", [f"strategy_candidate_repair_missing_id:{upper}"])
    return parsed


parse_strategy_scan_replacements = parse_strategy_candidate_replacements


def merge_strategy_scan_replacements(
    candidates_raw: List[Dict[str, Any]],
    replacements: Dict[str, Dict[str, Any]],
    *,
    valid_raw: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for item in candidates_raw:
        cid = _norm_text(item.get("id")).upper()
        if cid in valid_raw:
            merged.append(copy.deepcopy(valid_raw[cid]))
        elif cid in replacements:
            merged.append(copy.deepcopy(replacements[cid]))
        else:
            merged.append(copy.deepcopy(item))
    merged.sort(key=lambda row: _norm_text(row.get("id")).upper())
    return merged


def finalize_strategy_scan_candidates(valid_candidates: Dict[str, StrategyCandidate]) -> List[StrategyCandidate]:
    if len(valid_candidates) != 12:
        raise StageParseError("strategy_scan", ["strategy_scan_wrong_count"])
    if not filter_eligible_strategy_candidates(valid_candidates.values()):
        raise StageParseError("strategy_scan", ["strategy_scan_no_eligible_candidates"])
    return [valid_candidates[cid] for cid in STRATEGY_IDS]


def _run_focused_strategy_candidate_repair(
    *,
    model_caller: StrategyScanModelCaller,
    product_name: str,
    product_description: str,
    invalid: Dict[str, List[str]],
    candidate_items: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    invalid_ids = sorted(invalid.keys())
    user_prompt = build_strategy_scan_replacement_user_prompt(
        product_name=product_name,
        product_description=product_description,
        invalid_candidates=invalid,
        candidate_items=candidate_items,
    )
    last_reasons: List[str] = []
    for parse_attempt in (1, 2):
        replacement_raw = model_caller(
            STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM,
            user_prompt,
            stage="strategy_candidate_repair",
        )
        try:
            return parse_strategy_candidate_replacements(
                replacement_raw,
                allowed_ids=invalid_ids,
            )
        except StageParseError as exc:
            last_reasons = exc.reasons
            logger.warning(
                "BUILDER1_STRATEGY_CANDIDATE_REPAIR_PARSE_RETRY attempt=%s reasons=%s",
                parse_attempt,
                last_reasons,
            )
    raise StageParseError("strategy_candidate_repair", last_reasons or ["strategy_candidate_repair_failed"])


def ensure_strategy_scan_from_raw(
    raw_payload: object,
    *,
    product_name: str,
    product_description: str,
    model_caller: StrategyScanModelCaller,
) -> List[StrategyCandidate]:
    candidates_raw = parse_strategy_scan_structure(raw_payload)

    for repair_attempt in (1, 2):
        invalid, valid_candidates, valid_raw = validate_strategy_scan_set(
            candidates_raw,
            product_description=product_description,
        )
        if not invalid:
            logger.info("BUILDER1_STRATEGY_SCAN_OK candidates=12")
            return finalize_strategy_scan_candidates(valid_candidates)

        logger.info(
            "BUILDER1_STRATEGY_SCAN_REPAIR invalidCount=%s attempt=%s ids=%s",
            len(invalid),
            repair_attempt,
            ",".join(sorted(invalid)),
        )

        candidate_items = {_norm_text(item.get("id")).upper(): item for item in candidates_raw}
        try:
            replacements = _run_focused_strategy_candidate_repair(
                model_caller=model_caller,
                product_name=product_name,
                product_description=product_description,
                invalid=invalid,
                candidate_items=candidate_items,
            )
        except StageParseError:
            if repair_attempt == 2:
                raise
            continue

        candidates_raw = merge_strategy_scan_replacements(
            candidates_raw,
            replacements,
            valid_raw=valid_raw,
        )

    invalid, _valid, _valid_raw = validate_strategy_scan_set(
        candidates_raw,
        product_description=product_description,
    )
    if invalid:
        raise StageParseError(
            "strategy_candidate_repair",
            ["strategy_candidate_repair_exhausted"],
        )
    return finalize_strategy_scan_candidates(_valid)


def parse_strategy_scan(raw_payload: object, *, product_description: str) -> List[StrategyCandidate]:
    """Parse and fully validate a complete 12-candidate strategy scan."""
    candidates_raw = parse_strategy_scan_structure(raw_payload)
    invalid, valid_candidates, _valid_raw = validate_strategy_scan_set(
        candidates_raw,
        product_description=product_description,
    )
    if invalid:
        all_reasons = [reason for reasons in invalid.values() for reason in reasons]
        raise StageParseError("strategy_scan", all_reasons)
    return finalize_strategy_scan_candidates(valid_candidates)
