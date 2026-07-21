"""Builder1 physical-stage evaluation normalization, validation, and repair helpers."""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder1_product_shot_methodology import PHYSICAL_PRODUCT_SHOT_REJECTION_CODES

logger = logging.getLogger(__name__)

PHYSICAL_REJECTION_CODES = frozenset(PHYSICAL_PRODUCT_SHOT_REJECTION_CODES)
PHYSICAL_REJECTION_CODE_LIST: Tuple[str, ...] = tuple(sorted(PHYSICAL_REJECTION_CODES))

_PER_EVALUATION_DERIVATION_CODES = frozenset(
    {
        "physical_conventional_product_shot",
        "physical_collapses_without_product",
        "physical_no_external_object",
        "physical_decorative_presentation_only",
    }
)

_REPAIRABLE_EVALUATION_ERRORS = frozenset(
    {
        "physical_evaluation_ineligible_without_codes",
        "physical_evaluation_contradictory",
    }
)


def _norm_id(value: object) -> str:
    return str(value or "").strip().upper()


def extract_raw_rejection_codes(item: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    for key in ("rejectionCodes", "ineligibilityCodes"):
        raw = item.get(key)
        if not isinstance(raw, list):
            continue
        for code in raw:
            text = str(code or "").strip()
            if text:
                codes.append(text)
    return codes


def _dedupe_valid_codes(codes: List[str]) -> List[str]:
    seen: set[str] = set()
    valid: List[str] = []
    for code in codes:
        if code not in PHYSICAL_REJECTION_CODES:
            continue
        if code in seen:
            continue
        seen.add(code)
        valid.append(code)
    return valid


def derive_rejection_codes_from_evaluation_booleans(item: Dict[str, Any]) -> List[str]:
    """Deterministic one-to-one derivation from explicit evaluation booleans only."""
    if bool(item.get("eligible")):
        return []

    derived: List[str] = []
    if not item.get("clearerThanConventionalProductShot"):
        derived.append("physical_conventional_product_shot")
    if not item.get("survivesProductRemoval"):
        derived.append("physical_collapses_without_product")
    if not item.get("supportsTransferredObject"):
        derived.append("physical_no_external_object")
    if not item.get("distinctiveToBrand"):
        derived.append("physical_decorative_presentation_only")
    return [code for code in derived if code in _PER_EVALUATION_DERIVATION_CODES]


def normalize_physical_evaluation_item(
    item: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    normalized = copy.deepcopy(item)
    actions: List[str] = []

    raw_codes = extract_raw_rejection_codes(normalized)
    unknown = [code for code in raw_codes if code not in PHYSICAL_REJECTION_CODES]
    if unknown:
        actions.append("removed_unknown_codes")
    if "ineligibilityCodes" in normalized and "rejectionCodes" not in item:
        actions.append("mapped_ineligibility_codes")

    valid_codes = _dedupe_valid_codes(raw_codes)
    eligible = bool(normalized.get("eligible"))

    if eligible and valid_codes:
        valid_codes = []
        actions.append("cleared_codes_for_eligible_candidate")

    if not eligible and not valid_codes:
        derived = derive_rejection_codes_from_evaluation_booleans(normalized)
        if derived:
            valid_codes = derived
            actions.append("derived_codes_from_booleans")

    normalized["rejectionCodes"] = valid_codes
    normalized.pop("ineligibilityCodes", None)
    return normalized, actions


def validate_normalized_physical_evaluation_item(
    item: Dict[str, Any],
    *,
    candidate_id: str,
) -> List[str]:
    reasons: List[str] = []
    cid = _norm_id(candidate_id)
    eligible = bool(item.get("eligible"))
    codes = list(item.get("rejectionCodes") or [])

    if eligible and codes:
        reasons.append(f"physical_evaluation_contradictory:{cid}")
    if not eligible and not codes:
        reasons.append(f"physical_evaluation_ineligible_without_codes:{cid}")
        logger.info(
            "BUILDER1_PHYSICAL_CANDIDATE_INVALID candidateId=%s "
            "errorCode=physical_evaluation_ineligible_without_codes normalizationApplied=true",
            cid,
        )
    return reasons


def normalize_physical_evaluations_in_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    obj = copy.deepcopy(payload)
    evaluations_raw = obj.get("physicalEvaluations")
    if not isinstance(evaluations_raw, list):
        return obj, {}

    normalized_items: List[Dict[str, Any]] = []
    action_log: Dict[str, List[str]] = {}
    for item in evaluations_raw:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        cid = _norm_id(item.get("candidateId"))
        normalized_item, actions = normalize_physical_evaluation_item(item)
        normalized_items.append(normalized_item)
        if actions:
            action_log[cid] = actions
            logger.info(
                "BUILDER1_PHYSICAL_EVALUATION_NORMALIZATION candidateId=%s actions=%s "
                "deterministicNormalizationApplied=true",
                cid,
                ",".join(actions),
            )
    obj["physicalEvaluations"] = normalized_items
    return obj, action_log


def extract_repairable_evaluation_error_ids(reasons: List[str]) -> Dict[str, List[str]]:
    invalid: Dict[str, List[str]] = {}
    for reason in reasons:
        prefix = None
        cid = ""
        if reason.startswith("physical_evaluation_ineligible_without_codes:"):
            prefix = "physical_evaluation_ineligible_without_codes"
            cid = reason.split(":", 1)[-1]
        elif reason.startswith("physical_evaluation_contradictory:"):
            prefix = "physical_evaluation_contradictory"
            cid = reason.split(":", 1)[-1]
        if prefix and cid:
            invalid.setdefault(_norm_id(cid), []).append(prefix)
    return invalid


def is_repairable_physical_evaluation_parse_error(reasons: List[str]) -> bool:
    return bool(reasons) and all(
        reason.split(":", 1)[0] in _REPAIRABLE_EVALUATION_ERRORS for reason in reasons
    )


def parse_physical_evaluation_replacements(
    raw_payload: object,
    *,
    allowed_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

    allowed = {_norm_id(cid) for cid in allowed_ids}
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("physical_evaluation_repair", ["physical_evaluation_repair_not_object"]) from exc

    evaluations_raw = obj.get("physicalEvaluations")
    if not isinstance(evaluations_raw, list):
        raise StageParseError("physical_evaluation_repair", ["physical_evaluation_repair_missing_evaluations"])

    replacements: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    for item in evaluations_raw:
        if not isinstance(item, dict):
            raise StageParseError("physical_evaluation_repair", ["physical_evaluation_repair_entry_not_object"])
        cid = _norm_id(item.get("candidateId"))
        if cid not in allowed:
            raise StageParseError(
                "physical_evaluation_repair",
                [f"physical_evaluation_repair_unexpected_id:{cid}"],
            )
        if cid in seen:
            raise StageParseError(
                "physical_evaluation_repair",
                [f"physical_evaluation_repair_duplicate_id:{cid}"],
            )
        seen.add(cid)
        normalized_item, _actions = normalize_physical_evaluation_item(item)
        replacements[cid] = normalized_item

    missing = sorted(allowed - seen)
    if missing:
        raise StageParseError(
            "physical_evaluation_repair",
            [f"physical_evaluation_repair_missing_id:{cid}" for cid in missing],
        )
    return replacements


def merge_physical_evaluation_replacements(
    payload: Dict[str, Any],
    replacements: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    obj = copy.deepcopy(payload)
    merged: List[Dict[str, Any]] = []
    for item in obj.get("physicalEvaluations") or []:
        if not isinstance(item, dict):
            merged.append(item)
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid in replacements:
            merged.append(copy.deepcopy(replacements[cid]))
        else:
            merged.append(copy.deepcopy(item))
    obj["physicalEvaluations"] = merged
    return obj


def parse_brand_physical_with_evaluation_recovery(
    raw_payload: object,
    *,
    model_caller: Any,
    run_stage: Callable[..., Any],
    visibility_policy: Any,
    repair_context: Dict[str, Any],
    product_description: str = "",
    product_name_resolved: str = "",
) -> Any:
    from engine.builder1_final_stages import parse_brand_physical_output
    from engine.builder1_planning_contract import (
        STAGE_PHYSICAL_EVALUATION_REPAIR_SYSTEM,
        build_physical_evaluation_repair_user_prompt,
    )
    from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

    obj = coerce_json_dict(raw_payload)
    obj, _action_log = normalize_physical_evaluations_in_payload(obj)

    try:
        return parse_brand_physical_output(
            obj,
            product_description=product_description,
            product_name_resolved=product_name_resolved,
            visibility_policy=visibility_policy,
        )
    except StageParseError as exc:
        if not is_repairable_physical_evaluation_parse_error(exc.reasons):
            raise
        last_reasons = list(exc.reasons)

    evaluation_items = {
        _norm_id(item.get("candidateId")): item
        for item in (obj.get("physicalEvaluations") or [])
        if isinstance(item, dict)
    }

    for repair_attempt in (1, 2):
        invalid_reasons = extract_repairable_evaluation_error_ids(last_reasons)
        invalid_ids = sorted(invalid_reasons.keys())
        if not invalid_ids:
            raise StageParseError("brand_physical", last_reasons)

        user_prompt = build_physical_evaluation_repair_user_prompt(
            invalid_candidate_ids=invalid_ids,
            invalid_reasons=invalid_reasons,
            evaluation_items=evaluation_items,
            strategic_problem=str(repair_context.get("strategic_problem") or ""),
            relative_advantage=str(repair_context.get("relative_advantage") or ""),
            brand_slogan=str(repair_context.get("brand_slogan") or ""),
            implied_action=str(repair_context.get("implied_action") or ""),
            conceptual=repair_context.get("conceptual") or {},
        )

        def _parse(raw: object):
            return parse_physical_evaluation_replacements(
                raw,
                allowed_ids=invalid_ids,
            )

        logger.info(
            "BUILDER1_PHYSICAL_EVALUATION_REPAIR_START attempt=%s candidateIds=%s reasonCodes=%s",
            repair_attempt,
            ",".join(invalid_ids),
            ",".join(sorted({code for codes in invalid_reasons.values() for code in codes})),
        )
        try:
            replacements = run_stage(
                "physical_evaluation_repair",
                model_caller,
                STAGE_PHYSICAL_EVALUATION_REPAIR_SYSTEM,
                user_prompt,
                _parse,
            )
        except StageParseError as repair_exc:
            logger.info(
                "BUILDER1_PHYSICAL_EVALUATION_REPAIR_FAILED attempt=%s reasonCodes=%s",
                repair_attempt,
                repair_exc.reasons,
            )
            if repair_attempt == 2:
                raise StageParseError("brand_physical", repair_exc.reasons) from repair_exc
            continue

        obj = merge_physical_evaluation_replacements(obj, replacements)
        obj, _action_log = normalize_physical_evaluations_in_payload(obj)
        try:
            result = parse_brand_physical_output(
                obj,
                product_description=product_description,
                product_name_resolved=product_name_resolved,
                visibility_policy=visibility_policy,
            )
            logger.info(
                "BUILDER1_PHYSICAL_EVALUATION_REPAIR_OK attempt=%s candidateIds=%s",
                repair_attempt,
                ",".join(sorted(replacements.keys())),
            )
            return result
        except StageParseError as exc2:
            last_reasons = exc2.reasons
            logger.info(
                "BUILDER1_PHYSICAL_EVALUATION_REPAIR_FAILED attempt=%s reasonCodes=%s",
                repair_attempt,
                exc2.reasons,
            )
            if not is_repairable_physical_evaluation_parse_error(exc2.reasons):
                raise
            if repair_attempt == 2:
                raise

    raise StageParseError("brand_physical", last_reasons or ["physical_evaluation_repair_exhausted"])
