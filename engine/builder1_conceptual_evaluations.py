"""Builder1 conceptual-stage evaluation normalization, validation, and repair helpers."""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

from engine.builder1_product_shot_methodology import CONCEPTUAL_PRODUCT_SHOT_REJECTION_CODES

logger = logging.getLogger(__name__)

CONCEPTUAL_REJECTION_CODES = frozenset(
    {
        "concept_not_derived_from_slogan_action",
        "concept_does_not_express_advantage",
        "concept_not_visually_clear",
        "concept_not_series_generative",
        "concept_not_brand_ownable",
        "concept_not_category_relevant",
        "concept_not_image_executable",
        "concept_random_object_first",
        "concept_rewrites_slogan",
        "concept_requires_operational_change",
    }
    | CONCEPTUAL_PRODUCT_SHOT_REJECTION_CODES
)

CONCEPTUAL_REJECTION_CODE_LIST: Tuple[str, ...] = tuple(sorted(CONCEPTUAL_REJECTION_CODES))

_REPAIRABLE_EVALUATION_ERRORS = frozenset(
    {
        "conceptual_stage_evaluation_ineligible_without_codes",
        "conceptual_stage_evaluation_contradictory",
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
        if code not in CONCEPTUAL_REJECTION_CODES:
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
    perception = str(item.get("perceptionToCreate") or "").strip()
    implied_law = str(item.get("impliedPhysicalLaw") or "").strip()

    if not item.get("derivedFromSelectedSloganAction") or not implied_law:
        derived.append("concept_not_derived_from_slogan_action")
    if not item.get("expressesRelativeAdvantage"):
        derived.append("concept_does_not_express_advantage")
    if not item.get("visuallyClear") or not perception:
        derived.append("concept_not_visually_clear")
    if not item.get("seriesGenerative"):
        derived.append("concept_not_series_generative")
    if not item.get("brandOwnable"):
        derived.append("concept_not_brand_ownable")
    if not item.get("categoryRelevant"):
        derived.append("concept_not_category_relevant")
    if not item.get("executableByImageModel"):
        derived.append("concept_not_image_executable")
    if not item.get("avoidsProductShotBias"):
        derived.append("concept_conventional_product_shot")
    if not item.get("distinctiveToBrand"):
        derived.append("concept_not_distinctive")

    if bool(item.get("productEvidenceRequired")):
        if not str(item.get("productEvidenceReason") or "").strip():
            derived.append("concept_collapses_without_product")
    else:
        if not item.get("survivesProductRemoval"):
            derived.append("concept_collapses_without_product")
        if not item.get("supportsTransferredObject"):
            derived.append("concept_no_transferred_object_path")

    return list(dict.fromkeys(derived))


def normalize_conceptual_evaluation_item(
    item: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    normalized = copy.deepcopy(item)
    actions: List[str] = []

    raw_codes = extract_raw_rejection_codes(normalized)
    unknown = [code for code in raw_codes if code not in CONCEPTUAL_REJECTION_CODES]
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


def validate_normalized_conceptual_evaluation_item(
    item: Dict[str, Any],
    *,
    candidate_id: str,
) -> List[str]:
    reasons: List[str] = []
    cid = _norm_id(candidate_id)
    eligible = bool(item.get("eligible"))
    codes = list(item.get("rejectionCodes") or [])

    if eligible and codes:
        reasons.append(f"conceptual_stage_evaluation_contradictory:{cid}")
    if not eligible and not codes:
        reasons.append(f"conceptual_stage_evaluation_ineligible_without_codes:{cid}")
        logger.info(
            "BUILDER1_CONCEPTUAL_CANDIDATE_INVALID candidateId=%s errorCode=conceptual_stage_evaluation_ineligible_without_codes normalizationApplied=true",
            cid,
        )
    return reasons


def normalize_conceptual_evaluations_in_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    obj = copy.deepcopy(payload)
    evaluations_raw = obj.get("evaluations")
    if not isinstance(evaluations_raw, list):
        return obj, {}

    normalized_items: List[Dict[str, Any]] = []
    action_log: Dict[str, List[str]] = {}
    for item in evaluations_raw:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        cid = _norm_id(item.get("candidateId"))
        normalized_item, actions = normalize_conceptual_evaluation_item(item)
        normalized_items.append(normalized_item)
        if actions:
            action_log[cid] = actions
            logger.info(
                "BUILDER1_CONCEPTUAL_NORMALIZATION candidateId=%s actions=%s deterministicNormalizationApplied=true",
                cid,
                ",".join(actions),
            )
    obj["evaluations"] = normalized_items
    return obj, action_log


def extract_repairable_evaluation_error_ids(reasons: List[str]) -> Dict[str, List[str]]:
    invalid: Dict[str, List[str]] = {}
    for reason in reasons:
        prefix = None
        cid = ""
        if reason.startswith("conceptual_stage_evaluation_ineligible_without_codes:"):
            prefix = "conceptual_stage_evaluation_ineligible_without_codes"
            cid = reason.split(":", 1)[-1]
        elif reason.startswith("conceptual_stage_evaluation_contradictory:"):
            prefix = "conceptual_stage_evaluation_contradictory"
            cid = reason.split(":", 1)[-1]
        if prefix and cid:
            invalid.setdefault(_norm_id(cid), []).append(prefix)
    return invalid


def is_repairable_evaluation_parse_error(reasons: List[str]) -> bool:
    return any(
        reason.split(":", 1)[0] in _REPAIRABLE_EVALUATION_ERRORS
        for reason in reasons
    )


def parse_conceptual_evaluation_replacements(
    raw_payload: object,
    *,
    allowed_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

    allowed = {_norm_id(cid) for cid in allowed_ids}
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("conceptual_evaluation_repair", ["conceptual_evaluation_repair_not_object"]) from exc

    evaluations_raw = obj.get("evaluations")
    if not isinstance(evaluations_raw, list):
        raise StageParseError("conceptual_evaluation_repair", ["conceptual_evaluation_repair_missing_evaluations"])

    replacements: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    for item in evaluations_raw:
        if not isinstance(item, dict):
            raise StageParseError("conceptual_evaluation_repair", ["conceptual_evaluation_repair_entry_not_object"])
        cid = _norm_id(item.get("candidateId"))
        if cid not in allowed:
            raise StageParseError(
                "conceptual_evaluation_repair",
                [f"conceptual_evaluation_repair_unexpected_id:{cid}"],
            )
        if cid in seen:
            raise StageParseError(
                "conceptual_evaluation_repair",
                [f"conceptual_evaluation_repair_duplicate_id:{cid}"],
            )
        seen.add(cid)
        normalized_item, _actions = normalize_conceptual_evaluation_item(item)
        replacements[cid] = normalized_item

    missing = sorted(allowed - seen)
    if missing:
        raise StageParseError(
            "conceptual_evaluation_repair",
            [f"conceptual_evaluation_repair_missing_id:{cid}" for cid in missing],
        )
    return replacements


def merge_conceptual_evaluation_replacements(
    payload: Dict[str, Any],
    replacements: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    obj = copy.deepcopy(payload)
    merged: List[Dict[str, Any]] = []
    for item in obj.get("evaluations") or []:
        if not isinstance(item, dict):
            merged.append(item)
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid in replacements:
            merged.append(copy.deepcopy(replacements[cid]))
        else:
            merged.append(copy.deepcopy(item))
    obj["evaluations"] = merged
    return obj
