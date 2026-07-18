"""
Builder1 slogan scan, selection, and quality gate — fixed before conceptual generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from engine.builder1_client_boundary import validate_brand_physical_boundary_text
from engine.builder1_plan_parser import _norm_text, _word_count
from engine.builder1_plan_spec import BRAND_SLOGAN_MAX_WORDS
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

SLOGAN_IDS = [f"L{i:02d}" for i in range(1, 7)]
SLOGAN_TRANSFER_RISKS = {"low", "medium", "high"}

SLOGAN_REJECTION_CODES = frozenset(
    {
        "slogan_not_derived_from_advantage",
        "slogan_generic",
        "slogan_descriptive_only",
        "slogan_not_ownable",
        "slogan_not_credible",
        "slogan_no_implied_action",
        "slogan_not_campaign_generative",
        "slogan_requires_future_capability",
    }
)

_GENERIC_SLOGAN_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bquality without compromise\b", "slogan_generic"),
    (r"\bthe smart choice\b", "slogan_generic"),
    (r"\bthinking differently\b", "slogan_generic"),
    (r"\bthe best service\b", "slogan_generic"),
    (r"\bmoving forward together\b", "slogan_generic"),
    (r"\bmade for you\b", "slogan_generic"),
    (r"\ba world of possibilities\b", "slogan_generic"),
    (r"\bbuilt for success\b", "slogan_generic"),
    (r"\byour trusted partner\b", "slogan_generic"),
    (r"\bexcellence in everything\b", "slogan_generic"),
    (r"\bwe deliver results\b", "slogan_generic"),
    (r"\bthe future is now\b", "slogan_generic"),
)

_DESCRIPTIVE_ONLY_PATTERNS: Tuple[str, ...] = (
    r"^premium [a-z ]+$",
    r"^quality [a-z ]+$",
    r"^professional [a-z ]+$",
    r"^advanced [a-z ]+$",
    r"^innovative [a-z ]+$",
)


@dataclass
class SloganCandidate:
    id: str
    brand_slogan: str
    derivation_from_advantage: str
    implied_action: str
    why_ownable: str
    why_natural_in_language: str
    competitor_transfer_risk: str
    campaign_generative_power: str


@dataclass
class SloganSelection:
    selected_candidate_id: str
    selection_reason: str
    scores: Dict[str, int]


def is_slogan_rejection(codes: List[str]) -> bool:
    return any(code in SLOGAN_REJECTION_CODES for code in codes)


def _scan_generic_slogan(slogan: str) -> Optional[str]:
    lowered = slogan.strip().lower()
    for pattern, code in _GENERIC_SLOGAN_PATTERNS:
        if re.search(pattern, lowered, re.I):
            return code
    if len(lowered.split()) <= 2 and re.search(
        r"\b(quality|service|trust|best|smart|future|together|premium|excellence)\b", lowered
    ):
        return "slogan_generic"
    for pattern in _DESCRIPTIVE_ONLY_PATTERNS:
        if re.match(pattern, lowered):
            return "slogan_descriptive_only"
    return None


def parse_slogan_scan(raw_payload: object) -> List[SloganCandidate]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_scan", ["slogan_scan_not_object"]) from exc

    candidates_raw = obj.get("candidates")
    if not isinstance(candidates_raw, list):
        raise StageParseError("slogan_scan", ["slogan_scan_candidates_not_list"])
    if any(isinstance(c, str) for c in candidates_raw):
        reasons.append("slogan_scan_string_candidate")

    parsed: List[SloganCandidate] = []
    seen: set[str] = set()
    for item in candidates_raw:
        if not isinstance(item, dict):
            reasons.append("slogan_scan_candidate_not_object")
            continue
        cid = _norm_text(item.get("id")).upper()
        if cid not in SLOGAN_IDS:
            reasons.append("slogan_scan_invalid_id")
        if cid in seen:
            reasons.append("slogan_scan_duplicate_id")
        seen.add(cid)

        brand_slogan = _norm_text(item.get("brandSlogan"))
        derivation = _norm_text(item.get("derivationFromAdvantage"))
        implied_action = _norm_text(item.get("impliedAction"))
        why_ownable = _norm_text(item.get("whyOwnable"))
        why_natural = _norm_text(item.get("whyNaturalInLanguage"))
        transfer_risk = _norm_text(item.get("competitorTransferRisk")).lower()
        generative = _norm_text(item.get("campaignGenerativePower"))

        if not all([brand_slogan, derivation, implied_action, why_ownable, why_natural, generative]):
            reasons.append("slogan_scan_candidate_incomplete")
        if transfer_risk not in SLOGAN_TRANSFER_RISKS:
            reasons.append("slogan_scan_invalid_transfer_risk")
        if brand_slogan and _word_count(brand_slogan) > BRAND_SLOGAN_MAX_WORDS:
            reasons.append("slogan_scan_slogan_too_long")

        generic_code = _scan_generic_slogan(brand_slogan) if brand_slogan else None
        if generic_code:
            reasons.append(generic_code)

        parsed.append(
            SloganCandidate(
                id=cid,
                brand_slogan=brand_slogan,
                derivation_from_advantage=derivation,
                implied_action=implied_action,
                why_ownable=why_ownable,
                why_natural_in_language=why_natural,
                competitor_transfer_risk=transfer_risk,
                campaign_generative_power=generative,
            )
        )

    if len(parsed) != 6:
        reasons.append("slogan_scan_wrong_count")

    if reasons:
        raise StageParseError("slogan_scan", reasons)
    parsed.sort(key=lambda c: c.id)
    return parsed


def parse_slogan_selection(
    raw_payload: object,
    candidates: List[SloganCandidate],
) -> Tuple[SloganSelection, SloganCandidate]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_selection", ["slogan_selection_not_object"]) from exc

    selected_id = _norm_text(obj.get("selectedCandidateId")).upper()
    by_id = {c.id: c for c in candidates}
    if selected_id not in by_id:
        reasons.append("slogan_selection_invalid_id")

    selection_reason = _norm_text(obj.get("selectionReason"))
    scores = obj.get("scores")
    if not selection_reason:
        reasons.append("slogan_selection_incomplete")
    if not isinstance(scores, dict):
        reasons.append("slogan_selection_missing_scores")

    if reasons:
        raise StageParseError("slogan_selection", reasons)

    return (
        SloganSelection(
            selected_candidate_id=selected_id,
            selection_reason=selection_reason,
            scores={str(k): int(v) for k, v in scores.items()},
        ),
        by_id[selected_id],
    )


def validate_selected_slogan(
    slogan: SloganCandidate,
    *,
    relative_advantage: str,
    product_description: str = "",
) -> List[str]:
    """Mandatory quality gate before conceptual_scan."""
    reasons: List[str] = []
    advantage_tokens = {
        token
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", relative_advantage.lower())
        if len(token) >= 4
    }
    slogan_tokens = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", slogan.brand_slogan.lower()))
    derivation_tokens = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", slogan.derivation_from_advantage.lower()))
    if advantage_tokens and not (advantage_tokens & (slogan_tokens | derivation_tokens)):
        reasons.append("slogan_not_derived_from_advantage")

    generic = _scan_generic_slogan(slogan.brand_slogan)
    if generic:
        reasons.append(generic)

    if slogan.competitor_transfer_risk == "high":
        reasons.append("slogan_not_ownable")

    if len(slogan.implied_action.split()) < 2:
        reasons.append("slogan_no_implied_action")

    if len(slogan.campaign_generative_power.split()) < 4:
        reasons.append("slogan_not_campaign_generative")

    reasons.extend(
        validate_brand_physical_boundary_text(
            brand_slogan=slogan.brand_slogan,
            slogan_action=slogan.implied_action,
            campaign_rationale=slogan.derivation_from_advantage,
            physical_generator_campaign_role="",
            product_description=product_description,
        )
    )
    return list(dict.fromkeys(reasons))
