"""
Builder1 single-path strategy/slogan final-result parsing.

Builder1 exposes one final strategic path and one final slogan — not candidate lists.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from engine.builder1_plan_parser import _norm_text
from engine.builder1_slogan_stage import (
    SLOGAN_TRANSFER_RISKS,
    SloganCandidate,
    SloganSelection,
    validate_selected_slogan,
)
from engine.builder1_staged_parsers import (
    StageParseError,
    StrategyCandidate,
    StrategySelection,
    coerce_json_dict,
)
from engine.builder1_strategy_scan import validate_strategy_candidate_item

FINAL_STRATEGY_ID = "FINAL"
FINAL_SLOGAN_ID = "FINAL"

DEFAULT_STRATEGY_SCORES: Dict[str, int] = {
    "truth": 8,
    "briefSupport": 8,
    "advertisingExecutability": 8,
    "noConsultationDependency": 8,
    "noMaterialImplementationCost": 8,
    "relevance": 8,
    "distinctiveness": 8,
    "brandOwnership": 8,
    "persuasiveStrength": 8,
    "seriesPotential": 8,
    "conceptualActionPotential": 8,
}

DEFAULT_SLOGAN_SCORES: Dict[str, int] = {
    "directAdvantageExpression": 8,
    "naturalness": 8,
    "memorability": 8,
    "credibility": 8,
    "brandOwnership": 8,
    "competitorTransferResistance": 8,
    "actionClarity": 8,
    "campaignGenerativePower": 8,
}


def _extract_legacy_selected_item(section: Dict[str, Any], *, id_key: str = "selectedCandidateId") -> Dict[str, Any]:
    candidates = section.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise StageParseError("strategy_slogan_stage", ["legacy_candidate_section_empty"])
    selected_id = _norm_text(section.get(id_key)).upper()
    for item in candidates:
        if isinstance(item, dict) and _norm_text(item.get("id")).upper() == selected_id:
            merged = copy.deepcopy(item)
            merged["selectionReason"] = (
                _norm_text(section.get("selectionReason")) or "Legacy selected path"
            )
            return merged
    first = candidates[0]
    if not isinstance(first, dict):
        raise StageParseError("strategy_slogan_stage", ["legacy_candidate_section_invalid"])
    merged = copy.deepcopy(first)
    merged["selectionReason"] = _norm_text(section.get("selectionReason")) or "Legacy first candidate"
    return merged


def coerce_legacy_strategy_section(strategy_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize stored legacy candidate payloads to a single final strategy object."""
    if isinstance(strategy_raw.get("candidates"), list):
        return _extract_legacy_selected_item(strategy_raw)
    return strategy_raw


def coerce_legacy_slogan_section(slogan_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize stored legacy candidate payloads to a single final slogan object."""
    if isinstance(slogan_raw.get("candidates"), list):
        return _extract_legacy_selected_item(slogan_raw)
    return slogan_raw


def normalize_stored_strategy_slogan_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Read old stored plans containing candidate arrays without crashing."""
    normalized = copy.deepcopy(obj)
    strategy = normalized.get("strategy")
    if isinstance(strategy, dict):
        normalized["strategy"] = coerce_legacy_strategy_section(strategy)
    slogan = normalized.get("slogan")
    if isinstance(slogan, dict):
        normalized["slogan"] = coerce_legacy_slogan_section(slogan)
    return normalized


def _reject_exposed_candidate_fields(section: Dict[str, Any], *, label: str) -> None:
    forbidden = ("candidates", "evaluations", "selectedCandidateId", "candidateReviews")
    for key in forbidden:
        if key in section:
            raise StageParseError(
                "strategy_slogan_stage",
                [f"{label}:forbidden_candidate_field:{key}"],
            )


def parse_strategy_final_section(
    raw_payload: object,
    *,
    product_description: str,
    allow_legacy_candidates: bool = False,
) -> Tuple[StrategySelection, StrategyCandidate]:
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_slogan_stage", ["strategy:not_object"]) from exc

    if allow_legacy_candidates and isinstance(obj.get("candidates"), list):
        obj = coerce_legacy_strategy_section(obj)
    else:
        _reject_exposed_candidate_fields(obj, label="strategy")

    item = copy.deepcopy(obj)
    item["id"] = FINAL_STRATEGY_ID
    result = validate_strategy_candidate_item(
        item,
        candidate_id=FINAL_STRATEGY_ID,
        product_description=product_description,
        exact_sigs=set(),
    )
    if result.candidate is None:
        raise StageParseError(
            "strategy_slogan_stage",
            [f"strategy:{reason}" for reason in result.reasons],
        )

    selection_reason = _norm_text(obj.get("selectionReason")) or "Final campaign strategy"
    selection = StrategySelection(
        selected_candidate_id=FINAL_STRATEGY_ID,
        selection_reason=selection_reason,
        strategy_family=result.candidate.lens,
        scores=dict(DEFAULT_STRATEGY_SCORES),
    )
    return selection, result.candidate


def parse_slogan_final_section(
    raw_payload: object,
    *,
    relative_advantage: str,
    product_description: str,
    detected_language: str,
    allow_legacy_candidates: bool = False,
) -> Tuple[SloganSelection, SloganCandidate]:
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_slogan_stage", ["slogan:not_object"]) from exc

    if allow_legacy_candidates and isinstance(obj.get("candidates"), list):
        obj = coerce_legacy_slogan_section(obj)
    else:
        _reject_exposed_candidate_fields(obj, label="slogan")

    brand_slogan = _norm_text(obj.get("brandSlogan"))
    derivation = _norm_text(obj.get("derivationFromAdvantage"))
    implied_action = _norm_text(obj.get("impliedAction"))
    why_ownable = _norm_text(obj.get("whyOwnable"))
    why_natural = _norm_text(obj.get("whyNaturalInLanguage"))
    transfer_risk = _norm_text(obj.get("competitorTransferRisk")).lower()
    generative = _norm_text(obj.get("campaignGenerativePower"))
    reasons: List[str] = []
    if transfer_risk not in SLOGAN_TRANSFER_RISKS:
        reasons.append("slogan:invalid_structure")
    if not all([brand_slogan, derivation, implied_action, why_ownable, why_natural, generative]):
        reasons.append("slogan:incomplete")
    if reasons:
        raise StageParseError("strategy_slogan_stage", reasons)

    selected = SloganCandidate(
        id=FINAL_SLOGAN_ID,
        brand_slogan=brand_slogan,
        derivation_from_advantage=derivation,
        implied_action=implied_action,
        why_ownable=why_ownable,
        why_natural_in_language=why_natural,
        competitor_transfer_risk=transfer_risk,
        campaign_generative_power=generative,
    )
    rejections = validate_selected_slogan(
        selected,
        relative_advantage=relative_advantage,
        product_description=product_description,
        detected_language=detected_language,
    )
    if rejections:
        raise StageParseError("strategy_slogan_stage", [f"slogan:{code}" for code in rejections])

    selection_reason = _norm_text(obj.get("selectionReason")) or "Final brand slogan"
    selection = SloganSelection(
        selected_candidate_id=FINAL_SLOGAN_ID,
        selection_reason=selection_reason,
        scores=dict(DEFAULT_SLOGAN_SCORES),
    )
    return selection, selected
