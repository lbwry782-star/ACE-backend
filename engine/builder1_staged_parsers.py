"""
Builder1 staged planning parsers (intermediate + final assembly).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from engine.builder1_plan_parser import (
    _check_unsupported_claims,
    _norm_key,
    _norm_text,
    _parse_graphic_generator,
    _parse_series_generator,
    _reject_legacy_fields,
    _word_count,
    check_unsupported_evidence,
    validate_series_plan_structure,
)
from engine.builder1_client_boundary import (
    StrategyBoundaryFields,
    filter_eligible_strategy_candidates,
    parse_strategy_boundary_fields,
    strategy_candidate_is_eligible,
    validate_conceptual_boundary_text,
    validate_strategy_candidate_text_boundary,
)
from engine.builder1_plan_spec import (
    BRAND_SLOGAN_MAX_WORDS,
    RELATIVE_ADVANTAGE_SOURCES,
    WEAK_CONCEPTUAL_TERMS,
    Builder1SeriesPlan,
)

SUPPORTED_LANGUAGES = {"he", "en", "ar", "ru", "fr", "de", "es", "it", "pt", "nl"}

CLAIM_RISKS = {"low", "medium", "high"}

STRATEGY_IDS = [f"S{i:02d}" for i in range(1, 13)]
CONCEPTUAL_IDS = [f"C{i:02d}" for i in range(1, 7)]

@dataclass
class StrategyCandidate:
    id: str
    lens: str
    strategic_problem: str
    relative_advantage: str
    brief_support: str
    advantage_source: str
    claim_risk: str
    campaign_executable_now: bool = True
    requires_client_consultation: bool = False
    client_action_level: str = "none"
    implementation_cost_level: str = "none"
    simple_strategic_action: Optional[str] = None


@dataclass
class StrategySelection:
    selected_candidate_id: str
    selection_reason: str
    strategy_family: str
    scores: Dict[str, int]


@dataclass
class ConceptualCandidate:
    id: str
    generator: str
    action: str
    input: str
    transformation: str
    result: str
    why_it_expresses_advantage: str
    series_potential: str


@dataclass
class ConceptualSelection:
    selected_candidate_id: str
    selection_reason: str
    scores: Dict[str, int]


class StageParseError(ValueError):
    def __init__(self, stage: str, reasons: List[str]):
        self.stage = stage
        self.reasons = reasons
        super().__init__(f"{stage}: {';'.join(reasons)}")


def coerce_json_dict(raw_payload: object) -> Dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no_json_object")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("model_output_not_object")
        return obj
    raise ValueError("model_output_not_object")


def detect_brief_language(product_description: str, product_name: str = "") -> str:
    combined = f"{product_name} {product_description}"
    if re.search(r"[\u0590-\u05FF]", combined):
        return "he"
    if re.search(r"[\u0600-\u06FF]", combined):
        return "ar"
    if re.search(r"[\u0400-\u04FF]", combined):
        return "ru"
    return "en"


def parse_strategy_scan(raw_payload: object, *, product_description: str) -> List[StrategyCandidate]:
    from engine.builder1_strategy_scan import parse_strategy_scan as _parse_strategy_scan

    return _parse_strategy_scan(raw_payload, product_description=product_description)


def parse_strategy_selection(
    raw_payload: object,
    candidates: List[StrategyCandidate],
) -> Tuple[StrategySelection, StrategyCandidate]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_selection", ["strategy_selection_not_object"]) from exc

    selected_id = _norm_text(obj.get("selectedCandidateId")).upper()
    by_id = {c.id: c for c in candidates}
    if selected_id not in by_id:
        reasons.append("strategy_selection_invalid_id")
    elif not strategy_candidate_is_eligible(by_id[selected_id]):
        reasons.append("strategy_selection_ineligible_candidate")

    selection_reason = _norm_text(obj.get("selectionReason"))
    strategy_family = _norm_text(obj.get("strategyFamily"))
    scores = obj.get("scores")
    if not selection_reason or not strategy_family:
        reasons.append("strategy_selection_incomplete")
    if not isinstance(scores, dict):
        reasons.append("strategy_selection_missing_scores")

    if reasons:
        raise StageParseError("strategy_selection", reasons)

    selected = by_id[selected_id]
    return (
        StrategySelection(
            selected_candidate_id=selected_id,
            selection_reason=selection_reason,
            strategy_family=strategy_family,
            scores={str(k): int(v) for k, v in scores.items()},
        ),
        selected,
    )


def parse_conceptual_scan(raw_payload: object, *, product_description: str = "") -> List[ConceptualCandidate]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("conceptual_scan", ["conceptual_scan_not_object"]) from exc

    candidates_raw = obj.get("candidates")
    if not isinstance(candidates_raw, list):
        raise StageParseError("conceptual_scan", ["conceptual_scan_candidates_not_list"])
    if any(isinstance(c, str) for c in candidates_raw):
        reasons.append("conceptual_scan_string_candidate")

    parsed: List[ConceptualCandidate] = []
    seen_ids: set[str] = set()

    for item in candidates_raw:
        if not isinstance(item, dict):
            reasons.append("conceptual_scan_candidate_not_object")
            continue
        cid = _norm_text(item.get("id")).upper()
        if cid not in CONCEPTUAL_IDS:
            reasons.append("conceptual_scan_invalid_id")
        if cid in seen_ids:
            reasons.append("conceptual_scan_duplicate_id")
        seen_ids.add(cid)

        generator = _norm_text(item.get("generator"))
        action = _norm_text(item.get("action"))
        inp = _norm_text(item.get("input"))
        transform = _norm_text(item.get("transformation"))
        result = _norm_text(item.get("result"))
        why = _norm_text(item.get("whyItExpressesAdvantage"))
        series_pot = _norm_text(item.get("seriesPotential"))

        if not all([generator, action, inp, transform, result, why, series_pot]):
            reasons.append("conceptual_scan_candidate_incomplete")
        if _norm_key(generator) in WEAK_CONCEPTUAL_TERMS:
            reasons.append("conceptual_scan_candidate_too_vague")

        reasons.extend(
            validate_conceptual_boundary_text(
                generator=generator,
                action=action,
                result=result,
                why=why,
                product_description=product_description,
            )
        )

        parsed.append(
            ConceptualCandidate(
                id=cid,
                generator=generator,
                action=action,
                input=inp,
                transformation=transform,
                result=result,
                why_it_expresses_advantage=why,
                series_potential=series_pot,
            )
        )

    if len(parsed) != 6:
        reasons.append("conceptual_scan_wrong_count")

    if reasons:
        raise StageParseError("conceptual_scan", reasons)
    parsed.sort(key=lambda c: c.id)
    return parsed


def parse_conceptual_selection(
    raw_payload: object,
    candidates: List[ConceptualCandidate],
) -> Tuple[ConceptualSelection, ConceptualCandidate]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("conceptual_selection", ["conceptual_selection_not_object"]) from exc

    selected_id = _norm_text(obj.get("selectedCandidateId")).upper()
    by_id = {c.id: c for c in candidates}
    if selected_id not in by_id:
        reasons.append("conceptual_selection_invalid_id")

    selection_reason = _norm_text(obj.get("selectionReason"))
    scores = obj.get("scores")
    if not selection_reason:
        reasons.append("conceptual_selection_incomplete")
    if not isinstance(scores, dict):
        reasons.append("conceptual_selection_missing_scores")

    if reasons:
        raise StageParseError("conceptual_selection", reasons)

    return (
        ConceptualSelection(
            selected_candidate_id=selected_id,
            selection_reason=selection_reason,
            scores={str(k): int(v) for k, v in scores.items()},
        ),
        by_id[selected_id],
    )


def parse_final_campaign_output(
    raw_payload: object,
    *,
    expected_ad_count: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """Parse stage-5 creative output only (no scans, no request-controlled fields)."""
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        return {}, ["final_campaign_not_object"]

    for forbidden in (
        "strategyCandidateScan",
        "conceptualGeneratorScan",
        "format",
        "adCount",
        "detectedLanguage",
        "strategicProblem",
        "relativeAdvantage",
        "conceptualGenerator",
    ):
        if forbidden in obj:
            reasons.append(f"final_campaign_forbidden_field:{forbidden}")

    _reject_legacy_fields(obj, reasons)

    required = [
        "productNameResolved",
        "brandSlogan",
        "sloganDerivation",
        "sloganAction",
        "physicalGenerator",
        "physicalGeneratorNaturalPurpose",
        "physicalGeneratorCampaignRole",
        "campaignRationale",
    ]
    for field in required:
        if not _norm_text(obj.get(field)):
            reasons.append(f"missing_{field}")

    brand_slogan = _norm_text(obj.get("brandSlogan"))
    if brand_slogan and _word_count(brand_slogan) > BRAND_SLOGAN_MAX_WORDS:
        reasons.append("brand_slogan_too_long")

    medium_participates = obj.get("mediumParticipates")
    if not isinstance(medium_participates, bool):
        reasons.append("medium_participates_not_boolean")
    medium_role = _norm_text(obj.get("mediumRole"))
    if medium_participates and not medium_role:
        reasons.append("medium_role_required_when_participates")
    if not medium_participates and medium_role:
        reasons.append("medium_role_forbidden_when_not_participates")

    graphic = _parse_graphic_generator(obj.get("graphicGenerator"), reasons)
    series_gen = _parse_series_generator(obj.get("seriesGenerator"), reasons)

    ads_raw = obj.get("ads")
    if not isinstance(ads_raw, list) or len(ads_raw) != expected_ad_count:
        reasons.append("ads_length_mismatch")

    if reasons:
        return obj, reasons

    return obj, []


def assemble_builder1_series_plan(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    detected_language: str,
    exploration_seed: str,
    strategy: StrategyCandidate,
    strategy_selection: StrategySelection,
    conceptual: ConceptualCandidate,
    final_creative: Dict[str, Any],
) -> Builder1SeriesPlan:
    """Merge server-authoritative fields with selected strategy/concept and final creative."""
    evidence = strategy.brief_support
    if check_unsupported_evidence(evidence, product_description):
        raise StageParseError("assemble", ["unsupported_evidence_claim"])

    assembled: Dict[str, Any] = {
        "productName": product_name,
        "productDescription": product_description,
        "format": format_value,
        "adCount": ad_count,
        "detectedLanguage": detected_language,
        "productNameResolved": _norm_text(final_creative.get("productNameResolved")) or product_name or "Product",
        "strategicProblem": strategy.strategic_problem,
        "strategicProblemEvidence": evidence,
        "relativeAdvantage": strategy.relative_advantage,
        "relativeAdvantageSource": strategy.advantage_source,
        "relativeAdvantageBriefSupport": strategy.brief_support,
        "relativeAdvantageClaimRisk": strategy.claim_risk,
        "problemAdvantageLink": f"{strategy.relative_advantage} addresses {strategy.strategic_problem}",
        "brandSlogan": final_creative.get("brandSlogan"),
        "sloganDerivation": final_creative.get("sloganDerivation"),
        "sloganAction": final_creative.get("sloganAction"),
        "conceptualGenerator": conceptual.generator,
        "conceptualGeneratorAction": conceptual.action,
        "conceptualGeneratorInput": conceptual.input,
        "conceptualGeneratorTransformation": conceptual.transformation,
        "conceptualGeneratorResult": conceptual.result,
        "conceptualGeneratorWhyItExpressesAdvantage": conceptual.why_it_expresses_advantage,
        "physicalGenerator": final_creative.get("physicalGenerator"),
        "physicalGeneratorNaturalPurpose": final_creative.get("physicalGeneratorNaturalPurpose"),
        "physicalGeneratorCampaignRole": final_creative.get("physicalGeneratorCampaignRole"),
        "graphicGenerator": final_creative.get("graphicGenerator"),
        "seriesGenerator": final_creative.get("seriesGenerator"),
        "mediumParticipates": final_creative.get("mediumParticipates", False),
        "mediumRole": final_creative.get("mediumRole", ""),
        "campaignRationale": final_creative.get("campaignRationale"),
        "ads": final_creative.get("ads"),
    }

    plan, reasons = validate_series_plan_structure(
        assembled,
        expected_format=format_value,
        expected_ad_count=ad_count,
        product_name=product_name,
        product_description=product_description,
        require_internal_scans=False,
    )
    if plan is None:
        raise StageParseError("assemble", reasons)
    return plan
