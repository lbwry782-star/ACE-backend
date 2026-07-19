"""Builder1 consolidated same-layer planning stages (strategy, slogan, conceptual)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.builder1_client_boundary import strategy_candidate_is_eligible
from engine.builder1_planning_contract import (
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_SLOGAN_STAGE_SYSTEM,
    STAGE_STRATEGY_STAGE_SYSTEM,
    build_conceptual_scan_repair_prompt,
    build_conceptual_stage_user_prompt,
    build_slogan_stage_user_prompt,
    build_strategy_stage_user_prompt,
)
from engine.builder1_slogan_stage import (
    SloganCandidate,
    SloganSelection,
)
from engine.builder1_slogan_stage_parser import parse_consolidated_slogan_stage_response
from engine.builder1_product_shot_methodology import CONCEPTUAL_PRODUCT_SHOT_REJECTION_CODES
from engine.builder1_staged_parsers import (
    CONCEPTUAL_IDS,
    STRATEGY_IDS,
    ConceptualCandidate,
    StageParseError,
    StrategyCandidate,
    StrategyCandidateReview,
    StrategySelection,
    coerce_json_dict,
    parse_conceptual_scan,
)
from engine.builder1_strategy_scan import ensure_strategy_scan_from_raw
from engine.builder1_strategy_selection import (
    StrategySelectionExhausted,
    validate_selected_strategy_gate,
)

logger = logging.getLogger(__name__)

PlanningModelCaller = Callable[..., object]
RunStageFn = Callable[..., Any]

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


@dataclass
class ConceptualCandidateReview:
    candidate_id: str
    perception_to_create: str
    implied_physical_law: str
    derived_from_selected_slogan_action: bool
    expresses_relative_advantage: bool
    visually_clear: bool
    series_generative: bool
    brand_ownable: bool
    category_relevant: bool
    executable_by_image_model: bool
    survives_product_removal: bool
    avoids_product_shot_bias: bool
    supports_transferred_object: bool
    distinctive_to_brand: bool
    product_evidence_required: bool
    product_evidence_reason: str
    eligible: bool
    rejection_codes: List[str]


@dataclass
class ConceptualSelection:
    selected_candidate_id: str
    selection_reason: str


@dataclass
class Builder1UpstreamSnapshot:
    product_name_resolved: str
    strategic_problem: str
    relative_advantage: str
    brand_slogan: str
    implied_action: str
    selected_slogan_id: str
    conceptual_generator: str
    selected_conceptual_id: str
    physical_generator: str
    graphic_layout_template: str
    graphic_recurring_device: str


def build_conceptual_lineage(
    *,
    selected_slogan: Any,
    selected_conceptual: Any,
) -> Dict[str, str]:
    """Server-owned lineage metadata for structural final integrity checks."""
    return {
        "selectedConceptCandidateId": str(getattr(selected_conceptual, "id", "") or "").strip().upper(),
        "sourceSloganCandidateId": str(getattr(selected_slogan, "id", "") or "").strip().upper(),
        "fixedBrandSlogan": str(getattr(selected_slogan, "brand_slogan", "") or ""),
        "fixedImpliedAction": str(getattr(selected_slogan, "implied_action", "") or ""),
    }


def _norm_id(value: object) -> str:
    return str(value or "").strip().upper()


def _parse_strategy_evaluations(
    raw_payload: object,
    *,
    expected_ids: List[str],
) -> Dict[str, StrategyCandidateReview]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_stage", ["strategy_stage_not_object"]) from exc

    evaluations_raw = obj.get("evaluations")
    if not isinstance(evaluations_raw, list):
        raise StageParseError("strategy_stage", ["strategy_stage_missing_evaluations"])

    expected = {_norm_id(cid) for cid in expected_ids}
    parsed: Dict[str, StrategyCandidateReview] = {}
    seen: set[str] = set()
    for item in evaluations_raw:
        if not isinstance(item, dict):
            reasons.append("strategy_stage_evaluation_not_object")
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid not in expected:
            reasons.append(f"strategy_stage_evaluation_unknown_id:{cid}")
            continue
        if cid in seen:
            reasons.append(f"strategy_stage_evaluation_duplicate_id:{cid}")
            continue
        seen.add(cid)
        rejection_codes = [
            str(code)
            for code in (item.get("rejectionCodes") or [])
            if str(code).strip()
        ]
        eligible_flag = bool(item.get("eligible"))
        review = StrategyCandidateReview(
            candidate_id=cid,
            grounded_in_brief=bool(item.get("groundedInBrief")),
            advantage_currently_true=bool(item.get("advantageCurrentlyTrue")),
            executable_now=bool(item.get("executableNow")),
            requires_material_investment=bool(item.get("requiresMaterialInvestment")),
            requires_client_consultation=bool(item.get("requiresClientConsultation")),
            requires_business_transformation=bool(item.get("requiresBusinessTransformation")),
            brand_ownable=bool(item.get("brandOwnable")),
            category_relevant=bool(item.get("categoryRelevant")),
            eligible=eligible_flag,
            rejection_codes=rejection_codes,
        )
        if eligible_flag and rejection_codes:
            reasons.append(f"strategy_stage_evaluation_contradictory:{cid}")
        if not eligible_flag and not rejection_codes:
            reasons.append(f"strategy_stage_evaluation_ineligible_without_codes:{cid}")
        parsed[cid] = review

    if seen != expected:
        for missing in sorted(expected - seen):
            reasons.append(f"strategy_stage_evaluation_missing_id:{missing}")

    if reasons:
        raise StageParseError("strategy_stage", reasons)
    return parsed


def _pick_strongest_eligible_strategy(
    *,
    candidates: List[StrategyCandidate],
    reviews: Dict[str, StrategyCandidateReview],
    preferred_id: Optional[str] = None,
) -> StrategyCandidate:
    by_id = {c.id: c for c in candidates}
    ordered = [_norm_id(cid) for cid in STRATEGY_IDS if _norm_id(cid) in by_id]
    if preferred_id:
        preferred = _norm_id(preferred_id)
        review = reviews.get(preferred)
        candidate = by_id.get(preferred)
        if (
            candidate
            and review
            and review.eligible
            and strategy_candidate_is_eligible(candidate)
            and not validate_selected_strategy_gate(candidate, review)
        ):
            return candidate
    for cid in ordered:
        candidate = by_id[cid]
        review = reviews.get(cid)
        if not review or not review.eligible:
            continue
        if not strategy_candidate_is_eligible(candidate):
            continue
        if validate_selected_strategy_gate(candidate, review):
            continue
        return candidate
    raise StrategySelectionExhausted()


def process_strategy_stage_response(
    raw_payload: object,
    *,
    product_name: str,
    product_description: str,
    model_caller: PlanningModelCaller,
) -> Tuple[StrategySelection, StrategyCandidate, List[StrategyCandidate], Dict[str, StrategyCandidateReview]]:
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_stage", ["strategy_stage_not_object"]) from exc

    def _repair_caller(system: str, user: str, **kwargs: Any) -> object:
        stage = kwargs.get("stage") or "strategy_candidate_repair"
        if "stage" in kwargs:
            return model_caller(system, user, stage=stage)
        return model_caller(system, user)

    candidates = ensure_strategy_scan_from_raw(
        obj,
        product_name=product_name,
        product_description=product_description,
        model_caller=_repair_caller,
    )
    reviews = _parse_strategy_evaluations(obj, expected_ids=STRATEGY_IDS)
    selected_id = _norm_id(obj.get("selectedCandidateId"))
    selection_reason = str(obj.get("selectionReason") or "").strip() or "Strongest eligible strategy"
    selected = _pick_strongest_eligible_strategy(
        candidates=candidates,
        reviews=reviews,
        preferred_id=selected_id,
    )
    if selected.id != selected_id:
        logger.info(
            "BUILDER1_STRATEGY_STAGE_RESELECT modelSelected=%s localSelected=%s",
            selected_id,
            selected.id,
        )
    selection = StrategySelection(
        selected_candidate_id=selected.id,
        selection_reason=selection_reason,
        strategy_family=selected.lens,
        scores=dict(DEFAULT_STRATEGY_SCORES),
    )
    return selection, selected, candidates, reviews


def run_strategy_stage(
    run_stage: RunStageFn,
    model_caller: PlanningModelCaller,
    *,
    product_name: str,
    product_description: str,
    detected_language: str,
    lens_order: List[str],
    exploration_seed: str,
) -> Tuple[StrategySelection, StrategyCandidate, List[StrategyCandidate], Dict[str, StrategyCandidateReview]]:
    user_prompt = build_strategy_stage_user_prompt(
        product_name=product_name,
        product_description=product_description,
        detected_language=detected_language,
        lens_order=lens_order,
        exploration_seed=exploration_seed,
    )

    def _parse(raw: object):
        return process_strategy_stage_response(
            raw,
            product_name=product_name,
            product_description=product_description,
            model_caller=model_caller,
        )

    return run_stage(
        "strategy_stage",
        model_caller,
        STAGE_STRATEGY_STAGE_SYSTEM,
        user_prompt,
        _parse,
    )


def process_slogan_stage_response(
    raw_payload: object,
) -> Tuple[SloganSelection, SloganCandidate, List[SloganCandidate]]:
    candidates, _evaluations, selected_id, selection_reason = parse_consolidated_slogan_stage_response(
        raw_payload
    )
    selected = next(candidate for candidate in candidates if candidate.id == selected_id)
    return (
        SloganSelection(
            selected_candidate_id=selected.id,
            selection_reason=selection_reason,
            scores=dict(DEFAULT_SLOGAN_SCORES),
        ),
        selected,
        candidates,
    )


def run_slogan_stage(
    run_stage: RunStageFn,
    model_caller: PlanningModelCaller,
    *,
    selected_strategy: StrategyCandidate,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
) -> Tuple[SloganSelection, SloganCandidate, List[SloganCandidate]]:
    user_prompt = build_slogan_stage_user_prompt(
        product_name_resolved=product_name_resolved,
        product_description=product_description,
        detected_language=detected_language,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brief_support=selected_strategy.brief_support,
    )

    def _parse(raw: object):
        return process_slogan_stage_response(raw)

    return run_stage(
        "slogan_stage",
        model_caller,
        STAGE_SLOGAN_STAGE_SYSTEM,
        user_prompt,
        _parse,
    )


def _parse_conceptual_evaluations(
    raw_payload: object,
    *,
    expected_ids: List[str],
) -> Dict[str, ConceptualCandidateReview]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("conceptual_stage", ["conceptual_stage_not_object"]) from exc

    evaluations_raw = obj.get("evaluations")
    if not isinstance(evaluations_raw, list):
        raise StageParseError("conceptual_stage", ["conceptual_stage_missing_evaluations"])

    expected = {_norm_id(cid) for cid in expected_ids}
    parsed: Dict[str, ConceptualCandidateReview] = {}
    seen: set[str] = set()
    for item in evaluations_raw:
        if not isinstance(item, dict):
            reasons.append("conceptual_stage_evaluation_not_object")
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid not in expected:
            reasons.append(f"conceptual_stage_evaluation_unknown_id:{cid}")
            continue
        if cid in seen:
            reasons.append(f"conceptual_stage_evaluation_duplicate_id:{cid}")
            continue
        seen.add(cid)
        rejection_codes = [
            str(code)
            for code in (item.get("rejectionCodes") or [])
            if str(code).strip() in CONCEPTUAL_REJECTION_CODES
        ]
        eligible_flag = bool(item.get("eligible"))
        if eligible_flag and rejection_codes:
            reasons.append(f"conceptual_stage_evaluation_contradictory:{cid}")
        if not eligible_flag and not rejection_codes:
            reasons.append(f"conceptual_stage_evaluation_ineligible_without_codes:{cid}")
        parsed[cid] = ConceptualCandidateReview(
            candidate_id=cid,
            perception_to_create=str(item.get("perceptionToCreate") or "").strip(),
            implied_physical_law=str(item.get("impliedPhysicalLaw") or "").strip(),
            derived_from_selected_slogan_action=bool(item.get("derivedFromSelectedSloganAction")),
            expresses_relative_advantage=bool(item.get("expressesRelativeAdvantage")),
            visually_clear=bool(item.get("visuallyClear")),
            series_generative=bool(item.get("seriesGenerative")),
            brand_ownable=bool(item.get("brandOwnable")),
            category_relevant=bool(item.get("categoryRelevant")),
            executable_by_image_model=bool(item.get("executableByImageModel")),
            survives_product_removal=bool(item.get("survivesProductRemoval")),
            avoids_product_shot_bias=bool(item.get("avoidsProductShotBias")),
            supports_transferred_object=bool(item.get("supportsTransferredObject")),
            distinctive_to_brand=bool(item.get("distinctiveToBrand")),
            product_evidence_required=bool(item.get("productEvidenceRequired")),
            product_evidence_reason=str(item.get("productEvidenceReason") or "").strip(),
            eligible=eligible_flag,
            rejection_codes=rejection_codes,
        )

    if seen != expected:
        for missing in sorted(expected - seen):
            reasons.append(f"conceptual_stage_evaluation_missing_id:{missing}")

    if reasons:
        raise StageParseError("conceptual_stage", reasons)
    return parsed


def _conceptual_gate_reasons(review: ConceptualCandidateReview) -> List[str]:
    reasons: List[str] = list(review.rejection_codes)
    if not review.eligible:
        if not reasons:
            reasons.append("concept_not_derived_from_slogan_action")
        return reasons
    if not review.perception_to_create:
        reasons.append("concept_not_visually_clear")
    if not review.implied_physical_law:
        reasons.append("concept_not_derived_from_slogan_action")
    if not review.derived_from_selected_slogan_action:
        reasons.append("concept_not_derived_from_slogan_action")
    if not review.expresses_relative_advantage:
        reasons.append("concept_does_not_express_advantage")
    if not review.visually_clear:
        reasons.append("concept_not_visually_clear")
    if not review.series_generative:
        reasons.append("concept_not_series_generative")
    if not review.brand_ownable:
        reasons.append("concept_not_brand_ownable")
    if not review.category_relevant:
        reasons.append("concept_not_category_relevant")
    if not review.executable_by_image_model:
        reasons.append("concept_not_image_executable")
    if review.product_evidence_required:
        if not review.product_evidence_reason:
            reasons.append("concept_collapses_without_product")
    else:
        if not review.survives_product_removal:
            reasons.append("concept_collapses_without_product")
        if not review.supports_transferred_object:
            reasons.append("concept_no_transferred_object_path")
    if not review.avoids_product_shot_bias:
        reasons.append("concept_conventional_product_shot")
    if not review.distinctive_to_brand:
        reasons.append("concept_not_distinctive")
    return list(dict.fromkeys(reasons))


def _repair_conceptual_candidates(
    *,
    candidates: List[ConceptualCandidate],
    broken_json: str,
    reasons: List[str],
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
    product_description: str,
    brand_slogan: str,
    implied_action: str,
) -> List[ConceptualCandidate]:
    def _parse(raw: object):
        return parse_conceptual_scan(
            raw,
            product_description=product_description,
            brand_slogan=brand_slogan,
            implied_action=implied_action,
        )

    return run_stage(
        "conceptual_candidate_repair",
        model_caller,
        STAGE_CONCEPTUAL_STAGE_SYSTEM,
        build_conceptual_scan_repair_prompt(broken_json=broken_json, reasons=reasons),
        _parse,
    )


def process_conceptual_stage_response(
    raw_payload: object,
    *,
    product_description: str,
    product_name_resolved: str,
    brand_slogan: str,
    implied_action: str,
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
) -> Tuple[ConceptualSelection, ConceptualCandidate, List[ConceptualCandidate]]:
    try:
        candidates = parse_conceptual_scan(
            raw_payload,
            product_description=product_description,
            brand_slogan=brand_slogan,
            implied_action=implied_action,
        )
    except StageParseError as exc:
        if exc.stage != "conceptual_scan":
            raise
        candidates = _repair_conceptual_candidates(
            candidates=[],
            broken_json=json.dumps(coerce_json_dict(raw_payload), ensure_ascii=False),
            reasons=exc.reasons,
            model_caller=model_caller,
            run_stage=run_stage,
            product_description=product_description,
            brand_slogan=brand_slogan,
            implied_action=implied_action,
        )

    evaluations = _parse_conceptual_evaluations(raw_payload, expected_ids=CONCEPTUAL_IDS)
    obj = coerce_json_dict(raw_payload)
    preferred_id = _norm_id(obj.get("selectedCandidateId"))
    selection_reason = str(obj.get("selectionReason") or "").strip() or "Strongest eligible concept"
    by_id = {c.id: c for c in candidates}

    eligible_ids = [
        cid
        for cid in CONCEPTUAL_IDS
        if cid in evaluations and not _conceptual_gate_reasons(evaluations[cid])
    ]
    if not eligible_ids:
        raise StageParseError("conceptual_stage", ["conceptual_stage_no_eligible_candidates"])

    selected: Optional[ConceptualCandidate] = None
    if preferred_id in eligible_ids:
        selected = by_id[preferred_id]
    else:
        for cid in CONCEPTUAL_IDS:
            if cid in eligible_ids:
                selected = by_id[cid]
                logger.info(
                    "BUILDER1_CONCEPTUAL_STAGE_RESELECT modelSelected=%s localSelected=%s",
                    preferred_id,
                    cid,
                )
                break

    if selected is None:
        raise StageParseError("conceptual_stage", ["conceptual_stage_no_eligible_candidates"])

    return (
        ConceptualSelection(
            selected_candidate_id=selected.id,
            selection_reason=selection_reason,
        ),
        selected,
        candidates,
    )


def run_conceptual_stage(
    run_stage: RunStageFn,
    model_caller: PlanningModelCaller,
    *,
    product_description: str,
    product_name_resolved: str,
    selected_strategy: StrategyCandidate,
    selected_slogan: SloganCandidate,
    exploration_seed: str,
) -> Tuple[ConceptualSelection, ConceptualCandidate, List[ConceptualCandidate]]:
    user_prompt = build_conceptual_stage_user_prompt(
        product_description=product_description,
        product_name_resolved=product_name_resolved,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brand_slogan=selected_slogan.brand_slogan,
        slogan_derivation=selected_slogan.derivation_from_advantage,
        implied_action=selected_slogan.implied_action,
        exploration_seed=exploration_seed,
    )

    def _parse(raw: object):
        return process_conceptual_stage_response(
            raw,
            product_description=product_description,
            product_name_resolved=product_name_resolved,
            brand_slogan=selected_slogan.brand_slogan,
            implied_action=selected_slogan.implied_action,
            model_caller=model_caller,
            run_stage=run_stage,
        )

    return run_stage(
        "conceptual_stage",
        model_caller,
        STAGE_CONCEPTUAL_STAGE_SYSTEM,
        user_prompt,
        _parse,
    )
