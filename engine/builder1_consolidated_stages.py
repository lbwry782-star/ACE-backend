"""Builder1 consolidated same-layer planning stages (strategy, slogan, conceptual)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.builder1_client_boundary import strategy_candidate_is_eligible
from engine.builder1_conceptual_evaluations import (
    CONCEPTUAL_REJECTION_CODES,
    extract_repairable_evaluation_error_ids,
    is_repairable_evaluation_parse_error,
    merge_conceptual_evaluation_replacements,
    normalize_conceptual_evaluation_item,
    normalize_conceptual_evaluations_in_payload,
    parse_conceptual_evaluation_replacements,
    validate_normalized_conceptual_evaluation_item,
)
from engine.builder1_planning_contract import (
    STAGE_CONCEPTUAL_EVALUATION_REPAIR_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_STRATEGY_SLOGAN_REPAIR_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
    STAGE_STRATEGY_STAGE_SYSTEM,
    build_conceptual_evaluation_repair_user_prompt,
    build_conceptual_scan_repair_prompt,
    build_conceptual_stage_user_prompt,
    build_strategy_slogan_repair_user_prompt,
    build_strategy_slogan_stage_user_prompt,
    build_strategy_stage_user_prompt,
)
from engine.builder1_staged_parsers import (
    CONCEPTUAL_IDS,
    StageParseError,
    ConceptualCandidate,
    ConceptualSelection,
    StrategyCandidate,
    StrategyCandidateReview,
    StrategySelection,
    coerce_json_dict,
    parse_conceptual_scan,
)
from engine.builder1_strategy_slogan_final import (
    FINAL_SLOGAN_ID,
    FINAL_STRATEGY_ID,
    parse_slogan_final_section,
    parse_strategy_final_section,
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


def process_strategy_stage_response(
    raw_payload: object,
    *,
    product_name: str,
    product_description: str,
    model_caller: PlanningModelCaller,
) -> Tuple[StrategySelection, StrategyCandidate, List[StrategyCandidate], Dict[str, StrategyCandidateReview]]:
    del product_name, model_caller
    strategy_selection, selected_strategy = parse_strategy_final_section(
        raw_payload,
        product_description=product_description,
    )
    return strategy_selection, selected_strategy, [selected_strategy], {}


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


def process_strategy_slogan_stage_response(
    raw_payload: object,
    *,
    product_name: str,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
) -> Tuple[
    StrategySelection,
    StrategyCandidate,
    List[StrategyCandidate],
    Dict[str, StrategyCandidateReview],
    Any,
    Any,
    List[Any],
]:
    del product_name, product_name_resolved, model_caller, run_stage
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_slogan_stage", ["strategy_slogan_stage_invalid_structure"]) from exc

    strategy_raw = obj.get("strategy")
    slogan_raw = obj.get("slogan")
    if not isinstance(strategy_raw, dict) or not isinstance(slogan_raw, dict):
        raise StageParseError("strategy_slogan_stage", ["strategy_slogan_stage_invalid_structure"])

    try:
        strategy_selection, selected_strategy = parse_strategy_final_section(
            strategy_raw,
            product_description=product_description,
        )
    except StageParseError as exc:
        if exc.stage != "strategy_slogan_stage":
            raise
        raise StageParseError(
            "strategy_slogan_stage",
            [f"strategy:{reason}" if not str(reason).startswith("strategy:") else reason for reason in exc.reasons],
        ) from exc

    logger.info("BUILDER1_STRATEGY_SECTION_OK finalStrategy=true")

    try:
        slogan_selection, selected_slogan = parse_slogan_final_section(
            slogan_raw,
            relative_advantage=selected_strategy.relative_advantage,
            product_description=product_description,
            detected_language=detected_language,
        )
    except StageParseError as exc:
        if exc.stage != "strategy_slogan_stage":
            raise
        raise

    logger.info("BUILDER1_SLOGAN_SECTION_OK finalSlogan=true")
    return (
        strategy_selection,
        selected_strategy,
        [selected_strategy],
        {},
        slogan_selection,
        selected_slogan,
        [selected_slogan],
    )


def run_strategy_slogan_stage(
    run_stage: RunStageFn,
    model_caller: PlanningModelCaller,
    *,
    product_name: str,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    lens_order: List[str],
    exploration_seed: str,
) -> Tuple[
    StrategySelection,
    StrategyCandidate,
    List[StrategyCandidate],
    Dict[str, StrategyCandidateReview],
    SloganSelection,
    SloganCandidate,
    List[SloganCandidate],
]:
    user_prompt = build_strategy_slogan_stage_user_prompt(
        product_name=product_name,
        product_description=product_description,
        detected_language=detected_language,
        lens_order=lens_order,
        exploration_seed=exploration_seed,
    )

    def _parse(raw: object):
        return process_strategy_slogan_stage_response(
            raw,
            product_name=product_name,
            product_name_resolved=product_name_resolved,
            product_description=product_description,
            detected_language=detected_language,
            model_caller=model_caller,
            run_stage=run_stage,
        )

    def _repair_builder(broken_json: str, reasons: List[str]) -> str:
        return build_strategy_slogan_repair_user_prompt(
            broken_json=broken_json,
            reasons=reasons,
            product_name=product_name_resolved,
            product_description=product_description,
        )

    return run_stage(
        "strategy_slogan_stage",
        model_caller,
        STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
        user_prompt,
        _parse,
        repair_builder=_repair_builder,
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
        normalized_item, _actions = normalize_conceptual_evaluation_item(item)
        rejection_codes = list(normalized_item.get("rejectionCodes") or [])
        eligible_flag = bool(normalized_item.get("eligible"))
        reasons.extend(
            validate_normalized_conceptual_evaluation_item(
                normalized_item,
                candidate_id=cid,
            )
        )
        parsed[cid] = ConceptualCandidateReview(
            candidate_id=cid,
            perception_to_create=str(normalized_item.get("perceptionToCreate") or "").strip(),
            implied_physical_law=str(normalized_item.get("impliedPhysicalLaw") or "").strip(),
            derived_from_selected_slogan_action=bool(normalized_item.get("derivedFromSelectedSloganAction")),
            expresses_relative_advantage=bool(normalized_item.get("expressesRelativeAdvantage")),
            visually_clear=bool(normalized_item.get("visuallyClear")),
            series_generative=bool(normalized_item.get("seriesGenerative")),
            brand_ownable=bool(normalized_item.get("brandOwnable")),
            category_relevant=bool(normalized_item.get("categoryRelevant")),
            executable_by_image_model=bool(normalized_item.get("executableByImageModel")),
            survives_product_removal=bool(normalized_item.get("survivesProductRemoval")),
            avoids_product_shot_bias=bool(normalized_item.get("avoidsProductShotBias")),
            supports_transferred_object=bool(normalized_item.get("supportsTransferredObject")),
            distinctive_to_brand=bool(normalized_item.get("distinctiveToBrand")),
            product_evidence_required=bool(normalized_item.get("productEvidenceRequired")),
            product_evidence_reason=str(normalized_item.get("productEvidenceReason") or "").strip(),
            eligible=eligible_flag,
            rejection_codes=rejection_codes,
        )

    if seen != expected:
        for missing in sorted(expected - seen):
            reasons.append(f"conceptual_stage_evaluation_missing_id:{missing}")

    if reasons:
        raise StageParseError("conceptual_stage", reasons)
    return parsed


def _run_conceptual_evaluation_repair(
    *,
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
    invalid_ids: List[str],
    invalid_reasons: Dict[str, List[str]],
    evaluation_items: Dict[str, Dict[str, Any]],
    candidates: List[ConceptualCandidate],
    product_description: str,
    brand_slogan: str,
    implied_action: str,
    relative_advantage: str,
    strategic_problem: str,
    repair_attempt: int,
) -> Dict[str, Dict[str, Any]]:
    user_prompt = build_conceptual_evaluation_repair_user_prompt(
        invalid_candidate_ids=invalid_ids,
        invalid_reasons=invalid_reasons,
        evaluation_items=evaluation_items,
        candidates=candidates,
        product_description=product_description,
        brand_slogan=brand_slogan,
        implied_action=implied_action,
        relative_advantage=relative_advantage,
        strategic_problem=strategic_problem,
    )

    def _parse(raw: object):
        return parse_conceptual_evaluation_replacements(
            raw,
            allowed_ids=invalid_ids,
        )

    logger.info(
        "BUILDER1_CONCEPTUAL_REPAIR_START attempt=%s candidateIds=%s reasonCodes=%s",
        repair_attempt,
        ",".join(sorted(invalid_ids)),
        ",".join(sorted({code for codes in invalid_reasons.values() for code in codes})),
    )
    return run_stage(
        "conceptual_evaluation_repair",
        model_caller,
        STAGE_CONCEPTUAL_EVALUATION_REPAIR_SYSTEM,
        user_prompt,
        _parse,
    )


def _ensure_conceptual_evaluations(
    raw_payload: object,
    *,
    candidates: List[ConceptualCandidate],
    product_description: str,
    brand_slogan: str,
    implied_action: str,
    relative_advantage: str,
    strategic_problem: str,
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
) -> Dict[str, ConceptualCandidateReview]:
    obj = coerce_json_dict(raw_payload)
    obj, _action_log = normalize_conceptual_evaluations_in_payload(obj)

    try:
        return _parse_conceptual_evaluations(obj, expected_ids=CONCEPTUAL_IDS)
    except StageParseError as exc:
        if not is_repairable_evaluation_parse_error(exc.reasons):
            raise
        last_reasons = list(exc.reasons)

    evaluation_items = {
        _norm_id(item.get("candidateId")): item
        for item in (obj.get("evaluations") or [])
        if isinstance(item, dict)
    }

    for repair_attempt in (1, 2):
        invalid_reasons = extract_repairable_evaluation_error_ids(last_reasons)
        invalid_ids = sorted(invalid_reasons.keys())
        if not invalid_ids:
            raise StageParseError("conceptual_stage", last_reasons)

        try:
            replacements = _run_conceptual_evaluation_repair(
                model_caller=model_caller,
                run_stage=run_stage,
                invalid_ids=invalid_ids,
                invalid_reasons=invalid_reasons,
                evaluation_items=evaluation_items,
                candidates=candidates,
                product_description=product_description,
                brand_slogan=brand_slogan,
                implied_action=implied_action,
                relative_advantage=relative_advantage,
                strategic_problem=strategic_problem,
                repair_attempt=repair_attempt,
            )
        except StageParseError as repair_exc:
            logger.info(
                "BUILDER1_CONCEPTUAL_REPAIR_FAILED attempt=%s reasonCodes=%s",
                repair_attempt,
                repair_exc.reasons,
            )
            if repair_attempt == 2:
                raise StageParseError("conceptual_stage", repair_exc.reasons) from repair_exc
            continue

        obj = merge_conceptual_evaluation_replacements(obj, replacements)
        obj, _action_log = normalize_conceptual_evaluations_in_payload(obj)
        try:
            reviews = _parse_conceptual_evaluations(obj, expected_ids=CONCEPTUAL_IDS)
            logger.info(
                "BUILDER1_CONCEPTUAL_REPAIR_OK attempt=%s candidateIds=%s",
                repair_attempt,
                ",".join(sorted(replacements.keys())),
            )
            return reviews
        except StageParseError as exc2:
            last_reasons = exc2.reasons
            logger.info(
                "BUILDER1_CONCEPTUAL_REPAIR_FAILED attempt=%s reasonCodes=%s",
                repair_attempt,
                exc2.reasons,
            )
            if repair_attempt == 2:
                raise

    raise StageParseError("conceptual_stage", last_reasons or ["conceptual_evaluation_repair_exhausted"])


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
    relative_advantage: str,
    strategic_problem: str,
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

    evaluations = _ensure_conceptual_evaluations(
        raw_payload,
        candidates=candidates,
        product_description=product_description,
        brand_slogan=brand_slogan,
        implied_action=implied_action,
        relative_advantage=relative_advantage,
        strategic_problem=strategic_problem,
        model_caller=model_caller,
        run_stage=run_stage,
    )
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
            relative_advantage=selected_strategy.relative_advantage,
            strategic_problem=selected_strategy.strategic_problem,
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
