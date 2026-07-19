"""
Repair Builder1 campaigns from brand_physical forward while preserving upstream work.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.builder1_campaign_integrity import make_upstream_snapshot, validate_builder1_campaign_integrity
from engine.builder1_failure_classification import validate_forbidden_plan_visibility
from engine.builder1_final_stages import (
    SeriesAdsOutput,
    assemble_builder1_campaign,
    parse_brand_physical_output,
    parse_graphic_system_output,
    parse_series_ads_output,
)
from engine.builder1_marketing_text_repair import ensure_series_ads_marketing_text
from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_to_store_dict
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    build_brand_physical_identity_retry_prompt,
    build_brand_physical_repair_prompt,
    build_brand_physical_user_prompt,
    build_graphic_system_repair_prompt,
    build_graphic_system_user_prompt,
    build_series_ads_repair_prompt,
    build_series_ads_user_prompt,
)
from engine.builder1_planner import (
    Builder1PlannerError,
    _brand_physical_to_dict,
    _conceptual_to_dict,
    _graphic_to_dict,
    _run_stage,
)
from engine.builder1_product_identity_guard import VISUAL_CONFLICT_PREFIXES
from engine.builder1_product_name import enforce_authoritative_product_name
from engine.builder1_product_visibility import ProductVisibilityPolicy, ProductVisibilitySource
from engine.builder1_slogan_stage import SloganCandidate
from engine.builder1_staged_parsers import ConceptualCandidate, StageParseError, StrategyCandidate, StrategySelection

logger = logging.getLogger(__name__)


def _reconstruct_upstream_from_plan(plan: Builder1SeriesPlan):
    internals = plan.planning_internals or {}
    lineage = internals.get("conceptualLineage") if isinstance(internals.get("conceptualLineage"), dict) else {}
    strategy = StrategyCandidate(
        id="S01",
        lens="preserved",
        strategic_problem=plan.strategic_problem,
        relative_advantage=plan.relative_advantage,
        brief_support=plan.strategic_problem_evidence,
        advantage_source=plan.relative_advantage_source,
        claim_risk=plan.relative_advantage_claim_risk,
    )
    strategy_selection = StrategySelection(
        selected_candidate_id="S01",
        selection_reason="preserved",
        strategy_family="preserved",
        scores={},
    )
    selected_slogan = SloganCandidate(
        id=str(lineage.get("sourceSloganCandidateId") or "G01"),
        brand_slogan=plan.brand_slogan,
        derivation_from_advantage=plan.slogan_derivation,
        implied_action=plan.slogan_action,
        why_ownable=str(internals.get("brandOwnershipReason") or ""),
        why_natural_in_language="preserved",
        competitor_transfer_risk=str(internals.get("competitorTransferTest") or "medium"),
        campaign_generative_power="preserved",
    )
    selected_conceptual = ConceptualCandidate(
        id=str(lineage.get("selectedConceptCandidateId") or "C01"),
        generator=plan.conceptual_generator,
        action=plan.conceptual_generator_action,
        input=plan.conceptual_generator_input,
        transformation=plan.conceptual_generator_transformation,
        result=plan.conceptual_generator_result,
        perception_to_create=str(internals.get("perceptionToCreate") or plan.conceptual_generator_result),
        implied_physical_law=str(internals.get("impliedPhysicalLaw") or plan.conceptual_generator_action),
        why_it_expresses_slogan=str(internals.get("conceptualGeneratorWhyItExpressesSlogan") or ""),
        why_it_expresses_advantage=plan.conceptual_generator_why_it_expresses_advantage,
        series_potential="preserved",
        brand_ownership_potential="preserved",
    )
    return strategy, strategy_selection, selected_slogan, selected_conceptual


def _run_brand_physical_with_identity_guard(
    *,
    model_caller: Any,
    user_prompt: str,
    parse_kwargs: Dict[str, Any],
    visibility_policy: ProductVisibilityPolicy,
) -> Any:
    identity_retry_used = False
    current_prompt = user_prompt

    def _parse(raw: object):
        return parse_brand_physical_output(
            raw,
            visibility_policy=visibility_policy,
            **parse_kwargs,
        )

    while True:
        try:
            return _run_stage(
                "brand_physical",
                model_caller,
                STAGE_BRAND_PHYSICAL_SYSTEM,
                current_prompt,
                _parse,
                repair_builder=lambda broken, reasons: build_brand_physical_repair_prompt(
                    broken_json=broken, reasons=reasons
                ),
            )
        except Builder1PlannerError as exc:
            message = str(exc)
            identity_failed = any(prefix in message for prefix in VISUAL_CONFLICT_PREFIXES) or (
                "physical_generator_is_product" in message
                or "physical_generator_is_packaging" in message
            )
            if identity_failed and not identity_retry_used:
                identity_retry_used = True
                current_prompt = build_brand_physical_identity_retry_prompt(base_user_prompt=user_prompt)
                logger.info("BUILDER1_BRAND_PHYSICAL_IDENTITY_RETRY")
                continue
            if identity_failed:
                raise Builder1PlannerError("brand_physical_product_identity_conflict") from exc
            raise


def repair_builder1_campaign_from_physical(
    plan: Builder1SeriesPlan,
    *,
    model_caller: Any,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> Builder1SeriesPlan:
    """Rerun brand_physical, graphic_system, and series_ads while preserving upstream identity."""
    strategy, strategy_selection, selected_slogan, selected_conceptual = _reconstruct_upstream_from_plan(plan)
    conceptual_fixed = _conceptual_to_dict(selected_conceptual)
    visibility_policy = ProductVisibilityPolicy.FORBIDDEN
    raw_policy = (plan.product_visibility_policy or "").strip().upper()
    try:
        visibility_policy = ProductVisibilityPolicy(raw_policy)
    except ValueError:
        pass
    visibility_source = ProductVisibilitySource.DEFAULT
    internals = plan.planning_internals or {}
    try:
        visibility_source = ProductVisibilitySource(str(internals.get("productVisibilitySource") or "default"))
    except ValueError:
        pass

    user_prompt = build_brand_physical_user_prompt(
        product_name_resolved=plan.product_name_resolved,
        product_description=plan.product_description,
        detected_language=plan.detected_language,
        format_value=plan.format,
        strategic_problem=plan.strategic_problem,
        relative_advantage=plan.relative_advantage,
        brand_slogan=plan.brand_slogan,
        slogan_derivation=plan.slogan_derivation,
        implied_action=plan.slogan_action,
        conceptual=conceptual_fixed,
        brand_guidelines=brand_guidelines,
        visibility_policy=visibility_policy.value,
    )

    brand_physical = _run_brand_physical_with_identity_guard(
        model_caller=model_caller,
        user_prompt=user_prompt,
        parse_kwargs={
            "product_description": plan.product_description,
            "product_name_resolved": plan.product_name_resolved,
        },
        visibility_policy=visibility_policy,
    )
    brand_physical = enforce_authoritative_product_name(
        brand_physical,
        product_name_resolved=plan.product_name_resolved,
    )
    brand_physical_dict = _brand_physical_to_dict(brand_physical)

    graphic = _run_stage(
        "graphic_system",
        model_caller,
        STAGE_GRAPHIC_SYSTEM_SYSTEM,
        build_graphic_system_user_prompt(
            product_description=plan.product_description,
            detected_language=plan.detected_language,
            relative_advantage=plan.relative_advantage,
            brand_slogan=plan.brand_slogan,
            conceptual=conceptual_fixed,
            brand_physical=brand_physical_dict,
            format_value=plan.format,
        ),
        parse_graphic_system_output,
        repair_builder=lambda broken, reasons: build_graphic_system_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )
    graphic_dict = _graphic_to_dict(graphic)

    upstream_snapshot = make_upstream_snapshot(
        product_name_resolved=plan.product_name_resolved,
        selected_strategy=strategy,
        selected_slogan=selected_slogan,
        selected_conceptual=selected_conceptual,
        brand_physical=brand_physical,
        graphic=graphic,
    )

    series_ads = _run_stage(
        "series_ads",
        model_caller,
        STAGE_SERIES_ADS_SYSTEM,
        build_series_ads_user_prompt(
            ad_count=plan.ad_count,
            format_value=plan.format,
            detected_language=plan.detected_language,
            strategic_problem=plan.strategic_problem,
            relative_advantage=plan.relative_advantage,
            brand_slogan=plan.brand_slogan,
            implied_action=plan.slogan_action,
            conceptual=conceptual_fixed,
            brand_physical=brand_physical_dict,
            graphic_generator=graphic_dict,
            visibility_policy=visibility_policy.value,
        ),
        lambda raw: parse_series_ads_output(
            raw,
            expected_ad_count=plan.ad_count,
            product_description=plan.product_description,
            visibility_policy=visibility_policy,
        ),
        repair_builder=lambda broken, reasons: build_series_ads_repair_prompt(
            broken_json=broken, reasons=reasons, ad_count=plan.ad_count
        ),
    )
    try:
        series_ads.ads = ensure_series_ads_marketing_text(
            series_ads.ads,
            detected_language=plan.detected_language,
            relative_advantage=plan.relative_advantage,
            product_name=plan.product_name_resolved,
            brand_slogan=plan.brand_slogan,
            model_caller=model_caller,
        )
    except StageParseError as exc:
        raise Builder1PlannerError("marketing_text_failed") from exc

    repaired = assemble_builder1_campaign(
        product_name=plan.product_name,
        product_description=plan.product_description,
        format_value=plan.format,
        ad_count=plan.ad_count,
        detected_language=plan.detected_language,
        exploration_seed=str(internals.get("explorationSeed") or "repair"),
        product_name_resolved=plan.product_name_resolved,
        strategy=strategy,
        strategy_selection=strategy_selection,
        selected_slogan=selected_slogan,
        conceptual=selected_conceptual,
        brand_physical=brand_physical,
        graphic=graphic,
        series_ads=series_ads,
        visibility_policy=visibility_policy,
        visibility_source=visibility_source,
    )

    visibility_reasons = validate_forbidden_plan_visibility(repaired)
    if visibility_reasons:
        raise Builder1PlannerError(f"physical_repair_visibility_failed:{';'.join(visibility_reasons[:5])}")

    integrity = validate_builder1_campaign_integrity(
        repaired,
        upstream=upstream_snapshot,
        detected_language=plan.detected_language,
    )
    if not integrity.ok or integrity.upstream_mutation:
        raise Builder1PlannerError(f"physical_repair_integrity_failed:{';'.join(integrity.reasons[:5])}")

    logger.info(
        "BUILDER1_PHYSICAL_REPAIR_OK campaignPhysical=%s campaignTransferred=%s",
        repaired.physical_generator,
        repaired.transferred_object,
    )
    return repaired
