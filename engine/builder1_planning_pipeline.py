"""Builder1 staged campaign pipeline execution (consolidated planning stages)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.builder1_campaign_integrity import (
    make_upstream_snapshot,
    validate_builder1_campaign_integrity,
)
from engine.builder1_consolidated_stages import (
    run_conceptual_stage,
    run_slogan_stage,
    run_strategy_stage,
)
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
    build_brand_physical_repair_prompt,
    build_brand_physical_user_prompt,
    build_graphic_system_repair_prompt,
    build_graphic_system_user_prompt,
    build_series_ads_repair_prompt,
    build_series_ads_user_prompt,
)
from engine.builder1_product_name import enforce_authoritative_product_name
from engine.builder1_slogan_stage import slogan_candidate_to_dict
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_strategy_selection import StrategySelectionExhausted

logger = logging.getLogger(__name__)


@dataclass
class Builder1PipelineContext:
    exploration_seed: str
    lens_order: List[str]
    strategy_selection: Any
    selected_strategy: Any
    slogan_dicts: List[Dict[str, Any]]
    selected_slogan: Any
    conc_dicts: List[Dict[str, Any]]
    selected_conceptual: Any
    conceptual_fixed: Dict[str, str]
    brand_physical: Any
    brand_physical_dict: Dict[str, Any]
    graphic: Any
    graphic_dict: Dict[str, Any]
    series_ads: SeriesAdsOutput
    plan: Builder1SeriesPlan
    upstream_snapshot: Any


def run_builder1_campaign_pipeline(
    *,
    normalized: Any,
    product_name_resolved: str,
    detected_language: str,
    exploration_seed: str,
    lens_order: List[str],
    model_caller: Any,
    brand_guidelines: Optional[Dict[str, Any]],
) -> Builder1PipelineContext:
    from engine.builder1_planner import (
        Builder1PlannerError,
        _brand_physical_to_dict,
        _conceptual_to_dict,
        _graphic_to_dict,
        _run_stage,
    )

    try:
        strategy_selection, selected_strategy, _strategy_candidates, _strategy_reviews = run_strategy_stage(
            _run_stage,
            model_caller,
            product_name=product_name_resolved,
            product_description=normalized.product_description,
            detected_language=detected_language,
            lens_order=lens_order,
            exploration_seed=exploration_seed,
        )
    except StrategySelectionExhausted as exc:
        raise Builder1PlannerError("strategy_stage_failed") from exc

    _slogan_selection, selected_slogan, slogan_candidates = run_slogan_stage(
        _run_stage,
        model_caller,
        selected_strategy=selected_strategy,
        product_name_resolved=product_name_resolved,
        product_description=normalized.product_description,
        detected_language=detected_language,
    )
    slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]

    _conceptual_selection, selected_conceptual, conceptual_candidates = run_conceptual_stage(
        _run_stage,
        model_caller,
        product_description=normalized.product_description,
        product_name_resolved=product_name_resolved,
        selected_strategy=selected_strategy,
        selected_slogan=selected_slogan,
        exploration_seed=exploration_seed,
    )
    conc_dicts = [
        {
            "id": c.id,
            "generator": c.generator,
            "action": c.action,
            "input": c.input,
            "transformation": c.transformation,
            "result": c.result,
            "whyItExpressesSlogan": c.why_it_expresses_slogan,
            "whyItExpressesAdvantage": c.why_it_expresses_advantage,
            "seriesPotential": c.series_potential,
            "brandOwnershipPotential": c.brand_ownership_potential,
        }
        for c in conceptual_candidates
    ]
    conceptual_fixed = _conceptual_to_dict(selected_conceptual)

    brand_physical = _run_stage(
        "brand_physical",
        model_caller,
        STAGE_BRAND_PHYSICAL_SYSTEM,
        build_brand_physical_user_prompt(
            product_name_resolved=product_name_resolved,
            product_description=normalized.product_description,
            detected_language=detected_language,
            format_value=normalized.format,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            brand_slogan=selected_slogan.brand_slogan,
            slogan_derivation=selected_slogan.derivation_from_advantage,
            implied_action=selected_slogan.implied_action,
            conceptual=conceptual_fixed,
            brand_guidelines=brand_guidelines,
        ),
        lambda raw: parse_brand_physical_output(
            raw, product_description=normalized.product_description
        ),
        repair_builder=lambda broken, reasons: build_brand_physical_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )
    brand_physical = enforce_authoritative_product_name(
        brand_physical,
        product_name_resolved=product_name_resolved,
    )
    brand_physical_dict = _brand_physical_to_dict(brand_physical)

    graphic = _run_stage(
        "graphic_system",
        model_caller,
        STAGE_GRAPHIC_SYSTEM_SYSTEM,
        build_graphic_system_user_prompt(
            product_description=normalized.product_description,
            detected_language=detected_language,
            relative_advantage=selected_strategy.relative_advantage,
            brand_slogan=selected_slogan.brand_slogan,
            conceptual=conceptual_fixed,
            brand_physical=brand_physical_dict,
            format_value=normalized.format,
        ),
        parse_graphic_system_output,
        repair_builder=lambda broken, reasons: build_graphic_system_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )
    graphic_dict = _graphic_to_dict(graphic)

    upstream_snapshot = make_upstream_snapshot(
        product_name_resolved=product_name_resolved,
        selected_strategy=selected_strategy,
        selected_slogan=selected_slogan,
        selected_conceptual=selected_conceptual,
        brand_physical=brand_physical,
        graphic=graphic,
    )

    series_ads = _run_series_stage_with_integrity(
        normalized=normalized,
        detected_language=detected_language,
        selected_strategy=selected_strategy,
        selected_slogan=selected_slogan,
        conceptual_fixed=conceptual_fixed,
        brand_physical_dict=brand_physical_dict,
        graphic_dict=graphic_dict,
        upstream_snapshot=upstream_snapshot,
        model_caller=model_caller,
        run_stage=_run_stage,
        series_retry_used=False,
    )

    plan = assemble_builder1_campaign(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        ad_count=normalized.ad_count,
        detected_language=detected_language,
        exploration_seed=exploration_seed,
        product_name_resolved=product_name_resolved,
        strategy=selected_strategy,
        strategy_selection=strategy_selection,
        selected_slogan=selected_slogan,
        conceptual=selected_conceptual,
        brand_physical=brand_physical,
        graphic=graphic,
        series_ads=series_ads,
    )

    plan_dict = series_plan_to_store_dict(plan)
    plan_dict["planningEvidence"] = {
        "sloganQualityValidated": True,
        "sloganDerivedFromAdvantageValidated": True,
        "semanticDerivationStandard": "overlap_not_required",
        "selectedSloganId": selected_slogan.id,
    }

    integrity = validate_builder1_campaign_integrity(
        plan,
        upstream=upstream_snapshot,
        detected_language=detected_language,
    )
    if integrity.upstream_mutation:
        logger.error(
            "BUILDER1_INTEGRITY_FAILED reasons=%s",
            integrity.reasons,
        )
        raise Builder1PlannerError("campaign_integrity_failed")
    if not integrity.ok:
        logger.error(
            "BUILDER1_INTEGRITY_FAILED reasons=%s",
            integrity.reasons,
        )
        raise Builder1PlannerError("campaign_integrity_failed")

    logger.info("BUILDER1_INTEGRITY_OK")

    return Builder1PipelineContext(
        exploration_seed=exploration_seed,
        lens_order=lens_order,
        strategy_selection=strategy_selection,
        selected_strategy=selected_strategy,
        slogan_dicts=slogan_dicts,
        selected_slogan=selected_slogan,
        conc_dicts=conc_dicts,
        selected_conceptual=selected_conceptual,
        conceptual_fixed=conceptual_fixed,
        brand_physical=brand_physical,
        brand_physical_dict=brand_physical_dict,
        graphic=graphic,
        graphic_dict=graphic_dict,
        series_ads=series_ads,
        plan=plan,
        upstream_snapshot=upstream_snapshot,
    )


def _run_series_stage_with_integrity(
    *,
    normalized: Any,
    detected_language: str,
    selected_strategy: Any,
    selected_slogan: Any,
    conceptual_fixed: Dict[str, str],
    brand_physical_dict: Dict[str, Any],
    graphic_dict: Dict[str, Any],
    upstream_snapshot: Any,
    model_caller: Any,
    run_stage: Any,
    series_retry_used: bool,
) -> SeriesAdsOutput:
    from engine.builder1_planner import Builder1PlannerError

    series_ads = run_stage(
        "series_ads",
        model_caller,
        STAGE_SERIES_ADS_SYSTEM,
        build_series_ads_user_prompt(
            ad_count=normalized.ad_count,
            format_value=normalized.format,
            detected_language=detected_language,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            brand_slogan=selected_slogan.brand_slogan,
            implied_action=selected_slogan.implied_action,
            conceptual=conceptual_fixed,
            brand_physical=brand_physical_dict,
            graphic_generator=graphic_dict,
        ),
        lambda raw: parse_series_ads_output(
            raw,
            expected_ad_count=normalized.ad_count,
            product_description=normalized.product_description,
        ),
        repair_builder=lambda broken, reasons: build_series_ads_repair_prompt(
            broken_json=broken, reasons=reasons, ad_count=normalized.ad_count
        ),
    )

    try:
        series_ads.ads = ensure_series_ads_marketing_text(
            series_ads.ads,
            detected_language=detected_language,
            relative_advantage=selected_strategy.relative_advantage,
            product_name=upstream_snapshot.product_name_resolved,
            brand_slogan=selected_slogan.brand_slogan,
            model_caller=model_caller,
        )
    except StageParseError as exc:
        raise Builder1PlannerError("marketing_text_failed") from exc

    if _series_ads_needs_retry(series_ads, expected_ad_count=normalized.ad_count) and not series_retry_used:
        logger.info("BUILDER1_SERIES_STAGE_RETRY reason=structural_failure")
        return _run_series_stage_with_integrity(
            normalized=normalized,
            detected_language=detected_language,
            selected_strategy=selected_strategy,
            selected_slogan=selected_slogan,
            conceptual_fixed=conceptual_fixed,
            brand_physical_dict=brand_physical_dict,
            graphic_dict=graphic_dict,
            upstream_snapshot=upstream_snapshot,
            model_caller=model_caller,
            run_stage=run_stage,
            series_retry_used=True,
        )
    return series_ads


def _series_ads_needs_retry(series_ads: SeriesAdsOutput, *, expected_ad_count: int) -> bool:
    if len(series_ads.ads) != expected_ad_count:
        return True
    indices: List[int] = []
    for ad in series_ads.ads:
        if not isinstance(ad, dict):
            return True
        try:
            indices.append(int(ad.get("index")))
        except (TypeError, ValueError):
            return True
    expected = list(range(1, expected_ad_count + 1))
    return sorted(indices) != expected or len(set(indices)) != len(indices)
