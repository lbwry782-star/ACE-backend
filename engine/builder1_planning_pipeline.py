"""Builder1 staged campaign pipeline execution and judge-targeted repair."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_SLOGAN_SCAN_SYSTEM,
    STAGE_SLOGAN_SELECT_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    build_brand_physical_repair_prompt,
    build_brand_physical_user_prompt,
    build_conceptual_scan_repair_prompt,
    build_conceptual_scan_user_prompt,
    build_conceptual_select_user_prompt,
    build_graphic_system_repair_prompt,
    build_graphic_system_user_prompt,
    build_series_ads_repair_prompt,
    build_series_ads_user_prompt,
    build_slogan_scan_repair_prompt,
    build_slogan_scan_user_prompt,
    build_slogan_select_user_prompt,
    build_strategy_select_user_prompt,
)
from engine.builder1_product_name import enforce_authoritative_product_name
from engine.builder1_staged_parsers import (
    StageParseError,
    filter_eligible_strategy_candidates,
    parse_conceptual_scan,
    parse_conceptual_selection,
    parse_strategy_selection,
)
from engine.builder1_slogan_quality import (
    SloganFullRescanRequired,
    execute_slogan_scan_through_selection,
)
from engine.builder1_slogan_stage import (
    parse_slogan_scan,
    parse_slogan_selection,
    slogan_candidate_to_dict,
    validate_selected_slogan,
)
from engine.builder1_strategy_judge import StrategyJudgeResult, judge_builder1_strategy

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
    judge_result: StrategyJudgeResult


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
        _run_strategy_scan_stage,
    )

    strategy_candidates = _run_strategy_scan_stage(
        model_caller,
        product_name=product_name_resolved,
        product_description=normalized.product_description,
        detected_language=detected_language,
        lens_order=lens_order,
        exploration_seed=exploration_seed,
    )

    cand_dicts = [
        {
            "id": c.id,
            "lens": c.lens,
            "strategicProblem": c.strategic_problem,
            "relativeAdvantage": c.relative_advantage,
            "briefSupport": c.brief_support,
            "advantageSource": c.advantage_source,
            "claimRisk": c.claim_risk,
            "campaignExecutableNow": c.campaign_executable_now,
            "requiresClientConsultation": c.requires_client_consultation,
            "clientActionLevel": c.client_action_level,
            "implementationCostLevel": c.implementation_cost_level,
            "simpleStrategicAction": c.simple_strategic_action,
        }
        for c in strategy_candidates
    ]
    eligible_strategy = filter_eligible_strategy_candidates(strategy_candidates)
    if not eligible_strategy:
        raise Builder1PlannerError("strategy_selection_failed")
    eligible_dicts = [c for c in cand_dicts if c["id"] in {e.id for e in eligible_strategy}]

    def _parse_selection(raw: object):
        return parse_strategy_selection(raw, eligible_strategy)

    strategy_selection, selected_strategy = _run_stage(
        "strategy_selection",
        model_caller,
        STAGE_STRATEGY_SELECT_SYSTEM,
        build_strategy_select_user_prompt(eligible_dicts, exploration_seed),
        _parse_selection,
    )

    slogan_candidates = _run_stage(
        "slogan_scan",
        model_caller,
        STAGE_SLOGAN_SCAN_SYSTEM,
        build_slogan_scan_user_prompt(
            product_name_resolved=product_name_resolved,
            product_description=normalized.product_description,
            detected_language=detected_language,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            brief_support=selected_strategy.brief_support,
        ),
        parse_slogan_scan,
        repair_builder=lambda broken, reasons: build_slogan_scan_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )
    slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]
    selected_slogan: Any = None

    def _run_slogan_pipeline_once(*, full_rescan_used: bool) -> bool:
        nonlocal slogan_candidates, slogan_dicts, selected_slogan
        try:
            selected_slogan, slogan_candidates = execute_slogan_scan_through_selection(
                slogan_candidates=slogan_candidates,
                selected_strategy=selected_strategy,
                product_name_resolved=product_name_resolved,
                product_description=normalized.product_description,
                detected_language=detected_language,
                model_caller=model_caller,
                run_stage=_run_stage,
                full_rescan_used=full_rescan_used,
            )
        except SloganFullRescanRequired:
            return False
        slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]
        return True

    if not _run_slogan_pipeline_once(full_rescan_used=False):
        logger.info("BUILDER1_SLOGAN_FULL_RESCAN reason=no_eligible_candidates")
        slogan_candidates = _run_stage(
            "slogan_scan",
            model_caller,
            STAGE_SLOGAN_SCAN_SYSTEM,
            build_slogan_scan_repair_prompt(
                broken_json=json.dumps({"candidates": slogan_dicts}, ensure_ascii=False),
                reasons=["no_eligible_slogan_candidates"],
            ),
            parse_slogan_scan,
        )
        slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]
        if not _run_slogan_pipeline_once(full_rescan_used=True):
            raise Builder1PlannerError("slogan_quality_gate_failed")

    conceptual_candidates = _run_stage(
        "conceptual_scan",
        model_caller,
        STAGE_CONCEPTUAL_SCAN_SYSTEM,
        build_conceptual_scan_user_prompt(
            product_description=normalized.product_description,
            product_name_resolved=product_name_resolved,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            brand_slogan=selected_slogan.brand_slogan,
            slogan_derivation=selected_slogan.derivation_from_advantage,
            implied_action=selected_slogan.implied_action,
            exploration_seed=exploration_seed,
        ),
        lambda raw: parse_conceptual_scan(
            raw,
            product_description=normalized.product_description,
            brand_slogan=selected_slogan.brand_slogan,
            implied_action=selected_slogan.implied_action,
        ),
        repair_builder=lambda broken, reasons: build_conceptual_scan_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
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

    def _parse_conc_selection(raw: object):
        return parse_conceptual_selection(raw, conceptual_candidates)

    _, selected_conceptual = _run_stage(
        "conceptual_selection",
        model_caller,
        STAGE_CONCEPTUAL_SELECT_SYSTEM,
        build_conceptual_select_user_prompt(conc_dicts),
        _parse_conc_selection,
    )
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

    series_ads = _run_stage(
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
            product_name=product_name_resolved,
            brand_slogan=selected_slogan.brand_slogan,
            model_caller=model_caller,
        )
    except StageParseError as exc:
        raise Builder1PlannerError("marketing_text_failed") from exc

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

    judge_result = judge_builder1_strategy(
        product_description=normalized.product_description,
        plan_dict=series_plan_to_store_dict(plan),
        model_caller=model_caller,
    )

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
        judge_result=judge_result,
    )


def apply_targeted_judge_repair(
    ctx: Builder1PipelineContext,
    *,
    normalized: Any,
    product_name_resolved: str,
    detected_language: str,
    model_caller: Any,
    brand_guidelines: Optional[Dict[str, Any]],
    repair_stage: str,
    rejection_codes: List[str],
) -> Builder1PipelineContext:
    from engine.builder1_planner import (
        Builder1PlannerError,
        _brand_physical_to_dict,
        _conceptual_to_dict,
        _graphic_to_dict,
        _run_stage,
    )

    selected_strategy = ctx.selected_strategy
    selected_slogan = ctx.selected_slogan
    selected_conceptual = ctx.selected_conceptual
    slogan_dicts = list(ctx.slogan_dicts)
    conc_dicts = list(ctx.conc_dicts)
    conceptual_fixed = dict(ctx.conceptual_fixed)
    brand_physical = ctx.brand_physical
    brand_physical_dict = dict(ctx.brand_physical_dict)
    graphic = ctx.graphic
    graphic_dict = dict(ctx.graphic_dict)
    series_ads = ctx.series_ads

    if repair_stage == "slogan_scan":
        slogan_candidates = _run_stage(
            "slogan_scan",
            model_caller,
            STAGE_SLOGAN_SCAN_SYSTEM,
            build_slogan_scan_repair_prompt(
                broken_json=json.dumps({"candidates": slogan_dicts}, ensure_ascii=False),
                reasons=rejection_codes,
            ),
            parse_slogan_scan,
        )
        selected_slogan, slogan_candidates = execute_slogan_scan_through_selection(
            slogan_candidates=slogan_candidates,
            selected_strategy=selected_strategy,
            product_name_resolved=product_name_resolved,
            product_description=normalized.product_description,
            detected_language=detected_language,
            model_caller=model_caller,
            run_stage=_run_stage,
            full_rescan_used=True,
        )
        slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]
    elif repair_stage == "conceptual_scan":
        conceptual_candidates = _run_stage(
            "conceptual_scan",
            model_caller,
            STAGE_CONCEPTUAL_SCAN_SYSTEM,
            build_conceptual_scan_repair_prompt(
                broken_json=json.dumps({"candidates": conc_dicts}, ensure_ascii=False),
                reasons=rejection_codes,
            ),
            lambda raw: parse_conceptual_scan(
                raw,
                product_description=normalized.product_description,
                brand_slogan=selected_slogan.brand_slogan,
                implied_action=selected_slogan.implied_action,
            ),
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

        def _parse_conc_selection(raw: object):
            return parse_conceptual_selection(raw, conceptual_candidates)

        _, selected_conceptual = _run_stage(
            "conceptual_selection",
            model_caller,
            STAGE_CONCEPTUAL_SELECT_SYSTEM,
            build_conceptual_select_user_prompt(conc_dicts),
            _parse_conc_selection,
        )
        conceptual_fixed = _conceptual_to_dict(selected_conceptual)
    elif repair_stage == "strategy_scan":
        raise Builder1PlannerError("final_judge_failed")
    elif repair_stage == "brand_physical":
        brand_physical = _run_stage(
            "brand_physical",
            model_caller,
            STAGE_BRAND_PHYSICAL_SYSTEM,
            build_brand_physical_repair_prompt(
                broken_json=json.dumps(brand_physical_dict, ensure_ascii=False),
                reasons=rejection_codes,
            ),
            lambda raw: parse_brand_physical_output(
                raw, product_description=normalized.product_description
            ),
        )
        brand_physical = enforce_authoritative_product_name(
            brand_physical,
            product_name_resolved=product_name_resolved,
        )
        brand_physical_dict = _brand_physical_to_dict(brand_physical)
    elif repair_stage == "graphic_system":
        graphic = _run_stage(
            "graphic_system",
            model_caller,
            STAGE_GRAPHIC_SYSTEM_SYSTEM,
            build_graphic_system_repair_prompt(
                broken_json=json.dumps(graphic_dict, ensure_ascii=False),
                reasons=rejection_codes,
            ),
            parse_graphic_system_output,
        )
        graphic_dict = _graphic_to_dict(graphic)
    elif repair_stage == "marketing_text":
        series_ads.ads = ensure_series_ads_marketing_text(
            series_ads.ads,
            detected_language=detected_language,
            relative_advantage=selected_strategy.relative_advantage,
            product_name=product_name_resolved,
            brand_slogan=selected_slogan.brand_slogan,
            model_caller=model_caller,
        )
    else:
        series_ads = _run_stage(
            "series_ads",
            model_caller,
            STAGE_SERIES_ADS_SYSTEM,
            build_series_ads_repair_prompt(
                broken_json=json.dumps(
                    {"seriesGenerator": series_ads.series_generator, "ads": series_ads.ads},
                    ensure_ascii=False,
                ),
                reasons=rejection_codes,
                ad_count=normalized.ad_count,
            ),
            lambda raw: parse_series_ads_output(
                raw,
                expected_ad_count=normalized.ad_count,
                product_description=normalized.product_description,
            ),
        )
        series_ads.ads = ensure_series_ads_marketing_text(
            series_ads.ads,
            detected_language=detected_language,
            relative_advantage=selected_strategy.relative_advantage,
            product_name=product_name_resolved,
            brand_slogan=selected_slogan.brand_slogan,
            model_caller=model_caller,
        )

    plan = assemble_builder1_campaign(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        ad_count=normalized.ad_count,
        detected_language=detected_language,
        exploration_seed=ctx.exploration_seed,
        product_name_resolved=product_name_resolved,
        strategy=selected_strategy,
        strategy_selection=ctx.strategy_selection,
        selected_slogan=selected_slogan,
        conceptual=selected_conceptual,
        brand_physical=brand_physical,
        graphic=graphic,
        series_ads=series_ads,
    )
    judge_result = judge_builder1_strategy(
        product_description=normalized.product_description,
        plan_dict=series_plan_to_store_dict(plan),
        model_caller=model_caller,
    )
    if not judge_result.passed:
        raise Builder1PlannerError("final_judge_failed")

    return Builder1PipelineContext(
        exploration_seed=ctx.exploration_seed,
        lens_order=ctx.lens_order,
        strategy_selection=ctx.strategy_selection,
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
        judge_result=judge_result,
    )
