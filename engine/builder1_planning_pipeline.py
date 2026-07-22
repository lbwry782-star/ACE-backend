"""Builder1 staged campaign pipeline execution (consolidated planning stages)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.builder1_campaign_integrity import (
    make_upstream_snapshot,
    validate_builder1_campaign_integrity,
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
from engine.builder1_idea_memory_pipeline import (
    persist_plan_idea_memory,
    run_brand_physical_with_memory_guard,
    run_conceptual_with_memory_guard,
    run_series_ads_with_memory_guard,
    run_strategy_slogan_with_memory_guard,
    stage_memory_block,
)
from engine.builder1_product_name import enforce_authoritative_product_name
from engine.builder1_product_visibility import (
    derive_product_visibility_policy,
    log_builder1_product_visibility_policy,
)
from engine.builder1_slogan_stage import slogan_candidate_to_dict
from engine.builder1_series_distinctness import (
    duplicate_assembly_reasons,
    validate_ad_execution_distinctness,
)
from engine.builder1_series_execution_repair import attempt_series_execution_repair
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_strategy_selection import StrategySelectionExhausted

logger = logging.getLogger(__name__)


def _run_graphic_system_stage(
    model_caller: Any,
    *,
    user_prompt: str,
    run_stage: Any,
) -> Any:
    from engine.builder1_graphic_contract import is_graphic_contract_mismatch
    from engine.builder1_planning_metrics import get_planning_metrics
    from engine.builder1_planning_profile import (
        execution_optimization_active,
        quality_model,
        resolve_stage_model,
        stage_model_override,
    )
    from engine.builder1_planner import Builder1PlannerError

    def _attempt() -> Any:
        return run_stage(
            "graphic_system",
            model_caller,
            STAGE_GRAPHIC_SYSTEM_SYSTEM,
            user_prompt,
            parse_graphic_system_output,
            repair_builder=lambda broken, reasons: build_graphic_system_repair_prompt(
                broken_json=broken, reasons=reasons
            ),
        )

    try:
        return _attempt()
    except Builder1PlannerError as exc:
        reasons = exc.reasons or []
        if exc.stage != "graphic_system" or not is_graphic_contract_mismatch(reasons):
            raise
        exec_model = resolve_stage_model("graphic_system")
        q_model = quality_model()
        if not execution_optimization_active() or exec_model == q_model:
            logger.error(
                "BUILDER1_GRAPHIC_CONTRACT_UNRESOLVED stage=graphic_system reasons=%s",
                reasons,
            )
            raise
        metrics = get_planning_metrics()
        if metrics is not None:
            metrics.record_stage_model_fallback("graphic_system")
        logger.info(
            "BUILDER1_STAGE_MODEL_FALLBACK stage=graphic_system fromModel=%s toModel=%s "
            "reason=execution_model_contract_failure",
            exec_model,
            q_model,
        )
        with stage_model_override({"graphic_system": q_model}):
            return _attempt()


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
    visibility_decision: Optional[Any] = None,
    campaign_id: str = "",
    idea_memory: Optional[Any] = None,
) -> Builder1PipelineContext:
    from engine.builder1_planner import (
        Builder1PlannerError,
        _brand_physical_to_dict,
        _conceptual_to_dict,
        _graphic_to_dict,
        _run_stage,
    )

    try:
        (
            strategy_selection,
            selected_strategy,
            _strategy_candidates,
            _strategy_reviews,
            _slogan_selection,
            selected_slogan,
            slogan_candidates,
        ) = run_strategy_slogan_with_memory_guard(
            _run_stage,
            model_caller,
            campaign_id=campaign_id,
            idea_memory=idea_memory,
            product_name=product_name_resolved,
            product_name_resolved=product_name_resolved,
            product_description=normalized.product_description,
            detected_language=detected_language,
            lens_order=lens_order,
            exploration_seed=exploration_seed,
            idea_memory_block=stage_memory_block(
                "strategy_slogan_stage", idea_memory, campaign_id=campaign_id
            ),
        )
    except StrategySelectionExhausted as exc:
        raise Builder1PlannerError("strategy_slogan_stage_failed") from exc

    slogan_dicts = [slogan_candidate_to_dict(c) for c in slogan_candidates]

    _conceptual_selection, selected_conceptual, conceptual_candidates = run_conceptual_with_memory_guard(
        _run_stage,
        model_caller,
        campaign_id=campaign_id,
        idea_memory=idea_memory,
        idea_memory_block=stage_memory_block("conceptual_stage", idea_memory, campaign_id=campaign_id),
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

    if visibility_decision is None:
        visibility_decision = derive_product_visibility_policy(
            product_name=product_name_resolved,
            product_description=normalized.product_description,
            brand_guidelines=brand_guidelines,
        )

    log_builder1_product_visibility_policy(
        policy=visibility_decision.policy,
        source=visibility_decision.source,
    )

    brand_physical_user_prompt = build_brand_physical_user_prompt(
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
        visibility_policy=visibility_decision.policy.value,
        idea_memory_block=stage_memory_block("brand_physical", idea_memory, campaign_id=campaign_id),
    )
    brand_physical = run_brand_physical_with_memory_guard(
        model_caller=model_caller,
        run_stage=_run_stage,
        campaign_id=campaign_id,
        idea_memory=idea_memory,
        user_prompt=brand_physical_user_prompt,
        parse_kwargs={
            "product_description": normalized.product_description,
            "product_name_resolved": product_name_resolved,
        },
        visibility_policy=visibility_decision.policy,
        repair_context={
            "strategic_problem": selected_strategy.strategic_problem,
            "relative_advantage": selected_strategy.relative_advantage,
            "brand_slogan": selected_slogan.brand_slogan,
            "implied_action": selected_slogan.implied_action,
            "conceptual": conceptual_fixed,
        },
    )
    brand_physical = enforce_authoritative_product_name(
        brand_physical,
        product_name_resolved=product_name_resolved,
    )
    brand_physical_dict = _brand_physical_to_dict(brand_physical)

    graphic = _run_graphic_system_stage(
        model_caller,
        user_prompt=build_graphic_system_user_prompt(
            product_description=normalized.product_description,
            detected_language=detected_language,
            relative_advantage=selected_strategy.relative_advantage,
            brand_slogan=selected_slogan.brand_slogan,
            conceptual=conceptual_fixed,
            brand_physical=brand_physical_dict,
            format_value=normalized.format,
            idea_memory_block=stage_memory_block("graphic_system", idea_memory, campaign_id=campaign_id),
        ),
        run_stage=_run_stage,
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
        visibility_policy=visibility_decision.policy,
        campaign_id=campaign_id,
        idea_memory=idea_memory,
    )

    plan = _assemble_campaign_with_duplicate_recovery(
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
        visibility_policy=visibility_decision.policy,
        visibility_source=visibility_decision.source,
        conceptual_fixed=conceptual_fixed,
        brand_physical_dict=brand_physical_dict,
        graphic_dict=graphic_dict,
        model_caller=model_caller,
        run_stage=_run_stage,
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
    from engine.builder1_failure_classification import validate_forbidden_plan_visibility

    visibility_conflicts = validate_forbidden_plan_visibility(plan)
    if visibility_conflicts:
        logger.error(
            "BUILDER1_INTEGRITY_FAILED reasons=%s",
            visibility_conflicts,
        )
        raise Builder1PlannerError("campaign_visibility_integrity_failed")
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

    persist_plan_idea_memory(
        plan,
        campaign_id=campaign_id,
        idea_memory=idea_memory,
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
    visibility_policy: Any,
    campaign_id: str = "",
    idea_memory: Optional[Any] = None,
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
            visibility_policy=getattr(visibility_policy, "value", str(visibility_policy)),
            idea_memory_block=stage_memory_block("series_ads", idea_memory, campaign_id=campaign_id),
        ),
        lambda raw: parse_series_ads_output(
            raw,
            expected_ad_count=normalized.ad_count,
            product_description=normalized.product_description,
            visibility_policy=visibility_policy,
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

    series_ads = run_series_ads_with_memory_guard(
        series_ads=series_ads,
        campaign_id=campaign_id,
        idea_memory=idea_memory,
        brand_slogan=selected_slogan.brand_slogan,
        conceptual=conceptual_fixed,
        brand_physical=brand_physical_dict,
        graphic_generator=graphic_dict,
        detected_language=detected_language,
        model_caller=model_caller,
        run_stage=run_stage,
    )

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
            visibility_policy=visibility_policy,
            campaign_id=campaign_id,
            idea_memory=idea_memory,
        )
    return series_ads


def _assemble_campaign_with_duplicate_recovery(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    detected_language: str,
    exploration_seed: str,
    product_name_resolved: str,
    strategy: Any,
    strategy_selection: Any,
    selected_slogan: Any,
    conceptual: Any,
    brand_physical: Any,
    graphic: Any,
    series_ads: SeriesAdsOutput,
    visibility_policy: Any,
    visibility_source: Any,
    conceptual_fixed: Dict[str, str],
    brand_physical_dict: Dict[str, Any],
    graphic_dict: Dict[str, Any],
    model_caller: Any,
    run_stage: Any,
) -> Builder1SeriesPlan:
    from engine.builder1_planner import Builder1PlannerError

    current_series = series_ads
    repair_used = False
    while True:
        try:
            return assemble_builder1_campaign(
                product_name=product_name,
                product_description=product_description,
                format_value=format_value,
                ad_count=ad_count,
                detected_language=detected_language,
                exploration_seed=exploration_seed,
                product_name_resolved=product_name_resolved,
                strategy=strategy,
                strategy_selection=strategy_selection,
                selected_slogan=selected_slogan,
                conceptual=conceptual,
                brand_physical=brand_physical,
                graphic=graphic,
                series_ads=current_series,
                visibility_policy=visibility_policy,
                visibility_source=visibility_source,
            )
        except StageParseError as exc:
            if exc.stage != "assemble":
                raise
            duplicate_reasons = duplicate_assembly_reasons(exc.reasons)
            if not duplicate_reasons or repair_used:
                raise Builder1PlannerError(
                    "series_assembly_failed",
                    reasons=exc.reasons,
                    stage="assemble",
                ) from exc
            _, findings = validate_ad_execution_distinctness(current_series.ads)
            logger.info(
                "BUILDER1_SERIES_EXECUTION_REPAIR_START duplicateReasons=%s",
                duplicate_reasons,
            )
            current_series = attempt_series_execution_repair(
                series_ads=current_series,
                duplicate_reasons=duplicate_reasons,
                findings=findings,
                brand_slogan=selected_slogan.brand_slogan,
                conceptual=conceptual_fixed,
                brand_physical=brand_physical_dict,
                graphic_generator=graphic_dict,
                detected_language=detected_language,
                model_caller=model_caller,
                run_stage=run_stage,
            )
            repair_used = True


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
