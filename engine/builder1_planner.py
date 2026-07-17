"""
Builder1 campaign-series planning entry point (staged pipeline).
"""
from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeAlias

from engine.builder1_final_stages import (
    BrandPhysicalOutput,
    SeriesAdsOutput,
    assemble_builder1_campaign,
    parse_brand_physical_output,
    parse_graphic_system_output,
    parse_series_ads_output,
)
from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_to_store_dict
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    build_brand_physical_repair_prompt,
    build_brand_physical_user_prompt,
    build_conceptual_scan_repair_prompt,
    build_conceptual_scan_user_prompt,
    build_conceptual_select_user_prompt,
    build_graphic_system_repair_prompt,
    build_graphic_system_user_prompt,
    build_product_name_resolution_repair_prompt,
    build_product_name_resolution_user_prompt,
    build_series_ads_repair_prompt,
    build_series_ads_user_prompt,
    build_strategy_scan_repair_prompt,
    build_strategy_scan_user_prompt,
    build_strategy_select_user_prompt,
    shuffled_exploration_lens_order,
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
)
from engine.builder1_marketing_text_repair import ensure_series_ads_marketing_text
from engine.builder1_product_name import (
    enforce_authoritative_product_name,
    parse_product_name_resolution,
)
from engine.builder1_strict_schema import StrictSchemaConfigurationError
from engine.builder1_staged_parsers import (
    StageParseError,
    detect_brief_language,
    filter_eligible_strategy_candidates,
    parse_conceptual_scan,
    parse_conceptual_selection,
    parse_strategy_selection,
)
from engine.builder1_strategy_scan import (
    STRATEGY_SCAN_REPLACEMENT_SYSTEM,
    ensure_strategy_scan_from_raw,
    is_global_strategy_scan_failure,
)
from engine.builder1_strategy_judge import (
    is_client_boundary_rejection,
    is_marketing_language_rejection,
    is_marketing_word_count_rejection,
    is_no_logo_rejection_code,
    judge_builder1_strategy,
)

logger = logging.getLogger(__name__)

PlanningModelCaller: TypeAlias = Callable[..., object]


class Builder1PlannerError(RuntimeError):
    pass


def _conceptual_to_dict(c: Any) -> dict[str, str]:
    return {
        "generator": c.generator,
        "action": c.action,
        "input": c.input,
        "transformation": c.transformation,
        "result": c.result,
        "whyItExpressesAdvantage": c.why_it_expresses_advantage,
        "seriesPotential": c.series_potential,
    }


def _brand_physical_to_dict(bp: BrandPhysicalOutput) -> Dict[str, Any]:
    return {
        "productNameResolved": bp.product_name_resolved,
        "brandSlogan": bp.brand_slogan,
        "sloganDerivation": bp.slogan_derivation,
        "sloganAction": bp.slogan_action,
        "physicalGenerator": bp.physical_generator,
        "physicalGeneratorNaturalPurpose": bp.physical_generator_natural_purpose,
        "physicalGeneratorCampaignRole": bp.physical_generator_campaign_role,
        "mediumParticipates": bp.medium_participates,
        "mediumRole": bp.medium_role,
        "campaignRationale": bp.campaign_rationale,
    }


def _graphic_to_dict(graphic: Any) -> Dict[str, Any]:
    return {
        "palette": {
            "dominant": graphic.palette.dominant,
            "secondary": graphic.palette.secondary,
            "accent": graphic.palette.accent,
            "background": graphic.palette.background,
            "text": graphic.palette.text,
        },
        "layoutTemplate": graphic.layout_template,
        "headlinePlacement": graphic.headline_placement,
        "headlineAlignment": graphic.headline_alignment,
        "headlineMaxWidthPercent": graphic.headline_max_width_percent,
        "brandBlockPlacement": graphic.brand_block_placement,
        "sloganPlacement": graphic.slogan_placement,
        "copySafeArea": {
            "side": graphic.copy_safe_area.side,
            "widthPercent": graphic.copy_safe_area.width_percent,
        },
        "typographyStyle": graphic.typography_style,
        "headlineScale": graphic.headline_scale,
        "brandScale": graphic.brand_scale,
        "sloganScale": graphic.slogan_scale,
        "imageStyle": graphic.image_style,
        "backgroundTreatment": graphic.background_treatment,
        "borderTreatment": graphic.border_treatment,
        "recurringGraphicDevice": graphic.recurring_graphic_device,
        "recurringGraphicDeviceRule": graphic.recurring_graphic_device_rule,
        "shapeLanguage": graphic.shape_language,
        "framingRule": graphic.framing_rule,
        "spacingRule": graphic.spacing_rule,
    }


def _raw_to_text(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _invoke_model_caller(
    model_caller: PlanningModelCaller,
    system_prompt: str,
    user_prompt: str,
    *,
    stage: Optional[str] = None,
) -> object:
    try:
        sig = inspect.signature(model_caller)
    except (TypeError, ValueError):
        return model_caller(system_prompt, user_prompt)
    if "stage" in sig.parameters:
        return model_caller(system_prompt, user_prompt, stage=stage)
    return model_caller(system_prompt, user_prompt)


def _run_stage(
    stage: str,
    model_caller: PlanningModelCaller,
    system_prompt: str,
    user_prompt: str,
    parse_fn: Callable[[object], Any],
    *,
    repair_builder: Optional[Callable[[str, List[str]], str]] = None,
) -> Any:
    last_reasons: List[str] = []
    last_raw = ""
    current_prompt = user_prompt
    for attempt in (1, 2):
        logger.info("BUILDER1_STAGE_START stage=%s attempt=%s", stage, attempt)
        try:
            raw = _invoke_model_caller(
                model_caller, system_prompt, current_prompt, stage=stage
            )
            last_raw = _raw_to_text(raw)
            result = parse_fn(raw)
            logger.info("BUILDER1_STAGE_PARSE_OK stage=%s", stage)
            return result
        except StrictSchemaConfigurationError as exc:
            logger.error(
                "BUILDER1_STRICT_SCHEMA_INVALID stage=%s paths=%s",
                stage,
                exc.errors[:5],
            )
            raise Builder1PlannerError(f"{stage}_failed") from exc
        except StageParseError as exc:
            last_reasons = exc.reasons
            logger.error("BUILDER1_STAGE_FAILED stage=%s reasons=%s", stage, last_reasons)
            if repair_builder and attempt == 1:
                logger.info("BUILDER1_STAGE_REPAIR stage=%s", stage)
                current_prompt = repair_builder(last_raw, last_reasons)
                try:
                    raw = _invoke_model_caller(
                        model_caller, system_prompt, current_prompt, stage=stage
                    )
                    last_raw = _raw_to_text(raw)
                    result = parse_fn(raw)
                    logger.info("BUILDER1_STAGE_PARSE_OK stage=%s after_repair", stage)
                    return result
                except StageParseError as exc2:
                    last_reasons = exc2.reasons
                    logger.error("BUILDER1_STAGE_FAILED stage=%s repair reasons=%s", stage, last_reasons)
            logger.info("BUILDER1_STAGE_RETRY stage=%s", stage)
            current_prompt = user_prompt
        except Exception as exc:
            logger.error("BUILDER1_STAGE_FAILED stage=%s err=%s", stage, exc)
            if attempt == 2:
                raise Builder1PlannerError(f"{stage}_failed") from exc
            logger.info("BUILDER1_STAGE_RETRY stage=%s", stage)
    raise Builder1PlannerError(f"{stage}_failed: {';'.join(last_reasons)}")


def _run_strategy_scan_stage(
    model_caller: PlanningModelCaller,
    *,
    product_name: str,
    product_description: str,
    detected_language: str,
    lens_order: List[str],
    exploration_seed: str,
) -> List[Any]:
    system_prompt = STAGE_STRATEGY_SCAN_SYSTEM
    user_prompt = build_strategy_scan_user_prompt(
        product_name=product_name,
        product_description=product_description,
        detected_language=detected_language,
        lens_order=lens_order,
        exploration_seed=exploration_seed,
    )
    last_reasons: List[str] = []
    raw: object = {}

    def _strategy_scan_replacement_caller(system: str, user: str) -> object:
        return _invoke_model_caller(
            model_caller,
            system,
            user,
            stage="strategy_scan",
        )

    for attempt in (1, 2):
        logger.info("BUILDER1_STAGE_START stage=strategy_scan attempt=%s", attempt)
        try:
            raw = _invoke_model_caller(
                model_caller,
                system_prompt,
                user_prompt,
                stage="strategy_scan",
            )
            return ensure_strategy_scan_from_raw(
                raw,
                product_name=product_name,
                product_description=product_description,
                model_caller=_strategy_scan_replacement_caller,
            )
        except StrictSchemaConfigurationError as exc:
            logger.error(
                "BUILDER1_STRICT_SCHEMA_INVALID stage=strategy_scan paths=%s",
                exc.errors[:5],
            )
            raise Builder1PlannerError("strategy_scan_failed") from exc
        except StageParseError as exc:
            last_reasons = exc.reasons
            logger.error("BUILDER1_STAGE_FAILED stage=strategy_scan reasons=%s", last_reasons)
            if is_global_strategy_scan_failure(last_reasons) and attempt == 1:
                logger.info("BUILDER1_STAGE_REPAIR stage=strategy_scan global")
                repair_prompt = build_strategy_scan_repair_prompt(
                    broken_json=_raw_to_text(raw),
                    reasons=last_reasons,
                )
                try:
                    raw = _invoke_model_caller(
                        model_caller,
                        system_prompt,
                        repair_prompt,
                        stage="strategy_scan",
                    )
                    return ensure_strategy_scan_from_raw(
                        raw,
                        product_name=product_name,
                        product_description=product_description,
                        model_caller=_strategy_scan_replacement_caller,
                    )
                except StageParseError as exc2:
                    last_reasons = exc2.reasons
                    logger.error(
                        "BUILDER1_STAGE_FAILED stage=strategy_scan global_repair reasons=%s",
                        last_reasons,
                    )
            if attempt == 2:
                raise Builder1PlannerError("strategy_scan_failed") from exc
            logger.info("BUILDER1_STAGE_RETRY stage=strategy_scan")
        except Exception as exc:
            logger.error("BUILDER1_STAGE_FAILED stage=strategy_scan err=%s", exc)
            if attempt == 2:
                raise Builder1PlannerError("strategy_scan_failed") from exc
            logger.info("BUILDER1_STAGE_RETRY stage=strategy_scan")
    raise Builder1PlannerError(f"strategy_scan_failed: {';'.join(last_reasons)}")


def _run_product_name_resolution_stage(
    model_caller: PlanningModelCaller,
    *,
    product_description: str,
    detected_language: str,
    brand_guidelines: Optional[Dict[str, Any]],
) -> str:
    system_prompt = STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM
    user_prompt = build_product_name_resolution_user_prompt(
        product_description=product_description,
        detected_language=detected_language,
        brand_guidelines=brand_guidelines,
    )
    return _run_stage(
        "product_name_resolution",
        model_caller,
        system_prompt,
        user_prompt,
        lambda raw: parse_product_name_resolution(
            raw,
            product_description=product_description,
            detected_language=detected_language,
        ),
        repair_builder=lambda broken, reasons: build_product_name_resolution_repair_prompt(
            broken_json=broken,
            reasons=reasons,
        ),
    )


def _resolve_builder1_product_name(
    model_caller: PlanningModelCaller,
    *,
    product_name: str,
    product_description: str,
    detected_language: str,
    brand_guidelines: Optional[Dict[str, Any]],
) -> str:
    if product_name:
        logger.info("BUILDER1_PRODUCT_NAME_SOURCE source=user")
        return product_name

    logger.info("BUILDER1_PRODUCT_NAME_GENERATION_START")
    try:
        resolved = _run_product_name_resolution_stage(
            model_caller,
            product_description=product_description,
            detected_language=detected_language,
            brand_guidelines=brand_guidelines,
        )
    except Builder1PlannerError as exc:
        logger.error("BUILDER1_PRODUCT_NAME_GENERATION_FAILED")
        raise Builder1PlannerError("product_name_generation_failed") from exc
    except StageParseError as exc:
        logger.error("BUILDER1_PRODUCT_NAME_GENERATION_FAILED reasons=%s", exc.reasons)
        raise Builder1PlannerError("product_name_generation_failed") from exc

    logger.info(
        "BUILDER1_PRODUCT_NAME_GENERATION_OK nameLength=%s lang=%s",
        len(resolved),
        detected_language,
    )
    logger.info("BUILDER1_PRODUCT_NAME_SOURCE source=generated")
    return resolved


def _judge_repair_stage(codes: List[str]) -> Optional[str]:
    if is_marketing_word_count_rejection(codes) or is_marketing_language_rejection(codes):
        return "marketing_text"
    if is_no_logo_rejection_code(codes):
        if any(
            code in codes
            for code in ("supplied_logo_displayed", "product_name_not_text_only")
        ):
            return "brand_physical"
        if "packaging_contains_brand_mark" in codes:
            return "series_ads"
        return "graphic_system"
    if is_client_boundary_rejection(codes):
        joined = " ".join(codes).lower()
        if "unsupported_future_capability" in joined or "marketing" in joined:
            return "series_ads"
        if any(
            code in codes
            for code in (
                "business_transformation_required",
                "advantage_not_currently_true",
                "client_consultation_required",
                "material_client_investment_required",
            )
        ):
            return "strategy_scan"
        return "brand_physical"
    joined = " ".join(codes).lower()
    if any(k in joined for k in ("graphic", "palette", "layout", "typography", "device")):
        return "graphic_system"
    if any(
        k in joined
        for k in (
            "series",
            "conceptual_execution",
            "physical_execution",
            "visual_execution",
            "scene_description",
            "new_contribution",
            "contribution",
        )
    ) and "marketing_copy" not in joined:
        return "series_ads"
    if any(k in joined for k in ("headline_too_long",)):
        return "marketing_text"
    if any(
        k in joined
        for k in ("slogan", "physical", "medium", "rationale", "brand", "unsupported_evidence")
    ):
        return "brand_physical"
    if any(k in joined for k in ("marketing_copy", "marketing_text")):
        return "marketing_text"
    return "series_ads"


def plan_builder1(
    product_name: object,
    product_description: object,
    format_value: object,
    model_caller: PlanningModelCaller,
    *,
    ad_count: int = 2,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> Builder1SeriesPlan:
    """Plan one Builder1 campaign via staged pipeline. No creative-output memory."""
    normalized = normalize_builder1_input(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
    )
    exploration_seed = str(uuid.uuid4())
    lens_order = shuffled_exploration_lens_order()
    detected_language = detect_brief_language(
        normalized.product_description,
        normalized.product_name,
    )

    logger.info(
        "BUILDER1_SERIES_PLANNING_START productName=%s format=%s adCount=%s seed=%s lang=%s",
        normalized.product_name,
        normalized.format,
        normalized.ad_count,
        exploration_seed,
        detected_language,
    )

    product_name_resolved = _resolve_builder1_product_name(
        model_caller,
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        detected_language=detected_language,
        brand_guidelines=brand_guidelines,
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

    conceptual_candidates = _run_stage(
        "conceptual_scan",
        model_caller,
        STAGE_CONCEPTUAL_SCAN_SYSTEM,
        build_conceptual_scan_user_prompt(
            product_description=normalized.product_description,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            exploration_seed=exploration_seed,
        ),
        lambda raw: parse_conceptual_scan(raw, product_description=normalized.product_description),
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
            "whyItExpressesAdvantage": c.why_it_expresses_advantage,
            "seriesPotential": c.series_potential,
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
            brand_slogan=brand_physical.brand_slogan,
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
        repair_stage = _judge_repair_stage(judge_result.rejection_reason_codes)
        logger.info(
            "BUILDER1_STAGE_REPAIR stage=%s_judge reasons=%s",
            repair_stage,
            judge_result.rejection_reason_codes,
        )
        try:
            if repair_stage == "brand_physical":
                brand_physical = _run_stage(
                    "brand_physical",
                    model_caller,
                    STAGE_BRAND_PHYSICAL_SYSTEM,
                    build_brand_physical_repair_prompt(
                        broken_json=json.dumps(brand_physical_dict, ensure_ascii=False),
                        reasons=judge_result.rejection_reason_codes,
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
                        reasons=judge_result.rejection_reason_codes,
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
                    brand_slogan=brand_physical.brand_slogan,
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
                        reasons=judge_result.rejection_reason_codes,
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
                    brand_slogan=brand_physical.brand_slogan,
                    model_caller=model_caller,
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
                conceptual=selected_conceptual,
                brand_physical=brand_physical,
                graphic=graphic,
                series_ads=series_ads,
            )
            judge2 = judge_builder1_strategy(
                product_description=normalized.product_description,
                plan_dict=series_plan_to_store_dict(plan),
                model_caller=model_caller,
            )
            if not judge2.passed:
                raise Builder1PlannerError("final_judge_failed")
        except Builder1PlannerError:
            raise
        except Exception as exc:
            raise Builder1PlannerError("final_judge_failed") from exc

    logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", plan.ad_count)
    return plan
