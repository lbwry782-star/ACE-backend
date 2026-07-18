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
from engine.builder1_creative_methodology import is_foundational_strategic_rejection, methodology_repair_stage
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_SLOGAN_SCAN_SYSTEM,
    STAGE_SLOGAN_SELECT_SYSTEM,
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
    build_slogan_scan_repair_prompt,
    build_slogan_scan_user_prompt,
    build_slogan_select_user_prompt,
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
from engine.builder1_slogan_stage import (
    SloganCandidate,
    parse_slogan_scan,
    parse_slogan_selection,
    validate_selected_slogan,
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
        "whyItExpressesSlogan": c.why_it_expresses_slogan,
        "whyItExpressesAdvantage": c.why_it_expresses_advantage,
        "seriesPotential": c.series_potential,
        "brandOwnershipPotential": c.brand_ownership_potential,
    }


def _brand_physical_to_dict(bp: BrandPhysicalOutput) -> Dict[str, Any]:
    return {
        "productNameResolved": bp.product_name_resolved,
        "physicalGenerator": bp.physical_generator,
        "physicalGeneratorNaturalPurpose": bp.physical_generator_natural_purpose,
        "physicalGeneratorCampaignRole": bp.physical_generator_campaign_role,
        "embodimentChoice": bp.embodiment_choice,
        "productVisibilityJustification": bp.product_visibility_justification,
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
        "sloganPlacementReason": graphic.slogan_placement_reason,
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
    methodology_stage = methodology_repair_stage(codes)
    if methodology_stage:
        return methodology_stage
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
    campaign_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Builder1SeriesPlan:
    """Plan one Builder1 campaign via staged pipeline. No creative-output memory."""
    from engine.builder1_planning_pipeline import (
        apply_targeted_judge_repair,
        run_builder1_campaign_pipeline,
    )

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

    strategic_restart_used = False
    ctx = run_builder1_campaign_pipeline(
        normalized=normalized,
        product_name_resolved=product_name_resolved,
        detected_language=detected_language,
        exploration_seed=exploration_seed,
        lens_order=lens_order,
        model_caller=model_caller,
        brand_guidelines=brand_guidelines,
    )

    while not ctx.judge_result.passed:
        codes = ctx.judge_result.rejection_reason_codes
        if is_foundational_strategic_rejection(codes):
            if strategic_restart_used:
                logger.error(
                    "BUILDER1_STRATEGIC_RESTART_FAILED campaignId=%s jobId=%s codes=%s",
                    campaign_id or "",
                    job_id or "",
                    codes,
                )
                raise Builder1PlannerError("planning_failed")
            logger.info(
                "BUILDER1_STRATEGIC_RESTART_START campaignId=%s jobId=%s",
                campaign_id or "",
                job_id or "",
            )
            logger.info(
                "BUILDER1_STRATEGIC_RESTART_REASON campaignId=%s jobId=%s codes=%s",
                campaign_id or "",
                job_id or "",
                codes,
            )
            exploration_seed = str(uuid.uuid4())
            lens_order = shuffled_exploration_lens_order()
            strategic_restart_used = True
            ctx = run_builder1_campaign_pipeline(
                normalized=normalized,
                product_name_resolved=product_name_resolved,
                detected_language=detected_language,
                exploration_seed=exploration_seed,
                lens_order=lens_order,
                model_caller=model_caller,
                brand_guidelines=brand_guidelines,
            )
            if ctx.judge_result.passed:
                logger.info(
                    "BUILDER1_STRATEGIC_RESTART_OK campaignId=%s jobId=%s seed=%s",
                    campaign_id or "",
                    job_id or "",
                    ctx.exploration_seed,
                )
                logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", ctx.plan.ad_count)
                return ctx.plan
            if is_foundational_strategic_rejection(ctx.judge_result.rejection_reason_codes):
                logger.error(
                    "BUILDER1_STRATEGIC_RESTART_FAILED campaignId=%s jobId=%s codes=%s",
                    campaign_id or "",
                    job_id or "",
                    ctx.judge_result.rejection_reason_codes,
                )
                raise Builder1PlannerError("planning_failed")
            codes = ctx.judge_result.rejection_reason_codes

        repair_stage = _judge_repair_stage(codes)
        logger.info(
            "BUILDER1_STAGE_REPAIR stage=%s_judge reasons=%s",
            repair_stage,
            codes,
        )
        try:
            ctx = apply_targeted_judge_repair(
                ctx,
                normalized=normalized,
                product_name_resolved=product_name_resolved,
                detected_language=detected_language,
                model_caller=model_caller,
                brand_guidelines=brand_guidelines,
                repair_stage=repair_stage or "series_ads",
                rejection_codes=codes,
            )
            break
        except Builder1PlannerError:
            raise
        except Exception as exc:
            raise Builder1PlannerError("final_judge_failed") from exc

    if strategic_restart_used and ctx.judge_result.passed:
        logger.info(
            "BUILDER1_STRATEGIC_RESTART_OK campaignId=%s jobId=%s seed=%s",
            campaign_id or "",
            job_id or "",
            ctx.exploration_seed,
        )

    logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", ctx.plan.ad_count)
    return ctx.plan
