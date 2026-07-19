"""
Builder1 campaign-series planning entry point (staged pipeline).
"""
from __future__ import annotations

import inspect
import json
import logging
import time
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
from engine.builder1_planning_metrics import (
    Builder1PlanningMetrics,
    get_planning_metrics,
    reset_planning_metrics,
    set_planning_metrics,
)
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
from engine.builder1_strategy_selection import StrategySelectionExhausted
from engine.builder1_strategy_scan import (
    ensure_strategy_scan_from_raw,
    is_global_strategy_scan_failure,
)

logger = logging.getLogger(__name__)

PlanningModelCaller: TypeAlias = Callable[..., object]


class Builder1PlannerError(RuntimeError):
    def __init__(self, message: str, *, reasons: Optional[List[str]] = None, stage: Optional[str] = None):
        self.reasons = list(reasons or [])
        self.stage = stage
        super().__init__(message)


def _conceptual_to_dict(c: Any) -> dict[str, str]:
    return {
        "generator": c.generator,
        "action": c.action,
        "input": c.input,
        "transformation": c.transformation,
        "result": c.result,
        "perceptionToCreate": getattr(c, "perception_to_create", "") or c.result,
        "impliedPhysicalLaw": getattr(c, "implied_physical_law", "") or c.action,
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
        "physicalGeneratorIsProduct": bp.physical_generator_is_product,
        "physicalGeneratorIsPackaging": bp.physical_generator_is_packaging,
        "worksWithoutProductVisible": bp.works_without_product_visible,
        "transferredObject": bp.transferred_object,
        "transferredObjectAction": bp.transferred_object_action,
        "whyClearerThanShowingProduct": bp.why_clearer_than_showing_product,
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
    metrics = get_planning_metrics()
    if metrics is not None:
        metrics.record_model_call(stage)
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
        metrics = get_planning_metrics()
        if metrics is not None and attempt == 2:
            metrics.record_stage_retry(stage)
        if metrics is not None:
            metrics.begin_stage(stage, attempt=attempt)
        logger.info("BUILDER1_STAGE_START stage=%s attempt=%s", stage, attempt)
        started_at = time.perf_counter()
        try:
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
                    if metrics is not None:
                        metrics.record_stage_repair(stage)
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
        finally:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "BUILDER1_STAGE_DURATION stage=%s attempt=%s durationMs=%s",
                stage,
                attempt,
                duration_ms,
            )
            if metrics is not None:
                metrics.end_stage(stage, attempt=attempt)
    raise Builder1PlannerError(
        f"{stage}_failed: {';'.join(last_reasons)}",
        reasons=last_reasons,
        stage=stage,
    )


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

    def _strategy_scan_replacement_caller(system: str, user: str, **kwargs: Any) -> object:
        return _invoke_model_caller(
            model_caller,
            system,
            user,
            stage=kwargs.get("stage") or "strategy_candidate_repair",
        )

    for attempt in (1, 2):
        metrics = get_planning_metrics()
        if metrics is not None:
            metrics.begin_stage("strategy_scan", attempt=attempt)
        logger.info("BUILDER1_STAGE_START stage=strategy_scan attempt=%s", attempt)
        try:
            raw = _invoke_model_caller(
                model_caller,
                system_prompt,
                user_prompt,
                stage="strategy_scan",
            )
            result = ensure_strategy_scan_from_raw(
                raw,
                product_name=product_name,
                product_description=product_description,
                model_caller=_strategy_scan_replacement_caller,
            )
            if metrics is not None:
                metrics.end_stage("strategy_scan", attempt=attempt)
            return result
        except StrictSchemaConfigurationError as exc:
            logger.error(
                "BUILDER1_STRICT_SCHEMA_INVALID stage=strategy_scan paths=%s",
                exc.errors[:5],
            )
            raise Builder1PlannerError("strategy_scan_failed") from exc
        except StageParseError as exc:
            last_reasons = exc.reasons
            if exc.stage == "strategy_candidate_repair":
                if metrics is not None:
                    metrics.end_stage("strategy_scan", attempt=attempt)
                raise Builder1PlannerError("strategy_candidate_repair_failed") from exc
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
                    if exc2.stage == "strategy_candidate_repair":
                        if metrics is not None:
                            metrics.end_stage("strategy_scan", attempt=attempt)
                        raise Builder1PlannerError("strategy_candidate_repair_failed") from exc2
                    logger.error(
                        "BUILDER1_STAGE_FAILED stage=strategy_scan global_repair reasons=%s",
                        last_reasons,
                    )
            elif not is_global_strategy_scan_failure(last_reasons):
                if metrics is not None:
                    metrics.end_stage("strategy_scan", attempt=attempt)
                raise Builder1PlannerError("strategy_scan_failed") from exc
            if attempt == 2:
                if metrics is not None:
                    metrics.end_stage("strategy_scan", attempt=attempt)
                raise Builder1PlannerError("strategy_scan_failed") from exc
            if metrics is not None:
                metrics.end_stage("strategy_scan", attempt=attempt)
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
    """Plan one Builder1 campaign via consolidated staged pipeline."""
    from engine.builder1_planning_pipeline import run_builder1_campaign_pipeline

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

    from engine.builder1_product_visibility import (
        ProductVisibilityPolicy,
        ProductVisibilitySource,
        derive_product_visibility_policy,
        log_builder1_product_visibility_policy,
    )

    visibility_decision = derive_product_visibility_policy(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        brand_guidelines=brand_guidelines,
    )
    log_builder1_product_visibility_policy(
        campaign_id=campaign_id or "",
        policy=visibility_decision.policy,
        source=visibility_decision.source,
    )

    metrics = Builder1PlanningMetrics(
        campaign_id=campaign_id or "",
        job_id=job_id or "",
        product_name_call_used=not bool(normalized.product_name),
    )
    metrics_token = set_planning_metrics(metrics)

    try:
        product_name_resolved = _resolve_builder1_product_name(
            model_caller,
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            detected_language=detected_language,
            brand_guidelines=brand_guidelines,
        )

        metrics.begin_pipeline_pass("initial")
        try:
            ctx = run_builder1_campaign_pipeline(
                normalized=normalized,
                product_name_resolved=product_name_resolved,
                detected_language=detected_language,
                exploration_seed=exploration_seed,
                lens_order=lens_order,
                model_caller=model_caller,
                brand_guidelines=brand_guidelines,
                visibility_decision=visibility_decision,
            )
        except StrategySelectionExhausted as exc:
            raise Builder1PlannerError("strategy_stage_failed") from exc
        finally:
            metrics.end_pipeline_pass()

        logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", ctx.plan.ad_count)
        return ctx.plan
    finally:
        metrics.log_summary()
        reset_planning_metrics(metrics_token)
