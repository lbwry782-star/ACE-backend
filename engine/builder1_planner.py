"""
Builder1 campaign-series planning entry point (staged pipeline).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, List, Optional, TypeAlias

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_planning_contract import (
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_FINAL_CAMPAIGN_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    build_conceptual_scan_repair_prompt,
    build_conceptual_scan_user_prompt,
    build_conceptual_select_user_prompt,
    build_final_campaign_repair_prompt,
    build_final_campaign_user_prompt,
    build_strategy_scan_repair_prompt,
    build_strategy_scan_user_prompt,
    build_strategy_select_user_prompt,
    shuffled_exploration_lens_order,
)
from engine.builder1_staged_parsers import (
    StageParseError,
    assemble_builder1_series_plan,
    detect_brief_language,
    parse_conceptual_scan,
    parse_conceptual_selection,
    parse_final_campaign_output,
    parse_strategy_scan,
    parse_strategy_selection,
)
from engine.builder1_strategy_judge import judge_builder1_strategy
from engine.builder1_plan_spec import series_plan_to_store_dict

logger = logging.getLogger(__name__)

PlanningModelCaller: TypeAlias = Callable[[str, str], object]


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


def _raw_to_text(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


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
            raw = model_caller(system_prompt, current_prompt)
            last_raw = _raw_to_text(raw)
            result = parse_fn(raw)
            logger.info("BUILDER1_STAGE_PARSE_OK stage=%s", stage)
            return result
        except StageParseError as exc:
            last_reasons = exc.reasons
            logger.error("BUILDER1_STAGE_FAILED stage=%s reasons=%s", stage, last_reasons)
            if repair_builder and attempt == 1:
                logger.info("BUILDER1_STAGE_REPAIR stage=%s", stage)
                current_prompt = repair_builder(last_raw, last_reasons)
                try:
                    raw = model_caller(system_prompt, current_prompt)
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

    # Stage 1 — strategy scan
    strategy_candidates = _run_stage(
        "strategy_scan",
        model_caller,
        STAGE_STRATEGY_SCAN_SYSTEM,
        build_strategy_scan_user_prompt(
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            detected_language=detected_language,
            lens_order=lens_order,
            exploration_seed=exploration_seed,
        ),
        lambda raw: parse_strategy_scan(raw, product_description=normalized.product_description),
        repair_builder=lambda broken, reasons: build_strategy_scan_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )

    # Stage 2 — strategy selection
    cand_dicts = [
        {
            "id": c.id,
            "lens": c.lens,
            "strategicProblem": c.strategic_problem,
            "relativeAdvantage": c.relative_advantage,
            "briefSupport": c.brief_support,
            "advantageSource": c.advantage_source,
            "claimRisk": c.claim_risk,
        }
        for c in strategy_candidates
    ]

    def _parse_selection(raw: object):
        sel, selected = parse_strategy_selection(raw, strategy_candidates)
        return sel, selected

    selection_result = _run_stage(
        "strategy_selection",
        model_caller,
        STAGE_STRATEGY_SELECT_SYSTEM,
        build_strategy_select_user_prompt(cand_dicts, exploration_seed),
        _parse_selection,
    )
    strategy_selection, selected_strategy = selection_result
    logger.info(
        "BUILDER1_STRATEGY_SELECTED id=%s family=%s",
        selected_strategy.id,
        strategy_selection.strategy_family,
    )

    # Stage 3 — conceptual scan
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
        parse_conceptual_scan,
        repair_builder=lambda broken, reasons: build_conceptual_scan_repair_prompt(
            broken_json=broken, reasons=reasons
        ),
    )

    # Stage 4 — conceptual selection
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
        sel, selected = parse_conceptual_selection(raw, conceptual_candidates)
        return sel, selected

    conc_selection_result = _run_stage(
        "conceptual_selection",
        model_caller,
        STAGE_CONCEPTUAL_SELECT_SYSTEM,
        build_conceptual_select_user_prompt(conc_dicts),
        _parse_conc_selection,
    )
    _, selected_conceptual = conc_selection_result

    conceptual_fixed = _conceptual_to_dict(selected_conceptual)

    # Stage 5 — final campaign construction
    def _parse_final(raw: object):
        creative, reasons = parse_final_campaign_output(
            raw, expected_ad_count=normalized.ad_count
        )
        if reasons:
            raise StageParseError("final_campaign", reasons)
        return creative

    final_user = build_final_campaign_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        ad_count=normalized.ad_count,
        format_value=normalized.format,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brief_support=selected_strategy.brief_support,
        advantage_source=selected_strategy.advantage_source,
        conceptual=conceptual_fixed,
        brand_guidelines=brand_guidelines,
    )

    try:
        final_creative = _run_stage(
            "final_campaign",
            model_caller,
            STAGE_FINAL_CAMPAIGN_SYSTEM,
            final_user,
            _parse_final,
            repair_builder=lambda broken, reasons: build_final_campaign_repair_prompt(
                broken_json=broken,
                reasons=reasons,
                ad_count=normalized.ad_count,
                strategic_problem=selected_strategy.strategic_problem,
                relative_advantage=selected_strategy.relative_advantage,
                conceptual=conceptual_fixed,
            ),
        )
    except Builder1PlannerError:
        raise

    plan = assemble_builder1_series_plan(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        ad_count=normalized.ad_count,
        detected_language=detected_language,
        exploration_seed=exploration_seed,
        strategy=selected_strategy,
        strategy_selection=strategy_selection,
        conceptual=selected_conceptual,
        final_creative=final_creative,
    )

    # Stage 6 — judge (content quality only)
    plan_dict = series_plan_to_store_dict(plan)
    judge_result = judge_builder1_strategy(
        product_description=normalized.product_description,
        plan_dict=plan_dict,
        model_caller=model_caller,
    )
    if not judge_result.passed:
        logger.info("BUILDER1_STAGE_REPAIR stage=final_campaign_judge reasons=%s", judge_result.rejection_reason_codes)
        repair_prompt = build_final_campaign_repair_prompt(
            broken_json=json.dumps(final_creative, ensure_ascii=False),
            reasons=judge_result.rejection_reason_codes,
            ad_count=normalized.ad_count,
            strategic_problem=selected_strategy.strategic_problem,
            relative_advantage=selected_strategy.relative_advantage,
            conceptual=conceptual_fixed,
        )
        try:
            raw = model_caller(STAGE_FINAL_CAMPAIGN_SYSTEM, repair_prompt)
            final_creative = _parse_final(raw)
            plan = assemble_builder1_series_plan(
                product_name=normalized.product_name,
                product_description=normalized.product_description,
                format_value=normalized.format,
                ad_count=normalized.ad_count,
                detected_language=detected_language,
                exploration_seed=exploration_seed,
                strategy=selected_strategy,
                strategy_selection=strategy_selection,
                conceptual=selected_conceptual,
                final_creative=final_creative,
            )
            judge2 = judge_builder1_strategy(
                product_description=normalized.product_description,
                plan_dict=series_plan_to_store_dict(plan),
                model_caller=model_caller,
            )
            if not judge2.passed:
                raise Builder1PlannerError("strategy_judge_failed")
        except Builder1PlannerError:
            raise
        except Exception as exc:
            raise Builder1PlannerError("strategy_judge_failed") from exc

    logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", plan.ad_count)
    return plan
