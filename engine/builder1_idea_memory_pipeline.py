"""
Builder1 idea-memory integration for the staged planning pipeline.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder1_consolidated_stages import (
    process_strategy_slogan_stage_response,
    run_conceptual_stage,
    run_strategy_slogan_stage,
)
from engine.builder1_final_stages import SeriesAdsOutput
from engine.builder1_idea_memory import (
    HistoricalDuplicateFinding,
    IdeaMemorySnapshot,
    _campaign_idea_payload,
    build_records_from_plan,
    build_stage_memory_block,
    compute_ad_execution_fingerprint,
    compute_campaign_idea_fingerprint,
    find_historical_duplicate,
    log_historical_duplicate,
    persist_idea_memory_records,
    register_completed_campaign_for_backfill,
)
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_planner import Builder1PlannerError
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_STRATEGY_SLOGAN_REPAIR_SYSTEM,
    build_conceptual_stage_user_prompt,
    build_strategy_slogan_repair_user_prompt,
)
from engine.builder1_physical_evaluations import parse_brand_physical_with_evaluation_recovery
from engine.builder1_series_distinctness import DuplicateExecutionFinding
from engine.builder1_series_execution_repair import attempt_series_execution_repair


def campaign_idea_fingerprint_from_upstream(
    *,
    strategic_problem: str,
    relative_advantage: str,
    brand_slogan: str,
    conceptual_generator: str = "",
    physical_generator: str = "",
    transferred_object: str = "",
    transferred_object_action: str = "",
) -> str:
    payload = _campaign_idea_payload(
        strategic_problem=strategic_problem,
        relative_advantage=relative_advantage,
        slogan=brand_slogan,
        conceptual_generator=conceptual_generator,
        physical_generator=physical_generator,
        transferred_object=transferred_object,
        transferred_object_action=transferred_object_action,
    )
    return compute_campaign_idea_fingerprint(payload)


def ad_execution_fingerprint_from_dict(ad: Dict[str, Any]) -> str:
    return compute_ad_execution_fingerprint(ad)


def _historical_repair_prompt(
    *,
    finding: HistoricalDuplicateFinding,
    product_name: str,
    product_description: str,
) -> str:
    return (
        "Historical duplicate detected. Replace ONLY the duplicated creative decision while preserving "
        "all valid frozen upstream constraints.\n"
        f"Duplicate type: {finding.duplicate_type}\n"
        f"Conflicting prior record: {finding.matching_record_id} from campaign {finding.matching_campaign_id}\n"
        f"Fingerprint: {finding.fingerprint}\n"
        "Do not reuse the same underlying campaign idea, conceptual mechanism, physical mechanism, "
        "or ad execution. Wording or palette changes alone are insufficient.\n"
        + build_strategy_slogan_repair_user_prompt(
            broken_json="{}",
            reasons=[f"builder1_historical_idea_duplicate:{finding.duplicate_type}"],
            product_name=product_name,
            product_description=product_description,
        )
    )


def _conceptual_historical_repair_prompt(
    *,
    finding: HistoricalDuplicateFinding,
    base_user_prompt: str,
) -> str:
    return (
        f"{base_user_prompt}\n"
        "Historical duplicate detected for conceptual generator.\n"
        f"Duplicate type: {finding.duplicate_type}\n"
        f"Conflicting prior record: {finding.matching_record_id} from campaign {finding.matching_campaign_id}\n"
        "Replace the selected conceptual generator with a clearly different central mechanism. "
        "Rewording the same mechanism is insufficient.\n"
    )


def _physical_historical_repair_prompt(
    *,
    finding: HistoricalDuplicateFinding,
    base_user_prompt: str,
) -> str:
    return (
        f"{base_user_prompt}\n"
        "Historical duplicate detected for physical generator / transferred action.\n"
        f"Duplicate type: {finding.duplicate_type}\n"
        f"Conflicting prior record: {finding.matching_record_id} from campaign {finding.matching_campaign_id}\n"
        "Replace the duplicated physical mechanism. Scene, crop, or palette changes alone are insufficient.\n"
    )


def run_strategy_slogan_with_memory_guard(
    run_stage: Callable[..., Any],
    model_caller: Any,
    *,
    campaign_id: str,
    idea_memory: Optional[IdeaMemorySnapshot],
    product_name: str,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    lens_order: List[str],
    exploration_seed: str,
    idea_memory_block: str,
):
    result = run_strategy_slogan_stage(
        run_stage,
        model_caller,
        product_name=product_name,
        product_name_resolved=product_name_resolved,
        product_description=product_description,
        detected_language=detected_language,
        lens_order=lens_order,
        exploration_seed=exploration_seed,
        idea_memory_block=idea_memory_block,
    )
    if idea_memory is None:
        return result
    strategy_selection, selected_strategy, sc, sr, slogan_selection, selected_slogan, sgc = result
    finding = find_historical_duplicate(
        stage="strategy_slogan_stage",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brand_slogan=selected_slogan.brand_slogan,
    )
    if not finding:
        return result
    log_historical_duplicate(finding, campaign_id=campaign_id, repair_attempted=True)
    repair_prompt = _historical_repair_prompt(
        finding=finding,
        product_name=product_name_resolved,
        product_description=product_description,
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

    repaired = run_stage(
        "strategy_slogan_repair",
        model_caller,
        STAGE_STRATEGY_SLOGAN_REPAIR_SYSTEM,
        repair_prompt,
        _parse,
    )
    strategy_selection, selected_strategy, sc, sr, slogan_selection, selected_slogan, sgc = repaired
    finding2 = find_historical_duplicate(
        stage="strategy_slogan_stage",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brand_slogan=selected_slogan.brand_slogan,
    )
    if finding2:
        raise Builder1PlannerError("builder1_historical_idea_duplicate")
    return repaired


def run_conceptual_with_memory_guard(
    run_stage: Callable[..., Any],
    model_caller: Any,
    *,
    campaign_id: str,
    idea_memory: Optional[IdeaMemorySnapshot],
    idea_memory_block: str,
    product_description: str,
    product_name_resolved: str,
    selected_strategy: Any,
    selected_slogan: Any,
    exploration_seed: str,
    **kwargs: Any,
):
    from engine.builder1_consolidated_stages import process_conceptual_stage_response

    result = run_conceptual_stage(
        run_stage,
        model_caller,
        product_description=product_description,
        product_name_resolved=product_name_resolved,
        selected_strategy=selected_strategy,
        selected_slogan=selected_slogan,
        exploration_seed=exploration_seed,
        idea_memory_block=idea_memory_block,
    )
    if idea_memory is None:
        return result
    _selection, selected_conceptual, _candidates = result
    finding = find_historical_duplicate(
        stage="conceptual_stage",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        conceptual_generator=selected_conceptual.generator,
    )
    if not finding:
        return result
    log_historical_duplicate(finding, campaign_id=campaign_id, repair_attempted=True)
    base_prompt = build_conceptual_stage_user_prompt(
        product_description=product_description,
        product_name_resolved=product_name_resolved,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brand_slogan=selected_slogan.brand_slogan,
        slogan_derivation=selected_slogan.derivation_from_advantage,
        implied_action=selected_slogan.implied_action,
        exploration_seed=exploration_seed,
        idea_memory_block=idea_memory_block,
    )
    repair_prompt = _conceptual_historical_repair_prompt(finding=finding, base_user_prompt=base_prompt)

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

    repaired = run_stage(
        "conceptual_candidate_repair",
        model_caller,
        STAGE_CONCEPTUAL_STAGE_SYSTEM,
        repair_prompt,
        _parse,
    )
    _selection2, selected_conceptual2, candidates2 = repaired
    finding2 = find_historical_duplicate(
        stage="conceptual_stage",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        conceptual_generator=selected_conceptual2.generator,
    )
    if finding2:
        raise Builder1PlannerError("builder1_historical_idea_duplicate")
    return repaired


def run_brand_physical_with_memory_guard(
    *,
    model_caller: Any,
    run_stage: Callable[..., Any],
    campaign_id: str,
    idea_memory: Optional[IdeaMemorySnapshot],
    user_prompt: str,
    parse_kwargs: Dict[str, Any],
    visibility_policy: Any,
    repair_context: Dict[str, Any],
) -> Any:
    from engine.builder1_physical_repair import _run_brand_physical_with_identity_guard

    brand_physical = _run_brand_physical_with_identity_guard(
        model_caller=model_caller,
        user_prompt=user_prompt,
        parse_kwargs=parse_kwargs,
        visibility_policy=visibility_policy,
        repair_context=repair_context,
    )
    if idea_memory is None:
        return brand_physical

    physical_generator = str(getattr(brand_physical, "physical_generator", "") or "")
    transferred_object = str(getattr(brand_physical, "transferred_object", "") or "")
    transferred_object_action = str(getattr(brand_physical, "transferred_object_action", "") or "")
    finding = find_historical_duplicate(
        stage="brand_physical",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        physical_generator=physical_generator,
        transferred_object=transferred_object,
        transferred_object_action=transferred_object_action,
    )
    if not finding:
        return brand_physical

    log_historical_duplicate(finding, campaign_id=campaign_id, repair_attempted=True)
    repair_prompt = _physical_historical_repair_prompt(finding=finding, base_user_prompt=user_prompt)

    def _parse(raw: object):
        return parse_brand_physical_with_evaluation_recovery(
            raw,
            model_caller=model_caller,
            run_stage=run_stage,
            visibility_policy=visibility_policy,
            repair_context=repair_context,
            **parse_kwargs,
        )

    repaired = run_stage(
        "physical_evaluation_repair",
        model_caller,
        STAGE_BRAND_PHYSICAL_SYSTEM,
        repair_prompt,
        _parse,
    )
    physical_generator2 = str(getattr(repaired, "physical_generator", "") or "")
    transferred_object2 = str(getattr(repaired, "transferred_object", "") or "")
    transferred_object_action2 = str(getattr(repaired, "transferred_object_action", "") or "")
    finding2 = find_historical_duplicate(
        stage="brand_physical",
        snapshot=idea_memory,
        exclude_campaign_id=campaign_id,
        physical_generator=physical_generator2,
        transferred_object=transferred_object2,
        transferred_object_action=transferred_object_action2,
    )
    if finding2:
        raise Builder1PlannerError("builder1_historical_idea_duplicate")
    return repaired


def run_series_ads_with_memory_guard(
    *,
    series_ads: SeriesAdsOutput,
    campaign_id: str,
    idea_memory: Optional[IdeaMemorySnapshot],
    brand_slogan: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    graphic_generator: Dict[str, Any],
    detected_language: str,
    model_caller: Any,
    run_stage: Callable[..., Any],
) -> SeriesAdsOutput:
    if idea_memory is None:
        return series_ads

    current = series_ads
    repair_used = False
    while True:
        duplicate_ad_indexes: List[int] = []
        findings: List[DuplicateExecutionFinding] = []
        duplicate_reasons: List[str] = []
        for ad in current.ads:
            if not isinstance(ad, dict):
                continue
            try:
                ad_index = int(ad.get("index"))
            except (TypeError, ValueError):
                continue
            fp = ad_execution_fingerprint_from_dict(ad)
            finding = find_historical_duplicate(
                stage="series_ads",
                snapshot=idea_memory,
                exclude_campaign_id=campaign_id,
                ad_execution_fingerprint=fp,
                proposed_execution=ad,
            )
            if not finding:
                continue
            log_historical_duplicate(
                finding,
                campaign_id=campaign_id,
                repair_attempted=not repair_used,
            )
            duplicate_ad_indexes.append(ad_index)
            duplicate_reasons.append("builder1_historical_idea_duplicate:execution_fingerprint")
            findings.append(
                DuplicateExecutionFinding(
                    reason="builder1_historical_idea_duplicate",
                    ad_index_a=ad_index,
                    ad_index_b=ad_index,
                    duplicate_type="execution_fingerprint",
                    compared_fields=("historical",),
                    normalized_values=(finding.fingerprint,),
                    fingerprint_hash=finding.fingerprint,
                    excluded_campaign_fields=(),
                )
            )
        if not duplicate_ad_indexes:
            return current
        if repair_used:
            raise Builder1PlannerError("builder1_historical_idea_duplicate")
        current = attempt_series_execution_repair(
            series_ads=current,
            duplicate_reasons=duplicate_reasons,
            findings=findings,
            brand_slogan=brand_slogan,
            conceptual=conceptual,
            brand_physical=brand_physical,
            graphic_generator=graphic_generator,
            detected_language=detected_language,
            model_caller=model_caller,
            run_stage=run_stage,
        )
        repair_used = True


def persist_plan_idea_memory(
    plan: Builder1SeriesPlan,
    *,
    campaign_id: str,
    idea_memory: Optional[IdeaMemorySnapshot],
) -> None:
    if idea_memory is None or not campaign_id:
        return
    records = build_records_from_plan(plan, scope=idea_memory.scope, campaign_id=campaign_id)
    persist_idea_memory_records(records, scope=idea_memory.scope)
    register_completed_campaign_for_backfill(
        campaign_id=campaign_id,
        scope=idea_memory.scope,
        ad_count=plan.ad_count,
    )


def stage_memory_block(
    stage: str,
    idea_memory: Optional[IdeaMemorySnapshot],
    *,
    campaign_id: str,
) -> str:
    if idea_memory is None:
        return ""
    return build_stage_memory_block(stage, idea_memory, exclude_campaign_id=campaign_id)
