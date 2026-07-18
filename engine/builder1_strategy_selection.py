"""Builder1 strategy selection review, gate, and bounded reselection."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.builder1_client_boundary import strategy_candidate_is_eligible
from engine.builder1_planning_contract import (
    STAGE_STRATEGY_SELECT_SYSTEM,
    build_strategy_select_user_prompt,
)
from engine.builder1_staged_parsers import (
    StrategyCandidate,
    StrategyCandidateReview,
    StrategySelection,
    parse_strategy_selection,
)

logger = logging.getLogger(__name__)

PlanningModelCaller = Callable[..., object]
RunStageFn = Callable[..., Any]


class StrategySelectionExhausted(Exception):
    """No eligible strategy candidate remains after bounded reselection."""


def validate_selected_strategy_gate(
    candidate: StrategyCandidate,
    review: StrategyCandidateReview,
) -> List[str]:
    reasons: List[str] = []
    if not review.eligible:
        reasons.extend(review.rejection_codes)
    if not review.grounded_in_brief:
        reasons.append("unsupported_evidence_claim")
    if not review.advantage_currently_true:
        reasons.append("advantage_not_currently_true")
    if not review.executable_now:
        reasons.append("unsupported_future_capability")
    if review.requires_material_investment:
        reasons.append("material_client_investment_required")
    if review.requires_client_consultation:
        reasons.append("client_consultation_required")
    if review.requires_business_transformation:
        reasons.append("business_transformation_required")
    if not review.brand_ownable:
        reasons.append("strategy_not_brand_ownable")
    if not review.category_relevant:
        reasons.append("category_relevance_patched")
    if not strategy_candidate_is_eligible(candidate):
        if candidate.implementation_cost_level == "material":
            reasons.append("material_client_investment_required")
        elif candidate.requires_client_consultation:
            reasons.append("client_consultation_required")
        elif candidate.client_action_level == "complex_required":
            reasons.append("business_transformation_required")
        elif not candidate.campaign_executable_now:
            reasons.append("advantage_not_currently_true")
    if review.candidate_id != candidate.id:
        reasons.append("strategy_selection_invalid_id")
    return list(dict.fromkeys(reasons))


def _run_strategy_selection_stage(
    run_stage: RunStageFn,
    model_caller: PlanningModelCaller,
    *,
    candidates: List[StrategyCandidate],
    eligible_ids: Set[str],
    exploration_seed: str,
) -> Tuple[StrategySelection, StrategyCandidate, Dict[str, StrategyCandidateReview]]:
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
        for c in candidates
        if c.id in eligible_ids
    ]

    def _parse(raw: object):
        return parse_strategy_selection(
            raw,
            candidates,
            eligible_ids=eligible_ids,
        )

    return run_stage(
        "strategy_selection",
        model_caller,
        STAGE_STRATEGY_SELECT_SYSTEM,
        build_strategy_select_user_prompt(cand_dicts, exploration_seed),
        _parse,
    )


def run_strategy_selection_with_gate(
    *,
    strategy_candidates: List[StrategyCandidate],
    eligible_ids: Set[str],
    exploration_seed: str,
    model_caller: PlanningModelCaller,
    run_stage: RunStageFn,
) -> Tuple[StrategySelection, StrategyCandidate, Dict[str, StrategyCandidateReview]]:
    if not eligible_ids:
        raise StrategySelectionExhausted()

    remaining = set(eligible_ids)
    reselection_used = False
    last: Optional[Tuple[StrategySelection, StrategyCandidate, Dict[str, StrategyCandidateReview]]] = None

    while remaining:
        try:
            last = _run_strategy_selection_stage(
                run_stage,
                model_caller,
                candidates=strategy_candidates,
                eligible_ids=remaining,
                exploration_seed=exploration_seed,
            )
        except Builder1PlannerError:
            raise

        selection, selected, reviews = last
        gate = validate_selected_strategy_gate(selected, reviews[selected.id])
        if not gate:
            logger.info(
                "BUILDER1_STRATEGY_SELECTION_OK candidateId=%s",
                selected.id,
            )
            return selection, selected, reviews

        logger.info(
            "BUILDER1_STRATEGY_RESELECTION rejectedCandidateId=%s reasons=%s",
            selected.id,
            gate,
        )
        remaining.discard(selected.id)
        if not remaining:
            break
        if reselection_used:
            break
        reselection_used = True

    raise StrategySelectionExhausted()
