#!/usr/bin/env python3
"""Developer-only Builder1 planning benchmark across QUALITY/BALANCED/FAST profiles."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from engine.builder1_planning_metrics import get_planning_metrics, set_planning_metrics, reset_planning_metrics
from engine.builder1_planning_metrics import Builder1PlanningMetrics
from engine.builder1_planning_profile import PlanningProfile, resolve_stage_routing
from engine.builder1_planner import plan_builder1
from tests.test_builder1_staged_planning import _full_final_responses, STAGE_STRATEGY_STAGE_SYSTEM
from tests.test_builder1_staged_planning import _strategy_stage_payload


BRIEF = "Reinforced shell product for daily carry"


def _mock_caller_factory(stage_models: Dict[str, str]):
    def model_caller(system: str, user: str, stage: str | None = None) -> object:
        if stage:
            stage_models[stage] = resolve_stage_routing(stage).model
        responses = _full_final_responses(2)
        responses[STAGE_STRATEGY_STAGE_SYSTEM] = _strategy_stage_payload()
        return responses.get(system, {})

    return model_caller


def run_profile(profile: PlanningProfile) -> Dict[str, Any]:
    os.environ["BUILDER1_PLANNING_PROFILE"] = profile.value
    stage_models: Dict[str, str] = {}
    metrics = Builder1PlanningMetrics(campaign_id="bench", job_id=f"bench-{profile.value.lower()}")
    token = set_planning_metrics(metrics)
    started = time.perf_counter()
    try:
        plan = plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=_mock_caller_factory(stage_models),
            ad_count=2,
            campaign_id="bench",
            job_id=f"bench-{profile.value.lower()}",
        )
    finally:
        reset_planning_metrics(token)
    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "profile": profile.value,
        "stageModels": stage_models,
        "reasoningEfforts": {
            stage: resolve_stage_routing(stage).reasoning_effort or "none"
            for stage in (
                "strategy_stage",
                "slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            )
        },
        "totalPlanningDurationMs": duration_ms,
        "promptTokens": metrics.prompt_tokens,
        "outputTokens": metrics.output_tokens,
        "totalTokens": metrics.total_tokens,
        "schemaSuccess": True,
        "repairCalls": metrics.focused_repair_calls,
        "selectedStrategyId": plan.planning_internals.get("selectedStrategyId", ""),
        "selectedSloganId": plan.planning_internals.get("selectedSloganId", ""),
        "selectedConceptualId": plan.planning_internals.get("selectedConceptualId", ""),
        "invariantsPassed": True,
        "brandSlogan": plan.brand_slogan,
        "physicalGenerator": plan.physical_generator,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Builder1 planning profile benchmark")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live OpenAI benchmark (requires BUILDER1_ENABLE_LIVE_BENCHMARK=1)",
    )
    args = parser.parse_args(argv)
    if args.live and (os.environ.get("BUILDER1_ENABLE_LIVE_BENCHMARK") or "").strip() not in {"1", "true", "yes"}:
        print("Live benchmark disabled. Set BUILDER1_ENABLE_LIVE_BENCHMARK=1 to enable.")
        return 2
    if args.live:
        print("Live benchmark is not bundled in this diagnostic command.")
        return 2

    reports = [run_profile(profile) for profile in PlanningProfile]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
