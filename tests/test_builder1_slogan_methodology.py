"""
Builder1 slogan-first creative methodology tests.

Run: python -m unittest tests.test_builder1_slogan_methodology -v
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_creative_methodology import (
    deterministic_methodology_checks,
    methodology_repair_stage,
)
from engine.builder1_final_stages import parse_brand_physical_output
from engine.builder1_planner import plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_SLOGAN_STAGE_SYSTEM,
    STAGE_STRATEGY_STAGE_SYSTEM,
    build_conceptual_stage_user_prompt,
)
from engine.builder1_slogan_stage import (
    parse_slogan_scan,
    parse_slogan_selection,
    validate_selected_slogan,
)
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT, deterministic_judge_checks
from engine.builder1_visual_prompt import build_visual_prompt
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _conceptual_scan_payload,
    _full_final_responses,
    _slogan_scan_payload,
    _strategy_scan_payload,
)
from tests.test_builder1_series import _base_campaign, _parse


class TestPlanningOrder(unittest.TestCase):
    def test_planner_source_order_has_strategy_slogan_before_conceptual(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module.run_builder1_campaign_pipeline)
        self.assertLess(source.index("run_strategy_slogan_stage"), source.index("run_conceptual_stage"))

    def test_conceptual_stage_prompt_receives_selected_slogan(self) -> None:
        prompt = build_conceptual_stage_user_prompt(
            product_description="Reinforced shell product",
            product_name_resolved="TestBrand",
            strategic_problem="Buyers doubt durability",
            relative_advantage="Survives daily drops",
            brand_slogan="Built To Last",
            slogan_derivation="From durability advantage",
            implied_action="Show impact survival visually",
            exploration_seed="seed-1",
        )
        self.assertIn("Fixed brand slogan: Built To Last", prompt)
        self.assertIn("Implied slogan action:", prompt)

    def test_brand_physical_parser_rejects_slogan_creation(self) -> None:
        payload = _brand_physical()
        payload["brandSlogan"] = "New Slogan"
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("brand_physical_must_not_create_slogan", str(ctx.exception))


class TestSloganQuality(unittest.TestCase):
    def test_advantage_derived_slogan_passes_gate(self) -> None:
        candidates = parse_slogan_scan(_slogan_scan_payload())
        _, selected = parse_slogan_selection(
            {"selectedCandidateId": "L01", "selectionReason": "Best", "scores": {"directAdvantageExpression": 9}},
            candidates,
        )
        reasons = validate_selected_slogan(
            selected,
            relative_advantage="Survives daily drops",
            product_description="Reinforced shell product",
        )
        self.assertEqual(reasons, [])

    def test_generic_slogan_rejected_in_scan(self) -> None:
        with self.assertRaises(Exception) as ctx:
            parse_slogan_scan(_slogan_scan_payload(generic=True))
        self.assertIn("slogan_generic", str(ctx.exception))

    def test_high_transfer_risk_rejected_at_gate(self) -> None:
        candidates = parse_slogan_scan(_slogan_scan_payload())
        risky = candidates[0]
        risky.competitor_transfer_risk = "high"
        reasons = validate_selected_slogan(
            risky,
            relative_advantage="Survives daily drops",
        )
        self.assertIn("slogan_not_ownable", reasons)

    def test_empty_implied_action_rejected(self) -> None:
        candidates = parse_slogan_scan(_slogan_scan_payload())
        broken = candidates[0]
        broken.implied_action = "x"
        reasons = validate_selected_slogan(broken, relative_advantage="Survives daily drops")
        self.assertIn("slogan_no_implied_action", reasons)


class TestConceptualAndPhysical(unittest.TestCase):
    def test_conceptual_scan_requires_slogan_fields(self) -> None:
        payload = _conceptual_scan_payload()
        payload["candidates"][0].pop("whyItExpressesSlogan")
        with self.assertRaises(Exception) as ctx:
            from engine.builder1_staged_parsers import parse_conceptual_scan

            parse_conceptual_scan(payload)
        self.assertIn("conceptual_scan_candidate_incomplete", str(ctx.exception))

    def test_literal_product_without_justification_fails_judge(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        store = {
            "brandSlogan": plan.brand_slogan,
            "sloganAction": plan.slogan_action,
            "conceptualGeneratorAction": plan.conceptual_generator_action,
            "conceptualGeneratorWhyItExpressesSlogan": "Shows slogan",
            "physicalGenerator": plan.physical_generator,
            "productVisibilityPolicy": "FORBIDDEN",
            "embodimentChoice": "literal",
            "productVisibilityJustification": "",
            "detectedLanguage": "en",
            "relativeAdvantage": plan.relative_advantage,
            "sloganDerivation": plan.slogan_derivation,
            "ads": [
                {
                    "index": 1,
                    "sceneDescription": "Scene",
                    "headline": None,
                    "marketingText": marketing_text_words(50),
                    "productVisible": True,
                }
            ],
        }
        self.assertIn("unauthorized_product_visibility", deterministic_methodology_checks(store))


class TestRepairRouting(unittest.TestCase):
    def test_generic_slogan_routes_to_slogan_scan(self) -> None:
        self.assertEqual(methodology_repair_stage(["slogan_generic"]), "slogan_scan")

    def test_weak_concept_routes_to_conceptual_scan(self) -> None:
        self.assertEqual(
            methodology_repair_stage(["conceptual_generator_not_derived_from_slogan"]),
            "conceptual_scan",
        )

    def test_weak_physical_routes_to_brand_physical(self) -> None:
        self.assertEqual(
            methodology_repair_stage(["physical_generator_not_derived_from_concept"]),
            "brand_physical",
        )

    def test_transferable_campaign_routes_to_strategy_scan(self) -> None:
        self.assertEqual(
            methodology_repair_stage(["campaign_transferable_to_competitor"]),
            "strategy_scan",
        )


class TestPlannerIntegration(unittest.TestCase):
    def test_plan_builder1_uses_fixed_slogan_from_strategy_slogan_stage(self) -> None:
        stage_order: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stage_order.append(stage)
            responses = _full_final_responses(2)
            return responses.get(system, {})

        plan = plan_builder1(
            product_name="TestBrand",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertIn("strategy_slogan_stage", stage_order)
        self.assertLess(stage_order.index("strategy_slogan_stage"), stage_order.index("conceptual_stage"))
        self.assertNotIn("slogan_stage", stage_order)
        self.assertEqual(plan.brand_slogan, "Built To Last")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("Fixed brand slogan (typography only): Built To Last.", prompt)


class TestRegressionAndBoundaries(unittest.TestCase):
    def test_brand_physical_system_forbids_slogan_creation(self) -> None:
        self.assertIn("Do NOT create, replace, or modify the brand slogan", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_integrity_validation_exists_without_judge(self) -> None:
        from engine.builder1_campaign_integrity import validate_builder1_campaign_integrity

        self.assertTrue(callable(validate_builder1_campaign_integrity))

    def test_builder2_unchanged(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_slogan_stage", text)


if __name__ == "__main__":
    unittest.main()
