"""
Builder1 creative methodology — perception-first, anti product-shot bias.

Run: python -m unittest tests.test_builder1_creative_methodology -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
from pathlib import Path
from typing import Any, Dict, List

from engine.builder1_consolidated_stages import (
    CONCEPTUAL_REJECTION_CODES,
    _conceptual_gate_reasons,
    _parse_conceptual_evaluations,
    process_conceptual_stage_response,
)
from engine.builder1_final_stages import parse_brand_physical_output
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    build_brand_physical_user_prompt,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_methodology_reasons import STRATEGY_STAGE_METHODOLOGY
from engine.builder1_product_shot_methodology import (
    BUILDER1_FORBIDDEN_PRODUCT_SHOT_LANGUAGE,
    BUILDER1_REMOVAL_TEST,
)
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _conceptual_evaluation,
    _conceptual_stage_payload,
    _full_final_responses,
    _physical_candidates,
)


BRIEF = "Reinforced shell product for daily carry"


class TestMethodologyPrompts(unittest.TestCase):
    def test_conceptual_stage_contains_idea_before_product(self) -> None:
        self.assertIn("attractive object", STAGE_CONCEPTUAL_STAGE_SYSTEM.lower())
        self.assertIn("object-first", STAGE_CONCEPTUAL_STAGE_SYSTEM.lower())

    def test_conceptual_stage_contains_removal_test(self) -> None:
        self.assertIn("REMOVAL TEST", STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn(BUILDER1_REMOVAL_TEST.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)

    def test_conceptual_stage_evaluates_product_shot_bias(self) -> None:
        self.assertIn("avoidsProductShotBias", STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn("removed", STAGE_CONCEPTUAL_STAGE_SYSTEM.lower())

    def test_brand_physical_requires_external_object_exploration(self) -> None:
        self.assertIn("physicalCandidates", STAGE_BRAND_PHYSICAL_SYSTEM)
        self.assertIn("different physicalWorld", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_brand_physical_asks_why_clearer_than_product(self) -> None:
        self.assertIn("whyClearerThanShowingProduct", STAGE_BRAND_PHYSICAL_SYSTEM)
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="TestBrand",
            product_description=BRIEF,
            detected_language="en",
            format_value="portrait",
            strategic_problem="Daily carry protection",
            relative_advantage="Survives daily drops",
            brand_slogan="Built To Last",
            slogan_derivation="From durability",
            implied_action="Show impact survival",
            conceptual={"generator": "Stress test"},
            visibility_policy="FORBIDDEN",
        )
        self.assertIn("clearer than showing the product", prompt.lower())

    def test_series_ads_preserves_transferred_object_logic(self) -> None:
        self.assertIn("same visual mechanism", STAGE_SERIES_ADS_SYSTEM.lower())
        self.assertIn("product-shot fallback", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_image_prompt_avoids_hero_product_language_under_forbidden(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(BUILDER1_FORBIDDEN_PRODUCT_SHOT_LANGUAGE.splitlines()[0].lower(), prompt.lower())
        self.assertNotIn("hero product", prompt.lower())
        self.assertIn("transferred external object", prompt.lower())
        self.assertIn("catalog packaging photography", prompt.lower())


class TestConceptualCandidateSelection(unittest.TestCase):
    def _evaluations(self, **overrides: Any) -> Dict[str, Any]:
        base = [_conceptual_evaluation(f"C{i:02d}") for i in range(1, 7)]
        if overrides:
            base[0] = _conceptual_evaluation("C01", **overrides)
        return base

    def test_conventional_product_shot_is_ineligible(self) -> None:
        payload = _conceptual_stage_payload()
        payload["evaluations"] = [
            _conceptual_evaluation(
                f"C{i:02d}",
                eligible=False,
                avoids_product_shot_bias=False,
                rejection_codes=["concept_conventional_product_shot"],
            )
            for i in range(1, 7)
        ]
        with self.assertRaises(Exception):
            process_conceptual_stage_response(
                payload,
                product_description=BRIEF,
                product_name_resolved="TestBrand",
                brand_slogan="Built To Last",
                implied_action="Show impact survival",
                model_caller=lambda *_a, **_k: {},
                run_stage=lambda *_a, **_k: [],
            )

    def test_beautified_package_without_mechanism_is_ineligible(self) -> None:
        review = _parse_conceptual_evaluations(
            {
                "evaluations": [
                    _conceptual_evaluation(
                        "C01",
                        eligible=False,
                        avoids_product_shot_bias=False,
                        supports_transferred_object=False,
                        rejection_codes=["concept_decorative_presentation_only"],
                    )
                ]
            },
            expected_ids=["C01"],
        )["C01"]
        reasons = _conceptual_gate_reasons(review)
        self.assertIn("concept_decorative_presentation_only", reasons)

    def test_multiplied_product_without_necessity_is_ineligible(self) -> None:
        self.assertIn("concept_product_shot_bias", CONCEPTUAL_REJECTION_CODES)
        review = _parse_conceptual_evaluations(
            {
                "evaluations": [
                    _conceptual_evaluation(
                        "C01",
                        eligible=False,
                        survives_product_removal=False,
                        avoids_product_shot_bias=False,
                        rejection_codes=["concept_product_shot_bias"],
                    )
                ]
            },
            expected_ids=["C01"],
        )["C01"]
        self.assertFalse(review.eligible)

    def test_transferred_external_object_can_be_eligible(self) -> None:
        review = _parse_conceptual_evaluations(
            {"evaluations": [_conceptual_evaluation("C01")]},
            expected_ids=["C01"],
        )["C01"]
        self.assertEqual(_conceptual_gate_reasons(review), [])

    def test_survives_product_removal_can_be_eligible(self) -> None:
        review = _parse_conceptual_evaluations(
            {
                "evaluations": [
                    _conceptual_evaluation("C01", survives_product_removal=True),
                ]
            },
            expected_ids=["C01"],
        )["C01"]
        self.assertTrue(review.survives_product_removal)
        self.assertEqual(_conceptual_gate_reasons(review), [])

    def test_collapses_without_product_is_rejected_without_evidence(self) -> None:
        review = _parse_conceptual_evaluations(
            {
                "evaluations": [
                    _conceptual_evaluation(
                        "C01",
                        eligible=True,
                        survives_product_removal=False,
                        supports_transferred_object=False,
                    )
                ]
            },
            expected_ids=["C01"],
        )["C01"]
        reasons = _conceptual_gate_reasons(review)
        self.assertIn("concept_collapses_without_product", reasons)
        self.assertIn("concept_no_transferred_object_path", reasons)


class TestProductEvidenceException(unittest.TestCase):
    def test_real_product_property_evidence_may_be_eligible(self) -> None:
        review = _parse_conceptual_evaluations(
            {
                "evaluations": [
                    _conceptual_evaluation(
                        "C01",
                        survives_product_removal=False,
                        supports_transferred_object=False,
                        product_evidence_required=True,
                        product_evidence_reason="Only the actual reinforced shell construction proves the mechanism.",
                    )
                ]
            },
            expected_ids=["C01"],
        )["C01"]
        reasons = _conceptual_gate_reasons(review)
        self.assertNotIn("concept_collapses_without_product", reasons)

    def test_product_visibility_requires_necessity_explanation(self) -> None:
        payload = _brand_physical()
        payload["productEvidenceRequired"] = True
        payload["productEvidenceReason"] = ""
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, product_description=BRIEF)
        self.assertIn("physical_missing_evidence_reason", str(ctx.exception))

    def test_attractive_presentation_alone_is_insufficient(self) -> None:
        payload = _brand_physical()
        payload["physicalEvaluations"][0]["eligible"] = False
        payload["physicalEvaluations"][0]["rejectionCodes"] = ["physical_decorative_presentation_only"]
        payload["selectedPhysicalCandidateId"] = "P01"
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, product_description=BRIEF)
        self.assertIn("physical_selected_not_eligible", str(ctx.exception))

    def test_product_visibility_remains_secondary_when_policy_permits(self) -> None:
        self.assertIn("product presence remains secondary", STAGE_BRAND_PHYSICAL_SYSTEM.lower())

    def test_forbidden_policy_still_blocks_product_depiction(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, product_description=BRIEF, visibility_policy="FORBIDDEN")
        self.assertIn("physical_generator_is_product", str(ctx.exception))


class TestSeriesAndRegression(unittest.TestCase):
    def test_transferred_object_family_required_in_brand_physical(self) -> None:
        payload = _brand_physical()
        payload["physicalCandidates"] = _physical_candidates(worlds=["kitchen", "kitchen", "kitchen", "kitchen"])
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, product_description=BRIEF)
        self.assertIn("physical_all_candidates_same_world", str(ctx.exception))

    def test_no_logo_rules_unchanged_in_series_prompt(self) -> None:
        self.assertIn("never request a logo", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_supplied_name_planning_remains_six_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 6)

    def test_generated_name_planning_remains_seven_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 7)

    def test_stage_order_unchanged_in_pipeline(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module.run_builder1_campaign_pipeline)
        self.assertLess(source.index("run_slogan_stage"), source.index("run_conceptual_stage"))
        self.assertLess(source.index("run_conceptual_stage"), source.index("build_brand_physical_user_prompt"))
        self.assertLess(source.index("build_brand_physical_user_prompt"), source.index("_run_graphic_system_stage"))
        self.assertLess(source.index("_run_graphic_system_stage"), source.index("_run_series_stage_with_integrity"))

    def test_no_final_server_creative_judge(self) -> None:
        from engine import builder1_planning_pipeline as pipeline_module
        from engine import builder1_planner as planner_module

        pipeline_source = inspect.getsource(pipeline_module.run_builder1_campaign_pipeline)
        planner_source = inspect.getsource(planner_module.plan_builder1)
        self.assertNotIn("judge_builder1_strategy", pipeline_source)
        self.assertNotIn("judge_builder1_strategy", planner_source)

    def test_frontend_unchanged(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertFalse(any(p.name.lower() == "frontend" for p in repo.iterdir() if p.is_dir()))

    def test_builder2_unchanged(self) -> None:
        engine_dir = Path(__file__).resolve().parents[1] / "engine"
        builder2_files = list(engine_dir.glob("builder2*.py"))
        self.assertTrue(builder2_files)

    def test_end_to_end_planning_still_completes(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan = plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertTrue(plan.transferred_object)


if __name__ == "__main__":
    unittest.main()
