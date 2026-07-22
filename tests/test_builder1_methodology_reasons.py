"""
Builder1 methodology reason-layer tests.

Run: python -m unittest tests.test_builder1_methodology_reasons -v
"""
from __future__ import annotations

import copy
import inspect
import re
import unittest
from pathlib import Path

from engine.builder1_image_compliance import IMAGE_COMPLIANCE_SYSTEM_PROMPT
from engine.builder1_methodology_reasons import (
    BRAND_PHYSICAL_STAGE_METHODOLOGY,
    CONCEPTUAL_STAGE_METHODOLOGY,
    GRAPHIC_GENERATOR_REASON,
    METHODOLOGY_EXAMPLE_TEACHING,
    NO_LOGO_REASON,
    POSITIVE_IMAGE_PROMPT_REASON,
    PRODUCT_SHOT_BIAS_REASON,
    SERIES_STAGE_METHODOLOGY,
    SLOGAN_STAGE_METHODOLOGY,
    STAGE_METHODOLOGY_BLOCKS,
    STRATEGY_PROBLEM_PERCEPTION,
    STRATEGY_RELATIVE_ADVANTAGE,
    STRATEGY_STAGE_METHODOLOGY,
    STRATEGY_TRUTH_AND_OPERATIONS,
    TRANSFERRED_OBJECT_REASON,
)
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_SLOGAN_STAGE_SYSTEM,
    STAGE_STRATEGY_STAGE_SYSTEM,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_product_shot_methodology import (
    BUILDER1_PERCEPTION_FIRST,
    BUILDER1_REMOVAL_TEST,
)
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _full_final_responses


BRIEF = "Reinforced shell product for daily carry"


def has_reason_layer(text: str) -> bool:
    lowered = text.lower()
    return (
        "why:" in lowered
        and ("failure prevented:" in lowered or "failure mode" in lowered)
        and "instead:" in lowered
        and "selection test:" in lowered
    )


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


class TestExplanationStructure(unittest.TestCase):
    def test_major_strategic_rules_include_reason(self) -> None:
        self.assertTrue(has_reason_layer(STRATEGY_PROBLEM_PERCEPTION))
        self.assertTrue(has_reason_layer(STRATEGY_RELATIVE_ADVANTAGE))
        self.assertTrue(has_reason_layer(STRATEGY_TRUTH_AND_OPERATIONS))

    def test_major_rules_describe_failure_mode(self) -> None:
        self.assertIn("Failure prevented:", STRATEGY_PROBLEM_PERCEPTION)
        self.assertIn("generic", STRATEGY_PROBLEM_PERCEPTION.lower())

    def test_major_rules_provide_positive_objective(self) -> None:
        self.assertIn("Instead:", STRATEGY_RELATIVE_ADVANTAGE)

    def test_major_rules_provide_selection_test(self) -> None:
        self.assertIn("Selection test:", PRODUCT_SHOT_BIAS_REASON)
        self.assertIn("?", PRODUCT_SHOT_BIAS_REASON)

    def test_formatting_rules_not_expanded_unnecessarily(self) -> None:
        self.assertNotIn("Selection test:", STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM)
        self.assertLess(word_count(STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM), 120)


class TestStrategyReasons(unittest.TestCase):
    def test_problem_perception_in_strategy_stage(self) -> None:
        self.assertIn("audience perception must change", STAGE_STRATEGY_STAGE_SYSTEM.lower())

    def test_relative_advantage_in_strategy_stage(self) -> None:
        self.assertIn("realistic alternative", STAGE_STRATEGY_STAGE_SYSTEM.lower())

    def test_truth_and_operations_in_strategy_stage(self) -> None:
        self.assertIn("operational change", STAGE_STRATEGY_STAGE_SYSTEM.lower())

    def test_generic_praise_identified_as_failure(self) -> None:
        self.assertIn("high quality", STRATEGY_STAGE_METHODOLOGY.lower())


class TestSloganReasons(unittest.TestCase):
    def test_slogan_after_advantage_present(self) -> None:
        self.assertIn("only after selecting the relative advantage", STAGE_SLOGAN_STAGE_SYSTEM.lower())

    def test_slogan_as_distillation_present(self) -> None:
        self.assertIn("distill", STAGE_SLOGAN_STAGE_SYSTEM.lower())

    def test_fixed_before_concept_present(self) -> None:
        self.assertIn("fixed before conceptual", STAGE_SLOGAN_STAGE_SYSTEM.lower())

    def test_generic_slogan_failure_explained(self) -> None:
        self.assertIn("generic inspiration", SLOGAN_STAGE_METHODOLOGY.lower())


class TestConceptReasons(unittest.TestCase):
    def test_implied_action_explanation_present(self) -> None:
        self.assertIn("implied by the slogan", SLOGAN_STAGE_METHODOLOGY.lower())

    def test_idea_before_object_present(self) -> None:
        self.assertIn("attractive object", CONCEPTUAL_STAGE_METHODOLOGY.lower())

    def test_removal_test_explained(self) -> None:
        self.assertTrue(has_reason_layer(BUILDER1_REMOVAL_TEST))

    def test_clarity_before_cleverness_present(self) -> None:
        self.assertIn("few seconds", CONCEPTUAL_STAGE_METHODOLOGY.lower())

    def test_distinctiveness_and_transfer_test_present(self) -> None:
        self.assertIn("competitor replace the Product Name", CONCEPTUAL_STAGE_METHODOLOGY)


class TestPhysicalReasons(unittest.TestCase):
    def test_transferred_object_rationale_present(self) -> None:
        self.assertTrue(has_reason_layer(TRANSFERRED_OBJECT_REASON))

    def test_product_shot_bias_rationale_present(self) -> None:
        self.assertIn("default conceptual or physical generator", STAGE_BRAND_PHYSICAL_SYSTEM.lower())
        self.assertIn("removed", STAGE_BRAND_PHYSICAL_SYSTEM.lower())

    def test_product_evidence_exception_explained(self) -> None:
        self.assertIn("productEvidenceRequired", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_examples_teach_mechanisms(self) -> None:
        self.assertIn("transferable mechanism", METHODOLOGY_EXAMPLE_TEACHING.lower())
        self.assertIn("not using a giraffe", METHODOLOGY_EXAMPLE_TEACHING.lower())


class TestGraphicAndSeriesReasons(unittest.TestCase):
    def test_graphic_generator_rationale_present(self) -> None:
        self.assertIn("stable visual language", STAGE_GRAPHIC_SYSTEM_SYSTEM.lower())

    def test_series_coherence_rationale_present(self) -> None:
        self.assertIn("one mechanism", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_variation_versus_repetition_present(self) -> None:
        self.assertIn("what meaningfully changes", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_restart_not_retrofit_present(self) -> None:
        self.assertIn("adapted from an idea", STAGE_SERIES_ADS_SYSTEM.lower())


class TestCopyAndImageReasons(unittest.TestCase):
    def test_optional_headline_rationale_present(self) -> None:
        self.assertIn("do not add a headline automatically", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_slogan_versus_headline_present(self) -> None:
        self.assertIn("permanent campaign signature", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_no_logo_rationale_present(self) -> None:
        self.assertIn("approved brand asset", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_positive_image_prompt_rationale_present(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(POSITIVE_IMAGE_PROMPT_REASON.splitlines()[0], prompt)

    def test_image_compliance_has_reason(self) -> None:
        self.assertIn("Why:", IMAGE_COMPLIANCE_SYSTEM_PROMPT)


class TestArchitectureRegression(unittest.TestCase):
    def test_no_new_model_call_added(self) -> None:
        from engine import builder1_planning_pipeline as pipeline_module

        source = inspect.getsource(pipeline_module.run_builder1_campaign_pipeline)
        self.assertNotIn("strategy_judge", source)
        self.assertNotIn("creative_judge", source)

    def test_supplied_name_five_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    def test_generated_name_six_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)

    def test_stage_order_unchanged(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module.run_builder1_campaign_pipeline)
        self.assertLess(source.index("run_strategy_slogan_with_memory_guard"), source.index("run_conceptual_with_memory_guard"))
        self.assertLess(source.index("run_conceptual_with_memory_guard"), source.index("build_brand_physical_user_prompt"))

    def test_candidate_counts_unchanged(self) -> None:
        self.assertIn("Exactly 12 candidates", STAGE_STRATEGY_STAGE_SYSTEM)
        self.assertIn("Exactly 6 candidates", STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn("physicalCandidates", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_fifty_marketing_words_enforced(self) -> None:
        self.assertIn("exactly 50 words", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_headline_max_unchanged(self) -> None:
        self.assertIn("headline to null", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_visibility_policy_server_owned(self) -> None:
        self.assertIn("server owns product visibility policy", STAGE_SERIES_ADS_SYSTEM.lower())

    def test_no_server_semantic_judge_in_pipeline(self) -> None:
        from engine import builder1_planning_pipeline as module
        from engine import builder1_planner as planner_module

        self.assertNotIn("judge_builder1_strategy", inspect.getsource(module.run_builder1_campaign_pipeline))
        self.assertNotIn("judge_builder1_strategy", inspect.getsource(planner_module.plan_builder1))

    def test_frontend_unchanged(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertFalse(any(p.name.lower() == "frontend" for p in repo.iterdir() if p.is_dir()))

    def test_builder2_unchanged(self) -> None:
        engine_dir = Path(__file__).resolve().parents[1] / "engine"
        self.assertTrue(list(engine_dir.glob("builder2*.py")))

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


class TestTokenEstimates(unittest.TestCase):
    def test_stage_blocks_are_bounded(self) -> None:
        limits = {"strategy_slogan_stage": 1040}
        for stage, block in STAGE_METHODOLOGY_BLOCKS.items():
            limit = limits.get(stage, 520)
            self.assertLess(word_count(block), limit, msg=stage)

    def test_perception_first_has_reason_layer(self) -> None:
        self.assertTrue(has_reason_layer(BUILDER1_PERCEPTION_FIRST))

    def test_no_logo_reason_shared_not_duplicated_in_full_essay(self) -> None:
        self.assertEqual(
            STAGE_SERIES_ADS_SYSTEM.count("approved brand asset"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
