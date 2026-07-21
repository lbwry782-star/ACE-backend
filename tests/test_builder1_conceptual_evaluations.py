"""Builder1 conceptual-stage evaluation normalization and repair tests."""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_consolidated_stages import (
    _parse_conceptual_evaluations,
    process_conceptual_stage_response,
)
from engine.builder1_conceptual_evaluations import (
    CONCEPTUAL_REJECTION_CODES,
    derive_rejection_codes_from_evaluation_booleans,
    merge_conceptual_evaluation_replacements,
    normalize_conceptual_evaluation_item,
    normalize_conceptual_evaluations_in_payload,
    parse_conceptual_evaluation_replacements,
)
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from engine.builder1_planner import plan_builder1
from engine.builder1_staged_parsers import StageParseError
from tests.test_builder1_staged_planning import (
    _conceptual_evaluation,
    _conceptual_stage_payload,
    _full_final_responses,
)

BRIEF = "Reinforced shell product for daily carry"


class TestConceptualEvaluationNormalization(unittest.TestCase):
    def test_eligible_candidate_with_empty_codes_passes(self) -> None:
        item, actions = normalize_conceptual_evaluation_item(_conceptual_evaluation("C01"))
        self.assertEqual(item["rejectionCodes"], [])
        self.assertEqual(actions, [])
        reviews = _parse_conceptual_evaluations(
            {"evaluations": [item]},
            expected_ids=["C01"],
        )
        self.assertTrue(reviews["C01"].eligible)

    def test_eligible_candidate_with_codes_is_normalized(self) -> None:
        item, actions = normalize_conceptual_evaluation_item(
            _conceptual_evaluation("C01", rejection_codes=["concept_not_visually_clear"])
        )
        self.assertEqual(item["rejectionCodes"], [])
        self.assertIn("cleared_codes_for_eligible_candidate", actions)

    def test_ineligible_candidate_with_valid_code_passes(self) -> None:
        item, _ = normalize_conceptual_evaluation_item(
            _conceptual_evaluation(
                "C01",
                eligible=False,
                avoids_product_shot_bias=False,
                rejection_codes=["concept_conventional_product_shot"],
            )
        )
        reviews = _parse_conceptual_evaluations(
            {"evaluations": [item]},
            expected_ids=["C01"],
        )
        self.assertFalse(reviews["C01"].eligible)
        self.assertEqual(reviews["C01"].rejection_codes, ["concept_conventional_product_shot"])

    def test_ineligible_without_codes_derives_from_booleans(self) -> None:
        raw = _conceptual_evaluation(
            "C01",
            eligible=False,
            avoids_product_shot_bias=False,
            supports_transferred_object=False,
        )
        item, actions = normalize_conceptual_evaluation_item(raw)
        self.assertIn("derived_codes_from_booleans", actions)
        self.assertTrue(item["rejectionCodes"])
        self.assertIn("concept_conventional_product_shot", item["rejectionCodes"])

    def test_unknown_codes_are_removed(self) -> None:
        item, actions = normalize_conceptual_evaluation_item(
            _conceptual_evaluation(
                "C01",
                eligible=False,
                avoids_product_shot_bias=False,
                rejection_codes=["made_up_code", "concept_conventional_product_shot"],
            )
        )
        self.assertIn("removed_unknown_codes", actions)
        self.assertEqual(item["rejectionCodes"], ["concept_conventional_product_shot"])

    def test_duplicate_valid_codes_are_deduplicated(self) -> None:
        raw = _conceptual_evaluation(
            "C01",
            eligible=False,
            avoids_product_shot_bias=False,
            rejection_codes=[
                "concept_conventional_product_shot",
                "concept_conventional_product_shot",
            ],
        )
        item, _ = normalize_conceptual_evaluation_item(raw)
        self.assertEqual(item["rejectionCodes"], ["concept_conventional_product_shot"])

    def test_ineligibility_codes_alias_is_mapped(self) -> None:
        raw = _conceptual_evaluation(
            "C01",
            eligible=False,
            avoids_product_shot_bias=False,
        )
        raw.pop("rejectionCodes", None)
        raw["ineligibilityCodes"] = ["concept_conventional_product_shot"]
        item, actions = normalize_conceptual_evaluation_item(raw)
        self.assertIn("mapped_ineligibility_codes", actions)
        self.assertEqual(item["rejectionCodes"], ["concept_conventional_product_shot"])

    def test_no_code_invented_from_free_text(self) -> None:
        raw = _conceptual_evaluation("C01", eligible=False)
        raw["rejectionCodes"] = []
        raw["perceptionToCreate"] = "This prose explains why it fails but has no mapped boolean signal"
        for key in (
            "derivedFromSelectedSloganAction",
            "expressesRelativeAdvantage",
            "visuallyClear",
            "seriesGenerative",
            "brandOwnable",
            "categoryRelevant",
            "executableByImageModel",
            "survivesProductRemoval",
            "avoidsProductShotBias",
            "supportsTransferredObject",
            "distinctiveToBrand",
        ):
            raw[key] = True
        codes = derive_rejection_codes_from_evaluation_booleans(raw)
        self.assertEqual(codes, [])


class TestConceptualEvaluationRepair(unittest.TestCase):
    def test_repaired_candidates_merge_by_id(self) -> None:
        payload = _conceptual_stage_payload()
        replacement = _conceptual_evaluation(
            "C02",
            eligible=False,
            avoids_product_shot_bias=False,
            rejection_codes=["concept_conventional_product_shot"],
        )
        merged = merge_conceptual_evaluation_replacements(
            payload,
            {"C02": replacement},
        )
        by_id = {item["candidateId"]: item for item in merged["evaluations"]}
        self.assertEqual(by_id["C02"]["rejectionCodes"], ["concept_conventional_product_shot"])
        self.assertEqual(by_id["C01"]["candidateId"], "C01")

    def test_ineligible_without_codes_triggers_targeted_repair(self) -> None:
        payload = _conceptual_stage_payload()
        broken = copy.deepcopy(payload)
        for evaluation in broken["evaluations"]:
            if evaluation["candidateId"] == "C02":
                evaluation["eligible"] = False
                evaluation["rejectionCodes"] = []
                evaluation["avoidsProductShotBias"] = True
                evaluation["supportsTransferredObject"] = True
                evaluation["derivedFromSelectedSloganAction"] = True
        repair_calls: List[str] = []

        def run_stage(stage, model_caller, system, user, parse_fn, **kwargs):
            repair_calls.append(stage)
            repaired = _conceptual_evaluation(
                "C02",
                eligible=False,
                avoids_product_shot_bias=False,
                rejection_codes=["concept_conventional_product_shot"],
            )
            return parse_fn({"evaluations": [repaired]})

        process_conceptual_stage_response(
            broken,
            product_description=BRIEF,
            product_name_resolved="TestBrand",
            brand_slogan="Built To Last",
            implied_action="Show impact survival",
            relative_advantage="Survives daily drops",
            strategic_problem="Buyers doubt durability",
            model_caller=lambda *_a, **_k: {},
            run_stage=run_stage,
        )
        self.assertEqual(repair_calls, ["conceptual_evaluation_repair"])

    def test_failed_targeted_repair_raises_for_stage_retry(self) -> None:
        payload = _conceptual_stage_payload()
        broken = copy.deepcopy(payload)
        broken["evaluations"][1]["eligible"] = False
        broken["evaluations"][1]["rejectionCodes"] = []
        broken["evaluations"][1]["avoidsProductShotBias"] = True
        broken["evaluations"][1]["supportsTransferredObject"] = True

        def run_stage(stage, model_caller, system, user, parse_fn, **kwargs):
            if stage == "conceptual_evaluation_repair":
                return parse_fn(
                    {
                        "evaluations": [
                            {
                                **_conceptual_evaluation("C02", eligible=False),
                                "rejectionCodes": [],
                                "avoidsProductShotBias": True,
                            }
                        ]
                    }
                )
            raise AssertionError(stage)

        with self.assertRaises(StageParseError):
            process_conceptual_stage_response(
                broken,
                product_description=BRIEF,
                product_name_resolved="TestBrand",
                brand_slogan="Built To Last",
                implied_action="Show impact survival",
                relative_advantage="Survives daily drops",
                strategic_problem="Buyers doubt durability",
                model_caller=lambda *_a, **_k: {},
                run_stage=run_stage,
            )

    def test_parse_repair_rejects_unexpected_ids(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_conceptual_evaluation_replacements(
                {"evaluations": [_conceptual_evaluation("C99", eligible=False, rejection_codes=["concept_not_visually_clear"])]},
                allowed_ids=["C01"],
            )
        self.assertIn("conceptual_evaluation_repair_unexpected_id:C99", ctx.exception.reasons)


class TestConceptualPipelineRegression(unittest.TestCase):
    def test_normal_expected_calls_remain_five(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    @patch.dict("os.environ", {"BUILDER1_PLANNING_PROFILE": "QUALITY"}, clear=False)
    def test_gpt_5_6_sol_remains_configured(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")

    def test_planning_completes_after_valid_conceptual_stage(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan = plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertIn("brand_physical", stages)
        self.assertIn("graphic_system", stages)
        self.assertIn("series_ads", stages)

    def test_planning_failure_does_not_reach_image_generation(self) -> None:
        from engine.builder1_planning_contract import STAGE_CONCEPTUAL_STAGE_SYSTEM
        from engine.builder1_planner import Builder1PlannerError

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_CONCEPTUAL_STAGE_SYSTEM:
                payload = copy.deepcopy(_conceptual_stage_payload())
                for evaluation in payload["evaluations"]:
                    evaluation["eligible"] = False
                    evaluation["rejectionCodes"] = []
                    evaluation["avoidsProductShotBias"] = True
                    evaluation["supportsTransferredObject"] = True
                    evaluation["derivedFromSelectedSloganAction"] = True
                return payload
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        def failing_repair(*_args, **_kwargs):
            raise StageParseError("conceptual_evaluation_repair", ["conceptual_evaluation_repair_failed"])

        with patch(
            "engine.builder1_consolidated_stages._run_conceptual_evaluation_repair",
            side_effect=failing_repair,
        ):
            with self.assertRaises(Builder1PlannerError):
                plan_builder1(
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    model_caller=model_caller,
                    ad_count=2,
                )


if __name__ == "__main__":
    unittest.main()
