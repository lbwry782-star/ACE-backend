"""Builder1 physical-stage evaluation normalization and repair tests."""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List

from engine.builder1_final_stages import parse_brand_physical_output
from engine.builder1_physical_evaluations import (
    PHYSICAL_REJECTION_CODES,
    derive_rejection_codes_from_evaluation_booleans,
    merge_physical_evaluation_replacements,
    normalize_physical_evaluation_item,
    normalize_physical_evaluations_in_payload,
    parse_brand_physical_with_evaluation_recovery,
    parse_physical_evaluation_replacements,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_NAME,
    Builder1PlanningMetrics,
    reset_planning_metrics,
    set_planning_metrics,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_planning_contract import STAGE_BRAND_PHYSICAL_SYSTEM
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _full_final_responses,
)

BRIEF = "Reinforced shell product for daily carry"


def _physical_evaluation(
    candidate_id: str,
    *,
    eligible: bool = True,
    rejection_codes: List[str] | None = None,
    clearer_than_product_shot: bool = True,
    survives_product_removal: bool = True,
    supports_transferred_object: bool = True,
    distinctive_to_brand: bool = True,
) -> Dict[str, Any]:
    return {
        "candidateId": candidate_id,
        "clearerThanConventionalProductShot": clearer_than_product_shot,
        "survivesProductRemoval": survives_product_removal,
        "supportsTransferredObject": supports_transferred_object,
        "distinctiveToBrand": distinctive_to_brand,
        "eligible": eligible,
        "rejectionCodes": list(rejection_codes or []),
    }


class TestPhysicalEvaluationNormalization(unittest.TestCase):
    def test_eligible_with_empty_codes_passes(self) -> None:
        item, actions = normalize_physical_evaluation_item(_physical_evaluation("P01"))
        self.assertEqual(item["rejectionCodes"], [])
        self.assertEqual(actions, [])
        payload = _brand_physical()
        payload["physicalEvaluations"] = [item]
        parse_brand_physical_output(payload, product_description=BRIEF)

    def test_eligible_with_codes_is_normalized(self) -> None:
        item, actions = normalize_physical_evaluation_item(
            _physical_evaluation("P01", rejection_codes=["physical_decorative_presentation_only"])
        )
        self.assertEqual(item["rejectionCodes"], [])
        self.assertIn("cleared_codes_for_eligible_candidate", actions)

    def test_ineligible_with_valid_code_passes(self) -> None:
        item, _ = normalize_physical_evaluation_item(
            _physical_evaluation(
                "P02",
                eligible=False,
                clearer_than_product_shot=False,
                rejection_codes=["physical_conventional_product_shot"],
            )
        )
        payload = _brand_physical()
        payload["physicalEvaluations"] = _brand_physical()["physicalEvaluations"]
        payload["physicalEvaluations"][1] = item
        parse_brand_physical_output(payload, product_description=BRIEF)

    def test_ineligible_without_codes_derives_from_booleans(self) -> None:
        item, actions = normalize_physical_evaluation_item(
            _physical_evaluation(
                "P02",
                eligible=False,
                clearer_than_product_shot=False,
                distinctive_to_brand=False,
            )
        )
        self.assertIn("derived_codes_from_booleans", actions)
        self.assertIn("physical_conventional_product_shot", item["rejectionCodes"])
        self.assertIn("physical_decorative_presentation_only", item["rejectionCodes"])

    def test_ineligibility_codes_alias_is_mapped(self) -> None:
        raw = _physical_evaluation("P02", eligible=False, clearer_than_product_shot=False)
        raw.pop("rejectionCodes", None)
        raw["ineligibilityCodes"] = ["physical_conventional_product_shot"]
        item, actions = normalize_physical_evaluation_item(raw)
        self.assertIn("mapped_ineligibility_codes", actions)
        self.assertEqual(item["rejectionCodes"], ["physical_conventional_product_shot"])

    def test_unknown_codes_are_removed(self) -> None:
        item, actions = normalize_physical_evaluation_item(
            _physical_evaluation(
                "P02",
                eligible=False,
                clearer_than_product_shot=False,
                rejection_codes=["made_up_code", "physical_conventional_product_shot"],
            )
        )
        self.assertIn("removed_unknown_codes", actions)
        self.assertEqual(item["rejectionCodes"], ["physical_conventional_product_shot"])

    def test_duplicate_valid_codes_are_deduplicated(self) -> None:
        item, _ = normalize_physical_evaluation_item(
            _physical_evaluation(
                "P02",
                eligible=False,
                clearer_than_product_shot=False,
                rejection_codes=[
                    "physical_conventional_product_shot",
                    "physical_conventional_product_shot",
                ],
            )
        )
        self.assertEqual(item["rejectionCodes"], ["physical_conventional_product_shot"])

    def test_eligibility_is_never_flipped(self) -> None:
        raw = _physical_evaluation("P02", eligible=False, clearer_than_product_shot=False)
        item, _ = normalize_physical_evaluation_item(raw)
        self.assertFalse(item["eligible"])

    def test_no_code_invented_from_prose_only(self) -> None:
        raw = _physical_evaluation("P02", eligible=False)
        raw["rejectionCodes"] = []
        for key in (
            "clearerThanConventionalProductShot",
            "survivesProductRemoval",
            "supportsTransferredObject",
            "distinctiveToBrand",
        ):
            raw[key] = True
        codes = derive_rejection_codes_from_evaluation_booleans(raw)
        self.assertEqual(codes, [])


class TestPhysicalEvaluationRepair(unittest.TestCase):
    def test_repaired_evaluations_merge_by_id(self) -> None:
        payload = _brand_physical()
        replacement = _physical_evaluation(
            "P02",
            eligible=False,
            clearer_than_product_shot=False,
            rejection_codes=["physical_conventional_product_shot"],
        )
        merged = merge_physical_evaluation_replacements(payload, {"P02": replacement})
        by_id = {item["candidateId"]: item for item in merged["physicalEvaluations"]}
        self.assertEqual(by_id["P02"]["rejectionCodes"], ["physical_conventional_product_shot"])
        self.assertEqual(by_id["P01"]["candidateId"], "P01")

    def test_ineligible_without_codes_triggers_targeted_repair(self) -> None:
        broken = copy.deepcopy(_brand_physical())
        broken["physicalEvaluations"][1]["eligible"] = False
        broken["physicalEvaluations"][1]["rejectionCodes"] = []
        broken["physicalEvaluations"][1]["clearerThanConventionalProductShot"] = True
        broken["physicalEvaluations"][1]["supportsTransferredObject"] = True
        broken["physicalEvaluations"][1]["distinctiveToBrand"] = True
        repair_calls: List[str] = []

        def run_stage(stage, model_caller, system, user, parse_fn, **kwargs):
            repair_calls.append(stage)
            repaired = _physical_evaluation(
                "P02",
                eligible=False,
                clearer_than_product_shot=False,
                rejection_codes=["physical_conventional_product_shot"],
            )
            return parse_fn({"physicalEvaluations": [repaired]})

        parse_brand_physical_with_evaluation_recovery(
            broken,
            model_caller=lambda *_a, **_k: {},
            run_stage=run_stage,
            visibility_policy="FORBIDDEN",
            repair_context={
                "strategic_problem": "Buyers doubt durability",
                "relative_advantage": "Survives daily drops",
                "brand_slogan": "Built To Last",
                "implied_action": "Show impact survival",
                "conceptual": {"generator": "Stress test"},
            },
            product_description=BRIEF,
        )
        self.assertEqual(repair_calls, ["physical_evaluation_repair"])

    def test_failed_targeted_repair_raises_for_full_retry(self) -> None:
        broken = copy.deepcopy(_brand_physical())
        broken["physicalEvaluations"][1]["eligible"] = False
        broken["physicalEvaluations"][1]["rejectionCodes"] = []
        broken["physicalEvaluations"][1]["clearerThanConventionalProductShot"] = True
        broken["physicalEvaluations"][1]["supportsTransferredObject"] = True
        broken["physicalEvaluations"][1]["distinctiveToBrand"] = True

        def run_stage(stage, model_caller, system, user, parse_fn, **kwargs):
            if stage == "physical_evaluation_repair":
                raise StageParseError("physical_evaluation_repair", ["physical_evaluation_repair_failed"])
            return parse_fn(raw)

        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_with_evaluation_recovery(
                broken,
                model_caller=lambda *_a, **_k: {},
                run_stage=run_stage,
                visibility_policy="FORBIDDEN",
                repair_context={},
                product_description=BRIEF,
            )
        self.assertEqual(ctx.exception.stage, "brand_physical")

    def test_valid_candidates_unchanged_during_repair_merge(self) -> None:
        payload = _brand_physical()
        original_p01 = copy.deepcopy(payload["physicalEvaluations"][0])
        replacement = _physical_evaluation(
            "P02",
            eligible=False,
            clearer_than_product_shot=False,
            rejection_codes=["physical_conventional_product_shot"],
        )
        merged = merge_physical_evaluation_replacements(payload, {"P02": replacement})
        self.assertEqual(merged["physicalEvaluations"][0], original_p01)


class TestPhysicalPipelineIntegration(unittest.TestCase):
    def test_successful_repair_allows_graphic_stage(self) -> None:
        responses = _full_final_responses(2)
        physical_payload = copy.deepcopy(_brand_physical())
        physical_payload["physicalEvaluations"][1]["eligible"] = False
        physical_payload["physicalEvaluations"][1]["rejectionCodes"] = []
        physical_payload["physicalEvaluations"][1]["clearerThanConventionalProductShot"] = True
        physical_payload["physicalEvaluations"][1]["supportsTransferredObject"] = True
        physical_payload["physicalEvaluations"][1]["distinctiveToBrand"] = True
        responses[STAGE_BRAND_PHYSICAL_SYSTEM] = physical_payload
        stages: List[str] = []
        repair_attempts = {"count": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            if stage == "physical_evaluation_repair":
                repair_attempts["count"] += 1
                repaired = _physical_evaluation(
                    "P02",
                    eligible=False,
                    clearer_than_product_shot=False,
                    rejection_codes=["physical_conventional_product_shot"],
                )
                return {"physicalEvaluations": [repaired]}
            return copy.deepcopy(responses.get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(repair_attempts["count"], 1)
        self.assertIn("physical_evaluation_repair", stages)
        self.assertIn("graphic_system", stages)

    def test_strategy_candidate_repair_is_counted_via_invoke_model_caller(self) -> None:
        from engine.builder1_planner import _invoke_model_caller

        metrics = Builder1PlanningMetrics()
        token = set_planning_metrics(metrics)
        try:

            def model_caller(system: str, user: str, stage: str | None = None) -> object:
                return {}

            _invoke_model_caller(
                model_caller,
                "system",
                "user",
                stage="strategy_candidate_repair",
            )
            self.assertEqual(metrics.strategy_candidate_repair_calls, 1)
            self.assertEqual(metrics.total_planning_model_calls, 1)
        finally:
            reset_planning_metrics(token)

    def test_physical_evaluation_repair_metric(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_model_call("strategy_slogan_stage")
        metrics.record_model_call("conceptual_stage")
        metrics.record_model_call("brand_physical")
        metrics.record_model_call("physical_evaluation_repair")
        metrics.record_model_call("graphic_system")
        metrics.record_model_call("series_ads")
        self.assertEqual(metrics.physical_evaluation_repair_calls, 1)
        self.assertEqual(metrics.physical_stage_calls, 1)
        self.assertEqual(metrics.total_planning_model_calls, 6)
        self.assertGreater(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_allowed_physical_codes_match_methodology(self) -> None:
        self.assertIn("physical_conventional_product_shot", PHYSICAL_REJECTION_CODES)
        self.assertIn("physical_no_external_object", PHYSICAL_REJECTION_CODES)


if __name__ == "__main__":
    unittest.main()
