"""
Builder1 strategy_scan input validation and candidate-level repair tests.

Run: python -m unittest tests.test_builder1_strategy_scan_repair -v
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_client_boundary import (
    normalize_simple_strategic_action_value,
    parse_strategy_boundary_fields,
    strategy_candidate_is_eligible,
)
from engine.builder1_input_normalizer import Builder1InputError, normalize_builder1_input
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_staged_parsers import StageParseError, parse_strategy_scan, parse_strategy_selection
from engine.builder1_strategy_scan import (
    STRATEGY_SCAN_REPLACEMENT_SYSTEM,
    ensure_strategy_scan_from_raw,
    merge_strategy_scan_replacements,
    parse_strategy_scan_replacements,
    validate_strategy_candidate_item,
    validate_strategy_scan_set,
)
from tests.test_builder1_staged_planning import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    _brand_physical,
    _early_stage_responses,
    _graphic,
    _series_ads,
    _strategy_scan_payload,
    _strategy_selection_payload,
)


BRIEF = "Reinforced shell product for daily carry"


class TestStrategyCandidateValidation(unittest.TestCase):
    def test_fabricated_statistics_rejected_with_candidate_id(self) -> None:
        item = _strategy_scan_payload()["candidates"][3]
        item = copy.deepcopy(item)
        item["briefSupport"] = "Research shows 87% of buyers prefer reinforced shells"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S04",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNone(result.candidate)
        self.assertTrue(
            any("strategy_scan_S04_unsupported_grounding_claim" in r for r in result.reasons)
        )

    def test_brief_grounded_candidate_passes(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["briefSupport"] = "The brief describes a reinforced shell for daily carry"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.reasons, [])

    def test_category_inference_without_empirical_claim_passes(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["briefSupport"] = "Category inference: reinforced shells typically protect everyday carry items"
        item["advantageSource"] = "category_inference"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNotNone(result.candidate)

    def test_none_action_with_none_client_action_passes(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["clientActionLevel"] = "none"
        item["simpleStrategicAction"] = None
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNotNone(result.candidate)

    def test_empty_string_action_normalizes_to_null(self) -> None:
        self.assertIsNone(normalize_simple_strategic_action_value(""))
        self.assertIsNone(normalize_simple_strategic_action_value("none"))
        fields, reasons = parse_strategy_boundary_fields(
            {
                "campaignExecutableNow": True,
                "requiresClientConsultation": False,
                "clientActionLevel": "none",
                "implementationCostLevel": "none",
                "simpleStrategicAction": "   ",
            },
            candidate_id="S01",
        )
        self.assertEqual(reasons, [])
        self.assertIsNone(fields.simple_strategic_action)

    def test_none_level_with_real_action_invalid(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["clientActionLevel"] = "none"
        item["simpleStrategicAction"] = "Add a new loyalty program"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNone(result.candidate)
        self.assertIn("strategy_scan_S01_invalid_simple_strategic_action", result.reasons)

    def test_simple_optional_with_valid_action_passes(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item.update(
            {
                "clientActionLevel": "simple_optional",
                "simpleStrategicAction": "Use the existing drop-test proof in every ad.",
            }
        )
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNotNone(result.candidate)
        self.assertTrue(strategy_candidate_is_eligible(result.candidate))

    def test_simple_optional_with_null_action_invalid(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item.update(
            {
                "clientActionLevel": "simple_optional",
                "simpleStrategicAction": None,
            }
        )
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIn("strategy_scan_S01_invalid_simple_strategic_action", result.reasons)

    def test_complex_required_ineligible(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["clientActionLevel"] = "complex_required"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertIsNotNone(result.candidate)
        self.assertFalse(strategy_candidate_is_eligible(result.candidate))

    def test_material_implementation_ineligible(self) -> None:
        item = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
        item["implementationCostLevel"] = "material"
        result = validate_strategy_candidate_item(
            item,
            candidate_id="S01",
            product_description=BRIEF,
            exact_sigs=set(),
        )
        self.assertFalse(strategy_candidate_is_eligible(result.candidate))


class TestStrategyScanCandidateRepair(unittest.TestCase):
    def test_one_invalid_candidate_triggers_focused_replacement(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0]["simpleStrategicAction"] = "Run a workshop first"
        payload["candidates"][7]["briefSupport"] = "Surveys show 90% prefer reinforced shells"

        replacement_calls: List[str] = []

        def replacement_caller(system: str, user: str, **kwargs: Any) -> Dict[str, Any]:
            replacement_calls.append(user)
            return {
                "replacements": [
                    {
                        **_strategy_scan_payload()["candidates"][0],
                        "simpleStrategicAction": None,
                    },
                    {
                        **_strategy_scan_payload()["candidates"][7],
                        "briefSupport": "Follows from brief reinforced shell mention",
                    },
                ]
            }

        result = ensure_strategy_scan_from_raw(
            payload,
            product_name="TestBrand",
            product_description=BRIEF,
            model_caller=replacement_caller,
        )
        self.assertEqual(len(result), 12)
        self.assertEqual(len(replacement_calls), 1)
        self.assertIn("S01", replacement_calls[0])
        self.assertIn("S08", replacement_calls[0])

    def test_valid_candidates_preserved_byte_for_byte(self) -> None:
        payload = _strategy_scan_payload()
        original_s02 = copy.deepcopy(payload["candidates"][1])
        payload["candidates"][0]["simpleStrategicAction"] = "Needs client workshop"

        invalid, _valid, valid_raw = validate_strategy_scan_set(
            payload["candidates"],
            product_description=BRIEF,
        )
        self.assertIn("S01", invalid)
        self.assertIn("S02", valid_raw)
        self.assertEqual(valid_raw["S02"], original_s02)

        replacements = {
            "S01": {
                **_strategy_scan_payload()["candidates"][0],
                "simpleStrategicAction": None,
            }
        }
        merged = merge_strategy_scan_replacements(
            payload["candidates"],
            replacements,
            valid_raw=valid_raw,
        )
        merged_by_id = {c["id"]: c for c in merged}
        self.assertEqual(merged_by_id["S02"], original_s02)

    def test_replacement_ids_must_match_invalid_ids(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_scan_replacements(
                {"replacements": [{"id": "S02", **_strategy_scan_payload()["candidates"][1]}]},
                allowed_ids=["S01"],
            )
        self.assertTrue(
            any("strategy_candidate_repair_unexpected_id" in r for r in ctx.exception.reasons)
        )

    def test_repaired_scan_reaches_strategy_selection(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0]["simpleStrategicAction"] = "Do something now"

        def replacement_caller(_system: str, _user: str, **kwargs: Any) -> Dict[str, Any]:
            fixed = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
            fixed["simpleStrategicAction"] = None
            return {"replacements": [fixed]}

        candidates = ensure_strategy_scan_from_raw(
            payload,
            product_name="TestBrand",
            product_description=BRIEF,
            model_caller=replacement_caller,
        )
        eligible = [c for c in candidates if strategy_candidate_is_eligible(c)]
        selection, _selected, _reviews = parse_strategy_selection(
            _strategy_selection_payload(selected_id="S01"),
            eligible,
        )
        self.assertEqual(selection.selected_candidate_id, "S01")


class TestPlannerStrategyScanRepairIntegration(unittest.TestCase):
    def test_repaired_strategy_scan_continues_staged_planning(self) -> None:
        broken_scan = _strategy_scan_payload()
        broken_scan["candidates"][0]["simpleStrategicAction"] = "Hold interviews first"

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STRATEGY_SCAN_REPLACEMENT_SYSTEM:
                fixed = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
                fixed["simpleStrategicAction"] = None
                return {"replacements": [fixed]}
            responses = _early_stage_responses(2)
            responses[STAGE_STRATEGY_SCAN_SYSTEM] = broken_scan
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            responses[STAGE_SERIES_ADS_SYSTEM] = _series_ads(2)
            return copy.deepcopy(responses.get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan = plan_builder1(
            product_name="TestBrand",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)

    def test_incremental_generation_does_not_rerun_planning(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            from engine.builder1_campaign_store import (
                clear_memory_store_for_tests,
                create_campaign_session,
                mark_ad_generated,
                try_acquire_generation_lock,
                validate_next_ad_request,
            )
            from tests.test_builder1_series import _base_campaign, _parse

            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="strategy-scan-next", plan=plan)
            try_acquire_generation_lock("strategy-scan-next", 1)
            mark_ad_generated("strategy-scan-next", 1)
            validate_next_ad_request("strategy-scan-next", 2)
            mock_plan.assert_not_called()


class TestBuilder2Unchanged(unittest.TestCase):
    def test_no_builder2_strategy_scan_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_strategy_scan", text)


class TestInputNormalizerAllowsEmptyNameForInternalCalls(unittest.TestCase):
    def test_normalize_does_not_require_product_name(self) -> None:
        normalized = normalize_builder1_input(
            "",
            BRIEF,
            "portrait",
            ad_count=2,
        )
        self.assertEqual(normalized.product_name, "")


if __name__ == "__main__":
    unittest.main()
