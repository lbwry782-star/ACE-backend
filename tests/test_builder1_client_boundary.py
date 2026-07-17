"""
Builder1 digital-agent client-implementation boundary tests.

Run: python -m unittest tests.test_builder1_client_boundary -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_client_boundary import (
    deterministic_client_boundary_checks,
    parse_strategy_boundary_fields,
    scan_text_for_prohibited_client_action,
    strategy_boundary_fields_is_eligible,
    strategy_candidate_is_eligible,
)
from engine.builder1_final_stages import parse_brand_physical_output, parse_series_ads_output
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_plan_spec import campaign_identity_to_dict
from engine.builder1_planner import plan_builder1
from engine.builder1_staged_parsers import (
    StageParseError,
    parse_conceptual_scan,
    parse_strategy_scan,
    parse_strategy_selection,
)
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_staged_planning import (
    LENSES,
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    _brand_physical,
    _conceptual_scan_payload,
    _early_stage_responses,
    _graphic,
    _series_ads,
    _strategy_scan_payload,
)


def _scan_payload_with_boundary_overrides(**overrides: Any) -> Dict[str, Any]:
    payload = _strategy_scan_payload()
    for idx, candidate in enumerate(payload["candidates"], start=1):
        candidate.update(
            {
                "campaignExecutableNow": True,
                "requiresClientConsultation": False,
                "clientActionLevel": "none",
                "implementationCostLevel": "none",
                "simpleStrategicAction": None,
            }
        )
        if idx == 1:
            candidate.update(overrides)
    return payload


class TestStrategyBoundaryFields(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_existing_brief_supported_advantage_passes(self) -> None:
        candidates = parse_strategy_scan(
            _strategy_scan_payload(),
            product_description=self.BRIEF,
        )
        self.assertTrue(strategy_candidate_is_eligible(candidates[0]))

    def test_perceptual_reframing_passes(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0]["relativeAdvantage"] = (
            "Present the reinforced shell as everyday confidence instead of bulk"
        )
        candidates = parse_strategy_scan(payload, product_description=self.BRIEF)
        self.assertTrue(strategy_candidate_is_eligible(candidates[0]))

    def test_optional_zero_cost_action_passes(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0].update(
            {
                "clientActionLevel": "simple_optional",
                "simpleStrategicAction": "Use the existing drop-test proof in every ad.",
            }
        )
        candidates = parse_strategy_scan(payload, product_description=self.BRIEF)
        self.assertTrue(strategy_candidate_is_eligible(candidates[0]))

    def test_new_paid_technical_system_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Launch a new paid technical system to track durability",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "client_implementation_too_complex")

    def test_new_dashboard_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Build a customer reporting dashboard",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "client_implementation_too_complex")

    def test_new_service_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Introduce a new service plan",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "business_transformation_required")

    def test_pricing_change_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Change pricing to win skeptical buyers",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "business_transformation_required")

    def test_discount_not_in_brief_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Offer a limited discount this month",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "unsupported_future_capability")

    def test_guarantee_not_in_brief_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Includes a lifetime guarantee",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "unsupported_future_capability")

    def test_staff_training_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Requires staff training before the claim is true",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "client_implementation_too_complex")

    def test_packaging_redesign_rejected(self) -> None:
        code = scan_text_for_prohibited_client_action(
            "Requires packaging redesign before launch",
            product_description=self.BRIEF,
        )
        self.assertEqual(code, "client_implementation_too_complex")

    def test_material_implementation_cost_rejected(self) -> None:
        fields, _ = parse_strategy_boundary_fields(
            {
                "campaignExecutableNow": True,
                "requiresClientConsultation": False,
                "clientActionLevel": "none",
                "implementationCostLevel": "material",
                "simpleStrategicAction": None,
            },
            candidate_id="S01",
        )
        self.assertFalse(strategy_boundary_fields_is_eligible(fields))

    def test_consultation_required_rejected(self) -> None:
        fields, _ = parse_strategy_boundary_fields(
            {
                "campaignExecutableNow": True,
                "requiresClientConsultation": True,
                "clientActionLevel": "none",
                "implementationCostLevel": "none",
                "simpleStrategicAction": None,
            },
            candidate_id="S01",
        )
        self.assertFalse(strategy_boundary_fields_is_eligible(fields))

    def test_complex_candidate_cannot_win_selection(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0].update(
            {
                "clientActionLevel": "complex_required",
                "implementationCostLevel": "material",
            }
        )
        candidates = parse_strategy_scan(payload, product_description=self.BRIEF)
        eligible = [c for c in candidates if strategy_candidate_is_eligible(c)]
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_selection(
                {
                    "selectedCandidateId": "S01",
                    "selectionReason": "Dramatic",
                    "strategyFamily": "durability",
                    "scores": {"truth": 9},
                },
                eligible,
            )
        self.assertIn("strategy_selection_invalid_id", ctx.exception.reasons)


class TestLaterStageBoundary(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_brand_physical_rejects_business_transformation(self) -> None:
        payload = _brand_physical()
        payload["campaignRationale"] = "Requires a full rebrand before the claim is true"
        with self.assertRaises(StageParseError):
            parse_brand_physical_output(payload, product_description=self.BRIEF)

    def test_series_ads_rejects_future_capability_as_current(self) -> None:
        payload = _series_ads(2)
        words = marketing_text_words(49, prefix="word")
        payload["ads"][0]["marketingText"] = f"{words} now includes reinforced protection."
        with self.assertRaises(StageParseError) as ctx:
            parse_series_ads_output(payload, expected_ad_count=2, product_description=self.BRIEF)
        self.assertTrue(
            any("unsupported_future_capability" in reason for reason in ctx.exception.reasons)
        )

    def test_simple_optional_action_not_consumer_promise(self) -> None:
        plan = {
            "productDescription": self.BRIEF,
            "relativeAdvantage": "Reinforced shell survives daily drops",
            "brandSlogan": "Built To Last",
            "campaignRationale": "Use the existing drop proof in ads.",
            "conceptualGeneratorAction": "Drop and survive",
            "ads": [{"index": 1, "marketingText": marketing_text_words(50)}],
        }
        self.assertEqual(deterministic_client_boundary_checks(plan), [])


class TestPlannerBoundaryIntegration(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_staged_planning_reaches_image_generation(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            responses = _early_stage_responses(2)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            responses[STAGE_SERIES_ADS_SYSTEM] = _series_ads(2)
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        calls: List[int] = []

        def image_caller(_p: str, _f: str) -> bytes:
            calls.append(1)
            return b"jpeg"

        generate_builder1_ad_image(plan, 1, image_caller)
        self.assertEqual(len(calls), 1)

    def test_incremental_ads_do_not_rerun_planning(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            from engine.builder1_campaign_store import (
                clear_memory_store_for_tests,
                create_campaign_session,
                mark_ad_generated,
                try_acquire_generation_lock,
                validate_next_ad_request,
            )
            from tests.test_builder1_series import _parse, _base_campaign

            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="boundary-next", plan=plan)
            try_acquire_generation_lock("boundary-next", 1)
            mark_ad_generated("boundary-next", 1)
            validate_next_ad_request("boundary-next", 2)


class TestPublicApiUnchanged(unittest.TestCase):
    def test_public_campaign_identity_has_no_internal_boundary_fields(self) -> None:
        from engine.builder1_final_stages import assemble_builder1_campaign, parse_brand_physical_output
        from engine.builder1_final_stages import parse_graphic_system_output, parse_series_ads_output
        from engine.builder1_staged_parsers import parse_conceptual_scan, parse_strategy_scan, parse_strategy_selection

        strategy_candidates = parse_strategy_scan(
            _strategy_scan_payload(),
            product_description="Reinforced shell product for daily carry",
        )
        selection, strategy = parse_strategy_selection(
            {
                "selectedCandidateId": "S01",
                "selectionReason": "Best",
                "strategyFamily": "durability",
                "scores": {"truth": 9},
            },
            strategy_candidates,
        )
        conceptual_candidates = parse_conceptual_scan(_conceptual_scan_payload())
        from engine.builder1_staged_parsers import parse_conceptual_selection

        _, conceptual = parse_conceptual_selection(
            {
                "selectedCandidateId": "C01",
                "selectionReason": "Best",
                "scores": {"advantageConnection": 9},
            },
            conceptual_candidates,
        )
        brand = parse_brand_physical_output(_brand_physical())
        graphic = parse_graphic_system_output(_graphic())
        series = parse_series_ads_output(_series_ads(2), expected_ad_count=2)
        plan = assemble_builder1_campaign(
            product_name="",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            ad_count=2,
            detected_language="en",
            exploration_seed="seed",
            strategy=strategy,
            strategy_selection=selection,
            conceptual=conceptual,
            brand_physical=brand,
            graphic=graphic,
            series_ads=series,
        )
        public = campaign_identity_to_dict(plan)
        for key in (
            "campaignExecutableNow",
            "requiresClientConsultation",
            "clientActionLevel",
            "implementationCostLevel",
            "simpleStrategicAction",
        ):
            self.assertNotIn(key, public)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_client_boundary_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_client_boundary", text)


if __name__ == "__main__":
    unittest.main()
