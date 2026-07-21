"""
Builder1 automatic product-name resolution tests.

Run: python -m unittest tests.test_builder1_product_name -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    mark_ad_generated,
    try_acquire_generation_lock,
    validate_next_ad_request,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_plan_spec import campaign_identity_to_dict
from engine.builder1_planning_contract import STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_product_name import parse_product_name_resolution, validate_resolved_product_name
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
    _brand_physical,
    _early_stage_responses,
    _full_final_responses,
    _graphic,
    _series_ads,
)


EN_BRIEF = "Reinforced shell product for daily carry"
HE_BRIEF = "מוצר עם מעטפת מחוזקת לשימוש יומיומי"
GENERATED_NAME = "ShellGuard"


def _planner_responses(
    *,
    ad_count: int = 2,
    generated_name: str = GENERATED_NAME,
    brand_name_override: str | None = None,
) -> Dict[str, Any]:
    responses = _full_final_responses(ad_count)
    responses[STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM] = {"productNameResolved": generated_name}
    brand = _brand_physical()
    brand["productNameResolved"] = brand_name_override or generated_name
    responses[STAGE_BRAND_PHYSICAL_SYSTEM] = brand
    return responses


class TestProductNameRouteValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())
        from app import app

        cls.client = app.test_client()

    @patch("app._builder1_executor.submit")
    def test_empty_product_name_with_description_is_accepted(self, mock_submit: Any) -> None:
        resp = self.client.post(
            "/api/builder1-generate",
            json={
                "productName": "",
                "productDescription": EN_BRIEF,
                "adCount": 2,
            },
        )
        self.assertEqual(resp.status_code, 202)
        self.assertIn("campaignId", resp.get_json())
        mock_submit.assert_called_once()

    @patch("app._builder1_executor.submit")
    def test_whitespace_product_name_with_description_is_accepted(self, mock_submit: Any) -> None:
        resp = self.client.post(
            "/api/builder1-generate",
            json={
                "productName": "   ",
                "productDescription": EN_BRIEF,
                "adCount": 2,
            },
        )
        self.assertEqual(resp.status_code, 202)
        mock_submit.assert_called_once()

    @patch("app._builder1_executor.submit")
    def test_empty_name_and_empty_description_rejected(self, mock_submit: Any) -> None:
        resp = self.client.post(
            "/api/builder1-generate",
            json={"productName": "", "productDescription": "", "adCount": 2},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "invalid_input")
        mock_submit.assert_not_called()


class TestProductNameResolutionStage(unittest.TestCase):
    def test_empty_product_name_invokes_naming_stage_once(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            calls.append(system)
            responses = _planner_responses(ad_count=2)
            return copy.deepcopy(responses.get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan = plan_builder1(
            product_name="",
            product_description=EN_BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.product_name_resolved, GENERATED_NAME)
        self.assertEqual(calls.count(STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM), 1)
        self.assertGreater(calls.count(STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM), 0)

    def test_supplied_product_name_bypasses_naming_stage(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            calls.append(system)
            responses = _planner_responses(ad_count=2, generated_name="IgnoredName")
            return copy.deepcopy(responses.get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan = plan_builder1(
            product_name="UserBrand",
            product_description=EN_BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.product_name_resolved, "UserBrand")
        self.assertNotIn(STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM, calls)

    def test_generated_name_reaches_strategy_scan(self) -> None:
        seen: Dict[str, str] = {}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM:
                seen["strategy_user_prompt"] = user
            responses = _planner_responses(ad_count=2)
            return copy.deepcopy(responses.get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan_builder1(
            product_name="",
            product_description=EN_BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertIn(GENERATED_NAME, seen["strategy_user_prompt"])

    def test_later_stage_cannot_rename_generated_name(self) -> None:
        plan = plan_builder1(
            product_name="",
            product_description=EN_BRIEF,
            format_value="portrait",
            model_caller=lambda system, user, stage=None: copy.deepcopy(
                _planner_responses(ad_count=2, brand_name_override="WrongBrand").get(
                    system, {"pass": True, "rejectionReasonCodes": []}
                )
            ),
            ad_count=2,
        )
        self.assertEqual(plan.product_name_resolved, GENERATED_NAME)

    def test_same_generated_name_in_all_ads_image_prompts_and_marketing_repair(self) -> None:
        marketing_names: List[str] = []
        original = __import__(
            "engine.builder1_marketing_text_repair",
            fromlist=["ensure_series_ads_marketing_text"],
        ).ensure_series_ads_marketing_text

        def capture(ads, **kwargs):
            marketing_names.append(kwargs["product_name"])
            return original(ads, **kwargs)

        with patch("engine.builder1_planner.ensure_series_ads_marketing_text", side_effect=capture):
            plan = plan_builder1(
                product_name="",
                product_description=EN_BRIEF,
                format_value="portrait",
                model_caller=lambda system, user, stage=None: copy.deepcopy(
                    _planner_responses(ad_count=2).get(system, {"pass": True, "rejectionReasonCodes": []})
                ),
                ad_count=2,
            )
        self.assertEqual(plan.product_name_resolved, GENERATED_NAME)
        self.assertTrue(all(name == GENERATED_NAME for name in marketing_names))
        for ad in plan.ads:
            prompt = build_visual_prompt(plan, ad)
            self.assertIn(GENERATED_NAME, prompt)

    def test_generated_name_stored_in_campaign_session(self) -> None:
        plan = plan_builder1(
            product_name="",
            product_description=EN_BRIEF,
            format_value="portrait",
            model_caller=lambda system, user, stage=None: copy.deepcopy(
                _planner_responses(ad_count=2).get(system, {"pass": True, "rejectionReasonCodes": []})
            ),
            ad_count=2,
        )
        clear_memory_store_for_tests()
        create_campaign_session(campaign_id="name-session", plan=plan)
        session = validate_next_ad_request("name-session", 1)
        self.assertEqual(session.plan.product_name_resolved, GENERATED_NAME)
        public = campaign_identity_to_dict(session.plan)
        self.assertEqual(public["productNameResolved"], GENERATED_NAME)

    def test_generate_again_does_not_rerun_naming(self) -> None:
        from dataclasses import replace

        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            clear_memory_store_for_tests()
            plan = replace(_parse(_base_campaign(2), 2), product_name_resolved=GENERATED_NAME)
            create_campaign_session(campaign_id="name-next", plan=plan)
            try_acquire_generation_lock("name-next", 1)
            mark_ad_generated("name-next", 1)
            validate_next_ad_request("name-next", 2)
            mock_plan.assert_not_called()

    def test_naming_failure_returns_product_name_generation_failed(self) -> None:
        copied = EN_BRIEF

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM:
                return {"productNameResolved": copied}
            return {"pass": True, "rejectionReasonCodes": []}

        with self.assertRaises(Builder1PlannerError) as ctx:
            plan_builder1(
                product_name="",
                product_description=EN_BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(str(ctx.exception), "product_name_generation_failed")


class TestProductNameLanguageRules(unittest.TestCase):
    def test_english_description_requires_english_name(self) -> None:
        reasons = validate_resolved_product_name(
            "שellGuard",
            product_description=EN_BRIEF,
            detected_language="en",
        )
        self.assertIn("product_name_resolution_wrong_language", reasons)

    def test_hebrew_description_allows_hebrew_or_english_name(self) -> None:
        self.assertEqual(
            validate_resolved_product_name(
                "מגן יומי",
                product_description=HE_BRIEF,
                detected_language="he",
            ),
            [],
        )
        self.assertEqual(
            validate_resolved_product_name(
                "DailyGuard",
                product_description=HE_BRIEF,
                detected_language="he",
            ),
            [],
        )

    def test_description_not_copied_verbatim(self) -> None:
        with self.assertRaises(Exception) as ctx:
            parse_product_name_resolution(
                {"productNameResolved": EN_BRIEF},
                product_description=EN_BRIEF,
                detected_language="en",
            )
        self.assertIn("product_name_resolution_copied_description", ctx.exception.reasons)

    def test_generic_category_label_rejected(self) -> None:
        with self.assertRaises(Exception) as ctx:
            parse_product_name_resolution(
                {"productNameResolved": "Restaurant"},
                product_description=EN_BRIEF,
                detected_language="en",
            )
        self.assertIn("product_name_resolution_generic_category", ctx.exception.reasons)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_no_builder2_product_name_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_product_name", text)


if __name__ == "__main__":
    unittest.main()
