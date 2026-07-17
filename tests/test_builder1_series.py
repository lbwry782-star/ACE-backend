"""
Builder1 campaign-series engine tests.

Run: python -m unittest tests.test_builder1_series -v
"""
from __future__ import annotations

import base64
import io
import unittest
import zipfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    release_generation_lock,
    try_acquire_generation_lock,
    validate_next_ad_request,
)
from engine.builder1_image_generator import ImageRateLimitError, generate_builder1_ad_image
from engine.builder1_input_normalizer import Builder1InputError, normalize_ad_count
from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_plan_spec import (
    ad_to_public_api_dict,
    campaign_identity_to_dict,
)
from engine.builder1_visual_prompt import build_campaign_graphic_identity_block, build_visual_prompt
from engine.builder1_zip import build_builder1_zip_bytes


def _graphic() -> Dict[str, Any]:
    return {
        "palette": {
            "dominant": "#111111",
            "secondary": "#EEEEEE",
            "accent": "#FF5500",
            "background": "#F5F5F5",
            "text": "#222222",
        },
        "layoutTemplate": "visual_right_copy_left",
        "headlinePlacement": "top_left",
        "headlineAlignment": "right",
        "headlineMaxWidthPercent": 34,
        "headlineColor": "#222222",
        "headlineTreatment": "plain",
        "brandBlockPlacement": "bottom_left",
        "sloganPlacement": "bottom_left",
        "copySafeArea": {"side": "left", "widthPercent": 38},
        "imageStyle": "editorial_photography",
        "backgroundTreatment": "solid",
        "borderTreatment": "none",
        "recurringGraphicDevice": "Orange corner bracket",
        "recurringGraphicDeviceRule": "Identical bracket appears on top-left of every ad",
        "framingRule": "Subject cropped with generous negative space on copy side",
    }


def _strategy_scan(ad_count: int = 2) -> Dict[str, Any]:
    lenses = [
        "economic",
        "perceptual",
        "emotional",
        "operational",
        "time",
        "accessibility",
        "expertise",
        "challenger_positioning",
        "participation",
        "simplicity",
    ]
    return {
        "candidates": [
            {
                "lens": lenses[i],
                "problem": f"Problem lens {i}",
                "advantage": f"Advantage lens {i}",
                "briefSupport": "From brief or category",
            }
            for i in range(10)
        ]
    }


def _base_campaign(ad_count: int = 2) -> Dict[str, Any]:
    ads = []
    for i in range(1, ad_count + 1):
        ads.append(
            {
                "index": i,
                "variationLabel": f"var-{i}",
                "newContribution": f"Contribution {i}",
                "physicalExecution": f"Object variant {i}",
                "visualExecution": f"Action variant {i}",
                "sceneDescription": f"Scene description {i}",
                "conceptualExecution": f"Perform stress-test action variant {i}",
                "conceptualActionProof": f"Proof {i} of shared transformation",
                "headline": None if i == 1 else f"Line {i}",
                "headlineNeededReason": "Visual needs support" if i > 1 else "Self-explanatory",
                "marketingText": f"Marketing {i}",
            }
        )
    return {
        "productNameResolved": "TestBrand",
        "detectedLanguage": "en",
        "format": "portrait",
        "adCount": ad_count,
        "strategyCandidateScan": _strategy_scan(ad_count),
        "strategicProblem": "Buyers doubt durability",
        "strategicProblemEvidence": "Category reviews cite breakage",
        "relativeAdvantage": "Survives daily drops",
        "relativeAdvantageSource": "observable_product_mechanism",
        "relativeAdvantageBriefSupport": "Brief mentions reinforced shell",
        "relativeAdvantageClaimRisk": "Low — grounded in product mechanism",
        "problemAdvantageLink": "Durability removes purchase fear",
        "brandSlogan": "Built To Last",
        "sloganDerivation": "From durability advantage",
        "sloganAction": "Trust everyday use",
        "conceptualGenerator": "Stress-test proof",
        "conceptualGeneratorAction": "Drop and survive",
        "conceptualGeneratorInput": "Everyday object",
        "conceptualGeneratorTransformation": "Subject survives impact",
        "conceptualGeneratorResult": "Visible proof of durability",
        "conceptualGeneratorWhyItExpressesAdvantage": "Shows survival not claims",
        "physicalGenerator": "Rubber ball family",
        "physicalGeneratorNaturalPurpose": "Bounce and absorb impact",
        "physicalGeneratorCampaignRole": "Impact survival metaphor",
        "graphicGenerator": _graphic(),
        "seriesGenerator": {
            "type": "situations",
            "principle": "Different drop contexts",
            "progression": "Escalating severity",
        },
        "mediumParticipates": False,
        "mediumRole": "",
        "campaignRationale": "Ownable durability story",
        "ads": ads,
    }


def _parse(data: Dict[str, Any], ad_count: int = 2):
    from engine.builder1_plan_parser import parse_builder1_series_plan

    return parse_builder1_series_plan(
        data,
        expected_format="portrait",
        expected_ad_count=ad_count,
        product_name="",
        product_description="Reinforced shell product for daily carry",
    )


class TestBuilder1SeriesParser(unittest.TestCase):
    def test_valid_2_ad_plan_parses(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        self.assertEqual(plan.ad_count, 2)

    def test_valid_4_ad_plan_parses(self) -> None:
        plan = _parse(_base_campaign(4), 4)
        self.assertEqual(len(plan.ads), 4)

    def test_ad_count_4_preserved(self) -> None:
        self.assertEqual(normalize_ad_count(4), 4)

    def test_ad_count_1_rejected(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count(1)

    def test_string_ad_count_rejected(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count("2")

    def test_unsupported_capability_rejected(self) -> None:
        data = _base_campaign(2)
        data["relativeAdvantage"] = "Live dashboard shows measurable sales increase"
        data["relativeAdvantageSource"] = "category_inference"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="Simple ad tool",
        )
        self.assertIn("unsupported_product_capability", reasons)

    def test_conceptual_equals_physical_rejected(self) -> None:
        data = _base_campaign(2)
        data["conceptualGenerator"] = data["physicalGenerator"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("conceptual_equals_physical_generator", reasons)

    def test_missing_conceptual_execution_rejected(self) -> None:
        data = _base_campaign(2)
        del data["ads"][0]["conceptualExecution"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("missing_conceptual_execution", reasons)

    def test_strategy_scan_requires_diverse_lenses(self) -> None:
        data = _base_campaign(2)
        data["strategyCandidateScan"] = {"candidates": [{"lens": "economic", "problem": "a", "advantage": "b"}]}
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("strategy_scan_insufficient_candidates", reasons)

    def test_graphic_generator_concrete_fields(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        g = plan.graphic_generator
        self.assertEqual(g.layout_template, "visual_right_copy_left")
        self.assertEqual(g.copy_safe_area.width_percent, 38)
        self.assertTrue(g.recurring_graphic_device)


class TestBuilder1VisualPrompt(unittest.TestCase):
    def test_identical_graphic_block_all_ads(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        block = build_campaign_graphic_identity_block(plan)
        prompts = [build_visual_prompt(plan, ad) for ad in plan.ads]
        for p in prompts:
            self.assertIn(block, p)
        self.assertIn(block, prompts[0])
        self.assertEqual(prompts[0].count("=== CAMPAIGN GRAPHIC IDENTITY"), 1)

    def test_conceptual_execution_in_each_prompt(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        for ad in plan.ads:
            prompt = build_visual_prompt(plan, ad)
            self.assertIn(ad.conceptual_execution, prompt)

    def test_copy_safe_area_reserved(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("Copy-safe area", prompt)
        self.assertIn("38%", prompt)


class TestBuilder1SingleImage(unittest.TestCase):
    def test_one_image_per_call(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        calls = []

        def caller(prompt: str, fmt: str) -> bytes:
            calls.append(1)
            return b"img"

        generate_builder1_ad_image(plan, 1, caller)
        self.assertEqual(len(calls), 1)

    def test_rate_limit_is_retryable(self) -> None:
        plan = _parse(_base_campaign(2), 2)

        def caller(_p: str, _f: str) -> bytes:
            raise ImageRateLimitError(retry_after_seconds=12)

        with self.assertRaises(ImageRateLimitError) as ctx:
            generate_builder1_ad_image(plan, 1, caller)
        self.assertEqual(ctx.exception.retry_after_seconds, 12)


class TestBuilder1CampaignSession(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_incremental_generation_flow(self) -> None:
        plan = _parse(_base_campaign(4), 4)
        create_campaign_session(campaign_id="c1", plan=plan)
        try_acquire_generation_lock("c1", 1)
        mark_ad_generated("c1", 1)
        session = get_campaign_session("c1")
        self.assertEqual(session.generated_indexes, [1])
        self.assertEqual(session.next_ad_index, 2)

    def test_next_index_conflict(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="c2", plan=plan)
        with self.assertRaises(CampaignStoreError) as ctx:
            validate_next_ad_request("c2", 2)
        self.assertEqual(ctx.exception.code, "campaign_index_conflict")

    def test_duplicate_generation_lock(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="c5", plan=plan)
        try_acquire_generation_lock("c5", 1)
        with self.assertRaises(CampaignStoreError) as ctx:
            try_acquire_generation_lock("c5", 1)
        self.assertEqual(ctx.exception.code, "campaign_generation_in_progress")
        release_generation_lock("c5")

    def test_campaign_complete_blocks_next(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="c3", plan=plan)
        try_acquire_generation_lock("c3", 1)
        mark_ad_generated("c3", 1)
        try_acquire_generation_lock("c3", 2)
        mark_ad_generated("c3", 2)
        session = get_campaign_session("c3")
        self.assertTrue(session.complete)
        with self.assertRaises(CampaignStoreError) as ctx:
            validate_next_ad_request("c3", 3)
        self.assertEqual(ctx.exception.code, "campaign_complete")


class TestBuilder1PublicResponse(unittest.TestCase):
    def test_no_internal_fields_in_public_campaign(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        public = campaign_identity_to_dict(plan)
        self.assertNotIn("strategyCandidateScan", public)
        self.assertNotIn("strategyJudgeResult", public)
        ad_public = ad_to_public_api_dict(plan.ads[0], visual_prompt="p", image_base64="x")
        self.assertNotIn("sceneDescription", ad_public)
        self.assertNotIn("variationLabel", ad_public)


class TestBuilder1PlannerIntegration(unittest.TestCase):
    def test_next_ad_does_not_call_planner(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner should not be called for next ad")
            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="c4", plan=plan)
            try_acquire_generation_lock("c4", 1)
            mark_ad_generated("c4", 1)
            validate_next_ad_request("c4", 2)


class TestBuilder1Zip(unittest.TestCase):
    def test_zip_creates_expected_files(self) -> None:
        img = base64.b64encode(b"fakejpeg").decode()
        payload = {
            "productName": "BrandX",
            "brandSlogan": "Stay Sharp",
            "ads": [
                {"index": 1, "imageBase64": img, "headline": None, "marketingText": "One"},
                {"index": 2, "imageBase64": img, "headline": "Hi", "marketingText": "Two"},
            ],
        }
        zbytes = build_builder1_zip_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            self.assertIn("campaign.txt", zf.namelist())
            self.assertIn("ad-01.jpg", zf.namelist())


if __name__ == "__main__":
    unittest.main()
