"""
Builder1 campaign-series engine tests.

Run: python -m unittest tests.test_builder1_series -v
"""
from __future__ import annotations

from tests.builder1_test_helpers import marketing_text_words, pass_compliance_reviewer

import base64
import io
import unittest
import zipfile
from typing import Any, Dict
from unittest.mock import patch

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
from engine.builder1_visual_prompt import (
    build_campaign_graphic_identity_block,
    build_text_to_render_block,
    build_visual_prompt,
)
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
        "brandBlockPlacement": "bottom_left",
        "sloganPlacement": "bottom_left",
        "copySafeArea": {"side": "left", "widthPercent": 38},
        "typographyStyle": "bold_geometric_sans",
        "headlineScale": "large",
        "brandScale": "small",
        "sloganScale": "medium",
        "imageStyle": "editorial_photography",
        "backgroundTreatment": "solid",
        "borderTreatment": "none",
        "recurringGraphicDevice": "Orange corner bracket",
        "recurringGraphicDeviceRule": "Identical bracket appears on top-left of every ad",
        "shapeLanguage": "Angular geometric frames",
        "framingRule": "Subject cropped with generous negative space on copy side",
        "spacingRule": "Wide outer margins with tight copy grouping",
        "sloganPlacementReason": "",
    }


def _strategy_scan() -> Dict[str, Any]:
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
        "specialization",
        "category_convention",
    ]
    families = [
        {"family": f"family_{i}", "candidateIndexes": [i], "score": 80 + i}
        for i in range(5)
    ]
    return {
        "candidates": [
            {
                "lens": lenses[i],
                "problem": f"Distinct problem {i}",
                "advantage": f"Distinct advantage {i}",
                "briefSupport": "From brief or category",
                "strategyFamily": f"family_{i % 5}",
            }
            for i in range(12)
        ],
        "families": families,
    }


def _conceptual_scan() -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "conceptualGenerator": f"Action mechanism {i}",
                "conceptualGeneratorAction": f"Perform action {i}",
                "conceptualGeneratorInput": "Everyday object",
                "conceptualGeneratorTransformation": f"Transform step {i}",
                "conceptualGeneratorResult": f"Visible proof {i}",
                "conceptualGeneratorWhyItExpressesAdvantage": "Shows advantage through action",
                "conceptualGeneratorSeriesPotential": "Supports multiple proofs",
            }
            for i in range(6)
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
                "marketingText": marketing_text_words(50, prefix=f"m{i}"),
            }
        )
    return {
        "productNameResolved": "TestBrand",
        "detectedLanguage": "en",
        "format": "portrait",
        "adCount": ad_count,
        "strategyCandidateScan": _strategy_scan(),
        "conceptualGeneratorScan": _conceptual_scan(),
        "strategyFamily": "durability_proof",
        "strategyScore": 88,
        "campaignExplorationSeed": "seed-abc-123",
        "selectionReason": "Strongest brief-supported family after seed tie-break",
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
        "transferredObject": "Rubber ball family",
        "transferredObjectAction": "Bounces after a drop without cracking",
        "productVisibilityPolicy": "FORBIDDEN",
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
        self.assertEqual(_parse(_base_campaign(2), 2).ad_count, 2)

    def test_strategy_families_required(self) -> None:
        data = _base_campaign(2)
        data["strategyCandidateScan"]["families"] = data["strategyCandidateScan"]["families"][:2]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("strategy_scan_insufficient_families", reasons)

    def test_conceptual_scan_required(self) -> None:
        data = _base_campaign(2)
        del data["conceptualGeneratorScan"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("conceptual_scan_missing", reasons)

    def test_vague_conceptual_in_scan_rejected(self) -> None:
        data = _base_campaign(2)
        data["conceptualGeneratorScan"]["candidates"][0]["conceptualGenerator"] = "visibility"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("conceptual_scan_candidate_too_vague", reasons)

    def test_unsupported_capability_rejected(self) -> None:
        data = _base_campaign(2)
        data["relativeAdvantage"] = "Live dashboard with transparency system"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="Simple ad tool",
        )
        self.assertIn("unsupported_product_capability", reasons)


class TestBuilder1VisualPrompt(unittest.TestCase):
    def test_complete_finished_advertisement(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("complete finished advertisement", prompt.lower())

    def test_no_global_no_text_constraint(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertNotIn("No written text, letters, words", prompt)

    def test_exact_brand_and_slogan_in_prompt(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn('"TestBrand"', prompt)
        self.assertIn('"Built To Last"', prompt)

    def test_headline_exact_when_present(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[1])
        self.assertIn('"Line 2"', prompt)

    def test_null_headline_no_placeholder(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        block = build_text_to_render_block(plan, plan.ads[0])
        self.assertIn("do not render any headline", block.lower())
        self.assertNotIn('Headline:\n"Placeholder"', block)

    def test_prohibits_extra_copy(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("Prohibit any text beyond", prompt)

    def test_identical_graphic_block(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        block = build_campaign_graphic_identity_block(plan)
        prompts = [build_visual_prompt(plan, ad) for ad in plan.ads]
        for p in prompts:
            self.assertIn(block, p)
            self.assertIn("#111111", p)
            self.assertIn("bold_geometric_sans", p)
            self.assertIn("Orange corner bracket", p)

    def test_full_frame_and_no_mockups(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("fills the entire image frame", prompt.lower())
        self.assertIn("billboard", prompt.lower())

    def test_conceptual_execution_in_prompt(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(plan.ads[0].conceptual_execution, prompt)
        self.assertIn(plan.relative_advantage, prompt)


class TestBuilder1SingleImage(unittest.TestCase):
    def test_one_image_per_call(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        calls = []

        def caller(prompt: str, fmt: str) -> bytes:
            calls.append(1)
            return b"img"

        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=pass_compliance_reviewer)
        self.assertEqual(len(calls), 1)

    def test_rate_limit_is_retryable(self) -> None:
        plan = _parse(_base_campaign(2), 2)

        def caller(_p: str, _f: str) -> bytes:
            raise ImageRateLimitError(retry_after_seconds=12)

        with self.assertRaises(ImageRateLimitError) as ctx:
            generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=pass_compliance_reviewer)
        self.assertEqual(ctx.exception.retry_after_seconds, 12)


class TestBuilder1CampaignSession(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_incremental_flow(self) -> None:
        plan = _parse(_base_campaign(4), 4)
        create_campaign_session(campaign_id="c1", plan=plan)
        try_acquire_generation_lock("c1", 1)
        mark_ad_generated("c1", 1)
        session = get_campaign_session("c1")
        self.assertEqual(session.next_ad_index, 2)

    def test_next_ad_no_planner(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="c2", plan=plan)
            try_acquire_generation_lock("c2", 1)
            mark_ad_generated("c2", 1)
            validate_next_ad_request("c2", 2)


class TestBuilder1PublicResponse(unittest.TestCase):
    def test_image_contains_final_copy_flag(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        ad = ad_to_public_api_dict(plan.ads[0], visual_prompt="p", image_base64="x")
        self.assertTrue(ad["imageContainsFinalCopy"])

    def test_no_internal_scan_in_campaign(self) -> None:
        public = campaign_identity_to_dict(_parse(_base_campaign(2), 2))
        self.assertNotIn("strategyCandidateScan", public)
        self.assertNotIn("conceptualGeneratorScan", public)


class TestBuilder1Zip(unittest.TestCase):
    def test_zip_still_works(self) -> None:
        img = base64.b64encode(b"fakejpeg").decode()
        text50 = marketing_text_words(50)
        zbytes = build_builder1_zip_bytes(
            {
                "productName": "BrandX",
                "brandSlogan": "Stay Sharp",
                "ads": [
                    {"index": 1, "imageBase64": img, "headline": None, "marketingText": text50},
                    {"index": 2, "imageBase64": img, "headline": "Hi", "marketingText": marketing_text_words(50, "two")},
                ],
            }
        )
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            self.assertIn("ad-01.jpg", zf.namelist())


if __name__ == "__main__":
    unittest.main()
