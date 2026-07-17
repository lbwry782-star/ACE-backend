"""
Builder1 campaign-series engine tests.

Run: python -m unittest tests.test_builder1_series -v
"""
from __future__ import annotations

import base64
import io
import unittest
import zipfile
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_image_generator import generate_builder1_series_images
from engine.builder1_input_normalizer import Builder1InputError, normalize_ad_count
from engine.builder1_plan_parser import (
    Builder1SeriesPlanParseError,
    parse_builder1_series_plan,
    validate_series_plan_structure,
)
from engine.builder1_plan_spec import (
    Builder1AdPlan,
    Builder1CompositionGrid,
    Builder1GraphicGenerator,
    Builder1SeriesGenerator,
    Builder1SeriesPlan,
    Builder1Typography,
)
from engine.builder1_visual_prompt import build_visual_prompt
from engine.builder1_zip import build_builder1_zip_bytes


def _graphic() -> Dict[str, Any]:
    return {
        "colorPalette": ["#111111", "#FFFFFF"],
        "typography": {
            "headlineStyle": "Bold sans",
            "sloganStyle": "Light sans",
            "brandStyle": "Small caps",
        },
        "composition": {
            "grid": "12-col",
            "visualArea": "top 70%",
            "copyArea": "bottom 30%",
            "alignment": "left",
            "sloganPlacement": "bottom-left",
            "brandPlacement": "bottom-right",
        },
        "imageStyle": "Clean studio photography",
        "spacing": "Generous margins",
        "visualTreatment": "High contrast",
        "backgroundTreatment": "Soft gradient",
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
        "strategicProblem": "Buyers doubt durability",
        "strategicProblemEvidence": "Category reviews cite breakage",
        "relativeAdvantage": "Survives daily drops",
        "problemAdvantageLink": "Durability removes purchase fear",
        "brandSlogan": "Built To Last",
        "sloganDerivation": "From durability advantage",
        "sloganAction": "Trust everyday use",
        "conceptualGenerator": "Stress-test proof",
        "conceptualGeneratorAction": "Show survival moments",
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


def _series_plan_from_dict(data: Dict[str, Any], ad_count: int = 2) -> Builder1SeriesPlan:
    return parse_builder1_series_plan(
        data,
        expected_format="portrait",
        expected_ad_count=ad_count,
        product_name="",
        product_description="Test product",
    )


class TestBuilder1SeriesParser(unittest.TestCase):
    def test_valid_2_ad_plan_parses(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(2), 2)
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(len(plan.ads), 2)

    def test_valid_4_ad_plan_parses(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(4), 4)
        self.assertEqual(len(plan.ads), 4)

    def test_ad_count_1_rejected_via_input(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count(1)

    def test_ad_count_5_rejected_via_input(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count(5)

    def test_string_ad_count_rejected(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count("2")

    def test_plan_count_mismatch_rejected(self) -> None:
        data = _base_campaign(2)
        data["adCount"] = 3
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=3,
            product_name="",
            product_description="x",
        )
        self.assertIn("ads_length_mismatch", reasons)

    def test_missing_slogan_rejected(self) -> None:
        data = _base_campaign(2)
        data["brandSlogan"] = ""
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("missing_brand_slogan", reasons)

    def test_slogan_over_max_words_rejected(self) -> None:
        data = _base_campaign(2)
        data["brandSlogan"] = "one two three four five six seven"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("brand_slogan_too_long", reasons)

    def test_missing_graphic_field_rejected(self) -> None:
        data = _base_campaign(2)
        del data["graphicGenerator"]["imageStyle"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("graphic_generator_missing_style_fields", reasons)

    def test_blank_headline_normalizes_to_null(self) -> None:
        data = _base_campaign(2)
        data["ads"][0]["headline"] = "   "
        plan = _series_plan_from_dict(data, 2)
        self.assertIsNone(plan.ads[0].headline)

    def test_headline_over_7_words_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["headline"] = "one two three four five six seven eight"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("headline_too_long", reasons)

    def test_duplicate_ad_indexes_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["index"] = 1
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("duplicate_ad_index", reasons)

    def test_duplicate_physical_executions_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["physicalExecution"] = data["ads"][0]["physicalExecution"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("duplicate_physical_execution", reasons)

    def test_duplicate_visual_executions_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["visualExecution"] = data["ads"][0]["visualExecution"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("duplicate_visual_execution", reasons)

    def test_duplicate_scene_descriptions_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["sceneDescription"] = data["ads"][0]["sceneDescription"]
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("duplicate_scene_description", reasons)

    def test_headline_only_variation_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][1]["physicalExecution"] = data["ads"][0]["physicalExecution"]
        data["ads"][1]["visualExecution"] = data["ads"][0]["visualExecution"]
        data["ads"][1]["sceneDescription"] = data["ads"][0]["sceneDescription"]
        data["ads"][1]["headline"] = "Different headline only"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertTrue(
            "headline_only_variation" in reasons or "duplicate_physical_execution" in reasons
        )

    def test_per_ad_slogan_rejected(self) -> None:
        data = _base_campaign(2)
        data["ads"][0]["brandSlogan"] = "Other"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("per_ad_slogan_forbidden", reasons)

    def test_old_object_ab_structure_rejected(self) -> None:
        data = _base_campaign(2)
        data["objectA"] = "microphone"
        data["objectB"] = "ice cream"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("legacy_field_present:objectA", reasons)

    def test_medium_role_inconsistency_rejected(self) -> None:
        data = _base_campaign(2)
        data["mediumParticipates"] = False
        data["mediumRole"] = "Billboard shows idea"
        _, reasons = validate_series_plan_structure(
            data,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description="x",
        )
        self.assertIn("medium_role_forbidden_when_not_participates", reasons)


class TestBuilder1VisualPrompt(unittest.TestCase):
    def _plan(self, medium: bool = False) -> Builder1SeriesPlan:
        data = _series_plan_from_dict(_base_campaign(2), 2)
        if medium:
            data = Builder1SeriesPlan(
                **{**data.__dict__, "medium_participates": True, "medium_role": "Billboard bends"}
            )
        return data

    def test_shared_graphic_language_in_every_prompt(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(2), 2)
        prompts = [build_visual_prompt(plan, ad) for ad in plan.ads]
        for p in prompts:
            self.assertIn("#111111", p)
            self.assertIn("Clean studio photography", p)
            self.assertIn("12-col", p)

    def test_distinct_executions_per_ad(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(2), 2)
        p1 = build_visual_prompt(plan, plan.ads[0])
        p2 = build_visual_prompt(plan, plan.ads[1])
        self.assertIn("Scene description 1", p1)
        self.assertIn("Scene description 2", p2)
        self.assertNotEqual(p1, p2)

    def test_medium_prohibition_when_false(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(2), 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("billboards", prompt.lower())

    def test_medium_role_when_true(self) -> None:
        data = _base_campaign(2)
        data["mediumParticipates"] = True
        data["mediumRole"] = "The billboard itself folds into a bridge"
        plan = _series_plan_from_dict(data, 2)
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("folds into a bridge", prompt)


class TestBuilder1SeriesImages(unittest.TestCase):
    def test_preserves_ad_index_order(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(3), 3)
        calls: List[int] = []

        def caller(prompt: str, fmt: str) -> bytes:
            for ad in plan.ads:
                if ad.scene_description in prompt:
                    calls.append(ad.index)
                    return f"img-{ad.index}".encode()
            return b"x"

        result = generate_builder1_series_images(plan, caller)
        self.assertEqual([r.index for r in result.images], [1, 2, 3])

    def test_one_failed_image_fails_campaign(self) -> None:
        plan = _series_plan_from_dict(_base_campaign(2), 2)

        def caller(prompt: str, fmt: str) -> bytes:
            if "Scene description 2" in prompt:
                raise RuntimeError("transient")
            return b"ok"

        with self.assertRaises(RuntimeError):
            generate_builder1_series_images(plan, caller)


class TestBuilder1MemoryDisconnect(unittest.TestCase):
    def test_planner_does_not_import_ace_usage_memory(self) -> None:
        import engine.builder1_planner as planner_mod

        source = planner_mod.__file__
        with open(source, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("ace_usage_memory", text)


class TestBuilder1ApiResponse(unittest.TestCase):
    def test_completed_result_shape(self) -> None:
        from engine.builder1_plan_spec import ad_plan_to_api_dict, campaign_identity_to_dict

        plan = _series_plan_from_dict(_base_campaign(3), 3)
        result = {
            "ok": True,
            "campaign": campaign_identity_to_dict(plan),
            "ads": [
                ad_plan_to_api_dict(ad, visual_prompt="p", image_base64="abc")
                for ad in plan.ads
            ],
        }
        self.assertIn("campaign", result)
        self.assertIn("ads", result)
        self.assertEqual(len(result["ads"]), 3)
        self.assertNotIn("imageBase64", result)
        self.assertNotIn("objectA", result["campaign"])

    def test_float_ad_count_rejected(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count(2.5)

    def test_bool_ad_count_rejected(self) -> None:
        with self.assertRaises(Builder1InputError):
            normalize_ad_count(True)

    def test_default_ad_count_is_2(self) -> None:
        self.assertEqual(normalize_ad_count(None), 2)


class TestBuilder1Zip(unittest.TestCase):
    def test_zip_creates_expected_files_in_order(self) -> None:
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
            names = zf.namelist()
            self.assertEqual(names, ["campaign.txt", "ad-01.jpg", "ad-01.txt", "ad-02.jpg", "ad-02.txt"])
            self.assertIn("Slogan: Stay Sharp", zf.read("campaign.txt").decode())


if __name__ == "__main__":
    unittest.main()
