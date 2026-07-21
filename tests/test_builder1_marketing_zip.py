"""
Builder1 marketing copy and single-ad ZIP tests.

Run: python -m unittest tests.test_builder1_marketing_zip -v
"""
from __future__ import annotations

import base64
import io
import unittest
import zipfile
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_marketing_copy import (
    MARKETING_TEXT_WORD_COUNT,
    count_marketing_words,
    normalize_marketing_text,
    validate_marketing_text_50_words,
)
from engine.builder1_marketing_text_repair import (
    MARKETING_TEXT_REPAIR_SYSTEM,
    ensure_series_ads_marketing_text,
)
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
)
from engine.builder1_zip import build_builder1_single_ad_zip_bytes, build_builder1_zip_bytes
from tests.builder1_test_helpers import (
    marketing_text_hebrew,
    marketing_text_with_punctuation,
    marketing_text_words,
)
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _early_stage_responses,
    _graphic,
    _series_ads,
)


class TestMarketingCopyValidation(unittest.TestCase):
    def test_exactly_fifty_words_required(self) -> None:
        validate_marketing_text_50_words(marketing_text_words(50))
        with self.assertRaises(ValueError):
            validate_marketing_text_50_words(marketing_text_words(49))

    def test_hebrew_word_count(self) -> None:
        self.assertEqual(count_marketing_words(marketing_text_hebrew(50)), 50)

    def test_english_word_count(self) -> None:
        self.assertEqual(count_marketing_words(marketing_text_words(50)), 50)

    def test_repeated_whitespace_normalized(self) -> None:
        text = "  word1   word2  " + marketing_text_words(48, prefix="w")
        self.assertEqual(count_marketing_words(text), 50)

    def test_punctuation_does_not_add_words(self) -> None:
        self.assertEqual(count_marketing_words(marketing_text_with_punctuation()), 50)


class TestMarketingTextRepair(unittest.TestCase):
    def _sample_ads(self, *, bad_index: int = 1, bad_count: int = 49) -> List[Dict[str, Any]]:
        ads = []
        for i in range(1, 3):
            ads.append(
                {
                    "index": i,
                    "variationLabel": f"var-{i}",
                    "newContribution": f"Contribution {i}",
                    "conceptualExecution": f"Action {i}",
                    "conceptualActionProof": f"Proof {i}",
                    "physicalExecution": f"Physical {i}",
                    "visualExecution": f"Visual {i}",
                    "sceneDescription": f"Scene {i}",
                    "headline": None if i == 1 else f"Line {i}",
                    "headlineNeededReason": "Needed",
                    "marketingText": marketing_text_words(bad_count if i == bad_index else 50),
                }
            )
        return ads

    def test_forty_nine_words_triggers_repair(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str) -> object:
            calls.append(system)
            return {"repairs": [{"index": 1, "marketingText": marketing_text_words(50)}]}

        result = ensure_series_ads_marketing_text(
            self._sample_ads(bad_count=49),
            detected_language="en",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(MARKETING_TEXT_REPAIR_SYSTEM, calls[0])
        self.assertEqual(count_marketing_words(result[0]["marketingText"]), 50)

    def test_fifty_one_words_triggers_repair(self) -> None:
        def model_caller(_s: str, _u: str) -> object:
            return {"repairs": [{"index": 1, "marketingText": marketing_text_words(50)}]}

        result = ensure_series_ads_marketing_text(
            self._sample_ads(bad_count=51),
            detected_language="en",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(count_marketing_words(result[0]["marketingText"]), 50)

    def test_repair_changes_only_marketing_text(self) -> None:
        ads = self._sample_ads(bad_count=49)
        original_scene = ads[0]["sceneDescription"]
        original_headline = ads[1]["headline"]

        def model_caller(_s: str, _u: str) -> object:
            return {"repairs": [{"index": 1, "marketingText": marketing_text_words(50)}]}

        result = ensure_series_ads_marketing_text(
            ads,
            detected_language="en",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(result[0]["sceneDescription"], original_scene)
        self.assertEqual(result[1]["headline"], original_headline)

    def test_all_four_ads_can_have_valid_text(self) -> None:
        ads = []
        for i in range(1, 5):
            ads.append(
                {
                    "index": i,
                    "variationLabel": f"var-{i}",
                    "newContribution": f"Contribution {i}",
                    "conceptualExecution": f"Action {i}",
                    "conceptualActionProof": f"Proof {i}",
                    "physicalExecution": f"Physical {i}",
                    "visualExecution": f"Visual {i}",
                    "sceneDescription": f"Scene {i}",
                    "headline": None,
                    "headlineNeededReason": "Needed",
                    "marketingText": marketing_text_words(49 if i == 2 else 50, prefix=f"w{i}"),
                }
            )

        def model_caller(_s: str, _u: str) -> object:
            return {"repairs": [{"index": 2, "marketingText": marketing_text_words(50, prefix="w2")}]}

        result = ensure_series_ads_marketing_text(
            ads,
            detected_language="en",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        for ad in result:
            self.assertEqual(count_marketing_words(ad["marketingText"]), 50)


class TestPlannerMarketingIntegration(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_repair_does_not_rerun_strategy_or_conceptual(self) -> None:
        counters = {"strategy": 0, "conceptual": 0, "repair": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM:
                counters["strategy"] += 1
            if system == STAGE_CONCEPTUAL_STAGE_SYSTEM:
                counters["conceptual"] += 1
            if system == MARKETING_TEXT_REPAIR_SYSTEM:
                counters["repair"] += 1
                return {"repairs": [{"index": 1, "marketingText": marketing_text_words(50, "r")},
                                    {"index": 2, "marketingText": marketing_text_words(50, "s")}]}
            if system == STAGE_SERIES_ADS_SYSTEM and "Repair ONLY" not in user:
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(49)
                payload["ads"][1]["marketingText"] = marketing_text_words(51)
                return payload
            responses = _early_stage_responses(2)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            if system == STAGE_SERIES_ADS_SYSTEM:
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(49)
                payload["ads"][1]["marketingText"] = marketing_text_words(51)
                return payload
            return responses.get(system, {})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(counters["strategy"], 1)
        self.assertEqual(counters["conceptual"], 1)
        self.assertGreaterEqual(counters["repair"], 1)
        for ad in plan.ads:
            self.assertEqual(count_marketing_words(ad.marketing_text), 50)


class TestSingleAdZip(unittest.TestCase):
    IMG = base64.b64encode(b"fakejpeg").decode()
    TEXT50 = marketing_text_words(50)

    def _payload(self, *, index: int = 1, marketing_text: str | None = None) -> Dict[str, Any]:
        return {
            "scope": "single_ad",
            "campaignId": "c1",
            "campaign": {
                "productNameResolved": "TestBrand",
                "brandSlogan": "Built To Last",
            },
            "ad": {
                "index": index,
                "headline": None,
                "marketingText": marketing_text or self.TEXT50,
                "imageBase64": self.IMG,
            },
        }

    def test_single_ad_zip_before_campaign_completion(self) -> None:
        zip_bytes, name = build_builder1_single_ad_zip_bytes(self._payload(index=1))
        self.assertEqual(name, "ad-01.zip")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            self.assertEqual(sorted(zf.namelist()), ["ad-01.jpg", "ad-01.txt"])

    def test_zip_contains_image_and_text(self) -> None:
        zip_bytes, _ = build_builder1_single_ad_zip_bytes(self._payload(index=2))
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            self.assertIn("ad-02.jpg", zf.namelist())
            self.assertIn("ad-02.txt", zf.namelist())

    def test_zip_text_contains_fifty_word_paragraph(self) -> None:
        zip_bytes, _ = build_builder1_single_ad_zip_bytes(self._payload())
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt = zf.read("ad-01.txt").decode("utf-8")
        self.assertIn("Product: TestBrand", txt)
        self.assertIn("Slogan: Built To Last", txt)
        self.assertIn(self.TEXT50, txt)

    def test_zip_text_excludes_internal_methodology(self) -> None:
        zip_bytes, _ = build_builder1_single_ad_zip_bytes(self._payload())
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt = zf.read("ad-01.txt").decode("utf-8").lower()
        for forbidden in ("strategic problem", "relative advantage", "conceptual generator", "visual prompt"):
            self.assertNotIn(forbidden, txt)

    def test_invalid_base64_rejected(self) -> None:
        payload = self._payload(index=2)
        payload["ad"]["imageBase64"] = "%%%not-base64%%%"
        with self.assertRaises(ValueError) as ctx:
            build_builder1_single_ad_zip_bytes(payload)
        self.assertEqual(str(ctx.exception), "invalid_image_base64")

    def test_full_campaign_zip_backward_compatible(self) -> None:
        zbytes = build_builder1_zip_bytes(
            {
                "productName": "BrandX",
                "brandSlogan": "Stay Sharp",
                "ads": [
                    {"index": 1, "imageBase64": self.IMG, "headline": None, "marketingText": self.TEXT50},
                    {"index": 2, "imageBase64": self.IMG, "headline": "Hi", "marketingText": marketing_text_words(50, "two")},
                ],
            }
        )
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            self.assertIn("campaign.txt", zf.namelist())


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_marketing_copy_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_marketing_copy", text)


if __name__ == "__main__":
    unittest.main()
