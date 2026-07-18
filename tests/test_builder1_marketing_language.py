"""
Builder1 marketing-text language validation and repair tests.

Run: python -m unittest tests.test_builder1_marketing_language -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_marketing_copy import (
    MARKETING_TEXT_WORD_COUNT,
    count_marketing_words,
    validate_marketing_text_language,
    validate_marketing_text_50_words,
)
from engine.builder1_marketing_text_repair import (
    MARKETING_TEXT_REPAIR_SYSTEM,
    ensure_series_ads_marketing_text,
)
from engine.builder1_creative_methodology import methodology_repair_stage
from engine.builder1_planner import plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
)
from engine.builder1_strategy_judge import (
    BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT,
    deterministic_judge_checks,
    judge_builder1_strategy,
)
from tests.builder1_test_helpers import (
    marketing_text_english_with_hebrew_brand,
    marketing_text_hebrew,
    marketing_text_hebrew_with_brand,
    marketing_text_words,
    pass_compliance_reviewer,
)
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _early_stage_responses,
    _graphic,
    _series_ads,
)


def _sample_ads(*, language_text: str, count: int = 2) -> List[Dict[str, Any]]:
    ads = []
    for i in range(1, count + 1):
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
                "marketingText": language_text if i == 1 else marketing_text_hebrew(50),
            }
        )
    return ads


class TestMarketingLanguageValidation(unittest.TestCase):
    def test_hebrew_paragraph_passes(self) -> None:
        result = validate_marketing_text_language(marketing_text_hebrew(50), "he")
        self.assertTrue(result["valid"])
        self.assertEqual(result["targetLanguage"], "he")
        self.assertEqual(result["dominantLanguage"], "he")

    def test_hebrew_with_english_brand_name_passes(self) -> None:
        result = validate_marketing_text_language(marketing_text_hebrew_with_brand(50), "he")
        self.assertTrue(result["valid"])

    def test_primarily_english_fails_for_hebrew_target(self) -> None:
        result = validate_marketing_text_language(marketing_text_words(50), "he")
        self.assertFalse(result["valid"])
        self.assertEqual(result["dominantLanguage"], "en")

    def test_english_paragraph_passes(self) -> None:
        result = validate_marketing_text_language(marketing_text_words(50), "en")
        self.assertTrue(result["valid"])
        self.assertEqual(result["dominantLanguage"], "en")

    def test_english_with_hebrew_brand_name_passes(self) -> None:
        result = validate_marketing_text_language(
            marketing_text_english_with_hebrew_brand(50),
            "en",
        )
        self.assertTrue(result["valid"])

    def test_primarily_hebrew_fails_for_english_target(self) -> None:
        result = validate_marketing_text_language(marketing_text_hebrew(50), "en")
        self.assertFalse(result["valid"])


class TestMarketingLanguageRepair(unittest.TestCase):
    def test_wrong_language_triggers_focused_repair(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str) -> object:
            calls.append(system)
            return {"repairs": [{"index": 1, "marketingText": marketing_text_hebrew(50)}]}

        result = ensure_series_ads_marketing_text(
            _sample_ads(language_text=marketing_text_words(50)),
            detected_language="he",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(MARKETING_TEXT_REPAIR_SYSTEM, calls[0])
        validate_marketing_text_50_words(result[0]["marketingText"])
        self.assertTrue(validate_marketing_text_language(result[0]["marketingText"], "he")["valid"])

    def test_language_repair_changes_only_marketing_text(self) -> None:
        ads = _sample_ads(language_text=marketing_text_words(50))
        original_scene = ads[0]["sceneDescription"]
        original_headline = ads[1]["headline"]
        original_execution = ads[0]["conceptualExecution"]

        def model_caller(_s: str, _u: str) -> object:
            return {"repairs": [{"index": 1, "marketingText": marketing_text_hebrew(50)}]}

        result = ensure_series_ads_marketing_text(
            ads,
            detected_language="he",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(result[0]["sceneDescription"], original_scene)
        self.assertEqual(result[1]["headline"], original_headline)
        self.assertEqual(result[0]["conceptualExecution"], original_execution)

    def test_repaired_hebrew_has_fifty_words(self) -> None:
        def model_caller(_s: str, _u: str) -> object:
            return {"repairs": [{"index": 1, "marketingText": marketing_text_hebrew(50)}]}

        result = ensure_series_ads_marketing_text(
            _sample_ads(language_text=marketing_text_words(50)),
            detected_language="he",
            relative_advantage="Distinct advantage",
            product_name="TestBrand",
            brand_slogan="Built To Last",
            model_caller=model_caller,
        )
        self.assertEqual(count_marketing_words(result[0]["marketingText"]), 50)

    def test_repaired_english_has_fifty_words(self) -> None:
        ads = _sample_ads(language_text=marketing_text_hebrew(50))
        ads[1]["marketingText"] = marketing_text_words(50)

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
        self.assertEqual(count_marketing_words(result[0]["marketingText"]), 50)


class TestPlannerLanguageIntegration(unittest.TestCase):
    HEBREW_BRIEF = "מגן מחוזק למוצר יומיומי לנשיאה"

    def test_marketing_copy_wrong_language_is_repaired_in_series_flow(self) -> None:
        ads = _sample_ads(language_text=marketing_text_hebrew(50))
        ads[1]["marketingText"] = marketing_text_words(50)

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
        self.assertEqual(count_marketing_words(result[0]["marketingText"]), 50)

    def test_valid_language_does_not_rerun_series_ads(self) -> None:
        series_calls = {"n": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_SERIES_ADS_SYSTEM:
                series_calls["n"] += 1
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(50)
                payload["ads"][1]["marketingText"] = marketing_text_words(50, prefix="b")
                return payload
            if system == MARKETING_TEXT_REPAIR_SYSTEM:
                return {
                    "repairs": [
                        {"index": 1, "marketingText": marketing_text_hebrew(50)},
                        {"index": 2, "marketingText": " ".join(f"ב{i}" for i in range(1, 51))},
                    ]
                }
            responses = _early_stage_responses(2)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            return responses.get(system, {})

        with patch(
            "engine.builder1_planner.detect_brief_language",
            return_value="he",
        ):
            plan = plan_builder1(
                product_name="",
                product_description=self.HEBREW_BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(series_calls["n"], 1)
        for ad in plan.ads:
            self.assertEqual(count_marketing_words(ad.marketing_text), MARKETING_TEXT_WORD_COUNT)
            self.assertTrue(validate_marketing_text_language(ad.marketing_text, "he")["valid"])

    def test_successful_language_repair_completes_planning_without_judge(self) -> None:
        repair_calls = {"n": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == MARKETING_TEXT_REPAIR_SYSTEM:
                repair_calls["n"] += 1
                return {
                    "repairs": [
                        {"index": 1, "marketingText": marketing_text_hebrew(50)},
                        {"index": 2, "marketingText": " ".join(f"ב{i}" for i in range(1, 51))},
                    ]
                }
            if system == STAGE_SERIES_ADS_SYSTEM:
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(50)
                payload["ads"][1]["marketingText"] = marketing_text_words(50, prefix="b")
                return payload
            responses = _early_stage_responses(2)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            return responses.get(system, {})

        with patch(
            "engine.builder1_planner.detect_brief_language",
            return_value="he",
        ):
            plan = plan_builder1(
                product_name="",
                product_description=self.HEBREW_BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertGreaterEqual(repair_calls["n"], 1)
        self.assertEqual(plan.ad_count, 2)

    def test_successful_repair_reaches_image_generation(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == MARKETING_TEXT_REPAIR_SYSTEM:
                return {
                    "repairs": [
                        {"index": 1, "marketingText": marketing_text_hebrew(50)},
                        {"index": 2, "marketingText": " ".join(f"ב{i}" for i in range(1, 51))},
                    ]
                }
            if system == STAGE_SERIES_ADS_SYSTEM:
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(50)
                payload["ads"][1]["marketingText"] = marketing_text_words(50, prefix="b")
                return payload
            responses = _early_stage_responses(2)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            return responses.get(system, {})

        with patch(
            "engine.builder1_planner.detect_brief_language",
            return_value="he",
        ):
            plan = plan_builder1(
                product_name="",
                product_description=self.HEBREW_BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        calls: List[int] = []

        def image_caller(_p: str, _f: str) -> bytes:
            calls.append(1)
            return b"jpeg"

        generate_builder1_ad_image(plan, 1, image_caller, compliance_reviewer=pass_compliance_reviewer)
        self.assertEqual(len(calls), 1)

    def test_language_repair_preserves_generators(self) -> None:
        original_brand = _brand_physical()
        original_graphic = _graphic()
        captured: Dict[str, Any] = {}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_BRAND_PHYSICAL_SYSTEM:
                captured["brand"] = copy.deepcopy(_brand_physical())
                return _brand_physical()
            if system == STAGE_GRAPHIC_SYSTEM_SYSTEM:
                captured["graphic"] = copy.deepcopy(_graphic())
                return _graphic()
            if system == BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT:
                return {"pass": False, "rejectionReasonCodes": ["marketing_copy_wrong_language"]}
            if system == MARKETING_TEXT_REPAIR_SYSTEM:
                return {
                    "repairs": [
                        {"index": 1, "marketingText": marketing_text_hebrew(50)},
                        {"index": 2, "marketingText": " ".join(f"ב{i}" for i in range(1, 51))},
                    ]
                }
            if system == STAGE_SERIES_ADS_SYSTEM:
                payload = _series_ads(2)
                payload["ads"][0]["marketingText"] = marketing_text_words(50)
                payload["ads"][1]["marketingText"] = marketing_text_words(50, prefix="b")
                return payload
            return copy.deepcopy(_early_stage_responses(2).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planner.detect_brief_language",
            return_value="he",
        ):
            plan = plan_builder1(
                product_name="",
                product_description=self.HEBREW_BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(plan.brand_slogan, "Built To Last")
        self.assertEqual(plan.physical_generator, original_brand["physicalGenerator"])
        self.assertEqual(plan.graphic_generator.palette.dominant, original_graphic["palette"]["dominant"])
        self.assertEqual(plan.series_generator.type, "situations")


class TestFourHebrewAds(unittest.TestCase):
    def test_four_hebrew_ads_pass_deterministic_checks(self) -> None:
        plan = {
            "detectedLanguage": "he",
            "brandSlogan": "בנוי לעמוד",
            "ads": [
                {
                    "index": i,
                    "marketingText": " ".join(f"ע{i}{j}" for j in range(1, 51)),
                }
                for i in range(1, 5)
            ],
        }
        self.assertEqual(deterministic_judge_checks(plan), [])

        def model_caller(_s: str, _u: str) -> object:
            return {"pass": True, "rejectionReasonCodes": []}

        result = judge_builder1_strategy(
            product_description="מוצר יומיומי",
            plan_dict=plan,
            model_caller=model_caller,
        )
        self.assertTrue(result.passed)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_marketing_language_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("validate_marketing_text_language", text)


if __name__ == "__main__":
    unittest.main()
