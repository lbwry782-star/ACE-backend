"""
Builder1 strategy judge tests — marketing copy length rules.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_marketing_copy import MARKETING_TEXT_WORD_COUNT, count_marketing_words
from engine.builder1_planner import plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
)
from engine.builder1_plan_spec import BRAND_SLOGAN_MAX_WORDS, HEADLINE_MAX_WORDS
from engine.builder1_strategy_judge import (
    BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT,
    deterministic_judge_checks,
    finalize_judge_result,
    judge_builder1_strategy,
    normalize_judge_rejection_codes,
)
from tests.builder1_test_helpers import marketing_text_hebrew, marketing_text_words
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _early_stage_responses,
    _graphic,
    _series_ads,
)


def _plan_dict(*, ad_count: int = 4, word_count: int = 50, headline: str | None = None) -> Dict[str, Any]:
    ads = []
    for i in range(1, ad_count + 1):
        ads.append(
            {
                "index": i,
                "headline": headline if i == 2 else None,
                "marketingText": marketing_text_words(word_count, prefix=f"a{i}"),
            }
        )
    return {
        "productNameResolved": "TestBrand",
        "brandSlogan": "Built To Last",
        "detectedLanguage": "en",
        "ads": ads,
    }


class TestStrategyJudgeMarketingCopy(unittest.TestCase):
    def test_hebrew_fifty_words_passes_deterministic(self) -> None:
        plan = _plan_dict(ad_count=1)
        plan["detectedLanguage"] = "he"
        plan["ads"][0]["marketingText"] = marketing_text_hebrew(50)
        self.assertEqual(deterministic_judge_checks(plan), [])

    def test_english_fifty_words_passes_deterministic(self) -> None:
        self.assertEqual(deterministic_judge_checks(_plan_dict(ad_count=2)), [])

    def test_forty_nine_words_fail_deterministic(self) -> None:
        reasons = deterministic_judge_checks(_plan_dict(ad_count=1, word_count=49))
        self.assertIn("marketing_copy_wrong_word_count", reasons)

    def test_fifty_one_words_fail_deterministic(self) -> None:
        reasons = deterministic_judge_checks(_plan_dict(ad_count=1, word_count=51))
        self.assertIn("marketing_copy_wrong_word_count", reasons)

    def test_judge_accepts_fifty_words_despite_stale_too_long_code(self) -> None:
        result = finalize_judge_result(
            model_pass=False,
            model_codes=["marketing_copy_too_long"],
            plan_dict=_plan_dict(ad_count=4),
            unsupported=False,
            raw={"pass": False, "rejectionReasonCodes": ["marketing_copy_too_long"]},
        )
        self.assertTrue(result.passed)
        self.assertNotIn("marketing_copy_too_long", result.rejection_reason_codes)

    def test_judge_does_not_emit_too_long_for_valid_fifty_words(self) -> None:
        codes = normalize_judge_rejection_codes(["marketing_copy_too_long"], _plan_dict(ad_count=4))
        self.assertEqual(codes, [])

    def test_headline_brevity_still_active(self) -> None:
        long_headline = " ".join(f"h{i}" for i in range(HEADLINE_MAX_WORDS + 2))
        plan = _plan_dict(ad_count=1)
        plan["ads"][0]["headline"] = long_headline
        reasons = deterministic_judge_checks(plan)
        self.assertIn("headline_too_long", reasons)

    def test_slogan_brevity_still_active(self) -> None:
        plan = _plan_dict(ad_count=1)
        plan["brandSlogan"] = " ".join(f"s{i}" for i in range(BRAND_SLOGAN_MAX_WORDS + 2))
        reasons = deterministic_judge_checks(plan)
        self.assertIn("brand_slogan_too_long", reasons)

    def test_image_brevity_rule_not_in_marketing_text_requirement(self) -> None:
        self.assertIn("exactly 50 words", BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT.lower())
        self.assertNotIn("marketing copy is too long for in-image rendering", BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT.lower())

    def test_four_ads_with_fifty_words_pass_judge(self) -> None:
        def model_caller(_s: str, _u: str) -> object:
            return {"pass": True, "rejectionReasonCodes": []}

        result = judge_builder1_strategy(
            product_description="Reinforced shell product",
            plan_dict=_plan_dict(ad_count=4),
            model_caller=model_caller,
        )
        self.assertTrue(result.passed)

    def test_unsupported_claim_still_fails(self) -> None:
        result = finalize_judge_result(
            model_pass=False,
            model_codes=["marketing_copy_unsupported_claim"],
            plan_dict=_plan_dict(ad_count=2),
            unsupported=True,
            raw={},
        )
        self.assertFalse(result.passed)
        self.assertIn("marketing_copy_unsupported_claim", result.rejection_reason_codes)


class TestPlannerJudgeIntegration(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_valid_fifty_words_do_not_rerun_series_ads_on_stale_judge_code(self) -> None:
        series_calls = {"n": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_SERIES_ADS_SYSTEM:
                series_calls["n"] += 1
                return _series_ads(4)
            if "strategy auditor" in system.lower() or system == BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT:
                return {"pass": False, "rejectionReasonCodes": ["marketing_copy_too_long"]}
            responses = _early_stage_responses(4)
            responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
            responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=4,
        )
        self.assertEqual(series_calls["n"], 1)
        for ad in plan.ads:
            self.assertEqual(count_marketing_words(ad.marketing_text), MARKETING_TEXT_WORD_COUNT)

    def test_valid_campaign_reaches_image_generation(self) -> None:
        def model_caller(system: str, _user: str, stage: str | None = None) -> object:
            if system == BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT:
                return {"pass": False, "rejectionReasonCodes": ["marketing_copy_too_long"]}
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


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_strategy_judge_imports(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_strategy_judge", text)


if __name__ == "__main__":
    unittest.main()
