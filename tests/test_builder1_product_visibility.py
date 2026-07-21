"""
Builder1 product visibility policy tests.

Run: python -m unittest tests.test_builder1_product_visibility -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List

from engine.builder1_final_stages import parse_brand_physical_output, parse_series_ads_output
from engine.builder1_image_compliance import (
    IMAGE_COMPLIANCE_VIOLATION_CODES,
    parse_image_compliance_response,
)
from engine.builder1_image_generator import VISIBILITY_VIOLATION_CODES, generate_builder1_ad_image
from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesGenerator
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    ProductVisibilitySource,
    build_product_visibility_image_block,
    derive_product_visibility_policy,
    enforce_series_ad_visibility_fields,
    explicit_product_visibility_requested,
)
from engine.builder1_visual_prompt import build_visual_prompt
from tests.builder1_test_helpers import marketing_text_words, pass_compliance_reviewer
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _brand_physical, _series_ads


BRIEF = "Reinforced shell product for daily carry"


class TestVisibilityPolicyDerivation(unittest.TestCase):
    def test_default_visibility_policy_is_forbidden(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.FORBIDDEN)
        self.assertEqual(decision.source, ProductVisibilitySource.DEFAULT)

    def test_product_description_alone_does_not_allow_visibility(self) -> None:
        self.assertFalse(
            explicit_product_visibility_requested(
                product_name="",
                product_description=BRIEF,
            )
        )

    def test_product_name_alone_does_not_allow_visibility(self) -> None:
        self.assertFalse(
            explicit_product_visibility_requested(
                product_name="CarryShell",
                product_description=BRIEF,
            )
        )

    def test_explicit_user_instruction_creates_secondary_exception(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=f"{BRIEF}. Please show the product in the ad.",
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION)
        self.assertEqual(decision.source, ProductVisibilitySource.EXPLICIT_USER_REQUEST)


class TestBrandPhysicalInvariants(unittest.TestCase):
    def test_physical_generator_cannot_be_product(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_generator_is_product", str(ctx.exception))

    def test_physical_generator_cannot_be_packaging(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsPackaging"] = True
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_generator_is_packaging", str(ctx.exception))

    def test_physical_generator_must_work_without_product_visible(self) -> None:
        payload = _brand_physical()
        payload["worksWithoutProductVisible"] = False
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_generator_requires_product_visible", str(ctx.exception))


class TestSeriesAdsVisibilityInjection(unittest.TestCase):
    def test_series_ads_produces_product_visible_false_by_default(self) -> None:
        result = parse_series_ads_output(
            _series_ads(2),
            expected_ad_count=2,
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        for ad in result.ads:
            self.assertFalse(ad["productVisible"])
            self.assertFalse(ad["packagingVisible"])
            self.assertFalse(ad["productIsMainVisual"])
            self.assertFalse(ad["productIsPhysicalGenerator"])

    def test_model_cannot_independently_enable_visibility(self) -> None:
        payload = _series_ads(2)
        payload["ads"][0]["productVisibilityRequired"] = True
        payload["ads"][0]["productVisibilityReason"] = "Brand clearer with product shot"
        result = parse_series_ads_output(
            payload,
            expected_ad_count=2,
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertFalse(result.ads[0]["productVisible"])


class TestImagePromptVisibility(unittest.TestCase):
    def _plan(self) -> "Builder1SeriesPlan":
        from engine.builder1_plan_spec import Builder1SeriesPlan

        return _parse(_base_campaign(2), 2)

    def test_image_prompt_excludes_product_and_packaging(self) -> None:
        prompt = build_visual_prompt(self._plan(), self._plan().ads[0])
        self.assertIn("ADVERTISED PRODUCT: not depicted", prompt)
        self.assertIn("PACKAGING: not depicted", prompt)
        self.assertNotIn(self._plan().product_description, prompt)

    def test_image_prompt_positively_describes_transferred_object(self) -> None:
        plan = self._plan()
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(f"MAIN VISUAL: {plan.transferred_object}", prompt)
        self.assertIn(f"ACTION: {plan.transferred_object_action}", prompt)

    def test_product_name_plain_text_only(self) -> None:
        prompt = build_visual_prompt(self._plan(), self._plan().ads[0])
        self.assertIn("plain readable advertising typography", prompt.lower())
        self.assertIn("Do not print the brand name on any object", prompt)


class TestComplianceVisibility(unittest.TestCase):
    def test_new_visibility_violation_codes_registered(self) -> None:
        for code in (
            "product_visible_without_explicit_request",
            "packaging_visible_without_explicit_request",
            "product_used_as_physical_generator",
            "product_used_as_main_visual",
        ):
            self.assertIn(code, IMAGE_COMPLIANCE_VIOLATION_CODES)
            self.assertIn(code, VISIBILITY_VIOLATION_CODES)

    def test_compliance_rejects_unauthorized_product_shot(self) -> None:
        result = parse_image_compliance_response(
            {
                "pass": False,
                "violations": ["product_visible_without_explicit_request"],
                "confidence": "high",
            }
        )
        self.assertFalse(result.passed)

    def test_no_logo_compliance_remains_active(self) -> None:
        result = parse_image_compliance_response(
            {
                "pass": False,
                "violations": ["invented_product_logo"],
                "confidence": "high",
            }
        )
        self.assertIn("invented_product_logo", result.violations)

    def test_product_violation_triggers_same_ad_regeneration_only(self) -> None:
        calls = {"gen": 0, "review": 0}
        plan = TestImagePromptVisibility()._plan()
        plan.product_description = "Reinforced shell bottle for daily carry"

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            calls["review"] += 1
            from engine.builder1_compliance_product_grounding import ComplianceProductMatch
            from engine.builder1_image_compliance import ImageComplianceResult, finalize_compliance_result
            from engine.builder1_compliance_adjudication import ComplianceEvidenceItem

            if calls["review"] == 1:
                return finalize_compliance_result(
                    reviewer_pass=False,
                    candidate_violations=["product_visible_without_explicit_request"],
                    evidence_items=[
                        ComplianceEvidenceItem(
                            code="product_visible_without_explicit_request",
                            confidence="high",
                            symbol_description="Visible reinforced shell bottle matching product description",
                        )
                    ],
                    overall_confidence="high",
                    series_plan=plan,
                    product_match=ComplianceProductMatch(
                        advertised_product_present=True,
                        product_match_basis="explicit_product_shape",
                        matched_visual_element="reinforced shell bottle",
                        relationship_to_advertised_product="actual_product",
                        product_match_explanation="Bottle matches advertised tangible product description.",
                    ),
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 2)
        self.assertEqual(calls["review"], 2)


class TestExplicitExceptionBlock(unittest.TestCase):
    def test_secondary_exception_block_allows_only_secondary_product(self) -> None:
        block = build_product_visibility_image_block(
            policy=ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
            transferred_object="Rubber ball family",
            transferred_object_action="Bounces after a drop",
            product_name="CarryShell",
        )
        self.assertIn("secondary contextual element", block.lower())
        self.assertIn("transferred physical generator remains the main visual", block.lower())


if __name__ == "__main__":
    unittest.main()
