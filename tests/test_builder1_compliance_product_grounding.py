"""
Builder1 image-compliance advertised-product grounding tests.

Run: python -m unittest tests.test_builder1_compliance_product_grounding -v
"""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from engine.builder1_compliance_adjudication import ComplianceEvidenceItem, adjudicate_compliance_review
from engine.builder1_compliance_product_grounding import (
    AdvertisedProductType,
    ComplianceProductMatch,
    classify_advertised_product_type,
    evaluate_product_visible_hard_support,
    is_human_visual_representation,
    is_inanimate_named_person_match,
    reference_image_actually_supplied,
)
from engine.builder1_image_compliance import (
    ImageComplianceError,
    ImageComplianceResult,
    finalize_compliance_result,
    parse_image_compliance_response,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_generator import MAX_INTERNAL_IMAGE_ATTEMPTS, generate_builder1_ad_image
from engine.builder1_product_visibility import ProductVisibilityPolicy
from tests.test_builder1_series import _base_campaign, _parse


def _default_product_match(**overrides: Any) -> dict[str, Any]:
    payload = {
        "advertisedProductPresent": False,
        "productMatchBasis": "none",
        "matchedVisualElement": "",
        "relationshipToAdvertisedProduct": "none",
        "productMatchExplanation": "",
    }
    payload.update(overrides)
    return payload


def _named_person_domino_plan(ad_count: int = 2):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.product_name = "אמיר גוטליב"
    plan.product_name_resolved = "אמיר גוטליב"
    plan.product_description = ""
    plan.product_visibility_policy = "FORBIDDEN"
    plan.physical_generator = "Domino tiles"
    plan.transferred_object = "Domino tiles"
    plan.transferred_object_action = "Tiles fall in a controlled chain reaction"
    plan.planning_internals = dict(plan.planning_internals or {})
    plan.planning_internals["productVisibilityPolicy"] = "FORBIDDEN"
    return plan


def _tangible_product_plan(ad_count: int = 2):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.product_name_resolved = "CarryShell"
    plan.product_description = "Reinforced shell bottle for daily carry"
    plan.product_visibility_policy = "FORBIDDEN"
    plan.physical_generator = "Rubber ball family"
    plan.transferred_object = "Rubber ball family"
    return plan


class TestAdvertisedProductClassification(unittest.TestCase):
    def test_hebrew_person_name_classified_as_named_person(self) -> None:
        self.assertEqual(
            classify_advertised_product_type(product_name="אמיר גוטליב"),
            AdvertisedProductType.NAMED_PERSON,
        )

    def test_tangible_product_description(self) -> None:
        self.assertEqual(
            classify_advertised_product_type(
                product_name="CarryShell",
                product_description="Reinforced shell bottle for daily carry",
            ),
            AdvertisedProductType.PACKAGED_PRODUCT,
        )


def _named_person_orit_plan(ad_count: int = 2):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.product_name = "אורי לב"
    plan.product_name_resolved = "אורי לב"
    plan.product_description = ""
    plan.product_visibility_policy = "FORBIDDEN"
    plan.physical_generator = "Access control motif"
    plan.transferred_object = "Access control motif"
    plan.planning_internals = dict(plan.planning_internals or {})
    plan.planning_internals["productVisibilityPolicy"] = "FORBIDDEN"
    plan.planning_internals["advertisedProductType"] = "named_person"
    return plan


class TestNamedPersonHumanPresenceGate(unittest.TestCase):
    def test_badge_not_product_visibility(self) -> None:
        plan = _named_person_orit_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Identification badge on lanyard",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="identification badge",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Badge identifies the named person.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)
        self.assertIn("possible_product_resemblance", result.advisories)

    def test_identification_card_not_product_visibility(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="identification card",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="ID card treated as person.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_tag_and_reader_not_product_visibility(self) -> None:
        self.assertTrue(is_inanimate_named_person_match("tag and reader"))
        plan = _named_person_orit_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Access tag and reader at entry",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="tag and reader",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Reader implies the named person.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)

    def test_sign_with_name_not_product_visibility(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="exact_product_text",
                matched_visual_element="sign containing אורי לב",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Name appears on sign.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
            evidence_description="Poster text with the advertised name",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_business_card_not_product_visibility(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="business card",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Card belongs to the person.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_associated_device_not_product_visibility(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="access control device",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Device associated with the person.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_actual_product_rejected_for_inanimate_match(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="reader",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Reader labeled actual product.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_explicit_prompt_basis_rejected_without_human(self) -> None:
        plan = _named_person_orit_plan()
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="identification badge",
                relationship_to_advertised_product="explicit_representation",
                product_match_explanation="Prompt mentioned the person.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "named_person_non_human_match")

    def test_supplied_reference_basis_rejected_without_actual_reference(self) -> None:
        plan = _named_person_orit_plan()
        self.assertFalse(reference_image_actually_supplied(plan))
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="supplied_reference_image",
                matched_visual_element="human portrait",
                relationship_to_advertised_product="explicit_representation",
                product_match_explanation="Model claimed supplied reference image.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "invalid_reference_image_basis")

    def test_human_portrait_can_still_be_true_match(self) -> None:
        plan = _named_person_orit_plan()
        plan.ads[0].physical_execution = "Portrait photograph of אורי לב centered in frame"
        self.assertTrue(
            is_human_visual_representation(
                matched_visual_element="portrait of אורי לב",
                product_match_explanation="Visible human portrait of advertised person.",
                advertised_product_name="אורי לב",
            )
        )
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Human portrait centered in frame",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="portrait of אורי לב",
                relationship_to_advertised_product="explicit_representation",
                product_match_explanation="Portrait explicitly requested in plan.",
            ),
        )
        self.assertIn("product_visible_without_explicit_request", result.hard_violations)

    def test_supplied_reference_with_actual_reference_and_human_match(self) -> None:
        plan = _named_person_orit_plan()
        plan.planning_internals["referenceImageSupplied"] = True
        self.assertTrue(reference_image_actually_supplied(plan))
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="supplied_reference_image",
                matched_visual_element="human portrait based on supplied reference",
                relationship_to_advertised_product="explicit_representation",
                product_match_explanation="Portrait matches supplied reference image.",
            ),
            advertised_type=AdvertisedProductType.NAMED_PERSON,
            series_plan=plan,
            confidence="high",
            evidence_description="Recognizable portrait of a human person",
        )
        self.assertTrue(supported)
        self.assertEqual(reason, "")

    def test_false_positive_suppression_prevents_regeneration(self) -> None:
        plan = _named_person_orit_plan()
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return finalize_compliance_result(
                reviewer_pass=False,
                candidate_violations=["product_visible_without_explicit_request"],
                evidence_items=[
                    ComplianceEvidenceItem(
                        code="product_visible_without_explicit_request",
                        confidence="high",
                        symbol_description="Identification badge",
                    )
                ],
                overall_confidence="high",
                series_plan=plan,
                product_match=ComplianceProductMatch(
                    advertised_product_present=True,
                    product_match_basis="explicit_prompt_identification",
                    matched_visual_element="identification badge",
                    relationship_to_advertised_product="actual_product",
                    product_match_explanation="Badge treated as person.",
                ),
            )

        result = generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 1)
        self.assertEqual(result.image_bytes, b"img")


class TestDominoFalsePositiveSuppression(unittest.TestCase):
    def test_named_person_domino_tiles_not_hard_violation(self) -> None:
        plan = _named_person_domino_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Domino tiles centered in composition",
                    location="center",
                    relationship_to_brand_text="none",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_product_shape",
                matched_visual_element="Domino tiles",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Central domino tiles treated as product.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)
        self.assertTrue(result.passed)

    def test_generic_person_without_portrait_request_not_violation(self) -> None:
        plan = _named_person_domino_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="medium",
                    symbol_description="Generic adult person in background",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="none",
                matched_visual_element="generic adult person",
                relationship_to_advertised_product="uncertain",
                product_match_explanation="Person resembles advertised name from appearance.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)

    def test_named_person_explicit_portrait_can_fail_when_grounded(self) -> None:
        plan = _named_person_domino_plan()
        plan.ads[0].physical_execution = "Portrait photograph of אמיר גוטליב centered in frame"
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Portrait of advertised person",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_prompt_identification",
                matched_visual_element="portrait of אמיר גוטליב",
                relationship_to_advertised_product="explicit_representation",
                product_match_explanation="Image prompt explicitly requested portrait of advertised person.",
            ),
        )
        self.assertIn("product_visible_without_explicit_request", result.hard_violations)


class TestCreativeObjectNotProduct(unittest.TestCase):
    def test_dominant_physical_generator_not_product(self) -> None:
        plan = _tangible_product_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="Large rubber ball family dominates frame",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=False,
                product_match_basis="none",
                matched_visual_element="Rubber ball family",
                relationship_to_advertised_product="creative_generator",
                product_match_explanation="Dominant object is approved physical generator.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)

    def test_centered_transferred_object_not_product(self) -> None:
        plan = _tangible_product_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=False,
                product_match_basis="none",
                matched_visual_element="Rubber ball family",
                relationship_to_advertised_product="transferred_object",
                product_match_explanation="Centered transferred object from plan.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)

    def test_thematic_prop_not_product(self) -> None:
        plan = _tangible_product_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=False,
                product_match_basis="none",
                matched_visual_element="protective shell motif sculpture",
                relationship_to_advertised_product="generic_prop",
                product_match_explanation="Thematic resemblance only.",
            ),
        )
        self.assertNotIn("product_visible_without_explicit_request", result.hard_violations)


class TestGroundedViolationsPreserved(unittest.TestCase):
    def test_tangible_product_visible_remains_hard(self) -> None:
        plan = _tangible_product_plan()
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    confidence="high",
                    symbol_description="CarryShell bottle shown as hero product",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_product_shape",
                matched_visual_element="CarryShell reinforced shell bottle",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Visible bottle matches advertised product description.",
            ),
        )
        self.assertIn("product_visible_without_explicit_request", result.hard_violations)

    def test_packaging_visible_remains_hard_for_tangible_product(self) -> None:
        plan = _tangible_product_plan()
        result = adjudicate_compliance_review(
            raw_violations=["packaging_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="packaging_visible_without_explicit_request",
                    confidence="high",
                )
            ],
            overall_confidence="high",
            series_plan=plan,
        )
        self.assertIn("packaging_visible_without_explicit_request", result.hard_violations)


class TestDeterministicGate(unittest.TestCase):
    def test_product_match_basis_none_suppressed(self) -> None:
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(product_match_basis="none"),
            advertised_type=AdvertisedProductType.TANGIBLE_PRODUCT,
            series_plan=_tangible_product_plan(),
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertEqual(reason, "ungrounded_product_match")

    def test_creative_generator_suppressed(self) -> None:
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_product_shape",
                matched_visual_element="Rubber ball family",
                relationship_to_advertised_product="creative_generator",
                product_match_explanation="Looks like product because it is central.",
            ),
            advertised_type=AdvertisedProductType.TANGIBLE_PRODUCT,
            series_plan=_tangible_product_plan(),
            confidence="high",
        )
        self.assertFalse(supported)
        self.assertIn(
            reason,
            {"relationship_creative_generator", "creative_generator_not_product"},
        )

    def test_actual_product_grounded_preserved(self) -> None:
        supported, reason = evaluate_product_visible_hard_support(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            product_match=ComplianceProductMatch(
                advertised_product_present=True,
                product_match_basis="explicit_product_shape",
                matched_visual_element="CarryShell bottle",
                relationship_to_advertised_product="actual_product",
                product_match_explanation="Matches advertised bottle product.",
            ),
            advertised_type=AdvertisedProductType.PACKAGED_PRODUCT,
            series_plan=_tangible_product_plan(),
            confidence="high",
        )
        self.assertTrue(supported)
        self.assertEqual(reason, "")


class TestRegenerationLoopBehavior(unittest.TestCase):
    def test_unsupported_finding_does_not_regenerate(self) -> None:
        plan = _named_person_domino_plan()
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return finalize_compliance_result(
                reviewer_pass=False,
                candidate_violations=["product_visible_without_explicit_request"],
                evidence_items=[
                    ComplianceEvidenceItem(
                        code="product_visible_without_explicit_request",
                        confidence="high",
                        symbol_description="Domino tiles in center",
                        relationship_to_brand_text="none",
                    )
                ],
                overall_confidence="high",
                series_plan=plan,
                product_match=ComplianceProductMatch(
                    advertised_product_present=True,
                    product_match_basis="none",
                    matched_visual_element="Domino tiles",
                    relationship_to_advertised_product="creative_generator",
                    product_match_explanation="Reviewer confused domino tiles with product.",
                ),
            )

        result = generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 1)
        self.assertEqual(result.index, 1)

    def test_grounded_violation_regenerates(self) -> None:
        plan = _tangible_product_plan()
        calls = {"gen": 0, "review": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            calls["review"] += 1
            if calls["review"] == 1:
                return finalize_compliance_result(
                    reviewer_pass=False,
                    candidate_violations=["product_visible_without_explicit_request"],
                    evidence_items=[
                        ComplianceEvidenceItem(
                            code="product_visible_without_explicit_request",
                            confidence="high",
                        )
                    ],
                    overall_confidence="high",
                    series_plan=plan,
                    product_match=ComplianceProductMatch(
                        advertised_product_present=True,
                        product_match_basis="explicit_product_shape",
                        matched_visual_element="CarryShell bottle",
                        relationship_to_advertised_product="actual_product",
                        product_match_explanation="Advertised bottle visible.",
                    ),
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 2)

    def test_false_positive_does_not_consume_three_attempts(self) -> None:
        plan = _named_person_domino_plan()
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            return finalize_compliance_result(
                reviewer_pass=False,
                candidate_violations=["product_visible_without_explicit_request"],
                evidence_items=[
                    ComplianceEvidenceItem(
                        code="product_visible_without_explicit_request",
                        confidence="high",
                        symbol_description="Domino tiles",
                    )
                ],
                overall_confidence="high",
                series_plan=plan,
                product_match=ComplianceProductMatch(
                    advertised_product_present=True,
                    product_match_basis="none",
                    matched_visual_element="Domino tiles",
                    relationship_to_advertised_product="creative_generator",
                    product_match_explanation="False positive",
                ),
            )

        result = generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertLess(calls["gen"], MAX_INTERNAL_IMAGE_ATTEMPTS)
        self.assertEqual(result.image_bytes, b"img")

    def test_ad_two_proceeds_after_ad_one_suppressed_false_positive(self) -> None:
        plan = _named_person_domino_plan(3)

        def pass_after_suppression(**_kwargs: Any):
            return finalize_compliance_result(
                reviewer_pass=False,
                candidate_violations=["product_visible_without_explicit_request"],
                evidence_items=[
                    ComplianceEvidenceItem(
                        code="product_visible_without_explicit_request",
                        confidence="high",
                        symbol_description="Domino tiles",
                    )
                ],
                overall_confidence="high",
                series_plan=plan,
                product_match=ComplianceProductMatch(
                    matched_visual_element="Domino tiles",
                    relationship_to_advertised_product="transferred_object",
                    product_match_explanation="Approved transferred object.",
                ),
            )

        ad1 = generate_builder1_ad_image(plan, 1, lambda _p, _f: b"a1", compliance_reviewer=pass_after_suppression)
        ad2 = generate_builder1_ad_image(plan, 2, lambda _p, _f: b"a2", compliance_reviewer=pass_after_suppression)
        self.assertEqual(ad1.image_bytes, b"a1")
        self.assertEqual(ad2.image_bytes, b"a2")


class TestNoExtraCalls(unittest.TestCase):
    def test_review_builder1_still_single_compliance_call(self) -> None:
        plan = _named_person_domino_plan()
        with patch(
            "engine.builder1_image_compliance._openai_compliance_review_call",
            return_value='{"pass": true, "violations": [], "confidence": "high"}',
        ) as mock_review:
            review_builder1_ad_image_compliance(
                b"img",
                product_name=plan.product_name_resolved,
                ad_index=1,
                series_plan=plan,
                reviewer=None,
            )
        self.assertEqual(mock_review.call_count, 1)

    def test_finalize_does_not_invoke_openai(self) -> None:
        with patch("engine.builder1_image_compliance._openai_compliance_review_call") as mock_review:
            finalize_compliance_result(
                reviewer_pass=False,
                candidate_violations=["product_visible_without_explicit_request"],
                evidence_items=[],
                overall_confidence="high",
                series_plan=_named_person_domino_plan(),
                product_match=ComplianceProductMatch(
                    matched_visual_element="Domino tiles",
                    relationship_to_advertised_product="transferred_object",
                ),
            )
        mock_review.assert_not_called()


class TestCanonicalResponseParsing(unittest.TestCase):
    def test_relationship_none_with_brand_text_none_not_hard(self) -> None:
        plan = _named_person_domino_plan()
        result = parse_image_compliance_response(
            {
                "reviewStatus": "completed",
                "hardViolations": ["product_visible_without_explicit_request"],
                "advisories": [],
                "evidence": [
                    {
                        "code": "product_visible_without_explicit_request",
                        "confidence": "high",
                        "evidenceType": "visual_context",
                        "description": "Domino tiles centered",
                        "location": "center",
                        "relationshipToBrandText": "none",
                        "symbolDescription": None,
                        "symbolLocation": None,
                        "relationshipToProductName": None,
                        "relationshipToSlogan": None,
                        "compactAndIsolated": None,
                        "enclosedAsBadgeOrSeal": None,
                        "repeatedAsBrandSignature": None,
                    }
                ],
                "overallConfidence": "high",
                "productMatch": _default_product_match(
                    advertisedProductPresent=True,
                    productMatchBasis="none",
                    matchedVisualElement="Domino tiles",
                    relationshipToAdvertisedProduct="none",
                    productMatchExplanation="Reviewer guess only.",
                ),
            },
            series_plan=plan,
        )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
