"""
Builder1 compliance adjudication and three-level image attempt tests.

Run: python -m unittest tests.test_builder1_compliance_adjudication -v
"""
from __future__ import annotations

import copy
import os
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_compliance_adjudication import (
    ComplianceEvidenceItem,
    adjudicate_compliance_review,
)
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    classify_compliance_failure,
)
from engine.builder1_image_compliance import (
    ImageComplianceError,
    ImageComplianceResult,
    finalize_compliance_result,
    parse_image_compliance_response,
)
from engine.builder1_image_generator import MAX_INTERNAL_IMAGE_ATTEMPTS, generate_builder1_ad_image
from engine.builder1_image_retry import (
    CORRECTION_PROFILE_MINIMAL_SAFE,
    build_minimal_safe_execution_block,
    union_advisories_for_ad,
    union_hard_violations_for_ad,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planning_profile import resolve_stage_model
from engine.builder1_planner import plan_builder1
from engine.builder1_product_modality import ProductModality, derive_product_modality
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _full_final_responses


def _plan(ad_count: int = 3):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.transferred_object = "Rubber ball family"
    plan.transferred_object_action = "Bounces after a controlled drop"
    plan.physical_generator = "Rubber ball family"
    plan.product_visibility_policy = "FORBIDDEN"
    plan.planning_internals = dict(plan.planning_internals or {})
    plan.planning_internals["productModality"] = ProductModality.PHYSICAL_PRODUCT.value
    return plan


def _service_plan(ad_count: int = 3):
    plan = _plan(ad_count)
    plan.product_description = "Digital advertising agent for automated campaign optimization"
    plan.planning_internals["productModality"] = ProductModality.DIGITAL_PRODUCT.value
    return plan


class TestProductRoleAdjudication(unittest.TestCase):
    def test_pixel_only_product_as_generator_is_advisory(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["product_used_as_physical_generator"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertEqual(result.hard_violations, [])
        self.assertIn("possible_product_resemblance", result.advisories)
        self.assertTrue(result.passed)

    def test_structured_generator_conflict_stays_hard(self) -> None:
        plan = _plan(3)
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        plan.product_description = "Lightweight running shoe for daily training"
        result = adjudicate_compliance_review(
            raw_violations=["product_used_as_physical_generator"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=plan,
        )
        self.assertIn("product_used_as_physical_generator", result.hard_violations)

    def test_physical_product_visible_hard_fails(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(code="product_visible_without_explicit_request", confidence="high")
            ],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertIn("product_visible_without_explicit_request", result.hard_violations)

    def test_service_not_inferred_from_advertising_imagery(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[],
            overall_confidence="medium",
            series_plan=_service_plan(3),
        )
        self.assertEqual(result.hard_violations, [])
        self.assertTrue(result.advisories)

    def test_digital_product_requires_concrete_evidence(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["product_visible_without_explicit_request"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="product_visible_without_explicit_request",
                    evidence_type="application_interface",
                    confidence="high",
                )
            ],
            overall_confidence="high",
            series_plan=_service_plan(3),
        )
        self.assertIn("product_visible_without_explicit_request", result.hard_violations)


class TestLogoEvidenceAdjudication(unittest.TestCase):
    def _logo_item(self, **kwargs: Any) -> ComplianceEvidenceItem:
        base = {
            "code": "logo_like_brand_symbol",
            "symbol_description": "Small circular icon",
            "symbol_location": "beside product name",
            "relationship_to_product_name": "adjacent lockup",
            "compact_and_isolated": True,
            "confidence": "high",
        }
        base.update(kwargs)
        return ComplianceEvidenceItem(**base)

    def test_small_symbol_beside_product_name_hard_fails(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["logo_like_brand_symbol"],
            evidence_items=[self._logo_item()],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertIn("logo_like_brand_symbol", result.hard_violations)

    def test_badge_or_seal_hard_fails(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["logo_like_brand_symbol"],
            evidence_items=[self._logo_item(enclosed_as_badge_or_seal=True, compact_and_isolated=False)],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertIn("logo_like_brand_symbol", result.hard_violations)

    def test_large_central_object_is_advisory(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["logo_like_brand_symbol"],
            evidence_items=[
                ComplianceEvidenceItem(
                    code="logo_like_brand_symbol",
                    symbol_description="Large rubber ball hero object",
                    symbol_location="center of composition",
                    confidence="high",
                )
            ],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertEqual(result.hard_violations, [])
        self.assertIn("possible_logo_like_shape", result.advisories)

    def test_low_confidence_logo_is_advisory(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["logo_like_brand_symbol"],
            evidence_items=[self._logo_item(confidence="low")],
            overall_confidence="low",
            series_plan=_plan(3),
        )
        self.assertEqual(result.hard_violations, [])
        self.assertIn("low_confidence_logo_identification", result.advisories)

    def test_missing_evidence_prevents_hard_logo_rejection(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["logo_like_brand_symbol"],
            evidence_items=[],
            overall_confidence="medium",
            series_plan=_plan(3),
        )
        self.assertEqual(result.hard_violations, [])
        self.assertIn("possible_logo_like_shape", result.advisories)


class TestComplianceContract(unittest.TestCase):
    def test_result_separates_hard_and_advisories(self) -> None:
        parsed = parse_image_compliance_response(
            {
                "reviewStatus": "completed",
                "hardViolations": ["invented_product_logo"],
                "advisories": ["possible_logo_like_shape"],
                "overallConfidence": "high",
                "evidence": [],
            }
        )
        self.assertIn("invented_product_logo", parsed.hard_violations or [])
        self.assertIn("possible_logo_like_shape", parsed.advisories or [])

    def test_advisory_only_image_passes(self) -> None:
        result = finalize_compliance_result(
            reviewer_pass=False,
            candidate_violations=["product_used_as_physical_generator"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.hard_violations, [])

    def test_objective_no_logo_still_fails(self) -> None:
        result = finalize_compliance_result(
            reviewer_pass=False,
            candidate_violations=["invented_product_logo"],
            evidence_items=[ComplianceEvidenceItem(code="invented_product_logo", confidence="high")],
            overall_confidence="high",
            series_plan=_plan(3),
        )
        self.assertFalse(result.passed)
        self.assertIn("invented_product_logo", result.hard_violations or [])


class TestThreeLevelAttempts(unittest.TestCase):
    def test_three_internal_attempts_before_failure(self) -> None:
        gen_calls = {"n": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            gen_calls["n"] += 1
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return ImageComplianceResult(
                passed=False,
                violations=["invented_product_logo"],
                hard_violations=["invented_product_logo"],
                raw_violations=["invented_product_logo"],
                confidence="high",
            )

        with self.assertRaises(ImageComplianceError):
            generate_builder1_ad_image(_plan(3), 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(gen_calls["n"], MAX_INTERNAL_IMAGE_ATTEMPTS)

    def test_advisory_only_does_not_consume_extra_attempt(self) -> None:
        gen_calls = {"n": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            gen_calls["n"] += 1
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return ImageComplianceResult(
                passed=False,
                violations=["product_used_as_physical_generator"],
                raw_violations=["product_used_as_physical_generator"],
                confidence="high",
            )

        result = generate_builder1_ad_image(_plan(3), 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(gen_calls["n"], 1)
        self.assertEqual(result.index, 1)

    def test_minimal_safe_block_preserves_core_elements(self) -> None:
        plan = _plan(3)
        block = build_minimal_safe_execution_block(series_plan=plan, ad_plan=plan.ads[0])
        self.assertIn("Rubber ball family", block)
        self.assertIn(plan.brand_slogan, block)
        self.assertIn("plain readable typography", block.lower())
        self.assertIn("MINIMAL SAFE EXECUTION", block)


class TestModalityDerivation(unittest.TestCase):
    def test_digital_agent_modality(self) -> None:
        self.assertEqual(
            derive_product_modality(product_description="AI digital advertising agent platform"),
            ProductModality.DIGITAL_PRODUCT,
        )


class TestRegression(unittest.TestCase):
    def test_supplied_name_planning_remains_six_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_planning_remains_seven_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)

    def test_quality_planning_on_o3_pro(self) -> None:
        with patch.dict(
            os.environ,
            {"BUILDER1_PLANNING_PROFILE": "QUALITY", "BUILDER1_QUALITY_MODEL": "o3-pro"},
            clear=False,
        ):
            self.assertEqual(resolve_stage_model("graphic_system"), "o3-pro")

    def test_builder2_unchanged(self) -> None:
        import os

        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine", "builder2_zip.py")))


if __name__ == "__main__":
    unittest.main()
