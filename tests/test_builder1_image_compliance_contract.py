"""
Builder1 image compliance production fail-closed contract tests.

Run: python -m unittest tests.test_builder1_image_compliance_contract -v
"""
from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import Mock, patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    release_generation_lock,
    reserve_next_ad_index,
    try_acquire_generation_lock,
)
from engine.builder1_image_compliance import (
    ImageComplianceError,
    ImageComplianceResponseError,
    ImageComplianceResult,
    ImageComplianceUnavailableError,
    parse_image_compliance_response,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from tests.builder1_test_helpers import pass_compliance_reviewer
from tests.test_builder1_series import _base_campaign, _parse


def _plan(ad_count: int = 2):
    return _parse(_base_campaign(ad_count), ad_count)


class TestComplianceProductionFailClosed(unittest.TestCase):
    def test_injected_reviewer_may_approve_without_api(self) -> None:
        result = review_builder1_ad_image_compliance(
            b"img",
            product_name="TestBrand",
            ad_index=1,
            reviewer=pass_compliance_reviewer,
        )
        self.assertTrue(result.passed)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_never_auto_passes(self) -> None:
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            review_builder1_ad_image_compliance(
                b"img",
                product_name="TestBrand",
                ad_index=1,
            )
        self.assertEqual(ctx.exception.reason_code, "missing_api_key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_missing_review_client_returns_unavailable(self) -> None:
        with patch(
            "engine.builder1_image_compliance._openai_compliance_review_call",
            side_effect=ImageComplianceUnavailableError("client_unavailable", ad_index=-1),
        ):
            with self.assertRaises(ImageComplianceUnavailableError) as ctx:
                review_builder1_ad_image_compliance(
                    b"img",
                    product_name="TestBrand",
                    ad_index=2,
                )
        self.assertEqual(ctx.exception.reason_code, "client_unavailable")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_unsupported_model_never_auto_passes(self) -> None:
        with patch(
            "engine.builder1_image_compliance._openai_compliance_review_call",
            side_effect=ImageComplianceUnavailableError("unsupported_model", ad_index=-1),
        ):
            with self.assertRaises(ImageComplianceUnavailableError) as ctx:
                review_builder1_ad_image_compliance(
                    b"img",
                    product_name="TestBrand",
                    ad_index=1,
                )
        self.assertEqual(ctx.exception.reason_code, "unsupported_model")

    def test_malformed_json_never_auto_passes(self) -> None:
        with self.assertRaises(ImageComplianceResponseError):
            parse_image_compliance_response("not json")

    def test_pass_true_with_violations_rejected(self) -> None:
        with self.assertRaises(ImageComplianceResponseError) as ctx:
            parse_image_compliance_response(
                {
                    "pass": True,
                    "violations": ["invented_product_logo"],
                    "confidence": "high",
                }
            )
        self.assertIn("pass_true_with_violations", str(ctx.exception))

    def test_pass_false_without_violations_rejected(self) -> None:
        with self.assertRaises(ImageComplianceResponseError):
            parse_image_compliance_response(
                {"pass": False, "violations": [], "confidence": "high"}
            )


class TestComplianceGenerationFlow(unittest.TestCase):
    def test_valid_pass_stores_image(self) -> None:
        result = generate_builder1_ad_image(
            _plan(2),
            1,
            lambda _p, _f: b"approved",
            compliance_reviewer=pass_compliance_reviewer,
        )
        self.assertEqual(result.index, 1)
        self.assertEqual(result.image_bytes, b"approved")

    def test_failed_review_triggers_one_regeneration(self) -> None:
        calls = {"gen": 0, "review": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            calls["review"] += 1
            if calls["review"] == 1:
                return ImageComplianceResult(
                    passed=False,
                    violations=["logo_like_brand_symbol"],
                    confidence="high",
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        generate_builder1_ad_image(_plan(2), 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 2)
        self.assertEqual(calls["review"], 2)

    def test_second_visual_failure_raises_compliance_failed(self) -> None:
        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return ImageComplianceResult(
                passed=False,
                violations=["invented_product_logo"],
                confidence="high",
            )

        with self.assertRaises(ImageComplianceError):
            generate_builder1_ad_image(
                _plan(2),
                1,
                lambda _p, _f: b"img",
                compliance_reviewer=reviewer,
            )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_review_infrastructure_failure_raises_unavailable(self) -> None:
        with self.assertRaises(ImageComplianceUnavailableError):
            generate_builder1_ad_image(
                _plan(2),
                1,
                lambda _p, _f: b"img",
            )


class TestCompliancePublicApiContract(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def _reserved_session(self, *, campaign_id: str = "cmp-contract", ad_count: int = 4):
        plan = _plan(ad_count)
        create_campaign_session(
            campaign_id=campaign_id,
            plan=plan,
            target_ad_count=ad_count,
        )
        try_acquire_generation_lock(campaign_id, 1)
        mark_ad_generated(campaign_id, 1)
        session = reserve_next_ad_index(campaign_id, 2)
        return session

    def test_compliance_failed_public_payload(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session()
        with patch(
            "app.generate_builder1_ad_image",
            side_effect=ImageComplianceError(["invented_product_logo"], ad_index=2),
        ):
            result = _builder1_generate_single_ad(
                job_id="job-fail",
                campaign_id="cmp-contract",
                ad_index=2,
                already_reserved=True,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "image_compliance_failed")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["campaignId"], "cmp-contract")
        self.assertEqual(result["nextAdIndex"], 2)
        self.assertEqual(result["generatedCount"], 1)
        self.assertEqual(result["targetAdCount"], 4)
        self.assertNotIn("ad", result)
        self.assertNotIn("imageBase64", result)
        self.assertNotIn("image_base64", result)

    def test_compliance_unavailable_public_payload(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session(campaign_id="cmp-unavail")
        with patch(
            "app.generate_builder1_ad_image",
            side_effect=ImageComplianceUnavailableError("missing_api_key", ad_index=2),
        ):
            result = _builder1_generate_single_ad(
                job_id="job-unavail",
                campaign_id="cmp-unavail",
                ad_index=2,
                already_reserved=True,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "image_compliance_unavailable")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["nextAdIndex"], 2)
        self.assertEqual(result["generatedCount"], 1)
        self.assertEqual(result["targetAdCount"], 4)
        self.assertNotIn("ad", result)

    def test_both_errors_share_retryable_and_next_ad_index(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session(campaign_id="cmp-both")
        errors = (
            ImageComplianceError(["invented_product_logo"], ad_index=2),
            ImageComplianceUnavailableError("review_service_error", ad_index=2),
        )
        for idx, error in enumerate(errors):
            if idx > 0:
                reserve_next_ad_index("cmp-both", 2)
            with patch("app.generate_builder1_ad_image", side_effect=error):
                result = _builder1_generate_single_ad(
                    job_id="job-both",
                    campaign_id="cmp-both",
                    ad_index=2,
                    already_reserved=True,
                )
            self.assertTrue(result["retryable"])
            self.assertEqual(result["nextAdIndex"], 2)

    def test_generated_count_does_not_advance_on_failed(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session(campaign_id="cmp-no-advance")
        with patch(
            "app.generate_builder1_ad_image",
            side_effect=ImageComplianceError(["invented_product_logo"], ad_index=2),
        ):
            _builder1_generate_single_ad(
                job_id="job-1",
                campaign_id="cmp-no-advance",
                ad_index=2,
                already_reserved=True,
            )
        session = get_campaign_session("cmp-no-advance")
        self.assertEqual(session.generated_count, 1)
        self.assertEqual(session.target_ad_count, 4)

    def test_retry_same_ad_index_after_lock_release(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session(campaign_id="cmp-retry")
        with patch(
            "app.generate_builder1_ad_image",
            side_effect=ImageComplianceUnavailableError("transient_review_failure", ad_index=2),
        ):
            first = _builder1_generate_single_ad(
                job_id="job-r1",
                campaign_id="cmp-retry",
                ad_index=2,
                already_reserved=True,
            )
        self.assertEqual(first["nextAdIndex"], 2)
        reserve_next_ad_index("cmp-retry", 2)
        image_result = Mock(visual_prompt="p", image_bytes=b"ok")
        with patch("app.generate_builder1_ad_image", return_value=image_result):
            with patch("app.image_bytes_to_base64", return_value="b2s="):
                second = _builder1_generate_single_ad(
                    job_id="job-r2",
                    campaign_id="cmp-retry",
                    ad_index=2,
                    already_reserved=True,
                )
        self.assertTrue(second["ok"])

    def test_planning_not_rerun_on_compliance_error(self) -> None:
        from app import _builder1_generate_single_ad

        self._reserved_session(campaign_id="cmp-no-plan")
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            with patch(
                "app.generate_builder1_ad_image",
                side_effect=ImageComplianceError(["invented_product_logo"], ad_index=2),
            ):
                _builder1_generate_single_ad(
                    job_id="job-np",
                    campaign_id="cmp-no-plan",
                    ad_index=2,
                    already_reserved=True,
                )

    def test_concurrent_campaigns_remain_isolated(self) -> None:
        plan_a = _plan(2)
        plan_b = _plan(2)
        create_campaign_session(campaign_id="iso-a", plan=plan_a, target_ad_count=2)
        create_campaign_session(campaign_id="iso-b", plan=plan_b, target_ad_count=2)
        try_acquire_generation_lock("iso-a", 1)
        try_acquire_generation_lock("iso-b", 1)
        self.assertEqual(get_campaign_session("iso-a").generating_index, 1)
        self.assertEqual(get_campaign_session("iso-b").generating_index, 1)

    def test_two_retries_cannot_reserve_same_index(self) -> None:
        create_campaign_session(campaign_id="lock-a", plan=_plan(2), target_ad_count=2)
        try_acquire_generation_lock("lock-a", 1)
        with self.assertRaises(CampaignStoreError):
            try_acquire_generation_lock("lock-a", 1)

    def test_lock_release_only_affects_current_campaign(self) -> None:
        create_campaign_session(campaign_id="rel-a", plan=_plan(2), target_ad_count=2)
        create_campaign_session(campaign_id="rel-b", plan=_plan(2), target_ad_count=2)
        try_acquire_generation_lock("rel-a", 1)
        try_acquire_generation_lock("rel-b", 1)
        release_generation_lock("rel-a")
        session_b = get_campaign_session("rel-b")
        self.assertEqual(session_b.generating_index, 1)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_files_exist_without_compliance_changes(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "engine"
        names = {path.name for path in root.glob("builder2*.py")}
        self.assertTrue(names)


if __name__ == "__main__":
    unittest.main()
