"""
Builder1 pending image store tests — size limits, persistence, cleanup.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    mark_ad_generated,
    reserve_next_ad_index,
)
from engine.builder1_pending_image_store import (
    PENDING_IMAGE_MAX_BYTES,
    PendingImageStoreError,
    clear_memory_pending_for_tests,
    delete_pending_image,
    load_pending_image,
    pending_image_reference,
    save_pending_image,
)
from tests.test_builder1_series import _base_campaign, _parse


def _plan(ad_count: int = 2):
    return _parse(_base_campaign(ad_count), ad_count)


class TestPendingImageStore(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_pending_for_tests()
        clear_memory_store_for_tests()

    def test_normal_pending_image_persists_and_reloads(self) -> None:
        ref = save_pending_image(
            campaign_id="cmp-pending",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"small-image",
            visual_prompt="prompt",
        )
        loaded = load_pending_image(ref)
        self.assertEqual(loaded.image_bytes, b"small-image")
        self.assertEqual(loaded.visual_prompt, "prompt")
        self.assertEqual(
            ref,
            pending_image_reference(campaign_id="cmp-pending", ad_index=1, plan_revision=1),
        )

    def test_oversized_pending_image_rejected_before_store_write(self) -> None:
        oversized = b"x" * (PENDING_IMAGE_MAX_BYTES + 1)
        with self.assertRaises(PendingImageStoreError) as ctx:
            save_pending_image(
                campaign_id="cmp-big",
                ad_index=1,
                plan_revision=1,
                image_bytes=oversized,
                visual_prompt="prompt",
            )
        self.assertEqual(ctx.exception.code, "pending_image_too_large")
        ref = pending_image_reference(campaign_id="cmp-big", ad_index=1, plan_revision=1)
        with self.assertRaises(PendingImageStoreError):
            load_pending_image(ref)

    def test_pending_image_cleaned_after_successful_ad_persistence(self) -> None:
        from engine.builder1_campaign_store import mark_compliance_review_required

        plan = _plan(2)
        create_campaign_session(campaign_id="cmp-clean", plan=plan, target_ad_count=2)
        ref = save_pending_image(
            campaign_id="cmp-clean",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"keep-until-ad",
            visual_prompt="prompt",
        )
        mark_compliance_review_required(
            "cmp-clean",
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
            visual_prompt="prompt",
        )
        reserve_next_ad_index("cmp-clean", 1, job_id="job-clean")
        mark_ad_generated("cmp-clean", 1)
        with self.assertRaises(PendingImageStoreError):
            load_pending_image(ref)

    def test_same_reference_overwrites_without_duplicate_keys(self) -> None:
        ref1 = save_pending_image(
            campaign_id="cmp-one",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"v1",
            visual_prompt="p1",
        )
        ref2 = save_pending_image(
            campaign_id="cmp-one",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"v2",
            visual_prompt="p2",
        )
        self.assertEqual(ref1, ref2)
        loaded = load_pending_image(ref1)
        self.assertEqual(loaded.image_bytes, b"v2")


class TestReviewOnlyStillZeroPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())
        sys.modules["openai"].OpenAI = MagicMock()
        httpx_module = MagicMock()
        httpx_module.Timeout = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("httpx", httpx_module)

    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_pending_for_tests()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_review_only_performs_zero_planning_and_image_calls(self) -> None:
        from engine.builder1_campaign_store import mark_compliance_review_required
        from tests.builder1_test_helpers import pass_compliance_reviewer, seed_builder1_image_job

        plan = _plan(3)
        create_campaign_session(campaign_id="cmp-ro-size", plan=plan, target_ad_count=3)
        ref = save_pending_image(
            campaign_id="cmp-ro-size",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"stored-image",
            visual_prompt="visual",
        )
        mark_compliance_review_required(
            "cmp-ro-size",
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
            visual_prompt="visual",
        )
        reserve_next_ad_index("cmp-ro-size", 1, job_id="job-ro-size")
        seed_builder1_image_job(
            job_id="job-ro-size",
            campaign_id="cmp-ro-size",
            ad_index=1,
            target_ad_count=3,
            plan_revision=1,
        )

        from app import _builder1_generate_review_only_ad

        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            with patch("app.generate_builder1_ad_image") as mock_image:
                with patch(
                    "app.review_builder1_ad_image_compliance",
                    return_value=pass_compliance_reviewer(image_bytes=b"stored-image"),
                ):
                    result = _builder1_generate_review_only_ad(
                        job_id="job-ro-size",
                        campaign_id="cmp-ro-size",
                        ad_index=1,
                        already_reserved=True,
                    )
        mock_plan.assert_not_called()
        mock_image.assert_not_called()
        self.assertTrue(result["ok"])
        delete_pending_image(ref)
