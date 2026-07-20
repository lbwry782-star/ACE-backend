"""
Builder1 generation lock ownership and initial-flow concurrency tests.

Run: python -m unittest tests.test_builder1_generation_lock -v
"""
from __future__ import annotations

import copy
import json
import threading
import unittest
from typing import List
from unittest.mock import patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    _load_raw,
    _memory_campaigns,
    _memory_lock,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    release_generation_lock,
    reserve_next_ad_index,
)
from engine.builder1_jobs_store import (
    clear_memory_jobs_for_tests,
    create_builder1_job,
    finalize_builder1_job,
    get_builder1_job,
)
from tests.builder1_test_helpers import pass_compliance_reviewer
from tests.test_builder1_series import _base_campaign, _parse


def _plan(ad_count: int, *, marker: str = "lock"):
    data = _base_campaign(ad_count)
    data["productNameResolved"] = f"Brand-{marker}"
    for ad in data["ads"]:
        ad["marketingText"] = " ".join([f"{marker}{i}" for i in range(1, 51)])
    return _parse(data, ad_count)


class TestInitialFlowOwnership(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_successful_planning_creates_campaign_without_active_lock(self) -> None:
        create_campaign_session(campaign_id="init-a", plan=_plan(2), target_ad_count=2)
        raw = _load_raw("init-a")
        assert raw is not None
        self.assertNotIn("generatingIndex", raw)
        session = get_campaign_session("init-a")
        self.assertIsNone(session.generating_index)
        self.assertEqual(session.next_ad_index, 1)
        self.assertEqual(session.generated_count, 0)

    def test_same_initial_job_reserves_ad_one_after_campaign_creation(self) -> None:
        create_campaign_session(campaign_id="init-b", plan=_plan(2), target_ad_count=2)
        session = reserve_next_ad_index("init-b", 1, job_id="job-initial")
        self.assertEqual(session.generating_index, 1)
        self.assertEqual(session.generating_lock_owner_job_id, "job-initial")
        self.assertIsNotNone(session.generating_lock_token)

    def test_same_initial_job_is_not_blocked_by_its_own_lock(self) -> None:
        create_campaign_session(campaign_id="init-c", plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index("init-c", 1, job_id="job-initial")
        session = reserve_next_ad_index("init-c", 1, job_id="job-initial")
        self.assertEqual(session.generating_index, 1)
        self.assertEqual(session.generating_lock_owner_job_id, "job-initial")

    def test_null_generating_index_artifact_does_not_block_memory_reserve(self) -> None:
        create_campaign_session(campaign_id="init-null", plan=_plan(2), target_ad_count=2)
        with _memory_lock:
            _memory_campaigns["init-null"]["generatingIndex"] = None
        session = reserve_next_ad_index("init-null", 1, job_id="job-null")
        self.assertEqual(session.generating_index, 1)

    @patch("app.generate_builder1_ad_image")
    @patch("app.plan_builder1")
    def test_production_shaped_initial_job_generates_ad_one(self, mock_plan, mock_image) -> None:
        from app import _builder1_generate_initial

        mock_plan.return_value = _plan(2, marker="prod")
        mock_image.return_value = type(
            "ImageResult",
            (),
            {"visual_prompt": "prompt", "image_bytes": b"img"},
        )()
        create_builder1_job(
            job_id="job-prod",
            campaign_id="cmp-prod",
            target_ad_count=2,
            stage="planning",
        )

        result = _builder1_generate_initial(
            job_id="job-prod",
            campaign_id="cmp-prod",
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_val="portrait",
            ad_count=2,
            brand_guidelines=None,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["generatedCount"], 1)
        self.assertEqual(result["nextAdIndex"], 2)
        self.assertEqual(result["targetAdCount"], 2)
        session = get_campaign_session("cmp-prod")
        self.assertEqual(session.generated_count, 1)
        self.assertIsNone(session.generating_index)

    @patch("app.generate_builder1_ad_image")
    @patch("app.plan_builder1")
    def test_initial_job_finalizes_success_not_generation_in_progress(self, mock_plan, mock_image) -> None:
        from app import _builder1_generate_initial, _builder1_finalize_job

        mock_plan.return_value = _plan(2, marker="done")
        mock_image.return_value = type(
            "ImageResult",
            (),
            {"visual_prompt": "prompt", "image_bytes": b"img"},
        )()
        create_builder1_job(
            job_id="job-done",
            campaign_id="cmp-done",
            target_ad_count=2,
            stage="planning",
        )
        result = _builder1_generate_initial(
            job_id="job-done",
            campaign_id="cmp-done",
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_val="portrait",
            ad_count=2,
            brand_guidelines=None,
        )
        create_builder1_job(job_id="job-done", campaign_id="cmp-done", target_ad_count=2)
        _builder1_finalize_job("job-done", result, target_ad_count=2)
        job = get_builder1_job("job-done")
        assert job is not None
        self.assertEqual(job["status"], "done")
        self.assertNotEqual(job.get("error"), "campaign_generation_in_progress")

    @patch("app.generate_builder1_ad_image")
    @patch("app.plan_builder1")
    def test_generated_count_increments_only_after_storage(self, mock_plan, mock_image) -> None:
        from app import _builder1_generate_initial

        mock_plan.return_value = _plan(2, marker="count")
        captured: List[int] = []

        def _image(plan, ad_index, caller, **kwargs):
            session = get_campaign_session("cmp-count")
            captured.append(session.generated_count)
            return type("ImageResult", (), {"visual_prompt": "prompt", "image_bytes": b"img"})()

        mock_image.side_effect = _image
        create_builder1_job(
            job_id="job-count",
            campaign_id="cmp-count",
            target_ad_count=2,
            stage="planning",
        )
        result = _builder1_generate_initial(
            job_id="job-count",
            campaign_id="cmp-count",
            product_name="CarryShell",
            product_description="Brief",
            format_val="portrait",
            ad_count=2,
            brand_guidelines=None,
        )
        self.assertEqual(captured, [0])
        self.assertEqual(result["generatedCount"], 1)


class TestTrueConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_different_job_blocked_while_ad_one_reserved(self) -> None:
        create_campaign_session(campaign_id="block-a", plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index("block-a", 1, job_id="owner-job")
        with self.assertRaises(CampaignStoreError) as ctx:
            reserve_next_ad_index("block-a", 1, job_id="other-job")
        self.assertEqual(ctx.exception.code, "campaign_generation_in_progress")

    def test_two_simultaneous_next_ad_requests_cannot_reserve_same_index(self) -> None:
        create_campaign_session(campaign_id="block-b", plan=_plan(3), target_ad_count=3)
        reserve_next_ad_index("block-b", 1, job_id="job-a")
        mark_ad_generated("block-b", 1)

        barrier = threading.Barrier(2)
        results: List[tuple[str, str]] = []

        def worker(job_id: str) -> None:
            barrier.wait()
            try:
                reserve_next_ad_index("block-b", 2, job_id=job_id)
                results.append((job_id, "ok"))
            except CampaignStoreError as exc:
                results.append((job_id, exc.code))

        t1 = threading.Thread(target=worker, args=("job-1",))
        t2 = threading.Thread(target=worker, args=("job-2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        winners = [job_id for job_id, code in results if code == "ok"]
        if winners:
            session = get_campaign_session("block-b")
            release_generation_lock(
                "block-b",
                job_id=winners[0],
                lock_token=session.generating_lock_token or "",
            )

        codes = [code for _, code in results]
        self.assertEqual(codes.count("ok"), 1)
        self.assertEqual(codes.count("campaign_generation_in_progress"), 1)

    def test_campaign_locks_do_not_cross_campaigns(self) -> None:
        create_campaign_session(campaign_id="iso-a", plan=_plan(2, marker="a"), target_ad_count=2)
        create_campaign_session(campaign_id="iso-b", plan=_plan(4, marker="b"), target_ad_count=4)
        reserve_next_ad_index("iso-a", 1, job_id="job-a")
        session_b = reserve_next_ad_index("iso-b", 1, job_id="job-b")
        self.assertEqual(session_b.generating_index, 1)


class TestLockOwnershipRelease(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_old_owner_cannot_release_newer_lock(self) -> None:
        create_campaign_session(campaign_id="token-a", plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index("token-a", 1, job_id="job-a", lock_token="token-a")
        session = get_campaign_session("token-a")
        with self.assertRaises(CampaignStoreError) as ctx:
            release_generation_lock("token-a", job_id="job-a", lock_token="wrong-token")
        self.assertEqual(ctx.exception.code, "campaign_lock_token_mismatch")
        self.assertEqual(session.generating_index, 1)

    def test_different_job_cannot_release_existing_lock(self) -> None:
        create_campaign_session(campaign_id="token-b", plan=_plan(2), target_ad_count=2)
        session = reserve_next_ad_index("token-b", 1, job_id="job-a")
        with self.assertRaises(CampaignStoreError) as ctx:
            release_generation_lock(
                "token-b",
                job_id="job-b",
                lock_token=session.generating_lock_token or "",
            )
        self.assertEqual(ctx.exception.code, "campaign_lock_owner_mismatch")

    def test_release_clears_generation_in_progress(self) -> None:
        create_campaign_session(campaign_id="token-c", plan=_plan(2), target_ad_count=2)
        session = reserve_next_ad_index("token-c", 1, job_id="job-c")
        release_generation_lock(
            "token-c",
            job_id="job-c",
            lock_token=session.generating_lock_token or "",
        )
        cleared = get_campaign_session("token-c")
        self.assertIsNone(cleared.generating_index)


class TestRedisNullArtifactRegression(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_create_campaign_session_json_has_no_generating_index_key(self) -> None:
        create_campaign_session(campaign_id="redis-shape", plan=_plan(2), target_ad_count=2)
        raw = _load_raw("redis-shape")
        assert raw is not None
        encoded = json.dumps(raw)
        self.assertNotIn('"generatingIndex": null', encoded)
        self.assertNotIn('"generatingIndex"', encoded)
        reserve_next_ad_index("redis-shape", 1, job_id="job-redis")
        self.assertEqual(get_campaign_session("redis-shape").generating_index, 1)


class TestGenerateNextOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())

    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    @patch("app.generate_builder1_ad_image")
    def test_generate_next_reserves_with_job_owner_before_worker(self, mock_image) -> None:
        from app import app

        mock_image.return_value = type("R", (), {"visual_prompt": "p", "image_bytes": b"x"})()
        create_campaign_session(campaign_id="next-owner", plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index("next-owner", 1, job_id="job-first")
        mark_ad_generated("next-owner", 1)

        client = app.test_client()
        with patch("app._builder1_executor.submit"):
            resp = client.post(
                "/api/builder1-generate-next",
                json={"campaignId": "next-owner", "expectedNextIndex": 2},
            )
        self.assertEqual(resp.status_code, 202)
        payload = resp.get_json()
        assert payload is not None
        session = get_campaign_session("next-owner")
        self.assertEqual(session.generating_index, 2)
        self.assertEqual(session.generating_lock_owner_job_id, payload["jobId"])


if __name__ == "__main__":
    unittest.main()
