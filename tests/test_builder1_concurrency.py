"""
Builder1 multi-campaign concurrency and isolation tests.

Run: python -m unittest tests.test_builder1_concurrency -v
"""
from __future__ import annotations

import copy
import threading
import unittest
from typing import Dict, List
from unittest.mock import patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    get_campaign_store_backend,
    mark_ad_generated,
    release_generation_lock,
    reserve_next_ad_index,
    validate_next_ad_request,
)
from engine.builder1_image_generator import ImageRateLimitError, generate_builder1_ad_image
from engine.builder1_jobs_store import (
    clear_memory_jobs_for_tests,
    create_builder1_job,
    finalize_builder1_job,
    get_builder1_job,
)
from tests.test_builder1_series import _base_campaign, _parse


def _plan(ad_count: int, *, marker: str):
    data = _base_campaign(ad_count)
    data["productNameResolved"] = f"Brand-{marker}"
    data["ads"] = copy.deepcopy(data["ads"])
    for ad in data["ads"]:
        ad["marketingText"] = " ".join([f"{marker}{i}" for i in range(1, 51)])
    return _parse(data, ad_count)


class TestInterleavedCampaignTargets(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_three_campaigns_keep_isolated_targets(self) -> None:
        specs = [("A", 2), ("B", 4), ("C", 3)]
        for marker, count in specs:
            create_campaign_session(
                campaign_id=f"campaign-{marker}",
                plan=_plan(count, marker=marker),
                target_ad_count=count,
            )

        for marker, count in specs:
            session = get_campaign_session(f"campaign-{marker}")
            self.assertEqual(session.target_ad_count, count)
            self.assertEqual(len(session.plan.ads), count)
            self.assertEqual(session.plan.product_name_resolved, f"Brand-{marker}")

    def test_interleaved_generation_respects_each_target(self) -> None:
        order = [("A", 2), ("B", 4), ("C", 3)]
        for marker, count in order:
            create_campaign_session(
                campaign_id=f"campaign-{marker}",
                plan=_plan(count, marker=marker),
                target_ad_count=count,
            )

        flow = [
            ("A", 1),
            ("B", 1),
            ("C", 1),
            ("A", 2),
            ("C", 2),
            ("B", 2),
            ("C", 3),
            ("B", 3),
            ("B", 4),
        ]
        for marker, ad_index in flow:
            cid = f"campaign-{marker}"
            reserve_next_ad_index(cid, ad_index)
            mark_ad_generated(cid, ad_index)

        self.assertTrue(get_campaign_session("campaign-A").complete)
        self.assertEqual(get_campaign_session("campaign-A").generated_count, 2)
        self.assertTrue(get_campaign_session("campaign-B").complete)
        self.assertEqual(get_campaign_session("campaign-B").generated_count, 4)
        self.assertTrue(get_campaign_session("campaign-C").complete)
        self.assertEqual(get_campaign_session("campaign-C").generated_count, 3)

        self.assertNotEqual(
            get_campaign_session("campaign-A").plan.product_name_resolved,
            get_campaign_session("campaign-B").plan.product_name_resolved,
        )


class TestAtomicReservation(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_duplicate_simultaneous_reservation_blocked(self) -> None:
        create_campaign_session(campaign_id="dup", plan=_plan(3, marker="dup"), target_ad_count=3)
        reserve_next_ad_index("dup", 1)
        mark_ad_generated("dup", 1)

        barrier = threading.Barrier(2)
        results: List[str] = []

        def worker() -> None:
            barrier.wait()
            try:
                reserve_next_ad_index("dup", 2)
                results.append("ok")
            except CampaignStoreError as exc:
                results.append(exc.code)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        release_generation_lock("dup")

        self.assertEqual(results.count("ok"), 1)
        self.assertEqual(results.count("campaign_generation_in_progress"), 1)

    def test_generated_count_cannot_exceed_target(self) -> None:
        create_campaign_session(campaign_id="cap", plan=_plan(2, marker="cap"), target_ad_count=2)
        reserve_next_ad_index("cap", 1)
        mark_ad_generated("cap", 1)
        reserve_next_ad_index("cap", 2)
        mark_ad_generated("cap", 2)
        session = get_campaign_session("cap")
        self.assertTrue(session.complete)
        with self.assertRaises(CampaignStoreError):
            validate_next_ad_request("cap", 3)


class Test429RetrySafety(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_429_releases_reservation_without_advancing_count(self) -> None:
        create_campaign_session(campaign_id="rate", plan=_plan(2, marker="rate"), target_ad_count=2)
        reserve_next_ad_index("rate", 1)
        release_generation_lock("rate")
        session = get_campaign_session("rate")
        self.assertEqual(session.generated_count, 0)
        self.assertEqual(session.next_ad_index, 1)
        self.assertEqual(session.target_ad_count, 2)

    def test_429_in_one_campaign_does_not_affect_another(self) -> None:
        create_campaign_session(campaign_id="a429", plan=_plan(2, marker="a"), target_ad_count=2)
        create_campaign_session(campaign_id="bok", plan=_plan(4, marker="b"), target_ad_count=4)
        reserve_next_ad_index("a429", 1)
        release_generation_lock("a429")
        reserve_next_ad_index("bok", 1)
        mark_ad_generated("bok", 1)
        self.assertEqual(get_campaign_session("bok").generated_count, 1)
        self.assertEqual(get_campaign_session("a429").generated_count, 0)


class TestJobIsolation(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_jobs_for_tests()

    def test_jobs_do_not_overwrite_each_other(self) -> None:
        create_builder1_job(job_id="job-a", campaign_id="camp-a", target_ad_count=2)
        create_builder1_job(job_id="job-b", campaign_id="camp-b", target_ad_count=4)
        finalize_builder1_job(
            "job-a",
            {"ok": True, "campaignId": "camp-a", "generatedCount": 1},
            target_ad_count=2,
        )
        finalize_builder1_job(
            "job-b",
            {"ok": True, "campaignId": "camp-b", "generatedCount": 1},
            target_ad_count=4,
        )
        job_a = get_builder1_job("job-a")
        job_b = get_builder1_job("job-b")
        assert job_a is not None and job_b is not None
        self.assertEqual(job_a["targetAdCount"], 2)
        self.assertEqual(job_b["targetAdCount"], 4)
        self.assertEqual(job_a["campaignId"], "camp-a")
        self.assertEqual(job_b["campaignId"], "camp-b")

    def test_job_campaign_mismatch_is_rejected(self) -> None:
        create_builder1_job(job_id="job-x", campaign_id="camp-x", target_ad_count=2)
        finalize_builder1_job(
            "job-x",
            {"ok": True, "campaignId": "camp-y", "generatedCount": 1},
            target_ad_count=2,
        )
        job = get_builder1_job("job-x")
        assert job is not None
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["error"], "job_campaign_mismatch")


class TestSharedStoreSimulation(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_worker_b_reads_campaign_created_by_worker_a(self) -> None:
        create_campaign_session(campaign_id="shared", plan=_plan(3, marker="shared"), target_ad_count=3)

        seen: Dict[str, int] = {}

        def worker_a() -> None:
            reserve_next_ad_index("shared", 1)
            mark_ad_generated("shared", 1)

        def worker_b() -> None:
            session = get_campaign_session("shared")
            seen["target"] = session.target_ad_count

        worker_a()
        worker_b()
        self.assertEqual(seen["target"], 3)
        self.assertEqual(get_campaign_store_backend(), "memory")


class TestAppLevelConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())

    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    @patch("app.generate_builder1_ad_image")
    def test_generate_next_reserves_before_async_job(self, mock_image) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())
        from app import app

        mock_image.return_value = type("R", (), {"visual_prompt": "p", "image_bytes": b"x"})()
        plan = _plan(2, marker="app")
        create_campaign_session(campaign_id="app-c", plan=plan, target_ad_count=2)
        reserve_next_ad_index("app-c", 1)
        mark_ad_generated("app-c", 1)

        client = app.test_client()
        with patch("app._builder1_executor.submit"):
            resp = client.post(
                "/api/builder1-generate-next",
                json={"campaignId": "app-c", "expectedNextIndex": 2},
            )
        self.assertEqual(resp.status_code, 202)
        session = get_campaign_session("app-c")
        self.assertEqual(session.generating_index, 2)
        self.assertEqual(session.target_ad_count, 2)


class TestImageGenerationIsolation(unittest.TestCase):
    def test_image_prompts_use_campaign_plan_only(self) -> None:
        plan_a = _plan(2, marker="imgA")
        plan_b = _plan(4, marker="imgB")
        calls: List[str] = []

        def caller(prompt: str, fmt: str) -> bytes:
            calls.append(prompt)
            return b"img"

        generate_builder1_ad_image(plan_a, 1, caller)
        generate_builder1_ad_image(plan_b, 1, caller)
        self.assertIn("Brand-imgA", calls[0])
        self.assertIn("Brand-imgB", calls[1])
        self.assertNotIn("Brand-imgB", calls[0])


class TestBuilder2Unchanged(unittest.TestCase):
    def test_no_builder2_concurrency_modules(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_campaign_store", text)


if __name__ == "__main__":
    unittest.main()
