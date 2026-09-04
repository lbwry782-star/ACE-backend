"""
Builder2 initial generate idempotency + cancel ownership — mocked/offline only.
"""
from __future__ import annotations

import os
import threading
import unittest
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder2_initial_generate_idempotency import (
    clear_memory_idempotency_for_tests,
    disable_memory_idempotency_store,
    enable_memory_idempotency_store,
    idempotency_ttl_seconds,
)
from engine.builder2_job_ownership import ownership_fields_for_job_create
from engine.builder2_tournament_recovery import disable_memory_recovery, enable_memory_recovery
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_video_allowance_store import (
    clear_memory_allowance_store_for_tests,
    disable_memory_allowance_store,
    enable_memory_allowance_store,
)
from engine.video_jobs_redis import (
    disable_memory_jobs,
    enable_memory_jobs,
    video_job_get_raw,
)


def _headers(*, batch: str = "batch-idem-a", auth: str = "Bearer token-a", request_id: str = "") -> Dict[str, str]:
    out = {"X-ACE-Batch-State": batch, "Authorization": auth}
    if request_id:
        out["X-ACE-Request-Id"] = request_id
    return out


def _payload(**overrides: Any) -> Dict[str, Any]:
    base = {"productName": "Product", "productDescription": "A valid product description.", "targetVideoCount": 1}
    base.update(overrides)
    return base


class Builder2IdempotencyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()
        enable_memory_allowance_store()
        enable_memory_idempotency_store()
        clear_memory_allowance_store_for_tests()
        clear_memory_idempotency_for_tests()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"
        os.environ["REDIS_URL"] = "redis://test"
        from app import app

        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        disable_memory_jobs()
        disable_memory_store()
        disable_memory_recovery()
        disable_memory_allowance_store()
        disable_memory_idempotency_store()
        clear_memory_allowance_store_for_tests()
        clear_memory_idempotency_for_tests()

    def _post_generate(
        self,
        *,
        request_id: str = "",
        batch: str = "batch-idem-a",
        payload: Dict[str, Any] | None = None,
    ) -> Any:
        body = payload or _payload()
        headers = _headers(batch=batch, request_id=request_id)
        with patch("app.redis_configured", return_value=True):
            return self.client.post("/api/generate-video", json=body, headers=headers)

    def _count_jobs(self) -> int:
        from engine.video_jobs_redis import _memory_job_hashes

        return len(_memory_job_hashes)


class TestInitialGenerateIdempotency(Builder2IdempotencyTestCase):
    def test_first_request_creates_one_job(self) -> None:
        rid = str(uuid.uuid4())
        resp = self._post_generate(request_id=rid)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(self._count_jobs(), 1)

    def test_replay_returns_same_job_and_allowance(self) -> None:
        rid = str(uuid.uuid4())
        first = self._post_generate(request_id=rid).get_json()
        second = self._post_generate(request_id=rid).get_json()
        self.assertEqual(first["jobId"], second["jobId"])
        self.assertEqual(first["videoAllowanceId"], second["videoAllowanceId"])
        self.assertTrue(second.get("idempotentReplay"))
        self.assertEqual(self._count_jobs(), 1)

    def test_replay_does_not_enqueue_again(self) -> None:
        rid = str(uuid.uuid4())
        with patch("engine.video_jobs_redis.get_redis") as redis_mock:
            pipe = MagicMock()
            redis_mock.return_value.pipeline.return_value = pipe
            self._post_generate(request_id=rid)
            first_lpush = pipe.lpush.call_count
            self._post_generate(request_id=rid)
            self.assertEqual(pipe.lpush.call_count, first_lpush)

    def test_fingerprint_conflict(self) -> None:
        rid = str(uuid.uuid4())
        self._post_generate(request_id=rid, payload=_payload(productDescription="First description text."))
        conflict = self._post_generate(request_id=rid, payload=_payload(productDescription="Different description."))
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.get_json().get("error"), "builder2_idempotency_conflict")

    def test_concurrent_same_request_id_one_job(self) -> None:
        rid = str(uuid.uuid4())
        results: List[str] = []
        barrier = threading.Barrier(2)

        def _worker() -> None:
            barrier.wait()
            resp = self._post_generate(request_id=rid)
            results.append(resp.get_json()["jobId"])

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(self._count_jobs(), 1)

    def test_lost_response_replay_recovers_job_a(self) -> None:
        rid = str(uuid.uuid4())
        first = self._post_generate(request_id=rid).get_json()
        job_a = first["jobId"]
        allowance_a = first["videoAllowanceId"]
        replay = self._post_generate(request_id=rid).get_json()
        self.assertEqual(replay["jobId"], job_a)
        self.assertEqual(replay["videoAllowanceId"], allowance_a)
        self.assertEqual(self._count_jobs(), 1)

    def test_idempotency_ttl_seven_days(self) -> None:
        self.assertEqual(idempotency_ttl_seconds(), 7 * 24 * 3600)

    def test_different_request_ids_create_two_jobs(self) -> None:
        r1 = self._post_generate(request_id=str(uuid.uuid4())).get_json()
        r2 = self._post_generate(request_id=str(uuid.uuid4())).get_json()
        self.assertNotEqual(r1["jobId"], r2["jobId"])
        self.assertEqual(self._count_jobs(), 2)

    def test_different_owners_same_request_id_isolated(self) -> None:
        rid = str(uuid.uuid4())
        a = self._post_generate(request_id=rid, batch="owner-a").get_json()
        b = self._post_generate(request_id=rid, batch="owner-b").get_json()
        self.assertNotEqual(a["jobId"], b["jobId"])
        self.assertEqual(self._count_jobs(), 2)

    def test_legacy_without_request_id_still_works(self) -> None:
        with patch("app.redis_configured", return_value=True):
            resp = self.client.post(
                "/api/generate-video",
                json=_payload(),
                headers={"X-ACE-Batch-State": "legacy-batch"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("ok"))
        self.assertNotIn("idempotentReplay", resp.get_json())


class TestGenerateVideoNextUnchanged(Builder2IdempotencyTestCase):
    def test_video_two_atomic_reservation_preserved(self) -> None:
        from engine.builder2_video_allowance import request_generate_video_next

        rid = str(uuid.uuid4())
        first = self._post_generate(request_id=rid, payload=_payload(targetVideoCount=2)).get_json()
        job1 = first["jobId"]
        allowance_id = first["videoAllowanceId"]
        from engine.video_jobs_redis import set_memory_job_hash

        set_memory_job_hash(
            job1,
            {
                "status": "done",
                "video_url": "https://example.test/v1.mp4",
                "product_description": "A valid product description.",
                **ownership_fields_for_job_create(
                    MagicMock(headers=_headers()),
                    _payload(),
                ),
                "videoAllowanceId": allowance_id,
                "videoIndex": "1",
            },
        )
        with patch("engine.builder2_video_allowance.video_job_create"):
            next_one = request_generate_video_next(
                video_allowance_id=allowance_id,
                request=MagicMock(headers=_headers()),
                public_base_url="https://example.test",
            )
            next_two = request_generate_video_next(
                video_allowance_id=allowance_id,
                request=MagicMock(headers=_headers()),
                public_base_url="https://example.test",
            )
        self.assertTrue(next_one.get("ok"))
        self.assertTrue(next_two.get("ok"))
        self.assertEqual(next_one["jobId"], next_two["jobId"])


class TestBuilder2CancelOwnership(Builder2IdempotencyTestCase):
    def _owned_job(self, job_id: str, *, batch: str = "batch-cancel-owner") -> None:
        from engine.video_jobs_redis import set_memory_job_hash

        fields = ownership_fields_for_job_create(
            MagicMock(headers=_headers(batch=batch)),
            _payload(),
        )
        set_memory_job_hash(
            job_id,
            {
                "status": "running",
                "product_name": "Product",
                "product_description": "desc",
                "public_base_url": "https://example.test",
                **fields,
            },
        )

    def test_owner_can_cancel(self) -> None:
        job_id = "job-owner-cancel"
        self._owned_job(job_id)
        with patch("app.redis_configured", return_value=True):
            resp = self.client.post(
                f"/api/builder2/jobs/{job_id}/cancel",
                headers=_headers(batch="batch-cancel-owner"),
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("outcome"), "cancelled")

    def test_wrong_owner_cannot_cancel(self) -> None:
        job_id = "job-other-cancel"
        self._owned_job(job_id, batch="batch-cancel-owner")
        with patch("app.redis_configured", return_value=True):
            resp = self.client.post(
                f"/api/builder2-jobs/{job_id}/cancel",
                headers=_headers(batch="batch-other"),
            )
        self.assertEqual(resp.status_code, 403)

    def test_both_cancel_aliases_protected(self) -> None:
        job_id = "job-alias-cancel"
        self._owned_job(job_id)
        with patch("app.redis_configured", return_value=True):
            r1 = self.client.post(
                f"/api/builder2/jobs/{job_id}/cancel",
                headers=_headers(batch="batch-other"),
            )
            r2 = self.client.post(
                f"/api/builder2-jobs/{job_id}/cancel",
                headers=_headers(batch="batch-other"),
            )
        self.assertEqual(r1.status_code, 403)
        self.assertEqual(r2.status_code, 403)

    def test_authorized_cancel_still_marks_cancelled(self) -> None:
        job_id = "job-cancel-state"
        self._owned_job(job_id)
        with patch("app.redis_configured", return_value=True):
            self.client.post(
                f"/api/builder2/jobs/{job_id}/cancel",
                headers=_headers(batch="batch-cancel-owner"),
            )
        raw = video_job_get_raw(job_id) or {}
        self.assertEqual(raw.get("status"), "cancelled")
        self.assertEqual(raw.get("cancelReason"), "frontend_refresh")


if __name__ == "__main__":
    unittest.main()
