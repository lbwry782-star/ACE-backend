"""
Builder2 video purchase allowance (1 or 2 customer videos) — mocked/offline only.
"""
from __future__ import annotations

import io
import os
import threading
import unittest
import zipfile
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

from engine.builder2_job_ownership import ownership_fields_for_job_create
from engine.builder2_video_allowance import (
    enrich_status_with_allowance,
    parse_target_video_count,
    request_generate_video_next,
    resolve_zip_payload_from_job,
)
from engine.builder2_video_allowance_store import (
    clear_memory_allowance_store_for_tests,
    disable_memory_allowance_store,
    enable_memory_allowance_store,
    get_video_allowance,
    reserve_video_two_slot,
)
from engine.builder2_tournament_recovery import disable_memory_recovery, enable_memory_recovery
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.video_jobs_redis import (
    disable_memory_jobs,
    enable_memory_jobs,
    set_memory_job_hash,
    video_job_get_raw,
)


def _fifty_word_copy(prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(1, 51))


def _mock_request(*, batch_state: str = "batch-allowance-a", authorization: str = "Bearer token-a") -> MagicMock:
    headers = {}
    if batch_state:
        headers["X-ACE-Batch-State"] = batch_state
    if authorization:
        headers["Authorization"] = authorization
    request = MagicMock()
    request.headers = headers
    request.url_root = "https://example.test/"
    return request


def _owned_job_hash(
    job_id: str,
    *,
    batch_state: str = "batch-allowance-a",
    product_name: str = "Product A",
    product_description: str = "Description A",
    status: str = "running",
    video_url: str = "",
    marketing_text: str = "",
    video_allowance_id: str = "",
    video_index: int = 0,
) -> Dict[str, str]:
    fields = ownership_fields_for_job_create(
        _mock_request(batch_state=batch_state),
        {"productDescription": product_description, "productName": product_name},
    )
    out = {
        "status": status,
        "product_name": product_name,
        "product_description": product_description,
        "video_url": video_url,
        "marketing_text": marketing_text,
        "public_base_url": "https://example.test",
        **fields,
    }
    if video_allowance_id:
        out["videoAllowanceId"] = video_allowance_id
    if video_index:
        out["videoIndex"] = str(video_index)
    return out


def _mark_job_done(
    job_id: str,
    *,
    video_url: str,
    marketing_text: str,
    **extra: str,
) -> None:
    existing = video_job_get_raw(job_id) or {}
    existing.update(
        {
            "status": "done",
            "video_url": video_url,
            "marketing_text": marketing_text,
            **extra,
        }
    )
    set_memory_job_hash(job_id, existing)


class Builder2VideoAllowanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()
        enable_memory_allowance_store()
        clear_memory_allowance_store_for_tests()
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
        clear_memory_allowance_store_for_tests()

    def _post_generate(
        self,
        *,
        batch_state: str = "batch-allowance-a",
        target_video_count: Any = None,
        product_name: str = "Product A",
        product_description: str = "Description A",
    ) -> Tuple[Any, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "productName": product_name,
            "productDescription": product_description,
        }
        if target_video_count is not None:
            payload["targetVideoCount"] = target_video_count
        with patch("app.redis_configured", return_value=True), patch(
            "app.video_job_create",
        ) as create_mock:
            response = self.client.post(
                "/api/generate-video",
                json=payload,
                headers={"X-ACE-Batch-State": batch_state, "Authorization": "Bearer token-a"},
            )
            body = response.get_json()
            return create_mock, body

    def _create_allowance_via_generate(
        self,
        *,
        target: int = 2,
        batch_state: str = "batch-allowance-a",
        product_name: str = "Product A",
        product_description: str = "Description A",
    ) -> Tuple[str, str, Any]:
        create_mock, body = self._post_generate(
            target_video_count=target,
            batch_state=batch_state,
            product_name=product_name,
            product_description=product_description,
        )
        self.assertTrue(body.get("ok"))
        allowance_id = body["videoAllowanceId"]
        job_id = body["jobId"]
        create_args = create_mock.call_args
        self.assertEqual(create_args[0][0], job_id)
        extra = create_args[1].get("extra_fields") or {}
        self.assertEqual(extra.get("videoAllowanceId"), allowance_id)
        self.assertEqual(extra.get("videoIndex"), "1")
        set_memory_job_hash(job_id, _owned_job_hash(job_id, batch_state=batch_state, video_allowance_id=allowance_id, video_index=1))
        return allowance_id, job_id, body


class TestTargetVideoCountParsing(unittest.TestCase):
    def test_missing_defaults_to_one(self) -> None:
        self.assertEqual(parse_target_video_count(None), (1, None))

    def test_one_and_two_accepted(self) -> None:
        self.assertEqual(parse_target_video_count(1), (1, None))
        self.assertEqual(parse_target_video_count(2), (2, None))
        self.assertEqual(parse_target_video_count("2"), (2, None))

    def test_invalid_values_rejected(self) -> None:
        for bad in (0, 3, 4, -1, [], {}, True, "x"):
            count, err = parse_target_video_count(bad)
            self.assertEqual(err, "invalid_target_video_count")


class TestInitialGeneration(Builder2VideoAllowanceTestCase):
    def test_initial_generation_creates_allowance_and_links_job(self) -> None:
        allowance_id, job_id, body = self._create_allowance_via_generate(target=2)
        self.assertEqual(body.get("targetVideoCount"), 2)
        self.assertEqual(body.get("videoIndex"), 1)
        stored = get_video_allowance(allowance_id)
        assert stored is not None
        self.assertEqual(stored["targetVideoCount"], 2)
        self.assertEqual(stored["productName"], "Product A")
        self.assertEqual(stored["productDescription"], "Description A")
        self.assertEqual(stored["videos"][0]["jobId"], job_id)

    def test_invalid_target_count_rejected(self) -> None:
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video",
                json={"productDescription": "desc", "targetVideoCount": 3},
                headers={"X-ACE-Batch-State": "batch-allowance-a"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json().get("error"), "invalid_target_video_count")


class TestDerivedAllowanceState(Builder2VideoAllowanceTestCase):
    def test_target_one_consumed_after_video_one_done(self) -> None:
        allowance_id, job_id, _ = self._create_allowance_via_generate(target=1)
        _mark_job_done(job_id, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        payload = enrich_status_with_allowance(job_id, video_job_get_raw(job_id), request=_mock_request())
        self.assertEqual(payload["generatedVideoCount"], 1)
        self.assertEqual(payload["remainingVideoCount"], 0)
        self.assertFalse(payload["canGenerateNext"])
        self.assertTrue(payload["consumed"])

    def test_target_two_after_video_one_done_allows_next(self) -> None:
        allowance_id, job_id, _ = self._create_allowance_via_generate(target=2)
        _mark_job_done(job_id, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        payload = enrich_status_with_allowance(job_id, video_job_get_raw(job_id), request=_mock_request())
        self.assertEqual(payload["generatedVideoCount"], 1)
        self.assertEqual(payload["remainingVideoCount"], 1)
        self.assertTrue(payload["canGenerateNext"])
        self.assertFalse(payload["consumed"])

    def test_target_two_consumed_after_both_done(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        with patch("app.redis_configured", return_value=True):
            next_resp = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        job2 = next_resp.get_json()["jobId"]
        if not video_job_get_raw(job2):
            set_memory_job_hash(job2, _owned_job_hash(job2, video_allowance_id=allowance_id, video_index=2))
        _mark_job_done(job2, video_url="https://example.test/v2.mp4", marketing_text=_fifty_word_copy("two"))
        payload = enrich_status_with_allowance(job2, video_job_get_raw(job2), request=_mock_request())
        self.assertEqual(payload["generatedVideoCount"], 2)
        self.assertFalse(payload["canGenerateNext"])
        self.assertTrue(payload["consumed"])
        videos = payload["videos"]
        self.assertEqual(videos[0]["videoIndex"], 1)
        self.assertTrue(videos[0]["finalVideoAvailable"])
        self.assertEqual(videos[1]["videoIndex"], 2)
        self.assertTrue(videos[1]["finalVideoAvailable"])


class TestGenerateVideoNext(Builder2VideoAllowanceTestCase):
    def test_video_two_uses_stored_product_snapshot(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(
            target=2,
            product_name="Frozen Name",
            product_description="Frozen Description",
        )
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        with patch("app.redis_configured", return_value=True), patch("app.video_job_create") as create_mock:
            response = self.client.post(
                "/api/generate-video-next",
                json={
                    "videoAllowanceId": allowance_id,
                    "productName": "Hacker",
                    "productDescription": "Changed",
                },
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json().get("error"), "product_input_not_allowed")

        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("videoIndex"), 2)
        job2 = body["jobId"]
        raw = video_job_get_raw(job2) or {}
        self.assertEqual(raw.get("product_name"), "Frozen Name")
        self.assertEqual(raw.get("product_description"), "Frozen Description")
        self.assertEqual(raw.get("videoIndex"), "2")

    def test_target_one_cannot_generate_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=1)
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json().get("error"), "target_video_count_not_two")

    def test_blocks_before_video_one_success(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json().get("error"), "video_one_not_complete")

    def test_failed_video_one_blocks_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        existing = video_job_get_raw(job1) or {}
        existing["status"] = "error"
        set_memory_job_hash(job1, existing)
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertEqual(response.get_json().get("error"), "video_one_not_complete")

    def test_running_video_one_blocks_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertEqual(response.get_json().get("error"), "video_one_not_complete")

    def test_third_video_never_created(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        with patch("app.redis_configured", return_value=True):
            first = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
            second = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertTrue(first.get_json().get("ok"))
        self.assertTrue(second.get_json().get("ok"))
        stored = get_video_allowance(allowance_id)
        assert stored is not None
        self.assertEqual(len(stored.get("videos") or []), 2)

    def test_concurrent_generate_next_creates_one_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        results: List[str] = []
        barrier = threading.Barrier(2)

        def _worker() -> None:
            barrier.wait()
            out = request_generate_video_next(
                video_allowance_id=allowance_id,
                request=_mock_request(),
                public_base_url="https://example.test",
            )
            if out.get("ok"):
                results.append(out["jobId"])

        with patch("engine.builder2_video_allowance.video_job_create"):
            t1 = threading.Thread(target=_worker)
            t2 = threading.Thread(target=_worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        self.assertEqual(len(set(results)), 1)
        stored = get_video_allowance(allowance_id)
        assert stored is not None
        self.assertEqual(len(stored.get("videos") or []), 2)


class TestOwnerIsolation(Builder2VideoAllowanceTestCase):
    def test_two_users_isolated(self) -> None:
        allowance_a, job_a, _ = self._create_allowance_via_generate(target=2, batch_state="batch-user-a")
        allowance_b, job_b, _ = self._create_allowance_via_generate(target=2, batch_state="batch-user-b")
        self.assertNotEqual(allowance_a, allowance_b)
        _mark_job_done(job_a, video_url="https://example.test/a1.mp4", marketing_text=_fifty_word_copy("a"))
        _mark_job_done(job_b, video_url="https://example.test/b1.mp4", marketing_text=_fifty_word_copy("b"))
        with patch("app.redis_configured", return_value=True):
            denied = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_a},
                headers={"X-ACE-Batch-State": "batch-user-b", "Authorization": "Bearer token-b"},
            )
        self.assertEqual(denied.status_code, 403)

    def test_same_user_two_allowances_isolated(self) -> None:
        allowance_one, job_one, _ = self._create_allowance_via_generate(target=1)
        allowance_two, job_two, _ = self._create_allowance_via_generate(target=2)
        self.assertNotEqual(allowance_one, allowance_two)
        self.assertEqual(get_video_allowance(allowance_one)["targetVideoCount"], 1)
        self.assertEqual(get_video_allowance(allowance_two)["targetVideoCount"], 2)


class TestRetentionAndZip(Builder2VideoAllowanceTestCase):
    def test_video_one_retained_after_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        text1 = _fifty_word_copy("one")
        url1 = "https://example.test/api/builder2-final-video/" + ("a" * 32)
        _mark_job_done(job1, video_url=url1, marketing_text=text1)
        with patch("app.redis_configured", return_value=True):
            next_body = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            ).get_json()
        job2 = next_body["jobId"]
        text2 = _fifty_word_copy("two")
        url2 = "https://example.test/api/builder2-final-video/" + ("b" * 32)
        set_memory_job_hash(job2, _owned_job_hash(job2, video_allowance_id=allowance_id, video_index=2))
        _mark_job_done(job2, video_url=url2, marketing_text=text2)
        payload = enrich_status_with_allowance(job2, video_job_get_raw(job2), request=_mock_request())
        self.assertEqual(payload["videos"][0]["videoUrl"], url1)
        self.assertEqual(payload["videos"][0]["marketingText"], text1)
        self.assertEqual(payload["videos"][1]["videoUrl"], url2)
        self.assertEqual(payload["videos"][1]["marketingText"], text2)

    @patch("engine.builder2_zip_download.fetch_builder2_zip_video_bytes")
    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_zip_download_per_job_without_paid_calls(self, mock_copy: Any, fetch_mock: Any) -> None:
        mock_copy.side_effect = AssertionError("no paid marketing generation during zip")
        fetch_mock.side_effect = lambda url: b"VIDEO:" + url.encode("utf-8")
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        text1 = _fifty_word_copy("one")
        url1 = "https://example.test/v1.mp4"
        _mark_job_done(job1, video_url=url1, marketing_text=text1)
        with patch("app.redis_configured", return_value=True):
            job2 = self.client.post(
                "/api/generate-video-next",
                json={"videoAllowanceId": allowance_id},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            ).get_json()["jobId"]
        text2 = _fifty_word_copy("two")
        url2 = "https://example.test/v2.mp4"
        set_memory_job_hash(job2, _owned_job_hash(job2, video_allowance_id=allowance_id, video_index=2))
        _mark_job_done(job2, video_url=url2, marketing_text=text2)

        zip1 = self.client.post(
            "/api/builder2-download-zip",
            json={"jobId": job1},
            headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
        )
        zip2 = self.client.post(
            "/api/builder2-download-zip",
            json={"jobId": job2},
            headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
        )
        self.assertEqual(zip1.status_code, 200)
        self.assertEqual(zip2.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(zip1.data), "r") as zf1:
            self.assertEqual(zf1.read("text.txt").decode("utf-8"), text1)
            self.assertEqual(zf1.read("ad.mp4"), b"VIDEO:" + url1.encode("utf-8"))
        with zipfile.ZipFile(io.BytesIO(zip2.data), "r") as zf2:
            self.assertEqual(zf2.read("text.txt").decode("utf-8"), text2)
            self.assertEqual(zf2.read("ad.mp4"), b"VIDEO:" + url2.encode("utf-8"))
        self.assertNotEqual(text1, text2)

    def test_wrong_owner_zip_denied(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=1)
        _mark_job_done(job1, video_url="https://example.test/v1.mp4", marketing_text=_fifty_word_copy("one"))
        with patch("app.redis_configured", return_value=True):
            response = self.client.post(
                "/api/builder2-download-zip",
                json={"jobId": job1},
                headers={"X-ACE-Batch-State": "batch-other", "Authorization": "Bearer other"},
            )
        self.assertEqual(response.status_code, 403)


class TestResumeCompatibility(Builder2VideoAllowanceTestCase):
    def test_resume_does_not_create_video_two(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        existing = video_job_get_raw(job1) or {}
        existing["status"] = "interrupted"
        existing["canResume"] = "1"
        set_memory_job_hash(job1, existing)
        with patch("app.redis_configured", return_value=True), patch(
            "engine.builder2_resume_service.request_builder2_resume",
            return_value={"ok": True, "jobId": job1, "status": "queued"},
        ):
            resume = self.client.post(
                "/api/builder2-resume",
                json={"jobId": job1},
                headers={"X-ACE-Batch-State": "batch-allowance-a", "Authorization": "Bearer token-a"},
            )
        self.assertTrue(resume.get_json().get("ok"))
        stored = get_video_allowance(allowance_id)
        assert stored is not None
        self.assertEqual(len(stored.get("videos") or []), 1)


class TestLegacyCompatibility(Builder2VideoAllowanceTestCase):
    def test_legacy_job_without_allowance_metadata(self) -> None:
        job_id = "legacy-job-no-allowance"
        set_memory_job_hash(
            job_id,
            {
                "status": "done",
                "video_url": "https://example.test/legacy.mp4",
                "marketing_text": _fifty_word_copy("legacy"),
                "product_description": "legacy desc",
                "postprocess_ran": "1",
            },
        )
        payload = enrich_status_with_allowance(job_id, video_job_get_raw(job_id), request=_mock_request())
        self.assertEqual(payload, {})

        with patch("app.redis_configured", return_value=True):
            response = self.client.get(f"/api/video-status?jobId={job_id}")
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertNotIn("videoAllowanceId", body)

    @patch("engine.builder2_zip_download.fetch_builder2_zip_video_bytes")
    def test_legacy_zip_contract_still_works(self, fetch_mock: Any) -> None:
        fetch_mock.return_value = b"\x00\x00\x00\x18ftypmp42"
        copy = _fifty_word_copy("legacy")
        response = self.client.post(
            "/api/builder2-download-zip",
            json={"videoUrl": "https://example.test/v.mp4", "marketingText": copy},
        )
        self.assertEqual(response.status_code, 200)


class TestReserveVideoTwoDirect(Builder2VideoAllowanceTestCase):
    def test_reserve_video_two_idempotent(self) -> None:
        allowance_id, job1, _ = self._create_allowance_via_generate(target=2)
        owner = get_video_allowance(allowance_id)["ownerContextRef"]
        first = reserve_video_two_slot(allowance_id, owner_context_ref=owner, job_id="job-two-a")
        second = reserve_video_two_slot(allowance_id, owner_context_ref=owner, job_id="job-two-b")
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertTrue(second.idempotent)
        self.assertEqual(first.job_id, second.job_id)


if __name__ == "__main__":
    unittest.main()
