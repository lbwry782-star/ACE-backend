"""
Builder2 GET /api/video-status ownership enforcement tests — offline/mocked only.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_job_ownership import ownership_fields_for_job_create
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_recovery import disable_memory_recovery, enable_memory_recovery
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash, video_job_get_raw


def _video_job_get_from_memory(job_id: str):
    data = video_job_get_raw(job_id)
    if not data:
        return None
    rp = (data.get("resolved_product_name") or "").strip()
    st = (data.get("status") or "running").strip()
    infra = (data.get("infrastructure_failure") or "").strip() == "1"
    icode = (data.get("interrupt_code") or "").strip()
    return {
        "status": st,
        "videoUrl": data.get("video_url") or "",
        "marketingText": data.get("marketing_text") or "",
        "overlayHeadline": data.get("overlay_headline") or "",
        "publicBaseUrl": data.get("public_base_url") or "",
        "postprocessRan": (data.get("postprocess_ran") or "").strip(),
        "error": data.get("error") or "",
        "interruptCode": icode,
        "infrastructureFailure": infra,
        "resolvedProductName": rp,
        "productNameResolved": rp,
        "productNameSource": (data.get("product_name_source") or "").strip(),
        "productDescription": (data.get("product_description") or "").strip(),
        "progressStage": (data.get("progressStage") or data.get("progress_stage") or "").strip(),
        "progressStartedAt": (data.get("progressStartedAt") or "").strip(),
        "builder": (data.get("builder") or "").strip(),
        "builder2ResumeContractVersion": (data.get("builder2ResumeContractVersion") or "").strip(),
        "ownerContextPresent": (data.get("ownerContextPresent") or "").strip() in {"1", "true", "True"},
        "cancelRequested": str(data.get("cancelRequested") or "").strip().lower() in {"1", "true", "yes"},
        "cancelRequestedAt": (data.get("cancelRequestedAt") or "").strip(),
        "cancelReason": (data.get("cancelReason") or "").strip(),
        "cancelledAt": (data.get("cancelledAt") or "").strip(),
    }


def _mock_request(*, batch_state: str = "", authorization: str = "") -> MagicMock:
    headers: Dict[str, str] = {}
    if batch_state:
        headers["X-ACE-Batch-State"] = batch_state
    if authorization:
        headers["Authorization"] = authorization
    request = MagicMock()
    request.headers = headers
    return request


def _owned_builder2_job_hash(
    job_id: str,
    *,
    batch_state: str = "owner-a",
    status: str = "done",
) -> Dict[str, str]:
    fields = ownership_fields_for_job_create(
        _mock_request(batch_state=batch_state),
        {"productDescription": "Builder2 ownership test product."},
    )
    return {
        "status": status,
        "product_description": "Builder2 ownership test product.",
        "product_name": "Test Product",
        "video_url": "https://example.test/api/builder2-final-video/" + ("a" * 32),
        "marketing_text": "Marketing copy for ownership test.",
        "resolved_product_name": "Test Product Resolved",
        "enqueued_ts": "1716192000",
        **fields,
    }


class TestBuilder2VideoStatusOwnershipRoute(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"
        from app import app

        self.app = app
        self.client = app.test_client()
        self._video_get_patch = patch("app.video_job_get", side_effect=_video_job_get_from_memory)
        self._stale_patch = patch("app.video_job_try_finalize_stale_running", return_value=False)
        self._postprocess_patch = patch("app.ensure_video_postprocessed_for_poll")
        self._video_get_patch.start()
        self._stale_patch.start()
        self._postprocess_patch.start()

    def tearDown(self) -> None:
        self._postprocess_patch.stop()
        self._stale_patch.stop()
        self._video_get_patch.stop()
        disable_memory_recovery()
        disable_memory_store()
        disable_memory_jobs()

    @patch("app.redis_configured", return_value=True)
    def test_matching_owner_returns_completed_payload(self, _redis: Any) -> None:
        job_id = "job-status-owner-match"
        set_memory_job_hash(job_id, _owned_builder2_job_hash(job_id, batch_state="owner-a"))
        response = self.client.get(
            f"/api/video-status?jobId={job_id}",
            headers={"X-ACE-Batch-State": "owner-a"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("status"), "done")
        self.assertTrue(body.get("videoUrl"))
        self.assertTrue(body.get("marketingText"))
        self.assertTrue(body.get("productNameResolved"))
        self.assertTrue(body.get("ownershipVerified"))

    @patch("app.redis_configured", return_value=True)
    def test_mismatched_owner_returns_403_without_sensitive_fields(self, _redis: Any) -> None:
        job_id = "job-status-owner-mismatch"
        set_memory_job_hash(job_id, _owned_builder2_job_hash(job_id, batch_state="owner-a", status="done"))
        response = self.client.get(
            f"/api/video-status?jobId={job_id}",
            headers={"X-ACE-Batch-State": "owner-b"},
        )
        self.assertEqual(response.status_code, 403)
        body = response.get_json()
        self.assertFalse(body.get("ok"))
        self.assertEqual(body.get("error"), "ownership_mismatch")
        self.assertEqual(body.get("jobId"), job_id)
        self.assertNotIn("videoUrl", body)
        self.assertNotIn("finalVideoUrl", body)
        self.assertNotIn("marketingText", body)
        self.assertNotIn("productNameResolved", body)
        self.assertNotIn("product_name_resolved", body)
        self.assertNotIn("status", body)

    @patch("app.redis_configured", return_value=True)
    def test_missing_owner_context_returns_403(self, _redis: Any) -> None:
        job_id = "job-status-owner-missing"
        set_memory_job_hash(job_id, _owned_builder2_job_hash(job_id, batch_state="owner-a", status="done"))
        response = self.client.get(f"/api/video-status?jobId={job_id}")
        self.assertEqual(response.status_code, 403)
        body = response.get_json()
        self.assertEqual(body.get("error"), "ownership_mismatch")
        self.assertNotIn("videoUrl", body)

    @patch("app.redis_configured", return_value=True)
    def test_cross_tenant_owner_a_job_owner_b_request(self, _redis: Any) -> None:
        job_id = "job-status-cross-tenant"
        set_memory_job_hash(job_id, _owned_builder2_job_hash(job_id, batch_state="tenant-a"))
        response = self.client.get(
            f"/api/video-status?jobId={job_id}",
            headers={"X-ACE-Batch-State": "tenant-b"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json().get("error"), "ownership_mismatch")

    @patch("app.redis_configured", return_value=True)
    def test_historical_pre_ownership_job_preserves_compatibility(self, _redis: Any) -> None:
        job_id = "job-status-historical"
        set_memory_job_hash(
            job_id,
            {
                "status": "done",
                "builder": "builder2",
                "builder2ResumeContractVersion": BUILDER2_RESUME_CONTRACT_VERSION,
                "video_url": "https://example.test/video.mp4",
                "marketing_text": "Historical marketing copy.",
                "resolved_product_name": "Historical Product",
            },
        )
        response = self.client.get(f"/api/video-status?jobId={job_id}")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("videoUrl"), "https://example.test/video.mp4")
        self.assertEqual(body.get("marketingText"), "Historical marketing copy.")

    @patch("app.redis_configured", return_value=True)
    def test_running_owned_job_mismatch_does_not_leak_progress(self, _redis: Any) -> None:
        job_id = "job-status-running-protected"
        set_memory_job_hash(job_id, _owned_builder2_job_hash(job_id, batch_state="owner-a", status="running"))
        response = self.client.get(
            f"/api/video-status?jobId={job_id}",
            headers={"X-ACE-Batch-State": "owner-b"},
        )
        self.assertEqual(response.status_code, 403)
        body = response.get_json()
        self.assertNotIn("progressStage", body)
        self.assertNotIn("canResume", body)
        self.assertNotIn("failureReason", body)

    @patch("app.redis_configured", return_value=True)
    def test_builder1_job_unchanged_without_ownership_headers(self, _redis: Any) -> None:
        job_id = "job-status-builder1"
        set_memory_job_hash(
            job_id,
            {
                "status": "done",
                "video_url": "https://example.test/builder1.mp4",
                "marketing_text": "Builder1 marketing.",
                "resolved_product_name": "Builder1 Product",
            },
        )
        response = self.client.get(f"/api/video-status?jobId={job_id}")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("videoUrl"), "https://example.test/builder1.mp4")
        self.assertEqual(body.get("marketingText"), "Builder1 marketing.")
        self.assertNotIn("ownershipVerified", body)


if __name__ == "__main__":
    unittest.main()
