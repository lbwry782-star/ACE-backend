"""
Builder2 marketing text delivery + ZIP download tests — offline/mocked only.
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_packaging_marketing_text import (
    count_packaging_marketing_words,
    ensure_builder2_packaging_marketing_text,
    is_insufficient_delivery_marketing_text,
)
from engine.builder2_resume_service import build_builder2_status_payload
from engine.builder2_tournament_recovery import disable_memory_recovery, enable_memory_recovery
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_zip import build_builder2_video_zip_bytes
from engine.builder2_zip_download import Builder2ZipVideoFetchError, fetch_builder2_zip_video_bytes
from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash
from tests.test_builder2_media_resume import _media_ready_state


def _fifty_word_copy() -> str:
    return " ".join(f"word{i}" for i in range(1, 51))


class TestPackagingMarketingText(unittest.TestCase):
    def test_reuses_sufficient_existing_text(self) -> None:
        existing = _fifty_word_copy()
        text, source = ensure_builder2_packaging_marketing_text(
            existing_text=existing,
            existing_source="delivery_existing",
            product_name="Product",
            product_description="desc",
            plan={"language": "en"},
        )
        self.assertEqual(text, existing)
        self.assertEqual(source, "delivery_existing")

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_generates_when_deterministic_fallback(self, mock_copy: Any) -> None:
        mock_copy.return_value = _fifty_word_copy()
        text, source = ensure_builder2_packaging_marketing_text(
            existing_text="Product. Video delivery.",
            existing_source="deterministic_fallback",
            product_name="Product",
            product_description="A useful product for daily life.",
            plan={"language": "en", "advertisingPromise": "Clear promise."},
        )
        mock_copy.assert_called_once()
        self.assertEqual(source, "packaging_copy")
        self.assertGreaterEqual(count_packaging_marketing_words(text), 45)

    def test_insufficient_detects_short_fallback(self) -> None:
        self.assertTrue(is_insufficient_delivery_marketing_text("Product. Video delivery.", source="deterministic_fallback"))
        self.assertFalse(is_insufficient_delivery_marketing_text(_fifty_word_copy(), source="packaging_copy"))


class TestStatusPayloadMarketingText(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_jobs()
        disable_memory_store()
        disable_memory_recovery()

    def test_completed_job_exposes_marketing_text(self) -> None:
        job_id = "job-marketing-status"
        copy = _fifty_word_copy()
        set_memory_job_hash(
            job_id,
            {
                "status": "done",
                "video_url": "https://example.test/api/builder2-final-video/" + ("a" * 32),
                "marketing_text": copy,
                "builder": "builder2",
            },
        )
        state = _media_ready_state(job_id=job_id)
        media = state.setdefault("mediaResume", {})
        media["marketingText"] = copy
        save_tournament_state(job_id, state)
        payload = build_builder2_status_payload(job_id)
        self.assertEqual(payload.get("marketingText"), copy)
        self.assertTrue(payload.get("videoUrl"))


class TestBuilder2ZipContents(unittest.TestCase):
    def test_zip_contains_ad_mp4_and_text_txt_only(self) -> None:
        video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        copy = _fifty_word_copy()
        archive = build_builder2_video_zip_bytes(video, copy)
        with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
            self.assertEqual(set(zf.namelist()), {"ad.mp4", "text.txt"})
            self.assertEqual(zf.read("ad.mp4"), video)
            self.assertEqual(zf.read("text.txt").decode("utf-8"), copy)

    def test_local_builder2_final_video_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token = "b" * 32
            root = os.path.join(tmp, "builder2_final")
            os.makedirs(root, exist_ok=True)
            payload = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 32
            path = os.path.join(root, f"{token}.mp4")
            with open(path, "wb") as handle:
                handle.write(payload)
            with patch.dict(os.environ, {"BUILDER2_FINAL_VIDEO_STORAGE_DIR": root}, clear=False):
                from engine.builder2_final_video_store import get_builder2_final_video_path

                self.assertTrue(get_builder2_final_video_path(token) is not None)
                fetched = fetch_builder2_zip_video_bytes(f"https://host/api/builder2-final-video/{token}")
                self.assertEqual(fetched, payload)

    @patch("engine.builder2_zip_download.httpx.Client")
    def test_http_fetch_error_is_controlled(self, client_cls: Any) -> None:
        client = MagicMock()
        client_cls.return_value.__enter__.return_value = client
        response = MagicMock()
        response.status_code = 502
        response.content = b""
        client.get.return_value = response
        with self.assertRaises(Builder2ZipVideoFetchError) as ctx:
            fetch_builder2_zip_video_bytes("https://remote.example/video.mp4")
        self.assertEqual(ctx.exception.code, "video_download_failed")
        self.assertEqual(ctx.exception.http_status, 502)


class TestBuilder2DownloadZipEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        from app import app

        self.app = app
        self.client = app.test_client()

    @patch("engine.builder2_zip_download.fetch_builder2_zip_video_bytes")
    def test_post_builder2_download_zip(self, fetch_mock: Any) -> None:
        fetch_mock.return_value = b"\x00\x00\x00\x18ftypmp42"
        copy = _fifty_word_copy()
        response = self.client.post(
            "/api/builder2-download-zip",
            json={"videoUrl": "https://example.test/api/builder2-final-video/" + ("c" * 32), "marketingText": copy},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.data), "r") as zf:
            self.assertEqual(zf.read("text.txt").decode("utf-8"), copy)
            self.assertTrue(zf.read("ad.mp4"))
        fetch_mock.assert_called_once()

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_zip_endpoint_does_not_generate_marketing_copy(self, mock_copy: Any) -> None:
        mock_copy.side_effect = AssertionError("marketing copy must not run during zip download")
        with patch("engine.builder2_zip_download.fetch_builder2_zip_video_bytes", return_value=b"video"):
            response = self.client.post(
                "/api/builder2-download-zip",
                json={"videoUrl": "https://example.test/v.mp4", "marketingText": "supplied copy"},
            )
        self.assertEqual(response.status_code, 200)


class TestParagraphNotTruncatedMidSentence(unittest.TestCase):
    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_finalize_paragraph_preserves_sentence_boundary(self, mock_copy: Any) -> None:
        sentence = " ".join(["This"] + [f"word{i}" for i in range(2, 46)] + ["works."])
        mock_copy.return_value = sentence
        text, _ = ensure_builder2_packaging_marketing_text(
            existing_text="",
            product_name="Product",
            product_description="desc",
            plan={"language": "en"},
        )
        self.assertTrue(text.endswith("."))
        self.assertGreaterEqual(count_packaging_marketing_words(text), 45)


if __name__ == "__main__":
    unittest.main()
