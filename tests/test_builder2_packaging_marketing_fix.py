"""
Builder2 packaging marketing copy fix — placeholder cleanup, Redis description, telemetry.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_packaging_marketing_text import (
    count_packaging_marketing_words,
    ensure_builder2_packaging_marketing_text,
    has_builder2_packaging_placeholder_residue,
    sanitize_builder2_packaging_marketing_text,
)
from engine.builder2_product_description_resolve import resolve_builder2_product_description_for_packaging
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash
from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps

URI_DESCRIPTION = (
    "סוכן פרסום דיגיטלי. המשתמש מזין את שם המוצר ואת תיאור המוצר ומקבל פרסומת למוצר."
)

URI_PRODUCTION_SUFFIX_COPY = (
    "כשיש לך רעיון ברור, אין סיבה שיישאר בגדר ניסוח. אורי לב מעניק לתיאור שלך נוכחות מקצועית, "
    "מדויקת ובטוחה, עם תוצאה שמרגישה מוכנה לחשיפה מהרגע הראשון. בלי פער בין מה שדמיינת לבין מה "
    "שהקהל מקבל, רק ביטוי שלם, חד ומשכנע, שמביא את הכוונה שלך היישר אל מרכז תשומת הלב (המוצר הזה)."
)


def _fifty_word_hebrew_copy() -> str:
    return " ".join(f"מילה{i}" for i in range(1, 51))


def _fifty_word_packaging_copy(*_args: Any, **_kwargs: Any) -> str:
    from engine.builder2_media_resume_guard import MediaResumeIsolationGuard

    MediaResumeIsolationGuard.record_packaging_marketing_copy_call(call_type="normal")
    return " ".join(f"word{i}" for i in range(1, 51))


class TestProductDescriptionResolve(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()

    def tearDown(self) -> None:
        disable_memory_jobs()

    def test_redis_fallback_when_state_missing_description(self) -> None:
        job_id = "job-uri-desc"
        set_memory_job_hash(
            job_id,
            {
                "status": "running",
                "product_description": URI_DESCRIPTION,
                "product_name": "אורי לב",
            },
        )
        state = {"jobId": job_id, "productNameResolved": "אורי לב"}
        resolved = resolve_builder2_product_description_for_packaging(job_id=job_id, state=state)
        self.assertEqual(resolved, URI_DESCRIPTION)


class TestPlaceholderSanitation(unittest.TestCase):
    def test_removes_production_placeholder_without_gpt(self) -> None:
        cleaned = sanitize_builder2_packaging_marketing_text(URI_PRODUCTION_SUFFIX_COPY)
        self.assertNotIn("המוצר הזה", cleaned)
        self.assertNotIn("(המוצר הזה)", cleaned)
        self.assertGreaterEqual(count_packaging_marketing_words(cleaned), 45)
        self.assertLessEqual(count_packaging_marketing_words(cleaned), 55)

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_legacy_placeholder_cleanup_no_gpt(self, mock_generate: Any) -> None:
        mock_generate.side_effect = AssertionError("GPT must not run for sanitizable placeholder copy")
        text, source = ensure_builder2_packaging_marketing_text(
            existing_text=URI_PRODUCTION_SUFFIX_COPY,
            existing_source="packaging_copy",
            product_name="אורי לב",
            product_description=URI_DESCRIPTION,
            plan={"language": "he"},
        )
        mock_generate.assert_not_called()
        self.assertEqual(source, "delivery_sanitized")
        self.assertFalse(has_builder2_packaging_placeholder_residue(text))

    def test_legitimate_parentheses_retained(self) -> None:
        copy = (
            "אורי לב מציע לך ניסוח (בדיוק כמו שדמיינת) שמכבד את הרעיון שלך ומעניק לו נוכחות "
            "מקצועית, ברורה ומשכנעת לאורך כל המסר, בלי לסטות מהכוונה המקורית שלך וללא הפתעות "
            "מיותרות בדרך אל הקהל שלך היום."
        )
        self.assertFalse(has_builder2_packaging_placeholder_residue(copy))
        self.assertIn("(בדיוק כמו שדמיינת)", copy)
        sanitized = sanitize_builder2_packaging_marketing_text(copy)
        self.assertIn("(בדיוק כמו שדמיינת)", sanitized)


class TestUriLevPackagingGeneration(unittest.TestCase):
    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_generation_receives_redis_description(self, mock_generate: Any) -> None:
        mock_generate.return_value = _fifty_word_hebrew_copy()
        enable_memory_jobs()
        job_id = "job-uri-packaging"
        try:
            set_memory_job_hash(job_id, {"status": "running", "product_description": URI_DESCRIPTION})
            state = {"jobId": job_id}
            ensure_builder2_packaging_marketing_text(
                existing_text="Product. Video delivery.",
                existing_source="deterministic_fallback",
                product_name="אורי לב",
                product_description=resolve_builder2_product_description_for_packaging(job_id=job_id, state=state),
                plan={"language": "he", "advertisingPromise": "הבטחה ברורה."},
            )
        finally:
            disable_memory_jobs()
        mock_generate.assert_called_once()
        _args, kwargs = mock_generate.call_args
        self.assertEqual(_args[0], "אורי לב")
        self.assertEqual(_args[1], URI_DESCRIPTION)
        self.assertNotIn("המוצר הזה", mock_generate.return_value)


class TestMediaResumeTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.capability_patch, self.closure_patch, self.publish_patch, self.publish_mock = (
            patch_media_pipeline_durable_finalization()
        )
        self.capability_patch.start()
        self.closure_patch.start()
        self.publish_patch.start()

    def tearDown(self) -> None:
        self.publish_patch.stop()
        self.closure_patch.stop()
        self.capability_patch.stop()
        disable_memory_store()
        MediaResumeIsolationGuard.end()

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy", side_effect=_fifty_word_packaging_copy)
    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(
        os.environ,
        {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"},
        clear=False,
    )
    def test_packaging_call_counted(self, _redis: Any, _packaging: Any) -> None:
        state = _media_ready_state(job_id="job-telemetry-one")
        report = run_one_media_resume(
            job_id="job-telemetry-one",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["marketingCopyCalls"], 1)
        self.assertEqual(
            report["totalOpenAIDispatchesThisRun"],
            int(report.get("marketingCopyCalls") or 0) + int(report.get("startImageNormalCalls") or 0),
        )
        self.assertEqual(report["totalReasoningCalls"], 0)

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(
        os.environ,
        {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"},
        clear=False,
    )
    def test_retry_reports_two_calls(self, _redis: Any, mock_generate: Any) -> None:
        calls = {"n": 0}

        def _side_effect(*_args: Any, **_kwargs: Any) -> str:
            calls["n"] += 1
            MediaResumeIsolationGuard.record_packaging_marketing_copy_call(
                call_type="retry" if calls["n"] > 1 else "normal"
            )
            return _fifty_word_hebrew_copy()

        mock_generate.side_effect = _side_effect
        state = _media_ready_state(job_id="job-telemetry-retry")
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.record_packaging_marketing_copy_call(call_type="normal")
        MediaResumeIsolationGuard.record_packaging_marketing_copy_call(call_type="retry")
        report = MediaResumeIsolationGuard.packaging_report()
        self.assertEqual(report["marketingCopyCalls"], 2)


class TestNoDuplicatePaidCalls(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.capability_patch, self.closure_patch, self.publish_patch, _ = patch_media_pipeline_durable_finalization()
        self.capability_patch.start()
        self.closure_patch.start()
        self.publish_patch.start()

    def tearDown(self) -> None:
        self.publish_patch.stop()
        self.closure_patch.stop()
        self.capability_patch.stop()
        disable_memory_store()

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(
        os.environ,
        {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"},
        clear=False,
    )
    def test_completed_job_zero_packaging_calls(self, _redis: Any, mock_generate: Any) -> None:
        mock_generate.side_effect = AssertionError("packaging must not run on completed reuse")
        state = _media_ready_state(job_id="job-no-dup")
        clean = _fifty_word_hebrew_copy()
        media = state.setdefault("mediaResume", {})
        media["marketingText"] = clean
        media["marketingCopySource"] = "packaging_copy"
        media["mediaResumeStatus"] = "completed"
        media["finalPublicUrl"] = "https://example.com/final.mp4"
        state["status"] = "completed"
        first = run_one_media_resume(job_id="job-no-dup", tournament_state=deepcopy(state), pipeline_deps=_mock_pipeline_deps())
        self.assertTrue(first["mediaReused"])
        self.assertEqual(first["marketingCopyCalls"], 0)

    @patch("engine.runway_video.generate_builder2_packaging_marketing_copy")
    def test_sanitizable_placeholder_zero_gpt(self, mock_generate: Any) -> None:
        mock_generate.side_effect = AssertionError("sanitizable placeholder must not trigger GPT")
        text, source = ensure_builder2_packaging_marketing_text(
            existing_text=URI_PRODUCTION_SUFFIX_COPY,
            existing_source="packaging_copy",
            product_name="אורי לב",
            product_description=URI_DESCRIPTION,
            plan={"language": "he"},
        )
        mock_generate.assert_not_called()
        self.assertEqual(source, "delivery_sanitized")
        self.assertFalse(has_builder2_packaging_placeholder_residue(text))


if __name__ == "__main__":
    unittest.main()
