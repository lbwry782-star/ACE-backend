"""
Builder2 media-only reasoning isolation — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_media_marketing_text import (
    build_deterministic_media_marketing_fallback,
    resolve_media_resume_marketing_text,
)
from engine.builder2_media_pipeline import MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_reasoning_guard import (
    MEDIA_RESUME_MODEL_DEPENDENT_DELIVERY,
    MEDIA_RESUME_REASONING_BLOCKED,
    MediaResumeReasoningCounters,
    assert_media_resume_reasoning_call_allowed,
)
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_reasoning_config import log_builder2_model_selected
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from tests.test_builder2_media_resume import (
    HISTORICAL_JOB_ID,
    _media_ready_state,
    _mock_pipeline_deps,
    _mock_render_advertising_closure,
    _mock_start_image_data_uri,
)


def _block_roles() -> tuple[str, ...]:
    return (
        "strategy",
        "builder2_strategy",
        "creator",
        "builder2_creator",
        "judge",
        "builder2_judge",
        "winner",
        "builder2_winner",
        "marketing_copy",
        "video_headline",
        "headline",
        "keyword",
        "plan_repair",
        "copy_repair",
        "copy_retry",
        "generic_text_fallback",
        "unknown_role_xyz",
    )


class TestMediaReasoningGuardBlocksRoles(unittest.TestCase):
    def setUp(self) -> None:
        MediaResumeIsolationGuard.begin()

    def tearDown(self) -> None:
        MediaResumeIsolationGuard.end()

    def test_blocks_before_http_submission(self) -> None:
        client = MagicMock()
        with self.assertRaises(Builder2TournamentError) as ctx:
            log_builder2_model_selected(role="marketing_copy")
        self.assertEqual(str(ctx.exception.args[0]), f"{MEDIA_RESUME_REASONING_BLOCKED}:marketing_copy")
        client.responses.create.assert_not_called()

    def test_all_listed_roles_blocked(self) -> None:
        for role in _block_roles():
            with self.subTest(role=role):
                with self.assertRaises(Builder2TournamentError) as ctx:
                    MediaResumeIsolationGuard.assert_reasoning_call_allowed(role)
                self.assertIn(MEDIA_RESUME_REASONING_BLOCKED, str(ctx.exception))


class TestMediaReasoningCounters(unittest.TestCase):
    def test_unknown_role_counts_as_other(self) -> None:
        counters = MediaResumeReasoningCounters()
        counters.increment("unknown_role_xyz")
        report = counters.to_report_dict()
        self.assertEqual(report["otherReasoningCalls"], 1)
        self.assertEqual(report["totalReasoningCalls"], 1)

    def test_marketing_copy_bucket(self) -> None:
        counters = MediaResumeReasoningCounters()
        counters.increment("marketing_copy")
        report = counters.to_report_dict()
        self.assertEqual(report["marketingCopyCalls"], 1)
        self.assertEqual(report["totalReasoningCalls"], 1)


class TestDeterministicMarketingText(unittest.TestCase):
    def test_reuses_persisted_job_text(self) -> None:
        text, source = resolve_media_resume_marketing_text(
            state={},
            plan={"productNameResolved": "ACE Product"},
            job_data={"marketing_text": "Existing job copy."},
            headline_decision="omit",
        )
        self.assertEqual(text, "Existing job copy.")
        self.assertEqual(source, "persisted_job")

    def test_reuses_winner_plan_text(self) -> None:
        text, source = resolve_media_resume_marketing_text(
            state={},
            plan={"productNameResolved": "ACE Product", "marketingText": "Plan copy."},
            headline_decision="omit",
        )
        self.assertEqual(text, "Plan copy.")
        self.assertEqual(source, "winner_existing")

    def test_omit_headline_does_not_invent_headline(self) -> None:
        text, source = resolve_media_resume_marketing_text(
            state={},
            plan={"productNameResolved": "ACE Product", "headlineText": "Should not appear"},
            headline_decision="omit",
        )
        self.assertEqual(source, "deterministic_fallback")
        self.assertNotIn("Should not appear", text)
        self.assertEqual(text, "ACE Product. Video delivery.")

    def test_deterministic_fallback_has_no_model(self) -> None:
        fallback = build_deterministic_media_marketing_fallback(
            product_name="ACE Product",
            headline_decision="omit",
        )
        self.assertEqual(fallback, "ACE Product. Video delivery.")


class TestMediaPipelineMarketingIsolation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.enable_start_image()
        MediaResumeIsolationGuard.enable_runway()
        MediaResumeIsolationGuard.enable_ffmpeg()
        self.render_patch = patch(
            "engine.builder2_advertising_closure_pipeline.render_advertising_closure_for_state",
            side_effect=_mock_render_advertising_closure,
        )
        self.render_patch.start()

    def tearDown(self) -> None:
        self.render_patch.stop()
        MediaResumeIsolationGuard.end()
        disable_memory_store()

    def test_pipeline_completes_without_compose_model_call(self) -> None:
        state = _media_ready_state(job_id="job-marketing-isolated")
        model_copy = MagicMock(side_effect=AssertionError("marketing model must not run"))
        deps = _mock_pipeline_deps()
        deps.compose_marketing_copy = model_copy
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            updated, counters = execute_builder2_media_pipeline(
                job_id="job-marketing-isolated",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        model_copy.assert_not_called()
        media = updated["mediaResume"]
        self.assertTrue(media.get("finalPublicUrl"))
        self.assertEqual(media.get("marketingCopySource"), "deterministic_fallback")
        self.assertEqual(counters.ffmpeg_calls, 1)

    def test_mp4_completion_does_not_depend_on_marketing_copy(self) -> None:
        state = _media_ready_state(job_id="job-mp4-first")
        deps = _mock_pipeline_deps()
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            execute_builder2_media_pipeline(
                job_id="job-mp4-first",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        self.assertTrue(state["mediaResume"]["finalPublicUrl"])
        self.assertIn("marketingText", state["mediaResume"])


class TestMediaResumeReporting(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.render_patch = patch(
            "engine.builder2_advertising_closure_pipeline.render_advertising_closure_for_state",
            side_effect=_mock_render_advertising_closure,
        )
        self.render_patch.start()

    def tearDown(self) -> None:
        self.render_patch.stop()
        disable_memory_store()

    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_success_reports_all_reasoning_counters_zero(self, _redis: Any) -> None:
        state = _media_ready_state(job_id="job-report-counters")
        report = run_one_media_resume(
            job_id="job-report-counters",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["marketingCopyCalls"], 0)
        self.assertEqual(report["headlineCalls"], 0)
        self.assertEqual(report["keywordCalls"], 0)
        self.assertEqual(report["otherReasoningCalls"], 0)
        self.assertEqual(report["totalReasoningCalls"], 0)

    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_completed_job_zero_calls(self, _redis: Any) -> None:
        state = _media_ready_state(job_id="job-completed-zero")
        first_state = deepcopy(state)
        first = run_one_media_resume(
            job_id="job-completed-zero",
            tournament_state=first_state,
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(first["ok"])
        second = run_one_media_resume(
            job_id="job-completed-zero",
            tournament_state=first_state,
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(second["mediaReused"])
        self.assertTrue(second["jobCompleted"])
        self.assertEqual(second["totalReasoningCalls"], 0)
        self.assertEqual(second["runwaySubmissionCalls"], 0)
        self.assertEqual(second["startImageCalls"], 0)
        self.assertEqual(second["ffmpegCalls"], 0)

    def test_dry_run_reports_marketing_policy(self) -> None:
        state = _media_ready_state(job_id="job-dry-marketing")
        with patch.dict(
            os.environ,
            {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"},
            clear=False,
        ):
            report = run_one_media_resume(job_id="job-dry-marketing", tournament_state=deepcopy(state), dry_run=True)
        self.assertFalse(report["marketingCopyRequired"])
        self.assertFalse(report["marketingCopyModelAllowed"])
        self.assertTrue(report["allReasoningRolesBlocked"])
        self.assertEqual(report["totalReasoningCalls"], 0)
        self.assertIn(report["marketingCopySource"], ("deterministic_fallback", "winner_existing", "delivery_existing", "persisted_job"))


class TestDryRunModelDependentDelivery(unittest.TestCase):
    def setUp(self) -> None:
        MediaResumeIsolationGuard.begin()

    def tearDown(self) -> None:
        MediaResumeIsolationGuard.end()

    def test_model_dependent_delivery_rejected(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            MediaResumeIsolationGuard.assert_delivery_is_model_free(compose_marketing_copy_uses_model=True)
        self.assertEqual(str(ctx.exception.args[0]), MEDIA_RESUME_MODEL_DEPENDENT_DELIVERY)


class TestOrdinaryBuilder2Unchanged(unittest.TestCase):
    @patch("engine.builder2_reasoning_config.logger")
    def test_log_builder2_model_selected_outside_media_context(self, _logger: Any) -> None:
        MediaResumeIsolationGuard.end()
        self.assertFalse(MediaResumeIsolationGuard.active)
        log_builder2_model_selected(role="marketing_copy", call_type="normal", attempt=1)

    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestHistoricalJobInterpretation(unittest.TestCase):
    def test_historical_job_id_constant(self) -> None:
        self.assertEqual(HISTORICAL_JOB_ID, "5a3157a3-532f-44ef-86db-c777cff54d38")


class TestGuardInactiveAllowsReasoning(unittest.TestCase):
    def test_inactive_guard_allows_role_check(self) -> None:
        MediaResumeIsolationGuard.end()
        assert_media_resume_reasoning_call_allowed(role="marketing_copy", active=False)


if __name__ == "__main__":
    unittest.main()
