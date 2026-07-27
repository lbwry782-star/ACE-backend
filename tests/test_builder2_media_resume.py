"""
Builder2 media-only resume tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_media_pipeline import MediaPipelineCounters, MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_resume import collect_media_resume_missing_paths, run_one_media_resume
from engine.builder2_media_resume_guard import MEDIA_RESUME_ISOLATION_ERROR, MediaResumeIsolationGuard
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import persist_winner_development_atomically
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_reasoning_resume import _candidate_id_for_prototype
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_resume import _historical_judged_state


HISTORICAL_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"
HISTORICAL_CANDIDATE_ID = "cand-1-summer_fan-1-57f415ca"


def _mock_start_image_data_uri() -> str:
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (1280, 720), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _mock_render_advertising_closure(**kwargs: Any) -> tuple[Dict[str, Any], Any]:
    from engine.builder2_advertising_closure_pipeline import AdvertisingClosureRenderCounters

    state = kwargs["state"]
    media = state.setdefault("mediaResume", {})
    final_url = "https://example.com/final-with-closure.mp4"
    media["finalVideoWithClosureUrl"] = final_url
    media["finalPublicUrl"] = final_url
    media["finalVideoPath"] = final_url
    media["advertisingClosureStatus"] = "completed"
    state["advertisingClosureStatus"] = "completed"
    return state, AdvertisingClosureRenderCounters(closure_ffmpeg_calls=1)


def _winner_plan_for_media(*, headline_decision: str = "omit") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    candidate["verbalPotential"] = {
        "decision": "not_needed",
        "reason": "The visible fan behavior communicates absence without a headline.",
    }
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision=headline_decision, winning_candidate=candidate, strategy=strategy))
    for key in ("prototypeId", "structureType", "visualParallelType", "coreCreativeMechanism"):
        if candidate.get(key) is not None:
            plan[key] = candidate[key]
    if isinstance(plan.get("preservationReference"), dict):
        plan["preservationReference"].update(
            {
                "prototypeId": candidate.get("prototypeId"),
                "structureType": candidate.get("structureType"),
                "visualParallelType": candidate.get("visualParallelType"),
                "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
            }
        )
    plan["headline"] = ""
    plan["headlineCoreKeyword"] = ""
    plan["productNameResolved"] = "ACE Product"
    plan["language"] = "he"
    plan["planInferenceMode"] = "builder2_tournament_winner_v1"
    if isinstance(plan.get("preservationReference"), dict):
        plan["preservationReference"]["strategyFoundationId"] = strategy.get("strategyFoundationId")
    return plan


def _media_ready_state(*, job_id: str = HISTORICAL_JOB_ID) -> Dict[str, Any]:
    from engine.builder2_winner_preservation_contract import (
        build_server_owned_winner_source_reference,
        build_winning_candidate_preservation_snapshot,
        process_winner_development_response,
    )

    state = _historical_judged_state(job_id=job_id)
    candidate_id = _candidate_id_for_prototype("summer_fan")
    if candidate_id not in state["candidates"]:
        for cid, rec in (state.get("candidates") or {}).items():
            if rec.get("prototypeId") == "summer_fan":
                candidate_id = cid
                break
    winner_rec = state["candidates"][candidate_id]
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or _candidate("summer_fan")
    strategy = state["strategyFoundation"] or _strategy(language="he")
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    raw_plan = _winner_plan_for_media(headline_decision="omit")
    winner_plan = process_winner_development_response(
        raw_plan,
        source_reference=source,
        winning_candidate=winning_candidate,
        preservation_snapshot=snapshot,
        winning_judgment={},
    )
    persist_winner_development_atomically(
        state,
        candidate_id=candidate_id,
        prototype_id="summer_fan",
        winner_plan=winner_plan,
        winning_candidate=winning_candidate,
        preservation_snapshot=snapshot,
    )
    state["winnerCandidateId"] = candidate_id
    state["winnerDevelopmentCandidateId"] = candidate_id
    state["winnerDevelopmentPrototypeId"] = "summer_fan"
    state["mediaContinuationRequired"] = True
    state["productName"] = "ACE Product"
    state["productDescription"] = "Product description"
    state["contentLanguage"] = "he"
    closure = (state.get("winnerDevelopmentPlan") or {}).get("advertisingClosure")
    if isinstance(closure, dict):
        state["advertisingClosure"] = dict(closure)
        state["advertisingClosureStatus"] = "approved"
        state["advertisingClosureSource"] = "winner_creator_candidate"
    return state


def _mock_pipeline_deps() -> MediaPipelineDeps:
    return MediaPipelineDeps(
        generate_start_image=lambda plan: _mock_start_image_data_uri(),
        submit_runway_task=lambda **kwargs: "task-mock-1",
        poll_runway_task=lambda **kwargs: ("SUCCEEDED", "https://runway/mock.mp4"),
        postprocess_video=lambda **kwargs: kwargs["runway_url"],
        compose_marketing_copy=lambda **kwargs: "Marketing copy",
    )


class TestMediaResumePrerequisites(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_persisted_winner_is_loaded(self) -> None:
        state = _media_ready_state(job_id="job-media-load")
        save_tournament_state("job-media-load", state)
        with patch.dict(os.environ, {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False):
            report = run_one_media_resume(job_id="job-media-load", dry_run=True)
        self.assertTrue(report["winnerLoaded"])
        self.assertEqual(report["headlineDecision"], "omit")

    def test_missing_winner_fails_before_paid_calls(self) -> None:
        state = _historical_judged_state(job_id="job-media-missing")
        state["mediaContinuationRequired"] = True
        missing = collect_media_resume_missing_paths(state)
        self.assertIn("winnerDevelopmentPlan", missing)
        report = run_one_media_resume(job_id="job-media-missing", tournament_state=state, dry_run=True)
        self.assertFalse(report["ok"])


class TestReasoningIsolation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_strategy_never_enabled(self) -> None:
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.strategy_generation_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            MediaResumeIsolationGuard.assert_reasoning_isolated()
        self.assertIn(MEDIA_RESUME_ISOLATION_ERROR, str(ctx.exception))
        MediaResumeIsolationGuard.end()

    def test_creator_judge_winner_never_enabled(self) -> None:
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.creator_generation_enabled = True
        MediaResumeIsolationGuard.judge_generation_enabled = True
        MediaResumeIsolationGuard.winner_development_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            MediaResumeIsolationGuard.assert_reasoning_isolated()
        self.assertIn("creatorGenerationEnabled", str(ctx.exception))
        MediaResumeIsolationGuard.end()

    def test_dry_run_makes_zero_image_runway_ffmpeg_calls(self) -> None:
        state = _media_ready_state(job_id="job-media-dry")
        with patch.dict(os.environ, {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False):
            report = run_one_media_resume(job_id="job-media-dry", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)


class TestDryRunValidation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_dry_run_validates_omit_headline(self) -> None:
        state = _media_ready_state(job_id="job-media-omit")
        with patch.dict(os.environ, {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False):
            report = run_one_media_resume(job_id="job-media-omit", tournament_state=deepcopy(state), dry_run=True)
        self.assertEqual(report["headlineDecision"], "omit")
        self.assertTrue(report["downstreamValidationAccepted"])

    def test_rich_visual_anchor_is_downstream_safe(self) -> None:
        state = _media_ready_state(job_id="job-media-anchor")
        plan = state["winnerDevelopmentPlan"]
        self.assertIsInstance(plan.get("visualAnchor"), (dict, str))
        with patch.dict(os.environ, {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False):
            report = run_one_media_resume(job_id="job-media-anchor", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["downstreamValidationAccepted"])

    def test_dry_run_reports_start_image_geometry(self) -> None:
        state = _media_ready_state(job_id="job-media-geometry")
        with patch.dict(os.environ, {"BUILDER2_MEDIA_RESUME_DRY_RUN": "true", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False):
            report = run_one_media_resume(job_id="job-media-geometry", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["startImageGeometry"]["imageGenerationSize"], "1536x1024")
        self.assertEqual(report["startImageGeometry"]["startImageOutputSize"], "1280x720")


class TestMediaPipelineIdempotency(unittest.TestCase):
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

    def test_existing_start_image_is_reused(self) -> None:
        state = _media_ready_state(job_id="job-media-start-reuse")
        state["mediaResume"] = {"startImageArtifact": _mock_start_image_data_uri(), "startImageStatus": "completed"}
        counters = MediaPipelineCounters()
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            _, counters = execute_builder2_media_pipeline(
                job_id="job-media-start-reuse",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                dry_run=False,
                deps=_mock_pipeline_deps(),
            )
        self.assertEqual(counters.start_image_calls, 0)

    def test_existing_runway_task_prevents_resubmission(self) -> None:
        state = _media_ready_state(job_id="job-media-runway-reuse")
        state["mediaResume"] = {
            "startImageArtifact": _mock_start_image_data_uri(),
            "runwayTaskId": "task-existing",
            "runwayVideoUrl": "https://runway/existing.mp4",
        }
        state["runway"] = {"taskId": "task-existing", "submissionState": "submitted", "startImageCompleted": True}
        deps = _mock_pipeline_deps()
        deps.submit_runway_task = unittest.mock.Mock(return_value="should-not-run")
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            _, counters = execute_builder2_media_pipeline(
                job_id="job-media-runway-reuse",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        deps.submit_runway_task.assert_not_called()
        self.assertTrue(counters.runway_polling_resumed)

    def test_completed_runway_output_is_reused(self) -> None:
        state = _media_ready_state(job_id="job-media-output-reuse")
        state["mediaResume"] = {
            "startImageArtifact": _mock_start_image_data_uri(),
            "runwayTaskId": "task-existing",
            "runwayVideoUrl": "https://runway/existing.mp4",
            "downloadedVideoPath": "https://runway/existing.mp4",
            "finalPublicUrl": "https://example.com/final-with-closure.mp4",
            "finalVideoWithClosureUrl": "https://example.com/final-with-closure.mp4",
            "mediaResumeStatus": "completed",
        }
        _, counters = execute_builder2_media_pipeline(
            job_id="job-media-output-reuse",
            state=state,
            plan=state["winnerDevelopmentPlan"],
            public_base_url="https://example.com",
            product_description="desc",
            deps=_mock_pipeline_deps(),
        )
        self.assertTrue(counters.media_reused)

    def test_failure_preserves_runway_task_id(self) -> None:
        state = _media_ready_state(job_id="job-media-failure")
        state["mediaResume"] = {"runwayTaskId": "task-preserve", "runwaySubmissionStatus": "submitted"}
        from engine.builder2_media_resume import _persist_media_failure

        _persist_media_failure(state, stage="waiting_for_runway", reason="timeout", task_id="task-preserve", paid_step_submitted=True)
        self.assertEqual(state["mediaResume"]["mediaFailure"]["runwayTaskId"], "task-preserve")
        self.assertIs(state.get("winnerDevelopmentPlan"), state["winnerDevelopmentPlan"])


class TestMediaResumeExecution(unittest.TestCase):
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

    @patch("engine.builder2_media_resume.video_job_mark_done")
    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch("engine.builder2_media_resume.save_tournament_state")
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_success_updates_job_record(self, _save: Any, _redis_cfg: Any, mark_done: Any) -> None:
        state = _media_ready_state(job_id="job-media-success")
        captured: Dict[str, Any] = {}

        def _capture(job_id: str, payload: Dict[str, Any]) -> None:
            captured.update(payload)

        _save.side_effect = _capture
        report = run_one_media_resume(
            job_id="job-media-success",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["finalVideoAvailable"])
        self.assertEqual(captured.get("status"), "completed")

    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_zero_reasoning_calls_on_media_path(self, _redis_cfg: Any) -> None:
        state = _media_ready_state(job_id="job-media-zero-reasoning")
        report = run_one_media_resume(
            job_id="job-media-zero-reasoning",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)
        self.assertEqual(report["judgeCalls"], 0)
        self.assertEqual(report["winnerCalls"], 0)
        self.assertEqual(report["marketingCopyCalls"], 0)
        self.assertEqual(report["totalReasoningCalls"], 0)

    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_rerun_reuses_completed_media(self, _redis_cfg: Any) -> None:
        state = _media_ready_state(job_id="job-media-rerun")
        first_state = deepcopy(state)
        first = run_one_media_resume(
            job_id="job-media-rerun",
            tournament_state=first_state,
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(first["ok"])
        second = run_one_media_resume(
            job_id="job-media-rerun",
            tournament_state=first_state,
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(second["mediaReused"])
        self.assertEqual(second["startImageCalls"], 0)
        self.assertEqual(second["runwaySubmissionCalls"], 0)


class TestPublicBaseUrlDryRunParity(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_missing_url_makes_dry_run_fail(self) -> None:
        state = _media_ready_state(job_id="job-media-dry-missing-url")
        with patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=True):
            report = run_one_media_resume(job_id="job-media-dry-missing-url", tournament_state=deepcopy(state), dry_run=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "configuration")
        self.assertEqual(report["failureReason"], "builder2_media_resume_not_configured:publicBaseUrl")

    def test_missing_url_makes_actual_run_fail_identically(self) -> None:
        state = _media_ready_state(job_id="job-media-actual-missing-url")
        with patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=True):
            dry = run_one_media_resume(job_id="job-media-actual-missing-url", tournament_state=deepcopy(state), dry_run=True)
            actual = run_one_media_resume(job_id="job-media-actual-missing-url", tournament_state=deepcopy(state), dry_run=False)
        self.assertEqual(dry["failureReason"], actual["failureReason"])
        self.assertEqual(dry["failureStage"], actual["failureStage"])
        self.assertEqual(actual["startImageCalls"], 0)
        self.assertEqual(actual["runwaySubmissionCalls"], 0)

    def test_ace_public_base_url_allows_dry_run_ready(self) -> None:
        state = _media_ready_state(job_id="job-media-ace-url")
        with patch.dict(
            os.environ,
            {
                "RUNWAY_API_KEY": "rk-test",
                "OPENAI_API_KEY": "sk-test",
                "ACE_PUBLIC_BASE_URL": "https://ace-backend-k1p6.onrender.com",
            },
            clear=True,
        ):
            report = run_one_media_resume(job_id="job-media-ace-url", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["publicBaseUrlSource"], "ACE_PUBLIC_BASE_URL")

    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_valid_url_enters_start_image_stage(self, _redis_cfg: Any) -> None:
        state = _media_ready_state(job_id="job-media-start-stage")
        deps = _mock_pipeline_deps()
        with patch(
            "engine.builder2_advertising_closure_pipeline.render_advertising_closure_for_state",
            side_effect=_mock_render_advertising_closure,
        ):
            report = run_one_media_resume(
                job_id="job-media-start-stage",
                tournament_state=deepcopy(state),
                dry_run=False,
                pipeline_deps=deps,
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["startImageCalls"], 1)
        self.assertEqual(report["startImageNormalCalls"], 1)
        self.assertEqual(report["startImageGeneratedCount"], 1)
        self.assertEqual(report["ffmpegCalls"], 1)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
