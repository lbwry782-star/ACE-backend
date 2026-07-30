"""
Builder2 media finalization correction and recovery tests.
"""
from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_advertising_closure_pipeline import render_advertising_closure_for_state
from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    ClosureRenderResult,
    render_builder2_advertising_closure_endcard,
    sanitize_ffmpeg_stderr,
)
from engine.builder2_media_finalization_contract import (
    backfill_legacy_headline_reference,
    finalization_recovery_eligible,
    validate_builder2_media_completion_contract,
)
from engine.builder2_media_finalization_download import SafeDownloadDiagnostics, safe_download_to_path
from engine.builder2_media_finalization_source import (
    SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
    FinalizationSourceDecision,
    resolve_finalization_source_decision,
)
from engine.builder2_local_headline_render import render_builder2_accepted_headline_overlay
from engine.builder2_media_finalization_failure_inspect import inspect_builder2_media_finalization_failure
from engine.builder2_media_finalization_resume import run_finalization_preflight, run_one_media_finalization_resume
from engine.builder2_media_pipeline import execute_builder2_media_pipeline
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    RAW_RUNWAY,
    _false_completion_state,
    _job_raw,
    verified_final_publication_media_fields,
)
from tests.builder2_preflight_test_helpers import patch_accepted_web_storage_capability
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps, _mock_start_image_data_uri
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store


def _valid_closure_result(**overrides: Any) -> ClosureRenderResult:
    return ClosureRenderResult(
        public_url=overrides.get("public_url", CLOSURE_URL),
        local_path=overrides.get("local_path", "/tmp/out.mp4"),
        measured_duration_seconds=overrides.get("measured_duration_seconds", 13.51),
        output_token=overrides.get("output_token", "tok" * 8),
        input_fingerprint=overrides.get("input_fingerprint", "abc"),
    )


def _mock_closure_render_write_output(*args: Any, **kwargs: Any) -> ClosureRenderResult:
    output_path = kwargs.get("output_path")
    measured = 13.51
    if output_path is not None:
        Path(output_path).write_bytes(b"x" * 128)
        return _valid_closure_result(local_path=str(output_path), public_url="", measured_duration_seconds=measured)
    return _valid_closure_result(public_url="", measured_duration_seconds=measured)


def _verified_publication_result(**overrides: Any) -> "FinalVideoPublicationResult":
    from engine.builder2_final_video_publication import FinalVideoPublicationResult

    public_url = overrides.pop("public_url", CLOSURE_URL)
    return FinalVideoPublicationResult(
        public_url=public_url,
        output_token=overrides.pop("output_token", "tok" * 8),
        route_family=overrides.pop("route_family", "api/builder2-final-video"),
        publication_accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        publication_reference_present=True,
        uploaded_byte_count=overrides.pop("uploaded_byte_count", 1028987),
        post_upload_verification_attempted=True,
        post_upload_verification_accepted=True,
        post_upload_http_status_code=200,
        post_upload_content_type="video/mp4",
        post_upload_content_length=1028987,
        artifact_fingerprint_verified=True,
        **overrides,
    )


def _pipeline_outcome_from_render(**kwargs: Any) -> "FinalizationPipelineOutcome":
    from engine.builder2_media_finalization_resume import FinalizationPipelineOutcome

    render_result = _valid_closure_result()
    if "state" in kwargs:
        media = kwargs["state"].setdefault("mediaResume", {})
        media.update(
            {
                "finalVideoWithClosureUrl": CLOSURE_URL,
                "finalPublicUrl": CLOSURE_URL,
                "finalVideoPath": CLOSURE_URL,
                "advertisingClosureRendered": True,
                "actualFinalVideoDurationSeconds": 13.51,
                "advertisingClosureStatus": "completed",
                "headlineReconstructionCompleted": True,
                "headlineArtifactSource": "deterministic_local_reconstruction_from_raw_runway",
                **verified_final_publication_media_fields(),
            }
        )
    if "report" in kwargs:
        kwargs["report"]["headlineFfmpegSubprocessCalls"] = 1
        kwargs["report"]["closureFfmpegSubprocessCalls"] = 1
        kwargs["report"]["totalFfmpegCalls"] = 2
        kwargs["report"]["ffmpegCalls"] = 2
        kwargs["report"]["publicationCalls"] = 1
        kwargs["report"]["postUploadVerificationAccepted"] = True
    return FinalizationPipelineOutcome(
        render_result=render_result,
        publication_result=_verified_publication_result(),
        public_url=CLOSURE_URL,
    )


def _production_shaped_plan_without_headline_text() -> Dict[str, Any]:
    return {
        "schemaVersion": WINNER_PLAN_SCHEMA_VERSION,
        "prototypeId": "forgot",
        "structureType": "continuous_event",
        "headlineDecision": {"decision": "use", "reasonSource": "judge", "reason": "Required."},
        "headline": "what you forgot stays with you",
        "headlineCoreKeyword": "forgot",
        "productNameResolved": "Forgot Product",
        "language": "he",
        "advertisingClosure": {
            "required": True,
            "productNameText": "Forgot Product",
            "sloganText": "SECRET SLOGAN TEXT",
            "language": "he",
        },
    }


def _production_shaped_state_without_headline_text() -> Dict[str, Any]:
    state = _false_completion_state(with_valid_closure=False)
    state["winnerDevelopmentPlan"] = _production_shaped_plan_without_headline_text()
    media = state["mediaResume"]
    media.pop("headlineArtifactUrl", None)
    return state


class TestClosureRenderErrors(unittest.TestCase):
    @patch("engine.builder2_closure_render.requests.get")
    @patch("engine.builder2_closure_render._default_font_path", return_value="/font.ttf")
    @patch("engine.builder2_closure_render._ffmpeg_bin", return_value="/ffmpeg")
    @patch("engine.builder2_closure_render._ffprobe_duration_seconds", return_value=10.0)
    @patch("engine.builder2_closure_render._input_has_audio", return_value=False)
    def test_called_process_error_raises_not_source_fallback(
        self,
        _audio: Any,
        _probe: Any,
        _ffmpeg: Any,
        _font: Any,
        get_req: Any,
    ) -> None:
        get_req.return_value = MagicMock(status_code=200, iter_content=lambda **k: [b"x"])
        get_req.return_value.raise_for_status = MagicMock()
        out = Path(tempfile.gettempdir()) / "closure_error_out.mp4"

        def runner(cmd: list[str], stage: str, category: str) -> None:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"Invalid filter graph")

        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            render_builder2_advertising_closure_endcard(
                "https://example.com/in.mp4",
                product_name="Product",
                slogan="Slogan",
                output_path=out,
                ffmpeg_runner=runner,
            )
        self.assertEqual(ctx.exception.stage, "card_generation")
        self.assertEqual(ctx.exception.return_code, 1)
        self.assertTrue(ctx.exception.stderr_tail)

    def test_sanitize_stderr_redacts_paths(self) -> None:
        text = sanitize_ffmpeg_stderr(b"Error reinitializing filters at /tmp/secret/in.mp4")
        self.assertNotIn("/tmp/secret/in.mp4", text)
        self.assertIn("<path>", text)

    @patch("engine.builder2_closure_render.requests.get")
    @patch("engine.builder2_closure_render._default_font_path", return_value="/font.ttf")
    @patch("engine.builder2_closure_render._ffmpeg_bin", return_value="/ffmpeg")
    @patch("engine.builder2_closure_render._ffprobe_duration_seconds", side_effect=[10.0, 3.5, 13.51])
    @patch("engine.builder2_closure_render._input_has_audio", return_value=False)
    def test_success_returns_distinct_result(
        self,
        _audio: Any,
        _probe: Any,
        _ffmpeg: Any,
        _font: Any,
        get_req: Any,
    ) -> None:
        out = Path(tempfile.gettempdir()) / "closure_success_out.mp4"
        get_req.return_value = MagicMock(status_code=200, iter_content=lambda **k: [b"x"])
        get_req.return_value.raise_for_status = MagicMock()

        def runner(cmd: list[str], stage: str, category: str) -> None:
            if cmd:
                Path(cmd[-1]).write_bytes(b"fake")

        result = render_builder2_advertising_closure_endcard(
            "https://example.com/in.mp4",
            product_name="Product",
            slogan="Slogan",
            output_path=out,
            ffmpeg_runner=runner,
        )
        self.assertEqual(result.public_url, "")
        self.assertEqual(result.local_path, str(out))
        self.assertAlmostEqual(result.measured_duration_seconds, 13.51, places=2)


class TestAdvertisingClosurePipelineSemantics(unittest.TestCase):
    def test_same_url_as_input_is_failure(self) -> None:
        state: Dict[str, Any] = {
            "mediaResume": {
                "rawRunwayVideoUrl": RAW_RUNWAY,
                "downloadedVideoPath": RAW_RUNWAY,
            },
            "advertisingClosure": {"required": True, "productNameText": "P", "sloganText": "S", "language": "he"},
        }
        plan = {"productNameResolved": "P", "headlineDecision": {"decision": "omit"}}

        def bad_render(*args: Any, **kwargs: Any) -> ClosureRenderResult:
            return _valid_closure_result(public_url=RAW_RUNWAY)

        with self.assertRaises(Builder2ClosureRenderError):
            render_advertising_closure_for_state(
                job_id=JOB_ID,
                state=state,
                plan=plan,
                closure=state["advertisingClosure"],
                public_base_url="https://ace.example.com",
                source_video_url=RAW_RUNWAY,
                render_endcard=bad_render,
            )

    def test_success_sets_rendered_and_actual_duration(self) -> None:
        state: Dict[str, Any] = {
            "mediaResume": {
                "rawRunwayVideoUrl": RAW_RUNWAY,
                "downloadedVideoPath": RAW_RUNWAY,
            },
            "advertisingClosure": {"required": True, "productNameText": "P", "sloganText": "S", "language": "he"},
        }
        plan = {"productNameResolved": "P", "headlineDecision": {"decision": "omit"}}

        def good_render(*args: Any, **kwargs: Any) -> ClosureRenderResult:
            return _valid_closure_result()

        updated, counters = render_advertising_closure_for_state(
            job_id=JOB_ID,
            state=state,
            plan=plan,
            closure=state["advertisingClosure"],
            public_base_url="https://ace.example.com",
            source_video_url=RAW_RUNWAY,
            render_endcard=good_render,
        )
        media = updated["mediaResume"]
        self.assertTrue(media["advertisingClosureRendered"])
        self.assertEqual(media["advertisingClosureStatus"], "completed")
        self.assertEqual(media["actualFinalVideoDurationSeconds"], 13.51)
        self.assertEqual(counters.closure_ffmpeg_calls, 1)


class TestMediaPipelineOrdering(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.enable_start_image()
        MediaResumeIsolationGuard.enable_runway()
        MediaResumeIsolationGuard.enable_ffmpeg()

    def tearDown(self) -> None:
        MediaResumeIsolationGuard.end()
        disable_memory_store()

    @patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: None)
    def test_headline_runs_before_closure(self, _patch: Any) -> None:
        from tests.builder2_durable_finalization_test_helpers import (
            accepted_web_storage_capability_result,
            durable_publication_result,
            mock_closure_render_result,
        )

        calls: list[str] = []
        state = _media_ready_state(job_id=JOB_ID)
        plan = state["winnerDevelopmentPlan"]
        state.pop("copyContractVersion", None)
        state.pop("builder2NewFormatVersion", None)
        plan.pop("copyContractVersion", None)
        plan.pop("builder2NewFormatVersion", None)
        plan["headlineDecision"] = {"decision": "use", "reasonSource": "judge", "reason": "Required."}
        plan["headlineText"] = "Headline text"
        plan["headlineTextRemainder"] = "remainder"
        media = state.setdefault("mediaResume", {})
        media.update(
            {
                "startImageArtifact": _mock_start_image_data_uri(),
                "startImageStatus": "completed",
                "runwayTaskId": "task-1",
                "runwayVideoUrl": RAW_RUNWAY,
                "downloadedVideoPath": RAW_RUNWAY,
                "rawRunwayVideoUrl": RAW_RUNWAY,
                "mediaResumeStatus": "running",
                "progressStage": "postprocessing_video",
            }
        )

        def postprocess(**kwargs: Any) -> str:
            calls.append("headline")
            return HEADLINE_URL

        def _capability(*_args: Any, **_kwargs: Any) -> Any:
            calls.append("capability")
            return accepted_web_storage_capability_result()

        def _render(source: str, **kwargs: Any) -> Any:
            calls.append(f"closure:{source}")
            return mock_closure_render_result(source, **kwargs)

        def _publish(*_args: Any, **_kwargs: Any) -> Any:
            calls.append("publish")
            return durable_publication_result(CLOSURE_URL)

        deps = _mock_pipeline_deps()
        deps.postprocess_video = lambda **kwargs: postprocess(**kwargs)
        with patch(
            "engine.builder2_durable_finalization.require_builder2_web_storage_capability",
            side_effect=_capability,
        ), patch(
            "engine.builder2_closure_render.render_builder2_advertising_closure_endcard",
            side_effect=_render,
        ), patch(
            "engine.builder2_durable_finalization.publish_builder2_durable_final_video",
            side_effect=_publish,
        ):
            execute_builder2_media_pipeline(
                job_id=JOB_ID,
                state=state,
                plan=plan,
                public_base_url="https://ace.example.com",
                product_description="desc",
                deps=deps,
            )
        self.assertEqual(calls[0], "headline")
        self.assertEqual(calls[1], "capability")
        self.assertTrue(calls[2].startswith("closure:"))
        self.assertIn(HEADLINE_URL, calls[2])
        self.assertEqual(calls[3], "publish")


class TestCompletionGate(unittest.TestCase):
    def test_headline_only_final_fails_contract(self) -> None:
        state = _false_completion_state(with_valid_closure=False)
        plan = state["winnerDevelopmentPlan"]
        ok, failure, failures = validate_builder2_media_completion_contract(
            state=state,
            plan=plan,
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(ok)
        self.assertTrue(failures)

    @patch("engine.builder2_media_resume._load_and_normalize_winner")
    @patch("engine.builder2_media_resume.redis_configured", return_value=False)
    @patch("engine.builder2_media_resume.save_tournament_state")
    @patch("engine.builder2_media_resume.execute_builder2_media_pipeline")
    @patch("engine.builder2_media_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_resume.load_tournament_state")
    def test_media_resume_blocks_mark_done_on_invalid_contract(
        self,
        load_state: Any,
        build_config: Any,
        pipeline: Any,
        save_state: Any,
        _redis: Any,
        load_winner: Any,
    ) -> None:
        load_winner.side_effect = lambda **kwargs: kwargs["state"]["winnerDevelopmentPlan"]
        state = _false_completion_state(with_valid_closure=False)
        state["mediaContinuationRequired"] = True
        load_state.return_value = deepcopy(state)
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        updated = deepcopy(state)
        media = updated["mediaResume"]
        media["finalPublicUrl"] = HEADLINE_URL
        media["finalVideoWithClosureUrl"] = HEADLINE_URL
        pipeline.return_value = (updated, MagicMock(
            start_image_calls=0,
            start_image_normal_calls=0,
            start_image_repair_calls=0,
            start_image_retry_calls=0,
            start_image_generated_count=0,
            start_image_reused=True,
            runway_submission_calls=0,
            runway_task_created_count=0,
            runway_polling_calls=0,
            runway_polling_resumed=True,
            ffmpeg_calls=2,
            media_reused=False,
        ))
        report = run_one_media_resume(job_id=JOB_ID, dry_run=False)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "finalization_contract")


class TestInspectorRecoveryFlags(unittest.TestCase):
    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_legacy_headline_recovery_flags(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertTrue(report["artifactIdentityGraph"]["jobMarkedDoneViaHeadlineArtifact"])
        self.assertTrue(report["headlineArtifactCanBeReused"])
        self.assertTrue(report["recoveryRequiresFFmpeg"])
        self.assertTrue(report["recoveryRequiresPublication"])


class TestLegacyHeadlineBackfill(unittest.TestCase):
    def test_backfill_from_false_completion_urls(self) -> None:
        state = _false_completion_state(with_valid_closure=False)
        url = backfill_legacy_headline_reference(state, job_video_url=HEADLINE_URL)
        self.assertEqual(url, HEADLINE_URL)
        self.assertEqual(state["mediaResume"]["headlineArtifactUrl"], HEADLINE_URL)


class TestFinalizationPreflight(unittest.TestCase):
    @patch_accepted_web_storage_capability()
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    def test_preflight_no_redis_writes(self, pipeline: Any, build_config: Any, _capability: Any) -> None:
        def _ok(**kwargs: Any) -> None:
            kwargs["report"]["ok"] = True
            kwargs["report"]["readyForFinalizationRecovery"] = True

        pipeline.side_effect = _ok
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _false_completion_state(with_valid_closure=False)
        report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["preflight"])
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["publicationCalls"], 0)


class TestDurationVerification(unittest.TestCase):
    def test_configured_duration_alone_fails_contract(self) -> None:
        state = _false_completion_state(with_valid_closure=False)
        state["mediaResume"]["finalVideoDurationSeconds"] = 13.5
        state["mediaResume"]["advertisingClosureRendered"] = True
        state["mediaResume"]["advertisingClosureStatus"] = "completed"
        ok, failure, failures = validate_builder2_media_completion_contract(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(ok)
        self.assertIn("actual_final_duration_missing", failures)

    def test_measured_duration_required_not_configured(self) -> None:
        from engine.builder2_closure_render import verify_builder2_final_video_duration

        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(10.042)
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_not_longer_than_visual")


class TestNoHeadlinePipelineOrdering(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.enable_start_image()
        MediaResumeIsolationGuard.enable_runway()
        MediaResumeIsolationGuard.enable_ffmpeg()

    def tearDown(self) -> None:
        MediaResumeIsolationGuard.end()
        disable_memory_store()

    @patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: None)
    def test_raw_runs_before_closure_without_headline(self, _patch: Any) -> None:
        from tests.builder2_durable_finalization_test_helpers import (
            accepted_web_storage_capability_result,
            durable_publication_result,
            mock_closure_render_result,
        )

        calls: list[str] = []
        state = _media_ready_state(job_id=JOB_ID)
        plan = state["winnerDevelopmentPlan"]
        media = state.setdefault("mediaResume", {})
        media.update(
            {
                "startImageArtifact": _mock_start_image_data_uri(),
                "startImageStatus": "completed",
                "runwayTaskId": "task-1",
                "runwayVideoUrl": RAW_RUNWAY,
                "downloadedVideoPath": RAW_RUNWAY,
                "rawRunwayVideoUrl": RAW_RUNWAY,
                "mediaResumeStatus": "running",
            }
        )

        def _capability(*_args: Any, **_kwargs: Any) -> Any:
            calls.append("capability")
            return accepted_web_storage_capability_result()

        def _render(source: str, **kwargs: Any) -> Any:
            calls.append(f"closure:{source}")
            return mock_closure_render_result(source, **kwargs)

        def _publish(*_args: Any, **_kwargs: Any) -> Any:
            calls.append("publish")
            return durable_publication_result(CLOSURE_URL)

        deps = _mock_pipeline_deps()
        deps.postprocess_video = lambda **kwargs: (_ for _ in ()).throw(AssertionError("headline must not run"))
        with patch(
            "engine.builder2_durable_finalization.require_builder2_web_storage_capability",
            side_effect=_capability,
        ), patch(
            "engine.builder2_closure_render.render_builder2_advertising_closure_endcard",
            side_effect=_render,
        ), patch(
            "engine.builder2_durable_finalization.publish_builder2_durable_final_video",
            side_effect=_publish,
        ):
            execute_builder2_media_pipeline(
                job_id=JOB_ID,
                state=state,
                plan=plan,
                public_base_url="https://ace.example.com",
                product_description="desc",
                deps=deps,
            )
        self.assertEqual(calls[0], "capability")
        self.assertIn(RAW_RUNWAY, calls[1])
        self.assertEqual(calls[2], "publish")


class TestClosureDiagnosticsSafety(unittest.TestCase):
    def test_failure_metadata_excludes_creative_text(self) -> None:
        state: Dict[str, Any] = {
            "mediaResume": {"rawRunwayVideoUrl": RAW_RUNWAY},
            "advertisingClosure": {
                "required": True,
                "productNameText": "SECRET PRODUCT",
                "sloganText": "SECRET SLOGAN",
                "language": "he",
            },
        }
        plan = state["winnerDevelopmentPlan"] = {
            "productNameResolved": "SECRET PRODUCT",
            "headlineDecision": {"decision": "omit"},
        }

        def bad_render(*args: Any, **kwargs: Any) -> ClosureRenderResult:
            raise Builder2ClosureRenderError(
                "builder2_closure_ffmpeg_failed",
                stage="concatenation",
                return_code=1,
                stderr_tail="filter graph invalid",
                command_category="ffmpeg_concat",
            )

        with self.assertRaises(Builder2ClosureRenderError):
            render_advertising_closure_for_state(
                job_id=JOB_ID,
                state=state,
                plan=plan,
                closure=state["advertisingClosure"],
                public_base_url="https://ace.example.com",
                source_video_url=RAW_RUNWAY,
                render_endcard=bad_render,
            )
        failure = state["mediaResume"]["advertisingClosureFailure"]
        self.assertNotIn("SECRET PRODUCT", str(failure))
        self.assertNotIn("SECRET SLOGAN", str(failure))
        self.assertEqual(failure["returnCode"], 1)


class TestFinalizationRecovery(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_media_finalization_resume.video_job_mark_done")
    @patch("engine.builder2_media_finalization_resume.save_tournament_state")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    def test_recovery_zero_openai_runway_image(
        self,
        _release: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
        save_state: Any,
        mark_done: Any,
    ) -> None:
        state = deepcopy(_false_completion_state(with_valid_closure=False))
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))

        def _render(**kwargs: Any):
            return _pipeline_outcome_from_render(**kwargs)

        pipeline.side_effect = _render
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["runwayPollingCalls"], 0)
        self.assertEqual(report["totalFfmpegCalls"], 2)
        self.assertEqual(report["publicationCalls"], 1)
        mark_done.assert_called_once()
        save_state.assert_called()

    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    def test_recovery_idempotent_when_valid_closure_exists(
        self,
        job_get_raw: Any,
        _release: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
    ) -> None:
        state = _false_completion_state(with_valid_closure=True)
        state["mediaResume"]["advertisingClosureRendered"] = True
        state["mediaResume"]["actualFinalVideoDurationSeconds"] = 13.51
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=CLOSURE_URL)
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertTrue(report["ok"])
        self.assertTrue(report["finalizationReused"])
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertEqual(report["publicationCalls"], 0)

    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=False)
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    def test_concurrent_recovery_blocked_by_lease(
        self,
        job_get_raw: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
    ) -> None:
        state = _false_completion_state(with_valid_closure=False)
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "lease")

    @patch("engine.builder2_media_finalization_resume.save_tournament_state")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    @patch("engine.builder2_media_finalization_resume.video_job_mark_done")
    def test_failed_recovery_remains_resumable(
        self,
        mark_done: Any,
        _release: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
        save_state: Any,
    ) -> None:
        state = deepcopy(_false_completion_state(with_valid_closure=False))
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))

        def _fail(**kwargs: Any) -> None:
            kwargs["report"]["failureStage"] = "concatenation"
            kwargs["report"]["failureReason"] = "builder2_closure_ffmpeg_failed"
            kwargs["report"]["originalFailureStage"] = "concatenation"
            kwargs["report"]["originalFailureCode"] = "builder2_closure_ffmpeg_failed"
            kwargs["report"]["originalFailureClass"] = "Builder2ClosureRenderError"

        pipeline.side_effect = _fail
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "concatenation")
        mark_done.assert_not_called()
        saved_state = save_state.call_args[0][1]
        self.assertTrue(saved_state.get("mediaContinuationRequired"))
        self.assertEqual(saved_state["mediaResume"]["advertisingClosureStatus"], "failed")
        self.assertEqual(saved_state["status"], "media_finalization_incomplete")
        eligible, missing = finalization_recovery_eligible(
            state=saved_state,
            plan=saved_state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(eligible, msg=f"expected eligible after failure, missing={missing}")
        self.assertEqual(saved_state["mediaResume"].get("finalizationFailureStage"), "concatenation")


class TestPreflightSynthetic(unittest.TestCase):
    @patch_accepted_web_storage_capability()
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.render_builder2_accepted_headline_overlay")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    def test_preflight_validates_10s_headline_to_12s_final(
        self,
        source_decision: Any,
        headline_render: Any,
        closure_render: Any,
        build_config: Any,
        _capability: Any,
    ) -> None:
        raw_path = Path(tempfile.gettempdir()) / "raw_preflight.mp4"
        headline_path = Path(tempfile.gettempdir()) / "headline_preflight.mp4"
        source_decision.return_value = FinalizationSourceDecision(
            source_kind=SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
            closure_input_path=raw_path,
            local_headline_render_required=True,
            legacy_headline_download_failed=True,
            legacy_headline_diagnostics=SafeDownloadDiagnostics(
                request_attempted=True,
                http_status_code=404,
                download_failure_class="HTTPError",
                download_failure_category="not_found",
                legacy_headline_artifact_unavailable=True,
            ),
            raw_runway_diagnostics=SafeDownloadDiagnostics(request_attempted=True, download_accepted=True),
        )
        headline_render.return_value = MagicMock(
            output_path=headline_path,
            measured_duration_seconds=10.042,
        )
        closure_render.side_effect = _mock_closure_render_write_output
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _false_completion_state(with_valid_closure=False)
        with patch("engine.builder2_media_finalization_resume._probe_duration", side_effect=[10.042, 10.042, 12.01]):
            report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["ok"])
        self.assertEqual(report["legacyHeadlineDownloadFailureCategory"], "not_found")
        self.assertTrue(report["rawRunwayFallbackAccepted"])
        self.assertTrue(report["localHeadlineRenderAccepted"])
        self.assertAlmostEqual(report["measuredHeadlineDurationSeconds"], 10.042, places=3)
        self.assertAlmostEqual(report["measuredFinalDurationSeconds"], 13.51, places=2)
        self.assertTrue(report["finalDurationAccepted"])
        self.assertEqual(report["headlineFfmpegCalls"], 1)
        self.assertEqual(report["closureFfmpegCalls"], 1)


class TestHeadlineDownloadFallback(unittest.TestCase):
    @patch("engine.builder2_media_finalization_download.requests.get")
    def test_404_classified_as_not_found(self, get_req: Any) -> None:
        response = MagicMock(status_code=404, history=[], url=HEADLINE_URL, headers={"Content-Type": "application/json"})
        response.raise_for_status.side_effect = __import__("requests").HTTPError(response=response)
        get_req.return_value = response
        path = Path(tempfile.gettempdir()) / "headline_404_test.mp4"
        diag = safe_download_to_path(HEADLINE_URL, path, validate_video=False)
        self.assertEqual(diag.download_failure_category, "not_found")
        self.assertTrue(diag.legacy_headline_artifact_unavailable)
        self.assertNotIn("abc123", json.dumps(diag.to_report_dict()))

    @patch("engine.builder2_media_finalization_download.requests.get")
    def test_410_classified_as_expired_or_gone(self, get_req: Any) -> None:
        response = MagicMock(status_code=410, history=[], url=HEADLINE_URL, headers={})
        response.raise_for_status.side_effect = __import__("requests").HTTPError(response=response)
        get_req.return_value = response
        path = Path(tempfile.gettempdir()) / "headline_410_test.mp4"
        diag = safe_download_to_path(HEADLINE_URL, path, validate_video=False)
        self.assertEqual(diag.download_failure_category, "expired_or_gone")

    @patch("engine.builder2_media_finalization_download.requests.get")
    def test_403_classified_as_forbidden(self, get_req: Any) -> None:
        response = MagicMock(status_code=403, history=[], url=HEADLINE_URL, headers={})
        response.raise_for_status.side_effect = __import__("requests").HTTPError(response=response)
        get_req.return_value = response
        path = Path(tempfile.gettempdir()) / "headline_403_test.mp4"
        diag = safe_download_to_path(HEADLINE_URL, path, validate_video=False)
        self.assertEqual(diag.download_failure_category, "forbidden")

    @patch("engine.builder2_media_finalization_source.safe_download_to_path")
    def test_source_selection_falls_back_to_raw_runway(self, download: Any) -> None:
        state = _false_completion_state(with_valid_closure=False)
        plan = state["winnerDevelopmentPlan"]
        work = Path(tempfile.mkdtemp())

        def _side_effect(url: str, path: Path, **kwargs: Any) -> SafeDownloadDiagnostics:
            if HEADLINE_URL in url:
                return SafeDownloadDiagnostics(
                    request_attempted=True,
                    http_status_code=404,
                    download_failure_category="not_found",
                    legacy_headline_artifact_unavailable=True,
                )
            return SafeDownloadDiagnostics(request_attempted=True, download_accepted=True)

        download.side_effect = _side_effect
        decision = resolve_finalization_source_decision(
            state=state,
            plan=plan,
            job_video_url=HEADLINE_URL,
            work_dir=work,
        )
        self.assertEqual(decision.source_kind, SOURCE_RAW_RUNWAY_LOCAL_HEADLINE)
        self.assertTrue(decision.local_headline_render_required)
        self.assertTrue(decision.legacy_headline_download_failed)

    @patch_accepted_web_storage_capability()
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.render_builder2_accepted_headline_overlay")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    def test_preflight_succeeds_when_legacy_headline_download_fails(
        self,
        source_decision: Any,
        headline_render: Any,
        closure_render: Any,
        build_config: Any,
        _capability: Any,
    ) -> None:
        raw_path = Path(tempfile.gettempdir()) / "raw_fallback.mp4"
        headline_path = Path(tempfile.gettempdir()) / "headline_fallback.mp4"
        source_decision.return_value = FinalizationSourceDecision(
            source_kind=SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
            closure_input_path=raw_path,
            local_headline_render_required=True,
            legacy_headline_download_failed=True,
            legacy_headline_diagnostics=SafeDownloadDiagnostics(
                request_attempted=True,
                http_status_code=404,
                download_failure_category="not_found",
                legacy_headline_artifact_unavailable=True,
            ),
            raw_runway_diagnostics=SafeDownloadDiagnostics(request_attempted=True, download_accepted=True),
        )
        headline_render.return_value = MagicMock(output_path=headline_path, measured_duration_seconds=10.042)
        closure_render.side_effect = _mock_closure_render_write_output
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _false_completion_state(with_valid_closure=False)
        with patch("engine.builder2_media_finalization_resume._probe_duration", side_effect=[10.042, 10.042, 12.01]):
            report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["ok"])
        self.assertTrue(report["readyForFinalizationRecovery"])
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["publicationCalls"], 0)

    def test_completion_gate_rejects_inaccessible_headline_only_url(self) -> None:
        state = _false_completion_state(with_valid_closure=False)
        ok, _, failures = validate_builder2_media_completion_contract(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(ok)
        self.assertIn("final_url_is_headline_only", failures)


class TestAcceptedHeadlineResolution(unittest.TestCase):
    def test_production_shaped_plan_has_use_headline_and_keyword_without_headline_text(self) -> None:
        plan = _production_shaped_plan_without_headline_text()
        self.assertEqual(plan["headlineDecision"]["decision"], "use")
        self.assertTrue(plan.get("headline"))
        self.assertTrue(plan.get("headlineCoreKeyword"))
        self.assertFalse(plan.get("headlineText"))

    @patch("engine.builder2_winner_downstream.apply_builder2_headline_composition")
    def test_canonical_resolution_derives_headline_text(self, compose: Any) -> None:
        from engine.builder2_winner_downstream import resolve_accepted_winner_headline_for_media

        def _compose(plan: Dict[str, Any]) -> None:
            plan["headlineText"] = "Forgot Product what you forgot stays with you"
            plan["headlineTextRemainder"] = "what you forgot stays with you"

        compose.side_effect = _compose
        resolution = resolve_accepted_winner_headline_for_media(_production_shaped_plan_without_headline_text())
        self.assertTrue(resolution.canonical_headline_resolution_accepted)
        self.assertEqual(resolution.canonical_headline_source, "derived_from_accepted_winner")
        self.assertGreater(resolution.canonical_headline_character_count, 0)
        compose.assert_called()

    def test_slogan_not_used_as_headline(self) -> None:
        from engine.builder2_winner_downstream import resolve_accepted_winner_headline_for_media

        plan = _production_shaped_plan_without_headline_text()
        plan.pop("headline", None)
        plan.pop("headlineCoreKeyword", None)
        resolution = resolve_accepted_winner_headline_for_media(plan)
        self.assertEqual(resolution.failure_code, "accepted_headline_missing")
        self.assertNotEqual(resolution.failure_code, "canonical_headline_composition_failed")

    def test_product_name_alone_not_accepted_as_headline(self) -> None:
        from engine.builder2_winner_downstream import resolve_accepted_winner_headline_for_media

        plan = _production_shaped_plan_without_headline_text()
        plan["headline"] = "Forgot Product"
        plan["headlineCoreKeyword"] = "forgot"
        resolution = resolve_accepted_winner_headline_for_media(plan)
        self.assertFalse(resolution.canonical_headline_resolution_accepted)

    @patch("engine.builder2_local_headline_render.render_local_video_headline_overlay")
    @patch("engine.video_bidi.prepare_ffmpeg_overlay_headline")
    def test_local_renderer_receives_non_empty_headline(self, overlay_prep: Any, local_render: Any) -> None:
        overlay_prep.return_value = MagicMock(
            text_plain="Forgot Product what you forgot stays with you",
            render_mode="plain_text",
            dual_latin="",
            dual_hebrew="",
        )
        local_render.return_value = MagicMock(
            output_path=Path(tempfile.gettempdir()) / "out.mp4",
            measured_duration_seconds=10.042,
        )
        result = render_builder2_accepted_headline_overlay(
            source_video_path=Path(tempfile.gettempdir()) / "raw.mp4",
            output_path=Path(tempfile.gettempdir()) / "headline.mp4",
            plan=_production_shaped_plan_without_headline_text(),
        )
        self.assertGreater(result.headline_resolution.canonical_headline_character_count, 0)
        self.assertNotEqual(local_render.call_args.kwargs["headline"], "")

    @patch_accepted_web_storage_capability()
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.render_builder2_accepted_headline_overlay")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    def test_preflight_runner_passes_derived_headline(
        self,
        source_decision: Any,
        headline_render: Any,
        closure_render: Any,
        build_config: Any,
        _capability: Any,
    ) -> None:
        raw_path = Path(tempfile.gettempdir()) / "raw_runner.mp4"
        source_decision.return_value = FinalizationSourceDecision(
            source_kind=SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
            closure_input_path=raw_path,
            local_headline_render_required=True,
        )
        headline_render.return_value = MagicMock(
            output_path=Path(tempfile.gettempdir()) / "headline_runner.mp4",
            measured_duration_seconds=10.042,
            headline_resolution=MagicMock(canonical_headline_resolution_accepted=True),
        )
        closure_render.side_effect = _mock_closure_render_write_output
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _production_shaped_state_without_headline_text()
        with patch("engine.builder2_media_finalization_resume._probe_duration", side_effect=[10.042, 10.042, 12.01]):
            report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["ok"])
        plan_arg = headline_render.call_args.kwargs["plan"]
        self.assertEqual(plan_arg, state["winnerDevelopmentPlan"])
        self.assertNotIn("SECRET SLOGAN TEXT", json.dumps(report))

    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    def test_empty_input_validation_does_not_count_ffmpeg_subprocess(self, source_decision: Any) -> None:
        from engine.builder2_media_finalization_resume import _execute_finalization_render_pipeline, _initial_report

        raw_path = Path(tempfile.gettempdir()) / "raw_empty.mp4"
        source_decision.return_value = FinalizationSourceDecision(
            source_kind=SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
            closure_input_path=raw_path,
            local_headline_render_required=True,
            raw_runway_diagnostics=SafeDownloadDiagnostics(request_attempted=True, download_accepted=True),
        )
        state = _production_shaped_state_without_headline_text()
        state["winnerDevelopmentPlan"] = {"headlineDecision": {"decision": "use"}, "productNameResolved": "X"}
        report = _initial_report(job_id=JOB_ID, preflight=True)
        with patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042):
            _execute_finalization_render_pipeline(
                job_id=JOB_ID,
                state=state,
                plan=state["winnerDevelopmentPlan"],
                job_video_url=HEADLINE_URL,
                report=report,
                preflight=True,
                public_base_url="https://ace.example.com",
            )
        self.assertEqual(report["headlineRenderAttempts"], 1)
        self.assertEqual(report["headlineFfmpegSubprocessCalls"], 0)
        self.assertEqual(report["headlineFfmpegCalls"], 0)

    def test_omit_decision_skips_headline_resolution_requirement(self) -> None:
        from engine.builder2_winner_downstream import resolve_accepted_winner_headline_for_media

        plan = _production_shaped_plan_without_headline_text()
        plan["headlineDecision"] = {"decision": "omit"}
        resolution = resolve_accepted_winner_headline_for_media(plan)
        self.assertFalse(resolution.headline_required)
        self.assertEqual(resolution.canonical_headline_source, "omitted_by_decision")

    @patch("engine.builder2_media_finalization_resume.run_finalization_preflight")
    @patch.dict(
        "os.environ",
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_main_cli_preflight_prints_json(self, preflight: Any) -> None:
        from engine.builder2_media_finalization_resume import main

        preflight.return_value = {"jobId": JOB_ID, "ok": True, "preflight": True, "readyForFinalizationRecovery": True}
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main([])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue().strip())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["readyForFinalizationRecovery"])


if __name__ == "__main__":
    unittest.main()
