"""
Builder2 Lyria durable storage + paid-call invariant tests — mock/offline only.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_closure_only_rerender import run_builder2_closure_only_rerender
from engine.builder2_lyria import (
    Builder2LyriaError,
    generate_builder2_music,
    music_artifact_is_valid,
)
from engine.builder2_lyria_artifact import resolve_existing_lyria_audio, resolve_lyria_audio_for_render
from engine.builder2_media_finalization_contract import validate_builder2_media_completion_contract
from engine.builder2_media_pipeline import MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_music_artifact_publication import (
    Builder2MusicPublicationError,
    MusicArtifactPublicationResult,
)
from engine.builder2_resume_contract import sync_builder2_stage_checkpoints_from_state
from engine.builder2_resume_resolver import _infer_resume_stage
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from tests.test_builder2_lyria import (
    _fake_mp3_bytes,
    _music_direction,
    _winner_plan_with_music,
)


def _fake_publish(local_path: Path, public_base_url: str, *, job_id: str = "", output_token: str | None = None) -> MusicArtifactPublicationResult:
    token = output_token or "abcd1234567890123456789012345678"
    base = public_base_url.rstrip("/")
    return MusicArtifactPublicationResult(
        music_artifact_url=f"{base}/api/builder2-music-artifact/{token}",
        output_token=token,
        publication_accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        uploaded_byte_count=int(local_path.stat().st_size),
        stored_byte_count=int(local_path.stat().st_size),
        artifact_fingerprint_verified=True,
    )


def _fake_download(*, music_artifact_url: str, job_id: str, session=None) -> Path:
    from engine.builder2_lyria_config import resolve_builder2_lyria_job_artifact_path

    dest = resolve_builder2_lyria_job_artifact_path(job_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_fake_mp3_bytes() * 20)
    return dest


class TestLyriaDurableStorage(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.tmp = tempfile.mkdtemp()
        self.web_store = Path(self.tmp) / "web_music"
        self.web_store.mkdir(parents=True, exist_ok=True)
        self.probe_patch = patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0)
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_LYRIA_ENABLED": "true",
            "BUILDER2_LYRIA_API_KEY": "test-key",
            "BUILDER2_LYRIA_ARTIFACT_DIR": "",
            "ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret",
        },
        clear=True,
    )
    def test_success_uploads_and_persists_durable_reference(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state: Dict[str, Any] = {"mediaResume": {}}
            plan = {"musicDirection": _music_direction()}

            def _fake_caller(**kwargs: Any) -> bytes:
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    result = generate_builder2_music(
                        job_id="job-durable-1",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                    )
            media = state["mediaResume"]
            self.assertFalse(result.reused)
            self.assertTrue(media["musicArtifactUrl"].startswith("https://ace.example.com/api/builder2-music-artifact/"))
            self.assertEqual(media["musicArtifactToken"], "abcd1234567890123456789012345678")
            self.assertNotIn("test-key", json.dumps(media))

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true", "BUILDER2_LYRIA_API_KEY": "test-key"}, clear=True)
    def test_local_reuse_on_same_worker(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            path = Path(self.tmp) / "job-local" / "soundtrack.mp3"
            path.parent.mkdir(parents=True)
            path.write_bytes(_fake_mp3_bytes() * 20)
            state = {
                "mediaResume": {
                    "musicGenerationStatus": "succeeded",
                    "musicArtifactPath": str(path),
                    "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
                    "musicGenerationAttempt": 1,
                }
            }
            calls = {"count": 0}

            def _fail_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                return b""

            with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                resolved = resolve_existing_lyria_audio(job_id="job-local", state=state)
            self.assertIsNotNone(resolved)
            self.assertTrue(resolved.reused)
            self.assertEqual(calls["count"], 0)

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_durable_download_when_local_missing(self) -> None:
        state = {
            "mediaResume": {
                "musicGenerationStatus": "succeeded",
                "musicArtifactPath": str(Path(self.tmp) / "missing.mp3"),
                "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
                "musicGenerationAttempt": 1,
            }
        }
        with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
            with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
                resolved = resolve_existing_lyria_audio(
                    job_id="job-download",
                    state=state,
                    downloader=_fake_download,
                )
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.recovered_from_durable)
        self.assertTrue(music_artifact_is_valid(resolved.local_path))

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_succeeded_missing_durable_fails_without_paid_call(self) -> None:
        state = {
            "mediaResume": {
                "musicGenerationStatus": "succeeded",
                "musicArtifactPath": "",
                "musicGenerationAttempt": 1,
            }
        }
        with self.assertRaises(Builder2LyriaError) as ctx:
            resolve_existing_lyria_audio(job_id="job-missing", state=state)
        self.assertEqual(str(ctx.exception.args[0]), "builder2_lyria_succeeded_artifact_missing")

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_generating_ambiguous_fails_without_paid_call(self) -> None:
        state = {
            "mediaResume": {
                "musicGenerationStatus": "generating",
                "musicGenerationAttempt": 1,
            }
        }
        with self.assertRaises(Builder2LyriaError) as ctx:
            resolve_existing_lyria_audio(job_id="job-ambiguous", state=state)
        self.assertEqual(str(ctx.exception.args[0]), "builder2_lyria_paid_call_outcome_unknown")
        self.assertEqual(state["mediaResume"]["musicGenerationStatus"], "paid_call_outcome_unknown")

    @patch.dict(
        os.environ,
        {"BUILDER2_LYRIA_ENABLED": "true", "BUILDER2_LYRIA_API_KEY": "test-key"},
        clear=True,
    )
    def test_failed_resume_allows_second_attempt(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state = {
                "mediaResume": {
                    "musicGenerationStatus": "failed",
                    "musicGenerationAttempt": 1,
                }
            }
            plan = {"musicDirection": _music_direction()}
            calls = {"count": 0}

            def _fake_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    generate_builder2_music(
                        job_id="job-retry",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                    )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(state["mediaResume"]["musicGenerationAttempt"], 2)

    @patch.dict(
        os.environ,
        {"BUILDER2_LYRIA_ENABLED": "true", "BUILDER2_LYRIA_API_KEY": "test-key"},
        clear=True,
    )
    def test_no_second_paid_call_after_success(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state = {
                "mediaResume": {
                    "musicGenerationStatus": "succeeded",
                    "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
                    "musicGenerationAttempt": 1,
                }
            }
            plan = {"musicDirection": _music_direction()}
            calls = {"count": 0}

            def _fake_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                    generate_builder2_music(
                        job_id="job-no-second",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                        downloader=_fake_download,
                    )
            self.assertEqual(calls["count"], 0)

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true", "BUILDER2_LYRIA_API_KEY": "test-key"}, clear=True)
    def test_single_invocation_one_api_call(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state: Dict[str, Any] = {"mediaResume": {}}
            plan = {"musicDirection": _music_direction()}
            calls = {"count": 0}

            def _fake_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    with self.assertRaises(Builder2LyriaError):
                        generate_builder2_music(
                            job_id="job-one-call",
                            state=state,
                            plan=plan,
                            public_base_url="https://ace.example.com",
                            api_caller=_fake_caller,
                            publisher=lambda *a, **k: (_ for _ in ()).throw(
                                Builder2MusicPublicationError("builder2_music_artifact_upload_failed")
                            ),
                        )
            self.assertEqual(calls["count"], 1)


class TestResumeResolverMusicStage(unittest.TestCase):
    @patch("engine.builder2_resume_resolver.resolve_complete_ad_resume_stage", return_value="rendering_advertising_closure")
    @patch("engine.builder2_resume_resolver.is_valid_persisted_winner_development", return_value=True)
    @patch("engine.builder2_resume_resolver.is_tournament_ready_for_winner_selection", return_value=True)
    @patch("engine.builder2_resume_resolver.missing_judge_prototype_ids", return_value=[])
    @patch("engine.builder2_resume_resolver.missing_creator_prototype_ids", return_value=[])
    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_generating_music_stage_when_music_incomplete(
        self,
        _complete_ad: Any,
        _creators: Any,
        _judges: Any,
        _ready: Any,
        _winner: Any,
    ) -> None:
        state = {
            "mediaContinuationRequired": True,
            "reasoningComplete": True,
            "strategyFoundation": {"strategyFoundationId": "sf-1"},
            "winnerCandidateId": "cand-1",
            "winnerDevelopmentPlan": {"advertisingClosure": {"sloganText": "Slogan"}},
            "advertisingClosureStatus": "approved",
            "mediaResume": {
                "startImageStatus": "completed",
                "startImageArtifact": "data:image/png;base64,abc",
                "runwayTaskId": "task-1",
                "runwayStatus": "succeeded",
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "downloadedVideoPath": "https://example.com/runway.mp4",
                "postprocessStatus": "completed",
                "headlineArtifactUrl": "https://ace.example.com/api/video-headline/token1234567890123456789012345678",
            },
        }
        stage = _infer_resume_stage(state, None)
        self.assertEqual(stage, "generating_music")

    @patch("engine.builder2_resume_resolver.resolve_complete_ad_resume_stage", return_value="rendering_advertising_closure")
    @patch("engine.builder2_resume_resolver.is_valid_persisted_winner_development", return_value=True)
    @patch("engine.builder2_resume_resolver.is_tournament_ready_for_winner_selection", return_value=True)
    @patch("engine.builder2_resume_resolver.missing_judge_prototype_ids", return_value=[])
    @patch("engine.builder2_resume_resolver.missing_creator_prototype_ids", return_value=[])
    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_music_complete_skips_to_closure(
        self,
        _complete_ad: Any,
        _creators: Any,
        _judges: Any,
        _ready: Any,
        _winner: Any,
    ) -> None:
        path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        path.write(_fake_mp3_bytes() * 20)
        path.flush()
        path.close()
        self.addCleanup(lambda: os.unlink(path.name))
        state = {
            "mediaContinuationRequired": True,
            "reasoningComplete": True,
            "strategyFoundation": {"strategyFoundationId": "sf-1"},
            "winnerCandidateId": "cand-1",
            "winnerDevelopmentPlan": {"advertisingClosure": {"sloganText": "Slogan"}},
            "advertisingClosureStatus": "approved",
            "mediaResume": {
                "startImageStatus": "completed",
                "startImageArtifact": "data:image/png;base64,abc",
                "runwayTaskId": "task-1",
                "runwayStatus": "succeeded",
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "downloadedVideoPath": "https://example.com/runway.mp4",
                "postprocessStatus": "completed",
                "musicGenerationStatus": "succeeded",
                "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
                "musicArtifactPath": path.name,
            },
        }
        stage = _infer_resume_stage(state, None)
        self.assertEqual(stage, "rendering_advertising_closure")

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_checkpoint_sync_generating_music(self) -> None:
        state = {
            "mediaResume": {
                "musicGenerationStatus": "succeeded",
                "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
            }
        }
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        checkpoints = state.get("stageCheckpoints") or {}
        self.assertEqual(checkpoints.get("generating_music", {}).get("status"), "completed")


class TestClosureOnlyRerenderMusic(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    @patch("engine.builder2_closure_only_rerender.save_tournament_state")
    @patch("engine.builder2_closure_only_rerender.publish_builder2_durable_final_video")
    @patch("engine.builder2_closure_only_rerender.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_closure_only_rerender.require_builder2_web_storage_capability")
    @patch("engine.builder2_closure_only_rerender.inspect_builder2_closure_rerender")
    def test_recovers_durable_music(
        self,
        inspect_mock: MagicMock,
        _cap: MagicMock,
        render_mock: MagicMock,
        pub_mock: MagicMock,
        _save: MagicMock,
    ) -> None:
        inspect_mock.return_value = {
            "closureOnlyRerenderEligible": True,
            "closureDurationContractSatisfied": True,
            "closureOnlyRerenderMissingFields": [],
        }
        from engine.builder2_closure_render import ClosureRenderResult

        render_mock.return_value = ClosureRenderResult(
            public_url="",
            local_path="/tmp/out.mp4",
            measured_duration_seconds=13.5,
            output_token="token",
            input_fingerprint="fp",
        )
        pub_mock.return_value = MagicMock(public_url="https://ace.example.com/final.mp4")
        state = {
            "jobId": "job-rerender",
            "winnerDevelopmentPlan": {
                "advertisingClosure": {"productNameText": "ACE", "sloganText": "Feel it.", "language": "en"},
            },
            "advertisingClosure": {"productNameText": "ACE", "sloganText": "Feel it.", "language": "en"},
            "mediaResume": {
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "musicGenerationStatus": "succeeded",
                "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
            },
        }
        with patch(
            "engine.builder2_lyria_artifact.resolve_lyria_audio_for_render",
            return_value="/tmp/recovered.mp3",
        ):
            with patch(
                "engine.builder2_closure_only_rerender.apply_builder2_durable_publication_fields",
                return_value="https://ace.example.com/final.mp4",
            ):
                report = run_builder2_closure_only_rerender(
                    job_id="job-rerender",
                    tournament_state=state,
                    public_base_url="https://ace.example.com",
                )
        self.assertTrue(report.get("ok"))
        _, kwargs = render_mock.call_args
        self.assertEqual(kwargs.get("lyria_audio_path"), "/tmp/recovered.mp3")

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    @patch("engine.builder2_closure_only_rerender.inspect_builder2_closure_rerender")
    def test_fails_when_durable_music_unavailable(self, inspect_mock: MagicMock) -> None:
        inspect_mock.return_value = {
            "closureOnlyRerenderEligible": True,
            "closureDurationContractSatisfied": True,
            "closureOnlyRerenderMissingFields": [],
        }
        state = {
            "jobId": "job-rerender-fail",
            "winnerDevelopmentPlan": {
                "advertisingClosure": {"productNameText": "ACE", "sloganText": "Feel it.", "language": "en"},
            },
            "advertisingClosure": {"productNameText": "ACE", "sloganText": "Feel it.", "language": "en"},
            "mediaResume": {
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "musicGenerationStatus": "succeeded",
            },
        }
        with patch(
            "engine.builder2_lyria_artifact.resolve_lyria_audio_for_render",
            side_effect=Builder2LyriaError("builder2_lyria_succeeded_artifact_missing"),
        ):
            with patch("engine.builder2_closure_only_rerender.require_builder2_web_storage_capability"):
                report = run_builder2_closure_only_rerender(
                    job_id="job-rerender-fail",
                    tournament_state=state,
                    public_base_url="https://ace.example.com",
                )
        self.assertFalse(report.get("ok"))
        self.assertEqual(report.get("failureReason"), "builder2_lyria_succeeded_artifact_missing")


class TestCompletionContractAudio(unittest.TestCase):
    def test_lyria_job_requires_audio_stream_flag(self) -> None:
        state = {
            "advertisingClosureStatus": "completed",
            "mediaResume": {
                "advertisingClosureStatus": "completed",
                "advertisingClosureRendered": True,
                "finalVideoWithClosureUrl": "https://ace.example.com/api/builder2-final-video/abcd1234567890123456789012345678",
                "finalPublicUrl": "https://ace.example.com/api/builder2-final-video/abcd1234567890123456789012345678",
                "finalPublicationVerificationAccepted": True,
                "finalPublicationDurableStorageConfirmed": True,
                "actualFinalVideoDurationSeconds": 13.5,
                "musicGenerationStatus": "succeeded",
                "musicArtifactUrl": "https://ace.example.com/api/builder2-music-artifact/abcd1234567890123456789012345678",
                "finalVideoHasAudioStream": False,
            },
        }
        plan = {"advertisingClosure": {"sloganText": "Slogan", "productNameText": "ACE", "language": "en"}}
        ok, failure, _ = validate_builder2_media_completion_contract(state=state, plan=plan, require_job_video_url_match=False)
        self.assertFalse(ok)
        self.assertEqual(failure, "builder2_final_video_missing_audio_stream")


class TestLyria503PipelineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=False)
    @patch.dict(
        os.environ,
        {
            "BUILDER2_LYRIA_ENABLED": "true",
            "BUILDER2_LYRIA_API_KEY": "test-key",
            "BUILDER2_LYRIA_ARTIFACT_DIR": "",
        },
        clear=True,
    )
    @patch("engine.builder2_lyria._sleep_lyria_auto_retry_delay")
    def test_pipeline_continues_after_503_then_success(self, sleep_mock: MagicMock, _start_image_mock: MagicMock) -> None:
        from engine.builder2_closure_render import ClosureRenderResult
        from tests.test_builder2_lyria import _fake_mp3_bytes, _music_direction, _winner_plan_with_music

        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state: Dict[str, Any] = {
                "mediaResume": {
                    "mediaResumeStatus": "running",
                    "runwayVideoUrl": "https://example.com/runway.mp4",
                    "downloadedVideoPath": "https://example.com/runway.mp4",
                    "runwayTaskId": "task-1",
                }
            }
            plan = _winner_plan_with_music()
            plan.setdefault(
                "advertisingClosure",
                {"productNameText": "ACE", "sloganText": "Feel the breeze.", "language": "en"},
            )
            state["advertisingClosure"] = plan["advertisingClosure"]
            deps = MediaPipelineDeps(
                generate_start_image=lambda plan: "",
                submit_runway_task=lambda **kwargs: "task-1",
                poll_runway_task=lambda **kwargs: ("succeeded", "https://example.com/runway.mp4"),
                postprocess_video=lambda **kwargs: kwargs["runway_url"],
                compose_marketing_copy=lambda **kwargs: "marketing",
            )
            calls = {"count": 0}

            def _fake_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                if calls["count"] == 1:
                    raise Builder2LyriaError("builder2_lyria_api_rejected", http_status=503)
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    with patch("engine.builder2_lyria.call_lyria_generate_content", side_effect=_fake_caller):
                        with patch("engine.builder2_media_pipeline.validate_builder2_pre_runway"):
                            with patch(
                                "engine.builder2_closure_render.render_builder2_advertising_closure_endcard"
                            ) as render_mock:
                                render_mock.return_value = ClosureRenderResult(
                                    public_url="",
                                    local_path="/tmp/out.mp4",
                                    measured_duration_seconds=13.5,
                                    output_token="token",
                                    input_fingerprint="fp",
                                )
                                with patch(
                                    "engine.builder2_durable_finalization.publish_builder2_durable_final_video"
                                ) as pub_mock:
                                    pub_mock.return_value = MagicMock(public_url="https://example.com/final.mp4")
                                    with patch(
                                        "engine.builder2_durable_finalization.apply_builder2_durable_publication_fields",
                                        return_value="https://example.com/final.mp4",
                                    ):
                                        with patch(
                                            "engine.builder2_durable_finalization.require_builder2_web_storage_capability"
                                        ):
                                            with patch(
                                                "engine.builder2_lyria.publish_builder2_music_artifact",
                                                _fake_publish,
                                            ):
                                                updated_state, _counters = execute_builder2_media_pipeline(
                                                    job_id="job-pipeline-503",
                                                    state=state,
                                                    plan=plan,
                                                    public_base_url="https://ace.example.com",
                                                    product_description="desc",
                                                    deps=deps,
                                                )
        self.assertEqual(calls["count"], 2)
        sleep_mock.assert_called_once()
        media = updated_state["mediaResume"]
        self.assertEqual(media["musicGenerationStatus"], "succeeded")
        self.assertEqual(media["musicGenerationAttempt"], 2)
        self.assertTrue(str(media.get("musicArtifactPath") or "").strip())
        _, kwargs = render_mock.call_args
        self.assertTrue(str(kwargs.get("lyria_audio_path") or "").strip())


class TestFlagFalseUnchanged(unittest.TestCase):
    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=False)
    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "false"}, clear=True)
    def test_pipeline_skips_lyria(self, _mock: MagicMock) -> None:
        from engine.builder2_closure_render import ClosureRenderResult

        state: Dict[str, Any] = {
            "mediaResume": {
                "mediaResumeStatus": "running",
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "downloadedVideoPath": "https://example.com/runway.mp4",
                "runwayTaskId": "task-1",
            }
        }
        plan = _winner_plan_with_music()
        plan.setdefault(
            "advertisingClosure",
            {"productNameText": "ACE", "sloganText": "Feel the breeze.", "language": "en"},
        )
        state["advertisingClosure"] = plan["advertisingClosure"]
        deps = MediaPipelineDeps(
            generate_start_image=lambda plan: "",
            submit_runway_task=lambda **kwargs: "task-1",
            poll_runway_task=lambda **kwargs: ("succeeded", "https://example.com/runway.mp4"),
            postprocess_video=lambda **kwargs: kwargs["runway_url"],
            compose_marketing_copy=lambda **kwargs: "marketing",
        )
        with patch("engine.builder2_media_pipeline.validate_builder2_pre_runway"):
            with patch("engine.builder2_closure_render.render_builder2_advertising_closure_endcard") as render_mock:
                render_mock.return_value = ClosureRenderResult(
                    public_url="",
                    local_path="/tmp/out.mp4",
                    measured_duration_seconds=13.5,
                    output_token="token",
                    input_fingerprint="fp",
                )
                with patch("engine.builder2_durable_finalization.publish_builder2_durable_final_video") as pub_mock:
                    pub_mock.return_value = MagicMock(public_url="https://example.com/final.mp4")
                    with patch(
                        "engine.builder2_durable_finalization.apply_builder2_durable_publication_fields",
                        return_value="https://example.com/final.mp4",
                    ):
                        with patch("engine.builder2_durable_finalization.require_builder2_web_storage_capability"):
                            execute_builder2_media_pipeline(
                                job_id="job-flag-false-2",
                                state=state,
                                plan=plan,
                                public_base_url="https://example.com",
                                product_description="desc",
                                deps=deps,
                            )
        _, kwargs = render_mock.call_args
        self.assertEqual(kwargs.get("lyria_audio_path"), "")


if __name__ == "__main__":
    unittest.main()
