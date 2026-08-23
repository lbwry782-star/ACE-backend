"""
Builder2 Google Lyria integration tests — mocks/fakes only, no paid API calls.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_lyria import (
    Builder2LyriaError,
    build_lyria_generate_content_payload,
    call_lyria_generate_content,
    extract_audio_bytes_from_generate_content_response,
    generate_builder2_music,
    music_artifact_is_valid,
)
from engine.builder2_lyria_config import (
    DEFAULT_BUILDER2_LYRIA_MODEL,
    resolve_builder2_lyria_api_key,
    resolve_builder2_lyria_enabled,
    resolve_builder2_lyria_generate_content_url,
    resolve_builder2_lyria_model,
)
from engine.builder2_media_pipeline import MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_music_direction import (
    build_lyria_request_prompt,
    validate_music_direction_for_lyria_media,
    validate_music_direction_shape,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_winner_persistence import persist_winner_development_atomically
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt


def _fake_mp3_bytes() -> bytes:
    return b"\xff\xfb" + b"\x00" * 128


def _fake_lyria_response_payload() -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/mpeg",
                                "data": base64.b64encode(_fake_mp3_bytes()).decode("ascii"),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _music_direction() -> Dict[str, Any]:
    return {
        "prompt": "Tense minimal electronic pulse matching a slow reveal.",
        "instrumentalOnly": True,
        "immediateStart": True,
    }


def _winner_plan_with_music() -> Dict[str, Any]:
    strategy = _strategy(language="en")
    candidate = _candidate("summer_fan")
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision="omit", winning_candidate=candidate, strategy=strategy))
    plan["musicDirection"] = _music_direction()
    return plan


class TestLyriaConfig(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_feature_flag_defaults_false(self) -> None:
        self.assertFalse(resolve_builder2_lyria_enabled())

    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "true"}, clear=True)
    def test_feature_flag_true(self) -> None:
        self.assertTrue(resolve_builder2_lyria_enabled())

    @patch.dict(os.environ, {}, clear=True)
    def test_model_default(self) -> None:
        self.assertEqual(resolve_builder2_lyria_model(), DEFAULT_BUILDER2_LYRIA_MODEL)

    @patch.dict(os.environ, {"BUILDER2_LYRIA_MODEL": "lyria-3-pro-preview"}, clear=True)
    def test_model_from_env(self) -> None:
        self.assertEqual(resolve_builder2_lyria_model(), "lyria-3-pro-preview")
        url = resolve_builder2_lyria_generate_content_url()
        self.assertIn("lyria-3-pro-preview", url)
        self.assertIn("generateContent", url)

    @patch.dict(os.environ, {"BUILDER2_LYRIA_API_KEY": "secret-key"}, clear=True)
    def test_api_key_env_only(self) -> None:
        self.assertEqual(resolve_builder2_lyria_api_key(), "secret-key")


class TestMusicDirection(unittest.TestCase):
    def test_shape_validation_optional(self) -> None:
        normalized = validate_music_direction_shape(_music_direction())
        self.assertEqual(normalized["instrumentalOnly"], True)

    def test_strict_media_validation(self) -> None:
        plan = {"musicDirection": _music_direction()}
        validated = validate_music_direction_for_lyria_media(plan)
        self.assertTrue(validated["instrumentalOnly"])

    def test_request_builder_adds_safety_constraints(self) -> None:
        creative, combined = build_lyria_request_prompt(_music_direction())
        self.assertIn("Tense minimal", creative)
        self.assertIn("Instrumental only", combined)
        self.assertIn("No vocals", combined)


class TestLyriaRestParsing(unittest.TestCase):
    def test_extracts_mp3_from_inline_data(self) -> None:
        payload = _fake_lyria_response_payload()
        audio = extract_audio_bytes_from_generate_content_response(payload)
        self.assertTrue(audio.startswith(b"\xff\xfb"))

    def test_missing_audio_fails(self) -> None:
        with self.assertRaises(Builder2LyriaError):
            extract_audio_bytes_from_generate_content_response({"candidates": [{"content": {"parts": [{"text": "nope"}]}}]})

    def test_payload_shape(self) -> None:
        body = build_lyria_generate_content_payload(combined_prompt="test prompt")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "test prompt")


class TestGenerateBuilder2Music(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_LYRIA_ENABLED": "true",
            "BUILDER2_LYRIA_API_KEY": "test-key",
            "BUILDER2_LYRIA_ARTIFACT_DIR": "",
        },
        clear=True,
    )
    def test_success_persists_artifact_without_api_key_in_state(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state: Dict[str, Any] = {"mediaResume": {}}
            plan = {"musicDirection": _music_direction()}
            fake_audio = _fake_mp3_bytes() * 20

            def _fake_caller(**kwargs: Any) -> bytes:
                self.assertEqual(kwargs["model"], resolve_builder2_lyria_model())
                self.assertNotIn("secret", json.dumps(kwargs))
                return fake_audio

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    from tests.test_builder2_lyria_durable import _fake_publish

                    result = generate_builder2_music(
                        job_id="job-lyria-1",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                    )
            self.assertFalse(result.reused)
            self.assertTrue(music_artifact_is_valid(result.artifact_path))
            media = state["mediaResume"]
            self.assertEqual(media["musicGenerationStatus"], "succeeded")
            self.assertTrue(str(media.get("musicArtifactUrl") or "").startswith("https://"))
            self.assertNotIn("BUILDER2_LYRIA_API_KEY", json.dumps(media))
            self.assertNotIn("test-key", json.dumps(media))

    @patch.dict(
        os.environ,
        {"BUILDER2_LYRIA_ENABLED": "true", "BUILDER2_LYRIA_API_KEY": "test-key", "BUILDER2_LYRIA_ARTIFACT_DIR": ""},
        clear=True,
    )
    def test_reuse_skips_second_call(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_LYRIA_ARTIFACT_DIR": self.tmp}, clear=False):
            state: Dict[str, Any] = {"mediaResume": {}}
            plan = {"musicDirection": _music_direction()}
            calls = {"count": 0}

            def _fake_caller(**kwargs: Any) -> bytes:
                calls["count"] += 1
                return _fake_mp3_bytes() * 20

            with patch("engine.builder2_lyria.patch_tournament_state", lambda job_id, fn: None):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    from tests.test_builder2_lyria_durable import _fake_publish

                    generate_builder2_music(
                        job_id="job-lyria-2",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                    )
                    generate_builder2_music(
                        job_id="job-lyria-2",
                        state=state,
                        plan=plan,
                        public_base_url="https://ace.example.com",
                        api_caller=_fake_caller,
                        publisher=_fake_publish,
                    )
            self.assertEqual(calls["count"], 1)


class TestMediaPipelineLyriaFlag(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()
        MediaResumeIsolationGuard.end()

    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=False)
    @patch.dict(os.environ, {"BUILDER2_LYRIA_ENABLED": "false"}, clear=True)
    def test_flag_false_skips_lyria(self, _start_image_mock: MagicMock) -> None:
        state: Dict[str, Any] = {
            "mediaResume": {
                "mediaResumeStatus": "running",
                "runwayVideoUrl": "https://example.com/runway.mp4",
                "downloadedVideoPath": "https://example.com/runway.mp4",
                "runwayTaskId": "task-1",
            }
        }
        plan = _winner_plan_with_music()
        plan.setdefault("advertisingClosure", {"productNameText": "ACE", "sloganText": "Feel the breeze.", "language": "en"})
        state["advertisingClosure"] = plan["advertisingClosure"]
        plan.setdefault("advertisingClosure", {"productNameText": "ACE", "sloganText": "Feel the breeze.", "language": "en"})
        state["advertisingClosure"] = plan["advertisingClosure"]
        deps = MediaPipelineDeps(
            generate_start_image=lambda plan: "",
            submit_runway_task=lambda **kwargs: "task-1",
            poll_runway_task=lambda **kwargs: ("succeeded", "https://example.com/runway.mp4"),
            postprocess_video=lambda **kwargs: kwargs["runway_url"],
            compose_marketing_copy=lambda **kwargs: "marketing",
        )
        from engine.builder2_closure_render import ClosureRenderResult

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
                                job_id="job-flag-false",
                                state=state,
                                plan=plan,
                                public_base_url="https://example.com",
                                product_description="desc",
                                deps=deps,
                            )
        _, kwargs = render_mock.call_args
        self.assertEqual(kwargs.get("lyria_audio_path"), "")


class TestWinnerPersistenceMusicDirection(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_music_direction_persisted_in_winner_plan(self) -> None:
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference
        from engine.builder2_winner_preservation_contract import build_winning_candidate_preservation_snapshot
        from engine.builder2_winner_preservation_contract import process_winner_development_response

        state: Dict[str, Any] = {"jobId": "job-winner-music", "candidates": {}}
        strategy = _strategy(language="en")
        candidate = _candidate("summer_fan")
        raw_plan = _winner_plan_with_music()
        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-1",
        )
        snapshot = build_winning_candidate_preservation_snapshot(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-1",
        )
        plan = process_winner_development_response(
            raw_plan,
            source_reference=source,
            winning_candidate=candidate,
            preservation_snapshot=snapshot,
            winning_judgment={},
        )
        persisted = persist_winner_development_atomically(
            state,
            candidate_id="cand-1",
            prototype_id="summer_fan",
            winner_plan=plan,
            winning_candidate=candidate,
            preservation_snapshot=snapshot,
        )
        self.assertEqual(persisted["musicDirection"]["prompt"], raw_plan["musicDirection"]["prompt"])


if __name__ == "__main__":
    unittest.main()
