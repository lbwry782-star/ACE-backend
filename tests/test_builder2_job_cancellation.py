"""
Builder2 job cancellation tests — offline/mocked only.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder2_job_cancellation import (
    CANCELLED_ERROR_CODE,
    CANCEL_REASON_FRONTEND_REFRESH,
    Builder2JobCancelledError,
    checkpoint_builder2_cancellation,
    is_builder2_job_cancelled,
    request_builder2_job_cancellation,
    video_job_mark_done_respecting_cancellation,
)
from engine.builder2_job_ownership import ownership_fields_for_job_create
from engine.builder2_lyria import Builder2LyriaError, generate_builder2_music
from engine.builder2_media_pipeline import MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_resume_service import request_builder2_resume
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_recovery import (
    disable_memory_recovery,
    enable_memory_recovery,
    mark_job_queued,
    register_recoverable_job,
    requeue_recoverable_job,
    set_memory_job_hash,
)
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    new_tournament_state,
    save_tournament_state,
)
from engine.video_jobs_redis import (
    QUEUE_KEY,
    disable_memory_jobs,
    enable_memory_jobs,
    video_job_create,
    video_job_get_raw,
)
from tests.test_builder2_media_resume import _media_ready_state
from tests.test_builder2_tournament import _strategy
from tests.test_builder2_lyria import _music_direction


def _builder2_job_hash(job_id: str, **extra: str) -> Dict[str, str]:
    fields = ownership_fields_for_job_create(MagicMock(headers={}), {"productDescription": "desc"})
    fields.update(
        {
            "status": "queued",
            "product_name": "Product",
            "product_description": "A product description for testing.",
            "public_base_url": "https://example.test",
            "canResume": "1",
        }
    )
    fields.update({k: str(v) for k, v in extra.items()})
    set_memory_job_hash(job_id, fields)
    return fields


class _CancellationTestBase(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"
        os.environ["BUILDER2_LYRIA_ENABLED"] = "true"

    def tearDown(self) -> None:
        disable_memory_jobs()
        disable_memory_store()
        disable_memory_recovery()
        os.environ.pop("BUILDER2_TOURNAMENT_ENABLED", None)
        os.environ.pop("BUILDER2_LYRIA_ENABLED", None)


class TestCancelEndpointContract(_CancellationTestBase):
    def test_cancel_queued_job(self) -> None:
        job_id = "job-cancel-queued"
        _builder2_job_hash(job_id, status="queued")
        mark_job_queued(job_id)

        result = request_builder2_job_cancellation(job_id, reason=CANCEL_REASON_FRONTEND_REFRESH)
        self.assertEqual(result["outcome"], "cancelled")
        raw = video_job_get_raw(job_id) or {}
        self.assertEqual(raw.get("status"), "cancelled")
        self.assertEqual(raw.get("cancelReason"), CANCEL_REASON_FRONTEND_REFRESH)
        self.assertTrue(is_builder2_job_cancelled(job_id))

    def test_idempotent_cancel(self) -> None:
        job_id = "job-cancel-idempotent"
        _builder2_job_hash(job_id, status="running")
        first = request_builder2_job_cancellation(job_id)
        second = request_builder2_job_cancellation(job_id)
        self.assertEqual(first["outcome"], "cancelled")
        self.assertEqual(second["outcome"], "already_cancelled")

    def test_already_completed(self) -> None:
        job_id = "job-cancel-done"
        _builder2_job_hash(job_id, status="done", video_url="https://example.test/v.mp4")
        result = request_builder2_job_cancellation(job_id)
        self.assertEqual(result["outcome"], "already_completed")

    def test_completion_race_cancel_then_complete(self) -> None:
        job_id = "job-race-cancel-first"
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)
        wrote = video_job_mark_done_respecting_cancellation(job_id, "https://x.test/v.mp4", "copy")
        self.assertFalse(wrote)
        self.assertEqual((video_job_get_raw(job_id) or {}).get("status"), "cancelled")

    def test_completion_race_complete_then_cancel(self) -> None:
        job_id = "job-race-complete-first"
        _builder2_job_hash(job_id, status="running")
        wrote = video_job_mark_done_respecting_cancellation(job_id, "https://x.test/v.mp4", "copy")
        self.assertTrue(wrote)
        result = request_builder2_job_cancellation(job_id)
        self.assertEqual(result["outcome"], "already_completed")

    def test_not_builder2_job_rejected(self) -> None:
        job_id = "job-builder1"
        set_memory_job_hash(
            job_id,
            {
                "status": "queued",
                "product_name": "P",
                "product_description": "D",
                "public_base_url": "https://example.test",
            },
        )
        result = request_builder2_job_cancellation(job_id)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "not_builder2_job")


class TestCancelBeforePaidStages(_CancellationTestBase):
    @patch("engine.builder2_tournament_manager.resolve_builder2_active_prototype_ids", return_value=["think_small"])
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_attempts_per_prototype_per_round", return_value=1)
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_max_rounds", return_value=1)
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_eliminations_per_round", return_value=0)
    @patch("engine.builder2_tournament_manager._generate_strategy")
    def test_cancel_before_strategy_no_openai(
        self,
        strategy_mock: MagicMock,
        *_mocks: Any,
    ) -> None:
        job_id = "job-before-strategy"
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)
        with self.assertRaises(Builder2JobCancelledError):
            run_builder2_tournament(
                job_id=job_id,
                product_name="Product",
                product_description="Desc",
                content_language="he",
                llm_client=MagicMock(),
            )
        strategy_mock.assert_not_called()

    @patch("engine.builder2_tournament_manager.resolve_builder2_active_prototype_ids", return_value=["think_small", "hero_journey"])
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_attempts_per_prototype_per_round", return_value=1)
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_max_rounds", return_value=1)
    @patch("engine.builder2_tournament_manager.resolve_builder2_tournament_eliminations_per_round", return_value=0)
    @patch("engine.builder2_tournament_manager.generate_creator_candidate")
    def test_cancel_between_creators(
        self,
        creator_mock: MagicMock,
        *_mocks: Any,
    ) -> None:
        job_id = "job-between-creators"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small", "hero_journey"], random_seed="seed")
        state["strategyFoundation"] = _strategy()
        state["status"] = "strategy_complete"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")

        def _creator_side_effect(**kwargs: Any) -> Any:
            request_builder2_job_cancellation(job_id)
            return "cand-1", {"candidateId": "cand-1", "prototypeId": kwargs["prototype_id"]}

        creator_mock.side_effect = _creator_side_effect
        with self.assertRaises(Builder2JobCancelledError):
            run_builder2_tournament(
                job_id=job_id,
                product_name="Product",
                product_description="Desc",
                content_language="he",
                llm_client=MagicMock(),
            )
        self.assertEqual(creator_mock.call_count, 1)

    @patch("engine.builder2_tournament_manager.develop_builder2_winning_candidate")
    @patch("engine.builder2_tournament_manager.select_global_winner", return_value="cand-1")
    @patch("engine.builder2_tournament_completion_gate.assert_tournament_ready_for_winner_selection")
    @patch("engine.builder2_tournament_completion_gate.invalidate_provisional_winner_if_incomplete")
    def test_cancel_before_winner_development(
        self,
        _inv: MagicMock,
        _assert_ready: MagicMock,
        _select: MagicMock,
        winner_mock: MagicMock,
    ) -> None:
        job_id = "job-before-winner"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="seed")
        state["strategyFoundation"] = _strategy()
        state["status"] = "tournament_complete"
        state["winnerCandidateId"] = "cand-1"
        state["candidates"] = {
            "cand-1": {
                "candidateId": "cand-1",
                "prototypeId": "think_small",
                "creatorOutput": {"candidateId": "cand-1"},
                "eligible": True,
                "creatorAcceptanceStatus": "accepted",
                "judgeStatus": "accepted",
                "judgmentId": "j-1",
            }
        }
        state["judgments"] = {"j-1": {"judgment": {"eligible": True}}}
        state["rounds"] = [{"roundIndex": 1, "roundComplete": True}]
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        with patch(
            "engine.builder2_tournament_manager.resolve_builder2_active_prototype_ids",
            return_value=["think_small"],
        ), patch(
            "engine.builder2_tournament_manager.resolve_builder2_tournament_max_rounds",
            return_value=1,
        ), patch(
            "engine.builder2_tournament_manager.resolve_builder2_tournament_attempts_per_prototype_per_round",
            return_value=1,
        ), patch(
            "engine.builder2_tournament_manager.resolve_builder2_tournament_eliminations_per_round",
            return_value=0,
        ), patch(
            "engine.builder2_tournament_manager.is_creator_contract_circuit_breaker_tripped",
            return_value=False,
        ), patch(
            "engine.builder2_tournament_manager.is_judge_contract_circuit_breaker_tripped",
            return_value=False,
        ):
            with self.assertRaises(Builder2JobCancelledError):
                run_builder2_tournament(
                    job_id=job_id,
                    product_name="Product",
                    product_description="Desc",
                    content_language="he",
                    llm_client=MagicMock(),
                )
        winner_mock.assert_not_called()


class TestMediaCancellation(_CancellationTestBase):
    def _plan_and_state(self, job_id: str) -> tuple:
        state = _media_ready_state(job_id=job_id)
        state["advertisingClosureStatus"] = "approved"
        save_tournament_state(job_id, state)
        plan = dict(state["winnerDevelopmentPlan"])
        plan["musicDirection"] = _music_direction()
        return state, plan

    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=True)
    @patch("engine.builder2_media_pipeline.validate_builder2_pre_runway")
    def test_cancel_before_start_image(self, _pre: MagicMock, _req: MagicMock) -> None:
        job_id = "job-before-start-image"
        state, plan = self._plan_and_state(job_id)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)
        start_calls: List[int] = []

        deps = MediaPipelineDeps(
            generate_start_image=lambda **_k: start_calls.append(1) or "data:image/png;base64,abc",
            submit_runway_task=lambda **_k: "task-1",
            poll_runway_task=lambda **_k: ("SUCCEEDED", "https://runway.test/v.mp4"),
            postprocess_video=lambda **_k: "https://example.test/headline.mp4",
            compose_marketing_copy=lambda **_k: "copy",
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            execute_builder2_media_pipeline(
                job_id=job_id,
                state=state,
                plan=plan,
                public_base_url="https://example.test",
                product_description="desc",
                deps=deps,
            )
        self.assertEqual(str(ctx.exception.args[0]), CANCELLED_ERROR_CODE)
        self.assertEqual(start_calls, [])

    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=False)
    @patch("engine.builder2_media_pipeline.validate_builder2_pre_runway")
    @patch("engine.builder2_lyria_config.resolve_builder2_lyria_enabled", return_value=True)
    def test_cancel_after_runway_stops_polling_and_lyria(
        self,
        _lyria_on: MagicMock,
        _pre: MagicMock,
        _req: MagicMock,
    ) -> None:
        job_id = "job-after-runway"
        state, plan = self._plan_and_state(job_id)
        media = state.setdefault("mediaResume", {})
        media["runwayTaskId"] = "task-existing"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")

        poll_calls = {"n": 0}
        lyria_calls: List[int] = []

        def _poll(**_k: Any) -> Any:
            poll_calls["n"] += 1
            request_builder2_job_cancellation(job_id)
            raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)

        deps = MediaPipelineDeps(
            generate_start_image=lambda **_k: "",
            submit_runway_task=lambda **_k: "task-1",
            poll_runway_task=_poll,
            postprocess_video=lambda **_k: "https://example.test/headline.mp4",
            compose_marketing_copy=lambda **_k: "copy",
        )
        with patch("engine.builder2_lyria.generate_builder2_music", side_effect=lambda **_k: lyria_calls.append(1)):
            with self.assertRaises(Builder2TournamentError) as ctx:
                execute_builder2_media_pipeline(
                    job_id=job_id,
                    state=state,
                    plan=plan,
                    public_base_url="https://example.test",
                    product_description="desc",
                    deps=deps,
                )
        self.assertEqual(str(ctx.exception.args[0]), CANCELLED_ERROR_CODE)
        self.assertEqual(lyria_calls, [])

    @patch("engine.builder2_lyria_config.resolve_builder2_lyria_enabled", return_value=True)
    @patch("engine.builder2_lyria.resolve_builder2_lyria_api_key", return_value="test-key")
    def test_lyria_503_cancel_before_attempt_2(self, *_m: Any) -> None:
        job_id = "job-lyria-503-cancel"
        state = _media_ready_state(job_id=job_id)
        plan = dict(state["winnerDevelopmentPlan"])
        plan["musicDirection"] = _music_direction()
        _builder2_job_hash(job_id, status="running")

        calls = {"n": 0}

        def _api(**_k: Any) -> bytes:
            calls["n"] += 1
            if calls["n"] == 1:
                request_builder2_job_cancellation(job_id)
                raise Builder2LyriaError("builder2_lyria_http_error", http_status=503)
            return b"\xff\xfb" + b"\x00" * 128

        with patch("engine.builder2_lyria._sleep_lyria_auto_retry_delay"):
            with patch("engine.builder2_lyria.write_mp3_artifact", return_value=MagicMock()):
                with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=30.0):
                    with patch("engine.builder2_lyria.publish_builder2_music_artifact") as pub_mock:
                        pub_mock.return_value = MagicMock(music_artifact_url="https://m.test/a.mp3", output_token="tok")
                        with self.assertRaises(Builder2JobCancelledError):
                            generate_builder2_music(
                                job_id=job_id,
                                state=state,
                                plan=plan,
                                api_caller=_api,
                            )
        self.assertEqual(calls["n"], 1)

    @patch("engine.builder2_media_pipeline.builder2_runway_requires_start_image", return_value=False)
    @patch("engine.builder2_media_pipeline.validate_builder2_pre_runway")
    @patch("engine.builder2_lyria_config.resolve_builder2_lyria_enabled", return_value=True)
    def test_cancel_before_lyria_zero_calls(self, _lyria_on: MagicMock, _pre: MagicMock, _req: MagicMock) -> None:
        job_id = "job-before-lyria"
        state, plan = self._plan_and_state(job_id)
        media = state.setdefault("mediaResume", {})
        media["runwayVideoUrl"] = "https://runway.test/v.mp4"
        media["downloadedVideoPath"] = "https://runway.test/v.mp4"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)
        lyria_calls: List[int] = []

        deps = MediaPipelineDeps(
            generate_start_image=lambda **_k: "",
            submit_runway_task=lambda **_k: "task-1",
            poll_runway_task=lambda **_k: ("SUCCEEDED", "https://runway.test/v.mp4"),
            postprocess_video=lambda **_k: "https://example.test/headline.mp4",
            compose_marketing_copy=lambda **_k: "copy",
        )
        with patch("engine.builder2_lyria.generate_builder2_music", side_effect=lambda **_k: lyria_calls.append(1)):
            with patch("engine.builder2_closure_render.render_builder2_advertising_closure_endcard") as render_mock:
                render_mock.return_value = MagicMock(
                    measured_duration_seconds=10.0,
                    output_token="tok",
                    duration_diagnostics=None,
                )
                with self.assertRaises(Builder2TournamentError):
                    execute_builder2_media_pipeline(
                        job_id=job_id,
                        state=state,
                        plan=plan,
                        public_base_url="https://example.test",
                        product_description="desc",
                        deps=deps,
                    )
        self.assertEqual(lyria_calls, [])


class TestInFlightAndResume(_CancellationTestBase):
    def test_in_flight_paid_call_returns_cancelled_not_next_stage(self) -> None:
        job_id = "job-in-flight"
        _builder2_job_hash(job_id, status="running")
        paid_started = {"v": False}

        def _paid(**_k: Any) -> Any:
            paid_started["v"] = True
            request_builder2_job_cancellation(job_id)
            return b"\xff\xfb" + b"\x00" * 64

        state = _media_ready_state(job_id=job_id)
        plan = dict(state["winnerDevelopmentPlan"])
        plan["musicDirection"] = _music_direction()
        with patch("engine.builder2_lyria.resolve_builder2_lyria_enabled", return_value=True):
            with patch("engine.builder2_lyria.resolve_builder2_lyria_api_key", return_value="k"):
                with patch("engine.builder2_lyria.write_mp3_artifact", return_value=MagicMock()):
                    with patch("engine.builder2_lyria.probe_mp3_duration_seconds", return_value=10.0):
                        with patch("engine.builder2_lyria.publish_builder2_music_artifact") as pub:
                            pub.return_value = MagicMock(music_artifact_url="u", output_token="t")
                            with self.assertRaises(Builder2JobCancelledError):
                                generate_builder2_music(job_id=job_id, state=state, plan=plan, api_caller=_paid)
        self.assertTrue(paid_started["v"])
        self.assertTrue(is_builder2_job_cancelled(job_id))
        self.assertEqual((video_job_get_raw(job_id) or {}).get("status"), "cancelled")

    def test_reasoning_resume_rejected(self) -> None:
        job_id = "job-resume-reasoning"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume

        report = run_controlled_complete_ad_reasoning_resume(job_id=job_id, tournament_state=state, acquire_lease=False)
        self.assertEqual(report.get("failureReason"), "builder2_job_cancelled")

    def test_media_resume_rejected(self) -> None:
        job_id = "job-resume-media"
        state = _media_ready_state(job_id=job_id)
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        from engine.builder2_media_resume import run_one_media_resume

        report = run_one_media_resume(job_id=job_id, tournament_state=state)
        self.assertEqual(report.get("failureReason"), "builder2_job_cancelled")

    def test_durable_resume_rejected(self) -> None:
        job_id = "job-durable-resume"
        state = _media_ready_state(job_id=job_id)
        state["status"] = "failed"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1", canResume="0")
        result = request_builder2_resume(job_id)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "builder2_job_cancelled")

    def test_recovery_does_not_requeue_cancelled(self) -> None:
        job_id = "job-recovery-cancel"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        register_recoverable_job(job_id)
        self.assertFalse(requeue_recoverable_job(job_id))


class TestBuilder1Isolation(_CancellationTestBase):
    def test_builder1_job_not_cancellable_via_builder2_endpoint(self) -> None:
        job_id = "builder1-job"
        set_memory_job_hash(
            job_id,
            {
                "status": "queued",
                "product_name": "P",
                "product_description": "D",
                "public_base_url": "https://example.test",
            },
        )
        result = request_builder2_job_cancellation(job_id)
        self.assertEqual(result.get("error"), "not_builder2_job")
        self.assertEqual((video_job_get_raw(job_id) or {}).get("status"), "queued")

    def test_checkpoint_no_job_id_is_noop(self) -> None:
        checkpoint_builder2_cancellation("", stage="noop")


if __name__ == "__main__":
    unittest.main()
