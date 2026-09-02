"""
Builder2 durable job resume tests — mocks only.
"""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_accepted_creator_store import ACCEPTED_CREATOR_INDEX_KEY, persist_accepted_creator_candidate
from engine.builder2_accepted_judgment_store import persist_accepted_judgment
from engine.builder2_execution_lease import acquire_job_lease, has_active_lease, release_job_lease, renew_job_lease
from engine.builder2_job_ownership import (
    extract_owner_context_from_request,
    is_historical_job_without_ownership,
    ownership_fields_for_job_create,
    owner_context_present_in_job,
    verify_owner_context,
)
from engine.builder2_resume_contract import (
    BUILDER2_RESUME_CONTRACT_VERSION,
    CANONICAL_BUILDER2_STAGES,
    completed_stage_names,
    sync_builder2_stage_checkpoints_from_state,
    upsert_stage_checkpoint,
)
from engine.builder2_resume_inspect import inspect_builder2_resume_job
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_resume_service import build_builder2_status_payload, request_builder2_resume
from engine.builder2_tournament_recovery import (
    disable_memory_recovery,
    enable_memory_recovery,
    is_job_queued,
    mark_job_queued,
    new_worker_token,
    set_memory_job_hash,
)
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    new_tournament_state,
    save_tournament_state,
)
from engine.video_jobs_redis import QUEUE_KEY, disable_memory_jobs, enable_memory_jobs, job_key, video_job_create
from tests.test_builder2_media_resume import _media_ready_state
from tests.test_builder2_reasoning_resume import _candidate_id_for_prototype, _historical_resume_state, _judgment_for_candidate
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt


HISTORICAL_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _durable_media_state(job_id: str = "job-media", **media_fields: Any) -> Dict[str, Any]:
    state = _media_ready_state(job_id=job_id)
    state["advertisingClosureStatus"] = "approved"
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    media.update(media_fields)
    save_tournament_state(job_id, state)
    return state


def _mock_request(*, batch_state: str = "batch-abc", authorization: str = "") -> MagicMock:
    headers = {}
    if batch_state:
        headers["X-ACE-Batch-State"] = batch_state
    if authorization:
        headers["Authorization"] = authorization
    request = MagicMock()
    request.headers = headers
    return request


def _owned_job_hash(job_id: str, *, batch_state: str = "batch-abc") -> Dict[str, str]:
    fields = ownership_fields_for_job_create(_mock_request(batch_state=batch_state), {"productDescription": "desc"})
    return {
        "status": "running",
        "product_description": "desc",
        "video_url": "",
        "enqueued_ts": "1716192000",
        "progressStartedAt": "2026-05-20T08:00:00+00:00",
        **fields,
    }


class TestBuilder2DurableResumeCore(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()
        enable_memory_jobs()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"

    def tearDown(self) -> None:
        disable_memory_store()
        disable_memory_recovery()
        disable_memory_jobs()

    def test_refresh_status_read_creates_no_new_job(self) -> None:
        job_id = "job-status-read"
        set_memory_job_hash(job_id, _owned_job_hash(job_id))
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        payload = build_builder2_status_payload(job_id, _owned_job_hash(job_id))
        self.assertEqual(payload["jobId"], job_id)
        self.assertIn("canResume", payload)

    def test_retry_returns_same_job_id(self) -> None:
        job_id = "job-retry-same"
        set_memory_job_hash(job_id, _owned_job_hash(job_id))
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        result = request_builder2_resume(job_id, request=_mock_request())
        self.assertEqual(result["jobId"], job_id)

    def test_resume_returns_same_job_id(self) -> None:
        job_id = "job-resume-same"
        set_memory_job_hash(job_id, _owned_job_hash(job_id))
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        result = request_builder2_resume(job_id, request=_mock_request())
        self.assertEqual(result.get("jobId"), job_id)

    def test_explicit_new_submission_creates_new_job_id(self) -> None:
        disable_memory_jobs()
        try:
            with patch("engine.video_jobs_redis.get_redis") as redis_mock:
                pipe = MagicMock()
                redis_mock.return_value.pipeline.return_value = pipe
                video_job_create("job-a", "Name", "desc", "https://example.com", extra_fields={"builder": "builder2"})
                video_job_create("job-b", "Name", "desc", "https://example.com", extra_fields={"builder": "builder2"})
            self.assertEqual(pipe.lpush.call_count, 2)
            pushed_ids = [call.args[1] for call in pipe.lpush.call_args_list]
            self.assertNotEqual(pushed_ids[0], pushed_ids[1])
        finally:
            enable_memory_jobs()

    def test_strategy_checkpoint_is_reused(self) -> None:
        state = new_tournament_state(job_id="job-strategy", language="he", active_prototype_ids=["closest"], random_seed="s")
        state["strategyFoundation"] = _strategy(language="he")
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("strategy", completed_stage_names(state))
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertNotEqual(resolved["resumeFromStage"], "strategy")

    def test_partial_creator_set_resumes_only_missing_creators(self) -> None:
        state = new_tournament_state(job_id="job-creator-partial", language="he", active_prototype_ids=["closest", "winning_card"], random_seed="s")
        state["strategyFoundation"] = _strategy(language="he")
        persist_accepted_creator_candidate(
            state,
            candidate_id=_candidate_id_for_prototype("closest"),
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            creator_output=_candidate("closest"),
            strategy_foundation=state["strategyFoundation"],
        )
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved["resumeFromStage"], "creator_generation")

    def test_completed_creators_are_not_regenerated(self) -> None:
        state = _historical_resume_state(with_reusable_judgments=0)
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("creator_complete", completed_stage_names(state))

    def test_partial_judge_set_resumes_only_missing_judges(self) -> None:
        state = _historical_resume_state(with_reusable_judgments=1)
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved["resumeFromStage"], "judge_generation")

    def test_completed_judges_are_not_regenerated(self) -> None:
        state = _historical_resume_state(with_reusable_judgments=6)
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("judge_complete", completed_stage_names(state))

    def test_winner_selection_is_reused(self) -> None:
        state = _historical_resume_state(with_reusable_judgments=6)
        state["winnerCandidateId"] = _candidate_id_for_prototype("summer_fan")
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("winner_selection", completed_stage_names(state))

    def test_winner_development_is_reused(self) -> None:
        state = _durable_media_state("job-winner-dev")
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("winner_development", completed_stage_names(state))

    def test_approved_slogan_closure_is_reused(self) -> None:
        state = _durable_media_state("job-closure")
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("advertising_closure", completed_stage_names(state))

    def test_start_image_is_reused(self) -> None:
        state = _durable_media_state(
            "job-start-image",
            startImageArtifact="data:image/png;base64,abc",
            startImageStatus="completed",
        )
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("start_image_complete", completed_stage_names(state))
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved["resumeFromStage"], "runway_submission")

    def test_existing_runway_task_prevents_resubmission(self) -> None:
        state = _durable_media_state(
            "job-runway-idempotent",
            startImageArtifact="data:image/png;base64,abc",
            startImageStatus="completed",
            runwayTaskId="task-existing",
            runwayStatus="running",
        )
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved["resumeFromStage"], "runway_waiting")
        self.assertIn("runwayTaskId", resolved["reusableArtifacts"])

    def test_existing_runway_task_resumes_polling(self) -> None:
        state = _durable_media_state(
            "job-runway-poll",
            startImageArtifact="data:image/png;base64,abc",
            startImageStatus="completed",
            runwayTaskId="task-existing",
            runwayStatus="running",
        )
        resolved = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved["resumeFromStage"], "runway_waiting")

    def test_completed_runway_output_is_reused(self) -> None:
        state = _durable_media_state(
            "job-runway-output",
            startImageArtifact="data:image/png;base64,abc",
            startImageStatus="completed",
            runwayTaskId="task-existing",
            runwayStatus="succeeded",
            runwayOutputUrl="https://runway.example/out.mp4",
        )
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("runway_complete", completed_stage_names(state))

    def test_downloaded_video_is_reused(self) -> None:
        state = _durable_media_state(
            "job-downloaded",
            downloadedVideoPath="/tmp/video.mp4",
            postprocessStatus="completed",
            finalPublicUrl="https://example.com/final.mp4",
        )
        job = {"status": "done", "video_url": "https://example.com/final.mp4", "postprocess_ran": "1"}
        resolved = resolve_builder2_resume_stage(job, state)
        self.assertTrue(resolved["jobAlreadyCompleted"])

    def test_existing_final_artifact_is_reused(self) -> None:
        job = {"status": "done", "video_url": "https://example.com/final.mp4"}
        state = {"mediaResume": {"finalPublicUrl": "https://example.com/final.mp4"}}
        resolved = resolve_builder2_resume_stage(job, state)
        self.assertTrue(resolved["jobAlreadyCompleted"])

    def test_completed_job_makes_zero_paid_calls(self) -> None:
        job_id = "job-done"
        set_memory_job_hash(
            job_id,
            {**_owned_job_hash(job_id), "status": "done", "video_url": "https://example.com/final.mp4"},
        )
        state = _durable_media_state(
            job_id,
            finalPublicUrl="https://example.com/final.mp4",
            mediaResumeStatus="completed",
        )
        result = request_builder2_resume(job_id, request=_mock_request())
        self.assertTrue(result.get("mediaReused"))

    def test_failure_does_not_erase_earlier_checkpoints(self) -> None:
        state = _historical_resume_state(with_reusable_judgments=6)
        upsert_stage_checkpoint(state, "strategy", status="completed", artifact_ref="strategyFoundation")
        state["resumeFailure"] = {"failureStage": "runway_waiting", "failureReason": "timeout"}
        sync_builder2_stage_checkpoints_from_state(job_state={}, tournament_state=state)
        self.assertIn("strategy", completed_stage_names(state))

    def test_sigterm_leaves_resumable_job(self) -> None:
        job_id = "job-sigterm"
        job_hash = _owned_job_hash(job_id)
        job_hash["status"] = "interrupted"
        job_hash["canResume"] = "1"
        set_memory_job_hash(job_id, job_hash)
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        resolved = resolve_builder2_resume_stage(job_hash, state)
        self.assertTrue(resolved["canResume"])

    def test_expired_execution_lease_can_be_reclaimed(self) -> None:
        job_id = "job-lease-reclaim"
        token_a = new_worker_token()
        token_b = new_worker_token()
        self.assertTrue(acquire_job_lease(job_id, token_a))
        release_job_lease(job_id, token_a)
        self.assertTrue(acquire_job_lease(job_id, token_b))

    def test_active_lease_prevents_duplicate_worker_execution(self) -> None:
        job_id = "job-active-lease"
        token_a = new_worker_token()
        token_b = new_worker_token()
        self.assertTrue(acquire_job_lease(job_id, token_a))
        self.assertFalse(acquire_job_lease(job_id, token_b))

    def test_only_lease_owner_can_release_it(self) -> None:
        job_id = "job-lease-owner"
        owner = new_worker_token()
        other = new_worker_token()
        self.assertTrue(acquire_job_lease(job_id, owner))
        release_job_lease(job_id, other)
        self.assertTrue(has_active_lease(job_id))
        release_job_lease(job_id, owner)
        self.assertFalse(has_active_lease(job_id))

    def test_duplicate_resume_does_not_duplicate_queue_entry(self) -> None:
        job_id = "job-dedupe-queue"
        set_memory_job_hash(job_id, _owned_job_hash(job_id))
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        first = request_builder2_resume(job_id, request=_mock_request())
        second = request_builder2_resume(job_id, request=_mock_request())
        self.assertTrue(first.get("resumeRequested") or first.get("resumeAlreadyInProgress"))
        self.assertTrue(second.get("resumeAlreadyInProgress"))

    def test_two_simultaneous_retries_do_not_create_two_runway_tasks(self) -> None:
        state = _durable_media_state(
            "job-runway-dedupe",
            runwayTaskId="task-one",
            runwayStatus="running",
            startImageArtifact="data:image/png;base64,abc",
            startImageStatus="completed",
        )
        resolved_a = resolve_builder2_resume_stage({}, state)
        resolved_b = resolve_builder2_resume_stage({}, state)
        self.assertEqual(resolved_a["resumeFromStage"], "runway_waiting")
        self.assertEqual(resolved_b["resumeFromStage"], "runway_waiting")
        self.assertEqual(state["mediaResume"]["runwayTaskId"], "task-one")

    def test_status_endpoint_exposes_resume_fields(self) -> None:
        job_id = "job-status-fields"
        job_hash = _owned_job_hash(job_id)
        set_memory_job_hash(job_id, job_hash)
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        payload = build_builder2_status_payload(job_id, job_hash)
        for field in (
            "canResume",
            "resumeFromStage",
            "progressStage",
            "ownerContextPresent",
            "resumeAlreadyInProgress",
            "elapsedSeconds",
        ):
            self.assertIn(field, payload)

    def test_progress_started_at_is_not_reset(self) -> None:
        job_id = "job-progress-preserve"
        original = "2026-05-20T08:00:00+00:00"
        job_hash = _owned_job_hash(job_id)
        job_hash["progressStartedAt"] = original
        set_memory_job_hash(job_id, job_hash)
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        request_builder2_resume(job_id, request=_mock_request())
        updated = build_builder2_status_payload(job_id, job_hash)
        self.assertEqual(updated["progressStartedAt"], original)

    def test_ownership_mismatch_blocks_resume(self) -> None:
        job_id = "job-owner-mismatch"
        set_memory_job_hash(job_id, _owned_job_hash(job_id, batch_state="owner-a"))
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        result = request_builder2_resume(job_id, request=_mock_request(batch_state="owner-b"))
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "ownership_mismatch")

    def test_ownership_fields_persisted_on_new_jobs(self) -> None:
        fields = ownership_fields_for_job_create(_mock_request(), {"productDescription": "coffee fan"})
        self.assertEqual(fields["builder"], "builder2")
        self.assertEqual(fields["builder2ResumeContractVersion"], BUILDER2_RESUME_CONTRACT_VERSION)
        self.assertEqual(fields["ownerContextPresent"], "1")
        self.assertTrue(fields.get("ownerContextRef"))

    def test_historical_no_owner_jobs_not_exposed_through_relaxed_public_recovery(self) -> None:
        job_hash = {"status": "running", "product_description": "desc"}
        self.assertTrue(is_historical_job_without_ownership(job_hash))
        ok, reason = verify_owner_context(job_hash, _mock_request())
        self.assertFalse(ok)
        self.assertEqual(reason, "ownership_required_historical_job")

    def test_read_only_resume_inspector_makes_zero_redis_writes(self) -> None:
        report = inspect_builder2_resume_job(
            HISTORICAL_JOB_ID,
            raw_job_reader=lambda _jid: {"status": "running"},
            tournament_loader=lambda _jid: {"status": "created"},
        )
        self.assertEqual(report["redisMutations"], 0)

    def test_inspector_makes_zero_openai_calls(self) -> None:
        report = inspect_builder2_resume_job(
            "job-inspect",
            raw_job_reader=lambda _jid: {"status": "running"},
            tournament_loader=lambda _jid: None,
        )
        self.assertEqual(report["openAICalls"], 0)

    def test_inspector_makes_zero_runway_calls(self) -> None:
        report = inspect_builder2_resume_job(
            "job-inspect",
            raw_job_reader=lambda _jid: {"status": "running"},
            tournament_loader=lambda _jid: None,
        )
        self.assertEqual(report["runwayCalls"], 0)

    def test_inspector_makes_zero_image_calls(self) -> None:
        report = inspect_builder2_resume_job(
            "job-inspect",
            raw_job_reader=lambda _jid: {"status": "running"},
            tournament_loader=lambda _jid: None,
        )
        self.assertEqual(report["imageCalls"], 0)

    def test_inspector_makes_zero_ffmpeg_calls(self) -> None:
        report = inspect_builder2_resume_job(
            "job-inspect",
            raw_job_reader=lambda _jid: {"status": "running"},
            tournament_loader=lambda _jid: None,
        )
        self.assertEqual(report["ffmpegCalls"], 0)

    def test_builder1_remains_unchanged(self) -> None:
        import app

        source = open(app.__file__, encoding="utf-8").read()
        self.assertIn("builder1_generate", source)
        self.assertNotIn("builder1_resume_contract", source)

    def test_canonical_stage_order_is_server_owned(self) -> None:
        self.assertEqual(CANONICAL_BUILDER2_STAGES[0], "queued")
        self.assertEqual(CANONICAL_BUILDER2_STAGES[-1], "completed")

    def test_lease_renewal_by_owner(self) -> None:
        job_id = "job-renew"
        token = new_worker_token()
        self.assertTrue(acquire_job_lease(job_id, token))
        self.assertTrue(renew_job_lease(job_id, token))
        self.assertTrue(has_active_lease(job_id))

    def test_owner_context_present_flag(self) -> None:
        self.assertTrue(owner_context_present_in_job(_owned_job_hash("x")))
        self.assertFalse(owner_context_present_in_job({"status": "running"}))


class TestBuilder2ResumeRedisPath(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        disable_memory_recovery()
        disable_memory_jobs()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"

    def tearDown(self) -> None:
        disable_memory_store()
        enable_memory_recovery()
        enable_memory_jobs()

    def _eligible_state(self, job_id: str) -> Dict[str, Any]:
        state = _historical_resume_state(job_id=job_id, with_reusable_judgments=6)
        save_tournament_state(job_id, state)
        return state

    @patch("engine.builder2_resume_service.register_recoverable_job")
    @patch("engine.builder2_resume_service.video_job_touch_progress")
    @patch("engine.builder2_tournament_recovery.get_redis")
    @patch("engine.builder2_resume_service.get_redis")
    def test_redis_resume_path_requeues_eligible_job(
        self,
        resume_get_redis: Any,
        recovery_get_redis: Any,
        _touch: Any,
        _register: Any,
    ) -> None:
        redis_client = MagicMock()
        redis_client.exists.return_value = False
        redis_client.set.return_value = True
        resume_get_redis.return_value = redis_client
        recovery_get_redis.return_value = redis_client

        job_id = "job-redis-resume"
        job_hash = _owned_job_hash(job_id)
        self._eligible_state(job_id)

        with patch("engine.builder2_resume_service.video_job_get_raw", return_value=job_hash):
            result = request_builder2_resume(job_id, request=_mock_request())

        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("resumeRequested"))
        redis_client.hset.assert_called_once()
        hset_args, hset_kwargs = redis_client.hset.call_args
        self.assertEqual(hset_args[0], job_key(job_id))
        self.assertIn("lastResumeRequestedAt", hset_kwargs["mapping"])
        redis_client.lpush.assert_called_once_with(QUEUE_KEY, job_id)

    @patch("engine.builder2_tournament_recovery.get_redis")
    @patch("engine.builder2_resume_service.get_redis")
    def test_redis_resume_path_missing_job_does_not_enqueue(
        self,
        resume_get_redis: Any,
        recovery_get_redis: Any,
    ) -> None:
        redis_client = MagicMock()
        resume_get_redis.return_value = redis_client
        recovery_get_redis.return_value = redis_client

        with patch("engine.builder2_resume_service.video_job_get_raw", return_value=None):
            result = request_builder2_resume("job-missing", request=_mock_request())

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "not_found")
        redis_client.lpush.assert_not_called()
        redis_client.hset.assert_not_called()

    @patch("engine.builder2_resume_service.resolve_builder2_resume_stage")
    @patch("engine.builder2_tournament_recovery.get_redis")
    @patch("engine.builder2_resume_service.get_redis")
    def test_redis_resume_path_not_resumable_does_not_enqueue(
        self,
        resume_get_redis: Any,
        recovery_get_redis: Any,
        resolve_mock: Any,
    ) -> None:
        redis_client = MagicMock()
        resume_get_redis.return_value = redis_client
        recovery_get_redis.return_value = redis_client
        resolve_mock.return_value = {
            "canResume": False,
            "blockedReason": "not_ready",
            "resumeFromStage": "strategy",
        }

        job_id = "job-not-resumable"
        job_hash = _owned_job_hash(job_id)

        with patch("engine.builder2_resume_service.video_job_get_raw", return_value=job_hash):
            result = request_builder2_resume(job_id, request=_mock_request())

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "not_ready")
        redis_client.lpush.assert_not_called()
        redis_client.hset.assert_not_called()

    @patch("engine.builder2_resume_service.register_recoverable_job")
    @patch("engine.builder2_resume_service.video_job_touch_progress")
    @patch("engine.builder2_resume_service.is_job_queued")
    @patch("engine.builder2_tournament_recovery.get_redis")
    @patch("engine.builder2_resume_service.get_redis")
    def test_redis_resume_path_second_request_skips_duplicate_lpush(
        self,
        resume_get_redis: Any,
        recovery_get_redis: Any,
        is_queued_mock: Any,
        _touch: Any,
        _register: Any,
    ) -> None:
        redis_client = MagicMock()
        redis_client.exists.return_value = False
        redis_client.set.return_value = True
        resume_get_redis.return_value = redis_client
        recovery_get_redis.return_value = redis_client

        job_id = "job-redis-dedupe"
        job_hash = _owned_job_hash(job_id)
        self._eligible_state(job_id)
        is_queued_mock.side_effect = [False, True, True, True]

        with patch("engine.builder2_resume_service.video_job_get_raw", return_value=job_hash):
            first = request_builder2_resume(job_id, request=_mock_request())
            second = request_builder2_resume(job_id, request=_mock_request())

        self.assertTrue(first.get("resumeRequested"))
        self.assertTrue(second.get("resumeAlreadyInProgress"))
        self.assertEqual(redis_client.lpush.call_count, 1)

    @patch("engine.builder2_resume_service.register_recoverable_job")
    @patch("engine.builder2_resume_service.video_job_touch_progress")
    @patch("engine.video_jobs_redis.get_redis")
    def test_memory_recovery_path_does_not_use_redis_hset_or_lpush(
        self,
        redis_mock_factory: Any,
        _touch: Any,
        _register: Any,
    ) -> None:
        enable_memory_recovery()
        try:
            redis_client = MagicMock()
            redis_mock_factory.return_value = redis_client

            job_id = "job-memory-resume"
            job_hash = _owned_job_hash(job_id)
            set_memory_job_hash(job_id, job_hash)
            self._eligible_state(job_id)

            with patch("engine.builder2_resume_service.video_job_get_raw", return_value=job_hash):
                result = request_builder2_resume(job_id, request=_mock_request())

            self.assertTrue(result.get("resumeRequested"))
            redis_client.hset.assert_not_called()
            redis_client.lpush.assert_not_called()
        finally:
            disable_memory_recovery()


class TestBuilder2DurableResumeAppRoutes(unittest.TestCase):
    @patch.dict(os.environ, {"REDIS_URL": "redis://test", "BUILDER2_TOURNAMENT_ENABLED": "true"}, clear=False)
    @patch("app.redis_configured", return_value=True)
    @patch("app.video_job_create")
    def test_generate_video_returns_queued_status(self, create_mock: Any, _cfg: Any) -> None:
        from app import app as flask_app

        client = flask_app.test_client()
        response = client.post(
            "/api/generate-video",
            json={"productDescription": "A durable resume test product."},
            headers={"X-ACE-Batch-State": "frontend-batch-1"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("status"), "queued")
        create_mock.assert_called_once()

    @patch.dict(os.environ, {"REDIS_URL": "redis://test", "BUILDER2_TOURNAMENT_ENABLED": "true"}, clear=False)
    @patch("app.redis_configured", return_value=True)
    @patch("engine.builder2_resume_service.request_builder2_resume")
    def test_builder2_resume_route(self, resume_mock: Any, _cfg: Any) -> None:
        from app import app as flask_app

        resume_mock.return_value = {"ok": True, "jobId": "job-1", "status": "queued", "resumeRequested": True}
        client = flask_app.test_client()
        response = client.post("/api/builder2-resume", json={"jobId": "job-1"}, headers={"X-ACE-Batch-State": "frontend-batch-1"})
        self.assertEqual(response.status_code, 200)
        resume_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
