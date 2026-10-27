"""Jobs halted by intentional_local_quit stay stopped after LOCAL is disconnected."""
from __future__ import annotations

import os
import unittest

from engine.ace_local_dev_config import reset_local_dev_config_for_tests
from engine.builder1_job_cancellation import (
    CANCEL_REASON_INTENTIONAL_LOCAL_QUIT,
    Builder1JobCancelledError,
    checkpoint_builder1_cancellation,
    should_block_builder1_execution,
)
from engine.builder_client_binding import strip_client_binding_fields
from engine.builder1_jobs_store import clear_memory_jobs_for_tests, create_builder1_job, get_builder1_job, update_builder1_job
from engine.builder1_paid_provider import run_paid_provider_call
from engine.builder1_paid_stage_guard import builder1_paid_stage_context
from engine.builder1_planning_resume import assess_builder1_planning_resume
from engine.builder2_job_cancellation import (
    accept_dequeued_builder2_job,
    checkpoint_builder2_cancellation,
    should_block_builder2_execution,
)
from engine.builder2_resume_service import request_builder2_resume
from engine.builder2_tournament_recovery import (
    disable_memory_recovery,
    enable_memory_recovery,
    requeue_recoverable_job,
)
from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash


class IntentionalLocalQuitWithoutLocalRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._redis = os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_LOCAL_DEV_ENABLED", None)
        reset_local_dev_config_for_tests()
        clear_memory_jobs_for_tests()
        enable_memory_jobs()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_jobs()
        disable_memory_recovery()
        clear_memory_jobs_for_tests()
        reset_local_dev_config_for_tests()
        if self._redis is None:
            os.environ.pop("REDIS_URL", None)
        else:
            os.environ["REDIS_URL"] = self._redis

    def test_builder1_quit_marker_blocks_paid_dispatch_without_local_modules(self) -> None:
        create_builder1_job(job_id="b1-quit", campaign_id="c1", target_ad_count=1)
        update_builder1_job(
            "b1-quit",
            haltRequested="1",
            haltReason=CANCEL_REASON_INTENTIONAL_LOCAL_QUIT,
            localStopStatus="halt_requested",
            status="cancelled",
        )
        self.assertTrue(should_block_builder1_execution("b1-quit"))
        with self.assertRaises(Builder1JobCancelledError):
            checkpoint_builder1_cancellation("b1-quit", campaign_id="c1", stage="paid")
        called = {"n": 0}

        def _dispatch() -> str:
            called["n"] += 1
            return "vendor"

        with builder1_paid_stage_context(job_id="b1-quit", campaign_id="c1"):
            with self.assertRaises(Builder1JobCancelledError):
                run_paid_provider_call("image", _dispatch)
        self.assertEqual(called["n"], 0)
        report = assess_builder1_planning_resume("b1-quit")
        self.assertIn("job_cancelled", report.get("rejectionReasons") or [])

    def test_builder2_quit_marker_blocks_resume_dequeue_and_recovery(self) -> None:
        set_memory_job_hash(
            "b2-quit",
            {
                "status": "cancelled",
                "cancelRequested": "1",
                "haltRequested": "1",
                "haltReason": CANCEL_REASON_INTENTIONAL_LOCAL_QUIT,
                "localStopStatus": "halt_requested",
                "builder": "builder2",
                "canResume": "0",
            },
        )
        self.assertTrue(should_block_builder2_execution("b2-quit"))
        with self.assertRaises(Exception):
            checkpoint_builder2_cancellation("b2-quit", stage="worker_execution_start")
        accepted = accept_dequeued_builder2_job("b2-quit")
        self.assertFalse(accepted.get("accepted"))
        self.assertEqual(accepted.get("reason"), "cancelled_or_halted")
        resumed = request_builder2_resume("b2-quit")
        self.assertFalse(resumed.get("ok"))
        self.assertEqual(resumed.get("error"), "builder2_job_cancelled")
        self.assertFalse(resumed.get("canResume"))
        self.assertFalse(requeue_recoverable_job("b2-quit"))

    def test_forged_binding_fields_stripped_without_local_modules(self) -> None:
        cleaned = strip_client_binding_fields(
            {
                "ownerContextRef": "owner-1",
                "localDevInstanceId": "forged-instance",
                "localDevBound": "1",
                "localDevStartGeneration": "9",
                "environment": "local",
                "aceEnvironment": "local",
                "localScope": "dev",
            }
        )
        self.assertEqual(cleaned.get("ownerContextRef"), "owner-1")
        for forged in (
            "localDevInstanceId",
            "localDevBound",
            "localDevStartGeneration",
            "environment",
            "aceEnvironment",
            "localScope",
        ):
            self.assertNotIn(forged, cleaned)
        create_builder1_job(
            job_id="b1-forge",
            campaign_id="c-forge",
            target_ad_count=1,
            ownership_fields={
                "ownerContextRef": "owner-1",
                "localDevInstanceId": "forged-instance",
                "localDevBound": "1",
            },
        )
        stored = get_builder1_job("b1-forge") or {}
        self.assertEqual(stored.get("ownerContextRef"), "owner-1")
        self.assertNotIn("localDevInstanceId", stored)
        self.assertNotIn("localDevBound", stored)


if __name__ == "__main__":
    unittest.main()
