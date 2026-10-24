"""Isolated shutdown gate builder activity (5.19.8) — no live Redis or media jobs."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from engine import builder1_jobs_store
from engine import builder1_campaign_store
from engine import video_jobs_redis
from engine.builder2_tournament_recovery import acquire_job_lease, disable_memory_recovery, enable_memory_recovery
from engine.ace_local_dev_shutdown_builder_activity import observe_builder_shutdown_activity
from engine.ace_local_dev_shutdown_gate import (
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_BLOCKED,
    GATE_WAITING_FOR_OPERATION,
    evaluate_shutdown_gate_status,
    observe_active_local_dev_operations,
    refresh_local_shutdown_gate,
    request_local_shutdown,
)


class BuilderActivityTempLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ace-shutdown-builder-")
        self.ace_root = Path(self._tmp.name) / "ACE"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.sessions_root = self.ace_root / "_backups" / "sessions"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        (self.local_dev / "managed-processes.json").write_text(
            json.dumps(
                {
                    "launcherSession": "20260101-120000",
                    "components": {"worker": {"pid": os.getpid()}},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self._tmp.cleanup()


class TestBuilderShutdownActivity(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_redis_url = os.environ.pop("REDIS_URL", None)
        builder1_jobs_store.clear_memory_jobs_for_tests()
        builder1_campaign_store.clear_memory_store_for_tests()
        video_jobs_redis.disable_memory_jobs()
        disable_memory_recovery()
        self.layout = BuilderActivityTempLayout()

    def tearDown(self) -> None:
        self.layout.close()
        builder1_jobs_store.clear_memory_jobs_for_tests()
        builder1_campaign_store.clear_memory_store_for_tests()
        video_jobs_redis.disable_memory_jobs()
        disable_memory_recovery()
        if self._saved_redis_url is not None:
            os.environ["REDIS_URL"] = self._saved_redis_url

    def test_idle_no_builder_observations(self) -> None:
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            execute_busy=False,
        )
        builder_rows = [o for o in scan["observations"] if str(o.get("kind", "")).startswith("BUILDER")]
        self.assertEqual(builder_rows, [])
        status, _ = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_SAFE_TO_STOP)

    def test_builder1_running_blocks(self) -> None:
        jid = uuid.uuid4().hex
        builder1_jobs_store.create_builder1_job(
            job_id=jid,
            campaign_id="camp-1",
            target_ad_count=3,
        )
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
        )
        kinds = {o.get("kind") for o in scan["observations"]}
        self.assertIn("BUILDER1", kinds)
        status, reasons = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_WAITING_FOR_OPERATION)
        self.assertTrue(any("builder_active" in r for r in reasons))

    def test_builder2_queued_blocks(self) -> None:
        video_jobs_redis.enable_memory_jobs()
        video_jobs_redis.video_job_create(
            "job-q",
            "product",
            "desc",
            "http://localhost/",
        )
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
        )
        self.assertTrue(any(o.get("kind") == "BUILDER2" for o in scan["observations"]))
        status, _ = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_WAITING_FOR_OPERATION)

    def test_worker_lease_busy(self) -> None:
        enable_memory_recovery()
        acquire_job_lease("lease-job", "worker-token")
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
        )
        self.assertTrue(any(o.get("kind") == "BUILDER2_WORKER" for o in scan["observations"]))

    def test_redis_unreachable_blocks(self) -> None:
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            redis_reachable=False,
        )
        self.assertFalse(scan.get("redisReachable"))
        status, reasons = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_SHUTDOWN_BLOCKED)
        self.assertIn("redis_unreachable", reasons)

    def test_worker_unreachable_with_pending_work_blocks(self) -> None:
        override = {
            "observations": [
                {
                    "kind": "BUILDER2_QUEUE",
                    "status": "ACTIVE",
                    "notes": ["queue_depth:1"],
                }
            ],
            "redisReachable": True,
            "workerReachable": False,
            "notes": [],
        }
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            builder_activity_override=override,
        )
        status, reasons = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_SHUTDOWN_BLOCKED)
        self.assertIn("worker_unreachable", reasons)

    def test_unknown_builder_status_blocks(self) -> None:
        override = {
            "observations": [
                {"kind": "BUILDER2", "status": "UNKNOWN", "jobId": "x", "notes": ["builder2_status:mystery"]}
            ],
            "redisReachable": True,
            "workerReachable": True,
            "notes": [],
        }
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            builder_activity_override=override,
        )
        status, _ = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_SHUTDOWN_BLOCKED)

    def test_terminal_builder_job_allows_safe(self) -> None:
        override = {
            "observations": [
                {"kind": "BUILDER1", "status": "IDLE", "jobId": "done-job", "notes": []},
            ],
            "redisReachable": True,
            "workerReachable": True,
            "notes": [],
        }
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            builder_activity_override=override,
        )
        blocking = [o for o in scan["observations"] if o.get("status") in ("ACTIVE", "UNKNOWN")]
        self.assertEqual(blocking, [])
        status, _ = evaluate_shutdown_gate_status(
            scan["observations"],
            shutdown_requested=True,
            scan_meta=scan,
        )
        self.assertEqual(status, GATE_SAFE_TO_STOP)

    def test_work_finishing_during_wait_becomes_safe(self) -> None:
        jid = uuid.uuid4().hex
        builder1_jobs_store.create_builder1_job(job_id=jid, campaign_id="c1", target_ad_count=1)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        waiting = refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session="20260101-120000",
            sessions_root=self.layout.sessions_root,
        )
        self.assertEqual(waiting["gateStatus"], GATE_WAITING_FOR_OPERATION)
        builder1_jobs_store.update_builder1_job(jid, status="done", stage="done")
        safe = refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session="20260101-120000",
            sessions_root=self.layout.sessions_root,
        )
        self.assertEqual(safe["gateStatus"], GATE_SAFE_TO_STOP)

    def test_new_work_after_safe_detected_on_refresh(self) -> None:
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        safe = refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session="20260101-120000",
            sessions_root=self.layout.sessions_root,
        )
        self.assertEqual(safe["gateStatus"], GATE_SAFE_TO_STOP)
        builder1_jobs_store.create_builder1_job(
            job_id=uuid.uuid4().hex,
            campaign_id="late",
            target_ad_count=2,
        )
        waiting = refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session="20260101-120000",
            sessions_root=self.layout.sessions_root,
        )
        self.assertEqual(waiting["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_observe_builder_module_classify_running(self) -> None:
        jid = uuid.uuid4().hex
        builder1_jobs_store.create_builder1_job(job_id=jid, campaign_id="c2", target_ad_count=1)
        block = observe_builder_shutdown_activity(ace_root=self.layout.ace_root)
        active = [o for o in block["observations"] if o.get("status") == "ACTIVE"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["kind"], "BUILDER1")


if __name__ == "__main__":
    unittest.main()
