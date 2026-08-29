"""
Builder1 request idempotency — duplicate mutation protection.
"""
from __future__ import annotations

import json
import os
import threading
import unittest
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder1_campaign_store import clear_memory_store_for_tests
from engine.builder1_job_ownership import extract_owner_context_from_request, ownership_fields_for_builder1_create
from engine.builder1_jobs_store import clear_memory_jobs_for_tests, create_builder1_job, get_builder1_job
from engine.builder1_request_idempotency import (
    OPERATION_GENERATE_NEXT,
    OPERATION_INITIAL_GENERATE,
    OPERATION_REPAIR_PHYSICAL,
    OPERATION_RETRY_IMAGE,
    SUBMISSION_CLAIM_LEASE_SECONDS,
    begin_builder1_idempotent_request,
    claim_submission_lease,
    clear_memory_idempotency_for_tests,
    execution_is_proven,
    finalize_idempotent_worker_dispatch,
    fingerprint_generate_next,
    fingerprint_initial_generate,
    fingerprint_repair_physical,
    fingerprint_retry_image,
    mark_builder1_worker_started,
    replay_response_from_record,
    should_recover_stale_submission,
    submission_claim_is_live,
)


def _mock_request(*, batch: str = "batch-a", auth: str = "Bearer token-a", request_id: str = "") -> MagicMock:
    headers = {"X-ACE-Batch-State": batch, "Authorization": auth}
    if request_id:
        headers["X-ACE-Request-Id"] = request_id
    req = MagicMock()
    req.headers = headers
    return req


class TestIdempotencyStore(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_idempotency_for_tests()
        clear_memory_jobs_for_tests()

    def test_first_request_creates_record(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate(
            {"productName": "P", "productDescription": "D", "format": "portrait", "adCount": 2},
            ad_count=2,
        )
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-1",
            campaign_id="camp-1",
        )
        self.assertEqual(begin.kind, "new")
        self.assertEqual(begin.record["jobId"], "job-1")

    def test_replay_returns_same_job(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        first = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-1",
            campaign_id="camp-1",
        )
        response = {"jobId": "job-1", "campaignId": "camp-1", "pollUrl": "/status"}
        claim_submission_lease(first.key, response=response)
        create_builder1_job(job_id="job-1", campaign_id="camp-1", target_ad_count=2, stage="planning")
        mark_builder1_worker_started("job-1", idempotency_key=first.key)
        second = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-OTHER",
            campaign_id="camp-OTHER",
        )
        self.assertEqual(second.kind, "replay")
        replay = replay_response_from_record(second.record)
        self.assertTrue(replay.get("idempotentReplay"))
        self.assertEqual(replay["jobId"], "job-1")

    def test_worker_submit_claimed_once(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-1",
            campaign_id="camp-1",
        )
        submits: List[bool] = []

        def _try_submit() -> None:
            claimed, _ = claim_submission_lease(begin.key, response={"jobId": "job-1"})
            submits.append(claimed)

        threads = [threading.Thread(target=_try_submit) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for x in submits if x), 1)

    def test_concurrent_begin_one_winner(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        results: List[str] = []
        lock = threading.Lock()

        def _worker(idx: int) -> None:
            begin = begin_builder1_idempotent_request(
                operation=OPERATION_INITIAL_GENERATE,
                request_id=rid,
                owner_context_ref="owner-a",
                request_fingerprint=fp,
                job_id=f"job-{idx}",
                campaign_id=f"camp-{idx}",
            )
            with lock:
                results.append(str(begin.record.get("jobId")))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(results)), 1)

    def test_same_request_id_different_payload_conflict(self) -> None:
        rid = str(uuid.uuid4())
        fp1 = fingerprint_initial_generate({"productDescription": "A"}, ad_count=2)
        fp2 = fingerprint_initial_generate({"productDescription": "B"}, ad_count=2)
        begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp1,
            job_id="job-1",
            campaign_id="camp-1",
        )
        second = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp2,
            job_id="job-2",
            campaign_id="camp-2",
        )
        self.assertEqual(second.kind, "conflict")

    def test_owner_isolation(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        a = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-a",
            campaign_id="camp-a",
        )
        b = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-b",
            request_fingerprint=fp,
            job_id="job-b",
            campaign_id="camp-b",
        )
        self.assertNotEqual(a.record["jobId"], b.record["jobId"])

    def test_recover_after_reserved_before_job(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        first = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-recover",
            campaign_id="camp-recover",
        )
        second = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-other",
            campaign_id="camp-other",
        )
        self.assertIn(second.kind, {"recover", "replay"})
        self.assertEqual(second.record["jobId"], first.record["jobId"])

    def test_generate_next_fingerprint_scoped(self) -> None:
        fp1 = fingerprint_generate_next(campaign_id="c1", expected_next_index=2)
        fp2 = fingerprint_generate_next(campaign_id="c1", expected_next_index=3)
        self.assertNotEqual(fp1, fp2)

    def test_retry_image_fingerprint(self) -> None:
        fp = fingerprint_retry_image(campaign_id="c1", retry_ad_index=2)
        self.assertTrue(fp)

    def test_repair_physical_fingerprint(self) -> None:
        fp = fingerprint_repair_physical(campaign_id="c1", retry_ad_index=1)
        self.assertTrue(fp)


class TestSubmissionLeaseRecovery(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_idempotency_for_tests()
        clear_memory_jobs_for_tests()

    def test_crash_after_claim_before_submit_recovers_same_job(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-crash",
            campaign_id="camp-crash",
        )
        response = {"jobId": "job-crash", "campaignId": "camp-crash", "status": "running"}
        claim_submission_lease(begin.key, response=response)
        create_builder1_job(job_id="job-crash", campaign_id="camp-crash", target_ad_count=2, stage="planning")

        with patch(
            "engine.builder1_request_idempotency.SUBMISSION_CLAIM_LEASE_SECONDS",
            0,
        ):
            self.assertTrue(should_recover_stale_submission(_load(begin.key), "job-crash"))

            submits: List[int] = []

            def _submit() -> None:
                submits.append(1)

            body, is_replay = finalize_idempotent_worker_dispatch(
                idem_key=begin.key,
                idem_record=begin.record,
                job_id="job-crash",
                response_body=response,
                submit_fn=_submit,
            )
        self.assertFalse(is_replay)
        self.assertEqual(body["jobId"], "job-crash")
        self.assertEqual(submits, [1])

    def test_worker_started_prevents_duplicate_submission(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-live",
            campaign_id="camp-live",
        )
        create_builder1_job(job_id="job-live", campaign_id="camp-live", target_ad_count=2, stage="planning")
        mark_builder1_worker_started("job-live", idempotency_key=begin.key)
        submits: List[int] = []

        body, is_replay = finalize_idempotent_worker_dispatch(
            idem_key=begin.key,
            idem_record=begin.record,
            job_id="job-live",
            response_body={"jobId": "job-live", "campaignId": "camp-live"},
            submit_fn=lambda: submits.append(1),
        )
        self.assertTrue(is_replay)
        self.assertEqual(submits, [])
        self.assertTrue(execution_is_proven(_load(begin.key), "job-live"))

    def test_live_claim_cannot_be_stolen(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-live-claim",
            campaign_id="camp-live-claim",
        )
        first_claim, _ = claim_submission_lease(begin.key, response={"jobId": "job-live-claim"})
        second_claim, _ = claim_submission_lease(begin.key, response={"jobId": "job-live-claim"})
        self.assertTrue(first_claim)
        self.assertFalse(second_claim)
        self.assertTrue(submission_claim_is_live(_load(begin.key)))

    def test_paid_stage_outcome_unknown_blocks_auto_resubmit(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-paid",
            campaign_id="camp-paid",
        )
        create_builder1_job(
            job_id="job-paid",
            campaign_id="camp-paid",
            target_ad_count=2,
            stage="planning",
        )
        from engine.builder1_jobs_store import update_builder1_job

        update_builder1_job(
            "job-paid",
            lastPaidStage="openai_image_generation",
            lastPaidStageStatus="outcome_unknown",
            status="error",
            error="paid_stage_outcome_unknown",
        )
        with patch("engine.builder1_request_idempotency.SUBMISSION_CLAIM_LEASE_SECONDS", 0):
            submits: List[int] = []
            body, is_replay = finalize_idempotent_worker_dispatch(
                idem_key=begin.key,
                idem_record=begin.record,
                job_id="job-paid",
                response_body={"jobId": "job-paid", "campaignId": "camp-paid"},
                submit_fn=lambda: submits.append(1),
            )
        self.assertTrue(is_replay)
        self.assertEqual(submits, [])

    def test_executor_enqueued_without_worker_start_recovers(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-enqueued",
            campaign_id="camp-enqueued",
        )
        create_builder1_job(job_id="job-enqueued", campaign_id="camp-enqueued", target_ad_count=2, stage="planning")
        claim_submission_lease(begin.key, response={"jobId": "job-enqueued"})
        from engine.builder1_request_idempotency import mark_executor_enqueued

        mark_executor_enqueued(begin.key)
        with patch("engine.builder1_request_idempotency.SUBMISSION_CLAIM_LEASE_SECONDS", 0):
            submits: List[int] = []
            body, is_replay = finalize_idempotent_worker_dispatch(
                idem_key=begin.key,
                idem_record=begin.record,
                job_id="job-enqueued",
                response_body={"jobId": "job-enqueued", "campaignId": "camp-enqueued"},
                submit_fn=lambda: submits.append(1),
            )
        self.assertFalse(is_replay)
        self.assertEqual(body["jobId"], "job-enqueued")
        self.assertEqual(submits, [1])

    def test_concurrent_replay_one_worker_path(self) -> None:
        rid = str(uuid.uuid4())
        fp = fingerprint_initial_generate({"productDescription": "D"}, ad_count=2)
        begin = begin_builder1_idempotent_request(
            operation=OPERATION_INITIAL_GENERATE,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id="job-conc",
            campaign_id="camp-conc",
        )
        create_builder1_job(job_id="job-conc", campaign_id="camp-conc", target_ad_count=2, stage="planning")
        submits: List[int] = []
        lock = threading.Lock()

        def _submit() -> None:
            with lock:
                submits.append(1)

        def _dispatch() -> None:
            finalize_idempotent_worker_dispatch(
                idem_key=begin.key,
                idem_record=begin.record,
                job_id="job-conc",
                response_body={"jobId": "job-conc", "campaignId": "camp-conc"},
                submit_fn=_submit,
            )

        threads = [threading.Thread(target=_dispatch) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(submits, [1])

    def _operation_recovery_same_job(self, operation: str, fp: str, job_id: str, campaign_id: str) -> None:
        rid = str(uuid.uuid4())
        begin = begin_builder1_idempotent_request(
            operation=operation,
            request_id=rid,
            owner_context_ref="owner-a",
            request_fingerprint=fp,
            job_id=job_id,
            campaign_id=campaign_id,
        )
        claim_submission_lease(begin.key, response={"jobId": job_id, "campaignId": campaign_id})
        create_builder1_job(job_id=job_id, campaign_id=campaign_id, target_ad_count=2, stage="planning")
        with patch("engine.builder1_request_idempotency.SUBMISSION_CLAIM_LEASE_SECONDS", 0):
            submits: List[int] = []
            body, is_replay = finalize_idempotent_worker_dispatch(
                idem_key=begin.key,
                idem_record=begin.record,
                job_id=job_id,
                response_body={"jobId": job_id, "campaignId": campaign_id},
                submit_fn=lambda: submits.append(1),
            )
        self.assertFalse(is_replay)
        self.assertEqual(body["jobId"], job_id)
        self.assertEqual(body["campaignId"], campaign_id)
        self.assertEqual(submits, [1])

    def test_generate_next_same_guarantees(self) -> None:
        fp = fingerprint_generate_next(campaign_id="camp-next", expected_next_index=2)
        self._operation_recovery_same_job(
            OPERATION_GENERATE_NEXT, fp, "job-next", "camp-next",
        )

    def test_retry_image_same_guarantees(self) -> None:
        fp = fingerprint_retry_image(campaign_id="camp-retry", retry_ad_index=1)
        self._operation_recovery_same_job(
            OPERATION_RETRY_IMAGE, fp, "job-retry", "camp-retry",
        )

    def test_repair_physical_same_guarantees(self) -> None:
        fp = fingerprint_repair_physical(campaign_id="camp-repair", retry_ad_index=1)
        self._operation_recovery_same_job(
            OPERATION_REPAIR_PHYSICAL, fp, "job-repair", "camp-repair",
        )

    def test_submission_claim_lease_default_is_bounded(self) -> None:
        self.assertEqual(SUBMISSION_CLAIM_LEASE_SECONDS, 120)


def _load(key: str) -> Dict[str, Any]:
    from engine.builder1_request_idempotency import _load_record

    return dict(_load_record(key) or {})


class TestIdempotencyFlaskRoutes(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_idempotency_for_tests()
        clear_memory_jobs_for_tests()
        clear_memory_store_for_tests()

    @patch.dict(os.environ, {"BUILDER1_REQUEST_ID_REQUIRED": "true"}, clear=False)
    def test_missing_request_id_rejected_in_production_mode(self) -> None:
        import app as app_module

        client = app_module.app.test_client()
        resp = client.post(
            "/api/builder1-generate",
            json={"productDescription": "desc", "adCount": 2},
            headers={"X-ACE-Batch-State": "b1", "Authorization": "Bearer x"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json().get("error"), "builder1_request_id_required")

    @patch.dict(os.environ, {}, clear=False)
    def test_generate_replay_via_flask(self) -> None:
        import app as app_module

        os.environ.pop("BUILDER1_PRODUCTION_MODE", None)
        os.environ.pop("BUILDER1_REQUEST_ID_REQUIRED", None)
        submit_calls: Dict[str, int] = {"count": 0}
        original_submit = app_module._builder1_executor.submit

        def _counting_submit(*args: Any, **kwargs: Any) -> Any:
            submit_calls["count"] += 1
            return original_submit(*args, **kwargs)

        rid = str(uuid.uuid4())
        headers = {
            "X-ACE-Batch-State": "idem-batch",
            "Authorization": "Bearer idem",
            "X-ACE-Request-Id": rid,
        }
        client = app_module.app.test_client()
        with patch.object(app_module._builder1_executor, "submit", side_effect=_counting_submit):
            with patch.object(app_module, "_builder1_run_initial_job", return_value=None):
                first = client.post(
                    "/api/builder1-generate",
                    json={"productDescription": "desc", "productName": "P", "adCount": 2},
                    headers=headers,
                )
                second = client.post(
                    "/api/builder1-generate",
                    json={"productDescription": "desc", "productName": "P", "adCount": 2},
                    headers=headers,
                )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        body1 = first.get_json()
        body2 = second.get_json()
        self.assertEqual(body1.get("jobId"), body2.get("jobId"))
        self.assertEqual(body1.get("campaignId"), body2.get("campaignId"))
        self.assertTrue(body2.get("idempotentReplay"))
        self.assertEqual(submit_calls["count"], 1)


class TestBuilder2UnchangedIdempotency(unittest.TestCase):
    def test_builder2_cancel_still_rejects_builder1_job(self) -> None:
        from engine.builder2_job_cancellation import request_builder2_job_cancellation
        from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash

        enable_memory_jobs()
        try:
            set_memory_job_hash("b1-like", {"status": "queued"})
            result = request_builder2_job_cancellation("b1-like")
            self.assertEqual(result.get("error"), "not_builder2_job")
        finally:
            disable_memory_jobs()


if __name__ == "__main__":
    unittest.main()
