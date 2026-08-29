"""
Builder1 production reliability — cancellation, ownership, stale lock, artifacts.
"""
from __future__ import annotations

import base64
import io
import os
import time
import unittest
import zipfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    persist_campaign_ad_artifact,
    reserve_next_ad_index,
    try_recover_stale_generation_lock,
    validate_next_ad_request,
)
from engine.builder1_campaign_completion import evaluate_campaign_completion
from engine.builder1_image_artifact_store import (
    read_builder1_image_artifact_bytes,
    write_builder1_image_artifact_bytes,
    ad_artifact_record,
)
from engine.builder1_job_cancellation import (
    CANCEL_REASON_FRONTEND_REFRESH,
    Builder1JobCancelledError,
    checkpoint_builder1_cancellation,
    finalize_builder1_job_respecting_cancellation,
    is_builder1_campaign_cancelled,
    is_builder1_job_cancelled,
    request_builder1_job_cancellation,
)
from engine.builder1_job_ownership import (
    extract_owner_context_from_request,
    ownership_fields_for_builder1_create,
    verify_owner_context,
)
from engine.builder1_jobs_store import (
    clear_memory_jobs_for_tests,
    create_builder1_job,
    get_builder1_job,
    is_builder1_job_stale,
    try_finalize_stale_builder1_job,
    update_builder1_job,
)
from engine.builder1_marketing_placeholders import (
    has_builder1_marketing_placeholder_residue,
    sanitize_builder1_marketing_placeholder_residue,
    validate_builder1_marketing_text_hygiene,
)
from engine.builder1_paid_stage_guard import builder1_paid_stage_context, checkpoint_before_paid_call
from engine.builder1_production_config import assert_builder1_production_ready, builder1_production_mode_enabled
from engine.builder1_zip import build_builder1_zip_from_campaign_session
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_series import _base_campaign, _parse


def _mock_request(*, batch: str = "batch-a", auth: str = "Bearer token-a") -> MagicMock:
    req = MagicMock()
    req.headers = {"X-ACE-Batch-State": batch, "Authorization": auth}
    return req


def _owned_job(job_id: str, campaign_id: str) -> Dict[str, str]:
    fields = ownership_fields_for_builder1_create(
        _mock_request(),
        {"productName": "P", "productDescription": "D"},
    )
    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=2,
        ownership_fields=fields,
    )
    return fields


class TestBuilder1Cancellation(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_cancel_running_job(self) -> None:
        job_id = "job-cancel-run"
        campaign_id = "camp-cancel"
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id=campaign_id, plan=plan, target_ad_count=2)
        _owned_job(job_id, campaign_id)
        result = request_builder1_job_cancellation(job_id, reason=CANCEL_REASON_FRONTEND_REFRESH)
        self.assertEqual(result["outcome"], "cancelled")
        self.assertTrue(is_builder1_job_cancelled(job_id))
        self.assertTrue(is_builder1_campaign_cancelled(campaign_id))

    def test_cancel_idempotent(self) -> None:
        job_id = "job-cancel-idem"
        _owned_job(job_id, "camp-idem")
        first = request_builder1_job_cancellation(job_id)
        second = request_builder1_job_cancellation(job_id)
        self.assertEqual(first["outcome"], "cancelled")
        self.assertEqual(second["outcome"], "already_cancelled")

    def test_cancel_already_completed(self) -> None:
        job_id = "job-done"
        _owned_job(job_id, "camp-done")
        update_builder1_job(job_id, status="done")
        result = request_builder1_job_cancellation(job_id)
        self.assertEqual(result["outcome"], "already_completed")

    def test_checkpoint_before_paid_raises(self) -> None:
        job_id = "job-checkpoint"
        _owned_job(job_id, "camp-checkpoint")
        request_builder1_job_cancellation(job_id)
        with builder1_paid_stage_context(job_id=job_id, campaign_id="camp-checkpoint"):
            with self.assertRaises(Builder1JobCancelledError):
                checkpoint_before_paid_call("strategy_stage")

    def test_cancel_wins_completion_race(self) -> None:
        job_id = "job-race"
        campaign_id = "camp-race"
        _owned_job(job_id, campaign_id)
        request_builder1_job_cancellation(job_id)
        wrote = finalize_builder1_job_respecting_cancellation(
            job_id,
            {"ok": True, "campaignId": campaign_id, "generatedCount": 1},
            target_ad_count=2,
        )
        self.assertFalse(wrote)
        self.assertEqual(get_builder1_job(job_id).get("status"), "cancelled")

    def test_cancelled_campaign_blocks_generate_next(self) -> None:
        job_id = "job-x"
        campaign_id = "camp-block"
        _owned_job(job_id, campaign_id)
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id=campaign_id, plan=plan, target_ad_count=2)
        request_builder1_job_cancellation(job_id)
        with self.assertRaises(CampaignStoreError) as ctx:
            validate_next_ad_request(campaign_id, 1)
        self.assertEqual(ctx.exception.code, "builder1_campaign_cancelled")


class TestStaleLockRecovery(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_stale_lock_cleared_when_owner_job_stale(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="camp-stale", plan=plan, target_ad_count=2)
        job_id = "owner-stale"
        _owned_job(job_id, "camp-stale")
        session = reserve_next_ad_index("camp-stale", 1, job_id=job_id)
        self.assertEqual(session.generating_index, 1)
        update_builder1_job(job_id, lastHeartbeatAt=time.time() - 10_000, status="running")
        from engine.builder1_campaign_store import _load_raw, _save_raw

        data = dict(_load_raw("camp-stale") or {})
        data["generatingLockAcquiredAt"] = time.time() - 10_000
        data["generatingLockHeartbeatAt"] = time.time() - 10_000
        _save_raw("camp-stale", data)
        self.assertTrue(is_builder1_job_stale(get_builder1_job(job_id) or {}))
        self.assertTrue(try_recover_stale_generation_lock("camp-stale"))
        self.assertIsNone(get_campaign_session("camp-stale").generating_index)

    def test_live_lock_not_stolen(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="camp-live", plan=plan, target_ad_count=2)
        job_id = "owner-live"
        _owned_job(job_id, "camp-live")
        reserve_next_ad_index("camp-live", 1, job_id=job_id)
        update_builder1_job(job_id, lastHeartbeatAt=time.time())
        self.assertFalse(try_recover_stale_generation_lock("camp-live"))
        self.assertEqual(get_campaign_session("camp-live").generating_index, 1)


class TestDurableArtifactsAndZip(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_artifact_survives_job_loss_and_zip_rebuild(self) -> None:
        plan_data = _base_campaign(2)
        for ad in plan_data["ads"]:
            ad["marketingText"] = marketing_text_words(50, prefix=f"w{ad['index']}")
        plan = _parse(plan_data, 2)
        create_campaign_session(campaign_id="camp-art", plan=plan, target_ad_count=2)
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 64
        for idx in (1, 2):
            rec = ad_artifact_record(campaign_id="camp-art", ad_index=idx, plan_revision=1)
            write_builder1_image_artifact_bytes(rec["token"], image_bytes)
            persist_campaign_ad_artifact("camp-art", ad_index=idx, artifact=rec)
            reserve_next_ad_index("camp-art", idx, job_id=f"job-{idx}")
            mark_ad_generated("camp-art", idx)
        clear_memory_jobs_for_tests()
        session = get_campaign_session("camp-art")
        zbytes = build_builder1_zip_from_campaign_session(session)
        with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
            names = sorted(zf.namelist())
            self.assertIn("ad-01.jpg", names)
            self.assertIn("ad-02.jpg", names)
            self.assertIn("campaign.txt", names)
        self.assertTrue(read_builder1_image_artifact_bytes(
            (session.ad_artifacts or {}).get("1", {}).get("token", "")
        ))


class TestOwnership(unittest.TestCase):
    def test_mismatch_denied(self) -> None:
        fields = ownership_fields_for_builder1_create(
            _mock_request(batch="a"),
            {"productName": "P", "productDescription": "D"},
        )
        ok, err = verify_owner_context(fields, _mock_request(batch="b"))
        self.assertFalse(ok)
        self.assertEqual(err, "ownership_mismatch")

    def test_match_allowed(self) -> None:
        req = _mock_request(batch="same")
        fields = ownership_fields_for_builder1_create(req, {"productName": "P", "productDescription": "D"})
        ok, err = verify_owner_context(fields, req)
        self.assertTrue(ok)
        self.assertIsNone(err)


class TestMarketingPlaceholders(unittest.TestCase):
    def test_placeholder_detected_and_sanitized(self) -> None:
        text = marketing_text_words(48, prefix="w") + " (המוצר הזה)"
        self.assertTrue(has_builder1_marketing_placeholder_residue(text))
        cleaned = sanitize_builder1_marketing_placeholder_residue(text)
        self.assertFalse(has_builder1_marketing_placeholder_residue(cleaned))

    def test_exact_fifty_preserved(self) -> None:
        text = marketing_text_words(50)
        validate_builder1_marketing_text_hygiene(text)


class TestCampaignCompletionGate(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_not_ready_until_all_artifacts(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        session = create_campaign_session(campaign_id="camp-ready", plan=plan, target_ad_count=2)
        report = evaluate_campaign_completion(session)
        self.assertFalse(report["campaignReady"])


class TestProductionRedisRequirement(unittest.TestCase):
    def test_production_requires_redis(self) -> None:
        with patch.dict(os.environ, {"BUILDER1_PRODUCTION_MODE": "true", "REDIS_URL": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                assert_builder1_production_ready()


class TestPaidOutcomeUnknown(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_jobs_for_tests()

    def test_transport_timeout_becomes_outcome_unknown_zero_retry(self) -> None:
        from engine.builder1_paid_provider import PaidStageOutcomeUnknownError, PaidStageRetryBlockedError, run_paid_provider_call

        job_id = "job-timeout"
        _owned_job(job_id, "camp-timeout")
        calls = {"count": 0}

        def _dispatch() -> str:
            calls["count"] += 1
            raise TimeoutError("request timed out")

        with builder1_paid_stage_context(job_id=job_id, campaign_id="camp-timeout"):
            with self.assertRaises(PaidStageOutcomeUnknownError):
                run_paid_provider_call("openai_image_generation", _dispatch)
            with self.assertRaises(PaidStageOutcomeUnknownError):
                run_paid_provider_call("openai_image_generation", _dispatch)

        self.assertEqual(calls["count"], 1)
        job = get_builder1_job(job_id) or {}
        self.assertEqual(job.get("lastPaidStageStatus"), "outcome_unknown")
        self.assertFalse(job.get("retryable", True))

    def test_connection_reset_becomes_outcome_unknown(self) -> None:
        from engine.builder1_paid_provider import PaidStageOutcomeUnknownError, run_paid_provider_call

        job_id = "job-reset"
        _owned_job(job_id, "camp-reset")

        def _dispatch() -> str:
            raise ConnectionResetError("connection reset by peer")

        with builder1_paid_stage_context(job_id=job_id):
            with self.assertRaises(PaidStageOutcomeUnknownError):
                run_paid_provider_call("strategy_stage", _dispatch)

        self.assertEqual(get_builder1_job(job_id).get("lastPaidStageStatus"), "outcome_unknown")

    def test_pre_submit_failure_allows_bounded_retry(self) -> None:
        from engine.builder1_paid_provider import run_paid_provider_call

        job_id = "job-pre"
        _owned_job(job_id, "camp-pre")
        calls = {"count": 0}

        def _pre_submit() -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("openai_unconfigured")

        def _dispatch() -> str:
            return "ok"

        with builder1_paid_stage_context(job_id=job_id):
            with self.assertRaises(ValueError):
                run_paid_provider_call("strategy_stage", _dispatch, pre_submit=_pre_submit)
            result = run_paid_provider_call("strategy_stage", _dispatch, pre_submit=_pre_submit)

        self.assertEqual(result, "ok")
        self.assertEqual(get_builder1_job(job_id).get("lastPaidStageStatus"), "succeeded")

    def test_reasoning_unknown_outcome_blocks_retry(self) -> None:
        from engine.builder1_paid_provider import PaidStageOutcomeUnknownError, run_paid_provider_call

        job_id = "job-reason"
        _owned_job(job_id, "camp-reason")

        def _timeout() -> str:
            raise TimeoutError("timeout")

        with builder1_paid_stage_context(job_id=job_id):
            with self.assertRaises(PaidStageOutcomeUnknownError):
                run_paid_provider_call("conceptual_stage", _timeout)
            with self.assertRaises(PaidStageOutcomeUnknownError):
                run_paid_provider_call("conceptual_stage", lambda: "should-not-run")

    def test_image_transport_no_second_paid_call(self) -> None:
        import app as app_module

        job_id = "job-img"
        _owned_job(job_id, "camp-img")

        class _TimeoutClient:
            class images:
                @staticmethod
                def generate(**_kwargs: Any) -> Any:
                    raise TimeoutError("transport timeout")

        with builder1_paid_stage_context(job_id=job_id):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
                with patch("openai.OpenAI", return_value=_TimeoutClient()):
                    with self.assertRaises(Exception):
                        app_module._builder1_image_caller("prompt", "portrait")

        self.assertEqual(get_builder1_job(job_id).get("lastPaidStageStatus"), "outcome_unknown")

    def test_submitted_state_blocks_blind_retry_after_restart(self) -> None:
        from engine.builder1_paid_provider import PaidStageRetryBlockedError, run_paid_provider_call

        job_id = "job-submitted"
        _owned_job(job_id, "camp-submitted")
        update_builder1_job(job_id, lastPaidStage="openai_image_generation", lastPaidStageStatus="submitted")

        with builder1_paid_stage_context(job_id=job_id):
            with self.assertRaises(PaidStageRetryBlockedError):
                run_paid_provider_call("openai_image_generation", lambda: b"bytes")


class TestProductionArtifactStorage(unittest.TestCase):
    def test_production_rejects_temp_storage(self) -> None:
        with patch.dict(
            os.environ,
            {"BUILDER1_PRODUCTION_MODE": "true", "REDIS_URL": "redis://localhost/0"},
            clear=False,
        ):
            with patch.dict(os.environ, {"BUILDER1_IMAGE_ARTIFACT_STORAGE_DIR": "", "VIDEO_HEADLINE_STORAGE_DIR": ""}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    assert_builder1_production_ready()
                self.assertEqual(str(ctx.exception), "builder1_production_requires_durable_artifact_storage")

    def test_production_accepts_configured_durable_root(self) -> None:
        durable_root = os.path.join(os.path.expanduser("~"), ".ace_builder1_test_storage")
        os.makedirs(durable_root, exist_ok=True)
        try:
            with patch.dict(
                os.environ,
                {
                    "BUILDER1_PRODUCTION_MODE": "true",
                    "REDIS_URL": "redis://localhost/0",
                    "BUILDER1_IMAGE_ARTIFACT_STORAGE_DIR": durable_root,
                },
                clear=False,
            ):
                assert_builder1_production_ready()
        finally:
            try:
                os.rmdir(durable_root)
            except OSError:
                pass

    def test_shared_headline_subdir_is_independent_namespace(self) -> None:
        import tempfile

        from engine.builder1_artifact_storage import resolve_builder1_image_storage_root

        td = tempfile.mkdtemp()
        with patch.dict(os.environ, {"VIDEO_HEADLINE_STORAGE_DIR": td, "BUILDER1_IMAGE_ARTIFACT_STORAGE_DIR": ""}, clear=False):
            root = resolve_builder1_image_storage_root()
            self.assertTrue(str(root).endswith("builder1_images"))
            self.assertNotEqual(root.name, "")


class TestArtifactWriteBeforeCompletion(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_missing_file_cannot_yield_campaign_ready(self) -> None:
        from engine.builder1_image_artifact_store import get_builder1_image_artifact_path

        plan_data = _base_campaign(2)
        for ad in plan_data["ads"]:
            ad["marketingText"] = marketing_text_words(50, prefix=f"w{ad['index']}")
        plan = _parse(plan_data, 2)
        create_campaign_session(campaign_id="camp-missing-file", plan=plan, target_ad_count=2)
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 32
        tokens = []
        for idx in (1, 2):
            rec = ad_artifact_record(campaign_id="camp-missing-file", ad_index=idx, plan_revision=1)
            write_builder1_image_artifact_bytes(rec["token"], image_bytes)
            tokens.append(rec["token"])
            persist_campaign_ad_artifact("camp-missing-file", ad_index=idx, artifact=rec)
            reserve_next_ad_index("camp-missing-file", idx, job_id=f"job-{idx}")
            mark_ad_generated("camp-missing-file", idx)
        path = get_builder1_image_artifact_path(tokens[0])
        self.assertIsNotNone(path)
        if path is not None:
            path.unlink(missing_ok=True)
        session = get_campaign_session("camp-missing-file")
        report = evaluate_campaign_completion(session)
        self.assertFalse(report["campaignReady"])
        self.assertIn(1, report["missingArtifacts"])

    def test_production_persist_requires_durable_write(self) -> None:
        import app as app_module

        with patch.dict(os.environ, {"BUILDER1_PRODUCTION_MODE": "true"}, clear=False):
            with patch.object(app_module, "persist_campaign_ad_artifact", wraps=persist_campaign_ad_artifact):
                with patch(
                    "engine.builder1_image_artifact_store.get_builder1_image_artifact_path",
                    return_value=None,
                ):
                    with self.assertRaises(RuntimeError):
                        app_module._builder1_persist_ad_artifact(
                            campaign_id="c1",
                            ad_index=1,
                            plan_revision=1,
                            image_bytes=b"\xff\xd8\xff",
                        )


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_cancel_still_rejects_builder1_job(self) -> None:
        from engine.builder2_job_cancellation import request_builder2_job_cancellation
        from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, set_memory_job_hash

        enable_memory_jobs()
        try:
            set_memory_job_hash(
                "builder1-like",
                {"status": "queued", "product_name": "P", "product_description": "D"},
            )
            result = request_builder2_job_cancellation("builder1-like")
            self.assertEqual(result.get("error"), "not_builder2_job")
        finally:
            disable_memory_jobs()


if __name__ == "__main__":
    unittest.main()
