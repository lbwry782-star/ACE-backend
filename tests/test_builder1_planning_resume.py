"""
Builder1 planning resume API + stage-scoped methodology invalidation tests.

Run: python -m unittest tests.test_builder1_planning_resume -v
"""
from __future__ import annotations

import os
import unittest
import uuid
from typing import Any, Dict
from unittest.mock import patch

import app as app_module
from engine.builder1_integrity_diagnostics import persist_integrity_failure_diagnostic
from engine.builder1_job_planning_request import build_planning_request_snapshot
from engine.builder1_jobs_store import (
    clear_memory_jobs_for_tests,
    create_builder1_job,
    update_builder1_job,
)
from engine.builder1_campaign_store import clear_memory_store_for_tests
from engine.builder1_planning_checkpoint import (
    CHECKPOINT_VERSION,
    STAGE_CHECKPOINT_CONTRACT_VERSIONS,
    PlanningCheckpointSession,
    build_planning_checkpoint_identity,
    delete_planning_checkpoint,
    load_planning_checkpoint_record,
    save_planning_checkpoint_record,
)
from engine.builder1_job_ownership import extract_owner_context_from_request
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_request_idempotency import clear_memory_idempotency_for_tests
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
)
from tests.test_builder1_staged_planning import (
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
    _full_final_responses,
)

BRIEF = "Reinforced shell product for daily carry"
JOB_ID = "job-resume-test"
CAMPAIGN_ID = "campaign-resume-test"


def _snapshot(**overrides: Any) -> Dict[str, Any]:
    base = build_planning_request_snapshot(
        product_name="CarryShell",
        product_description=BRIEF,
        format_value="portrait",
        ad_count=2,
        brand_guidelines=None,
    )
    base.update(overrides)
    return base


def _ownership_fields(headers: Dict[str, str]) -> Dict[str, str]:
    class _Req:
        pass

    _Req.headers = headers
    fields = extract_owner_context_from_request(_Req())
    fields["builder"] = "builder1"
    fields["builder1ContractVersion"] = "builder1_production_v1"
    return fields


def _seed_failed_planning_job(
    *,
    job_id: str = JOB_ID,
    campaign_id: str = CAMPAIGN_ID,
    with_checkpoint: bool = True,
    completed_stages: tuple[str, ...] = ("strategy_slogan_stage", "conceptual_stage"),
    headers: Dict[str, str] | None = None,
    use_real_checkpoint: bool = False,
) -> None:
    hdrs = headers or _request_headers()
    snapshot = _snapshot()
    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=2,
        stage="planning",
        ownership_fields=_ownership_fields(hdrs),
    )
    update_builder1_job(
        job_id,
        status="error",
        error="planning_failed",
        result={"ok": False, "error": "planning_failed", "message": "brand_physical_failed"},
        planningRequestSnapshot=snapshot,
    )
    if not with_checkpoint:
        return
    if use_real_checkpoint:
        responses = _full_final_responses(2)
        counts: Dict[str, int] = {}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            stage_name = stage or ""
            counts[stage_name] = counts.get(stage_name, 0) + 1
            mapping = {
                "product_name_resolution": responses[STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM],
                "strategy_slogan_stage": responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM],
                "conceptual_stage": responses[STAGE_CONCEPTUAL_STAGE_SYSTEM],
                "brand_physical": responses[STAGE_BRAND_PHYSICAL_SYSTEM],
                "graphic_system": responses[STAGE_GRAPHIC_SYSTEM_SYSTEM],
                "series_ads": responses[STAGE_SERIES_ADS_SYSTEM],
            }
            return mapping.get(stage_name, responses.get(system, {}))

        update_builder1_job(job_id, status="running", error="")
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with patch.object(app_module, "_openai_reasoning_planning_model_caller", side_effect=model_caller):
                try:
                    plan_builder1(
                        product_name=snapshot["productName"],
                        product_description=snapshot["productDescription"],
                        format_value=snapshot["format"],
                        model_caller=model_caller,
                        ad_count=snapshot["adCount"],
                        brand_guidelines=snapshot.get("brandGuidelines"),
                        campaign_id=campaign_id,
                        job_id=job_id,
                    )
                except Builder1PlannerError:
                    pass
        update_builder1_job(
            job_id,
            status="error",
            error="planning_failed",
            result={"ok": False, "error": "planning_failed", "message": "brand_physical_failed"},
        )
        return
    identity = build_planning_checkpoint_identity(
        job_id=job_id,
        campaign_id=campaign_id,
        product_name=snapshot["productName"],
        product_description=snapshot["productDescription"],
        format_value=snapshot["format"],
        ad_count=snapshot["adCount"],
        brand_guidelines=snapshot.get("brandGuidelines"),
    )
    session = PlanningCheckpointSession.open(identity)
    for stage in completed_stages:
        session.persist_stage(
            stage,
            output={"stage": stage, "payload": stage},
            dependency_fingerprint=f"dep-{stage}",
        )


def _request_headers(*, request_id: str | None = None) -> Dict[str, str]:
    rid = request_id or str(uuid.uuid4())
    return {
        "X-ACE-Batch-State": "resume-batch",
        "Authorization": "Bearer resume-token",
        "X-ACE-Request-Id": rid,
    }


class TestPlanningResumeApi(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_jobs_for_tests()
        clear_memory_store_for_tests()
        clear_memory_idempotency_for_tests()
        delete_planning_checkpoint(JOB_ID)
        os.environ.pop("BUILDER1_PRODUCTION_MODE", None)
        os.environ.pop("BUILDER1_OWNERSHIP_REQUIRED", None)
        os.environ["BUILDER1_REQUEST_ID_REQUIRED"] = "1"
        self.client = app_module.app.test_client()
        self._stage_counts: Dict[str, int] = {
            "strategy_slogan_stage": 0,
            "conceptual_stage": 0,
            "brand_physical": 0,
            "graphic_system": 0,
            "series_ads": 0,
            "product_name_resolution": 0,
        }
        self._responses = _full_final_responses(2)

    def tearDown(self) -> None:
        delete_planning_checkpoint(JOB_ID)
        clear_memory_idempotency_for_tests()
        clear_memory_jobs_for_tests()
        clear_memory_store_for_tests()
        os.environ.pop("BUILDER1_REQUEST_ID_REQUIRED", None)
        os.environ.pop("BUILDER1_OWNERSHIP_REQUIRED", None)

    def _model_caller(self, system: str, user: str, stage: str | None = None) -> object:
        stage_name = stage or ""
        if stage_name in self._stage_counts:
            self._stage_counts[stage_name] += 1
        mapping = {
            "product_name_resolution": self._responses[STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM],
            "strategy_slogan_stage": self._responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM],
            "conceptual_stage": self._responses[STAGE_CONCEPTUAL_STAGE_SYSTEM],
            "brand_physical": self._responses[STAGE_BRAND_PHYSICAL_SYSTEM],
            "graphic_system": self._responses[STAGE_GRAPHIC_SYSTEM_SYSTEM],
            "series_ads": self._responses[STAGE_SERIES_ADS_SYSTEM],
        }
        if stage_name in mapping:
            return mapping[stage_name]
        return self._responses.get(system, {})

    def test_resume_accepted_same_job_and_campaign(self) -> None:
        _seed_failed_planning_job()
        with patch.object(app_module._builder1_executor, "submit") as submit_mock:
            resp = self.client.post(
                "/api/builder1-resume-planning",
                json={"jobId": JOB_ID},
                headers=_request_headers(),
            )
        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("jobId"), JOB_ID)
        self.assertEqual(body.get("campaignId"), CAMPAIGN_ID)
        self.assertTrue(body.get("planningResume"))
        submit_mock.assert_called_once()
        self.assertEqual(submit_mock.call_args[0][1], JOB_ID)
        self.assertEqual(submit_mock.call_args[0][2], CAMPAIGN_ID)

    def test_resume_reuses_strategy_and_conceptual_three_new_calls(self) -> None:
        _seed_failed_planning_job(use_real_checkpoint=True)
        before = load_planning_checkpoint_record(JOB_ID)
        self.assertEqual(set((before or {}).get("completedStages", {}).keys()), {"strategy_slogan_stage", "conceptual_stage"})
        with patch.object(app_module._builder1_executor, "submit", side_effect=lambda fn, *a: fn(*a)):
            with patch.object(app_module, "_openai_reasoning_planning_model_caller", side_effect=self._model_caller):
                resp = self.client.post(
                    "/api/builder1-resume-planning",
                    json={"jobId": JOB_ID},
                    headers=_request_headers(),
                )
        self.assertEqual(resp.status_code, 202)
        after = load_planning_checkpoint_record(JOB_ID)
        completed = set((after or {}).get("completedStages", {}).keys())
        self.assertEqual(
            completed,
            {
                "strategy_slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            },
        )

    def test_resume_idempotent_same_request_id(self) -> None:
        _seed_failed_planning_job()
        rid = str(uuid.uuid4())
        headers = _request_headers(request_id=rid)
        submit_count = 0

        def _submit(fn, *args):
            nonlocal submit_count
            submit_count += 1

        with patch.object(app_module._builder1_executor, "submit", side_effect=_submit):
            first = self.client.post("/api/builder1-resume-planning", json={"jobId": JOB_ID}, headers=headers)
            second = self.client.post("/api/builder1-resume-planning", json={"jobId": JOB_ID}, headers=headers)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertTrue(second.get_json().get("idempotentReplay"))
        self.assertEqual(submit_count, 1)

    def test_resume_conflict_different_body_same_request_id(self) -> None:
        _seed_failed_planning_job()
        rid = str(uuid.uuid4())
        headers = _request_headers(request_id=rid)
        with patch.object(app_module._builder1_executor, "submit", return_value=None):
            first = self.client.post("/api/builder1-resume-planning", json={"jobId": JOB_ID}, headers=headers)
            second = self.client.post(
                "/api/builder1-resume-planning",
                json={"jobId": "other-job-id"},
                headers=headers,
            )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)

    def test_running_job_rejected(self) -> None:
        _seed_failed_planning_job()
        update_builder1_job(JOB_ID, status="running", error="")
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("job_already_running", resp.get_json().get("rejectionReasons", []))

    def test_cancelled_job_rejected(self) -> None:
        _seed_failed_planning_job()
        update_builder1_job(JOB_ID, status="cancelled", cancelRequested=True)
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("job_cancelled", resp.get_json().get("rejectionReasons", []))

    def test_missing_checkpoint_rejected(self) -> None:
        _seed_failed_planning_job(with_checkpoint=False)
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("planning_checkpoint_missing", resp.get_json().get("rejectionReasons", []))

    def test_integrity_job_not_eligible(self) -> None:
        _seed_failed_planning_job()
        update_builder1_job(
            JOB_ID,
            error="campaign_integrity_failed",
            result={"ok": False, "error": "planning_failed", "message": "campaign_integrity_failed"},
        )
        persist_integrity_failure_diagnostic(
            JOB_ID,
            {"reasons": ["upstream_mutation"], "rejectedPlan": {"productNameResolved": "X"}},
            campaign_id=CAMPAIGN_ID,
        )
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("integrity_rejection_not_planning_resume", resp.get_json().get("rejectionReasons", []))

    def test_image_inflight_blocks_resume(self) -> None:
        _seed_failed_planning_job()
        update_builder1_job(JOB_ID, lastPaidStage="openai_image_generation", lastPaidStageStatus="submitted")
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("image_provider_inflight", resp.get_json().get("rejectionReasons", []))

    def test_initial_generate_replay_not_planning_resume(self) -> None:
        rid = str(uuid.uuid4())
        headers = _request_headers(request_id=rid)
        with patch.object(app_module._builder1_executor, "submit", return_value=None):
            first = self.client.post(
                "/api/builder1-generate",
                json={"productDescription": BRIEF, "productName": "CarryShell", "adCount": 2},
                headers=headers,
            )
            job_id = first.get_json()["jobId"]
            update_builder1_job(
                job_id,
                status="error",
                error="planning_failed",
                result={"ok": False, "error": "planning_failed"},
            )
            replay = self.client.post(
                "/api/builder1-generate",
                json={"productDescription": BRIEF, "productName": "CarryShell", "adCount": 2},
                headers=headers,
            )
        self.assertTrue(replay.get_json().get("idempotentReplay"))
        self.assertNotIn("planningResume", replay.get_json())

    def test_ownership_mismatch_rejected(self) -> None:
        os.environ["BUILDER1_OWNERSHIP_REQUIRED"] = "1"
        _seed_failed_planning_job()
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers={
                "X-ACE-Batch-State": "other-batch",
                "Authorization": "Bearer other-token",
                "X-ACE-Request-Id": str(uuid.uuid4()),
            },
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json().get("error"), "ownership_mismatch")

    def test_media_started_job_rejected(self) -> None:
        from engine.builder1_campaign_store import _save_raw, get_campaign_session_raw

        _seed_failed_planning_job()
        raw = get_campaign_session_raw(CAMPAIGN_ID) or {
            "campaignId": CAMPAIGN_ID,
            "targetAdCount": 2,
            "generatedCount": 0,
        }
        raw["generatedCount"] = 1
        _save_raw(CAMPAIGN_ID, raw)
        resp = self.client.post(
            "/api/builder1-resume-planning",
            json={"jobId": JOB_ID},
            headers=_request_headers(),
        )
        self.assertIn("campaign_media_started", resp.get_json().get("rejectionReasons", []))


class TestStageScopedMethodologyInvalidation(unittest.TestCase):
    def setUp(self) -> None:
        delete_planning_checkpoint(JOB_ID)

    def tearDown(self) -> None:
        delete_planning_checkpoint(JOB_ID)

    def _seed_all_stages(self) -> PlanningCheckpointSession:
        identity = build_planning_checkpoint_identity(
            job_id=JOB_ID,
            campaign_id=CAMPAIGN_ID,
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            ad_count=2,
            brand_guidelines=None,
        )
        session = PlanningCheckpointSession.open(identity)
        for stage in (
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        ):
            session.persist_stage(stage, output={"stage": stage}, dependency_fingerprint=f"dep-{stage}")
        return session

    def _restore_flags(self, session: PlanningCheckpointSession) -> Dict[str, bool]:
        flags: Dict[str, bool] = {}
        for stage in (
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        ):
            restored = session.try_restore_stage(
                stage,
                dependency_fingerprint=f"dep-{stage}",
                deserialize=lambda payload: payload,
            )
            flags[stage] = restored is not None
        return flags

    def test_brand_physical_methodology_change_preserves_upstream(self) -> None:
        import engine.builder1_planning_checkpoint as cp

        self._seed_all_stages()
        original = cp.stage_methodology_fingerprint

        def _patched(stage: str) -> str:
            if stage == "brand_physical":
                return original(stage) + "_changed"
            return original(stage)

        with patch.object(cp, "stage_methodology_fingerprint", side_effect=_patched):
            session = PlanningCheckpointSession.open(
                build_planning_checkpoint_identity(
                    job_id=JOB_ID,
                    campaign_id=CAMPAIGN_ID,
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    ad_count=2,
                    brand_guidelines=None,
                )
            )
            flags = self._restore_flags(session)
        self.assertTrue(flags["strategy_slogan_stage"])
        self.assertTrue(flags["conceptual_stage"])
        self.assertFalse(flags["brand_physical"])

    def _methodology_flags(self, changed_stage: str) -> Dict[str, bool]:
        import engine.builder1_planning_checkpoint as cp

        self._seed_all_stages()
        original = cp.stage_methodology_fingerprint

        def _patched(stage: str) -> str:
            if stage == changed_stage:
                return original(stage) + "_changed"
            return original(stage)

        with patch.object(cp, "stage_methodology_fingerprint", side_effect=_patched):
            session = PlanningCheckpointSession.open(
                build_planning_checkpoint_identity(
                    job_id=JOB_ID,
                    campaign_id=CAMPAIGN_ID,
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    ad_count=2,
                    brand_guidelines=None,
                )
            )
            return self._restore_flags(session)

    def test_graphic_methodology_change_preserves_physical(self) -> None:
        flags = self._methodology_flags("graphic_system")
        self.assertTrue(flags["strategy_slogan_stage"])
        self.assertTrue(flags["conceptual_stage"])
        self.assertTrue(flags["brand_physical"])
        self.assertFalse(flags["graphic_system"])

    def test_conceptual_methodology_change_preserves_strategy(self) -> None:
        flags = self._methodology_flags("conceptual_stage")
        self.assertTrue(flags["strategy_slogan_stage"])
        self.assertFalse(flags["conceptual_stage"])

    def test_strategy_methodology_change_invalidates_all(self) -> None:
        flags = self._methodology_flags("strategy_slogan_stage")
        self.assertFalse(flags["strategy_slogan_stage"])

    def test_series_methodology_change_preserves_upstream_four(self) -> None:
        flags = self._methodology_flags("series_ads")
        self.assertTrue(flags["strategy_slogan_stage"])
        self.assertTrue(flags["conceptual_stage"])
        self.assertTrue(flags["brand_physical"])
        self.assertTrue(flags["graphic_system"])
        self.assertFalse(flags["series_ads"])

    def test_stage_contract_version_change_is_stage_scoped(self) -> None:
        self._seed_all_stages()
        record = load_planning_checkpoint_record(JOB_ID)
        record["completedStages"]["brand_physical"]["stageContractVersion"] = "1"
        save_planning_checkpoint_record(JOB_ID, record)
        session = PlanningCheckpointSession.open(
            build_planning_checkpoint_identity(
                job_id=JOB_ID,
                campaign_id=CAMPAIGN_ID,
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                ad_count=2,
                brand_guidelines=None,
            )
        )
        flags = self._restore_flags(session)
        self.assertTrue(flags["strategy_slogan_stage"])
        self.assertTrue(flags["conceptual_stage"])
        self.assertFalse(flags["brand_physical"])

    def test_request_fingerprint_change_rejects_identity(self) -> None:
        self._seed_all_stages()
        session = PlanningCheckpointSession.open(
            build_planning_checkpoint_identity(
                job_id=JOB_ID,
                campaign_id=CAMPAIGN_ID,
                product_name="OtherName",
                product_description=BRIEF,
                format_value="portrait",
                ad_count=2,
                brand_guidelines=None,
            )
        )
        restored = session.try_restore_stage(
            "strategy_slogan_stage",
            dependency_fingerprint="dep-strategy_slogan_stage",
            deserialize=lambda payload: payload,
        )
        self.assertIsNone(restored)

    def test_corrupt_stage_methodology_fingerprint_reruns_stage(self) -> None:
        self._seed_all_stages()
        record = load_planning_checkpoint_record(JOB_ID)
        record["completedStages"]["brand_physical"]["stageMethodologyFingerprint"] = "corrupt"
        save_planning_checkpoint_record(JOB_ID, record)
        session = PlanningCheckpointSession.open(
            build_planning_checkpoint_identity(
                job_id=JOB_ID,
                campaign_id=CAMPAIGN_ID,
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                ad_count=2,
                brand_guidelines=None,
            )
        )
        self.assertIsNone(
            session.try_restore_stage(
                "brand_physical",
                dependency_fingerprint="dep-brand_physical",
                deserialize=lambda payload: payload,
            )
        )

    def test_legacy_global_fingerprint_checkpoint_fails_safe(self) -> None:
        record = {
            "version": "builder1_planning_checkpoint_v1",
            "identity": {
                "jobId": JOB_ID,
                "campaignId": CAMPAIGN_ID,
                "requestFingerprint": build_planning_checkpoint_identity(
                    job_id=JOB_ID,
                    campaign_id=CAMPAIGN_ID,
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    ad_count=2,
                    brand_guidelines=None,
                ).request_fingerprint,
                "planningContractVersion": "builder1_production_v1",
                "methodologyFingerprint": "legacy-global",
            },
            "completedStages": {
                "strategy_slogan_stage": {
                    "status": "succeeded",
                    "stageContractVersion": STAGE_CHECKPOINT_CONTRACT_VERSIONS["strategy_slogan_stage"],
                    "dependencyFingerprint": "dep-strategy_slogan_stage",
                    "outputFingerprint": "abc",
                    "output": {"stage": "strategy_slogan_stage"},
                    "completedAt": 1.0,
                }
            },
        }
        save_planning_checkpoint_record(JOB_ID, record)
        session = PlanningCheckpointSession.open(
            build_planning_checkpoint_identity(
                job_id=JOB_ID,
                campaign_id=CAMPAIGN_ID,
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                ad_count=2,
                brand_guidelines=None,
            )
        )
        restored = session.try_restore_stage(
            "strategy_slogan_stage",
            dependency_fingerprint="dep-strategy_slogan_stage",
            deserialize=lambda payload: payload,
        )
        self.assertIsNone(restored)

    def test_identity_fields_exclude_methodology(self) -> None:
        identity = build_planning_checkpoint_identity(
            job_id=JOB_ID,
            campaign_id=CAMPAIGN_ID,
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            ad_count=2,
            brand_guidelines=None,
        )
        self.assertNotIn("methodologyFingerprint", identity.to_dict())
        self.assertEqual(CHECKPOINT_VERSION, "builder1_planning_checkpoint_v2")


if __name__ == "__main__":
    unittest.main()
