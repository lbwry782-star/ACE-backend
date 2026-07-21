"""
Builder1 retry-state routing and physical-repair execution tests.

Run: python -m unittest tests.test_builder1_retry_routing -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, List
from unittest.mock import Mock, patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    apply_repaired_campaign_plan,
    begin_physical_repair,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    mark_image_retry_required,
    mark_physical_repair_required,
    reserve_next_ad_index,
    validate_next_ad_request,
)
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    classify_compliance_failure,
)
from engine.builder1_jobs_store import clear_memory_jobs_for_tests, get_builder1_job
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planning_profile import resolve_stage_model
from engine.builder1_planner import plan_builder1
from engine.builder1_retry_state import (
    RETRY_MODE_IMAGE_ONLY,
    RETRY_MODE_REPAIR_FROM_PHYSICAL,
    public_retry_fields,
    resolve_authoritative_retry_mode,
)
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _full_final_responses


class TestRetryModeStorage(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def _plan(self, ad_count: int = 3):
        return _parse(_base_campaign(ad_count), ad_count)

    def test_physical_repair_required_persists_retry_mode(self) -> None:
        create_campaign_session(campaign_id="cmp-repair", plan=self._plan(3), target_ad_count=3)
        mark_physical_repair_required(
            "cmp-repair",
            failed_ad_index=2,
            violations=["product_used_as_physical_generator"],
        )
        session = get_campaign_session("cmp-repair")
        self.assertEqual(session.status, "physical_repair_required")
        self.assertEqual(session.retry_mode, RETRY_MODE_REPAIR_FROM_PHYSICAL)
        self.assertEqual(session.failed_ad_index, 2)
        self.assertEqual(session.plan_revision, 1)

    def test_image_retry_required_persists_retry_mode(self) -> None:
        create_campaign_session(campaign_id="cmp-image", plan=self._plan(3), target_ad_count=3)
        mark_image_retry_required("cmp-image", failed_ad_index=2, violations=["invented_product_logo"])
        session = get_campaign_session("cmp-image")
        self.assertEqual(session.retry_mode, RETRY_MODE_IMAGE_ONLY)
        self.assertEqual(session.status, "image_retry_required")

    def test_retry_mode_survives_json_roundtrip(self) -> None:
        create_campaign_session(campaign_id="cmp-serde", plan=self._plan(3), target_ad_count=3)
        mark_physical_repair_required("cmp-serde", failed_ad_index=2, violations=["x"])
        session = get_campaign_session("cmp-serde")
        reloaded = get_campaign_session("cmp-serde")
        self.assertEqual(reloaded.retry_mode, RETRY_MODE_REPAIR_FROM_PHYSICAL)
        self.assertEqual(reloaded.failed_ad_index, session.failed_ad_index)


class TestReservationRouting(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def _session(self, *, ad_count: int = 3, generated: List[int] | None = None) -> str:
        cid = "cmp-route"
        create_campaign_session(campaign_id=cid, plan=_parse(_base_campaign(ad_count), ad_count), target_ad_count=ad_count)
        for idx in generated or []:
            reserve_next_ad_index(cid, idx, job_id=f"job-{idx}")
            mark_ad_generated(cid, idx)
        return cid

    def test_physical_repair_required_blocks_reservation(self) -> None:
        cid = self._session(generated=[1])
        mark_physical_repair_required(cid, failed_ad_index=2, violations=["product_used_as_physical_generator"])
        with self.assertRaises(CampaignStoreError) as ctx:
            reserve_next_ad_index(cid, 2, job_id="job-block")
        self.assertEqual(ctx.exception.code, "physical_repair_not_completed")

    def test_image_only_retry_allows_reservation(self) -> None:
        cid = self._session(generated=[1])
        mark_image_retry_required(cid, failed_ad_index=2, violations=["logo_like_brand_symbol"])
        session = reserve_next_ad_index(cid, 2, job_id="job-image")
        self.assertEqual(session.generating_index, 2)

    def test_repaired_plan_increments_revision(self) -> None:
        cid = self._session(generated=[1])
        plan = get_campaign_session(cid).plan
        mark_physical_repair_required(cid, failed_ad_index=2, violations=["x"])
        updated = apply_repaired_campaign_plan(cid, plan)
        self.assertEqual(updated.plan_revision, 2)
        self.assertEqual(updated.retry_mode, RETRY_MODE_IMAGE_ONLY)
        self.assertEqual(updated.failed_ad_index, 2)


class TestGenerateNextRouting(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_generate_next_routes_physical_repair_without_reservation(self) -> None:
        from app import app

        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-next-repair", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-next-repair", 1, job_id="job-1")
        mark_ad_generated("cmp-next-repair", 1)
        mark_physical_repair_required(
            "cmp-next-repair",
            failed_ad_index=2,
            violations=["product_used_as_physical_generator"],
        )
        submitted: List[tuple[Any, ...]] = []

        def capture_submit(fn, *args):
            submitted.append((fn.__name__, args))

        with app.test_client() as client:
                with patch("app._builder1_executor") as executor:
                    executor.submit.side_effect = capture_submit
                    response = client.post(
                        "/api/builder1-generate-next",
                        json={"campaignId": "cmp-next-repair", "expectedNextIndex": 2},
                    )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["stage"], "repairing_physical")
        self.assertEqual(payload["retryMode"], RETRY_MODE_REPAIR_FROM_PHYSICAL)
        self.assertTrue(payload["repairInProgress"])
        session = get_campaign_session("cmp-next-repair")
        self.assertIsNone(session.generating_index)
        self.assertTrue(submitted)
        self.assertEqual(submitted[0][0], "_builder1_run_physical_repair_job")

    def test_generate_next_image_only_reserves_and_routes_image_job(self) -> None:
        from app import app

        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-next-image", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-next-image", 1, job_id="job-1")
        mark_ad_generated("cmp-next-image", 1)
        mark_image_retry_required("cmp-next-image", failed_ad_index=2, violations=["logo_like_brand_symbol"])
        submitted: List[tuple[Any, ...]] = []

        def capture_submit(fn, *args):
            submitted.append((fn.__name__, args))

        with app.test_client() as client:
            with patch("app._builder1_executor") as executor:
                executor.submit.side_effect = capture_submit
                response = client.post(
                    "/api/builder1-generate-next",
                    json={"campaignId": "cmp-next-image", "expectedNextIndex": 2},
                )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["stage"], "generating_images")
        self.assertEqual(submitted[0][0], "_builder1_run_next_job")
        self.assertEqual(get_campaign_session("cmp-next-image").generating_index, 2)


class TestImageJobGuards(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_stale_plan_revision_rejected_before_image(self) -> None:
        from app import _builder1_generate_single_ad
        from engine.builder1_jobs_store import create_builder1_job, update_builder1_job

        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-stale", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-stale", 1, job_id="job-1")
        mark_ad_generated("cmp-stale", 1)
        mark_image_retry_required("cmp-stale", failed_ad_index=2, violations=["x"])
        reserve_next_ad_index("cmp-stale", 2, job_id="job-stale")
        create_builder1_job(job_id="job-stale", campaign_id="cmp-stale", target_ad_count=3, stage="generating_images")
        update_builder1_job("job-stale", planRevision=1, retryAdIndex=2)
        apply_repaired_campaign_plan("cmp-stale", plan)
        result = _builder1_generate_single_ad(
            job_id="job-stale",
            campaign_id="cmp-stale",
            ad_index=2,
            already_reserved=True,
        )
        self.assertEqual(result["error"], "stale_plan_revision")

    def test_missing_plan_revision_rejected_before_image(self) -> None:
        from app import _builder1_generate_single_ad
        from engine.builder1_jobs_store import create_builder1_job

        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-missing-rev", plan=plan, target_ad_count=3)
        create_builder1_job(
            job_id="job-missing-rev",
            campaign_id="cmp-missing-rev",
            target_ad_count=3,
            stage="generating_images",
        )
        result = _builder1_generate_single_ad(
            job_id="job-missing-rev",
            campaign_id="cmp-missing-rev",
            ad_index=1,
            already_reserved=False,
        )
        self.assertEqual(result["error"], "missing_plan_revision")

    def test_physical_repair_pending_rejects_image_worker(self) -> None:
        from app import _builder1_generate_single_ad

        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-pending", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-pending", 1, job_id="job-1")
        mark_ad_generated("cmp-pending", 1)
        mark_physical_repair_required("cmp-pending", failed_ad_index=2, violations=["x"])
        begin_physical_repair("cmp-pending", job_id="job-repair")
        result = _builder1_generate_single_ad(
            job_id="job-repair",
            campaign_id="cmp-pending",
            ad_index=2,
            already_reserved=False,
        )
        self.assertEqual(result["error"], "physical_repair_not_completed")


class TestClassificationAccuracy(unittest.TestCase):
    def test_pixel_only_ambiguity_is_image_execution(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["product_visible_without_explicit_request", "product_used_as_physical_generator"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)
        self.assertFalse(evidence["structuredPlanConflict"])

    def test_structured_conflict_is_plan_contradiction(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        plan.product_description = "Lightweight running shoe for daily training"
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["product_visible_without_explicit_request"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.PLAN_CONTRADICTION)
        self.assertTrue(evidence["structuredPlanConflict"])


class TestGeneratedAdPreservation(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_mark_ad_generated_clears_retry_state(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-clear", plan=plan, target_ad_count=3)
        mark_image_retry_required("cmp-clear", failed_ad_index=1, violations=["x"])
        reserve_next_ad_index("cmp-clear", 1, job_id="job-1")
        mark_ad_generated("cmp-clear", 1)
        session = get_campaign_session("cmp-clear")
        self.assertEqual(session.generated_count, 1)
        self.assertIsNone(session.failed_ad_index)
        self.assertEqual(session.retry_mode, "none")

    def test_generated_count_preserved_during_repair_required(self) -> None:
        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-count", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-count", 1, job_id="job-1")
        mark_ad_generated("cmp-count", 1)
        mark_physical_repair_required("cmp-count", failed_ad_index=2, violations=["x"])
        session = get_campaign_session("cmp-count")
        self.assertEqual(session.generated_count, 1)
        self.assertEqual(session.failed_ad_index, 2)


class TestPlanningRegression(unittest.TestCase):
    def test_supplied_name_planning_remains_six_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_planning_remains_seven_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)

    def test_quality_profile_keeps_quality_model(self) -> None:
        import os

        with patch.dict(os.environ, {"BUILDER1_PLANNING_PROFILE": "QUALITY", "BUILDER1_PLANNING_MODEL": "gpt-5.6-sol"}, clear=False):
            self.assertEqual(resolve_stage_model("graphic_system"), "gpt-5.6-sol")


class TestPublicRetryFields(unittest.TestCase):
    def test_public_retry_fields_expose_actionable_state(self) -> None:
        clear_memory_store_for_tests()
        plan = _parse(_base_campaign(3), 3)
        create_campaign_session(campaign_id="cmp-public", plan=plan, target_ad_count=3)
        mark_physical_repair_required("cmp-public", failed_ad_index=2, violations=["x"])
        session = get_campaign_session("cmp-public")
        fields = public_retry_fields(session=session, retry_ad_index=2)
        self.assertTrue(fields["retryable"])
        self.assertEqual(fields["retryMode"], RETRY_MODE_REPAIR_FROM_PHYSICAL)
        self.assertEqual(fields["retryAdIndex"], 2)
        self.assertTrue(fields["planningComplete"])


if __name__ == "__main__":
    unittest.main()
