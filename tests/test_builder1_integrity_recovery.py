"""
Builder1 integrity-failure recovery tests.

Run: python -m unittest tests.test_builder1_integrity_recovery -v
"""
from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder1_integrity_diagnostics import (
    INTEGRITY_DIAGNOSTIC_JOB_FIELD,
    build_integrity_failure_diagnostic,
    persist_integrity_failure_diagnostic,
)
from engine.builder1_integrity_recovery import (
    apply_builder1_integrity_recovery,
    assess_builder1_integrity_recovery,
    revalidate_rejected_plan_dict,
)
from engine.builder1_jobs_store import create_builder1_job, get_builder1_job, update_builder1_job
from engine.builder1_campaign_store import clear_memory_store_for_tests, get_campaign_session_raw
from tests.test_builder1_literal_slogan_false_positive import _tsaad_tsaad_production_rejected_plan


class TestIntegrityRecovery(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        self.job_id = "job-integrity-recovery-test"
        self.campaign_id = "camp-integrity-recovery-test"
        create_builder1_job(
            job_id=self.job_id,
            campaign_id=self.campaign_id,
            target_ad_count=2,
            stage="planning",
        )

    def _persist_failed_job(self, *, plan: Dict[str, Any], still_fails: bool = False) -> None:
        reasons = ["literal_slogan_illustration"] if not still_fails else ["literal_slogan_illustration"]
        if still_fails:
            plan = dict(plan)
            plan["transferredObject"] = "Door"
            plan["physicalGenerator"] = "Door"
            plan["brandSlogan"] = "Opens every door"
            plan["whyClearerThanShowingProduct"] = "Shows a door because the slogan mentions opening doors"
            plan["planningInternals"] = {}
        diagnostic = build_integrity_failure_diagnostic(
            reasons=reasons,
            details=[{"code": "literal_slogan_illustration", "branch": "plan_object_lexical_match"}],
            rejected_plan=plan,
        )
        persist_integrity_failure_diagnostic(self.job_id, diagnostic, campaign_id=self.campaign_id)
        update_builder1_job(
            self.job_id,
            status="error",
            error="planning_failed",
            result={"ok": False, "error": "planning_failed", "message": "campaign_integrity_failed"},
            lastPaidStage="series_ads",
            lastPaidStageStatus="succeeded",
        )

    def test_production_plan_revalidates_cleanly(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        result = revalidate_rejected_plan_dict(plan)
        self.assertTrue(result["ok"], msg=str(result["reasons"]))

    def test_dry_run_performs_zero_paid_calls(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan())
        with patch("engine.builder1_planner.plan_builder1") as plan_mock:
            report = assess_builder1_integrity_recovery(self.job_id)
            plan_mock.assert_not_called()
        self.assertEqual(report["paidCallsPerformed"], 0)
        self.assertTrue(report["rejectedPlanPresent"])

    def test_dry_run_eligible_for_fixed_production_plan(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan())
        report = assess_builder1_integrity_recovery(self.job_id)
        self.assertTrue(report["recoveryEligible"], msg=str(report["eligibilityFailures"]))
        self.assertEqual(report["remainingIntegrityReasons"], [])

    def test_recovery_rejects_still_failing_plan(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan(), still_fails=True)
        report = assess_builder1_integrity_recovery(self.job_id)
        self.assertFalse(report["recoveryEligible"])
        self.assertIn("revalidation_failed", report["eligibilityFailures"])

    def test_recovery_rejects_missing_diagnostic(self) -> None:
        report = assess_builder1_integrity_recovery(self.job_id)
        self.assertFalse(report["recoveryEligible"])
        self.assertIn("integrity_failure_diagnostic_missing", report["eligibilityFailures"])

    def test_apply_creates_session_without_planning(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan())
        with patch("engine.builder1_planner.plan_builder1") as plan_mock:
            with patch("app._builder1_executor") as executor_mock:
                report = apply_builder1_integrity_recovery(
                    self.job_id,
                    enqueue_image_pipeline=True,
                )
                plan_mock.assert_not_called()
        self.assertTrue(report["applied"])
        self.assertTrue(report["sessionCreated"])
        self.assertIsNotNone(get_campaign_session_raw(self.campaign_id))
        executor_mock.submit.assert_called_once()

    def test_apply_never_repeats_planning(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan())
        with patch("engine.builder1_planner.plan_builder1") as plan_mock:
            apply_builder1_integrity_recovery(self.job_id, enqueue_image_pipeline=False)
        plan_mock.assert_not_called()

    def test_recovery_rejects_existing_media_state(self) -> None:
        self._persist_failed_job(plan=_tsaad_tsaad_production_rejected_plan())
        from engine.builder1_campaign_store import create_campaign_session
        from tests.test_builder1_series import _parse

        plan = _parse(_tsaad_tsaad_production_rejected_plan(), 2)
        create_campaign_session(campaign_id=self.campaign_id, plan=plan, target_ad_count=2)
        raw = get_campaign_session_raw(self.campaign_id) or {}
        raw["generatedCount"] = 1
        raw["generated"] = [{"index": 1}]
        from engine.builder1_campaign_store import _save_raw

        _save_raw(self.campaign_id, raw)
        report = assess_builder1_integrity_recovery(self.job_id)
        self.assertFalse(report["recoveryEligible"])
        self.assertIn("campaign_media_already_started", report["eligibilityFailures"])


if __name__ == "__main__":
    unittest.main()
