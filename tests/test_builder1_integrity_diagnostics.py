"""
Builder1 integrity-failure diagnostic persistence tests.

Run: python -m unittest tests.test_builder1_integrity_diagnostics -v
"""
from __future__ import annotations

import copy
import inspect
import json
import unittest
from unittest.mock import patch

from engine.builder1_campaign_integrity import validate_builder1_campaign_integrity
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from engine.builder1_integrity_diagnostics import (
    INTEGRITY_DIAGNOSTIC_JOB_FIELD,
    build_integrity_failure_diagnostic,
    get_integrity_failure_diagnostic,
    persist_integrity_failure_diagnostic,
)
from engine.builder1_jobs_store import JOB_TTL_SECONDS, create_builder1_job, get_builder1_job
from engine.builder1_literal_embodiment import scan_literal_embodiment_bias
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from tests.test_builder1_literal_embodiment import _literal_trap_plan_dict


def _minimal_upstream(**overrides: object):
    from engine.builder1_consolidated_stages import Builder1UpstreamSnapshot

    base = dict(
        product_name_resolved="RoutePro",
        strategic_problem="Buyers waste time on long routes",
        relative_advantage="Shorter navigation paths",
        brand_slogan="We shorten your way",
        implied_action="Make distances shorter",
        selected_slogan_id="S01",
        conceptual_generator="Distances shrink",
        selected_conceptual_id="C01",
        physical_generator="City road network",
        graphic_layout_template="",
        graphic_recurring_device="",
    )
    base.update(overrides)
    return Builder1UpstreamSnapshot(**base)


def _series_plan_from_dict(plan_dict: dict, *, ad_count: int = 2):
    from tests.test_builder1_series import _parse

    return _parse(plan_dict, ad_count)


class TestIntegrityDiagnosticPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.job_id = "job-integrity-diagnostic-test"
        self.campaign_id = "camp-integrity-diagnostic-test"
        create_builder1_job(
            job_id=self.job_id,
            campaign_id=self.campaign_id,
            target_ad_count=2,
            stage="planning",
        )

    def test_integrity_failure_persists_rejected_plan_snapshot(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        diagnostic = build_integrity_failure_diagnostic(
            reasons=["literal_slogan_illustration"],
            details=[{"code": "literal_slogan_illustration", "branch": "scan_shortening_route_family_plan"}],
            rejected_plan=plan_dict,
        )
        self.assertTrue(persist_integrity_failure_diagnostic(self.job_id, diagnostic, campaign_id=self.campaign_id))
        stored = get_integrity_failure_diagnostic(self.job_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        rejected = stored.get("rejectedPlan") or {}
        self.assertTrue(stored.get("rejectedPlanAvailable"))
        self.assertEqual(rejected.get("brandSlogan"), plan_dict["brandSlogan"])
        self.assertEqual(rejected.get("physicalGenerator"), plan_dict["physicalGenerator"])
        self.assertEqual(len(rejected.get("ads") or []), 2)

    def test_integrity_failure_persists_exact_reasons(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        reasons = scan_literal_embodiment_bias(plan_dict)
        diagnostic = build_integrity_failure_diagnostic(
            reasons=reasons,
            details=[],
            rejected_plan=plan_dict,
        )
        persist_integrity_failure_diagnostic(self.job_id, diagnostic)
        stored = get_integrity_failure_diagnostic(self.job_id)
        assert stored is not None
        self.assertEqual(stored.get("reasons"), reasons)

    def test_literal_slogan_illustration_persists_detector_branch_evidence(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        evidence: list[dict] = []
        reasons = scan_literal_embodiment_bias(plan_dict, evidence)
        self.assertIn("literal_slogan_illustration", reasons)
        self.assertTrue(any(item.get("branch") for item in evidence))
        literal_entries = [item for item in evidence if item.get("code") == "literal_slogan_illustration"]
        self.assertTrue(literal_entries)
        self.assertEqual(literal_entries[0].get("detector"), "literal_embodiment")

    def test_plan_level_violation_records_field_and_matched_terms(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        evidence: list[dict] = []
        scan_literal_embodiment_bias(plan_dict, evidence)
        plan_level = [
            item
            for item in evidence
            if item.get("level") == "plan" and item.get("field") and item.get("code") == "literal_slogan_illustration"
        ]
        self.assertTrue(plan_level)
        entry = plan_level[0]
        self.assertIn(entry.get("field"), {"transferredObject", "physicalGenerator", "transferredObjectAction", "planVisualBlob", "ads"})
        if entry.get("matchedTerms"):
            self.assertIsInstance(entry.get("matchedTerms"), list)

    def test_ad_level_violation_records_ad_index_and_field(self) -> None:
        plan_dict = copy.deepcopy(_literal_trap_plan_dict())
        plan_dict["transferredObject"] = "Compact travel suitcase"
        plan_dict["physicalGenerator"] = "Compact travel suitcase"
        evidence: list[dict] = []
        reasons = scan_literal_embodiment_bias(plan_dict, evidence)
        self.assertIn("literal_slogan_illustration", reasons)
        ad_level = [item for item in evidence if item.get("level") == "ad" and item.get("adIndex") is not None]
        self.assertTrue(ad_level)
        self.assertIn(ad_level[0].get("field"), {"adVisualBlob", "physicalExecution", "executionSubject", "structuredAdProof"})

    def test_same_plan_produces_identical_integrity_result_with_and_without_evidence(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        baseline = deterministic_builder1_integrity_checks(copy.deepcopy(plan_dict))
        evidence: list[dict] = []
        with_evidence = deterministic_builder1_integrity_checks(copy.deepcopy(plan_dict), integrity_evidence=evidence)
        self.assertEqual(baseline, with_evidence)
        passing = copy.deepcopy(plan_dict)
        passing["physicalGenerator"] = "Compact travel suitcase"
        passing["transferredObject"] = "Foldable city bicycle"
        passing["transferredObjectAction"] = "Bike folds into a tiny unexpected shape"
        passing["ads"] = [
            {
                **ad,
                "physicalExecution": "A cyclist folds a bright bicycle in a studio",
                "visualExecution": "Unexpected compact bicycle shape",
                "sceneDescription": "Minimal studio with one folded bicycle",
            }
            for ad in passing["ads"]
        ]
        pass_baseline = deterministic_builder1_integrity_checks(copy.deepcopy(passing))
        pass_evidence: list[dict] = []
        pass_with_evidence = deterministic_builder1_integrity_checks(
            copy.deepcopy(passing),
            integrity_evidence=pass_evidence,
        )
        self.assertEqual(pass_baseline, pass_with_evidence)

    def test_validate_builder1_campaign_integrity_includes_rejected_plan_on_failure(self) -> None:
        plan = _series_plan_from_dict(_literal_trap_plan_dict())
        upstream = _minimal_upstream()
        result = validate_builder1_campaign_integrity(plan, upstream=upstream, detected_language="en")
        self.assertFalse(result.ok)
        self.assertIn("literal_slogan_illustration", result.reasons)
        self.assertIsNotNone(result.rejected_plan_dict)
        assert result.rejected_plan_dict is not None
        self.assertEqual(result.rejected_plan_dict.get("brandSlogan"), plan.brand_slogan)

    def test_integrity_failure_path_does_not_persist_idea_memory(self) -> None:
        plan_dict = _literal_trap_plan_dict()
        plan = _series_plan_from_dict(plan_dict)
        upstream = _minimal_upstream()
        integrity = validate_builder1_campaign_integrity(plan, upstream=upstream, detected_language="en")
        self.assertFalse(integrity.ok)
        with patch("engine.builder1_planning_pipeline.persist_plan_idea_memory") as mock_persist_memory:
            diagnostic = build_integrity_failure_diagnostic(
                reasons=integrity.reasons,
                details=integrity.integrity_details or [],
                rejected_plan=integrity.rejected_plan_dict or plan_dict,
            )
            persist_integrity_failure_diagnostic(self.job_id, diagnostic, campaign_id=self.campaign_id)
            mock_persist_memory.assert_not_called()

    def test_rejected_plan_does_not_create_campaign_session(self) -> None:
        import app

        source = inspect.getsource(app._builder1_generate_initial)
        plan_idx = source.index("plan_builder1(")
        create_idx = source.index("create_campaign_session(")
        self.assertLess(plan_idx, create_idx)
        except_idx = source.index("except Builder1PlannerError")
        self.assertLess(except_idx, create_idx)

    def test_successful_integrity_result_has_no_rejected_plan(self) -> None:
        from tests.test_builder1_conceptual_lineage_integrity import _assemble, _upstream_for_plan

        plan = _assemble(ad_count=2)
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertTrue(result.ok, msg=str(result.reasons))
        self.assertIsNone(result.rejected_plan_dict)
        self.assertIsNone(result.integrity_details)

    def test_diagnostic_stored_on_job_field_not_in_public_result_shape(self) -> None:
        diagnostic = build_integrity_failure_diagnostic(
            reasons=["literal_slogan_illustration"],
            details=[],
            rejected_plan=_literal_trap_plan_dict(),
        )
        persist_integrity_failure_diagnostic(self.job_id, diagnostic)
        job = get_builder1_job(self.job_id)
        assert job is not None
        self.assertIn(INTEGRITY_DIAGNOSTIC_JOB_FIELD, job)
        self.assertNotIn("rejectedPlan", job.get("result") or {})

    def test_builder1_status_response_does_not_include_diagnostic_field(self) -> None:
        import app

        source = inspect.getsource(app.builder1_status)
        self.assertNotIn(INTEGRITY_DIAGNOSTIC_JOB_FIELD, source)
        self.assertNotIn("rejectedPlan", source)

    def test_planning_pipeline_wires_integrity_diagnostic_persistence(self) -> None:
        from engine.builder1_planning_pipeline import run_builder1_campaign_pipeline

        source = inspect.getsource(run_builder1_campaign_pipeline)
        self.assertIn("persist_integrity_failure_diagnostic", source)
        self.assertIn("job_id", source)
        self.assertIn("persist_plan_idea_memory", source)
        persist_idx = source.index("persist_plan_idea_memory")
        diagnostic_idx = source.index("persist_integrity_failure_diagnostic")
        self.assertLess(diagnostic_idx, persist_idx)

    def test_job_update_uses_existing_ttl(self) -> None:
        source = inspect.getsource(persist_integrity_failure_diagnostic)
        self.assertIn("update_builder1_job", source)
        from engine.builder1_jobs_store import update_builder1_job

        update_source = inspect.getsource(update_builder1_job)
        self.assertIn("JOB_TTL_SECONDS", update_source)
        self.assertEqual(JOB_TTL_SECONDS, 24 * 3600)

    def test_no_extra_model_calls_in_integrity_diagnostics_modules(self) -> None:
        from engine import builder1_integrity_diagnostics as diag_mod
        from engine import builder1_campaign_integrity as integrity_mod

        for source in (inspect.getsource(diag_mod), inspect.getsource(integrity_mod)):
            self.assertNotIn("model_caller", source)
            self.assertNotIn("openai", source.lower())

    def test_normal_supplied_name_planning_call_count_unchanged(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    def test_read_only_diagnostic_roundtrip_json_safe(self) -> None:
        diagnostic = build_integrity_failure_diagnostic(
            reasons=["literal_slogan_illustration"],
            details=[{"code": "literal_slogan_illustration", "branch": "scan_shortening_route_family_plan"}],
            rejected_plan=_literal_trap_plan_dict(),
        )
        encoded = json.dumps(diagnostic, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertTrue(decoded.get("rejectedPlanAvailable"))


if __name__ == "__main__":
    unittest.main()
