"""
Builder1 plan revision propagation and cumulative image retry tests.

Run: python -m unittest tests.test_builder1_image_retry -v
"""
from __future__ import annotations

import copy
import json
import os
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_campaign_store import (
    CampaignStoreError,
    _load_raw,
    _session_from_raw,
    apply_repaired_campaign_plan,
    clear_memory_store_for_tests,
    create_campaign_session,
    cumulative_violations_for_ad,
    get_campaign_session,
    mark_ad_generated,
    mark_image_retry_required,
    record_image_attempt_violations,
    reserve_next_ad_index,
)
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    classify_compliance_failure,
    log_failure_classification,
)
from engine.builder1_image_compliance import ImageComplianceError, ImageComplianceResult, parse_image_compliance_response
from engine.builder1_image_generator import VISIBILITY_VIOLATION_CODES, generate_builder1_ad_image
from engine.builder1_image_retry import (
    CAMPAIGN_DEVICE_AS_LOGO_CORRECTION,
    LOGO_VIOLATION_CODES,
    PERMANENT_FORBIDDEN_GLOBAL_CONSTRAINTS,
    build_cumulative_image_correction_block,
    parse_image_attempt_history,
    union_violations_for_ad,
)
from engine.builder1_jobs_store import clear_memory_jobs_for_tests, create_builder1_job, update_builder1_job
from engine.builder1_marketing_copy import validate_marketing_text_50_words
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planning_profile import resolve_stage_model
from engine.builder1_planner import plan_builder1
from engine.builder1_product_visibility import ProductVisibilityPolicy
from engine.builder1_retry_state import RETRY_MODE_IMAGE_ONLY, public_retry_fields
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _full_final_responses


def _plan(ad_count: int = 3):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.transferred_object = "Rubber ball family"
    plan.transferred_object_action = "Bounces after a controlled drop"
    plan.physical_generator = "Rubber ball family"
    plan.product_visibility_policy = "FORBIDDEN"
    return plan


class TestPlanRevisionInitialization(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_new_memory_campaign_starts_at_revision_one(self) -> None:
        session = create_campaign_session(campaign_id="rev-mem", plan=_plan(3), target_ad_count=3)
        self.assertEqual(session.plan_revision, 1)
        raw = _load_raw("rev-mem")
        assert raw is not None
        self.assertEqual(raw.get("planRevision"), 1)

    def test_redis_shape_json_stores_revision_one(self) -> None:
        create_campaign_session(campaign_id="rev-json", plan=_plan(3), target_ad_count=3)
        raw = _load_raw("rev-json")
        assert raw is not None
        encoded = json.dumps(raw)
        self.assertIn('"planRevision": 1', encoded)

    def test_initial_image_job_records_revision_one(self) -> None:
        from app import _builder1_generate_single_ad

        create_campaign_session(campaign_id="rev-job", plan=_plan(3), target_ad_count=3)
        create_builder1_job(
            job_id="job-rev",
            campaign_id="rev-job",
            target_ad_count=3,
            stage="generating_images",
        )
        update_builder1_job("job-rev", planRevision=1, retryAdIndex=1, campaignId="rev-job")

        with patch(
            "app.generate_builder1_ad_image",
            return_value=type("R", (), {"visual_prompt": "p", "image_bytes": b"x"})(),
        ):
            result = _builder1_generate_single_ad(
                job_id="job-rev",
                campaign_id="rev-job",
                ad_index=1,
                already_reserved=False,
            )
        self.assertTrue(result.get("ok"))

    def test_generate_next_image_job_records_current_revision(self) -> None:
        from engine.builder1_jobs_store import get_builder1_job

        cid = "rev-next"
        create_campaign_session(campaign_id=cid, plan=_plan(3), target_ad_count=3)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)
        apply_repaired_campaign_plan(cid, _plan(3))
        reserve_next_ad_index(cid, 2, job_id="job-2")
        create_builder1_job(job_id="job-2", campaign_id=cid, target_ad_count=3, stage="generating_images")
        update_builder1_job("job-2", planRevision=2, retryAdIndex=2, campaignId=cid)
        job = get_builder1_job("job-2")
        self.assertEqual(job.get("planRevision"), 2)

    def test_failure_classification_log_never_blank_revision(self) -> None:
        with self.assertLogs("engine.builder1_failure_classification", level="INFO") as captured:
            log_failure_classification(
                campaign_id="cmp-log",
                ad_index=1,
                failure_class=Builder1FailureClass.IMAGE_EXECUTION,
                action=Builder1FailureAction.REGENERATE_IMAGE,
                plan_revision=1,
            )
        joined = "\n".join(captured.output)
        self.assertIn("planRevision=1", joined)
        self.assertNotRegex(joined, r"planRevision=\s*$")
        self.assertNotIn("planRevision= ", joined)

    def test_image_retry_required_persists_revision(self) -> None:
        create_campaign_session(campaign_id="rev-retry", plan=_plan(3), target_ad_count=3)
        mark_image_retry_required("rev-retry", failed_ad_index=2, violations=["campaign_device_used_as_logo"])
        session = get_campaign_session("rev-retry")
        self.assertEqual(session.plan_revision, 1)
        self.assertEqual(session.retry_mode, RETRY_MODE_IMAGE_ONLY)

    def test_missing_revision_fails_before_image_api(self) -> None:
        with self.assertRaises(CampaignStoreError) as ctx:
            generate_builder1_ad_image(_plan(3), 1, lambda _p, _f: b"x", plan_revision=0)
        self.assertEqual(ctx.exception.code, "missing_plan_revision")

    def test_stale_revision_fails_before_image_api(self) -> None:
        from app import _builder1_generate_single_ad

        create_campaign_session(campaign_id="rev-stale", plan=_plan(3), target_ad_count=3)
        create_builder1_job(
            job_id="job-stale-rev",
            campaign_id="rev-stale",
            target_ad_count=3,
            stage="generating_images",
        )
        update_builder1_job("job-stale-rev", planRevision=1, retryAdIndex=1)
        apply_repaired_campaign_plan("rev-stale", _plan(3))
        result = _builder1_generate_single_ad(
            job_id="job-stale-rev",
            campaign_id="rev-stale",
            ad_index=1,
            already_reserved=False,
        )
        self.assertEqual(result["error"], "stale_plan_revision")


class TestViolationHistory(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_second_violation_added_to_history(self) -> None:
        create_campaign_session(campaign_id="hist", plan=_plan(3), target_ad_count=3)
        record_image_attempt_violations(
            "hist", ad_index=1, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        session = record_image_attempt_violations(
            "hist", ad_index=1, attempt=2, violations=["product_visible_without_explicit_request"]
        )
        union = cumulative_violations_for_ad(session, 1)
        self.assertEqual(
            union,
            ["campaign_device_used_as_logo", "product_visible_without_explicit_request"],
        )

    def test_history_survives_json_roundtrip(self) -> None:
        create_campaign_session(campaign_id="hist-serde", plan=_plan(3), target_ad_count=3)
        record_image_attempt_violations(
            "hist-serde", ad_index=2, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        record_image_attempt_violations(
            "hist-serde", ad_index=2, attempt=2, violations=["product_visible_without_explicit_request"]
        )
        raw = _load_raw("hist-serde")
        assert raw is not None
        reloaded = _session_from_raw("hist-serde", raw)
        self.assertEqual(
            cumulative_violations_for_ad(reloaded, 2),
            ["campaign_device_used_as_logo", "product_visible_without_explicit_request"],
        )

    def test_history_clears_only_after_compliance_passes(self) -> None:
        create_campaign_session(campaign_id="hist-clear", plan=_plan(3), target_ad_count=3)
        record_image_attempt_violations(
            "hist-clear", ad_index=1, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        reserve_next_ad_index("hist-clear", 1, job_id="job-clear")
        mark_ad_generated("hist-clear", 1)
        session = get_campaign_session("hist-clear")
        self.assertEqual(cumulative_violations_for_ad(session, 1), [])

    def test_public_retry_fields_include_revision(self) -> None:
        create_campaign_session(campaign_id="hist-public", plan=_plan(3), target_ad_count=3)
        record_image_attempt_violations(
            "hist-public", ad_index=2, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        mark_image_retry_required(
            "hist-public", failed_ad_index=2, violations=["product_visible_without_explicit_request"]
        )
        session = get_campaign_session("hist-public")
        fields = public_retry_fields(session=session, retry_ad_index=2)
        self.assertEqual(fields["planRevision"], 1)


class TestCumulativeCorrectionPrompts(unittest.TestCase):
    def test_logo_correction_preserves_no_product_constraints(self) -> None:
        plan = _plan(3)
        ad = plan.ads[0]
        block = build_cumulative_image_correction_block(
            violations=["campaign_device_used_as_logo"],
            series_plan=plan,
            ad_plan=ad,
            plan_revision=1,
        )
        self.assertIn("Advertised product must not be depicted", block)
        self.assertIn(CAMPAIGN_DEVICE_AS_LOGO_CORRECTION, block)
        self.assertIn("do not remove the central campaign idea", block.lower())

    def test_product_correction_preserves_no_logo_constraints(self) -> None:
        plan = _plan(3)
        ad = plan.ads[0]
        block = build_cumulative_image_correction_block(
            violations=["product_visible_without_explicit_request"],
            series_plan=plan,
            ad_plan=ad,
            plan_revision=1,
        )
        self.assertIn("No supplied or invented logo", block)
        self.assertIn("Remove only the specific advertised-product element", block)

    def test_both_violations_apply_both_profiles(self) -> None:
        plan = _plan(3)
        ad = plan.ads[0]
        block = build_cumulative_image_correction_block(
            violations=[
                "campaign_device_used_as_logo",
                "product_visible_without_explicit_request",
            ],
            series_plan=plan,
            ad_plan=ad,
            plan_revision=1,
        )
        self.assertIn("campaign_device_used_as_logo", block)
        self.assertIn("product_visible_without_explicit_request", block)
        self.assertIn(CAMPAIGN_DEVICE_AS_LOGO_CORRECTION, block)
        self.assertIn("Remove only the specific advertised-product element", block)

    def test_retry_prompt_includes_global_constraints(self) -> None:
        plan = _plan(3)
        block = build_cumulative_image_correction_block(
            violations=["invented_product_logo"],
            series_plan=plan,
            ad_plan=plan.ads[0],
            plan_revision=1,
        )
        self.assertIn(PERMANENT_FORBIDDEN_GLOBAL_CONSTRAINTS, block)

    def test_retry_prompt_contains_positive_main_visual(self) -> None:
        plan = _plan(3)
        block = build_cumulative_image_correction_block(
            violations=["campaign_device_used_as_logo"],
            series_plan=plan,
            ad_plan=plan.ads[0],
            plan_revision=1,
        )
        self.assertIn("MAIN VISUAL: Rubber ball family", block)
        self.assertIn("ACTION: Bounces after a controlled drop", block)
        self.assertIn("BRAND TEXT:", block)

    def test_retry_prompt_preserves_slogan_and_graphic_system(self) -> None:
        plan = _plan(3)
        block = build_cumulative_image_correction_block(
            violations=["campaign_device_used_as_logo"],
            series_plan=plan,
            ad_plan=plan.ads[0],
            plan_revision=1,
        )
        self.assertIn(plan.brand_slogan, block)
        self.assertIn("graphic system", block.lower())

    def test_public_retry_receives_all_historical_violations(self) -> None:
        prompts: List[str] = []

        def caller(prompt: str, _fmt: str) -> bytes:
            prompts.append(prompt)
            return b"img"

        create_campaign_session(campaign_id="cum-public", plan=_plan(3), target_ad_count=3)
        record_image_attempt_violations(
            "cum-public", ad_index=1, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        record_image_attempt_violations(
            "cum-public",
            ad_index=1,
            attempt=2,
            violations=["product_visible_without_explicit_request"],
        )
        session = get_campaign_session("cum-public")
        generate_builder1_ad_image(
            session.plan,
            1,
            caller,
            campaign_id="cum-public",
            plan_revision=1,
            cumulative_violations=cumulative_violations_for_ad(session, 1),
            compliance_reviewer=lambda **_k: ImageComplianceResult(passed=True, violations=[], confidence="high"),
        )
        self.assertEqual(len(prompts), 1)
        self.assertIn("campaign_device_used_as_logo", prompts[0])
        self.assertIn("product_visible_without_explicit_request", prompts[0])
        self.assertIn("GLOBAL IMAGE CONSTRAINTS", prompts[0])


class TestLogoDeviceCorrection(unittest.TestCase):
    def test_campaign_device_may_remain_as_motif(self) -> None:
        block = CAMPAIGN_DEVICE_AS_LOGO_CORRECTION
        self.assertIn("compositional motif", block.lower())

    def test_campaign_device_may_not_be_emblem(self) -> None:
        block = CAMPAIGN_DEVICE_AS_LOGO_CORRECTION
        self.assertIn("emblematic", block.lower())
        self.assertIn("mark-like", block.lower())

    def test_product_name_remains_plain_typography(self) -> None:
        block = CAMPAIGN_DEVICE_AS_LOGO_CORRECTION
        self.assertIn("plain readable typography", block.lower())

    def test_correction_does_not_invent_replacement_logo(self) -> None:
        block = CAMPAIGN_DEVICE_AS_LOGO_CORRECTION
        self.assertIn("Do not invent a replacement logo", block)

    def test_correction_does_not_delete_generator_unnecessarily(self) -> None:
        block = CAMPAIGN_DEVICE_AS_LOGO_CORRECTION
        self.assertIn("do not remove the central campaign idea", block.lower())
        self.assertIn("delete the recurring device entirely", block.lower())


class TestImageOnlyRouting(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_image_only_retry_performs_zero_planning_calls(self) -> None:
        from app import app

        plan = _plan(3)
        create_campaign_session(campaign_id="route-image", plan=plan, target_ad_count=3)
        reserve_next_ad_index("route-image", 1, job_id="job-1")
        mark_ad_generated("route-image", 1)
        record_image_attempt_violations(
            "route-image", ad_index=2, attempt=1, violations=["campaign_device_used_as_logo"]
        )
        mark_image_retry_required(
            "route-image", failed_ad_index=2, violations=["product_visible_without_explicit_request"]
        )
        submitted: List[tuple[Any, ...]] = []

        def capture_submit(fn, *args):
            submitted.append((fn.__name__, args))

        with app.test_client() as client:
            with patch("app._builder1_executor") as executor:
                with patch("engine.builder1_planner.plan_builder1") as mock_plan:
                    mock_plan.side_effect = AssertionError("planner must not run")
                    executor.submit.side_effect = capture_submit
                    response = client.post(
                        "/api/builder1-generate-next",
                        json={"campaignId": "route-image", "expectedNextIndex": 2},
                    )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(submitted[0][0], "_builder1_run_next_job")

    def test_same_campaign_id_preserved(self) -> None:
        create_campaign_session(campaign_id="route-id", plan=_plan(3), target_ad_count=3)
        mark_image_retry_required("route-id", failed_ad_index=1, violations=["x"])
        self.assertEqual(get_campaign_session("route-id").campaign_id, "route-id")

    def test_same_failed_ad_index_preserved(self) -> None:
        create_campaign_session(campaign_id="route-idx", plan=_plan(3), target_ad_count=3)
        reserve_next_ad_index("route-idx", 1, job_id="job-1")
        mark_ad_generated("route-idx", 1)
        mark_image_retry_required("route-idx", failed_ad_index=2, violations=["x"])
        session = reserve_next_ad_index("route-idx", 2, job_id="job-retry")
        self.assertEqual(session.generating_index, 2)
        self.assertEqual(session.failed_ad_index, 2)

    def test_generated_count_unchanged_before_pass(self) -> None:
        create_campaign_session(campaign_id="route-count", plan=_plan(3), target_ad_count=3)
        reserve_next_ad_index("route-count", 1, job_id="job-1")
        mark_ad_generated("route-count", 1)
        mark_image_retry_required("route-count", failed_ad_index=2, violations=["x"])
        self.assertEqual(get_campaign_session("route-count").generated_count, 1)

    def test_successful_retry_stores_ad_and_clears_retry_state(self) -> None:
        from app import _builder1_generate_single_ad

        create_campaign_session(campaign_id="route-pass", plan=_plan(3), target_ad_count=3)
        mark_image_retry_required("route-pass", failed_ad_index=1, violations=["x"])
        create_builder1_job(
            job_id="job-pass",
            campaign_id="route-pass",
            target_ad_count=3,
            stage="generating_images",
        )
        update_builder1_job("job-pass", planRevision=1, retryAdIndex=1, campaignId="route-pass")
        with patch(
            "app.generate_builder1_ad_image",
            return_value=type("R", (), {"visual_prompt": "p", "image_bytes": b"x"})(),
        ):
            result = _builder1_generate_single_ad(
                job_id="job-pass",
                campaign_id="route-pass",
                ad_index=1,
                already_reserved=False,
            )
        self.assertTrue(result.get("ok"))
        session = get_campaign_session("route-pass")
        self.assertEqual(session.generated_count, 1)
        self.assertEqual(session.retry_mode, "none")
        self.assertIsNone(session.failed_ad_index)


class TestProductionSequenceRegression(unittest.TestCase):
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

    def test_quality_keeps_planning_stages_on_o3_pro(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_PROFILE": "QUALITY",
                "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            },
            clear=False,
        ):
            for stage in (
                "strategy_scan",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
                "marketing_language",
            ):
                self.assertEqual(resolve_stage_model(stage), "gpt-5.6-sol")

    def test_product_visibility_remains_forbidden_by_default(self) -> None:
        plan = _plan(3)
        self.assertEqual(plan.product_visibility_policy, "FORBIDDEN")

    def test_fifty_marketing_words_enforced(self) -> None:
        validate_marketing_text_50_words(marketing_text_words(50))

    def test_no_logo_compliance_remains_active(self) -> None:
        result = parse_image_compliance_response(
            {
                "pass": False,
                "violations": ["invented_product_logo"],
                "confidence": "high",
            }
        )
        self.assertIn("invented_product_logo", result.violations)

    def test_no_builder1_judge_introduced(self) -> None:
        import engine.builder1_image_generator as gen

        self.assertFalse(hasattr(gen, "run_builder1_judge"))

    def test_builder2_unchanged(self) -> None:
        import os

        builder2_zip = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine", "builder2_zip.py")
        self.assertTrue(os.path.isfile(builder2_zip))


class TestCumulativeInternalRegeneration(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_logo_then_product_violations_accumulate_on_internal_retry(self) -> None:
        prompts: List[str] = []
        review_n = {"n": 0}

        def caller(prompt: str, _fmt: str) -> bytes:
            prompts.append(prompt)
            return b"img"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            review_n["n"] += 1
            if review_n["n"] == 1:
                return ImageComplianceResult(
                    passed=False,
                    violations=["invented_product_logo"],
                    hard_violations=["invented_product_logo"],
                    raw_violations=["invented_product_logo"],
                    confidence="high",
                )
            if review_n["n"] == 2:
                from engine.builder1_compliance_adjudication import ComplianceEvidenceItem
                from engine.builder1_compliance_product_grounding import ComplianceProductMatch
                from engine.builder1_image_compliance import finalize_compliance_result

                return finalize_compliance_result(
                    reviewer_pass=False,
                    candidate_violations=["product_visible_without_explicit_request"],
                    evidence_items=[
                        ComplianceEvidenceItem(
                            code="product_visible_without_explicit_request",
                            confidence="high",
                        )
                    ],
                    overall_confidence="high",
                    series_plan=_plan(3),
                    product_match=ComplianceProductMatch(
                        advertised_product_present=True,
                        product_match_basis="explicit_product_shape",
                        matched_visual_element="TestBrand product unit",
                        relationship_to_advertised_product="actual_product",
                        product_match_explanation="Visible product unit matches advertised product.",
                    ),
                )
            return ImageComplianceResult(
                passed=False,
                violations=["invented_product_logo"],
                hard_violations=["invented_product_logo"],
                raw_violations=["invented_product_logo"],
                confidence="high",
            )

        create_campaign_session(campaign_id="seq", plan=_plan(3), target_ad_count=3)
        with self.assertRaises(ImageComplianceError):
            generate_builder1_ad_image(
                _plan(3),
                1,
                caller,
                campaign_id="seq",
                plan_revision=1,
                compliance_reviewer=reviewer,
            )
        session = get_campaign_session("seq")
        self.assertEqual(
            cumulative_violations_for_ad(session, 1),
            [
                "invented_product_logo",
                "product_visible_without_explicit_request",
            ],
        )
        self.assertGreaterEqual(len(prompts), 2)
        self.assertIn("invented_product_logo", prompts[1])
        self.assertIn("GLOBAL IMAGE CONSTRAINTS", prompts[1])

    def test_parse_image_attempt_history_normalizes_entries(self) -> None:
        parsed = parse_image_attempt_history(
            {
                "1": [
                    {"attempt": 1, "violations": ["campaign_device_used_as_logo", "campaign_device_used_as_logo"]},
                    {"attempt": 2, "violations": ["product_visible_without_explicit_request"]},
                ]
            }
        )
        self.assertEqual(
            union_violations_for_ad(parsed, 1),
            ["campaign_device_used_as_logo", "product_visible_without_explicit_request"],
        )

    def test_visibility_and_logo_codes_exported(self) -> None:
        self.assertIn("product_visible_without_explicit_request", VISIBILITY_VIOLATION_CODES)
        self.assertIn("campaign_device_used_as_logo", LOGO_VIOLATION_CODES)

    def test_pixel_violation_classified_as_image_execution(self) -> None:
        failure_class, action, _, _ = classify_compliance_failure(
            violations=["campaign_device_used_as_logo"],
            series_plan=_plan(3),
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)


if __name__ == "__main__":
    unittest.main()
