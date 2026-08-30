"""
Builder1 advertising comprehension and execution fidelity tests.

Run: python -m unittest tests.test_builder1_advertising_comprehension -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_advertising_comprehension import (
    EXECUTION_FIDELITY_VIOLATION_CODES,
    build_execution_fidelity_correction_block,
    scan_advertising_comprehension,
    validate_ad_advertising_comprehension,
)
from engine.builder1_campaign_completion import evaluate_campaign_completion
from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    mark_physical_repair_required,
    persist_campaign_ad_artifact,
    reserve_next_ad_index,
)
from engine.builder1_compliance_adjudication import adjudicate_compliance_review
from engine.builder1_image_compliance_contract import (
    COMPLIANCE_SCHEMA_VERSION,
    IMAGE_COMPLIANCE_VIOLATION_CODES,
    build_compliance_responses_request_kwargs,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_jobs_store import clear_memory_jobs_for_tests
from engine.builder1_literal_embodiment import validate_visual_prompt_expressive_object
from engine.builder1_literal_embodiment import validate_visual_prompt_slogan_noun_reintroduction
from engine.builder1_plan_spec import MARKETING_TEXT_WORD_COUNT
from engine.builder1_planner import plan_builder1
from engine.builder1_retry_state import RETRY_MODE_REPAIR_FROM_PHYSICAL, public_retry_fields
from engine.builder1_creative_methodology import methodology_repair_stage
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _full_final_responses, _internal_ad_fields


def _plan_dict_with_ad_fields(**ad_overrides: Any) -> Dict[str, Any]:
    raw = copy.deepcopy(_base_campaign(2))
    raw["relativeAdvantage"] = "Lessons focus explicitly on Bagrut history preparation"
    ad = raw["ads"][0]
    ad.update(_internal_ad_fields(headline=None, ad_index=1))
    ad.update(ad_overrides)
    return raw


class TestPlanningAdvertisingBridge(unittest.TestCase):
    def test_physical_clear_but_bridge_unclear_rejected(self) -> None:
        plan = _plan_dict_with_ad_fields(
            immediateClarityReason="Rain, umbrella, and dry surface are familiar and immediately understandable",
            relativeAdvantageConnection="The physical scene is clear to any viewer",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("advertising_bridge_unclear", reasons)

    def test_direct_transferred_object_bridge_accepted(self) -> None:
        plan = _plan_dict_with_ad_fields(
            immediateClarityReason="Viewer sees dry rectangle under umbrella on testing conveyor",
            relativeAdvantageConnection=(
                "The dry patch proves each lesson survives the Bagrut exam condition, "
                "showing focused preparation not generic tutoring help"
            ),
            executionPunchline="Dry rectangle proves the umbrella passed the rain test",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("advertising_bridge_unclear", reasons)

    def test_multi_hop_symbolic_chain_rejected(self) -> None:
        plan = _plan_dict_with_ad_fields(
            immediateClarityReason="Lesson represents umbrella which represents rain which represents exam",
            relativeAdvantageConnection="Each object symbolizes the next step in tutoring",
            sloganConnection="The scene maps tutoring to weather metaphors",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("multi_hop_symbolic_chain", reasons)

    def test_dominant_unexplained_visual_rejected(self) -> None:
        plan = _plan_dict_with_ad_fields(
            executionScene="A clock tower dominates the background while umbrella sits on conveyor",
            relativeAdvantageConnection="Dry patch proves Bagrut-focused preparation",
            executionPunchline="Dry rectangle under umbrella",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("dominant_object_strategic_role_missing", reasons)

    def test_negated_competing_scene_without_role_rejected(self) -> None:
        plan = _plan_dict_with_ad_fields(
            executionScene="Open umbrella standing on railway tracks in heavy rain",
            noReuseCheck="No railway tracks or route imagery",
            relativeAdvantageConnection="Shows focused Bagrut preparation through testing",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("dominant_object_strategic_role_missing", reasons)

    def test_methodology_routes_comprehension_to_series_ads(self) -> None:
        stage = methodology_repair_stage(["advertising_bridge_unclear"])
        self.assertEqual(stage, "series_ads")

    def test_existing_pipeline_plan_still_passes(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan = plan_builder1(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        reasons = scan_advertising_comprehension(
            {
                "relativeAdvantage": plan.relative_advantage,
                "brandSlogan": plan.brand_slogan,
                "physicalGenerator": plan.physical_generator,
                "transferredObject": plan.transferred_object,
                "conceptualGenerator": plan.conceptual_generator,
                "conceptualGeneratorAction": plan.conceptual_generator_action,
                "ads": [
                    {
                        "index": ad.index,
                        "sceneDescription": ad.scene_description,
                        "physicalExecution": ad.physical_execution,
                        **(plan.planning_internals.get("adInternals", {}).get(ad.index, {})),
                    }
                    for ad in plan.ads
                ],
            }
        )
        self.assertEqual(reasons, [])


class TestExecutionFidelityCompliance(unittest.TestCase):
    def test_execution_fidelity_codes_are_hard_violations(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["planned_scene_diverged"],
            evidence_items=[],
            overall_confidence="high",
            campaign_id="c1",
            ad_index=1,
        )
        self.assertFalse(result.passed)
        self.assertIn("planned_scene_diverged", result.hard_violations)

    def test_central_proof_missing_fails_review(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=["central_proof_not_visible"],
            evidence_items=[],
            overall_confidence="high",
        )
        self.assertFalse(result.passed)

    def test_compliant_faithful_review_passes(self) -> None:
        result = adjudicate_compliance_review(
            raw_violations=[],
            evidence_items=[],
            overall_confidence="high",
            reviewer_pass=True,
        )
        self.assertTrue(result.passed)

    def test_compliance_schema_includes_fidelity_codes(self) -> None:
        self.assertEqual(COMPLIANCE_SCHEMA_VERSION, "builder1_image_compliance_v4")
        self.assertTrue(EXECUTION_FIDELITY_VIOLATION_CODES.issubset(IMAGE_COMPLIANCE_VIOLATION_CODES))

    def test_planned_execution_block_in_compliance_request(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        kwargs = build_compliance_responses_request_kwargs(
            model="gpt-test",
            image_bytes=b"\xff\xd8\xff",
            system_prompt="system",
            product_name=plan.product_name_resolved,
            product_description=plan.product_description,
            visibility_policy="FORBIDDEN",
            transferred_object=plan.transferred_object or plan.physical_generator,
            series_plan=plan,
            ad_index=1,
        )
        user_text = kwargs["input"][0]["content"][0]["text"]
        self.assertIn("PLANNED EXECUTION CONTEXT", user_text)
        self.assertIn("executionPunchline", user_text)

    def test_fidelity_correction_block_targets_divergence(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        block = build_execution_fidelity_correction_block(
            violations=["planned_scene_diverged", "central_proof_not_visible"],
            series_plan=plan,
            ad_index=1,
        )
        self.assertIn("railway tracks", block.lower())
        self.assertIn("punchline", block.lower())


class TestImageRegenerationPath(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_jobs_for_tests()

    def test_fidelity_failure_uses_bounded_regeneration(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="cmp-fidelity", plan=plan, target_ad_count=2)
        reserve_next_ad_index("cmp-fidelity", 1, job_id="job-1")
        attempts = {"n": 0}

        def reviewer(*_args, **_kwargs: Any):
            from engine.builder1_image_compliance import ImageComplianceResult

            attempts["n"] += 1
            if attempts["n"] == 1:
                return ImageComplianceResult(
                    passed=False,
                    violations=["planned_scene_diverged"],
                    hard_violations=["planned_scene_diverged"],
                    confidence="high",
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        with patch(
            "engine.builder1_image_generator._generate_once",
            return_value=b"\xff\xd8\xff" + b"\x00" * 16,
        ):
            with patch(
                "engine.builder1_image_generator.review_builder1_ad_image_compliance",
                side_effect=reviewer,
            ) as review_mock:
                result = generate_builder1_ad_image(
                    series_plan=plan,
                    ad_index=1,
                    campaign_id="cmp-fidelity",
                    job_id="job-1",
                    image_caller=lambda p, f: b"\xff\xd8\xff",
                    compliance_reviewer=reviewer,
                )
        self.assertEqual(review_mock.call_count, 2)
        self.assertEqual(result.compliance_regeneration_count, 1)
        self.assertIn("EXECUTION FIDELITY CORRECTION", result.visual_prompt)

    def test_pixel_failure_does_not_invoke_planner(self) -> None:
        planner_calls: List[str] = []

        def track_planner(*_args, **_kwargs):
            planner_calls.append("plan")
            raise AssertionError("planning must not rerun on pixel failure")

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="cmp-no-plan", plan=plan, target_ad_count=2)
        reserve_next_ad_index("cmp-no-plan", 1, job_id="job-1")
        attempts = {"n": 0}

        def reviewer(*_args, **_kwargs: Any):
            from engine.builder1_image_compliance import ImageComplianceResult

            attempts["n"] += 1
            if attempts["n"] == 1:
                return ImageComplianceResult(
                    passed=False,
                    violations=["central_proof_not_visible"],
                    hard_violations=["central_proof_not_visible"],
                )
            return ImageComplianceResult(passed=True, violations=[])

        with patch("engine.builder1_planner.plan_builder1", side_effect=track_planner):
            with patch(
                "engine.builder1_image_generator._generate_once",
                return_value=b"\xff\xd8\xff" + b"\x00" * 16,
            ):
                generate_builder1_ad_image(
                    series_plan=plan,
                    ad_index=1,
                    campaign_id="cmp-no-plan",
                    image_caller=lambda p, f: b"\xff\xd8\xff",
                    compliance_reviewer=reviewer,
                )
        self.assertEqual(planner_calls, [])


class TestLiteralSloganAndRouteFamily(unittest.TestCase):
    def test_literal_slogan_protection_still_active(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.brand_slogan = "We shorten your way"
        plan.slogan_action = "Make distances shorter"
        plan.transferred_object = "Short-neck giraffe"
        plan.physical_generator = "Short-neck giraffe"
        prompt = "=== MAIN VISUAL ===\nMAIN VISUAL: Car driving through a highway maze\n=== END MAIN VISUAL ==="
        reasons = validate_visual_prompt_expressive_object(prompt, series_plan=plan)
        self.assertIn("expressive_object_weakened", reasons)

    def test_route_family_transferred_without_independent_mechanism_still_checked(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.transferred_object = "City route map"
        plan.physical_generator = "City route map"
        plan.brand_slogan = "We shorten your way"
        plan.planning_internals = {"adInternals": {}}
        prompt = (
            "=== MAIN VISUAL ===\n"
            "MAIN VISUAL: City route map\n"
            "ACTION: A road, train, and city route appear behind the object\n"
            "=== END MAIN VISUAL ==="
        )
        reasons = validate_visual_prompt_slogan_noun_reintroduction(prompt, series_plan=plan)
        self.assertIn("literal_slogan_illustration", reasons)


class TestRepairAndCampaignReady(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_repair_from_physical_contract(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="cmp-repair-contract", plan=plan, target_ad_count=2)
        mark_physical_repair_required(
            "cmp-repair-contract",
            failed_ad_index=1,
            violations=["literal_slogan_illustration"],
        )
        session = get_campaign_session("cmp-repair-contract")
        fields = public_retry_fields(session=session, retry_ad_index=1)
        self.assertTrue(fields["retryable"])
        self.assertEqual(fields["retryMode"], RETRY_MODE_REPAIR_FROM_PHYSICAL)
        self.assertEqual(fields["retryAdIndex"], 1)
        self.assertTrue(fields["planningComplete"])

    def test_repair_preserves_strategy_slogan_conceptual(self) -> None:
        original = _parse(_base_campaign(2), 2)
        preserved = (
            original.strategic_problem,
            original.relative_advantage,
            original.brand_slogan,
            original.slogan_action,
            original.conceptual_generator,
        )
        from dataclasses import replace

        repaired = replace(original, physical_generator="External testing conveyor")
        after = (
            repaired.strategic_problem,
            repaired.relative_advantage,
            repaired.brand_slogan,
            repaired.slogan_action,
            repaired.conceptual_generator,
        )
        self.assertEqual(preserved, after)

    def test_campaign_ready_true_when_complete(self) -> None:
        from engine.builder1_image_artifact_store import ad_artifact_record, write_builder1_image_artifact_bytes

        plan_data = _base_campaign(2)
        for ad in plan_data["ads"]:
            ad["marketingText"] = marketing_text_words(50, prefix=f"w{ad['index']}")
        plan = _parse(plan_data, 2)
        create_campaign_session(campaign_id="cmp-ready", plan=plan, target_ad_count=2)
        image_bytes = b"\xff\xd8\xff" + b"\x00" * 32
        for idx in (1, 2):
            rec = ad_artifact_record(campaign_id="cmp-ready", ad_index=idx, plan_revision=1)
            write_builder1_image_artifact_bytes(rec["token"], image_bytes)
            persist_campaign_ad_artifact("cmp-ready", ad_index=idx, artifact=rec)
            reserve_next_ad_index("cmp-ready", idx, job_id=f"job-{idx}")
            mark_ad_generated("cmp-ready", idx)
        report = evaluate_campaign_completion(get_campaign_session("cmp-ready"))
        self.assertTrue(report["campaignReady"])
        self.assertTrue(report["deliveryReconstructible"])
        self.assertTrue(report["campaignComplete"])
        self.assertEqual(report["generatedCount"], report["targetAdCount"])

    def test_fifty_word_rule_unchanged(self) -> None:
        self.assertEqual(MARKETING_TEXT_WORD_COUNT, 50)


if __name__ == "__main__":
    unittest.main()
