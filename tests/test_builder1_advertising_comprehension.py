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
    CATEGORY_INTEGRITY_VIOLATION_CODES,
    EXECUTION_FIDELITY_VIOLATION_CODES,
    assess_everyday_familiarity,
    build_analogy_repair_guidance_block,
    build_execution_fidelity_correction_block,
    build_planned_execution_compliance_block,
    detect_competing_category_visual,
    detect_mechanism_not_observable,
    detect_public_analogy_too_complex,
    scan_advertising_comprehension,
    validate_ad_advertising_comprehension,
)
from engine.builder1_idea_memory import (
    IdeaMemoryRecord,
    IdeaMemoryScope,
    IdeaMemorySnapshot,
    build_physical_family_novelty_block,
    build_stage_memory_block,
    classify_physical_generator_family,
)
from engine.builder1_planning_contract import build_brand_physical_repair_prompt
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    classify_compliance_failure,
    plan_has_category_integrity_violation,
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
    raw["strategicProblem"] = "Parents may undervalue divided classroom attention"
    raw["productDescription"] = "Private history tutor for Bagrut preparation"
    raw["productName"] = "Amir Gottlieb"
    raw["productNameResolved"] = "Amir Gottlieb"
    for i, ad in enumerate(raw["ads"], start=1):
        ad.update(_internal_ad_fields(headline=ad.get("headline"), ad_index=i))
    raw["ads"][0].update(ad_overrides)
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


def _tutor_plan_dict(**ad_overrides: Any) -> Dict[str, Any]:
    plan = _plan_dict_with_ad_fields(**ad_overrides)
    plan["productDescription"] = "Private HISTORY tutor preparing students for Bagrut"
    plan["relativeAdvantage"] = (
        "Amir's teaching time and attention are dedicated to one student and Bagrut preparation"
    )
    plan["physicalGenerator"] = "Sports camera autofocus continuously following one gymnast"
    plan["transferredObject"] = plan["physicalGenerator"]
    plan["transferredObjectAction"] = "Autofocus locks on one gymnast while others stay blurred"
    return plan


def _tutor_series_plan(**ad_overrides: Any):
    data = copy.deepcopy(_base_campaign(2))
    data["productDescription"] = "Private HISTORY tutor preparing students for Bagrut"
    data["relativeAdvantage"] = (
        "Amir's teaching time and attention are dedicated to one student and Bagrut preparation"
    )
    data["physicalGenerator"] = "Sports camera autofocus continuously following one gymnast"
    data["transferredObject"] = data["physicalGenerator"]
    for i, ad in enumerate(data["ads"], start=1):
        ad.update(_internal_ad_fields(headline=ad.get("headline"), ad_index=i))
    data["ads"][0].update(ad_overrides)
    return _parse(data, 2)


class TestCategoryIntegrity(unittest.TestCase):
    def test_history_tutor_gymnastics_execution_rejected(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Young gymnast on gymnastics training floor with other gymnasts blurred",
            executionSubject="Gymnast in sharp focus",
            physicalExecution="Sports action photo of gymnast on floor",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("competing_category_visual", reasons)

    def test_history_tutor_music_lesson_rejected(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Child playing piano with violin in background",
            executionSubject="Piano student at keyboard",
            physicalExecution="Music lesson scene with piano",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("competing_category_visual", reasons)

    def test_instructional_service_vs_instructional_service_rejected(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Art studio with canvas easel and painting lesson",
            physicalExecution="Painting instruction at easel in studio",
        )
        self.assertTrue(
            detect_competing_category_visual(plan_dict=plan, ad=plan["ads"][0])
        )

    def test_cross_domain_metaphor_without_competing_service_may_pass(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Industrial testing conveyor with open umbrella under rain",
            executionSubject="Umbrella on conveyor with visible dry rectangle",
            physicalExecution="Umbrella tested on conveyor belt under rain",
            physicalGenerator="Open umbrella on industrial testing conveyor",
            transferredObject="Open umbrella on industrial testing conveyor",
            immediateClarityReason=(
                "Dry rectangle under umbrella proves the test condition while rain wets surroundings"
            ),
            relativeAdvantageConnection=(
                "Each lesson is tested against Bagrut readiness like the dry patch proves the umbrella passed"
            ),
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("competing_category_visual", reasons)

    def test_category_relevance_reason_cannot_override_integrity_failure(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Gymnastics training floor with gymnast in focus",
            categoryRelevanceReason=(
                "Does not show teacher or classroom; uses external camera system to illustrate educational benefit"
            ),
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("competing_category_visual", reasons)

    def test_planning_routes_category_failure_to_brand_physical(self) -> None:
        self.assertEqual(
            methodology_repair_stage(["competing_category_visual"]),
            "brand_physical",
        )
        self.assertEqual(
            methodology_repair_stage(["advertising_mechanism_not_observable"]),
            "brand_physical",
        )


class TestObservableMechanism(unittest.TestCase):
    def test_autofocus_claim_with_blur_result_only_rejected(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="One gymnast sharp with blurred gymnasts on floor",
            executionSubject="Gymnast in focus",
            immediateClarityReason="Sharp foreground against blurred movement is a familiar photography effect",
        )
        self.assertTrue(
            detect_mechanism_not_observable(plan_dict=plan, ad=plan["ads"][0])
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("advertising_mechanism_not_observable", reasons)

    def test_observable_mechanism_with_device_visible_passes(self) -> None:
        plan = _tutor_plan_dict(
            physicalGenerator="Sports camera with visible autofocus box tracking one gymnast",
            transferredObject="Sports camera with visible autofocus box tracking one gymnast",
            executionScene="Camera viewfinder showing autofocus box locked on one gymnast",
            executionSubject="Camera screen with autofocus indicator on gymnast",
            physicalExecution="Visible camera display with focus reticle on one athlete",
            immediateClarityReason="Viewer sees the autofocus box lock onto one gymnast on the camera screen",
            relativeAdvantageConnection="Autofocus tracking one subject proves dedicated individual attention",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("advertising_mechanism_not_observable", reasons)

    def test_visual_effect_only_clarity_insufficient(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Gymnast sharp, others blurred",
            immediateClarityReason="Sharp landing against blurred movement is a familiar sports photography effect",
            relativeAdvantageConnection="The photo looks dynamic",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("advertising_bridge_unclear", reasons)


class TestCategoryComplianceRouting(unittest.TestCase):
    def test_plan_category_failure_routes_to_physical_repair(self) -> None:
        plan = _tutor_series_plan(
            executionScene="Gymnast on gymnastics floor",
            physicalExecution="Gymnastics training photo",
        )
        self.assertTrue(plan_has_category_integrity_violation(plan))
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["competing_category_visual"],
            hard_violations=["competing_category_visual"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.PLAN_CONTRADICTION)
        self.assertEqual(action, Builder1FailureAction.REPAIR_FROM_PHYSICAL)
        self.assertTrue(evidence.get("planCategoryIntegrityFailure"))

    def test_pixel_only_fidelity_still_regenerates_image(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["planned_scene_diverged"],
            hard_violations=["planned_scene_diverged"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)
        self.assertFalse(evidence.get("planCategoryIntegrityFailure"))

    def test_category_codes_in_compliance_schema(self) -> None:
        self.assertTrue(CATEGORY_INTEGRITY_VIOLATION_CODES.issubset(EXECUTION_FIDELITY_VIOLATION_CODES))

    def test_compliance_adjudication_hard_fails_category(self) -> None:
        from engine.builder1_compliance_adjudication import adjudicate_compliance_review

        result = adjudicate_compliance_review(
            raw_violations=["competing_category_visual"],
            evidence_items=[],
            overall_confidence="high",
        )
        self.assertFalse(result.passed)
        self.assertIn("competing_category_visual", result.hard_violations)


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


class TestPublicAnalogySimplicity(unittest.TestCase):
    def test_magnet_cross_domain_with_direct_bridge_may_pass(self) -> None:
        plan = _tutor_plan_dict(
            physicalGenerator="Large magnet attracting one metal sphere",
            transferredObject="Large magnet attracting one metal sphere",
            transferredObjectAction="Magnet pulls one metal sphere while others stay untouched",
            executionScene="Magnet on table pulling one metal sphere toward it",
            executionSubject="Magnet and one attracted sphere",
            physicalExecution="Visible magnet pull on one metal object",
            immediateClarityReason="Viewer sees the magnet pull one metal sphere while other spheres stay still",
            relativeAdvantageConnection=(
                "Pulling one object proves Amir dedicates teaching time and attention to one student"
            ),
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("competing_category_visual", reasons)
        self.assertNotIn("public_analogy_too_complex", reasons)

    def test_autofocus_technical_chain_rejected(self) -> None:
        plan = _tutor_plan_dict(
            immediateClarityReason=(
                "Autofocus plane tracks one gymnast through sensor feedback calibration loop"
            ),
            relativeAdvantageConnection=(
                "Optical tracking represents personalized adaptation through dynamic correction"
            ),
            sloganConnection="Gymnast maps to autofocus which maps to tutoring",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertTrue(
            "public_analogy_too_complex" in reasons
            or "advertising_mechanism_not_observable" in reasons
            or "multi_hop_symbolic_chain" in reasons
        )

    def test_two_sentence_simple_bridge_passes(self) -> None:
        plan = _tutor_plan_dict(
            physicalGenerator="Open umbrella on testing conveyor",
            transferredObject="Open umbrella on testing conveyor",
            executionScene="Umbrella on conveyor with dry rectangle under rain",
            immediateClarityReason="Rain wets everything except the dry rectangle protected by the umbrella",
            relativeAdvantageConnection=(
                "Each lesson is tested against Bagrut readiness like the dry patch proves focused preparation"
            ),
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("public_analogy_too_complex", reasons)

    def test_familiar_effect_only_clarity_rejected(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Gymnast sharp, others blurred",
            immediateClarityReason="Sharp landing against blurred movement is a familiar sports photography effect",
            relativeAdvantageConnection="The photo looks dynamic",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("advertising_bridge_unclear", reasons)

    def test_simple_physical_preferred_over_technical_explanation(self) -> None:
        plan = _tutor_plan_dict(
            physicalGenerator="Magnet selecting one metal object",
            transferredObject="Magnet selecting one metal object",
            executionScene="Magnet visibly pulls one metal ball",
            immediateClarityReason="Viewer sees magnet pull one metal ball while others remain still",
            relativeAdvantageConnection="One attracted object proves dedicated attention to one student",
        )
        self.assertFalse(detect_public_analogy_too_complex(plan_dict=plan, ad=plan["ads"][0]))

    def test_three_plus_hidden_mappings_rejected(self) -> None:
        plan = _tutor_plan_dict(
            immediateClarityReason="Gymnast represents autofocus represents sensor loop represents tutoring",
            relativeAdvantageConnection="Each step symbolizes the next mapping in the chain",
            sloganConnection="Scene maps gymnastics to camera to education",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("public_analogy_too_complex", reasons)

    def test_category_integrity_does_not_require_literal_category_imagery(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Magnet on desk pulling one metal sphere",
            physicalExecution="Magnet attracting one sphere",
            categoryRelevanceReason="Uses external magnet metaphor rather than classroom imagery",
        )
        self.assertFalse(detect_competing_category_visual(plan_dict=plan, ad=plan["ads"][0]))

    def test_cross_domain_not_rejected_merely_for_domain_difference(self) -> None:
        plan = _tutor_plan_dict(
            executionScene="Parachute slowing one package while others fall fast",
            physicalExecution="Parachute descent with one slowed package",
            immediateClarityReason="Viewer sees one package descend slowly under an open parachute",
            relativeAdvantageConnection="Slowing one package proves individualized pacing for one student",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertNotIn("competing_category_visual", reasons)

    def test_everyday_familiarity_universal_for_magnet(self) -> None:
        self.assertEqual(
            assess_everyday_familiarity("Magnet pulls one metal object"),
            "universal",
        )

    def test_everyday_familiarity_technical_for_autofocus(self) -> None:
        level = assess_everyday_familiarity(
            "Autofocus plane with sensor feedback calibration optical tracking"
        )
        self.assertIn(level, ("specialized", "technical"))

    def test_repair_guidance_demands_simpler_analogy_not_literal_category(self) -> None:
        block = build_analogy_repair_guidance_block(["public_analogy_too_complex"])
        self.assertIn("Do NOT replace the visual with literal product/category imagery", block)
        self.assertIn("SAME relative advantage", block)

    def test_brand_physical_repair_prompt_includes_analogy_guidance(self) -> None:
        prompt = build_brand_physical_repair_prompt(
            broken_json="{}",
            reasons=["public_analogy_too_complex"],
        )
        self.assertIn("ANALOGY REPAIR", prompt)
        self.assertNotIn("show the product/category directly", prompt.lower())

    def test_public_analogy_not_recoverable_in_fidelity_codes(self) -> None:
        self.assertIn("public_analogy_not_recoverable", EXECUTION_FIDELITY_VIOLATION_CODES)

    def test_compliance_block_includes_public_comprehension(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        block = build_planned_execution_compliance_block(plan, ad_index=1)
        self.assertIn("Public comprehension", block)
        self.assertIn("public_analogy_not_recoverable", block)

    def test_methodology_routes_public_analogy_to_brand_physical(self) -> None:
        self.assertEqual(
            methodology_repair_stage(["public_analogy_too_complex"]),
            "brand_physical",
        )

    def test_observable_causal_magnet_passes_mechanism_check(self) -> None:
        plan = _tutor_plan_dict(
            physicalGenerator="Magnet attracting one metal sphere",
            transferredObject="Magnet attracting one metal sphere",
            executionScene="Magnet pulls one metal sphere on a table",
            immediateClarityReason="Viewer sees the magnet pull one sphere while others stay still",
            relativeAdvantageConnection="One pulled object proves dedicated attention to one student",
        )
        self.assertFalse(detect_mechanism_not_observable(plan_dict=plan, ad=plan["ads"][0]))


class TestIdeaMemoryPhysicalFamilyNovelty(unittest.TestCase):
    def _record(self, **overrides: Any) -> IdeaMemoryRecord:
        base = {
            "schema_version": 1,
            "record_id": "r1",
            "campaign_id": "c-prior",
            "ad_index": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "strategic_problem": "sp",
            "relative_advantage": "ra",
            "slogan": "s",
            "conceptual_generator": "cg",
            "physical_generator": "Open umbrella on industrial testing conveyor",
            "transferred_object": "Open umbrella on industrial testing conveyor",
            "transferred_object_action": "Umbrella tested under rain",
            "graphic_summary": "g",
            "conceptual_execution": "ce",
            "physical_execution": "pe",
            "visual_execution": "ve",
            "scene_description": "sd",
            "execution_subject": "es",
            "execution_action": "ea",
            "execution_object_state": "eos",
            "execution_scene": "esc",
            "execution_punchline": "ep",
            "campaign_idea_fingerprint": "fp",
            "ad_execution_fingerprint": "afp",
        }
        base.update(overrides)
        return IdeaMemoryRecord(**base)

    def test_classify_conveyor_family(self) -> None:
        self.assertEqual(
            classify_physical_generator_family("Quality control conveyor inspection station"),
            "conveyor_production",
        )

    def test_repeated_conveyor_family_gets_novelty_penalty(self) -> None:
        snapshot = IdeaMemorySnapshot(
            scope=IdeaMemoryScope(),
            records=[
                self._record(record_id="r1"),
                self._record(record_id="r2", ad_index=2),
            ],
        )
        block = build_physical_family_novelty_block(snapshot)
        self.assertIn("conveyor_production", block)
        self.assertIn("appeared 2 times", block)

    def test_stage_memory_injects_family_novelty_for_brand_physical(self) -> None:
        snapshot = IdeaMemorySnapshot(
            scope=IdeaMemoryScope(),
            records=[self._record(record_id="r1"), self._record(record_id="r2", ad_index=2)],
        )
        block = build_stage_memory_block("brand_physical", snapshot)
        self.assertIn("conveyor_production", block)

    def test_conveyor_still_allowed_without_hard_ban(self) -> None:
        snapshot = IdeaMemorySnapshot(scope=IdeaMemoryScope(), records=[self._record()])
        block = build_physical_family_novelty_block(snapshot)
        self.assertIn("NOT banned", block)

    def test_conceptual_stage_excludes_physical_family_novelty(self) -> None:
        snapshot = IdeaMemorySnapshot(
            scope=IdeaMemoryScope(),
            records=[self._record(record_id="r1"), self._record(record_id="r2", ad_index=2)],
        )
        block = build_stage_memory_block("conceptual_stage", snapshot)
        self.assertIn("conceptualGenerator", block)
        self.assertNotIn("conveyor_production", block)
        self.assertNotIn("physical-generator families", block)

    def test_series_ads_receives_physical_family_novelty(self) -> None:
        snapshot = IdeaMemorySnapshot(
            scope=IdeaMemoryScope(),
            records=[self._record(record_id="r1"), self._record(record_id="r2", ad_index=2)],
        )
        block = build_stage_memory_block("series_ads", snapshot)
        self.assertIn("conveyor_production", block)
        self.assertIn("executionPunchline", block)


class TestMethodologyPurityAndCallCount(unittest.TestCase):
    def test_supplied_product_name_planning_call_count_unchanged(self) -> None:
        from engine.builder1_planning_metrics import (
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
            NORMAL_PLANNING_CALLS_WITH_NAME,
            get_planning_metrics,
        )
        from engine.builder1_planner import plan_builder1
        from tests.test_builder1_staged_planning import _full_final_responses

        captured: dict[str, object] = {}
        real_reset = __import__(
            "engine.builder1_planning_metrics", fromlist=["reset_planning_metrics"]
        ).reset_planning_metrics

        def capture_reset(token) -> None:
            metrics = get_planning_metrics()
            if metrics is not None:
                captured["metrics"] = metrics
            real_reset(token)

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        with patch("engine.builder1_planner.reset_planning_metrics", side_effect=capture_reset):
            plan_builder1(
                product_name="CarryShell",
                product_description="Reinforced shell product for daily carry",
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        metrics = captured.get("metrics")
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(metrics.product_name_call_used)
        self.assertEqual(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_NAME)
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)

    def test_generated_name_adds_only_product_name_resolution_call(self) -> None:
        from engine.builder1_planning_metrics import (
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
            get_planning_metrics,
        )
        from engine.builder1_planner import plan_builder1
        from tests.test_builder1_staged_planning import _full_final_responses

        captured: dict[str, object] = {}
        real_reset = __import__(
            "engine.builder1_planning_metrics", fromlist=["reset_planning_metrics"]
        ).reset_planning_metrics

        def capture_reset(token) -> None:
            metrics = get_planning_metrics()
            if metrics is not None:
                captured["metrics"] = metrics
            real_reset(token)

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        with patch("engine.builder1_planner.reset_planning_metrics", side_effect=capture_reset):
            plan_builder1(
                product_name="",
                product_description="Reinforced shell product for daily carry",
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        metrics = captured.get("metrics")
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertTrue(metrics.product_name_call_used)
        self.assertEqual(metrics.product_name_stage_calls, 1)
        self.assertEqual(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)

    def test_no_new_mandatory_planning_stage_constant(self) -> None:
        from engine.builder1_planning_metrics import (
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
            NORMAL_PLANNING_CALLS_WITH_NAME,
        )

        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME - NORMAL_PLANNING_CALLS_WITH_NAME, 1)

    def test_conceptual_generator_memory_remains_conceptual_only(self) -> None:
        snapshot = IdeaMemorySnapshot(
            scope=IdeaMemoryScope(),
            records=[
                IdeaMemoryRecord(
                    schema_version=1,
                    record_id="r1",
                    campaign_id="c1",
                    ad_index=1,
                    created_at="2026-01-01T00:00:00Z",
                    strategic_problem="sp",
                    relative_advantage="ra",
                    slogan="s",
                    conceptual_generator="Testing readiness through controlled conditions",
                    physical_generator="Open umbrella on industrial testing conveyor",
                    transferred_object="Open umbrella on industrial testing conveyor",
                    transferred_object_action="Umbrella tested under rain",
                    graphic_summary="g",
                    conceptual_execution="ce",
                    physical_execution="pe",
                    visual_execution="ve",
                    scene_description="sd",
                    execution_subject="es",
                    execution_action="ea",
                    execution_object_state="eos",
                    execution_scene="esc",
                    execution_punchline="ep",
                    campaign_idea_fingerprint="fp",
                    ad_execution_fingerprint="afp",
                )
            ],
        )
        block = build_stage_memory_block("conceptual_stage", snapshot)
        self.assertIn("Testing readiness", block)
        self.assertNotIn("physical-generator families", block)
        self.assertNotIn("conveyor_production", block)


if __name__ == "__main__":
    unittest.main()
