"""
Builder2 single-slogan copy contract and metaphorical embodiment tests — mocks only.
"""
from __future__ import annotations

import logging
import os
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from engine.builder2_metaphorical_embodiment_contract import (
    CREATOR_METAPHOR_FIELDS,
    DEPRECATED_EXAMPLE_ONLY_CREATOR_FIELDS,
    JUDGE_METAPHOR_FIELDS,
    PROTOTYPE_EMBODIMENT_MODE_HINTS,
    apply_metaphorical_eligibility_rules,
    candidate_literal_execution_detected,
    validate_creator_metaphorical_embodiment,
    validate_judge_metaphorical_embodiment,
)
from engine.builder2_resume_service import build_builder2_status_payload
from engine.builder2_single_slogan_contract import (
    BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
    apply_single_slogan_winner_normalization,
    builder2_requires_headline_overlay,
    is_single_slogan_contract,
    normalize_legacy_dual_copy,
    resolve_canonical_slogan_text,
    validate_single_slogan_completion,
    validate_single_slogan_plan_contract,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.builder2_methodology_fixtures import (
    complete_ad_creator_extras,
    creative_embodiment_context_collision_extras,
    creative_embodiment_media_replacement_extras,
    creative_embodiment_think_small_extras,
    metaphorical_embodiment_absence_extras,
    metaphorical_embodiment_creator_extras,
    metaphorical_embodiment_judge_extras,
    metaphorical_embodiment_shortening_extras,
    methodology_strategy_extras,
    single_slogan_contract_extras,
)
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL, verified_final_publication_media_fields
from engine.builder2_media_resume import run_one_media_resume
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps


class TestSingleSloganContract(unittest.TestCase):
    def test_new_tournament_state_has_copy_contract(self) -> None:
        from engine.builder2_tournament_store import new_tournament_state

        state = new_tournament_state(
            job_id="job-copy-contract",
            language="he",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        self.assertEqual(state["copyContractVersion"], BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION)
        self.assertTrue(is_single_slogan_contract(state=state))

    def test_winner_normalization_forces_single_slogan(self) -> None:
        plan: dict[str, Any] = {
            "copyContractVersion": BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
            "builder2NewFormatVersion": "builder2_complete_ad_v1",
            "headlineDecision": {"decision": "use", "reason": "old", "reasonSource": "model"},
            "headlineText": "Different headline",
            "headline": "Different headline",
            "advertisingClosure": complete_ad_creator_extras()["advertisingClosure"],
        }
        candidate = complete_ad_creator_extras()
        candidate.update(metaphorical_embodiment_creator_extras())
        apply_single_slogan_winner_normalization(plan, winning_candidate=candidate)
        self.assertFalse(builder2_requires_headline_overlay(plan=plan))
        self.assertEqual(
            resolve_canonical_slogan_text(plan=plan),
            plan["advertisingClosure"]["sloganText"],
        )
        self.assertTrue(plan.get("headlineCompatibilityAlias"))

    def test_legacy_dual_copy_normalization_prefers_closure(self) -> None:
        plan = {
            "headlineText": "Headline sentence",
            "advertisingClosure": {
                "required": True,
                "productNameText": "Brand",
                "sloganText": "Closure slogan",
                "language": "he",
                "presentationMode": "end_card",
                "durationSeconds": 2.0,
                "noLogo": True,
            },
        }
        legacy = normalize_legacy_dual_copy(plan=plan)
        self.assertEqual(legacy["canonicalSloganText"], "Closure slogan")
        self.assertTrue(legacy["legacyCopyNormalized"])

    def test_two_distinct_messages_fail_validation(self) -> None:
        plan = {
            "copyContractVersion": BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
            "builder2NewFormatVersion": "builder2_complete_ad_v1",
            "headlineDecision": {"decision": "use", "reason": "x", "reasonSource": "model"},
            "headline": "First message",
            "headlineText": "First message",
            "advertisingClosure": complete_ad_creator_extras(slogan_text="Second message")["advertisingClosure"],
            "sloganUnderstandsWithoutPriorCopy": True,
        }
        ok, failures = validate_single_slogan_plan_contract(plan)
        self.assertFalse(ok)
        self.assertTrue(failures)

    def test_depends_on_earlier_copy_rejected(self) -> None:
        plan = single_slogan_contract_extras()
        plan.update(complete_ad_creator_extras())
        plan["dependsOnEarlierCopy"] = True
        ok, failures = validate_single_slogan_plan_contract(plan)
        self.assertFalse(ok)
        self.assertIn("slogan_depends_on_earlier_copy", failures)


class TestMetaphoricalEmbodiment(unittest.TestCase):
    def test_creator_graph_without_transformation_invalid(self) -> None:
        candidate = {
            "coreVisualIdea": "A growth graph shows lead quantity",
            "visualMechanism": "Dashboard metrics rise on screen",
            **metaphorical_embodiment_creator_extras(),
        }
        candidate["metaphoricalEmbodiment"] = {
            **metaphorical_embodiment_creator_extras()["metaphoricalEmbodiment"],
            "literalSymbolsRejectedOrTransformed": "Uses the graph directly",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_creator_metaphorical_closeness_valid(self) -> None:
        candidate = {
            "coreVisualIdea": "Two people close the distance between them",
            "visualMechanism": "Physical nearness expresses strategic closeness",
            **metaphorical_embodiment_creator_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_shortening_metaphor_without_quantity_or_quality_fields(self) -> None:
        candidate = {
            "coreVisualIdea": "A ribbon is shortened beside a longer segment",
            "visualMechanism": "Material shortening expresses faster delivery",
            **metaphorical_embodiment_shortening_extras(),
        }
        metaphor = candidate["metaphoricalEmbodiment"]
        self.assertNotIn("quantityEmbodiment", metaphor)
        self.assertNotIn("qualityOrAdvantageEmbodiment", metaphor)
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_absence_metaphor_without_repeated_superior_objects(self) -> None:
        candidate = {
            "coreVisualIdea": "Activity continues in darkness",
            "visualMechanism": "Missing light reveals the forgotten step",
            **metaphorical_embodiment_absence_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="forgot")

    def test_deprecated_quantity_quality_fields_rejected(self) -> None:
        candidate = {
            **metaphorical_embodiment_creator_extras(),
        }
        candidate["metaphoricalEmbodiment"] = {
            **metaphorical_embodiment_creator_extras()["metaphoricalEmbodiment"],
            "quantityEmbodiment": "Several ordinary pens repeat the same role",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_judge_rejects_literal_execution(self) -> None:
        judgment = {
            "eligible": True,
            "disqualifiers": [],
            **metaphorical_embodiment_judge_extras(),
        }
        judgment["metaphoricalEmbodimentAssessment"]["literalExecutionDetected"] = True
        judgment["metaphoricalEmbodimentAssessment"]["literalPresentationMeaningfullyTransformed"] = False
        judgment["metaphoricalEmbodimentAssessment"]["creativeEmbodimentAccepted"] = False
        updated = apply_metaphorical_eligibility_rules(judgment)
        self.assertFalse(updated["eligible"])

    def test_slogan_cannot_rescue_literal_visual(self) -> None:
        candidate = {
            "coreVisualIdea": "CRM dashboard with lead counters",
            "openingFrameDescription": "Numbers fill the screen",
        }
        self.assertTrue(candidate_literal_execution_detected(candidate))

    def test_judge_evaluates_physical_embodiment_not_domain_departure(self) -> None:
        judgment = {
            **metaphorical_embodiment_judge_extras(),
        }
        judgment["metaphoricalEmbodimentAssessment"]["physicalEmbodimentMatchesStrategicRelationship"] = False
        updated = apply_metaphorical_eligibility_rules({**judgment, "eligible": True, "disqualifiers": []})
        self.assertFalse(updated["eligible"])
        self.assertIn("physical_embodiment_mismatch_rejected", updated["disqualifiers"])

    def test_judge_accepts_visible_product_without_external_world(self) -> None:
        judgment = {
            **metaphorical_embodiment_judge_extras(),
        }
        updated = apply_metaphorical_eligibility_rules({**judgment, "eligible": True, "disqualifiers": []})
        self.assertTrue(updated["eligible"])

    def test_judge_visual_bridge_validation(self) -> None:
        judgment = {
            **metaphorical_embodiment_judge_extras(),
        }
        validate_judge_metaphorical_embodiment(judgment)


class TestFreshProductionSingleSloganIntegration(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.capability_patch, self.closure_patch, self.publish_patch, self.publish_mock = (
            patch_media_pipeline_durable_finalization(CLOSURE_URL)
        )
        self.capability_patch.start()
        self.closure_patch.start()
        self.publish_patch.start()

    def tearDown(self) -> None:
        self.publish_patch.stop()
        self.closure_patch.stop()
        self.capability_patch.stop()
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret",
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "ACE_PUBLIC_BASE_URL": "https://ace.example.com",
        },
        clear=True,
    )
    @patch("engine.builder2_media_resume.save_tournament_state")
    def test_fresh_production_skips_headline_and_renders_one_slogan(self, _save: Any) -> None:
        state = _media_ready_state(job_id="job-single-slogan-fresh")
        state.update(single_slogan_contract_extras())
        state["builder2NewFormatVersion"] = "builder2_complete_ad_v1"
        plan = state["winnerDevelopmentPlan"]
        plan.update(single_slogan_contract_extras())
        plan["advertisingClosure"] = complete_ad_creator_extras()["advertisingClosure"]
        apply_single_slogan_winner_normalization(plan, winning_candidate=complete_ad_creator_extras())

        postprocess_calls: list[str] = []
        deps = _mock_pipeline_deps()
        deps.postprocess_video = lambda **kwargs: postprocess_calls.append("headline") or kwargs["runway_url"]

        report = run_one_media_resume(
            job_id="job-single-slogan-fresh",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=deps,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(postprocess_calls, [])
        self.assertTrue(report["finalVideoAvailable"])
        saved = _save.call_args[0][1]
        media = saved["mediaResume"]
        self.assertTrue(media.get("headlineOverlaySkipped"))
        self.assertTrue(media.get("sloganRenderedExactlyOnce"))
        self.assertIn("/api/builder2-final-video/", media["finalPublicUrl"])

    @patch("engine.builder2_execution_lease.has_active_lease", return_value=False)
    @patch("engine.builder2_resume_service.is_job_queued", return_value=False)
    @patch("engine.builder2_resume_service.has_active_lease", return_value=False)
    def test_completed_status_exposes_video_url(self, _a: Any, _b: Any, _c: Any) -> None:
        state = _media_ready_state(job_id="job-single-slogan-status")
        state.update(single_slogan_contract_extras())
        media = state.setdefault("mediaResume", {})
        media.update(
            {
                "finalPublicUrl": CLOSURE_URL,
                "finalVideoWithClosureUrl": CLOSURE_URL,
                **verified_final_publication_media_fields(),
                "headlineOverlaySkipped": True,
                "sloganRenderedExactlyOnce": True,
            }
        )
        payload = build_builder2_status_payload(
            "job-single-slogan-status",
            {"status": "done", "video_url": CLOSURE_URL, "builder": "builder2"},
            tournament_state=state,
        )
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["videoUrl"], CLOSURE_URL)

    def test_completion_contract_rejects_headline_overlay(self) -> None:
        state = _media_ready_state(job_id="job-headline-overlay-fail")
        plan = state["winnerDevelopmentPlan"]
        plan.update(single_slogan_contract_extras())
        media = state.setdefault("mediaResume", {})
        media.update(
            {
                "headlinePostprocessStatus": "completed",
                "headlineArtifactUrl": "https://ace.example.com/api/video-headline/token",
                **verified_final_publication_media_fields(),
            }
        )
        failures = validate_single_slogan_completion(state=state, plan=plan, media=media)
        self.assertIn("headline_overlay_rendered_under_single_slogan_contract", failures)


class TestSafeLogging(unittest.TestCase):
    def test_single_slogan_metadata_does_not_log_copy_text(self) -> None:
        plan = single_slogan_contract_extras(slogan_text="Secret advertising sentence")
        plan["advertisingClosure"] = complete_ad_creator_extras(slogan_text="Secret advertising sentence")["advertisingClosure"]
        with self.assertLogs("engine.builder2_single_slogan_contract", level="INFO") as logs:
            from engine.builder2_single_slogan_contract import log_single_slogan_safe_metadata

            log_single_slogan_safe_metadata(plan=plan, job_id="job-log-safe")
        joined = "\n".join(logs.output)
        self.assertNotIn("Secret advertising sentence", joined)
        self.assertIn("BUILDER2_SINGLE_SLOGAN_METADATA", joined)


class TestCreativeEmbodimentContract(unittest.TestCase):
    def test_external_metaphor_may_pass(self) -> None:
        candidate = {
            "coreVisualIdea": "A ribbon is shortened beside a longer segment",
            "visualMechanism": "Material shortening expresses faster delivery",
            **metaphorical_embodiment_shortening_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_think_small_passes_with_advertised_product_itself(self) -> None:
        candidate = {
            "coreVisualIdea": "A compact car sits tiny in a vast white field",
            "visualMechanism": "Scale and composition invert smallness into advantage",
            "visualParallelType": "side_by_side",
            **_prototype_application_bundle("think_small"),
            **creative_embodiment_think_small_extras(),
        }
        metaphor = candidate["metaphoricalEmbodiment"]
        self.assertNotIn("metaphoricalWorld", metaphor)
        self.assertNotIn("metaphoricalPhysicalFamily", metaphor)
        self.assertEqual(metaphor["creativeEmbodimentMode"], "transformed_product")
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="think_small")

    def test_think_small_rejects_size_diagram(self) -> None:
        candidate = {
            "coreVisualIdea": "A size comparison diagram shows the car is smaller",
            "visualMechanism": "Measurement labels prove compactness",
            **creative_embodiment_think_small_extras(),
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="think_small")

    def test_think_small_rejects_ordinary_untransformed_product_photo(self) -> None:
        candidate = {
            "coreVisualIdea": "An ordinary product photo of the compact car on a street",
            "visualMechanism": "Standard catalog presentation",
            **creative_embodiment_think_small_extras(),
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="think_small")

    def test_media_replacement_passes_without_external_object_family(self) -> None:
        candidate = {
            "coreVisualIdea": "The storefront window display becomes the proof",
            "visualMechanism": "The medium itself carries the argument",
            "visualParallelType": "medium_as_object",
            **_prototype_application_bundle("winning_card"),
            **creative_embodiment_media_replacement_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="winning_card")

    def test_context_transformation_passes_with_visible_product(self) -> None:
        candidate = {
            "coreVisualIdea": "The product remains visible inside an extreme colliding environment",
            "visualMechanism": "Context collision reveals the product role",
            "visualParallelType": "context_collision",
            **creative_embodiment_context_collision_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_deprecated_metaphor_world_fields_rejected(self) -> None:
        candidate = {
            **metaphorical_embodiment_creator_extras(),
        }
        candidate["metaphoricalEmbodiment"] = {
            **metaphorical_embodiment_creator_extras()["metaphoricalEmbodiment"],
            "metaphoricalWorld": "A separate fantasy world",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_every_active_gold_prototype_can_satisfy_contract(self) -> None:
        from engine.builder2_prototypes import active_prototype_ids

        extras_by_prototype = {
            "winning_card": creative_embodiment_media_replacement_extras(),
            "summer_fan": metaphorical_embodiment_creator_extras(
                strategic_perception="Motion creates a missing association.",
                creative_embodiment_mode="transformed_action_or_motion",
            ),
            "greenpeace_essential_pairing": metaphorical_embodiment_creator_extras(
                strategic_perception="Two essences belong together emotionally.",
                creative_embodiment_mode="essential_pairing",
            ),
            "forgot": metaphorical_embodiment_absence_extras(),
            "closest": metaphorical_embodiment_creator_extras(),
            "think_small": creative_embodiment_think_small_extras(),
        }
        for prototype_id in sorted(active_prototype_ids()):
            candidate = {
                "coreVisualIdea": f"Prototype {prototype_id} embodiment",
                "visualMechanism": "Visible physical proof",
                **extras_by_prototype[prototype_id],
            }
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id=prototype_id)
            self.assertIn(prototype_id, PROTOTYPE_EMBODIMENT_MODE_HINTS)


class TestGenericMetaphorContract(unittest.TestCase):
    def test_quantity_and_quality_not_mandatory_creator_fields(self) -> None:
        self.assertNotIn("quantityEmbodiment", CREATOR_METAPHOR_FIELDS)
        self.assertNotIn("qualityOrAdvantageEmbodiment", CREATOR_METAPHOR_FIELDS)
        self.assertNotIn("qualityEmbodiment", CREATOR_METAPHOR_FIELDS)
        self.assertIn("physicalEmbodiment", CREATOR_METAPHOR_FIELDS)
        self.assertIn("creativeEmbodimentMode", CREATOR_METAPHOR_FIELDS)
        self.assertIn("transformationMechanism", CREATOR_METAPHOR_FIELDS)
        self.assertNotIn("metaphoricalWorld", CREATOR_METAPHOR_FIELDS)
        self.assertNotIn("metaphoricalPhysicalFamily", CREATOR_METAPHOR_FIELDS)

    def test_quantity_and_quality_not_mandatory_judge_fields(self) -> None:
        self.assertNotIn("sameOrParallelFamilyAccepted", JUDGE_METAPHOR_FIELDS)
        self.assertNotIn("metaphoricalWorldDistinctFromBusinessDomain", JUDGE_METAPHOR_FIELDS)
        self.assertIn("creativeEmbodimentAccepted", JUDGE_METAPHOR_FIELDS)
        self.assertIn("physicalEmbodimentMatchesStrategicRelationship", JUDGE_METAPHOR_FIELDS)

    def test_deprecated_fields_listed_as_example_only(self) -> None:
        self.assertIn("quantityEmbodiment", DEPRECATED_EXAMPLE_ONLY_CREATOR_FIELDS)
        self.assertIn("sameOrParallelObjectFamily", DEPRECATED_EXAMPLE_ONLY_CREATOR_FIELDS)

    def test_example_objects_not_stored_as_preferred_solutions(self) -> None:
        import json

        from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
        from tests.builder2_methodology_fixtures import metaphorical_embodiment_creator_extras

        example_markers = (
            "fountain pen",
            "matchbox",
            "zippo",
            "v formation",
            "short-neck giraffe",
            "quantity versus quality",
        )
        fixture_blob = json.dumps(metaphorical_embodiment_creator_extras()).lower()
        prompt = build_creator_required_keys_prompt_text(prototype_id="closest").lower()
        for marker in example_markers:
            self.assertNotIn(marker, fixture_blob, msg=f"default fixture must not encode {marker!r}")
            self.assertNotIn(marker, prompt, msg=f"Creator schema prompt must not prefer {marker!r}")

    def test_prompts_do_not_require_example_objects(self) -> None:
        from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
        from engine.builder2_prototypes import require_prototype

        prompt = build_creator_required_keys_prompt_text(prototype_id="closest")
        self.assertNotIn("quantityEmbodiment", prompt.split("Do not use")[0])
        self.assertIn("sloganBridgeToBusinessMeaning", prompt)
        self.assertIn("Do not use quantityEmbodiment", prompt)

    def test_graph_literal_execution_still_rejectable(self) -> None:
        candidate = {
            "coreVisualIdea": "A growth graph shows lead quantity",
            "visualMechanism": "Dashboard metrics rise on screen",
            **metaphorical_embodiment_creator_extras(),
        }
        candidate["metaphoricalEmbodiment"] = {
            **metaphorical_embodiment_creator_extras()["metaphoricalEmbodiment"],
            "literalSymbolsRejectedOrTransformed": "Uses the graph directly",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_single_slogan_contract_unchanged(self) -> None:
        self.assertEqual(BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION, "builder2_single_slogan_v1")

    def test_creator_prompt_requires_metaphorical_embodiment(self) -> None:
        from engine.builder2_prototypes import require_prototype
        from engine.builder2_tournament_prompts import build_creator_prompt

        prompt = build_creator_prompt(
            product_name="Brand",
            product_description="desc",
            language="he",
            strategy_foundation=methodology_strategy_extras(),
            prototype=require_prototype("closest"),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="standard",
        )
        self.assertIn("metaphoricalEmbodiment", prompt)
        self.assertIn("literalSymbolsRejectedOrTransformed", prompt)
        self.assertIn("dependsOnEarlierCopy=false", prompt)

    def test_judge_prompt_requires_metaphor_assessment(self) -> None:
        from engine.builder2_prototypes import require_prototype
        from engine.builder2_tournament_prompts import build_judge_prompt

        prompt = build_judge_prompt(
            product_name="Brand",
            product_description="desc",
            language="he",
            strategy_foundation=methodology_strategy_extras(),
            prototype=require_prototype("closest"),
            candidate={"prototypeId": "closest", "verbalPotential": {"decision": "available"}},
            candidate_id="cand-1",
        )
        self.assertIn("creativeEmbodimentAccepted", prompt)
        self.assertIn("physicalEmbodimentMatchesStrategicRelationship", prompt)
        self.assertNotIn("sameOrParallelFamilyAccepted", prompt)
        self.assertNotIn("metaphoricalWorldDistinctFromBusinessDomain", prompt)

    def test_creator_schema_does_not_require_external_world(self) -> None:
        candidate = {
            "coreVisualIdea": "The actual product remains visible while composition transforms meaning",
            "visualMechanism": "Scale inversion expresses the strategic perception",
            **creative_embodiment_think_small_extras(),
        }
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="think_small")


def _prototype_application_bundle(prototype_id: str) -> dict[str, Any]:
    from tests.builder2_methodology_fixtures import _prototype_application

    return _prototype_application(prototype_id)


class TestBuilder1Isolation(unittest.TestCase):
    def test_builder1_modules_untouched(self) -> None:
        import engine.builder1_planner  # noqa: F401

        self.assertTrue(hasattr(__import__("engine.builder1_planner"), "builder1_planner"))


if __name__ == "__main__":
    unittest.main()
