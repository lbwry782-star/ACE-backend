"""
Builder2 no-logo policy contract tests — mocks only.
"""
from __future__ import annotations

import inspect
import os
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from engine.builder2_no_logo_contract import (
    BUILDER2_NO_LOGO_POLICY_VERSION,
    apply_logo_eligibility_rules,
    build_builder2_no_logo_visual_policy_block,
    contains_third_party_brand_reference,
    judgment_rejects_logo_policy,
    normalize_builder2_media_prompt_text,
    validate_creator_logo_policy,
    validate_judge_logo_policy,
    validate_no_logo_completion,
)
from engine.builder2_resume_service import build_builder2_status_payload
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, new_tournament_state
from engine.builder2_winner_downstream import (
    build_builder2_start_frame_image_prompt,
    build_continuous_event_runway_prompt,
)
from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.builder2_methodology_fixtures import (
    complete_ad_creator_extras,
    logo_policy_creator_extras,
    logo_policy_judge_extras,
    metaphorical_embodiment_creator_extras,
    metaphorical_embodiment_judge_extras,
    methodology_strategy_extras,
    single_slogan_contract_extras,
)
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL, verified_final_publication_media_fields
from engine.builder2_media_resume import run_one_media_resume
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps


class TestNoLogoPolicyContract(unittest.TestCase):
    def test_new_tournament_state_has_logo_policy(self) -> None:
        state = new_tournament_state(
            job_id="job-logo-policy",
            language="he",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        self.assertEqual(state["logoPolicyVersion"], BUILDER2_NO_LOGO_POLICY_VERSION)
        self.assertFalse(state["logosAllowed"])
        self.assertTrue(state["plainTextAdvertisedNameOnly"])

    def test_creator_advertised_logo_invalid(self) -> None:
        candidate = {
            "coreVisualIdea": "Generic unbranded scene",
            **metaphorical_embodiment_creator_extras(),
        }
        candidate["logoPolicyReport"] = {
            **logo_policy_creator_extras()["logoPolicyReport"],
            "advertisedLogoRequested": True,
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_logo_policy(candidate, assigned_prototype_id="closest", product_name="ACE Product")

    def test_creator_third_party_logo_reference_invalid(self) -> None:
        candidate = {
            "coreVisualIdea": "A Nike shirt proves athletic motion",
            **metaphorical_embodiment_creator_extras(),
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_logo_policy(candidate, assigned_prototype_id="closest", product_name="ACE Product")

    def test_creator_invented_logo_invalid(self) -> None:
        candidate = {
            "coreVisualIdea": "A custom logo emblem appears on the package",
            **metaphorical_embodiment_creator_extras(),
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_logo_policy(candidate, assigned_prototype_id="closest", product_name="ACE Product")

    def test_creator_generic_unbranded_scene_valid(self) -> None:
        candidate = {
            "coreVisualIdea": "Generic unbranded objects in a neutral room",
            **metaphorical_embodiment_creator_extras(),
        }
        validate_creator_logo_policy(candidate, assigned_prototype_id="closest", product_name="ACE Product")

    def test_judge_cannot_approve_logo_dependent_candidate(self) -> None:
        judgment = {
            **metaphorical_embodiment_judge_extras(),
            "eligible": True,
            "disqualifiers": [],
        }
        judgment["logoPolicyAssessment"]["logoDependentMeaning"] = True
        updated = apply_logo_eligibility_rules(judgment)
        self.assertFalse(updated["eligible"])
        self.assertIn("logo_dependent_meaning", updated["disqualifiers"])

    def test_manager_blocks_logo_policy_failure(self) -> None:
        judgment = {**metaphorical_embodiment_judge_extras()}
        judgment["logoPolicyAssessment"]["logoFreeExecutionAccepted"] = False
        self.assertTrue(judgment_rejects_logo_policy(judgment))

    def test_prompt_block_present_in_runway_and_start_image(self) -> None:
        plan = {
            "structureType": "continuous_event",
            "coreVisualIdea": "Two people close distance in a neutral room",
            "videoPrompt": "Continuous human gesture in one room",
            "openingFrameDescription": "Two people with space between them",
            "sequence": {"beginning": "One step forward", "development": "Distance closes", "resolution": "They meet"},
            "runwayFeasibility": {
                "mainSubject": "Two people",
                "mainAction": "One person steps forward",
                "location": "Neutral room",
                "openingFrame": "Visible distance between two people",
            },
            "visualAnchor": {"description": "The closing distance"},
        }
        block = build_builder2_no_logo_visual_policy_block(compact=True)
        runway = build_continuous_event_runway_prompt(plan, duration_seconds=7)
        start = build_builder2_start_frame_image_prompt(plan, duration_seconds=7)
        self.assertIn("NO-LOGO:", runway)
        self.assertIn("NO-LOGO:", start)
        self.assertNotIn("Nike", runway)
        self.assertNotIn("Zippo", start)

    def test_prompt_normalization_replaces_commercial_references(self) -> None:
        normalized = normalize_builder2_media_prompt_text("A Zippo lighter beside an iPhone on a Nike shirt")
        self.assertNotIn("Zippo", normalized)
        self.assertNotIn("iPhone", normalized)
        self.assertNotIn("Nike", normalized)
        self.assertTrue(contains_third_party_brand_reference("A Nike shirt"))

    def test_closure_renderer_has_no_logo_asset_parameters(self) -> None:
        from engine.builder2_closure_render import render_builder2_advertising_closure_endcard

        signature = inspect.signature(render_builder2_advertising_closure_endcard)
        self.assertNotIn("logo", " ".join(signature.parameters))

    def test_completion_rejects_logo_asset_used(self) -> None:
        failures = validate_no_logo_completion(
            state=single_slogan_contract_extras(),
            plan=single_slogan_contract_extras(),
            media={"logoAssetUsed": True, "brandNameRenderedAsPlainText": True, "logoPolicyVersion": BUILDER2_NO_LOGO_POLICY_VERSION},
        )
        self.assertIn("logo_asset_used_under_no_logo_policy", failures)

    def test_completion_rejects_brand_graphic_rendered(self) -> None:
        failures = validate_no_logo_completion(
            state=single_slogan_contract_extras(),
            plan=single_slogan_contract_extras(),
            media={
                "brandGraphicRendered": True,
                "brandNameRenderedAsPlainText": True,
                "logoPolicyVersion": BUILDER2_NO_LOGO_POLICY_VERSION,
            },
        )
        self.assertIn("brand_graphic_rendered_under_no_logo_policy", failures)

    def test_judge_logo_assessment_schema(self) -> None:
        judgment = {**metaphorical_embodiment_judge_extras()}
        validate_judge_logo_policy(judgment)

    def test_single_slogan_contract_unchanged(self) -> None:
        from engine.builder2_single_slogan_contract import BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION

        self.assertEqual(BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION, "builder2_single_slogan_v1")


class TestFreshProductionNoLogoIntegration(unittest.TestCase):
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
    def test_fresh_production_logo_free_closure_and_publication(self, _save: Any) -> None:
        state = _media_ready_state(job_id="job-no-logo-fresh")
        state.update(single_slogan_contract_extras())
        state["builder2NewFormatVersion"] = "builder2_complete_ad_v1"
        plan = state["winnerDevelopmentPlan"]
        plan.update(single_slogan_contract_extras())
        plan["advertisingClosure"] = complete_ad_creator_extras()["advertisingClosure"]

        report = run_one_media_resume(
            job_id="job-no-logo-fresh",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(report["ok"])
        saved = _save.call_args[0][1]
        media = saved["mediaResume"]
        self.assertTrue(media.get("brandNameRenderedAsPlainText"))
        self.assertFalse(media.get("brandGraphicRendered"))
        self.assertFalse(media.get("logoAssetUsed"))
        self.assertTrue(media.get("logoPolicySatisfied"))
        self.assertTrue(media.get("headlineOverlaySkipped"))
        self.assertTrue(media.get("sloganRenderedExactlyOnce"))
        self.assertIn("/api/builder2-final-video/", media["finalPublicUrl"])

    @patch("engine.builder2_execution_lease.has_active_lease", return_value=False)
    @patch("engine.builder2_resume_service.is_job_queued", return_value=False)
    @patch("engine.builder2_resume_service.has_active_lease", return_value=False)
    def test_status_payload_done_with_durable_url(self, _a: Any, _b: Any, _c: Any) -> None:
        state = _media_ready_state(job_id="job-no-logo-status")
        state.update(single_slogan_contract_extras())
        media = state.setdefault("mediaResume", {})
        media.update(
            {
                "finalPublicUrl": CLOSURE_URL,
                "finalVideoWithClosureUrl": CLOSURE_URL,
                **verified_final_publication_media_fields(),
                "headlineOverlaySkipped": True,
                "sloganRenderedExactlyOnce": True,
                "brandNameRenderedAsPlainText": True,
                "brandGraphicRendered": False,
                "logoAssetUsed": False,
                "logoPolicySatisfied": True,
                "logoPolicyVersion": BUILDER2_NO_LOGO_POLICY_VERSION,
            }
        )
        payload = build_builder2_status_payload(
            "job-no-logo-status",
            {"status": "done", "video_url": CLOSURE_URL, "builder": "builder2"},
            tournament_state=state,
        )
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["videoUrl"], CLOSURE_URL)


class TestBuilder1Isolation(unittest.TestCase):
    def test_builder1_untouched(self) -> None:
        import engine.builder1_planner  # noqa: F401

        self.assertTrue(hasattr(__import__("engine.builder1_planner"), "builder1_planner"))


if __name__ == "__main__":
    unittest.main()
