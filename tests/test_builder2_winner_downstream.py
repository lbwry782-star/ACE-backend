"""
Builder2 winner downstream adapter tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_tournament_contracts import Builder2TournamentError, WINNER_PLAN_SCHEMA_VERSION
from engine.builder2_winner_downstream import (
    Builder2WinnerDownstreamError,
    build_builder2_start_frame_image_prompt,
    build_continuous_event_runway_prompt,
    build_variation_montage_runway_prompt,
    compose_builder2_headline_text,
    get_visual_anchor_description,
    normalize_builder2_winner_downstream,
    validate_builder2_pre_runway,
)
from engine.builder2_winner_plan import validate_and_normalize_builder2_winner_plan
from engine.builder2_winner_development import normalize_winner_plan_for_runway
from engine.runway_video import RunwayVideoMVPError, _generate_one_video_mvp_body
from engine.video_planning import build_runway_prompt_from_plan
from engine.video_start_image import build_ace_start_frame_image_prompt
from tests.test_builder2_tournament import _winner_plan
from tests.test_builder2_tournament_corrections import _winner_plan as _corrections_winner_plan


def _canonical_continuous_plan(*, headline_decision: str = "include", headline: str | None = None) -> Dict[str, Any]:
    plan = _winner_plan(language="he")
    plan["productNameResolved"] = "קופי"
    plan["prototypeId"] = "greenpeace_essential_pairing"
    plan["structureType"] = "continuous_event"
    plan["sceneVariations"] = []
    plan["visualAnchor"] = {
        "description": "The reusable cup appears beside the disposable one.",
        "whyEssential": "It proves the pairing visually.",
        "appearsBeforeOrDuringResolution": True,
    }
    if headline is None:
        headline = "מכירים את הקבוע שלכם." if headline_decision == "include" else ""
    plan["headline"] = headline
    plan["headlineCoreKeyword"] = "קבוע"
    plan["headlineDecision"] = {
        "decision": headline_decision,
        "reason": "Explicit headline decision for downstream tests.",
    }
    plan["headlineForm"] = "none" if headline_decision == "omit" else "direct"
    if headline_decision == "omit":
        plan["headline"] = ""
        plan["headlineCoreKeyword"] = ""
    return plan


class Builder2WinnerDownstreamVisualAnchorTests(unittest.TestCase):
    def test_canonical_object_anchor_works(self) -> None:
        plan = _canonical_continuous_plan()
        desc = get_visual_anchor_description(plan)
        self.assertIn("reusable cup", desc)

    def test_legacy_string_anchor_works(self) -> None:
        plan = _winner_plan()
        desc = get_visual_anchor_description(plan)
        self.assertTrue(desc)

    def test_dict_anchor_never_strip_crash(self) -> None:
        plan = _canonical_continuous_plan()
        normalized = validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_runway_prompt_from_plan(normalized)
        self.assertIn("Visual anchor:", prompt)

    def test_missing_description_fails_with_exact_code(self) -> None:
        plan = _canonical_continuous_plan()
        plan["visualAnchor"] = {"whyEssential": "missing description"}
        with self.assertRaises(Builder2WinnerDownstreamError) as ctx:
            get_visual_anchor_description(plan)
        self.assertEqual(ctx.exception.code, "builder2_winner_downstream_invalid:visualAnchor.description")

    def test_why_essential_not_substituted_for_description(self) -> None:
        plan = _canonical_continuous_plan()
        plan["visualAnchor"] = {"whyEssential": "Only why", "description": ""}
        with self.assertRaises(Builder2WinnerDownstreamError):
            get_visual_anchor_description(plan)

    def test_anchor_object_in_runway_prompt(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_continuous_event_runway_prompt(plan, duration_seconds=7)
        self.assertIn("reusable cup", prompt)

    def test_anchor_object_in_start_image_prompt(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_builder2_start_frame_image_prompt(plan, duration_seconds=7)
        self.assertIn("Opening moment", prompt)
        self.assertNotIn("embrace", prompt.lower())


class Builder2ContinuousEventTests(unittest.TestCase):
    def test_temporal_beats_prompt(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_runway_prompt_from_plan(plan)
        self.assertIn("one continuous", prompt.lower())
        self.assertIn("Beginning:", prompt)
        self.assertIn("Development:", prompt)
        self.assertIn("Resolution:", prompt)
        self.assertNotRegex(prompt.lower(), r"\bmontage of\b")

    def test_structure_remains_continuous_event(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertEqual(plan["structureType"], "continuous_event")

    def test_metadata_remains_builder2_schema(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertEqual(plan["schemaVersion"], WINNER_PLAN_SCHEMA_VERSION)
        self.assertNotEqual(plan.get("schemaVersion"), "variation_montage_v4")

    def test_not_labeled_variation_montage_v4(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertNotIn("variation_montage_v4", str(plan.get("schemaVersion")))

    def test_temporal_beat_semantics_preserved(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertEqual(plan["sceneSequenceSemantics"], "temporal_beats")


class Builder2MontageDownstreamTests(unittest.TestCase):
    def _montage_plan(self) -> Dict[str, Any]:
        plan = _winner_plan()
        plan["structureType"] = "variation_montage"
        plan["sceneVariations"] = [
            {"description": "Two people stand apart.", "familyId": "nearness"},
            {"description": "One step closes the distance.", "familyId": "nearness"},
        ]
        plan["visualFamily"] = {"familyDefinition": "human closeness gestures", "recurringMotif": "closing distance"}
        plan["visualFamilyConsistency"] = {"recurringMotif": "closing distance"}
        return plan

    def test_two_structured_variations_work(self) -> None:
        normalized = validate_and_normalize_builder2_winner_plan(
            self._montage_plan(),
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        self.assertEqual(len(normalized["sceneVariations"]), 2)

    def test_four_variations_work(self) -> None:
        plan = self._montage_plan()
        plan["sceneVariations"] = [
            {"description": f"Beat {i}.", "familyId": "nearness"} for i in range(4)
        ]
        normalized = validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        self.assertEqual(len(normalized["sceneVariations"]), 4)

    def test_invalid_nested_variation_fails_with_index(self) -> None:
        plan = self._montage_plan()
        plan["sceneVariations"][1] = {"familyId": "nearness"}
        with self.assertRaises(Builder2TournamentError):
            validate_and_normalize_builder2_winner_plan(
                plan,
                product_name="ACE Product",
                product_description="desc",
                content_language="en",
            )

    def test_continuous_not_routed_through_montage_logic(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_runway_prompt_from_plan(plan)
        self.assertNotIn("Variation moments:", prompt)


class Builder2HeadlineCompositionTests(unittest.TestCase):
    def test_hebrew_product_name_not_duplicated(self) -> None:
        full, rem = compose_builder2_headline_text("קופי", "קופי. מכירים את הקבוע שלכם.")
        self.assertEqual(full.count("קופי"), 1)
        self.assertEqual(rem, "מכירים את הקבוע שלכם.")

    def test_english_product_name_not_duplicated(self) -> None:
        full, rem = compose_builder2_headline_text("Nike", "Nike. Just do it.")
        self.assertEqual(full.lower().count("nike"), 1)
        self.assertEqual(rem, "Just do it.")

    def test_punctuation_variation_not_duplicated(self) -> None:
        full, rem = compose_builder2_headline_text("אורי לב", "אורי לב הרעיון הראשון.")
        self.assertNotIn("אורי לב אורי לב", full)

    def test_product_excluded_from_remainder_word_count(self) -> None:
        from engine.video_planning import _headline_remainder_word_count

        _, rem = compose_builder2_headline_text("קופי", "קופי. one two three four five six seven")
        self.assertLessEqual(_headline_remainder_word_count(rem), 7)

    def test_include_path_still_overlays(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(headline="מכירים את הקבוע שלכם."),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertTrue(plan["headlineText"].startswith("קופי"))

    def test_omit_path_never_overlays_fields(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(headline_decision="omit"),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertEqual(plan["headlineDecision"], "omit")
        self.assertEqual(plan["headlineText"], "")


class Builder2PreRunwayValidationTests(unittest.TestCase):
    def test_valid_continuous_passes(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        validate_builder2_pre_runway(plan)

    def test_invalid_anchor_fails_before_generation(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        plan["visualAnchor"] = {"whyEssential": "x"}
        with self.assertRaises(Builder2WinnerDownstreamError) as ctx:
            validate_builder2_pre_runway(plan)
        self.assertIn("visualAnchor", ctx.exception.code)

    def test_invalid_sequence_fails(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        plan["sequence"] = {"beginning": "x", "development": "", "resolution": "z"}
        with self.assertRaises(Builder2WinnerDownstreamError):
            validate_builder2_pre_runway(plan)

    @patch("engine.runway_video.load_tournament_state", return_value=None)
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
        },
        clear=False,
    )
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.generate_video_start_image_data_uri")
    @patch("engine.runway_video._create_image_to_video_task")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "קופי"))
    def test_no_paid_task_after_validation_failure(
        self,
        _product,
        image_task,
        start_image,
        tournament,
        _phase,
        _redis_name,
        _load_state,
    ) -> None:
        bad = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        bad["visualAnchor"] = {}
        tournament.return_value = bad
        with self.assertRaises(RunwayVideoMVPError):
            _generate_one_video_mvp_body("קופי", "desc", job_id="job-pre-runway-fail")
        image_task.assert_not_called()
        start_image.assert_not_called()


class Builder2StartImageTests(unittest.TestCase):
    def test_opening_represents_beginning(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertIn("opening moment", prompt.lower())
        self.assertIn("not the final resolution", prompt.lower())

    def test_resolution_not_leaked(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertNotIn(plan["sequence"]["resolution"], prompt)

    def test_nested_fields_do_not_crash(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        self.assertTrue(build_ace_start_frame_image_prompt(plan))


class Builder2ProductionRegressionTests(unittest.TestCase):
    @patch("engine.runway_video.load_tournament_state", return_value=None)
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
        },
        clear=False,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once", return_value={"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]})
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-regression")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "קופי"))
    def test_greenpeace_include_regression(
        self,
        _product,
        tournament,
        _start_image,
        image_task,
        _poll,
        _sleep,
        _promise,
        _copy,
        _overlay,
        _phase,
        _redis_name,
        _load_state,
    ) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(headline="מכירים את הקבוע שלכם."),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        tournament.return_value = plan
        url = _generate_one_video_mvp_body("קופי", "desc", job_id="job-greenpeace-include")
        self.assertTrue(url)
        self.assertEqual(plan["headlineText"].count("קופי"), 1)
        image_task.assert_called_once()

    @patch("engine.runway_video.load_tournament_state", return_value=None)
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
        },
        clear=False,
    )
    @patch("engine.runway_video.postprocess_video_headline")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once", return_value={"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]})
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-omit")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "קופי"))
    def test_greenpeace_omit_regression(
        self,
        _product,
        tournament,
        _start_image,
        image_task,
        _poll,
        _sleep,
        _promise,
        _copy,
        overlay_mock,
        _phase,
        _redis_name,
        _load_state,
    ) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _canonical_continuous_plan(headline_decision="omit"),
            product_name="קופי",
            product_description="desc",
            content_language="he",
        )
        tournament.return_value = plan
        url = _generate_one_video_mvp_body("קופי", "desc", job_id="job-greenpeace-omit")
        self.assertTrue(url)
        overlay_mock.assert_not_called()
        image_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
