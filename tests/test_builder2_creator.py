"""
Builder2 Creator tests — parsing, validation, repair, retry, tournament continuation.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_creator import (
    generate_creator_candidate,
    normalize_creator_raw,
    validate_creator_candidate,
    validate_creator_purity,
)
from engine.builder2_tournament_contracts import CANDIDATE_SCHEMA_VERSION, Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _judgment, _strategy


def _valid_candidate(prototype_id: str = "closest", *, language_structure: str = "continuous_event") -> Dict[str, Any]:
    cand = _candidate(prototype_id, structure=language_structure)
    cand["creatorReport"]["goldPrototypeUsed"] = prototype_id
    cand["creatorReport"]["visualParallelType"] = cand["visualParallelType"]
    return cand


class TestCreatorValidation(unittest.TestCase):
    def test_valid_english_candidate_passes(self) -> None:
        validate_creator_candidate(
            _valid_candidate("closest"),
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
        )

    def test_valid_hebrew_candidate_passes(self) -> None:
        cand = _valid_candidate("closest")
        cand["conceptSummary"] = "הוכחה ויזואלית ברורה לקרבה."
        cand["sevenSecondStructure"]["beginning"] = "שני אנשים עומדים במרחק."
        validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
        )

    def test_continuous_event_passes(self) -> None:
        validate_creator_candidate(
            _valid_candidate("closest", language_structure="continuous_event"),
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
        )

    def test_variation_montage_passes(self) -> None:
        cand = _valid_candidate("closest", language_structure="variation_montage")
        validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
        )

    def test_wrong_schema_version(self) -> None:
        cand = _valid_candidate("closest")
        cand["schemaVersion"] = "wrong"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")
        self.assertEqual(ctx.exception.args[0], "builder2_creator_schema_invalid:schemaVersion")

    def test_wrong_structure_type(self) -> None:
        cand = _valid_candidate("closest")
        cand["structureType"] = "montage"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")
        self.assertEqual(ctx.exception.args[0], "builder2_creator_schema_invalid:structureType")

    def test_wrong_continuity_risk(self) -> None:
        cand = _valid_candidate("closest")
        cand["runwayFeasibility"]["continuityRisk"] = "very_low"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")
        self.assertEqual(ctx.exception.args[0], "builder2_creator_schema_invalid:runwayFeasibility.continuityRisk")

    def test_string_generation_risks_normalized(self) -> None:
        cand = _valid_candidate("closest")
        cand["runwayFeasibility"]["generationRisks"] = "possible hand overlap"
        normalized = normalize_creator_raw(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
        )
        self.assertEqual(normalized["runwayFeasibility"]["generationRisks"], ["possible hand overlap"])
        validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")

    def test_string_boolean_normalized(self) -> None:
        cand = _valid_candidate("closest")
        cand["silentVerification"]["understandableWithoutAudio"] = "true"
        validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")

    def test_common_structure_alias_normalized(self) -> None:
        cand = _valid_candidate("closest")
        cand["structureType"] = "continuous event"
        validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")


class TestCreatorPurity(unittest.TestCase):
    def test_other_candidate_rejected(self) -> None:
        cand = _valid_candidate("closest")
        cand["conceptSummary"] = "Better than the previous candidate."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_purity(cand)
        self.assertEqual(ctx.exception.args[0], "builder2_creator_purity_violation:mentions_other_candidate")

    def test_judge_score_rejected(self) -> None:
        cand = _valid_candidate("closest")
        cand["conceptSummary"] = "The judge score would be higher with more motion."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_purity(cand)
        self.assertEqual(ctx.exception.args[0], "builder2_creator_purity_violation:mentions_judge_score")

    def test_tournament_ranking_rejected(self) -> None:
        cand = _valid_candidate("closest")
        cand["conceptSummary"] = "This would improve tournament standing."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_purity(cand)
        self.assertEqual(ctx.exception.args[0], "builder2_creator_purity_violation:mentions_tournament_ranking")

    def test_ordinary_judge_word_allowed(self) -> None:
        cand = _valid_candidate("closest")
        cand["conceptSummary"] = "The visual judge of distance closes quickly."
        validate_creator_purity(cand)


class TestCreatorGenerationFlow(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_empty_response(self) -> None:
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_creator_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                round_index=1,
                attempt_number=1,
                runway_mode="image_to_video",
                llm_client=lambda **kwargs: "",
                state=state,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_creator_empty_response")

    def test_malformed_response(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_creator_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                round_index=1,
                attempt_number=1,
                runway_mode="image_to_video",
                llm_client=lambda **kwargs: "not-json",
            )
        self.assertEqual(ctx.exception.args[0], "builder2_creator_malformed_response")

    def test_repairable_schema_gets_repair(self) -> None:
        calls: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            if len(calls) == 1:
                bad = _valid_candidate("closest")
                bad["schemaVersion"] = "wrong"
                return bad
            return _valid_candidate("closest")

        generate_creator_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(state["metrics"]["creatorRepairCalls"], 1)

    def test_purity_triggers_clean_retry_not_repair(self) -> None:
        calls = {"count": 0}
        prompts: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any):
            calls["count"] += 1
            prompts.append(kwargs.get("prompt", ""))
            if calls["count"] == 1:
                bad = _valid_candidate("closest")
                bad["conceptSummary"] = "Better than the previous candidate."
                return bad
            return _valid_candidate("closest")

        generate_creator_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(state["metrics"]["creatorRetryCalls"], 1)
        self.assertEqual(state["metrics"].get("creatorRepairCalls", 0), 0)
        self.assertIn("clean retry", prompts[1].lower())
        self.assertNotIn("Invalid structured output", prompts[1])


class TestCreatorTournamentContinuation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_first_prototype_failure_does_not_abort(self) -> None:
        calls: List[str] = []

        def llm(**kwargs: Any):
            calls.append(kwargs.get("role", ""))
            if kwargs.get("role") == "builder2_creator":
                prototype_id = "closest"
                for pid in ("closest", "winning_card", "forgot"):
                    if f"Assigned prototype ID: {pid}" in kwargs.get("prompt", ""):
                        prototype_id = pid
                        break
                if prototype_id == "closest":
                    bad = _valid_candidate("closest")
                    bad["schemaVersion"] = "wrong"
                    return bad
                return _valid_candidate(prototype_id)
            if kwargs.get("role") == "builder2_judge":
                candidate_id = "unknown"
                for token in kwargs.get("prompt", "").split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip()
                        break
                return _judgment(candidate_id)
            if kwargs.get("role") == "builder2_strategy":
                return _strategy()
            if kwargs.get("role") == "builder2_winner":
                from tests.test_builder2_tournament import _winner_plan

                return _winner_plan()
            raise AssertionError(kwargs.get("role"))

        with patch.dict(
            os.environ,
            {
                "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card,forgot",
                "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
            },
            clear=True,
        ):
            run_builder2_tournament(
                job_id="job-continue",
                product_name="Product",
                product_description="desc",
                content_language="en",
                llm_client=llm,
                rng_seed="seed",
            )

        state = load_tournament_state("job-continue")
        assert state is not None
        rejected = [c for c in state["candidates"].values() if c.get("validationStatus") == "creator_rejected"]
        judged = [c for c in state["candidates"].values() if c.get("judgmentId")]
        self.assertEqual(len(rejected), 1)
        self.assertGreaterEqual(len(judged), 2)
        self.assertTrue(state.get("winnerCandidateId"))
        self.assertNotIn("builder2_judge", [c for c in calls if c == "builder2_judge" and "closest" in str(c)])

    def test_all_creator_failures_raise_no_valid_candidate(self) -> None:
        def llm(**kwargs: Any):
            if kwargs.get("role") == "builder2_creator":
                bad = _valid_candidate("closest")
                bad["schemaVersion"] = "wrong"
                return bad
            if kwargs.get("role") == "builder2_strategy":
                return _strategy()
            raise AssertionError(kwargs.get("role"))

        with patch.dict(
            os.environ,
            {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
            clear=True,
        ):
            with self.assertRaises(Builder2TournamentError) as ctx:
                run_builder2_tournament(
                    job_id="job-all-fail",
                    product_name="Product",
                    product_description="desc",
                    content_language="en",
                    llm_client=llm,
                    rng_seed="seed",
                )
        self.assertEqual(ctx.exception.args[0], "builder2_tournament_no_valid_candidate")


class TestProductionRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card,forgot",
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-regression")
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    def test_strategy_ok_closest_invalid_then_continue(
        self,
        _product_name,
        _start_image,
        _text_task,
        _image_task,
        poll_mock,
        _sleep,
        _promise,
        _copy,
        _post,
        _phase,
        _redis_name,
    ) -> None:
        poll_mock.return_value = {"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]}
        calls: List[str] = []

        def llm(**kwargs: Any):
            calls.append(kwargs.get("role", ""))
            if kwargs.get("role") == "builder2_strategy":
                return _strategy()
            if kwargs.get("role") == "builder2_creator":
                prototype_id = "closest"
                for pid in ("closest", "winning_card", "forgot"):
                    if f"Assigned prototype ID: {pid}" in kwargs.get("prompt", ""):
                        prototype_id = pid
                        break
                if prototype_id == "closest":
                    bad = _valid_candidate("closest")
                    bad["schemaVersion"] = "wrong"
                    return bad
                return _valid_candidate(prototype_id)
            if kwargs.get("role") == "builder2_judge":
                candidate_id = "unknown"
                for token in kwargs.get("prompt", "").split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip()
                        break
                return _judgment(candidate_id, total_hint=90)
            if kwargs.get("role") == "builder2_winner":
                from tests.test_builder2_tournament import _winner_plan

                return _winner_plan()
            raise AssertionError(kwargs.get("role"))

        def _run_tournament(**kwargs: Any):
            kwargs["llm_client"] = llm
            return run_builder2_tournament(**kwargs)

        with patch("engine.runway_video.run_builder2_tournament", side_effect=_run_tournament):
            from engine.runway_video import _generate_one_video_mvp_body

            _generate_one_video_mvp_body("Product", "desc", job_id="job-regression")

        state = load_tournament_state("job-regression")
        assert state is not None
        rejected = [c for c in state["candidates"].values() if c.get("validationStatus") == "creator_rejected"]
        judged = [c for c in state["candidates"].values() if c.get("judgmentId")]
        self.assertEqual(len(rejected), 1)
        self.assertGreaterEqual(len(judged), 2)
        self.assertTrue(state.get("winnerDevelopmentPlan") or state.get("winnerCandidateId"))
        _image_task.assert_called_once()


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
