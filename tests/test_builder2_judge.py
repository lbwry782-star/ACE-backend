"""
Builder2 Judge tests — parsing, validation, repair, retry, tournament continuation.
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_judge import (
    calculate_judge_total,
    judge_candidate,
    validate_judge_purity,
    validate_judge_response,
)
from engine.builder2_tournament_contracts import (
    JUDGE_SCORE_RANGES,
    JUDGMENT_SCHEMA_VERSION,
    Builder2TournamentError,
    compare_candidate_rankings,
)
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from tests.test_builder2_tournament import (
    TournamentMockLLM,
    _candidate,
    _judgment,
    _strategy,
    _winner_plan,
)


def _valid_judgment(candidate_id: str = "cand-1", *, eligible: bool = True) -> Dict[str, Any]:
    return _judgment(candidate_id, eligible=eligible)


class TestJudgeValidation(unittest.TestCase):
    def test_valid_english_judgment_passes(self) -> None:
        parsed, total, scores = validate_judge_response(_valid_judgment("cand-1"), candidate_id="cand-1")
        self.assertTrue(parsed["eligible"])
        self.assertEqual(total, calculate_judge_total(scores))

    def test_valid_hebrew_judgment_passes(self) -> None:
        judgment = _valid_judgment("cand-he")
        judgment["verdict"] = "הוכחה ויזואלית ברורה."
        judgment["strengths"] = ["מנגנון ברור"]
        judgment["weaknesses"] = ["חלשות קלה"]
        judgment["prototypeQualityComparison"] = "יישום שיטה ללא העתקה."
        parsed, total, _ = validate_judge_response(judgment, candidate_id="cand-he")
        self.assertGreater(total, 0)

    def test_empty_output_code(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-1",
                candidate=_candidate("closest"),
                llm_client=lambda **kwargs: "",
            )
        self.assertEqual(ctx.exception.args[0], "builder2_judge_empty_response")

    def test_malformed_json_code(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-1",
                candidate=_candidate("closest"),
                llm_client=lambda **kwargs: "not-json",
            )
        self.assertEqual(ctx.exception.args[0], "builder2_judge_malformed_response")

    def test_wrong_schema_version(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["schemaVersion"] = "wrong"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:schemaVersion")

    def test_wrong_candidate_id(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["candidateId"] = "cand-other"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_validation_failed:candidateId")

    def test_missing_score_field(self) -> None:
        bad = _valid_judgment("cand-1")
        del bad["scores"]["silentVisualClarity"]
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:scores.silentVisualClarity")

    def test_unknown_score_field_rejected(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["scores"]["bonusPoints"] = 5
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:scores")

    def test_model_total_rejected(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["totalScore"] = 999
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:scores.total")

    def test_string_boolean_fails_validation(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["eligible"] = "true"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:eligible")

    def test_string_array_fails_validation(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["strengths"] = "one strength"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:strengths")

    def test_numeric_string_score_fails_validation(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["scores"]["silentVisualClarity"] = "12"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_score_invalid:silentVisualClarity")

    def test_integer_float_normalized(self) -> None:
        bad = _valid_judgment("cand-1")
        bad["scores"]["silentVisualClarity"] = 12.0
        _, total, scores = validate_judge_response(bad, candidate_id="cand-1")
        self.assertEqual(scores["silentVisualClarity"], 12)
        self.assertEqual(total, calculate_judge_total(scores))


class TestJudgeScoreRanges(unittest.TestCase):
    def test_every_minimum_score_accepted(self) -> None:
        for name, (low, _high) in JUDGE_SCORE_RANGES.items():
            judgment = _valid_judgment("cand-1")
            judgment["scores"][name] = low
            validate_judge_response(judgment, candidate_id="cand-1")

    def test_every_maximum_score_accepted(self) -> None:
        for name, (_low, high) in JUDGE_SCORE_RANGES.items():
            judgment = _valid_judgment("cand-1")
            judgment["scores"][name] = high
            validate_judge_response(judgment, candidate_id="cand-1")

    def test_score_below_minimum_rejected(self) -> None:
        judgment = _valid_judgment("cand-1")
        judgment["scores"]["silentVisualClarity"] = -1
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(judgment, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_score_invalid:silentVisualClarity")

    def test_score_above_maximum_rejected(self) -> None:
        judgment = _valid_judgment("cand-1")
        judgment["scores"]["silentVisualClarity"] = 18
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(judgment, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_score_invalid:silentVisualClarity")

    def test_server_total_calculated(self) -> None:
        judgment = _valid_judgment("cand-1")
        _, total, scores = validate_judge_response(judgment, candidate_id="cand-1")
        self.assertEqual(total, sum(scores.values()))
        self.assertNotIn("total", judgment["scores"])

    def test_model_cannot_override_total(self) -> None:
        judgment = _valid_judgment("cand-1")
        judgment["scores"]["totalScore"] = 999
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(judgment, candidate_id="cand-1")
        self.assertEqual(ctx.exception.args[0], "builder2_judge_schema_invalid:scores.total")

    def test_tie_break_fields_use_validated_scores(self) -> None:
        better = {
            "candidateId": "a",
            "totalScore": 90,
            "tieScores": {"silentVisualClarity": 14, "problemAdvantageIntegrity": 18, "runwayFeasibility": 9},
            "completedAt": "2026-01-02T00:00:00+00:00",
            "eligible": True,
        }
        weaker = {
            "candidateId": "b",
            "totalScore": 70,
            "tieScores": {"silentVisualClarity": 10, "problemAdvantageIntegrity": 12, "runwayFeasibility": 7},
            "completedAt": "2026-01-03T00:00:00+00:00",
            "eligible": True,
        }
        self.assertGreater(compare_candidate_rankings(better, weaker), 0)


class TestValidNegativeJudgment(unittest.TestCase):
    def test_eligible_false_stored_normally(self) -> None:
        judgment = _valid_judgment("cand-neg", eligible=False)
        parsed, total, scores = validate_judge_response(judgment, candidate_id="cand-neg")
        self.assertFalse(parsed["eligible"])
        self.assertTrue(parsed["disqualifiers"])
        self.assertEqual(total, calculate_judge_total(scores))

    def test_negative_does_not_trigger_repair(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            return _valid_judgment("cand-neg", eligible=False)

        judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-neg",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(state["metrics"].get("judgeRepairCalls", 0), 0)

    def test_negative_cannot_win(self) -> None:
        state = {
            "candidates": {
                "c1": {
                    "candidateId": "c1",
                    "eligible": False,
                    "validationStatus": "accepted",
                    "totalScore": 95,
                    "tieScores": {},
                    "completedAt": "2026-01-01T00:00:00+00:00",
                }
            }
        }
        with self.assertRaises(Builder2TournamentError):
            select_global_winner(state)


class TestJudgePurity(unittest.TestCase):
    def test_redesign_rejected(self) -> None:
        bad = _valid_judgment("c1")
        bad["verdict"] = "We should redesign the candidate entirely."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_purity(bad)
        self.assertEqual(ctx.exception.args[0], "builder2_judge_purity_violation:redesigns_candidate")

    def test_invented_creator_intent_rejected(self) -> None:
        bad = _valid_judgment("c1")
        bad["verdict"] = "The creator probably meant a different mechanism."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_purity(bad)
        self.assertEqual(ctx.exception.args[0], "builder2_judge_purity_violation:invents_creator_intent")

    def test_unseen_candidate_comparison_rejected(self) -> None:
        bad = _valid_judgment("c1")
        bad["verdict"] = "Compared to other candidates this is weaker."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_purity(bad)
        self.assertEqual(ctx.exception.args[0], "builder2_judge_purity_violation:compares_unseen_candidates")

    def test_tournament_ranking_rejected(self) -> None:
        bad = _valid_judgment("c1")
        bad["verdict"] = "This would hurt tournament standing."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_purity(bad)
        self.assertEqual(ctx.exception.args[0], "builder2_judge_purity_violation:mentions_tournament_ranking")

    def test_ordinary_critique_allowed(self) -> None:
        bad = _valid_judgment("c1")
        bad["verdict"] = "An alternative framing could be clearer, but the judge mechanism works."
        validate_judge_purity(bad)


class TestJudgeCorrectionFlow(unittest.TestCase):
    def test_repairable_schema_gets_repair(self) -> None:
        calls: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            if len(calls) == 1:
                bad = _valid_judgment("cand-1")
                bad["schemaVersion"] = "wrong"
                return bad
            return _valid_judgment("cand-1")

        judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(state["metrics"]["judgeRepairCalls"], 1)

    def test_successful_repair_produces_valid_judgment(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            if calls["count"] == 1:
                bad = _valid_judgment("cand-1")
                bad["eligible"] = "true"
                return bad
            return _valid_judgment("cand-1")

        _, judgment, total, _ = judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        self.assertTrue(judgment["eligible"])
        self.assertGreater(total, 0)

    def test_failed_repair_does_not_loop(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            bad = _valid_judgment("cand-1")
            bad["schemaVersion"] = "wrong"
            return bad

        with self.assertRaises(Builder2TournamentError):
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-1",
                candidate=_candidate("closest"),
                llm_client=llm,
                state=state,
            )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(state["metrics"]["judgeRepairCalls"], 1)

    def test_purity_triggers_clean_retry(self) -> None:
        calls = {"count": 0}
        prompts: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any):
            calls["count"] += 1
            prompts.append(kwargs.get("prompt", ""))
            if calls["count"] == 1:
                bad = _valid_judgment("cand-1")
                bad["verdict"] = "Compared to other candidates this is weaker."
                return bad
            return _valid_judgment("cand-1")

        judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        self.assertEqual(state["metrics"]["judgeRetryCalls"], 1)
        self.assertEqual(state["metrics"].get("judgeRepairCalls", 0), 0)
        self.assertIn("clean retry", prompts[1].lower())
        self.assertNotIn("Invalid structured output", prompts[1])

    def test_failed_retry_marks_unavailable(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any):
            calls["count"] += 1
            bad = _valid_judgment("cand-1")
            bad["verdict"] = "Compared to other candidates this is weaker."
            return bad

        with self.assertRaises(Builder2TournamentError) as ctx:
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-1",
                candidate=_candidate("closest"),
                llm_client=llm,
                state=state,
            )
        self.assertEqual(calls["count"], 2)
        self.assertIn("builder2_judge_purity_violation", ctx.exception.args[0])
        diagnostics = state["judgeDiagnosticsByCandidate"]["cand-1"]
        self.assertTrue(diagnostics["cleanRetryAttempted"])

    def test_maximum_correction_calls_enforced(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any):
            calls["count"] += 1
            bad = _valid_judgment("cand-1")
            bad["schemaVersion"] = "wrong"
            return bad

        with self.assertRaises(Builder2TournamentError):
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-1",
                candidate=_candidate("closest"),
                llm_client=llm,
                state=state,
            )
        self.assertLessEqual(calls["count"], 2)


class TestJudgeTournamentContinuation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _prototype_from_prompt(self, prompt: str) -> str:
        for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
            if f"Assigned prototype ID: {pid}" in prompt or f'"prototypeId": "{pid}"' in prompt:
                return pid
            if pid in prompt and "goldPrototypeUsed" in prompt:
                return pid
        for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
            if pid in prompt:
                return pid
        return "closest"

    def test_first_judge_failure_does_not_abort(self) -> None:
        judge_calls = {"count": 0}

        def llm(**kwargs: Any):
            role = kwargs.get("role")
            if role == "builder2_strategy":
                return _strategy()
            if role == "builder2_creator":
                prototype_id = self._prototype_from_prompt(kwargs.get("prompt", ""))
                return _candidate(prototype_id)
            if role == "builder2_judge":
                judge_calls["count"] += 1
                candidate_id = "unknown"
                for token in kwargs.get("prompt", "").split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip().strip(",")
                        break
                if judge_calls["count"] == 1:
                    bad = _valid_judgment(candidate_id)
                    bad["eligible"] = "true"
                    return bad
                if "repair" in kwargs.get("prompt", "").lower():
                    bad = _valid_judgment(candidate_id)
                    bad["eligible"] = "false"
                    bad["disqualifiers"] = ["still broken"]
                    return bad
                return _valid_judgment(candidate_id, eligible=True)
            if role == "builder2_winner":
                return _winner_plan()
            raise AssertionError(role)

        with patch.dict(
            os.environ,
            {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS), "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
            clear=True,
        ):
            run_builder2_tournament(
                job_id="job-judge-continue",
                product_name="Product",
                product_description="desc",
                content_language="en",
                llm_client=llm,
                rng_seed="seed-judge",
            )

        state = load_tournament_state("job-judge-continue")
        assert state is not None
        unavailable = [c for c in state["candidates"].values() if c.get("validationStatus") == "judge_unavailable"]
        eligible = [c for c in state["candidates"].values() if c.get("eligible")]
        self.assertEqual(len(unavailable), 1)
        self.assertGreaterEqual(len(eligible), 1)
        self.assertTrue(state.get("winnerCandidateId"))

    def test_all_judgments_unavailable_raises_no_valid_candidate(self) -> None:
        def llm(**kwargs: Any):
            role = kwargs.get("role")
            if role == "builder2_strategy":
                return _strategy()
            if role == "builder2_creator":
                return _candidate("closest")
            if role == "builder2_judge":
                return "not-json"
            raise AssertionError(role)

        with patch.dict(
            os.environ,
            {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
            clear=True,
        ):
            with self.assertRaises(Builder2TournamentError) as ctx:
                run_builder2_tournament(
                    job_id="job-all-judge-fail",
                    product_name="Product",
                    product_description="desc",
                    content_language="en",
                    llm_client=llm,
                    rng_seed="seed",
                )
        self.assertEqual(ctx.exception.args[0], "builder2_tournament_no_valid_candidate")

    def test_unavailable_candidate_cannot_win(self) -> None:
        state = {
            "candidates": {
                "c1": {
                    "candidateId": "c1",
                    "eligible": False,
                    "validationStatus": "judge_unavailable",
                    "totalScore": None,
                    "tieScores": {},
                    "completedAt": "2026-01-01T00:00:00+00:00",
                }
            }
        }
        with self.assertRaises(Builder2TournamentError):
            select_global_winner(state)

    def test_metrics_counted(self) -> None:
        state = {"jobId": "job-a", "tournamentId": "t1", "metrics": {}}
        other = {"jobId": "job-b", "tournamentId": "t2", "metrics": {}}

        def llm(**kwargs: Any):
            if kwargs.get("role") == "builder2_judge":
                return _valid_judgment("cand-1")
            raise AssertionError(kwargs.get("role"))

        judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        with self.assertRaises(Builder2TournamentError):
            judge_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                prototype_id="closest",
                candidate_id="cand-2",
                candidate=_candidate("closest"),
                llm_client=lambda **kwargs: "",
                state=other,
            )
        self.assertEqual(state["metrics"]["judgeCalls"], 1)
        self.assertEqual(other["metrics"]["judgeUnavailableCandidates"], 1)
        self.assertEqual(other["metrics"]["judgeRejectedResponses"], 1)


class TestProductionJudgeRegression(unittest.TestCase):
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
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-judge-regression")
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    def test_greenpeace_judge_invalid_then_continue(
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

        def llm(**kwargs: Any):
            role = kwargs.get("role")
            prompt = kwargs.get("prompt", "")
            if role == "builder2_strategy":
                return _strategy()
            if role == "builder2_creator":
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                    if f'"prototypeId": "{pid}"' in prompt or f"Assigned prototype ID: {pid}" in prompt:
                        prototype_id = pid
                        break
                return _candidate(prototype_id)
            if role == "builder2_judge":
                candidate_id = "unknown"
                for token in prompt.split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip().strip(",")
                        break
                if "greenpeace_essential_pairing" not in prompt:
                    return _valid_judgment(candidate_id, eligible=True)
                if "repair" in prompt.lower():
                    bad = _valid_judgment(candidate_id)
                    bad["schemaVersion"] = "still_wrong"
                    return bad
                if "clean retry" in prompt.lower():
                    return _valid_judgment(candidate_id, eligible=True)
                bad = _valid_judgment(candidate_id)
                bad["eligible"] = "true"
                return bad
            if role == "builder2_winner":
                return _winner_plan()
            raise AssertionError(role)

        def _run_tournament(**kwargs: Any):
            kwargs["llm_client"] = llm
            return run_builder2_tournament(**kwargs)

        with patch("engine.runway_video.run_builder2_tournament", side_effect=_run_tournament):
            from engine.runway_video import _generate_one_video_mvp_body

            _generate_one_video_mvp_body("Product", "desc", job_id="job-judge-regression")

        state = load_tournament_state("job-judge-regression")
        assert state is not None
        greenpeace_unavailable = [
            c
            for c in state["candidates"].values()
            if c.get("prototypeId") == "greenpeace_essential_pairing"
            and c.get("validationStatus") == "judge_unavailable"
        ]
        judged = [c for c in state["candidates"].values() if c.get("judgmentId")]
        self.assertEqual(len(greenpeace_unavailable), 1)
        self.assertGreaterEqual(len(judged), 1)
        self.assertTrue(state.get("winnerCandidateId"))
        self.assertGreaterEqual(len(state.get("initialActivePrototypeIds") or []), 6)
        _image_task.assert_called_once()


class TestJudgeNormalization(unittest.TestCase):
    def test_confidence_percentage_normalized(self) -> None:
        judgment = _valid_judgment("c1")
        judgment["confidence"] = 82
        parsed, _, _ = validate_judge_response(judgment, candidate_id="c1")
        self.assertAlmostEqual(parsed["confidence"], 0.82)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
