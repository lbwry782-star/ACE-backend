"""
Builder2 Judge contract, circuit breaker, and preflight tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_judge import (
    collect_judge_structural_errors,
    judge_candidate,
    validate_judge_response,
)
from engine.builder2_judge_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE,
    assert_judge_contract_available,
    is_judge_contract_circuit_breaker_tripped,
    record_judge_contract_failure,
)
from engine.builder2_judge_core_contract import resolve_creator_verbal_decision
from engine.builder2_judge_preflight import (
    DEFAULT_PREFLIGHT_CANDIDATE_ID,
    DEFAULT_PREFLIGHT_JOB_ID,
    run_one_isolated_judge_preflight,
)
from engine.builder2_judge_preflight_guard import JudgePreflightIsolationGuard, PREFLIGHT_ISOLATION_ERROR
from engine.builder2_methodology_validation import validate_judge_methodology
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_prompts import build_judge_prompt
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, new_tournament_state, save_tournament_state
from engine.builder2_prototypes import require_prototype
from tests.builder2_methodology_fixtures import methodology_judgment_extras
from tests.test_builder2_tournament import (
    _candidate,
    _judgment,
    _strategy,
    _winner_plan_from_prompt,
)


def _valid_judgment(candidate_id: str = "cand-1", *, eligible: bool = True) -> Dict[str, Any]:
    return _judgment(candidate_id, eligible=eligible)


def _negative_verbal_judgment(candidate_id: str = "cand-neg", *, eligible: bool = False) -> Dict[str, Any]:
    judgment = _valid_judgment(candidate_id, eligible=eligible)
    judgment["verbalLayerAssessment"] = {
        "applicability": "available",
        "keywordBornFromVisual": False,
        "visualMeaningIsClear": False,
        "strategicMeaningIsClear": False,
        "twoMeaningsReinforceEachOther": False,
        "notes": "The verbal layer does not follow from the visible mechanism.",
    }
    if not eligible:
        judgment["disqualifiers"] = ["weak_verbal_layer"]
        judgment["weaknesses"] = ["Verbal layer fails the born-from-visual test."]
    return judgment


def _persisted_preflight_state(*, candidate_id: str = DEFAULT_PREFLIGHT_CANDIDATE_ID) -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    candidate["verbalPotential"] = {
        "decision": "not_needed",
        "reason": "The visible fan behavior communicates absence without a headline.",
    }
    state = new_tournament_state(
        job_id=DEFAULT_PREFLIGHT_JOB_ID,
        language="he",
        active_prototype_ids=list(DEFAULT_ACTIVE_PROTOTYPE_IDS),
        random_seed="judge-preflight-seed",
    )
    state["productName"] = "Preflight Product"
    state["productDescription"] = "Preflight description"
    state["contentLanguage"] = "he"
    state["strategyFoundation"] = strategy
    state["candidates"][candidate_id] = {
        "candidateId": candidate_id,
        "prototypeId": "summer_fan",
        "roundIndex": 1,
        "attemptNumber": 1,
        "creatorOutput": candidate,
        "validationStatus": "accepted",
        "status": "accepted",
        "judgmentId": None,
        "eligible": False,
        "totalScore": None,
        "tieScores": {},
        "completedAt": "2026-01-01T00:00:00+00:00",
    }
    return state


class TestJudgeBooleanValidity(unittest.TestCase):
    def test_keyword_born_from_visual_false_is_valid(self) -> None:
        judgment = _valid_judgment("c1")
        judgment["verbalLayerAssessment"]["keywordBornFromVisual"] = False
        judgment["verbalLayerAssessment"]["notes"] = "Keyword does not follow the visible mechanism."
        parsed, _, _ = validate_judge_response(judgment, candidate_id="c1", candidate=_candidate("closest"))
        self.assertFalse(parsed["verbalLayerAssessment"]["keywordBornFromVisual"])

    def test_visual_meaning_is_clear_false_is_valid(self) -> None:
        judgment = _valid_judgment("c1")
        judgment["verbalLayerAssessment"]["visualMeaningIsClear"] = False
        judgment["verbalLayerAssessment"]["notes"] = "Visual meaning is muddy."
        validate_judge_response(judgment, candidate_id="c1", candidate=_candidate("closest"))

    def test_all_verbal_booleans_false_are_valid(self) -> None:
        judgment = _negative_verbal_judgment("c-neg", eligible=True)
        judgment["weaknesses"] = ["Verbal layer is weak."]
        parsed, _, _ = validate_judge_response(judgment, candidate_id="c-neg", candidate=_candidate("closest"))
        verbal = parsed["verbalLayerAssessment"]
        self.assertFalse(verbal["keywordBornFromVisual"])
        self.assertFalse(verbal["visualMeaningIsClear"])

    def test_eligible_false_is_valid_stored_judgment(self) -> None:
        judgment = _negative_verbal_judgment("c-neg", eligible=False)
        parsed, total, scores = validate_judge_response(judgment, candidate_id="c-neg", candidate=_candidate("closest"))
        self.assertFalse(parsed["eligible"])
        self.assertTrue(parsed["disqualifiers"])
        self.assertGreater(total, 0)
        self.assertEqual(len(scores), 8)


class TestJudgeStructuralRepairRouting(unittest.TestCase):
    def test_missing_verbal_object_is_structural(self) -> None:
        judgment = _valid_judgment("c1")
        judgment.pop("verbalLayerAssessment")
        errors = collect_judge_structural_errors(
            judgment,
            candidate_id="c1",
            candidate=_candidate("closest"),
        )
        self.assertTrue(any("verbalLayerAssessment" == item.split(":", 1)[-1] or item.endswith(":verbalLayerAssessment") for item in errors))

    def test_string_false_is_structural(self) -> None:
        judgment = _valid_judgment("c1")
        judgment["verbalLayerAssessment"]["keywordBornFromVisual"] = "false"
        errors = collect_judge_structural_errors(
            judgment,
            candidate_id="c1",
            candidate=_candidate("closest"),
        )
        self.assertIn("builder2_judge_schema_invalid:verbalLayerAssessment.keywordBornFromVisual", errors)

    def test_boolean_false_does_not_trigger_repair(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            return _negative_verbal_judgment("cand-neg", eligible=False)

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

    def test_negative_judgment_is_not_unavailable_on_success(self) -> None:
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        _, judgment, _, _ = judge_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            candidate_id="cand-neg",
            candidate=_candidate("closest"),
            llm_client=lambda **kwargs: _negative_verbal_judgment("cand-neg", eligible=False),
            state=state,
        )
        self.assertFalse(judgment["eligible"])


class TestJudgeConditionalVerbalLayer(unittest.TestCase):
    def test_creator_not_needed_allows_null_booleans(self) -> None:
        candidate = _candidate("summer_fan")
        candidate["verbalPotential"] = {"decision": "not_needed", "reason": "Visual proof is enough."}
        judgment = _valid_judgment("c1")
        judgment["verbalLayerAssessment"] = {
            "applicability": "not_needed",
            "keywordBornFromVisual": None,
            "visualMeaningIsClear": None,
            "strategicMeaningIsClear": None,
            "twoMeaningsReinforceEachOther": None,
            "notes": "Verbal analysis is not applicable because the visual proof stands alone.",
        }
        validate_judge_methodology(judgment, candidate=candidate)

    def test_creator_not_found_remains_judgeable(self) -> None:
        candidate = _candidate("closest")
        candidate["verbalPotential"] = {"decision": "not_found", "reason": "No defensible keyword exists."}
        judgment = _valid_judgment("c1", eligible=False)
        judgment["disqualifiers"] = ["no_defensible_keyword"]
        judgment["verbalLayerAssessment"] = {
            "applicability": "not_found",
            "keywordBornFromVisual": None,
            "visualMeaningIsClear": None,
            "strategicMeaningIsClear": None,
            "twoMeaningsReinforceEachOther": None,
            "notes": "Creator found no defensible keyword; absence is a weakness.",
        }
        validate_judge_methodology(judgment, candidate=candidate)


class TestJudgeCoherenceAndTournamentOutcomes(unittest.TestCase):
    def test_contradictory_negative_assessment_requires_notes_or_retry(self) -> None:
        judgment = _valid_judgment("c1", eligible=True)
        judgment["verbalLayerAssessment"]["keywordBornFromVisual"] = False
        judgment["verbalLayerAssessment"]["notes"] = ""
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_response(judgment, candidate_id="c1", candidate=_candidate("closest"))
        self.assertIn("verbalLayerAssessment.notes", str(ctx.exception))

    def test_all_valid_negative_judgments_yield_no_eligible_candidate(self) -> None:
        state = {
            "candidates": {
                f"c{i}": {
                    "candidateId": f"c{i}",
                    "eligible": False,
                    "validationStatus": "accepted",
                    "judgmentId": f"j{i}",
                    "totalScore": 10 + i,
                    "tieScores": {},
                    "completedAt": f"2026-01-0{i}T00:00:00+00:00",
                }
                for i in range(1, 7)
            }
        }
        with self.assertRaises(Builder2TournamentError) as ctx:
            select_global_winner(state)
        self.assertEqual(ctx.exception.args[0], "builder2_tournament_no_eligible_candidate")

    def test_mixed_unavailable_still_no_valid_candidate(self) -> None:
        state = {
            "candidates": {
                "c1": {
                    "candidateId": "c1",
                    "eligible": False,
                    "validationStatus": "accepted",
                    "judgmentId": "j1",
                    "totalScore": 10,
                    "tieScores": {},
                    "completedAt": "2026-01-01T00:00:00+00:00",
                },
                "c2": {
                    "candidateId": "c2",
                    "eligible": False,
                    "validationStatus": "accepted",
                    "judgmentId": None,
                    "totalScore": None,
                    "tieScores": {},
                    "completedAt": "2026-01-02T00:00:00+00:00",
                },
            }
        }
        with self.assertRaises(Builder2TournamentError) as ctx:
            select_global_winner(state)
        self.assertEqual(ctx.exception.args[0], "builder2_tournament_no_valid_candidate")


class TestJudgeCircuitBreaker(unittest.TestCase):
    def test_two_common_failures_trip_breaker(self) -> None:
        state: Dict[str, Any] = {}
        record_judge_contract_failure(
            state,
            candidate_id="c1",
            error_paths=["verbalLayerAssessment", "verbalLayerAssessment.keywordBornFromVisual"],
        )
        self.assertFalse(is_judge_contract_circuit_breaker_tripped(state))
        record_judge_contract_failure(
            state,
            candidate_id="c2",
            error_paths=["verbalLayerAssessment", "verbalLayerAssessment.visualMeaningIsClear"],
        )
        self.assertTrue(is_judge_contract_circuit_breaker_tripped(state))
        with self.assertRaises(Builder2TournamentError) as ctx:
            assert_judge_contract_available(state)
        self.assertIn(SYSTEMIC_FAILURE_CODE, str(ctx.exception))

    def test_remaining_judge_calls_stop_after_breaker(self) -> None:
        enable_memory_store()
        try:
            state = new_tournament_state(
                job_id="job-judge-breaker",
                language="en",
                active_prototype_ids=["closest", "summer_fan"],
                random_seed="seed-judge-breaker",
            )
            state["strategyFoundation"] = _strategy()
            save_tournament_state("job-judge-breaker", state)

            calls = {"judge": 0}

            def llm(**kwargs: Any):
                role = kwargs.get("role")
                prompt = kwargs.get("prompt", "")
                if role == "builder2_strategy":
                    return _strategy()
                if role == "builder2_creator":
                    prototype_id = "closest"
                    if "summer_fan" in prompt:
                        prototype_id = "summer_fan"
                    return _candidate(prototype_id)
                if role == "builder2_judge":
                    calls["judge"] += 1
                    candidate_id = "unknown"
                    for token in prompt.split():
                        if token.startswith("cand-"):
                            candidate_id = token.strip().strip(",")
                            break
                    bad = _valid_judgment(candidate_id)
                    bad.pop("verbalLayerAssessment")
                    return bad
                raise AssertionError(role)

            with patch.dict(
                os.environ,
                {
                    "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,summer_fan",
                    "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
                },
                clear=True,
            ):
                with self.assertRaises(Builder2TournamentError) as ctx:
                    run_builder2_tournament(
                        job_id="job-judge-breaker",
                        product_name="Product",
                        product_description="desc",
                        content_language="en",
                        llm_client=llm,
                        rng_seed="seed-judge-breaker",
                    )
            self.assertIn(SYSTEMIC_FAILURE_CODE, str(ctx.exception))
            self.assertLessEqual(calls["judge"], 3)
        finally:
            disable_memory_store()

    def test_healthy_six_judge_path_has_zero_repairs(self) -> None:
        enable_memory_store()
        try:
            metrics_holder: Dict[str, Any] = {}

            def llm(**kwargs: Any):
                role = kwargs.get("role")
                prompt = kwargs.get("prompt", "")
                if role == "builder2_strategy":
                    return _strategy()
                if role == "builder2_creator":
                    prototype_id = "closest"
                    for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                        if pid in prompt:
                            prototype_id = pid
                            break
                    return _candidate(prototype_id, prompt=prompt)
                if role == "builder2_judge":
                    candidate_id = "unknown"
                    for token in prompt.split():
                        if token.startswith("cand-"):
                            candidate_id = token.strip().strip(",")
                            break
                    return _valid_judgment(candidate_id, eligible=True)
                if role == "builder2_winner":
                    return _winner_plan_from_prompt(prompt)
                raise AssertionError(role)

            with patch.dict(
                os.environ,
                {
                    "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
                    "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
                },
                clear=True,
            ):
                run_builder2_tournament(
                    job_id="job-judge-healthy",
                    product_name="Product",
                    product_description="desc",
                    content_language="en",
                    llm_client=llm,
                    rng_seed="seed-judge-healthy",
                )
            final_state = load_tournament_state("job-judge-healthy")
            assert final_state is not None
            metrics = final_state.get("metrics") or {}
            self.assertEqual(metrics.get("judgeRepairCalls", 0), 0)
            self.assertEqual(metrics.get("judgeRetryCalls", 0), 0)
            self.assertEqual(metrics.get("totalReasoningCalls"), 14)
        finally:
            disable_memory_store()


class TestJudgePreflight(unittest.TestCase):
    def setUp(self) -> None:
        JudgePreflightIsolationGuard.begin()

    def tearDown(self) -> None:
        JudgePreflightIsolationGuard.end()

    def test_preflight_loads_persisted_candidate(self) -> None:
        state = _persisted_preflight_state()
        report = run_one_isolated_judge_preflight(
            job_id=DEFAULT_PREFLIGHT_JOB_ID,
            candidate_id=DEFAULT_PREFLIGHT_CANDIDATE_ID,
            tournament_state=state,
            llm_client=lambda **kwargs: _valid_judgment(DEFAULT_PREFLIGHT_CANDIDATE_ID, eligible=True),
        )
        self.assertTrue(report["judgeAccepted"])
        self.assertEqual(report["candidateSource"], "requested_persisted_candidate")
        self.assertEqual(report["judgeNormalCalls"], 1)
        self.assertEqual(report["judgeRepairCalls"], 0)

    def test_preflight_zero_strategy_creator_winner_runway_ffmpeg(self) -> None:
        state = _persisted_preflight_state()
        report = run_one_isolated_judge_preflight(
            job_id=DEFAULT_PREFLIGHT_JOB_ID,
            tournament_state=state,
            llm_client=lambda **kwargs: _valid_judgment(DEFAULT_PREFLIGHT_CANDIDATE_ID),
        )
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)
        self.assertEqual(report["winnerCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)

    def test_isolation_guard_blocks_side_paths(self) -> None:
        JudgePreflightIsolationGuard.strategy_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            JudgePreflightIsolationGuard.assert_safe_before_paid_call()
        self.assertIn(PREFLIGHT_ISOLATION_ERROR, str(ctx.exception))


class TestJudgePromptParity(unittest.TestCase):
    def test_prompt_mentions_conditional_verbal_contract(self) -> None:
        candidate = _candidate("summer_fan")
        candidate["verbalPotential"] = {"decision": "not_needed", "reason": "Visual proof only."}
        prompt = build_judge_prompt(
            product_name="Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(language="he"),
            prototype=require_prototype("summer_fan"),
            candidate=candidate,
            candidate_id="cand-1",
        )
        self.assertIn("not_needed", prompt)
        self.assertIn("verbalLayerAssessment", prompt)
        self.assertEqual(resolve_creator_verbal_decision(candidate), "not_needed")


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
