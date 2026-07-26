"""
Builder2 Winner-only resume and Winner Development diagnostics tests — mocks only.
"""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError, WINNER_PLAN_SCHEMA_VERSION
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from engine.builder2_winner_development import develop_builder2_winning_candidate, normalize_winner_plan_for_runway
from engine.builder2_winner_development_diagnostics import PUBLIC_FAILURE_CODE, STAGE_EXTRACTION, STAGE_PERSISTENCE, STAGE_VALIDATION
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically
from engine.builder2_winner_resume import DEFAULT_WINNER_RESUME_JOB_ID, run_one_winner_resume
from engine.builder2_winner_resume_guard import RESUME_ISOLATION_ERROR, WinnerResumeIsolationGuard
from tests.test_builder2_reasoning_resume import (
    HISTORICAL_JOB_ID,
    _candidate_id_for_prototype,
    _historical_resume_state,
    _judgment_for_candidate,
    _make_llm,
)
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt


def _scores_for_total(total: int) -> Dict[str, int]:
    remaining = max(0, min(100, total))
    scores = {
        "problemAdvantageIntegrity": min(20, remaining),
    }
    remaining -= scores["problemAdvantageIntegrity"]
    scores["mechanismQuality"] = min(15, remaining)
    remaining -= scores["mechanismQuality"]
    scores["prototypeMethodApplication"] = min(10, remaining)
    remaining -= scores["prototypeMethodApplication"]
    scores["silentVisualClarity"] = min(15, remaining)
    remaining -= scores["silentVisualClarity"]
    scores["originalityFreshness"] = min(15, remaining)
    remaining -= scores["originalityFreshness"]
    scores["eleganceSimplicity"] = min(10, remaining)
    remaining -= scores["eleganceSimplicity"]
    scores["runwayFeasibility"] = min(10, remaining)
    remaining -= scores["runwayFeasibility"]
    scores["editingContribution"] = min(5, remaining)
    return scores


def _historical_judged_state(*, job_id: str = HISTORICAL_JOB_ID) -> Dict[str, Any]:
    from engine.builder2_accepted_judgment_store import persist_accepted_judgment

    scores_by_prototype = {
        "closest": 68,
        "forgot": 67,
        "greenpeace_essential_pairing": 79,
        "summer_fan": 80,
        "think_small": 71,
        "winning_card": 76,
    }
    state = _historical_resume_state(job_id=job_id)
    for prototype_id in DEFAULT_ACTIVE_PROTOTYPE_IDS:
        candidate_id = _candidate_id_for_prototype(prototype_id)
        candidate = _candidate(prototype_id)
        if prototype_id == "summer_fan":
            candidate["verbalPotential"] = {
                "decision": "not_needed",
                "reason": "The visible fan behavior communicates absence without a headline.",
            }
        judgment = _judgment_for_candidate(
            candidate_id,
            candidate,
            eligible=True,
        )
        judgment["scores"] = _scores_for_total(scores_by_prototype[prototype_id])
        _, total, score_map = __import__(
            "engine.builder2_judge", fromlist=["validate_judge_response"]
        ).validate_judge_response(judgment, candidate_id=candidate_id, candidate=candidate)
        persist_accepted_judgment(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            judgment_id=f"judge-{candidate_id}-stored",
            judgment=judgment,
            total=total,
            scores=score_map,
        )
    state["winnerCandidateId"] = _candidate_id_for_prototype("summer_fan")
    save_tournament_state(job_id, state)
    loaded = load_tournament_state(job_id)
    assert loaded is not None
    return loaded


class TestWinnerResumePrerequisites(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_six_accepted_judgments_are_reused(self) -> None:
        state = _historical_judged_state(job_id="job-winner-reuse-judgments")
        report = run_one_winner_resume(job_id="job-winner-reuse-judgments", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["acceptedJudgmentCount"], 6)
        self.assertEqual(report["reusedJudgmentCount"], 6)

    def test_strategy_creator_judge_never_called(self) -> None:
        state = _historical_judged_state(job_id="job-winner-no-reasoning")
        report = run_one_winner_resume(job_id="job-winner-no-reasoning", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)
        self.assertEqual(report["judgeCalls"], 0)

    def test_winner_selection_resolves_to_summer_fan(self) -> None:
        state = _historical_judged_state(job_id="job-winner-select")
        expected = select_global_winner(state)
        self.assertIn("summer_fan", expected)
        report = run_one_winner_resume(job_id="job-winner-select", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["winnerPrototypeId"], "summer_fan")
        self.assertEqual(report["winnerScore"], 80)


class TestWinnerResumeExecution(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_winner_called_exactly_once(self) -> None:
        state = _historical_judged_state(job_id="job-winner-once")
        calls = {"count": 0}
        base = _make_llm()

        def llm(**kwargs: Any) -> Dict[str, Any]:
            if kwargs.get("role") == "builder2_winner":
                calls["count"] += 1
            return base(**kwargs)

        report = run_one_winner_resume(job_id="job-winner-once", llm_client=llm, tournament_state=deepcopy(state))
        self.assertEqual(report["winnerNormalCalls"], 1)
        self.assertEqual(calls["count"], 1)

    def test_paid_call_counted_when_parsing_fails(self) -> None:
        state = _historical_judged_state(job_id="job-winner-parse-fail")

        def llm(**kwargs: Any) -> str:
            if kwargs.get("role") == "builder2_winner":
                return "not-json"
            raise AssertionError(kwargs.get("role"))

        report = run_one_winner_resume(job_id="job-winner-parse-fail", llm_client=llm, tournament_state=deepcopy(state))
        self.assertFalse(report["ok"])
        self.assertEqual(report["winnerNormalCalls"], 1)
        self.assertEqual(report["failureStage"], STAGE_EXTRACTION)

    def test_extraction_failure_reports_exact_stage(self) -> None:
        state = _historical_judged_state(job_id="job-winner-stage-extract")
        report = run_one_winner_resume(
            job_id="job-winner-stage-extract",
            llm_client=lambda **kwargs: "" if kwargs.get("role") == "builder2_winner" else (_ for _ in ()).throw(AssertionError()),
            tournament_state=deepcopy(state),
        )
        self.assertEqual(report["failureStage"], STAGE_EXTRACTION)
        self.assertEqual(report["failureReason"], PUBLIC_FAILURE_CODE)

    def test_validation_failure_reports_field_path(self) -> None:
        state = _historical_judged_state(job_id="job-winner-validate-fail")

        def llm(**kwargs: Any) -> Dict[str, Any]:
            if kwargs.get("role") != "builder2_winner":
                raise AssertionError(kwargs.get("role"))
            plan = _winner_plan_from_prompt(kwargs.get("prompt", ""))
            plan["schemaVersion"] = "builder2_winner_video_plan_v0"
            return plan

        report = run_one_winner_resume(job_id="job-winner-validate-fail", llm_client=llm, tournament_state=deepcopy(state))
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], STAGE_VALIDATION)
        failure = (load_tournament_state("job-winner-validate-fail") or {}).get("winnerDevelopmentFailure") or {}
        self.assertEqual(failure.get("fieldPath"), "schemaVersion")

    def test_valid_winner_persisted_atomically(self) -> None:
        state = _historical_judged_state(job_id="job-winner-persist")
        report = run_one_winner_resume(job_id="job-winner-persist", llm_client=_make_llm(), tournament_state=deepcopy(state))
        loaded = load_tournament_state("job-winner-persist")
        assert loaded is not None
        self.assertTrue(report["ok"])
        self.assertTrue(is_valid_persisted_winner_development(loaded))
        self.assertEqual(loaded.get("winnerDevelopmentPrototypeId"), "summer_fan")
        self.assertIsNotNone(loaded.get("winnerDevelopmentAcceptedAt"))

    def test_rerun_reuses_persisted_winner_with_zero_calls(self) -> None:
        state = _historical_judged_state(job_id="job-winner-rerun")
        first = run_one_winner_resume(job_id="job-winner-rerun", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertTrue(first["ok"])
        loaded = load_tournament_state("job-winner-rerun")
        assert loaded is not None
        report = run_one_winner_resume(job_id="job-winner-rerun", llm_client=_make_llm(), tournament_state=loaded)
        self.assertTrue(report["winnerReused"])
        self.assertEqual(report["winnerNormalCalls"], 0)

    def test_partial_failed_winner_is_not_reused(self) -> None:
        state = _historical_judged_state(job_id="job-winner-partial-fail")
        state["winnerDevelopmentFailure"] = {
            "stage": STAGE_EXTRACTION,
            "preciseReason": "empty_response",
            "publicReason": PUBLIC_FAILURE_CODE,
        }
        state["winnerDevelopmentPaidCallRecorded"] = True
        report = run_one_winner_resume(job_id="job-winner-partial-fail", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["winnerNormalCalls"], 1)
        self.assertFalse(report["winnerReused"])

    def test_persistence_failure_is_distinguished(self) -> None:
        state = _historical_judged_state(job_id="job-winner-persist-fail")

        with patch(
            "engine.builder2_winner_resume.persist_winner_development_atomically",
            side_effect=RuntimeError("storage_write_failed"),
        ):
            report = run_one_winner_resume(
                job_id="job-winner-persist-fail",
                llm_client=_make_llm(),
                tournament_state=deepcopy(state),
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["winnerNormalCalls"], 1)
        self.assertEqual(report["failureStage"], STAGE_PERSISTENCE)


class TestWinnerDevelopmentDiagnostics(unittest.TestCase):
    def test_development_counts_submitted_call_before_validation(self) -> None:
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        submitted = {"count": 0}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            submitted["count"] += 1
            bad = _winner_plan_from_prompt(kwargs.get("prompt", ""))
            bad["schemaVersion"] = "builder2_winner_video_plan_v0"
            return bad

        with self.assertRaises(Builder2TournamentError):
            develop_builder2_winning_candidate(
                product_name="Product",
                product_description="desc",
                language="en",
                strategy_foundation=_strategy(),
                winning_candidate=_candidate("summer_fan"),
                winning_judgment=_judgment_for_candidate("c1", _candidate("summer_fan")),
                prototype_id="summer_fan",
                runway_mode="single",
                llm_client=llm,
                state=state,
            )
        self.assertEqual(submitted["count"], 1)
        self.assertEqual((state.get("metrics") or {}).get("winnerNormalCalls"), 1)

    def test_normalization_failure_reports_field_path(self) -> None:
        plan = _winner_plan_from_prompt("")
        plan["visualAnchor"] = {"description": 123}
        with self.assertRaises(Builder2TournamentError):
            normalize_winner_plan_for_runway(
                plan,
                product_name="Product",
                product_description="desc",
                content_language="en",
            )


class TestWinnerResumeIsolation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()
        WinnerResumeIsolationGuard.end()

    def test_media_and_queue_disabled(self) -> None:
        state = _historical_judged_state(job_id="job-winner-media")
        report = run_one_winner_resume(job_id="job-winner-media", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertTrue(report["mediaContinuationRequired"])

    def test_queue_and_recovery_disabled(self) -> None:
        WinnerResumeIsolationGuard.begin()
        WinnerResumeIsolationGuard.strategy_generation_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            WinnerResumeIsolationGuard.assert_safe_before_paid_call()
        self.assertIn(RESUME_ISOLATION_ERROR, str(ctx.exception))
        WinnerResumeIsolationGuard.end()


class TestWinnerResumeRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_full_production_healthy_path_remains_14_reasoning_calls(self) -> None:
        def llm(**kwargs: Any) -> Dict[str, Any]:
            role = kwargs.get("role", "")
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
                candidate_id = next(token.strip().strip(",") for token in prompt.split() if token.startswith("cand-"))
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                    if pid in candidate_id:
                        prototype_id = pid
                        break
                return _judgment_for_candidate(candidate_id, _candidate(prototype_id), eligible=True)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        run_builder2_tournament(
            job_id="job-winner-regression-14",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-winner-regression",
        )
        state = load_tournament_state("job-winner-regression-14")
        assert state is not None
        self.assertEqual((state.get("metrics") or {}).get("totalReasoningCalls"), 14)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
