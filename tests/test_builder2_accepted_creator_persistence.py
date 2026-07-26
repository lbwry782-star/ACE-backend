"""
Builder2 accepted Creator persistence and preflight tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_accepted_creator_store import (
    backfill_accepted_creator_index,
    load_accepted_creator_candidate,
    persist_accepted_creator_candidate,
    update_candidate_judge_state,
)
from engine.builder2_judge_preflight import (
    DEFAULT_PREFLIGHT_CANDIDATE_ID,
    DEFAULT_PREFLIGHT_JOB_ID,
    run_one_isolated_judge_preflight,
)
from engine.builder2_reasoning_pipeline_preflight import run_one_isolated_reasoning_pipeline_preflight
from engine.builder2_reasoning_pipeline_preflight_guard import (
    PREFLIGHT_ISOLATION_ERROR,
    PipelinePreflightIsolationGuard,
)
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    new_tournament_state,
    save_tournament_state,
)
from tests.test_builder2_judge_contract import _valid_judgment
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt


HISTORICAL_JOB_ID = DEFAULT_PREFLIGHT_JOB_ID
HISTORICAL_CANDIDATE_ID = DEFAULT_PREFLIGHT_CANDIDATE_ID


def _persisted_job_with_judge_unavailable_candidate() -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    state = new_tournament_state(
        job_id=HISTORICAL_JOB_ID,
        language="he",
        active_prototype_ids=list(DEFAULT_ACTIVE_PROTOTYPE_IDS),
        random_seed="persist-test",
    )
    state["strategyFoundation"] = strategy
    state["productName"] = "Product"
    state["productDescription"] = "desc"
    state["candidates"][HISTORICAL_CANDIDATE_ID] = {
        "candidateId": HISTORICAL_CANDIDATE_ID,
        "prototypeId": "summer_fan",
        "roundIndex": 1,
        "attemptNumber": 1,
        "creatorOutput": deepcopy(candidate),
        "creatorAcceptanceStatus": "accepted",
        "validationStatus": "judge_unavailable",
        "judgeStatus": "unavailable",
        "status": "judge_unavailable",
        "creatorCandidateValid": True,
        "judgmentId": None,
        "eligible": False,
        "totalScore": None,
        "tieScores": {},
        "failureReason": "builder2_judge_validation_failed:verbalLayerAssessment",
        "completedAt": "2026-01-01T00:00:00+00:00",
    }
    backfill_accepted_creator_index(state)
    return state


class TestAcceptedCreatorPersistence(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_accepted_creator_persisted_before_judge(self) -> None:
        state = new_tournament_state(
            job_id="job-persist-order",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["strategyFoundation"] = _strategy()
        candidate = _candidate("closest")
        persist_accepted_creator_candidate(
            state,
            candidate_id="cand-1",
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            creator_output=candidate,
            strategy_foundation=state["strategyFoundation"],
        )
        save_tournament_state("job-persist-order", state)
        loaded = load_tournament_state("job-persist-order")
        assert loaded is not None
        self.assertIn("cand-1", loaded.get("acceptedCreatorCandidates") or {})
        snapshot = load_accepted_creator_candidate(job_id="job-persist-order", candidate_id="cand-1", tournament_state=loaded)
        self.assertEqual(snapshot["candidateId"], "cand-1")
        self.assertIsInstance(snapshot["creatorOutput"], dict)

    def test_judge_rejection_does_not_delete_creator_snapshot(self) -> None:
        state = _persisted_job_with_judge_unavailable_candidate()
        original = deepcopy(state["candidates"][HISTORICAL_CANDIDATE_ID]["creatorOutput"])
        update_candidate_judge_state(
            state,
            candidate_id=HISTORICAL_CANDIDATE_ID,
            judge_status="unavailable",
            failure_reason="builder2_judge_validation_failed:verbalLayerAssessment",
        )
        save_tournament_state(HISTORICAL_JOB_ID, state)
        snapshot = load_accepted_creator_candidate(
            job_id=HISTORICAL_JOB_ID,
            candidate_id=HISTORICAL_CANDIDATE_ID,
            tournament_state=state,
        )
        self.assertEqual(snapshot["creatorOutput"]["prototypeId"], original["prototypeId"])
        self.assertEqual(state["acceptedCreatorCandidates"][HISTORICAL_CANDIDATE_ID]["creatorOutput"]["prototypeId"], original["prototypeId"])

    def test_tournament_failure_preserves_accepted_candidates(self) -> None:
        state = _persisted_job_with_judge_unavailable_candidate()
        state["status"] = "failed"
        state["error"] = "builder2_tournament_no_valid_candidate"
        save_tournament_state(HISTORICAL_JOB_ID, state)
        loaded = load_tournament_state(HISTORICAL_JOB_ID)
        assert loaded is not None
        self.assertIn(HISTORICAL_CANDIDATE_ID, loaded.get("acceptedCreatorCandidates") or {})

    def test_restart_preserves_accepted_candidates(self) -> None:
        state = _persisted_job_with_judge_unavailable_candidate()
        save_tournament_state(HISTORICAL_JOB_ID, state)
        loaded = load_tournament_state(HISTORICAL_JOB_ID)
        assert loaded is not None
        self.assertIn(HISTORICAL_CANDIDATE_ID, loaded.get("acceptedCreatorCandidates") or {})

    def test_recovery_preserves_accepted_candidates(self) -> None:
        from engine.builder2_tournament_recovery import disable_memory_recovery, enable_memory_recovery, register_recoverable_job

        state = _persisted_job_with_judge_unavailable_candidate()
        save_tournament_state(HISTORICAL_JOB_ID, state)
        enable_memory_recovery()
        try:
            register_recoverable_job(HISTORICAL_JOB_ID)
        finally:
            disable_memory_recovery()
        reloaded = load_tournament_state(HISTORICAL_JOB_ID)
        assert reloaded is not None
        snapshot = load_accepted_creator_candidate(
            job_id=HISTORICAL_JOB_ID,
            candidate_id=HISTORICAL_CANDIDATE_ID,
            tournament_state=reloaded,
        )
        self.assertEqual(snapshot["candidateId"], HISTORICAL_CANDIDATE_ID)

        state = _persisted_job_with_judge_unavailable_candidate()
        snapshot = load_accepted_creator_candidate(
            job_id=HISTORICAL_JOB_ID,
            candidate_id=HISTORICAL_CANDIDATE_ID,
            tournament_state=state,
        )
        self.assertEqual(snapshot["candidateId"], HISTORICAL_CANDIDATE_ID)

    def test_missing_candidate_fails_with_zero_model_calls(self) -> None:
        state = new_tournament_state(
            job_id="job-missing",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["strategyFoundation"] = _strategy()
        save_tournament_state("job-missing", state)
        calls: List[str] = []

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            raise AssertionError("no model calls expected")

        report = run_one_isolated_judge_preflight(
            job_id="job-missing",
            candidate_id="cand-missing",
            tournament_state=load_tournament_state("job-missing"),
            llm_client=llm,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(calls, [])
        self.assertEqual(report["judgeNormalCalls"], 0)

    def test_requested_candidate_id_reported_correctly_when_missing(self) -> None:
        state = new_tournament_state(
            job_id="job-report",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["strategyFoundation"] = _strategy()
        report = run_one_isolated_judge_preflight(
            job_id="job-report",
            candidate_id="cand-explicit",
            tournament_state=state,
            llm_client=lambda **kwargs: _valid_judgment("cand-explicit"),
        )
        self.assertEqual(report["requestedCandidateId"], "cand-explicit")
        self.assertIsNone(report["resolvedCandidateId"])
        self.assertEqual(report["candidateSource"], "missing")

    def test_alternate_candidate_only_when_explicitly_enabled(self) -> None:
        state = _persisted_job_with_judge_unavailable_candidate()
        report = run_one_isolated_judge_preflight(
            job_id=HISTORICAL_JOB_ID,
            candidate_id="cand-other",
            allow_alternate=True,
            tournament_state=state,
            llm_client=lambda **kwargs: _valid_judgment(HISTORICAL_CANDIDATE_ID),
        )
        self.assertEqual(report["requestedCandidateId"], "cand-other")
        self.assertEqual(report["resolvedCandidateId"], HISTORICAL_CANDIDATE_ID)
        self.assertEqual(report["candidateSource"], "alternate_persisted_accepted_candidate")

    def test_judge_only_preflight_never_creates_creator(self) -> None:
        calls: List[str] = []

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            if kwargs.get("role") == "builder2_judge":
                return _valid_judgment(HISTORICAL_CANDIDATE_ID)
            raise AssertionError(kwargs.get("role"))

        state = _persisted_job_with_judge_unavailable_candidate()
        report = run_one_isolated_judge_preflight(
            job_id=HISTORICAL_JOB_ID,
            candidate_id=HISTORICAL_CANDIDATE_ID,
            tournament_state=state,
            llm_client=llm,
        )
        self.assertTrue(report["judgeAccepted"])
        self.assertEqual(report["creatorCalls"], 0)
        self.assertNotIn("builder2_creator", calls)


class PipelineMockLLM:
    calls: List[str] = []

    def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
        PipelineMockLLM.calls.append(role)
        if role == "builder2_strategy":
            return _strategy(language="he")
        if role == "builder2_creator":
            cand = _candidate("think_small")
            marker = "strategyFoundationId (return exactly): "
            if marker in prompt:
                start = prompt.index(marker) + len(marker)
                cand["strategyFoundationId"] = prompt[start:].split("\n", 1)[0].strip().strip('"').strip("'")
            return cand
        if role == "builder2_judge":
            return _valid_judgment("cand-pipeline-think_small-1", eligible=False)
        if role == "builder2_winner":
            raise AssertionError("winner must not run in pipeline preflight")
        raise AssertionError(role)


class TestReasoningPipelinePreflight(unittest.TestCase):
    def setUp(self) -> None:
        PipelineMockLLM.calls = []
        PipelinePreflightIsolationGuard.begin()

    def tearDown(self) -> None:
        PipelinePreflightIsolationGuard.end()

    def test_pipeline_runs_one_strategy_creator_judge(self) -> None:
        report = run_one_isolated_reasoning_pipeline_preflight(
            product_name="Product",
            product_description="desc",
            content_language="he",
            prototype_id="think_small",
            llm_client=PipelineMockLLM(),
        )
        self.assertTrue(report["strategyAccepted"])
        self.assertTrue(report["creatorAccepted"])
        self.assertTrue(report["creatorPersisted"])
        self.assertTrue(report["judgeAccepted"])
        self.assertFalse(report["judgeEligible"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["strategyCalls"], 1)
        self.assertEqual(report["creatorNormalCalls"], 1)
        self.assertEqual(report["judgeNormalCalls"], 1)
        self.assertEqual(
            report["strategyCalls"] + report["creatorNormalCalls"] + report["judgeNormalCalls"],
            3,
        )

    def test_pipeline_never_runs_media_or_winner(self) -> None:
        report = run_one_isolated_reasoning_pipeline_preflight(
            product_name="Product",
            product_description="desc",
            content_language="he",
            llm_client=PipelineMockLLM(),
        )
        self.assertEqual(report["winnerCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertNotIn("builder2_winner", PipelineMockLLM.calls)

    def test_pipeline_isolation_guard(self) -> None:
        PipelinePreflightIsolationGuard.runway_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            PipelinePreflightIsolationGuard.assert_safe_before_paid_call()
        self.assertIn(PREFLIGHT_ISOLATION_ERROR, str(ctx.exception))

    def test_valid_negative_judge_is_accepted(self) -> None:
        report = run_one_isolated_reasoning_pipeline_preflight(
            product_name="Product",
            product_description="desc",
            content_language="he",
            llm_client=PipelineMockLLM(),
        )
        self.assertTrue(report["judgeAccepted"])
        self.assertFalse(report["judgeEligible"])
        self.assertTrue(report["ok"])


class TestProductionPathPreservation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    def test_full_production_happy_path_remains_fourteen_calls(self) -> None:
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

        run_builder2_tournament(
            job_id="job-14-persist",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-14-persist",
        )
        state = load_tournament_state("job-14-persist")
        assert state is not None
        self.assertEqual((state.get("metrics") or {}).get("totalReasoningCalls"), 14)
        accepted_index = state.get("acceptedCreatorCandidates") or {}
        self.assertEqual(len(accepted_index), 6)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
