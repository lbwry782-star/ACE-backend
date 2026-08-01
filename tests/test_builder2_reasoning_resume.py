"""
Builder2 reasoning-only resume tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_accepted_creator_store import load_accepted_creator_candidate
from engine.builder2_accepted_judgment_store import (
    audit_reusable_accepted_judgment,
    persist_accepted_judgment,
)
from engine.builder2_judge_circuit_breaker import JUDGE_BREAKER_CONTRACT_VERSION, SYSTEMIC_FAILURE_CODE
from engine.builder2_judge_preflight import DEFAULT_PREFLIGHT_CANDIDATE_ID, DEFAULT_PREFLIGHT_JOB_ID
from engine.builder2_reasoning_resume import (
    DEFAULT_RESUME_JOB_ID,
    run_one_reasoning_resume,
    validate_reasoning_resume_state,
)
from engine.builder2_reasoning_resume_guard import (
    RESUME_ISOLATION_ERROR,
    ReasoningResumeIsolationGuard,
)
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    new_tournament_state,
    save_tournament_state,
)
from tests.test_builder2_judge_contract import _negative_verbal_judgment, _valid_judgment
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt


HISTORICAL_JOB_ID = DEFAULT_RESUME_JOB_ID


def _judgment_for_candidate(
    candidate_id: str,
    candidate: Dict[str, Any],
    *,
    eligible: bool = True,
    score_boost: int = 0,
    schema_version: str | None = None,
) -> Dict[str, Any]:
    judgment = _valid_judgment(candidate_id, eligible=eligible)
    verbal = candidate.get("verbalPotential") if isinstance(candidate.get("verbalPotential"), dict) else {}
    decision = str(verbal.get("decision") or "available")
    if decision == "not_needed":
        judgment["verbalLayerAssessment"] = {
            "applicability": "not_needed",
            "keywordBornFromVisual": None,
            "visualMeaningIsClear": None,
            "strategicMeaningIsClear": None,
            "twoMeaningsReinforceEachOther": None,
            "notes": "Verbal analysis is not applicable because the visual proof stands alone.",
        }
    elif decision == "not_found":
        judgment["verbalLayerAssessment"] = {
            "applicability": "not_found",
            "keywordBornFromVisual": None,
            "visualMeaningIsClear": None,
            "strategicMeaningIsClear": None,
            "twoMeaningsReinforceEachOther": None,
            "notes": "Creator found no defensible keyword; absence is a weakness.",
        }
        if not eligible:
            judgment["disqualifiers"] = ["no_defensible_keyword"]
    if score_boost:
        judgment["scores"]["silentVisualClarity"] = score_boost
    if not eligible and not judgment.get("disqualifiers"):
        judgment["disqualifiers"] = ["ineligible_without_reason"]
        judgment["weaknesses"] = ["Mechanism does not clear the eligible bar."]
    if schema_version is not None:
        judgment["schemaVersion"] = schema_version
    return judgment


def _candidate_id_for_prototype(prototype_id: str) -> str:
    return f"cand-1-{prototype_id}-1-resume"


def _historical_resume_state(
    *,
    job_id: str = HISTORICAL_JOB_ID,
    with_reusable_judgments: int = 0,
    with_ineligible_judgment: bool = False,
    with_invalid_judgment: bool = False,
    with_winner_plan: bool = False,
    interrupted_after: int = 0,
) -> Dict[str, Any]:
    from engine.builder2_accepted_creator_store import persist_accepted_creator_candidate

    strategy = _strategy(language="he")
    state = new_tournament_state(
        job_id=job_id,
        language="he",
        active_prototype_ids=list(DEFAULT_ACTIVE_PROTOTYPE_IDS),
        random_seed="reasoning-resume-seed",
    )
    state["productName"] = "Resume Product"
    state["productDescription"] = "Resume description"
    state["contentLanguage"] = "he"
    state["strategyFoundation"] = strategy
    # Historical resume fixtures represent jobs created before single-slogan contract.
    state.pop("copyContractVersion", None)
    state.pop("builder2NewFormatVersion", None)
    state["currentRound"] = 1
    state["status"] = "round_complete"

    for index, prototype_id in enumerate(DEFAULT_ACTIVE_PROTOTYPE_IDS):
        candidate_id = _candidate_id_for_prototype(prototype_id)
        candidate = _candidate(prototype_id)
        if prototype_id == "summer_fan":
            candidate["verbalPotential"] = {
                "decision": "not_needed",
                "reason": "The visible fan behavior communicates absence without a headline.",
            }
        persist_accepted_creator_candidate(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            round_index=1,
            attempt_number=1,
            creator_output=candidate,
            strategy_foundation=strategy,
        )
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": 1,
            "attemptNumber": 1,
            "creatorOutput": deepcopy(candidate),
            "creatorSnapshot": deepcopy(candidate),
            "creatorAcceptanceStatus": "accepted",
            "validationStatus": "accepted",
            "judgeStatus": "pending",
            "status": "accepted",
            "judgmentId": None,
            "eligible": False,
            "totalScore": None,
            "tieScores": {},
            "completedAt": "2026-01-01T00:00:00+00:00",
        }

        should_persist = index < with_reusable_judgments or (interrupted_after and index < interrupted_after)
        if should_persist:
            eligible = not with_ineligible_judgment or index > 0
            if with_invalid_judgment and index == 0:
                judgment = _judgment_for_candidate(
                    candidate_id,
                    candidate,
                    eligible=True,
                    schema_version="builder2_judgment_v0",
                )
                state["judgments"][f"judge-{candidate_id}-stored"] = {
                    "judgmentId": f"judge-{candidate_id}-stored",
                    "candidateId": candidate_id,
                    "judgment": judgment,
                    "totalScore": 80,
                    "scores": judgment["scores"],
                    "eligible": True,
                    "completedAt": "2026-01-01T00:00:00+00:00",
                }
                cand = state["candidates"][candidate_id]
                cand["judgeStatus"] = "accepted"
                cand["judgmentId"] = f"judge-{candidate_id}-stored"
                cand["judgmentSnapshot"] = deepcopy(judgment)
                cand["eligible"] = True
                cand["totalScore"] = 80
                cand["tieScores"] = deepcopy(judgment["scores"])
                continue
            if with_ineligible_judgment and index == 0:
                judgment = _negative_verbal_judgment(candidate_id, eligible=False)
                if prototype_id == "summer_fan":
                    judgment["verbalLayerAssessment"] = {
                        "applicability": "not_needed",
                        "keywordBornFromVisual": None,
                        "visualMeaningIsClear": None,
                        "strategicMeaningIsClear": None,
                        "twoMeaningsReinforceEachOther": None,
                        "notes": "Verbal layer is weak for this silent proof.",
                    }
            else:
                judgment = _judgment_for_candidate(
                    candidate_id,
                    candidate,
                    eligible=eligible,
                    score_boost=10 + index,
                )
            _, total, scores = __import__(
                "engine.builder2_judge", fromlist=["validate_judge_response"]
            ).validate_judge_response(judgment, candidate_id=candidate_id, candidate=candidate)
            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=f"judge-{candidate_id}-stored",
                judgment=judgment,
                total=total,
                scores=scores,
            )

    if with_winner_plan:
        for prototype_id in DEFAULT_ACTIVE_PROTOTYPE_IDS:
            candidate_id = _candidate_id_for_prototype(prototype_id)
            if (state.get("acceptedJudgments") or {}).get(candidate_id):
                continue
            candidate = _candidate(prototype_id)
            judgment = _judgment_for_candidate(candidate_id, candidate, eligible=True)
            _, total, scores = __import__(
                "engine.builder2_judge", fromlist=["validate_judge_response"]
            ).validate_judge_response(judgment, candidate_id=candidate_id, candidate=candidate)
            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=f"judge-{candidate_id}-stored",
                judgment=judgment,
                total=total,
                scores=scores,
            )
        winner_id = _candidate_id_for_prototype("closest")
        state["winnerCandidateId"] = winner_id
        state["winnerDevelopmentPlan"] = _winner_plan_from_prompt("")

    save_tournament_state(job_id, state)
    loaded = load_tournament_state(job_id)
    assert loaded is not None
    return loaded


def _make_llm(*, eligible_all: bool = True, winner_only: bool = False):
    def llm(**kwargs: Any) -> Dict[str, Any]:
        role = kwargs.get("role", "")
        prompt = kwargs.get("prompt", "")
        if role == "builder2_judge":
            candidate_id = "unknown"
            for token in prompt.split():
                if token.startswith("cand-"):
                    candidate_id = token.strip().strip(",")
                    break
            prototype_id = "closest"
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if pid in candidate_id:
                    prototype_id = pid
                    break
            candidate = _candidate(prototype_id)
            if prototype_id == "summer_fan":
                candidate["verbalPotential"] = {
                    "decision": "not_needed",
                    "reason": "The visible fan behavior communicates absence without a headline.",
                }
            eligible = eligible_all
            if not eligible_all and prototype_id == DEFAULT_ACTIVE_PROTOTYPE_IDS[0]:
                eligible = False
            return _judgment_for_candidate(candidate_id, candidate, eligible=eligible)
        if role == "builder2_winner":
            return _winner_plan_from_prompt(prompt)
        if winner_only:
            raise AssertionError(f"unexpected role during resume: {role}")
        raise AssertionError(role)

    return llm


class TestReasoningResumeValidation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_existing_strategy_loaded_not_regenerated(self) -> None:
        state = _historical_resume_state(job_id="job-resume-strategy")
        llm = _make_llm()
        report = run_one_reasoning_resume(job_id="job-resume-strategy", llm_client=llm, tournament_state=deepcopy(state))
        self.assertTrue(report["strategyLoaded"])
        self.assertEqual(report["strategyCalls"], 0)
        loaded = load_tournament_state("job-resume-strategy")
        assert loaded is not None
        self.assertIsInstance(loaded.get("strategyFoundation"), dict)

    def test_six_accepted_creator_snapshots_loaded(self) -> None:
        state = _historical_resume_state(job_id="job-resume-six")
        valid, missing = validate_reasoning_resume_state(state)
        self.assertTrue(valid, missing)
        self.assertEqual(len(state.get("acceptedCreatorCandidates") or {}), 6)

    def test_missing_creator_state_fails_before_paid_calls(self) -> None:
        state = new_tournament_state(
            job_id="job-resume-missing",
            language="he",
            active_prototype_ids=list(DEFAULT_ACTIVE_PROTOTYPE_IDS),
            random_seed="seed",
        )
        state["strategyFoundation"] = _strategy(language="he")
        llm = _make_llm()
        report = run_one_reasoning_resume(job_id="job-resume-missing", llm_client=llm, tournament_state=state)
        self.assertFalse(report["ok"])
        self.assertEqual(report["judgeNormalCalls"], 0)
        self.assertEqual(report["strategyCalls"], 0)


class TestReasoningResumeJudgments(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_accepted_judgments_are_reused(self) -> None:
        state = _historical_resume_state(job_id="job-resume-reuse", with_reusable_judgments=2)
        report = run_one_reasoning_resume(
            job_id="job-resume-reuse",
            llm_client=_make_llm(),
            tournament_state=deepcopy(state),
        )
        self.assertEqual(report["reusedJudgmentCount"], 2)
        self.assertEqual(report["judgeNormalCalls"], 4)

    def test_invalid_historical_judgments_not_reused(self) -> None:
        state = _historical_resume_state(job_id="job-resume-invalid-judgment", with_reusable_judgments=1, with_invalid_judgment=True)
        report = run_one_reasoning_resume(
            job_id="job-resume-invalid-judgment",
            llm_client=_make_llm(),
            tournament_state=deepcopy(state),
        )
        self.assertEqual(report["reusedJudgmentCount"], 0)
        self.assertEqual(report["judgeNormalCalls"], 6)

    def test_interrupted_resume_skips_already_accepted_judgments(self) -> None:
        state = _historical_resume_state(job_id="job-resume-interrupted", interrupted_after=3)
        report = run_one_reasoning_resume(
            job_id="job-resume-interrupted",
            llm_client=_make_llm(),
            tournament_state=deepcopy(state),
        )
        self.assertEqual(report["reusedJudgmentCount"], 3)
        self.assertEqual(report["judgeNormalCalls"], 3)

    def test_each_new_judgment_persisted_immediately(self) -> None:
        state = _historical_resume_state(job_id="job-resume-immediate")
        persisted_counts: List[int] = []
        base_llm = _make_llm()

        def llm(**kwargs: Any) -> Dict[str, Any]:
            role = kwargs.get("role", "")
            if role == "builder2_judge":
                loaded = load_tournament_state("job-resume-immediate")
                assert loaded is not None
                persisted_counts.append(len(loaded.get("acceptedJudgments") or {}))
            return base_llm(**kwargs)

        run_one_reasoning_resume(job_id="job-resume-immediate", llm_client=llm, tournament_state=deepcopy(state))
        self.assertEqual(persisted_counts, [0, 1, 2, 3, 4, 5])

    def test_valid_eligible_false_judgment_stored(self) -> None:
        state = _historical_resume_state(job_id="job-resume-ineligible-one", with_reusable_judgments=1, with_ineligible_judgment=True)
        report = run_one_reasoning_resume(
            job_id="job-resume-ineligible-one",
            llm_client=_make_llm(eligible_all=True),
            tournament_state=deepcopy(state),
        )
        loaded = load_tournament_state("job-resume-ineligible-one")
        assert loaded is not None
        ineligible_id = _candidate_id_for_prototype(DEFAULT_ACTIVE_PROTOTYPE_IDS[0])
        self.assertFalse(loaded["candidates"][ineligible_id]["eligible"])
        self.assertEqual(loaded["candidates"][ineligible_id]["judgeStatus"], "accepted")
        self.assertTrue(report["ok"])


class TestReasoningResumeWinner(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_all_six_ineligible_produces_no_eligible_candidate(self) -> None:
        state = _historical_resume_state(job_id="job-resume-all-ineligible")
        for prototype_id in DEFAULT_ACTIVE_PROTOTYPE_IDS:
            candidate_id = _candidate_id_for_prototype(prototype_id)
            candidate = _candidate(prototype_id)
            if prototype_id == "summer_fan":
                candidate["verbalPotential"] = {
                    "decision": "not_needed",
                    "reason": "The visible fan behavior communicates absence without a headline.",
                }
            judgment = _judgment_for_candidate(candidate_id, candidate, eligible=False)
            _, total, scores = __import__(
                "engine.builder2_judge", fromlist=["validate_judge_response"]
            ).validate_judge_response(judgment, candidate_id=candidate_id, candidate=candidate)
            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=f"judge-{candidate_id}-ineligible",
                judgment=judgment,
                total=total,
                scores=scores,
            )
        save_tournament_state("job-resume-all-ineligible", state)
        report = run_one_reasoning_resume(
            job_id="job-resume-all-ineligible",
            llm_client=_make_llm(),
            tournament_state=deepcopy(state),
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureReason"], "builder2_tournament_no_eligible_candidate")
        self.assertEqual(report["winnerCalls"], 0)

    def test_winner_selection_uses_production_ranking(self) -> None:
        state = _historical_resume_state(job_id="job-resume-ranking")
        for index, prototype_id in enumerate(DEFAULT_ACTIVE_PROTOTYPE_IDS):
            candidate_id = _candidate_id_for_prototype(prototype_id)
            candidate = _candidate(prototype_id)
            if prototype_id == "summer_fan":
                candidate["verbalPotential"] = {
                    "decision": "not_needed",
                    "reason": "The visible fan behavior communicates absence without a headline.",
                }
            judgment = _judgment_for_candidate(candidate_id, candidate, eligible=True, score_boost=5 + index)
            _, total, scores = __import__(
                "engine.builder2_judge", fromlist=["validate_judge_response"]
            ).validate_judge_response(judgment, candidate_id=candidate_id, candidate=candidate)
            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=f"judge-{candidate_id}-rank",
                judgment=judgment,
                total=total,
                scores=scores,
            )
        save_tournament_state("job-resume-ranking", state)
        expected = select_global_winner(state)
        report = run_one_reasoning_resume(
            job_id="job-resume-ranking",
            llm_client=_make_llm(winner_only=True),
            tournament_state=deepcopy(state),
        )
        loaded = load_tournament_state("job-resume-ranking")
        assert loaded is not None
        self.assertEqual(loaded.get("winnerCandidateId"), expected)
        self.assertTrue(report["winnerSelected"])

    def test_winner_development_runs_exactly_once(self) -> None:
        state = _historical_resume_state(job_id="job-resume-winner-once")
        winner_calls = {"count": 0}
        base_llm = _make_llm()

        def llm(**kwargs: Any) -> Dict[str, Any]:
            if kwargs.get("role") == "builder2_winner":
                winner_calls["count"] += 1
            return base_llm(**kwargs)

        report = run_one_reasoning_resume(job_id="job-resume-winner-once", llm_client=llm, tournament_state=deepcopy(state))
        self.assertEqual(report["winnerCalls"], 1)
        self.assertEqual(winner_calls["count"], 1)

    def test_rerun_does_not_repeat_winner_development(self) -> None:
        state = _historical_resume_state(job_id="job-resume-winner-rerun", with_winner_plan=True)
        report = run_one_reasoning_resume(job_id="job-resume-winner-rerun", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["winnerCalls"], 0)
        self.assertTrue(report["ok"])


class TestReasoningResumeIsolation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()
        ReasoningResumeIsolationGuard.end()

    def test_strategy_and_creator_call_counts_zero(self) -> None:
        state = _historical_resume_state(job_id="job-resume-counts")
        report = run_one_reasoning_resume(job_id="job-resume-counts", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)

    def test_judge_normal_calls_at_most_six(self) -> None:
        state = _historical_resume_state(job_id="job-resume-judge-max")
        report = run_one_reasoning_resume(job_id="job-resume-judge-max", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertLessEqual(report["judgeNormalCalls"], 6)

    def test_healthy_path_has_zero_judge_repairs(self) -> None:
        state = _historical_resume_state(job_id="job-resume-no-repair")
        report = run_one_reasoning_resume(job_id="job-resume-no-repair", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["judgeRepairCalls"], 0)

    def test_media_call_counts_zero(self) -> None:
        state = _historical_resume_state(job_id="job-resume-media")
        report = run_one_reasoning_resume(job_id="job-resume-media", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertTrue(report["mediaContinuationRequired"])

    def test_queue_and_recovery_disabled(self) -> None:
        ReasoningResumeIsolationGuard.begin()
        ReasoningResumeIsolationGuard.ordinary_queue_enabled = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            ReasoningResumeIsolationGuard.assert_safe_before_judge()
        self.assertIn(RESUME_ISOLATION_ERROR, str(ctx.exception))

    def test_systemic_judge_circuit_breaker_stops_remaining_calls(self) -> None:
        state = _historical_resume_state(job_id="job-resume-breaker")
        state["judgeContractCircuitBreaker"] = {
            "contractVersion": JUDGE_BREAKER_CONTRACT_VERSION,
            "currentContractTripped": True,
            "currentTrippedReason": "contract_failure",
            "currentRepeatedFieldPaths": ["verbalLayerAssessment"],
            "postRepairFailures": [],
            "currentCandidateFailurePaths": {},
        }
        report = run_one_reasoning_resume(job_id="job-resume-breaker", llm_client=_make_llm(), tournament_state=state)
        self.assertFalse(report["ok"])
        self.assertIn(SYSTEMIC_FAILURE_CODE, str(report.get("failureReason")))
        self.assertEqual(report["judgeNormalCalls"], 0)


class TestReasoningResumeRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_full_normal_production_regression_remains_14_reasoning_calls(self) -> None:
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
                candidate = _candidate(prototype_id)
                return _judgment_for_candidate(candidate_id, candidate, eligible=True)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        run_builder2_tournament(
            job_id="job-resume-regression-14",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-resume-regression",
        )
        state = load_tournament_state("job-resume-regression-14")
        assert state is not None
        self.assertEqual((state.get("metrics") or {}).get("totalReasoningCalls"), 14)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestJudgmentReuseAudit(unittest.TestCase):
    def test_judge_unavailable_not_reusable(self) -> None:
        from tests.test_builder2_accepted_creator_persistence import _persisted_job_with_judge_unavailable_candidate

        state = _persisted_job_with_judge_unavailable_candidate()
        snapshot = load_accepted_creator_candidate(
            job_id=DEFAULT_PREFLIGHT_JOB_ID,
            candidate_id=DEFAULT_PREFLIGHT_CANDIDATE_ID,
            tournament_state=state,
        )
        reusable, reason = audit_reusable_accepted_judgment(
            state,
            candidate_id=DEFAULT_PREFLIGHT_CANDIDATE_ID,
            creator_snapshot=snapshot,
            strategy_foundation=state["strategyFoundation"],
        )
        self.assertFalse(reusable)
        self.assertEqual(reason, "judgeStatus_not_accepted")


if __name__ == "__main__":
    unittest.main()
