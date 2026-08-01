"""
Builder2 resume-plan parity and judge-only recovery after offline Creator recovery.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_creator_recovery import run_offline_creator_recovery_batch
from engine.builder2_complete_ad_reasoning_resume import (
    DEFAULT_MAX_CALLS,
    run_controlled_complete_ad_reasoning_resume,
    validate_controlled_complete_ad_preconditions,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import (
    RESUME_STAGE_JUDGE_GENERATION,
    evaluate_complete_ad_reasoning_executor_preconditions,
    resolve_complete_ad_canonical_resume_plan,
)
from engine.builder2_judge_only_resume import DEFAULT_JUDGE_ONLY_MAX_CALLS, run_judge_only_reasoning_resume
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION, NORMAL_REASONING_CALL_BUDGET
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_completion_gate import accepted_judgment_count
from engine.builder2_tournament_config import resolve_builder2_tournament_max_rounds
from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    save_tournament_state,
)
from tests.builder2_methodology_fixtures import logo_policy_creator_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_creator_metaphor_regression import _production_shaped_metaphor_extras
from tests.test_builder2_creator_slogan_repair import _candidate_with_slogan
from tests.test_builder2_tournament import _judgment


_PROTOTYPES = [
    "winning_card",
    "summer_fan",
    "forgot",
    "closest",
    "think_small",
    "greenpeace_essential_pairing",
]
_RELATIVE_ADVANTAGE = "קרבה אישית שמבינה את הלקוח"
_FAILURE_REASON = (
    "builder2_creator_literal_execution_without_transformation:"
    "metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed"
)


def _six_rejected_pre_recovery_state(*, status: str = "error") -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "jobId": "job-recovered-judge-resume",
        "tournamentId": "tournament-recovered-judge",
        "builder2ResumeContractVersion": BUILDER2_RESUME_CONTRACT_VERSION,
        "builder2NewFormatVersion": BUILDER2_NEW_FORMAT_VERSION,
        "schemaVersion": TOURNAMENT_STATE_SCHEMA_VERSION,
        "status": status,
        "failureReason": _FAILURE_REASON,
        "failureStage": "creator_generation",
        "productDescription": "Recovered product",
        "contentLanguage": "he",
        "initialActivePrototypeIds": list(_PROTOTYPES),
        "activePrototypeIds": list(_PROTOTYPES),
        "strategyFoundation": {
            "productNameResolved": "אורי לב",
            "relativeAdvantage": {"statement": _RELATIVE_ADVANTAGE},
            "language": "he",
        },
        "acceptedCreatorCandidates": {},
        "acceptedJudgments": {},
        "candidates": {},
        "judgments": {},
        "rejectedCreatorParsedResponses": {},
    }
    for prototype_id in _PROTOTYPES:
        candidate_id = f"cand-1-{prototype_id}-1-test"
        candidate = _candidate_with_slogan(
            prototype_id=prototype_id,
            product_name="אורי לב",
            slogan_text="קרוב אליך יותר ממה שחשבת",
        )
        candidate["candidateId"] = candidate_id
        candidate["advertisingSloganFormulation"]["relativeAdvantageSource"] = _RELATIVE_ADVANTAGE
        candidate.update(logo_policy_creator_extras(advertised_entity_name="אורי לב"))
        candidate.update(_production_shaped_metaphor_extras())
        state["rejectedCreatorParsedResponses"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": 1,
            "attemptNumber": 1,
            "parsed": deepcopy(candidate),
            "failureReason": _FAILURE_REASON,
        }
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": 1,
            "attemptNumber": 1,
            "validationStatus": "creator_rejected",
            "status": "creator_rejected",
            "failureReason": _FAILURE_REASON,
        }
    return state


def _apply_offline_recovery(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_offline_creator_recovery_batch(state, product_name="אורי לב")


def _recovered_six_creator_state(*, status: str = "error") -> Dict[str, Any]:
    state = _six_rejected_pre_recovery_state(status=status)
    _apply_offline_recovery(state)
    return state


def _make_judge_side_effect() -> Any:
    def judge_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
        candidate_id = kwargs["candidate_id"]
        judgment = _judgment(candidate_id, total_hint=80, eligible=True)
        scores = dict(judgment["scores"])
        total = 80
        return f"judge-{candidate_id}", judgment, total, scores

    return judge_side_effect


class TestCanonicalResumePlanParity(unittest.TestCase):
    def test_six_creators_zero_judges_resolves_judge_generation(self) -> None:
        state = _recovered_six_creator_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["acceptedCreatorCount"], 6)
        self.assertEqual(plan["acceptedJudgmentCount"], 0)
        self.assertEqual(plan["missingCreatorPrototypeIds"], [])
        self.assertEqual(len(plan["missingJudgmentPrototypeIds"]), 6)
        self.assertEqual(plan["resolvedResumeStage"], RESUME_STAGE_JUDGE_GENERATION)
        self.assertTrue(plan["resumeEligible"])
        self.assertTrue(plan["executorWouldAcceptState"])
        self.assertFalse(plan["creatorsWouldDispatch"])
        self.assertFalse(plan["strategyWouldDispatch"])
        self.assertFalse(plan["winnerWouldDispatch"])
        self.assertFalse(plan["mediaWouldDispatch"])
        self.assertEqual(plan["judgeCallsPlanned"], 6)

    def test_inspector_and_executor_agree_on_recovered_state(self) -> None:
        state = _recovered_six_creator_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        ok, reason, executor_plan = evaluate_complete_ad_reasoning_executor_preconditions(state)
        self.assertTrue(ok, reason)
        self.assertEqual(plan["executorWouldAcceptState"], executor_plan["executorWouldAcceptState"])
        self.assertEqual(plan["resolvedResumeStage"], executor_plan["resolvedResumeStage"])
        self.assertEqual(plan["missingJudgmentPrototypeIds"], executor_plan["missingJudgmentPrototypeIds"])

    def test_inspector_safe_implies_executor_preconditions_pass(self) -> None:
        state = _recovered_six_creator_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(plan["resumeEligible"])
        self.assertTrue(plan["executorWouldAcceptState"])
        self.assertTrue(ok, reason)

    def test_five_five_pattern_still_supported(self) -> None:
        state = _six_prototype_state(judged=5, creators=5)
        state["jobId"] = "job-five-five"
        state["tournamentId"] = "t-five-five"
        state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
        state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
        state["strategyFoundation"] = {"productNameResolved": "Product", "language": "he"}
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok, reason)

    def test_unsupported_partial_state_rejected_by_both(self) -> None:
        state = _six_prototype_state(judged=4, creators=4)
        state["jobId"] = "job-bad"
        state["tournamentId"] = "t-bad"
        state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
        state["strategyFoundation"] = {"productNameResolved": "Product", "language": "he"}
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        ok, reason, _ = evaluate_complete_ad_reasoning_executor_preconditions(state)
        self.assertFalse(ok)
        self.assertFalse(plan["executorWouldAcceptState"])
        self.assertIn("unexpected_partial_state", reason or "")

    def test_missing_prototype_ids_semantics_split(self) -> None:
        state = _recovered_six_creator_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["missingCreatorPrototypeIds"], [])
        self.assertEqual(set(plan["missingJudgmentPrototypeIds"]), set(_PROTOTYPES))
        self.assertEqual(set(plan["missingPrototypeIds"]), set(_PROTOTYPES))


class TestOfflineRecoveryResumeTransition(unittest.TestCase):
    def test_offline_recovery_clears_terminal_error_and_sets_judge_stage(self) -> None:
        state = _six_rejected_pre_recovery_state(status="error")
        report = _apply_offline_recovery(state)
        self.assertTrue(report["stateMutated"])
        self.assertEqual(report["acceptedBefore"], 0)
        self.assertEqual(report["acceptedAfter"], 6)
        self.assertEqual(report["rejectedBefore"], 6)
        self.assertEqual(report["rejectedAfter"], 0)
        self.assertEqual(state["status"], "paused_for_reasoning_resume")
        self.assertEqual(state["progressStage"], RESUME_STAGE_JUDGE_GENERATION)
        self.assertIsNone(state["failureReason"])
        self.assertTrue(report["readyForJudges"])
        self.assertTrue(report["reasoningResumePossible"])

    def test_offline_recovery_idempotent(self) -> None:
        state = _six_rejected_pre_recovery_state(status="error")
        _apply_offline_recovery(state)
        second = run_offline_creator_recovery_batch(state)
        self.assertFalse(second["stateMutated"])

    def test_job_id_resolves_tournament_id_after_recovery(self) -> None:
        state = _six_rejected_pre_recovery_state()
        _apply_offline_recovery(state)
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["jobId"], "job-recovered-judge-resume")
        self.assertEqual(plan["tournamentId"], "tournament-recovered-judge")

    def test_creator_identities_preserved_after_recovery(self) -> None:
        state = _six_rejected_pre_recovery_state()
        original_ids = {
            prototype_id: f"cand-1-{prototype_id}-1-test" for prototype_id in _PROTOTYPES
        }
        _apply_offline_recovery(state)
        for prototype_id, candidate_id in original_ids.items():
            rec = state["candidates"][candidate_id]
            self.assertEqual(rec["prototypeId"], prototype_id)
            self.assertEqual(rec["validationStatus"], "accepted")


class TestJudgeOnlyResumeFlow(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _run_judge_resume(
        self,
        state: Dict[str, Any],
        *,
        max_calls: int = 6,
        use_controlled: bool = False,
    ) -> Dict[str, Any]:
        save_tournament_state(state["jobId"], deepcopy(state))
        patches = [
            patch("engine.builder2_complete_ad_reasoning_resume.judge_candidate", side_effect=_make_judge_side_effect()),
            patch("engine.builder2_judge_only_resume.acquire_job_lease", return_value=True),
            patch("engine.builder2_judge_only_resume.release_job_lease"),
            patch("engine.builder2_judge_only_resume.redis_configured", return_value=True),
            patch("engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True),
            patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"),
            patch("engine.builder2_complete_ad_reasoning_resume.redis_configured", return_value=True),
            patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}),
            patch("engine.builder2_judge_only_resume.video_job_get_raw", return_value={}),
        ]
        for item in patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in patches])
        if use_controlled:
            return run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=max_calls,
                acquire_lease=False,
            )
        return run_judge_only_reasoning_resume(job_id=state["jobId"], max_calls=max_calls, acquire_lease=False)

    def test_controlled_resume_accepts_recovered_state_without_openai(self) -> None:
        state = _recovered_six_creator_state(status="error")
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok, reason)

    def test_judge_only_dispatches_six_without_creator_or_strategy(self) -> None:
        state = _recovered_six_creator_state(status="error")
        with patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate"
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock:
            report = self._run_judge_resume(state, max_calls=6)
        self.assertTrue(report.get("ok"))
        self.assertEqual(report.get("judgeCallsDispatched"), 6)
        self.assertEqual(report.get("creatorCallsThisRun"), 0)
        self.assertEqual(report.get("winnerCallsThisRun"), 0)
        self.assertTrue(report.get("readyForWinnerDevelopment"))
        self.assertFalse(report.get("winnerDevelopmentStarted"))
        self.assertTrue(report.get("stoppedBeforeMedia"))
        creator_mock.assert_not_called()
        winner_mock.assert_not_called()

    def test_controlled_resume_judge_path_respects_three_call_cap(self) -> None:
        state = _recovered_six_creator_state(status="error")
        report = self._run_judge_resume(state, max_calls=DEFAULT_MAX_CALLS, use_controlled=True)
        self.assertTrue(report.get("ok"))
        self.assertEqual(report.get("judgeCallsDispatched"), 3)
        self.assertTrue(report.get("callBudgetExhausted"))
        self.assertEqual(len(report.get("remainingMissingJudgmentPrototypeIds") or []), 3)

    def test_partial_judge_resume_retries_only_remaining(self) -> None:
        state = _recovered_six_creator_state(status="error")
        first = self._run_judge_resume(state, max_calls=2)
        self.assertEqual(first.get("judgeCallsDispatched"), 2)
        self.assertFalse(first.get("readyForWinnerDevelopment"))
        self.assertEqual(len(first.get("remainingMissingJudgmentPrototypeIds") or []), 4)
        stored = load_tournament_state(state["jobId"])
        assert stored is not None
        self.assertEqual(accepted_judgment_count(stored), 2)
        second = self._run_judge_resume(stored, max_calls=6)
        self.assertEqual(second.get("judgeCallsDispatched"), 4)
        self.assertTrue(second.get("readyForWinnerDevelopment"))

    def test_accepted_judges_not_repeated_on_retry(self) -> None:
        state = _recovered_six_creator_state(status="error")
        save_tournament_state(state["jobId"], deepcopy(state))
        with patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=_make_judge_side_effect(),
        ) as judge_mock, patch("engine.builder2_judge_only_resume.acquire_job_lease", return_value=True), patch(
            "engine.builder2_judge_only_resume.release_job_lease"
        ), patch("engine.builder2_judge_only_resume.redis_configured", return_value=True), patch(
            "engine.builder2_judge_only_resume.video_job_get_raw", return_value={}
        ):
            first = run_judge_only_reasoning_resume(job_id=state["jobId"], max_calls=3, acquire_lease=False)
            self.assertEqual(first.get("judgeCallsDispatched"), 3)
            first_count = judge_mock.call_count
            second = run_judge_only_reasoning_resume(job_id=state["jobId"], max_calls=6, acquire_lease=False)
            self.assertEqual(second.get("judgeCallsDispatched"), 3)
            self.assertEqual(judge_mock.call_count - first_count, 3)
            self.assertEqual(judge_mock.call_count, 6)

    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    def test_inspector_parity_fields(
        self,
        _raw_job: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        state = _recovered_six_creator_state(status="error")
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertTrue(report["ok"])
        self.assertTrue(report["executorWouldAcceptState"])
        self.assertIsNone(report["executorRejectionReason"])
        self.assertEqual(report["acceptedCreatorCount"], 6)
        self.assertEqual(len(report["missingJudgmentPrototypeIds"]), 6)
        self.assertEqual(report["missingCreatorPrototypeIds"], [])
        self.assertEqual(report["judgeCallsPlanned"], 6)
        self.assertFalse(report["creatorsWouldDispatch"])
        self.assertFalse(report["strategyWouldDispatch"])
        self.assertFalse(report["winnerWouldDispatch"])
        self.assertFalse(report["mediaWouldDispatch"])
        self.assertEqual(report["paidCalls"], 0)
        self.assertFalse(report["stateMutated"])

    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    def test_repeated_inspect_causes_no_mutations(
        self,
        _raw_job: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        state = _recovered_six_creator_state(status="error")
        read_raw.return_value = deepcopy(state)
        with read_only_builder2_inspection() as counter:
            inspect_builder2_complete_ad_resume(state["jobId"])
            inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertEqual(counter.redis_mutations, 0)


class TestResumeBudgetSemantics(unittest.TestCase):
    def test_controlled_default_max_calls_is_three_role_bound(self) -> None:
        self.assertEqual(DEFAULT_MAX_CALLS, 3)

    def test_judge_only_default_max_calls_is_six(self) -> None:
        self.assertEqual(DEFAULT_JUDGE_ONLY_MAX_CALLS, 6)

    def test_normal_reasoning_budget_unchanged(self) -> None:
        self.assertEqual(NORMAL_REASONING_CALL_BUDGET, 14)
        self.assertEqual(resolve_builder2_tournament_max_rounds(), 1)

    def test_failed_resume_before_dispatch_zero_paid_calls(self) -> None:
        state = _six_prototype_state(judged=4, creators=4)
        state["jobId"] = "job-fail-pre-dispatch"
        state["tournamentId"] = "t-fail"
        state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
        state["strategyFoundation"] = {"productNameResolved": "Product", "language": "he"}
        with patch("engine.builder2_complete_ad_reasoning_resume.redis_configured", return_value=True), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}
        ), patch("engine.builder2_complete_ad_reasoning_resume.judge_candidate") as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate"
        ) as creator_mock:
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=state,
                acquire_lease=False,
            )
        self.assertFalse(report.get("ok"))
        self.assertIn("unexpected_partial_state", report.get("failureReason") or "")
        self.assertEqual(report.get("totalReasoningCallsThisRun"), 0)
        judge_mock.assert_not_called()
        creator_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
