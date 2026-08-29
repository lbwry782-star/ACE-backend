"""
Builder2 mixed partial resume planning, execution, and inspector tests.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import (
    run_controlled_complete_ad_reasoning_resume,
    validate_controlled_complete_ad_preconditions,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import (
    RESUME_STAGE_MIXED_PARTIAL,
    evaluate_complete_ad_reasoning_executor_preconditions,
    resolve_complete_ad_canonical_resume_plan,
)
from engine.builder2_judge_grounding_failure_inspect import inspect_judge_grounding_failures
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from tests.builder2_methodology_fixtures import methodology_strategy_evidence_extras
from tests.test_builder2_tournament import _candidate, _judgment


_PROTOTYPES = [
    "winning_card",
    "summer_fan",
    "forgot",
    "closest",
    "think_small",
    "greenpeace_essential_pairing",
]
_MISSING_CREATORS = ["winning_card", "summer_fan", "forgot", "greenpeace_essential_pairing"]
_ACCEPTED_CREATORS = ["closest", "think_small"]
_SPARSE_DESCRIPTION = "An AI application that creates advertising ideas for small businesses."


def _strategy_block() -> Dict[str, Any]:
    return validate_strategy_foundation(
        methodology_strategy_evidence_extras(
            tournament_id="9d789e1e-7e4a-4ef4-b72e-642da8083788",
            product_name="אורי לב",
            product_description=_SPARSE_DESCRIPTION,
        ),
        product_name="אורי לב",
        product_description=_SPARSE_DESCRIPTION,
    )


def _production_mixed_partial_state(*, status: str = "failed") -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "jobId": "e369b792-9988-4087-b054-38a713966918",
        "tournamentId": "9d789e1e-7e4a-4ef4-b72e-642da8083788",
        "builder2ResumeContractVersion": BUILDER2_RESUME_CONTRACT_VERSION,
        "builder2NewFormatVersion": BUILDER2_NEW_FORMAT_VERSION,
        "schemaVersion": TOURNAMENT_STATE_SCHEMA_VERSION,
        "status": status,
        "failureReason": "builder2_judge_contract_systemic_failure",
        "failureStage": "judge_generation",
        "productDescription": _SPARSE_DESCRIPTION,
        "contentLanguage": "he",
        "initialActivePrototypeIds": list(_PROTOTYPES),
        "activePrototypeIds": list(_PROTOTYPES),
        "strategyFoundation": _strategy_block(),
        "acceptedCreatorCandidates": {},
        "acceptedJudgments": {},
        "candidates": {},
        "judgments": {},
        "metrics": {"judgeRepairCalls": 1},
        "judgeContractCircuitBreaker": {
            "tripped": True,
            "trippedReason": "shared_structural_contract_field",
            "repeatedFieldPaths": ["factualGroundingAssessment.productClaimFactuallyGrounded"],
        },
        "judgeDiagnosticsByCandidate": {},
    }
    for prototype_id in _ACCEPTED_CREATORS:
        candidate_id = f"cand-1-{prototype_id}-1-test"
        candidate = _candidate(prototype_id)
        candidate["candidateId"] = candidate_id
        state["acceptedCreatorCandidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "validationStatus": "accepted",
            "creatorOutput": candidate,
        }
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "creatorAcceptanceStatus": "accepted",
            "validationStatus": "accepted",
            "status": "accepted",
            "judgeStatus": "pending",
            "creatorOutput": candidate,
            "creatorSnapshot": candidate,
            "creatorFactuallyGrounded": True,
            "newProductClaimsIntroduced": [],
            "judgmentId": None,
            "eligible": False,
            "judgeDiagnostics": {
                "responseReceived": prototype_id == "closest",
                "repairAttempted": prototype_id == "closest",
                "failureFieldPaths": ["factualGroundingAssessment.productClaimFactuallyGrounded"]
                if prototype_id == "closest"
                else [],
                "failureReason": "builder2_judge_validation_failed:factualGroundingAssessment.productClaimFactuallyGrounded"
                if prototype_id == "closest"
                else None,
            },
        }
        state["judgeDiagnosticsByCandidate"][candidate_id] = dict(state["candidates"][candidate_id]["judgeDiagnostics"])
    return state


def _make_judge_side_effect(*, eligible: bool = True) -> Any:
    def judge_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
        candidate_id = kwargs["candidate_id"]
        judgment = _judgment(candidate_id, total_hint=70, eligible=eligible)
        return f"judge-{candidate_id}", judgment, 70, dict(judgment["scores"])

    return judge_side_effect


class TestMixedPartialResumePlanning(unittest.TestCase):
    def test_production_state_is_resumable(self) -> None:
        state = _production_mixed_partial_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertTrue(plan["resumeEligible"], plan.get("executorRejectionReason"))
        self.assertTrue(plan["executorWouldAcceptState"])
        self.assertEqual(plan["resolvedResumeStage"], RESUME_STAGE_MIXED_PARTIAL)
        self.assertFalse(plan["strategyWouldDispatch"])
        self.assertEqual(plan["acceptedCreatorCount"], 2)
        self.assertEqual(plan["acceptedJudgmentCount"], 0)
        self.assertEqual(plan["remainingCreatorNormalCalls"], 4)
        self.assertEqual(plan["remainingJudgeNormalCalls"], 6)
        self.assertTrue(plan["winnerNormalCallConditional"])
        self.assertEqual(plan["minimumAdditionalNormalReasoningCalls"], 10)
        self.assertEqual(plan["maximumAdditionalNormalReasoningCalls"], 11)

    def test_resume_plan_by_prototype(self) -> None:
        state = _production_mixed_partial_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        by_proto = plan["resumePlanByPrototype"]
        self.assertEqual(by_proto["closest"]["creatorAction"], "reuse")
        self.assertEqual(by_proto["closest"]["judgeAction"], "dispatch")
        self.assertEqual(by_proto["think_small"]["creatorAction"], "reuse")
        self.assertEqual(by_proto["think_small"]["judgeAction"], "dispatch")
        self.assertEqual(by_proto["winning_card"]["creatorAction"], "dispatch")
        self.assertEqual(by_proto["winning_card"]["judgeAction"], "dispatch_after_creator")
        self.assertEqual(by_proto["summer_fan"]["creatorAction"], "dispatch")
        self.assertEqual(by_proto["forgot"]["creatorAction"], "dispatch")
        self.assertEqual(by_proto["greenpeace_essential_pairing"]["creatorAction"], "dispatch")

    def test_failed_unrelated_state_still_rejected(self) -> None:
        state = _production_mixed_partial_state()
        state["acceptedCreatorCandidates"] = {}
        state["candidates"] = {}
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertFalse(plan["executorWouldAcceptState"])

    def test_executor_preconditions_accept_production_state(self) -> None:
        state = _production_mixed_partial_state()
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok, reason)


class TestMixedPartialResumeExecution(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_execution_reuses_existing_creators_and_dispatches_judges_first(self) -> None:
        state = _production_mixed_partial_state()
        save_tournament_state(state["jobId"], deepcopy(state))
        creator_calls: Dict[str, int] = {"count": 0}
        judge_calls: Dict[str, int] = {"count": 0}

        def creator_side_effect(**kwargs: Any):
            creator_calls["count"] += 1
            prototype_id = kwargs["prototype_id"]
            candidate_id = f"cand-1-{prototype_id}-1-new"
            candidate = _candidate(prototype_id)
            candidate["candidateId"] = candidate_id
            return candidate_id, candidate

        def judge_side_effect(**kwargs: Any):
            judge_calls["count"] += 1
            candidate_id = kwargs["candidate_id"]
            judgment = _judgment(candidate_id, eligible=True)
            return f"judge-{candidate_id}", judgment, 70, dict(judgment["scores"])

        with patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=creator_side_effect,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=judge_side_effect,
        ), patch("engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch("engine.builder2_complete_ad_reasoning_resume.redis_configured", return_value=True), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=2,
                acquire_lease=False,
            )
        self.assertTrue(report.get("ok"))
        self.assertEqual(creator_calls["count"], 0)
        self.assertEqual(judge_calls["count"], 2)
        stored = load_tournament_state(state["jobId"])
        assert stored is not None
        self.assertEqual(accepted_creator_count(stored), 2)
        self.assertEqual(accepted_judgment_count(stored), 2)

    def test_negative_judgment_persists_and_continues(self) -> None:
        state = _production_mixed_partial_state()
        save_tournament_state(state["jobId"], deepcopy(state))

        def judge_side_effect(**kwargs: Any):
            candidate_id = kwargs["candidate_id"]
            judgment = _judgment(candidate_id, eligible=False)
            judgment["disqualifiers"] = ["unsupported_viewer_inference"]
            return f"judge-{candidate_id}", judgment, 55, dict(judgment["scores"])

        with patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=judge_side_effect,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=AssertionError("creator should not be called"),
        ), patch("engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch("engine.builder2_complete_ad_reasoning_resume.redis_configured", return_value=True), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=2,
                acquire_lease=False,
            )
        self.assertTrue(report.get("ok"))
        stored = load_tournament_state(state["jobId"])
        assert stored is not None
        judged = [c for c in stored["candidates"].values() if c.get("judgmentId")]
        self.assertGreaterEqual(len(judged), 1)
        self.assertTrue(all(not c.get("eligible") for c in judged if c.get("judgmentId")))


class TestJudgeInspectorLegacyReporting(unittest.TestCase):
    def test_legacy_unpersisted_response_not_structurally_invalid(self) -> None:
        state = _production_mixed_partial_state()
        report = inspect_judge_grounding_failures(state)
        self.assertEqual(report["attemptedJudgeCount"], 2)
        closest = next(item for item in report["attempts"] if item["prototypeId"] == "closest")
        think_small = next(item for item in report["attempts"] if item["prototypeId"] == "think_small")
        self.assertTrue(closest["legacyResponseNotPersisted"])
        self.assertIsNone(closest["responseStructurallyValidUnderCorrectedContract"])
        self.assertEqual(closest["structuralErrors"], [])
        self.assertFalse(closest["offlinePersistencePossible"])
        self.assertIsNone(closest["falseBooleanMisclassifiedAsValidationFailure"])
        self.assertTrue(closest["repairDispatched"])
        self.assertFalse(think_small["repairDispatched"])
        self.assertEqual(report["historicalCircuitBreakerReason"], "shared_structural_contract_field")
        self.assertEqual(report["structurallyInvalidResponseCount"], 0)

    def test_inspector_is_read_only(self) -> None:
        state = _production_mixed_partial_state()
        with read_only_builder2_inspection() as counter:
            inspect_judge_grounding_failures(state)
            inspect_judge_grounding_failures(state)
        self.assertEqual(counter.redis_mutations, 0)


class TestCompleteAdResumeInspectorFields(unittest.TestCase):
    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    def test_production_plan_fields(self, _raw_job: Any, read_raw: Any, _redis: Any) -> None:
        state = _production_mixed_partial_state()
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertTrue(report["ok"])
        self.assertTrue(report["executorWouldAcceptState"])
        self.assertEqual(report["acceptedCreatorCount"], 2)
        self.assertEqual(report["remainingCreatorNormalCalls"], 4)
        self.assertEqual(report["remainingJudgeNormalCalls"], 6)
        self.assertEqual(report["normalCallsBeforeWinner"], 10)
        self.assertEqual(len(report["missingCreatorPrototypeIds"]), 4)
        self.assertEqual(report["missingJudgmentPrototypeIds"], ["closest", "think_small"])
        self.assertEqual(len(report["actionableMissingJudgmentPrototypeIds"]), 2)
        self.assertEqual(report["remainingJudgeNormalCalls"], 6)
        self.assertEqual(report["resumePlanByPrototype"]["closest"]["judgeAction"], "dispatch")
        self.assertEqual(report["resumePlanByPrototype"]["winning_card"]["creatorAction"], "dispatch")
        self.assertFalse(report["strategyWouldDispatch"])


if __name__ == "__main__":
    unittest.main()
