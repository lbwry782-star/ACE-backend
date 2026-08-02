"""
Builder2 Judge repair response ledger, inspector, and salvage tests.
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_judge import judge_candidate_structural_repair
from engine.builder2_judge_grounding_failure_inspect import inspect_judge_grounding_failures
from engine.builder2_judge_pending_repair import (
    REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE,
    repair_judge_call_must_not_repeat,
    resolve_judge_repair_resume_context,
)
from engine.builder2_judge_repair_offline_salvage import assess_repair_attempt_salvage, salvage_repair_judgment_offline
from engine.builder2_judge_repair_response_inspect import inspect_judge_repair_response
from engine.builder2_judge_response_ledger import append_judge_response_attempt, find_latest_attempt, repair_attempts
from engine.builder2_judge_structural_repair_classifier import is_judge_structural_repairable
from engine.builder2_strategy_evidence_grounding_contract import build_default_judge_factual_grounding_assessment
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from tests.test_builder2_judge_pending_repair import _closest_empty_assessment_state, _grounded_judgment
from tests.test_builder2_tournament import _candidate


class TestRepairResponseLedgerPersistence(unittest.TestCase):
    def test_repair_persisted_before_validation_failure(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        normal = find_latest_attempt(state, closest_id, call_type="normal")
        assert normal is not None
        source_parsed = normal.get("parsedResponse")
        assert isinstance(source_parsed, dict)

        def invoke_side_effect(**kwargs: Any) -> str:
            repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=False))
            repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(
                notes="Repair notes."
            )
            repaired["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
            return json.dumps(repaired, ensure_ascii=False)

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=invoke_side_effect):
            judgment_id, judgment, _, _ = judge_candidate_structural_repair(
                product_name="Product",
                product_description="Sparse product",
                language="he",
                strategy_foundation=state["strategyFoundation"],
                prototype_id="closest",
                candidate_id=closest_id,
                candidate=_candidate("closest"),
                source_judgment_id=str(normal.get("judgmentId")),
                source_parsed=source_parsed,
                source_parsed_fingerprint=str(normal.get("parsedResponseFingerprint")),
                structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
                state=state,
            )
        repair = find_latest_attempt(state, closest_id, call_type="repair")
        self.assertIsNotNone(repair)
        self.assertTrue(repair.get("parsedResponseAvailable"))
        self.assertEqual(repair.get("callType"), "repair")
        self.assertFalse(judgment["eligible"])

    def test_failed_repair_remains_in_ledger(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        normal = find_latest_attempt(state, closest_id, call_type="normal")
        assert normal is not None
        source_parsed = normal.get("parsedResponse")
        assert isinstance(source_parsed, dict)

        def invoke_side_effect(**kwargs: Any) -> str:
            broken = copy.deepcopy(_grounded_judgment(closest_id, eligible=True))
            broken["factualGroundingAssessment"] = {"notes": "missing booleans"}
            return json.dumps(broken, ensure_ascii=False)

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=invoke_side_effect):
            with self.assertRaises(Builder2TournamentError):
                judge_candidate_structural_repair(
                    product_name="Product",
                    product_description="Sparse product",
                    language="he",
                    strategy_foundation=state["strategyFoundation"],
                    prototype_id="closest",
                    candidate_id=closest_id,
                    candidate=_candidate("closest"),
                    source_judgment_id=str(normal.get("judgmentId")),
                    source_parsed=source_parsed,
                    source_parsed_fingerprint=str(normal.get("parsedResponseFingerprint")),
                    structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
                    state=state,
                )
        repair = find_latest_attempt(state, closest_id, call_type="repair")
        self.assertIsNotNone(repair)
        self.assertFalse(repair.get("accepted"))
        self.assertTrue(repair.get("validationFailureReason"))

    def test_normal_and_repair_coexist(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        append_judge_response_attempt(
            state,
            candidate_id=closest_id,
            judgment_id="judge-repair-test",
            call_type="repair",
            response_text='{"candidateId":"x"}',
            parsed={"candidateId": closest_id, "eligible": False, "factualGroundingAssessment": {}},
            source_judgment_id="judge-normal",
        )
        self.assertEqual(len(repair_attempts(state, closest_id)), 1)
        self.assertIsNotNone(find_latest_attempt(state, closest_id, call_type="normal"))


class TestRepairValidationAndSalvage(unittest.TestCase):
    def test_false_boolean_is_not_repairable(self) -> None:
        parsed = _grounded_judgment("c1", eligible=False)
        parsed["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
        self.assertFalse(
            is_judge_structural_repairable(
                "builder2_judge_validation_failed",
                "factualGroundingAssessment.productClaimFactuallyGrounded",
                parsed=parsed,
            )
        )

    def test_offline_salvage_zero_calls(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=False))
        repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(notes="Salvage.")
        repaired["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
        append_judge_response_attempt(
            state,
            candidate_id=closest_id,
            judgment_id="judge-repair-salvage",
            call_type="repair",
            response_text=json.dumps(repaired, ensure_ascii=False),
            parsed=repaired,
            source_judgment_id="judge-normal",
        )
        result = salvage_repair_judgment_offline(state, candidate_id=closest_id, dry_run=False)
        self.assertTrue(result.get("salvaged"))
        self.assertEqual(result.get("paidCalls"), 0)
        self.assertFalse(result["eligible"])


class TestRepairInspectorAndResume(unittest.TestCase):
    def test_inspector_lists_repair_attempt(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=False))
        repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(notes="Inspect.")
        append_judge_response_attempt(
            state,
            candidate_id=closest_id,
            judgment_id="judge-repair-inspect",
            call_type="repair",
            response_text=json.dumps(repaired, ensure_ascii=False),
            parsed=repaired,
        )
        state["candidates"][closest_id]["pendingJudgeRepair"]["repairDispatched"] = True
        report = inspect_judge_grounding_failures(state)
        self.assertEqual(report["repairJudgeResponseCount"], 1)
        self.assertEqual(report["repairParsedResponsePersistedCount"], 1)
        repair_attempt = next(item for item in report["attempts"] if item.get("callType") == "repair")
        self.assertEqual(repair_attempt["callType"], "repair")

    def test_unrecoverable_state_blocks_automatic_retry(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        pending = state["candidates"][closest_id]["pendingJudgeRepair"]
        pending["repairDispatched"] = True
        pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
        pending["repairOutcomeUnrecoverable"] = True
        state["metrics"] = {"judgeRepairCalls": 1}
        ctx = resolve_judge_repair_resume_context(state, closest_id)
        self.assertEqual(ctx["kind"], "unrecoverable")
        self.assertTrue(repair_judge_call_must_not_repeat(state, closest_id))
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertFalse(plan["executorWouldAcceptState"])
        closest_plan = plan["resumePlanByPrototype"]["closest"]
        self.assertEqual(closest_plan["judgeAction"], "repair_response_unrecoverable")

    def test_inspect_unrecoverable_repair_response(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        state["candidates"][closest_id]["pendingJudgeRepair"]["repairDispatched"] = True
        state["metrics"] = {"judgeRepairCalls": 1}
        report = inspect_judge_repair_response(state, candidate_id=closest_id)
        self.assertTrue(report["repairDispatchRecorded"])
        self.assertFalse(report["parsedRepairResponseAvailable"])
        self.assertEqual(report["unavailableReason"], "builder2_judge_repair_response_unavailable")


class TestRepairResumeIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_resume_does_not_repeat_dispatched_repair(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        state["candidates"][closest_id]["pendingJudgeRepair"]["repairDispatched"] = True
        state["metrics"] = {"judgeRepairCalls": 1}
        save_tournament_state(state["jobId"], copy.deepcopy(state))
        with patch(
            "engine.builder2_judge._invoke_judge_model",
            side_effect=AssertionError("repair must not run"),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease",
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.redis_configured",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw",
            return_value={},
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=1,
                acquire_lease=False,
            )
        self.assertFalse(report.get("ok"))


if __name__ == "__main__":
    unittest.main()
