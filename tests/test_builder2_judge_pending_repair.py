"""
Builder2 Judge pending repair, structural classifier, and circuit-breaker isolation tests.
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_judge import (
    collect_judge_structural_errors,
    judge_candidate,
    judge_candidate_structural_repair,
)
from engine.builder2_judge_circuit_breaker import (
    current_breaker_evidence_count,
    is_current_judge_contract_circuit_breaker_tripped,
    is_judge_contract_circuit_breaker_tripped,
    legacy_breaker_evidence_excluded_count,
    record_judge_contract_failure,
)
from engine.builder2_judge_pending_repair import (
    normal_judge_call_must_not_repeat,
    pending_judge_repair_candidate_ids,
    persist_pending_judge_repair,
    resolve_judge_repair_resume_context,
    resolve_pending_judge_repair,
)
from engine.builder2_judge_structural_repair_classifier import (
    classify_judge_structural_repair,
    is_factual_grounding_object_structurally_defective,
    is_judge_structural_repairable,
)
from engine.builder2_judge_grounding_failure_inspect import inspect_judge_grounding_failures
from engine.builder2_strategy_evidence_grounding_contract import build_default_judge_factual_grounding_assessment
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from tests.builder2_methodology_fixtures import methodology_judge_factual_grounding_extras, methodology_strategy_evidence_extras
from tests.test_builder2_mixed_partial_resume import _production_mixed_partial_state
from tests.test_builder2_tournament import _candidate, _judgment


_SPARSE = "An AI application that creates advertising ideas for small businesses."


def _strategy() -> Dict[str, Any]:
    from engine.builder2_strategy import validate_strategy_foundation

    return validate_strategy_foundation(
        methodology_strategy_evidence_extras(product_description=_SPARSE),
        product_name="ACE Product",
        product_description=_SPARSE,
    )


def _grounded_judgment(candidate_id: str, *, eligible: bool = True) -> Dict[str, Any]:
    judgment = _judgment(candidate_id, eligible=eligible)
    judgment.update(methodology_judge_factual_grounding_extras())
    return judgment


def _closest_empty_assessment_state() -> Dict[str, Any]:
    state = _production_mixed_partial_state()
    closest_id = ""
    closest_record: Dict[str, Any] = {}
    for cid, rec in (state.get("candidates") or {}).items():
        if isinstance(rec, dict) and rec.get("prototypeId") == "closest":
            closest_id = str(cid)
            closest_record = rec
            break
    if not closest_id:
        raise AssertionError("closest candidate missing from production fixture")
    production_closest_id = "cand-1-closest-1-c4ba148f"
    if closest_id != production_closest_id:
        state["candidates"][production_closest_id] = dict(closest_record)
        state["candidates"][production_closest_id]["candidateId"] = production_closest_id
        state["candidates"].pop(closest_id, None)
        if closest_id in state.get("acceptedCreatorCandidates", {}):
            state["acceptedCreatorCandidates"][production_closest_id] = dict(state["acceptedCreatorCandidates"].pop(closest_id))
            state["acceptedCreatorCandidates"][production_closest_id]["candidateId"] = production_closest_id
        closest_id = production_closest_id
    parsed = _grounded_judgment(closest_id, eligible=False)
    parsed["factualGroundingAssessment"] = {}
    parsed["scores"] = dict(parsed.get("scores") or {})
    response_text = json.dumps(parsed, ensure_ascii=False)
    import hashlib

    response_fp = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    parsed_fp = hashlib.sha256(
        json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    state["acceptedCreatorCandidates"][closest_id] = {
        "candidateId": closest_id,
        "prototypeId": "closest",
        "validationStatus": "accepted",
        "creatorOutput": state["candidates"][closest_id]["creatorOutput"],
    }
    state["judgeResponseLedgerByCandidate"] = {
        closest_id: [
            {
                "candidateId": closest_id,
                "judgmentId": "judge-cand-1-closest-1-c4ba148f-017d6914",
                "callType": "normal",
                "responseAvailable": True,
                "parsedResponseAvailable": True,
                "parsedResponse": parsed,
                "responseFingerprint": response_fp,
                "parsedResponseFingerprint": parsed_fp,
                "validationFailureField": "factualGroundingAssessment",
                "validationFailureReason": "builder2_judge_validation_failed:factualGroundingAssessment",
                "structuralErrors": ["builder2_judge_validation_failed:factualGroundingAssessment"],
            }
        ]
    }
    persist_pending_judge_repair(
        state,
        candidate_id=closest_id,
        normal_entry=state["judgeResponseLedgerByCandidate"][closest_id][0],
        structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
    )
    closest_record = state["candidates"][closest_id]
    closest_record["validationStatus"] = "accepted"
    closest_record["status"] = "accepted"
    closest_record["judgeStatus"] = "pending"
    closest_record.pop("judgeFailure", None)
    for candidate_id, record in (state.get("candidates") or {}).items():
        if isinstance(record, dict) and record.get("prototypeId") == "think_small":
            record["validationStatus"] = "accepted"
            record["status"] = "accepted"
            record["judgeStatus"] = "pending"
            record.pop("judgeFailure", None)
    state["candidates"][closest_id]["judgeDiagnostics"] = {
        "responseReceived": True,
        "repairAttempted": False,
        "failureFieldPaths": ["factualGroundingAssessment"],
        "failureReason": "builder2_judge_validation_failed:factualGroundingAssessment",
    }
    state["judgeDiagnosticsByCandidate"][closest_id] = dict(state["candidates"][closest_id]["judgeDiagnostics"])
    breaker = state.setdefault("judgeContractCircuitBreaker", {})
    breaker.setdefault(
        "candidateFailurePaths",
        {
            "cand-1-think_small-1-c6196416": ["factualGroundingAssessment.productClaimFactuallyGrounded"],
        },
    )
    return state


class TestStructuralRepairClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _strategy()

    def test_empty_object_is_structural_and_repairable(self) -> None:
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"] = {}
        self.assertTrue(is_factual_grounding_object_structurally_defective(parsed))
        decision = classify_judge_structural_repair(
            "builder2_judge_validation_failed",
            "factualGroundingAssessment",
            parsed=parsed,
        )
        self.assertTrue(decision["repairable"])
        self.assertEqual(decision["decision"], "structural_defect")

    def test_valid_false_is_not_repairable(self) -> None:
        parsed = _grounded_judgment("c1", eligible=False)
        parsed["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
        parsed["factualGroundingAssessment"]["notes"] = "Unsupported claim."
        self.assertFalse(
            is_judge_structural_repairable(
                "builder2_judge_validation_failed",
                "factualGroundingAssessment.productClaimFactuallyGrounded",
                parsed=parsed,
            )
        )

    def test_missing_gate_is_repairable(self) -> None:
        parsed = _grounded_judgment("c1")
        del parsed["factualGroundingAssessment"]["productClaimFactuallyGrounded"]
        errors = collect_judge_structural_errors(
            parsed,
            candidate_id="c1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(errors)
        self.assertTrue(
            is_judge_structural_repairable(
                "builder2_judge_validation_failed",
                "factualGroundingAssessment.productClaimFactuallyGrounded",
                parsed=parsed,
            )
        )

    def test_wrong_type_is_repairable(self) -> None:
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"]["noUnsupportedFeatureClaim"] = "false"
        self.assertTrue(
            is_judge_structural_repairable(
                "builder2_judge_validation_failed",
                "factualGroundingAssessment.noUnsupportedFeatureClaim",
                parsed=parsed,
            )
        )


class TestPendingRepairPlanning(unittest.TestCase):
    def test_closest_pending_repair_plan(self) -> None:
        state = _closest_empty_assessment_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        closest = plan["resumePlanByPrototype"]["closest"]
        self.assertEqual(closest["judgeAction"], "dispatch_repair")
        self.assertFalse(closest["normalJudgeCallRequired"])
        self.assertTrue(closest["repairJudgeCallRequired"])
        self.assertEqual(plan["remainingCreatorNormalCalls"], 4)
        self.assertEqual(plan["remainingJudgeNormalCalls"], 5)
        self.assertEqual(plan["requiredJudgeRepairCalls"], 1)
        self.assertEqual(plan["totalPaidCallsBeforeWinner"], 10)
        self.assertEqual(plan["minimumAdditionalPaidReasoningCalls"], 10)

    def test_think_small_still_normal_judge(self) -> None:
        state = _closest_empty_assessment_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        entry = plan["resumePlanByPrototype"]["think_small"]
        self.assertEqual(entry["judgeAction"], "dispatch")
        self.assertEqual(entry["normalJudgeCalls"], 1)


class TestPendingRepairExecution(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_resume_dispatches_repair_not_normal(self) -> None:
        state = _closest_empty_assessment_state()
        save_tournament_state(state["jobId"], copy.deepcopy(state))
        calls = {"normal": 0, "repair": 0}

        def invoke_side_effect(**kwargs: Any) -> str:
            if kwargs.get("call_type") == "repair":
                calls["repair"] += 1
                repaired = copy.deepcopy(_grounded_judgment("cand-1-closest-1-c4ba148f", eligible=False))
                repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(
                    notes="Repaired structural assessment."
                )
                repaired["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
                return json.dumps(repaired, ensure_ascii=False)
            calls["normal"] += 1
            return json.dumps(_grounded_judgment("c1"), ensure_ascii=False)

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=invoke_side_effect), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease",
            return_value=True,
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.redis_configured",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw",
            return_value={},
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=lambda **kwargs: (
                f"cand-1-{kwargs['prototype_id']}-1-new",
                _candidate(kwargs["prototype_id"]),
            ),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=1,
                acquire_lease=False,
            )
        self.assertTrue(report.get("ok"))
        self.assertEqual(calls["repair"], 1)
        self.assertEqual(calls["normal"], 0)
        stored = load_tournament_state(state["jobId"])
        assert stored is not None
        ledger = stored["judgeResponseLedgerByCandidate"]["cand-1-closest-1-c4ba148f"]
        self.assertEqual(ledger[0]["parsedResponseFingerprint"], state["judgeResponseLedgerByCandidate"]["cand-1-closest-1-c4ba148f"][0]["parsedResponseFingerprint"])
        self.assertTrue(any(item.get("callType") == "repair" for item in ledger))

    def test_repeated_resume_does_not_repeat_repair(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=False))
        repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(notes="Repaired.")
        state["judgeResponseLedgerByCandidate"][closest_id].append(
            {
                "candidateId": closest_id,
                "judgmentId": "judge-repair-closest",
                "callType": "repair",
                "responseAvailable": True,
                "parsedResponseAvailable": True,
                "parsedResponse": repaired,
                "repairResponseAccepted": True,
                "accepted": True,
                "sourceJudgmentId": "judge-cand-1-closest-1-c4ba148f-017d6914",
            }
        )
        save_tournament_state(state["jobId"], copy.deepcopy(state))
        stored = load_tournament_state(state["jobId"])
        assert stored is not None
        ctx = resolve_judge_repair_resume_context(stored, closest_id)
        self.assertIn(ctx.get("kind"), {"unresolved_salvageable", "none"})


class TestCircuitBreakerIsolation(unittest.TestCase):
    def test_legacy_breaker_visible_but_excluded(self) -> None:
        state = _closest_empty_assessment_state()
        self.assertTrue(state["judgeContractCircuitBreaker"]["tripped"])
        self.assertFalse(is_current_judge_contract_circuit_breaker_tripped(state))
        self.assertGreater(legacy_breaker_evidence_excluded_count(state), 0)

    def test_one_current_failure_does_not_combine_with_legacy(self) -> None:
        state = _closest_empty_assessment_state()
        record_judge_contract_failure(
            state,
            candidate_id="cand-1-closest-1-c4ba148f",
            error_paths=["factualGroundingAssessment"],
            parsed={"factualGroundingAssessment": {}},
        )
        self.assertFalse(is_current_judge_contract_circuit_breaker_tripped(state))
        self.assertEqual(current_breaker_evidence_count(state), 1)

    def test_two_current_same_object_paths_trip_breaker(self) -> None:
        state: Dict[str, Any] = {"judgeContractCircuitBreaker": {}}
        parsed = {"factualGroundingAssessment": {}}
        record_judge_contract_failure(
            state,
            candidate_id="c1",
            error_paths=["factualGroundingAssessment"],
            parsed=parsed,
        )
        record_judge_contract_failure(
            state,
            candidate_id="c2",
            error_paths=["factualGroundingAssessment"],
            parsed=parsed,
        )
        self.assertTrue(is_current_judge_contract_circuit_breaker_tripped(state))


class TestJudgeInspectorPendingRepair(unittest.TestCase):
    def test_empty_assessment_reports_repair_blocked_and_pending(self) -> None:
        state = _closest_empty_assessment_state()
        report = inspect_judge_grounding_failures(state)
        closest = next(item for item in report["attempts"] if item["prototypeId"] == "closest" and item["callType"] == "normal")
        self.assertFalse(closest["responseStructurallyValidUnderCorrectedContract"])
        self.assertTrue(closest["repairNecessaryUnderCorrectedContract"])
        self.assertFalse(closest["repairDispatched"])
        self.assertTrue(closest["pendingRepairEligible"])
        self.assertTrue(closest["normalCallMustNotRepeat"])
        self.assertEqual(closest["structuralRepairClassifierDecision"], "structural_defect")


if __name__ == "__main__":
    unittest.main()
