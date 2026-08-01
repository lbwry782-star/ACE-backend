"""
Builder2 Judge factual-grounding handling — structural validity vs substantive eligibility.
"""
from __future__ import annotations

import copy
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_judge import (
    collect_judge_structural_errors,
    judge_candidate,
    validate_judge_response,
)
from engine.builder2_judge_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE,
    is_judge_contract_circuit_breaker_tripped,
    record_judge_contract_failure,
)
from engine.builder2_judge_grounding_failure_inspect import inspect_judge_grounding_failures
from engine.builder2_judge_grounding_offline_recovery import recover_candidate_judgment_offline
from engine.builder2_strategy_evidence_grounding_contract import (
    build_default_judge_factual_grounding_assessment,
    collect_failed_factual_grounding_gates,
)
from engine.builder2_tournament_contracts import Builder2TournamentError, JUDGMENT_SCHEMA_VERSION
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, new_tournament_state
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from tests.builder2_methodology_fixtures import methodology_judge_factual_grounding_extras, methodology_strategy_evidence_extras
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _judgment, _strategy, _winner_plan_from_prompt


SPARSE_DESCRIPTION = "An AI application that creates advertising ideas for small businesses."


def _evidence_strategy() -> Dict[str, Any]:
    from engine.builder2_strategy import validate_strategy_foundation

    return validate_strategy_foundation(
        methodology_strategy_evidence_extras(product_description=SPARSE_DESCRIPTION),
        product_name="ACE Product",
        product_description=SPARSE_DESCRIPTION,
    )


def _judgment_with_grounding(candidate_id: str, *, eligible: bool = True) -> Dict[str, Any]:
    judgment = _judgment(candidate_id, eligible=eligible)
    judgment.update(methodology_judge_factual_grounding_extras())
    return judgment


def _set_gate(judgment: Dict[str, Any], gate: str, value: Any) -> None:
    judgment.setdefault("factualGroundingAssessment", build_default_judge_factual_grounding_assessment())
    judgment["factualGroundingAssessment"][gate] = value


class TestFactualGroundingStructuralValidity(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _evidence_strategy()

    def test_all_gates_true_valid_and_eligible(self) -> None:
        judgment, total, scores = validate_judge_response(
            _judgment_with_grounding("cand-1", eligible=True),
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(judgment["eligible"])
        self.assertGreater(total, 0)
        self.assertEqual(len(scores), 8)

    def test_product_claim_false_is_structurally_valid_ineligible(self) -> None:
        judgment = _judgment_with_grounding("cand-1", eligible=True)
        _set_gate(judgment, "productClaimFactuallyGrounded", False)
        judgment["factualGroundingAssessment"]["notes"] = "Viewer-facing mechanism implies unsupported feedback workflow."
        parsed, _, _ = validate_judge_response(
            judgment,
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertFalse(parsed["eligible"])
        self.assertIn("productClaimFactuallyGrounded", collect_failed_factual_grounding_gates(parsed["factualGroundingAssessment"]))

    def test_each_gate_false_is_structurally_valid(self) -> None:
        for gate in (
            "noUnsupportedFeatureClaim",
            "noCategoryConventionPresentedAsProductFact",
            "viewerWouldNotInferUnsupportedCapability",
            "relativeAdvantageEvidenceAccepted",
        ):
            with self.subTest(gate=gate):
                judgment = _judgment_with_grounding("cand-1", eligible=True)
                _set_gate(judgment, gate, False)
                judgment["factualGroundingAssessment"]["notes"] = f"Gate failed: {gate}."
                parsed, _, _ = validate_judge_response(
                    judgment,
                    candidate_id="cand-1",
                    candidate=_candidate("closest"),
                    strategy_foundation=self.strategy,
                )
                self.assertFalse(parsed["eligible"])

    def test_missing_gate_is_structural(self) -> None:
        judgment = _judgment_with_grounding("cand-1")
        del judgment["factualGroundingAssessment"]["productClaimFactuallyGrounded"]
        errors = collect_judge_structural_errors(
            judgment,
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(any("productClaimFactuallyGrounded" in item for item in errors))

    def test_string_false_is_structural(self) -> None:
        judgment = _judgment_with_grounding("cand-1")
        _set_gate(judgment, "productClaimFactuallyGrounded", "false")
        errors = collect_judge_structural_errors(
            judgment,
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(any("productClaimFactuallyGrounded" in item for item in errors))

    def test_null_gate_is_structural(self) -> None:
        judgment = _judgment_with_grounding("cand-1")
        _set_gate(judgment, "noUnsupportedFeatureClaim", None)
        errors = collect_judge_structural_errors(
            judgment,
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(any("noUnsupportedFeatureClaim" in item for item in errors))


class TestFactualGroundingRepairAndCircuitBreaker(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _evidence_strategy()

    def test_negative_judgment_does_not_dispatch_repair(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            judgment = _judgment_with_grounding("cand-1", eligible=False)
            _set_gate(judgment, "viewerWouldNotInferUnsupportedCapability", False)
            judgment["factualGroundingAssessment"]["notes"] = "Metaphor implies feedback shaping."
            judgment["disqualifiers"] = ["unsupported_viewer_inference"]
            return judgment

        _, judgment, _, _ = judge_candidate(
            product_name="Product",
            product_description=SPARSE_DESCRIPTION,
            language="en",
            strategy_foundation=self.strategy,
            prototype_id="closest",
            candidate_id="cand-1",
            candidate=_candidate("closest"),
            llm_client=llm,
            state=state,
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(state["metrics"].get("judgeRepairCalls", 0), 0)
        self.assertFalse(judgment["eligible"])

    def test_two_negative_judgments_do_not_trip_breaker(self) -> None:
        state: Dict[str, Any] = {}
        record_judge_contract_failure(
            state,
            candidate_id="c1",
            error_paths=["factualGroundingAssessment.productClaimFactuallyGrounded"],
        )
        record_judge_contract_failure(
            state,
            candidate_id="c2",
            error_paths=["factualGroundingAssessment.noUnsupportedFeatureClaim"],
        )
        self.assertFalse(is_judge_contract_circuit_breaker_tripped(state))

    def test_two_malformed_responses_trip_breaker(self) -> None:
        state: Dict[str, Any] = {}
        record_judge_contract_failure(
            state,
            candidate_id="c1",
            error_paths=["verbalLayerAssessment", "verbalLayerAssessment.keywordBornFromVisual"],
        )
        record_judge_contract_failure(
            state,
            candidate_id="c2",
            error_paths=["verbalLayerAssessment", "verbalLayerAssessment.visualMeaningIsClear"],
        )
        self.assertTrue(is_judge_contract_circuit_breaker_tripped(state))


class TestFactualGroundingTournamentBehavior(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_negative_judgment_persists_and_tournament_continues(self) -> None:
        judge_calls = {"count": 0}

        def llm(**kwargs: Any):
            role = kwargs.get("role")
            prompt = kwargs.get("prompt", "")
            if role == "builder2_strategy":
                return _evidence_strategy()
            if role == "builder2_creator":
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS[:2]:
                    if pid in prompt:
                        prototype_id = pid
                        break
                return _candidate(prototype_id, prompt=prompt)
            if role == "builder2_judge":
                judge_calls["count"] += 1
                candidate_id = "unknown"
                for token in prompt.split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip().strip(",")
                        break
                if judge_calls["count"] == 1:
                    judgment = _judgment_with_grounding(candidate_id, eligible=False)
                    _set_gate(judgment, "productClaimFactuallyGrounded", False)
                    judgment["factualGroundingAssessment"]["notes"] = "Unsupported feedback inference."
                    judgment["disqualifiers"] = ["unsupported_product_capability"]
                    return judgment
                return _judgment_with_grounding(candidate_id, eligible=True)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        with patch.dict(
            os.environ,
            {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS[:2]), "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
            clear=True,
        ):
            run_builder2_tournament(
                job_id="job-factual-negative-continue",
                product_name="Product",
                product_description=SPARSE_DESCRIPTION,
                content_language="en",
                llm_client=llm,
                rng_seed="seed-factual-negative",
            )
        state = load_tournament_state("job-factual-negative-continue")
        assert state is not None
        ineligible = [c for c in state["candidates"].values() if c.get("judgmentId") and not c.get("eligible")]
        eligible = [c for c in state["candidates"].values() if c.get("eligible")]
        self.assertEqual(len(ineligible), 1)
        self.assertGreaterEqual(len(eligible), 1)
        self.assertTrue(state.get("winnerCandidateId"))

    def test_all_ineligible_ends_gracefully(self) -> None:
        state = {
            "candidates": {
                f"c{i}": {
                    "candidateId": f"c{i}",
                    "eligible": False,
                    "creatorAcceptanceStatus": "accepted",
                    "judgeStatus": "accepted",
                    "validationStatus": "accepted",
                    "judgmentId": f"j{i}",
                    "totalScore": 10 + i,
                    "tieScores": {},
                    "completedAt": f"2026-01-0{i}T00:00:00+00:00",
                }
                for i in range(1, 3)
            },
            "acceptedCreatorCandidates": {
                f"c{i}": {"candidateId": f"c{i}", "validationStatus": "accepted", "creatorOutput": {"prototypeId": "closest"}}
                for i in range(1, 3)
            },
        }
        with self.assertRaises(Builder2TournamentError) as ctx:
            select_global_winner(state)
        self.assertEqual(ctx.exception.args[0], "builder2_no_factually_eligible_candidate")


class TestFactualGroundingInspectorAndRecovery(unittest.TestCase):
    def test_inspector_reports_ledger_attempt(self) -> None:
        strategy = _evidence_strategy()
        candidate = _candidate("closest")
        parsed = _judgment_with_grounding("cand-1", eligible=False)
        _set_gate(parsed, "productClaimFactuallyGrounded", False)
        parsed["factualGroundingAssessment"]["notes"] = "Unsupported feedback metaphor."
        state = new_tournament_state(
            job_id="inspect-job",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["strategyFoundation"] = strategy
        state["candidates"] = {
            "cand-1": {
                "candidateId": "cand-1",
                "prototypeId": "closest",
                "creatorAcceptanceStatus": "accepted",
                "validationStatus": "accepted",
                "judgeStatus": "unavailable",
                "creatorOutput": candidate,
                "creatorSnapshot": candidate,
            }
        }
        state["judgeResponseLedgerByCandidate"] = {
            "cand-1": [
                {
                    "judgmentId": "judge-cand-1-test",
                    "callType": "normal",
                    "responseAvailable": True,
                    "parsedResponseAvailable": True,
                    "parsedResponse": parsed,
                    "responseFingerprint": "abc",
                    "parsedResponseFingerprint": "def",
                    "validationFailureField": "factualGroundingAssessment.productClaimFactuallyGrounded",
                    "validationFailureReason": "builder2_judge_validation_failed:factualGroundingAssessment.productClaimFactuallyGrounded",
                    "structuralErrors": [],
                }
            ]
        }
        report = inspect_judge_grounding_failures(state)
        self.assertEqual(report["attemptedJudgeCount"], 1)
        attempt = report["attempts"][0]
        self.assertTrue(attempt["responseStructurallyValidUnderCorrectedContract"])
        self.assertFalse(attempt["judgmentWouldBeEligibleUnderCorrectedContract"])
        self.assertTrue(attempt["offlineRevalidationPossible"])
        self.assertTrue(attempt["offlinePersistencePossible"])

    def test_offline_recovery_preserves_false_and_scores(self) -> None:
        strategy = _evidence_strategy()
        candidate = _candidate("closest")
        parsed = _judgment_with_grounding("cand-1", eligible=False)
        _set_gate(parsed, "viewerWouldNotInferUnsupportedCapability", False)
        parsed["factualGroundingAssessment"]["notes"] = "Viewer would infer unsupported feedback loop."
        parsed["disqualifiers"] = ["unsupported_viewer_inference"]
        state = new_tournament_state(
            job_id="recover-job",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["strategyFoundation"] = strategy
        state["candidates"] = {
            "cand-1": {
                "candidateId": "cand-1",
                "prototypeId": "closest",
                "creatorAcceptanceStatus": "accepted",
                "validationStatus": "accepted",
                "judgeStatus": "unavailable",
                "creatorOutput": candidate,
                "creatorSnapshot": candidate,
            }
        }
        state["judgeResponseLedgerByCandidate"] = {
            "cand-1": [
                {
                    "judgmentId": "judge-cand-1-recover",
                    "callType": "normal",
                    "parsedResponse": parsed,
                    "parsedResponseFingerprint": "fp-recover",
                }
            ]
        }
        result = recover_candidate_judgment_offline(state, candidate_id="cand-1")
        self.assertTrue(result["recovered"])
        self.assertFalse(result["eligible"])
        self.assertEqual(state["candidates"]["cand-1"]["judgmentId"], "judge-cand-1-recover")
        self.assertFalse(state["candidates"]["cand-1"]["eligible"])
        self.assertIsNotNone(state["candidates"]["cand-1"]["totalScore"])


if __name__ == "__main__":
    unittest.main()
