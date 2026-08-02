"""
Builder2 Judge factual-grounding output schema, fingerprint persistence, and resume-plan tests.
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_judge import judge_candidate, judge_candidate_structural_repair
from engine.builder2_judge_core_contract import (
    JUDGE_FACTUAL_GROUNDING_GATE_FIELDS,
    build_judge_factual_grounding_prompt_text,
)
from engine.builder2_judge_factual_grounding_output_schema import (
    BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1,
    REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES,
    assert_judge_factual_grounding_output_schema_contract,
    build_factual_grounding_assessment_json_schema,
    factual_grounding_assessment_satisfies_schema_contract,
    serialized_judge_output_schema_for_tests,
)
from engine.builder2_judge_grounding_failure_inspect import inspect_judge_grounding_failures
from engine.builder2_judge_pending_repair import resolve_pending_judge_repair
from engine.builder2_judge_response_ledger import (
    append_judge_response_attempt,
    finalize_judge_response_validation,
    resolve_parsed_response_fingerprint,
)
from engine.builder2_judge_unavailable_resolution_contract import PRODUCTION_CLOSEST_CANDIDATE_ID
from engine.builder2_strategy_evidence_grounding_contract import build_default_judge_factual_grounding_assessment
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_prompts import build_judge_prompt, build_judge_repair_prompt
from engine.builder2_prototypes import require_prototype
from tests.builder2_methodology_fixtures import methodology_judge_factual_grounding_extras, methodology_strategy_evidence_extras
from tests.test_builder2_judge_unavailable_resolution import _apply_resolution, _unrecoverable_closest_state
from tests.test_builder2_judge_pending_repair import _closest_empty_assessment_state, _grounded_judgment, _strategy
from tests.test_builder2_tournament import _candidate, _judgment


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _think_small_production_state() -> Dict[str, Any]:
    from engine.builder2_judge_circuit_breaker import JUDGE_BREAKER_CONTRACT_VERSION
    from engine.builder2_judge_response_ledger import backfill_parsed_response_fingerprint
    from engine.builder2_judge_pending_repair import persist_pending_judge_repair

    state = _unrecoverable_closest_state()
    _apply_resolution(state)
    think_small_id = "cand-1-think_small-1-c6196416"
    for candidate_id in list((state.get("candidates") or {}).keys()):
        record = state["candidates"].get(candidate_id)
        if isinstance(record, dict) and record.get("prototypeId") == "think_small" and candidate_id != think_small_id:
            state["candidates"].pop(candidate_id, None)
            state.get("acceptedCreatorCandidates", {}).pop(candidate_id, None)
    candidate = _candidate("think_small")
    candidate["candidateId"] = think_small_id
    state["acceptedCreatorCandidates"][think_small_id] = {
        "candidateId": think_small_id,
        "prototypeId": "think_small",
        "validationStatus": "accepted",
        "creatorOutput": candidate,
    }
    state["candidates"][think_small_id] = {
        "candidateId": think_small_id,
        "prototypeId": "think_small",
        "creatorAcceptanceStatus": "accepted",
        "validationStatus": "accepted",
        "status": "accepted",
        "judgeStatus": "pending",
        "creatorOutput": candidate,
    }
    parsed = _grounded_judgment(think_small_id, eligible=False)
    parsed["factualGroundingAssessment"] = {}
    entry: Dict[str, Any] = {
        "candidateId": think_small_id,
        "attemptId": "judge-attempt-think-small-001",
        "judgmentId": "judge-cand-1-think_small-1-c6196416-90ce86f5",
        "callType": "normal",
        "responseAvailable": True,
        "parsedResponseAvailable": True,
        "parsedResponse": parsed,
        "responseFingerprint": "e6738e9c64dfc906671ef8dc0b998da19804ef3cbf711d761030865cc95d85a8",
        "parsedResponseFingerprint": "",
        "validationFailureField": "factualGroundingAssessment",
        "validationFailureReason": "builder2_judge_validation_failed:factualGroundingAssessment",
        "structuralErrors": ["builder2_judge_validation_failed:factualGroundingAssessment"],
    }
    backfill_parsed_response_fingerprint(entry)
    state["judgeResponseLedgerByCandidate"] = {think_small_id: [entry]}
    persist_pending_judge_repair(
        state,
        candidate_id=think_small_id,
        normal_entry=entry,
        structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
    )
    breaker = state.setdefault("judgeContractCircuitBreaker", {})
    breaker["contractVersion"] = JUDGE_BREAKER_CONTRACT_VERSION
    breaker["currentCandidateFailurePaths"] = {think_small_id: ["factualGroundingAssessment"]}
    breaker["currentContractTripped"] = False
    breaker["legacyCandidateFailurePaths"] = {
        PRODUCTION_CLOSEST_CANDIDATE_ID: [
            "factualGroundingAssessment",
            "factualGroundingAssessment.productClaimFactuallyGrounded",
            "factualGroundingAssessment.noUnsupportedFeatureClaim",
            "factualGroundingAssessment.notes",
        ]
    }
    return state


class TestJudgeFactualGroundingOutputSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _strategy()

    def test_serialized_normal_schema_rejects_empty_object(self) -> None:
        schema = serialized_judge_output_schema_for_tests(strategy_foundation=self.strategy)
        assert schema is not None
        assessment_schema = schema["properties"]["factualGroundingAssessment"]
        self.assertFalse(factual_grounding_assessment_satisfies_schema_contract({}))
        self.assertEqual(set(assessment_schema["required"]), set(REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES))

    def test_serialized_repair_schema_rejects_empty_object(self) -> None:
        schema = serialized_judge_output_schema_for_tests(strategy_foundation=self.strategy)
        assert schema is not None
        self.assertFalse(factual_grounding_assessment_satisfies_schema_contract({}))

    def test_all_five_boolean_gates_required(self) -> None:
        schema = build_factual_grounding_assessment_json_schema()
        for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
            self.assertIn(gate, schema["required"])
            self.assertEqual(schema["properties"][gate]["type"], "boolean")

    def test_boolean_false_accepted(self) -> None:
        assessment = build_default_judge_factual_grounding_assessment(notes="Unsupported claim.")
        assessment["productClaimFactuallyGrounded"] = False
        self.assertTrue(factual_grounding_assessment_satisfies_schema_contract(assessment))

    def test_notes_required(self) -> None:
        assessment = build_default_judge_factual_grounding_assessment()
        del assessment["notes"]
        self.assertFalse(factual_grounding_assessment_satisfies_schema_contract(assessment))

    def test_normal_and_repair_share_contract(self) -> None:
        normal = serialized_judge_output_schema_for_tests(strategy_foundation=self.strategy)
        repair = serialized_judge_output_schema_for_tests(strategy_foundation=self.strategy)
        self.assertEqual(normal, repair)

    def test_serialized_schema_contains_nested_properties(self) -> None:
        schema = serialized_judge_output_schema_for_tests(strategy_foundation=self.strategy)
        assert schema is not None
        assert_judge_factual_grounding_output_schema_contract(schema)
        props = schema["properties"]["factualGroundingAssessment"]["properties"]
        self.assertTrue(props)

    def test_prompt_and_schema_field_names_match(self) -> None:
        prompt = build_judge_factual_grounding_prompt_text()
        for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
            self.assertIn(gate, prompt)
        self.assertIn("notes", prompt)
        judge_prompt = build_judge_prompt(
            product_name="Product",
            product_description="Sparse",
            language="he",
            strategy_foundation=self.strategy,
            prototype=require_prototype("think_small"),
            candidate=_candidate("think_small"),
            candidate_id="cand-1-think_small-1-c6196416",
        )
        self.assertIn("factualGroundingAssessment is mandatory", judge_prompt)

    def test_empty_object_fails_structural_validation(self) -> None:
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"] = {}
        from engine.builder2_judge import collect_judge_structural_errors

        errors = collect_judge_structural_errors(
            parsed,
            candidate_id="c1",
            candidate=_candidate("think_small"),
            strategy_foundation=self.strategy,
        )
        self.assertTrue(any("factualGroundingAssessment" in item for item in errors))


class TestParsedFingerprintPersistence(unittest.TestCase):
    def test_fingerprint_persisted_before_validation_failure(self) -> None:
        state: Dict[str, Any] = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"] = {}
        attempt_id = append_judge_response_attempt(
            state,
            candidate_id="c1",
            judgment_id="j1",
            call_type="normal",
            response_text=json.dumps(parsed),
            parsed=parsed,
        )
        finalize_judge_response_validation(
            state,
            candidate_id="c1",
            attempt_id=attempt_id,
            validation_failure_field="factualGroundingAssessment",
            validation_failure_reason="builder2_judge_validation_failed:factualGroundingAssessment",
            accepted=False,
        )
        entry = state["judgeResponseLedgerByCandidate"]["c1"][0]
        self.assertTrue(entry.get("parsedResponseFingerprint"))

    def test_failed_normal_retains_fingerprint(self) -> None:
        state: Dict[str, Any] = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"] = {}
        attempt_id = append_judge_response_attempt(
            state,
            candidate_id="c1",
            judgment_id="j1",
            call_type="normal",
            response_text=json.dumps(parsed),
            parsed=parsed,
        )
        before = resolve_parsed_response_fingerprint(state["judgeResponseLedgerByCandidate"]["c1"][0])["effective"]
        finalize_judge_response_validation(
            state,
            candidate_id="c1",
            attempt_id=attempt_id,
            accepted=False,
            validation_failure_field="factualGroundingAssessment",
        )
        after = resolve_parsed_response_fingerprint(state["judgeResponseLedgerByCandidate"]["c1"][0])["effective"]
        self.assertEqual(before, after)
        self.assertTrue(after)

    def test_failed_repair_retains_fingerprint(self) -> None:
        state: Dict[str, Any] = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        parsed = _grounded_judgment("c1")
        parsed["factualGroundingAssessment"] = {"notes": "missing booleans"}
        attempt_id = append_judge_response_attempt(
            state,
            candidate_id="c1",
            judgment_id="j-repair",
            call_type="repair",
            response_text=json.dumps(parsed),
            parsed=parsed,
        )
        before = resolve_parsed_response_fingerprint(state["judgeResponseLedgerByCandidate"]["c1"][0])["effective"]
        finalize_judge_response_validation(
            state,
            candidate_id="c1",
            attempt_id=attempt_id,
            accepted=False,
            validation_failure_field="factualGroundingAssessment",
        )
        after = resolve_parsed_response_fingerprint(state["judgeResponseLedgerByCandidate"]["c1"][0])["effective"]
        self.assertEqual(before, after)

    def test_inspector_derives_fingerprint_from_persisted_json(self) -> None:
        state = _think_small_production_state()
        think_small_id = "cand-1-think_small-1-c6196416"
        entry = state["judgeResponseLedgerByCandidate"][think_small_id][0]
        entry["parsedResponseFingerprint"] = ""
        report = inspect_judge_grounding_failures(state)
        attempt = next(item for item in report["attempts"] if item.get("prototypeId") == "think_small")
        self.assertTrue(attempt.get("parsedFingerprintDerivationPossible"))
        self.assertTrue(attempt.get("parsedFingerprintDerived"))
        self.assertFalse(attempt.get("pendingRepairBlockedByMissingFingerprint"))

    def test_pending_repair_refuses_mismatched_fingerprint(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        normal = state["judgeResponseLedgerByCandidate"][closest_id][0]
        source_parsed = normal["parsedResponse"]
        with patch("engine.builder2_judge._invoke_judge_model", return_value=json.dumps(_grounded_judgment(closest_id))):
            with self.assertRaises(Builder2TournamentError):
                judge_candidate_structural_repair(
                    product_name="Product",
                    product_description="Sparse",
                    language="he",
                    strategy_foundation=state["strategyFoundation"],
                    prototype_id="closest",
                    candidate_id=closest_id,
                    candidate=_candidate("closest"),
                    source_judgment_id=str(normal.get("judgmentId")),
                    source_parsed=source_parsed,
                    source_parsed_fingerprint="deadbeef",
                    structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
                    state=state,
                )

    def test_pending_repair_allows_derived_fingerprint(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        normal = state["judgeResponseLedgerByCandidate"][closest_id][0]
        normal["parsedResponseFingerprint"] = ""
        source_parsed = normal["parsedResponse"]
        derived = resolve_parsed_response_fingerprint(normal)["derived"]

        def invoke_side_effect(**kwargs: Any) -> str:
            repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=False))
            repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(
                notes="Repair notes."
            )
            repaired["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
            return json.dumps(repaired, ensure_ascii=False)

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=invoke_side_effect):
            _, judgment, _, _ = judge_candidate_structural_repair(
                product_name="Product",
                product_description="Sparse",
                language="he",
                strategy_foundation=state["strategyFoundation"],
                prototype_id="closest",
                candidate_id=closest_id,
                candidate=_candidate("closest"),
                source_judgment_id=str(normal.get("judgmentId")),
                source_parsed=source_parsed,
                source_parsed_fingerprint=str(derived),
                structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
                state=state,
            )
        self.assertFalse(judgment["eligible"])


class TestThinkSmallResumePlan(unittest.TestCase):
    def test_think_small_repair_only_not_normal(self) -> None:
        state = _think_small_production_state()
        think_small_id = "cand-1-think_small-1-c6196416"
        pending = resolve_pending_judge_repair(state, think_small_id)
        self.assertIsNotNone(pending)
        self.assertTrue(pending.get("repairRequired"))
        self.assertFalse(pending.get("repairDispatched"))
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        prototype_plan = (plan.get("resumePlanByPrototype") or {}).get("think_small") or {}
        self.assertEqual(prototype_plan.get("creatorAction"), "reuse")
        self.assertEqual(prototype_plan.get("judgeAction"), "dispatch_repair")
        self.assertEqual(prototype_plan.get("normalJudgeCalls"), 0)
        self.assertEqual(prototype_plan.get("repairJudgeCalls"), 1)
        self.assertEqual(prototype_plan.get("sourceJudgmentId"), "judge-cand-1-think_small-1-c6196416-90ce86f5")
        self.assertTrue(_clean(prototype_plan.get("sourceParsedResponseFingerprint")))

    def test_repair_false_boolean_persists_ineligible(self) -> None:
        state = _closest_empty_assessment_state()
        closest_id = "cand-1-closest-1-c4ba148f"
        normal = state["judgeResponseLedgerByCandidate"][closest_id][0]

        def invoke_side_effect(**kwargs: Any) -> str:
            repaired = copy.deepcopy(_grounded_judgment(closest_id, eligible=True))
            repaired["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(notes="Negative.")
            repaired["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
            return json.dumps(repaired, ensure_ascii=False)

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=invoke_side_effect):
            _, judgment, _, _ = judge_candidate_structural_repair(
                product_name="Product",
                product_description="Sparse",
                language="he",
                strategy_foundation=state["strategyFoundation"],
                prototype_id="closest",
                candidate_id=closest_id,
                candidate=_candidate("closest"),
                source_judgment_id=str(normal.get("judgmentId")),
                source_parsed=normal["parsedResponse"],
                source_parsed_fingerprint=str(normal.get("parsedResponseFingerprint")),
                structural_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
                state=state,
            )
        self.assertFalse(judgment["eligible"])

    def test_current_breaker_evidence_remains_one(self) -> None:
        state = _think_small_production_state()
        from engine.builder2_judge_circuit_breaker import current_breaker_evidence_count, legacy_breaker_evidence_excluded_count

        self.assertEqual(current_breaker_evidence_count(state), 1)
        self.assertGreaterEqual(legacy_breaker_evidence_excluded_count(state), 0)

    def test_closest_resolved_unavailable_zero_calls(self) -> None:
        state = _think_small_production_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        prototype_plan = (plan.get("resumePlanByPrototype") or {}).get("closest") or {}
        self.assertEqual(prototype_plan.get("judgeAction"), "resolved_unavailable")
        self.assertEqual(prototype_plan.get("normalJudgeCalls"), 0)
        self.assertEqual(prototype_plan.get("repairJudgeCalls"), 0)

    def test_remaining_call_counts(self) -> None:
        state = _think_small_production_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan.get("remainingCreatorNormalCalls"), 4)
        self.assertEqual(plan.get("remainingJudgeNormalCalls"), 4)
        self.assertEqual(plan.get("requiredJudgeRepairCalls"), 1)
        self.assertEqual(plan.get("totalNormalCallsBeforeWinner"), 8)
        self.assertEqual(plan.get("totalRequiredRepairCallsBeforeWinner"), 1)
        self.assertEqual(plan.get("totalPaidCallsBeforeWinner"), 9)
        self.assertEqual(plan.get("conditionalWinnerNormalCalls"), 1)
        self.assertEqual(plan.get("minimumAdditionalPaidReasoningCalls"), 9)
        self.assertEqual(plan.get("maximumAdditionalPaidReasoningCallsWithoutFutureRepairs"), 10)
        self.assertTrue(plan.get("possibleRepairCallsNotIncluded"))
        self.assertEqual(plan.get("perInvocationCallLimit"), 3)

    def test_schema_version_constant(self) -> None:
        self.assertEqual(BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1, "builder2_judge_factual_grounding_output_schema_v1")

    def test_repair_prompt_mentions_mandatory_assessment(self) -> None:
        strategy = _strategy()
        prompt = build_judge_repair_prompt(
            product_name="Product",
            product_description="Sparse",
            language="he",
            strategy_foundation=strategy,
            prototype=require_prototype("think_small"),
            candidate=_candidate("think_small"),
            candidate_id="cand-1-think_small-1-c6196416",
            invalid_output={"factualGroundingAssessment": {}},
            validation_failures=["builder2_judge_validation_failed:factualGroundingAssessment"],
        )
        self.assertIn("factualGroundingAssessment is mandatory", prompt)

    def test_judge_candidate_persists_fingerprint_before_validation_failure(self) -> None:
        state: Dict[str, Any] = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        strategy = _strategy()

        def llm(**kwargs: Any) -> Dict[str, Any]:
            judgment = _judgment("c1")
            judgment.update(methodology_judge_factual_grounding_extras())
            judgment["factualGroundingAssessment"] = {}
            return judgment

        with patch("engine.builder2_judge._invoke_judge_model", side_effect=lambda **kwargs: json.dumps(llm())):
            with self.assertRaises(Builder2TournamentError):
                judge_candidate(
                    product_name="Product",
                    product_description="Sparse",
                    language="he",
                    strategy_foundation=strategy,
                    prototype_id="think_small",
                    candidate_id="c1",
                    candidate=_candidate("think_small"),
                    state=state,
                    single_attempt_only=True,
                )
        entry = state["judgeResponseLedgerByCandidate"]["c1"][0]
        self.assertTrue(entry.get("parsedResponseFingerprint"))
