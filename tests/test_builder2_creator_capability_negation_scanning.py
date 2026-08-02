"""
Builder2 Creator capability negation scanning and production winning_card salvage tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from typing import Any, Dict

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    REJECTED_CREATOR_RESPONSE_HISTORY_KEY,
    can_offline_revalidate_rejected_creator,
    offline_revalidate_and_accept_rejected_creator,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_creator_grounding_failure_inspect import inspect_creator_grounding_failure
from engine.builder2_creator_grounding_offline_recovery import (
    PRODUCTION_WINNING_CARD_PARSED_FINGERPRINT,
    recover_creator_grounding_offline,
)
from engine.builder2_strategy_evidence_grounding_contract import (
    CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION,
    detect_capabilities_in_text,
    scan_capability_occurrences,
    scan_texts_for_unsupported_capabilities,
    validate_creator_evidence_grounding,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from tests.test_builder2_creator_grounding_failure_inspect import (
    _CANDIDATE_ID,
    _JOB_ID,
    _SPARSE_HE,
    _strategy,
    _winning_card_rejected,
    _winning_card_state,
)
from tests.test_builder2_judge_factual_grounding_output_schema import _think_small_production_state
from tests.test_builder2_judge_pending_repair import _grounded_judgment
from tests.test_builder2_mixed_partial_resume import _production_mixed_partial_state
from tests.test_builder2_tournament import _judgment


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _salvaged_production_resume_state() -> Dict[str, Any]:
    state = _think_small_production_state()
    winning_payload = copy.deepcopy(_production_winning_card_state()[REJECTED_CREATOR_PARSED_INDEX_KEY][_CANDIDATE_ID])
    parsed = winning_payload["parsed"]
    product_name = _clean(state["strategyFoundation"].get("productNameResolved")) or "ACE Product"
    parsed["advertisingClosure"]["productNameText"] = product_name
    parsed["advertisingClosure"]["productName"] = product_name
    parsed.setdefault("logoPolicyReport", {})["advertisedEntityName"] = product_name
    state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {_CANDIDATE_ID: winning_payload}
    state["candidates"][_CANDIDATE_ID] = copy.deepcopy(_production_winning_card_state()["candidates"][_CANDIDATE_ID])
    state["creatorDiagnosticsByCandidate"] = {
        **_production_winning_card_state().get("creatorDiagnosticsByCandidate", {}),
    }
    offline_revalidate_and_accept_rejected_creator(state, candidate_id=_CANDIDATE_ID, product_name=product_name)

    think_small_id = "cand-1-think_small-1-c6196416"
    judgment = _grounded_judgment(think_small_id, eligible=True)
    judgment_id = "judge-cand-1-think_small-1-c6196416-accepted"
    state.setdefault("acceptedJudgments", {})[think_small_id] = {
        "judgmentId": judgment_id,
        "candidateId": think_small_id,
        "judgment": judgment,
        "eligible": True,
    }
    state.setdefault("judgments", {})[judgment_id] = {
        "judgmentId": judgment_id,
        "candidateId": think_small_id,
        "judgment": judgment,
        "eligible": True,
    }
    think_small_record = state["candidates"][think_small_id]
    think_small_record["judgmentId"] = judgment_id
    think_small_record["judgeStatus"] = "accepted"
    think_small_record["eligible"] = True
    state.pop("pendingJudgeRepairByCandidate", None)
    return state


PRODUCTION_MECHANISM_SCAN_SUMMARY = (
    "הוויזואל חושף את שכבת התיווך האנושית שבדרך כלל נותרת מאחורי משטח "
    "הפרסום, אך אינו מייחס לה תהליך, אופטימיזציה או תוצאה."
)


def _parsed_fingerprint(parsed: Dict[str, Any]) -> str:
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _production_winning_card_state() -> Dict[str, Any]:
    state = _winning_card_state(
        mechanism="A playing card reveals the named professional address clearly.",
        self_report=[],
    )
    parsed = state[REJECTED_CREATOR_PARSED_INDEX_KEY][_CANDIDATE_ID]["parsed"]
    parsed["creatorReport"]["mechanismScanSummary"] = PRODUCTION_MECHANISM_SCAN_SUMMARY
    parsed["newProductClaimsIntroduced"] = []
    return state


class TestCapabilityNegationScanning(unittest.TestCase):
    def test_explicit_hebrew_negation_suppresses_claim(self) -> None:
        text = PRODUCTION_MECHANISM_SCAN_SUMMARY
        self.assertIn("optimization", detect_capabilities_in_text(text))
        self.assertEqual(scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[]), [])

    def test_attribution_denial_produces_no_claim(self) -> None:
        text = "אינו מייחס לה תהליך, אופטימיזציה או תוצאה."
        self.assertEqual(scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[]), [])

    def test_without_result_promise_produces_no_claim(self) -> None:
        text = "בלי להבטיח תוצאה מדידה."
        self.assertEqual(scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[]), [])

    def test_no_evidence_support_produces_no_claim(self) -> None:
        text = "אין מידע שמוכיח ליווי שוטף."
        self.assertEqual(scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[]), [])

    def test_positive_optimization_remains_rejected(self) -> None:
        text = "אורי לב מבצע אופטימיזציה לקמפיינים."
        hits = scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[])
        self.assertEqual([hit["capability"] for hit in hits], ["optimization"])

    def test_positive_learning_remains_rejected(self) -> None:
        text = "הסוכן לומד מהתוצאות ומשפר את הפרסום."
        hits = scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[])
        self.assertIn("performance_learning", [hit["capability"] for hit in hits])

    def test_mixed_negated_and_positive_emits_only_positive(self) -> None:
        text = "אינו מייחס אופטימיזציה, אך כולל ליווי שוטף."
        hits = scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[])
        self.assertEqual([hit["capability"] for hit in hits], ["collaborative_iteration"])

    def test_unrelated_clause_negation_does_not_suppress_positive(self) -> None:
        text = "אינו מייחס אופטימיזציה. הסוכן מספק ליווי שוטף."
        hits = scan_texts_for_unsupported_capabilities([("x", text)], allowed_capabilities=[])
        self.assertEqual([hit["capability"] for hit in hits], ["collaborative_iteration"])

    def test_creator_report_field_uses_same_classifier(self) -> None:
        occurrences = scan_capability_occurrences(
            PRODUCTION_MECHANISM_SCAN_SUMMARY,
            allowed_capabilities=[],
            field_path="creatorReport.mechanismScanSummary",
        )
        self.assertTrue(occurrences)
        self.assertTrue(all(not item["productClaimEmitted"] for item in occurrences))

    def test_finished_concept_field_uses_same_classifier(self) -> None:
        text = "אורי לב מבצע אופטימיזציה לקמפיינים."
        occurrences = scan_capability_occurrences(text, allowed_capabilities=[], field_path="coreCreativeMechanism")
        self.assertTrue(any(item["productClaimEmitted"] for item in occurrences))


class TestProductionWinningCardSalvage(unittest.TestCase):
    def test_production_response_revalidates_successfully(self) -> None:
        state = _production_winning_card_state()
        ok, reason = can_offline_revalidate_rejected_creator(
            state,
            candidate_id=_CANDIDATE_ID,
            product_name="אורי לב",
        )
        self.assertTrue(ok, reason)

    def test_parsed_fingerprint_remains_unchanged_by_scanner_fix(self) -> None:
        state = _production_winning_card_state()
        parsed = state[REJECTED_CREATOR_PARSED_INDEX_KEY][_CANDIDATE_ID]["parsed"]
        before = _parsed_fingerprint(parsed)
        can_offline_revalidate_rejected_creator(state, candidate_id=_CANDIDATE_ID, product_name="אורי לב")
        after = _parsed_fingerprint(parsed)
        self.assertEqual(before, after)

    def test_historical_rejection_preserved_on_accept(self) -> None:
        state = _production_winning_card_state()
        offline_revalidate_and_accept_rejected_creator(state, candidate_id=_CANDIDATE_ID, product_name="אורי לב")
        history = state.get(REJECTED_CREATOR_RESPONSE_HISTORY_KEY) or {}
        self.assertIn(_CANDIDATE_ID, history)
        self.assertEqual(
            history[_CANDIDATE_ID]["originalFailureReason"],
            "builder2_creator_validation_failed:newProductClaimsIntroduced",
        )

    def test_accepted_creator_persistence_idempotent(self) -> None:
        state = _production_winning_card_state()
        parsed = state[REJECTED_CREATOR_PARSED_INDEX_KEY][_CANDIDATE_ID]["parsed"]
        fingerprint = _parsed_fingerprint(parsed)
        first = recover_creator_grounding_offline(
            state,
            candidate_id=_CANDIDATE_ID,
            expected_fingerprint=fingerprint,
            dry_run=True,
        )
        second = recover_creator_grounding_offline(
            state,
            candidate_id=_CANDIDATE_ID,
            expected_fingerprint=fingerprint,
            dry_run=True,
        )
        self.assertTrue(first["recovered"])
        self.assertTrue(second["recovered"])
        self.assertEqual(first["reason"], "builder2_creator_grounding_offline_recovery_dry_run_ok")
        self.assertEqual(second["reason"], "builder2_creator_grounding_offline_recovery_dry_run_ok")

    def test_inspector_reports_negated_false_positive(self) -> None:
        state = _production_winning_card_state()
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertFalse(report["rejectionStillValidUnderCurrentContract"])
        self.assertEqual(report["inspectionConclusion"], "false_positive_negated_capability_mention")
        self.assertTrue(report["deterministicScannerAccepted"])
        self.assertTrue(report["factualGroundingValidationAccepted"])
        self.assertEqual(report["scannerDerivedNewProductClaimsIntroduced"], [])
        self.assertTrue(report["offlineRevalidationPossible"])
        self.assertTrue(report["replacementCreatorCallCanBeAvoided"])
        self.assertEqual(report["originalRejectionComponent"], "server_scanner")
        self.assertEqual(report["prototypeId"], "winning_card")

    def test_no_replacement_creator_after_salvage(self) -> None:
        state = _production_mixed_partial_state()
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = _production_winning_card_state()[REJECTED_CREATOR_PARSED_INDEX_KEY]
        state["candidates"].update(_production_winning_card_state()["candidates"])
        state["creatorDiagnosticsByCandidate"] = _production_winning_card_state()["creatorDiagnosticsByCandidate"]
        state["strategyFoundation"] = _strategy()
        offline_revalidate_and_accept_rejected_creator(state, candidate_id=_CANDIDATE_ID, product_name="אורי לב")
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["resumePlanByPrototype"]["winning_card"]["creatorAction"], "reuse")
        self.assertEqual(plan["resumePlanByPrototype"]["winning_card"]["judgeAction"], "dispatch")

    def test_resume_plan_counts_after_salvage(self) -> None:
        state = _salvaged_production_resume_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["remainingCreatorNormalCalls"], 3)
        self.assertEqual(plan["remainingJudgeNormalCalls"], 4)
        self.assertEqual(plan["requiredJudgeRepairCalls"], 0)
        self.assertEqual(plan["totalNormalCallsBeforeWinner"], 7)
        self.assertEqual(plan["conditionalWinnerNormalCalls"], 1)
        self.assertEqual(plan["minimumAdditionalPaidReasoningCalls"], 7)
        self.assertEqual(plan["maximumAdditionalPaidReasoningCallsWithoutFutureRepairs"], 8)
        self.assertEqual(plan["perInvocationCallLimit"], 3)

    def test_think_small_and_closest_plan_unchanged(self) -> None:
        state = _salvaged_production_resume_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["resumePlanByPrototype"]["think_small"]["creatorAction"], "reuse")
        self.assertEqual(plan["resumePlanByPrototype"]["think_small"]["judgeAction"], "reuse")
        self.assertEqual(plan["resumePlanByPrototype"]["closest"]["creatorAction"], "reuse")
        self.assertEqual(plan["resumePlanByPrototype"]["closest"]["judgeAction"], "resolved_unavailable")

    def test_offline_recovery_dry_run_with_fingerprint_guard(self) -> None:
        state = _production_winning_card_state()
        parsed = state[REJECTED_CREATOR_PARSED_INDEX_KEY][_CANDIDATE_ID]["parsed"]
        fingerprint = _parsed_fingerprint(parsed)
        result = recover_creator_grounding_offline(
            state,
            candidate_id=_CANDIDATE_ID,
            expected_fingerprint=fingerprint,
            dry_run=True,
        )
        self.assertTrue(result["recovered"])
        self.assertEqual(result["reason"], "builder2_creator_grounding_offline_recovery_dry_run_ok")

    def test_existing_unsupported_claim_tests_still_reject(self) -> None:
        strategy = _strategy()
        candidate = _winning_card_rejected(mechanism="Feedback enters and the ad improves after revision.")
        with self.assertRaises(Builder2TournamentError):
            validate_creator_evidence_grounding(candidate, strategy_foundation=strategy)

    def test_resume_inspector_marks_offline_revalidatable(self) -> None:
        state = _production_mixed_partial_state()
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = _production_winning_card_state()[REJECTED_CREATOR_PARSED_INDEX_KEY]
        state["candidates"].update(_production_winning_card_state()["candidates"])
        state["creatorDiagnosticsByCandidate"] = _production_winning_card_state()["creatorDiagnosticsByCandidate"]
        state["strategyFoundation"] = _strategy()
        from unittest.mock import patch

        with patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True), patch(
            "engine.builder2_complete_ad_resume_inspect._read_raw",
            return_value=copy.deepcopy(state),
        ), patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={}):
            report = inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertTrue(report["rejectedCreatorOfflineRevalidatable"])


class TestNegationClassifierDetails(unittest.TestCase):
    def test_production_sentence_classified_as_explicit_negation(self) -> None:
        text = PRODUCTION_MECHANISM_SCAN_SUMMARY
        occurrences = scan_capability_occurrences(
            text,
            allowed_capabilities=[],
            field_path="creatorReport.mechanismScanSummary",
        )
        optimization = [item for item in occurrences if item["capability"] == "optimization"]
        self.assertTrue(optimization)
        self.assertEqual(optimization[0]["occurrenceClassification"], CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION)


if __name__ == "__main__":
    unittest.main()
