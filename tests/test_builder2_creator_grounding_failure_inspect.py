"""
Builder2 Creator grounding failure inspector tests.
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_creator_grounding_failure_inspect import (
    inspect_creator_grounding_failure,
    inspect_creator_grounding_failure_for_job,
)
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_strategy_evidence_grounding_contract import BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from tests.builder2_methodology_fixtures import (
    complete_ad_creator_extras,
    logo_policy_creator_extras,
    methodology_candidate_extras,
    methodology_strategy_evidence_extras,
    methodology_strategy_identity_for,
)
from tests.test_builder2_mixed_partial_resume import _production_mixed_partial_state
from tests.test_builder2_tournament import _candidate, _deep_merge


_SPARSE_HE = "סוכן פרסום דיגיטלי"
_CANDIDATE_ID = "cand-1-winning_card-1-577b91f2"
_JOB_ID = "e369b792-9988-4087-b054-38a713966918"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _strategy() -> Dict[str, Any]:
    strategy = validate_strategy_foundation(
        methodology_strategy_evidence_extras(
            tournament_id="9d789e1e-7e4a-4ef4-b72e-642da8083788",
            product_name="אורי לב",
            product_description=_SPARSE_HE,
        ),
        product_name="אורי לב",
        product_description=_SPARSE_HE,
    )
    strategy["productNameResolved"] = "אורי לב"
    return strategy


def _winning_card_rejected(*, mechanism: str, self_report: list[str] | None = None) -> Dict[str, Any]:
    strategy = _strategy()
    candidate = _deep_merge(_candidate("winning_card"), methodology_candidate_extras("winning_card", strategy=strategy))
    candidate.update(methodology_strategy_identity_for(strategy))
    relative_advantage = _clean(strategy["relativeAdvantage"]["statement"])
    candidate["creatorReport"] = _deep_merge(
        candidate["creatorReport"],
        {
            "problemPerception": strategy["problemPerception"]["statement"],
            "relativeAdvantage": relative_advantage,
            "mechanismScanSummary": strategy["mechanismScan"]["discoveredMechanism"],
        },
    )
    ad_extras = complete_ad_creator_extras(
        product_name="אורי לב",
        language="he",
        relative_advantage_source=relative_advantage,
    )
    candidate["advertisingClosure"] = ad_extras["advertisingClosure"]
    candidate["advertisingSloganFormulation"] = ad_extras["advertisingSloganFormulation"]
    candidate.update(logo_policy_creator_extras(advertised_entity_name="אורי לב"))
    candidate["candidateId"] = _CANDIDATE_ID
    candidate["coreCreativeMechanism"] = mechanism
    candidate["conceptSummary"] = mechanism
    if self_report is not None:
        candidate["newProductClaimsIntroduced"] = list(self_report)
    return candidate


def _winning_card_state(*, mechanism: str, self_report: list[str] | None = None) -> Dict[str, Any]:
    state = _production_mixed_partial_state()
    state["jobId"] = _JOB_ID
    parsed = _winning_card_rejected(mechanism=mechanism, self_report=self_report)
    state["strategyFoundation"] = _strategy()
    state["productDescription"] = _SPARSE_HE
    state["productName"] = "אורי לב"
    state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {
        _CANDIDATE_ID: {
            "candidateId": _CANDIDATE_ID,
            "prototypeId": "winning_card",
            "roundIndex": 1,
            "attemptNumber": 1,
            "parsed": parsed,
            "failureReason": "builder2_creator_validation_failed:newProductClaimsIntroduced",
            "callType": "normal",
            "storedAt": "2026-05-20T00:00:00+00:00",
        }
    }
    state["candidates"][_CANDIDATE_ID] = {
        "candidateId": _CANDIDATE_ID,
        "prototypeId": "winning_card",
        "validationStatus": "creator_rejected",
        "status": "creator_rejected",
        "failureReason": "builder2_creator_validation_failed:newProductClaimsIntroduced",
        "attemptNumber": 1,
    }
    state["creatorDiagnosticsByCandidate"] = {
        _CANDIDATE_ID: {
            "responseReceived": True,
            "responseLength": 8534,
            "failureReason": "builder2_creator_validation_failed:newProductClaimsIntroduced",
            "failureFieldPaths": ["newProductClaimsIntroduced"],
        }
    }
    return state


class TestCreatorGroundingFailureInspect(unittest.TestCase):
    def test_rejected_response_found_without_mutation(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        before = copy.deepcopy(state)
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertEqual(state, before)
        self.assertFalse(report["stateMutated"])
        self.assertTrue(report["parsedResponseAvailable"])
        self.assertEqual(report["responseLocation"], REJECTED_CREATOR_PARSED_INDEX_KEY)

    def test_raw_and_parsed_availability_reported_independently(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["parsedResponseAvailable"])
        self.assertFalse(report["rawResponseAvailable"])
        self.assertIsNone(report["responseFingerprint"])
        self.assertTrue(report["parsedResponseFingerprint"])

    def test_exact_new_product_claims_preserved(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertIn("feedback", report["newProductClaimsIntroduced"])
        self.assertIn("revision", report["newProductClaimsIntroduced"])

    def test_source_fields_reported_for_claims(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["claimAnalyses"])
        self.assertTrue(any(item.get("fieldPath") for item in report["claimAnalyses"]))

    def test_actual_unsupported_capability_remains_rejected(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["rejectionStillValidUnderCurrentContract"])
        self.assertEqual(report["rejectionComponent"], "server_scanner")
        self.assertEqual(report["inspectionConclusion"], "valid_rejection_actual_unsupported_claim")

    def test_grounded_identity_not_treated_as_new_claim(self) -> None:
        mechanism = "אורי לב is shown as a named digital-advertising professional address on a playing card."
        state = _winning_card_state(mechanism=mechanism)
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertEqual(report["newProductClaimsIntroduced"], [])
        self.assertFalse(report["rejectionStillValidUnderCurrentContract"])

    def test_visual_metaphor_distinguished_from_capability(self) -> None:
        mechanism = "A playing card flips in slow motion while feedback icons orbit the frame."
        state = _winning_card_state(mechanism=mechanism)
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        if report["newProductClaimsIntroduced"]:
            self.assertTrue(
                any(item.get("isVisualMetaphorNotProductAssertion") for item in report["claimAnalyses"])
                or report["inspectionConclusion"] == "false_positive_visual_or_strategic_metaphor"
            )

    def test_category_convention_not_promoted_to_product_fact(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["categoryConventionDependencies"])

    def test_empty_claim_array_handled(self) -> None:
        state = _winning_card_state(mechanism="A playing card reveals the product name clearly.", self_report=[])
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertEqual(report["newProductClaimsIntroduced"], [])

    def test_self_report_vs_scanner_disagreement_exposed(self) -> None:
        state = _winning_card_state(
            mechanism="A card reveals the product name.",
            self_report=["optimization"],
        )
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["contradictionDetected"])
        self.assertIn("optimization", report["creatorProvidedNewProductClaimsIntroduced"])

    def test_structural_and_semantic_rejection_separated(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertTrue(report["responseStructurallyValidUnderCurrentContract"])
        self.assertEqual(report["structuralErrorCount"], 0)
        self.assertFalse(report["factualGroundingValidationAccepted"])

    def test_missing_parsed_reported_unknown_not_empty_object(self) -> None:
        state = _production_mixed_partial_state()
        report = inspect_creator_grounding_failure(
            state,
            candidate_id="cand-missing",
            prototype_id="winning_card",
        )
        self.assertFalse(report["parsedResponseAvailable"])
        self.assertIsNone(report["newProductClaimsIntroduced"])
        self.assertEqual(report["inspectionConclusion"], "response_unavailable")

    def test_zero_mutations_and_paid_calls(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertEqual(report["paidCalls"], 0)
        self.assertEqual(report["openAICalls"], 0)
        self.assertFalse(report["stateMutated"])

    def test_offline_revalidation_blocked_reason_exposed(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertFalse(report["offlineRevalidationPossible"])
        self.assertFalse(report["replacementCreatorCallCanBeAvoided"])
        self.assertIn("newProductClaimsIntroduced", _clean(report["offlineRevalidationBlockedReason"]))

    def test_replacement_creator_currently_planned(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["resumePlanByPrototype"]["winning_card"]["creatorAction"], "dispatch")
        self.assertTrue(report["replacementCreatorCallCurrentlyPlanned"])

    def test_strategy_contract_version_reported(self) -> None:
        state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
        report = inspect_creator_grounding_failure(state, candidate_id=_CANDIDATE_ID, prototype_id="winning_card")
        self.assertEqual(report["strategyEvidenceGroundingContractVersion"], BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION)

    def test_memory_store_job_wrapper(self) -> None:
        enable_memory_store()
        try:
            state = _winning_card_state(mechanism="Feedback enters and the ad improves after revision.")
            save_tournament_state(_JOB_ID, state)
            report = inspect_creator_grounding_failure_for_job(_JOB_ID, candidate_id=_CANDIDATE_ID)
            self.assertTrue(report["ok"])
            self.assertTrue(report["parsedResponseAvailable"])
        finally:
            disable_memory_store()

    def test_builder1_unchanged(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder1*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder2_creator_grounding_failure_inspect", text)


class TestPreservationRegression(unittest.TestCase):
    def test_six_prototypes_remain_configurable(self) -> None:
        self.assertEqual(len(DEFAULT_ACTIVE_PROTOTYPE_IDS), 6)
