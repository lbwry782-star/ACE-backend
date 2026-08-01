"""
Builder2 Strategy evidence grounding contract tests.
"""
from __future__ import annotations

import copy
import unittest

from engine.builder2_complete_ad_contract import apply_semantic_eligibility_rules
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_strategy_evidence_grounding_contract import (
    BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
    apply_strategy_evidence_grounding,
    build_default_judge_factual_grounding_assessment,
    build_product_input_audit,
    detect_capabilities_in_text,
    scan_texts_for_unsupported_capabilities,
    validate_creator_evidence_grounding,
    validate_strategy_evidence_grounding,
)
from engine.builder2_tournament_contracts import Builder2TournamentError, STRATEGY_SCHEMA_VERSION
from engine.builder2_methodology_validation import validate_judge_methodology
from tests.builder2_methodology_fixtures import (
    methodology_candidate_extras,
    methodology_judgment_extras,
    methodology_strategy_evidence_extras,
    methodology_strategy_extras,
)
from tests.test_builder2_tournament import _candidate, _strategy


SPARSE_PRODUCT_DESCRIPTION = (
    "אורי לב is an AI application that creates advertising ideas for small businesses."
)
FEEDBACK_PRODUCT_DESCRIPTION = (
    "An AI application that creates advertisements and accepts user feedback to produce a revised advertisement."
)


def _sparse_strategy(*, with_feedback_claim: bool = False) -> dict:
    description = FEEDBACK_PRODUCT_DESCRIPTION if with_feedback_claim else SPARSE_PRODUCT_DESCRIPTION
    strategy = methodology_strategy_evidence_extras(
        tournament_id="grounding-test",
        product_name="אורי לב",
        product_description=description,
    )
    if with_feedback_claim:
        strategy["relativeAdvantage"]["statement"] = (
            "The product improves the advertisement after receiving user feedback."
        )
        strategy["relativeAdvantage"]["relativeAdvantageInferenceLevel"] = "explicit"
        strategy["relativeAdvantage"]["relativeAdvantageFactuallyGrounded"] = True
    return strategy


class TestProductInputAudit(unittest.TestCase):
    def test_sparse_input_marks_unknown_market_and_sparse_density(self) -> None:
        audit = build_product_input_audit(
            product_name="אורי לב",
            product_description=SPARSE_PRODUCT_DESCRIPTION,
        )
        self.assertEqual(audit["productMarketStatus"], "unknown")
        self.assertIn(audit["productInformationDensity"], {"sparse", "minimal"})
        self.assertFalse(audit["feedbackCapabilitySupplied"])

    def test_explicit_feedback_input_is_detected(self) -> None:
        audit = build_product_input_audit(
            product_name="ACE",
            product_description=FEEDBACK_PRODUCT_DESCRIPTION,
        )
        self.assertTrue(audit["feedbackCapabilitySupplied"])


class TestStrategyEvidenceGrounding(unittest.TestCase):
    def test_sparse_strategy_rejects_feedback_relative_advantage(self) -> None:
        strategy = _sparse_strategy()
        strategy["relativeAdvantage"]["statement"] = (
            "The product improves advertising after user feedback and revision rounds."
        )
        strategy = apply_strategy_evidence_grounding(
            strategy,
            product_name="אורי לב",
            product_description=SPARSE_PRODUCT_DESCRIPTION,
        )
        with self.assertRaises(Builder2TournamentError):
            validate_strategy_evidence_grounding(
                strategy,
                product_name="אורי לב",
                product_description=SPARSE_PRODUCT_DESCRIPTION,
            )

    def test_sparse_strategy_allows_access_advantage(self) -> None:
        strategy = _sparse_strategy()
        strategy["relativeAdvantage"]["statement"] = (
            "Small businesses gain access to AI-generated advertising ideas from supplied business information."
        )
        validated = validate_strategy_foundation(
            strategy,
            product_name="אורי לב",
            product_description=SPARSE_PRODUCT_DESCRIPTION,
        )
        self.assertEqual(
            validated["strategyEvidenceGrounding"]["contractVersion"],
            BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
        )
        self.assertTrue(validated["relativeAdvantage"]["relativeAdvantageFactuallyGrounded"])

    def test_explicit_feedback_may_enter_relative_advantage(self) -> None:
        strategy = _sparse_strategy(with_feedback_claim=True)
        validated = validate_strategy_foundation(
            strategy,
            product_name="ACE",
            product_description=FEEDBACK_PRODUCT_DESCRIPTION,
        )
        self.assertIn("feedback", validated["strategyEvidenceGrounding"]["allowedCapabilities"])

    def test_category_convention_alone_does_not_become_allowed_capability(self) -> None:
        strategy = _sparse_strategy()
        self.assertNotIn("revision", strategy["strategyEvidenceGrounding"]["allowedCapabilities"])


class TestCreatorAndJudgeGrounding(unittest.TestCase):
    def test_creator_rejects_unsupported_feature(self) -> None:
        strategy = validate_strategy_foundation(
            _sparse_strategy(),
            product_name="אורי לב",
            product_description=SPARSE_PRODUCT_DESCRIPTION,
        )
        candidate = _candidate("summer_fan")
        candidate.update(methodology_candidate_extras(prototype_id="summer_fan", strategy=strategy))
        candidate["coreCreativeMechanism"] = "Feedback enters and the ad improves after revision."
        with self.assertRaises(Builder2TournamentError):
            validate_creator_evidence_grounding(candidate, strategy_foundation=strategy)

    def test_judge_factual_gate_disqualifies_unsupported_claim(self) -> None:
        strategy = validate_strategy_foundation(
            _sparse_strategy(),
            product_name="אורי לב",
            product_description=SPARSE_PRODUCT_DESCRIPTION,
        )
        candidate = _candidate("summer_fan")
        candidate.update(methodology_candidate_extras(prototype_id="summer_fan", strategy=strategy))
        candidate["coreCreativeMechanism"] = "Feedback enters and the ad improves after revision."
        judgment = {
            "methodologyVersion": strategy["methodologyVersion"],
            "candidateId": "cand-test",
            "eligible": True,
            "disqualifiers": [],
            **methodology_judgment_extras(prototype_id="summer_fan"),
        }
        judgment["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment(
            notes="Unsupported feedback capability detected in candidate mechanism."
        )
        judgment["factualGroundingAssessment"]["productClaimFactuallyGrounded"] = False
        judgment["factualGroundingAssessment"]["noUnsupportedFeatureClaim"] = False
        validate_judge_methodology(
            judgment,
            candidate=candidate,
            strategy_foundation=strategy,
            product_input=strategy["strategyEvidenceGrounding"]["productInputAudit"],
        )
        self.assertFalse(judgment["eligible"])


class TestLegacyCompatibility(unittest.TestCase):
    def test_legacy_strategy_without_evidence_contract_still_validates(self) -> None:
        legacy = methodology_strategy_extras(tournament_id="legacy-test")
        legacy["schemaVersion"] = STRATEGY_SCHEMA_VERSION
        validate_strategy_foundation(legacy, compatibility_mode=True)


class TestInspectorHelpers(unittest.TestCase):
    def test_detect_feedback_in_pottery_metaphor_text(self) -> None:
        text = (
            "Feedback enters the scene, the clay is reshaped, and the advertising improves through revision."
        )
        self.assertIn("feedback", detect_capabilities_in_text(text))
        hits = scan_texts_for_unsupported_capabilities(
            [("coreCreativeMechanism", text)],
            allowed_capabilities=[],
        )
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
