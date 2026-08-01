"""
Builder2 Creator metaphor regression tests — slogan-quality commit must not break literal-symbol validation.
"""
from __future__ import annotations

import copy
import unittest

from engine.builder2_advertising_slogan_quality_contract import (
    BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
    validate_creator_advertising_slogan_formulation,
)
from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
from engine.builder2_complete_ad_creator_recovery import run_offline_creator_recovery_batch
from engine.builder2_metaphorical_embodiment_contract import (
    BUILDER2_LITERAL_SYMBOL_DISPOSITION_CONTRACT_VERSION,
    LITERAL_SYMBOL_DISPOSITION_FIELD,
    VALID_LITERAL_SYMBOL_DISPOSITIONS,
    validate_creator_metaphorical_embodiment,
)
from engine.builder2_methodology_validation import validate_creator_methodology
from engine.builder2_new_format_config import NORMAL_REASONING_CALL_BUDGET
from engine.builder2_tournament_config import (
    resolve_builder2_active_prototype_ids,
    resolve_builder2_tournament_max_rounds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import new_tournament_state
from tests.builder2_methodology_fixtures import (
    complete_ad_creator_extras,
    metaphorical_embodiment_creator_extras,
    single_slogan_contract_extras,
)


def _production_shaped_metaphor_extras() -> dict:
    return {
        "metaphoricalEmbodiment": {
            "strategicPerception": "Personal warmth and direct fit matter more than generic scale.",
            "obviousLiteralVisualSymbols": ["CRM dashboard", "lead counter"],
            "literalSymbolsRejectedOrTransformed": (
                "Show closeness through physical presence and eye contact instead of interface metrics."
            ),
            "creativeEmbodimentMode": "transformed_action_or_motion",
            "embodimentSubjectOrWorld": "A consultant meeting a client in a quiet office",
            "physicalEmbodiment": "The consultant closes the visible distance and listens closely",
            "embodiedStrategicRelationship": "Human proximity expresses strategic fit",
            "visiblePhysicalRelationship": "The closing distance is readable before copy",
            "transformationMechanism": "Physical closeness replaces dashboard proof",
            "whyTheVisualIsNotLiteralExplanation": "Bodies carry the meaning, not a screen",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan completes the closeness already visible",
        },
        "visualBridgeAssessment": {
            "centralVisibleDetail": "The consultant leaning in to listen",
            "sloganConnectionToVisibleDetail": "The slogan names the closeness already visible",
            "sloganConnectionToRelativeAdvantage": "The slogan closes the fit advantage",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
    }


def _base_candidate(**overrides) -> dict:
    candidate = {
        "schemaVersion": "builder2_candidate_v1",
        "methodologyVersion": "builder2_methodology_v1",
        "prototypeId": "closest",
        "structureType": "single_scene",
        "visualParallelType": "replacement",
        "coreCreativeMechanism": "Physical closeness expresses fit",
        "visualMechanism": "Distance closes between two people",
        "coreVisualIdea": "Two people close the visible distance between them",
        "openingFrameDescription": "A quiet office with two people facing each other",
        **single_slogan_contract_extras(),
        **complete_ad_creator_extras(),
        **_production_shaped_metaphor_extras(),
    }
    candidate.update(overrides)
    return candidate


def _test_state(*, active_prototype_ids: list[str] | None = None) -> dict:
    state = new_tournament_state(
        job_id="job-test",
        language="he",
        active_prototype_ids=active_prototype_ids or ["closest"],
        random_seed="metaphor-regression-test",
    )
    state["strategyFoundation"] = {
        "productNameResolved": "אורי לב",
        "relativeAdvantage": {"statement": "קרבה אישית שמבינה את הלקוח"},
        "language": "he",
    }
    state["productName"] = "אורי לב"
    state["productDescription"] = "desc"
    return state


class TestBuilder2CreatorMetaphorRegression(unittest.TestCase):
    def test_declared_symbols_without_reject_keyword_accepted_when_execution_clean(self) -> None:
        candidate = _base_candidate()
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_declared_symbols_alone_do_not_count_as_execution_hits(self) -> None:
        candidate = _base_candidate()
        metaphor = candidate["metaphoricalEmbodiment"]
        hits_before = metaphor["obviousLiteralVisualSymbols"]
        self.assertTrue(hits_before)
        validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_literal_dashboard_execution_without_transformation_rejected(self) -> None:
        candidate = _base_candidate(
            coreVisualIdea="CRM dashboard with lead counters",
            visualMechanism="Numbers rise on the interface",
        )
        candidate["metaphoricalEmbodiment"] = {
            "strategicPerception": "Lead volume is the message",
            "obviousLiteralVisualSymbols": ["CRM dashboard"],
            "literalSymbolsRejectedOrTransformed": "Uses the dashboard directly",
            "creativeEmbodimentMode": "external_metaphor",
            "embodimentSubjectOrWorld": "Office screen",
            "physicalEmbodiment": "Dashboard counters fill the frame",
            "embodiedStrategicRelationship": "Higher numbers mean success",
            "visiblePhysicalRelationship": "Counters dominate the screen",
            "transformationMechanism": "Shows the dashboard unchanged",
            "whyTheVisualIsNotLiteralExplanation": "The dashboard itself explains the claim",
            "understandableBeforeSlogan": True,
            "sloganBridgeToBusinessMeaning": "The slogan labels the dashboard",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_not_present_disposition_requires_no_declared_symbols(self) -> None:
        candidate = _base_candidate()
        candidate["metaphoricalEmbodiment"] = {
            **candidate["metaphoricalEmbodiment"],
            LITERAL_SYMBOL_DISPOSITION_FIELD: "not_present",
            "obviousLiteralVisualSymbols": ["CRM dashboard"],
        }
        state = _test_state()
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(
                candidate,
                assigned_prototype_id="closest",
                tournament_state=state,
            )

    def test_rejected_disposition_accepts_declared_symbols_with_evidence(self) -> None:
        candidate = _base_candidate()
        candidate["metaphoricalEmbodiment"] = {
            **candidate["metaphoricalEmbodiment"],
            LITERAL_SYMBOL_DISPOSITION_FIELD: "rejected",
        }
        state = _test_state()
        validate_creator_metaphorical_embodiment(
            candidate,
            assigned_prototype_id="closest",
            tournament_state=state,
        )

    def test_transformed_disposition_requires_evidence(self) -> None:
        candidate = _base_candidate()
        candidate["metaphoricalEmbodiment"] = {
            **candidate["metaphoricalEmbodiment"],
            LITERAL_SYMBOL_DISPOSITION_FIELD: "transformed",
            "literalSymbolsRejectedOrTransformed": "",
            "transformationMechanism": "",
            "whyTheVisualIsNotLiteralExplanation": "",
            "physicalEmbodiment": "",
        }
        state = _test_state()
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(
                candidate,
                assigned_prototype_id="closest",
                tournament_state=state,
            )

    def test_untransformed_disposition_rejected(self) -> None:
        candidate = _base_candidate()
        candidate["metaphoricalEmbodiment"] = {
            **candidate["metaphoricalEmbodiment"],
            LITERAL_SYMBOL_DISPOSITION_FIELD: "untransformed",
        }
        state = _test_state()
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(
                candidate,
                assigned_prototype_id="closest",
                tournament_state=state,
            )

    def test_legacy_declared_symbols_without_explanation_rejected(self) -> None:
        candidate = _base_candidate()
        candidate["metaphoricalEmbodiment"] = {
            **candidate["metaphoricalEmbodiment"],
            "literalSymbolsRejectedOrTransformed": "",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_metaphorical_embodiment(candidate, assigned_prototype_id="closest")

    def test_slogan_quality_and_metaphor_both_validate(self) -> None:
        relative_advantage = "קרבה אישית שמבינה את הלקוח"
        candidate = _base_candidate()
        candidate.update(
            complete_ad_creator_extras(
                product_name="אורי לב",
                slogan_text="קרוב אליך יותר ממה שחשבת",
                relative_advantage_source=relative_advantage,
            )
        )
        candidate.update(metaphorical_embodiment_creator_extras())
        candidate["metaphoricalEmbodiment"][LITERAL_SYMBOL_DISPOSITION_FIELD] = "rejected"
        state = _test_state()
        state["strategyFoundation"]["relativeAdvantage"] = {"statement": relative_advantage}
        validate_creator_metaphorical_embodiment(
            candidate,
            assigned_prototype_id="closest",
            tournament_state=state,
        )
        validate_creator_advertising_slogan_formulation(
            candidate,
            strategy_foundation=state["strategyFoundation"],
            product_name="אורי לב",
        )

    def test_all_six_prototype_prompts_include_disposition_contract(self) -> None:
        for prototype_id in resolve_builder2_active_prototype_ids():
            prompt = build_creator_required_keys_prompt_text(prototype_id=prototype_id)
            self.assertIn(LITERAL_SYMBOL_DISPOSITION_FIELD, prompt)
            for token in VALID_LITERAL_SYMBOL_DISPOSITIONS:
                self.assertIn(token, prompt)

    def test_shared_prompt_contract_is_prototype_agnostic(self) -> None:
        prompts = [
            build_creator_required_keys_prompt_text(prototype_id=prototype_id)
            for prototype_id in resolve_builder2_active_prototype_ids()
        ]
        self.assertGreaterEqual(len(prompts), 6)
        for prompt in prompts:
            self.assertIn("literalSymbolDisposition", prompt)
            self.assertIn("advertisingSloganFormulation", prompt)

    def test_slogan_prompt_addition_preserves_metaphor_wording(self) -> None:
        prompt = build_creator_required_keys_prompt_text(prototype_id="closest")
        self.assertIn("literalSymbolsRejectedOrTransformed", prompt)
        self.assertIn("obviousLiteralVisualSymbols", prompt)
        self.assertIn("Creative embodiment (mandatory)", prompt)

    def test_advertising_slogan_quality_still_mandatory(self) -> None:
        candidate = _base_candidate()
        candidate.pop("advertisingSloganFormulation", None)
        state = _test_state()
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(
                candidate,
                assigned_prototype_id="closest",
                strategy_foundation=state["strategyFoundation"],
                tournament_state=state,
            )

    def test_descriptive_slogan_still_rejected(self) -> None:
        candidate = _base_candidate()
        candidate["advertisingSloganFormulation"] = {
            **candidate["advertisingSloganFormulation"],
            "finalSloganText": candidate["advertisingClosure"]["sloganText"],
            "merelyDescriptive": True,
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_advertising_slogan_formulation(
                candidate,
                strategy_foundation={"relativeAdvantage": {"statement": "יתרון"}},
                product_name="אורי לב",
            )

    def test_offline_recovery_batch_idempotent(self) -> None:
        from tests.test_builder2_creator_slogan_repair import _candidate_with_slogan

        state = _test_state(active_prototype_ids=list(resolve_builder2_active_prototype_ids()))
        state.pop("metaphoricalEmbodimentContractVersion", None)
        relative_advantage = "קרבה אישית שמבינה את הלקוח"
        state["strategyFoundation"]["relativeAdvantage"] = {"statement": relative_advantage}
        candidate = _candidate_with_slogan(
            prototype_id="closest",
            product_name="אורי לב",
            slogan_text="קרוב אליך יותר ממה שחשבת",
        )
        candidate["advertisingSloganFormulation"]["relativeAdvantageSource"] = relative_advantage
        from tests.builder2_methodology_fixtures import logo_policy_creator_extras

        candidate.update(logo_policy_creator_extras(advertised_entity_name="אורי לב"))
        candidate.update(_production_shaped_metaphor_extras())
        candidate_id = "cand-closest-1"
        candidate["candidateId"] = candidate_id
        state["rejectedCreatorParsedResponses"] = {
            candidate_id: {
                "candidateId": candidate_id,
                "prototypeId": "closest",
                "roundIndex": 1,
                "attemptNumber": 1,
                "parsed": copy.deepcopy(candidate),
                "failureReason": (
                    "builder2_creator_literal_execution_without_transformation:"
                    "metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed"
                ),
            }
        }
        first = run_offline_creator_recovery_batch(state)
        self.assertTrue(first["stateMutated"], first.get("results"))
        self.assertEqual(first["acceptedAfter"], 1)
        second = run_offline_creator_recovery_batch(state)
        self.assertFalse(second["stateMutated"])
        self.assertEqual(second["acceptedAfter"], 1)

    def test_configurable_budget_and_rounds_unchanged(self) -> None:
        self.assertEqual(NORMAL_REASONING_CALL_BUDGET, 14)
        self.assertEqual(resolve_builder2_tournament_max_rounds(), 1)

    def test_new_tournament_state_stamps_both_contract_versions(self) -> None:
        state = new_tournament_state(
            job_id="job-v",
            language="he",
            active_prototype_ids=["closest"],
            random_seed="version-stamp-test",
        )
        self.assertEqual(
            state["advertisingSloganQualityContractVersion"],
            BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
        )
        self.assertEqual(
            state["metaphoricalEmbodimentContractVersion"],
            BUILDER2_LITERAL_SYMBOL_DISPOSITION_CONTRACT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
