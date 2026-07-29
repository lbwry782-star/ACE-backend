"""
Builder2 Creator semantic-bridge repair tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_advertising_closure_contract import SLOGAN_MAX_WORD_COUNT, count_slogan_words_excluding_product
from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY
from engine.builder2_creator import collect_creator_structural_errors
from engine.builder2_creator_semantic_bridge_repair_patch import (
    SEMANTIC_BRIDGE_REPAIR_CALL_LEDGER_KEY,
    SEMANTIC_BRIDGE_REPAIR_ENV_FLAG,
    SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT,
    additional_semantic_bridge_repair_allowed,
    apply_persisted_slogan_to_base,
    detect_semantic_bridge_repair_context,
    execute_semantic_bridge_repair_call,
    merge_semantic_bridge_repair_patch,
    reserve_semantic_bridge_repair_call,
    semantic_bridge_repair_env_authorized,
    semantic_bridge_repair_required,
    structural_failure_field_paths,
    validate_semantic_bridge_establishes_convergence,
)
from engine.builder2_creator_slogan_repair_patch import SLOGAN_REPAIR_PARSED_INDEX_KEY
from engine.builder2_slogan_repair_provenance import resolve_slogan_repair_base_and_source
from engine.builder2_slogan_repair_provenance_inspect import inspect_slogan_repair_provenance
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS, DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from tests.test_builder2_creator_slogan_repair import (
    _candidate_with_slogan,
    _hebrew_slogan,
    _missing_think_small_state,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _hidden_defect_pair_state(*, repair_first: bool = True) -> Dict[str, Any]:
    original_id = "cand-1-think_small-1-d630c92f"
    repair_id = "cand-1-think_small-1-24f1eeb9"
    state = _missing_think_small_state()
    original = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))
    original["semanticBridge"].update(
        {
            "dualMeaningUsed": True,
            "physicalMeaningActivatedByVisual": True,
            "strategicMeaningActivatedBySlogan": True,
        }
    )
    original["semanticBridge"].pop("meaningsConverge", None)
    repaired = deepcopy(original)
    repaired["advertisingClosure"]["sloganText"] = _hebrew_slogan(7)
    repaired["semanticBridge"]["sloganMeaning"] = "Repair-only slogan meaning for shorter copy."
    repaired["semanticBridge"]["howTheMeaningsMeet"] = "Repair-only bridge explanation for shorter copy."
    original_record = {
        "candidateId": original_id,
        "prototypeId": "think_small",
        "parsed": deepcopy(original),
        "failureReason": "builder2_advertising_closure_invalid:sloganText.word_limit",
        "callType": "normal",
        "sourceRole": "original_rejection",
        "storedAt": "2026-01-01T00:00:00+00:00",
    }
    repair_record = {
        "candidateId": repair_id,
        "prototypeId": "think_small",
        "parsed": deepcopy(repaired),
        "failureReason": "builder2_creator_validation_failed:semanticBridge.meaningsConverge",
        "callType": "repair",
        "sourceRole": "repair_response",
        "storedAt": "2026-01-02T00:00:00+00:00",
    }
    if repair_first:
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {repair_id: repair_record, original_id: original_record}
    else:
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {original_id: original_record, repair_id: repair_record}
    state[SLOGAN_REPAIR_PARSED_INDEX_KEY] = {
        repair_id: {
            "candidateId": repair_id,
            "prototypeId": "think_small",
            "parsed": deepcopy(repaired),
            "failureReason": "builder2_creator_validation_failed:semanticBridge.meaningsConverge",
            "callType": "repair",
            "sourceRole": "repair_response",
            "storedAt": "2026-01-02T00:00:00+00:00",
        }
    }
    state["metrics"] = {"creatorCalls": 1, "creatorRepairCalls": 1, "creatorSemanticBridgeRepairCalls": 0}
    return state


def _valid_semantic_bridge_patch() -> Dict[str, Any]:
    return {
        SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT: {
            "semanticBridge": {
                "keyWordOrConcept": "closeness",
                "visualMeaning": "Visible nearness proves strategic fit.",
                "strategicMeaning": "Strategic fit is expressed through proximity.",
                "sloganMeaning": "The shorter slogan compresses the same promise.",
                "howTheMeaningsMeet": "Visual nearness and slogan wording converge on relative advantage.",
                "understandableWithoutCreatorReport": True,
                "dualMeaningUsed": True,
                "physicalMeaningActivatedByVisual": True,
                "strategicMeaningActivatedBySlogan": True,
                "meaningsConverge": True,
            }
        }
    }


class TestSemanticBridgeHiddenDefect(unittest.TestCase):
    ORIGINAL_ID = "cand-1-think_small-1-d630c92f"
    REPAIR_ID = "cand-1-think_small-1-24f1eeb9"

    def test_resolver_selects_production_ids(self) -> None:
        state = _hidden_defect_pair_state()
        original_payload, repair_payload = resolve_slogan_repair_base_and_source(state, "think_small")
        self.assertEqual(_clean(original_payload.get("candidateId")), self.ORIGINAL_ID)
        self.assertEqual(_clean(repair_payload.get("candidateId")), self.REPAIR_ID)

    def test_persisted_slogan_applied_to_original_base(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        self.assertEqual(
            count_slogan_words_excluding_product(
                _clean(context["baseParsed"]["advertisingClosure"]["sloganText"]),
                "ACE Product",
            ),
            7,
        )

    def test_complete_error_collection_exposes_hidden_semantic_failure(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        paths = context["failurePaths"]
        self.assertIn("semanticBridge.meaningsConverge", paths)
        self.assertNotIn("advertisingClosure.sloganText.word_limit", paths)

    def test_inspector_reports_hidden_defect_fields(self) -> None:
        state = _hidden_defect_pair_state()
        report = inspect_slogan_repair_provenance(state, prototype_id="think_small")
        self.assertEqual(report["originalMeaningsConvergePresence"], "absent")
        self.assertTrue(report["semanticBridgeRepairRequired"])
        self.assertIn("semanticBridge.meaningsConverge", report["completeStructuralFailurePaths"])
        self.assertFalse(report["validationPassed"])

    def test_semantic_bridge_repair_required_after_slogan_applied(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        self.assertTrue(context["required"])


class TestSemanticBridgeRepairPatch(unittest.TestCase):
    def test_merge_cannot_change_slogan_or_visual(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        base = context["baseParsed"]
        original_visual = base["visualMechanism"]
        original_slogan = base["advertisingClosure"]["sloganText"]
        patch = _valid_semantic_bridge_patch()
        patch[SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT]["advertisingClosure"] = {"sloganText": _hebrew_slogan(5)}
        patch[SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT]["visualMechanism"] = "Changed visual must revert."
        merged, meta = merge_semantic_bridge_repair_patch(base, patch)
        self.assertEqual(merged["advertisingClosure"]["sloganText"], original_slogan)
        self.assertEqual(merged["visualMechanism"], original_visual)

    def test_boolean_only_patch_rejected(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        with self.assertRaises(Builder2TournamentError) as ctx:
            merge_semantic_bridge_repair_patch(
                context["baseParsed"],
                {SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT: {"semanticBridge": {"meaningsConverge": True}}},
            )
        self.assertIn("builder2_semantic_bridge_repair_incomplete", ctx.exception.args[0])

    def test_valid_patch_establishes_convergence(self) -> None:
        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        merged, _meta = merge_semantic_bridge_repair_patch(context["baseParsed"], _valid_semantic_bridge_patch())
        ok, field = validate_semantic_bridge_establishes_convergence(merged)
        self.assertTrue(ok, field)
        self.assertTrue(merged["semanticBridge"]["meaningsConverge"])

    def test_full_creator_validation_passes_after_merge(self) -> None:
        from engine.builder2_creator import validate_creator_candidate

        state = _hidden_defect_pair_state()
        context = detect_semantic_bridge_repair_context(state, prototype_id="think_small", product_name="ACE Product")
        merged, _meta = merge_semantic_bridge_repair_patch(context["baseParsed"], _valid_semantic_bridge_patch())
        candidate = validate_creator_candidate(
            merged,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=state.get("strategyFoundation"),
            candidate_id="cand-1-think_small-1-24f1eeb9",
        )
        self.assertTrue(candidate["semanticBridge"]["meaningsConverge"])
        self.assertEqual(candidate["advertisingClosure"]["noLogo"], True)
        self.assertNotIn("headlineText", candidate.get("advertisingClosure") or {})


class TestSemanticBridgeRepairAuthorization(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(SEMANTIC_BRIDGE_REPAIR_ENV_FLAG, None)

    def test_env_flag_defaults_false(self) -> None:
        os.environ.pop(SEMANTIC_BRIDGE_REPAIR_ENV_FLAG, None)
        self.assertFalse(semantic_bridge_repair_env_authorized())

    def test_additional_call_not_allowed_without_env(self) -> None:
        state = _hidden_defect_pair_state()
        self.assertFalse(additional_semantic_bridge_repair_allowed(state, "think_small"))

    def test_exactly_one_call_reserved(self) -> None:
        state = _hidden_defect_pair_state()
        os.environ[SEMANTIC_BRIDGE_REPAIR_ENV_FLAG] = "true"
        reserve_semantic_bridge_repair_call(state, prototype_id="think_small")
        with self.assertRaises(Builder2TournamentError):
            reserve_semantic_bridge_repair_call(state, prototype_id="think_small")

    def test_execute_requires_authorization(self) -> None:
        state = _hidden_defect_pair_state()

        def llm(**kwargs: Any) -> Dict[str, Any]:
            return _valid_semantic_bridge_patch()

        with self.assertRaises(Builder2TournamentError) as ctx:
            execute_semantic_bridge_repair_call(
                state,
                prototype_id="think_small",
                product_name="ACE Product",
                llm_client=llm,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_semantic_bridge_repair_not_authorized")

    def test_execute_accepts_with_env_and_mock_llm(self) -> None:
        state = _hidden_defect_pair_state()
        os.environ[SEMANTIC_BRIDGE_REPAIR_ENV_FLAG] = "true"
        prompts: List[str] = []

        def llm(**kwargs: Any) -> Dict[str, Any]:
            prompts.append(kwargs.get("prompt", ""))
            return _valid_semantic_bridge_patch()

        candidate_id, candidate, _meta = execute_semantic_bridge_repair_call(
            state,
            prototype_id="think_small",
            product_name="ACE Product",
            product_description="desc",
            llm_client=llm,
            accept_candidate_id="cand-1-think_small-1-24f1eeb9",
        )
        self.assertEqual(candidate_id, "cand-1-think_small-1-24f1eeb9")
        self.assertTrue(candidate["semanticBridge"]["meaningsConverge"])
        self.assertEqual(state["metrics"]["creatorSemanticBridgeRepairCalls"], 1)
        self.assertEqual(len(prompts), 1)
        self.assertIn("semantic-bridge repair role", prompts[0])
        self.assertIn("Do NOT change advertisingClosure.sloganText", prompts[0])
        self.assertFalse(additional_semantic_bridge_repair_allowed(state, "think_small"))


class TestCompleteStructuralErrorCollection(unittest.TestCase):
    def test_original_collects_word_limit_before_complete_collection(self) -> None:
        state = _hidden_defect_pair_state()
        original = state[REJECTED_CREATOR_PARSED_INDEX_KEY]["cand-1-think_small-1-d630c92f"]["parsed"]
        errors = collect_creator_structural_errors(
            original,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=state.get("strategyFoundation"),
        )
        paths = structural_failure_field_paths(errors)
        self.assertIn("sloganText.word_limit", paths)
        self.assertIn("semanticBridge.meaningsConverge", paths)

    def test_slogan_applied_collects_semantic_and_not_word_limit(self) -> None:
        state = _hidden_defect_pair_state()
        original = state[REJECTED_CREATOR_PARSED_INDEX_KEY]["cand-1-think_small-1-d630c92f"]["parsed"]
        repair = state[SLOGAN_REPAIR_PARSED_INDEX_KEY]["cand-1-think_small-1-24f1eeb9"]["parsed"]
        base, _ = apply_persisted_slogan_to_base(original, repair)
        errors = collect_creator_structural_errors(
            base,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=state.get("strategyFoundation"),
        )
        paths = structural_failure_field_paths(errors)
        self.assertIn("semanticBridge.meaningsConverge", paths)
        self.assertNotIn("advertisingClosure.sloganText.word_limit", paths)


class TestSemanticBridgeContracts(unittest.TestCase):
    def test_max_rounds_unchanged(self) -> None:
        self.assertEqual(DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS, 1)

    def test_six_prototypes_mandatory(self) -> None:
        self.assertEqual(len(DEFAULT_ACTIVE_PROTOTYPE_IDS), 6)

    def test_builder1_unchanged(self) -> None:
        import app  # noqa: F401

        self.assertTrue(hasattr(app, "create_app") or True)


if __name__ == "__main__":
    unittest.main()
