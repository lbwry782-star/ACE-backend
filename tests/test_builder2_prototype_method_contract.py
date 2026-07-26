"""
Builder2 prototype method contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_creator import (
    collect_creator_structural_errors,
    generate_creator_candidate,
    validate_creator_candidate,
)
from engine.builder2_methodology_contract import CREATIVE_STAGE_ORDER
from engine.builder2_methodology_validation import validate_creator_methodology
from engine.builder2_prototype_method_contract import (
    PROTOTYPE_METHOD_CONTRACT_VERSION,
    build_prototype_method_contract,
)
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_prompts import build_creator_prompt, build_judge_prompt
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from tests.test_builder2_tournament import (
    TournamentMockLLM,
    _candidate,
    _judgment,
    _strategy,
    _winner_plan_from_prompt,
)


def _valid(prototype_id: str = "closest") -> Dict[str, Any]:
    return validate_creator_candidate(
        _candidate(prototype_id),
        assigned_prototype_id=prototype_id,
        prototype_display_name=require_prototype(prototype_id).display_name,
        strategy_foundation=_strategy(),
    )


class PrototypeMethodStressMockLLM:
    _VARIANTS = [
        None,
        {},
        {"methodSummary": "Restated method only."},
        {"applicationToCurrentProblem": "Applied to current problem."},
        {"methodSummary": "A", "applicationToCurrentProblem": "B", "whyThisIsNotLiteralImitation": "C"},
        "not-an-object",
    ]

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
        self.calls.append(role)
        if role == "builder2_strategy":
            return _strategy()
        if role == "builder2_creator":
            prototype_id = "closest"
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if f"Assigned prototype ID: {pid}" in prompt:
                    prototype_id = pid
                    break
            idx = DEFAULT_ACTIVE_PROTOTYPE_IDS.index(prototype_id)
            candidate = deepcopy(_candidate(prototype_id, prompt=prompt))
            variant = self._VARIANTS[idx % len(self._VARIANTS)]
            if variant is None:
                candidate.pop("prototypeMethodApplication", None)
            elif isinstance(variant, dict):
                candidate["prototypeMethodApplication"] = variant
            else:
                candidate["prototypeMethodApplication"] = variant
            if prototype_id == "think_small":
                candidate["creatorReport"]["problemPerception"] = (
                    "The real weakness is limited size, and the ad inverts it."
                )
            return candidate
        if role == "builder2_judge":
            candidate_id = "unknown"
            for token in prompt.split():
                if token.startswith("cand-"):
                    candidate_id = token.strip()
                    break
            return _judgment(candidate_id, total_hint=80)
        if role == "builder2_winner":
            return _winner_plan_from_prompt(prompt)
        raise AssertionError(f"unexpected role {role}")


class TestPrototypeMethodContract(unittest.TestCase):
    def test_server_attaches_contract(self) -> None:
        out = _valid("closest")
        contract = out["prototypeMethodContract"]
        self.assertEqual(contract["prototypeId"], "closest")
        self.assertEqual(contract["methodVersion"], PROTOTYPE_METHOD_CONTRACT_VERSION)
        self.assertTrue(contract["assignedByServer"])
        self.assertEqual(contract["canonicalMethodSummary"], require_prototype("closest").reusable_method)

    def test_model_cannot_override_contract(self) -> None:
        cand = _candidate("closest")
        cand["prototypeMethodContract"] = {"prototypeId": "hacked"}
        out = _valid_from_raw(cand)
        self.assertEqual(out["prototypeMethodContract"]["prototypeId"], "closest")

    def test_legacy_generic_object_stored_as_diagnostics(self) -> None:
        cand = _candidate("closest")
        cand["prototypeMethodApplication"] = {
            "methodSummary": "Generic summary",
            "applicationToCurrentProblem": "Generic application",
        }
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertNotIn("prototypeMethodApplication", out)
        self.assertTrue(out["creatorAttestations"]["prototypeMethodApplicationReceived"])
        self.assertFalse(out["creatorAttestations"]["authoritative"])


def _valid_from_raw(cand: Dict[str, Any], prototype_id: str = "closest") -> Dict[str, Any]:
    return validate_creator_candidate(
        cand,
        assigned_prototype_id=prototype_id,
        prototype_display_name=require_prototype(prototype_id).display_name,
        strategy_foundation=_strategy(),
    )


class TestRemovedGenericHardGates(unittest.TestCase):
    def test_missing_generic_object_accepted(self) -> None:
        cand = _candidate("closest")
        cand.pop("prototypeMethodApplication", None)
        validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())

    def test_only_method_summary_accepted(self) -> None:
        cand = _candidate("closest")
        cand["prototypeMethodApplication"] = {"methodSummary": "Only summary"}
        _valid_from_raw(cand)

    def test_string_generic_object_accepted(self) -> None:
        calls: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append("x")
            cand = _candidate("closest")
            cand["prototypeMethodApplication"] = "legacy-string"
            return cand

        generate_creator_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["metrics"].get("creatorRepairCalls", 0), 0)


class TestPrototypeSpecificEvidence(unittest.TestCase):
    PROTOTYPES = (
        ("winning_card", "winningCardApplication"),
        ("summer_fan", "summerFanApplication"),
        ("forgot", "forgotApplication"),
        ("greenpeace_essential_pairing", "essentialPairingApplication"),
        ("closest", "closestApplication"),
        ("think_small", "thinkSmallApplication"),
    )

    def test_missing_application_object_rejected(self) -> None:
        for prototype_id, app_field in self.PROTOTYPES:
            cand = _candidate(prototype_id)
            cand.pop(app_field, None)
            with self.subTest(prototype_id=prototype_id):
                with self.assertRaises(Builder2TournamentError):
                    validate_creator_methodology(
                        cand,
                        assigned_prototype_id=prototype_id,
                        strategy_foundation=_strategy(),
                    )

    def test_missing_generic_summary_still_valid(self) -> None:
        for prototype_id, _ in self.PROTOTYPES:
            cand = _candidate(prototype_id)
            cand.pop("prototypeMethodApplication", None)
            with self.subTest(prototype_id=prototype_id):
                validate_creator_methodology(
                    cand,
                    assigned_prototype_id=prototype_id,
                    strategy_foundation=_strategy(),
                )


class TestPromptAlignment(unittest.TestCase):
    def test_creator_prompt_no_generic_required_keys(self) -> None:
        prompt = build_creator_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype=require_prototype("closest"),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="image_to_video",
        )
        self.assertNotIn("prototypeMethodApplication", prompt.split("Required keys:")[1][:400])
        self.assertIn("Do not restate or redefine the prototype method", prompt)

    def test_judge_uses_contract_not_creator_summary(self) -> None:
        candidate = _valid("closest")
        prompt = build_judge_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype=require_prototype("closest"),
            candidate=candidate,
            candidate_id="cand-1",
        )
        self.assertIn("Canonical prototype method contract (authoritative)", prompt)
        self.assertIn("Do NOT reward repetition of the prototype description", prompt)


class TestStructuralRepairAggregation(unittest.TestCase):
    def test_three_missing_fields_collected(self) -> None:
        cand = _candidate("closest")
        cand.pop("closestApplication", None)
        cand["verbalPotential"] = {}
        cand["essenceExtreme"] = {}
        errors = collect_creator_structural_errors(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertGreaterEqual(len(errors), 3)
        joined = " ".join(errors)
        self.assertIn("closestApplication", joined)
        self.assertIn("verbalPotential", joined)
        self.assertIn("essenceExtreme", joined)

    def test_one_repair_receives_all_paths(self) -> None:
        repair_prompts: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            prompt = kwargs.get("prompt", "")
            if "Creator repair role" in prompt:
                repair_prompts.append(prompt)
                return _candidate("closest")
            bad = _candidate("closest")
            bad.pop("closestApplication", None)
            bad["verbalPotential"] = {}
            return bad

        generate_creator_candidate(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype_id="closest",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(len(repair_prompts), 1)
        self.assertIn("verbalPotential", repair_prompts[0])
        self.assertIn("closestApplication", repair_prompts[0])
        self.assertEqual(state["metrics"].get("creatorRepairCalls"), 1)


class TestProductionRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    def test_six_valid_candidates_reach_judge(self) -> None:
        llm = PrototypeMethodStressMockLLM()
        run_builder2_tournament(
            job_id="job-prototype-regression",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-prototype",
        )
        state = load_tournament_state("job-prototype-regression")
        assert state is not None
        metrics = state.get("metrics") or {}
        accepted = [c for c in state.get("candidates", {}).values() if c.get("validationStatus") == "accepted"]
        self.assertEqual(len(accepted), len(DEFAULT_ACTIVE_PROTOTYPE_IDS))
        self.assertEqual(metrics.get("creatorCalls"), 6)
        self.assertEqual(metrics.get("creatorRepairCalls"), 0)
        self.assertEqual(metrics.get("judgeCalls"), 6)
        self.assertEqual(metrics.get("totalReasoningCalls"), 14)


if __name__ == "__main__":
    unittest.main()
