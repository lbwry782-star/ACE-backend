"""
Builder2 creative-order contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_creative_order_contract import (
    CREATIVE_ORDER_CONTRACT_VERSION,
    build_creative_order_contract,
    finalize_creator_order_metadata,
)
from engine.builder2_creator import generate_creator_candidate, validate_creator_candidate
from engine.builder2_methodology_contract import CREATIVE_STAGE_ORDER
from engine.builder2_methodology_validation import validate_creator_methodology
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_prompts import build_creator_prompt
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from tests.test_builder2_tournament import (
    TournamentMockLLM,
    _candidate,
    _judgment,
    _strategy,
    _winner_plan_from_prompt,
)


def _valid_candidate(prototype_id: str = "closest") -> Dict[str, Any]:
    return validate_creator_candidate(
        _candidate(prototype_id),
        assigned_prototype_id=prototype_id,
        prototype_display_name=require_prototype(prototype_id).display_name,
        strategy_foundation=_strategy(),
    )


class AttestationStressMockLLM:
    """Returns methodologically valid candidates with varied legacy attestations."""

    _VARIANTS = [
        None,
        {
            "visualCameBeforeKeyword": False,
            "runwayCheckCameBeforeKeyword": False,
            "headlineWasNotStartingPoint": False,
        },
        {
            "visualCameBeforeKeyword": "false",
            "runwayCheckCameBeforeKeyword": "true",
        },
        {
            "visualCameBeforeKeyword": True,
        },
        {
            "visualCameBeforeKeyword": False,
            "runwayCheckCameBeforeKeyword": True,
            "headlineWasNotStartingPoint": True,
        },
        {
            "visualCameBeforeKeyword": "true",
            "runwayCheckCameBeforeKeyword": "false",
            "headlineWasNotStartingPoint": "true",
        },
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
            attestation = self._VARIANTS[idx % len(self._VARIANTS)]
            if attestation is None:
                candidate.pop("creativeOrderConfirmation", None)
            else:
                candidate["creativeOrderConfirmation"] = attestation
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
            prototype_id = "closest"
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if pid in prompt:
                    prototype_id = pid
                    break
            return _judgment(candidate_id, total_hint=80)
        if role == "builder2_winner":
            return _winner_plan_from_prompt(prompt)
        raise AssertionError(f"unexpected role {role}")


class TestCreativeOrderAttestationHandling(unittest.TestCase):
    def test_missing_attestation_accepted(self) -> None:
        cand = _candidate("closest")
        cand.pop("creativeOrderConfirmation", None)
        validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertIn("creativeOrderContract", out)
        self.assertNotIn("creativeOrderConfirmation", out)

    def test_all_false_booleans_accepted(self) -> None:
        cand = _candidate("closest")
        cand["creativeOrderConfirmation"] = {
            "visualCameBeforeKeyword": False,
            "runwayCheckCameBeforeKeyword": False,
            "headlineWasNotStartingPoint": False,
        }
        validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertEqual(out["creatorAttestations"]["authoritative"], False)
        self.assertFalse(out["creatorAttestations"]["creativeOrderConfirmation"]["visualCameBeforeKeyword"])

    def test_partial_object_accepted(self) -> None:
        cand = _candidate("closest")
        cand["creativeOrderConfirmation"] = {"visualCameBeforeKeyword": True}
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertTrue(out["creatorAttestations"]["creativeOrderConfirmationReceived"])

    def test_string_booleans_do_not_trigger_repair(self) -> None:
        calls: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            cand = _candidate("closest")
            cand["creativeOrderConfirmation"] = {
                "visualCameBeforeKeyword": "false",
                "runwayCheckCameBeforeKeyword": "true",
                "headlineWasNotStartingPoint": "true",
            }
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
        self.assertEqual(state["metrics"].get("creatorRetryCalls", 0), 0)

    def test_legacy_stored_as_non_authoritative_diagnostics(self) -> None:
        out = _valid_candidate("closest")
        self.assertIn("creatorAttestations", out)
        self.assertFalse(out["creatorAttestations"]["authoritative"])
        self.assertNotIn("creativeOrderConfirmation", out)


class TestCreativeOrderContract(unittest.TestCase):
    def test_server_attaches_contract(self) -> None:
        out = _valid_candidate("closest")
        contract = out["creativeOrderContract"]
        self.assertEqual(contract["version"], CREATIVE_ORDER_CONTRACT_VERSION)
        self.assertTrue(contract["enforcedByCreatorPrompt"])

    def test_contract_uses_canonical_stage_order(self) -> None:
        contract = build_creative_order_contract()
        self.assertEqual(tuple(contract["stageOrder"]), CREATIVE_STAGE_ORDER)

    def test_model_cannot_override_contract(self) -> None:
        cand = _candidate("closest")
        cand["creativeOrderContract"] = {"version": "hacked", "stageOrder": ["headline_first"]}
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertEqual(out["creativeOrderContract"]["version"], CREATIVE_ORDER_CONTRACT_VERSION)
        self.assertEqual(tuple(out["creativeOrderContract"]["stageOrder"]), CREATIVE_STAGE_ORDER)


class TestCreativeOrderRealEnforcement(unittest.TestCase):
    def test_born_from_visible_mechanism_false_triggers_clean_retry(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            cand = _candidate("closest")
            cand["verbalPotential"]["bornFromVisibleMechanism"] = calls["count"] > 1
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
        self.assertEqual(calls["count"], 2)
        self.assertEqual(state["metrics"].get("creatorRetryCalls"), 1)
        self.assertEqual(state["metrics"].get("creatorRepairCalls", 0), 0)

    def test_missing_verbal_potential_is_normalized_without_repair(self) -> None:
        calls = {"count": 0}
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            cand = _candidate("closest")
            cand.pop("verbalPotential")
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
        self.assertEqual(calls["count"], 1)
        self.assertEqual(state["metrics"].get("creatorRepairCalls", 0), 0)

    def test_final_headline_fields_remain_prohibited(self) -> None:
        cand = _candidate("closest")
        cand["headlineText"] = "Buy now"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_candidate(
                cand,
                assigned_prototype_id="closest",
                prototype_display_name="Closest",
                strategy_foundation=_strategy(),
            )

    def test_runway_feasibility_guards_remain_mandatory(self) -> None:
        cand = _candidate("closest")
        cand["runwayFeasibility"]["requiresImpossibleMorphing"] = True
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())

    def test_strategy_identity_required(self) -> None:
        cand = _candidate("closest")
        cand["strategyFoundationId"] = "wrong"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())

    def test_prompt_does_not_require_self_attestation(self) -> None:
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
        self.assertNotIn("creativeOrderConfirmation", prompt)
        self.assertIn("do not self-certify internal reasoning order", prompt.lower())

    def test_think_small_still_invalid_without_weakness(self) -> None:
        cand = _candidate("think_small")
        cand["thinkSmallApplication"]["evidenceTheWeaknessIsReal"] = "An invented cosmetic weakness"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="think_small", strategy_foundation=_strategy())

    def test_closest_still_requires_application(self) -> None:
        cand = _candidate("closest")
        cand.pop("closestApplication", None)
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())

    def test_greenpeace_still_invalid_shape_only(self) -> None:
        cand = _candidate("greenpeace_essential_pairing")
        cand["essentialPairingApplication"]["notMerelyAppearance"] = "Shape only pairing"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(
                cand,
                assigned_prototype_id="greenpeace_essential_pairing",
                strategy_foundation=_strategy(),
            )


class TestCreativeOrderProductionRegression(unittest.TestCase):
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
    def test_six_valid_candidates_reach_judge_without_attestation_repairs(self) -> None:
        llm = AttestationStressMockLLM()
        run_builder2_tournament(
            job_id="job-attestation-regression",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-attestation",
        )
        state = load_tournament_state("job-attestation-regression")
        assert state is not None
        metrics = state.get("metrics") or {}
        accepted = [
            rec
            for rec in state.get("candidates", {}).values()
            if rec.get("validationStatus") == "accepted"
        ]
        self.assertEqual(len(accepted), len(DEFAULT_ACTIVE_PROTOTYPE_IDS))
        judged = [rec for rec in accepted if rec.get("judgmentId")]
        self.assertEqual(len(judged), len(DEFAULT_ACTIVE_PROTOTYPE_IDS))
        self.assertTrue(state.get("winnerCandidateId"))
        self.assertTrue(state.get("winnerDevelopmentPlan"))
        self.assertEqual(metrics.get("creatorCalls"), 6)
        self.assertEqual(metrics.get("creatorRepairCalls"), 0)
        self.assertEqual(metrics.get("creatorRetryCalls"), 0)
        self.assertEqual(metrics.get("judgeCalls"), 6)
        self.assertEqual(metrics.get("winnerDevelopmentCalls"), 1)
        self.assertEqual(metrics.get("totalReasoningCalls"), 14)
        for rec in accepted:
            contract = (rec.get("creatorOutput") or {}).get("creativeOrderContract") or {}
            self.assertEqual(contract.get("version"), CREATIVE_ORDER_CONTRACT_VERSION)
            self.assertNotIn("creativeOrderConfirmation", rec.get("creatorOutput") or {})


class TestCreativeOrderTournamentContinuation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_MAX_ROUNDS": "1", "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,think_small"},
        clear=True,
    )
    def test_one_invalid_candidate_does_not_abort_tournament(self) -> None:
        class MixedLLM(TournamentMockLLM):
            def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
                if role == "builder2_creator" and "Assigned prototype ID: think_small" in prompt:
                    bad = _candidate("think_small")
                    bad.pop("verbalPotential")
                    return bad
                return super().__call__(role=role, model=model, prompt=prompt)

        llm = MixedLLM()
        run_builder2_tournament(
            job_id="job-partial-fail",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-partial",
        )
        state = load_tournament_state("job-partial-fail")
        assert state is not None
        self.assertTrue(state.get("winnerCandidateId"))
        rejected = [
            rec
            for rec in state.get("candidates", {}).values()
            if rec.get("validationStatus") != "accepted"
        ]
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
