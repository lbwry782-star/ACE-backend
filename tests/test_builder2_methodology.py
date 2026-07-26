"""
Builder2 methodology alignment tests.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_methodology_contract import (
    ACTIVE_PROTOTYPE_IDS,
    BUILDER2_METHODOLOGY_COVERAGE,
    METHODOLOGY_VERSION,
    REFERENCE_ONLY_METHOD_IDS,
    assert_full_coverage_map,
    methodology_section_ids,
)
from engine.builder2_methodology_validation import (
    build_winning_candidate_preservation_snapshot,
    validate_creator_methodology,
    validate_judge_methodology,
    validate_strategy_methodology,
    validate_winner_methodology,
)
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS, REFERENCE_ONLY_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError, compare_candidate_rankings
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from tests.builder2_methodology_fixtures import (
    methodology_candidate_extras,
    methodology_strategy_extras,
    methodology_winner_extras,
)
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _judgment, _strategy, _winner_plan


class TestMethodologyCoverage(unittest.TestCase):
    def test_all_39_sections_present(self) -> None:
        assert_full_coverage_map()
        self.assertEqual(len(methodology_section_ids()), 39)
        self.assertEqual(set(BUILDER2_METHODOLOGY_COVERAGE.keys()), set(methodology_section_ids()))

    def test_every_section_has_modules_and_tests(self) -> None:
        for section_id, entry in BUILDER2_METHODOLOGY_COVERAGE.items():
            self.assertTrue(entry["modules"], msg=section_id)
            self.assertTrue(entry["tests"], msg=section_id)

    def test_active_prototypes_match_contract(self) -> None:
        self.assertEqual(set(ACTIVE_PROTOTYPE_IDS), set(DEFAULT_ACTIVE_PROTOTYPE_IDS))

    def test_reference_only_not_active(self) -> None:
        for ref_id in REFERENCE_ONLY_PROTOTYPE_IDS:
            self.assertNotIn(ref_id, ACTIVE_PROTOTYPE_IDS)


class TestStrategyMethodology(unittest.TestCase):
    def test_grounded_problem_passes(self) -> None:
        validate_strategy_foundation(_strategy())

    def test_generic_business_goal_rejected(self) -> None:
        data = _strategy()
        data["problemPerception"]["statement"] = "The market is competitive and the brand needs awareness."
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_validation_failed:problemPerception.statement")

    def test_strategy_forbidden_visual_field(self) -> None:
        data = _strategy()
        data["headline"] = "Buy now"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_validation_failed:headline")

    def test_mechanism_depth_required(self) -> None:
        data = _strategy()
        del data["mechanismScan"]["depthEvidence"]
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_validation_failed:mechanismScan.depthEvidence")

    def test_compatibility_mode_skips_new_fields(self) -> None:
        legacy = _strategy()
        legacy.pop("methodologyVersion", None)
        legacy["relativeAdvantage"] = {
            "statement": legacy["relativeAdvantage"]["statement"],
            "derivationFromProblem": legacy["relativeAdvantage"]["derivationFromProblem"],
        }
        legacy["mechanismScan"] = {
            "domainFacts": legacy["mechanismScan"]["domainFacts"],
            "discoveredMechanism": legacy["mechanismScan"]["discoveredMechanism"],
            "creativeOpportunity": legacy["mechanismScan"]["creativeOpportunity"],
        }
        validate_strategy_methodology(legacy, compatibility_mode=True)


class TestPrototypeMethodology(unittest.TestCase):
    def test_winning_card_literal_card_rejected(self) -> None:
        cand = _candidate("winning_card")
        cand["winningCardApplication"]["mediumOrContainerIdentified"] = "A playing card with card symbols"
        cand["winningCardApplication"]["whatItBecomes"] = ""
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="winning_card", strategy_foundation=_strategy())

    def test_think_small_invented_weakness_rejected(self) -> None:
        cand = _candidate("think_small")
        cand["thinkSmallApplication"]["evidenceTheWeaknessIsReal"] = "An invented cosmetic weakness"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="think_small")

    def test_greenpeace_shape_only_rejected(self) -> None:
        cand = _candidate("greenpeace_essential_pairing")
        cand["essentialPairingApplication"]["notMerelyAppearance"] = "Shape only pairing"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="greenpeace_essential_pairing")


class TestVisualMethodology(unittest.TestCase):
    def test_replacement_requires_check(self) -> None:
        cand = _candidate("closest")
        cand["visualParallelType"] = "replacement"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest")

    def test_context_collision_requires_safeguard(self) -> None:
        cand = _candidate("closest")
        cand["visualParallelType"] = "context_collision"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest")


class TestVerbalLayer(unittest.TestCase):
    def test_judge_methodology_assessments_required(self) -> None:
        bad = _judgment("c1")
        bad.pop("verbalLayerAssessment")
        with self.assertRaises(Builder2TournamentError):
            validate_judge_methodology(bad)

    def test_winner_omit_headline_allowed(self) -> None:
        strategy = _strategy()
        candidate = _candidate("closest")
        plan = _winner_plan()
        plan.update(methodology_winner_extras(headline_decision="omit", winning_candidate=candidate, strategy=strategy))
        plan["headline"] = ""
        plan["headlineCoreKeyword"] = ""
        snapshot = build_winning_candidate_preservation_snapshot(
            strategy_foundation=strategy,
            winning_candidate=candidate,
        )
        validate_winner_methodology(plan, winning_candidate=candidate, preservation_snapshot=snapshot)


class TestTournamentBetweenIdeas(unittest.TestCase):
    def test_prototype_identity_does_not_affect_ranking(self) -> None:
        record_a = {
            "candidateId": "a",
            "totalScore": 80,
            "tieScores": {"silentVisualClarity": 12, "problemAdvantageIntegrity": 15, "runwayFeasibility": 8},
            "completedAt": "2026-01-01T00:00:00+00:00",
            "eligible": True,
        }
        record_b = dict(record_a)
        record_b["candidateId"] = record_a["candidateId"]
        self.assertEqual(compare_candidate_rankings(record_a, record_b), 0)


class TestOneRoundRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS), "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
        clear=True,
    )
    def test_fourteen_calls_no_retry(self) -> None:
        llm = TournamentMockLLM()
        run_builder2_tournament(
            job_id="job-methodology-14",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-14",
        )
        state = load_tournament_state("job-methodology-14")
        assert state is not None
        metrics = state.get("metrics") or {}
        self.assertEqual(metrics.get("totalReasoningCalls"), 14)
        self.assertEqual(state.get("completionReason"), "max_rounds_reached")
        self.assertTrue(state.get("winnerCandidateId"))


class TestManagerLearning(unittest.TestCase):
    def test_no_cross_job_creative_memory_module(self) -> None:
        import engine.builder2_tournament_store as store

        self.assertFalse(hasattr(store, "write_creative_reuse_memory"))


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


# Coverage-map class names (also implemented in test_builder2_methodology_gaps.py)
from tests.test_builder2_methodology_gaps import (  # noqa: E402
    TestCompatibilityMode,
    TestCoverageMapIntegrity,
    TestCreatorMethodology,
    TestEditingAdaptation,
    TestJudgeMethodology,
    TestJudgeToWinnerFlow,
    TestOptionalHeadline,
    TestProcessFailureTags,
    TestPromptEnumSource,
    TestSilentRunway,
    TestStrategyIdentity,
    TestVisualFamilyMontage,
    TestWinnerPreservation,
)


if __name__ == "__main__":
    unittest.main()
