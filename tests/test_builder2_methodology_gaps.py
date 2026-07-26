"""
Builder2 methodology gap-closure tests (extends test_builder2_methodology.py).
"""
from __future__ import annotations

import importlib
import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_methodology_contract import (
    BUILDER2_METHODOLOGY_COVERAGE,
    INTEREST_PRIORITY_ORDER,
    VALID_STRUCTURE_TYPES,
    VALID_VISUAL_PARALLEL_TYPES,
    prompt_enum_list,
    resolve_coverage_test_target,
)
from engine.builder2_methodology_validation import (
    build_winning_candidate_preservation_snapshot,
    infer_process_failure_tag,
    validate_creator_methodology,
    validate_judge_methodology,
    validate_strategy_identity,
    validate_winner_methodology,
)
from engine.builder2_strategy_identity import assign_strategy_foundation_identity
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_prompts import build_creator_prompt, build_judge_prompt, build_winner_development_prompt
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    ensure_methodology_compatibility_decided,
    load_tournament_state,
    new_tournament_state,
    record_process_failure_tag,
    save_tournament_state,
)
from engine.builder2_prototypes import require_prototype
from engine.builder2_winner_plan import validate_and_normalize_builder2_winner_plan
from tests.builder2_methodology_fixtures import methodology_strategy_extras, methodology_winner_extras
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _judgment, _strategy, _winner_plan


class TestCreatorMethodology(unittest.TestCase):
    def test_interest_first_in_creator_prompt(self) -> None:
        strategy = _strategy()
        prompt = build_creator_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=strategy,
            prototype=require_prototype("closest"),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="text_to_video",
        )
        expected = " → ".join(INTEREST_PRIORITY_ORDER)
        self.assertIn(f"Interest priority: {expected}", prompt)

    def test_verbal_potential_required(self) -> None:
        cand = _candidate("closest")
        cand.pop("verbalPotential")
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())
        self.assertIn("verbalPotential", str(ctx.exception.args[0]))

    def test_strategy_identity_required(self) -> None:
        strategy = _strategy()
        cand = _candidate("closest")
        cand["strategyFoundationId"] = "wrong-id"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=strategy)
        self.assertEqual(ctx.exception.args[0], "builder2_creator_validation_failed:strategyFoundationId")

    def test_attestation_not_required_for_methodology(self) -> None:
        cand = _candidate("closest")
        cand.pop("creativeOrderConfirmation", None)
        validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())


class TestJudgeMethodology(unittest.TestCase):
    def test_interest_first_in_judge_prompt(self) -> None:
        strategy = _strategy()
        prompt = build_judge_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=strategy,
            prototype=require_prototype("closest"),
            candidate=_candidate("closest"),
            candidate_id="cand-1",
        )
        expected = " → ".join(INTEREST_PRIORITY_ORDER)
        self.assertIn(f"Interest priority for qualitative assessment: {expected}", prompt)


class TestSilentRunway(unittest.TestCase):
    def test_runway_infeasible_morph_rejected(self) -> None:
        cand = _candidate("closest")
        cand["runwayFeasibility"]["requiresImpossibleMorphing"] = True
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())


class TestEditingAdaptation(unittest.TestCase):
    def test_builder1_adaptation_requires_fields(self) -> None:
        cand = _candidate("closest")
        cand["sourceConcept"] = {"type": "builder1_adaptation"}
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())


class TestCoverageMapIntegrity(unittest.TestCase):
    def test_referenced_test_targets_exist(self) -> None:
        seen: set[str] = set()
        for entry in BUILDER2_METHODOLOGY_COVERAGE.values():
            for target in entry["tests"]:
                if target in seen:
                    continue
                seen.add(target)
                try:
                    importlib.import_module(target)
                    continue
                except ModuleNotFoundError:
                    pass
                module_path, name = resolve_coverage_test_target(target)
                module = importlib.import_module(module_path)
                self.assertTrue(
                    hasattr(module, name),
                    msg=f"Missing coverage target {target}",
                )


class TestStrategyIdentity(unittest.TestCase):
    def test_same_id_across_candidates(self) -> None:
        strategy = _strategy()
        expected = strategy["strategyFoundationId"]
        for prototype_id in ("closest", "think_small", "forgot"):
            cand = _candidate(prototype_id)
            cand["strategyFoundationId"] = expected
            validate_strategy_identity(expected_strategy_foundation_id=expected, candidate=cand)

    def test_wrong_id_rejected(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_strategy_identity(
                expected_strategy_foundation_id=_strategy()["strategyFoundationId"],
                candidate={"strategyFoundationId": "other-job-id"},
            )

    def test_resume_reuses_id(self) -> None:
        enable_memory_store()
        try:
            strategy = assign_strategy_foundation_identity(methodology_strategy_extras(), tournament_id="t-resume")
            state = new_tournament_state(
                job_id="job-resume",
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            )
            state["strategyFoundation"] = strategy
            save_tournament_state("job-resume", state)
            loaded = load_tournament_state("job-resume")
            assert loaded is not None
            self.assertEqual(
                loaded["strategyFoundation"]["strategyFoundationId"],
                strategy["strategyFoundationId"],
            )
        finally:
            disable_memory_store()


class TestCompatibilityMode(unittest.TestCase):
    def test_new_job_cannot_enter_compatibility_mode(self) -> None:
        enable_memory_store()
        try:
            state = new_tournament_state(
                job_id="job-new",
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            )
            decided = ensure_methodology_compatibility_decided(state, is_new_job=True)
            self.assertFalse(decided)
            self.assertFalse(state.get("methodologyCompatibilityMode"))
        finally:
            disable_memory_store()

    def test_legacy_resume_may_enter_compatibility_mode(self) -> None:
        enable_memory_store()
        try:
            state = new_tournament_state(
                job_id="job-legacy",
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            )
            state["strategyFoundation"] = {"productNameResolved": "X", "language": "en"}
            save_tournament_state("job-legacy", state)
            loaded = load_tournament_state("job-legacy")
            assert loaded is not None
            decided = ensure_methodology_compatibility_decided(loaded, is_new_job=False)
            self.assertTrue(decided)
            self.assertEqual(loaded.get("methodologyCompatibilityReason"), "persisted_pre_methodology_state")
        finally:
            disable_memory_store()


class TestProcessFailureTags(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_category_only_no_creative_text(self) -> None:
        state = new_tournament_state(
            job_id="job-tags",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        record_process_failure_tag(
            state,
            "builder2_creator_validation_failed:strategyFoundationId headline rescue scene object",
        )
        tags = state.get("processFailureTags") or []
        self.assertEqual(tags, ["strategy_identity_mismatch"])
        blob = " ".join(tags).lower()
        for forbidden in ("headline", "scene", "object", "rescue"):
            self.assertNotIn(forbidden, blob)

    def test_deduplicated(self) -> None:
        state = new_tournament_state(
            job_id="job-tags-dedupe",
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        record_process_failure_tag(state, "builder2_strategy_not_grounded")
        record_process_failure_tag(state, "builder2_strategy_not_grounded")
        self.assertEqual(state["processFailureTags"].count("problem_not_grounded"), 1)


class TestJudgeToWinnerFlow(unittest.TestCase):
    def test_winner_prompt_includes_judgment_and_refinement_rule(self) -> None:
        strategy = _strategy()
        candidate = _candidate("closest")
        judgment = _judgment("cand-1")
        snapshot = build_winning_candidate_preservation_snapshot(
            strategy_foundation=strategy,
            winning_candidate=candidate,
        )
        prompt = build_winner_development_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=strategy,
            winning_candidate=candidate,
            winning_judgment=judgment,
            prototype=require_prototype("closest"),
            runway_mode="text_to_video",
            preservation_snapshot=snapshot,
        )
        self.assertIn("Refine the winning execution only.", prompt)
        self.assertIn("Valid Judge judgment for this winning candidate only:", prompt)
        self.assertNotIn("tournament standing", prompt.lower())


class TestWinnerPreservation(unittest.TestCase):
    def test_changed_prototype_rejected(self) -> None:
        strategy = _strategy()
        candidate = _candidate("closest")
        plan = _winner_plan()
        snapshot = build_winning_candidate_preservation_snapshot(
            strategy_foundation=strategy,
            winning_candidate=candidate,
        )
        plan["prototypeId"] = "think_small"
        plan["preservationReference"]["prototypeId"] = "think_small"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_winner_methodology(plan, winning_candidate=candidate, preservation_snapshot=snapshot)
        self.assertIn("prototypeId", str(ctx.exception.args[0]))


class TestOptionalHeadline(unittest.TestCase):
    def test_continuous_event_omit_normalizes(self) -> None:
        plan = _winner_plan()
        plan.update(methodology_winner_extras(headline_decision="omit"))
        plan["headline"] = ""
        plan["headlineCoreKeyword"] = ""
        normalized = validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        self.assertEqual(normalized.get("headlineDecision"), "omit")
        self.assertEqual(normalized.get("headlineText"), "")


class TestVisualFamilyMontage(unittest.TestCase):
    def test_montage_requires_two_to_four_variations(self) -> None:
        cand = _candidate("closest", structure="variation_montage")
        cand["sceneVariations"] = ["one beat only"]
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())

    def test_continuous_event_single_sequence_ok(self) -> None:
        cand = _candidate("closest", structure="continuous_event")
        validate_creator_methodology(cand, assigned_prototype_id="closest", strategy_foundation=_strategy())


class TestPromptEnumSource(unittest.TestCase):
    def test_creator_prompt_enums_from_canonical_constants(self) -> None:
        prompt = build_creator_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype=require_prototype("closest"),
            candidate_id="c1",
            attempt_number=1,
            runway_mode="text_to_video",
        )
        self.assertIn(prompt_enum_list(VALID_STRUCTURE_TYPES), prompt)
        self.assertIn(prompt_enum_list(VALID_VISUAL_PARALLEL_TYPES), prompt)


if __name__ == "__main__":
    unittest.main()
