"""
Builder2 product brief v2 production guard tests.

Run: python -m unittest tests.test_builder2_product_brief_production_guard -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder2_post_strategy_isolation import prompt_contains_raw_product_description
from engine.builder2_product_brief_production_guard import (
    PRODUCT_BRIEF_MODE_LEGACY_COMPAT,
    PRODUCT_BRIEF_MODE_V2_SELECTED,
    build_product_input_block_for_prompt,
    collect_v2_taxonomy_missing_fields,
    ensure_product_brief_mode_decided,
    has_complete_v2_product_brief_taxonomy,
    is_persisted_pre_v2_product_brief_state,
    resolve_product_brief_mode,
    validate_v2_product_brief_taxonomy_for_new_production,
)
from engine.builder2_product_semantic_brief import BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_prompts import build_creator_prompt, build_judge_prompt, build_winner_development_prompt
from engine.builder2_tournament_store import new_tournament_state
from tests.builder2_methodology_fixtures import methodology_strategy_evidence_extras
from tests.test_builder2_product_description_selection_chain import (
    RAW_PERFUME,
    _bosa_strategy,
    _legacy_strategy_without_taxonomy,
    _prototype,
)
from tests.test_builder2_tournament import _candidate, _judgment


def _incomplete_v2_brief_strategy() -> Dict[str, Any]:
    strategy = copy.deepcopy(_bosa_strategy())
    brief = strategy["strategyEvidenceGrounding"]["productSemanticBrief"]
    brief.pop("discardedFacts", None)
    return strategy


class TestV2TaxonomyDetection(unittest.TestCase):
    def test_complete_v2_brief_detected(self) -> None:
        brief = _bosa_strategy()["strategyEvidenceGrounding"]["productSemanticBrief"]
        self.assertTrue(has_complete_v2_product_brief_taxonomy(brief))
        self.assertEqual(collect_v2_taxonomy_missing_fields(brief), [])

    def test_legacy_v1_brief_not_v2(self) -> None:
        brief = _legacy_strategy_without_taxonomy()["strategyEvidenceGrounding"]["productSemanticBrief"]
        self.assertFalse(has_complete_v2_product_brief_taxonomy(brief))
        self.assertIn("briefVersion", collect_v2_taxonomy_missing_fields(brief))

    def test_missing_bucket_detected(self) -> None:
        brief = {
            "briefVersion": BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2,
            "essentialFacts": [{"id": "e1", "text": "fact"}],
            "supportingEvidence": [],
            "mandatoryConstraints": [],
        }
        missing = collect_v2_taxonomy_missing_fields(brief)
        self.assertIn("discardedFacts", missing)


class TestNewJobValidation(unittest.TestCase):
    def test_new_strategy_must_have_v2_taxonomy(self) -> None:
        strategy = methodology_strategy_evidence_extras(
            tournament_id="guard-new",
            product_name="Test",
            product_description="A product description for testing.",
        )
        validate_strategy_foundation(
            strategy,
            product_name="Test",
            product_description="A product description for testing.",
        )

    def test_wrong_brief_version_fails_validation(self) -> None:
        from engine.builder2_strategy_evidence_grounding_contract import validate_strategy_evidence_grounding

        strategy = _bosa_strategy()
        strategy["strategyEvidenceGrounding"]["productSemanticBrief"]["briefVersion"] = (
            "builder2_product_semantic_brief_v1"
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_evidence_grounding(
                strategy,
                product_name="בוסה",
                product_description=RAW_PERFUME,
            )
        self.assertIn("v2_taxonomy_incomplete.briefVersion", str(ctx.exception.args[0]))

    def test_validate_helper_raises_for_incomplete_brief(self) -> None:
        brief = _incomplete_v2_brief_strategy()["strategyEvidenceGrounding"]["productSemanticBrief"]
        with self.assertRaises(Builder2TournamentError):
            validate_v2_product_brief_taxonomy_for_new_production(brief)


class TestJobModeDecision(unittest.TestCase):
    def test_new_job_always_v2_selected(self) -> None:
        state = new_tournament_state(job_id="job-new", language="en", active_prototype_ids=["closest"], random_seed="seed")
        mode = ensure_product_brief_mode_decided(state, is_new_job=True)
        self.assertEqual(mode, PRODUCT_BRIEF_MODE_V2_SELECTED)
        self.assertTrue(state.get("productBriefModeDecided"))

    def test_resumed_legacy_job_enters_legacy_compat(self) -> None:
        state = new_tournament_state(job_id="job-legacy", language="en", active_prototype_ids=["closest"], random_seed="seed")
        state["strategyFoundation"] = _legacy_strategy_without_taxonomy()
        mode = ensure_product_brief_mode_decided(state, is_new_job=False)
        self.assertEqual(mode, PRODUCT_BRIEF_MODE_LEGACY_COMPAT)
        self.assertTrue(is_persisted_pre_v2_product_brief_state(state))

    def test_resumed_v2_job_stays_v2_selected(self) -> None:
        state = new_tournament_state(job_id="job-v2", language="en", active_prototype_ids=["closest"], random_seed="seed")
        state["strategyFoundation"] = _bosa_strategy()
        mode = ensure_product_brief_mode_decided(state, is_new_job=False)
        self.assertEqual(mode, PRODUCT_BRIEF_MODE_V2_SELECTED)


class TestPromptIsolationGuard(unittest.TestCase):
    def test_new_v2_job_creator_prompt_excludes_raw_description(self) -> None:
        strategy = _bosa_strategy()
        state = {"productBriefMode": PRODUCT_BRIEF_MODE_V2_SELECTED, "productBriefModeDecided": True}
        prompt = build_creator_prompt(
            product_name="בוסה",
            product_description=RAW_PERFUME,
            language="he",
            strategy_foundation=strategy,
            prototype=_prototype(),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="silent",
            state=state,
        )
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))
        self.assertIn("Post-Strategy creative input", prompt)

    def test_new_v2_job_judge_and_winner_exclude_raw_description(self) -> None:
        strategy = _bosa_strategy()
        state = {"productBriefMode": PRODUCT_BRIEF_MODE_V2_SELECTED, "productBriefModeDecided": True}
        candidate = _candidate("greenpeace_essential_pairing")
        for builder in (
            lambda: build_judge_prompt(
                product_name="בוסה",
                product_description=RAW_PERFUME,
                language="he",
                strategy_foundation=strategy,
                prototype=_prototype(),
                candidate=candidate,
                candidate_id="cand-1",
                state=state,
            ),
            lambda: build_winner_development_prompt(
                product_name="בוסה",
                product_description=RAW_PERFUME,
                language="he",
                strategy_foundation=strategy,
                winning_candidate=candidate,
                winning_judgment=_judgment("cand-1"),
                prototype=_prototype(),
                runway_mode="silent",
                preservation_snapshot={},
                state=state,
            ),
        ):
            prompt = builder()
            self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_legacy_compat_job_may_use_raw_description(self) -> None:
        strategy = _legacy_strategy_without_taxonomy()
        state = {"productBriefMode": PRODUCT_BRIEF_MODE_LEGACY_COMPAT, "productBriefModeDecided": True}
        prompt = build_creator_prompt(
            product_name="Legacy",
            product_description=RAW_PERFUME,
            language="he",
            strategy_foundation=strategy,
            prototype=_prototype(),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="silent",
            state=state,
        )
        self.assertIn("<product_description>", prompt)

    def test_v2_selected_without_taxonomy_fails_closed(self) -> None:
        strategy = _legacy_strategy_without_taxonomy()
        with self.assertRaises(Builder2TournamentError) as ctx:
            build_product_input_block_for_prompt(
                strategy_foundation=strategy,
                product_description=RAW_PERFUME,
                product_brief_mode=PRODUCT_BRIEF_MODE_V2_SELECTED,
            )
        self.assertEqual(str(ctx.exception.args[0]), "builder2_product_brief_v2_taxonomy_required")

    def test_resolve_mode_without_state_defaults_v2_selected(self) -> None:
        self.assertEqual(
            resolve_product_brief_mode(strategy_foundation=_bosa_strategy()),
            PRODUCT_BRIEF_MODE_V2_SELECTED,
        )


class TestBuilder1Untouched(unittest.TestCase):
    def test_guard_module_is_builder2_only(self) -> None:
        import engine.builder2_product_brief_production_guard as guard

        self.assertNotIn("builder1", guard.__file__ or "")


if __name__ == "__main__":
    unittest.main()
