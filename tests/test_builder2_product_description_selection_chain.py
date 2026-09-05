"""
Builder2 product-description selection chain — zero-cost structural tests.

Run: python -m unittest tests.test_builder2_product_description_selection_chain -v
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict

from engine.builder2_essential_fact_fusion import (
    BUILDER2_CULTURAL_CONTEXT,
    BUILDER2_ESSENTIAL_FACT_FUSION,
    apply_fusion_eligibility_rules,
    fusion_required_for_brief,
    judgment_rejects_essential_fact_fusion,
)
from engine.builder2_fact_selection import NEGATIVE_SELECTION_METHODOLOGY, validate_fact_selection_brief
from engine.builder2_post_strategy_isolation import (
    build_slim_strategy_foundation_for_prompts,
    prompt_contains_discarded_facts,
    prompt_contains_raw_product_description,
    prompt_contains_source_description_leak,
    strategy_json_for_post_strategy_prompt,
)
from engine.builder2_product_semantic_brief import (
    BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2,
    get_product_semantic_brief,
    summarize_brief_for_creative_prompt,
)
from engine.builder2_strategy import validate_strategy_foundation
from engine.builder2_strategy_evidence_grounding_contract import (
    BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
)
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_prompts import (
    build_creator_prompt,
    build_creator_repair_prompt,
    build_judge_prompt,
    build_judge_repair_prompt,
    build_strategy_prompt,
    build_winner_development_prompt,
    build_winner_headline_repair_prompt,
)
from tests.builder2_methodology_fixtures import (
    methodology_candidate_extras,
    methodology_creator_fusion_extras,
    methodology_judgment_extras,
    methodology_judge_fusion_extras,
    methodology_strategy_evidence_extras,
    methodology_strategy_extras,
)
from tests.test_builder2_tournament import _candidate, _judgment


RAW_PERFUME = (
    "בושם לגברים מתוצרת ישראל.\n"
    "בקבוק 100 מ״ל.\n"
    "החברה נוסדה בשנת 2019.\n"
    "המחסן נמצא בחיפה.\n"
    "בעל החברה אוהב ג'אז."
)
DISCARDED_JAZZ = "בעל החברה אוהב ג'אז"
DISCARDED_WAREHOUSE = "המחסן נמצא בחיפה"
DISCARDED_FOUNDED = "החברה נוסדה בשנת 2019"
DISCARDED_BOTTLE = "בקבוק 100 מ״ל"
PRODUCT_NAME = "בוסה"

RAW_SHELL = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue. "
    "Unrelated advantage: waterproof coating option. Founded in 2019. Warehouse in Haifa."
)


def _bosa_brief_overrides() -> Dict[str, Any]:
    return {
        "briefVersion": BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2,
        "sourceDescription": RAW_PERFUME,
        "essentialFacts": [
            {"id": "e1", "text": "בושם לגברים"},
            {"id": "e2", "text": "מיוצר בישראל"},
        ],
        "supportingEvidence": [
            {"id": "s1", "text": "תיאור המוצר מציין במפורש שהוא מתוצרת ישראל"},
        ],
        "mandatoryConstraints": [
            {"id": "m1", "text": "Silent video without dialogue"},
            {"id": "m2", "text": "No logo in frame"},
        ],
        "discardedFacts": [
            {"id": "d1", "text": DISCARDED_BOTTLE},
            {"id": "d2", "text": DISCARDED_FOUNDED},
            {"id": "d3", "text": DISCARDED_WAREHOUSE},
            {"id": "d4", "text": DISCARDED_JAZZ},
        ],
        "explicitFacts": [
            {"id": "e1", "text": "בושם לגברים"},
            {"id": "e2", "text": "מיוצר בישראל"},
        ],
    }


def _bosa_strategy(*, include_taxonomy: bool = True) -> Dict[str, Any]:
    strategy = methodology_strategy_evidence_extras(
        tournament_id="selection-chain",
        product_name=PRODUCT_NAME,
        product_description=RAW_PERFUME,
    )
    if include_taxonomy:
        brief = strategy["strategyEvidenceGrounding"]["productSemanticBrief"]
        brief.update(_bosa_brief_overrides())
    return validate_strategy_foundation(
        strategy,
        product_name=PRODUCT_NAME,
        product_description=RAW_PERFUME,
    )


def _legacy_strategy_without_taxonomy() -> Dict[str, Any]:
    strategy = methodology_strategy_extras(tournament_id="legacy-selection")
    strategy["schemaVersion"] = "builder2_strategy_v1"
    strategy["strategyEvidenceGrounding"] = {
        "contractVersion": BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
        "productMarketStatus": "existing_product",
        "productInformationDensity": "moderate",
        "explicitProductFacts": [{"text": "Legacy product fact"}],
        "safeStrategicInterpretations": [],
        "categoryConventions": [],
        "unsupportedAssumptions": [],
        "allowedCapabilities": [],
        "productInputAudit": {"productDescription": RAW_PERFUME},
        "productSemanticBrief": {
            "briefVersion": "builder2_product_semantic_brief_v1",
            "sourceDescription": RAW_PERFUME,
            "explicitFacts": [{"id": "f1", "text": "Legacy product fact"}],
            "licensedImplications": [],
            "restrictedCapabilities": [],
            "allowedCapabilities": [],
        },
    }
    return strategy


def _prototype():
    from engine.builder2_prototypes import get_prototype

    return get_prototype("greenpeace_essential_pairing")


def _creator_prompt(strategy: Dict[str, Any]) -> str:
    return build_creator_prompt(
        product_name=PRODUCT_NAME,
        product_description=RAW_PERFUME,
        language="he",
        strategy_foundation=strategy,
        prototype=_prototype(),
        candidate_id="cand-1",
        attempt_number=1,
        runway_mode="silent",
    )


def _judge_prompt(strategy: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    return build_judge_prompt(
        product_name=PRODUCT_NAME,
        product_description=RAW_PERFUME,
        language="he",
        strategy_foundation=strategy,
        prototype=_prototype(),
        candidate=candidate,
        candidate_id="cand-1",
    )


def _winner_prompt(strategy: Dict[str, Any], candidate: Dict[str, Any], judgment: Dict[str, Any]) -> str:
    return build_winner_development_prompt(
        product_name=PRODUCT_NAME,
        product_description=RAW_PERFUME,
        language="he",
        strategy_foundation=strategy,
        winning_candidate=candidate,
        winning_judgment=judgment,
        prototype=_prototype(),
        runway_mode="silent",
        preservation_snapshot={},
    )


def _bad_lamp_judgment() -> Dict[str, Any]:
    base = _judgment("cand-bad", eligible=False)
    base.update(methodology_judgment_extras(prototype_id="greenpeace_essential_pairing"))
    base["essentialFactFusionAssessment"] = {
        "productCategoryEssentialFactPreserved": False,
        "relativeAdvantageEssentialFactPreserved": True,
        "productCategoryAppliedInVisualMechanism": False,
        "relativeAdvantageAppliedInVisualMechanism": True,
        "factsIntegratedIntoOneMechanism": False,
        "advantageVisualizedWithoutProductApplication": True,
        "unselectedFactIntroduced": False,
        "fusionRequired": True,
        "fusionEligible": False,
        "notes": "Generic foreign-to-Israeli lamp swap expresses advantage without perfume/category mechanism.",
    }
    return apply_fusion_eligibility_rules(base)


def _fused_judgment() -> Dict[str, Any]:
    base = _judgment("cand-good", eligible=True)
    base.update(methodology_judgment_extras(prototype_id="greenpeace_essential_pairing"))
    base.update(
        methodology_judge_fusion_extras(
            fusion_required=True,
            notes="Perfume category and Israeli identity fuse in one integrated visual mechanism.",
        )
    )
    assessment = base["essentialFactFusionAssessment"]
    assessment["fusionRequired"] = True
    assessment["fusionEligible"] = True
    return apply_fusion_eligibility_rules(base)


class TestStrategySelection(unittest.TestCase):
    def test_strategy_receives_full_raw_product_description(self) -> None:
        prompt = build_strategy_prompt(
            product_name=PRODUCT_NAME,
            product_description=RAW_PERFUME,
            language="he",
        )
        self.assertIn(RAW_PERFUME.strip().split("\n")[0], prompt)
        self.assertIn("<product_description>", prompt)

    def test_strategy_emits_fact_taxonomy(self) -> None:
        prompt = build_strategy_prompt(
            product_name=PRODUCT_NAME,
            product_description=RAW_PERFUME,
            language="he",
        )
        for bucket in ("essentialFacts", "supportingEvidence", "mandatoryConstraints", "discardedFacts"):
            self.assertIn(bucket, prompt)
        self.assertIn(NEGATIVE_SELECTION_METHODOLOGY, prompt)

    def test_irrelevant_fact_becomes_discarded(self) -> None:
        brief = get_product_semantic_brief(_bosa_strategy())
        discarded = [item["text"] for item in brief.get("discardedFacts") or []]
        self.assertIn(DISCARDED_JAZZ, discarded)

    def test_discarded_fact_stored_for_audit(self) -> None:
        brief = get_product_semantic_brief(_bosa_strategy())
        stored = json.dumps(brief, ensure_ascii=False)
        self.assertIn(DISCARDED_WAREHOUSE, stored)


class TestPostStrategyIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _bosa_strategy()
        self.brief = get_product_semantic_brief(self.strategy)
        self.candidate = _candidate("greenpeace_essential_pairing")
        self.candidate.update(methodology_candidate_extras("greenpeace_essential_pairing", strategy=self.strategy))

    def test_discarded_fact_not_in_creator_prompt(self) -> None:
        prompt = _creator_prompt(self.strategy)
        self.assertFalse(prompt_contains_discarded_facts(prompt, self.brief))

    def test_raw_description_not_in_creator_prompt(self) -> None:
        prompt = _creator_prompt(self.strategy)
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_source_description_not_leaked_in_creator_strategy_json(self) -> None:
        prompt = _creator_prompt(self.strategy)
        slim = strategy_json_for_post_strategy_prompt(self.strategy)
        self.assertNotIn("sourceDescription", slim)
        self.assertNotIn("productInputAudit", slim)
        self.assertFalse(prompt_contains_source_description_leak(prompt))

    def test_judge_prompt_excludes_raw_description(self) -> None:
        prompt = _judge_prompt(self.strategy, self.candidate)
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_judge_prompt_excludes_discarded_facts(self) -> None:
        prompt = _judge_prompt(self.strategy, self.candidate)
        self.assertFalse(prompt_contains_discarded_facts(prompt, self.brief))

    def test_winner_development_excludes_raw_description(self) -> None:
        prompt = _winner_prompt(self.strategy, self.candidate, _fused_judgment())
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_winner_development_excludes_discarded_facts(self) -> None:
        prompt = _winner_prompt(self.strategy, self.candidate, _fused_judgment())
        self.assertFalse(prompt_contains_discarded_facts(prompt, self.brief))

    def test_essential_facts_reach_creator_judge_winner(self) -> None:
        essential = "בושם לגברים"
        for prompt in (
            _creator_prompt(self.strategy),
            _judge_prompt(self.strategy, self.candidate),
            _winner_prompt(self.strategy, self.candidate, _fused_judgment()),
        ):
            self.assertIn(essential, prompt)

    def test_mandatory_constraints_survive(self) -> None:
        prompt = _creator_prompt(self.strategy)
        self.assertIn("Silent video without dialogue", prompt)

    def test_restricted_capabilities_remain_enforced(self) -> None:
        brief = summarize_brief_for_creative_prompt(self.brief)
        self.assertIn("restrictedCapabilities", json.dumps(brief, ensure_ascii=False))

    def test_slim_strategy_view_fields(self) -> None:
        slim = build_slim_strategy_foundation_for_prompts(self.strategy)
        self.assertIn("problemPerception", slim)
        self.assertIn("relativeAdvantage", slim)
        self.assertIn("productSemanticBrief", slim)
        self.assertNotIn("sourceDescription", json.dumps(slim, ensure_ascii=False))
        creative = slim["productSemanticBrief"]
        self.assertIn("essentialFacts", creative)
        self.assertNotIn("discardedFacts", creative)


class TestEssentialFactFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = _bosa_strategy()
        self.brief = get_product_semantic_brief(self.strategy)

    def test_fusion_required_for_bosa_brief(self) -> None:
        self.assertTrue(fusion_required_for_brief(self.brief))

    def test_bosa_bad_lamp_ineligible(self) -> None:
        judgment = _bad_lamp_judgment()
        self.assertTrue(judgment_rejects_essential_fact_fusion(judgment))
        self.assertFalse(judgment.get("eligible"))

    def test_bosa_fused_candidate_may_remain_eligible(self) -> None:
        judgment = _fused_judgment()
        self.assertFalse(judgment_rejects_essential_fact_fusion(judgment))
        self.assertTrue(judgment.get("eligible"))

    def test_product_led_valid_mechanism_passes(self) -> None:
        judgment = _fused_judgment()
        judgment["essentialFactFusionAssessment"]["productCategoryAppliedInVisualMechanism"] = True
        judgment["essentialFactFusionAssessment"]["factsIntegratedIntoOneMechanism"] = True
        judgment = apply_fusion_eligibility_rules(judgment)
        self.assertFalse(judgment_rejects_essential_fact_fusion(judgment))

    def test_generic_advantage_only_analogy_fails(self) -> None:
        self.assertTrue(judgment_rejects_essential_fact_fusion(_bad_lamp_judgment()))

    def test_select_global_winner_blocks_fusion_reject(self) -> None:
        state = {
            "candidates": {
                "cand-bad": {
                    "eligible": True,
                    "creatorAcceptanceStatus": "accepted",
                    "judgmentId": "j-bad",
                    "totalScore": 90,
                },
                "cand-good": {
                    "eligible": True,
                    "creatorAcceptanceStatus": "accepted",
                    "judgmentId": "j-good",
                    "totalScore": 80,
                },
            },
            "judgments": {
                "j-bad": {"judgment": _bad_lamp_judgment()},
                "j-good": {"judgment": _fused_judgment()},
            },
        }
        self.assertEqual(select_global_winner(state), "cand-good")

    def test_creator_fusion_methodology_present(self) -> None:
        prompt = _creator_prompt(self.strategy)
        self.assertIn(BUILDER2_ESSENTIAL_FACT_FUSION, prompt)
        self.assertIn("essentialFactFusionEvidence", prompt)

    def test_cultural_context_in_creator_and_judge(self) -> None:
        creator_prompt = _creator_prompt(self.strategy)
        candidate = _candidate("greenpeace_essential_pairing")
        judge_prompt = _judge_prompt(self.strategy, candidate)
        self.assertIn(BUILDER2_CULTURAL_CONTEXT, creator_prompt)
        self.assertIn("Assess cultural meaning in context", judge_prompt)


class TestContractsPreserved(unittest.TestCase):
    def test_no_logo_contract_in_creator_prompt(self) -> None:
        prompt = _creator_prompt(_bosa_strategy())
        self.assertIn("No-logo policy", prompt)

    def test_silent_movie_contract_in_creator_prompt(self) -> None:
        prompt = _creator_prompt(_bosa_strategy())
        self.assertIn("silent", prompt.casefold())

    def test_negative_selection_validation_rejects_duplicates(self) -> None:
        brief = {
            "essentialFacts": [{"id": "e1", "text": "Same fact"}],
            "discardedFacts": [{"id": "d1", "text": "Same fact"}],
        }
        reasons = validate_fact_selection_brief(brief, strict=True)
        self.assertTrue(any("duplicate" in reason for reason in reasons))


class TestResumeRepairIsolation(unittest.TestCase):
    def test_creator_repair_preserves_isolation(self) -> None:
        strategy = _bosa_strategy()
        prompt = build_creator_repair_prompt(
            product_name=PRODUCT_NAME,
            product_description=RAW_PERFUME,
            language="he",
            strategy_foundation=strategy,
            prototype=_prototype(),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="silent",
            invalid_output=_candidate("greenpeace_essential_pairing"),
            validation_failures=["builder2_creator_validation_failed:creatorReport.mechanismScanSummary"],
        )
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_judge_repair_preserves_isolation(self) -> None:
        strategy = _bosa_strategy()
        candidate = _candidate("greenpeace_essential_pairing")
        prompt = build_judge_repair_prompt(
            product_name=PRODUCT_NAME,
            product_description=RAW_PERFUME,
            language="he",
            strategy_foundation=strategy,
            prototype=_prototype(),
            candidate=candidate,
            candidate_id="cand-1",
            invalid_output=_judgment("cand-1"),
            validation_failures=["builder2_judge_validation_failed:essentialFactFusionAssessment.notes"],
        )
        self.assertFalse(prompt_contains_raw_product_description(prompt, RAW_PERFUME))

    def test_headline_repair_does_not_leak_source_description(self) -> None:
        strategy = _bosa_strategy()
        prompt = build_winner_headline_repair_prompt(
            product_name=PRODUCT_NAME,
            language="he",
            strategy_foundation=strategy,
            winning_candidate=_candidate("greenpeace_essential_pairing"),
            winning_judgment=_fused_judgment(),
            parsed_winner_plan={"headlineForm": "direct", "headlineDecision": {"decision": "use"}},
            validation_failures=["builder2_winner_validation_failed:headline"],
        )
        slim = strategy_json_for_post_strategy_prompt(strategy)
        self.assertNotIn("sourceDescription", slim)
        self.assertFalse(prompt_contains_source_description_leak(prompt))


class TestCompatibility(unittest.TestCase):
    def test_legacy_strategy_allows_raw_description_fallback(self) -> None:
        from engine.builder2_product_brief_production_guard import PRODUCT_BRIEF_MODE_LEGACY_COMPAT

        strategy = _legacy_strategy_without_taxonomy()
        state = {"productBriefMode": PRODUCT_BRIEF_MODE_LEGACY_COMPAT, "productBriefModeDecided": True}
        prompt = build_creator_prompt(
            product_name=PRODUCT_NAME,
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

    def test_isolation_not_required_without_essential_facts(self) -> None:
        from engine.builder2_post_strategy_isolation import post_strategy_isolation_required
        from engine.builder2_product_brief_production_guard import PRODUCT_BRIEF_MODE_LEGACY_COMPAT

        strategy = _legacy_strategy_without_taxonomy()
        self.assertFalse(
            post_strategy_isolation_required(
                strategy,
                product_brief_mode=PRODUCT_BRIEF_MODE_LEGACY_COMPAT,
            )
        )
        brief = get_product_semantic_brief(strategy, product_description=RAW_PERFUME)
        self.assertFalse(fusion_required_for_brief(brief))


class TestBuilder1Untouched(unittest.TestCase):
    def test_builder1_modules_not_imported_by_builder2_selection(self) -> None:
        import engine.builder2_post_strategy_isolation as isolation
        import engine.builder2_fact_selection as selection
        import engine.builder2_essential_fact_fusion as fusion

        for module in (isolation, selection, fusion):
            source_path = module.__file__ or ""
            self.assertNotIn("builder1", source_path)


class TestFusionEvidenceValidation(unittest.TestCase):
    def test_creator_fusion_evidence_fixture(self) -> None:
        extras = methodology_creator_fusion_extras()
        evidence = extras["creatorReport"]["essentialFactFusionEvidence"]
        self.assertFalse(evidence["usedFactOutsideSelectedBrief"])

    def test_judge_detects_unselected_fact_introduction(self) -> None:
        judgment = _fused_judgment()
        judgment["essentialFactFusionAssessment"]["unselectedFactIntroduced"] = True
        judgment["essentialFactFusionAssessment"]["fusionEligible"] = False
        judgment = apply_fusion_eligibility_rules(judgment)
        self.assertTrue(judgment_rejects_essential_fact_fusion(judgment))


if __name__ == "__main__":
    unittest.main()
