"""
Builder2 methodology contract — canonical creative methodology coverage map.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Tuple

from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS, REFERENCE_ONLY_PROTOTYPE_IDS

METHODOLOGY_VERSION = "builder2_methodology_v1"

CREATIVE_STAGE_ORDER: Tuple[str, ...] = (
    "problem_perception",
    "relative_advantage",
    "mechanism_scan",
    "assigned_prototype_method",
    "visual_parallel",
    "runway_feasibility",
    "video_structure",
    "verbal_mechanism",
    "optional_headline",
)

ACTIVE_PROTOTYPE_IDS: Tuple[str, ...] = DEFAULT_ACTIVE_PROTOTYPE_IDS

REFERENCE_ONLY_METHOD_IDS: FrozenSet[str] = REFERENCE_ONLY_PROTOTYPE_IDS | frozenset(
    {
        "shared_word_line_mechanism",
        "old_commercial_code_inversion",
    }
)

# Authoritative Builder2 enum source — import these from this module elsewhere.
VALID_HEADLINE_DECISIONS: FrozenSet[str] = frozenset({"include", "omit"})

VALID_HEADLINE_FORMS: FrozenSet[str] = frozenset(
    {"expression_replacement", "direct", "planned_contradiction", "other", "none"}
)

VALID_STRUCTURE_TYPES: FrozenSet[str] = frozenset({"continuous_event", "variation_montage"})

VALID_CONTINUITY_RISK: FrozenSet[str] = frozenset({"low", "medium", "high"})

VALID_VISUAL_PARALLEL_TYPES: FrozenSet[str] = frozenset(
    {
        "replacement",
        "side_by_side",
        "motion_similarity",
        "motion",
        "physical_behavior",
        "graphic_similarity",
        "graphic",
        "structural",
        "context_collision",
        "context_replacement",
        "media_replacement",
        "medium_as_object",
        "essential_pairing",
        "spatial_proximity",
        "proximity",
        "consequence_embodiment",
        "consequence",
        "omission",
        "other",
    }
)

VISUAL_PARALLEL_CANONICAL_ALIASES: Dict[str, str] = {
    "motion": "motion_similarity",
    "graphic": "graphic_similarity",
    "proximity": "spatial_proximity",
    "consequence": "consequence_embodiment",
}

VALID_SOURCE_CONCEPT_TYPES: FrozenSet[str] = frozenset({"native_builder2", "builder1_adaptation"})

VALID_REPLACEMENT_CARRIERS: FrozenSet[str] = frozenset(
    {"placement", "behavior", "motion", "use", "framing", "camera_reveal", "other"}
)

VALID_PROVOCATION_RISK: FrozenSet[str] = frozenset({"none", "low", "medium", "high"})

PROCESS_FAILURE_TAGS: Tuple[str, ...] = (
    "problem_not_grounded",
    "advantage_not_derived",
    "mechanism_too_surface",
    "prototype_surface_copy",
    "visual_not_silent",
    "runway_infeasible",
    "headline_rescuing_visual",
    "visual_family_incoherent",
    "strategy_identity_mismatch",
    "winner_mechanism_changed",
    "winner_downstream_type_mismatch",
    "headline_composition_invalid",
    "pre_runway_contract_invalid",
)

METHODOLOGY_LOG_EVENTS: Tuple[str, ...] = (
    "BUILDER2_METHODOLOGY_VERSION_SELECTED",
    "BUILDER2_STRATEGY_METHODOLOGY_VALIDATED",
    "BUILDER2_CREATOR_METHODOLOGY_VALIDATED",
    "BUILDER2_PROTOTYPE_METHOD_VALIDATED",
    "BUILDER2_CREATIVE_ORDER_VALIDATED",
    "BUILDER2_VISUAL_FAMILY_VALIDATED",
    "BUILDER2_PARTICIPATION_VALIDATED",
    "BUILDER2_VERBAL_POTENTIAL_VALIDATED",
    "BUILDER2_HEADLINE_DECISION_VALIDATED",
    "BUILDER2_JUDGE_METHODOLOGY_VALIDATED",
    "BUILDER2_WINNER_MECHANISM_PRESERVED",
    "BUILDER2_METHODOLOGY_COMPATIBILITY_MODE",
)

STRATEGY_FORBIDDEN_TOP_LEVEL_KEYS: FrozenSet[str] = frozenset(
    {
        "visualParallelType",
        "videoPrompt",
        "headline",
        "headlineText",
        "headlineCoreKeyword",
        "keyword",
        "assignedPrototypeId",
        "prototypeId",
        "candidateId",
        "structureType",
        "sevenSecondStructure",
    }
)

GENERIC_BUSINESS_GOAL_PATTERNS: Tuple[str, ...] = (
    "wants more customers",
    "needs more customers",
    "needs awareness",
    "market is competitive",
    "needs creative advertising",
    "increase brand awareness",
)

TOURNAMENT_MOTTO = "The tournament is between ideas, not prototypes."

INTEREST_PRIORITY_ORDER: Tuple[str, ...] = (
    "interesting",
    "realistic",
    "silently_verifiable",
    "mechanism_faithful",
    "keyword_isolated",
    "runway_simple",
    "everyday",
)

SECTION_IDS: Tuple[str, ...] = tuple(f"{index:02d}_{name}" for index, name in enumerate(
    [
        "scope_and_terminology",
        "builder2_purpose",
        "builder2_as_methodologist",
        "core_principle",
        "fixed_creative_order",
        "problem_perception",
        "grounding",
        "relative_advantage",
        "problem_advantage_integrity",
        "mechanism_scan",
        "mechanism_scan_vs_visual_parallel",
        "fixed_strategic_foundation",
        "role_of_gold_prototypes",
        "random_prototype_inspiration",
        "tournament_between_ideas",
        "active_prototype_set",
        "winning_card_method",
        "summer_fan_method",
        "forgot_method",
        "greenpeace_essential_pairing",
        "closest_method",
        "think_small_method",
        "reference_only_methods",
        "shared_visual_verbal_mechanism",
        "old_vaseline_market_code_inversion",
        "context_collision",
        "visual_parallel_families",
        "replacement_principle",
        "essence_extreme",
        "visual_family_consistency",
        "participation_first_and_visual_anchor",
        "silent_movie_and_structure",
        "runway_feasibility_before_wording",
        "interest_first",
        "verbal_parallel",
        "headline_rules_and_optional_headline",
        "editing_and_builder1_adaptation",
        "creator_judge_reports_and_purity",
        "manager_winner_and_learning",
    ],
    start=1,
))


def _entry(
    *,
    modules: Tuple[str, ...],
    enforcement: str,
    tests: Tuple[str, ...],
) -> Dict[str, Any]:
    return {"modules": modules, "enforcement": enforcement, "tests": tests}


BUILDER2_METHODOLOGY_COVERAGE: Dict[str, Dict[str, Any]] = {
    "01_scope_and_terminology": _entry(
        modules=("engine/builder2_methodology_contract.py",),
        enforcement="isolation_contract",
        tests=("tests.test_builder2_methodology.TestBuilder1Isolation",),
    ),
    "02_builder2_purpose": _entry(
        modules=("engine/builder2_tournament_prompts.py", "engine/builder2_methodology_contract.py"),
        enforcement="prompt_and_contract",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "03_builder2_as_methodologist": _entry(
        modules=("engine/builder2_strategy.py", "engine/builder2_methodology_validation.py"),
        enforcement="strategy_forbidden_fields",
        tests=("tests.test_builder2_methodology.TestStrategyMethodology",),
    ),
    "04_core_principle": _entry(
        modules=("engine/builder2_methodology_contract.py", "engine/builder2_tournament_prompts.py"),
        enforcement="creative_stage_order",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "05_fixed_creative_order": _entry(
        modules=("engine/builder2_creator.py", "engine/builder2_methodology_validation.py"),
        enforcement="creative_order_confirmation",
        tests=("tests.test_builder2_methodology.TestCreatorMethodology",),
    ),
    "06_problem_perception": _entry(
        modules=("engine/builder2_strategy.py", "engine/builder2_methodology_validation.py"),
        enforcement="strategy_problem_validation",
        tests=("tests.test_builder2_methodology.TestStrategyMethodology",),
    ),
    "07_grounding": _entry(
        modules=("engine/builder2_strategy.py", "engine/builder2_tournament_contracts.py"),
        enforcement="grounding_type_and_evidence",
        tests=("tests.test_builder2_strategy",),
    ),
    "08_relative_advantage": _entry(
        modules=("engine/builder2_strategy.py", "engine/builder2_methodology_validation.py"),
        enforcement="relative_advantage_fields",
        tests=("tests.test_builder2_methodology.TestStrategyMethodology",),
    ),
    "09_problem_advantage_integrity": _entry(
        modules=("engine/builder2_judge.py", "engine/builder2_tournament_prompts.py"),
        enforcement="judge_score_and_assessment",
        tests=("tests.test_builder2_judge", "tests.test_builder2_methodology.TestJudgeMethodology"),
    ),
    "10_mechanism_scan": _entry(
        modules=("engine/builder2_strategy.py", "engine/builder2_methodology_validation.py"),
        enforcement="mechanism_scan_depth",
        tests=("tests.test_builder2_methodology.TestStrategyMethodology",),
    ),
    "11_mechanism_scan_vs_visual_parallel": _entry(
        modules=("engine/builder2_creator.py", "engine/builder2_methodology_validation.py"),
        enforcement="visual_mechanism_separation",
        tests=("tests.test_builder2_methodology.TestCreatorMethodology",),
    ),
    "12_fixed_strategic_foundation": _entry(
        modules=("engine/builder2_tournament_manager.py", "engine/builder2_methodology_validation.py"),
        enforcement="strategy_identity_check",
        tests=("tests.test_builder2_methodology.TestStrategyMethodology",),
    ),
    "13_role_of_gold_prototypes": _entry(
        modules=("engine/builder2_prototypes.py", "engine/builder2_methodology_validation.py"),
        enforcement="prototype_method_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "14_random_prototype_inspiration": _entry(
        modules=("engine/builder2_tournament_manager.py", "engine/builder2_tournament_config.py"),
        enforcement="deck_shuffle_all_active",
        tests=("tests.test_builder2_tournament_corrections.TestOneRoundBehavior",),
    ),
    "15_tournament_between_ideas": _entry(
        modules=("engine/builder2_tournament_contracts.py", "engine/builder2_methodology_contract.py"),
        enforcement="ranking_without_prototype_bonus",
        tests=("tests.test_builder2_methodology.TestTournamentBetweenIdeas",),
    ),
    "16_active_prototype_set": _entry(
        modules=("engine/builder2_tournament_config.py", "engine/builder2_prototypes.py"),
        enforcement="active_prototype_ids",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "17_winning_card_method": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="winning_card_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "18_summer_fan_method": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="summer_fan_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "19_forgot_method": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="forgot_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "20_greenpeace_essential_pairing": _entry(
        modules=("engine/builder2_methodology_validation.py", "engine/builder2_creator.py"),
        enforcement="essential_pairing_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "21_closest_method": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="closest_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "22_think_small_method": _entry(
        modules=("engine/builder2_methodology_validation.py", "engine/builder2_creator.py"),
        enforcement="think_small_application",
        tests=("tests.test_builder2_methodology.TestPrototypeMethodology",),
    ),
    "23_reference_only_methods": _entry(
        modules=("engine/builder2_tournament_config.py", "engine/builder2_prototypes.py"),
        enforcement="reference_only_exclusion",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "24_shared_visual_verbal_mechanism": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="verbal_potential",
        tests=("tests.test_builder2_methodology.TestVerbalLayer",),
    ),
    "25_old_vaseline_market_code_inversion": _entry(
        modules=("engine/builder2_tournament_prompts.py", "engine/builder2_methodology_contract.py"),
        enforcement="strategy_prompt_guidance",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "26_context_collision": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="context_collision_safeguard",
        tests=("tests.test_builder2_methodology.TestVisualMethodology",),
    ),
    "27_visual_parallel_families": _entry(
        modules=("engine/builder2_tournament_contracts.py", "engine/builder2_creator.py"),
        enforcement="visual_parallel_enum",
        tests=("tests.test_builder2_methodology.TestVisualMethodology",),
    ),
    "28_replacement_principle": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="replacement_check",
        tests=("tests.test_builder2_methodology.TestVisualMethodology",),
    ),
    "29_essence_extreme": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="essence_extreme_fields",
        tests=("tests.test_builder2_methodology.TestCreatorMethodology",),
    ),
    "30_visual_family_consistency": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="visual_family_consistency",
        tests=("tests.test_builder2_methodology.TestVisualMethodology",),
    ),
    "31_participation_first_and_visual_anchor": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="participation_and_anchor",
        tests=("tests.test_builder2_methodology.TestCreatorMethodology",),
    ),
    "32_silent_movie_and_structure": _entry(
        modules=("engine/builder2_creator.py", "engine/builder2_methodology_validation.py"),
        enforcement="silent_and_structure",
        tests=("tests.test_builder2_methodology.TestSilentRunway",),
    ),
    "33_runway_feasibility_before_wording": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="runway_before_verbal",
        tests=("tests.test_builder2_methodology.TestSilentRunway",),
    ),
    "34_interest_first": _entry(
        modules=("engine/builder2_tournament_prompts.py", "engine/builder2_methodology_contract.py"),
        enforcement="judge_rubric_prompt",
        tests=("tests.test_builder2_methodology.TestMethodologyCoverage",),
    ),
    "35_verbal_parallel": _entry(
        modules=("engine/builder2_methodology_validation.py", "engine/builder2_judge.py"),
        enforcement="verbal_potential_and_judge_assessment",
        tests=("tests.test_builder2_methodology.TestVerbalLayer",),
    ),
    "36_headline_rules_and_optional_headline": _entry(
        modules=("engine/builder2_winner_plan.py", "engine/builder2_methodology_validation.py"),
        enforcement="headline_decision_winner_only",
        tests=("tests.test_builder2_methodology.TestVerbalLayer",),
    ),
    "37_editing_and_builder1_adaptation": _entry(
        modules=("engine/builder2_methodology_validation.py",),
        enforcement="editing_and_source_concept",
        tests=("tests.test_builder2_methodology.TestEditingAdaptation",),
    ),
    "38_creator_judge_reports_and_purity": _entry(
        modules=("engine/builder2_creator.py", "engine/builder2_judge.py"),
        enforcement="purity_and_reports",
        tests=("tests.test_builder2_creator", "tests.test_builder2_judge"),
    ),
    "39_manager_winner_and_learning": _entry(
        modules=("engine/builder2_tournament_manager.py", "engine/builder2_tournament_store.py"),
        enforcement="manager_persistence_no_cross_job_memory",
        tests=("tests.test_builder2_methodology.TestManagerLearning", "tests.test_builder2_tournament_corrections"),
    ),
}


def methodology_section_ids() -> Tuple[str, ...]:
    return SECTION_IDS


def assert_full_coverage_map() -> None:
    missing = [sid for sid in SECTION_IDS if sid not in BUILDER2_METHODOLOGY_COVERAGE]
    if missing:
        raise AssertionError(f"Missing methodology coverage entries: {missing}")


def prompt_enum_list(values: Tuple[str, ...] | FrozenSet[str]) -> str:
    """Comma-separated sorted enum list for prompts — always derived from canonical constants."""
    return ", ".join(sorted(values))


def resolve_coverage_test_target(target: str) -> tuple[str, str]:
    """
    Resolve a coverage-map test reference to (module_path, class_or_function_name).
    Accepts dotted paths like tests.test_builder2_methodology.TestStrategyMethodology.
    """
    if "." not in target:
        raise ValueError(f"Invalid coverage test target: {target!r}")
    module_path, _, name = target.rpartition(".")
    if not module_path or not name:
        raise ValueError(f"Invalid coverage test target: {target!r}")
    return module_path, name
