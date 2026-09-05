"""
Builder2 product semantic brief production guard — v2 taxonomy vs legacy compatibility.

New production jobs MUST commit the full v2 fact-selection taxonomy and MUST NOT silently
fall back to raw productDescription post-Strategy prompts. Legacy compatibility is allowed
only for genuinely persisted pre-v2 strategyFoundation state on resumed jobs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from engine.builder2_product_semantic_brief import (
    BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2,
    get_product_semantic_brief,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

PRODUCT_BRIEF_MODE_V2_SELECTED = "v2_selected"
PRODUCT_BRIEF_MODE_LEGACY_COMPAT = "legacy_compat"

V2_REQUIRED_BUCKETS: tuple[str, ...] = (
    "essentialFacts",
    "supportingEvidence",
    "mandatoryConstraints",
    "discardedFacts",
)

_LOG_PREFIX = "BUILDER2_PRODUCT_BRIEF_MODE"


def _strategy_brief(strategy_foundation: Dict[str, Any]) -> Dict[str, Any]:
    return get_product_semantic_brief(strategy_foundation)


def has_complete_v2_product_brief_taxonomy(brief: Dict[str, Any]) -> bool:
    if not isinstance(brief, dict):
        return False
    if str(brief.get("briefVersion") or "").strip() != BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2:
        return False
    for bucket in V2_REQUIRED_BUCKETS:
        if not isinstance(brief.get(bucket), list):
            return False
    essential = brief.get("essentialFacts") or []
    return bool(essential)


def strategy_has_v2_product_brief_taxonomy(strategy_foundation: Dict[str, Any]) -> bool:
    return has_complete_v2_product_brief_taxonomy(_strategy_brief(strategy_foundation))


def collect_v2_taxonomy_missing_fields(brief: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not isinstance(brief, dict):
        return ["productSemanticBrief"]
    if str(brief.get("briefVersion") or "").strip() != BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2:
        missing.append("briefVersion")
    for bucket in V2_REQUIRED_BUCKETS:
        if not isinstance(brief.get(bucket), list):
            missing.append(bucket)
    if isinstance(brief.get("essentialFacts"), list) and not brief.get("essentialFacts"):
        missing.append("essentialFacts_empty")
    return list(dict.fromkeys(missing))


def validate_v2_product_brief_taxonomy_for_new_production(
    brief: Dict[str, Any],
    *,
    compatibility_mode: bool = False,
) -> None:
    if compatibility_mode:
        return
    missing = collect_v2_taxonomy_missing_fields(brief)
    if missing:
        raise Builder2TournamentError(
            f"builder2_strategy_validation_failed:strategyEvidenceGrounding.productSemanticBrief.v2_taxonomy_incomplete.{missing[0]}"
        )


def is_persisted_pre_v2_product_brief_state(state: Dict[str, Any]) -> bool:
    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        return False
    block = strategy.get("strategyEvidenceGrounding")
    if not isinstance(block, dict):
        return True
    brief = block.get("productSemanticBrief")
    if not isinstance(brief, dict):
        return True
    return not has_complete_v2_product_brief_taxonomy(brief)


def log_product_brief_mode(
    state: Optional[Dict[str, Any]],
    *,
    mode: str,
    reason: str = "",
) -> None:
    job_id = (state or {}).get("jobId") or ""
    tournament_id = (state or {}).get("tournamentId") or ""
    if reason:
        logger.info("%s mode=%s jobId=%s tournamentId=%s reason=%s", _LOG_PREFIX, mode, job_id, tournament_id, reason)
    else:
        logger.info("%s mode=%s jobId=%s tournamentId=%s", _LOG_PREFIX, mode, job_id, tournament_id)


def ensure_product_brief_mode_decided(state: Dict[str, Any], *, is_new_job: bool) -> str:
    """
    Decide and persist job-local product brief mode once.
    New jobs always require v2_selected.
    """
    if is_new_job:
        state["productBriefMode"] = PRODUCT_BRIEF_MODE_V2_SELECTED
        state["productBriefModeDecided"] = True
        state.pop("productBriefModeReason", None)
        log_product_brief_mode(state, mode=PRODUCT_BRIEF_MODE_V2_SELECTED)
        return PRODUCT_BRIEF_MODE_V2_SELECTED

    if state.get("productBriefModeDecided"):
        mode = str(state.get("productBriefMode") or PRODUCT_BRIEF_MODE_V2_SELECTED)
        return mode

    if is_persisted_pre_v2_product_brief_state(state):
        state["productBriefMode"] = PRODUCT_BRIEF_MODE_LEGACY_COMPAT
        state["productBriefModeReason"] = "persisted_pre_v2_product_brief"
        log_product_brief_mode(
            state,
            mode=PRODUCT_BRIEF_MODE_LEGACY_COMPAT,
            reason=state["productBriefModeReason"],
        )
    else:
        state["productBriefMode"] = PRODUCT_BRIEF_MODE_V2_SELECTED
        state.pop("productBriefModeReason", None)
        log_product_brief_mode(state, mode=PRODUCT_BRIEF_MODE_V2_SELECTED)

    state["productBriefModeDecided"] = True
    return str(state.get("productBriefMode") or PRODUCT_BRIEF_MODE_V2_SELECTED)


def resolve_product_brief_mode(
    *,
    strategy_foundation: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    explicit_mode: Optional[str] = None,
) -> str:
    if explicit_mode in (PRODUCT_BRIEF_MODE_V2_SELECTED, PRODUCT_BRIEF_MODE_LEGACY_COMPAT):
        return explicit_mode
    if state is not None:
        if not state.get("productBriefModeDecided"):
            return ensure_product_brief_mode_decided(state, is_new_job=False)
        return str(state.get("productBriefMode") or PRODUCT_BRIEF_MODE_V2_SELECTED)
    if strategy_has_v2_product_brief_taxonomy(strategy_foundation):
        return PRODUCT_BRIEF_MODE_V2_SELECTED
    return PRODUCT_BRIEF_MODE_V2_SELECTED


def post_strategy_isolation_required(
    strategy_foundation: Dict[str, Any],
    *,
    product_brief_mode: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    mode = resolve_product_brief_mode(
        strategy_foundation=strategy_foundation,
        state=state,
        explicit_mode=product_brief_mode,
    )
    if mode == PRODUCT_BRIEF_MODE_LEGACY_COMPAT:
        return False
    return True


def assert_v2_taxonomy_before_post_strategy_prompt(
    strategy_foundation: Dict[str, Any],
    *,
    product_brief_mode: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    mode = resolve_product_brief_mode(
        strategy_foundation=strategy_foundation,
        state=state,
        explicit_mode=product_brief_mode,
    )
    if mode == PRODUCT_BRIEF_MODE_LEGACY_COMPAT:
        return
    if not strategy_has_v2_product_brief_taxonomy(strategy_foundation):
        raise Builder2TournamentError("builder2_product_brief_v2_taxonomy_required")


def build_product_input_block_for_prompt(
    *,
    strategy_foundation: Dict[str, Any],
    product_description: str,
    product_brief_mode: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    from engine.builder2_post_strategy_isolation import (
        format_post_strategy_product_input_block,
    )
    from engine.builder2_product_semantic_brief import format_product_description_data_block

    mode = resolve_product_brief_mode(
        strategy_foundation=strategy_foundation,
        state=state,
        explicit_mode=product_brief_mode,
    )
    if mode == PRODUCT_BRIEF_MODE_LEGACY_COMPAT:
        return format_product_description_data_block(product_description)
    assert_v2_taxonomy_before_post_strategy_prompt(
        strategy_foundation,
        product_brief_mode=mode,
        state=state,
    )
    return format_post_strategy_product_input_block(strategy_foundation)
