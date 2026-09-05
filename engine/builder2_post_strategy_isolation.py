"""
Builder2 post-Strategy creative isolation.

After Strategy commits productSemanticBrief fact selection, downstream reasoning
roles must not receive unrestricted raw productDescription or audit-only fields.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from engine.builder2_product_semantic_brief import (
    BUILDER2_PRODUCT_SEMANTIC_BRIEF_V2,
    get_product_semantic_brief,
    summarize_brief_for_creative_prompt,
)

POST_STRATEGY_ISOLATION_NOTICE = (
    "Post-Strategy creative input (selection boundary — use only the committed semantic brief "
    "and frozen strategy fields; do not reopen raw productDescription or discardedFacts):"
)


def has_fact_selection_taxonomy(brief: Dict[str, Any]) -> bool:
    from engine.builder2_product_brief_production_guard import has_complete_v2_product_brief_taxonomy

    return has_complete_v2_product_brief_taxonomy(brief)


def post_strategy_isolation_required(
    strategy_foundation: Dict[str, Any],
    *,
    product_brief_mode: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    from engine.builder2_product_brief_production_guard import post_strategy_isolation_required as _guard_required

    return _guard_required(
        strategy_foundation,
        product_brief_mode=product_brief_mode,
        state=state,
    )


def _selected_fact_text_keys(brief: Dict[str, Any]) -> set[str]:
    from engine.builder2_fact_selection import _dedupe_key

    keys: set[str] = set()
    for bucket in ("essentialFacts", "supportingEvidence", "mandatoryConstraints"):
        for item in brief.get(bucket) or []:
            text = str(item.get("text") if isinstance(item, dict) else item or "").strip()
            if text:
                keys.add(_dedupe_key(text))
    return keys


def _sanitize_relative_advantage_for_prompt(
    relative_advantage: Dict[str, Any],
    brief: Dict[str, Any],
) -> Dict[str, Any]:
    from engine.builder2_fact_selection import _dedupe_key

    out = copy.deepcopy(relative_advantage)
    allowed = _selected_fact_text_keys(brief)
    if not allowed:
        return out
    filtered: List[Any] = []
    for item in out.get("relativeAdvantageEvidence") or []:
        text = str(item.get("text") if isinstance(item, dict) else item or "").strip()
        if text and _dedupe_key(text) in allowed:
            filtered.append(item)
    if filtered:
        out["relativeAdvantageEvidence"] = filtered
    else:
        out["relativeAdvantageEvidence"] = [
            item.get("text") if isinstance(item, dict) else item
            for item in (brief.get("essentialFacts") or [])[:3]
        ]
    return out


def build_slim_strategy_foundation_for_prompts(strategy_foundation: Dict[str, Any]) -> Dict[str, Any]:
    """Frozen strategic fields + creative brief buckets only — no raw audit leakage."""
    brief = get_product_semantic_brief(strategy_foundation)
    creative_brief = summarize_brief_for_creative_prompt(brief)
    slim: Dict[str, Any] = {
        "schemaVersion": strategy_foundation.get("schemaVersion"),
        "methodologyVersion": strategy_foundation.get("methodologyVersion"),
        "productNameResolved": strategy_foundation.get("productNameResolved"),
        "language": strategy_foundation.get("language"),
        "strategyFoundationId": strategy_foundation.get("strategyFoundationId"),
        "strategyFoundationDigest": strategy_foundation.get("strategyFoundationDigest"),
        "problemPerception": copy.deepcopy(strategy_foundation.get("problemPerception") or {}),
        "relativeAdvantage": _sanitize_relative_advantage_for_prompt(
            strategy_foundation.get("relativeAdvantage") or {},
            brief,
        ),
        "mechanismScan": copy.deepcopy(strategy_foundation.get("mechanismScan") or {}),
        "productSemanticBrief": creative_brief,
    }
    return slim


def format_post_strategy_product_input_block(strategy_foundation: Dict[str, Any]) -> str:
    brief = get_product_semantic_brief(strategy_foundation)
    body = summarize_brief_for_creative_prompt(brief)
    return f"{POST_STRATEGY_ISOLATION_NOTICE}\n{json.dumps(body, ensure_ascii=False, indent=2)}\n"


def strategy_json_for_post_strategy_prompt(
    strategy_foundation: Dict[str, Any],
    *,
    product_brief_mode: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    from engine.builder2_product_brief_production_guard import (
        assert_v2_taxonomy_before_post_strategy_prompt,
        resolve_product_brief_mode,
        PRODUCT_BRIEF_MODE_LEGACY_COMPAT,
    )

    mode = resolve_product_brief_mode(
        strategy_foundation=strategy_foundation,
        state=state,
        explicit_mode=product_brief_mode,
    )
    if mode == PRODUCT_BRIEF_MODE_LEGACY_COMPAT:
        payload = strategy_foundation
    else:
        assert_v2_taxonomy_before_post_strategy_prompt(
            strategy_foundation,
            product_brief_mode=mode,
            state=state,
        )
        payload = build_slim_strategy_foundation_for_prompts(strategy_foundation)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def prompt_contains_raw_product_description(prompt: str, raw_product_description: str) -> bool:
    raw = str(raw_product_description or "").strip()
    if not raw:
        return False
    if raw in prompt:
        return True
    markers = (
        f"<product_description>\n{raw}",
        "<product_description>",
    )
    return any(marker in prompt for marker in markers)


def prompt_contains_discarded_facts(prompt: str, brief: Dict[str, Any]) -> bool:
    if not isinstance(brief, dict):
        return False
    for item in brief.get("discardedFacts") or []:
        text = str(item.get("text") if isinstance(item, dict) else item or "").strip()
        if text and text in prompt:
            return True
    return False


def prompt_contains_source_description_leak(prompt: str) -> bool:
    return "sourceDescription" in prompt or "productInputAudit" in prompt
