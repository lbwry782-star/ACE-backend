"""
Builder2 negative selection — strategic fact-role taxonomy for productSemanticBrief.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from engine.builder2_product_semantic_brief import _clean, _normalize_fact_items

NEGATIVE_SELECTION_METHODOLOGY = """
NEGATIVE SELECTION — classify every material product fact before creative development:

1. essentialFacts — product/category identity, chosen strategic problem, relative advantage,
   or facts that must survive into the advertising proposition. May participate in Essential Fact Fusion.

2. supportingEvidence — grounds Strategy but does not need to appear in the visual mechanism.

3. mandatoryConstraints — execution constraints (language, silent video, no-logo, show/hide product,
   factual capability limits, explicit user instructions). Obey downstream; do not dramatize.

4. discardedFacts — true but strategically irrelevant to THIS campaign. Audit/diagnostics only.
   Never creative input downstream.

Selection is inclusion AND exclusion. Discard facts that are true but unnecessary — not facts that
are boring. A mundane fact may be essential when Strategy selects it as the advantage.
""".strip()

_SEGMENT_SPLIT_RE = re.compile(r"[\n\r]+|[.;]|(?:^\s*[-*•]\s+)", re.MULTILINE)
_MIN_SEGMENT_CHARS = 8
_OVER_SELECTION_MIN_RAW_SEGMENTS = 4


def _dedupe_key(text: str) -> str:
    return _clean(text).casefold()


def count_raw_factual_segments(product_description: str) -> int:
    text = _clean(product_description)
    if not text:
        return 0
    segments = [_clean(part) for part in _SEGMENT_SPLIT_RE.split(text)]
    return len([segment for segment in segments if len(segment) >= _MIN_SEGMENT_CHARS])


def find_cross_bucket_duplicates(brief: Dict[str, Any]) -> List[str]:
    seen: dict[str, str] = {}
    duplicates: List[str] = []
    buckets = (
        ("essentialFacts", brief.get("essentialFacts") or []),
        ("supportingEvidence", brief.get("supportingEvidence") or []),
        ("mandatoryConstraints", brief.get("mandatoryConstraints") or []),
        ("discardedFacts", brief.get("discardedFacts") or []),
    )
    for bucket_name, items in buckets:
        for item in items:
            text = _clean(item.get("text") if isinstance(item, dict) else item)
            key = _dedupe_key(text)
            if not key:
                continue
            prior = seen.get(key)
            if prior and prior != bucket_name:
                duplicates.append(
                    f"productSemanticBrief:duplicate_fact_across_{prior}_and_{bucket_name}"
                )
            else:
                seen[key] = bucket_name
    return list(dict.fromkeys(duplicates))


def validate_over_selection(brief: Dict[str, Any], *, product_description: str) -> List[str]:
    reasons: List[str] = []
    raw_segments = count_raw_factual_segments(product_description)
    if raw_segments < _OVER_SELECTION_MIN_RAW_SEGMENTS:
        return reasons
    essential = _normalize_fact_items(brief.get("essentialFacts"))
    discarded = _normalize_fact_items(brief.get("discardedFacts"))
    if not discarded and len(essential) >= raw_segments:
        reasons.append("productSemanticBrief:over_selection_no_discards_when_raw_is_multifact")
    return reasons


def validate_fact_selection_brief(
    brief: Dict[str, Any],
    *,
    product_description: str = "",
    strict: bool = True,
) -> List[str]:
    reasons: List[str] = []
    essential = _normalize_fact_items(brief.get("essentialFacts"))
    if strict and not essential:
        reasons.append("productSemanticBrief:essentialFacts_empty")
    reasons.extend(find_cross_bucket_duplicates(brief))
    if strict and product_description.strip():
        reasons.extend(validate_over_selection(brief, product_description=product_description))
    return list(dict.fromkeys(reasons))


def normalize_fact_selection_on_brief(
    brief: Dict[str, Any],
    *,
    product_description: str = "",
) -> Dict[str, Any]:
    """Ensure selection buckets exist; migrate legacy explicitFacts when needed."""
    out = dict(brief)
    essential = _normalize_fact_items(out.get("essentialFacts"))
    if not essential:
        essential = _normalize_fact_items(out.get("explicitFacts"))
    supporting = _normalize_fact_items(out.get("supportingEvidence"))
    mandatory = _normalize_fact_items(out.get("mandatoryConstraints"))
    discarded = _normalize_fact_items(out.get("discardedFacts"))
    out["essentialFacts"] = essential
    out["supportingEvidence"] = supporting
    out["mandatoryConstraints"] = mandatory
    out["discardedFacts"] = discarded
    if essential and not out.get("explicitFacts"):
        out["explicitFacts"] = list(essential)
    elif essential:
        out["explicitFacts"] = _normalize_fact_items(out.get("explicitFacts")) or list(essential)
    return out
