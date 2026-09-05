"""
Builder1 negative selection — classify and discard strategically irrelevant facts.

Selection decides what to keep AND what not to use. discardedFacts are stored for
audit/diagnostics only — never injected into downstream creative prompts.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from engine.builder1_essential_fact_fusion import (
    _CATEGORY_IDENTITY_MARKERS,
    _contains_marker,
    classify_essential_fact,
    partition_essential_facts,
)
from engine.builder1_plan_parser import _norm_text
from engine.builder1_product_identity_guard import extract_product_category_identities
from engine.builder1_selected_creative_brief import SelectedCreativeBrief

NEGATIVE_SELECTION_METHODOLOGY = """
NEGATIVE SELECTION — classify every material fact before creative development:

1. essentialFacts — necessary to define product/category identity, the selected strategic
   problem, the selected relative advantage, or the proposition the campaign must express.
   May participate in Essential Fact Fusion. Do NOT dump every true detail here.

2. supportingEvidence — substantiates the chosen direction but does NOT need to appear in
   the visual mechanism. Supports reasoning and validation only.

3. mandatoryConstraints — execution constraints (language, show/hide product, factual limits,
   appearance requirements, explicit user instructions). Obey downstream; do not dramatize.

4. discardedFacts — true but strategically irrelevant to THIS campaign after selection.
   Explicitly exclude facts that do not serve the chosen strategy, required product/category
   identity, or mandatory constraints. A mundane fact may still be essential when it IS the
   advantage (e.g. 100ml when size is the strategy; made-in-Israel when local origin is).

Negative selection rule: discard facts that are true but unnecessary — not facts that are boring.
The selected creative brief should normally be narrower than the raw description when the raw
brief contains several unrelated details.

discardedFacts are for audit/diagnostics only. They must NOT re-enter creative reasoning.
""".strip()

CREATIVE_BRIEF_PROMPT_BUCKETS = (
    "essentialFacts",
    "supportingEvidence",
    "mandatoryConstraints",
)

_SEGMENT_SPLIT_RE = re.compile(r"[\n\r]+|[.;]|(?:^\s*[-*•]\s+)", re.MULTILINE)
_MIN_SEGMENT_CHARS = 8
_OVER_SELECTION_MIN_RAW_SEGMENTS = 4


def _dedupe_key(text: str) -> str:
    return _norm_text(text).casefold()


def count_raw_factual_segments(product_description: str) -> int:
    """Structural count of distinct factual clauses in raw productDescription."""
    text = _norm_text(product_description)
    if not text:
        return 0
    segments = [_norm_text(part) for part in _SEGMENT_SPLIT_RE.split(text)]
    return len([segment for segment in segments if len(segment) >= _MIN_SEGMENT_CHARS])


def find_cross_bucket_duplicates(brief: SelectedCreativeBrief) -> List[str]:
    """Return rejection codes for the same fact appearing in multiple buckets."""
    seen: dict[str, str] = {}
    duplicates: List[str] = []
    bucket_items = (
        ("essentialFacts", brief.essential_facts),
        ("supportingEvidence", brief.supporting_evidence),
        ("mandatoryConstraints", brief.mandatory_constraints),
        ("discardedFacts", brief.discarded_facts),
    )
    for bucket_name, items in bucket_items:
        for item in items:
            key = _dedupe_key(item)
            if not key:
                continue
            prior = seen.get(key)
            if prior and prior != bucket_name:
                duplicates.append(
                    f"selectedCreativeBrief:duplicate_fact_across_{prior}_and_{bucket_name}"
                )
            else:
                seen[key] = bucket_name
    return list(dict.fromkeys(duplicates))


def validate_over_selection(
    brief: SelectedCreativeBrief,
    *,
    product_description: str,
) -> List[str]:
    """
    Reject obvious over-selection: nearly every raw segment marked essential with no discards.
    """
    reasons: List[str] = []
    raw_segments = count_raw_factual_segments(product_description)
    if raw_segments < _OVER_SELECTION_MIN_RAW_SEGMENTS:
        return reasons

    essential_count = len(brief.essential_facts)
    kept_count = essential_count + len(brief.supporting_evidence) + len(brief.mandatory_constraints)

    if not brief.discarded_facts and essential_count >= raw_segments:
        reasons.append("selectedCreativeBrief:over_selection_no_discards_when_raw_is_multifact")

    if essential_count >= max(6, raw_segments - 1) and kept_count >= raw_segments:
        if not brief.discarded_facts:
            reasons.append("selectedCreativeBrief:over_selection_essential_facts_span_full_raw")

    return reasons


def validate_under_selection(
    brief: SelectedCreativeBrief,
    *,
    product_description: str,
) -> List[str]:
    """Reject when product/category identity required by raw brief is missing from essentials."""
    reasons: List[str] = []
    if not brief.essential_facts:
        return ["selectedCreativeBrief:essentialFacts_empty"]

    if not _raw_requires_product_category_identity(product_description):
        return reasons

    category_facts, _, _ = partition_essential_facts(brief.essential_facts)
    if category_facts:
        return reasons

    has_category_essential = any(
        classify_essential_fact(fact) == "category_identity" for fact in brief.essential_facts
    )
    if not has_category_essential:
        reasons.append("selectedCreativeBrief:under_selection_missing_product_category_identity")
    return reasons


def _raw_requires_product_category_identity(product_description: str) -> bool:
    text = _norm_text(product_description)
    if not text:
        return False
    if extract_product_category_identities(product_description=text):
        return True
    if _contains_marker(text, _CATEGORY_IDENTITY_MARKERS):
        return True
    return classify_essential_fact(text) == "category_identity"


def validate_negative_selection(
    brief: SelectedCreativeBrief,
    *,
    product_description: str = "",
    strict: bool = True,
) -> List[str]:
    """
    Structural negative-selection contract checks.
    Legacy plans may omit discardedFacts; strict=False skips over-selection heuristics.
    """
    reasons: List[str] = []
    reasons.extend(find_cross_bucket_duplicates(brief))
    reasons.extend(validate_under_selection(brief, product_description=product_description))

    if strict and product_description.strip():
        reasons.extend(validate_over_selection(brief, product_description=product_description))

    return list(dict.fromkeys(reasons))


def creative_prompt_contains_discarded_facts(prompt: str, brief: SelectedCreativeBrief) -> bool:
    """True when any discarded fact text appears in a creative-stage prompt block."""
    if not brief.discarded_facts:
        return False
    for fact in brief.discarded_facts:
        text = _norm_text(fact)
        if text and text in prompt:
            return True
    return False


def fusion_uses_only_essential_facts(brief: SelectedCreativeBrief) -> bool:
    """Essential Fact Fusion partitions essentialFacts only — supporting/mandatory/discarded excluded."""
    category, advantage, general = partition_essential_facts(brief.essential_facts)
    supporting_keys = {_dedupe_key(item) for item in brief.supporting_evidence}
    mandatory_keys = {_dedupe_key(item) for item in brief.mandatory_constraints}
    discarded_keys = {_dedupe_key(item) for item in brief.discarded_facts}

    for fact in category + advantage + general:
        key = _dedupe_key(fact)
        if key in supporting_keys or key in mandatory_keys or key in discarded_keys:
            return False
    return True
