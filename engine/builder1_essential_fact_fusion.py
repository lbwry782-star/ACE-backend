"""
Builder1 essential fact fusion / selection preservation — methodology + deterministic QA.

After selectedCreativeBrief commits essential facts, downstream stages must fuse
product/category identity with relative advantage in one creative mechanism.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from engine.builder1_integrity_diagnostics import record_integrity_evidence
from engine.builder1_product_identity_guard import extract_product_category_identities
from engine.builder1_selected_creative_brief import (
    SelectedCreativeBrief,
    selected_creative_brief_from_planning_internals,
)

ESSENTIAL_FACT_FUSION_REJECTION_CODES = frozenset(
    {
        "relative_advantage_without_product_application",
    }
)

BUILDER1_ESSENTIAL_FACT_FUSION = """
ESSENTIAL FACT FUSION / SELECTION PRESERVATION — mandatory after selectedCreativeBrief:
Selection is not complete merely because essential facts were chosen.
When two or more selected essential facts jointly define the advertising proposition,
Builder1 must FUSE them into one creative mechanism — not visualize only the relative
advantage while the product/category identity disappears.

Essential Fact Fusion applies ONLY to essentialFacts — not supportingEvidence,
mandatoryConstraints, or discardedFacts.

Identify:
1. Which essential fact defines WHAT THE PRODUCT/CATEGORY IS.
2. Which essential fact defines WHY THIS PRODUCT IS ADVANTAGEOUS / DIFFERENT.

Before choosing an external analogy, ask:
"Can the product/category fact and the relative-advantage fact fuse into one visual
object, action, transformation, comparison, or product-integrated analogy?"

Prefer expressing the relative advantage THROUGH the product/category rather than
separately from it.

Desired structure:
PRODUCT/CATEGORY IDENTITY + RELATIVE ADVANTAGE → ONE INTEGRATED CREATIVE MECHANISM

Not:
RELATIVE ADVANTAGE → GENERIC EXTERNAL VISUALIZATION while the product/category vanishes.

An analogy does NOT have to show the literal product packshot.
When product/category identity is essential, the mechanism must remain an analogy OF
the product/category — transformed, reinterpreted, or product-integrated — not merely
an external demonstration of the advantage alone.

This is NOT "always show the product."
Conceptual preservation is required; literal depiction is not.
""".strip()

BUILDER1_ESSENTIAL_FACT_FUSION_TEST = """
ESSENTIAL FACT FUSION TEST — answer before accepting the conceptual generator:
A. What selected fact tells us what product/category this is?
B. What selected fact creates the relative advantage?
C. Where does fact A exist in the creative mechanism (not merely in copy)?
D. Where does fact B exist in the creative mechanism?
E. Do A and B interact or fuse into one advertising idea?
F. If the product name were removed from copy, would the visual mechanism still connect
   to the advertised product/category?

If C has no answer: REJECT.
If the visual expresses B but not A: REJECT as relative_advantage_without_product_application.
If A and B are merely side-by-side without meaningful fusion: keep searching.
""".strip()

BUILDER1_ESSENTIAL_FACT_CREATIVE_SEARCH_ORDER = """
CREATIVE SEARCH ORDER when product/category identity + relative advantage are both essential:
FIRST — search for fusion inside the product/category itself (product-led or integrated).
SECOND — search for a product-integrated analogy.
THIRD — search for an external analogy that still clearly represents the product/category.
Do NOT jump directly to a generic external visualization of the advantage alone.
""".strip()

BUILDER1_ESSENTIAL_FACT_CULTURAL_CONTEXT = """
CULTURAL CONTEXT — evaluate symbolism in the intended market context.
Do not automatically import cultural interpretations from another country.

For Israeli-market advertising specifically:
Military/IDF-related objects, shapes, equipment, vocabulary, or visual references must
NOT automatically be classified as militant, aggressive, extremist, or inappropriate
merely because they are military-associated.

Evaluate actual Israeli cultural meaning, tone, execution, target market, and context.
A familiar Israeli military object may function as local identity, shared cultural memory,
everyday national reference, or Israeli visual shorthand — not advocacy of militarism.

Still reject genuinely violent, threatening, hateful, extremist, or otherwise inappropriate
executions. ASSESS CULTURAL MEANING IN CONTEXT — not MILITARY SYMBOL = AUTOMATIC REJECTION.
""".strip()

BUILDER1_ESSENTIAL_FACT_FUSION_METHODOLOGY = "\n\n".join(
    [
        BUILDER1_ESSENTIAL_FACT_FUSION,
        BUILDER1_ESSENTIAL_FACT_FUSION_TEST,
        BUILDER1_ESSENTIAL_FACT_CREATIVE_SEARCH_ORDER,
        BUILDER1_ESSENTIAL_FACT_CULTURAL_CONTEXT,
    ]
)

_CATEGORY_IDENTITY_MARKERS = frozenset(
    {
        "perfume",
        "fragrance",
        "cologne",
        "aftershave",
        "deodorant",
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "sandal",
        "sandals",
        "bag",
        "bags",
        "bottle",
        "bottles",
        "food",
        "meal",
        "coffee",
        "tea",
        "wine",
        "beer",
        "car",
        "cars",
        "phone",
        "device",
        "service",
        "software",
        "app",
        "insurance",
        "bank",
        "clinic",
        "restaurant",
        "hotel",
        "cosmetic",
        "cosmetics",
        "skincare",
        "makeup",
        "בושם",
        "ניחוח",
        "ריח",
        "לגברים",
        "לנשים",
        "גברים",
        "נשים",
        "נעל",
        "נעליים",
        "מוצר",
        "קטגוריה",
        "קטגוריית",
    }
)

_ADVANTAGE_FACT_MARKERS = frozenset(
    {
        "israel",
        "israeli",
        "local",
        "locally",
        "domestic",
        "imported",
        "foreign",
        "made",
        "origin",
        "produced",
        "manufactured",
        "alternative",
        "cheaper",
        "faster",
        "better",
        "unique",
        "only",
        "first",
        "new",
        "quality",
        "premium",
        "affordable",
        "convenient",
        "available",
        "ישראל",
        "ישראלי",
        "מקומי",
        "מיוצר",
        "תוצרת",
        "ייצור",
        "זול",
        "מהיר",
        "איכות",
        "חלופה",
        "ייחודי",
        "זמין",
    }
)

_COPY_ONLY_FIELD_KEYS = frozenset(
    {
        "brandSlogan",
        "sloganAction",
        "marketingText",
        "marketing_text",
        "headline",
        "productNameResolved",
        "productName",
    }
)

_VISUAL_MECHANISM_AD_FIELDS = (
    "physicalExecution",
    "visualExecution",
    "sceneDescription",
    "conceptualExecution",
    "executionSubject",
    "executionAction",
    "executionObjectState",
    "executionScene",
)

_PLAN_MECHANISM_FIELDS = (
    "conceptualGenerator",
    "conceptualGeneratorAction",
    "physicalGenerator",
    "transferredObject",
    "transferredObjectAction",
    "campaignRationale",
    "whyClearerThanShowingProduct",
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _tokenize(text: str) -> Set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", _norm(text))
        if token
    }


def _contains_marker(text: str, markers: Iterable[str]) -> bool:
    lowered = _norm(text).casefold()
    for marker in markers:
        token = marker.casefold()
        if len(token) <= 3:
            if re.search(rf"(?<![\w\u0590-\u05FF]){re.escape(token)}(?![\w\u0590-\u05FF])", lowered):
                return True
            continue
        if token in lowered:
            return True
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return True
    return False


def _fact_tokens(fact: str) -> Set[str]:
    return _tokenize(fact)


def classify_essential_fact(fact: str) -> str:
    """Return category_identity | advantage | general."""
    text = _norm(fact)
    if not text:
        return "general"
    category_hits = int(_contains_marker(text, _CATEGORY_IDENTITY_MARKERS))
    category_hits += len(extract_product_category_identities(product_description=text))
    advantage_hits = int(_contains_marker(text, _ADVANTAGE_FACT_MARKERS))
    if category_hits and not advantage_hits:
        return "category_identity"
    if advantage_hits and not category_hits:
        return "advantage"
    if category_hits >= advantage_hits and category_hits > 0:
        return "category_identity"
    if advantage_hits > 0:
        return "advantage"
    return "general"


def partition_essential_facts(facts: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    category: List[str] = []
    advantage: List[str] = []
    general: List[str] = []
    for fact in facts:
        text = _norm(fact)
        if not text:
            continue
        kind = classify_essential_fact(text)
        if kind == "category_identity":
            category.append(text)
        elif kind == "advantage":
            advantage.append(text)
        else:
            general.append(text)
    return category, advantage, general


def _brief_from_plan(plan_dict: Mapping[str, Any]) -> Optional[SelectedCreativeBrief]:
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    if not isinstance(internals, dict):
        return None
    return selected_creative_brief_from_planning_internals(internals)


def _product_led_route(plan_dict: Mapping[str, Any]) -> bool:
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    if not isinstance(internals, dict):
        internals = {}
    if bool(internals.get("physicalGeneratorIsProduct")):
        return True
    assessment = internals.get("directProductRouteAssessment")
    if isinstance(assessment, dict):
        route = _norm(assessment.get("recommendedRoute")).upper()
        if route in {"PRODUCT_LED", "PRODUCT_INTEGRATED_ANALOGY"}:
            return True
    return False


def _visual_mechanism_blob(plan_dict: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for field in _PLAN_MECHANISM_FIELDS:
        value = _norm(plan_dict.get(field))
        if value:
            parts.append(value)
    ads = plan_dict.get("ads")
    if isinstance(ads, list):
        internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
        ad_internals = internals.get("adInternals") if isinstance(internals, dict) else {}
        for ad in ads:
            if not isinstance(ad, dict):
                continue
            idx = ad.get("index")
            extra: Dict[str, Any] = {}
            if isinstance(ad_internals, dict) and idx is not None:
                extra = ad_internals.get(idx) or ad_internals.get(str(idx)) or {}
                if not isinstance(extra, dict):
                    extra = {}
            for field in _VISUAL_MECHANISM_AD_FIELDS:
                value = _norm(ad.get(field) or extra.get(field))
                if value:
                    parts.append(value)
            for field in (
                "categoryRelevanceReason",
                "relativeAdvantageConnection",
                "conceptualActionProof",
                "singleChangedPropertyOrAction",
                "executionPunchline",
            ):
                value = _norm(extra.get(field) or ad.get(field))
                if value:
                    parts.append(value)
    return " ".join(parts)


def _category_tokens_from_facts(category_facts: Sequence[str], product_description: str = "") -> Set[str]:
    tokens: Set[str] = set()
    for fact in category_facts:
        tokens.update(_fact_tokens(fact))
        tokens.update(extract_product_category_identities(product_description=fact))
    if product_description:
        tokens.update(extract_product_category_identities(product_description=product_description))
        tokens.update(_tokenize(product_description))
    return {token for token in tokens if len(token) >= 3}


def _advantage_tokens_from_facts(
    advantage_facts: Sequence[str],
    relative_advantage: str,
) -> Set[str]:
    tokens: Set[str] = set()
    for fact in advantage_facts:
        tokens.update(_fact_tokens(fact))
    tokens.update(_tokenize(relative_advantage))
    return {token for token in tokens if len(token) >= 3}


def _token_hits(text: str, tokens: Set[str]) -> Set[str]:
    lowered = _norm(text).casefold()
    hits: Set[str] = set()
    for token in tokens:
        if len(token) < 3:
            continue
        if token in lowered or re.search(rf"\b{re.escape(token)}\b", lowered):
            hits.add(token)
    return hits


def _category_present_in_mechanism(
    *,
    mechanism_blob: str,
    category_facts: Sequence[str],
    product_description: str,
) -> Tuple[bool, Set[str]]:
    tokens = _category_tokens_from_facts(category_facts, product_description=product_description)
    marker_present = _contains_marker(mechanism_blob, _CATEGORY_IDENTITY_MARKERS)
    hits = _token_hits(mechanism_blob, tokens)
    if hits or marker_present:
        return True, hits
    return False, hits


def _advantage_present_in_mechanism(
    *,
    mechanism_blob: str,
    advantage_facts: Sequence[str],
    relative_advantage: str,
) -> Tuple[bool, Set[str]]:
    tokens = _advantage_tokens_from_facts(advantage_facts, relative_advantage)
    marker_present = _contains_marker(mechanism_blob, _ADVANTAGE_FACT_MARKERS)
    hits = _token_hits(mechanism_blob, tokens)
    if hits or marker_present:
        return True, hits
    return False, hits


def _fusion_rationale_present(plan_dict: Mapping[str, Any], category_facts: Sequence[str]) -> bool:
    rationale_blob = " ".join(
        _norm(plan_dict.get(field))
        for field in (
            "campaignRationale",
            "whyClearerThanShowingProduct",
            "conceptualGenerator",
            "conceptualGeneratorAction",
        )
    )
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    if isinstance(internals, dict):
        for key in ("categoryRelevanceReason", "whyClearerThanShowingProduct"):
            value = internals.get(key)
            if isinstance(value, str):
                rationale_blob = f"{rationale_blob} {value}"
        ad_internals = internals.get("adInternals")
        if isinstance(ad_internals, dict):
            for entry in ad_internals.values():
                if not isinstance(entry, dict):
                    continue
                for key in ("categoryRelevanceReason", "relativeAdvantageConnection", "conceptualActionProof"):
                    value = _norm(entry.get(key))
                    if value:
                        rationale_blob = f"{rationale_blob} {value}"
    category_tokens = _category_tokens_from_facts(category_facts)
    if _token_hits(rationale_blob, category_tokens):
        return True
    fusion_markers = (
        "product-integrated",
        "product integrated",
        "category",
        "perfume",
        "fragrance",
        "בושם",
        "קטגור",
        "of the product",
        "of this product",
        "product/category",
        "same category",
        "men's",
        "לגברים",
    )
    lowered = rationale_blob.casefold()
    return any(marker in lowered for marker in fusion_markers)


def fusion_required_for_brief(brief: SelectedCreativeBrief, *, relative_advantage: str = "") -> bool:
    category, advantage, _general = partition_essential_facts(brief.essential_facts)
    return bool(category and advantage)


def format_essential_fact_fusion_prompt_block(
    brief: Optional[SelectedCreativeBrief],
    *,
    relative_advantage: str = "",
) -> str:
    if brief is None or not brief.essential_facts:
        return ""
    category, advantage, _general = partition_essential_facts(brief.essential_facts)
    lines = [BUILDER1_ESSENTIAL_FACT_FUSION_METHODOLOGY, ""]
    if fusion_required_for_brief(brief, relative_advantage=relative_advantage):
        lines.append("Selected essential facts require fusion in the creative mechanism:")
        if category:
            lines.append("Product/category identity facts:")
            for fact in category:
                lines.append(f"- {fact}")
        if advantage:
            lines.append("Relative-advantage facts:")
            for fact in advantage:
                lines.append(f"- {fact}")
        if relative_advantage:
            lines.append(f"Fixed relative advantage: {relative_advantage}")
        lines.append("Apply the Essential Fact Fusion Test before selecting conceptual or physical routes.")
    return "\n".join(lines)


def scan_essential_fact_fusion(
    plan_dict: Mapping[str, Any],
    *,
    integrity_evidence: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Deterministic QA — reject advantage-only mechanisms when category facts were essential."""
    brief = _brief_from_plan(plan_dict)
    if brief is None or not brief.essential_facts:
        return []

    relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
    if not fusion_required_for_brief(brief, relative_advantage=relative_advantage):
        return []

    category_facts, advantage_facts, _general = partition_essential_facts(brief.essential_facts)
    if not category_facts:
        return []

    product_description = _norm(plan_dict.get("productDescription"))
    mechanism_blob = _visual_mechanism_blob(plan_dict)
    advantage_present, advantage_hits = _advantage_present_in_mechanism(
        mechanism_blob=mechanism_blob,
        advantage_facts=advantage_facts,
        relative_advantage=relative_advantage,
    )
    category_present, category_hits = _category_present_in_mechanism(
        mechanism_blob=mechanism_blob,
        category_facts=category_facts,
        product_description=product_description,
    )

    if category_present:
        return []
    if _product_led_route(plan_dict) and _fusion_rationale_present(plan_dict, category_facts):
        return []
    if _fusion_rationale_present(plan_dict, category_facts) and advantage_present:
        return []

    if not advantage_present:
        return []

    record_integrity_evidence(
        integrity_evidence,
        code="relative_advantage_without_product_application",
        detector="essential_fact_fusion",
        branch="category_identity_missing_from_mechanism",
        level="plan",
        field="physicalGenerator",
        field_value_preview=_norm(plan_dict.get("physicalGenerator") or plan_dict.get("transferredObject")),
        reason=(
            "Selected product/category essential facts are not preserved in the visual mechanism; "
            "the plan expresses relative advantage without product/category application."
        ),
        extra={
            "categoryFacts": list(category_facts),
            "advantageFacts": list(advantage_facts),
            "categoryHits": sorted(category_hits),
            "advantageHits": sorted(advantage_hits),
        },
    )
    return ["relative_advantage_without_product_application"]


def essential_fact_fusion_repair_stage(codes: Sequence[str]) -> Optional[str]:
    unique = list(dict.fromkeys(codes))
    if "relative_advantage_without_product_application" in unique:
        return "conceptual_scan"
    return None
