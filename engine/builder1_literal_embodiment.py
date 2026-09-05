"""
Builder1 concept-first embodiment guard — reject literal product/slogan illustration.

Deterministic checks only; no extra model calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from engine.builder1_integrity_diagnostics import record_integrity_evidence
from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_identity_guard import (
    _exact_visual_object_is_product_name,
    extract_product_category_identities,
)

BUILDER1_CONCEPT_FIRST_RULE = """
CONCEPT FIRST — PRODUCT OPTIONAL — LITERAL OBJECT OPTIONAL:
The ad must communicate the perceptual idea clearly. The visual and wording complement each other.
The concept matters more than literal illustration.

It is NOT required to show:
- the advertised product
- the product category
- the literal object named in the slogan
- the literal environment implied by the product or slogan wording

Prefer the strongest visual embodiment of the concept — which may be the product/category itself when it carries
a genuine advertising mechanism, or an external proxy when that adds a capability the direct route cannot match.

Mandatory decision before choosing any visual object:
1. What perceptual/business idea must be understood?
2. Can the product/category express it directly with a real advertising mechanism (not merely a packshot)?
3. If not, what is the clearest external embodiment?
4. Is showing the literal slogan object necessary?
5. Would a different external object communicate the concept more clearly ONLY IF the direct product route is weaker?
If the direct product route is equally strong, prefer it over unnecessary cross-domain translation.

Bad: slogan about shortening the way → road, maze, car, route, navigation map.
Better: another long thing becoming short — short-neck giraffe, short snake, shortened ruler/rope/ladder/queue.

Bad: product promise in slogan → visual repeats the same category object with no mechanism.
Better: the product/category participating in a transformation — OR a clearer external embodiment when direct proof is weak.

Ownability comes from the recurring conceptual mechanism, visual law, transformation, and graphic language —
not from literal product imagery alone, and not from analogy merely because a structural parallel exists.
""".strip()

BUILDER1_EXPRESSIVE_OBJECT_DECISION = """
Distinguish three roles:
1. Advertised product — optional; show only when it is truly the strongest embodiment.
2. Literal slogan object — optional; do not copy slogan nouns into the main visual by default.
3. Strongest expressive object — usually an external proxy; prefer this when it communicates the idea more clearly.

When #3 differs from #1 and #2, usually choose #3.
Reject executions that merely illustrate slogan wording instead of physically proving the perception.
The replacement object must stay simple, immediately readable, visually clear, and persuasive — not obscure art-school symbolism.
""".strip()

BUILDER1_SERIES_EXTERNAL_OBJECT_RULE = """
In a series, every ad must continue the same conceptual law but search for distinct external embodiments.
Do not trap the series in one literal family (for example maze, road, car, route, product category props).
Each ad should express the same concept through a different clear external object or instance —
not through repeated literal category or slogan-noun illustration.
""".strip()

BUILDER1_IMAGE_EXPRESSIVE_OBJECT_RULE = """
The image prompt must preserve the selected external expressive object as MAIN VISUAL.
Do not collapse the chosen concept back into the advertised product, product category, slogan noun,
road/path/maze/car thinking, or other literal illustration unless that object was explicitly selected
as the strongest embodiment in planning.
Do not add the slogan's concrete nouns back into the scene unless the structured plan explicitly selected them.
""".strip()

BUILDER1_SLOGAN_COMPLEMENTARITY_RULE = """
SLOGAN / VISUAL COMPLEMENTARITY — mandatory:
The slogan communicates the verbal layer. The visual communicates the underlying concept.
These two layers must complement each other, not duplicate each other.

Preferred structure:
- slogan names or frames the promise
- visual proves, embodies, exaggerates, or reveals the promise

Avoid:
- slogan says "shorter way" and image shows a shorter road
Prefer:
- slogan says "shorter way" and image shows a surprising object whose defining length has been reduced

The viewer should receive one complete idea from the combination.
Neither element should simply caption the other.
""".strip()

BUILDER1_SLOGAN_LITERALNESS_SCAN = """
MANDATORY SLOGAN-LITERALNESS SCAN — at conceptual, physical, and series stages:
Identify important slogan content words: nouns, concrete objects, actions, places, category words, obvious visual associations.

For each proposed execution, test:
1. Was the main visual object selected mainly because its name appears in the slogan?
2. Is the image merely illustrating the sentence?
3. Would the slogan and image communicate almost the same information?
4. Does the image add a new visual proof, analogy, transformation, or punchline?
5. Is there a clearer external object that expresses the underlying perception?

Reject or replace when questions 1–3 are true and the visual adds no independent conceptual value.
Do not reject based on word overlap alone — use the structured explanation of why the object was selected.
Literal use is allowed only when it is creatively essential, not merely convenient.

Required transformation:
slogan → underlying perception → conceptual law → strongest expressive object → visual execution
""".strip()

LITERAL_EMBODIMENT_REJECTION_CODES = frozenset(
    {
        "literal_slogan_illustration",
        "literal_slogan_object_depiction",
        "literal_category_depiction",
        "literal_product_embodiment",
        "slogan_word_illustration",
        "series_literal_category_trap",
        "expressive_object_weakened",
    }
)

_SLOGAN_ILLUSTRATION_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
        "with",
        "your",
        "our",
        "we",
        "you",
        "is",
        "are",
        "be",
        "that",
        "this",
        "from",
        "into",
        "on",
        "in",
        "at",
        "of",
        "it",
        "as",
        "by",
        "not",
        "no",
        "all",
        "more",
        "most",
        "best",
        "new",
        "now",
        "just",
        "only",
        "can",
        "will",
        "has",
        "have",
        "get",
        "make",
        "made",
        "brand",
        "product",
        "service",
        "quality",
        "better",
        "always",
        "every",
        "when",
        "where",
        "what",
        "how",
        "why",
        "who",
        "של",
        "עם",
        "על",
        "את",
        "זה",
        "זו",
        "היא",
        "הוא",
        "שלך",
        "שלנו",
        "תמיד",
        "יותר",
        "הכי",
        "רק",
        "גם",
        "כל",
        "לא",
        "כן",
    }
)

_SHORTENING_CONCEPT_TERMS = frozenset(
    {
        "short",
        "shorter",
        "shorten",
        "shortened",
        "shortening",
        "shrink",
        "shrinks",
        "shrunk",
        "reduce",
        "reduced",
        "condense",
        "compact",
        "distance",
        "distances",
        "way",
        "route",
        "path",
        "journey",
        "travel",
        "trip",
        "commute",
        "time",
        "faster",
        "quick",
        "quickly",
        "shortcut",
        "shortcuts",
        "long",
        "longer",
        "length",
        "קצר",
        "קצרה",
        "קצרים",
        "קיצור",
        "דרך",
        "מרחק",
        "מהיר",
        "מהירות",
    }
)

_LITERAL_ROUTE_FAMILY = frozenset(
    {
        "road",
        "roads",
        "route",
        "routes",
        "path",
        "paths",
        "highway",
        "highways",
        "street",
        "streets",
        "maze",
        "mazes",
        "labyrinth",
        "car",
        "cars",
        "vehicle",
        "vehicles",
        "truck",
        "trucks",
        "drive",
        "driving",
        "driver",
        "journey",
        "navigation",
        "gps",
        "map",
        "maps",
        "traffic",
        "intersection",
        "bridge",
        "freeway",
        "lane",
        "lanes",
        "commute",
        "commuter",
        "transit",
        "subway",
        "train",
        "rail",
        "signpost",
        "signposts",
        "crossroad",
        "crossroads",
        "roundabout",
        "detour",
    }
)

_ABSTRACT_NAVIGATION_TOKENS = frozenset(
    {
        "way",
        "ways",
        "route",
        "routes",
        "path",
        "paths",
        "journey",
        "distance",
        "distances",
        "travel",
        "trip",
        "shortcut",
        "shortcuts",
    }
)

_AD_VISUAL_FIELDS = (
    "physicalExecution",
    "visualExecution",
    "sceneDescription",
    "conceptualExecution",
    "executionSubject",
    "executionAction",
    "executionObjectState",
    "executionScene",
    "executionPunchline",
)

_STRUCTURED_PLAN_PROOF_FIELDS = (
    "whyClearerThanShowingProduct",
    "conceptualGeneratorWhyItExpressesSlogan",
    "campaignRationale",
)

_STRUCTURED_AD_PROOF_FIELDS = (
    "singleChangedPropertyOrAction",
    "newContribution",
    "conceptualActionProof",
    "immediateClarityReason",
    "executionPunchline",
    "sloganConnection",
    "relativeAdvantageConnection",
    "categoryRelevanceReason",
    "distinctFromOtherAdsReason",
)

_STRUCTURED_INDEPENDENT_PROOF_AD_FIELDS = (
    "conceptualActionProof",
    "categoryRelevanceReason",
    "relativeAdvantageConnection",
    "immediateClarityReason",
    "singleChangedPropertyOrAction",
)

_MIN_STRUCTURED_PROOF_CHARS = 20

_CAPTION_ONLY_MARKERS = (
    "illustrates the slogan",
    "literal depiction",
    "shows the word",
    "same noun",
    "visual version of the slogan",
    "matching object",
    "direct illustration",
    "because the slogan mentions",
    "because the slogan says",
    "shows a door because",
    "shows the slogan noun",
    "merely illustrating",
    "simply caption",
)

_INDEPENDENT_PROOF_MARKERS = (
    "transform",
    "transformation",
    "analogy",
    "unexpected",
    "surprising",
    "normally",
    "proof",
    "punchline",
    "changed property",
    "made short",
    "shortened",
    "reduced length",
    "independent visual",
    "external object",
    "physically proves",
    "embodies the perception",
    "not by repeating",
    "without repeating",
    "breakthrough",
    "visibly shorter",
)

_HEBREW_INDEPENDENT_PROOF_MARKERS = (
    "מנגנון",
    "הוכחה",
    "עצמא",
    "אובייקט חיצוני",
    "בלי לחזור",
    "לא ממש",
    "הפצלה",
    "מחיצ",
    "מבטא",
    "מראה",
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _tokenize(text: str) -> Set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", _norm(text))
        if token.casefold() not in _SLOGAN_ILLUSTRATION_STOPWORDS
    }


def extract_public_slogan_content_tokens(*, slogan: str) -> Set[str]:
    """User-visible brand slogan tokens — canonical source for lexical literalism checks."""
    return _tokenize(slogan)


def extract_internal_slogan_action_tokens(*, implied_action: str) -> Set[str]:
    """Internal strategic slogan-action tokens — diagnostic only, not standalone rejection evidence."""
    return _tokenize(implied_action)


def extract_slogan_content_tokens(*, slogan: str, implied_action: str = "") -> Set[str]:
    """Combined slogan + action tokens (shortening/route scans and legacy callers)."""
    return _tokenize(f"{slogan} {implied_action}")


def implies_shortening_or_distance_concept(*texts: str) -> bool:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return False
    return any(term in combined for term in _SHORTENING_CONCEPT_TERMS)


def matched_literal_route_terms(text: str) -> List[str]:
    lowered = _norm(text).casefold()
    if not lowered:
        return []
    matched: List[str] = []
    for term in _LITERAL_ROUTE_FAMILY:
        for match in re.finditer(rf"\b{re.escape(term)}\b", lowered):
            window = lowered[max(0, match.start() - 40) : match.end()]
            if re.search(
                rf"\b(?:no|not|without|never|avoid|excluding)\b[^.]{{0,40}}\b{re.escape(term)}\b",
                window,
            ):
                continue
            matched.append(term)
    for phrase in (
        "road trip",
        "traffic jam",
        "car park",
        "parking lot",
        "city map",
        "navigation app",
        "maze runner",
        "dead end",
        "one way street",
    ):
        if phrase in lowered:
            matched.append(phrase)
    return list(dict.fromkeys(matched))


def contains_literal_route_family(text: str) -> bool:
    lowered = _norm(text).casefold()
    if not lowered:
        return False
    for term in _LITERAL_ROUTE_FAMILY:
        for match in re.finditer(rf"\b{re.escape(term)}\b", lowered):
            window = lowered[max(0, match.start() - 40) : match.end()]
            if re.search(
                rf"\b(?:no|not|without|never|avoid|excluding)\b[^.]{{0,40}}\b{re.escape(term)}\b",
                window,
            ):
                continue
            return True
    phrases = (
        "road trip",
        "traffic jam",
        "car park",
        "parking lot",
        "city map",
        "navigation app",
        "maze runner",
        "dead end",
        "one way street",
    )
    return any(phrase in lowered for phrase in phrases)


def _plan_visual_blob(plan_dict: Mapping[str, Any]) -> str:
    parts = [
        _norm(plan_dict.get("physicalGenerator")),
        _norm(plan_dict.get("transferredObject")),
        _norm(plan_dict.get("transferredObjectAction")),
        _norm(plan_dict.get("conceptualGenerator")),
        _norm(plan_dict.get("conceptualGeneratorAction")),
    ]
    for ad in plan_dict.get("ads") or []:
        if isinstance(ad, dict):
            parts.extend(_norm(ad.get(field)) for field in _AD_VISUAL_FIELDS)
    return " ".join(part for part in parts if part)


def _ad_visual_blob(ad: Mapping[str, Any]) -> str:
    return " ".join(_norm(ad.get(field)) for field in _AD_VISUAL_FIELDS if _norm(ad.get(field)))


def _slogan_object_overlap_tokens(*, slogan_tokens: Set[str], object_text: str) -> Set[str]:
    object_tokens = _tokenize(object_text)
    overlap = slogan_tokens & object_tokens
    if not overlap:
        return set()
    concrete_overlap = overlap - _ABSTRACT_NAVIGATION_TOKENS - _SHORTENING_CONCEPT_TERMS
    if concrete_overlap:
        return concrete_overlap
    abstract_overlap = overlap & _ABSTRACT_NAVIGATION_TOKENS
    if abstract_overlap and contains_literal_route_family(object_text):
        return abstract_overlap
    return set()


def _literal_slogan_noun_in_object(*, slogan_tokens: Set[str], object_text: str) -> bool:
    return bool(_slogan_object_overlap_tokens(slogan_tokens=slogan_tokens, object_text=object_text))


def _planning_internals_dict(plan_dict: Mapping[str, Any]) -> Dict[str, Any]:
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    return dict(internals) if isinstance(internals, dict) else {}


def _merged_ad_dict(plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(ad)
    ad_internals = _planning_internals_dict(plan_dict).get("adInternals")
    if not isinstance(ad_internals, dict):
        return merged
    extra = ad_internals.get(ad.get("index")) or ad_internals.get(str(ad.get("index")))
    if isinstance(extra, dict):
        merged.update(extra)
    return merged


def _structured_plan_proof_text(plan_dict: Mapping[str, Any]) -> str:
    parts: List[str] = []
    internals = _planning_internals_dict(plan_dict)
    for field in _STRUCTURED_PLAN_PROOF_FIELDS:
        parts.append(_norm(plan_dict.get(field)))
        parts.append(_norm(internals.get(field)))
    return " ".join(part for part in parts if part)


def _structured_ad_proof_text(ad: Mapping[str, Any]) -> str:
    return " ".join(_norm(ad.get(field)) for field in _STRUCTURED_AD_PROOF_FIELDS if _norm(ad.get(field)))


def _substantial_proof_segment(text: str) -> bool:
    return len(_norm(text)) >= _MIN_STRUCTURED_PROOF_CHARS


def _structured_ad_independent_proof(ad: Mapping[str, Any]) -> bool:
    filled = sum(
        1
        for field in _STRUCTURED_INDEPENDENT_PROOF_AD_FIELDS
        if _substantial_proof_segment(str(ad.get(field) or ""))
    )
    return filled >= 2


def _structured_plan_independent_proof(plan_dict: Mapping[str, Any]) -> bool:
    internals = _planning_internals_dict(plan_dict)
    rationale = _norm(plan_dict.get("campaignRationale"))
    why_clearer = _norm(internals.get("whyClearerThanShowingProduct") or plan_dict.get("whyClearerThanShowingProduct"))
    why_slogan = _norm(
        internals.get("conceptualGeneratorWhyItExpressesSlogan") or plan_dict.get("conceptualGeneratorWhyItExpressesSlogan")
    )
    if _substantial_proof_segment(rationale) and (
        _substantial_proof_segment(why_clearer) or _substantial_proof_segment(why_slogan)
    ):
        return True
    ads = [ad for ad in (plan_dict.get("ads") or []) if isinstance(ad, dict)]
    if ads and all(_structured_ad_independent_proof(_merged_ad_dict(plan_dict, ad)) for ad in ads[:2]):
        return True
    return False


def _collect_independent_proof_text(
    plan_dict: Mapping[str, Any],
    *,
    ad: Optional[Mapping[str, Any]] = None,
) -> str:
    parts = [_structured_plan_proof_text(plan_dict)]
    if ad is not None:
        parts.append(_structured_ad_proof_text(_merged_ad_dict(plan_dict, ad)))
    else:
        for raw_ad in plan_dict.get("ads") or []:
            if isinstance(raw_ad, dict):
                parts.append(_structured_ad_proof_text(_merged_ad_dict(plan_dict, raw_ad)))
    return " ".join(part for part in parts if part)


def _has_independent_visual_proof(
    plan_dict: Mapping[str, Any],
    *,
    ad: Optional[Mapping[str, Any]] = None,
    rationale_text: str = "",
) -> bool:
    if ad is not None and _structured_ad_independent_proof(_merged_ad_dict(plan_dict, ad)):
        return True
    if _structured_plan_independent_proof(plan_dict):
        return True
    combined = _collect_independent_proof_text(plan_dict, ad=ad)
    if combined and _claims_independent_visual_proof(combined):
        return True
    if rationale_text and _claims_independent_visual_proof(rationale_text):
        return True
    return False


def _claims_caption_only_illustration(text: str) -> bool:
    lowered = _norm(text).casefold()
    return any(marker in lowered for marker in _CAPTION_ONLY_MARKERS)


def _claims_independent_visual_proof(text: str) -> bool:
    lowered = _norm(text).casefold()
    if not lowered:
        return False
    if _claims_caption_only_illustration(lowered):
        return False
    if contains_literal_route_family(lowered) and not any(
        marker in lowered
        for marker in (
            "unexpected",
            "surprising",
            "external object",
            "normally",
            "analogy",
            "without repeating",
            "not by repeating",
            "visibly shorter",
            "made short",
            "shortened",
        )
    ):
        return False
    return any(marker in lowered for marker in _INDEPENDENT_PROOF_MARKERS) or any(
        marker in lowered for marker in _HEBREW_INDEPENDENT_PROOF_MARKERS
    )


def _single_public_overlap_is_literal_object_name(*, overlap: Set[str], object_text: str) -> bool:
    if len(overlap) != 1:
        return False
    token = next(iter(overlap))
    object_tokens = _tokenize(object_text)
    if object_tokens == {token}:
        return True
    return _norm(object_text).casefold() == token.casefold()


def _public_overlap_triggers_literal_rejection(
    *,
    public_overlap: Set[str],
    object_text: str,
    rationale_text: str,
) -> bool:
    if not public_overlap:
        return False
    if _claims_caption_only_illustration(rationale_text):
        return True
    if len(public_overlap) >= 2:
        return True
    return _single_public_overlap_is_literal_object_name(overlap=public_overlap, object_text=object_text)


def _plan_has_creative_literal_justification(plan_dict: Mapping[str, Any]) -> bool:
    combined = " ".join(
        [
            _collect_independent_proof_text(plan_dict),
            _norm(plan_dict.get("conceptualGenerator")),
            _norm(plan_dict.get("conceptualGeneratorAction")),
            _norm(plan_dict.get("transferredObjectAction")),
        ]
    )
    return _claims_independent_visual_proof(combined)


def _ad_has_independent_visual_proof(plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> bool:
    return _has_independent_visual_proof(plan_dict, ad=ad)


def _object_selected_from_lexical_match(
    *,
    public_slogan_tokens: Set[str],
    object_text: str,
    plan_dict: Mapping[str, Any],
    rationale_text: str,
    ad: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not object_text:
        return False
    if _has_independent_visual_proof(plan_dict, ad=ad, rationale_text=rationale_text):
        return False
    if _claims_caption_only_illustration(rationale_text):
        return True
    public_overlap = _slogan_object_overlap_tokens(
        slogan_tokens=public_slogan_tokens,
        object_text=object_text,
    )
    if not public_overlap:
        return False
    return _public_overlap_triggers_literal_rejection(
        public_overlap=public_overlap,
        object_text=object_text,
        rationale_text=rationale_text,
    )


def _detect_literal_slogan_illustration(
    plan_dict: Mapping[str, Any],
    evidence_out: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    slogan = _norm(plan_dict.get("brandSlogan"))
    slogan_action = _norm(plan_dict.get("sloganAction"))
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))
    physical = _norm(plan_dict.get("physicalGenerator"))
    conceptual = _norm(plan_dict.get("conceptualGenerator"))
    conceptual_action = _norm(plan_dict.get("conceptualGeneratorAction"))
    public_slogan_tokens = extract_public_slogan_content_tokens(slogan=slogan)
    plan_proof = _structured_plan_proof_text(plan_dict)
    full_plan_proof = _collect_independent_proof_text(plan_dict)
    shortening_concept = implies_shortening_or_distance_concept(
        slogan,
        slogan_action,
        conceptual,
        conceptual_action,
        transferred,
    )
    creative_literal_ok = _plan_has_creative_literal_justification(plan_dict)
    slogan_token_list = sorted(public_slogan_tokens)

    for field_name, field_text in (
        ("transferredObject", transferred),
        ("physicalGenerator", physical),
    ):
        if _object_selected_from_lexical_match(
            public_slogan_tokens=public_slogan_tokens,
            object_text=field_text,
            plan_dict=plan_dict,
            rationale_text=plan_proof,
        ):
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="plan_object_lexical_match",
                level="plan",
                field=field_name,
                slogan_tokens=slogan_token_list,
                matched_terms=sorted(
                    _slogan_object_overlap_tokens(
                        slogan_tokens=public_slogan_tokens,
                        object_text=field_text,
                    )
                ),
                independent_visual_proof_absent=not _has_independent_visual_proof(
                    plan_dict,
                    rationale_text=full_plan_proof,
                ),
                field_value_preview=field_text,
                reason="Physical/transferred object selected mainly from public-slogan lexical overlap without independent visual proof.",
                extra={
                    "internalActionOnlyOverlap": sorted(
                        _slogan_object_overlap_tokens(
                            slogan_tokens=extract_internal_slogan_action_tokens(implied_action=slogan_action),
                            object_text=field_text,
                        )
                        - _slogan_object_overlap_tokens(
                            slogan_tokens=public_slogan_tokens,
                            object_text=field_text,
                        )
                    ),
                },
            )
            return True

    if shortening_concept and not creative_literal_ok:
        for field_name, field_text in (
            ("transferredObject", transferred),
            ("physicalGenerator", physical),
            ("transferredObjectAction", _norm(plan_dict.get("transferredObjectAction"))),
        ):
            if contains_literal_route_family(field_text):
                record_integrity_evidence(
                    evidence_out,
                    code="literal_slogan_illustration",
                    detector="literal_embodiment",
                    branch="shortening_concept_route_family_plan",
                    level="plan",
                    field=field_name,
                    slogan_tokens=slogan_token_list,
                    matched_terms=matched_literal_route_terms(field_text),
                    independent_visual_proof_absent=True,
                    field_value_preview=field_text,
                    reason="Shortening/distance concept with literal route/path family on plan field and no creative literal justification.",
                )
                return True

    external_selected = bool(transferred) and not contains_literal_route_family(transferred)
    for ad in plan_dict.get("ads") or []:
        if not isinstance(ad, dict):
            continue
        ad_index = ad.get("index")
        ad_proof = _structured_ad_proof_text(ad)
        if _claims_caption_only_illustration(ad_proof):
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="ad_caption_only_illustration",
                level="ad",
                field="structuredAdProof",
                ad_index=int(ad_index) if ad_index is not None else None,
                slogan_tokens=slogan_token_list,
                independent_visual_proof_absent=True,
                field_value_preview=ad_proof,
                reason="Ad structured proof reads as caption-only slogan illustration.",
            )
            return True
        blob = _ad_visual_blob(ad)
        if contains_literal_route_family(blob) and not _ad_has_independent_visual_proof(plan_dict, ad):
            if external_selected or shortening_concept:
                record_integrity_evidence(
                    evidence_out,
                    code="literal_slogan_illustration",
                    detector="literal_embodiment",
                    branch="ad_route_family_without_proof",
                    level="ad",
                    field="adVisualBlob",
                    ad_index=int(ad_index) if ad_index is not None else None,
                    slogan_tokens=slogan_token_list,
                    matched_terms=matched_literal_route_terms(blob),
                    independent_visual_proof_absent=True,
                    field_value_preview=blob,
                    reason="Ad visual fields contain literal route/path family without independent visual proof.",
                )
                return True
        exec_field = "executionSubject" if _norm(ad.get("executionSubject")) else "physicalExecution"
        exec_text = _norm(ad.get("executionSubject") or ad.get("physicalExecution"))
        if _object_selected_from_lexical_match(
            public_slogan_tokens=public_slogan_tokens,
            object_text=exec_text,
            plan_dict=plan_dict,
            rationale_text=ad_proof,
            ad=ad,
        ):
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="ad_execution_lexical_match",
                level="ad",
                field=exec_field,
                ad_index=int(ad_index) if ad_index is not None else None,
                slogan_tokens=slogan_token_list,
                matched_terms=sorted(
                    _slogan_object_overlap_tokens(
                        slogan_tokens=public_slogan_tokens,
                        object_text=exec_text,
                    )
                ),
                independent_visual_proof_absent=not _has_independent_visual_proof(
                    plan_dict,
                    ad=ad,
                    rationale_text=ad_proof,
                ),
                field_value_preview=exec_text,
                reason="Ad execution subject selected mainly from public-slogan lexical overlap without independent visual proof.",
            )
            return True

    return False


def _resolve_literal_product_embodiment_context(plan_dict: Mapping[str, Any]) -> Dict[str, Any]:
    internals = plan_dict.get("planningInternals")
    if not isinstance(internals, dict):
        internals = {}

    is_product = bool(
        plan_dict.get("physicalGeneratorIsProduct") or internals.get("physicalGeneratorIsProduct")
    )
    is_packaging = bool(
        plan_dict.get("physicalGeneratorIsPackaging") or internals.get("physicalGeneratorIsPackaging")
    )

    assessment_raw = plan_dict.get("directProductRouteAssessment") or internals.get(
        "directProductRouteAssessment"
    )
    route = ""
    mechanism_available = False
    if isinstance(assessment_raw, dict):
        route = _norm(assessment_raw.get("recommendedRoute")).upper()
        mechanism_available = bool(assessment_raw.get("productLedAdvertisingMechanismAvailable"))
    elif assessment_raw is not None and hasattr(assessment_raw, "recommended_route"):
        route_value = getattr(assessment_raw.recommended_route, "value", assessment_raw.recommended_route)
        route = _norm(route_value).upper()
        mechanism_available = bool(
            getattr(assessment_raw, "product_led_advertising_mechanism_available", False)
        )

    return {
        "is_product": is_product,
        "is_packaging": is_packaging,
        "route": route,
        "mechanism_available": mechanism_available,
    }


def _product_name_mentioned_in_field(field_text: str, product_name: str) -> bool:
    text = _norm(field_text)
    name = _norm(product_name)
    if not text or not name or len(name) < 4:
        return False
    return name.casefold() in text.casefold()


def _record_literal_product_embodiment_evidence(
    evidence_out: Optional[List[Dict[str, Any]]],
    *,
    field: str,
    field_text: str,
    product_name: str,
    route: str,
    is_product: bool,
    is_packaging: bool,
    branch: str,
    embodiment_basis: str,
    reason: str,
) -> None:
    record_integrity_evidence(
        evidence_out,
        code="literal_product_embodiment",
        detector="literal_embodiment",
        branch=branch,
        level="plan",
        field=field,
        matched_terms=[product_name],
        field_value_preview=field_text,
        reason=reason,
        extra={
            "productIdentity": product_name,
            "normalizedVisualObject": _norm(field_text),
            "recommendedRoute": route or None,
            "physicalGeneratorIsProduct": is_product,
            "physicalGeneratorIsPackaging": is_packaging,
            "embodimentBasis": embodiment_basis,
        },
    )


def _detect_literal_product_embodiment(
    plan_dict: Mapping[str, Any],
    evidence_out: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Detect actual product embodiment — not mere product-name mention in copy/label prose."""
    product_name = _norm(plan_dict.get("productNameResolved") or plan_dict.get("productName"))
    if len(product_name) < 4:
        return []

    ctx = _resolve_literal_product_embodiment_context(plan_dict)
    route = str(ctx["route"])
    is_product = bool(ctx["is_product"])
    is_packaging = bool(ctx["is_packaging"])
    mechanism_available = bool(ctx["mechanism_available"])

    transferred = _norm(plan_dict.get("transferredObject"))
    physical = _norm(plan_dict.get("physicalGenerator"))

    if route == "PRODUCT_LED" and is_product and not is_packaging and mechanism_available:
        return []

    reasons: List[str] = []
    for field_name, field_text in (("transferredObject", transferred), ("physicalGenerator", physical)):
        if not field_text:
            continue

        if _exact_visual_object_is_product_name(field_value=field_text, product_name=product_name):
            _record_literal_product_embodiment_evidence(
                evidence_out,
                field=field_name,
                field_text=field_text,
                product_name=product_name,
                route=route,
                is_product=is_product,
                is_packaging=is_packaging,
                branch="exact_visual_object_is_product_name",
                embodiment_basis="exact_visual_object_match",
                reason=(
                    "Physical/transferred object field equals the product name — the advertised product "
                    "is the visual generator rather than typography or signage."
                ),
            )
            reasons.append("literal_product_embodiment")
            continue

        if not _product_name_mentioned_in_field(field_text, product_name):
            continue

        if field_name == "physicalGenerator" and route == "ANALOGY_LED" and not is_product:
            continue

        if is_product and route in {"PRODUCT_LED", "PRODUCT_INTEGRATED_ANALOGY"}:
            continue

        if not route:
            continue

        if route == "ANALOGY_LED" and not is_product:
            if field_name != "transferredObject":
                continue
            _record_literal_product_embodiment_evidence(
                evidence_out,
                field=field_name,
                field_text=field_text,
                product_name=product_name,
                route=route,
                is_product=is_product,
                is_packaging=is_packaging,
                branch="analogy_led_transferred_object_product_identity",
                embodiment_basis="product_name_in_transferred_object",
                reason=(
                    "Product identity appears in transferredObject under ANALOGY_LED while "
                    "physicalGeneratorIsProduct is false — external generator contract violated."
                ),
            )
            reasons.append("literal_product_embodiment")
            continue

        if not is_product and route != "PRODUCT_LED":
            _record_literal_product_embodiment_evidence(
                evidence_out,
                field=field_name,
                field_text=field_text,
                product_name=product_name,
                route=route,
                is_product=is_product,
                is_packaging=is_packaging,
                branch="forbidden_product_identity_in_object_field",
                embodiment_basis="product_name_in_visual_object_descriptor",
                reason=(
                    "Product identity appears in the visual object descriptor while route/methodology "
                    "does not approve direct product embodiment."
                ),
            )
            reasons.append("literal_product_embodiment")

    return list(dict.fromkeys(reasons))


def scan_brand_physical_early_literal_product_embodiment(
    *,
    product_name_resolved: str,
    brand_physical: Any,
) -> List[str]:
    """Deterministic early gate after brand_physical — skips ambiguous substring-only cases."""
    assessment = getattr(brand_physical, "direct_product_route_assessment", None)
    assessment_dict = (
        assessment.to_dict()
        if assessment is not None and hasattr(assessment, "to_dict")
        else None
    )
    plan_dict: Dict[str, Any] = {
        "productNameResolved": _norm(product_name_resolved)
        or _norm(getattr(brand_physical, "product_name_resolved", "")),
        "physicalGenerator": _norm(getattr(brand_physical, "physical_generator", "")),
        "transferredObject": _norm(getattr(brand_physical, "transferred_object", "")),
        "physicalGeneratorIsProduct": bool(getattr(brand_physical, "physical_generator_is_product", False)),
        "physicalGeneratorIsPackaging": bool(
            getattr(brand_physical, "physical_generator_is_packaging", False)
        ),
        "directProductRouteAssessment": assessment_dict,
    }
    return _detect_literal_product_embodiment(plan_dict)


def scan_literal_embodiment_bias(
    plan_dict: Mapping[str, Any],
    evidence_out: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Deterministic QA for over-literal product/slogan/category embodiment."""
    reasons: List[str] = []
    slogan = _norm(plan_dict.get("brandSlogan"))
    slogan_action = _norm(plan_dict.get("sloganAction"))
    product_name = _norm(plan_dict.get("productNameResolved") or plan_dict.get("productName"))
    product_description = _norm(plan_dict.get("productDescription"))
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))
    physical = _norm(plan_dict.get("physicalGenerator"))
    conceptual = _norm(plan_dict.get("conceptualGenerator"))
    conceptual_action = _norm(plan_dict.get("conceptualGeneratorAction"))
    slogan_tokens = extract_slogan_content_tokens(slogan=slogan, implied_action=slogan_action)
    shortening_concept = implies_shortening_or_distance_concept(
        slogan,
        slogan_action,
        conceptual,
        conceptual_action,
        transferred,
    )

    reasons.extend(_detect_literal_product_embodiment(plan_dict, evidence_out))

    for identity in extract_product_category_identities(product_description=product_description):
        for field_text in (transferred, physical):
            if field_text and re.search(rf"\b{re.escape(identity)}\b", field_text, re.I):
                reasons.append("literal_category_depiction")

    for field_text in (transferred, physical):
        if field_text and _literal_slogan_noun_in_object(slogan_tokens=slogan_tokens, object_text=field_text):
            if contains_literal_route_family(field_text) or field_text.casefold() in slogan.casefold():
                reasons.append("literal_slogan_object_depiction")

    if shortening_concept:
        slogan_token_list = sorted(slogan_tokens)
        for field_name, field_text in (
            ("transferredObject", transferred),
            ("physicalGenerator", physical),
            ("transferredObjectAction", _norm(plan_dict.get("transferredObjectAction"))),
        ):
            if contains_literal_route_family(field_text):
                record_integrity_evidence(
                    evidence_out,
                    code="literal_slogan_illustration",
                    detector="literal_embodiment",
                    branch="scan_shortening_route_family_plan",
                    level="plan",
                    field=field_name,
                    slogan_tokens=slogan_token_list,
                    matched_terms=matched_literal_route_terms(field_text),
                    independent_visual_proof_absent=True,
                    field_value_preview=field_text,
                    reason="Shortening concept with literal route/path family on plan field.",
                )
                reasons.append("slogan_word_illustration")
                reasons.append("literal_slogan_illustration")

    ads = [ad for ad in (plan_dict.get("ads") or []) if isinstance(ad, dict)]
    external_selected = bool(transferred) and not contains_literal_route_family(transferred)
    slogan_token_list = sorted(slogan_tokens)
    if external_selected:
        for ad in ads:
            ad_index = ad.get("index")
            blob = _ad_visual_blob(ad)
            if contains_literal_route_family(blob) and not _ad_has_independent_visual_proof(plan_dict, ad):
                record_integrity_evidence(
                    evidence_out,
                    code="literal_slogan_illustration",
                    detector="literal_embodiment",
                    branch="scan_external_selected_ad_route_family",
                    level="ad",
                    field="adVisualBlob",
                    ad_index=int(ad_index) if ad_index is not None else None,
                    slogan_tokens=slogan_token_list,
                    matched_terms=matched_literal_route_terms(blob),
                    independent_visual_proof_absent=True,
                    field_value_preview=blob,
                    reason="External object selected but ad visual fields contain literal route/path family without proof.",
                )
                reasons.append("literal_slogan_object_depiction")
                reasons.append("literal_slogan_illustration")
                break

    if len(ads) >= 2 and shortening_concept:
        literal_family_ads = sum(
            1
            for ad in ads
            if contains_literal_route_family(_ad_visual_blob(ad)) and not _ad_has_independent_visual_proof(plan_dict, ad)
        )
        if literal_family_ads >= 2:
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="scan_series_literal_category_trap_multi_ad",
                level="plan",
                field="ads",
                slogan_tokens=slogan_token_list,
                independent_visual_proof_absent=True,
                reason="Multiple ads use literal route/path family without independent visual proof.",
            )
            reasons.append("series_literal_category_trap")
            reasons.append("literal_slogan_illustration")
        elif literal_family_ads >= 1 and not external_selected:
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="scan_series_literal_category_trap_single_ad",
                level="plan",
                field="ads",
                slogan_tokens=slogan_token_list,
                independent_visual_proof_absent=True,
                reason="Series uses literal route/path family without external object selection.",
            )
            reasons.append("series_literal_category_trap")
            reasons.append("literal_slogan_illustration")

    if not external_selected and shortening_concept and contains_literal_route_family(_plan_visual_blob(plan_dict)):
        if "slogan_word_illustration" not in reasons:
            record_integrity_evidence(
                evidence_out,
                code="literal_slogan_illustration",
                detector="literal_embodiment",
                branch="scan_plan_visual_blob_route_family",
                level="plan",
                field="planVisualBlob",
                slogan_tokens=slogan_token_list,
                matched_terms=matched_literal_route_terms(_plan_visual_blob(plan_dict)),
                independent_visual_proof_absent=True,
                reason="Plan-wide visual blob contains literal route/path family under shortening concept.",
            )
            reasons.append("slogan_word_illustration")
            reasons.append("literal_slogan_illustration")

    if _detect_literal_slogan_illustration(plan_dict, evidence_out):
        reasons.append("literal_slogan_illustration")

    return list(dict.fromkeys(reasons))


def scan_series_plan_literal_embodiment(series_plan: Builder1SeriesPlan) -> List[str]:
    internals = series_plan.planning_internals or {}
    ad_internals_map = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    ads_payload: List[Dict[str, Any]] = []
    for ad in series_plan.ads:
        extra = {}
        if isinstance(ad_internals_map, dict):
            extra = ad_internals_map.get(ad.index) or ad_internals_map.get(str(ad.index)) or {}
        payload = {
            "physicalExecution": ad.physical_execution,
            "visualExecution": ad.visual_execution,
            "sceneDescription": ad.scene_description,
            "conceptualExecution": ad.conceptual_execution,
        }
        if isinstance(extra, dict):
            for key in (
                "executionSubject",
                "executionAction",
                "executionObjectState",
                "executionScene",
                "executionPunchline",
                "conceptualActionProof",
                "categoryRelevanceReason",
                "relativeAdvantageConnection",
                "immediateClarityReason",
                "singleChangedPropertyOrAction",
                "sloganConnection",
            ):
                if extra.get(key):
                    payload[key] = extra.get(key)
        ads_payload.append(payload)
    plan_dict: Dict[str, Any] = {
        "brandSlogan": series_plan.brand_slogan,
        "sloganAction": series_plan.slogan_action,
        "productNameResolved": series_plan.product_name_resolved,
        "productDescription": series_plan.product_description,
        "physicalGenerator": series_plan.physical_generator,
        "transferredObject": series_plan.transferred_object,
        "transferredObjectAction": series_plan.transferred_object_action,
        "conceptualGenerator": series_plan.conceptual_generator,
        "conceptualGeneratorAction": series_plan.conceptual_generator_action,
        "campaignRationale": series_plan.campaign_rationale,
        "ads": ads_payload,
    }
    if internals:
        plan_dict["planningInternals"] = dict(internals)
    return scan_literal_embodiment_bias(plan_dict)


def validate_visual_prompt_expressive_object(
    prompt: str,
    *,
    series_plan: Builder1SeriesPlan,
    ad_index: int = 1,
) -> List[str]:
    reasons: List[str] = []
    transferred = _norm(series_plan.transferred_object or series_plan.physical_generator)
    if not transferred or not prompt:
        return reasons

    start = prompt.find("=== MAIN VISUAL")
    end = prompt.find("=== END MAIN VISUAL", start + 1) if start >= 0 else -1
    main_visual = prompt[start:end] if start >= 0 and end > start else prompt
    focus_lines = [
        line.strip().casefold()
        for line in main_visual.splitlines()
        if line.strip().startswith(("MAIN VISUAL:", "ACTION:", "Composition execution:"))
    ]
    lowered_main = " ".join(focus_lines) if focus_lines else main_visual.casefold()

    if contains_literal_route_family(lowered_main):
        reasons.append("expressive_object_weakened")

    transferred_tokens = _tokenize(transferred)
    significant = {token for token in transferred_tokens if len(token) >= 4}
    if significant and not any(token in lowered_main for token in significant):
        if "MAIN VISUAL:" in main_visual:
            reasons.append("expressive_object_weakened")

    for identity in extract_product_category_identities(product_description=series_plan.product_description):
        if identity in lowered_main and identity not in transferred.casefold():
            reasons.append("expressive_object_weakened")

    reasons.extend(
        validate_visual_prompt_slogan_noun_reintroduction(prompt, series_plan=series_plan)
    )

    return list(dict.fromkeys(reasons))


def validate_visual_prompt_slogan_noun_reintroduction(
    prompt: str,
    *,
    series_plan: Builder1SeriesPlan,
) -> List[str]:
    transferred = _norm(series_plan.transferred_object or series_plan.physical_generator)
    if not transferred or not prompt:
        return []

    start = prompt.find("=== MAIN VISUAL")
    end = prompt.find("=== END MAIN VISUAL", start + 1) if start >= 0 else -1
    main_visual = prompt[start:end] if start >= 0 and end > start else ""
    if not main_visual:
        return []

    focus_lines = [
        line.strip().casefold()
        for line in main_visual.splitlines()
        if line.strip().startswith(("MAIN VISUAL:", "ACTION:", "Composition execution:"))
    ]
    lowered_main = " ".join(focus_lines)
    if not lowered_main:
        return []

    slogan_tokens = extract_slogan_content_tokens(
        slogan=series_plan.brand_slogan,
        implied_action=series_plan.slogan_action,
    )
    transferred_tokens = _tokenize(transferred)
    discarded_tokens = {
        token
        for token in slogan_tokens
        if token not in transferred_tokens
        and token not in _SHORTENING_CONCEPT_TERMS
        and token not in _ABSTRACT_NAVIGATION_TOKENS
        and len(token) >= 4
    }
    reasons: List[str] = []
    for token in sorted(discarded_tokens):
        if re.search(rf"\b{re.escape(token)}\b", lowered_main):
            reasons.append("literal_slogan_illustration")
            break

    if contains_literal_route_family(lowered_main):
        reasons.append("literal_slogan_illustration")

    return list(dict.fromkeys(reasons))


def literal_embodiment_repair_stage(codes: Sequence[str]) -> str | None:
    unique = list(dict.fromkeys(codes))
    if any(
        code in unique
        for code in (
            "literal_slogan_illustration",
            "literal_slogan_object_depiction",
            "slogan_word_illustration",
        )
    ):
        return "brand_physical"
    if any(code in unique for code in ("literal_category_depiction", "literal_product_embodiment")):
        return "brand_physical"
    if "series_literal_category_trap" in unique:
        return "series_ads"
    if "expressive_object_weakened" in unique:
        return "series_ads"
    return None
