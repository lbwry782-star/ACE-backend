"""
Builder1 concept-first embodiment guard — reject literal product/slogan illustration.

Deterministic checks only; no extra model calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_identity_guard import extract_product_category_identities

BUILDER1_CONCEPT_FIRST_RULE = """
CONCEPT FIRST — PRODUCT OPTIONAL — LITERAL OBJECT OPTIONAL:
The ad must communicate the perceptual idea clearly. The visual and wording complement each other.
The concept matters more than literal illustration.

It is NOT required to show:
- the advertised product
- the product category
- the literal object named in the slogan
- the literal environment implied by the product or slogan wording

Prefer the strongest visual embodiment of the concept — often a different external object,
proxy, metaphor, or surprising but simple embodiment.

Mandatory decision before choosing any visual object:
1. What perceptual/business idea must be understood?
2. What is the clearest, simplest way to embody that idea visually?
3. Is showing the product necessary?
4. Is showing the literal slogan object necessary?
5. Would a different external object communicate the concept more clearly, simply, or memorably?
If #5 is yes, prefer the external object.

Bad: slogan about shortening the way → road, maze, car, route, navigation map.
Better: another long thing becoming short — short-neck giraffe, short snake, shortened ruler/rope/ladder/queue.

Bad: product promise in slogan → visual repeats the same category object.
Better: a clearer, more surprising embodiment of the same underlying idea.

Ownability comes from the recurring conceptual mechanism, visual law, transformation, and graphic language —
not from literal product imagery.
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
    "distinctFromOtherAdsReason",
)

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


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _tokenize(text: str) -> Set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", _norm(text))
        if token.casefold() not in _SLOGAN_ILLUSTRATION_STOPWORDS
    }


def extract_slogan_content_tokens(*, slogan: str, implied_action: str = "") -> Set[str]:
    return _tokenize(f"{slogan} {implied_action}")


def implies_shortening_or_distance_concept(*texts: str) -> bool:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return False
    return any(term in combined for term in _SHORTENING_CONCEPT_TERMS)


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


def _literal_slogan_noun_in_object(*, slogan_tokens: Set[str], object_text: str) -> bool:
    object_tokens = _tokenize(object_text)
    overlap = slogan_tokens & object_tokens
    if not overlap:
        return False
    concrete_overlap = overlap - _ABSTRACT_NAVIGATION_TOKENS - _SHORTENING_CONCEPT_TERMS
    if concrete_overlap:
        return True
    abstract_overlap = overlap & _ABSTRACT_NAVIGATION_TOKENS
    return bool(abstract_overlap and contains_literal_route_family(object_text))


def _structured_plan_proof_text(plan_dict: Mapping[str, Any]) -> str:
    return " ".join(_norm(plan_dict.get(field)) for field in _STRUCTURED_PLAN_PROOF_FIELDS)


def _structured_ad_proof_text(ad: Mapping[str, Any]) -> str:
    return " ".join(_norm(ad.get(field)) for field in _STRUCTURED_AD_PROOF_FIELDS)


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
    return any(marker in lowered for marker in _INDEPENDENT_PROOF_MARKERS)


def _plan_has_creative_literal_justification(plan_dict: Mapping[str, Any]) -> bool:
    combined = " ".join(
        [
            _structured_plan_proof_text(plan_dict),
            _norm(plan_dict.get("conceptualGenerator")),
            _norm(plan_dict.get("conceptualGeneratorAction")),
            _norm(plan_dict.get("transferredObjectAction")),
        ]
    )
    return _claims_independent_visual_proof(combined)


def _ad_has_independent_visual_proof(ad: Mapping[str, Any]) -> bool:
    return _claims_independent_visual_proof(_structured_ad_proof_text(ad))


def _object_selected_from_lexical_match(
    *,
    slogan_tokens: Set[str],
    object_text: str,
    rationale_text: str,
) -> bool:
    if not object_text:
        return False
    if _claims_independent_visual_proof(rationale_text):
        return False
    if _claims_caption_only_illustration(rationale_text):
        return True
    if not _literal_slogan_noun_in_object(slogan_tokens=slogan_tokens, object_text=object_text):
        return False
    return not _claims_independent_visual_proof(rationale_text)


def _detect_literal_slogan_illustration(plan_dict: Mapping[str, Any]) -> bool:
    slogan = _norm(plan_dict.get("brandSlogan"))
    slogan_action = _norm(plan_dict.get("sloganAction"))
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))
    physical = _norm(plan_dict.get("physicalGenerator"))
    conceptual = _norm(plan_dict.get("conceptualGenerator"))
    conceptual_action = _norm(plan_dict.get("conceptualGeneratorAction"))
    slogan_tokens = extract_slogan_content_tokens(slogan=slogan, implied_action=slogan_action)
    plan_proof = _structured_plan_proof_text(plan_dict)
    shortening_concept = implies_shortening_or_distance_concept(
        slogan,
        slogan_action,
        conceptual,
        conceptual_action,
        transferred,
    )
    creative_literal_ok = _plan_has_creative_literal_justification(plan_dict)

    for field_text in (transferred, physical):
        if _object_selected_from_lexical_match(
            slogan_tokens=slogan_tokens,
            object_text=field_text,
            rationale_text=plan_proof,
        ):
            return True

    if shortening_concept and not creative_literal_ok:
        for field_text in (transferred, physical, _norm(plan_dict.get("transferredObjectAction"))):
            if contains_literal_route_family(field_text):
                return True

    external_selected = bool(transferred) and not contains_literal_route_family(transferred)
    for ad in plan_dict.get("ads") or []:
        if not isinstance(ad, dict):
            continue
        ad_proof = _structured_ad_proof_text(ad)
        if _claims_caption_only_illustration(ad_proof):
            return True
        blob = _ad_visual_blob(ad)
        if contains_literal_route_family(blob) and not _ad_has_independent_visual_proof(ad):
            if external_selected or shortening_concept:
                return True
        if _object_selected_from_lexical_match(
            slogan_tokens=slogan_tokens,
            object_text=_norm(ad.get("executionSubject") or ad.get("physicalExecution")),
            rationale_text=ad_proof,
        ):
            return True

    return False


def scan_literal_embodiment_bias(plan_dict: Mapping[str, Any]) -> List[str]:
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

    if product_name and len(product_name) >= 4:
        for field_text in (transferred, physical):
            if field_text and product_name.casefold() in field_text.casefold():
                reasons.append("literal_product_embodiment")

    for identity in extract_product_category_identities(product_description=product_description):
        for field_text in (transferred, physical):
            if field_text and re.search(rf"\b{re.escape(identity)}\b", field_text, re.I):
                reasons.append("literal_category_depiction")

    for field_text in (transferred, physical):
        if field_text and _literal_slogan_noun_in_object(slogan_tokens=slogan_tokens, object_text=field_text):
            if contains_literal_route_family(field_text) or field_text.casefold() in slogan.casefold():
                reasons.append("literal_slogan_object_depiction")

    if shortening_concept:
        for field_text in (transferred, physical, _norm(plan_dict.get("transferredObjectAction"))):
            if contains_literal_route_family(field_text):
                reasons.append("slogan_word_illustration")
                reasons.append("literal_slogan_illustration")

    ads = [ad for ad in (plan_dict.get("ads") or []) if isinstance(ad, dict)]
    external_selected = bool(transferred) and not contains_literal_route_family(transferred)
    if external_selected:
        for ad in ads:
            if contains_literal_route_family(_ad_visual_blob(ad)) and not _ad_has_independent_visual_proof(ad):
                reasons.append("literal_slogan_object_depiction")
                reasons.append("literal_slogan_illustration")
                break

    if len(ads) >= 2 and shortening_concept:
        literal_family_ads = sum(
            1
            for ad in ads
            if contains_literal_route_family(_ad_visual_blob(ad)) and not _ad_has_independent_visual_proof(ad)
        )
        if literal_family_ads >= 2:
            reasons.append("series_literal_category_trap")
            reasons.append("literal_slogan_illustration")
        elif literal_family_ads >= 1 and not external_selected:
            reasons.append("series_literal_category_trap")
            reasons.append("literal_slogan_illustration")

    if not external_selected and shortening_concept and contains_literal_route_family(_plan_visual_blob(plan_dict)):
        if "slogan_word_illustration" not in reasons:
            reasons.append("slogan_word_illustration")
            reasons.append("literal_slogan_illustration")

    if _detect_literal_slogan_illustration(plan_dict):
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
        "ads": ads_payload,
    }
    return scan_literal_embodiment_bias(plan_dict)


def validate_visual_prompt_expressive_object(
    prompt: str,
    *,
    series_plan: Builder1SeriesPlan,
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

    if contains_literal_route_family(transferred):
        return reasons

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

    if contains_literal_route_family(transferred):
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
