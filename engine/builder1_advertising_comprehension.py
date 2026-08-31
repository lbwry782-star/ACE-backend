"""
Builder1 advertising comprehension — deterministic planning and compliance context.

Distinguishes physical mechanism clarity from advertising bridge clarity, rejects
multi-hop proxy chains, and validates dominant-object strategic roles without
adding paid model calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from engine.builder1_integrity_diagnostics import record_integrity_evidence
from engine.builder1_plan_spec import Builder1SeriesPlan

ADVERTISING_COMPREHENSION_REJECTION_CODES = frozenset(
    {
        "advertising_bridge_unclear",
        "multi_hop_symbolic_chain",
        "dominant_object_strategic_role_missing",
        "competing_category_visual",
        "advertising_mechanism_not_observable",
        "public_analogy_too_complex",
    }
)

PLAN_PHYSICAL_REPAIR_CODES = frozenset(
    {
        "competing_category_visual",
        "advertising_mechanism_not_observable",
        "public_analogy_too_complex",
    }
)

CATEGORY_INTEGRITY_VIOLATION_CODES = frozenset(
    {
        "competing_category_visual",
        "advertising_mechanism_not_observable",
    }
)

EXECUTION_FIDELITY_VIOLATION_CODES = frozenset(
    {
        "planned_scene_diverged",
        "planned_mechanism_diverged",
        "central_proof_not_visible",
        "unintended_dominant_interpretation",
        "advertising_bridge_not_recoverable",
        "relative_advantage_not_expressed",
        "visual_slogan_mechanism_mismatch",
        "dominant_subject_diverged",
        "public_analogy_not_recoverable",
    }
    | CATEGORY_INTEGRITY_VIOLATION_CODES
)

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "they",
        "their",
        "there",
        "where",
        "when",
        "what",
        "which",
        "while",
        "about",
        "through",
        "because",
        "viewer",
        "visual",
        "scene",
        "shows",
        "show",
        "visible",
        "clearly",
        "instant",
        "instantly",
        "immediately",
        "understand",
        "understandable",
        "every",
        "other",
        "variant",
        "subject",
        "action",
        "state",
        "punchline",
        "proof",
        "object",
        "physical",
        "advertising",
        "product",
        "service",
        "brand",
        "campaign",
        "lesson",
        "lessons",
    }
)

_PHYSICAL_ONLY_CLARITY_MARKERS = (
    "familiar",
    "understandable",
    "everyone knows",
    "object mechanics",
    "literal scene",
    "merely familiar",
    "only familiar",
    "are familiar",
    "physically familiar",
)

_ADVANTAGE_BRIDGE_MARKERS = (
    "viewer infers",
    "viewer understands",
    "advertising meaning",
    "relative advantage",
    "therefore the",
    "so the viewer",
    "means for",
    "shows why",
    "proves the advantage",
    "connects the visual",
    "without the slogan",
    "about the product",
    "about the service",
    "infers about",
    "understands that",
    "advertised offer",
    "specifically expresses",
    "expresses the advantage",
    "not merely",
    "not just an interesting",
)

_SYMBOLIC_MAPPING_MARKERS = (
    "represents",
    "symbolizes",
    "symbolises",
    "stands for",
    "maps to",
    "equivalent to",
    "metaphor for",
    "analogy to",
    "like a",
    "as if",
    "translates to",
    "→",
    "->",
)

_COMPETING_INTERPRETATION_TERMS = frozenset(
    {
        "railway",
        "rail",
        "rails",
        "track",
        "tracks",
        "train",
        "road",
        "route",
        "routes",
        "path",
        "paths",
        "maze",
        "map",
        "navigation",
        "highway",
        "station",
    }
)

_STRATEGIC_ROLE_MARKERS = (
    "proves",
    "shows",
    "expresses",
    "role",
    "because",
    "tests",
    "test",
    "survives",
    "demonstrates",
    "means",
    "readiness",
    "advantage",
    "focus",
    "focused",
    "preparation",
    "mechanism",
    "punchline",
)

_UNEXPLAINED_DOMINANT_NOUNS = frozenset(
    {
        "clock",
        "calendar",
        "tower",
        "train",
        "railway",
        "tracks",
        "road",
        "map",
        "maze",
    }
)

_GENERIC_EXECUTION_RE = re.compile(
    r"^(subject|object|scene|action|state|punchline)\s+variant\s+\d+$",
    re.IGNORECASE,
)

# Competing commercial service categories — generalized clusters, not object blacklists.
_SERVICE_CATEGORY_SIGNALS: Dict[str, frozenset[str]] = {
    "academic_tutoring": frozenset(
        {
            "tutor",
            "tutoring",
            "teacher",
            "teaching",
            "lesson",
            "lessons",
            "student",
            "students",
            "classroom",
            "exam",
            "bagrut",
            "history",
            "math",
            "mathematics",
            "physics",
            "chemistry",
            "academic",
            "schoolwork",
            "homework",
            "preparation",
            "curriculum",
            "היסטוריה",
            "בגרות",
            "שיעור",
            "שיעורים",
            "מורה",
            "תלמיד",
            "תלמידים",
            "לימוד",
            "חינוך",
            "הכנה",
            "פרטי",
        }
    ),
    "sports_coaching": frozenset(
        {
            "gymnast",
            "gymnastics",
            "gymnasium",
            "athlete",
            "athletic",
            "sports",
            "sport",
            "coach",
            "coaching",
            "training",
            "arena",
            "vault",
            "beam",
            "floor",
            "exercise",
            "apparatus",
            "parallel",
            "bars",
            "mat",
            "התעמלות",
            "התעמל",
            "התעמלן",
            "ספורט",
            "אימון",
            "מאמן",
            "מסלול",
            "קורה",
        }
    ),
    "music_instruction": frozenset(
        {
            "piano",
            "violin",
            "guitar",
            "music",
            "orchestra",
            "musician",
            "instrument",
            "symphony",
            "conductor",
            "מוזיקה",
            "פסנתר",
            "כינור",
            "גיטרה",
            "נגינה",
        }
    ),
    "driving_instruction": frozenset(
        {
            "driving lesson",
            "driving instructor",
            "driving school",
            "driver",
            "license",
            "steering",
            "wheel",
            "נהיגה",
            "רישיון",
            "נהג",
            "מורה נהיגה",
        }
    ),
    "art_instruction": frozenset(
        {
            "art",
            "artist",
            "painting",
            "drawing",
            "canvas",
            "easel",
            "sculpture",
            "studio",
            "אמנות",
            "ציור",
            "מכחול",
        }
    ),
    "language_tutoring": frozenset(
        {
            "language",
            "english",
            "french",
            "spanish",
            "hebrew",
            "vocabulary",
            "grammar",
            "conversation",
            "fluent",
            "שפה",
            "אנגלית",
            "עברית",
            "דיבור",
        }
    ),
    "fitness_coaching": frozenset(
        {
            "fitness",
            "gym",
            "workout",
            "trainer",
            "personal",
            "weights",
            "yoga",
            "pilates",
            "crossfit",
            "כושר",
            "מאמן",
            "אימון",
        }
    ),
    "culinary_instruction": frozenset(
        {
            "cooking",
            "culinary",
            "chef",
            "kitchen",
            "recipe",
            "baking",
            "restaurant",
            "cuisine",
            "בישול",
            "מטבח",
            "שף",
        }
    ),
    "legal_services": frozenset(
        {
            "lawyer",
            "attorney",
            "legal",
            "court",
            "litigation",
            "law",
            "counsel",
            "עורך",
            "דין",
            "משפט",
        }
    ),
    "financial_accounting": frozenset(
        {
            "accountant",
            "accounting",
            "bookkeeping",
            "tax",
            "finance",
            "audit",
            "cpa",
            "חשבונאות",
            "רואה",
            "חשבון",
        }
    ),
    "dental_care": frozenset(
        {
            "dentist",
            "dental",
            "teeth",
            "orthodont",
            "clinic",
            "שיניים",
            "רופא",
            "שיניים",
        }
    ),
    "automotive_repair": frozenset(
        {
            "mechanic",
            "garage",
            "automotive",
            "repair",
            "engine",
            "workshop",
            "מוסך",
            "מכונאי",
            "תיקון",
        }
    ),
}

_INSTRUCTIONAL_SERVICE_CLUSTERS = frozenset(
    {
        "academic_tutoring",
        "sports_coaching",
        "music_instruction",
        "driving_instruction",
        "art_instruction",
        "language_tutoring",
        "fitness_coaching",
        "culinary_instruction",
    }
)

_PROFESSIONAL_SERVICE_CLUSTERS = frozenset(
    {
        "legal_services",
        "financial_accounting",
        "dental_care",
        "automotive_repair",
    }
    | _INSTRUCTIONAL_SERVICE_CLUSTERS
)

_VISUAL_EFFECT_ONLY_MARKERS = (
    "photography effect",
    "photo effect",
    "familiar photo",
    "familiar photograph",
    "sports photography",
    "action photography",
    "action photo",
    "sharp foreground",
    "blurred background",
    "background blur",
    "depth of field",
    "bokeh",
    "out of focus",
    "motion blur",
    "soft background",
    "sharp subject",
    "תופעת צילום",
    "צילום ספורט",
    "צילום פעולה",
    "מטושטש",
    "רקע מטושטש",
    "חד בחזית",
    "נחיתה חדה",
    "אפקט צילום",
)

_MECHANISM_DEVICE_MARKERS = (
    "autofocus",
    "auto-focus",
    "auto focus",
    "focus system",
    "tracking system",
    "follows one",
    "continuously follow",
    "continuously track",
    "camera lens",
    "camera screen",
    "viewfinder",
    "focus lock",
    "focus indicator",
    "focus box",
    "focus reticle",
    "lens",
    "camera",
    "camcorder",
    "view screen",
    "display shows focus",
    "מצלמה",
    "מיקוד",
    "מיקוד אוטומטי",
    "מערכת מיקוד",
    "עוקב",
    "עוקבת",
    "מסך מיקוד",
)

_RESULT_ONLY_MARKERS = _VISUAL_EFFECT_ONLY_MARKERS + (
    "sharp while others blur",
    "one sharp",
    "others blurred",
    "only one in focus",
    "אחד חד",
    "אחרים מטושטשים",
)

_FAMILIARITY_ONLY_CLARITY_MARKERS = (
    "familiar photographic effect",
    "familiar photograph",
    "familiar photo effect",
    "familiar sports photography",
    "known industrial process",
    "viewer recognizes the object",
    "viewer recognizes",
    "recognizable object",
    "familiar effect",
    "familiar process",
    "תופעת צילום מוכרת",
    "תהליך תעשייתי מוכר",
)

_TECHNICAL_VOCABULARY_MARKERS = (
    "autofocus plane",
    "sensor feedback",
    "calibration",
    "optical tracking",
    "throughput",
    "industrial inspection",
    "dynamic correction loop",
    "feedback loop",
    "quality-control conveyor",
    "sensor calibration",
    "optical tracking system",
    "mechanical transmission",
    "transmission system",
    "מיקוד אוטומטי",
    "לולאת משוב",
    "כיול",
    "מעקב אופטי",
)

_UNIVERSAL_EVERYDAY_MARKERS = (
    "magnet",
    "umbrella",
    "domino",
    "balloon",
    "key opens",
    "key unlock",
    "lock and key",
    "parachute",
    "door opens",
    "door closes",
    "train on track",
    "fits into",
    "one object receives",
    "attracts",
    "attract",
    "pulls",
    "pull",
    "blocks rain",
    "keeps dry",
    "מגנט",
    "מטריה",
    "דומינו",
    "בלון",
    "מפתח",
    "מנעול",
    "צנח",
)

_COMMON_EVERYDAY_MARKERS = (
    "conveyor",
    "scale",
    "filter",
    "sort",
    "separate",
    "weigh",
    "balance",
    "mirror",
    "shadow",
    "light",
    "weight",
    "מסוע",
    "מאזניים",
    "מסנן",
)

_OBSERVABLE_CAUSAL_ACTION_MARKERS = (
    "attract",
    "attracts",
    "pull",
    "pulls",
    "pulling",
    "push",
    "pushes",
    "block",
    "blocks",
    "blocking",
    "open",
    "opens",
    "opening",
    "close",
    "closes",
    "fall",
    "falls",
    "falling",
    "inflate",
    "inflates",
    "deflate",
    "fit",
    "fits",
    "catch",
    "catches",
    "drop",
    "drops",
    "stick",
    "sticks",
    "separate",
    "separates",
    "keep dry",
    "stays dry",
    "one receives",
    "only one",
    "מושך",
    "נמשך",
    "דוחף",
    "חוסם",
    "נפתח",
    "נסגר",
    "נופל",
    "מתנפח",
)

_PHYSICAL_ACTION_MARKERS = _OBSERVABLE_CAUSAL_ACTION_MARKERS + (
    "visible",
    "shows",
    "see",
    "viewer sees",
    "under rain",
    "under water",
    "on track",
    "רואים",
    "רואה",
    "נראה",
)

# Hebrew verb inflections for ordinary visible physical actions (word-boundary safe).
_HEBREW_OBSERVABLE_CAUSAL_ACTION_RE = re.compile(
    r"(?<![\u0590-\u05FF])("
    r"עול(?:ה|ים|ות)|"
    r"יורד(?:|ה|ת|ים|ות)|"
    r"מעלה|מרים|מוריד|"
    r"מניח(?:|ים|ות)?|מונח|"
    r"נופל(?:|ה|ת|ים|ות)?|"
    r"דוחף|דוחפת|"
    r"מושך|מושכת|"
    r"נ(?:ע|וע)(?:|ה|ת|ים|ות)?|זז(?:|ה|ת|ים|ות)?|"
    r"נפתח(?:|ה|ת|ים|ות)?|"
    r"נסגר(?:|ה|ת|ים|ות)?|"
    r"חוצ(?:ה|ים|ות)|"
    r"נוט(?:ה|ים|ות)|"
    r"מתאז(?:ן|נ(?:|ה|ת|ים|ות)?)|איזון"
    r")(?![\u0590-\u05FF])",
    re.UNICODE,
)

_HEBREW_VISIBLE_ACTION_RE = re.compile(
    r"(?<![\u0590-\u05FF])(רוא(?:ה|ים|ות)|נרא(?:ה|ים|ות))(?![\u0590-\u05FF])",
    re.UNICODE,
)

_ALL_TECHNICAL_FAMILIARITY_MARKERS: tuple[str, ...] = (
    _TECHNICAL_VOCABULARY_MARKERS + _MECHANISM_DEVICE_MARKERS
)

_INSTRUCTIONAL_SCENE_MARKERS = (
    "gymnast",
    "gymnastics",
    "gymnasium",
    "training floor",
    "apparatus",
    "vault",
    "beam",
    "parallel bars",
    "piano",
    "keyboard",
    "violin",
    "orchestra",
    "easel",
    "canvas",
    "art studio",
    "driving lesson",
    "steering wheel",
    "dental chair",
    "courtroom",
    "התעמלות",
    "רצפת אימון",
    "פסנתר",
    "כינור",
    "סטודיו",
    "מורה נהיגה",
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _significant_tokens(text: str) -> Set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", _norm(text))
        if token.casefold() not in _STOPWORDS
    }


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = _significant_tokens(left)
    right_tokens = _significant_tokens(right)
    return bool(left_tokens & right_tokens)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = _norm(text).casefold()
    return any(marker in lowered for marker in markers)


def _non_overlapping_marker_matches(
    text: str,
    markers: Sequence[str],
) -> List[Dict[str, Any]]:
    """Find marker spans; overlapping hits on the same text count once (longest marker wins)."""
    lowered = _norm(text).casefold()
    if not lowered:
        return []
    raw: List[tuple[int, int, str]] = []
    for marker in markers:
        needle = marker.casefold()
        if not needle:
            continue
        start = 0
        while True:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            raw.append((idx, idx + len(needle), marker))
            start = idx + len(needle)
    if not raw:
        return []
    raw.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: List[tuple[int, int, str]] = []
    for start, end, marker in raw:
        if not merged:
            merged.append((start, end, marker))
            continue
        prev_start, prev_end, prev_marker = merged[-1]
        if start >= prev_end:
            merged.append((start, end, marker))
            continue
        new_end = max(prev_end, end)
        best = marker if len(marker) >= len(prev_marker) else prev_marker
        merged[-1] = (prev_start, new_end, best)
    return [
        {
            "marker": marker,
            "start": start,
            "end": end,
            "matchedSubstring": lowered[start:end],
        }
        for start, end, marker in merged
    ]


def count_technical_familiarity_occurrences(text: str) -> tuple[int, List[Dict[str, Any]]]:
    """Overlap-safe technical/mechanism occurrence count for familiarity assessment."""
    matches = _non_overlapping_marker_matches(text, _ALL_TECHNICAL_FAMILIARITY_MARKERS)
    enriched: List[Dict[str, Any]] = []
    for item in matches:
        marker = item["marker"]
        family = (
            "technical_vocab"
            if marker in _TECHNICAL_VOCABULARY_MARKERS
            else "mechanism_device"
        )
        enriched.append({**item, "family": family})
    return len(enriched), enriched


def collect_common_familiarity_matches(text: str) -> List[Dict[str, str]]:
    lowered = _norm(text).casefold()
    hits: List[Dict[str, str]] = []
    for marker in _COMMON_EVERYDAY_MARKERS:
        if marker in lowered:
            hits.append({"marker": marker, "family": "common_everyday"})
    return hits


def _contains_observable_causal_action(text: str) -> bool:
    lowered = _norm(text).casefold()
    if _HEBREW_OBSERVABLE_CAUSAL_ACTION_RE.search(lowered):
        return True
    return _contains_any(text, _OBSERVABLE_CAUSAL_ACTION_MARKERS)


def _contains_physical_action(text: str) -> bool:
    lowered = _norm(text).casefold()
    if _HEBREW_OBSERVABLE_CAUSAL_ACTION_RE.search(lowered):
        return True
    if _HEBREW_VISIBLE_ACTION_RE.search(lowered):
        return True
    return _contains_any(text, _PHYSICAL_ACTION_MARKERS)


def _count_symbolic_mappings(*texts: str) -> int:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return 0
    count = 0
    for marker in _SYMBOLIC_MAPPING_MARKERS:
        count += len(re.findall(re.escape(marker), combined))
    count += len(re.findall(r"\bmeans\b", combined))
    return count


def _ad_internal_fields(plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> Dict[str, Any]:
    index = ad.get("index")
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    ad_internals = internals.get("adInternals") if isinstance(internals, dict) else {}
    merged = dict(ad)
    if isinstance(ad_internals, dict) and index is not None:
        per_ad = ad_internals.get(index) or ad_internals.get(str(index))
        if isinstance(per_ad, dict):
            merged = {**merged, **per_ad}
    return merged


def _dominant_execution_tokens(*, plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> Set[str]:
    fields = _ad_internal_fields(plan_dict, ad)
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))
    candidates = [
        transferred,
        _norm(fields.get("executionSubject")),
        _norm(fields.get("executionAction")),
        _norm(fields.get("executionScene")),
        _norm(ad.get("physicalExecution")),
        _norm(ad.get("sceneDescription")),
    ]
    tokens: Set[str] = set()
    for text in candidates:
        if not text or _GENERIC_EXECUTION_RE.match(text):
            continue
        tokens.update(_significant_tokens(text))
    return tokens


def _negated_competing_terms(no_reuse_check: str) -> Set[str]:
    lowered = _norm(no_reuse_check).casefold()
    negated: Set[str] = set()
    if not lowered:
        return negated
    for term in _COMPETING_INTERPRETATION_TERMS:
        if re.search(rf"\b(?:no|not|without|never|avoid|excluding)\b[^.]{{0,24}}\b{re.escape(term)}\b", lowered):
            negated.add(term)
        if re.search(rf"\b{re.escape(term)}\b[^.]{{0,24}}\b(?:not|forbidden|excluded)\b", lowered):
            negated.add(term)
    return negated


def _scene_mentions_competing_term(text: str, term: str) -> bool:
    lowered = _norm(text).casefold()
    if not lowered:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", lowered))


def _strategic_role_explained(term: str, *texts: str) -> bool:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return False
    if term not in combined:
        return False
    window_pattern = rf".{{0,80}}\b{re.escape(term)}\b.{{0,80}}"
    match = re.search(window_pattern, combined)
    if not match:
        return False
    window = match.group(0)
    return _contains_any(window, _STRATEGIC_ROLE_MARKERS) or _contains_any(combined, _ADVANTAGE_BRIDGE_MARKERS)


def _category_tokens(text: str) -> Set[str]:
    lowered = _norm(text).casefold()
    return set(re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", lowered))


def _active_service_clusters(text: str, *, min_hits: int = 1) -> Set[str]:
    lowered = _norm(text).casefold()
    if not lowered:
        return set()
    tokens = _category_tokens(lowered)
    active: Set[str] = set()
    for cluster_id, signals in _SERVICE_CATEGORY_SIGNALS.items():
        hits = 0
        for signal in signals:
            signal_norm = signal.casefold()
            if " " in signal_norm:
                if signal_norm in lowered:
                    hits += 1
            elif signal_norm in tokens:
                hits += 1
        if hits >= min_hits:
            active.add(cluster_id)
    return active


def _advertised_service_context(plan_dict: Mapping[str, Any]) -> str:
    return " ".join(
        _norm(plan_dict.get(key))
        for key in (
            "productDescription",
            "productName",
            "productNameResolved",
            "relativeAdvantage",
            "strategicProblem",
        )
        if _norm(plan_dict.get(key))
    )


def _visual_execution_context(fields: Mapping[str, Any], ad: Mapping[str, Any]) -> str:
    return " ".join(
        _norm(fields.get(key)) or _norm(ad.get(key))
        for key in (
            "executionScene",
            "executionSubject",
            "executionAction",
            "executionObjectState",
            "executionPunchline",
            "physicalExecution",
            "visualExecution",
            "sceneDescription",
        )
        if (_norm(fields.get(key)) or _norm(ad.get(key)))
    )


def _generator_mechanism_context(plan_dict: Mapping[str, Any]) -> str:
    return " ".join(
        _norm(plan_dict.get(key))
        for key in (
            "physicalGenerator",
            "transferredObject",
            "transferredObjectAction",
            "physicalGeneratorCampaignRole",
            "physicalGeneratorNaturalPurpose",
        )
        if _norm(plan_dict.get(key))
    )


def _bridge_connects_advantage(
    *,
    bridge: str,
    relative_advantage: str,
    slogan_connection: str,
    punchline: str,
    fields: Mapping[str, Any],
    ad: Mapping[str, Any],
) -> bool:
    return bool(
        bridge
        and (
            _contains_any(bridge, _ADVANTAGE_BRIDGE_MARKERS)
            or _token_overlap(relative_advantage, bridge)
            or _token_overlap(relative_advantage, slogan_connection)
            or _token_overlap(relative_advantage, punchline)
            or (
                _contains_any(bridge, ("proves", "shows", "demonstrates", "expresses", "means"))
                and (
                    _token_overlap(bridge, _norm(fields.get("singleChangedPropertyOrAction")))
                    or _token_overlap(bridge, _norm(fields.get("conceptualActionProof")))
                    or _token_overlap(bridge, _norm(ad.get("conceptualExecution")))
                )
            )
        )
    )


def assess_everyday_familiarity(*texts: str) -> str:
    """Semantic everyday familiarity: universal > common > specialized > technical."""
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return "common"
    if _contains_any(combined, _UNIVERSAL_EVERYDAY_MARKERS):
        return "universal"
    technical_hits, _ = count_technical_familiarity_occurrences(combined)
    if technical_hits >= 2:
        return "technical"
    if technical_hits >= 1:
        return "specialized"
    if _contains_any(combined, _COMMON_EVERYDAY_MARKERS):
        return "common"
    return "common"


def _immediate_clarity_insufficient(
    *,
    immediate: str,
    bridge: str,
    relative_advantage: str,
    execution_blob: str,
) -> bool:
    if not immediate:
        return False
    familiarity_only = _contains_any(immediate, _FAMILIARITY_ONLY_CLARITY_MARKERS) or (
        _contains_any(immediate, _PHYSICAL_ONLY_CLARITY_MARKERS)
        and not _contains_physical_action(immediate)
    )
    if not familiarity_only:
        return False
    describes_action = _contains_physical_action(f"{immediate} {execution_blob}")
    connects_advantage = _token_overlap(relative_advantage, immediate) or _token_overlap(
        relative_advantage, bridge
    )
    return not describes_action or not connects_advantage


def _passes_two_sentence_test(
    *,
    immediate: str,
    bridge: str,
    relative_advantage: str,
    execution_blob: str,
) -> bool:
    physical_sentence = f"{immediate} {execution_blob}".strip()
    if not physical_sentence:
        return False
    if not _contains_physical_action(physical_sentence):
        if assess_everyday_familiarity(physical_sentence) in ("specialized", "technical"):
            return False
    if not bridge:
        return False
    if len(re.findall(r"[.!?]", bridge)) > 2:
        return False
    return _token_overlap(relative_advantage, bridge) or _contains_any(
        bridge, _ADVANTAGE_BRIDGE_MARKERS
    )


def _public_analogy_branch_reason(branch: str) -> str:
    reasons = {
        "mapping_count_ge_3_without_bridge": "Three or more symbolic mappings without a recoverable advantage bridge.",
        "technical_familiarity_without_simple_physical_event": (
            "Technical familiarity without a simple observable physical event and clear bridge."
        ),
        "specialized_mapping_ge_2_without_bridge": (
            "Specialized vocabulary with two or more mappings and no recoverable bridge."
        ),
        "technical_vocab_without_simple_or_bridge": (
            "Technical vocabulary present without simple physical event or bridge."
        ),
        "two_sentence_fail_with_familiarity_or_mapping": (
            "Two-sentence public comprehension test failed with specialized/technical familiarity or multi-mapping."
        ),
        "immediate_clarity_insufficient": (
            "Immediate clarity names familiarity or static scene without action and advantage connection."
        ),
    }
    return reasons.get(branch, branch)


def _evaluate_public_analogy_complexity(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[bool, Optional[str], Dict[str, Any]]:
    relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
    immediate = _norm(fields.get("immediateClarityReason"))
    bridge = _norm(fields.get("relativeAdvantageConnection"))
    slogan_connection = _norm(fields.get("sloganConnection"))
    punchline = _norm(fields.get("executionPunchline"))
    generator_blob = _generator_mechanism_context(plan_dict)
    execution_blob = _visual_execution_context(fields, ad)
    combined_explanation = " ".join(
        part for part in (immediate, bridge, slogan_connection, generator_blob) if part
    )
    physical_blob = f"{execution_blob} {immediate}".strip()
    familiarity_blob = f"{combined_explanation} {execution_blob}".strip()

    bridge_ok = _bridge_connects_advantage(
        bridge=bridge,
        relative_advantage=relative_advantage,
        slogan_connection=slogan_connection,
        punchline=punchline,
        fields=fields,
        ad=ad,
    )
    mapping_count = _count_symbolic_mappings(
        immediate,
        bridge,
        slogan_connection,
        _norm(plan_dict.get("conceptualGenerator")),
        _norm(plan_dict.get("conceptualGeneratorAction")),
        _norm(fields.get("conceptualExecution")),
    )
    familiarity = assess_everyday_familiarity(combined_explanation, execution_blob)
    _, technical_matches = count_technical_familiarity_occurrences(familiarity_blob)
    common_matches = collect_common_familiarity_matches(familiarity_blob)
    observable_causal = _contains_observable_causal_action(physical_blob)
    physical_action_detected = _contains_physical_action(physical_blob)
    simple_physical_event = observable_causal and physical_action_detected
    two_sentence_passed = _passes_two_sentence_test(
        immediate=immediate,
        bridge=bridge,
        relative_advantage=relative_advantage,
        execution_blob=execution_blob,
    )

    context: Dict[str, Any] = {
        "mappingCount": mapping_count,
        "bridgeOk": bridge_ok,
        "twoSentenceTestPassed": two_sentence_passed,
        "everydayFamiliarity": familiarity,
        "simplePhysicalEvent": simple_physical_event,
        "observableCausal": observable_causal,
        "physicalActionDetected": physical_action_detected,
        "technicalMatches": technical_matches,
        "commonMatches": common_matches,
    }

    if simple_physical_event and bridge_ok and mapping_count < 3:
        return False, None, context

    if mapping_count >= 3 and not bridge_ok:
        return True, "mapping_count_ge_3_without_bridge", context

    if familiarity == "technical" and not (bridge_ok and simple_physical_event):
        return True, "technical_familiarity_without_simple_physical_event", context

    if familiarity == "specialized" and mapping_count >= 2 and not bridge_ok:
        return True, "specialized_mapping_ge_2_without_bridge", context

    if _contains_any(combined_explanation, _TECHNICAL_VOCABULARY_MARKERS):
        if not simple_physical_event and not bridge_ok:
            return True, "technical_vocab_without_simple_or_bridge", context

    if not two_sentence_passed:
        if familiarity in ("specialized", "technical") or mapping_count >= 2:
            return True, "two_sentence_fail_with_familiarity_or_mapping", context

    if _immediate_clarity_insufficient(
        immediate=immediate,
        bridge=bridge,
        relative_advantage=relative_advantage,
        execution_blob=execution_blob,
    ):
        return True, "immediate_clarity_insufficient", context

    return False, None, context


def detect_public_analogy_too_complex(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
    fields: Optional[Mapping[str, Any]] = None,
    integrity_evidence: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """True when the planned analogy needs too many inferential steps for a general-public viewer."""
    merged_fields = dict(fields or _ad_internal_fields(plan_dict, ad))
    too_complex, branch, context = _evaluate_public_analogy_complexity(
        plan_dict=plan_dict,
        ad=ad,
        fields=merged_fields,
    )
    if too_complex and branch:
        record_integrity_evidence(
            integrity_evidence,
            code="public_analogy_too_complex",
            detector="advertising_comprehension",
            branch=branch,
            reason=_public_analogy_branch_reason(branch),
            level="ad",
            ad_index=int(ad.get("index") or 0) or None,
            field="relativeAdvantageConnection",
            field_value_preview=_norm(merged_fields.get("relativeAdvantageConnection")),
            extra=context,
        )
    return too_complex


def detect_competing_category_visual(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
    fields: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when dominant execution reads as advertising a different service category."""
    merged_fields = dict(fields or _ad_internal_fields(plan_dict, ad))
    advertised_text = _advertised_service_context(plan_dict)
    visual_text = _visual_execution_context(merged_fields, ad)

    advertised_clusters = _active_service_clusters(advertised_text, min_hits=1)
    visual_clusters = _active_service_clusters(visual_text, min_hits=1)

    advertised_prof = advertised_clusters & _PROFESSIONAL_SERVICE_CLUSTERS
    visual_prof = visual_clusters & _PROFESSIONAL_SERVICE_CLUSTERS
    if not advertised_prof or not visual_prof:
        return False

    competing_visual = visual_prof - advertised_prof
    if not competing_visual:
        return False

    # Cross-domain objects alone are not a category failure — require a plausible
    # instructional/professional service scene, not merely an external metaphor object.
    if competing_visual & _INSTRUCTIONAL_SERVICE_CLUSTERS:
        return _contains_any(visual_text, _INSTRUCTIONAL_SCENE_MARKERS)

    return bool(competing_visual)


def detect_mechanism_not_observable(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
    fields: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when claimed physical mechanism is not recoverable from planned execution."""
    merged_fields = dict(fields or _ad_internal_fields(plan_dict, ad))
    generator_blob = _generator_mechanism_context(plan_dict)
    execution_blob = _visual_execution_context(merged_fields, ad)
    immediate = _norm(merged_fields.get("immediateClarityReason"))

    mechanism_claimed = _contains_any(generator_blob, _MECHANISM_DEVICE_MARKERS)
    if not mechanism_claimed:
        return False

    mechanism_visible = _contains_any(execution_blob, _MECHANISM_DEVICE_MARKERS)
    if mechanism_visible:
        return False

    if _contains_any(execution_blob, _OBSERVABLE_CAUSAL_ACTION_MARKERS):
        return False

    if _contains_any(execution_blob, _RESULT_ONLY_MARKERS) or _contains_any(
        immediate, _VISUAL_EFFECT_ONLY_MARKERS
    ):
        return True

    return False


def scan_plan_category_integrity(plan_dict: Mapping[str, Any]) -> List[str]:
    """Deterministic category-integrity and observable-mechanism scan for a stored plan."""
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return []
    reasons: List[str] = []
    for ad in ads:
        if isinstance(ad, dict):
            reasons.extend(
                validate_ad_category_integrity(plan_dict=plan_dict, ad=ad)
            )
    return list(dict.fromkeys(reasons))


def scan_plan_physical_repair_reasons(plan_dict: Mapping[str, Any]) -> List[str]:
    """Planning-level violations that should trigger physical/analogy repair."""
    return [
        code
        for code in scan_advertising_comprehension(plan_dict)
        if code in PLAN_PHYSICAL_REPAIR_CODES
    ]


def validate_ad_category_integrity(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    fields = _ad_internal_fields(plan_dict, ad)
    if detect_competing_category_visual(plan_dict=plan_dict, ad=ad, fields=fields):
        reasons.append("competing_category_visual")
    if detect_mechanism_not_observable(plan_dict=plan_dict, ad=ad, fields=fields):
        reasons.append("advertising_mechanism_not_observable")
    return reasons


def validate_ad_advertising_comprehension(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
    integrity_evidence: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    reasons: List[str] = []
    fields = _ad_internal_fields(plan_dict, ad)
    relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
    immediate = _norm(fields.get("immediateClarityReason"))
    bridge = _norm(fields.get("relativeAdvantageConnection"))
    slogan_connection = _norm(fields.get("sloganConnection"))
    punchline = _norm(fields.get("executionPunchline"))
    no_reuse = _norm(fields.get("noReuseCheck"))
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))

    bridge_ok = _bridge_connects_advantage(
        bridge=bridge,
        relative_advantage=relative_advantage,
        slogan_connection=slogan_connection,
        punchline=punchline,
        fields=fields,
        ad=ad,
    )

    if immediate and bridge:
        physical_only = (
            _contains_any(immediate, _PHYSICAL_ONLY_CLARITY_MARKERS)
            and not _token_overlap(relative_advantage, immediate)
        )
        if physical_only and not bridge_ok:
            reasons.append("advertising_bridge_unclear")

    if immediate and _contains_any(immediate, _VISUAL_EFFECT_ONLY_MARKERS):
        clarity_is_effect_only = not (
            _contains_any(immediate, _ADVANTAGE_BRIDGE_MARKERS)
            or _token_overlap(relative_advantage, immediate)
            or _token_overlap(relative_advantage, bridge)
            or _contains_any(immediate, _MECHANISM_DEVICE_MARKERS)
        )
        if clarity_is_effect_only and not bridge_ok:
            if "advertising_bridge_unclear" not in reasons:
                reasons.append("advertising_bridge_unclear")

    mapping_count = _count_symbolic_mappings(
        immediate,
        bridge,
        slogan_connection,
        _norm(plan_dict.get("conceptualGenerator")),
        _norm(plan_dict.get("conceptualGeneratorAction")),
        _norm(fields.get("conceptualExecution")),
    )
    if mapping_count >= 3 and not bridge_ok:
        reasons.append("multi_hop_symbolic_chain")
        if "public_analogy_too_complex" not in reasons:
            reasons.append("public_analogy_too_complex")

    scene = " ".join(
        _norm(fields.get(field))
        for field in ("executionScene", "executionSubject", "executionAction", "executionObjectState")
    )
    scene = f"{scene} {_norm(ad.get('sceneDescription'))}".strip()
    strategic_blob = " ".join(
        part for part in (bridge, slogan_connection, punchline, transferred, no_reuse) if part
    ).casefold()
    for term in _UNEXPLAINED_DOMINANT_NOUNS:
        if term in scene.casefold() and term not in strategic_blob and term not in transferred.casefold():
            reasons.append("dominant_object_strategic_role_missing")
            break

    negated = _negated_competing_terms(no_reuse)
    scene_blob = " ".join(
        _norm(fields.get(field))
        for field in ("executionScene", "executionSubject", "executionAction", "executionObjectState")
    )
    for term in negated:
        if _scene_mentions_competing_term(scene_blob, term) and not _strategic_role_explained(
            term,
            bridge,
            punchline,
            slogan_connection,
        ):
            reasons.append("dominant_object_strategic_role_missing")
            break

    reasons.extend(validate_ad_category_integrity(plan_dict=plan_dict, ad=ad))

    if detect_public_analogy_too_complex(
        plan_dict=plan_dict,
        ad=ad,
        fields=fields,
        integrity_evidence=integrity_evidence,
    ):
        if "public_analogy_too_complex" not in reasons:
            reasons.append("public_analogy_too_complex")

    return list(dict.fromkeys(reasons))


def scan_advertising_comprehension(
    plan_dict: Mapping[str, Any],
    integrity_evidence: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return []
    reasons: List[str] = []
    for ad in ads:
        if isinstance(ad, dict):
            reasons.extend(
                validate_ad_advertising_comprehension(
                    plan_dict=plan_dict,
                    ad=ad,
                    integrity_evidence=integrity_evidence,
                )
            )
    return list(dict.fromkeys(reasons))


def _ad_internals_for_index(series_plan: Builder1SeriesPlan, ad_index: int) -> Dict[str, Any]:
    internals = series_plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    raw = ad_internals.get(ad_index) or ad_internals.get(str(ad_index)) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _ad_plan_fields(series_plan: Builder1SeriesPlan, ad_index: int) -> Dict[str, Any]:
    ad = next((item for item in series_plan.ads if item.index == ad_index), None)
    if ad is None:
        return {}
    return {
        "index": ad.index,
        "physicalExecution": ad.physical_execution,
        "visualExecution": ad.visual_execution,
        "sceneDescription": ad.scene_description,
        "conceptualExecution": ad.conceptual_execution,
    }


def build_planned_execution_compliance_block(
    series_plan: Builder1SeriesPlan,
    *,
    ad_index: int = 1,
) -> str:
    """Authoritative planned execution context for the existing compliance review call."""
    internals = _ad_internals_for_index(series_plan, ad_index)
    ad_fields = _ad_plan_fields(series_plan, ad_index)
    lines = [
        "=== PLANNED EXECUTION CONTEXT (AUTHORITATIVE — JUDGE FIDELITY, NOT BEAUTY) ===",
        f'relativeAdvantage: "{series_plan.relative_advantage}"',
        f'brandSlogan: "{series_plan.brand_slogan}"',
        f'sloganAction: "{series_plan.slogan_action}"',
        f'physicalGenerator: "{series_plan.physical_generator}"',
        f'transferredObject: "{series_plan.transferred_object or series_plan.physical_generator}"',
        f'transferredObjectAction: "{series_plan.transferred_object_action or series_plan.physical_generator_campaign_role}"',
        f'conceptualGenerator: "{series_plan.conceptual_generator}"',
        f"executionSubject: {_norm(internals.get('executionSubject') or ad_fields.get('physicalExecution'))}",
        f"executionAction: {_norm(internals.get('executionAction'))}",
        f"executionObjectState: {_norm(internals.get('executionObjectState'))}",
        f"executionScene: {_norm(internals.get('executionScene') or ad_fields.get('sceneDescription'))}",
        f"executionPunchline: {_norm(internals.get('executionPunchline'))}",
        f"immediateClarityReason: {_norm(internals.get('immediateClarityReason'))}",
        f"relativeAdvantageConnection: {_norm(internals.get('relativeAdvantageConnection'))}",
        f"sloganConnection: {_norm(internals.get('sloganConnection'))}",
        f"noReuseCheck: {_norm(internals.get('noReuseCheck'))}",
        f"categoryRelevanceReason: {_norm(internals.get('categoryRelevanceReason'))}",
        f"advertisedProductDescription: {series_plan.product_description}",
        "",
        "Category integrity (mandatory — separate from categoryRelevanceReason):",
        "- Before reading copy, what business/service/category would a normal viewer think this ad sells?",
        "- Fail ONLY when pixels naturally read as advertising a DIFFERENT commercial service/profession/category.",
        "- Cross-domain metaphor objects (magnet, umbrella, parachute, domino) are NOT failures by themselves.",
        "- Reject when the scene looks like another teachable discipline or service business",
        "  (e.g. history tutor vs gymnastics coaching arena, not history tutor vs magnet attracting one object).",
        "- categoryRelevanceReason does NOT override category integrity failure.",
        "",
        "Public comprehension (mandatory — same review call):",
        "- Can a general-public viewer identify the physical event immediately?",
        "- Is the intended causal mechanism visible/recoverable without specialist knowledge?",
        "- Does the image require technical/professional vocabulary to interpret?",
        "- Can the relative-advantage bridge plausibly be recovered from pixels?",
        "- Fail when generation made a simple planned analogy look technical or confusing.",
        "",
        "Observable mechanism (mandatory):",
        "- Compare approved physicalGenerator/transferredObjectAction to pixels.",
        "- The causal relationship carrying the idea must be recoverable (magnet pulling, umbrella blocking rain).",
        "- Fail when the plan depends on a hidden device/process but pixels show only its aesthetic result",
        "  (e.g. autofocus camera claimed but only sharp-subject/blur-background sports photo visible).",
        "- Do NOT require diagrams, labels, or explanatory arrows.",
        "",
        "Evaluate whether the generated pixels faithfully execute this approved mechanism.",
        "This is NOT aesthetic criticism.",
        "Fail when concrete evidence shows:",
        "- planned object/scene became a materially different object/context (e.g. conveyor vs railway tracks)",
        "- central proof/punchline is absent or visually ambiguous",
        "- dominant subject diverged from the planned transferred object or execution subject",
        "- an unintended dominant interpretation appears (especially one explicitly denied in noReuseCheck)",
        "- a normal viewer cannot recover the relative advantage from pixels alone (slogan may reinforce, not carry all meaning)",
        "- pixels and slogan communicate unrelated mechanisms",
        "",
        "Category/mechanism/public hard violation codes (same review call):",
        "competing_category_visual, advertising_mechanism_not_observable, public_analogy_not_recoverable",
        "",
        "Execution fidelity hard violation codes:",
        ", ".join(sorted(EXECUTION_FIDELITY_VIOLATION_CODES)),
        "=== END PLANNED EXECUTION CONTEXT ===",
    ]
    return "\n".join(lines)


ANALOGY_REPAIR_VIOLATION_CODES = frozenset(
    {
        "public_analogy_too_complex",
        "competing_category_visual",
        "advertising_mechanism_not_observable",
        "multi_hop_symbolic_chain",
        "advertising_bridge_unclear",
        "public_analogy_not_recoverable",
    }
)


def build_analogy_repair_guidance_block(violations: Sequence[str]) -> str:
    """Deterministic repair guidance — simpler analogy, never literal category depiction."""
    codes = {str(code).strip() for code in violations if str(code).strip()}
    if not codes & ANALOGY_REPAIR_VIOLATION_CODES:
        return ""
    lines = [
        "=== ANALOGY REPAIR (MANDATORY) ===",
        "Preserve strategy, relative advantage, slogan, and conceptual generator.",
        "Do NOT replace the visual with literal product/category imagery (teacher, classroom, book, exam, product in use).",
        "Find a simpler, more universally understood physical analogy for the SAME relative advantage.",
        "Prefer widely understood everyday mechanisms (magnet attracts, door opens, umbrella blocks rain)",
        "over technical/industrial/optical systems unless instantly obvious to a general-public viewer.",
        "Pass the child-comprehension heuristic: one simple sentence for what physically happens,",
        "one simple sentence for why that expresses the relative advantage.",
        "=== END ANALOGY REPAIR ===",
    ]
    return "\n".join(lines)


def build_execution_fidelity_correction_block(
    *,
    violations: List[str],
    series_plan: Builder1SeriesPlan,
    ad_index: int = 1,
) -> str:
    fidelity = [code for code in violations if code in EXECUTION_FIDELITY_VIOLATION_CODES]
    if not fidelity:
        return ""
    internals = _ad_internals_for_index(series_plan, ad_index)
    scene = _norm(internals.get("executionScene"))
    punchline = _norm(internals.get("executionPunchline"))
    no_reuse = _norm(internals.get("noReuseCheck"))
    transferred = series_plan.transferred_object or series_plan.physical_generator
    lines = [
        "=== EXECUTION FIDELITY CORRECTION (MANDATORY) ===",
        f"Preserve the approved campaign plan. Regenerate pixels only.",
        f"MAIN VISUAL must remain: {transferred}",
        f"Planned scene/context: {scene or '(see approved plan)'}",
        f"Central proof/punchline must be immediately visible: {punchline or '(see approved plan)'}",
    ]
    if no_reuse:
        lines.append(f"Explicit exclusions from approved plan: {no_reuse}")
    if "planned_scene_diverged" in fidelity or "planned_mechanism_diverged" in fidelity:
        lines.append(
            "Render the approved industrial/testing/conveyor context exactly — do NOT substitute railway tracks, roads, or route imagery unless explicitly planned."
        )
    if "central_proof_not_visible" in fidelity:
        lines.append(
            "Make the planned proof/punchline visually dominant and unambiguous in the final pixels."
        )
    if "unintended_dominant_interpretation" in fidelity or "dominant_subject_diverged" in fidelity:
        lines.append(
            "Remove competing unintended interpretations. The dominant subject must match the approved transferred object and strategic mechanism."
        )
    if "advertising_bridge_not_recoverable" in fidelity or "relative_advantage_not_expressed" in fidelity:
        lines.append(
            "Ensure the visible mechanism makes the relative advantage recoverable without requiring hidden symbolic translation."
        )
    if "public_analogy_not_recoverable" in fidelity:
        lines.append(
            "Restore immediate public comprehension: show a simple causal physical event a general viewer can read in one glance."
        )
    lines.extend(
        [
            "Do not change Product Name, slogan, palette, or graphic system.",
            "=== END EXECUTION FIDELITY CORRECTION ===",
        ]
    )
    return "\n".join(lines)
