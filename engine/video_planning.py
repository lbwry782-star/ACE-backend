"""
ACE video engine — GPT-5.6 Sol planning layer (isolated from image /preview /generate).

Produces a structured plan for Runway prompt assembly. On failure, video jobs abort (no generic Runway prompt).
Future: richer ACE video engine (e.g. two outputs); keep this module inspectable and logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import httpx
from openai import OpenAI

from engine.video_language import (
    normalize_video_content_language,
    video_language_display_name,
)

_MAX_HEADLINE_REMAINDER_WORDS = 7

# Scene / video_prompt must not imply surreal or impossible visuals.
_SCENE_FORBIDDEN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("surreal", re.compile(r"\bsurreal\w*\b", re.I)),
    ("dreamlike", re.compile(r"\bdream[\s-]?like\b", re.I)),
    ("fantasy", re.compile(r"\bfantas(y|ical)\b", re.I)),
    ("magic", re.compile(r"\bmagic(al)?\b", re.I)),
    ("impossible_physics", re.compile(r"\bimpossible\s+physics\b", re.I)),
    ("talking_object", re.compile(r"\btalking\s+(object|objects|item|items)\b", re.I)),
    ("animated_object", re.compile(r"\banimated\s+(object|objects|item|items)\b", re.I)),
    ("floating_object", re.compile(r"\bfloating\s+(object|objects|symbol|symbols)\b", re.I)),
    ("science_fiction", re.compile(r"\bscience[\s-]?fiction\b|\bsci[\s-]?fi\b", re.I)),
    ("levitat", re.compile(r"\blevitat\w*\b", re.I)),
    ("teleport", re.compile(r"\bteleport\w*\b", re.I)),
    ("morph", re.compile(r"\bmorph(?:s|ed|ing)?\b", re.I)),
    ("symbolic_object", re.compile(r"\bsymbolic\s+(object|objects|float|floating)\b", re.I)),
    ("heart_shaped", re.compile(r"\bheart[\s-]?shaped\b", re.I)),
]

# Literal industry / category words — forbidden as headlineCoreKeyword (headline may still mention them).
_WEAK_INDUSTRY_KEYWORDS: FrozenSet[str] = frozenset(
    {
        "advertising",
        "marketing",
        "digital",
        "campaign",
        "story",
        "service",
        "strategy",
        "agency",
        "brand",
        "branding",
        "social",
        "media",
        "content",
        "promotion",
        "promo",
        "creative",
        "copy",
        "copywriting",
        "seo",
        "ads",
        "ad",
        "פרסום",
        "שיווק",
        "דיגיטלי",
        "דיגיטל",
        "קמפיין",
        "סטורי",
        "שירות",
        "אסטרטגיה",
        "פרסומת",
        "פרסומות",
        "מדיה",
        "תוכן",
        "קידום",
        "קידוםמכירות",
    }
)

_KEYWORD_FILLER_WORDS: FrozenSet[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "very",
        "most",
        "more",
        "less",
        "so",
        "too",
        "also",
        "just",
        "only",
        "even",
        "still",
        "already",
        "yet",
        "than",
        "then",
        "there",
        "here",
        "when",
        "where",
        "how",
        "why",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "if",
        "not",
        "no",
        "yes",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "about",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "between",
        "among",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "again",
        "once",
        "ever",
        "never",
        "always",
        "often",
        "sometimes",
        "usually",
        "really",
        "quite",
        "rather",
        "such",
        "same",
        "other",
        "another",
        "each",
        "every",
        "all",
        "any",
        "some",
        "many",
        "much",
        "few",
        "little",
        "own",
        "new",
        "old",
        "good",
        "best",
        "better",
        "well",
        "one",
        "two",
        "first",
        "last",
        "next",
        "ה",
        "ו",
        "ש",
        "כ",
        "ל",
        "מ",
        "ב",
        "ע",
        "על",
        "אל",
        "את",
        "עם",
        "של",
        "זה",
        "זו",
        "זאת",
        "הוא",
        "היא",
        "הם",
        "הן",
        "אני",
        "אתה",
        "את",
        "אנחנו",
        "אתם",
        "אתן",
        "כי",
        "אם",
        "או",
        "גם",
        "רק",
        "עוד",
        "כבר",
        "לא",
        "כן",
        "מאוד",
        "הכי",
        "יותר",
        "פחות",
        "כל",
        "כמה",
        "איזה",
        "איזו",
        "מה",
        "מי",
        "איך",
        "למה",
        "מתי",
        "איפה",
        "כאן",
        "שם",
        "עכשיו",
        "תמיד",
        "לעולם",
        "אף",
        "פעם",
    }
)

# Appended to Runway promptText in runway_video after text-policy sanitize (hard constraint).
RUNWAY_PHYSICS_REALISM_CONSTRAINT = (
    "PHYSICAL REALISM: All motion must obey real-world resistance, weight, and contact between surfaces. "
    "No frictionless sliding, gliding, drifting, or floating movement. Show grip, pressure, and resisted motion only."
)


def _is_weak_industry_keyword(keyword: str) -> bool:
    return _normalize_keyword_token(keyword) in _WEAK_INDUSTRY_KEYWORDS


# Entire headline rejected when meaning depends on a fixed phrase/idiom (not only the keyword token).
_HEADLINE_PHRASE_DEPENDENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"שופכ\w*\s+אור", re.I), "hebrew_shafach_or"),
    (re.compile(r"זורק\w*\s+אור", re.I), "hebrew_zorek_or"),
    (re.compile(r"ישר\s+לב", re.I), "hebrew_yashar_lev"),
    (re.compile(r"על\s+המפה", re.I), "hebrew_al_hamapa"),
    (re.compile(r"פותח\w*\s+דלת\s+להזדמנ", re.I), "hebrew_opens_door_opportunities"),
    (re.compile(r"\bshed\s+light\b", re.I), "english_shed_light"),
    (re.compile(r"\bcast(s|ing)?\s+light\b", re.I), "english_cast_light"),
    (re.compile(r"\bon\s+the\s+map\b", re.I), "english_on_the_map"),
    (re.compile(r"\bstraight\s+to\s+the\s+heart\b", re.I), "english_straight_to_heart"),
    (re.compile(r"\bopens?\s+(a\s+)?door\s+to\s+opportunit", re.I), "english_door_to_opportunities"),
]

# Headline phrase + keyword pairs where the keyword meaning depends on the collocation (not standalone).
_HEADLINE_KEYWORD_COLLOCATION_REJECT: List[Tuple[re.Pattern, FrozenSet[str], str]] = [
    (re.compile(r"שופכ\w*\s+אור", re.I), frozenset({"אור"}), "hebrew_shafach_or"),
    (re.compile(r"זורק\w*\s+אור", re.I), frozenset({"אור"}), "hebrew_zorek_or"),
    (re.compile(r"ישר\s+לב", re.I), frozenset({"לב"}), "hebrew_yashar_lev"),
    (re.compile(r"על\s+המפה", re.I), frozenset({"מפה"}), "hebrew_al_hamapa"),
    (re.compile(r"\bshed\s+light\b", re.I), frozenset({"light"}), "english_shed_light"),
    (re.compile(r"\bcast(s|ing)?\s+light\b", re.I), frozenset({"light"}), "english_cast_light"),
    (re.compile(r"\bthrow(s|ing)?\s+light\b", re.I), frozenset({"light"}), "english_throw_light"),
    (re.compile(r"\bbring(s|ing)?\s+.+\s+light\b", re.I), frozenset({"light"}), "english_bring_light"),
    (re.compile(r"\bshines?\s+a\s+light\b", re.I), frozenset({"light"}), "english_shine_a_light"),
    (re.compile(r"\bon\s+the\s+map\b", re.I), frozenset({"map"}), "english_on_the_map"),
]

# Builder2 videos are silent — scene/video prose must not depend on audio to be understood.
_SCENE_REQUIRES_SOUND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("shouting", re.compile(r"\bshout(s|ing|ed)?\b", re.I)),
    ("screaming", re.compile(r"\bscream(s|ing|ed)?\b", re.I)),
    ("whispering", re.compile(r"\bwhisper(s|ing|ed)?\b", re.I)),
    ("singing", re.compile(r"\bsing(s|ing|s)?\b", re.I)),
    ("loudspeaker", re.compile(r"\bloudspeaker\b|\bpa\s+system\b", re.I)),
    ("music_playing", re.compile(r"\bmusic\s+playing\b|\bloud\s+music\b", re.I)),
    ("alarm_sound", re.compile(r"\balarm\s+(sound|ring|sounding|goes\s+off)\b", re.I)),
    ("bell_ringing", re.compile(r"\bbell(s)?\s+(ring|ringing)\b", re.I)),
    ("crowd_cheering", re.compile(r"\bcheer(s|ing|ed)?\b", re.I)),
    ("engine_roaring", re.compile(r"\b(roar|roaring)\b", re.I)),
    ("loudly", re.compile(r"\bloudly\b", re.I)),
    ("broadcasting", re.compile(r"\bbroadcast(ing|s)?\b", re.I)),
    ("hebrew_shouting", re.compile(r"צועק|צורח|צריח", re.I)),
    ("hebrew_whispering", re.compile(r"לוחש|לחש", re.I)),
    ("hebrew_singing", re.compile(r"\bשר(ים|ה)?\b|שירה", re.I)),
    ("hebrew_alarm", re.compile(r"אזעקה|צלצול", re.I)),
    ("hebrew_barking", re.compile(r"נביח", re.I)),
    ("hebrew_loud", re.compile(r"בקול\s+רם|רועש", re.I)),
]


def _headline_depends_on_fixed_phrase(headline: str) -> Optional[str]:
    """Return rule label when headline meaning depends on a fixed phrase/idiom; else None."""
    for rx, label in _HEADLINE_PHRASE_DEPENDENT_PATTERNS:
        if rx.search(headline or ""):
            return label
    return None


def _keyword_depends_on_headline_phrase(headline: str, keyword: str) -> Optional[str]:
    """Return rule label when keyword meaning likely depends on headline collocation; else None."""
    kw_norm = _normalize_keyword_token(keyword)
    if not kw_norm:
        return None
    for rx, forbidden_kws, label in _HEADLINE_KEYWORD_COLLOCATION_REJECT:
        if not rx.search(headline or ""):
            continue
        if kw_norm in {_normalize_keyword_token(k) for k in forbidden_kws}:
            return label
    return None


def scene_fields_imply_forbidden_surrealism(blob: str) -> Optional[str]:
    """Return a rule label if scene/video prose matches forbidden surreal semantics; else None."""
    if not (blob or "").strip():
        return None
    for label, rx in _SCENE_FORBIDDEN_PATTERNS:
        if rx.search(blob):
            return label
    return None


def scene_fields_require_sound_to_verify(blob: str) -> Optional[str]:
    """Return a rule label if scene/video meaning depends on audio (Builder2 is silent); else None."""
    if not (blob or "").strip():
        return None
    for label, rx in _SCENE_REQUIRES_SOUND_PATTERNS:
        if rx.search(blob):
            return label
    return None


def _headline_remainder_word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def _assemble_headline_full(product_name: str, remainder: str) -> str:
    pn = (product_name or "").strip()
    rem = " ".join((remainder or "").split())
    if not pn:
        return rem
    if not rem:
        return pn
    return f"{pn} {rem}"


def _normalize_keyword_token(word: str) -> str:
    return (word or "").strip().strip(".,!?;:\"'()[]").lower()


def _headline_contains_core_keyword(headline: str, keyword: str) -> bool:
    kw = _normalize_keyword_token(keyword)
    if not kw:
        return False
    for word in (headline or "").split():
        wn = _normalize_keyword_token(word)
        if wn == kw or kw in wn or wn in kw:
            return True
    return kw in _normalize_keyword_token(headline)


# Light English glosses for non-keyword headline words that leak into English videoPrompt.
_HEADLINE_WORD_EN_GLOSS: Dict[str, FrozenSet[str]] = {
    "דרך": frozenset({"path", "road", "route", "walkway", "trail"}),
    "גשר": frozenset({"bridge"}),
    "דלת": frozenset({"door"}),
    "לב": frozenset({"heart"}),
    "בית": frozenset({"home", "house"}),
    "מפתח": frozenset({"key", "lock"}),
    "מפה": frozenset({"map"}),
    "אור": frozenset({"light", "lighting"}),
}

_SCENE_FEMININE_SUBJECT = re.compile(
    r"\b(woman|women|girl|girls|female|lady|ladies|beautiful\s+woman)\b|(?:^|\s)(אישה|בחורה|נערה)(?:\s|$)",
    re.I,
)
_SCENE_MASCULINE_SUBJECT = re.compile(
    r"\b(man|men|boy|boys|male|guy|guys)\b|(?:^|\s)(גבר|בחור)(?:\s|$)",
    re.I,
)
_MASCULINE_CONTEXT_HINT = re.compile(
    r"\b(man|men|male|guy|גבר|בחור)\b"
    r"|(?:^|\s)(מגיע|פותח|מוביל|מביא|בונה|יוצר|נותן)(?:\s|$)"
    r"|(?:^|\s)(אורי|דני|מאיר|יוסי|רועי|גיא|נועם|איתי|עומר)(?:\s|$)",
    re.I,
)
_FEMININE_CONTEXT_HINT = re.compile(
    r"\b(woman|women|female|girl|lady|אישה|בחורה|נערה|גברת)\b"
    r"|(?:^|\s)(מגיעה|פותחת|מובילה|מביאה|בונה|יוצרת|נותנת)(?:\s|$)"
    r"|(?:^|\s)(שרה|רונית|דנה|מיכל|נועה|יעל|הילה)(?:\s|$)",
    re.I,
)


def _hebrew_lemma_light(word: str) -> str:
    w = (word or "").strip()
    while len(w) > 2 and w[0] in "בלמכשהו":
        w = w[1:]
    return _normalize_keyword_token(w)


def _headline_words_excluding_keyword(headline: str, keyword: str) -> List[str]:
    kw_norm = _normalize_keyword_token(keyword)
    lemmas: List[str] = []
    seen: set[str] = set()
    for raw in (headline or "").split():
        wn = _normalize_keyword_token(raw)
        lemma = _hebrew_lemma_light(raw)
        if not wn or wn == kw_norm or lemma == kw_norm:
            continue
        if wn in _KEYWORD_FILLER_WORDS or lemma in _KEYWORD_FILLER_WORDS:
            continue
        for candidate in (wn, lemma):
            if len(candidate) <= 1:
                continue
            if candidate not in seen:
                seen.add(candidate)
                lemmas.append(candidate)
    return lemmas


def _scene_leaks_non_keyword_headline_word(
    headline: str, keyword: str, scene_blob: str
) -> Optional[str]:
    """Light check: scene/video must not import other headline words beyond the keyword."""
    other_words = _headline_words_excluding_keyword(headline, keyword)
    if not other_words:
        return None
    blob = (scene_blob or "").lower()
    for w in other_words:
        if re.search(rf"(?:^|\s){re.escape(w)}(?:\s|$)", blob, re.I):
            return f"headline_word_in_scene:{w}"
        for gloss in _HEADLINE_WORD_EN_GLOSS.get(w, ()):
            if re.search(rf"\b{re.escape(gloss)}\b", blob, re.I):
                return f"headline_gloss_in_scene:{w}->{gloss}"
    return None


def _implied_subject_gender(product_name: str, product_description: str, headline: str) -> Optional[str]:
    """Return m/f when product or copy lightly implies subject gender; else None."""
    combined = f"{product_name or ''} {product_description or ''} {headline or ''}"
    fem = bool(_FEMININE_CONTEXT_HINT.search(combined))
    masc = bool(_MASCULINE_CONTEXT_HINT.search(combined))
    if fem and not masc:
        return "f"
    if masc and not fem:
        return "m"
    return None


def _scene_gender_mismatch(
    product_name: str,
    product_description: str,
    headline: str,
    scene_blob: str,
) -> Optional[str]:
    implied = _implied_subject_gender(product_name, product_description, headline)
    if not implied:
        return None
    if implied == "m" and _SCENE_FEMININE_SUBJECT.search(scene_blob or ""):
        return "masculine_context_feminine_subject"
    if implied == "f" and _SCENE_MASCULINE_SUBJECT.search(scene_blob or ""):
        return "feminine_context_masculine_subject"
    return None


_BARE_OBJECT_CORE_VISUAL_TERMS: FrozenSet[str] = frozenset(
    {
        "compass",
        "bridge",
        "key",
        "keys",
        "house",
        "door",
        "map",
        "nest",
        "מצפן",
        "גשר",
        "מפתח",
        "בית",
        "דלת",
        "מפה",
        "קן",
    }
)

_KEYWORD_LITERAL_OBJECT_TOKENS: Dict[str, Tuple[str, ...]] = {
    "מצפן": ("compass", "מצפן"),
    "compass": ("compass",),
    "מפתח": ("key", "keys", "lock", "מפתח", "מנעול"),
    "key": ("key", "keys", "lock"),
    "גשר": ("bridge", "גשר", "footbridge"),
    "bridge": ("bridge", "footbridge"),
    "בית": ("house", "building exterior", "house exterior", "בית"),
    "house": ("house", "building exterior", "house exterior"),
    "home": ("house", "building exterior", "house exterior"),
    "דלת": ("door", "דלת"),
    "door": ("door",),
    "מפה": ("map", "מפה"),
    "map": ("map",),
}


def _core_visual_is_bare_object(core_visual: str) -> Optional[str]:
    """Reject coreVisualIdea that names only the physical object, not the underlying idea."""
    normalized = re.sub(r"\s+", " ", (core_visual or "").strip().lower())
    if not normalized:
        return None
    if normalized in _BARE_OBJECT_CORE_VISUAL_TERMS:
        return f"bare_object:{normalized}"
    tokens = re.sub(r"[^\w\s\u0590-\u05ff-]", " ", normalized).split()
    if len(tokens) == 1 and tokens[0] in _BARE_OBJECT_CORE_VISUAL_TERMS:
        return f"bare_object:{tokens[0]}"
    return None


def _literal_object_tokens_for_keyword(keyword: str) -> Tuple[str, ...]:
    kw = _normalize_keyword_token(keyword)
    for key, tokens in _KEYWORD_LITERAL_OBJECT_TOKENS.items():
        if _normalize_keyword_token(key) == kw:
            return tokens
    if kw in _BARE_OBJECT_CORE_VISUAL_TERMS:
        return (kw,)
    return ()


def _variation_mentions_object_token(variation: str, token: str) -> bool:
    blob = (variation or "").lower()
    token = (token or "").lower()
    if not token:
        return False
    if re.search(r"[\u0590-\u05ff]", token):
        return token in blob
    return bool(re.search(rf"\b{re.escape(token)}\b", blob, re.I))


def _variations_object_repetition_violation(keyword: str, variations: List[str]) -> Optional[str]:
    """Same physical object may appear in at most one variation."""
    tokens = _literal_object_tokens_for_keyword(keyword)
    if not tokens:
        return None
    hits = 0
    for var in variations:
        if any(_variation_mentions_object_token(var, t) for t in tokens):
            hits += 1
    if hits > 1:
        return f"object_in_{hits}_variations:max_1"
    if hits == len(variations) and len(variations) >= 2:
        return "object_in_every_variation"
    return None


from engine.ad_promise_memory import (
    angle_seed_for_attempt,
    build_promise_diversity_addon,
    compute_product_hash,
    forbidden_promises_for_prompt,
    increment_promise_stat,
    load_ad_promise_history,
    maybe_soft_reset_promise_memory,
)

logger = logging.getLogger(__name__)

# Safe preview length for logs (no secrets; truncated model output)
_LOG_PREVIEW_CHARS = 240


class VideoPlanningTimeoutError(Exception):
    """Hard wall-clock deadline exceeded waiting for o3 planning (see fetch_video_plan_o3)."""

def _text_model() -> str:
    from engine.builder2_reasoning_config import resolve_builder2_reasoning_model

    return resolve_builder2_reasoning_model()


# HTTP read timeout for the planning API call (seconds). Slightly raised default vs older 120s to cut false timeouts.
_VIDEO_PLAN_TIMEOUT = float((os.environ.get("VIDEO_PLANNER_TIMEOUT_SECONDS") or "150").strip() or "150")
# Wall-clock cap for the whole planning call (thread join); must exceed client read timeout so the SDK can surface errors
_VIDEO_PLAN_HARD_SECONDS = float(
    (os.environ.get("VIDEO_PLANNER_HARD_TIMEOUT_SECONDS") or str(_VIDEO_PLAN_TIMEOUT + 45.0)).strip()
    or str(_VIDEO_PLAN_TIMEOUT + 45.0)
)
_VIDEO_PLAN_MODEL_RETRY_BACKOFF_S = float(
    (os.environ.get("VIDEO_PLAN_MODEL_RETRY_BACKOFF_S") or "3").strip() or "3"
)
_VIDEO_PLAN_MODEL_MAX_ATTEMPTS = 2


def _builder2_video_duration_seconds() -> int:
    from engine.builder2_runway_config import resolve_builder2_video_duration_seconds

    return resolve_builder2_video_duration_seconds()


def _planner_duration_instruction_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        f"VIDEO DURATION (mandatory — server-configured {n} seconds total):\n"
        f"- Design a complete silent video that fits naturally within {n} seconds.\n"
        f"- The complete montage, including all sceneVariations and the final visual resolution, "
        f"must be understandable within {n} seconds.\n"
        f"- {n} seconds is the TOTAL video length — not {n} seconds per variation.\n"
        "- Keep transitions short and clear; do not add scenes just to fill time.\n"
        "- Do not artificially stretch a simple action; allow enough time for visual anchor/resolution.\n"
        f"- Do not front-load all action in the first few seconds and leave the remainder empty. "
        f"Use the full {n} seconds only when it strengthens clarity and pacing.\n"
        "- If 3-4 variations cannot fit clearly within the total duration, choose 2 stronger variations "
        "(still within 2-4 range).\n"
        "- Silent video: every moment must remain visually verifiable without sound.\n\n"
    )


def _json_keys_block() -> str:
    n = _builder2_video_duration_seconds()
    return f"""
Return one JSON object only (no markdown, no prose).

Keys:
- productNameResolved, headline, headlineCoreKeyword, coreVisualIdea, sceneVariations, videoPrompt, language (all strings except sceneVariations)
- sceneVariations: JSON array of 2-4 strings — brief independent variations of the same coreVisualIdea

Flow (mandatory order — internal only; output final JSON only):
1) Read product name + product description.
2) headline: direct advertising advantage; remainder ONLY; no productNameResolved inside headline; up to 7 words; reject phrase-dependent headlines.
3) headlineCoreKeyword: exactly ONE standalone semantic word from headline.
4) coreVisualIdea: the ESSENCE of the keyword — then its strongest extreme visual embodiment (not literal meaning, not bare object).
5) sceneVariations: 2-4 variations within ONE visual family of that essence — same motif, not dictionary examples.
6) videoPrompt: English {n}-second montage of those family-consistent variations; short, clear, realistic, visually distinct; no headline burn-in.

Empty product name → invent productNameResolved.

Before the JSON: revision pass (headline → keyword → essence → extreme embodiment → visual family → variations → montage videoPrompt).

Failure only: {{"planningFailure":"planning_failed_invalid_plan"}}
"""


def _video_plan_model_retry_backoff_s() -> float:
    return max(2.0, min(_VIDEO_PLAN_MODEL_RETRY_BACKOFF_S, 5.0))


def _is_transient_plan_model_call_error(exc: BaseException) -> bool:
    """True for API/network timeouts where one immediate retry may succeed."""
    err_type = type(exc).__name__
    if err_type in (
        "APITimeoutError",
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ConnectError",
        "RemoteProtocolError",
    ):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    return False


def _can_retry_plan_model_call(deadline_monotonic: Optional[float], backoff_s: float) -> bool:
    """Keep retry inside overall planning deadline with room for another call."""
    if deadline_monotonic is None:
        return True
    remaining = deadline_monotonic - time.monotonic()
    min_call_window = min(_VIDEO_PLAN_TIMEOUT, 60.0)
    return remaining > backoff_s + min_call_window


def _responses_create_with_plan_retry(
    client: OpenAI,
    *,
    model: str,
    input_text: str,
    reasoning: dict,
    deadline_monotonic: Optional[float] = None,
    role: str = "video_planning",
    base_call_type: str = "normal",
):
    """
    Up to two identical planning model calls; one retry on transient timeout only.
    Raises VideoPlanningTimeoutError if hard deadline is exceeded.
    """
    backoff_s = _video_plan_model_retry_backoff_s()
    last_exc: Optional[BaseException] = None

    from engine.builder2_reasoning_config import log_builder2_model_selected

    for attempt in range(1, _VIDEO_PLAN_MODEL_MAX_ATTEMPTS + 1):
        call_type = base_call_type if attempt == 1 else "retry"
        log_builder2_model_selected(role=role, call_type=call_type, attempt=attempt)
        logger.info("VIDEO_PLAN_MODEL_CALL_ATTEMPT attempt=%s", attempt)
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise VideoPlanningTimeoutError()
        try:
            response = client.responses.create(
                model=model,
                input=input_text,
                reasoning=reasoning,
            )
            logger.info("VIDEO_PLAN_MODEL_CALL_SUCCESS attempt=%s", attempt)
            return response
        except VideoPlanningTimeoutError:
            raise
        except Exception as e:
            last_exc = e
            transient = _is_transient_plan_model_call_error(e)
            if not transient:
                raise
            logger.warning(
                "VIDEO_PLAN_MODEL_CALL_TIMEOUT attempt=%s err_type=%s err=%s",
                attempt,
                type(e).__name__,
                e,
            )
            if attempt >= _VIDEO_PLAN_MODEL_MAX_ATTEMPTS:
                logger.warning(
                    "VIDEO_PLAN_MODEL_CALL_FINAL_FAIL err_type=%s err=%s",
                    type(e).__name__,
                    e,
                )
                raise
            if not _can_retry_plan_model_call(deadline_monotonic, backoff_s):
                logger.warning(
                    "VIDEO_PLAN_MODEL_CALL_FINAL_FAIL err_type=%s err=%s reason=deadline_insufficient_for_retry",
                    type(e).__name__,
                    e,
                )
                raise
            logger.info("VIDEO_PLAN_MODEL_CALL_RETRY attempt=%s", attempt + 1)
            time.sleep(backoff_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("plan_model_call_no_response")


def _video_plan_planner_description_limit() -> int:
    raw = (os.environ.get("VIDEO_PLANNER_MAX_DESCRIPTION_CHARS") or "2200").strip() or "2200"
    try:
        n = int(raw)
    except ValueError:
        n = 2200
    return max(400, min(n, 48000))


def _planner_essence_extreme_block() -> str:
    return (
        "ESSENCE EXTREME RULE (mandatory — variation_montage_v4):\n"
        "- Do NOT ask: \"What does this keyword mean?\"\n"
        "- DO ask: \"What is the most extreme visual expression of the ESSENCE of this keyword?\"\n"
        "- Goal is NOT literal meaning — goal is the purest visual embodiment of the keyword's essence.\n\n"
        "PROCESS:\n"
        "STEP A — Find the essence (not the object, not literal definition).\n"
        "STEP B — Ask: \"What is the strongest visual image that expresses this essence in the most extreme form?\"\n"
        "STEP C — coreVisualIdea names that extreme visual embodiment.\n"
        "STEP D — Choose ONE visual family for that embodiment; sceneVariations stay inside that family.\n"
        "STEP E — sceneVariations are 2-4 independent variations within the same visual family.\n\n"
        "ESSENCE EXAMPLES (keyword → essence — NOT object):\n"
        "- דלת — NOT door. Essence: opening / passage / transition.\n"
        "- קרוב — NOT standing nearby. Essence: maximum connection.\n"
        "- עזרה — NOT helping generically. Essence: small support preventing collapse.\n"
        "- בית — NOT house. Essence: protected belonging.\n\n"
        "EXTREME EMBODIMENT EXAMPLES:\n"
        "- קרוב — WEAK: two people standing near. BETTER: hug (strongest visual of closeness).\n"
        "- דלת — WEAK: wooden door opening. BETTER: clear opening beyond dense leaves; bright sky through cave opening; "
        "sunlight through gap in clouds — the opening itself is more \"door\" than the physical object.\n"
        "- עזרה — WEAK: person helping another. BETTER: small support holding an enormous weight.\n"
        "- בית — WEAK: house exterior. BETTER: bird nest containing eggs (purest visual of home).\n\n"
        "MONTAGE RULE:\n"
        "- sceneVariations must be variations of the extreme visual ESSENCE — not variations of an object.\n"
        "- BAD: door, door, door, door. GOOD: different forms of openings.\n"
        "- BAD: bridge, bridge, bridge, bridge. GOOD: one connection family (e.g. people reaching toward each other).\n\n"
        "ESSENCE EXTREME SELF-CHECK (rewrite if any answer is NO):\n"
        "1) Did I identify the essence?\n"
        "2) Did I find the strongest visual embodiment of the essence?\n"
        "3) Is the embodiment stronger than showing the object itself?\n"
        "4) If the object disappeared completely, would the idea still be understood?\n"
        "5) Are the variations expressions of the essence rather than the object?\n"
        "6) Do all variations belong to the same visual family (side-by-side frame test)?\n\n"
        "GOAL: viewer should not think \"I saw a door\" — viewer should think \"I felt what a door means.\"\n\n"
    )


def _planner_essence_before_object_block() -> str:
    return (
        "ESSENCE BEFORE OBJECT (mandatory — variation_montage_v4):\n"
        "- Physical objects are optional. If essence can be expressed more powerfully WITHOUT the object, prefer non-object.\n"
        "- Do NOT automatically show door, key, bridge, compass, house if the essence shows better without them.\n"
        "- coreVisualIdea must name essence embodiment — REJECT bare object nouns: compass, bridge, key, house, door.\n"
        "- Same physical object in at most ONE variation (when an object appears at all).\n\n"
        "VARIATION EXAMPLES:\n"
        "- דלת / opening: opening beyond leaves; sky through cave mouth; sunlight through cloud gap — no wooden door needed.\n"
        "- מצפן / direction: compass needle once OR none; path fork; boat changing course; arrival at destination.\n"
        "- גשר / connection — visual family \"people reaching toward each other\": friends joining hands, human chain, leaning across table, child reaching parent — NOT rescue + river + dinner + handshake mix.\n\n"
    )


def _planner_visual_family_consistency_block() -> str:
    return (
        "VISUAL FAMILY CONSISTENCY (mandatory — variation_montage_v4):\n"
        "- sceneVariations must not merely express the same idea — they must belong to the SAME visual family.\n"
        "- After coreVisualIdea, choose ONE visual family. Every variation must remain inside that family.\n"
        "- Goal: viewer feels \"I am seeing the same visual idea expressed several different ways\" — one recurring motif.\n"
        "- NOT: three unrelated scenes that happen to express the same abstract idea.\n\n"
        "PROBLEM (reject):\n"
        "- Keyword גשר / idea: connection. Var 1: mountain rescue. Var 2: chain across river. Var 3: dinner table. Var 4: handshake. "
        "Same idea, different visual families — montage feels inconsistent.\n\n"
        "GOOD (קרוב / hug visual family):\n"
        "- friends hugging, family hugging, couple hugging, elderly friends hugging — same family, different variations.\n"
        "- Side-by-side frozen frames must look like one visual family.\n\n"
        "EXAMPLE גשר / human connection:\n"
        "- coreVisualIdea: human connection. Visual family: people reaching toward each other.\n"
        "- GOOD: friends reaching hands together; people forming human chain; people leaning across table; child reaching parent.\n"
        "- BAD: mountain rescue + river chain + dinner table + business handshake.\n\n"
        "MONTAGE FEEL: different expressions of ONE visual motif — NOT dictionary examples of an abstract idea.\n\n"
        "SELF-CHECK: freeze each variation frame side by side — same visual family? If NO, rewrite.\n\n"
        "PRIORITY: 1) Essence  2) Strong embodiment  3) Visual family consistency  4) Interest  "
        "5) Realistic  6) Silent-video  7) Simplicity.\n\n"
    )


def _planner_variation_montage_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        "VARIATION MONTAGE MODE (mandatory — Builder2 variation_montage_v4):\n"
        "- Do NOT generate a single scene. Generate 2-4 very short variations of the same extreme visual ESSENCE.\n"
        f"- Total video duration remains {n} seconds — montage of quick related moments.\n"
        "- Flow: headline → keyword → essence → coreVisualIdea → visual family → sceneVariations → videoPrompt.\n\n"
        "CORE VISUAL IDEA + VISUAL FAMILY:\n"
        "- coreVisualIdea = strongest extreme visual embodiment of the keyword's essence.\n"
        "- Then choose ONE visual family; all sceneVariations must stay inside that family.\n"
        "- Examples: קרוב→maximum connection/hug; דלת→opening/passage; עזרה→small support under enormous weight; "
        "בית→protected belonging/nest; גשר→connection; מצפן→finding direction.\n\n"
        "SCENE VARIATIONS (sceneVariations array, 2-4 items):\n"
        "- Each item is a variation within the SAME visual family — same motif, different subjects/contexts.\n"
        "- Must feel like קרוב→hug→friends/family/couple/elderly hugging — NOT mixed rescue/table/handshake families.\n"
        "- NO story, NO plot progression. Same physical object in at most ONE variation.\n\n"
        "EXAMPLE keyword גשר / visual family: people reaching toward each other:\n"
        "1) friends reaching hands together  2) people forming a human chain  "
        "3) people leaning across a table  4) child reaching toward parent\n\n"
        "EXAMPLE keyword קרוב / maximum connection:\n"
        "1) elderly friends hugging  2) young couple hugging  3) parent and child hugging  4) friends greeting with a hug\n\n"
        "EXAMPLE keyword דלת / opening:\n"
        "1) clear opening visible beyond dense leaves  2) bright sky through cave opening  "
        "3) sunlight visible through gap in clouds  4) light flooding through a dark passage\n\n"
        "VIDEO PROMPT:\n"
        f"- Describe a {n}-second montage of 2-4 variations within one visual family.\n"
        "- Goal: one recurring visual motif — viewer remembers the motif, not scattered dictionary scenes.\n"
        "- Headline overlay at end (downstream) — do NOT burn headline into videoPrompt.\n\n"
    )


def _planner_headline_rules_block() -> str:
    return (
        "HEADLINE RULES:\n"
        "- The headline is the direct expression of the primary advertising advantage implied by product name + description.\n"
        "- Prefer headlines that contain a single strong metaphorical word (e.g. close, bridge, door, path, heart, home, key, step, light, connection).\n"
        "- Avoid literal industry/category words whenever possible (e.g. advertising, marketing, digital, campaign, story, service, strategy).\n"
        "- Remainder only — no product name inside headline. Up to 7 words.\n\n"
        "headlineCoreKeyword RULES:\n"
        "- Exactly one word from the headline — a single standalone semantic word.\n"
        "- The keyword MUST preserve its intended meaning when completely isolated from the rest of the headline.\n"
        "- REJECT entire headline if meaning depends on a fixed phrase, idiom, or collocation — even when a keyword token looks valid.\n"
        "- SELF-CHECK after keyword selection: remove every other word; ask: "
        '"Does the same core advertising idea still exist?" If not → reject headline and generate a new one.\n'
        "- Must support a vivid universal everyday human association scene.\n"
        "- FORBIDDEN as keyword: advertising, marketing, digital, campaign, story, service, strategy (and Hebrew equivalents).\n"
        "- ACCEPT examples: קרוב, דלת, דרך (standalone meaningful).\n"
        '- REJECT headline "שופך אור על המסר שלך" (phrase "שופך אור").\n'
        '- REJECT headline "מגיע ישר ללב" (phrase "ישר ללב", not standalone "לב").\n'
        '- REJECT headline "שם אותך על המפה" (phrase "על המפה").\n'
        '- REJECT headline "פותח דלת להזדמנויות" (idiomatic phrase, not standalone "דלת").\n\n'
    )


def _planner_headline_phrase_dependency_block() -> str:
    return (
        "HEADLINE PHRASE DEPENDENCY RULE (mandatory):\n"
        "- Reject headlines where the intended meaning depends on a fixed phrase, collocation, idiom, or expression.\n"
        "- The keyword must carry the core advertising idea independently — not smuggled in via a multi-word phrase.\n"
        "- After selecting headlineCoreKeyword, remove every other word and verify the same core idea remains.\n"
        "- If not, reject the headline and generate a new headline.\n\n"
        "REJECT HEADLINE EXAMPLES:\n"
        '- "שופך אור על המסר שלך" — meaning from "שופך אור".\n'
        '- "מגיע ישר ללב" — meaning from "ישר ללב", not standalone "לב".\n'
        '- "שם אותך על המפה" — meaning from "על המפה".\n'
        '- "פותח דלת להזדמנויות" — meaning primarily from the idiomatic phrase.\n\n'
    )


def _planner_standalone_keyword_block() -> str:
    return (
        "STANDALONE KEYWORD RULE (mandatory for coreVisualIdea + sceneVariations):\n"
        "- headlineCoreKeyword must be a single standalone semantic word that defines semantic territory for scene search.\n"
        "- The scene must be generated from the standalone keyword itself — NEVER from a phrase containing the keyword.\n"
        "- Reject any candidate keyword whose meaning only works inside a larger phrase, expression, idiom, or collocation.\n\n"
        "ACCEPT / REJECT EXAMPLES:\n"
        '- "הכי קרוב למשרד פרסום" → ACCEPT "קרוב" (isolated meaning preserved).\n'
        '- "פותח לך דלת לקהל הנכון" → ACCEPT "דלת" (isolated meaning preserved).\n'
        '- "הדרך הקצרה לקהל שלך" → ACCEPT "דרך" (isolated meaning preserved).\n'
        '- "שופך אור על המסר שלך" → REJECT headline (phrase "שופך אור").\n'
        '- "מגיע ישר ללב" → REJECT headline (phrase "ישר ללב").\n\n'
        "If no headline passes phrase + standalone checks, rewrite the headline — do not force a phrase-dependent keyword.\n\n"
    )


def _planner_strict_keyword_isolation_block() -> str:
    return (
        "STRICT KEYWORD ISOLATION RULE (mandatory for coreVisualIdea + sceneVariations + videoPrompt):\n"
        "- coreVisualIdea and every sceneVariation must come from headlineCoreKeyword ONLY.\n"
        "- IGNORE every other word in the headline when creating the scene.\n"
        "- FORBIDDEN: building the scene from two headline words or from a headline phrase containing the keyword.\n"
        "- SELF-CHECK: remove the full headline; look only at headlineCoreKeyword; ask: "
        '"Would I choose the same scene from this single word alone?" If not → reject scene and recreate from keyword only.\n\n'
        "EXAMPLES (keyword territory — stay isolated from other headline words):\n"
        '- Keyword "מוביל" — BAD: person leading someone on a path (uses another headline word). '
        "GOOD: one person confidently walking first in front of others, visibly leading.\n"
        '- Keyword "דרך" — WEAK/literal fallback: person walking on a path. '
        "STRONGER: brisk fitness walk in stylish everyday clothes (not sportswear) — still movement/דרך territory.\n"
        '- Keyword "גשר" — WEAK/object: person crosses bridge repeatedly. '
        "STRONG: connection → helping hand across gap, groups meeting, shared table — bridge object at most once.\n\n"
    )


def _planner_gender_consistency_block() -> str:
    return (
        "GENDER CONSISTENCY RULE (mandatory for sceneVariations + videoPrompt):\n"
        "- Main human subject must NOT contradict grammatical gender implied by product_name, product_description, or headline.\n"
        "- When gender is unclear or not essential: use neutral wording — a person, an adult, someone, a human figure.\n"
        "- Do NOT randomly choose woman/man unless clearly required.\n"
        "- Hebrew: if product/headline implies masculine → use אדם, גבר, or neutral English \"a person\"; "
        "do NOT use woman/girl/female/אישה/בחורה.\n"
        "- Hebrew: if product/headline implies feminine → use אישה or neutral English \"a person\"; "
        "do NOT use man/male/גבר.\n"
        "- Preferred default: neutral subject unless gender is clearly required.\n"
        '- Example product "אורי לב" + masculine headline → BAD: woman as main subject representing the product.\n\n'
    )


def _planner_final_checklist_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        "FINAL CHECKLIST (before returning JSON):\n"
        "1) headlineCoreKeyword is exactly one standalone word.\n"
        "2) coreVisualIdea is the extreme visual embodiment of the keyword's essence — not literal meaning, not bare object.\n"
        "3) sceneVariations has 2-4 items within ONE visual family of that essence — not other headline words.\n"
        "4) side-by-side frame test: all variations feel like the same visual family.\n"
        "5) variations are essence-expressions — not object repetitions; same object ≤ once.\n"
        "6) variations are independent — no story arc or plot progression.\n"
        f"7) videoPrompt is a {n}-second montage of one visual-family motif; no headline phrase leakage.\n"
        "8) main subject gender does not contradict product/headline when gendered subjects appear.\n"
        "9) silent-video verifiable; realistic; same essence and family across variations.\n"
        "10) ESSENCE EXTREME + VISUAL FAMILY self-checks passed.\n\n"
    )


def _planner_interest_first_block() -> str:
    return (
        "INTEREST FIRST + ESSENCE EXTREME (primary coreVisualIdea + variation selection — mandatory):\n"
        "- Among valid essence embodiments, prefer the most extreme, memorable, visually powerful form.\n"
        "- NOT literal meaning, NOT obvious object shots, NOT mechanical keyword illustration.\n"
        "- coreVisualIdea and every variation must still be: realistic, simple, visually understandable, "
        "physically possible, silent-video compatible.\n"
        "- Valid montage: viewer feels the MEANING of the keyword — not \"I saw an object.\"\n\n"
        "KEYWORD ROLE:\n"
        "- headlineCoreKeyword defines territory — find ESSENCE first, then extreme embodiment.\n"
        "- Ask: \"What is the most extreme visual expression of this essence?\" — NOT \"What does the word mean?\"\n"
        "- Stay keyword-isolated: essence comes from the keyword alone, never from other headline words.\n\n"
        "ESSENCE EXTREME (not literal / not object-only):\n"
        '- "קרוב" — literal: standing near. EXTREME: hug / maximum connection.\n'
        '- "דלת" — literal: wooden door. EXTREME: openings — light beyond leaves, sky through cave, sun through cloud gap.\n'
        '- "עזרה" — literal: person helping. EXTREME: small support holding enormous weight.\n'
        '- "בית" — literal: house exterior. EXTREME: nest with eggs / protected belonging.\n'
        '- "גשר" — BAD: rescue + river chain + dinner + handshake (mixed families). '
        'GOOD: one family "people reaching toward each other" — joining hands, human chain, lean across table, child to parent.\n\n'
        "VISUAL FAMILY TEST: freeze frames side by side — same family? If not, rewrite.\n\n"
        "IMPORTANT LIMIT — extreme must NOT come from:\n"
        "fantasy, surrealism, impossible events, dream logic, visual tricks, symbolism requiring explanation.\n\n"
        "FINAL PRIORITY ORDER:\n"
        "1) Essence  2) Strong visual embodiment  3) Visual family consistency  4) Interesting  "
        "5) Realistic  6) Silent-video verifiable  7) Simple  8) Non-object when stronger.\n\n"
    )


def _planner_scene_association_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        "SCENE ASSOCIATION RULE (mandatory for coreVisualIdea + sceneVariations + videoPrompt):\n"
        "- Convert headlineCoreKeyword to ESSENCE, extreme embodiment, then ONE visual family (see ESSENCE EXTREME + VISUAL FAMILY rules).\n"
        "- coreVisualIdea comes ONLY from keyword territory — never from a multi-word phrase in the headline.\n"
        "- Do NOT default to literal dictionary meaning or the physical object.\n"
        "- sceneVariations = 2-4 variations within ONE visual family of that essence — not mixed families.\n"
        f"- Instantly recognizable and emotionally understandable within {n} seconds.\n\n"
        "BAD vs GOOD (keyword → essence → extreme embodiment → variations):\n"
        '- "קרוב" — BAD: standing near. GOOD: maximum connection → hug variations.\n'
        '- "דלת" — BAD: wooden door ×4. GOOD: opening → leaves parting to light, cave mouth to sky, cloud gap to sun.\n'
        '- "עזרה" — BAD: generic helping. GOOD: small support under enormous weight; hand steadying heavy load.\n'
        '- "בית" — BAD: house exterior ×4. GOOD: protected belonging → nest with eggs, bird returning, sheltered young.\n'
        '- "גשר" — BAD: rescue + river + dinner + handshake (mixed families). '
        'GOOD: visual family "people reaching" — hands together, human chain, lean across table, child to parent.\n\n'
        "When multiple valid embodiments exist, apply ESSENCE EXTREME + VISUAL FAMILY CONSISTENCY.\n"
        "Each variation is an independent expression — NO story arc, NO plot progression.\n\n"
    )


def _planner_video_prompt_simplicity_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        "VIDEO PROMPT SIMPLICITY RULE (mandatory for videoPrompt only):\n"
        f"- Execute sceneConcept as the SIMPLEST possible {n}-second stock-video moment — interesting but not complex.\n"
        "- Feel like clean stock footage — NOT a short film, NOT a plot, NOT an ad storyboard.\n"
        "- One main human subject whenever possible. One clear action. One location. Single continuous moment.\n"
        "- Interest comes from style, context, or human detail — NOT from extra story beats or choreography.\n"
        "- NO secondary story beat, NO plot resolution, NO complex choreography.\n"
        "- NO meeting another person unless the keyword absolutely requires it (e.g. hugging needs two people).\n"
        "- NO handshake unless the keyword itself requires it.\n"
        "- NO added business symbolism, NO \"they continue together\", NO \"and then\" sequences.\n\n"
        "PREFER (videoPrompt patterns — choose interesting variant when valid):\n"
        "- A person crosses a bridge in an elegant everyday setting.\n"
        "- A person opens a door and enters a welcoming home.\n"
        "- A person does a confident brisk walk in stylish everyday clothes.\n"
        "- Two people hug.\n\n"
        "KEYWORD \"גשר\" / bridge — videoPrompt examples:\n"
        '- GOOD: "A person crosses a small footbridge in soft morning light. Natural movement. Simple cinematic shot. No text."\n'
        '- WEAK/literal: "A person crosses a bridge."\n'
        '- BAD: "A person crosses a bridge, meets someone halfway, shakes hands, and they walk together."\n\n'
        "Strip sceneConcept to one visual essence — maximize interest within one simple moment.\n\n"
    )


def _planner_silent_video_verifiability_block() -> str:
    return (
        "SILENT VIDEO VERIFIABILITY RULE (mandatory for coreVisualIdea + sceneVariations + videoPrompt):\n"
        "- Builder2 videos are SILENT. Every variation moment must be visually verifiable with NO sound.\n"
        "- If a muted viewer cannot tell the action occurred, reject that variation and choose a visual alternative.\n"
        "- A muted viewer must reach the same interpretation as a hearing viewer for each moment.\n\n"
        "REJECT (audio-dependent):\n"
        "- shouting, screaming, whispering, singing, loudspeaker/PA, music playing loudly, alarms, bell ringing, "
        "crowd cheering, barking as the key event, engine roaring, any event whose meaning depends on audio.\n\n"
        "ALLOW (visually provable):\n"
        "- opening a door, crossing a bridge, hugging, handshake, entering a home, walking a path, "
        "turning a volume knob, turning on a light, moving a lever, raising a flag, lifting an object, helping another person.\n\n"
        "KEYWORD \"מגביר\" / amplify example:\n"
        "- BAD: woman shouting loudly (loudness is not visible).\n"
        "- GOOD: a hand rotates a volume knob from low to high (increase is visually observable).\n\n"
        "FINAL CHECK — plan is valid only if:\n"
        "1) keyword stands independently  2) headline does not depend on a fixed phrase  "
        "3) coreVisualIdea comes from the keyword itself  4) each variation is visually understandable  "
        "5) each variation is visually provable without sound  6) muted viewer reaches the same interpretation.\n\n"
    )


def _planner_keyword_scene_flow_block() -> str:
    n = _builder2_video_duration_seconds()
    return (
        "BUILDER2 VARIATION MONTAGE FLOW (mandatory; do not narrate in JSON):\n"
        "STEP 1 — Read product_name and product_description.\n"
        "STEP 2 — headline (see HEADLINE RULES).\n"
        "STEP 3 — headlineCoreKeyword (see STANDALONE KEYWORD RULE).\n"
        "STEP 4 — Find essence of headlineCoreKeyword; set coreVisualIdea to its strongest extreme visual embodiment.\n"
        "STEP 5 — Choose ONE visual family for coreVisualIdea; sceneVariations: 2-4 variations inside that family (object ≤ once).\n"
        f"STEP 6 — videoPrompt: {n}-second montage of that visual-family motif.\n\n"
        + _planner_duration_instruction_block()
        + _planner_variation_montage_block()
        + _planner_essence_extreme_block()
        + _planner_essence_before_object_block()
        + _planner_visual_family_consistency_block()
        + _planner_headline_rules_block()
        + _planner_headline_phrase_dependency_block()
        + _planner_standalone_keyword_block()
        + _planner_strict_keyword_isolation_block()
        + _planner_interest_first_block()
        + _planner_scene_association_block()
        + _planner_silent_video_verifiability_block()
        + _planner_gender_consistency_block()
        + _planner_final_checklist_block()
        + "MONTAGE RULES (sceneVariations + videoPrompt):\n"
        "- Realistic, simple, physically possible; visually verifiable without sound.\n"
        "- Same essence AND same visual family across all variations.\n"
        "- Variations = different expressions of one visual motif — not dictionary examples.\n"
        "FORBIDDEN: mixed visual families, literal keyword illustration, object-only montages, surreal, story arcs, headline burn-in.\n\n"
    )


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    n = _builder2_video_duration_seconds()
    return (
        f"ACE Builder2 video planning — variation montage ({_VIDEO_PLAN_SCHEMA_VERSION}). "
        f"Language {lang_name} ({lang}). "
        "product → headline → headlineCoreKeyword → essence → coreVisualIdea (extreme embodiment) → sceneVariations → videoPrompt. "
        f"{n}-second montage of 2-4 variations within one visual family; essence + family consistency; object ≤ once; all existing rules apply. "
        'Planner refusal: {"planningFailure":"planning_failed_invalid_plan"}'
    )


def _log_output_preview(raw: str, prefix: str = "VIDEO_PLAN output_preview") -> None:
    if not raw:
        return
    preview = raw.strip().replace("\n", " ")[:_LOG_PREVIEW_CHARS]
    logger.info("%s len=%s preview=%r", prefix, len(raw), preview)


def _repair_loose_json(s: str) -> str:
    """Remove trailing commas and stray BOM that often break json.loads."""
    t = s.strip()
    if t.startswith("\ufeff"):
        t = t.lstrip("\ufeff")
    # Trailing commas before } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def _strip_markdown_fences(text: str) -> str:
    """Remove ``` or ```json fences; tolerate missing closing fence."""
    t = text.strip()
    if t.startswith("\ufeff"):
        t = t.lstrip("\ufeff")
    lower_start = t[:12].lower()
    if lower_start.startswith("```json"):
        t = t[7:].lstrip()
    elif t.startswith("```"):
        t = t[3:].lstrip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    if t.endswith("```"):
        t = t[: -3].rstrip()
    return t.strip()


def _parse_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    text = _strip_markdown_fences(raw)
    # Drop leading prose before first {
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    text = _repair_loose_json(text)

    def _try_load(s: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(s)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    got = _try_load(text)
    if got is not None:
        return got

    # Brace-balanced slice from first { to last }
    end = text.rfind("}")
    start = text.find("{")
    if start >= 0 and end > start:
        slice_ = _repair_loose_json(text[start : end + 1])
        got = _try_load(slice_)
        if got is not None:
            return got

    return None


def _extract_responses_output_text(response: Any) -> str:
    """
    Prefer output_text; if empty, concatenate output_text parts from response.output (reasoning models).
    """
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: List[str] = []
    for block in getattr(response, "output", None) or []:
        contents = getattr(block, "content", None)
        if contents is None and isinstance(block, dict):
            contents = block.get("content")
        if not contents:
            continue
        for c in contents:
            ct = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
            if ct == "output_text":
                txt = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                if txt:
                    chunks.append(str(txt))
    return "".join(chunks).strip()


def _word_limit(s: str, max_words: int) -> str:
    words = (s or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


_VIDEO_PLAN_SCHEMA_VERSION = "variation_montage_v4"
_SCENE_VARIATIONS_MIN = 2
_SCENE_VARIATIONS_MAX = 4

_PLANNER_SELF_FAILURE_CODES: FrozenSet[str] = frozenset(
    {"planning_failed_invalid_plan", "planning_failed_no_valid_scene"}
)


def _build_scene_plan_repair_input(
    *,
    base_attempt_input: str,
    product_name: str,
    product_description: str,
    previous_plan: Dict[str, Any],
    reason: str,
) -> str:
    return (
        f"{base_attempt_input}\n\n"
        f"REPAIR REQUEST (one retry): The previous plan failed validation ({reason}).\n"
        "Keep the same product name and product description.\n"
        "Fix headline, headlineCoreKeyword, coreVisualIdea, sceneVariations, and videoPrompt to satisfy all rules.\n"
        "Find keyword ESSENCE first; coreVisualIdea must be its strongest extreme visual embodiment — not literal meaning, not bare object.\n"
        "sceneVariations must be 2-4 variations within ONE visual family of the essence — not mixed families (e.g. no rescue + table + handshake mix); same object in at most ONE variation.\n"
        "headlineCoreKeyword must be standalone — reject phrase-dependent headlines/keywords; rewrite headline if needed.\n"
        "coreVisualIdea and all variations must come from headlineCoreKeyword ONLY — ignore all other headline words.\n"
        "Use gender-neutral subject (a person) unless product/headline clearly requires gender; no gender contradiction.\n"
        "For sceneVariations + videoPrompt: visually provable without sound; muted viewer must understand each moment.\n"
        f"For videoPrompt: {_builder2_video_duration_seconds()}-second montage of the variations — short, clear, realistic, visually distinct moments.\n"
        "Return the same required JSON shape only.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        "Previous invalid plan (for correction):\n"
        f"{json.dumps(previous_plan, ensure_ascii=False)}\n"
    )


def _build_keyword_scene_fallback_plan(
    *,
    product_name: str,
    content_language: str,
) -> Dict[str, Any]:
    """Deterministic realistic human-scene fallback when planner repair fails."""
    lang = normalize_video_content_language(content_language)
    pn = (product_name or "").strip() or "ACE Product"
    headline = "הכי קרוב למשרד פרסום" if lang == "he" else "Always One Step Ahead"
    keyword = "קרוב" if lang == "he" else "Ahead"
    core_visual = "embrace" if lang == "en" else "חיבוק"
    variations = (
        [
            "elderly friends hugging",
            "young couple hugging",
            "parent and child hugging",
        ]
        if lang == "en"
        else [
            "חברים מבוגרים מתחבקים",
            "זוג צעיר מתחבק",
            "הורה וילד מתחבקים",
        ]
    )
    video_prompt = (
        f"A {_builder2_video_duration_seconds()}-second realistic montage of three short hug moments: elderly friends hugging; "
        "a young couple hugging; a parent and child hugging. Quick cuts, same embrace idea, "
        "natural movement. Simple cinematic shots. No text."
    )
    scene_joined = " | ".join(variations)
    return {
        "productNameResolved": pn,
        "headline": headline,
        "headlineCoreKeyword": keyword,
        "coreVisualIdea": core_visual,
        "sceneVariations": variations,
        "sceneConcept": scene_joined,
        "videoPrompt": video_prompt,
        "language": lang,
        "planInferenceMode": "deterministic_variation_montage_v4_fallback",
    }


def _runway_language_visual_constraints(plan: Dict[str, Any]) -> str:
    """Short language-consistent cue for the video model (no headline burn-in)."""
    lang = str(plan.get("language") or "").strip().lower()
    if lang == "he":
        return (
            "LANGUAGE-CONSISTENT VISUALS: If a setting appears, keep backgrounds generic; "
            "do not foreground English-only storefront lettering or foreign-script signage as the hero element."
        )
    return (
        "LANGUAGE-CONSISTENT VISUALS: If a setting appears, keep backgrounds generic; "
        "do not foreground non-English street or storefront lettering as the hero element."
    )


_RUNWAY_STYLE_TAIL = (
    "No logos, no packaging typography, no on-screen words, no headline burn-in. Single clean commercial look."
)

_RUNWAY_SCENE_TAIL_MARKERS: Tuple[str, ...] = (
    "Scene (follow exactly):",
    "Scene continuation (follow exactly):",
    "Montage (follow exactly):",
    "Montage continuation (follow exactly):",
    "Scene:",
)


def _runway_variation_montage_camera_focus() -> Tuple[str, str]:
    """Configurable-duration montage of related visual variations (Builder2)."""
    n = _builder2_video_duration_seconds()
    return (
        f"MANDATORY: one {n}-second realistic montage of 2-4 very short moments within the SAME visual family and extreme essence. "
        "Natural pacing from action start through brief development to a clear visual anchor or resolution; no dead seconds at the end. "
        "Each moment is a variation of one recurring visual motif — not scattered dictionary scenes. "
        "Quick cuts between family-consistent variations. Each moment clear and readable. "
        "NO story arc, NO cause-and-effect, NO plot progression — independent expressions of the same idea. "
        "No dialogue or audio required; no on-screen text. "
        "No surreal motion, no fantasy, no impossible physics.",
        f"variation_montage_{n}s",
    )


def _runway_human_scene_camera_focus() -> Tuple[str, str]:
    """Stable cinematic framing for realistic human scenes."""
    return (
        "MANDATORY: single continuous realistic human scene. Stable cinematic camera with gentle natural movement; "
        "subjects stay readable. No surreal motion, no morphing, no impossible physics, no cuts.",
        "human_scene_stable",
    )


# snake_case / alternate keys from some models → camelCase
_PLAN_KEY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("product_name_resolved", "productNameResolved"),
    ("headline_core_keyword", "headlineCoreKeyword"),
    ("core_visual_idea", "coreVisualIdea"),
    ("scene_concept", "sceneConcept"),
    ("video_prompt", "videoPrompt"),
    ("headline_text", "headline"),
    ("video_prompt_core", "videoPrompt"),
)


def _coerce_scene_variations(raw: Any) -> List[str]:
    """Normalize sceneVariations to 0..N non-empty strings."""
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        lines = re.split(r"\n+|\s*;\s*", text)
        cleaned: List[str] = []
        for line in lines:
            line = re.sub(r"^\s*\d+[\).\:-]\s*", "", line.strip())
            if line:
                cleaned.append(line)
        return cleaned
    return []


def _coerce_plan_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill camelCase keys from snake_case duplicates when the canonical key is missing or empty."""
    out = dict(data)
    for alt, canon in _PLAN_KEY_ALIASES:
        cur = out.get(canon)
        empty = cur is None or (isinstance(cur, str) and not cur.strip())
        alt_val = out.get(alt)
        if empty and alt_val is not None and str(alt_val).strip():
            out[canon] = alt_val
    if not out.get("sceneVariations"):
        coerced = _coerce_scene_variations(out.get("sceneVariations"))
        if not coerced:
            coerced = _coerce_scene_variations(out.get("scene_variations"))
        if coerced:
            out["sceneVariations"] = coerced
    return out


def _plan_scene_variations_list(plan: Dict[str, Any]) -> List[str]:
    variations = _coerce_scene_variations(plan.get("sceneVariations"))
    if variations:
        return variations
    scene = (plan.get("sceneConcept") or "").strip()
    if not scene:
        return []
    if " | " in scene:
        return [part.strip() for part in scene.split(" | ") if part.strip()]
    return [scene]


def validate_and_normalize_plan(
    data: Dict[str, Any],
    *,
    planner_deadline_monotonic: Optional[float] = None,
    product_name: str = "",
    product_description: str = "",
    content_language: str = "he",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Builder2 keyword-scene planning — structural validation only.
    Returns (plan, None) or (None, reason_code) for fail-fast logging.
    """
    logger.info("VIDEO_PLAN_SERVER_CREATIVE_GATE=disabled")
    logger.info("VIDEO_PLAN_SERVER_VALIDATION_SCOPE=keyword_scene_structural")

    if not data:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=no_payload")
        return None, "planning_failed_incomplete_plan"

    data = _coerce_plan_keys(data)

    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)

    pn = (data.get("productNameResolved") or "").strip()
    headline_rem = (data.get("headline") or "").strip()
    core_kw = (data.get("headlineCoreKeyword") or "").strip()
    core_visual = (data.get("coreVisualIdea") or "").strip()
    variations = _coerce_scene_variations(data.get("sceneVariations"))
    video_prompt = (data.get("videoPrompt") or "").strip()
    lang_raw = str(data.get("language") or "").strip()
    if not lang_raw:
        lang_raw = normalize_video_content_language(content_language)

    if not pn:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_product_name")
        return None, "planning_failed_incomplete_plan"
    if not headline_rem:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_headline")
        return None, "planning_failed_incomplete_plan"
    if not core_kw:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_headline_core_keyword")
        return None, "planning_failed_incomplete_plan"
    if not core_visual:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_core_visual_idea")
        return None, "planning_failed_incomplete_plan"
    if len(variations) < _SCENE_VARIATIONS_MIN or len(variations) > _SCENE_VARIATIONS_MAX:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=invalid_scene_variations_count count=%s",
            len(variations),
        )
        return None, "planning_failed_invalid_variations"
    if not video_prompt:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_video_prompt")
        return None, "planning_failed_incomplete_plan"

    scene = " | ".join(variations)

    if planner_deadline_monotonic is not None and time.monotonic() >= planner_deadline_monotonic:
        logger.error("VIDEO_PLAN_DEADLINE_EXCEEDED stage=validate")
        raise VideoPlanningTimeoutError()

    if _headline_remainder_word_count(headline_rem) > _MAX_HEADLINE_REMAINDER_WORDS:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_too_long")
        return None, "planning_failed_headline_too_long"

    headline_phrase_rule = _headline_depends_on_fixed_phrase(headline_rem)
    if headline_phrase_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=phrase_dependent_headline rule=%s",
            headline_phrase_rule,
        )
        return None, "planning_failed_phrase_dependent_headline"

    kw_tokens = [t for t in core_kw.split() if t]
    if len(kw_tokens) != 1:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=invalid_headline_core_keyword_count")
        return None, "planning_failed_invalid_keyword"
    if _normalize_keyword_token(kw_tokens[0]) in _KEYWORD_FILLER_WORDS:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_core_keyword_is_filler")
        return None, "planning_failed_invalid_keyword"
    if _is_weak_industry_keyword(kw_tokens[0]):
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_core_keyword_is_weak_industry")
        return None, "planning_failed_weak_industry_keyword"
    phrase_rule = _keyword_depends_on_headline_phrase(headline_rem, core_kw)
    if phrase_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=phrase_dependent_keyword rule=%s",
            phrase_rule,
        )
        return None, "planning_failed_phrase_dependent_keyword"
    if not _headline_contains_core_keyword(headline_rem, core_kw):
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_core_keyword_not_in_headline")
        return None, "planning_failed_invalid_keyword"

    scene_blob = "\n".join([core_visual, scene, video_prompt, *variations])
    bad_scene = scene_fields_imply_forbidden_surrealism(scene_blob)
    if bad_scene:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=forbidden_surreal_scene rule=%s",
            bad_scene,
        )
        return None, "planning_failed_surreal_scene"

    bad_sound = scene_fields_require_sound_to_verify(scene_blob)
    if bad_sound:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=sound_dependent_scene rule=%s",
            bad_sound,
        )
        return None, "planning_failed_sound_dependent_scene"

    gender_ctx_name = (product_name or "").strip() or pn
    leak_rule = _scene_leaks_non_keyword_headline_word(headline_rem, core_kw, scene_blob)
    if leak_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=keyword_not_isolated rule=%s",
            leak_rule,
        )
        return None, "planning_failed_keyword_not_isolated"

    gender_rule = _scene_gender_mismatch(
        gender_ctx_name, (product_description or "").strip(), headline_rem, scene_blob
    )
    if gender_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=gender_mismatch rule=%s",
            gender_rule,
        )
        return None, "planning_failed_gender_mismatch"

    bare_object_rule = _core_visual_is_bare_object(core_visual)
    if bare_object_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=object_not_idea rule=%s",
            bare_object_rule,
        )
        return None, "planning_failed_object_not_idea"

    object_rep_rule = _variations_object_repetition_violation(core_kw, variations)
    if object_rep_rule:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=object_repetition rule=%s",
            object_rep_rule,
        )
        return None, "planning_failed_object_repetition"

    pn_for_headline = (product_name or "").strip() or pn
    headline_full = _assemble_headline_full(pn_for_headline, headline_rem)
    opening_fd = (variations[0] if variations else core_visual)[:400]

    logger.info('VIDEO_PLAN_HEADLINE="%s"', headline_rem[:260])
    logger.info('VIDEO_PLAN_CORE_KEYWORD="%s"', core_kw[:120])
    logger.info('VIDEO_PLAN_CORE_VISUAL_IDEA="%s"', core_visual[:200])
    logger.info("VIDEO_PLAN_SCENE_VARIATIONS count=%s", len(variations))
    logger.info('VIDEO_PLAN_SCENE_CONCEPT="%s"', scene[:260])

    return {
        "productNameResolved": pn,
        "advertisingPromise": headline_rem,
        "headline": headline_rem,
        "headlineText": headline_full,
        "headlineTextRemainder": headline_rem,
        "headlineCoreKeyword": core_kw,
        "coreVisualIdea": core_visual,
        "sceneVariations": variations,
        "sceneConcept": scene,
        "videoPrompt": video_prompt,
        "videoPromptCore": video_prompt.strip(),
        "language": lang_raw,
        "headlineDecision": "include_product_name",
        "planInferenceMode": str(data.get("planInferenceMode") or "").strip(),
        "openingFrameDescription": opening_fd,
    }, None


def _scene_plan_digest(scene: str, keyword: str) -> str:
    raw = f"{(scene or '').strip()}\n{(keyword or '').strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_video_job_plan_integrity(plan: Dict[str, Any]) -> None:
    """Structured keyword-scene + headline fields for every validated plan (video job trace)."""
    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)
    logger.info(
        'VIDEO_PLAN_INTEGRITY headline="%s"',
        (plan.get("headline") or plan.get("headlineTextRemainder") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY headlineCoreKeyword="%s"',
        (plan.get("headlineCoreKeyword") or "")[:120],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY coreVisualIdea="%s"',
        (plan.get("coreVisualIdea") or "")[:200],
    )
    logger.info(
        "VIDEO_PLAN_INTEGRITY sceneVariations_count=%s",
        len(_plan_scene_variations_list(plan)),
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY sceneConcept="%s"',
        (plan.get("sceneConcept") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY headlineDecision=%s headlineText="%s"',
        plan.get("headlineDecision"),
        (plan.get("headlineText") or "")[:160],
    )
    logger.info(
        'VIDEO_PLAN_OPENING_FRAME="%s"',
        ((plan.get("openingFrameDescription") or "")[:200]),
    )


_RUNWAY_STRUCT_REQUIRED_KEYS: Tuple[str, ...] = (
    "productNameResolved",
    "advertisingPromise",
    "headlineText",
    "headlineCoreKeyword",
    "coreVisualIdea",
    "sceneConcept",
    "videoPromptCore",
    "openingFrameDescription",
)


def video_plan_struct_ok_for_runway(plan: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Structural sanity only (post-canonical); no creative judgment."""
    if not plan:
        return False, "no_plan"
    for k in _RUNWAY_STRUCT_REQUIRED_KEYS:
        if not str(plan.get(k) or "").strip():
            return False, f"missing_{k}"
    variations = _plan_scene_variations_list(plan)
    if len(variations) < _SCENE_VARIATIONS_MIN or len(variations) > _SCENE_VARIATIONS_MAX:
        return False, "invalid_sceneVariations"
    return True, ""


def log_plan_summary(plan: Dict[str, Any]) -> None:
    """Concise server-side log of the chosen plan (no full prompts, no secrets)."""
    logger.info(
        'VIDEO_PLAN productNameResolved="%s"',
        (plan.get("productNameResolved") or "")[:120],
    )
    logger.info(
        "VIDEO_PLAN_SUMMARY headlineCoreKeyword=%s language=%s",
        plan.get("headlineCoreKeyword"),
        plan.get("language"),
    )
    logger.info(
        'VIDEO_PLAN_CORE_VISUAL_IDEA="%s"',
        (plan.get("coreVisualIdea") or "")[:200],
    )
    logger.info(
        "VIDEO_PLAN_SCENE_VARIATIONS count=%s",
        len(_plan_scene_variations_list(plan)),
    )
    logger.info(
        'VIDEO_PLAN_SCENE_CONCEPT="%s"',
        (plan.get("sceneConcept") or "")[:260],
    )
    logger.info(
        "VIDEO_PLAN scene_digest=%s",
        _scene_plan_digest(
            str(plan.get("sceneConcept") or ""),
            str(plan.get("headlineCoreKeyword") or ""),
        ),
    )


def _log_video_plan_post_ok_diagnostics(plan: Dict[str, Any]) -> None:
    """Post-success creative diagnostics for retrospective ad-concept review (logging only)."""
    product_resolved = (plan.get("productNameResolved") or "").strip()
    headline_full = (plan.get("headlineText") or "").strip()
    headline_remainder = (plan.get("headlineTextRemainder") or "").strip()
    if not headline_remainder and product_resolved and headline_full.startswith(product_resolved + " "):
        headline_remainder = headline_full[len(product_resolved) + 1 :].strip()

    logger.info("VIDEO_PLAN_DIAG productNameResolved=%s", product_resolved[:200])
    logger.info(
        "VIDEO_PLAN_DIAG headlineCoreKeyword=%s",
        (plan.get("headlineCoreKeyword") or "")[:120],
    )
    logger.info(
        "VIDEO_PLAN_DIAG coreVisualIdea=%s",
        (plan.get("coreVisualIdea") or "")[:200],
    )
    logger.info(
        "VIDEO_PLAN_DIAG sceneVariations_count=%s",
        len(_plan_scene_variations_list(plan)),
    )
    logger.info("VIDEO_PLAN_DIAG sceneConcept=%s", (plan.get("sceneConcept") or "")[:300])
    logger.info("VIDEO_PLAN_DIAG headlineText=%s", headline_full[:300])
    logger.info(
        "VIDEO_PLAN_DIAG headline_remainder=%s",
        (plan.get("headline") or headline_remainder)[:300],
    )
    logger.info(
        "VIDEO_PLAN_DIAG videoPrompt=%s",
        (plan.get("videoPrompt") or plan.get("videoPromptCore") or "")[:400],
    )


def _video_plan_reasoning_payload() -> dict:
    from engine.builder2_reasoning_config import build_builder2_reasoning_payload

    return build_builder2_reasoning_payload()


def _return_plan_with_promise_persist(
    plan: Optional[Dict[str, Any]],
    *,
    product_name: str,
    product_description: str,
    session_id: str,
    fallback_used: bool = False,
) -> Optional[Dict[str, Any]]:
    if fallback_used:
        ph = compute_product_hash(product_name, product_description)
        increment_promise_stat(
            ph,
            "fallback_used_count",
            1,
            product_name=product_name,
            product_description=product_description,
        )
    # advertisingPromise is persisted only after a successful video generation (see runway_video).
    return plan


def _planner_deadline_guard(
    deadline_monotonic: Optional[float], *, stage: str, has_valid_plan: bool = False
) -> None:
    if deadline_monotonic is None:
        return
    now = time.monotonic()
    if now < deadline_monotonic:
        return
    logger.error(
        "VIDEO_PLAN_DEADLINE_EXCEEDED stage=%s has_valid_plan=%s",
        stage,
        str(has_valid_plan).lower(),
    )
    raise VideoPlanningTimeoutError()


def _fetch_video_plan_o3_sync(
    product_name: str,
    product_description: str,
    content_language: str = "he",
    *,
    deadline_monotonic: Optional[float] = None,
    session_id: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    One o3 planner call; structural normalization in validate_and_normalize_plan.
    Returns (plan, "") on success, or (None, reason_code).
    """
    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)
    logger.info("VIDEO_PLAN_SEARCH_ORDER=variation_montage_v4")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_PLAN_FAIL_NO_API_KEY")
        return None, "planning_failed_model_call"

    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise VideoPlanningTimeoutError()

    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    model = _text_model()
    desc_src = (product_description or "").strip()
    desc_limit = _video_plan_planner_description_limit()
    if len(desc_src) > desc_limit:
        desc_for_model = (
            desc_src[:desc_limit].rstrip()
            + "\n…[planner excerpt; full description is unchanged for Runway downstream]"
        )
        desc_truncated = True
    else:
        desc_for_model = desc_src
        desc_truncated = False
    user_block = f"""Product name (may be empty): {product_name or "(empty)"}
Product description:
{desc_for_model}

Language: {lang_name} ({lang}).

{_planner_keyword_scene_flow_block()}
{_json_keys_block()}
"""
    instructions = _build_video_planner_instructions(lang)
    ph = compute_product_hash(product_name, product_description)
    increment_promise_stat(
        ph,
        "recent_generations_count",
        1,
        product_name=product_name,
        product_description=product_description,
    )
    logger.info("AD_PROMISE_MEMORY_LOAD_BEFORE_GENERATION hash=%s", ph)
    history = load_ad_promise_history(product_name, product_description)
    logger.info(
        "VIDEO_PLAN_MEMORY_USED_FOR_DIVERSITY=%s",
        str(bool(history)).lower(),
    )
    forbid_hist = forbidden_promises_for_prompt(history, 3)
    promise_addon = build_promise_diversity_addon(
        forbid_hist,
        angle_seed_for_attempt(0, 0),
    )
    if len(promise_addon) > 1200:
        promise_addon = promise_addon[:1200].rstrip() + "\n…"
    attempt_input = instructions + "\n\n" + user_block + promise_addon
    _t = min(30.0, _VIDEO_PLAN_TIMEOUT)
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(connect=_t, read=_VIDEO_PLAN_TIMEOUT, write=_t, pool=_t),
        max_retries=0,
    )

    logger.info("VIDEO_PLAN_REQUEST_START model=%s", model)
    logger.info("VIDEO_PLAN_REQUEST_TIMEOUT_S=%s", _VIDEO_PLAN_TIMEOUT)
    if deadline_monotonic is not None:
        logger.info(
            "VIDEO_PLAN_OVERALL_DEADLINE_S remaining=%.3f",
            max(0.0, deadline_monotonic - time.monotonic()),
        )

    logger.info("VIDEO_PLAN_PROMPT_PROFILE=short")
    logger.info(
        "VIDEO_PLAN_PLANNER_DESC_CHARS original=%s planner_body=%s truncated=%s",
        len(desc_src),
        len(desc_for_model),
        str(desc_truncated).lower(),
    )
    logger.info("VIDEO_PLAN_PROMPT_LEN=%s", len(attempt_input))
    try:
        response = _responses_create_with_plan_retry(
            client,
            model=model,
            input_text=attempt_input,
            reasoning=_video_plan_reasoning_payload(),
            deadline_monotonic=deadline_monotonic,
            role="video_planning",
            base_call_type="normal",
        )
    except VideoPlanningTimeoutError:
        raise
    except Exception as e:
        err_type = type(e).__name__
        logger.warning(
            "VIDEO_PLAN_FAIL_MODEL_CALL model=%s err_type=%s err=%s",
            model,
            err_type,
            e,
        )
        logger.info("VIDEO_PLAN_RESPONSE_OK=false")
        return None, "planning_failed_model_call"

    try:
        raw = _extract_responses_output_text(response)
        if not raw:
            logger.error("VIDEO_PLAN_FAIL_EMPTY_OUTPUT model=%s", model)
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, "planning_failed_malformed_response"

        _log_output_preview(raw)

        parsed = _parse_json_from_response(raw)
        if not parsed:
            logger.error("VIDEO_PLAN_FAIL_JSON_PARSE model=%s", model)
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, "planning_failed_malformed_response"

        pf_raw = str(parsed.get("planningFailure") or "").strip()
        if pf_raw:
            detail = str(parsed.get("planningFailureDetail") or "").replace('"', "'")[:260]
            code = (
                pf_raw
                if pf_raw in _PLANNER_SELF_FAILURE_CODES
                else "planning_failed_invalid_plan"
            )
            logger.info('VIDEO_PLAN_PLANNER_SELF_REJECT code=%s detail="%s"', code, detail or "(none)")
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, code

        plan, v_err = validate_and_normalize_plan(
            parsed,
            planner_deadline_monotonic=deadline_monotonic,
            product_name=product_name,
            product_description=product_description,
            content_language=content_language,
        )
        if not plan:
            last_v_err = (v_err or "").strip() or "planning_failed_incomplete_plan"
            logger.info("VIDEO_PLAN_REPAIR_REQUESTED reason=%s", last_v_err)
            repair_input = _build_scene_plan_repair_input(
                base_attempt_input=attempt_input,
                product_name=product_name,
                product_description=product_description,
                previous_plan=parsed,
                reason=last_v_err,
            )
            try:
                repair_response = _responses_create_with_plan_retry(
                    client,
                    model=model,
                    input_text=repair_input,
                    reasoning=_video_plan_reasoning_payload(),
                    deadline_monotonic=deadline_monotonic,
                    role="video_planning_repair",
                    base_call_type="repair",
                )
                repair_raw = _extract_responses_output_text(repair_response)
                repair_parsed = _parse_json_from_response(repair_raw or "")
                if repair_parsed and not str(repair_parsed.get("planningFailure") or "").strip():
                    repaired_plan, repaired_err = validate_and_normalize_plan(
                        repair_parsed,
                        planner_deadline_monotonic=deadline_monotonic,
                        product_name=product_name,
                        product_description=product_description,
                        content_language=content_language,
                    )
                    if repaired_plan:
                        plan = repaired_plan
                        logger.info("VIDEO_PLAN_REPAIR_OK reason=%s", last_v_err)
            except VideoPlanningTimeoutError:
                raise
            except Exception as e:
                logger.warning(
                    "VIDEO_PLAN_REPAIR_FAILED reason=%s err_type=%s err=%s",
                    last_v_err,
                    type(e).__name__,
                    e,
                )

            if not plan:
                logger.warning("VIDEO_PLAN_FALLBACK_TRIGGERED reason=%s", last_v_err)
                fallback_raw = _build_keyword_scene_fallback_plan(
                    product_name=(parsed.get("productNameResolved") or product_name),
                    content_language=content_language,
                )
                fallback_plan, fallback_err = validate_and_normalize_plan(
                    fallback_raw,
                    planner_deadline_monotonic=deadline_monotonic,
                    product_name=product_name,
                    product_description=product_description,
                    content_language=content_language,
                )
                if fallback_plan:
                    logger.info("VIDEO_PLAN_FALLBACK_USED=true")
                    plan = fallback_plan
                else:
                    logger.error(
                        "VIDEO_PLAN_FALLBACK_USED=false reason=%s",
                        (fallback_err or "").strip() or "planning_failed_incomplete_plan",
                    )
                    logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                    return None, last_v_err

        log_plan_summary(plan)
        logger.info("VIDEO_PLAN_OK model=%s", model)
        _log_video_plan_post_ok_diagnostics(plan)
        logger.info("VIDEO_PLAN_RESPONSE_OK=true")
        return _return_plan_with_promise_persist(
            plan,
            product_name=product_name,
            product_description=product_description,
            session_id=session_id,
        ), ""
    except VideoPlanningTimeoutError:
        raise
    except Exception as e:
        logger.warning(
            "VIDEO_PLAN_FAIL_EXCEPTION phase=post_create err_type=%s err=%s",
            type(e).__name__,
            e,
        )
        logger.info("VIDEO_PLAN_RESPONSE_OK=false")
        return None, "planning_failed_malformed_response"


def fetch_video_plan_o3(
    product_name: str,
    product_description: str,
    content_language: str = "he",
    *,
    session_id: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Fetch plan from o3 under a hard wall-clock deadline; structural normalization only.
    On deadline exceeded, raises VideoPlanningTimeoutError (caller must fail the job).
    Returns (plan, failure_reason); failure_reason is empty on success.
    """
    deadline = time.monotonic() + _VIDEO_PLAN_HARD_SECONDS
    ph = compute_product_hash(product_name, product_description)
    maybe_soft_reset_promise_memory(
        ph, product_name=product_name, product_description=product_description
    )
    logger.info("AD_PROMISE_MEMORY_SESSION_AGNOSTIC=true")
    logger.info("AD_PROMISE_MEMORY_SCOPE global_product_level=true")
    logger.info("AD_PROMISE_MEMORY_PERSISTENT_STORE=true")
    logger.info(
        "VIDEO_TIMING_STAGE_START stage=planning jobId=%s",
        (session_id or "").strip() or "(none)",
    )
    t_plan_outer0 = time.monotonic()
    try:
        plan, fail_reason = _fetch_video_plan_o3_sync(
            product_name,
            product_description,
            content_language,
            deadline_monotonic=deadline,
            session_id=session_id,
        )
        if plan is None:
            increment_promise_stat(
                ph,
                "planning_failed_count",
                1,
                product_name=product_name,
                product_description=product_description,
            )
        logger.info(
            "VIDEO_TIMING_STAGE_END stage=planning jobId=%s elapsed_ms=%.1f ok=%s",
            (session_id or "").strip() or "(none)",
            (time.monotonic() - t_plan_outer0) * 1000.0,
            str(plan is not None).lower(),
        )
        return plan, fail_reason
    except VideoPlanningTimeoutError:
        increment_promise_stat(
            ph,
            "planning_failed_count",
            1,
            product_name=product_name,
            product_description=product_description,
        )
        logger.info("VIDEO_PLAN_RESPONSE_OK=false")
        logger.error(
            "VIDEO_PLAN_FAIL_TIMEOUT hard_seconds=%s (VIDEO_PLANNER_HARD_TIMEOUT_SECONDS or planner+45)",
            _VIDEO_PLAN_HARD_SECONDS,
        )
        logger.info("VIDEO_PLAN_TIMEOUT_FINAL no_valid_plan_before_deadline=true")
        logger.info("VIDEO_JOB_STEP step=plan_video timeout")
        logger.info(
            "VIDEO_TIMING_STAGE_END stage=planning jobId=%s elapsed_ms=%.1f ok=false reason=timeout",
            (session_id or "").strip() or "(none)",
            (time.monotonic() - t_plan_outer0) * 1000.0,
        )
        raise


_RUNWAY_PROMPT_MAX_CHARS = 1000


def _finalize_runway_prompt(headline_prefix: str, body: str) -> Tuple[str, bool]:
    """
    Join optional prefix + body. If over max length, truncate body so a leading prefix survives when present.
    Runway prompts do not include headline burn-in (headline is applied server-side after generation).
    Preserves Physical interaction + CONTACT EXECUTION tail before dropping style filler.
    Returns (final_string, was_truncated).
    """
    body = (body or "").strip()
    hp = (headline_prefix or "").strip()
    if hp:
        full = f"{hp} {body}".strip()
    else:
        full = body
    if len(full) <= _RUNWAY_PROMPT_MAX_CHARS:
        return full, False
    if hp:
        sep = " "
        room = _RUNWAY_PROMPT_MAX_CHARS - len(hp) - len(sep)
        if room < 32:
            out = full[: _RUNWAY_PROMPT_MAX_CHARS]
            return out, True
        trimmed_body = body[:room]
        return f"{hp}{sep}{trimmed_body}", True

    tail_idx = -1
    for marker in _RUNWAY_SCENE_TAIL_MARKERS:
        j = body.find(marker)
        if j >= 0 and (tail_idx < 0 or j < tail_idx):
            tail_idx = j
    if tail_idx >= 0:
        pref, tail = body[:tail_idx], body[tail_idx:]
        budget = _RUNWAY_PROMPT_MAX_CHARS
        tail_work = tail
        if _RUNWAY_STYLE_TAIL in tail_work and len(tail_work) > budget:
            tail_work = tail_work.replace(_RUNWAY_STYLE_TAIL, "").strip()
        if len(tail_work) <= budget:
            room = budget - len(tail_work)
            pref_keep = pref[-room:] if room > 0 else ""
            return f"{pref_keep}{tail_work}".strip(), True
        return tail_work[:budget].rstrip(), True
    return full[:_RUNWAY_PROMPT_MAX_CHARS], True


def _sentence_invites_visible_text(sentence: str) -> bool:
    """
    True if this sentence likely instructs the video model to render readable text/UI (drop it).
    Conservative: keep sentences that are clearly negations (no/forbidden/without … text).
    """
    sl = sentence.lower().strip()
    if not sl:
        return False
    if re.search(
        r"\b(no|never|not|without|don't|do not|forbidden|avoid|must not|zero)\s+"
        r"(?:readable\s+)?(?:text|letters|words|logos?|captions?|headlines?|titles?|subtitles?|watermarks?|signage|labels?)\b",
        sl,
    ):
        return False
    if "no text" in sl or "no letters" in sl or "no logos" in sl or "no readable" in sl:
        return False
    if "no caption" in sl or "no subtitles" in sl or "no headline" in sl:
        return False
    danger_snippets = (
        "include a headline",
        "include the headline",
        "include headline",
        "headline in the",
        "headline on",
        "headline must",
        "title card",
        "on-screen text",
        "readable text",
        "show the text",
        "display the text",
        "show text",
        "text overlay",
        "lower third",
        "chyron",
        "watermark",
        "packaging text",
        "signage",
        "brand name on",
        "spell out",
        "written on",
        "letters on screen",
        "words on screen",
        "typography in the",
        "feature the name",
        "burn-in",
        "burn in",
        "show caption",
        "add caption",
        "open captions",
        "closed caption",
        "add subtitles",
        "show subtitles",
    )
    return any(d in sl for d in danger_snippets)


def sanitize_runway_prompt_for_video_text_policy(prompt: str) -> Tuple[str, bool]:
    """
    Last-line defense before Runway: drop sentences that invite on-screen text; trim length.
    Returns (sanitized_prompt, was_modified).
    """
    original = (prompt or "").strip()
    if not original:
        return "", False

    chunks = re.split(r"(?<=[.!?])\s+", original)
    kept: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if _sentence_invites_visible_text(c):
            continue
        kept.append(c)
    out = " ".join(kept).strip()
    out = re.sub(r"\s+", " ", out)
    if len(out) > _RUNWAY_PROMPT_MAX_CHARS:
        out = out[:_RUNWAY_PROMPT_MAX_CHARS]
    if not out:
        out = (
            "Cinematic commercial motion only; no readable text, letters, logos, captions, "
            "or labels in-frame."
        )
    return out, out != original


def _plan_video_prompt_text(plan: Dict[str, Any]) -> str:
    return (plan.get("videoPrompt") or plan.get("videoPromptCore") or "").strip()


def _build_runway_prompt_compact_fallback(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Shorter ACE→Runway bridge if the detailed builder fails."""
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")
    motion, _ = _runway_variation_montage_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        lang_vis,
        motion,
        f"Montage (follow exactly): {scene_prompt}",
        _RUNWAY_STYLE_TAIL,
    ]
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def _format_variation_montage_prompt(plan: Dict[str, Any], scene_prompt: str) -> str:
    core_visual = (plan.get("coreVisualIdea") or "").strip()
    variations = _plan_scene_variations_list(plan)
    chunks: List[str] = []
    if core_visual:
        chunks.append(f"Core visual idea: {core_visual}.")
    if variations:
        numbered = "; ".join(f"({i + 1}) {v}" for i, v in enumerate(variations))
        chunks.append(f"Variation moments: {numbered}.")
    chunks.append(f"Montage direction: {scene_prompt}")
    return " ".join(chunks)


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Runway prompt from variation-montage plan: configurable-duration montage of related moments.
    No headline burn-in.
    """
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")

    motion, _ = _runway_variation_montage_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    montage_body = _format_variation_montage_prompt(plan, scene_prompt)
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"{motion} "
        f"Montage (follow exactly): {montage_body}. "
        f"{_RUNWAY_STYLE_TAIL}"
    )
    out, trunc = _finalize_runway_prompt("", body)
    if not out.strip():
        raise ValueError("empty prompt")
    return out, trunc


def _build_continuous_event_runway_prompt(plan: Dict[str, Any]) -> str:
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")
    sequence = plan.get("sequence") or {}
    n = _builder2_video_duration_seconds()
    lang_vis = _runway_language_visual_constraints(plan)
    anchor = (plan.get("visualAnchor") or "").strip()
    opening = (plan.get("openingFrameDescription") or sequence.get("beginning") or "").strip()
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"MANDATORY: one continuous {n}-second realistic event in a single location with one primary action. "
        "Natural pacing from opening physical state through development to a clear visual resolution; "
        "no montage, no multiple clips, no unrelated cuts, no dead seconds at the end. "
        f"Opening state: {opening}. "
    )
    if anchor:
        body += f"Visual anchor: {anchor}. "
    if sequence:
        body += (
            f"Development: {sequence.get('development', '')}. "
            f"Resolution: {sequence.get('resolution', '')}. "
        )
    body += f"Continuous event (follow exactly): {scene_prompt}. {_RUNWAY_STYLE_TAIL}"
    out, _ = _finalize_runway_prompt("", body)
    if not out.strip():
        raise ValueError("empty prompt")
    logger.info("RUNWAY_PROMPT path=continuous_event")
    return out


def build_runway_prompt_from_plan(plan: Dict[str, Any]) -> str:
    """
    ACE plan → Runway promptText. Prefers the detailed creative-direction builder; on any failure,
    uses a compact fallback so callers stay stable.
    """
    if (plan.get("structureType") or "").strip() == "continuous_event":
        return _build_continuous_event_runway_prompt(plan)

    headline_decision = (plan.get("headlineDecision") or "no_headline").strip()
    headline_text = (plan.get("headlineText") or "").strip()
    headline_present = headline_decision != "no_headline" and bool(headline_text)

    try:
        out, truncated = _build_runway_prompt_detailed(plan)
        path = "detailed"
    except Exception as e:
        logger.warning("RUNWAY_PROMPT detailed_builder_failed (%s); using compact fallback", e)
        out, truncated = _build_runway_prompt_compact_fallback(plan)
        path = "compact_fallback"

    logger.info(
        "RUNWAY_PROMPT final_len=%s truncated=%s runway_burn_in_headline=%s headline_in_plan=%s headline_text=%r path=%s",
        len(out),
        truncated,
        False,
        headline_present,
        (headline_text[:120] + "…") if len(headline_text) > 120 else headline_text,
        path,
    )
    _, motion_mode = _runway_variation_montage_camera_focus()
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
    return out


def _build_runway_interaction_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Runway prompt when a start frame is supplied — continue the variation montage."""
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")

    motion_focus, _ = _runway_variation_montage_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    montage_body = _format_variation_montage_prompt(plan, scene_prompt)
    n = _builder2_video_duration_seconds()
    scene = (
        f"{lang_vis} "
        f"The first frame is supplied as the start image; continue the same {n}-second variation montage. "
        f"{motion_focus} "
        f"Montage continuation (follow exactly): {montage_body}"
    )
    scene += " No logos or packaging type. Single clean commercial look."

    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{scene}"
    )
    out, trunc = _finalize_runway_prompt("", body)
    if not out.strip():
        raise ValueError("empty prompt")
    return out, trunc


def _build_runway_interaction_prompt_compact_fallback(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Shorter start-frame bridge if the detailed interaction builder fails."""
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")
    lang_vis = _runway_language_visual_constraints(plan)
    motion_focus, _ = _runway_variation_montage_camera_focus()
    motion = (
        f"{lang_vis} "
        "Start frame supplied; continue the variation montage. "
        f"{motion_focus} "
        f"Montage: {scene_prompt}."
    )
    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        motion,
    ]
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def build_runway_interaction_prompt_from_plan(plan: Dict[str, Any]) -> str:
    """
    ACE plan → Runway promptText when promptImage is the generated start frame: interaction/motion only.
    """
    headline_decision = (plan.get("headlineDecision") or "no_headline").strip()
    headline_text = (plan.get("headlineText") or "").strip()
    headline_present = headline_decision != "no_headline" and bool(headline_text)

    try:
        out, truncated = _build_runway_interaction_prompt_detailed(plan)
        path = "interaction_detailed"
    except Exception as e:
        logger.warning("RUNWAY_PROMPT interaction_detailed_failed (%s); using interaction compact fallback", e)
        out, truncated = _build_runway_interaction_prompt_compact_fallback(plan)
        path = "interaction_compact_fallback"

    logger.info(
        "RUNWAY_PROMPT final_len=%s truncated=%s runway_burn_in_headline=%s headline_in_plan=%s headline_text=%r path=%s",
        len(out),
        truncated,
        False,
        headline_present,
        (headline_text[:120] + "…") if len(headline_text) > 120 else headline_text,
        path,
    )
    _, motion_mode = _runway_variation_montage_camera_focus()
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
    out = f"{out.rstrip()} {RUNWAY_PHYSICS_REALISM_CONSTRAINT}".strip()
    return out
