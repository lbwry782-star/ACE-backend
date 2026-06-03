"""
ACE video engine — o3-pro planning layer (isolated from image /preview /generate).

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


def scene_fields_imply_forbidden_surrealism(blob: str) -> Optional[str]:
    """Return a rule label if scene/video prose matches forbidden surreal semantics; else None."""
    if not (blob or "").strip():
        return None
    for label, rx in _SCENE_FORBIDDEN_PATTERNS:
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

# Match codebase: o4-mini maps to o3-pro
def _text_model() -> str:
    m = (os.environ.get("VIDEO_PLANNER_MODEL") or os.environ.get("OPENAI_TEXT_MODEL", "") or "").strip() or "o3-pro"
    return "o3-pro" if m == "o4-mini" else m


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
):
    """
    Up to two identical planning model calls; one retry on transient timeout only.
    Raises VideoPlanningTimeoutError if hard deadline is exceeded.
    """
    backoff_s = _video_plan_model_retry_backoff_s()
    last_exc: Optional[BaseException] = None

    for attempt in range(1, _VIDEO_PLAN_MODEL_MAX_ATTEMPTS + 1):
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


_JSON_KEYS = """
Return one JSON object only (no markdown, no prose).

Keys (all strings): productNameResolved, headline, headlineCoreKeyword, sceneConcept, videoPrompt, language

Flow (mandatory order — internal only; output final JSON only):
1) Read product name + product description.
2) headline: direct expression of the primary advertising advantage implied by the product; remainder ONLY — do NOT include productNameResolved inside headline. Hebrew, English, or mixed. Up to 7 words. Prefer one strong metaphorical word; avoid literal industry/category words when possible.
3) headlineCoreKeyword: exactly ONE word — the strongest metaphorical word in headline; must appear in headline; must be capable of generating a realistic everyday human scene; never articles/prepositions/conjunctions/fillers; never literal industry words (e.g. advertising, marketing, digital, campaign, story, service, strategy).
4) sceneConcept: the literal real-world interpretation of headlineCoreKeyword in everyday human life — NOT a metaphor for the full headline.
5) videoPrompt: English cinematic direction for Runway — completely realistic, physical, everyday; describes sceneConcept only; no fantasy/surrealism/symbolic objects/impossible events; no readable on-screen text.

Empty product name → invent productNameResolved.

Before the JSON: one silent internal revision pass (headline → keyword → scene → prompt); output final JSON only.

Failure only: {"planningFailure":"planning_failed_invalid_plan"}
"""


def _planner_headline_rules_block() -> str:
    return (
        "HEADLINE RULES:\n"
        "- The headline is the direct expression of the primary advertising advantage implied by product name + description.\n"
        "- Prefer headlines that contain a single strong metaphorical word (e.g. close, bridge, door, path, heart, home, key, step, light, connection).\n"
        "- Avoid literal industry/category words whenever possible (e.g. advertising, marketing, digital, campaign, story, service, strategy).\n"
        "- Remainder only — no product name inside headline. Up to 7 words.\n\n"
        "headlineCoreKeyword RULES:\n"
        "- Exactly one word from the headline.\n"
        "- The strongest metaphorical word — capable of generating a realistic everyday scene.\n"
        "- FORBIDDEN as keyword: advertising, marketing, digital, campaign, story, service, strategy (and Hebrew equivalents).\n"
        "- STRONGER examples: close/קרוב, bridge, door/דלת, path/דרך, heart, home, key, step, light, connection.\n\n"
    )


def _planner_keyword_scene_flow_block() -> str:
    return (
        "BUILDER2 KEYWORD-SCENE FLOW v2 (mandatory; do not narrate in JSON):\n"
        "STEP 1 — Read product_name and product_description.\n"
        "STEP 2 — headline: direct advertising advantage expression (see HEADLINE RULES).\n"
        "STEP 3 — headlineCoreKeyword: strongest metaphorical word in headline.\n"
        "STEP 4 — sceneConcept: literal real-world interpretation of headlineCoreKeyword only.\n"
        "STEP 5 — videoPrompt: Runway-ready realistic scene from sceneConcept.\n\n"
        + _planner_headline_rules_block()
        + "SCENE EXAMPLES:\n"
        '- Headline "הכי קרוב למשרד פרסום" → keyword "קרוב" → scene: two people warmly embracing after meeting.\n'
        '- Headline "פותחים לך דלת להזדמנויות" → keyword "דלת" → scene: a person opening a front door and welcoming someone inside.\n'
        '- Headline "קיצור הדרך ליותר לקוחות" → keyword "דרך" → scene: a person choosing a shorter walking path in a park.\n\n'
        "SCENE RULES (sceneConcept + videoPrompt):\n"
        "- Realistic, everyday, simple, human, physically possible.\n"
        "- Literal real-world interpretation of the keyword — not surreal symbolism.\n"
        "- A viewer who has NOT seen the headline should see a normal realistic human situation.\n"
        "FORBIDDEN: surreal, dreamlike, fantasy, magic, impossible physics, talking/animated/floating/symbolic objects, "
        "impossible events, science fiction, abstract visual concepts.\n"
        "HEADLINE DISPLAY (downstream): scene plays first; headline overlay appears at the end — do NOT burn headline into videoPrompt.\n\n"
    )


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    return (
        f"ACE Builder2 video planning — keyword-scene v2 (no advertisingGoal stage). "
        f"Language {lang_name} ({lang}). "
        "product → headline → headlineCoreKeyword → sceneConcept → videoPrompt. "
        "Scene realism is mandatory. "
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


_VIDEO_PLAN_SCHEMA_VERSION = "keyword_scene_v2"

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
        "Fix headline, headlineCoreKeyword, sceneConcept, and videoPrompt to satisfy all rules.\n"
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
    scene = (
        "שני אנשים מתחבקים בחום לאחר שנפגשו שוב"
        if lang == "he"
        else "Two people warmly embracing after finally meeting again"
    )
    video_prompt = (
        "A completely realistic everyday scene of two people warmly embracing after finally meeting again. "
        "Natural human behavior. Real-world environment. Stable cinematic camera. "
        "No fantasy. No surrealism. No symbolic objects. No impossible events. No readable text in-frame."
    )
    return {
        "productNameResolved": pn,
        "headline": headline,
        "headlineCoreKeyword": keyword,
        "sceneConcept": scene,
        "videoPrompt": video_prompt,
        "language": lang,
        "planInferenceMode": "deterministic_keyword_scene_fallback",
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
    "Scene:",
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
    ("scene_concept", "sceneConcept"),
    ("video_prompt", "videoPrompt"),
    ("headline_text", "headline"),
    ("video_prompt_core", "videoPrompt"),
)


def _coerce_plan_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill camelCase keys from snake_case duplicates when the canonical key is missing or empty."""
    out = dict(data)
    for alt, canon in _PLAN_KEY_ALIASES:
        cur = out.get(canon)
        empty = cur is None or (isinstance(cur, str) and not cur.strip())
        alt_val = out.get(alt)
        if empty and alt_val is not None and str(alt_val).strip():
            out[canon] = alt_val
    return out


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
    scene = (data.get("sceneConcept") or "").strip()
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
    if not scene:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_scene_concept")
        return None, "planning_failed_incomplete_plan"
    if not video_prompt:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_video_prompt")
        return None, "planning_failed_incomplete_plan"

    if planner_deadline_monotonic is not None and time.monotonic() >= planner_deadline_monotonic:
        logger.error("VIDEO_PLAN_DEADLINE_EXCEEDED stage=validate")
        raise VideoPlanningTimeoutError()

    if _headline_remainder_word_count(headline_rem) > _MAX_HEADLINE_REMAINDER_WORDS:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_too_long")
        return None, "planning_failed_headline_too_long"

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
    if not _headline_contains_core_keyword(headline_rem, core_kw):
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_core_keyword_not_in_headline")
        return None, "planning_failed_invalid_keyword"

    scene_blob = "\n".join([scene, video_prompt])
    bad_scene = scene_fields_imply_forbidden_surrealism(scene_blob)
    if bad_scene:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=forbidden_surreal_scene rule=%s",
            bad_scene,
        )
        return None, "planning_failed_surreal_scene"

    pn_for_headline = (product_name or "").strip() or pn
    headline_full = _assemble_headline_full(pn_for_headline, headline_rem)
    opening_fd = scene[:400]

    logger.info('VIDEO_PLAN_HEADLINE="%s"', headline_rem[:260])
    logger.info('VIDEO_PLAN_CORE_KEYWORD="%s"', core_kw[:120])
    logger.info('VIDEO_PLAN_SCENE_CONCEPT="%s"', scene[:260])

    return {
        "productNameResolved": pn,
        "advertisingPromise": headline_rem,
        "headline": headline_rem,
        "headlineText": headline_full,
        "headlineTextRemainder": headline_rem,
        "headlineCoreKeyword": core_kw,
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


def _reasoning_effort() -> str:
    raw = (os.environ.get("VIDEO_PLANNER_REASONING_EFFORT") or "low").strip().lower()
    return raw if raw in ("low", "medium") else "low"


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
    logger.info("VIDEO_PLAN_SEARCH_ORDER=keyword_scene_v2")
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
{_JSON_KEYS}
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
            reasoning={"effort": _reasoning_effort()},
            deadline_monotonic=deadline_monotonic,
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
                    reasoning={"effort": _reasoning_effort()},
                    deadline_monotonic=deadline_monotonic,
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
    motion, _ = _runway_human_scene_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        lang_vis,
        motion,
        scene_prompt,
        _RUNWAY_STYLE_TAIL,
    ]
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Runway prompt from keyword-scene plan: videoPrompt + realistic human-scene camera rules.
    No headline burn-in; advertising goal is not injected into Runway promptText.
    """
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")

    motion, _ = _runway_human_scene_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"{motion} "
        f"Scene (follow exactly): {scene_prompt}. "
        f"{_RUNWAY_STYLE_TAIL}"
    )
    out, trunc = _finalize_runway_prompt("", body)
    if not out.strip():
        raise ValueError("empty prompt")
    return out, trunc


def build_runway_prompt_from_plan(plan: Dict[str, Any]) -> str:
    """
    ACE plan → Runway promptText. Prefers the detailed creative-direction builder; on any failure,
    uses a compact fallback so callers stay stable.
    """
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
    _, motion_mode = _runway_human_scene_camera_focus()
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
    return out


def _build_runway_interaction_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Runway prompt when a start frame is supplied — motion continues the realistic human scene."""
    scene_prompt = _plan_video_prompt_text(plan)
    if not scene_prompt:
        raise ValueError("missing videoPrompt")

    motion_focus, _ = _runway_human_scene_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    scene = (
        f"{lang_vis} "
        "The first frame is supplied as the start image; continue the same realistic human scene. "
        f"{motion_focus} "
        f"Scene continuation (follow exactly): {scene_prompt}"
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
    motion_focus, _ = _runway_human_scene_camera_focus()
    motion = (
        f"{lang_vis} "
        "Start frame supplied; continue the realistic human scene. "
        f"{motion_focus} "
        f"Scene: {scene_prompt}."
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
    _, motion_mode = _runway_human_scene_camera_focus()
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
    out = f"{out.rstrip()} {RUNWAY_PHYSICS_REALISM_CONSTRAINT}".strip()
    return out
