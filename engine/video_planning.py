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
import unicodedata
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import httpx
from openai import OpenAI

from engine.video_language import normalize_video_content_language, video_language_display_name

from engine.ad_promise_memory import (
    angle_seed_for_attempt,
    build_promise_diversity_addon,
    compute_product_hash,
    forbidden_promises_for_prompt,
    increment_promise_stat,
    is_promise_too_similar,
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

# Logged for diagnostics only; mode is decided from abInteractionType + branch validity (see validate_and_normalize_plan).
_VIDEO_REPLACEMENT_INTERACTION_SCORE_THRESHOLD = 85


_JSON_KEYS = """
OUTPUT FORMAT (strict)
- Return ONE JSON object only.
- Use the exact camelCase keys below. Do not omit required keys; use "" only where allowed.
- Do NOT wrap in markdown code fences. Do NOT add prose before or after the JSON.

Field notes:
- morphologicalReason: Why each object fits the advertising story (job language where noted). A and B need not look alike.
- objectPairViewerClarityOk / objectPairIdentityDistinctOk / identityDistinctnessNote: as before.
- discoveryInteractionSummary: short English phrase describing the CLASSIC or MEANINGFUL interaction you used to find objectB.
- visibleAdditionalInteractionSummary: short English phrase describing the additional creative interaction that will actually be shown in the video.
- visibleMotionScript: English. Full motion description for the additional interaction only (what the camera will see).

Required keys (all strings except where noted):
{
  "productNameResolved": string,
  "advertisingPromise": string,
  "objectA": string,
  "objectB": string,
  "morphologicalReason": string,
  "promiseReason": string,
  "replacementDirection": "B_replaces_A" or "A_replaces_B",
  "preservedBackgroundFrom": "A" or "B",
  "shortReplacementScript": string,
  "headlineDecision": "include_product_name" or "product_name_only" or "no_headline",
  "headlineText": string,
  "replacementOpeningFrameDescription": string,
  "replacementMotionScript": string,
  "sideBySideOpeningFrameDescription": string,
  "sideBySideMotionScript": string,
  "objectPairViewerClarityOk": boolean,
  "objectPairIdentityDistinctOk": boolean,
  "identityDistinctnessNote": string,
  "abInteractionType": "classic" or "meaningful",
  "discoveryInteractionSummary": string,
  "visibleAdditionalInteractionSummary": string,
  "visibleMotionScript": string
}
"""


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    return f"""You are the ACE video planning engine.

LANGUAGE
- Job: {lang_name} ({lang}), grounded in product name + product description. advertisingPromise, headlineText, shortReplacementScript, morphologicalReason, promiseReason: primarily {lang_name}. Loanwords/brands (AI, SaaS, etc.) OK.
- objectA, objectB: short English noun phrases (classic physical objects only).
- replacementOpeningFrameDescription, replacementMotionScript, sideBySideOpeningFrameDescription, sideBySideMotionScript: English only (no exceptions).

FOUNDATION
- The advertisingPromise must be derived only from the product name + the product description (no generic metaphor-first search).

OBJECT DISCOVERY (work strictly in this order — matches server validation)
1) Derive advertisingPromise only from the product name + the product description.
2) Choose objectA: objectA must be a classic, defined, physical object, visually depictable, and grounded in advertisingPromise.
3) Discover objectB:
   - objectB must be a classic, defined, physical object.
   - objectB must be grounded in advertisingPromise.
   - objectB must have a CLASSIC or MEANINGFUL discovery interaction with objectA (you label this using abInteractionType).
   - objectB must ALSO enable a SECOND, ADDITIONAL physical interaction with objectA that:
       * is not the classic discovery interaction,
       * is not the meaningful discovery interaction,
       * and is not directly derived from the advertisingPromise.
   Stop as soon as you find such an objectB.

VISIBLE INTERACTION (video content)
- The video must show ONLY the additional interaction between objectA and objectB.
- The discovery interaction (classic/meaningful) must NOT appear in the video.
- The visible interaction must:
  - be physically clear and visually understandable,
  - feel unexpected relative to the two objects,
  - not directly restate or illustrate the advertisingPromise.
- The camera moves in a gentle half-orbit around the pair throughout the shot.

OBJECT RULES
- Accepted objects: physical, classic, clearly defined, visually depictable.
- Rejected objects: abstract concepts, verbs, adjectives, sentence fragments, promise fragments, benefits/outcomes as “objects”, non-physical nouns.

DISCOVERY LABEL (no mode)
- abInteractionType is ONLY a label for the discovery interaction between objectA and objectB:
  - "classic": familiar, inherent, canonical real-world pairing (bee and flower, straw and cup, dog and bone, key and lock).
  - "meaningful": clear physical interaction that is not a special canonical classic pair.
- Do NOT use abInteractionType to change the video structure; the server uses only the additional interaction you describe in the English fields.

MEMORY (diversity only)
- If the server lists prior advertisingPromise lines for this product, use that list only to reduce repetition and prefer unused valid solutions.
- Memory must not introduce unrelated objects, must not override advertising-promise grounding from the product, and must not override interaction logic.

CREATIVE RULES (mandatory)
1) Derive advertisingPromise from the product name and the product description (in {lang_name} per field rules above).
2) objectA and objectB must each be clearly connected to the advertising promise; they do not need similar shapes.
3) Classic, defined, physical objects in classic situations only.
4) Filter text, logos, brands, written labels, vague environments, non-physical situations.
5) There are no secondary objects in video: plan only objectA and objectB as physical subjects.
6) Output abInteractionType exactly as required. Output BOTH creative branches (replacement* and sideBySide*) fully.

REPLACEMENT branch (English fields — used only if server selects REPLACEMENT)
- Exactly one primary is visible on camera (per replacementDirection); the other primary must never appear. Motion describes only that visible primary and the environment implied by preservedBackgroundFrom.
- Motion: English, concrete verbs, visually obvious cause/effect; no transformation/morph language.

SIDE_BY SIDE branch (English fields — used only if server selects SIDE_BY_SIDE)
- Opening: A and B both visible from frame 1; tight unified composition; close or slightly overlapping; same angle/scale/world; NO replacement framing.
- Motion: clear physical A↔B interaction (viewer-instant read). Avoid abstract-only or purely symbolic motion. No morphing, swapping, disappearance, wide empty split layout, cuts, or multi-shot story.

IDENTITY DISTINCTNESS
- objectPairIdentityDistinctOk / identityDistinctnessNote: A and B must be clearly different objects, not variants of the same thing. Reject near-twin ambiguous pairs per prior rules.

PROMISE DIVERSITY (persistent product memory — diversity only)
- The advertisingPromise MUST be materially different from all prior promises the server lists for this product in this request. Do not restate speed, accuracy, security, ease, savings, or growth as the main story if that angle was already used—pick a genuinely new core idea (not wording tricks). Use this only for diversity; do not let it override product grounding or object/interaction rules.

OBJECTS + SEARCH
- Iconic primaries only. Compare multiple B candidates; never trade promise clarity for weaker interaction.

PAIRING + PROMISE
- replacementDirection: set which primary is visible when REPLACEMENT is selected (B_replaces_A vs A_replaces_B). Set preservedBackgroundFrom to the side whose world/background should read in-frame.

HEADLINE (overlay metadata only—never pixels in video)
- headlineDecision: include_product_name | product_name_only | no_headline. headlineText: ≤7 words {lang_name}. Never burn headline into videoPrompt fields.

TEXT-FREE VIDEO
- No readable text in motion/opening descriptions or shortReplacementScript.

QUALITY
- morphologicalReason in {lang_name}. Both English branch descriptions must be concrete and policy-compliant.

"""


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


def _normalize_object_identifier_for_compare(s: str) -> str:
    """Lowercase NFC label for equality checks (underscores/hyphens → space, collapse spaces)."""
    t = unicodedata.normalize("NFC", (s or "").strip().lower())
    t = re.sub(r"[-_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _primaries_differ_norm(oa: str, ob: str) -> bool:
    return _normalize_object_identifier_for_compare(oa) != _normalize_object_identifier_for_compare(ob)


# Packaging / logistics primaries — not iconic enough as standalone A/B subjects for replacement clarity.
_WEAK_ICONIC_OBJECT_PHRASES: Tuple[str, ...] = (
    "shoe box",
    "shoebox",
    "shipping box",
    "mailer box",
    "product box",
    "cardboard box",
    "pizza box",
    "takeout box",
    "take-out box",
    "moving box",
    "storage box",
    "delivery box",
    "shipping carton",
    "shopping bag",
    "grocery bag",
    "produce bag",
    "plastic mailer",
    "bubble mailer",
    "padded envelope",
    "mailing envelope",
    "poly mailer",
    "shipping envelope",
    "bubble envelope",
    "product packaging",
    "generic box",
    "generic bag",
    "generic container",
    "empty box",
    "plain box",
    "plain bag",
    "storage bin",
    "plastic bin",
)

_TRIVIAL_BOX_LABELS = frozenset(
    {"box", "a box", "plain box", "empty box", "the box", "generic box", "a plain box"}
)


def _normalize_object_label_for_trivial_check(s: str) -> str:
    t = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _object_label_is_trivially_weak(s: str) -> bool:
    return _normalize_object_label_for_trivial_check(s) in _TRIVIAL_BOX_LABELS


def _object_string_has_weak_packaging_phrase(s: str) -> bool:
    low = (s or "").lower()
    return any(p in low for p in _WEAK_ICONIC_OBJECT_PHRASES)


def _object_pair_fails_weak_identity_heuristic(oa: str, ob: str) -> bool:
    """
    True → reject plan: A or B reads as packaging / weak-identity primary (server-side guardrail).
    """
    if _object_string_has_weak_packaging_phrase(oa) or _object_string_has_weak_packaging_phrase(ob):
        return True
    if _object_label_is_trivially_weak(oa) or _object_label_is_trivially_weak(ob):
        return True
    if re.search(
        r"\bgeneric\s+(?:box|bag|container|package|packaging)\b",
        (oa or "").lower(),
    ) or re.search(
        r"\bgeneric\s+(?:box|bag|container|package|packaging)\b",
        (ob or "").lower(),
    ):
        return True
    return False


def _identity_distinctness_norm_pair_key(oa: str, ob: str) -> Tuple[str, str]:
    a = _normalize_object_identifier_for_compare(oa)
    b = _normalize_object_identifier_for_compare(ob)
    return tuple(sorted((a, b)))


# Order-independent normalized labels; server guardrail for known near-twin ambiguous swaps.
_IDENTITY_TOO_CLOSE_NORM_PAIRS: FrozenSet[Tuple[str, str]] = frozenset(
    {
        _identity_distinctness_norm_pair_key("pencil", "stylus"),
        _identity_distinctness_norm_pair_key("pen", "stylus"),
        _identity_distinctness_norm_pair_key("ballpoint pen", "stylus"),
        _identity_distinctness_norm_pair_key("monitor", "tv"),
        _identity_distinctness_norm_pair_key("monitor", "television"),
        _identity_distinctness_norm_pair_key("tv", "television"),
        _identity_distinctness_norm_pair_key("gift box", "shoe box"),
    }
)


def _object_pair_identity_too_close_heuristic(oa: str, ob: str) -> bool:
    """True → reject: known near-twin pair where replacement tends to read as one object category."""
    return _identity_distinctness_norm_pair_key(oa, ob) in _IDENTITY_TOO_CLOSE_NORM_PAIRS


# REPLACEMENT enforcement: validated replacement branch must not name the absent primary as on-screen.
_REPLACEMENT_MOTION_REQUIRED_VERBS: Tuple[str, ...] = (
    "use",
    "uses",
    "using",
    "interact",
    "interacts",
    "trigger",
    "triggers",
    "activate",
    "activates",
    "resolve",
    "resolves",
    "change",
    "changes",
    "respond",
    "responds",
    "transform",
    "transforms",
    "solve",
    "solves",
)


def _norm_words_for_presence(s: str) -> List[str]:
    t = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return [w for w in t.split() if w]


def _contains_object_tokens(text: str, label: str) -> bool:
    words = _norm_words_for_presence(label)
    if not words:
        return False
    low = (text or "").lower()
    return any(len(w) >= 3 and re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)


def _replacement_motion_is_meaningful(script: str, visible_primary: str, absent_primary: str) -> bool:
    """
    REPLACEMENT motion: visible primary appears in script; absent primary must not read as on-screen.
    """
    s = (script or "").strip().lower()
    if len(s) < 40:
        return False
    if not _contains_object_tokens(s, visible_primary):
        return False
    if _contains_object_tokens(s, absent_primary):
        return False
    if not any(v in s for v in _REPLACEMENT_MOTION_REQUIRED_VERBS):
        return False
    decorative_only = ("idle", "floating", "ambient", "aesthetic", "beauty shot", "loop")
    if any(x in s for x in decorative_only):
        return False
    return True


def _object_connects_to_advertising_goal(object_label: str, promise_context: str) -> bool:
    """
    Each primary must tie to the advertising goal using promise + promiseReason + morphologicalReason text.
    When the context has no extractable Latin tokens (e.g. Hebrew-only), skip strict token matching.
    """
    o = (object_label or "").strip().lower()
    ctx = (promise_context or "").strip().lower()
    if not o:
        return False
    if not ctx:
        return True
    if o in ctx:
        return True
    ctx_ascii = set(re.findall(r"[a-z]{3,}", ctx))
    if not ctx_ascii:
        return True
    for w in _norm_words_for_presence(o):
        if len(w) >= 3 and w in ctx_ascii:
            return True
        if len(w) >= 4 and any(w in t or t in w for t in ctx_ascii if len(t) >= 4):
            return True
    return any(t in o for t in ctx_ascii if len(t) >= 5)


def _object_grounded_in_advertising_promise(object_label: str, apromise: str) -> bool:
    """True iff object label is visually/conceptually tied to the advertising promise text only (product flow)."""
    o = (object_label or "").strip()
    p = (apromise or "").strip()
    if not o or not p:
        return False
    lo = o.lower()
    lp = p.lower()
    if lo in lp or lp in lo:
        return True
    o_toks = set(re.findall(r"[\w\u0590-\u05FF]{3,}", lo, flags=re.UNICODE))
    p_toks = set(re.findall(r"[\w\u0590-\u05FF]{3,}", lp, flags=re.UNICODE))
    if o_toks & p_toks:
        return True
    for w in o_toks:
        if len(w) >= 3 and w in lp:
            return True
    for w in p_toks:
        if len(w) >= 3 and w in lo:
            return True
    return False


def _planning_text_tokens(s: str) -> Set[str]:
    return set(re.findall(r"[\w\u0590-\u05FF]{3,}", (s or "").lower(), flags=re.UNICODE))


def _advertising_promise_from_product(
    apromise: str, product_name: str, product_description: str
) -> bool:
    """True iff the advertising promise is plausibly derived from product name + description (token overlap)."""
    p = (apromise or "").strip()
    if not p:
        return False
    blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not blob:
        return False
    blob_l = blob.lower()
    if p.lower() in blob_l:
        return True
    ptoks = _planning_text_tokens(p)
    btoks = _planning_text_tokens(blob)
    if ptoks & btoks:
        return True
    for w in ptoks:
        if len(w) >= 3 and w in blob_l:
            return True
    return False


def _repaired_advertising_promise_for_product(
    original: str, product_name: str, product_description: str
) -> Tuple[str, bool]:
    """
    Ensure advertisingPromise passes _advertising_promise_from_product.
    Returns (promise_text, repaired) where repaired is True iff the string was changed.
    """
    o = (original or "").strip()
    if _advertising_promise_from_product(o, product_name, product_description):
        return o, False
    pn = (product_name or "").strip()
    pd = (product_description or "").strip()
    if not pn and not pd:
        return o, False
    parts = [x for x in (pn, pd) if x]
    candidate = " ".join(parts).strip()[:480] or o[:480] or "Product"
    if not _advertising_promise_from_product(candidate, product_name, product_description):
        candidate = (f"{pn} — {pd}".strip() if (pn or pd) else candidate)[:480]
    return candidate, True


def _promise_augment_notebook_pen(promise: str, product_name: str) -> str:
    """Append explicit concrete anchors so English object labels ground in the promise."""
    pn = (product_name or "").strip()
    base = (promise or "").strip()
    frag = f" {pn}: notebook and pen." if pn else " Notebook and pen."
    return (base + frag).strip()[:480]


def _fallback_abc_grounding_ok(
    promise: str, oa: str, ob: str, product_name: str, product_description: str
) -> Tuple[bool, bool, bool, bool, bool]:
    """Returns (pf, ga, gb, pa, pb) product-grounded promise + A/B promise-grounded + physical."""
    pf = _advertising_promise_from_product(promise, product_name, product_description)
    ga = _object_grounded_in_advertising_promise(oa, promise)
    gb = _object_grounded_in_advertising_promise(ob, promise)
    pa, _ = _object_label_is_physical_classic(oa)
    pb, _ = _object_label_is_physical_classic(ob)
    return pf, ga, gb, pa, pb


def _log_fallback_repair_diagnostics(
    *,
    original_promise: str,
    final_promise: str,
    original_oa: str,
    original_ob: str,
    final_oa: str,
    final_ob: str,
    promise_text_changed: bool,
    pf: bool,
    ga: bool,
    gb: bool,
    pa: bool,
    pb: bool,
    stale_objects_after_promise_repair: bool,
    sbs_script_ok: bool,
) -> None:
    """Structured logs for salvage; callers gate acceptance on coherent package."""
    coherent = bool(pf and ga and gb and pa and pb and sbs_script_ok)
    logger.info("VIDEO_PLAN_FALLBACK_COHERENT_PACKAGE=%s", str(coherent).lower())
    logger.info(
        "VIDEO_PLAN_FALLBACK_REPAIRED_PROMISE_ONLY=%s",
        str(bool(promise_text_changed and stale_objects_after_promise_repair)).lower(),
    )
    triple = bool(pf and ga and gb)
    logger.info(
        "VIDEO_PLAN_FALLBACK_REPAIRED_PROMISE_FROM_PRODUCT=%s",
        str(bool(triple)).lower(),
    )
    logger.info(
        "VIDEO_PLAN_FALLBACK_REPAIRED_OBJECT_A=%s",
        str(
            _normalize_object_identifier_for_compare(final_oa)
            != _normalize_object_identifier_for_compare(original_oa)
        ).lower(),
    )
    logger.info(
        "VIDEO_PLAN_FALLBACK_REPAIRED_OBJECT_B=%s",
        str(
            _normalize_object_identifier_for_compare(final_ob)
            != _normalize_object_identifier_for_compare(original_ob)
        ).lower(),
    )
    full = coherent and (
        promise_text_changed
        or _normalize_object_identifier_for_compare(final_oa)
        != _normalize_object_identifier_for_compare(original_oa)
        or _normalize_object_identifier_for_compare(final_ob)
        != _normalize_object_identifier_for_compare(original_ob)
    )
    logger.info("VIDEO_PLAN_FALLBACK_FULL_REPAIR=%s", str(full).lower())


def _parse_norm_ab_interaction_type(raw: Any) -> str:
    """Return exactly 'classic' | 'meaningful' | ''."""
    s = str(raw or "").strip().lower()
    if s == "classic":
        return "classic"
    if s == "meaningful":
        return "meaningful"
    return ""


_OBJECT_LABEL_MAX_CHARS = 48
_OBJECT_LABEL_MAX_TOKENS = 4

_NON_PHYSICAL_EN_TOKENS: FrozenSet[str] = frozenset(
    {
        "optimization",
        "optimizing",
        "optimized",
        "testing",
        "tests",
        "test",
        "improvement",
        "improvements",
        "improving",
        "improved",
        "efficiency",
        "efficient",
        "performance",
        "intelligence",
        "automation",
        "automated",
        "autonomy",
        "autonomies",
        "results",
        "result",
        "outcomes",
        "outcome",
        "productivity",
        "scalability",
        "scalable",
        "reliability",
        "insights",
        "insight",
        "analytics",
        "visibility",
        "transparency",
        "velocity",
        "acceleration",
        "growth",
        "innovation",
        "innovations",
        "excellence",
        "quality",
        "qualities",
        "capabilities",
        "capability",
        "agility",
        "alignment",
        "alignments",
        "strategy",
        "strategies",
        "management",
        "operations",
        "operation",
        "process",
        "processes",
        "workflow",
        "workflows",
        "solution",
        "solutions",
        "value",
        "values",
        "benefit",
        "benefits",
        "experience",
        "experiences",
        "success",
        "successes",
        "engagement",
        "framework",
        "platform",
        "platforms",
        "ecosystem",
        "ecosystems",
        "empowerment",
        "software",
        "system",
        "systems",
        "data",
        "api",
        "saas",
        "cloud",
        "digital",
        "digitization",
        "digitalization",
        "transformation",
        "transformations",
        "monitoring",
        "reporting",
        "processing",
        "onboarding",
        "orchestration",
        "integration",
        "metrics",
        "metric",
        "kpi",
        "kpis",
        "roi",
        "revenue",
        "profitability",
        "compliance",
        "governance",
        "security",
        "privacy",
        "encryption",
        "authentication",
        "authorization",
        "latency",
        "throughput",
        "bandwidth",
        "capacity",
        "utilization",
        "adoption",
        "retention",
        "churn",
        "conversion",
        "conversions",
        "optimization",
        "benchmark",
        "benchmarks",
        "scorecard",
    }
)

_NON_PHYSICAL_HE_TOKENS: FrozenSet[str] = frozenset(
    {
        "בדיקות",
        "בדיקה",
        "אוטונומיות",
        "אוטונומיה",
        "תוצאות",
        "תוצאה",
        "שמשפרות",
        "שיפור",
        "שיפורים",
        "ביצועים",
        "ביצוע",
        "יעילות",
        "יעיל",
        "חדשנות",
        "חדשני",
        "איכות",
        "אסטרטגיה",
        "אסטרטגיות",
        "ניהול",
        "ניהולי",
        "אנליטיקה",
        "שקיפות",
        "תובנות",
        "תובנה",
        "מטרות",
        "מטרה",
        "יעדים",
        "יעד",
        "פתרונות",
        "פתרון",
        "שדרוג",
        "שדרוגים",
        "אינטגרציה",
        "דיגיטלי",
        "דיגיטלית",
        "עסקים",
        "עסקי",
        "שירות",
        "שירותים",
        "מערכות",
        "מערכת",
        "תהליך",
        "תהליכים",
        "אוטומציה",
        "אוטומטיות",
        "למידה",
        "חכמה",
        "חכמות",
        "יצירתיות",
        "מקצועיות",
        "מומחיות",
        "עצמאות",
        "זמינות",
        "גמישות",
        "אמינות",
        "יציבות",
        "מודיעין",
        "יכולות",
        "יכולת",
        "הצלחה",
        "הצלחות",
        "חוויה",
        "חוויות",
        "ערך",
        "ערכים",
        "יתרון",
        "יתרונות",
        "חסרונות",
        "סיכון",
        "סיכונים",
        "ניתוח",
        "ניתוחים",
        "דוחות",
        "דוח",
        "מדדים",
        "מדד",
        "יעול",
        "יעולים",
    }
)

_EN_ABSTRACT_SUFFIX_EXCEPTIONS: FrozenSet[str] = frozenset(
    {
        "document",
        "moment",
        "basement",
        "segment",
        "filament",
        "cement",
        "equipment",
        "attachment",
        "ornament",
        "garment",
        "pavement",
        "apartment",
        "shipment",
        "implement",
        "implements",
        "movement",
        "ointment",
        "sediment",
        "supplement",
        "nutriment",
        "lament",
        "torment",
        "element",
        "pigment",
        "figment",
    }
)


def _object_label_tokens_for_physical_check(label: str) -> List[str]:
    out: List[str] = []
    for part in (label or "").split():
        p = re.sub(r'^[^\w\u0590-\u05FF]+|[^\w\u0590-\u05FF]+$', "", part)
        if p:
            out.append(unicodedata.normalize("NFC", p))
    return out


def _english_token_abstract_by_suffix(token_lower: str) -> bool:
    if len(token_lower) < 6 or token_lower in _EN_ABSTRACT_SUFFIX_EXCEPTIONS:
        return False
    for suf in ("tion", "sion", "ness", "ment", "ity", "ance", "ence", "ship", "hood"):
        if token_lower.endswith(suf):
            return True
    if token_lower.endswith("ing") and len(token_lower) >= 7:
        if token_lower in (
            "building",
            "ceiling",
            "lighting",
            "flooring",
            "roofing",
            "railing",
            "fencing",
            "string",
            "spring",
            "ring",
        ):
            return False
        return True
    return False


def _object_label_is_physical_classic(label: str) -> Tuple[bool, str]:
    """
    Hard gate: object labels must be depictable physical things, not abstract/process/benefit words.
    Returns (ok, failure_token_or_reason_code).
    """
    raw = unicodedata.normalize("NFC", (label or "").strip())
    if not raw:
        return False, "empty"
    if any(ch in raw for ch in "\n\r\t,;:–—"):
        return False, "clause_or_list_punctuation"
    if len(raw) > _OBJECT_LABEL_MAX_CHARS:
        return False, "too_long"
    tokens = _object_label_tokens_for_physical_check(raw)
    if not tokens:
        return False, "no_tokens"
    if len(tokens) > _OBJECT_LABEL_MAX_TOKENS:
        return False, "too_many_tokens"
    for tok in tokens:
        tl = tok.lower()
        if tl in _NON_PHYSICAL_EN_TOKENS:
            return False, tok
        if tok in _NON_PHYSICAL_HE_TOKENS or tl in _NON_PHYSICAL_HE_TOKENS:
            return False, tok
        he_only = "".join(ch for ch in tok if "\u0590" <= ch <= "\u05FF")
        if he_only and he_only in _NON_PHYSICAL_HE_TOKENS:
            return False, tok
        lat = re.findall(r"[a-z]{2,}", tl)
        for w in lat:
            if w in _NON_PHYSICAL_EN_TOKENS:
                return False, tok
            if _english_token_abstract_by_suffix(w):
                return False, tok
        if re.fullmatch(r"[a-z]{2,}", tl) and _english_token_abstract_by_suffix(tl):
            return False, tok
    return True, ""


def _validate_object_pair_physical(oa: str, ob: str) -> Tuple[bool, str, str]:
    """Returns (ok, field_name, offending_value)."""
    for field, val in (("objectA", oa), ("objectB", ob)):
        ok, hit = _object_label_is_physical_classic(val)
        if not ok:
            return False, field, val
    return True, "", ""


def _score_replacement_interaction_strength(
    rms: str, rep_open: str, visible_primary: str, absent_primary: str
) -> int:
    """
    0–100 score for REPLACEMENT readiness (diagnostics only). Used after absent-primary absence checks.
    """
    if _contains_object_tokens(rep_open, absent_primary) or _contains_object_tokens(
        rms, absent_primary
    ):
        return 0
    if not _replacement_motion_is_meaningful(rms, visible_primary, absent_primary):
        return min(72, 32 + len((rms or "").strip()) // 5)
    joined = f"{rms} {rep_open}".lower()
    score = 80
    if len((rms or "").strip()) >= 95:
        score += 4
    elif len((rms or "").strip()) < 55:
        score -= 10
    concrete = (
        "push",
        "pull",
        "lift",
        "lower",
        "turn",
        "rotate",
        "press",
        "slide",
        "tilt",
        "strike",
        "tap",
        "wrap",
        "hold",
        "grasp",
        "guide",
        "contact",
        "pulls",
        "pushes",
    )
    if sum(1 for w in concrete if w in joined) >= 2:
        score += 6
    vague = (
        "symbolic",
        "metaphor",
        "abstract",
        "suggests",
        "implies",
        "represents an idea",
        "emotional resonance",
        "ambient mood",
    )
    if any(v in joined for v in vague):
        score -= 15
    return int(max(0, min(100, score)))


def _pair_retry_key(oa: str, ob: str) -> str:
    a = _normalize_object_identifier_for_compare(oa)
    b = _normalize_object_identifier_for_compare(ob)
    return "|".join(sorted((a, b)))


def _pair_is_too_similar_to_rejected(
    oa: str, ob: str, rejected_pairs: Set[str]
) -> Tuple[bool, str, str]:
    """
    Reject exact/near-identical conceptual pairs across retries to avoid looped weak candidates.
    Returns (too_similar, reason, prior_pair) where prior_pair is the rejected key that triggered the block, else "".
    """
    key = _pair_retry_key(oa, ob)
    if key in rejected_pairs:
        return True, "exact_pair_repeat", key
    cur_tokens = set(key.replace("|", " ").split())
    weak_cur = cur_tokens & _SBS_WEAK_FAMILY_RETRY_TOKENS
    for prev in rejected_pairs:
        prev_tokens = set(prev.replace("|", " ").split())
        if weak_cur and (weak_cur & prev_tokens):
            return True, "weak_sbs_family_token_overlap", prev
    if not cur_tokens:
        return False, "", ""
    for prev in rejected_pairs:
        prev_tokens = set(prev.replace("|", " ").split())
        if not prev_tokens:
            continue
        overlap = len(cur_tokens & prev_tokens) / float(max(1, len(cur_tokens | prev_tokens)))
        if overlap >= 0.55:
            return True, "near_identical_family_repeat", prev
    return False, "", ""


def _side_by_side_motion_is_meaningful(script: str, object_a: str, object_b: str) -> bool:
    """
    SIDE_BY_SIDE: clear physical A↔B interaction (does not need to quote the advertisingPromise).
    Reject comparison-only, decorative-only, or abstract/symbolic-only motion.
    """
    s = (script or "").strip().lower()
    if len(s) < 45:
        return False
    if not _contains_object_tokens(s, object_a):
        return False
    if not _contains_object_tokens(s, object_b):
        return False
    if not any(v in s for v in _REPLACEMENT_MOTION_REQUIRED_VERBS):
        return False
    forbidden = (
        "side by side only",
        "comparison only",
        "no interaction",
        "idle",
        "ambient",
        "aesthetic",
        "beauty shot",
        "independent movement",
    )
    if any(x in s for x in forbidden):
        return False
    abstract_only = (
        "purely symbolic",
        "symbol only",
        "abstract metaphor",
        "represents the idea",
        "suggests meaning without",
        "emotional subtext only",
    )
    if any(x in s for x in abstract_only):
        return False
    return True


# SIDE_BY_SIDE_SHAPE_ENFORCEMENT: tall vertical-axis + top-mass silhouettes (tree, umbrella, lamppost, …)
_VERTICAL_AXIS_OBJECT_LEXEMES: Tuple[str, ...] = (
    "umbrella",
    "parasol",
    "tree",
    "oak",
    "pine",
    "palm",
    "birch",
    "spruce",
    "fir",
    "cedar",
    "willow",
    "elm",
    "maple",
    "cypress",
    "redwood",
    "sapling",
    "christmas tree",
    "mushroom",
    "toadstool",
    "lamppost",
    "lamp post",
    "lamp-post",
    "street lamp",
    "streetlight",
    "street light",
    "flagpole",
    "flag pole",
    "obelisk",
    "minaret",
    "spire",
    "lighthouse",
    "cactus",
    "rocket",
    "totem",
)


def _object_label_vertical_axis_top_mass(label: str) -> bool:
    """True when the object label suggests a vertical stem + upper mass (morphological vertical-axis read)."""
    low = (label or "").lower()
    return any(tok in low for tok in _VERTICAL_AXIS_OBJECT_LEXEMES)


_SIDE_BY_SIDE_VERTICAL_OPENING_ENFORCEMENT = (
    "SIDE_BY_SIDE_SHAPE_ENFORCEMENT: Both primaries are upright, vertically aligned on parallel axes, "
    "comparable height and scale, same vertical orientation; mass reads toward the top along a shared vertical axis "
    "(e.g. tree trunk vs umbrella handle / lamppost stem)."
)

_SIDE_BY_SIDE_VERTICAL_MOTION_ENFORCEMENT = (
    "Preserve upright vertical alignment for both primaries; motion stays subtle and must not tip either object "
    "off-vertical or break the shared-axis morphological comparison."
)


def _runway_vertical_axis_hard_constraints_english() -> str:
    """Hard Runway text policy when shapeAlignment=vertical_axis (SIDE_BY_SIDE)."""
    return (
        " HARD CONSTRAINT (upright vertical subjects): Both primary objects remain fully upright and vertical, "
        "sharing parallel vertical axes at comparable scale. "
        "The umbrella stands upright, fully vertical, like a tree trunk; the handle is straight and vertical; "
        "the canopy sits on top like a tree canopy. "
        "Forbidden: no leaning umbrella; no umbrella lying on the ground; no diagonal umbrella orientation; "
        "no tilt; do not lean; do not place either primary on the ground for support. "
        "Do not tilt, do not lean, do not place on ground."
    )


# SIDE_BY_SIDE: mandatory smooth half-orbit camera around the paired composition (never optional; not object-motion-only).
_SBS_HALF_ORBIT_CAMERA = "half_orbit"

_SBS_HALF_ORBIT_PLAN_DESCRIPTION = (
    "Camera: smooth half-orbit (half-circle path) around both side-by-side primaries as one stable paired composition; "
    "orbit intent is around the pair together; both remain in frame; move is smooth, medium-slow, readable, centered—not aggressive. "
    "Optional tiny subject motion is minor only and does not replace the camera half-orbit."
)

_SBS_HALF_ORBIT_RUNWAY_APPEND = (
    " MANDATORY CAMERA (SIDE_BY_SIDE — NOT OPTIONAL): The two primaries are side by side as ONE paired composition. "
    "The camera MUST perform a smooth half-orbit—a controlled half-circle path around that pair—so the viewer sees the pairing "
    "from continuously changing angles across the entire shot (calm advertising reveal in 3D). "
    "FORBIDDEN: static camera; nearly static camera; relying only on micro-flicker or tiny object motion without this orbit; "
    "dramatic fast moves; chaotic spin; full 360; handheld shaky cam; losing either object out of frame; cuts; scene changes. "
    "Small object/subject motion may appear as minor motion only—it must NOT replace the mandatory half-orbit. "
    "Half-orbit is smooth, medium-slow, stable, centered on the pair; both objects stay visible and readable throughout."
)


def _runway_side_by_side_half_orbit_preamble() -> str:
    """Opening clause for Runway SIDE_BY_SIDE scene text."""
    return (
        "MANDATORY: one continuous shot—the camera executes a smooth half-orbit (half-circle) around the paired side-by-side composition "
        "so the view angle changes continuously; not a locked-off still. "
    )


def _runway_side_by_side_interaction_half_orbit_focus() -> str:
    """Motion paragraph for start-frame / interaction SIDE_BY_SIDE prompts."""
    return (
        "MANDATORY: video motion is a smooth half-orbit (half-circle camera path) around the two side-by-side subjects as one pair—"
        "continuously changing viewing angle; both stay fully in frame and readable. "
        "Tiny subject motion is optional minor motion only; do not substitute it for the orbit. "
        "No static camera, no morph, no swap, no cuts. "
    )


def _parse_viewer_clarity_ok(raw: Any) -> Optional[bool]:
    """True / False from JSON bool or common string forms; None = missing or invalid."""
    if raw is True:
        return True
    if raw is False:
        return False
    s = str(raw or "").strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


def _norm_enum(value: Any, allowed: List[str], default: str) -> str:
    v = (str(value) if value is not None else "").strip()
    return v if v in allowed else default


def _norm_ab_side(value: Any, default: str) -> str:
    v = (str(value) if value is not None else "").strip().upper()
    return v if v in ("A", "B") else default


def _fuzzy_replacement_direction(raw: Any) -> str:
    """Map common model variants to B_replaces_A | A_replaces_B | ''."""
    s = str(raw or "").strip()
    if not s:
        return ""
    if s in ("B_replaces_A", "A_replaces_B"):
        return s
    if re.search(r"\bB\s+REPLAC(?:ES|ING)\s+A\b", s, re.I):
        return "B_replaces_A"
    if re.search(r"\bA\s+REPLAC(?:ES|ING)\s+B\b", s, re.I):
        return "A_replaces_B"
    u = re.sub(r"\s+", "_", s.upper()).replace("-", "_")
    if u == "B_REPLACES_A":
        return "B_replaces_A"
    if u == "A_REPLACES_B":
        return "A_replaces_B"
    return ""


def _fuzzy_headline_decision_raw(raw: Any) -> str:
    """Normalize common variants before strict enum check."""
    s = str(raw or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in ("no_headline", "noheadline", "none", "without_headline", "no_headline_text", "headline_none"):
        return "no_headline"
    if s in ("include_product_name", "include_product", "with_product_name"):
        return "include_product_name"
    if s in ("product_name_only", "product_only", "name_only"):
        return "product_name_only"
    return str(raw or "").strip()


def _norm_video_visual_mode(raw: Any) -> Optional[str]:
    """REPLACEMENT | SIDE_BY_SIDE, or None if invalid."""
    s = re.sub(r"\s+", "_", str(raw or "").strip().lower())
    s = s.replace("-", "_")
    if s in ("replacement", "replace"):
        return "REPLACEMENT"
    if s in ("side_by_side", "sidebyside", "side_by_side_mode", "sxs"):
        return "SIDE_BY_SIDE"
    return None


def _is_side_by_side_plan(plan: Dict[str, Any]) -> bool:
    return _norm_video_visual_mode(plan.get("videoVisualMode")) == "SIDE_BY_SIDE"


# snake_case / alternate keys from some models → camelCase
_PLAN_KEY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("product_name_resolved", "productNameResolved"),
    ("advertising_promise", "advertisingPromise"),
    ("object_a", "objectA"),
    ("object_b", "objectB"),
    ("morphological_reason", "morphologicalReason"),
    ("promise_reason", "promiseReason"),
    ("replacement_direction", "replacementDirection"),
    ("preserved_background_from", "preservedBackgroundFrom"),
    ("short_replacement_script", "shortReplacementScript"),
    ("headline_decision", "headlineDecision"),
    ("headline_text", "headlineText"),
    ("video_prompt_core", "videoPromptCore"),
    ("replacement_opening_frame_description", "replacementOpeningFrameDescription"),
    ("replacement_motion_script", "replacementMotionScript"),
    ("side_by_side_opening_frame_description", "sideBySideOpeningFrameDescription"),
    ("side_by_side_motion_script", "sideBySideMotionScript"),
    ("object_pair_viewer_clarity_ok", "objectPairViewerClarityOk"),
    ("object_pair_identity_distinct_ok", "objectPairIdentityDistinctOk"),
    ("identity_distinctness_note", "identityDistinctnessNote"),
    ("ab_interaction_type", "abInteractionType"),
    ("shape_alignment", "shapeAlignment"),
    ("side_by_side_camera_motion", "sideBySideCameraMotion"),
    ("side_by_side_camera_motion_description", "sideBySideCameraMotionDescription"),
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
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Final ACE video engine validator (no MODE / REPLACEMENT / SIDE_BY_SIDE).
    Returns (plan, None) or (None, reason_code) for logging.

    Failure reasons (non-exhaustive):
    - missing_replacementMotionScript | missing_sideBySideMotionScript
    - missing_replacementOpeningFrameDescription | missing_sideBySideOpeningFrameDescription
    - missing_advertisingPromise | missing_objectA_or_B
    - object_pair_weak_identity | object_pair_viewer_clarity_not_affirmed | identity_too_close
    - non_physical_object | advertising_promise_not_from_product
    - object_a_not_grounded_in_promise | object_b_not_grounded_in_promise
    - invalid_ab_interaction_type
    - no_additional_interaction | visible_interaction_overlaps_promise
    """
    if not data:
        return None, "missing_replacementMotionScript"

    data = _coerce_plan_keys(data)

    # Visible interaction script (single additional interaction; REPLACEMENT branch kept only for compatibility).
    rms = (data.get("replacementMotionScript") or "").strip()
    sbs_ms = (data.get("sideBySideMotionScript") or "").strip()
    legacy_core = (data.get("videoPromptCore") or "").strip()
    if not rms and legacy_core:
        rms = legacy_core
    rep_open = (data.get("replacementOpeningFrameDescription") or "").strip()
    sbs_open = (data.get("sideBySideOpeningFrameDescription") or "").strip()
    if not rms:
        return None, "missing_replacementMotionScript"
    if not sbs_ms:
        return None, "missing_sideBySideMotionScript"
    if not rep_open:
        return None, "missing_replacementOpeningFrameDescription"
    if not sbs_open:
        return None, "missing_sideBySideOpeningFrameDescription"

    # Discovery vs visible interaction separation.
    disc_summary = (data.get("discoveryInteractionSummary") or "").strip()
    vis_summary = (data.get("visibleAdditionalInteractionSummary") or "").strip()
    vis_motion = (data.get("visibleMotionScript") or "").strip()
    if not vis_summary or not vis_motion:
        logger.info('VIDEO_PLAN_DISCOVERY_INTERACTION_SUMMARY="%s"', disc_summary[:260])
        logger.info('VIDEO_PLAN_VISIBLE_INTERACTION_SUMMARY="%s"', vis_summary[:260])
        logger.info("VIDEO_PLAN_VISIBLE_INTERACTION_DISTINCT_FROM_DISCOVERY=false")
        logger.info("VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_CLASSIC=false")
        logger.info("VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_MEANINGFUL=false")
        logger.info("VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_PROMISE=false")
        logger.info("VIDEO_PLAN_REJECT_REASON=visible_interaction_missing")
        return None, "visible_interaction_missing"

    # advertisingPromise from model; if omitted, allow promiseReason (same model output) as fallback
    apromise = (data.get("advertisingPromise") or "").strip()
    if not apromise:
        apromise = (data.get("promiseReason") or "").strip()
    if not apromise:
        return None, "missing_advertisingPromise"

    if not _advertising_promise_from_product(apromise, product_name, product_description):
        logger.info("VIDEO_PLAN_PROMISE_FROM_PRODUCT=false")
        logger.info("VIDEO_PLAN_REJECT_REASON=advertising_promise_not_from_product")
        return None, "advertising_promise_not_from_product"
    logger.info("VIDEO_PLAN_PROMISE_FROM_PRODUCT=true")

    oa = (data.get("objectA") or "").strip()
    ob = (data.get("objectB") or "").strip()
    if not oa or not ob:
        return None, "missing_objectA_or_B"

    if _object_pair_fails_weak_identity_heuristic(oa, ob):
        logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=false")
        return None, "object_pair_weak_identity"

    oa_phys, _ = _object_label_is_physical_classic(oa)
    ob_phys, _ = _object_label_is_physical_classic(ob)
    logger.info("VIDEO_PLAN_OBJECT_A_PHYSICAL=%s", str(oa_phys).lower())
    logger.info("VIDEO_PLAN_OBJECT_B_PHYSICAL=%s", str(ob_phys).lower())
    q_ok, bad_field, bad_val = _validate_object_pair_physical(oa, ob)
    if not q_ok:
        safe_val = (bad_val or "").replace('"', "'")[:200]
        logger.info('VIDEO_PLAN_REJECT_BAD_OBJECT field=%s value="%s"', bad_field, safe_val)
        logger.info("VIDEO_PLAN_REJECT_REASON=non_physical_object")
        return None, "non_physical_object"

    logger.info(
        "VIDEO_PLAN_OBJECTS_PHYSICAL=%s",
        str(oa_phys and ob_phys).lower(),
    )

    pn = (data.get("productNameResolved") or "").strip() or "Product"

    hd_cand = _fuzzy_headline_decision_raw(data.get("headlineDecision"))
    if hd_cand not in ("include_product_name", "product_name_only", "no_headline"):
        hd_cand = str(data.get("headlineDecision") or "").strip()
    headline_decision = _norm_enum(
        hd_cand,
        ["include_product_name", "product_name_only", "no_headline"],
        "no_headline",
    )
    raw_headline = (data.get("headlineText") or "").strip()
    if headline_decision == "no_headline":
        headline_text = ""
    else:
        headline_text = _word_limit(raw_headline, 7)

    # Legacy fields kept for downstream compatibility (not used for mode decisions).
    repl_raw = data.get("replacementDirection")
    repl_fuzz = _fuzzy_replacement_direction(repl_raw)
    repl = repl_fuzz if repl_fuzz in ("B_replaces_A", "A_replaces_B") else _norm_enum(
        repl_raw, ["B_replaces_A", "A_replaces_B"], "A_replaces_B"
    )

    bg = _norm_ab_side(data.get("preservedBackgroundFrom"), "A")

    clarity_raw = data.get("objectPairViewerClarityOk")
    if _parse_viewer_clarity_ok(clarity_raw) is not True:
        logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=false")
        return None, "object_pair_viewer_clarity_not_affirmed"

    logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=true")

    id_note_raw = (data.get("identityDistinctnessNote") or "").strip()
    if _object_pair_identity_too_close_heuristic(oa, ob):
        logger.info("VIDEO_PLAN_IDENTITY_DISTINCTNESS_OK=false")
        logger.info(
            'VIDEO_PLAN_IDENTITY_DISTINCTNESS_NOTE="%s"',
            "server_near_twin_pair",
        )
        logger.info("VIDEO_PLAN_REJECT_REASON=identity_too_close")
        return None, "identity_too_close"

    id_distinct_raw = data.get("objectPairIdentityDistinctOk")
    if _parse_viewer_clarity_ok(id_distinct_raw) is not True:
        logger.info("VIDEO_PLAN_IDENTITY_DISTINCTNESS_OK=false")
        logger.info(
            'VIDEO_PLAN_IDENTITY_DISTINCTNESS_NOTE="%s"',
            (id_note_raw or "objectPairIdentityDistinctOk_not_true")[:300],
        )
        logger.info("VIDEO_PLAN_REJECT_REASON=identity_too_close")
        return None, "identity_too_close"

    logger.info("VIDEO_PLAN_IDENTITY_DISTINCTNESS_OK=true")
    logger.info(
        'VIDEO_PLAN_IDENTITY_DISTINCTNESS_NOTE="%s"',
        (id_note_raw or "")[:300],
    )

    ga = _object_grounded_in_advertising_promise(oa, apromise)
    gb = _object_grounded_in_advertising_promise(ob, apromise)
    logger.info("VIDEO_PLAN_OBJECT_A_PROMISE_GROUNDED=%s", str(ga).lower())
    logger.info("VIDEO_PLAN_OBJECT_B_PROMISE_GROUNDED=%s", str(gb).lower())
    if not ga:
        logger.info("VIDEO_PLAN_OBJECT_A_SELECTED_FROM_PROMISE=false")
        logger.info("VIDEO_PLAN_OBJECT_B_FOUND=%s", str(gb).lower())
        logger.info("VIDEO_PLAN_REJECT_REASON=object_a_not_grounded_in_promise")
        return None, "object_a_not_grounded_in_promise"
    if not gb:
        logger.info("VIDEO_PLAN_OBJECT_A_SELECTED_FROM_PROMISE=true")
        logger.info("VIDEO_PLAN_OBJECT_B_FOUND=false")
        logger.info("VIDEO_PLAN_REJECT_REASON=object_b_not_grounded_in_promise")
        return None, "object_b_not_grounded_in_promise"

    logger.info("VIDEO_PLAN_OBJECT_A_SELECTED_FROM_PROMISE=true")
    logger.info("VIDEO_PLAN_OBJECT_B_FOUND=true")

    logger.info(
        "VIDEO_OBJECT_SELECTION_SOURCE objectA=%s objectB=%s based_on=advertisingPromise",
        oa,
        ob,
    )

    if planner_deadline_monotonic is not None and time.monotonic() >= planner_deadline_monotonic:
        logger.error("VIDEO_PLAN_DEADLINE_EXCEEDED stage=pre_mode_decision")
        raise VideoPlanningTimeoutError()

    logger.info("VIDEO_PLANNING_FAST_PATH_USED=true")

    # Discovery interaction type (classic | meaningful) – reasoning label only, no mode.
    interaction_type = _parse_norm_ab_interaction_type(data.get("abInteractionType"))
    if interaction_type not in ("classic", "meaningful"):
        logger.info("VIDEO_PLAN_INTERACTION_TYPE=invalid")
        logger.info("VIDEO_PLAN_REJECT_REASON=invalid_ab_interaction_type")
        return None, "invalid_ab_interaction_type"
    logger.info("VIDEO_PLAN_INTERACTION_TYPE=%s", interaction_type)
    logger.info("VIDEO_PLAN_DISCOVERY_INTERACTION=%s", interaction_type)
    logger.info('VIDEO_PLAN_DISCOVERY_INTERACTION_SUMMARY="%s"', disc_summary[:260])

    # Additional visible interaction: must be clear, physical, and not directly express the promise
    # or collapse back into the discovery interaction.
    def _visible_additional_interaction_ok(
        opening: str,
        motion: str,
        oa_label: str,
        ob_label: str,
        promise: str,
        discovery_text: str,
        discovery_type: str,
    ) -> Tuple[bool, str, Dict[str, bool]]:
        txt = f"{opening or ''} {motion or ''}".strip()
        if not txt:
            return False, "no_additional_interaction", {
                "leaks_classic": False,
                "leaks_meaningful": False,
                "leaks_promise": False,
                "distinct_from_discovery": False,
            }
        low = txt.lower()
        if not _contains_object_tokens(txt, oa_label) or not _contains_object_tokens(txt, ob_label):
            return False, "no_additional_interaction", {
                "leaks_classic": False,
                "leaks_meaningful": False,
                "leaks_promise": False,
                "distinct_from_discovery": False,
            }
        # Basic physicality: reuse SIDE_BY_SIDE meaningfulness heuristic as a proxy for readable physical action.
        if not _side_by_side_motion_is_meaningful(txt, oa_label, ob_label):
            return False, "no_additional_interaction", {
                "leaks_classic": False,
                "leaks_meaningful": False,
                "leaks_promise": False,
                "distinct_from_discovery": False,
            }

        # Independence from discovery interaction summary: disallow near-equivalence.
        leaks_classic = False
        leaks_meaningful = False
        disc = (discovery_text or "").strip().lower()
        if disc:
            dtoks = _planning_text_tokens(disc)
            vtoks = _planning_text_tokens(low)
            if dtoks and vtoks:
                overlap_d = len(dtoks & vtoks) / float(max(1, len(dtoks | vtoks)))
                if overlap_d >= 0.5:
                    if discovery_type == "classic":
                        leaks_classic = True
                    else:
                        leaks_meaningful = True

        # Canonical leakage for known classic pairs (e.g. pen+notebook writing).
        leaks_canonical = False
        if discovery_type == "classic":
            a_head = _classic_interaction_head_token(oa_label)
            b_head = _classic_interaction_head_token(ob_label)
            pair = frozenset((a_head, b_head))
            # Simple heuristics for a few canonical pairs; extend conservatively.
            if pair == frozenset({"pen", "notebook"}):
                if any(w in low for w in ("write", "writes", "writing", "note", "notes")):
                    leaks_canonical = True
            elif pair == frozenset({"bee", "flower"}):
                if any(w in low for w in ("nectar", "pollen", "land", "landing", "drink", "drinks", "drinking")):
                    leaks_canonical = True
            elif pair == frozenset({"straw", "cup"}):
                if any(w in low for w in ("sip", "sips", "sipping", "drink", "drinks", "drinking")):
                    leaks_canonical = True

        # Independence from advertising promise: disallow strong token overlap.
        leaks_promise = False
        p = (promise or "").strip().lower()
        if p:
            ptoks = _planning_text_tokens(p)
            vtoks = _planning_text_tokens(low)
            if ptoks and vtoks:
                overlap = len(ptoks & vtoks) / float(max(1, len(ptoks | vtoks)))
                if overlap >= 0.5:
                    leaks_promise = True

        distinct = not (leaks_classic or leaks_meaningful or leaks_canonical or leaks_promise)
        if leaks_promise:
            return False, "visible_interaction_not_distinct", {
                "leaks_classic": leaks_classic or leaks_canonical,
                "leaks_meaningful": leaks_meaningful,
                "leaks_promise": True,
                "distinct_from_discovery": distinct,
            }
        if leaks_classic or leaks_meaningful or leaks_canonical:
            return False, "visible_interaction_matches_discovery", {
                "leaks_classic": leaks_classic or leaks_canonical,
                "leaks_meaningful": leaks_meaningful,
                "leaks_promise": leaks_promise,
                "distinct_from_discovery": distinct,
            }
        return True, "", {
            "leaks_classic": False,
            "leaks_meaningful": False,
            "leaks_promise": False,
            "distinct_from_discovery": True,
        }

    visible_ok, v_reason, leak_flags = _visible_additional_interaction_ok(
        vis_summary,
        vis_motion,
        oa,
        ob,
        apromise,
        disc_summary,
        interaction_type,
    )
    logger.info(
        "VIDEO_PLAN_HAS_ADDITIONAL_INTERACTION=%s",
        str(visible_ok).lower(),
    )
    logger.info(
        "VIDEO_PLAN_VISIBLE_INTERACTION_VALID=%s",
        str(visible_ok).lower(),
    )
    logger.info(
        'VIDEO_PLAN_VISIBLE_INTERACTION_SUMMARY="%s"',
        vis_summary[:260],
    )
    logger.info(
        "VIDEO_PLAN_VISIBLE_INTERACTION_DISTINCT_FROM_DISCOVERY=%s",
        str(leak_flags.get("distinct_from_discovery", False)).lower(),
    )
    logger.info(
        "VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_CLASSIC=%s",
        str(leak_flags.get("leaks_classic", False)).lower(),
    )
    logger.info(
        "VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_MEANINGFUL=%s",
        str(leak_flags.get("leaks_meaningful", False)).lower(),
    )
    logger.info(
        "VIDEO_PLAN_VISIBLE_INTERACTION_LEAKS_PROMISE=%s",
        str(leak_flags.get("leaks_promise", False)).lower(),
    )
    if not visible_ok:
        reason = v_reason or "visible_interaction_not_distinct"
        logger.info("VIDEO_PLAN_REJECT_REASON=%s", reason)
        return None, reason

    # Single canonical additional-interaction script; keep legacy fields for downstream.
    core = f"{vis_motion}{_SBS_HALF_ORBIT_RUNWAY_APPEND}".strip()
    opening_fd = sbs_open
    silhouette_similarity = 0.0
    shape_alignment = ""
    side_by_side_camera_motion = _SBS_HALF_ORBIT_CAMERA
    side_by_side_camera_motion_description = _SBS_HALF_ORBIT_PLAN_DESCRIPTION

    return {
        "productNameResolved": pn,
        "advertisingPromise": apromise,
        "objectA": oa,
        "objectB": ob,
        "morphologicalReason": (data.get("morphologicalReason") or "").strip(),
        "promiseReason": (data.get("promiseReason") or "").strip(),
        "replacementDirection": repl,
        "preservedBackgroundFrom": bg,
        "shortReplacementScript": (data.get("shortReplacementScript") or "").strip(),
        "headlineDecision": headline_decision,
        "headlineText": headline_text,
        "replacementOpeningFrameDescription": rep_open,
        "replacementMotionScript": rms,
        "sideBySideOpeningFrameDescription": sbs_open,
        "sideBySideMotionScript": sbs_ms,
        "videoPromptCore": core,
        "openingFrameDescription": opening_fd,
        # Visual mode fields kept for transport compatibility only (no mode logic).
        "videoVisualMode": "SIDE_BY_SIDE",
        "chosenMode": "SIDE_BY_SIDE",
        "silhouetteSimilarity": silhouette_similarity,
        "interactionScore": float(silhouette_similarity),
        "shapeAlignment": shape_alignment,
        "sideBySideCameraMotion": side_by_side_camera_motion,
        "sideBySideCameraMotionDescription": side_by_side_camera_motion_description,
        "abInteractionType": interaction_type,
    }, None


def video_plan_required_fields_for_runway(plan: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Hard gate before Runway: validated plan dict must include all ACE video structural fields
    and a non-empty end headline (video jobs always use ffmpeg overlay copy).
    Returns (ok, reason_code) with reason_code for logs only when ok is False.
    """
    if not plan:
        return False, "no_plan"
    if not (plan.get("advertisingPromise") or "").strip():
        return False, "missing_advertisingPromise"
    if not (plan.get("objectA") or "").strip():
        return False, "missing_objectA"
    if not (plan.get("objectB") or "").strip():
        return False, "missing_objectB"
    rd = (plan.get("replacementDirection") or "").strip()
    if rd not in ("B_replaces_A", "A_replaces_B"):
        return False, "invalid_replacementDirection"
    hd = (plan.get("headlineDecision") or "").strip()
    if hd not in ("include_product_name", "product_name_only", "no_headline"):
        return False, "invalid_headlineDecision"
    if hd == "no_headline":
        return False, "headlineDecision_no_headline_forbidden_for_video"
    if not (plan.get("headlineText") or "").strip():
        return False, "missing_headlineText"
    if not (plan.get("videoPromptCore") or "").strip():
        return False, "missing_videoPromptCore"
    if plan.get("silhouetteSimilarity") is None:
        return False, "missing_silhouetteSimilarity"
    if (plan.get("abInteractionType") or "").strip() not in ("classic", "meaningful"):
        return False, "missing_or_invalid_abInteractionType"
    vm = _norm_video_visual_mode(plan.get("videoVisualMode"))
    if vm is None:
        return False, "missing_or_invalid_videoVisualMode"
    return True, ""


def _object_pair_digest(oa: str, ob: str) -> str:
    """Short stable hash for diversity debugging (not cryptographic)."""
    raw = f"{(oa or '').strip()}\n{(ob or '').strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_video_job_plan_integrity(plan: Dict[str, Any]) -> None:
    """Structured A/B + promise + headline fields for every validated plan (video job trace)."""
    logger.info(
        'VIDEO_PLAN_INTEGRITY advertisingPromise="%s"',
        (plan.get("advertisingPromise") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY objectA="%s" objectB="%s"',
        plan.get("objectA"),
        plan.get("objectB"),
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


def log_plan_summary(plan: Dict[str, Any]) -> None:
    """Concise server-side log of the chosen plan (no full prompts, no secrets)."""
    logger.info(
        'VIDEO_PLAN productNameResolved="%s"',
        (plan.get("productNameResolved") or "")[:120],
    )
    logger.info(
        "VIDEO_PLAN_SUMMARY objectA=%s objectB=%s abInteractionType=%s",
        plan.get("objectA"),
        plan.get("objectB"),
        plan.get("abInteractionType"),
    )
    logger.info(
        "VIDEO_PLAN pair_digest=%s",
        _object_pair_digest(str(plan.get("objectA") or ""), str(plan.get("objectB") or "")),
    )
    logger.info(
        'VIDEO_PLAN morphologicalReason_preview="%s"',
        (plan.get("morphologicalReason") or "")[:200],
    )


def _reasoning_effort() -> str:
    raw = (os.environ.get("VIDEO_PLANNER_REASONING_EFFORT") or "low").strip().lower()
    return raw if raw in ("low", "medium") else "low"


_VIDEO_PLAN_RETRY_INTERACTION_MAX = int(
    (os.environ.get("VIDEO_PLAN_RETRY_INTERACTION_MAX") or "2").strip() or "2"
)

# When remaining wall-clock budget drops below this, skip further model retries and emit a conservative plan.
_VIDEO_PLAN_EMERGENCY_REMAINING_S = float(
    (os.environ.get("VIDEO_PLANNER_EMERGENCY_REMAINING_S") or "35").strip() or "35"
)

# Tokens that often produced low-silhouette / weak-interaction SIDE_BY_SIDE loops; block reuse after rejection.
_SBS_WEAK_FAMILY_RETRY_TOKENS: FrozenSet[str] = frozenset(
    {
        "megaphone",
        "bullhorn",
        "loudspeaker",
        "rocket",
        "missile",
        "horn",
    }
)

_EMERGENCY_TEXT_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "our",
        "out",
        "get",
        "has",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "that",
        "this",
        "with",
        "from",
        "your",
        "will",
        "have",
        "been",
        "into",
        "more",
        "than",
        "what",
        "when",
        "which",
        "their",
        "there",
        "these",
        "those",
        "each",
        "also",
        "only",
        "very",
        "just",
        "even",
        "such",
        "same",
        "over",
        "most",
        "other",
        "some",
        "about",
        "after",
        "before",
        "under",
        "above",
        "between",
        "through",
        "during",
        "while",
        "where",
        "here",
        "idea",
        "time",
        "life",
        "work",
        "best",
        "next",
        "first",
        "last",
        "help",
        "need",
        "make",
        "made",
        "take",
        "come",
        "give",
        "gives",
        "really",
        "true",
        "full",
        "high",
        "low",
        "wide",
        "deep",
        "long",
        "short",
        "team",
        "user",
        "data",
        "flow",
        "sales",
        "cloud",
        "tool",
        "saas",
        "product",
        "service",
        "digital",
        "business",
        "system",
        "solution",
        "experience",
        "customer",
        "value",
        "growth",
        "speed",
    }
)

# Undirected canonical token pairs for classic-interaction-first fallback search (head tokens).
_CLASSIC_INTERACTION_TOKEN_PAIRS: FrozenSet[FrozenSet[str]] = frozenset(
    {
        frozenset({"bee", "flower"}),
        frozenset({"dog", "bone"}),
        frozenset({"cat", "mouse"}),
        frozenset({"lock", "key"}),
        frozenset({"cup", "straw"}),
        frozenset({"bottle", "cap"}),
        frozenset({"pen", "paper"}),
        frozenset({"pencil", "eraser"}),
        frozenset({"shoe", "lace"}),
        frozenset({"needle", "thread"}),
        frozenset({"hammer", "nail"}),
        frozenset({"brush", "paint"}),
        frozenset({"arrow", "target"}),
        frozenset({"bow", "arrow"}),
        frozenset({"fish", "hook"}),
        frozenset({"bird", "nest"}),
        frozenset({"guitar", "pick"}),
        frozenset({"camera", "lens"}),
        frozenset({"notebook", "pen"}),
        frozenset({"plug", "socket"}),
        frozenset({"toothbrush", "toothpaste"}),
        frozenset({"salt", "pepper"}),
        frozenset({"bread", "knife"}),
        frozenset({"cup", "saucer"}),
        frozenset({"chair", "table"}),
        frozenset({"clock", "battery"}),
        frozenset({"umbrella", "handle"}),
        frozenset({"bucket", "mop"}),
        frozenset({"soap", "sponge"}),
        frozenset({"wine", "cork"}),
        frozenset({"envelope", "stamp"}),
    }
)


def _classic_interaction_head_token(label: str) -> str:
    toks = _object_label_tokens_for_physical_check(label)
    if not toks:
        return ""
    t0 = toks[0].lower()
    aliases = {
        "biro": "pen",
        "ballpoint": "pen",
        "ballpen": "pen",
        "notepad": "notebook",
        "memo": "notebook",
        "dartboard": "target",
        "dart": "arrow",
    }
    return aliases.get(t0, t0)


def _classic_interaction_pair(oa: str, ob: str) -> bool:
    a = _classic_interaction_head_token(oa)
    b = _classic_interaction_head_token(ob)
    if not a or not b or a == b:
        return False
    return frozenset((a, b)) in _CLASSIC_INTERACTION_TOKEN_PAIRS


def _ranked_physical_candidates_from_promise(
    apromise: str, product_name: str, product_description: str
) -> List[str]:
    """Single-token physical labels grounded in the promise, ranked by product+promise relevance."""
    ap = (apromise or "").strip()
    if not ap:
        return []
    pn = (product_name or "").strip()
    pd = (product_description or "").strip()
    blob = f"{ap} {pd} {pn}".strip()
    blob_l = blob.lower()
    ap_l = ap.lower()
    scored: List[Tuple[int, str, str]] = []
    seen_lower: set[str] = set()
    for m in re.finditer(r"[\w\u0590-\u05FF]{3,}", blob, flags=re.UNICODE):
        w = m.group(0).strip()
        if not w:
            continue
        wl = w.lower()
        if wl in _EMERGENCY_TEXT_STOPWORDS or wl in seen_lower:
            continue
        if not _object_grounded_in_advertising_promise(w, ap):
            continue
        ok, _ = _object_label_is_physical_classic(w)
        if not ok:
            continue
        seen_lower.add(wl)
        score = 0
        if wl in ap_l:
            score += 2
        if pn and wl in pn.lower():
            score += 2
        if pd and wl in pd.lower():
            score += 1
        scored.append((score, len(w), w))
    scored.sort(key=lambda t: (-t[0], t[1], t[2].lower()))
    ordered: List[str] = []
    seen2: set[str] = set()
    for _, __, w in scored:
        wl = w.lower()
        if wl not in seen2:
            seen2.add(wl)
            ordered.append(w)
    return ordered


def _search_ordered_fallback_validated_plan(
    parsed_ctx: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    log_context: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Fallback follows the same order as the planner: promise from product → objectA from promise →
    objectB classic-first → objectB meaningful-second → coherent package only via validate.
    Returns (validated_plan_or_None, template_name).
    """
    logger.info("VIDEO_PLAN_FALLBACK_PACKAGE_VALID=false")
    c = _coerce_plan_keys(parsed_ctx or {})
    op0 = (c.get("advertisingPromise") or c.get("promiseReason") or "").strip() or "the advertising promise"
    oa0 = (c.get("objectA") or "").strip() or "object A"
    ob0 = (c.get("objectB") or "").strip() or "object B"
    rep, promise_text_changed = _repaired_advertising_promise_for_product(
        op0, product_name, product_description
    )
    if not _advertising_promise_from_product(rep, product_name, product_description):
        logger.info("VIDEO_PLAN_FALLBACK_BLOCKED reason=advertising_promise_not_from_product")
        logger.info("VIDEO_PLAN_OBJECT_B_FOUND=false")
        return None, ""

    pn = (c.get("productNameResolved") or "").strip() or (product_name or "").strip() or "Product"
    raw_hl = (c.get("headlineText") or "").strip() or pn
    headline_text = _word_limit(raw_hl, 7)

    def build_attempt(oa: str, ob: str, ab_type: str) -> Tuple[Dict[str, Any], str]:
        bucket = _promise_bucket(rep)
        template_name_i, template_body = _fallback_template_for_bucket(bucket)
        sbs_motion = template_body.format(A=oa, B=ob, promise=rep)
        sbs_open = (
            f"Opening intent: {oa} and {ob} appear together in one stable composition, "
            "with immediate clear physical interaction between A and B."
        )
        rep_open = (
            f"Replacement intent: only {oa} is visible on camera; the partner primary is absent from the frame; "
            "background follows preservedBackgroundFrom=A."
        )
        rms = (
            f"The {oa} uses clear physical motion and direct contact with the environment; "
            "the partner primary stays fully off-screen; the change responds visibly using the scene, "
            "supporting the advertising promise."
        )
        attempt_i: Dict[str, Any] = {
            "productNameResolved": pn,
            "advertisingPromise": rep,
            "objectA": oa,
            "objectB": ob,
            "morphologicalReason": (c.get("morphologicalReason") or f"{log_context}_ordered_fallback").strip(),
            "promiseReason": (c.get("promiseReason") or "").strip(),
            "replacementDirection": "A_replaces_B",
            "preservedBackgroundFrom": "A",
            "shortReplacementScript": (c.get("shortReplacementScript") or "").strip(),
            "headlineDecision": "include_product_name",
            "headlineText": headline_text,
            "objectPairViewerClarityOk": True,
            "objectPairIdentityDistinctOk": True,
            "identityDistinctnessNote": f"{log_context}_ordered_fallback",
            "replacementOpeningFrameDescription": rep_open,
            "replacementMotionScript": rms,
            "sideBySideOpeningFrameDescription": sbs_open,
            "sideBySideMotionScript": sbs_motion,
            "abInteractionType": ab_type,
        }
        return attempt_i, template_name_i

    def try_pair(oa: str, ob: str, ab_type: str, phase_log: str) -> Tuple[Optional[Dict[str, Any]], str]:
        if _object_pair_identity_too_close_heuristic(oa, ob):
            return None, ""
        att, tmpl = build_attempt(oa, ob, ab_type)
        plan_i, err = validate_and_normalize_plan(
            att,
            planner_deadline_monotonic=None,
            product_name=product_name,
            product_description=product_description,
        )
        if plan_i:
            logger.info("VIDEO_PLAN_OBJECT_B_SEARCH_PHASE=%s", phase_log)
            logger.info("VIDEO_PLAN_OBJECT_B_FOUND=true")
            logger.info("VIDEO_PLAN_FALLBACK_PACKAGE_VALID=true")
            p_changed_final = (
                unicodedata.normalize("NFC", rep).strip()
                != unicodedata.normalize("NFC", op0).strip()
            )
            oa_f = str(plan_i.get("objectA") or "").strip()
            ob_f = str(plan_i.get("objectB") or "").strip()
            pf, ga, gb, pa, pb = _fallback_abc_grounding_ok(
                str(plan_i.get("advertisingPromise") or rep),
                oa_f,
                ob_f,
                product_name,
                product_description,
            )
            _, tb = _fallback_template_for_bucket(_promise_bucket(rep))
            sbs_probe = tb.format(A=oa_f, B=ob_f, promise=str(plan_i.get("advertisingPromise") or rep))
            sbs_ok = _side_by_side_motion_is_meaningful(sbs_probe, oa_f, ob_f)
            stale_after = bool(promise_text_changed and not (ga and gb and pa and pb))
            _log_fallback_repair_diagnostics(
                original_promise=op0,
                final_promise=str(plan_i.get("advertisingPromise") or rep),
                original_oa=oa0,
                original_ob=ob0,
                final_oa=oa_f,
                final_ob=ob_f,
                promise_text_changed=p_changed_final,
                pf=pf,
                ga=ga,
                gb=gb,
                pa=pa,
                pb=pb,
                stale_objects_after_promise_repair=stale_after,
                sbs_script_ok=sbs_ok,
            )
            return plan_i, tmpl
        if err:
            logger.info("VIDEO_PLAN_FALLBACK_TRY_REJECT reason=%s", err)
        return None, tmpl

    candidates = _ranked_physical_candidates_from_promise(rep, product_name, product_description)
    template_name = ""

    def run_phases(cand: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
        nonlocal template_name
        if len(cand) < 2:
            return None, ""
        logger.info("VIDEO_PLAN_OBJECT_B_SEARCH_PHASE=classic")
        logger.info("VIDEO_PLAN_OBJECT_A_SELECTED_FROM_PROMISE=true")
        for oa in cand:
            for ob in cand:
                if _normalize_object_identifier_for_compare(oa) == _normalize_object_identifier_for_compare(ob):
                    continue
                if not _classic_interaction_pair(oa, ob):
                    continue
                pln, tmpl = try_pair(oa, ob, "classic", "classic")
                template_name = tmpl or template_name
                if pln:
                    return pln, tmpl
        logger.info("VIDEO_PLAN_OBJECT_B_SEARCH_PHASE=meaningful")
        logger.info("VIDEO_PLAN_OBJECT_A_SELECTED_FROM_PROMISE=true")
        for oa in cand:
            for ob in cand:
                if _normalize_object_identifier_for_compare(oa) == _normalize_object_identifier_for_compare(ob):
                    continue
                pln, tmpl = try_pair(oa, ob, "meaningful", "meaningful")
                template_name = tmpl or template_name
                if pln:
                    return pln, tmpl
        return None, ""

    rep_initial = rep
    plan, tmpl = run_phases(candidates)
    if plan:
        return plan, tmpl

    rep_aug = _promise_augment_notebook_pen(rep_initial, (product_name or "").strip())
    if rep_aug != rep_initial and _advertising_promise_from_product(
        rep_aug, product_name, product_description
    ):
        rep = rep_aug
        candidates_aug = _ranked_physical_candidates_from_promise(
            rep_aug, product_name, product_description
        )
        if len(candidates_aug) < 2 and re.search(r"\bnotebook\b", rep_aug.lower()) and re.search(
            r"\bpen\b", rep_aug.lower()
        ):
            ok_nb, _ = _object_label_is_physical_classic("notebook")
            ok_pen, _ = _object_label_is_physical_classic("pen")
            if (
                ok_nb
                and ok_pen
                and _object_grounded_in_advertising_promise("notebook", rep_aug)
                and _object_grounded_in_advertising_promise("pen", rep_aug)
            ):
                candidates_aug = ["notebook", "pen"]
        plan, tmpl = run_phases(candidates_aug)
        if plan:
            return plan, tmpl

    logger.info("VIDEO_PLAN_OBJECT_B_FOUND=false")
    logger.info("VIDEO_PLAN_FALLBACK_BLOCKED reason=ordered_fallback_no_coherent_package")
    return None, template_name


def _emergency_object_pair_from_advertising_text(
    apromise: str, product_description: str, product_name: str
) -> Tuple[str, str]:
    """
    Emergency fallback objects: derived only from advertising promise + product text (no fixed generic pair).
    """
    ap = (apromise or "").strip()
    blob = f"{ap} {(product_description or '').strip()} {(product_name or '').strip()}"
    words: List[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[\w\u0590-\u05FF]{3,}", blob, flags=re.UNICODE):
        w = m.group(0)
        wl = w.lower()
        if wl in _EMERGENCY_TEXT_STOPWORDS or wl in seen:
            continue
        if ap and _object_grounded_in_advertising_promise(w, ap):
            words.append(w)
            seen.add(wl)
        if len(words) >= 2:
            break
    if len(words) < 2 and ap:
        for m in re.finditer(r"[\w\u0590-\u05FF]{3,}", ap, flags=re.UNICODE):
            w = m.group(0)
            wl = w.lower()
            if wl in _EMERGENCY_TEXT_STOPWORDS or wl in seen:
                continue
            words.append(w)
            seen.add(wl)
            if len(words) >= 2:
                break
    if len(words) < 2:
        parts = [x.strip() for x in re.split(r"[,;]", ap) if len(x.strip()) >= 3]
        if len(parts) >= 2:
            words = [parts[0][:48].strip(), parts[1][:48].strip()]
        elif len(parts) == 1 and len(parts[0]) >= 6:
            half = len(parts[0]) // 2
            words = [parts[0][:half].strip(), parts[0][half:].strip()]
        else:
            pn_words = [
                w
                for w in re.findall(
                    r"[\w\u0590-\u05FF]{3,}", (product_name or "").lower(), flags=re.UNICODE
                )
                if w not in _EMERGENCY_TEXT_STOPWORDS
            ]
            pd_words = [
                w
                for w in re.findall(
                    r"[\w\u0590-\u05FF]{3,}", (product_description or "").lower(), flags=re.UNICODE
                )
                if w not in _EMERGENCY_TEXT_STOPWORDS
            ]
            if len(pn_words) >= 2:
                words = [pn_words[0], pn_words[1]]
            elif pn_words and pd_words:
                words = [pn_words[0], pd_words[0]]
            elif len(pd_words) >= 2:
                words = [pd_words[0], pd_words[1]]
            else:
                words = ["primary_subject", "support_subject"]
    return words[0], words[1]


def _promise_bucket(promise: str) -> str:
    p = (promise or "").lower()
    if any(k in p for k in ("speed", "fast", "momentum", "quick", "velocity")):
        return "speed"
    if any(k in p for k in ("precision", "control", "accur", "align", "stable")):
        return "precision"
    if any(k in p for k in ("protect", "safe", "shield", "secure")):
        return "protection"
    if any(k in p for k in ("power", "boost", "ampl", "strong")):
        return "amplification"
    if any(k in p for k in ("clarity", "clear", "reveal", "discover", "uncover")):
        return "clarity"
    if any(k in p for k in ("growth", "uplift", "rise", "lift")):
        return "growth"
    return "generic"


def _fallback_template_for_bucket(bucket: str) -> Tuple[str, str]:
    templates = {
        "speed": (
            "launch_acceleration",
            "{A} launches {B} into a visible acceleration arc, and {B} reacts immediately, expressing: {promise}.",
        ),
        "precision": (
            "guidance_alignment",
            "{A} guides {B} into precise alignment, and {B} reacts by locking into place, expressing: {promise}.",
        ),
        "protection": (
            "shielding_response",
            "{A} protects {B} from a clear visible risk cue, and {B} reacts safely, expressing: {promise}.",
        ),
        "amplification": (
            "boosting_power",
            "{A} amplifies {B} into a visibly stronger state, and {B} reacts with clear output change, expressing: {promise}.",
        ),
        "clarity": (
            "reveal_clarity",
            "{A} triggers a reveal on {B} so hidden details become clear, and {B} reacts immediately, expressing: {promise}.",
        ),
        "growth": (
            "uplift_growth",
            "{A} lifts {B} into a clear upward state change, and {B} responds visibly, expressing: {promise}.",
        ),
        "generic": (
            "cooperative_resolution",
            "{A} and {B} cooperate to resolve a simple visible situation, with clear cause and reaction between them, expressing: {promise}.",
        ),
    }
    return templates.get(bucket, templates["generic"])


def _build_deterministic_side_by_side_plan_from_parsed(
    parsed: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """
    Layer 3+4 deterministic salvage: validate model-shaped plan, then ordered fallback
    (promise from product → A from promise → B classic-first → B meaningful) with full validation.
    Returns (plan_or_none, template_name, guaranteed_delivery_used).
    """
    c = _coerce_plan_keys(parsed or {})
    oa = (c.get("objectA") or "").strip() or "object A"
    ob = (c.get("objectB") or "").strip() or "object B"
    promise = (c.get("advertisingPromise") or c.get("promiseReason") or "").strip() or "the advertising promise"

    bucket = _promise_bucket(promise)
    template_name, template_body = _fallback_template_for_bucket(bucket)
    sbs_motion = template_body.format(A=oa, B=ob, promise=promise)
    sbs_open = (
        f"Opening intent: {oa} and {ob} are visible together in one stable composition, "
        "with immediate meaningful interaction between A and B."
    )
    c["sideBySideMotionScript"] = sbs_motion
    c["sideBySideOpeningFrameDescription"] = sbs_open
    c["advertisingPromise"] = promise
    if not (c.get("productNameResolved") or "").strip():
        c["productNameResolved"] = (product_name or "").strip() or "Product"
    if not (c.get("replacementMotionScript") or "").strip():
        c["replacementMotionScript"] = (
            f"The {oa} uses clear physical motion and direct contact with the environment; "
            "the partner primary stays fully off-screen; the change responds visibly using the scene, "
            "supporting the advertising promise."
        )
    if not (c.get("replacementOpeningFrameDescription") or "").strip():
        c["replacementOpeningFrameDescription"] = (
            f"Replacement intent: only {oa} is visible on camera; the partner primary is absent from the frame; "
            "background follows preservedBackgroundFrom=A."
        )
    if not (c.get("headlineDecision") or "").strip():
        c["headlineDecision"] = "include_product_name"
    if not (c.get("headlineText") or "").strip():
        c["headlineText"] = c["productNameResolved"]
    if not (c.get("objectPairViewerClarityOk") or False):
        c["objectPairViewerClarityOk"] = True
    if not (c.get("objectPairIdentityDistinctOk") or False):
        c["objectPairIdentityDistinctOk"] = True
    if not (c.get("identityDistinctnessNote") or "").strip():
        c["identityDistinctnessNote"] = "deterministic_salvage"

    c["abInteractionType"] = "meaningful"
    plan, _ = validate_and_normalize_plan(
        c, product_name=product_name, product_description=product_description
    )
    if plan:
        return plan, template_name, False

    # Layer 4: same search order as planner; returns only a fully validated coherent package.
    salvage, tmpl = _search_ordered_fallback_validated_plan(
        c,
        product_name=product_name,
        product_description=product_description,
        log_context="deterministic",
    )
    if not salvage:
        return None, template_name, True
    return salvage, tmpl or template_name, True


def _build_emergency_side_by_side_plan(
    parsed: Optional[Dict[str, Any]],
    *,
    product_name: str,
    product_description: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Near-deadline fallback: same ordered package as the planner (promise → A → B classic → B meaningful),
    validated in one path (no independent field repair).
    """
    c = _coerce_plan_keys(parsed or {})
    plan, template_name = _search_ordered_fallback_validated_plan(
        c,
        product_name=product_name,
        product_description=product_description,
        log_context="emergency",
    )
    return plan, template_name or ""


def _finalize_emergency_fallback(
    last_parsed: Optional[Dict[str, Any]],
    *,
    product_name: str,
    product_description: str,
    deadline_monotonic: Optional[float],
    model: str,
) -> Optional[Dict[str, Any]]:
    """Log + build the deadline-aware emergency SIDE_BY_SIDE plan, or None if no valid plan."""
    remaining_s = (
        max(0.0, deadline_monotonic - time.monotonic()) if deadline_monotonic is not None else -1.0
    )
    logger.info("VIDEO_PLAN_FALLBACK_LAYER_ENTERED layer=emergency_deadline_or_timeout")
    logger.info("VIDEO_PLAN_EMERGENCY_FALLBACK_ENTERED remaining_s=%.3f", remaining_s)
    emergency_plan, template_name = _build_emergency_side_by_side_plan(
        last_parsed, product_name=product_name, product_description=product_description
    )
    if not emergency_plan:
        logger.error(
            "VIDEO_PLAN_EMERGENCY_NO_VALID_PLAN template=%s",
            template_name or "(none)",
        )
        return None
    pair_k = _pair_retry_key(
        str(emergency_plan.get("objectA") or ""),
        str(emergency_plan.get("objectB") or ""),
    )
    logger.info("VIDEO_PLAN_EMERGENCY_FALLBACK_CHOSEN pair=%s", pair_k)
    logger.info("VIDEO_PLAN_FALLBACK_TEMPLATE_SELECTED template=%s", template_name)
    logger.info(
        "VIDEO_PLAN_EMERGENCY_FALLBACK_PAIR_SELECTED objectA=%s objectB=%s",
        emergency_plan.get("objectA"),
        emergency_plan.get("objectB"),
    )
    logger.info("VIDEO_PLAN_GUARANTEED_DELIVERY_MODE entered=true")
    logger.info("VIDEO_PLAN_RECOVERED_FROM_VALIDATION_FAILURE=true")
    logger.info("VIDEO_PLAN_EMERGENCY_FALLBACK_OK=true")
    logger.info("VIDEO_PLAN_OK model=%s", model)
    logger.info("VIDEO_PLAN_RESPONSE_OK=true")
    return emergency_plan


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
) -> Optional[Dict[str, Any]]:
    """
    Single planning model call returning a validated plan dict, or None on any failure (no generic video fallback).
    """
    logger.info("VIDEO_PLAN_SEARCH_ORDER=A_then_B_classic_then_meaningful")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_PLAN_FAIL_NO_API_KEY")
        return None

    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    model = _text_model()
    user_block = f"""Product name (may be empty): {product_name or "(empty)"}
Product description:
{product_description}

Locked output language for all user-facing plan fields (from description classification): {lang_name} ({lang})

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
    rejected_promises: List[str] = []
    promise_reject_count = 0
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

    # Hard cap: two planner calls total (fast convergence).
    max_attempts = min(2, 1 + max(0, _VIDEO_PLAN_RETRY_INTERACTION_MAX))
    last_parsed: Optional[Dict[str, Any]] = None
    last_v_err = ""
    rejected_pairs: Set[str] = set()
    for attempt in range(max_attempts):
        if (
            deadline_monotonic is not None
            and last_parsed is not None
            and (deadline_monotonic - time.monotonic()) <= _VIDEO_PLAN_EMERGENCY_REMAINING_S
        ):
            return _return_plan_with_promise_persist(
                _finalize_emergency_fallback(
                    last_parsed,
                    product_name=product_name,
                    product_description=product_description,
                    deadline_monotonic=deadline_monotonic,
                    model=model,
                ),
                product_name=product_name,
                product_description=product_description,
                session_id=session_id,
                fallback_used=True,
            )
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            if last_parsed is not None:
                return _return_plan_with_promise_persist(
                    _finalize_emergency_fallback(
                        last_parsed,
                        product_name=product_name,
                        product_description=product_description,
                        deadline_monotonic=deadline_monotonic,
                        model=model,
                    ),
                    product_name=product_name,
                    product_description=product_description,
                    session_id=session_id,
                    fallback_used=True,
                )
            _planner_deadline_guard(deadline_monotonic, stage=f"retry_{attempt+1}_start")
        logger.info("VIDEO_PLAN_RETRY_STAGE start attempt=%s/%s", attempt + 1, max_attempts)
        retry_tail = ""
        if rejected_pairs:
            rejected_preview = ", ".join(sorted(rejected_pairs)[:4])
            retry_tail = (
                "\n\nRetry constraints:\n"
                "- Do NOT reuse previously rejected object pairs (same or near-identical families): "
                + rejected_preview
                + ".\n"
                "- Choose clearly different object families than the rejected pairs.\n"
            )
        forbid_hist = forbidden_promises_for_prompt(history, 10)
        forbid_extra = [x for x in rejected_promises if x.strip()][-6:]
        promise_addon = build_promise_diversity_addon(
            forbid_hist + forbid_extra,
            angle_seed_for_attempt(attempt, promise_reject_count),
        )
        attempt_input = instructions + "\n\n" + user_block + promise_addon + retry_tail
        logger.info("VIDEO_PLAN_PROMPT_LEN=%s", len(attempt_input))
        try:
            response = client.responses.create(
                model=model,
                input=attempt_input,
                reasoning={"effort": _reasoning_effort()},
            )
        except Exception as e:
            err_type = type(e).__name__
            logger.warning(
                "VIDEO_PLAN_FAIL_MODEL_CALL model=%s err_type=%s err=%s",
                model,
                err_type,
                e,
            )
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None

        try:
            raw = _extract_responses_output_text(response)
            if not raw:
                logger.error("VIDEO_PLAN_FAIL_EMPTY_OUTPUT model=%s", model)
                logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                return None

            _log_output_preview(raw)

            parsed = _parse_json_from_response(raw)
            if not parsed:
                logger.error("VIDEO_PLAN_FAIL_JSON_PARSE model=%s", model)
                logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                return None
            last_parsed = parsed
            parsed_c = _coerce_plan_keys(parsed)
            oa_cand = (parsed_c.get("objectA") or "").strip()
            ob_cand = (parsed_c.get("objectB") or "").strip()
            if oa_cand and ob_cand:
                too_similar, sim_reason, prior_pair = _pair_is_too_similar_to_rejected(
                    oa_cand, ob_cand, rejected_pairs
                )
                if too_similar:
                    cand_k = _pair_retry_key(oa_cand, ob_cand)
                    if sim_reason == "exact_pair_repeat":
                        logger.info("VIDEO_PLAN_REJECTED_PAIR_MEMORY_HIT pair=%s", cand_k)
                    else:
                        logger.info(
                            "VIDEO_PLAN_REJECTED_NEAR_DUPLICATE pair=%s prior_pair=%s reason=%s",
                            cand_k,
                            prior_pair,
                            sim_reason,
                        )
                    logger.info(
                        "VIDEO_PLAN_REJECTED_PAIR_DEDUPE pair=%s reason=%s",
                        cand_k,
                        sim_reason,
                    )
                    logger.info(
                        "VIDEO_PLAN_RETRY attempt=%s reason=pair_too_similar_to_rejected",
                        attempt + 1,
                    )
                    logger.info("VIDEO_PLAN_RETRY_STAGE done attempt=%s result=retry", attempt + 1)
                    continue

            cand_promise = (parsed_c.get("advertisingPromise") or "").strip()
            if cand_promise:
                bad_p, psim, pkind, pdetail = is_promise_too_similar(
                    cand_promise, history, rejected_promises
                )
                if bad_p:
                    if pkind == "concept_match":
                        logger.info(
                            "VIDEO_PROMISE_REJECTED_CONCEPT_MATCH reason=%s similarity=%.3f",
                            pdetail or "concept_buckets",
                            psim,
                        )
                        increment_promise_stat(
                            ph,
                            "conceptual_match_rejections",
                            1,
                            product_name=product_name,
                            product_description=product_description,
                        )
                    else:
                        logger.info(
                            "VIDEO_PROMISE_REJECTED_DUPLICATE similarity=%.3f kind=%s",
                            psim,
                            pkind,
                        )
                        increment_promise_stat(
                            ph,
                            "duplicate_rejections",
                            1,
                            product_name=product_name,
                            product_description=product_description,
                        )
                    rejected_promises.append(cand_promise)
                    promise_reject_count += 1
                    logger.info(
                        "VIDEO_PLAN_RETRY attempt=%s reason=advertising_promise_diversity",
                        attempt + 1,
                    )
                    logger.info(
                        "VIDEO_PLAN_RETRY_STAGE done attempt=%s result=promise_reject",
                        attempt + 1,
                    )
                    continue

            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                return _return_plan_with_promise_persist(
                    _finalize_emergency_fallback(
                        last_parsed,
                        product_name=product_name,
                        product_description=product_description,
                        deadline_monotonic=deadline_monotonic,
                        model=model,
                    ),
                    product_name=product_name,
                    product_description=product_description,
                    session_id=session_id,
                    fallback_used=True,
                )

            plan, v_err = validate_and_normalize_plan(
                parsed,
                planner_deadline_monotonic=deadline_monotonic,
                product_name=product_name,
                product_description=product_description,
            )
            if not plan:
                last_v_err = (v_err or "").strip()
                if oa_cand and ob_cand:
                    rejected_key = _pair_retry_key(oa_cand, ob_cand)
                    rejected_pairs.add(rejected_key)
                    logger.info("VIDEO_PLAN_REJECTED_PAIR_MEMORY_ADD pair=%s reason=%s", rejected_key, last_v_err)
                if (
                    last_v_err
                    in (
                        "side_by_side_interaction_not_meaningful",
                        "replacement_branch_invalid_for_classic",
                        "invalid_ab_interaction_type",
                        "advertising_promise_not_from_product",
                    )
                    and attempt < max_attempts - 1
                ):
                    logger.info(
                        "VIDEO_PLAN_RETRY attempt=%s reason=interaction_not_meaningful",
                        attempt + 1,
                    )
                    logger.info("VIDEO_PLAN_RETRY_STAGE done attempt=%s result=retry", attempt + 1)
                    continue
                if v_err == "identity_too_close":
                    logger.info("VIDEO_PLAN_ABORTED reason=identity_too_close")
                else:
                    logger.error("VIDEO_PLAN_FAIL_VALIDATION reason=%s", v_err or "unknown")
                logger.info("VIDEO_PLAN_REJECT_REASON=%s", last_v_err or "validation_failed")
                logger.info("VIDEO_PLAN_RETRY_STAGE done attempt=%s result=invalid", attempt + 1)
                break

            log_plan_summary(plan)
            logger.info("VIDEO_PLAN_OK model=%s", model)
            logger.info("VIDEO_PLAN_RETRY_STAGE done attempt=%s result=accepted", attempt + 1)
            logger.info("VIDEO_PLAN_RESPONSE_OK=true")
            return _return_plan_with_promise_persist(
                plan,
                product_name=product_name,
                product_description=product_description,
                session_id=session_id,
            )
        except VideoPlanningTimeoutError:
            if last_parsed is not None:
                return _return_plan_with_promise_persist(
                    _finalize_emergency_fallback(
                        last_parsed,
                        product_name=product_name,
                        product_description=product_description,
                        deadline_monotonic=deadline_monotonic,
                        model=model,
                    ),
                    product_name=product_name,
                    product_description=product_description,
                    session_id=session_id,
                    fallback_used=True,
                )
            raise
        except Exception as e:
            logger.warning(
                "VIDEO_PLAN_FAIL_EXCEPTION phase=post_create err_type=%s err=%s",
                type(e).__name__,
                e,
            )
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None

    if last_parsed and last_v_err in (
        "side_by_side_interaction_not_meaningful",
        "replacement_branch_invalid_for_classic",
        "invalid_ab_interaction_type",
        "advertising_promise_not_from_product",
    ):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return _return_plan_with_promise_persist(
                _finalize_emergency_fallback(
                    last_parsed,
                    product_name=product_name,
                    product_description=product_description,
                    deadline_monotonic=deadline_monotonic,
                    model=model,
                ),
                product_name=product_name,
                product_description=product_description,
                session_id=session_id,
                fallback_used=True,
            )
        logger.info("VIDEO_PLAN_FALLBACK_LAYER_ENTERED layer=deterministic_salvage")
        salvage_plan, template_name, guaranteed_mode = _build_deterministic_side_by_side_plan_from_parsed(
            last_parsed,
            product_name=product_name,
            product_description=product_description,
            content_language=content_language,
        )
        logger.info("VIDEO_PLAN_FALLBACK_TEMPLATE_SELECTED template=%s", template_name)
        if salvage_plan:
            if guaranteed_mode:
                logger.info("VIDEO_PLAN_GUARANTEED_DELIVERY_MODE entered=true")
            logger.info("VIDEO_PLAN_RECOVERED_FROM_VALIDATION_FAILURE=true")
            logger.info("VIDEO_PLAN_OK model=%s", model)
            logger.info("VIDEO_PLAN_RESPONSE_OK=true")
            return _return_plan_with_promise_persist(
                salvage_plan,
                product_name=product_name,
                product_description=product_description,
                session_id=session_id,
                fallback_used=True,
            )

    if last_parsed is not None:
        return _return_plan_with_promise_persist(
            _finalize_emergency_fallback(
                last_parsed,
                product_name=product_name,
                product_description=product_description,
                deadline_monotonic=deadline_monotonic,
                model=model,
            ),
            product_name=product_name,
            product_description=product_description,
            session_id=session_id,
            fallback_used=True,
        )
    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
    return None


def fetch_video_plan_o3(
    product_name: str,
    product_description: str,
    content_language: str = "he",
    *,
    session_id: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Fetch and validate plan under one authoritative hard wall-clock deadline.
    On deadline exceeded, raises VideoPlanningTimeoutError (caller must fail the job).
    """
    deadline = time.monotonic() + _VIDEO_PLAN_HARD_SECONDS
    ph = compute_product_hash(product_name, product_description)
    maybe_soft_reset_promise_memory(
        ph, product_name=product_name, product_description=product_description
    )
    logger.info("AD_PROMISE_MEMORY_SESSION_AGNOSTIC=true")
    logger.info("AD_PROMISE_MEMORY_SCOPE global_product_level=true")
    logger.info("AD_PROMISE_MEMORY_PERSISTENT_STORE=true")
    try:
        plan = _fetch_video_plan_o3_sync(
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
        return plan
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
        raise


_RUNWAY_PROMPT_MAX_CHARS = 1000


def _finalize_runway_prompt(headline_prefix: str, body: str) -> Tuple[str, bool]:
    """
    Join optional prefix + body. If over max length, truncate body so a leading prefix survives when present.
    Runway prompts do not include headline burn-in (headline is applied server-side after generation).
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


def _build_runway_prompt_compact_fallback(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Shorter ACE→Runway bridge if the detailed builder fails; keeps prior behavior."""
    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()

    if _is_side_by_side_plan(plan):
        parts = [
            "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
            f"Side-by-side: both {oa} and {ob} as one pair; MANDATORY smooth half-orbit camera around the pair per Action—not static.",
            f"Scene: {core}" if core else "",
            f"Beat: {script}" if script else "",
        ]
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            parts.append(_runway_vertical_axis_hard_constraints_english())
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    else:
        parts = [
            "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
            f"Scene: {core}" if core else "",
            f"Replacement: {script}" if script else "",
        ]
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def _replacement_visible_and_absent(plan: Dict[str, Any]) -> Tuple[str, str]:
    """Which primary is on camera vs absent for REPLACEMENT framing (from replacementDirection)."""
    rd = (plan.get("replacementDirection") or "").strip()
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    if rd == "B_replaces_A":
        return ob, oa
    return oa, ob


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Compact ACE→Runway prompt. Headline rule is first when present so truncation never drops it.
    """
    rd = (plan.get("replacementDirection") or "").strip()
    if rd not in ("B_replaces_A", "A_replaces_B"):
        raise ValueError("invalid replacementDirection")

    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    if not oa or not ob:
        raise ValueError("missing object A or B")

    pbg = (plan.get("preservedBackgroundFrom") or "A").strip().upper()
    if pbg not in ("A", "B"):
        raise ValueError("invalid preservedBackgroundFrom")

    promise = (plan.get("advertisingPromise") or "").strip()
    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    if not core:
        raise ValueError("missing videoPromptCore")

    if _is_side_by_side_plan(plan):
        ofd = (plan.get("openingFrameDescription") or "").strip()
        open_block = f"Opening intent: {ofd} " if ofd else ""
        motion_pre = _runway_side_by_side_half_orbit_preamble()
        scene = (
            f"{open_block}"
            f"SIDE_BY_SIDE (no replacement): single continuous shot; {motion_pre}"
            f"tight unified composition; {oa} and {ob} "
            f"both visible from the first frame, close together or slightly overlapping, same world and scale; "
            f"promise: {promise}. No morphing, swapping, disappearance, or cuts. "
            f"Action: {core}"
        )
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            scene += _runway_vertical_axis_hard_constraints_english()
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    else:
        vis, absent = _replacement_visible_and_absent(plan)
        scene = (
            f"REPLACEMENT: only {vis} is visible on camera; {absent} must never appear in-frame. "
            f"Background follows preservedBackgroundFrom={pbg}. "
            f"Motion uses only {vis} plus environment to express the replacement relationship between the two primaries; "
            f"no extra stand-in objects. promise: {promise}. One smooth shot, no cuts. Action: {core}"
        )
    if script:
        scene += f" Beat: {script}"
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
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=half_orbit")
    return out


def _build_runway_interaction_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Runway promptText when promptImage is a pre-generated ACE start frame: motion / interaction only
    (replacement already visible in frame 1).
    """
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    if not oa or not ob:
        raise ValueError("missing object A or B")

    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("visibleMotionScript") or plan.get("shortReplacementScript") or "").strip()
    if not core:
        raise ValueError("missing videoPromptCore")

    motion_focus = _runway_side_by_side_interaction_half_orbit_focus()
    scene = (
        f"The first frame is supplied as the start image; it already shows {oa} and {ob} together, "
        f"both clearly visible and balanced. {motion_focus}"
        f"Action: {core}"
    )
    if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
        scene += _runway_vertical_axis_hard_constraints_english()
        logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    if script:
        scene += f" Beat: {script}"
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
    """Shorter interaction-only bridge if the detailed interaction builder fails."""
    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    rd = (plan.get("replacementDirection") or "").strip()
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()

    if _is_side_by_side_plan(plan):
        motion = (
            f"Both {oa} and {ob} side by side; MANDATORY smooth half-orbit camera around the pair per Action; "
            f"motion only; start frame supplied."
        )
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            motion += " " + _runway_vertical_axis_hard_constraints_english()
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    elif rd in ("B_replaces_A", "A_replaces_B"):
        vis, absent = _replacement_visible_and_absent(plan)
        motion = (
            f"Replacement motion: only {vis} moves on camera; {absent} absent; "
            f"no stand-in objects beyond environment; motion only; start frame supplied."
        )
    else:
        motion = "Motion only; start frame supplied."

    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        motion,
        f"Action: {core}" if core else "",
        f"Beat: {script}" if script else "",
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
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=half_orbit")
    return out
