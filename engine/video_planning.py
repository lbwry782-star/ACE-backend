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

_JSON_KEYS = """
Return ONE JSON object only. No markdown fences. No prose outside JSON.

SUCCESS (all strings, all required):
{
  "productNameResolved": string,
  "objectA": string,
  "objectB": string,
  "interactionSummary": string,
  "interactionScript": string,
  "advertisingPromise": string,
  "headlineText": string
}

FAILURE (no other keys):
{ "planningFailure": "planning_failed_no_valid_interaction" }
"""


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    return f"""ACE video planner — single shot, half-orbit camera around A+B. Job language: {lang_name} ({lang}).
objectA/objectB/interactionSummary/interactionScript: short English. productNameResolved, advertisingPromise, headlineText: {lang_name}. If product name is empty, invent productNameResolved (English if job is English; Hebrew job may use Hebrew or English).

CORE: (1) Choose A from the product. (2) Find B from the product. (3) Accept B only when A↔B creates the advertising promise — do NOT invent the promise first; it is born from the interaction. (4) interactionSummary + interactionScript = the only on-screen interaction. (5) headlineText starts exactly "<productNameResolved>," then up to 7 words total, derived from that interaction.

OBJECTS: Physical, clear, classic; no text/logos/UI/readable content as the idea. TV/phone/billboard OK only as objects, not message surfaces.

ANTI-BANAL: No obvious default use (e.g. pen writing, brush painting). Pick a clear but non-obvious physical interaction.

FAIL: If no valid A+B interaction, return exactly: {{"planningFailure":"planning_failed_no_valid_interaction"}}
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


_VIDEO_PLAN_SCHEMA_VERSION = "single_interaction_v3"

_PLANNER_SELF_FAILURE_CODES: FrozenSet[str] = frozenset({"planning_failed_no_valid_interaction"})


def _object_grounded_in_product_blob(
    object_label: str, product_name: str, product_description: str
) -> bool:
    blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not blob:
        return False
    return _object_grounded_in_advertising_promise(object_label, blob)


_INFERENCE_MODES_LOG: FrozenSet[str] = frozenset(
    {
        "literals",
        "domain_inference",
        "functional_inference",
        "commercial_world_inference",
    }
)

_MIN_REASON_CHARS_FOR_INFERENCE = 20


def _token_overlap_product(reason_or_interaction: str, product_name: str, product_description: str) -> int:
    blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not blob or not (reason_or_interaction or "").strip():
        return 0
    a = _planning_text_tokens(reason_or_interaction)
    b = _planning_text_tokens(blob)
    return len(a & b)


def _reason_grounds_object_in_product(
    reason: str, product_name: str, product_description: str
) -> bool:
    """Reason text ties the object to the product without requiring the object noun in the product text."""
    r = (reason or "").strip()
    if len(r) < _MIN_REASON_CHARS_FOR_INFERENCE:
        return False
    blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not blob:
        return False
    rt = _planning_text_tokens(r)
    shared = rt & _planning_text_tokens(blob)
    if not shared:
        return False
    if any(len(w) >= 4 for w in shared):
        return True
    return len(shared) >= 2


def _interaction_grounds_scene_in_product(
    interaction_blob: str, product_name: str, product_description: str
) -> bool:
    """Interaction/summary text is anchored in the product (so inferred objects are not random)."""
    if not (interaction_blob or "").strip():
        return False
    blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not blob:
        return False
    it = _planning_text_tokens(interaction_blob)
    bt = _planning_text_tokens(blob)
    shared = it & bt
    if not shared:
        return False
    if any(len(w) >= 4 for w in shared):
        return True
    return len(shared) >= 2


def _object_grounded_inferential(
    object_label: str,
    reason: str,
    interaction_blob: str,
    product_name: str,
    product_description: str,
) -> bool:
    """
    Object is grounded in the product via: literal mention, inferential reason, or interaction bridge.
    """
    if _object_grounded_in_product_blob(object_label, product_name, product_description):
        return True
    if _reason_grounds_object_in_product(reason, product_name, product_description):
        return True
    if _contains_object_tokens(interaction_blob, object_label) and _interaction_grounds_scene_in_product(
        interaction_blob, product_name, product_description
    ):
        return True
    return False


def _derive_object_inference_mode(
    strict_a: bool,
    strict_b: bool,
    oa_r: str,
    ob_r: str,
    product_name: str,
    product_description: str,
) -> str:
    """Server-side label for how object labels relate to literal product wording (logging)."""
    literal_count = int(bool(strict_a)) + int(bool(strict_b))
    if literal_count >= 2:
        return "literals"
    if literal_count == 1:
        return "functional_inference"
    ra = _token_overlap_product(oa_r, product_name, product_description)
    rb = _token_overlap_product(ob_r, product_name, product_description)
    if ra >= 2 and rb >= 2:
        return "domain_inference"
    return "commercial_world_inference"


def _interaction_covers_both_objects(script: str, summary: str, oa: str, ob: str) -> bool:
    blob = f"{summary or ''} {script or ''}".strip()
    if not blob:
        return False
    return _contains_object_tokens(blob, oa) and _contains_object_tokens(blob, ob)


_UI_TEXT_FORBIDDEN = re.compile(
    r"\b(ui|ux|gui|interface|app\s+screen|webpage|website|caption|subtitle|"
    r"lower third|watermark|qr\s*code|barcode|logo|brand\s+mark|packaging\s+text|"
    r"label\s+text|readable\s+text|on-?screen\s+text)\b",
    re.I,
)

# Idea depends on a surface meant to carry text/graphics/UI, not the object-as-prop alone.
_MESSAGE_SURFACE_FORBIDDEN = re.compile(
    r"\b("
    r"poster\s+with\s+text|readable\s+poster|read(ing)?\s+the\s+(poster|billboard|sign|banner)|"
    r"billboard\s+(text|message|copy|ad)|text\s+on\s+(the\s+)?(screen|display|phone|monitor|tv|television)|"
    r"notification\s+text|sms\s+text|subtitle|closed\s+captions|"
    r"swipe(s)?\s+the\s+ui|app\s+interface|home\s+screen\s+icons|"
    r"printed\s+(headline|slogan|copy|message)|"
    r"message\s+on\s+screen|display(s)?\s+readable|readable\s+content\s+on|"
    r"qr\s*code\s+scanned\s+for\s+message|scan\s+the\s+ad"
    r")\b",
    re.I,
)

_BANAL_OBJ_TOKEN_ALIASES: Dict[str, str] = {
    "biro": "pen",
    "ballpen": "pen",
    "ballpoint": "pen",
    "notepad": "notebook",
    "memo": "notebook",
    "journal": "notebook",
    "puppy": "dog",
    "puppies": "dog",
    "paintbrush": "brush",
    "cup": "cup",
    "mug": "cup",
    "glass": "cup",
    "blossom": "flower",
    "bloom": "flower",
}

def _object_tokens_for_banal_check(oa: str, ob: str) -> Set[str]:
    toks: Set[str] = set()
    for lab in (oa, ob):
        for w in re.findall(r"[\w\u0590-\u05FF]{3,}", (lab or "").lower(), flags=re.UNICODE):
            toks.add(_BANAL_OBJ_TOKEN_ALIASES.get(w, w))
    return toks


def _interaction_is_banal_obvious(oa: str, ob: str, interaction_lower: str) -> bool:
    """True if A/B matches a known cliché pair and interaction text matches the default/obvious use."""
    t = _object_tokens_for_banal_check(oa, ob)
    s = interaction_lower
    # Pen + writing surface + write/sketch
    if (t & frozenset({"pen"})) and (t & frozenset({"paper", "page", "sheet", "notebook", "notepad", "parchment"})):
        if any(w in s for w in ("write", "writes", "writing", "written", "scribes", "scribbling", "sketch", "sketches", "doodle", "doodles")):
            return True
    # Brush + canvas/paper + paint
    if (t & frozenset({"brush", "paintbrush"})) and (t & frozenset({"canvas", "easel", "panel", "paper"})):
        if any(w in s for w in ("paint", "paints", "painting", "stroke", "strokes")):
            return True
    # Dog + bone + eat/chew
    if (t & frozenset({"dog", "puppy"})) and ("bone" in t):
        if any(w in s for w in ("chew", "chews", "chewing", "gnaw", "gnaws", "eat", "eats", "eating", "chomp", "lick", "licks")):
            return True
    # Straw + cup + sip/drink
    if ("straw" in t) and (t & frozenset({"cup", "mug", "glass"})):
        if any(w in s for w in ("sip", "sips", "sipping", "drink", "drinks", "drinking", "slurp", "slurps")):
            return True
    # Bee + flower + nectar/pollinate/land
    if ("bee" in t) and (t & frozenset({"flower", "blossom", "bloom", "petal"})):
        if any(
            w in s
            for w in (
                "nectar",
                "pollen",
                "pollinate",
                "pollinates",
                "pollinating",
                "landing",
                "lands",
                "land on",
            )
        ):
            return True
    return False


def _interaction_message_surface_dependency(s: str) -> bool:
    return _MESSAGE_SURFACE_FORBIDDEN.search(s or "") is not None


def _interaction_avoids_text_dependency(s: str) -> bool:
    return _UI_TEXT_FORBIDDEN.search(s or "") is None


_GENERIC_HEADLINE_PHRASES: FrozenSet[str] = frozenset(
    {
        "the best",
        "best ever",
        "amazing",
        "incredible",
        "game changer",
        "game-changer",
        "must have",
        "you need",
        "love it",
        "next level",
        "perfect choice",
        "so good",
        "truly great",
        "world class",
        "world-class",
    }
)


def _headline_avoids_generic_praise(headline: str, product_resolved: str) -> bool:
    """False if the part after '<name>,' is empty or mostly generic marketing praise."""
    p = (product_resolved or "").strip()
    h = (headline or "").strip()
    if not p or not h.lower().startswith(p.lower() + ","):
        return True
    rest = h[len(p) + 1 :].strip().lower()
    if len(rest) < 3:
        return False
    if any(g in rest for g in _GENERIC_HEADLINE_PHRASES):
        return False
    return True


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


def _promise_emergent_from_interaction(
    apromise: str,
    interaction_blob: str,
    product_blob: str,
) -> Tuple[bool, str]:
    pt = _planning_text_tokens(apromise)
    it = _planning_text_tokens(interaction_blob)
    bt = _planning_text_tokens(product_blob)
    if len(pt) < 2 or len(it) < 2:
        return False, "planning_failed_promise_not_emergent"
    inter = pt & it
    if not inter:
        return False, "planning_failed_promise_not_emergent"
    overlap_ratio = len(inter) / float(max(1, len(pt)))
    if overlap_ratio < 0.12 and len(inter) < 2:
        return False, "planning_failed_promise_not_emergent"
    prod_only = len(pt & bt) / float(max(1, len(pt)))
    if prod_only >= 0.85 and overlap_ratio < 0.2:
        return False, "planning_failed_promise_not_emergent"
    return True, ""


_GENERIC_PROMISE_SNIPPETS: FrozenSet[str] = frozenset(
    {
        "better results",
        "best quality",
        "great experience",
        "amazing product",
        "the future",
        "work smarter",
        "grow faster",
        "unlock potential",
    }
)


def _advertising_promise_seems_prewritten(apromise: str) -> bool:
    s = (apromise or "").strip().lower()
    if len(s) < 16:
        return True
    for g in _GENERIC_PROMISE_SNIPPETS:
        if g in s:
            return True
    return False


def _headline_prefix_ok(headline: str, product_resolved: str) -> bool:
    p = (product_resolved or "").strip()
    h = (headline or "").strip()
    if not p or not h:
        return False
    return h.startswith(p + ",")


def _headline_word_count_ok(headline: str) -> bool:
    words = [w for w in (headline or "").strip().split() if w]
    return len(words) <= 7


def _headline_derived_from_interaction(headline: str, interaction_blob: str) -> bool:
    ht = _planning_text_tokens(headline)
    it = _planning_text_tokens(interaction_blob)
    if not ht or not it:
        return False
    return bool(ht & it)


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
    " MANDATORY CAMERA (NOT OPTIONAL): The two physical objects form one paired composition in frame. "
    "The camera MUST perform a smooth half-orbit—a controlled half-circle path around that pair—so the viewer sees the interaction "
    "from continuously changing angles across the entire shot (calm advertising reveal in 3D). "
    "FORBIDDEN: static camera; nearly static camera; relying only on micro-flicker or tiny object motion without this orbit; "
    "dramatic fast moves; chaotic spin; full 360; handheld shaky cam; losing either object out of frame; cuts; scene changes. "
    "Small object/subject motion may appear as minor motion only—it must NOT replace the mandatory half-orbit. "
    "Half-orbit is smooth, medium-slow, stable, centered on the pair; both objects stay visible and readable throughout."
)


def _runway_side_by_side_interaction_half_orbit_focus() -> str:
    """Mandatory half-orbit camera around the two-object composition (ACE single interaction)."""
    return (
        "MANDATORY: the camera performs a smooth half-orbit (half-circle path) around the two physical objects as one composition—"
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
    """Legacy REPLACEMENT | SIDE_BY_SIDE, or ACE_SINGLE_INTERACTION; None if invalid."""
    s = re.sub(r"\s+", "_", str(raw or "").strip().lower())
    s = s.replace("-", "_")
    if s in ("replacement", "replace"):
        return "REPLACEMENT"
    if s in ("side_by_side", "sidebyside", "side_by_side_mode", "sxs"):
        return "SIDE_BY_SIDE"
    if s in ("ace_single_interaction", "ace_single"):
        return "ACE_SINGLE_INTERACTION"
    return None


def _is_side_by_side_plan(plan: Dict[str, Any]) -> bool:
    """True when both primaries are visible together (v3 single interaction or legacy SIDE_BY_SIDE)."""
    vm = _norm_video_visual_mode(plan.get("videoVisualMode"))
    if vm == "ACE_SINGLE_INTERACTION":
        return True
    return vm == "SIDE_BY_SIDE"


# snake_case / alternate keys from some models → camelCase
_PLAN_KEY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("product_name_resolved", "productNameResolved"),
    ("advertising_promise", "advertisingPromise"),
    ("object_a", "objectA"),
    ("object_b", "objectB"),
    ("object_a_reason", "objectAReason"),
    ("object_b_reason", "objectBReason"),
    ("interaction_summary", "interactionSummary"),
    ("interaction_script", "interactionScript"),
    ("object_inference_mode", "objectInferenceMode"),
    ("literal_object_count", "literalObjectCount"),
    ("promise_derivation", "promiseDerivation"),
    ("headline_text", "headlineText"),
    ("headline_derivation", "headlineDerivation"),
    ("video_prompt_core", "videoPromptCore"),
    ("replacement_opening_frame_description", "replacementOpeningFrameDescription"),
    ("replacement_motion_script", "replacementMotionScript"),
    ("side_by_side_opening_frame_description", "sideBySideOpeningFrameDescription"),
    ("side_by_side_motion_script", "sideBySideMotionScript"),
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
    ACE video engine v3: one physical A↔B interaction; promise and headline must follow the interaction.
    Returns (plan, None) or (None, reason_code) for fail-fast logging.
    """
    if not data:
        return None, "planning_failed_invalid_objects"

    data = _coerce_plan_keys(data)

    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)

    pn = (data.get("productNameResolved") or "").strip()
    oa = (data.get("objectA") or "").strip()
    ob = (data.get("objectB") or "").strip()
    oa_r = (data.get("objectAReason") or "").strip()
    ob_r = (data.get("objectBReason") or "").strip()
    int_sum = (data.get("interactionSummary") or "").strip()
    int_script = (data.get("interactionScript") or "").strip()
    apromise = (data.get("advertisingPromise") or "").strip()
    pderiv = (data.get("promiseDerivation") or "").strip()
    headline = (data.get("headlineText") or "").strip()
    hderiv = (data.get("headlineDerivation") or "").strip()
    lang_raw = str(data.get("language") or "").strip().lower()
    if lang_raw not in ("he", "en"):
        lang_raw = normalize_video_content_language(content_language)

    if not pn or not oa or not ob:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_invalid_objects")
        return None, "planning_failed_invalid_objects"
    if not int_sum or not int_script:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_no_valid_interaction")
        return None, "planning_failed_no_valid_interaction"
    if not apromise:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_promise_not_emergent")
        return None, "planning_failed_promise_not_emergent"
    if not headline:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_headline_invalid")
        return None, "planning_failed_headline_invalid"

    if planner_deadline_monotonic is not None and time.monotonic() >= planner_deadline_monotonic:
        logger.error("VIDEO_PLAN_DEADLINE_EXCEEDED stage=validate")
        raise VideoPlanningTimeoutError()

    if _object_pair_fails_weak_identity_heuristic(oa, ob):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_invalid_objects")
        return None, "planning_failed_invalid_objects"

    oa_phys, _ = _object_label_is_physical_classic(oa)
    ob_phys, _ = _object_label_is_physical_classic(ob)
    q_ok, bad_field, bad_val = _validate_object_pair_physical(oa, ob)
    if not (oa_phys and ob_phys and q_ok):
        logger.info('VIDEO_PLAN_REJECT_BAD_OBJECT field=%s value="%s"', bad_field, (bad_val or "")[:120])
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_invalid_objects")
        return None, "planning_failed_invalid_objects"

    strict_a = _object_grounded_in_product_blob(oa, product_name, product_description)
    strict_b = _object_grounded_in_product_blob(ob, product_name, product_description)
    literal_count = int(bool(strict_a)) + int(bool(strict_b))
    inter_blob_ground = f"{int_sum} {int_script}"
    derived_mode = _derive_object_inference_mode(
        strict_a, strict_b, oa_r, ob_r, product_name, product_description
    )
    planner_mode = str(data.get("objectInferenceMode") or "").strip()
    logger.info("VIDEO_PLAN_OBJECT_INFERENCE_MODE=%s", derived_mode)
    if planner_mode in _INFERENCE_MODES_LOG and planner_mode != derived_mode:
        logger.info("VIDEO_PLAN_OBJECT_INFERENCE_MODE_PLANNER_INPUT=%s", planner_mode)
    logger.info("VIDEO_PLAN_LITERAL_OBJECT_COUNT=%s", literal_count)

    ga = _object_grounded_inferential(oa, oa_r, inter_blob_ground, product_name, product_description)
    gb = _object_grounded_inferential(ob, ob_r, inter_blob_ground, product_name, product_description)
    logger.info("VIDEO_PLAN_OBJECT_GROUNDEDNESS_OK=%s", str(ga and gb).lower())
    if not ga or not gb:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_invalid_objects")
        return None, "planning_failed_invalid_objects"

    if not _interaction_covers_both_objects(int_script, int_sum, oa, ob):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_no_valid_interaction")
        return None, "planning_failed_no_valid_interaction"
    if not _side_by_side_motion_is_meaningful(f"{int_sum} {int_script}", oa, ob):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_no_valid_interaction")
        return None, "planning_failed_no_valid_interaction"

    inter_low = f"{int_sum} {int_script}".lower()
    banal_hit = _interaction_is_banal_obvious(oa, ob, inter_low)
    msg_surface_hit = _interaction_message_surface_dependency(inter_low)
    logger.info("VIDEO_PLAN_BANAL_INTERACTION=%s", str(banal_hit).lower())
    logger.info("VIDEO_PLAN_MESSAGE_SURFACE_DEPENDENCY=%s", str(msg_surface_hit).lower())
    if msg_surface_hit:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_message_surface_dependency")
        return None, "planning_failed_message_surface_dependency"
    if banal_hit:
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_banal_interaction")
        return None, "planning_failed_banal_interaction"
    if not _interaction_avoids_text_dependency(inter_low):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_no_valid_interaction")
        return None, "planning_failed_no_valid_interaction"

    product_blob = f"{(product_name or '').strip()}\n{(product_description or '').strip()}".strip()
    if not _advertising_promise_from_product(apromise, product_name, product_description):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_promise_not_emergent")
        return None, "planning_failed_promise_not_emergent"
    if _advertising_promise_seems_prewritten(apromise):
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_promise_not_emergent")
        return None, "planning_failed_promise_not_emergent"

    emergent_ok, emergent_reason = _promise_emergent_from_interaction(
        apromise, f"{int_sum} {int_script}", product_blob
    )
    logger.info("VIDEO_PLAN_PROMISE_EMERGENT=%s", str(emergent_ok).lower())
    if not emergent_ok:
        logger.info("VIDEO_PLAN_REJECT_REASON=%s", emergent_reason)
        return None, emergent_reason

    if not _headline_prefix_ok(headline, pn):
        logger.info("VIDEO_PLAN_HEADLINE_PREFIX_OK=false")
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_headline_invalid")
        return None, "planning_failed_headline_invalid"
    if not _headline_word_count_ok(headline):
        logger.info("VIDEO_PLAN_HEADLINE_PREFIX_OK=true")
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_headline_invalid")
        return None, "planning_failed_headline_invalid"
    if not _headline_derived_from_interaction(headline, f"{int_sum} {int_script}"):
        logger.info("VIDEO_PLAN_HEADLINE_PREFIX_OK=true")
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_headline_invalid")
        return None, "planning_failed_headline_invalid"
    if not _headline_avoids_generic_praise(headline, pn):
        logger.info("VIDEO_PLAN_HEADLINE_PREFIX_OK=true")
        logger.info("VIDEO_PLAN_REJECT_REASON=planning_failed_headline_invalid")
        return None, "planning_failed_headline_invalid"
    logger.info("VIDEO_PLAN_HEADLINE_PREFIX_OK=true")

    opening_fd = (
        f"Single continuous shot: {oa} and {ob} are both visible together in one stable composition; "
        "the camera performs a smooth half-orbit around the pair."
    )
    core = f"{int_script}{_SBS_HALF_ORBIT_RUNWAY_APPEND}".strip()

    logger.info('VIDEO_PLAN_INTERACTION_SUMMARY="%s"', int_sum[:260])
    logger.info('VIDEO_PLAN_PROMISE="%s"', apromise[:260])
    logger.info('VIDEO_PLAN_HEADLINE="%s"', headline[:200])

    return {
        "productNameResolved": pn,
        "objectA": oa,
        "objectB": ob,
        "objectAReason": oa_r or "",
        "objectBReason": ob_r or "",
        "interactionSummary": int_sum,
        "interactionScript": int_script,
        "advertisingPromise": apromise,
        "promiseDerivation": pderiv or "",
        "headlineText": headline,
        "headlineDerivation": hderiv or "",
        "language": lang_raw,
        "objectInferenceMode": derived_mode,
        "literalObjectCount": literal_count,
        "objectGroundednessOk": True,
        "headlineDecision": "include_product_name",
        "replacementDirection": "A_replaces_B",
        "preservedBackgroundFrom": "A",
        "shortReplacementScript": "",
        "replacementOpeningFrameDescription": "",
        "replacementMotionScript": int_script,
        "sideBySideOpeningFrameDescription": opening_fd,
        "sideBySideMotionScript": int_script,
        "videoPromptCore": core,
        "openingFrameDescription": opening_fd,
        "videoVisualMode": "ACE_SINGLE_INTERACTION",
        "chosenMode": "ACE_SINGLE_INTERACTION",
        "silhouetteSimilarity": 0.0,
        "interactionScore": 0.0,
        "shapeAlignment": "",
        "sideBySideCameraMotion": _SBS_HALF_ORBIT_CAMERA,
        "sideBySideCameraMotionDescription": _SBS_HALF_ORBIT_PLAN_DESCRIPTION,
    }, None


def video_plan_required_fields_for_runway(plan: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Hard gate before Runway: v3 single-interaction plan + headline overlay fields.
    Returns (ok, reason_code) with reason_code for logs only when ok is False.
    """
    if not plan:
        return False, "no_plan"
    pn = (plan.get("productNameResolved") or "").strip()
    if not pn:
        return False, "planning_failed_invalid_objects"
    if not (plan.get("objectA") or "").strip() or not (plan.get("objectB") or "").strip():
        return False, "planning_failed_invalid_objects"
    if not (plan.get("interactionScript") or "").strip():
        return False, "planning_failed_no_valid_interaction"
    if not (plan.get("videoPromptCore") or "").strip():
        return False, "planning_failed_no_valid_interaction"
    if not (plan.get("advertisingPromise") or "").strip():
        return False, "planning_failed_promise_not_emergent"
    hd = (plan.get("headlineDecision") or "").strip()
    if hd not in ("include_product_name", "product_name_only", "no_headline"):
        return False, "planning_failed_headline_invalid"
    if hd == "no_headline":
        return False, "planning_failed_headline_invalid"
    ht = (plan.get("headlineText") or "").strip()
    if not ht:
        return False, "planning_failed_headline_invalid"
    if not _headline_prefix_ok(ht, pn):
        return False, "planning_failed_headline_invalid"
    if not _headline_word_count_ok(ht):
        return False, "planning_failed_headline_invalid"
    vm = _norm_video_visual_mode(plan.get("videoVisualMode"))
    if vm != "ACE_SINGLE_INTERACTION":
        return False, "planning_failed_invalid_objects"
    return True, ""


def _object_pair_digest(oa: str, ob: str) -> str:
    """Short stable hash for diversity debugging (not cryptographic)."""
    raw = f"{(oa or '').strip()}\n{(ob or '').strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_video_job_plan_integrity(plan: Dict[str, Any]) -> None:
    """Structured A/B + interaction + promise + headline fields for every validated plan (video job trace)."""
    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)
    logger.info(
        'VIDEO_PLAN_INTEGRITY advertisingPromise="%s"',
        (plan.get("advertisingPromise") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY interactionSummary="%s"',
        (plan.get("interactionSummary") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY objectA="%s" objectB="%s"',
        plan.get("objectA"),
        plan.get("objectB"),
    )
    logger.info(
        "VIDEO_PLAN_INTEGRITY objectInferenceMode=%s literalObjectCount=%s objectGroundednessOk=%s",
        plan.get("objectInferenceMode"),
        plan.get("literalObjectCount"),
        plan.get("objectGroundednessOk"),
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
        "VIDEO_PLAN_SUMMARY objectA=%s objectB=%s language=%s",
        plan.get("objectA"),
        plan.get("objectB"),
        plan.get("language"),
    )
    logger.info(
        'VIDEO_PLAN_INTERACTION_SUMMARY="%s"',
        (plan.get("interactionSummary") or "")[:260],
    )
    logger.info(
        "VIDEO_PLAN pair_digest=%s",
        _object_pair_digest(str(plan.get("objectA") or ""), str(plan.get("objectB") or "")),
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
    One strong planner call + strict validation. No salvage, no emergency merge, no deterministic repair.
    Returns (plan, "") on success, or (None, reason_code).
    """
    logger.info("VIDEO_PLAN_SCHEMA_VERSION=%s", _VIDEO_PLAN_SCHEMA_VERSION)
    logger.info("VIDEO_PLAN_SEARCH_ORDER=single_interaction_v3")
    default_fail = "planning_failed_invalid_objects"
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_PLAN_FAIL_NO_API_KEY")
        return None, default_fail

    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise VideoPlanningTimeoutError()

    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    model = _text_model()
    user_block = f"""Product name (may be empty): {product_name or "(empty)"}
Product description:
{product_description}

Language: {lang_name} ({lang}).

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
    forbid_hist = forbidden_promises_for_prompt(history, 4)
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
        return None, default_fail

    try:
        raw = _extract_responses_output_text(response)
        if not raw:
            logger.error("VIDEO_PLAN_FAIL_EMPTY_OUTPUT model=%s", model)
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, default_fail

        _log_output_preview(raw)

        parsed = _parse_json_from_response(raw)
        if not parsed:
            logger.error("VIDEO_PLAN_FAIL_JSON_PARSE model=%s", model)
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, default_fail

        pf_raw = str(parsed.get("planningFailure") or "").strip()
        if pf_raw:
            detail = str(parsed.get("planningFailureDetail") or "").replace('"', "'")[:260]
            code = (
                pf_raw
                if pf_raw in _PLANNER_SELF_FAILURE_CODES
                else "planning_failed_no_valid_interaction"
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
            last_v_err = (v_err or "").strip() or default_fail
            logger.error("VIDEO_PLAN_FAIL_VALIDATION reason=%s", last_v_err)
            logger.info("VIDEO_PLAN_REJECT_REASON=%s", last_v_err)
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None, last_v_err

        log_plan_summary(plan)
        logger.info("VIDEO_PLAN_OK model=%s", model)
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
        return None, default_fail


def fetch_video_plan_o3(
    product_name: str,
    product_description: str,
    content_language: str = "he",
    *,
    session_id: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Fetch and validate plan under one authoritative hard wall-clock deadline.
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
    """Shorter ACE→Runway bridge if the detailed builder fails."""
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    motion = _runway_side_by_side_interaction_half_orbit_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        lang_vis,
        f"Single continuous shot: {oa} and {ob}. {motion}",
        f"Physical interaction: {script}" if script else "",
    ]
    if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
        parts.append(_runway_vertical_axis_hard_constraints_english())
        logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Runway prompt from the validated v3 plan only: objectA, objectB, interactionScript, half-orbit camera.
    No planning prose, promise text, or alternate interaction modes in the model prompt.
    """
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    if not oa or not ob or not script:
        raise ValueError("missing objectA/objectB/interactionScript")

    motion = _runway_side_by_side_interaction_half_orbit_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"Single continuous shot. Two physical objects: {oa} and {ob}. "
        f"{motion}"
        f"Physical interaction (follow exactly): {script}. "
        "No logos, no packaging typography, no on-screen words. Single clean commercial look."
    )
    if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
        body += _runway_vertical_axis_hard_constraints_english()
        logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
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
    Runway promptText when promptImage is a pre-generated ACE start frame: motion / interaction only.
    """
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    if not oa or not ob or not script:
        raise ValueError("missing objectA/objectB/interactionScript")

    motion_focus = _runway_side_by_side_interaction_half_orbit_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    scene = (
        f"{lang_vis} "
        f"The first frame is supplied as the start image; it already shows {oa} and {ob} together, "
        f"both clearly visible and balanced. {motion_focus}"
        f"Physical interaction (follow exactly): {script}"
    )
    if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
        scene += _runway_vertical_axis_hard_constraints_english()
        logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
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
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    lang_vis = _runway_language_visual_constraints(plan)
    motion = (
        f"{lang_vis} "
        f"Start frame supplied; {oa} and {ob} already visible together. "
        f"{_runway_side_by_side_interaction_half_orbit_focus()}"
        f"Physical interaction: {script}."
    )
    if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
        motion += " " + _runway_vertical_axis_hard_constraints_english()
        logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
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
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=half_orbit")
    return out
