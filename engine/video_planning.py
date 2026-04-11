"""
ACE video engine — o3-pro planning layer (isolated from image /preview /generate).

Produces a structured plan for Runway prompt assembly. On failure, video jobs abort (no generic Runway prompt).
Future: richer ACE video engine (e.g. two outputs); keep this module inspectable and logged.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from engine.video_language import normalize_video_content_language, video_language_display_name

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
OUTPUT FORMAT (strict)
- Return ONE JSON object only.
- Use the exact camelCase keys below. Do not omit required keys; use "" only where allowed.
- Do NOT wrap in markdown code fences. Do NOT add prose before or after the JSON.

Field notes:
- morphologicalReason: Why A/B match in whole-object form + why viewer instantly reads “B replaced A” (iconic identity).
- objectPairViewerClarityOk: boolean true only if both primaries are iconic standalone objects and replacement is legible; if not, pick another pair—never false to bypass.

Required keys (all strings except where noted):
{
  "productNameResolved": string,
  "advertisingPromise": string,
  "objectA": string,
  "objectA_secondary": string,
  "objectB": string,
  "objectB_secondary": string,
  "morphologicalReason": string,
  "promiseReason": string,
  "replacementDirection": "B_replaces_A" or "A_replaces_B",
  "preservedBackgroundFrom": "A" or "B",
  "preservedSecondaryFrom": "A" or "B",
  "shortReplacementScript": string,
  "headlineDecision": "include_product_name" or "product_name_only" or "no_headline",
  "headlineText": string,
  "videoPromptCore": string,
  "objectPairViewerClarityOk": boolean
}
"""


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    return f"""You are the ACE video planning engine.

LANGUAGE
- Job: {lang_name} ({lang}), from product description only (Hebrew or English). advertisingPromise, headlineText, shortReplacementScript, morphologicalReason, promiseReason: primarily {lang_name}. Loanwords/brands (AI, SaaS, etc.) OK. objectA/objectB/objectA_secondary/objectB_secondary: short English nouns allowed for morphology.

PIPELINE (fixed)
- Frame 1 already shows replacement: B in A’s role, A’s background, A’s secondary, A’s pose; motion continues B with A’s secondary. Choose objectA and objectB using PHOTOGRAPHIC SIMILARITY (below)—the only selection rule for visual fit; viewer must feel B belongs in A’s place before any verbal explanation.

PHOTOGRAPHIC SIMILARITY (object pair selection — single rule)
Object A and Object B must be selected based on PHOTOGRAPHIC SIMILARITY.

Definition of PHOTOGRAPHIC SIMILARITY (strict priority order):

1) SHAPE / OUTLINE (highest priority)
   - Objects must have highly similar silhouettes.
   - Replacement must remain immediately legible.

2) COLOR
   - Prefer similar dominant color families.

3) MATERIAL / TEXTURE
   - Prefer similar surface qualities (metal, plastic, matte, glossy, etc.).

4) PHOTOGRAPHIC FEEL
   - Lighting, finish, and overall visual character should feel coherent.

REPLACEMENT CLARITY (hard)
- After replacement, the viewer must clearly understand: “Object B has replaced Object A” (or the chosen direction per replacementDirection). If that clarity is weak, reject the pair.

OBJECTS
- Iconic concrete physical primaries; no brands/logos/text-as-object/vague environments. Secondary: separate concrete prop in the same everyday scene; not part of the main (no label/packaging-line-as-secondary). A from product description (whole-object grasp, not contour trivia).
- If objectA and objectB differ, objectA_secondary and objectB_secondary MUST differ: each primary gets its own classic contextual prop (not the same noun under underscores vs spaces). Reusing one secondary for both sides when A≠B is invalid.

ICONIC + VIEWER CLARITY (HARD)
- Photographic similarity alone is NOT enough. Reject weak B: packaging subtype, utilitarian generic variant, non-iconic identity, ambiguous swap (“which box?”). Examples to reject: gift box↔shoe box; generic↔shipping/cardboard/product box; generic bag↔shopping bag as weak B; mailer/logistics identity vs named object. VIEWER TEST: instant “B replaced A”; discard weak-B even if photographically close—no shoe-box/mailer shortcuts.

SEARCH
- Compare multiple serious candidates; do not stop at first OK pair. Prefer less clichéd pairs only if photographic fit stays equally strong. B must satisfy (a) strong photographic-similarity fit to A AND (b) promise fit—never trade photographic fit for promise.

REPLACEMENT + PROMISE
- Prefer B_replaces_A (keep A bg + secondary). Else A_replaces_B. Set replacementDirection, preservedBackgroundFrom, preservedSecondaryFrom. advertisingPromise from product description.

HEADLINE (ffmpeg overlay metadata only—never pixels in generated video)
- headlineDecision: include_product_name | product_name_only | no_headline. headlineText: ≤7 words {lang_name}; "" iff no_headline. Invent productNameResolved if name empty. Never instruct rendering headline/name in-frame.

TEXT-FREE VIDEO
- Generated frames: zero readable text (captions, UI, signs, packaging type, logos, watermarks, numbers as graphics). No readable strings in videoPromptCore or shortReplacementScript. Brands only in advertisingPromise/promiseReason/headline metadata.

VIDEO (videoPromptCore)
- Pictorial scene/motion: cinematic commercial, smooth camera, objects/light/materials. No headline in core; no on-screen text/logos.

QUALITY
- shortReplacementScript, morphologicalReason, promiseReason in {lang_name}. morphologicalReason: whole-object iconic clarity; explain how A/B satisfy PHOTOGRAPHIC SIMILARITY (priority order above). videoPromptCore: intuitive replacement over long exposition.

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


def _secondaries_violate_distinct_rule(oa: str, ob: str, oa_sec: str, ob_sec: str) -> bool:
    """True when A≠B but secondaries match after trivial normalization (invalid plan)."""
    if not _primaries_differ_norm(oa, ob):
        return False
    return _normalize_object_identifier_for_compare(oa_sec) == _normalize_object_identifier_for_compare(
        ob_sec
    )


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


# snake_case / alternate keys from some models → camelCase
_PLAN_KEY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("product_name_resolved", "productNameResolved"),
    ("advertising_promise", "advertisingPromise"),
    ("object_a", "objectA"),
    ("object_a_secondary", "objectA_secondary"),
    ("object_b", "objectB"),
    ("object_b_secondary", "objectB_secondary"),
    ("morphological_reason", "morphologicalReason"),
    ("promise_reason", "promiseReason"),
    ("replacement_direction", "replacementDirection"),
    ("preserved_background_from", "preservedBackgroundFrom"),
    ("preserved_secondary_from", "preservedSecondaryFrom"),
    ("short_replacement_script", "shortReplacementScript"),
    ("headline_decision", "headlineDecision"),
    ("headline_text", "headlineText"),
    ("video_prompt_core", "videoPromptCore"),
    ("object_pair_viewer_clarity_ok", "objectPairViewerClarityOk"),
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


def validate_and_normalize_plan(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Return (plan, None) or (None, reason_code) for logging.
    reason_code: missing_videoPromptCore | missing_advertisingPromise | missing_objectA_or_B | missing_object_secondary
    | object_pair_weak_identity | object_pair_viewer_clarity_not_affirmed | secondary_objects_not_distinct
    """
    if not data:
        return None, "missing_videoPromptCore"

    data = _coerce_plan_keys(data)

    core = (data.get("videoPromptCore") or "").strip()
    if not core:
        return None, "missing_videoPromptCore"

    # advertisingPromise from model; if omitted, allow promiseReason (same model output) as fallback
    apromise = (data.get("advertisingPromise") or "").strip()
    if not apromise:
        apromise = (data.get("promiseReason") or "").strip()
    if not apromise:
        return None, "missing_advertisingPromise"

    oa = (data.get("objectA") or "").strip()
    ob = (data.get("objectB") or "").strip()
    if not oa or not ob:
        return None, "missing_objectA_or_B"

    if _object_pair_fails_weak_identity_heuristic(oa, ob):
        logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=false")
        return None, "object_pair_weak_identity"

    oa_sec = (data.get("objectA_secondary") or "").strip()
    ob_sec = (data.get("objectB_secondary") or "").strip()
    if not oa_sec or not ob_sec:
        return None, "missing_object_secondary"

    if _secondaries_violate_distinct_rule(oa, ob, oa_sec, ob_sec):
        logger.info("VIDEO_PLAN_SECONDARY_DISTINCT_OK=false")
        return None, "secondary_objects_not_distinct"

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

    repl_raw = data.get("replacementDirection")
    repl_fuzz = _fuzzy_replacement_direction(repl_raw)
    repl = repl_fuzz if repl_fuzz in ("B_replaces_A", "A_replaces_B") else _norm_enum(
        repl_raw, ["B_replaces_A", "A_replaces_B"], "B_replaces_A"
    )

    bg = _norm_ab_side(data.get("preservedBackgroundFrom"), "A")
    sec = _norm_ab_side(data.get("preservedSecondaryFrom"), "A")

    clarity_raw = data.get("objectPairViewerClarityOk")
    if _parse_viewer_clarity_ok(clarity_raw) is not True:
        logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=false")
        return None, "object_pair_viewer_clarity_not_affirmed"

    logger.info("VIDEO_PLAN_OBJECT_CLARITY_OK=true")

    return {
        "productNameResolved": pn,
        "advertisingPromise": apromise,
        "objectA": oa,
        "objectA_secondary": oa_sec,
        "objectB": ob,
        "objectB_secondary": ob_sec,
        "morphologicalReason": (data.get("morphologicalReason") or "").strip(),
        "promiseReason": (data.get("promiseReason") or "").strip(),
        "replacementDirection": repl,
        "preservedBackgroundFrom": bg,
        "preservedSecondaryFrom": sec,
        "shortReplacementScript": (data.get("shortReplacementScript") or "").strip(),
        "headlineDecision": headline_decision,
        "headlineText": headline_text,
        "videoPromptCore": core,
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
    if not (plan.get("objectA_secondary") or "").strip():
        return False, "missing_objectA_secondary"
    if not (plan.get("objectB") or "").strip():
        return False, "missing_objectB"
    if not (plan.get("objectB_secondary") or "").strip():
        return False, "missing_objectB_secondary"
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
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    oa_sec = (plan.get("objectA_secondary") or "").strip()
    ob_sec = (plan.get("objectB_secondary") or "").strip()
    if _secondaries_violate_distinct_rule(oa, ob, oa_sec, ob_sec):
        logger.info("VIDEO_PLAN_SECONDARY_DISTINCT_OK=false")
        return False, "secondary_objects_not_distinct"
    logger.info("VIDEO_PLAN_SECONDARY_DISTINCT_OK=true")
    return True, ""


def _object_pair_digest(oa: str, ob: str) -> str:
    """Short stable hash for diversity debugging (not cryptographic)."""
    raw = f"{(oa or '').strip()}\n{(ob or '').strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_video_job_plan_integrity(plan: Dict[str, Any]) -> None:
    """Structured A/B/sub-object + promise + headline fields for every validated plan (video job trace)."""
    logger.info(
        'VIDEO_PLAN_INTEGRITY advertisingPromise="%s"',
        (plan.get("advertisingPromise") or "")[:260],
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY objectA="%s" objectA_secondary="%s" objectB="%s" objectB_secondary="%s"',
        plan.get("objectA"),
        plan.get("objectA_secondary"),
        plan.get("objectB"),
        plan.get("objectB_secondary"),
    )
    logger.info(
        "VIDEO_PLAN_INTEGRITY replacementDirection=%s preservedBackgroundFrom=%s preservedSecondaryFrom=%s",
        plan.get("replacementDirection"),
        plan.get("preservedBackgroundFrom"),
        plan.get("preservedSecondaryFrom"),
    )
    logger.info(
        'VIDEO_PLAN_INTEGRITY headlineDecision=%s headlineText="%s"',
        plan.get("headlineDecision"),
        (plan.get("headlineText") or "")[:160],
    )


def log_plan_summary(plan: Dict[str, Any]) -> None:
    """Concise server-side log of the chosen plan (no full prompts, no secrets)."""
    logger.info(
        'VIDEO_PLAN productNameResolved="%s"',
        (plan.get("productNameResolved") or "")[:120],
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


def _fetch_video_plan_o3_sync(
    product_name: str,
    product_description: str,
    content_language: str = "he",
) -> Optional[Dict[str, Any]]:
    """
    Single planning model call returning a validated plan dict, or None on any failure (no generic video fallback).
    """
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
    full_input = instructions + "\n\n" + user_block
    _t = min(30.0, _VIDEO_PLAN_TIMEOUT)
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(connect=_t, read=_VIDEO_PLAN_TIMEOUT, write=_t, pool=_t),
        max_retries=0,
    )

    logger.info("VIDEO_PLAN_REQUEST_START model=%s", model)
    logger.info("VIDEO_PLAN_REQUEST_TIMEOUT_S=%s", _VIDEO_PLAN_TIMEOUT)
    logger.info("VIDEO_PLAN_PROMPT_LEN=%s", len(full_input))

    try:
        response = client.responses.create(
            model=model,
            input=full_input,
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

        plan, v_err = validate_and_normalize_plan(parsed)
        if not plan:
            if v_err == "secondary_objects_not_distinct":
                logger.info("VIDEO_PLAN_ABORTED reason=secondary_objects_not_distinct")
            if v_err == "missing_object_secondary":
                logger.error("VIDEO_PLAN_FAIL_STRUCTURE reason=%s", v_err)
            else:
                logger.error("VIDEO_PLAN_FAIL_VALIDATION reason=%s", v_err or "unknown")
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            return None

        log_plan_summary(plan)
        logger.info("VIDEO_PLAN_OK model=%s", model)
        logger.info("VIDEO_PLAN_RESPONSE_OK=true")
        return plan
    except Exception as e:
        logger.warning(
            "VIDEO_PLAN_FAIL_EXCEPTION phase=post_create err_type=%s err=%s",
            type(e).__name__,
            e,
        )
        logger.info("VIDEO_PLAN_RESPONSE_OK=false")
        return None


def fetch_video_plan_o3(
    product_name: str,
    product_description: str,
    content_language: str = "he",
) -> Optional[Dict[str, Any]]:
    """
    Same as _fetch_video_plan_o3_sync but with a hard wall-clock deadline so the worker cannot hang here.
    On deadline exceeded, raises VideoPlanningTimeoutError (caller must fail the job).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fetch_video_plan_o3_sync, product_name, product_description, content_language)
        try:
            return fut.result(timeout=_VIDEO_PLAN_HARD_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
            logger.error(
                "VIDEO_PLAN_FAIL_TIMEOUT hard_seconds=%s (VIDEO_PLANNER_HARD_TIMEOUT_SECONDS or planner+45)",
                _VIDEO_PLAN_HARD_SECONDS,
            )
            logger.info("VIDEO_JOB_STEP step=plan_video timeout")
            raise VideoPlanningTimeoutError()


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

    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        f"Scene: {core}" if core else "",
        f"Replacement: {script}" if script else "",
    ]
    return _finalize_runway_prompt("", " ".join(p for p in parts if p))


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Compact ACE→Runway prompt. Headline rule is first when present so truncation never drops it.
    """
    rd = (plan.get("replacementDirection") or "").strip()
    if rd not in ("B_replaces_A", "A_replaces_B"):
        raise ValueError("invalid replacementDirection")

    oa = (plan.get("objectA") or "").strip()
    oas = (plan.get("objectA_secondary") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    obs = (plan.get("objectB_secondary") or "").strip()
    if not oa or not ob:
        raise ValueError("missing object A or B")

    pbg = (plan.get("preservedBackgroundFrom") or "A").strip().upper()
    psf = (plan.get("preservedSecondaryFrom") or "A").strip().upper()
    if pbg not in ("A", "B") or psf not in ("A", "B"):
        raise ValueError("invalid preserved side markers")

    promise = (plan.get("advertisingPromise") or "").strip()
    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    if not core:
        raise ValueError("missing videoPromptCore")

    a_setup = f"{oa} + {oas}" if oas else oa
    b_setup = f"{ob} + {obs}" if obs else ob

    if rd == "B_replaces_A":
        scene = (
            f"Start: replacement already visible — {b_setup} in {oa}'s place, bg {pbg}, secondary {psf}, promise: {promise}. "
            f"Motion: {ob} with A's secondary; one smooth shot, no cuts. "
            f"Action: {core}"
        )
    else:
        scene = (
            f"Start: replacement already visible — {a_setup} in {ob}'s place, bg {pbg}, secondary {psf}, promise: {promise}. "
            f"Motion: {oa} with B's secondary; one smooth shot, no cuts. "
            f"Action: {core}"
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
    return out


def _build_runway_interaction_prompt_detailed(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Runway promptText when promptImage is a pre-generated ACE start frame: motion / interaction only
    (replacement already visible in frame 1).
    """
    rd = (plan.get("replacementDirection") or "").strip()
    if rd not in ("B_replaces_A", "A_replaces_B"):
        raise ValueError("invalid replacementDirection")

    oa = (plan.get("objectA") or "").strip()
    oas = (plan.get("objectA_secondary") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    obs = (plan.get("objectB_secondary") or "").strip()
    if not oa or not ob:
        raise ValueError("missing object A or B")

    pbg = (plan.get("preservedBackgroundFrom") or "A").strip().upper()
    psf = (plan.get("preservedSecondaryFrom") or "A").strip().upper()
    if pbg not in ("A", "B") or psf not in ("A", "B"):
        raise ValueError("invalid preserved side markers")

    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    if not core:
        raise ValueError("missing videoPromptCore")

    if rd == "B_replaces_A":
        sec = oas or "the contextual secondary object"
        scene = (
            f"The first frame is supplied as the start image; replacement is already complete. "
            f"Video motion only: {ob} interacts with {sec} in that fixed composition (background side {pbg}, secondary side {psf}); "
            f"subtle natural movement and camera; do not depict transformation, morphing, or {oa} becoming {ob}. "
            f"Action: {core}"
        )
    else:
        sec = obs or "the contextual secondary object"
        scene = (
            f"The first frame is supplied as the start image; replacement is already complete. "
            f"Video motion only: {oa} interacts with {sec} in that fixed composition (background side {pbg}, secondary side {psf}); "
            f"subtle natural movement and camera; do not depict transformation, morphing, or {ob} becoming {oa}. "
            f"Action: {core}"
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


def _build_runway_interaction_prompt_compact_fallback(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Shorter interaction-only bridge if the detailed interaction builder fails."""
    core = (plan.get("videoPromptCore") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()
    rd = (plan.get("replacementDirection") or "").strip()
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    oas = (plan.get("objectA_secondary") or "").strip()
    obs = (plan.get("objectB_secondary") or "").strip()

    if rd == "B_replaces_A":
        motion = f"{ob} with {oas or 'secondary'}; motion only; start frame supplied."
    elif rd == "A_replaces_B":
        motion = f"{oa} with {obs or 'secondary'}; motion only; start frame supplied."
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
    return out
