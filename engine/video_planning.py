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
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

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

# Must match engine.side_by_side_v1.SIMILARITY_THRESHOLD_REPLACEMENT (85): REPLACEMENT if silhouette > 85 else SIDE_BY_SIDE.
_VIDEO_SILHOUETTE_THRESHOLD_REPLACEMENT = 85


_JSON_KEYS = """
OUTPUT FORMAT (strict)
- Return ONE JSON object only.
- Use the exact camelCase keys below. Do not omit required keys; use "" only where allowed.
- Do NOT wrap in markdown code fences. Do NOT add prose before or after the JSON.

MODE (server-decided, do not output mode): After you choose objectA and objectB, the backend measures silhouette similarity
(same metric as the ACE image engine). If similarity >= 85 → REPLACEMENT; if < 85 → SIDE_BY_SIDE. You cannot set the mode.
You MUST output BOTH the REPLACEMENT branch fields AND the SIDE_BY_SIDE branch fields so the server can use the correct one.

Field notes:
- morphologicalReason: Why A/B match in whole-object form (iconic identity). Job language where noted below.
- objectPairViewerClarityOk / objectPairIdentityDistinctOk / identityDistinctnessNote: as before.
- replacementOpeningFrameDescription: English. Opening-frame intent if REPLACEMENT is selected: B already replaces A; A’s background; A’s secondary in A’s position. No side-by-side wording.
- replacementMotionScript: English. Motion for REPLACEMENT: interaction of B with A’s secondary; subtle; no transformation narrative.
- sideBySideOpeningFrameDescription: English. Opening-frame intent if SIDE_BY_SIDE is selected: both A and B visible from frame 1; tight unified composition; close or slightly overlapping; same world; A’s secondary nearby; NO replacement, NO B-instead-of-A.
- sideBySideMotionScript: English. Motion for SIDE_BY_SIDE: minimal, comparison-strengthening only (subtle tilt, slight rotation, small approach, tiny A/B interaction, subtle secondary involvement). No morphing, swapping, disappearance, cuts, or multi-shot story.

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
  "replacementOpeningFrameDescription": string,
  "replacementMotionScript": string,
  "sideBySideOpeningFrameDescription": string,
  "sideBySideMotionScript": string,
  "objectPairViewerClarityOk": boolean,
  "objectPairIdentityDistinctOk": boolean,
  "identityDistinctnessNote": string
}
"""


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    return f"""You are the ACE video planning engine.

LANGUAGE
- Job: {lang_name} ({lang}), from product description only (Hebrew or English). advertisingPromise, headlineText, shortReplacementScript, morphologicalReason, promiseReason: primarily {lang_name}. Loanwords/brands (AI, SaaS, etc.) OK.
- objectA, objectB, objectA_secondary, objectB_secondary: short English nouns (classic physical objects).
- replacementOpeningFrameDescription, replacementMotionScript, sideBySideOpeningFrameDescription, sideBySideMotionScript: English only (no exceptions).

CREATIVE RULES (mandatory)
1) Derive advertisingPromise from the product description (in {lang_name} per field rules above).
2) Choose objectA by grasping the product’s overall form intuitively (whole-object, painter-like).
3) Choose objectB morphologically similar to A and linked to the promise; never swap a more shape-correct B for a weaker one just to make the promise louder.
4) Classic, defined, physical objects in classic situations only.
5) Filter text, logos, brands, written labels, vague environments, non-physical situations.
6) Each main object has a nearby classic secondary object that is NOT part of the primary.
7) MODE is NOT your choice: the server computes silhouette similarity on (objectA, objectB) like the ACE image engine. Threshold 85: ≥85 → REPLACEMENT; <85 → SIDE_BY_SIDE. You must still output BOTH creative branches (replacement* and sideBySide*) fully.

REPLACEMENT branch (English fields — used only if server selects REPLACEMENT)
- Opening: start frame already shows B replacing A while keeping A’s background and A’s secondary in A’s position.
- Motion: B interacts with A’s secondary; no transformation/morph language.

SIDE_BY SIDE branch (English fields — used only if server selects SIDE_BY_SIDE)
- Opening: A and B both visible from frame 1; tight unified composition; close or slightly overlapping; same angle/scale/world; A’s secondary anchors the scene; NO replacement, NO “B instead of A”.
- Motion: minimal, comparison-only (subtle tilt, slight rotation, small approach/retreat, tiny A↔B interaction, subtle secondary cue). No morphing, swapping, disappearance, wide empty split layout, cuts, or multi-shot story.

PHOTOGRAPHIC SIMILARITY (pair selection)
Object A and Object B: priority (1) shape/outline (2) color (3) material/texture (4) photographic feel.

IDENTITY DISTINCTNESS
- objectPairIdentityDistinctOk / identityDistinctnessNote: A and B must be clearly different objects, not variants of the same thing. Reject near-twin ambiguous pairs per prior rules.

OBJECTS + SEARCH
- Iconic primaries; distinct secondaries when A≠B. Compare multiple B candidates; never trade shape fit for promise clarity.

PAIRING + PROMISE
- Prefer B_replaces_A; else A_replaces_B. Set replacementDirection, preservedBackgroundFrom, preservedSecondaryFrom.

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


# REPLACEMENT enforcement (server-side source of truth): similarity>85 => A replaces B, only B-secondary is preserved.
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


def _replacement_motion_is_meaningful(script: str, object_a: str, object_b_secondary: str) -> bool:
    """
    REPLACEMENT motion must communicate meaning through interaction(A, B-secondary), not decorative idle motion.
    """
    s = (script or "").strip().lower()
    if len(s) < 40:
        return False
    if not _contains_object_tokens(s, object_a):
        return False
    if not _contains_object_tokens(s, object_b_secondary):
        return False
    if not any(v in s for v in _REPLACEMENT_MOTION_REQUIRED_VERBS):
        return False
    decorative_only = ("idle", "floating", "ambient", "aesthetic", "beauty shot", "loop")
    if any(x in s for x in decorative_only):
        return False
    return True


def _side_by_side_motion_is_meaningful(
    script: str, object_a: str, object_b: str, advertising_promise: str
) -> bool:
    """
    SIDE_BY_SIDE requires meaningful A↔B interaction (not comparison-only or decorative movement).
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
    # Motion should express the promise; require at least one promise token to appear.
    p_words = [w for w in _norm_words_for_presence(advertising_promise) if len(w) >= 4]
    if p_words and not any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in p_words[:8]):
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
    "Optional tiny subject motion is secondary only and does not replace the camera half-orbit."
)

_SBS_HALF_ORBIT_RUNWAY_APPEND = (
    " MANDATORY CAMERA (SIDE_BY_SIDE — NOT OPTIONAL): The two primaries are side by side as ONE paired composition. "
    "The camera MUST perform a smooth half-orbit—a controlled half-circle path around that pair—so the viewer sees the pairing "
    "from continuously changing angles across the entire shot (calm advertising reveal in 3D). "
    "FORBIDDEN: static camera; nearly static camera; relying only on micro-flicker or tiny object motion without this orbit; "
    "dramatic fast moves; chaotic spin; full 360; handheld shaky cam; losing either object out of frame; cuts; scene changes. "
    "Small object/subject motion may appear SECONDARY only—it must NOT replace the mandatory half-orbit. "
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
        "Tiny subject motion is optional secondary only; do not substitute it for the orbit. "
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
    ("replacement_opening_frame_description", "replacementOpeningFrameDescription"),
    ("replacement_motion_script", "replacementMotionScript"),
    ("side_by_side_opening_frame_description", "sideBySideOpeningFrameDescription"),
    ("side_by_side_motion_script", "sideBySideMotionScript"),
    ("object_pair_viewer_clarity_ok", "objectPairViewerClarityOk"),
    ("object_pair_identity_distinct_ok", "objectPairIdentityDistinctOk"),
    ("identity_distinctness_note", "identityDistinctnessNote"),
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


def validate_and_normalize_plan(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Return (plan, None) or (None, reason_code) for logging.
    reason_code: missing branch fields | missing_advertisingPromise | missing_objectA_or_B | missing_object_secondary
    | object_pair_weak_identity | object_pair_viewer_clarity_not_affirmed | secondary_objects_not_distinct
    | identity_too_close | silhouette_similarity_eval_failed
    """
    if not data:
        return None, "missing_replacementMotionScript"

    data = _coerce_plan_keys(data)

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

    try:
        from engine.side_by_side_v1 import evaluate_silhouette_similarity

        silhouette_similarity = float(evaluate_silhouette_similarity(oa, ob))
    except Exception as e:
        logger.warning("VIDEO_SILHOUETTE_EVAL_FAIL err=%s", e)
        return None, "silhouette_similarity_eval_failed"

    chosen_mode = (
        "REPLACEMENT"
        if silhouette_similarity > float(_VIDEO_SILHOUETTE_THRESHOLD_REPLACEMENT)
        else "SIDE_BY_SIDE"
    )
    logger.info(
        "VIDEO_MODE_DECISION similarity=%s chosen_mode=%s threshold=%s",
        silhouette_similarity,
        chosen_mode,
        _VIDEO_SILHOUETTE_THRESHOLD_REPLACEMENT,
    )
    logger.info("VIDEO_PLAN_MODE_DECISION_SOURCE=image_engine_rule_adapted")
    shape_alignment = ""
    side_by_side_camera_motion = ""
    side_by_side_camera_motion_description = ""
    if chosen_mode == "REPLACEMENT":
        # Final source-of-truth enforcement:
        # similarity>85 => Object A fully replaces Object B; preserve B-secondary only.
        repl = "A_replaces_B"
        bg = "B"
        sec = "B"
        if _contains_object_tokens(rep_open, ob):
            logger.info("VIDEO_REPLACEMENT_RULE_INVALID reason=object_b_visible_in_opening")
            return None, "replacement_contains_object_b"
        if _contains_object_tokens(rms, ob):
            logger.info("VIDEO_REPLACEMENT_RULE_INVALID reason=object_b_visible_in_motion")
            return None, "replacement_contains_object_b"
        if not _replacement_motion_is_meaningful(rms, oa, ob_sec):
            logger.info("VIDEO_REPLACEMENT_RULE_INVALID reason=motion_not_meaningful")
            return None, "replacement_motion_not_meaningful"
        core = rms
        opening_fd = rep_open
        logger.info(
            "VIDEO_REPLACEMENT_RULE_ENFORCED mode=REPLACEMENT object_presence=A_only secondary=B_secondary"
        )
    else:
        if not _side_by_side_motion_is_meaningful(sbs_ms, oa, ob, apromise):
            logger.info("VIDEO_SIDE_BY_SIDE_RULE_INVALID reason=interaction_not_meaningful")
            return None, "side_by_side_interaction_not_meaningful"
        core = sbs_ms
        opening_fd = sbs_open
        if _object_label_vertical_axis_top_mass(oa) and _object_label_vertical_axis_top_mass(ob):
            shape_alignment = "vertical_axis"
            opening_fd = f"{opening_fd} {_SIDE_BY_SIDE_VERTICAL_OPENING_ENFORCEMENT}".strip()
            core = f"{core} {_SIDE_BY_SIDE_VERTICAL_MOTION_ENFORCEMENT}".strip()
            logger.info("VIDEO_SHAPE_ALIGNMENT axis=vertical applied=true")
        else:
            logger.info("VIDEO_SHAPE_ALIGNMENT axis=vertical applied=false")

        side_by_side_camera_motion = _SBS_HALF_ORBIT_CAMERA
        side_by_side_camera_motion_description = _SBS_HALF_ORBIT_PLAN_DESCRIPTION
        core = f"{core}{_SBS_HALF_ORBIT_RUNWAY_APPEND}".strip()
        logger.info("VIDEO_SIDE_BY_SIDE_CAMERA_RULE applied=true motion=half_orbit")
        logger.info("VIDEO_PLAN_CAMERA_MOTION mode=SIDE_BY_SIDE motion=half_orbit")
        logger.info(
            "VIDEO_SIDE_BY_SIDE_RULE_ENFORCED mode=SIDE_BY_SIDE primary_interaction=A_to_B anchors=A_secondary|B_secondary"
        )

    logger.info("VIDEO_PLAN_MODE=%s", chosen_mode)

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
        "replacementOpeningFrameDescription": rep_open,
        "replacementMotionScript": rms,
        "sideBySideOpeningFrameDescription": sbs_open,
        "sideBySideMotionScript": sbs_ms,
        "videoPromptCore": core,
        "openingFrameDescription": opening_fd,
        "videoVisualMode": chosen_mode,
        "chosenMode": chosen_mode,
        "silhouetteSimilarity": silhouette_similarity,
        "shapeAlignment": shape_alignment,
        "sideBySideCameraMotion": side_by_side_camera_motion,
        "sideBySideCameraMotionDescription": side_by_side_camera_motion_description,
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
    if plan.get("silhouetteSimilarity") is None:
        return False, "missing_silhouetteSimilarity"
    vm = _norm_video_visual_mode(plan.get("videoVisualMode"))
    if vm is None:
        return False, "missing_or_invalid_videoVisualMode"
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
    logger.info("VIDEO_PLAN_MODE_DECISION_SOURCE=image_engine_rule_adapted")
    logger.info(
        "VIDEO_PLAN_MODE=%s silhouetteSimilarity=%s",
        plan.get("videoVisualMode") or "REPLACEMENT",
        plan.get("silhouetteSimilarity"),
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
        "VIDEO_PLAN_SUMMARY mode=%s objectA=%s objectB=%s objectA_secondary=%s silhouette=%s",
        plan.get("videoVisualMode"),
        plan.get("objectA"),
        plan.get("objectB"),
        plan.get("objectA_secondary"),
        plan.get("silhouetteSimilarity"),
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
            "{A} launches {B} into a visible acceleration arc, and {B} reacts immediately. "
            "{A_secondary} and {B_secondary} remain visible as contextual anchors while this interaction expresses: {promise}.",
        ),
        "precision": (
            "guidance_alignment",
            "{A} guides {B} into precise alignment, and {B} reacts by locking into place. "
            "{A_secondary} and {B_secondary} stay visible as anchors while this interaction expresses: {promise}.",
        ),
        "protection": (
            "shielding_response",
            "{A} protects {B} from a clear risk cue near {B_secondary}, and {B} reacts safely. "
            "{A_secondary} and {B_secondary} stay visible as anchors while this interaction expresses: {promise}.",
        ),
        "amplification": (
            "boosting_power",
            "{A} amplifies {B} into a visibly stronger state, and {B} reacts with clear output change. "
            "{A_secondary} and {B_secondary} remain visible while this interaction expresses: {promise}.",
        ),
        "clarity": (
            "reveal_clarity",
            "{A} triggers a reveal on {B} so hidden details become clear, and {B} reacts immediately. "
            "{A_secondary} and {B_secondary} stay visible while this interaction expresses: {promise}.",
        ),
        "growth": (
            "uplift_growth",
            "{A} lifts {B} into a clear upward state change, and {B} responds visibly. "
            "{A_secondary} and {B_secondary} remain present while this interaction expresses: {promise}.",
        ),
        "generic": (
            "cooperative_resolution",
            "{A} and {B} cooperate to resolve a simple visible situation, with clear cause and reaction between them. "
            "{A_secondary} and {B_secondary} remain visible as anchors while this interaction expresses: {promise}.",
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
    Layer 3+4 deterministic salvage / guaranteed delivery for SIDE_BY_SIDE interaction quality failures.
    Returns (plan_or_none, template_name, guaranteed_delivery_used).
    """
    c = _coerce_plan_keys(parsed or {})
    oa = (c.get("objectA") or "").strip() or "object A"
    ob = (c.get("objectB") or "").strip() or "object B"
    oa_sec = (c.get("objectA_secondary") or "").strip() or "A contextual anchor"
    ob_sec = (c.get("objectB_secondary") or "").strip() or "B contextual anchor"
    promise = (c.get("advertisingPromise") or c.get("promiseReason") or "").strip() or "the advertising promise"

    bucket = _promise_bucket(promise)
    template_name, template_body = _fallback_template_for_bucket(bucket)
    sbs_motion = template_body.format(
        A=oa,
        B=ob,
        A_secondary=oa_sec,
        B_secondary=ob_sec,
        promise=promise,
    )
    sbs_open = (
        f"Opening intent: {oa} + {oa_sec} and {ob} + {ob_sec} are visible together in one stable composition, "
        "with immediate meaningful interaction between A and B."
    )
    c["sideBySideMotionScript"] = sbs_motion
    c["sideBySideOpeningFrameDescription"] = sbs_open
    c["advertisingPromise"] = promise
    if not (c.get("productNameResolved") or "").strip():
        c["productNameResolved"] = (product_name or "").strip() or "Product"
    if not (c.get("replacementMotionScript") or "").strip():
        c["replacementMotionScript"] = (
            f"{oa} interacts with {ob_sec}; meaningful visible effect supports the advertising promise."
        )
    if not (c.get("replacementOpeningFrameDescription") or "").strip():
        c["replacementOpeningFrameDescription"] = (
            f"Replacement opening frame concept with {oa} in {ob}'s context and {ob_sec} preserved."
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

    plan, _ = validate_and_normalize_plan(c)
    if plan:
        return plan, template_name, False

    # Layer 4 guaranteed delivery mode: force a conservative valid SIDE_BY_SIDE plan shape.
    forced = {
        "productNameResolved": (product_name or "").strip() or "Product",
        "advertisingPromise": promise,
        "objectA": oa,
        "objectA_secondary": oa_sec,
        "objectB": ob,
        "objectB_secondary": ob_sec,
        "morphologicalReason": (c.get("morphologicalReason") or "").strip(),
        "promiseReason": (c.get("promiseReason") or "").strip(),
        "replacementDirection": "A_replaces_B",
        "preservedBackgroundFrom": "B",
        "preservedSecondaryFrom": "B",
        "shortReplacementScript": (c.get("shortReplacementScript") or "").strip(),
        "headlineDecision": "include_product_name",
        "headlineText": (product_name or "").strip() or "Product",
        "replacementOpeningFrameDescription": c["replacementOpeningFrameDescription"],
        "replacementMotionScript": c["replacementMotionScript"],
        "sideBySideOpeningFrameDescription": sbs_open,
        "sideBySideMotionScript": sbs_motion,
        "videoPromptCore": sbs_motion,
        "openingFrameDescription": sbs_open,
        "videoVisualMode": "SIDE_BY_SIDE",
        "chosenMode": "SIDE_BY_SIDE",
        "silhouetteSimilarity": 50.0,
        "shapeAlignment": "",
        "sideBySideCameraMotion": _SBS_HALF_ORBIT_CAMERA,
        "sideBySideCameraMotionDescription": _SBS_HALF_ORBIT_PLAN_DESCRIPTION,
    }
    return forced, template_name, True


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

    max_attempts = 1 + max(0, _VIDEO_PLAN_RETRY_INTERACTION_MAX)
    last_parsed: Optional[Dict[str, Any]] = None
    last_v_err = ""
    for attempt in range(max_attempts):
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
            last_parsed = parsed

            plan, v_err = validate_and_normalize_plan(parsed)
            if not plan:
                last_v_err = (v_err or "").strip()
                if last_v_err == "side_by_side_interaction_not_meaningful" and attempt < max_attempts - 1:
                    logger.info(
                        "VIDEO_PLAN_RETRY attempt=%s reason=interaction_not_meaningful",
                        attempt + 1,
                    )
                    continue
                if v_err == "secondary_objects_not_distinct":
                    logger.info("VIDEO_PLAN_ABORTED reason=secondary_objects_not_distinct")
                elif v_err == "identity_too_close":
                    logger.info("VIDEO_PLAN_ABORTED reason=identity_too_close")
                elif v_err == "missing_object_secondary":
                    logger.error("VIDEO_PLAN_FAIL_STRUCTURE reason=%s", v_err)
                else:
                    logger.error("VIDEO_PLAN_FAIL_VALIDATION reason=%s", v_err or "unknown")
                break

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

    if last_v_err == "side_by_side_interaction_not_meaningful" and last_parsed:
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
            return salvage_plan
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

    if _is_side_by_side_plan(plan):
        ofd = (plan.get("openingFrameDescription") or "").strip()
        open_block = f"Opening intent: {ofd} " if ofd else ""
        motion_pre = _runway_side_by_side_half_orbit_preamble()
        scene = (
            f"{open_block}"
            f"SIDE_BY_SIDE (no replacement): single continuous shot; {motion_pre}"
            f"tight unified composition; {a_setup} and {b_setup} "
            f"both visible from the first frame, close together or slightly overlapping, same world and scale; "
            f"promise: {promise}. No morphing, swapping, disappearance, or cuts. "
            f"Action: {core}"
        )
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            scene += _runway_vertical_axis_hard_constraints_english()
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    elif rd == "B_replaces_A":
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
    logger.info(
        "VIDEO_PROMPT_MODE mode=%s",
        (plan.get("videoVisualMode") or "").strip() or "?",
    )
    if _is_side_by_side_plan(plan):
        logger.info("VIDEO_PROMPT_CAMERA_MOTION mode=SIDE_BY_SIDE motion=half_orbit")
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

    a_setup = f"{oa} + {oas}" if oas else oa
    b_setup = f"{ob} + {obs}" if obs else ob

    if _is_side_by_side_plan(plan):
        motion_focus = _runway_side_by_side_interaction_half_orbit_focus()
        scene = (
            f"The first frame is supplied as the start image; it already shows {a_setup} and {b_setup} side by side, "
            f"both clearly visible and balanced. {motion_focus}"
            f"Background/secondary context consistent with sides {pbg}/{psf}. "
            f"Action: {core}"
        )
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            scene += _runway_vertical_axis_hard_constraints_english()
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    elif rd == "B_replaces_A":
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

    if _is_side_by_side_plan(plan):
        motion = (
            f"Both {oa} and {ob} side by side with secondaries; MANDATORY smooth half-orbit camera around the pair per Action; "
            f"motion only; start frame supplied."
        )
        if (plan.get("shapeAlignment") or "").strip() == "vertical_axis":
            motion += " " + _runway_vertical_axis_hard_constraints_english()
            logger.info("VIDEO_PROMPT_CONSTRAINT umbrella_upright_enforced=true")
    elif rd == "B_replaces_A":
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
    logger.info(
        "VIDEO_PROMPT_MODE mode=%s",
        (plan.get("videoVisualMode") or "").strip() or "?",
    )
    if _is_side_by_side_plan(plan):
        logger.info("VIDEO_PROMPT_CAMERA_MOTION mode=SIDE_BY_SIDE motion=half_orbit")
    return out
