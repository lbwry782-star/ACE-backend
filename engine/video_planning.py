"""
ACE video engine — o3-pro planning layer (isolated from image /preview /generate).

Produces a structured plan for Runway prompt assembly. On failure, callers fall back to simple prompts.
Future: richer ACE video engine (e.g. two outputs); keep this module inspectable and logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

# Safe preview length for logs (no secrets; truncated model output)
_LOG_PREVIEW_CHARS = 240

# Match codebase: o4-mini maps to o3-pro
def _text_model() -> str:
    m = (os.environ.get("VIDEO_PLANNER_MODEL") or os.environ.get("OPENAI_TEXT_MODEL", "") or "").strip() or "o3-pro"
    return "o3-pro" if m == "o4-mini" else m


_VIDEO_PLAN_TIMEOUT = float((os.environ.get("VIDEO_PLANNER_TIMEOUT_SECONDS") or "120").strip() or "120")


_JSON_KEYS = """
OUTPUT FORMAT (strict)
- Return ONE JSON object only.
- Use the exact camelCase keys below. Do not omit required keys; use "" only where allowed.
- Do NOT wrap in markdown code fences. Do NOT add prose before or after the JSON.

Field notes (must follow in JSON values):
- morphologicalReason: Briefly justify why A and B are extremely close in overall form (whole-object, painterly grasp — not contour trivia alone).

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
  "videoPromptCore": string
}
"""


def _build_video_planner_instructions() -> str:
    return """You are the ACE video planning engine. All user-facing strings must be in English.

VIDEO PIPELINE (non-negotiable)
- The generative video opens from a first frame that already shows the replacement state: B in A's role, with A's background, A's secondary object, and A's spatial position. The rest of the video shows B interacting with A's secondary in that composition.
- Therefore Object A and Object B must be EXTREMELY morphologically similar in overall form. This is a core engine rule, not optional. If B is only moderately similar, the replacement will not read; the viewer must feel B belongs in A's place instantly, before any conceptual explanation.

CONTEXT
- ACE is an ad-generation product. Output is one short commercial video concept for a generative video model.
- Objects must be concrete, iconic physical nouns (classic situations). No brands, logos, text-as-object, or vague environments (e.g. "forest", "skyline") as primaries.
- Each main object has a nearby secondary object that is its classic context (not part of the main object): e.g. can+straw, dog+bone, bee+flower. Secondaries are not literal parts of the main object.
- Object A is chosen from the product description with an intuitive grasp of overall form — like a painter sensing the whole object, not technical contour-only tracing.

EXTREME MORPHOLOGICAL SIMILARITY (A vs B)
- Object B must be very, very close to Object A in overall silhouette, proportion, and massing. Shape correctness is STRONGLY preferred over verbal explicitness of the advertising promise.
- Do NOT accept a merely "interesting" conceptual link. Do NOT accept a B that is only somewhat similar. Do NOT choose a less shape-correct B because it makes the promise easier to explain in words.
- The replacement must be legible at first glance: the viewer should feel B can convincingly occupy A's place before the mind finishes the conceptual leap.

SEARCH RULE FOR B
- If a candidate B is not morphologically strong enough, keep searching. Do not stop at the first clever conceptual pairing. Stop only when B is BOTH: (a) extremely morphologically close to A in whole form, AND (b) plausibly connected to the advertising promise.
- Never swap a better shape match for a weaker one to favor the promise.

- Derive the advertising promise (advertisingPromise) from the product description.

REPLACEMENT
- Prefer: Object B replaces Object A while keeping A's background, A's secondary, and A's position.
- If impossible: Object A replaces Object B while keeping B's background, B's secondary, and B's position.
- Set replacementDirection to B_replaces_A or A_replaces_B accordingly.
- Set preservedBackgroundFrom and preservedSecondaryFrom to A or B matching what is preserved in the preferred case.

HEADLINE (for end-of-video on-screen text only; NOT for body copy)
- headlineDecision: include_product_name | product_name_only | no_headline — choose whether the visuals alone already convey the promise.
- headlineText: English, maximum 7 words. Empty string if and only if headlineDecision is no_headline.
- If product name is missing in input, invent a concise productNameResolved and you may use it in headline when appropriate.

VIDEO (for videoPromptCore)
- Describe scene and motion only: cinematic commercial, smooth camera, product-focused, modern lighting.
- Do NOT put the headline text inside videoPromptCore. videoPromptCore is the main action only; headline is specified separately via headlineText/headlineDecision.
- No on-screen text, logos, or readable words during the main action in this core description.

QUALITY
- Prefer morphological correctness and viewer intuition over explicit verbal explanation in videoPromptCore.
- shortReplacementScript: a brief plain-English line describing the A/B connection for the replacement shot.
- morphologicalReason: REQUIRED to state briefly why A and B are extremely close in overall form (whole-object, painterly grasp — not edge-matching alone). Be strict and concrete.
- promiseReason: short note on how B still ties to the advertising promise without weakening the shape requirement.

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
    reason_code: missing_videoPromptCore | missing_advertisingPromise | missing_objectA_or_B
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

    return {
        "productNameResolved": pn,
        "advertisingPromise": apromise,
        "objectA": oa,
        "objectA_secondary": (data.get("objectA_secondary") or "").strip(),
        "objectB": ob,
        "objectB_secondary": (data.get("objectB_secondary") or "").strip(),
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


def log_plan_summary(plan: Dict[str, Any]) -> None:
    """Concise server-side log of the chosen plan (no full prompts, no secrets)."""
    logger.info(
        'VIDEO_PLAN productNameResolved="%s" promise="%s"',
        (plan.get("productNameResolved") or "")[:120],
        (plan.get("advertisingPromise") or "")[:160],
    )
    logger.info(
        'VIDEO_PLAN objects A="%s" A_sub="%s" B="%s" B_sub="%s" repl=%s bg=%s sec=%s',
        plan.get("objectA"),
        plan.get("objectA_secondary"),
        plan.get("objectB"),
        plan.get("objectB_secondary"),
        plan.get("replacementDirection"),
        plan.get("preservedBackgroundFrom"),
        plan.get("preservedSecondaryFrom"),
    )
    logger.info(
        'VIDEO_PLAN headline_decision=%s headline="%s"',
        plan.get("headlineDecision"),
        (plan.get("headlineText") or "")[:80],
    )


def _reasoning_effort() -> str:
    raw = (os.environ.get("VIDEO_PLANNER_REASONING_EFFORT") or "low").strip().lower()
    return raw if raw in ("low", "medium") else "low"


def fetch_video_plan_o3(product_name: str, product_description: str) -> Optional[Dict[str, Any]]:
    """
    Single o3-pro call returning a validated plan dict, or None on any failure (caller uses fallback).
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_PLAN_FAIL_NO_API_KEY")
        return None

    model = _text_model()
    user_block = f"""Product name (may be empty): {product_name or "(empty)"}
Product description:
{product_description}

{_JSON_KEYS}
"""
    instructions = _build_video_planner_instructions()
    full_input = instructions + "\n\n" + user_block
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(_VIDEO_PLAN_TIMEOUT),
        max_retries=0,
    )
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
        return None

    try:
        raw = _extract_responses_output_text(response)
        if not raw:
            logger.error("VIDEO_PLAN_FAIL_EMPTY_OUTPUT model=%s", model)
            return None

        _log_output_preview(raw)

        parsed = _parse_json_from_response(raw)
        if not parsed:
            logger.error("VIDEO_PLAN_FAIL_JSON_PARSE model=%s", model)
            return None

        plan, v_err = validate_and_normalize_plan(parsed)
        if not plan:
            logger.error("VIDEO_PLAN_FAIL_VALIDATION reason=%s", v_err or "unknown")
            return None

        log_plan_summary(plan)
        logger.info("VIDEO_PLAN_OK model=%s", model)
        return plan
    except Exception as e:
        logger.warning(
            "VIDEO_PLAN_FAIL_EXCEPTION phase=post_create err_type=%s err=%s",
            type(e).__name__,
            e,
        )
        return None


_RUNWAY_PROMPT_MAX_CHARS = 1000


def _truncate_runway_prompt(s: str) -> str:
    if len(s) <= _RUNWAY_PROMPT_MAX_CHARS:
        return s
    logger.info("RUNWAY_PROMPT truncated from len=%s to %s", len(s), _RUNWAY_PROMPT_MAX_CHARS)
    return s[:_RUNWAY_PROMPT_MAX_CHARS]


def _build_runway_prompt_compact_fallback(plan: Dict[str, Any]) -> str:
    """Shorter ACE→Runway bridge if the detailed builder fails; keeps prior behavior."""
    core = (plan.get("videoPromptCore") or "").strip()
    headline_decision = plan.get("headlineDecision") or "no_headline"
    headline_text = (plan.get("headlineText") or "").strip()
    product = (plan.get("productNameResolved") or "").strip()
    script = (plan.get("shortReplacementScript") or "").strip()

    parts: List[str] = [
        "English language commercial video. Cinematic lighting, smooth camera movement, modern advertising style.",
        "During the main action: no visible readable words, no logos, no packaging print on screen.",
        "Scene and motion:",
        core,
    ]
    if script:
        parts.append(f"Replacement beat: {script}")
    if headline_decision != "no_headline" and headline_text:
        parts.append(
            "Only in the final moment after motion settles: one clean end-frame line burned into the picture, "
            f"exactly: {headline_text}"
        )
        if headline_decision == "include_product_name" and product and product.lower() not in headline_text.lower():
            parts.append(f"End frame should align with product: {product}.")
    else:
        parts.append("No title cards, supers, or captions; purely visual finish.")

    return _truncate_runway_prompt(" ".join(p for p in parts if p))


def _build_runway_prompt_detailed(plan: Dict[str, Any]) -> str:
    """
    Precise creative-direction style prompt: opening, secondary, replacement, tone, end-state, optional end headline.
    Raises ValueError if the plan lacks required fields for a coherent prompt.
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
    headline_decision = (plan.get("headlineDecision") or "no_headline").strip()
    headline_text = (plan.get("headlineText") or "").strip()
    product = (plan.get("productNameResolved") or "").strip()

    if not core:
        raise ValueError("missing videoPromptCore")

    a_setup = f"{oa} with its classic contextual prop: {oas}" if oas else oa
    b_setup = f"{ob} with its classic contextual prop: {obs}" if obs else ob

    # Opening + replacement choreography (single shot, no montage)
    if rd == "B_replaces_A":
        opening = (
            f"Open on one clear, elegant setup: {a_setup}, immediately readable and iconic. "
            f"The background and spatial world stay tied to side {pbg}; the secondary/context placement stays coherent with side {psf}. "
            f"The advertising idea to feel (without explaining in words): {promise}. "
        )
        transform = (
            f"The main motion is one continuous replacement: {ob} takes over the exact role, position, and silhouette-read of {oa} — "
            f"{oa} leaves the frame entirely as {ob} occupies that place — while the surrounding environment and the contextual secondary "
            f"remain locked to the preserved composition. The transformation must read as one legible REPLACEMENT event, smooth and premium, not a hard cut. "
        )
    else:
        opening = (
            f"Open on one clear, elegant setup: {b_setup}, immediately readable and iconic. "
            f"The background and spatial world stay tied to side {pbg}; the secondary/context placement stays coherent with side {psf}. "
            f"The advertising idea to feel: {promise}. "
        )
        transform = (
            f"The main motion is one continuous replacement: {oa} takes over the exact role, position, and silhouette-read of {ob} — "
            f"{ob} yields as {oa} occupies that place — while the surrounding environment and the contextual secondary "
            f"remain locked to the preserved composition. Same rules: one legible REPLACEMENT, smooth, premium, no gratuitous cuts. "
        )

    director = f"Creative motion and framing: {core}"
    beat = f"Replacement beat (director note): {script}" if script else ""

    tone = (
        "Visual style: clean English-language commercial aesthetic; soft cinematic light; restrained camera move; "
        "avoid clutter, avoid generic stock phrasing, avoid multiple scene changes — one visual idea, immediately grasped."
    )

    integrity = (
        "Keep the object pair and the required secondary context central; do not drift into generic scenery as the subject. "
        "No logos, no brand marks, no packaging typography, no decorative type, no irrelevant props."
    )

    # Mid-video: no readable words / logos (worded without "text" when no headline, per ACE brief)
    if headline_decision == "no_headline":
        mid_rule = (
            "Throughout the moving shot: purely pictorial storytelling — no title cards, no supers, no captions, no readable packaging, no logos."
        )
        end_rule = "Resolve on a clean held frame with no title cards or supers."
    else:
        if not headline_text:
            raise ValueError("headline expected but headlineText empty")
        mid_rule = (
            "Throughout the replacement motion: no title cards, no supers, no captions, no readable packaging, no logos — "
            "the movement itself carries meaning until it fully settles."
        )
        safe_line = headline_text.replace('"', "'")
        end_rule = (
            f"After the action fully settles and the picture holds still, only then — as the final beat — add a single short line "
            f"integrated into the frame (not subtitles): exactly «{safe_line}». "
            "Centered or clearly placed, highly readable, one line only, no paragraph, no extra copy, no earlier flashes."
        )
        if headline_decision == "include_product_name" and product and product.lower() not in safe_line.lower():
            end_rule += f" The line should still feel aligned with {product} as the offering."

    pieces = [opening, transform, director, beat, tone, integrity, mid_rule, end_rule]
    out = " ".join(p for p in pieces if p).strip()
    if not out:
        raise ValueError("empty prompt")
    return _truncate_runway_prompt(out)


def build_runway_prompt_from_plan(plan: Dict[str, Any]) -> str:
    """
    ACE plan → Runway promptText. Prefers the detailed creative-direction builder; on any failure,
    uses a compact fallback so callers stay stable.
    """
    headline_decision = (plan.get("headlineDecision") or "no_headline").strip()
    headline_text = (plan.get("headlineText") or "").strip()
    headline_present = headline_decision != "no_headline" and bool(headline_text)
    repl = (plan.get("replacementDirection") or "").strip()

    try:
        out = _build_runway_prompt_detailed(plan)
        logger.info(
            "RUNWAY_PROMPT build_ok repl=%s headline_present=%s headline=%s",
            repl or "?",
            headline_present,
            (headline_text[:100] + "…") if len(headline_text) > 100 else headline_text,
        )
        return out
    except Exception as e:
        logger.warning("RUNWAY_PROMPT detailed_builder_failed (%s); using compact fallback", e)
        fb = _build_runway_prompt_compact_fallback(plan)
        logger.info(
            "RUNWAY_PROMPT fallback_ok repl=%s headline_present=%s headline=%s",
            repl or "?",
            headline_present,
            (headline_text[:100] + "…") if len(headline_text) > 100 else headline_text,
        )
        return fb
