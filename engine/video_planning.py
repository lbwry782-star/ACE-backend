"""
ACE video engine — o3-pro planning layer (isolated from image /preview /generate).

Produces a structured plan for Runway prompt assembly. On failure, callers fall back to simple prompts.
Future: richer ACE video engine (e.g. two outputs); keep this module inspectable and logged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

# Match codebase: o4-mini maps to o3-pro
def _text_model() -> str:
    m = (os.environ.get("VIDEO_PLANNER_MODEL") or os.environ.get("OPENAI_TEXT_MODEL", "") or "").strip() or "o3-pro"
    return "o3-pro" if m == "o4-mini" else m


_VIDEO_PLAN_TIMEOUT = float((os.environ.get("VIDEO_PLANNER_TIMEOUT_SECONDS") or "120").strip() or "120")


_JSON_KEYS = """
You MUST respond with a single JSON object only (no markdown, no commentary), with exactly these keys and types:
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

CONTEXT
- ACE is an ad-generation product. Output is one short commercial video concept for a generative video model.
- Objects must be concrete, iconic physical nouns (classic situations). No brands, logos, text-as-object, or vague environments (e.g. "forest", "skyline") as primaries.
- Each main object has a nearby secondary object that is its classic context (not part of the main object): e.g. can+straw, dog+bone, bee+flower. Secondaries are not literal parts of the main object.
- Object A is chosen from the product description with an intuitive grasp of overall form (like a painter), not narrow contour trivia.
- Object B is morphologically similar to A. Prefer stronger shape match over forcing the advertising promise; the viewer completes the link. Stop when B is strongly correct morphologically AND plausibly tied to the promise — never swap a better shape match for a weaker one just to verbalize the promise.
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
- morphologicalReason and promiseReason: short planner notes (why B matches A; why B fits the promise).

"""


def _parse_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    if text.startswith("```json"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if len(lines) > 2 and lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # try to find outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _word_limit(s: str, max_words: int) -> str:
    words = (s or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _norm_enum(value: Any, allowed: List[str], default: str) -> str:
    v = (str(value) if value is not None else "").strip()
    return v if v in allowed else default


def validate_and_normalize_plan(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a normalized plan dict or None if unusable."""
    if not data:
        return None
    core = (data.get("videoPromptCore") or "").strip()
    if not core:
        return None
    apromise = (data.get("advertisingPromise") or "").strip()
    oa = (data.get("objectA") or "").strip()
    ob = (data.get("objectB") or "").strip()
    if not apromise or not oa or not ob:
        return None
    pn = (data.get("productNameResolved") or "").strip() or "Product"
    headline_decision = _norm_enum(
        data.get("headlineDecision"),
        ["include_product_name", "product_name_only", "no_headline"],
        "no_headline",
    )
    raw_headline = (data.get("headlineText") or "").strip()
    if headline_decision == "no_headline":
        headline_text = ""
    else:
        headline_text = _word_limit(raw_headline, 7)
    repl = _norm_enum(data.get("replacementDirection"), ["B_replaces_A", "A_replaces_B"], "B_replaces_A")
    bg = _norm_enum(data.get("preservedBackgroundFrom"), ["A", "B"], "A")
    sec = _norm_enum(data.get("preservedSecondaryFrom"), ["A", "B"], "A")
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
    }


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


def fetch_video_plan_o3(product_name: str, product_description: str) -> Optional[Dict[str, Any]]:
    """
    Single o3-pro call returning a validated plan dict, or None on any failure (caller uses fallback).
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_PLAN skip: OPENAI_API_KEY missing")
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
            reasoning={"effort": "low"},
        )
        raw = (response.output_text or "").strip()
        if not raw:
            logger.error("VIDEO_PLAN empty model output")
            return None
        parsed = _parse_json_from_response(raw)
        if not parsed:
            logger.error("VIDEO_PLAN json_parse_failed")
            return None
        plan = validate_and_normalize_plan(parsed)
        if not plan:
            logger.error("VIDEO_PLAN validation_failed")
            return None
        log_plan_summary(plan)
        return plan
    except Exception as e:
        logger.warning("VIDEO_PLAN o3_failed fallback will be used: %s", e)
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
