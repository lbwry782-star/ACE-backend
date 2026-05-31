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

from engine.ace_usage_memory import (
    get_used_headlines,
    get_used_object_a,
    remember_headline,
    remember_object_a as remember_object_a_ace,
)
from engine.video_language import (
    bilingual_en_he_headline_tail_struct_ok,
    english_headline_tail_after_product_no_separator_punct,
    headline_product_includes_hebrew_letters,
    hebrew_script_product_headline_tail_struct_ok,
    normalize_video_content_language,
    product_name_is_latin_only_for_bilingual_headline,
    video_language_display_name,
)
from engine.video_plan_objects import video_plan_object_blob_implies_graphic_text_content

# Semantic (regex) gate: interaction prose must not imply frictionless / floaty motion between objects.
# Word-boundary + stem patterns — not a single substring match.
_PHYSICAL_INTERACTION_MOTION_FORBIDDEN: List[Tuple[str, re.Pattern]] = [
    ("slide", re.compile(r"\b(slides?|sliding|slid)\b", re.I)),
    ("glide", re.compile(r"\b(glides?|gliding|glided)\b", re.I)),
    ("drift", re.compile(r"\b(drifts?|drifting|drifted)\b", re.I)),
    ("frictionless", re.compile(r"\bfrictionless\w*\b", re.I)),
    ("zero_no_friction", re.compile(r"\b(zero|no)\s+friction\b", re.I)),
    ("weightless", re.compile(r"\bweightless\w*\b", re.I)),
    ("effortless_motion", re.compile(r"\beffortless(ly)?\s+(motion|movement|contact|touch|sliding|gliding|push|pull|drag)\b", re.I)),
    ("float_motion", re.compile(r"\b(floats?|floating)\b.*\b(motion|movement|together|apart|contact)\b", re.I)),
    ("hover", re.compile(r"\b(hover|hovers|hovering)\b", re.I)),
    ("levitate", re.compile(r"\blevitat\w*\b", re.I)),
    ("glide_slide_across", re.compile(r"\b(glides?|slides?)\s+(across|along)\b", re.I)),
    ("silky_smooth", re.compile(r"\b(silky|buttery)\s+smooth\b", re.I)),
    ("ice_smooth", re.compile(r"\b(like\s+)?ice\b.*\b(smooth|slide|glide)\b", re.I)),
]

# Appended to Runway promptText in runway_video after text-policy sanitize (hard constraint).
RUNWAY_PHYSICS_REALISM_CONSTRAINT = (
    "PHYSICAL REALISM: All motion must obey real-world resistance, weight, and contact between surfaces. "
    "No frictionless sliding, gliding, drifting, or floating movement. Show grip, pressure, and resisted motion only."
)


def interaction_fields_imply_frictionless_or_floaty_motion(blob: str) -> Optional[str]:
    """
    Return a rule label if interaction-related prose matches forbidden motion semantics; else None.
    Scans interactionSummary, interactionScript, objectAReason, objectBReason only (not headline, not promise).
    """
    if not (blob or "").strip():
        return None
    for label, rx in _PHYSICAL_INTERACTION_MOTION_FORBIDDEN:
        if rx.search(blob):
            return label
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

Keys (all strings): productNameResolved, objectA, objectB, interactionSummary, interactionScript,
advertisingPromise, headlineText

One physical A↔B interaction in a single shot. objectA/B + interaction*: short English; other fields: request language. Empty product name → invent productNameResolved.
Objects must be simple physical items — never posters, printed graphics, readable books, signage, labels, or other text/graphic carriers (see OBJECTS block below).

Headline: required non-empty headlineText. ≤7 words; express the advertisingPromise and interpret the interaction (meaning), not a literal description of the shot. Build the remainder via rhyming object substitution per the HEADLINE block below. headlineText format: productNameResolved + one normal space + remainder — no comma or other separator punctuation. Video overlay may render name and remainder as separate visuals; keep this field as plain text with no bidi control characters. Language-specific headline rules are in the user block below.

Before the JSON: one silent internal revision pass only (pair, realism, cliché default, physics, motion clarity, headline); output final JSON only — no explanations.

Failure only: {"planningFailure":"planning_failed_no_valid_interaction"}

Emit a complete plan only if: (i) object A is grounded in the product; (ii) object B was accepted only after a physically plausible, non-surreal A↔B interaction was identified; (iii) advertisingPromise emerged together with that interaction (not pre-decided, not reverse-engineered from a preset slogan); (iv) headline interprets the interaction per the headline rules; (v) object rules below are satisfied; (vi) interactionSummary + interactionScript (+ object reasons) describe grounded contact with resistance — no frictionless sliding/gliding/floating-style motion (see PHYSICAL REALISM block).
"""


def _planner_causal_reasoning_block() -> str:
    """
    Planner-only: causal order with simultaneous discovery — goal emerges when a valid interaction is found.
    Server does not score or rewrite creative content; this steers the model.
    """
    return (
        "CAUSAL CREATIVE FLOW (mandatory internal reasoning; do not narrate it in the JSON):\n"
        "1. Read product name and description. Select object A from them (grounded, simple, physical).\n"
        "2. Explore candidate objects B. For each candidate, check whether a REALISTIC, PHYSICAL, NON-SURREAL "
        "interaction between A and B is possible in the real world (theoretically possible; not fantasy, not dream-logic).\n"
        "3. The advertising goal is NOT known in advance. At the exact moment you recognize a valid A↔B interaction, "
        "an advertisingPromise MUST emerge from that interaction — it justifies and belongs to that pair. "
        "Causality is preserved (the promise is why B is the right second object), but discovery is simultaneous: "
        "interaction recognition and goal emergence happen together.\n"
        "4. Accept B ONLY if such a goal emerges. If no valid interaction or no emerging goal, reject that B and continue searching. "
        "Do NOT pick a goal first and then force objects to fit. Do NOT invent conceptual matches without a real physical interaction.\n"
        "5. interactionSummary and interactionScript must describe only that single plausible interaction. advertisingPromise must align with it.\n"
        "6. Interactions may be unusual but must remain physically plausible — not impossible, not surreal.\n"
        "7. Motion between A and B must show contact, weight, and resistance — never frictionless sliding, gliding, drifting, "
        "floating/hovering movement, or zero-friction motion.\n\n"
    )


def _planner_physical_realism_block() -> str:
    """Planner: forbid floaty / frictionless interaction language (server also validates interaction fields)."""
    return (
        "PHYSICAL REALISM (interactionScript, interactionSummary, objectAReason, objectBReason):\n"
        "The shot must look physically grounded. The A↔B interaction must involve clear contact, grip, pressure, or resisted motion — "
        "weight and surface resistance visible.\n"
        "FORBIDDEN interaction language: frictionless motion; smooth sliding or gliding between objects; drifting; "
        "floating or hovering movement; zero friction; weightless contact; effortless physical motion between A and B; "
        "ice-like glide.\n"
        "PREFER verbs like: pressing, pushing, pulling, gripping, placing, bracing, tightening, turning, steady sliding only when "
        "resistance is obvious (e.g. dragged with friction), not frictionless.\n"
        "Do not describe motion that would read as floating, gliding on air, or sliding with no drag.\n\n"
    )


def _planner_object_selection_rules_block() -> str:
    """Hard rules for objectA/objectB: no graphic-communication objects (planner + server validation)."""
    return (
        "OBJECTS (strict): Choose two simple, physical, everyday items. They must not be communicative media.\n"
        "HARD FORBIDDEN — any object whose primary purpose is to hold or show graphics or text, including: "
        "posters; printed images/graphics; photographs with visible imagery; paintings with visible imagery; "
        "magazines; newspapers; books that are open, readable, or text-bearing; screens or monitors if visible content is described; "
        "signs; labels; packaging described with visible design/text/logos; flyers; brochures; infographics; charts/diagrams as displays; "
        "scoreboards; LED message boards; greeting cards; certificates; barcodes; branded packaging.\n"
        "ALLOWED examples (physical only, no described on-surface content): empty picture/photo frame; blank paper; "
        "screen/monitor/TV/phone as a device only with NO visible content described; empty billboard structure; billboard with no ad/copy described; "
        "closed book with no readable text described.\n"
        "Critical: a poster is always forbidden; an empty frame is allowed. A screen is allowed only if you do NOT describe what is on the screen.\n"
        "Do not pick objects that imply readable text or illustrative content anywhere in objectA, objectB, objectAReason, objectBReason, "
        "interactionSummary, or interactionScript.\n\n"
    )


def _planner_headline_rhyming_substitution_block() -> str:
    """Rhyming object-substitution headline rule (Builder2 video planning only)."""
    return (
        "HEADLINE (rhyming object substitution — mandatory for headlineText remainder):\n"
        "1. First find an existing familiar expression, idiom, proverb, or well-known phrase that expresses the advertisingPromise.\n"
        "2. The original expression must NOT already contain the name or core word of Object A or Object B.\n"
        "3. Choose exactly one word inside that expression.\n"
        "4. Replace that one word with the name of Object A or Object B (natural headline-language form).\n"
        "5. The replacement is valid only if the inserted object name rhymes with, or is phonetically very close to, the replaced original word.\n"
        "6. The result must still feel like a recognizable twist on the original expression.\n"
        "7. Do not add extra words before, inside, or after the twisted expression except productNameResolved as the headlineText prefix.\n"
        "8. headlineText format: productNameResolved + one normal space + final remainder.\n"
        "9. ≤7 words total in headlineText.\n"
        "10. headlineText must express the advertisingPromise and interpret the interaction — not a literal shot description.\n"
        "11. If no strong rhyme / phonetic substitution exists, choose another expression. Do not force a weak rhyme.\n"
        "12. Do NOT pick an expression that already contains the object word before substitution.\n"
        "13. The final substituted headline must express the advertisingPromise.\n"
        "14. Prefer the strongest case: the substitution itself should be the expression of the advertisingPromise.\n"
        "15. The viewer should feel that replacing the original word with Object A or Object B is exactly what creates the advertising meaning.\n"
        "16. It is not enough that the original expression expresses the promise, or that the final phrase sounds clever.\n"
        "17. The best headline has all three: (a) the original expression is recognizable, (b) the object-word substitution is visible and phonetically strong, (c) the substitution itself makes the advertisingPromise understandable.\n"
        "18. If the substitution is only a pun but does not carry the advertisingPromise, reject it and choose another expression/substitution.\n\n"
    )


def _planner_headline_rules_user_block(lang_code: str) -> str:
    """Extra headline constraints appended to the planner user block (language-specific)."""
    rhyme_block = _planner_headline_rhyming_substitution_block()
    if normalize_video_content_language(lang_code) != "he":
        return (
            "Headline (non-Hebrew request): headlineText is required. It must start with productNameResolved, "
            "then exactly one normal ASCII space, then the remainder phrase. "
            "Language should follow product language/context (English or mixed when context naturally requires it). "
            f"{rhyme_block}"
            "No comma, colon, dash, dot, or semicolon between name and tail.\n\n"
        )
    return (
        "Headline (Hebrew request): headlineText is required. It must start with productNameResolved, "
        "then exactly one normal space, then the rest of the headline. "
        "No comma, middle dot (·), bullet, colon, dash, or semicolon between the name and the tail — only that single space. "
        "Language may be Hebrew/English/mixed only as justified by product language/context (do not force one language). "
        f"{rhyme_block}"
        "Do not translate the product name. Works the same whether productNameResolved is English (Latin) or Hebrew script. "
        "Do not insert bidi control characters in JSON.\n\n"
    )


def _build_video_planner_instructions(content_language: str = "he") -> str:
    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    he_head = ""
    if lang == "he":
        he_head = (
            "Hebrew headline: productNameResolved then one space then Hebrew tail (no punctuation separator); no bidi marks in JSON. "
        )
    return (
        f"ACE video: one continuous shot; camera = smooth half-orbit around the two objects (path only). "
        f"Language {lang_name} ({lang}). "
        f"{he_head}"
        "Everyday complementary objects grounded in the product; reject the first cliché default; "
        "objectA/objectB must never be graphic-communication items (posters, prints, readable media, signage with copy, etc.). "
        "Creative rule: advertisingPromise emerges together with the chosen A↔B interaction — never choose the promise before "
        "a valid physical interaction exists; never reverse-reason from a preset goal to objects. "
        "Physical realism: interaction motion must show contact, weight, and resistance — no frictionless sliding or floaty movement. "
        "Stay realistic. One A↔B interaction only (no alternate layouts). "
        'Planner refusal: {"planningFailure":"planning_failed_no_valid_interaction"}'
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


_VIDEO_PLAN_SCHEMA_VERSION = "single_interaction_v3"

_PLANNER_SELF_FAILURE_CODES: FrozenSet[str] = frozenset({"planning_failed_no_valid_interaction"})


def _build_unrealistic_physics_repair_input(
    *,
    base_attempt_input: str,
    product_name: str,
    product_description: str,
    advertising_promise: str,
    previous_plan: Dict[str, Any],
) -> str:
    return (
        f"{base_attempt_input}\n\n"
        "REPAIR REQUEST (one retry): The previous plan violated physical realism.\n"
        "Keep the same product name, product description, and advertising goal/promise.\n"
        "Remove any sliding/gliding/drifting/floating/frictionless/weightless/hovering motion.\n"
        "Create grounded physical contact and use resistance, weight, friction, support, impact, or contact-based motion.\n"
        "Return the same required JSON shape only.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Advertising goal/promise to keep: {advertising_promise}\n"
        "Previous invalid plan (for correction):\n"
        f"{json.dumps(previous_plan, ensure_ascii=False)}\n"
    )


def _build_invalid_objects_repair_input(
    *,
    base_attempt_input: str,
    product_name: str,
    product_description: str,
    advertising_promise: str,
    previous_plan: Dict[str, Any],
) -> str:
    return (
        f"{base_attempt_input}\n\n"
        "REPAIR REQUEST (one retry): The previous plan used invalid objects.\n"
        "Keep the same product name, product description, and advertising goal/promise.\n"
        "Objects must be concrete, defined, classic physical everyday objects.\n"
        "Forbidden object families: lump, blob, foam, clay, modeling clay, dough, putty, slime, gel, mud, paste, powder, sand pile, raw material, amorphous mass, undefined material.\n"
        "Do not use soft material whose purpose is only to receive an imprint.\n"
        "Do not solve the interaction by pressing Object A into a soft material.\n"
        "Choose two defined everyday objects that can interact physically without relying on deformation of an amorphous material.\n"
        "Failed examples (forbidden): sneaker -> lump of modeling clay; sneaker -> foam.\n"
        "Replace any invalid object with a clear everyday object while preserving one single-interaction video structure.\n"
        "Return the same required JSON shape only.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Advertising goal/promise to keep: {advertising_promise}\n"
        "Previous invalid plan (for correction):\n"
        f"{json.dumps(previous_plan, ensure_ascii=False)}\n"
    )


def _build_headline_hebrew_product_tail_repair_input(
    *,
    base_attempt_input: str,
    product_name: str,
    previous_plan: Dict[str, Any],
) -> str:
    pn = (previous_plan.get("productNameResolved") or product_name or "").strip()
    bad_headline = (previous_plan.get("headlineText") or "").strip()
    return (
        f"{base_attempt_input}\n\n"
        "REPAIR REQUEST (one retry): Fix headlineText format only.\n"
        "Do NOT change objectA, objectB, objectAReason, objectBReason, interactionSummary, "
        "interactionScript, advertisingPromise, promiseDerivation, or productNameResolved.\n"
        "headlineText must start with productNameResolved, then exactly one normal ASCII space, then a non-empty Hebrew remainder.\n"
        "The Hebrew remainder must contain at least one Hebrew letter, no comma, no bidi marks, "
        "and no forbidden separator punctuation between the name and tail.\n"
        "Preserve the creative headline meaning from the invalid headline when possible.\n"
        "Return the same required JSON shape only.\n"
        f"productNameResolved: {pn or '(empty)'}\n"
        f"Invalid headlineText: {bad_headline or '(empty)'}\n"
        "Previous plan (change headlineText only):\n"
        f"{json.dumps(previous_plan, ensure_ascii=False)}\n"
    )


def _build_physics_safe_fallback_plan(
    *,
    product_name: str,
    product_description: str,
    content_language: str,
    advertising_promise: str,
) -> Dict[str, Any]:
    """
    Deterministic, physically grounded fallback plan used only when physics validation + repair fail.
    Keeps one simple contact interaction with gravity-consistent motion.
    """
    lang = normalize_video_content_language(content_language)
    pn = (product_name or "").strip() or "ACE Product"
    promise = (advertising_promise or "").strip()
    if not promise:
        promise = (
            "תוצאה יציבה שנשארת לאורך זמן" if lang == "he" else "A stable result that lasts over time"
        )
    if lang == "he":
        headline = f"{pn} יציב לאורך זמן"
    else:
        headline = f"{pn} Built to last"
    return {
        "productNameResolved": pn,
        "objectA": "product item",
        "objectB": "wooden shelf",
        "objectAReason": "Directly represents the marketed product in a simple physical form.",
        "objectBReason": "Stable everyday support object that creates clear, realistic contact.",
        "interactionSummary": "A hand places the product item on a wooden shelf and presses down to seat it firmly.",
        "interactionScript": "Single continuous shot. The product item is placed on a wooden shelf, then a hand presses it down and releases while both objects remain in stable contact with visible resistance.",
        "advertisingPromise": promise,
        "promiseDerivation": (
            "נגזר מפעולה פיזית יציבה וברורה של מגע, לחץ והתייצבות."
            if lang == "he"
            else "Derived from a stable physical action with clear contact, pressure, and settling."
        ),
        "headlineText": _word_limit(headline, 7),
        "headlineDerivation": (
            "הכותרת מבטאת תוצאה מתמשכת שנובעת ממגע יציב."
            if lang == "he"
            else "Headline expresses an enduring outcome emerging from stable contact."
        ),
        "language": lang,
        "objectInferenceMode": "deterministic_physics_fallback",
        "literalObjectCount": 2,
        "headlineDecision": "include_product_name",
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


def _headline_prefix_ok(headline: str, product_resolved: str) -> bool:
    p = (product_resolved or "").strip()
    h = (headline or "").strip()
    if not p or not h:
        return False
    if h == p:
        return True
    return h.startswith(p + " ")


def _headline_word_count_ok(headline: str) -> bool:
    words = [w for w in (headline or "").strip().split() if w]
    return len(words) <= 7


_HEADLINE_HEBREW_TAIL_FORBIDDEN = frozenset("•.:;-–—−\u2212\u00b7,")
_HEADLINE_HEBREW_TAIL_BIDI = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _headline_hebrew_product_tail_would_fail(headline: str, product_name: str) -> bool:
    h_norm = (headline or "").strip()
    pn = (product_name or "").strip()
    if not h_norm or not pn:
        return True
    if h_norm == pn:
        return not hebrew_script_product_headline_tail_struct_ok("")
    if not h_norm.startswith(pn + " "):
        return True
    tail_hs = h_norm[len(pn) :].lstrip()
    return not hebrew_script_product_headline_tail_struct_ok(tail_hs)


def _extract_headline_creative_remainder(headline: str, product_name: str) -> str:
    h = (headline or "").strip()
    pn = (product_name or "").strip()
    if not h or not pn:
        return ""
    if h == pn:
        return ""
    if h.startswith(pn + " "):
        return h[len(pn) + 1 :].strip()
    if h.startswith(pn):
        return h[len(pn) :].lstrip(" \t•.:;-–—−,")
    idx = h.find(pn)
    if idx >= 0:
        before = h[:idx].strip()
        after = h[idx + len(pn) :].strip().lstrip(" \t•.:;-–—−,")
        merged = " ".join(p for p in (before, after) if p).strip()
        if merged:
            return merged
    return h


def _sanitize_hebrew_product_headline_tail(tail: str) -> str:
    cleaned: list[str] = []
    for ch in (tail or "").strip():
        if ch in _HEADLINE_HEBREW_TAIL_FORBIDDEN or ch in _HEADLINE_HEBREW_TAIL_BIDI:
            if ch == ",":
                cleaned.append(" ")
            continue
        cleaned.append(ch)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _repair_headline_hebrew_product_tail_struct(headline: str, product_name: str) -> Optional[str]:
    pn = (product_name or "").strip()
    if not pn:
        return None
    remainder = _sanitize_hebrew_product_headline_tail(
        _extract_headline_creative_remainder(headline, pn)
    )
    if not remainder:
        return None
    repaired = " ".join(f"{pn} {remainder}".split())
    if not _headline_prefix_ok(repaired, pn):
        return None
    tail = repaired[len(pn) :].lstrip()
    if not hebrew_script_product_headline_tail_struct_ok(tail):
        return None
    if not _headline_word_count_ok(repaired):
        words = repaired.split()
        if len(words) > 7:
            repaired = " ".join(words[:7])
            tail = repaired[len(pn) :].lstrip()
            if not hebrew_script_product_headline_tail_struct_ok(tail):
                return None
        else:
            return None
    return repaired


_HEADLINE_HEBREW_GENERIC_FALLBACK_TAIL = "פותח הזדמנות חדשה"


def _tail_contains_product_name(tail: str, product_name: str) -> bool:
    t = " ".join((tail or "").strip().lower().split())
    pn = " ".join((product_name or "").strip().lower().split())
    if not t or not pn:
        return False
    return pn in t


def _safe_hebrew_product_headline_tail(tail: str, product_name: str) -> str:
    t = _sanitize_hebrew_product_headline_tail(tail)
    if not t or _tail_contains_product_name(t, product_name):
        return ""
    if not hebrew_script_product_headline_tail_struct_ok(t):
        return ""
    return t


def _hebrew_tail_from_advertising_promise(advertising_promise: str, product_name: str) -> str:
    raw = _sanitize_hebrew_product_headline_tail(advertising_promise or "")
    if not raw:
        return ""
    hebrew_words: list[str] = []
    for word in raw.split():
        if re.search(r"[\u0590-\u05FF]", word) and not re.search(r"[A-Za-z]", word):
            hebrew_words.append(word)
    candidate = " ".join(hebrew_words[:4]).strip()
    return _safe_hebrew_product_headline_tail(candidate, product_name)


def _fit_hebrew_product_headline_to_word_limit(headline: str, product_name: str) -> Optional[str]:
    pn = (product_name or "").strip()
    h = (headline or "").strip()
    if not pn or not h or not _headline_prefix_ok(h, pn):
        return None
    words = h.split()
    if len(words) <= 7:
        return h
    pn_words = pn.split()
    tail_budget = max(1, 7 - len(pn_words))
    tail_words = h[len(pn) :].lstrip().split()[:tail_budget]
    rebuilt = " ".join(f"{pn} {' '.join(tail_words)}".split())
    tail = rebuilt[len(pn) :].lstrip()
    if not hebrew_script_product_headline_tail_struct_ok(tail):
        return None
    return rebuilt


def _build_headline_hebrew_product_tail_final_fallback(
    *,
    headline: str,
    product_name: str,
    advertising_promise: str,
) -> Optional[str]:
    pn = (product_name or "").strip()
    if not pn:
        logger.info("VIDEO_HEADLINE_STRUCT_FINAL_FALLBACK_FAIL")
        return None

    logger.info("VIDEO_HEADLINE_STRUCT_FINAL_FALLBACK_START")
    remainder_sources = (
        _sanitize_hebrew_product_headline_tail(
            _extract_headline_creative_remainder(headline, pn)
        ),
        _hebrew_tail_from_advertising_promise(advertising_promise, pn),
        _HEADLINE_HEBREW_GENERIC_FALLBACK_TAIL,
    )
    for raw_tail in remainder_sources:
        safe_tail = _safe_hebrew_product_headline_tail(raw_tail, pn)
        if not safe_tail:
            continue
        rebuilt = " ".join(f"{pn} {safe_tail}".split())
        fitted = _fit_hebrew_product_headline_to_word_limit(rebuilt, pn)
        if not fitted or _headline_hebrew_product_tail_would_fail(fitted, pn):
            continue
        logger.info("VIDEO_HEADLINE_STRUCT_FINAL_FALLBACK_OK headlineText=%s", fitted[:200])
        return fitted

    logger.info("VIDEO_HEADLINE_STRUCT_FINAL_FALLBACK_FAIL")
    return None


def _rescue_plan_headline_hebrew_product_tail(
    plan_data: Dict[str, Any],
    *,
    planner_deadline_monotonic: Optional[float],
    product_name: str,
    product_description: str,
    content_language: str,
) -> Optional[Dict[str, Any]]:
    merged = dict(plan_data)
    fallback_headline = _build_headline_hebrew_product_tail_final_fallback(
        headline=(merged.get("headlineText") or "").strip(),
        product_name=(merged.get("productNameResolved") or product_name).strip(),
        advertising_promise=(merged.get("advertisingPromise") or "").strip(),
    )
    if not fallback_headline:
        return None
    merged["headlineText"] = fallback_headline
    rescued, _ = validate_and_normalize_plan(
        merged,
        planner_deadline_monotonic=planner_deadline_monotonic,
        product_name=product_name,
        product_description=product_description,
        content_language=content_language,
    )
    return rescued


# Mandatory smooth half-orbit camera around the two-object composition (ACE single interaction).
_ACE_HALF_ORBIT_RUNWAY_APPEND = (
    " MANDATORY CAMERA (NOT OPTIONAL): The two physical objects form one paired composition in frame. "
    "The camera MUST perform a smooth half-orbit—a controlled half-circle path around that pair—so the viewer sees the interaction "
    "from continuously changing angles across the entire shot (calm advertising reveal in 3D). "
    "FORBIDDEN: static camera; nearly static camera; relying only on micro-flicker or tiny object motion without this orbit; "
    "dramatic fast moves; chaotic spin; full 360; handheld shaky cam; losing either object out of frame; cuts; scene changes. "
    "Small object/subject motion may appear as minor motion only—it must NOT replace the mandatory half-orbit. "
    "Half-orbit is smooth, medium-slow, stable, centered on the pair; both objects stay visible and readable throughout."
)


def _runway_ace_half_orbit_focus() -> str:
    """Mandatory half-orbit camera around the two-object composition (ACE single interaction)."""
    return (
        "MANDATORY: the camera performs a smooth half-orbit (half-circle path) around the two physical objects as one composition—"
        "continuously changing viewing angle; both stay fully in frame and readable. "
        "Tiny subject motion is optional minor motion only; do not substitute it for the orbit. "
        "No static camera, no morph, no swap, no cuts. "
    )


_RUNWAY_STYLE_TAIL = (
    "No logos, no packaging typography, no on-screen words. Single clean commercial look."
)

_RUNWAY_CONTACT_EXECUTION_CLAUSE = (
    "CONTACT EXECUTION: The active object must maintain full visible contact with the main body/center of the passive object during the scripted action. "
    "The action must happen across or over the passive object, not merely touch, tap, graze, or bump its edge or corner. "
    "Show sustained pressure, resistance, and visible effect throughout the motion. "
    "If the action is rolling, dragging, pushing, sweeping, pressing, or rubbing, the active object must travel across the passive object's central surface, not stop at the edge."
)

_RUNWAY_INTERACTION_TAIL_MARKERS: Tuple[str, ...] = (
    "Physical interaction (follow exactly):",
    "Physical interaction:",
    "CONTACT EXECUTION:",
)

_PRECISE_ACTION_CAMERA_RULES: List[Tuple[str, re.Pattern]] = [
    ("press", re.compile(r"\bpress(?:es|ed|ing)?\b", re.I)),
    ("roll", re.compile(r"\broll(?:s|ed|ing)?\b", re.I)),
    ("drag", re.compile(r"\bdrag(?:s|ged|ging)?\b", re.I)),
    ("push", re.compile(r"\bpush(?:es|ed|ing)?\b", re.I)),
    ("sweep", re.compile(r"\bsweep(?:s|swept|ing)?\b", re.I)),
    ("rub", re.compile(r"\brub(?:s|bed|bing)?\b", re.I)),
    ("across", re.compile(r"\bacross\b", re.I)),
    ("over", re.compile(r"\bover\b", re.I)),
    ("carve", re.compile(r"\bcarv(?:e|es|ed|ing)\b", re.I)),
    ("cut", re.compile(r"\bcut(?:s|ting)?\b", re.I)),
    ("imprint", re.compile(r"\bimprint(?:s|ed|ing)?\b", re.I)),
    ("insert", re.compile(r"\binsert(?:s|ed|ing)?\b", re.I)),
    ("align", re.compile(r"\balign(?:s|ed|ing)?\b", re.I)),
    ("stamp", re.compile(r"\bstamp(?:s|ed|ing)?\b", re.I)),
    ("squeeze", re.compile(r"\bsqueez(?:e|es|ed|ing)\b", re.I)),
]


def _interaction_text_for_camera_focus(plan: Dict[str, Any]) -> str:
    summary = (plan.get("interactionSummary") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    return f"{summary} {script}".strip()


def _runway_camera_motion_focus(plan: Dict[str, Any]) -> Tuple[str, str]:
    """
    Default: half-orbit.
    For single precise interactions that must read instantly, use stable visibility-first framing.
    Returns (motion_text, motion_mode_label).
    """
    interaction_blob = _interaction_text_for_camera_focus(plan)
    for _, rx in _PRECISE_ACTION_CAMERA_RULES:
        if rx.search(interaction_blob):
            return (
                "MANDATORY: stable locked camera framing focused on the exact contact/action point so the precise action reads instantly. "
                "Keep both objects visible and readable throughout. No orbit, no sweeping camera path, no cuts.",
                "stable_precise_action",
            )
    return _runway_ace_half_orbit_focus(), "half_orbit"


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
    ACE video engine v3 — server: structural validation (required fields, JSON shape via coerce,
    headline prefix + word-count, no graphic/text-display objects in A/B/interaction fields).
    Causal ordering and goal emergence are specified in planner prompts only; the server does not
    score or rewrite creative choices. o3-pro is the sole creative authority for object choice,
    interaction, promise, and headline wording.
    Returns (plan, None) or (None, reason_code) for fail-fast logging.
    """
    logger.info("VIDEO_PLAN_SERVER_CREATIVE_GATE=disabled")
    logger.info("VIDEO_PLAN_SERVER_VALIDATION_SCOPE=structural_plus_no_graphic_media_objects")

    if not data:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=no_payload")
        return None, "planning_failed_incomplete_plan"

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
    lang_raw = str(data.get("language") or "").strip()
    if not lang_raw:
        lang_raw = normalize_video_content_language(content_language)

    if not pn or not oa or not ob:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_objects_or_product_name")
        return None, "planning_failed_incomplete_plan"
    if not int_sum or not int_script:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_interaction_fields")
        return None, "planning_failed_incomplete_plan"
    if not apromise:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_advertising_promise")
        return None, "planning_failed_incomplete_plan"
    if not headline:
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=missing_headline_text")
        return None, "planning_failed_incomplete_plan"

    if planner_deadline_monotonic is not None and time.monotonic() >= planner_deadline_monotonic:
        logger.error("VIDEO_PLAN_DEADLINE_EXCEEDED stage=validate")
        raise VideoPlanningTimeoutError()

    if not _headline_prefix_ok(headline, pn):
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_prefix_format")
        return None, "planning_failed_incomplete_plan"

    lang_norm = normalize_video_content_language(lang_raw)
    if lang_norm == "he" and product_name_is_latin_only_for_bilingual_headline(pn):
        h_norm = headline.strip()
        if h_norm == pn:
            tail_he = ""
        else:
            if not h_norm.startswith(pn + " "):
                logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_bilingual_en_he_tail")
                return None, "planning_failed_incomplete_plan"
            tail_he = h_norm[len(pn) :].lstrip()
        if not bilingual_en_he_headline_tail_struct_ok(tail_he):
            logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_bilingual_en_he_tail")
            return None, "planning_failed_incomplete_plan"

    if (
        lang_norm == "he"
        and headline_product_includes_hebrew_letters(pn)
        and not product_name_is_latin_only_for_bilingual_headline(pn)
    ):
        if _headline_hebrew_product_tail_would_fail(headline, pn):
            logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_START reason=headline_hebrew_product_tail")
            repaired_headline = _repair_headline_hebrew_product_tail_struct(headline, pn)
            if repaired_headline:
                headline = repaired_headline
                data["headlineText"] = repaired_headline
            if not _headline_hebrew_product_tail_would_fail(headline, pn):
                logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_OK")
            else:
                logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_FAIL")
                fallback_headline = _build_headline_hebrew_product_tail_final_fallback(
                    headline=headline,
                    product_name=pn,
                    advertising_promise=apromise,
                )
                if fallback_headline:
                    headline = fallback_headline
                    data["headlineText"] = fallback_headline
                if _headline_hebrew_product_tail_would_fail(headline, pn):
                    logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_hebrew_product_tail")
                    return None, "headline_hebrew_product_tail"

    if lang_norm == "en":
        h_norm = headline.strip()
        if h_norm != pn and h_norm.startswith(pn + " "):
            tail_en = h_norm[len(pn) :].lstrip()
            if not english_headline_tail_after_product_no_separator_punct(tail_en):
                logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_en_tail_separator_punct")
                return None, "planning_failed_incomplete_plan"

    if not _headline_word_count_ok(headline):
        logger.info("VIDEO_PLAN_STRUCT_INCOMPLETE reason=headline_word_count")
        return None, "planning_failed_incomplete_plan"

    _object_blob = "\n".join([oa, ob, oa_r, ob_r, int_sum, int_script])
    _bad_obj = video_plan_object_blob_implies_graphic_text_content(_object_blob)
    if _bad_obj:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=invalid_graphic_content_objects rule=%s",
            _bad_obj,
        )
        return None, "planning_failed_invalid_objects"

    _physics_blob = "\n".join([int_sum, int_script, oa_r, ob_r])
    _bad_physics = interaction_fields_imply_frictionless_or_floaty_motion(_physics_blob)
    if _bad_physics:
        logger.info(
            "VIDEO_PLAN_STRUCT_INCOMPLETE reason=unrealistic_interaction_motion rule=%s",
            _bad_physics,
        )
        return None, "planning_failed_unrealistic_physics"

    _lit_raw = data.get("literalObjectCount")
    if isinstance(_lit_raw, bool):
        literal_pass = int(_lit_raw)
    elif isinstance(_lit_raw, int):
        literal_pass = _lit_raw
    elif isinstance(_lit_raw, float):
        literal_pass = int(_lit_raw)
    elif isinstance(_lit_raw, str) and _lit_raw.strip().lstrip("-").isdigit():
        literal_pass = int(_lit_raw.strip())
    else:
        literal_pass = 0

    opening_fd = (
        f"Single continuous shot: {oa} and {ob} are both visible together in one stable composition; "
        "the camera performs a smooth half-orbit around the pair."
    )
    core = f"{int_script}{_ACE_HALF_ORBIT_RUNWAY_APPEND}".strip()

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
        "objectInferenceMode": str(data.get("objectInferenceMode") or "").strip(),
        "literalObjectCount": literal_pass,
        "objectGroundednessOk": True,
        "headlineDecision": (
            str(data.get("headlineDecision") or "").strip() or "include_product_name"
        ),
        "videoPromptCore": core,
        "openingFrameDescription": opening_fd,
    }, None


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


_RUNWAY_STRUCT_REQUIRED_KEYS: Tuple[str, ...] = (
    "productNameResolved",
    "objectA",
    "objectB",
    "interactionScript",
    "advertisingPromise",
    "headlineText",
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


def _log_video_plan_post_ok_diagnostics(plan: Dict[str, Any]) -> None:
    """Post-success creative diagnostics for retrospective ad-concept review (logging only)."""
    product_resolved = (plan.get("productNameResolved") or "").strip()
    headline_full = (plan.get("headlineText") or "").strip()
    headline_remainder = headline_full
    if product_resolved and headline_full.startswith(product_resolved + " "):
        headline_remainder = headline_full[len(product_resolved) + 1 :].strip()

    logger.info("VIDEO_PLAN_DIAG productNameResolved=%s", product_resolved[:200])
    logger.info("VIDEO_PLAN_DIAG objectA=%s", (plan.get("objectA") or "")[:200])
    logger.info("VIDEO_PLAN_DIAG objectB=%s", (plan.get("objectB") or "")[:200])
    logger.info(
        "VIDEO_PLAN_DIAG interactionSummary=%s",
        (plan.get("interactionSummary") or "")[:300],
    )
    logger.info(
        "VIDEO_PLAN_DIAG advertisingPromise=%s",
        (plan.get("advertisingPromise") or "")[:300],
    )
    logger.info("VIDEO_PLAN_DIAG headlineText=%s", headline_full[:300])
    logger.info("VIDEO_PLAN_DIAG headline_remainder=%s", headline_remainder[:300])

    optional_fields = (
        ("headlineOriginalExpression", "original_expression"),
        ("headlineReplacedWord", "replaced_word"),
        ("headlineReplacementObject", "replacement_object"),
        ("headlineRhymeReason", "rhyme_reason"),
        ("headline_original_expression", "original_expression"),
        ("headline_replaced_word", "replaced_word"),
        ("headline_replacement_object", "replacement_object"),
        ("headline_rhyme_reason", "rhyme_reason"),
        ("final_headline_remainder", "final_headline_remainder"),
    )
    logged_diag_keys: set[str] = set()
    for plan_key, log_key in optional_fields:
        if log_key in logged_diag_keys:
            continue
        value = (plan.get(plan_key) or "").strip()
        if value:
            logger.info("VIDEO_PLAN_DIAG %s=%s", log_key, value[:300])
            logged_diag_keys.add(log_key)


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
    logger.info("VIDEO_PLAN_SEARCH_ORDER=single_interaction_v3")
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

{_planner_object_selection_rules_block()}
{_planner_causal_reasoning_block()}
{_planner_physical_realism_block()}
{_planner_headline_rules_user_block(lang)}
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
    used_object_a = get_used_object_a("builder2")
    used_headlines = get_used_headlines("builder2")
    logger.info(
        "VIDEO_PLAN_MEMORY_OBJECT_A_USED engine=builder2 count=%s",
        len(used_object_a),
    )
    logger.info(
        "VIDEO_PLAN_MEMORY_HEADLINES_USED engine=builder2 count=%s",
        len(used_headlines),
    )
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
    object_a_memory_addon = ""
    if used_object_a:
        object_a_memory_addon = (
            "\n\nObject A memory for Builder2 (avoid reusing these Object A values):\n"
            f"- {', '.join(used_object_a)}\n"
            "- Avoid reusing any Object A listed above.\n"
        )
    headline_memory_addon = ""
    if used_headlines:
        headline_memory_addon = (
            "\n\nHeadline idea memory for Builder2 (avoid reusing these headline ideas):\n"
            f"- {', '.join(used_headlines)}\n"
            "- Avoid reusing headline ideas listed above.\n"
        )
    attempt_input = (
        instructions
        + "\n\n"
        + user_block
        + promise_addon
        + object_a_memory_addon
        + headline_memory_addon
    )
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
    logger.info("VIDEO_HEADLINE_RULE=rhyming_object_substitution")
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
            last_v_err = (v_err or "").strip() or "planning_failed_incomplete_plan"
            if last_v_err == "headline_hebrew_product_tail":
                logger.info("VIDEO_PLAN_REPAIR_REQUESTED reason=%s", last_v_err)
                repair_input = _build_headline_hebrew_product_tail_repair_input(
                    base_attempt_input=attempt_input,
                    product_name=product_name,
                    previous_plan=parsed,
                )
                try:
                    repair_response = _responses_create_with_plan_retry(
                        client,
                        model=model,
                        input_text=repair_input,
                        reasoning={"effort": _reasoning_effort()},
                        deadline_monotonic=deadline_monotonic,
                    )
                except VideoPlanningTimeoutError:
                    raise
                except Exception as e:
                    logger.warning(
                        "VIDEO_PLAN_REPAIR_FAILED reason=%s err_type=%s err=%s",
                        "planning_failed_model_call",
                        type(e).__name__,
                        e,
                    )
                    logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_FAIL")
                    rescued = _rescue_plan_headline_hebrew_product_tail(
                        parsed,
                        planner_deadline_monotonic=deadline_monotonic,
                        product_name=product_name,
                        product_description=product_description,
                        content_language=content_language,
                    )
                    if rescued:
                        plan = rescued
                        logger.info(
                            "VIDEO_PLAN_REPAIR_OK reason=headline_hebrew_product_tail_final_fallback"
                        )
                    else:
                        logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                        logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                        return None, "planning_failed_incomplete_plan"
                else:
                    repair_raw = _extract_responses_output_text(repair_response)
                    repair_parsed = _parse_json_from_response(repair_raw or "")
                    if not repair_parsed:
                        logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_FAIL")
                        rescued = _rescue_plan_headline_hebrew_product_tail(
                            parsed,
                            planner_deadline_monotonic=deadline_monotonic,
                            product_name=product_name,
                            product_description=product_description,
                            content_language=content_language,
                        )
                        if rescued:
                            plan = rescued
                            logger.info(
                                "VIDEO_PLAN_REPAIR_OK reason=headline_hebrew_product_tail_final_fallback"
                            )
                        else:
                            logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                            logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                            return None, "planning_failed_incomplete_plan"
                    else:
                        merged = dict(parsed)
                        new_ht = (repair_parsed.get("headlineText") or "").strip()
                        if new_ht:
                            merged["headlineText"] = new_ht
                        repaired_plan, repaired_err = validate_and_normalize_plan(
                            merged,
                            planner_deadline_monotonic=deadline_monotonic,
                            product_name=product_name,
                            product_description=product_description,
                            content_language=content_language,
                        )
                        if repaired_plan:
                            plan = repaired_plan
                            logger.info("VIDEO_PLAN_REPAIR_OK reason=headline_hebrew_product_tail")
                        else:
                            logger.info("VIDEO_HEADLINE_STRUCT_REPAIR_FAIL")
                            rescued = _rescue_plan_headline_hebrew_product_tail(
                                merged,
                                planner_deadline_monotonic=deadline_monotonic,
                                product_name=product_name,
                                product_description=product_description,
                                content_language=content_language,
                            )
                            if rescued:
                                plan = rescued
                                logger.info(
                                    "VIDEO_PLAN_REPAIR_OK reason=headline_hebrew_product_tail_final_fallback"
                                )
                            else:
                                logger.error(
                                    "VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s",
                                    (repaired_err or last_v_err).strip(),
                                )
                                logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                                return None, "planning_failed_incomplete_plan"
            elif last_v_err in {
                "planning_failed_unrealistic_physics",
                "planning_failed_invalid_objects",
            }:
                logger.info("VIDEO_PLAN_REPAIR_REQUESTED reason=%s", last_v_err)
                if last_v_err == "planning_failed_unrealistic_physics":
                    repair_input = _build_unrealistic_physics_repair_input(
                        base_attempt_input=attempt_input,
                        product_name=product_name,
                        product_description=product_description,
                        advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
                        previous_plan=parsed,
                    )
                else:
                    repair_input = _build_invalid_objects_repair_input(
                        base_attempt_input=attempt_input,
                        product_name=product_name,
                        product_description=product_description,
                        advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
                        previous_plan=parsed,
                    )
                try:
                    repair_response = _responses_create_with_plan_retry(
                        client,
                        model=model,
                        input_text=repair_input,
                        reasoning={"effort": _reasoning_effort()},
                        deadline_monotonic=deadline_monotonic,
                    )
                except VideoPlanningTimeoutError:
                    raise
                except Exception as e:
                    logger.warning(
                        "VIDEO_PLAN_REPAIR_FAILED reason=%s err_type=%s err=%s",
                        "planning_failed_model_call",
                        type(e).__name__,
                        e,
                    )
                    logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                    if last_v_err == "planning_failed_unrealistic_physics":
                        logger.warning("VIDEO_PLAN_FALLBACK_TRIGGERED reason=physics_failed")
                        fallback_raw = _build_physics_safe_fallback_plan(
                            product_name=(parsed.get("productNameResolved") or product_name),
                            product_description=product_description,
                            content_language=content_language,
                            advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
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
                            return None, last_v_err
                    else:
                        return None, last_v_err

                repair_raw = _extract_responses_output_text(repair_response)
                repair_parsed = _parse_json_from_response(repair_raw or "")
                if not repair_parsed:
                    logger.warning(
                        "VIDEO_PLAN_REPAIR_FAILED reason=%s",
                        "planning_failed_malformed_response",
                    )
                    logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                    if last_v_err == "planning_failed_unrealistic_physics":
                        logger.warning("VIDEO_PLAN_FALLBACK_TRIGGERED reason=physics_failed")
                        fallback_raw = _build_physics_safe_fallback_plan(
                            product_name=(parsed.get("productNameResolved") or product_name),
                            product_description=product_description,
                            content_language=content_language,
                            advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
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
                            return None, last_v_err
                    else:
                        return None, last_v_err

                repair_pf_raw = str(repair_parsed.get("planningFailure") or "").strip()
                if repair_pf_raw:
                    repair_code = (
                        repair_pf_raw
                        if repair_pf_raw in _PLANNER_SELF_FAILURE_CODES
                        else "planning_failed_no_valid_interaction"
                    )
                    logger.warning("VIDEO_PLAN_REPAIR_FAILED reason=%s", repair_code)
                    logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                    if last_v_err == "planning_failed_unrealistic_physics":
                        logger.warning("VIDEO_PLAN_FALLBACK_TRIGGERED reason=physics_failed")
                        fallback_raw = _build_physics_safe_fallback_plan(
                            product_name=(parsed.get("productNameResolved") or product_name),
                            product_description=product_description,
                            content_language=content_language,
                            advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
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
                            return None, last_v_err
                    else:
                        return None, last_v_err

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
                else:
                    logger.warning(
                        "VIDEO_PLAN_REPAIR_FAILED reason=%s",
                        (repaired_err or "").strip() or "planning_failed_incomplete_plan",
                    )
                    logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                    logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                    if last_v_err == "planning_failed_unrealistic_physics":
                        logger.warning("VIDEO_PLAN_FALLBACK_TRIGGERED reason=physics_failed")
                        fallback_raw = _build_physics_safe_fallback_plan(
                            product_name=(parsed.get("productNameResolved") or product_name),
                            product_description=product_description,
                            content_language=content_language,
                            advertising_promise=(parsed.get("advertisingPromise") or "").strip(),
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
                            return None, last_v_err
                    else:
                        return None, last_v_err
            else:
                logger.error("VIDEO_PLAN_FAIL_STRUCT_NORMALIZE reason=%s", last_v_err)
                logger.info("VIDEO_PLAN_RESPONSE_OK=false")
                return None, last_v_err

        log_plan_summary(plan)
        logger.info("VIDEO_PLAN_OK model=%s", model)
        _log_video_plan_post_ok_diagnostics(plan)
        logger.info("VIDEO_PLAN_RESPONSE_OK=true")
        object_a_value = (plan.get("objectA") or "").strip()
        if object_a_value:
            remember_object_a_ace("builder2", object_a_value)
        headline_full = (plan.get("headlineText") or "").strip()
        product_resolved = (plan.get("productNameResolved") or "").strip()
        headline_without_product = headline_full
        if product_resolved and headline_full.startswith(product_resolved + " "):
            headline_without_product = headline_full[len(product_resolved) + 1 :].strip()
        if headline_without_product:
            remember_headline("builder2", headline_without_product)
        logger.info(
            "VIDEO_HEADLINE_RHYME final_headline_remainder=%s",
            headline_without_product[:200],
        )
        hderiv = (plan.get("headlineDerivation") or "").strip()
        if hderiv:
            logger.info("VIDEO_HEADLINE_RHYME headlineDerivation=%s", hderiv[:300])
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
    for marker in _RUNWAY_INTERACTION_TAIL_MARKERS:
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


def _build_runway_prompt_compact_fallback(plan: Dict[str, Any]) -> Tuple[str, bool]:
    """Shorter ACE→Runway bridge if the detailed builder fails."""
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    motion, _ = _runway_camera_motion_focus(plan)
    lang_vis = _runway_language_visual_constraints(plan)
    parts = [
        "VISUAL POLICY: No readable text, letters, words, logos, captions, labels, signage, or title cards in-frame.",
        lang_vis,
        f"Single continuous shot: {oa} and {ob}. {motion}",
        f"Physical interaction: {script}" if script else "",
    ]
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

    motion, _ = _runway_camera_motion_focus(plan)
    lang_vis = _runway_language_visual_constraints(plan)
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"Single continuous shot. Two physical objects: {oa} and {ob}. "
        f"{motion}"
        f"Physical interaction (follow exactly): {script}. "
        f"{_RUNWAY_CONTACT_EXECUTION_CLAUSE} "
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
    _, motion_mode = _runway_camera_motion_focus(plan)
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
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

    motion_focus, _ = _runway_camera_motion_focus(plan)
    lang_vis = _runway_language_visual_constraints(plan)
    scene = (
        f"{lang_vis} "
        f"The first frame is supplied as the start image; it already shows {oa} and {ob} together, "
        f"both clearly visible and balanced. {motion_focus}"
        f"Physical interaction (follow exactly): {script}"
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
    """Shorter interaction-only bridge if the detailed interaction builder fails."""
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()
    script = (plan.get("interactionScript") or "").strip()
    lang_vis = _runway_language_visual_constraints(plan)
    motion_focus, _ = _runway_camera_motion_focus(plan)
    motion = (
        f"{lang_vis} "
        f"Start frame supplied; {oa} and {ob} already visible together. "
        f"{motion_focus}"
        f"Physical interaction: {script}."
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
    _, motion_mode = _runway_camera_motion_focus(plan)
    logger.info("VIDEO_PROMPT_CAMERA_MOTION motion=%s", motion_mode)
    out = f"{out.rstrip()} {RUNWAY_PHYSICS_REALISM_CONSTRAINT}".strip()
    return out
