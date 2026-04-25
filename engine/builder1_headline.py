"""
Builder1 headline text via o3-pro (not burned into images).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from openai import OpenAI

from engine.builder1_memory import get_builder1_memory_snapshot, remember_headline_text

logger = logging.getLogger(__name__)


def _parse_json_object(raw: str) -> dict[str, Any]:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
    t = t.strip()
    if t.lower().startswith("```json"):
        t = t[7:].lstrip()
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model_output_not_object")
    return obj


def generate_builder1_headline_o3(
    *,
    product_name_resolved: str,
    detected_language: str,
    advertising_promise: str,
    object_a: str,
    object_a_secondary: str,
    object_b: str,
    mode_decision: str,
    visual_description: str,
    visual_prompt: str,
) -> dict[str, str]:
    resolved = (product_name_resolved or "").strip()
    if not resolved:
        raise ValueError("missing_product_name_resolved")

    memory = get_builder1_memory_snapshot()
    remembered_headlines = memory.get("headline_text") or []
    recent_headlines = remembered_headlines[-10:]
    logger.info(
        "BUILDER1_MEMORY_INJECTED_TO_HEADLINE headline_count=%s recent_headlines=%r",
        len(remembered_headlines),
        recent_headlines,
    )

    system = (
        "Return exactly one JSON object, no markdown, no prose. Keys only:\n"
        '{"headlineProductName":"...","headlineText":"...","headlineFull":"..."}\n'
        "Rules:\n"
        "- Write in the same language as detectedLanguage (he or en).\n"
        "- The advertising promise is already resolved inside the visual. The headline must not create or restate it.\n"
        "- Generate headlineText ONLY from the visual: objectA, objectB, their overlap/interaction, visualDescription, and visualPrompt.\n"
        "- advertisingPromise is used only to validate the choice, NOT to generate wording.\n"
        "- Prefer an existing expression or familiar phrase when possible.\n"
        "- Use a word/meaning/inflection that emerges from the visual content.\n"
        "- The expression must feel like something people already say.\n"
        "- The correct headline should feel like the viewer recognizes a known expression that suddenly makes sense because of the visual.\n"
        "- A correct headline must not only describe what is seen, but express a meaning or implication that emerges from the visual.\n"
        "- headlineProductName must exactly match the given productNameResolved string.\n"
        "- The model controls headlineText only; headlineProductName is fixed by backend.\n"
        "- Do not include product description inside headlineProductName.\n"
        "- headlineText is the slogan/title phrase only (do not repeat the product name in headlineText).\n"
        "- headlineFull must be exactly headlineProductName, one ASCII space, then headlineText.\n"
        "- headlineFull must be at most 7 words total (count words on headlineFull).\n"
        "- Product name must stay as the first headline line and is visually larger; do not move/alter it.\n"
        "- Do not create a slogan that merely explains the product.\n"
        "- Do NOT write explanatory slogans or direct benefit statements such as: 'מכפיל את הווליום הדיגיטלי', 'מגביר את הקול', or any direct phrasing of the advertising promise.\n"
        "- Phrases like 'מסך מול מסך' are too descriptive and lack meaning.\n"
        "- The headline should feel like a familiar expression that gains meaning from the visual, not just mirrors it.\n"
        "- Prefer concise wordplay tied to the visual cues.\n"
        "- Headline memory excludes product names; compare only the slogan/title phrase part.\n"
        "- Do not reuse any previous headlineText from memory.\n"
        "- Do not reuse the same familiar expression; choose a fresh phrasing.\n"
        "Example:\n"
        "- Product name: אורי לב\n"
        "- Product description: סוכן פרסום\n"
        "- Object A: empty vertical advertising sign\n"
        "- Object A secondary: sign pole\n"
        "- Object B: king of hearts card\n"
        "- Mode: REPLACEMENT\n"
        "- Visual: king-of-hearts-card sign on a pole\n"
        "- Good headline:\n"
        "  אורי לב\n"
        "  קלף בפרסום\n"
    )
    user = (
        f"detectedLanguage: {detected_language}\n"
        f"productNameResolved: {resolved}\n"
        f"advertisingPromise: {advertising_promise}\n"
        f"objectA: {object_a}\n"
        f"objectASecondary: {object_a_secondary}\n"
        f"objectB: {object_b}\n"
        f"modeDecision: {mode_decision}\n"
        f"visualDescription: {visual_description}\n"
        f"visualPrompt: {visual_prompt}\n"
        f"headlineTextMemoryToAvoid: {', '.join(remembered_headlines)}\n"
    )
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    combined = f"{system.strip()}\n\n{user.strip()}"
    response = client.responses.create(
        model="o3-pro",
        input=combined,
        reasoning={"effort": "low"},
    )
    out_text = getattr(response, "output_text", None) or ""
    data = _parse_json_object(out_text)
    hpn = resolved
    htt = (data.get("headlineText") or "").strip()
    if not hpn or not htt:
        raise ValueError("headline_empty_field")
    hfull = " ".join(f"{hpn} {htt}".split())
    norm_full = hfull
    if len(norm_full.split()) > 7:
        raise ValueError("headline_too_long")
    logger.info("BUILDER1_MEMORY_HEADLINE_REMEMBER_CALL headline_text=%r", htt)
    remember_headline_text(htt)
    return {
        "headlineProductName": hpn,
        "headlineText": htt,
        "headlineFull": hfull,
    }
