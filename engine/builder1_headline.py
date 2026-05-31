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

from engine.ace_usage_memory import get_used_headlines, remember_headline

logger = logging.getLogger(__name__)

_MODE_SIDE_BY_SIDE = "SIDE_BY_SIDE"
_MODE_REPLACEMENT = "REPLACEMENT"


def _norm_object_name(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _builder1_headline_rhyming_substitution_block() -> str:
    return (
        "HEADLINE (rhyming object substitution — mandatory for headlineText remainder):\n"
        "1. First find an existing familiar expression, idiom, proverb, or well-known phrase that expresses the advertisingPromise.\n"
        "2. The original expression must NOT already contain the name or core word of Object A or Object B.\n"
        "3. Choose exactly one word inside that expression.\n"
        "4. Replace that one word with the name of Object A or Object B (natural headline-language form).\n"
        "5. The replacement must be based on strong phonetic similarity, not rhyme alone.\n"
        "6. The inserted Object A/Object B name must preserve most of the sound structure of the replaced word.\n"
        "7. Prefer replacements where the object name differs by only one small sound, syllable, or letter cluster.\n"
        "8. Simple end-rhyme is not enough.\n"
        "9. If the audience cannot immediately hear the original expression behind the twist, reject it.\n"
        "10. Invalid weak substitution example: original word חשבון → replacement object סבון — only end-rhyme; too much of the original word is lost; the original expression is not immediately recognizable.\n"
        "11. If only a weak rhyme exists, choose another expression/object substitution instead. Do not force a weak substitution.\n"
        "12. The result must still feel like a recognizable twist on the original expression.\n"
        "13. The final headline remainder must visibly differ from the original expression.\n"
        "14. The viewer must immediately notice the substituted object word.\n"
        "15. Do not use a substitution if the final phrase reads exactly like the original expression.\n"
        "16. Do not replace a word with an object name that is already hidden inside the original expression or naturally contained across adjacent letters.\n"
        "17. The replacement must create a visible, readable twist, not only an internal spelling explanation.\n"
        "18. Invalid unchanged-substitution example: original_expression השלם גדול מסך חלקיו, replaced_word סך, replacement_object מסך, final_headline_remainder השלם גדול מסך חלקיו — the final phrase reads exactly like the original expression; no visible twist; forbidden.\n"
        "19. Do not add extra words before, inside, or after the twisted expression.\n"
        "20. headlineProductName must exactly match productNameResolved (backend-fixed). headlineText is the twisted expression remainder only — do not repeat the product name in headlineText.\n"
        "21. headlineFull must be exactly headlineProductName, one ASCII space, then headlineText. ≤7 words total on headlineFull.\n"
        "22. headlineText must express the advertisingPromise through the visual interaction — not by restating the promise literally.\n"
        "23. Do NOT pick an expression that already contains the object word before substitution.\n"
    )


def _builder1_headline_mode_block(mode_decision: str) -> str:
    mode = (mode_decision or "").strip().upper()
    if mode == _MODE_REPLACEMENT:
        return (
            "Mode REPLACEMENT:\n"
            "- The replacement object may be Object A or Object B only.\n"
            "- Object A secondary is visual/context only — do NOT use it for headline substitution.\n"
            "- Object A and Object B names must be distinct; if they are the same or normalize to the same name, this pair is invalid.\n"
        )
    return (
        "Mode SIDE_BY_SIDE:\n"
        "- The replacement object may be Object A or Object B.\n"
    )


def _log_headline_rhyme_diagnostics(data: dict[str, Any], *, final_remainder: str) -> None:
    logger.info("BUILDER1_HEADLINE_RHYME final_headline_remainder=%s", final_remainder[:200])
    for key, log_key in (
        ("headlineOriginalExpression", "original_expression"),
        ("headlineReplacedWord", "replaced_word"),
        ("headlineReplacementObject", "replacement_object"),
        ("headlineRhymeReason", "rhyme_reason"),
    ):
        value = (data.get(key) or "").strip()
        if value:
            logger.info("BUILDER1_HEADLINE_RHYME %s=%s", log_key, value[:300])


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

    mode = (mode_decision or "").strip().upper()
    a_norm = _norm_object_name(object_a)
    b_norm = _norm_object_name(object_b)
    if mode == _MODE_REPLACEMENT and a_norm and b_norm and a_norm == b_norm:
        raise ValueError("headline_replacement_object_a_b_identical")

    used_headlines_ace = get_used_headlines("builder1")
    recent_headlines_ace = used_headlines_ace[-10:]
    logger.info(
        "BUILDER1_MEMORY_INJECTED_TO_HEADLINE_ACE headline_count=%s recent_headlines=%r",
        len(used_headlines_ace),
        recent_headlines_ace,
    )

    rhyme_block = _builder1_headline_rhyming_substitution_block()
    mode_block = _builder1_headline_mode_block(mode)
    system = (
        "Return exactly one JSON object, no markdown, no prose. Required keys only:\n"
        '{"headlineProductName":"...","headlineText":"...","headlineFull":"..."}\n'
        "Optional diagnostic keys (may be omitted): headlineOriginalExpression, headlineReplacedWord, "
        "headlineReplacementObject, headlineRhymeReason.\n"
        "Rules:\n"
        "- Write in the same language as detectedLanguage (he or en).\n"
        "- The advertising promise is already resolved inside the visual. The headline must not create or restate it literally.\n"
        "- Use objectA, objectB, visualDescription, and visualPrompt as context for how the twisted expression connects to the visual.\n"
        f"{rhyme_block}"
        f"{mode_block}"
        "- headlineProductName must exactly match the given productNameResolved string.\n"
        "- The model controls headlineText only; headlineProductName is fixed by backend.\n"
        "- Do not include product description inside headlineProductName.\n"
        "- Product name must stay as the first headline line and is visually larger; do not move/alter it.\n"
        "- Do not create a slogan that merely explains the product.\n"
        "- Do NOT write explanatory slogans or direct benefit statements such as: 'מכפיל את הווליום הדיגיטלי', 'מגביר את הקול', or any direct phrasing of the advertising promise.\n"
        "- Phrases like 'מסך מול מסך' are too descriptive and lack meaning.\n"
        "- The headline is not a literal visual description.\n"
        "- Headline memory excludes product names; compare only the slogan/title phrase part.\n"
        "- Do not reuse any previous headlineText from memory.\n"
        "- Do not reuse the same familiar expression; choose a fresh phrasing.\n"
        "Example:\n"
        "- Product name: אורי לב\n"
        "- Object A: empty vertical advertising sign\n"
        "- Object A secondary: sign pole\n"
        "- Object B: king of hearts card\n"
        "- Mode: REPLACEMENT\n"
        "- Visual: king-of-hearts-card sign on a pole\n"
        "- Original expression (before substitution): כלב בפרסום\n"
        "- Replaced word: כלב → replacement object: קלף (Object B; one small sound/letter shift; original expression still immediately audible)\n"
        "- Invalid weak substitution (do NOT use): חשבון → סבון — only end-rhyme; original expression not immediately recognizable\n"
        "- Invalid unchanged substitution (do NOT use): original_expression השלם גדול מסך חלקיו → final_headline_remainder השלם גדול מסך חלקיו — reads exactly like the original; no visible twist\n"
        "- Good headline:\n"
        "  headlineProductName: אורי לב\n"
        "  headlineText: קלף בפרסום\n"
        "  headlineFull: אורי לב קלף בפרסום\n"
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
        f"headlineTextMemoryToAvoidACE: {', '.join(used_headlines_ace)}\n"
    )
    logger.info("BUILDER1_HEADLINE_RULE=rhyming_object_substitution")
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
    headline_without_product_name = htt
    hfull_norm = " ".join((hfull or "").split())
    hpn_norm = " ".join((hpn or "").split())
    if hpn_norm and hfull_norm.startswith(f"{hpn_norm} "):
        headline_without_product_name = hfull_norm[len(hpn_norm) + 1 :].strip()
    if (headline_without_product_name or "").strip():
        remember_headline("builder1", headline_without_product_name)
    _log_headline_rhyme_diagnostics(data, final_remainder=headline_without_product_name)
    return {
        "headlineProductName": hpn,
        "headlineText": htt,
        "headlineFull": hfull,
    }
