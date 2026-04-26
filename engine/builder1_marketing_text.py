"""
Builder1 marketing text generation (~50 words) via o3-pro.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeAlias

logger = logging.getLogger(__name__)

MarketingTextModelCaller: TypeAlias = Callable[[str, str], str]


def _normalize_paragraph(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def generate_builder1_marketing_text_o3(
    *,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    advertising_promise: str,
    object_a: str,
    object_a_secondary: str,
    object_b: str,
    mode_decision: str,
    visual_description: str,
    visual_prompt: str,
    headline_product_name: str,
    headline_text: str,
    headline_full: str,
    model_caller: MarketingTextModelCaller,
) -> str:
    system_prompt = (
        "Write one short marketing paragraph only.\n"
        "Rules:\n"
        "- About 50 words.\n"
        "- Use advertisingPromise only as a validation signal, not as a wording source.\n"
        "- Derive the text from the final visual and headline meaning only.\n"
        "- Derive specifically from: objectA/objectB visual relationship, overlap/replacement/interaction, and objectASecondary when relevant.\n"
        "- The text should feel like a short extension of the ad idea.\n"
        "- Do not explain the product directly.\n"
        "- Do not restate the advertising promise.\n"
        "- Do not write generic benefit copy.\n"
        "- Be specific and non-generic.\n"
        "- Use the same language as detectedLanguage.\n"
        "- One plain paragraph only.\n"
        "- Do not write direct benefit statements.\n"
        "- Do not describe the product in generic marketing language.\n"
        "- Do not simply explain what the user's business does.\n"
        "- No hashtags.\n"
        "- No bullets.\n"
        "- No quotes.\n"
        "- No markdown.\n"
        "- Return plain text only.\n"
    )
    user_prompt = (
        f"productNameResolved: {product_name_resolved}\n"
        f"productDescription: {product_description}\n"
        f"detectedLanguage: {detected_language}\n"
        f"advertisingPromise: {advertising_promise}\n"
        f"objectA: {object_a}\n"
        f"objectASecondary: {object_a_secondary}\n"
        f"objectB: {object_b}\n"
        f"modeDecision: {mode_decision}\n"
        f"visualDescription: {visual_description}\n"
        f"visualPrompt: {visual_prompt}\n"
        f"headlineProductName: {headline_product_name}\n"
        f"headlineText: {headline_text}\n"
        f"headlineFull: {headline_full}\n"
    )
    raw = model_caller(system_prompt, user_prompt)
    text = _normalize_paragraph(raw)
    if not text:
        raise ValueError("marketing_text_empty")
    logger.info(
        "BUILDER1_MARKETING_TEXT_OK word_count=%s detected_language=%r",
        len(text.split()),
        detected_language,
    )
    return text
