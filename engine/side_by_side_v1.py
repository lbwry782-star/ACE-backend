"""Builder1 side-by-side pipeline scaffold (no model calls yet)."""

from __future__ import annotations

import json
import os

import httpx
from openai import OpenAI


def mode_from_similarity(morphological_similarity: int) -> str:
    """REPLACEMENT if morphological similarity >= 85, otherwise SIDE_BY_SIDE."""
    return "REPLACEMENT" if morphological_similarity >= 85 else "SIDE_BY_SIDE"


def get_concept_from_o3(product_name: str, product_description: str) -> dict:
    """One o3-pro call: structured concept JSON for Builder1. Raises ValueError if JSON is invalid."""
    pn = (product_name or "").strip() or "Product"
    desc = (product_description or "").strip() or "No description provided."
    user_input = f"""Product name: {pn}
Product description: {desc}

Return ONLY valid JSON with exactly these keys and no others (no markdown fences):
{{ "objectA": "<string>", "objectB": "<string>", "advertisingPromise": "<string>", "morphologicalSimilarity": <integer 0-100>, "reasoning": "<string>" }}

Rules for the model:
- Choose Object A from the product name and description by grasping its overall physical form intuitively, like a painter, not as a technical contour only.
- Then find Object B with the strongest possible morphological similarity to A.
- Stop only when B also introduces an additional conceptual reason, which is the advertising promise (the advertisingPromise field).
- Do not choose a weaker B just to make the advertising promise more obvious.
- Trust viewer intuition.
- objectA and objectB must be physical, simple, everyday, clearly defined. No text, logos, or brands. No vague environments or non-physical situations.
- Keep reasoning short.
"""

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    response = client.responses.create(
        model="o3-pro",
        input=user_input,
        reasoning={"effort": "low"},
    )
    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw:
        raise ValueError("BUILDER1_CONCEPT_O3: empty output_text from o3-pro")

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    if cleaned.startswith("```json"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"BUILDER1_CONCEPT_O3: invalid JSON from o3-pro: {e}") from e

    required = (
        "objectA",
        "objectB",
        "advertisingPromise",
        "morphologicalSimilarity",
        "reasoning",
    )
    if not isinstance(data, dict):
        raise ValueError("BUILDER1_CONCEPT_O3: parsed JSON root is not an object")
    if set(data.keys()) != set(required):
        raise ValueError(
            f"BUILDER1_CONCEPT_O3: JSON keys must be exactly {list(required)}, got {list(data.keys())}"
        )

    for key in ("objectA", "objectB", "advertisingPromise", "reasoning"):
        val = data[key]
        if not isinstance(val, str):
            raise ValueError(
                f"BUILDER1_CONCEPT_O3: {key} must be a string, got {type(val).__name__}"
            )

    sim = data["morphologicalSimilarity"]
    try:
        sim_i = int(sim)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"BUILDER1_CONCEPT_O3: morphologicalSimilarity must be an integer 0-100, got {sim!r}"
        ) from e
    if sim_i < 0 or sim_i > 100:
        raise ValueError(f"BUILDER1_CONCEPT_O3: morphologicalSimilarity out of range: {sim_i}")

    return {
        "objectA": data["objectA"],
        "objectB": data["objectB"],
        "advertisingPromise": data["advertisingPromise"],
        "morphologicalSimilarity": sim_i,
        "reasoning": data["reasoning"],
    }


def build_image_prompt(objectA: str, objectB: str, mode: str) -> str:
    """Build gpt-image-1.5 photorealistic prompt (placeholder)."""
    return ""


def generate_headline(
    objectA: str, objectB: str, advertisingPromise: str, product_name: str
) -> str:
    """Generate headline via o3-pro (placeholder)."""
    return ""


def generate_marketing_text(headline: str, advertisingPromise: str) -> str:
    """Generate ~50 word marketing text (placeholder)."""
    return ""


def generate_builder1_ad(product_name: str, product_description: str) -> dict:
    """Orchestrate Builder1 pipeline (placeholder)."""
    _ = product_name, product_description
    return {
        "objectA": "",
        "objectB": "",
        "advertisingPromise": "",
        "mode": mode_from_similarity(0),
        "headline": "",
        "imagePrompt": "",
        "marketingText": "",
    }
