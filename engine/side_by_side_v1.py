"""Builder1 side-by-side pipeline scaffold (no model calls yet)."""

from __future__ import annotations


def mode_from_similarity(morphological_similarity: int) -> str:
    """REPLACEMENT if morphological similarity >= 85, otherwise SIDE_BY_SIDE."""
    return "REPLACEMENT" if morphological_similarity >= 85 else "SIDE_BY_SIDE"


def get_concept_from_o3(product_name: str, product_description: str) -> dict:
    """Call o3-pro for structured concept JSON (placeholder)."""
    return {}


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
