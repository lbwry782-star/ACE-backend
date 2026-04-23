"""
Tiny disconnected dry examples for the Builder1 scaffold pipeline.
"""
from __future__ import annotations

from engine.builder1_pipeline_dry import Builder1DryRunResult, run_builder1_dry_pipeline

EXAMPLE_SIDE_BY_SIDE_PAYLOAD = {
    "productNameResolved": "AeroSip Bottle",
    "detectedLanguage": "en",
    "advertisingPromise": "Feels light and effortless to carry all day.",
    "objectA": "stainless steel water bottle",
    "objectASecondary": "carabiner clip",
    "objectB": "sleek thermos flask",
    "visualSimilarityScore": 72,
    "modeDecision": "SIDE_BY_SIDE",
    "visualDescription": "Clean product shot with matching curved silhouettes and subtle overlap.",
}

EXAMPLE_REPLACEMENT_PAYLOAD = {
    "productNameResolved": "PureSlice Knife",
    "detectedLanguage": "en",
    "advertisingPromise": "Cuts with precise control and confidence.",
    "objectA": "chef knife",
    "objectASecondary": "wooden cutting board",
    "objectB": "precision scalpel",
    "visualSimilarityScore": 91,
    "modeDecision": "REPLACEMENT",
    "visualDescription": "Hero product close-up emphasizing crisp metallic edges and exact alignment.",
}

EXAMPLE_SIDE_BY_SIDE_INPUT = {
    "product_name": "AeroSip",
    "product_description": "Insulated reusable bottle for everyday commuting and workouts.",
    "format_value": "landscape",
}

EXAMPLE_REPLACEMENT_INPUT = {
    "product_name": "PureSlice",
    "product_description": "Professional kitchen knife designed for accurate, smooth slicing.",
    "format_value": "portrait",
}


def build_side_by_side_example() -> Builder1DryRunResult:
    return run_builder1_dry_pipeline(
        product_name=EXAMPLE_SIDE_BY_SIDE_INPUT["product_name"],
        product_description=EXAMPLE_SIDE_BY_SIDE_INPUT["product_description"],
        format_value=EXAMPLE_SIDE_BY_SIDE_INPUT["format_value"],
        model_payload=EXAMPLE_SIDE_BY_SIDE_PAYLOAD,
    )


def build_replacement_example() -> Builder1DryRunResult:
    return run_builder1_dry_pipeline(
        product_name=EXAMPLE_REPLACEMENT_INPUT["product_name"],
        product_description=EXAMPLE_REPLACEMENT_INPUT["product_description"],
        format_value=EXAMPLE_REPLACEMENT_INPUT["format_value"],
        model_payload=EXAMPLE_REPLACEMENT_PAYLOAD,
    )
