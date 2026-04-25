"""
Tiny disconnected end-to-end demo for the Builder1 generate flow.
"""
from __future__ import annotations

from engine.builder1_generate_flow import Builder1GenerateResult, generate_builder1_ad

DEMO_PLANNING_PAYLOAD = {
    "productNameResolved": "AeroSip Bottle",
    "detectedLanguage": "en",
    "advertisingPromise": "Feels light and effortless to carry all day.",
    "objectA": "stainless steel water bottle",
    "objectASecondary": "carabiner clip",
    "objectB": "sleek thermos flask",
    "visualSimilarityScore": 88,
    "modeDecision": "REPLACEMENT",
    "visualDescription": "Clean product shot with crisp edges and minimal studio styling.",
}


def fake_planning_model_caller(system_prompt: str, user_prompt: str) -> object:
    del system_prompt, user_prompt
    return DEMO_PLANNING_PAYLOAD


def fake_image_caller(prompt: str, format_value: str) -> bytes:
    del prompt, format_value
    return b"demo-image-bytes"


def build_demo_ad() -> Builder1GenerateResult:
    return generate_builder1_ad(
        product_name="AeroSip",
        product_description="Insulated reusable bottle for everyday commuting and workouts.",
        format_value="landscape",
        planning_model_caller=fake_planning_model_caller,
        image_caller=fake_image_caller,
    )
