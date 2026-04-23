"""
Disconnected Builder1 planning contract for a future single o3-pro call.
"""
from __future__ import annotations

BUILDER1_PLANNING_SYSTEM_PROMPT: str = """
Return strict JSON only, with no markdown and no extra keys.
Output must contain exactly these fields:
- productNameResolved
- detectedLanguage
- advertisingPromise
- objectA
- objectASecondary
- objectB
- visualSimilarityScore
- modeDecision
- visualDescription

Rules:
- Choose Object A from product name + product description.
- Grasp overall form like a painter, not only technical contour.
- Find Object B with strong morphological similarity to Object A.
- Stop when Object B also expresses the advertising promise.
- Filter out text, logos, brands, generic environments, and unclear/non-physical situations.
- All objects must be classic, defined, physical objects.
- detectedLanguage must be "he" or "en".
- modeDecision must be "REPLACEMENT" if visualSimilarityScore >= 85; otherwise "SIDE_BY_SIDE".
""".strip()


BUILDER1_PLANNING_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "productNameResolved",
        "detectedLanguage",
        "advertisingPromise",
        "objectA",
        "objectASecondary",
        "objectB",
        "visualSimilarityScore",
        "modeDecision",
        "visualDescription",
    ],
    "properties": {
        "productNameResolved": {"type": "string"},
        "detectedLanguage": {"type": "string", "enum": ["he", "en"]},
        "advertisingPromise": {"type": "string"},
        "objectA": {"type": "string"},
        "objectASecondary": {"type": "string"},
        "objectB": {"type": "string"},
        "visualSimilarityScore": {"type": "integer", "minimum": 0, "maximum": 100},
        "modeDecision": {"type": "string", "enum": ["REPLACEMENT", "SIDE_BY_SIDE"]},
        "visualDescription": {"type": "string"},
    },
}


def build_builder1_planning_user_prompt(
    product_name: str, product_description: str, format_value: str
) -> str:
    return (
        f"Product name: {product_name}\n"
        f"Product description: {product_description}\n"
        f"Format: {format_value}\n"
        "Scope reminder: composition, headline rendering, and 50-word text are out of scope for this stage."
    )
