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
- objectASecondary must be the classic physical companion/context object of objectA only (for example: can+straw, dog+bone, bee+flower, sign+pole).
- Never choose objectASecondary as a product, promise, explanation, slogan, benefit, or abstract concept object.
- objectB must be physically distinct from objectA; do not use synonyms or near-identical variants (for example: megaphone vs bullhorn is forbidden).
- For REPLACEMENT planning, explicitly plan a final visual where objectA is absent, objectB replaces objectA in objectA's position/context, and objectASecondary remains visible interacting with objectB.
- REPLACEMENT requires replacement-grade similarity, not just general silhouette similarity.
- Score 85+ only when objectB can literally occupy objectA's exact physical role, pose, position, and objectASecondary interaction without reconfiguring the scene.
- objectASecondary interaction must still make physical sense as originally paired with objectA.
- If objectB needs a different grip, support, usage posture, or a different interaction from objectASecondary, similarity is below 85 and modeDecision must be SIDE_BY_SIDE.
- Shared cone/bell shape alone is not enough for REPLACEMENT.
- Forbidden example: megaphone + hand -> trumpet must be SIDE_BY_SIDE, not REPLACEMENT, because hand/grip/usage changes and the viewer will not read it as replacement.
- Forbidden example: laptop computer + table -> open magazine must not be REPLACEMENT merely because both are open/flat/rectangular/book-like; this should be SIDE_BY_SIDE unless true replacement-grade continuity is proven.
- If unsure whether replacement-grade conditions are met, score below 85 and choose SIDE_BY_SIDE.
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
        "visualSimilarityScore": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Conservative replacement-grade score. Use 85+ only if objectB can literally replace objectA "
                "in the same role/pose/position with objectASecondary interaction unchanged; otherwise below 85."
            ),
        },
        "modeDecision": {
            "type": "string",
            "enum": ["REPLACEMENT", "SIDE_BY_SIDE"],
            "description": (
                "Must be REPLACEMENT only when visualSimilarityScore >= 85 under replacement-grade rules; "
                "if unsure or any reconfiguration is required, choose SIDE_BY_SIDE."
            ),
        },
        "visualDescription": {"type": "string"},
    },
}


def build_builder1_planning_user_prompt(
    product_name: str,
    product_description: str,
    format_value: str,
    remembered_object_a: list[str] | None = None,
) -> str:
    memory_lines = ""
    if remembered_object_a:
        memory_lines = (
            "Object A memory (avoid reusing or near-equivalent ideas):\n"
            f"- previous_object_a: {', '.join(remembered_object_a)}\n"
            "- Do not reuse any previous Object A from memory.\n"
            "- Avoid objects that are essentially the same as remembered Object A values.\n"
            "- Choose a fresh Object A.\n"
        )
    return (
        f"Product name: {product_name}\n"
        f"Product description: {product_description}\n"
        f"Format: {format_value}\n"
        f"{memory_lines}"
        "Scope reminder: composition, headline rendering, and 50-word text are out of scope for this stage."
    )
