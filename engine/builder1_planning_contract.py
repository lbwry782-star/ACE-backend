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
- Choose Object A and Object B the way a painter perceives them: grasp overall form like a painter, not only technical contour or silhouette.
- Find Object B with strong morphological similarity to Object A.
- A painter should perceive Object A and Object B as nearly the same physical form before any conceptual meaning is considered.
- Among morphologically valid candidates, the advertising promise may help choose Object B — but advertising promise must NEVER increase visualSimilarityScore.

VISUAL SIMILARITY MUST BE BASED ON (only these factors may raise visualSimilarityScore):
- overall form
- mass distribution
- proportions
- volume
- dominant shapes
- physical structure
- physical occupation of space
- physical role in the scene
- whether Object B can physically replace Object A in the exact same position and interaction

DO NOT COUNT toward visualSimilarityScore (forbidden score inflation):
- shared function
- shared purpose
- shared category
- shared user interaction
- shared verb
- shared action
- shared name
- shared word root
- rhyme
- metaphor
- advertising promise
- headline potential
- conceptual similarity

Before assigning visualSimilarityScore above 90:
- Ignore all meaning, function, branding, language, and advertising context.
- Judge only each object as a silent white sculpture.
- If the score would fall below 90 under that sculpture-only test, REPLACEMENT is forbidden and modeDecision must be SIDE_BY_SIDE.

VALID HIGH REPLACEMENT-GRADE SIMILARITY EXAMPLES (only when physical replacement continuity is plausible):
- road cone ↔ carrot
- dart ↔ nail
- microphone ↔ ice cream cone

INVALID HIGH-SIMILARITY / FORBIDDEN 90+ EXAMPLES (similarity driven mainly by function, language, symbolism, or concept — must be SIDE_BY_SIDE or a new pair):
- computer keyboard ↔ piano keys
- computer mouse ↔ real mouse
- newspaper ↔ website
- camera ↔ eye

- Filter out text, logos, brands, generic environments, and unclear/non-physical situations.
- All objects must be classic, defined, physical objects.
- objectASecondary must be the classic physical companion/context object of objectA only (for example: can+straw, dog+bone, bee+flower, sign+pole).
- Object A must have a nearby secondary object that is its classic physical companion/context; objectASecondary is not part of objectA.
- Never choose objectASecondary as a product, promise, explanation, slogan, benefit, or abstract concept object.
- objectASecondary is intended for REPLACEMENT continuity and must stay usable when objectB replaces objectA.
- objectB must be physically distinct from objectA; do not use synonyms or near-identical variants (for example: megaphone vs bullhorn is forbidden).
- For REPLACEMENT planning, explicitly plan a final visual where objectA is absent, objectB replaces objectA in objectA's position/context, and objectASecondary remains visible interacting with objectB.
- In REPLACEMENT, objectB must keep objectA's background/context while appearing in objectA's original position.
- In REPLACEMENT, objectASecondary must remain visible and interact with objectB in a way that demonstrates objectB's nature, as if objectASecondary were naturally paired with objectB.

REPLACEMENT visualDescription (mandatory when modeDecision is REPLACEMENT):
- visualDescription must describe the final rendered image only — written as if Object A no longer exists.
- Describe only the final state: Object B, objectASecondary, and final pose, position, scale, lighting, and composition.
- A reader of visualDescription must be unable to tell that Object A was ever present.
- Object A must not be visually described as present in the scene.
- Object A may appear only in negative instructions (for example: "Do not show Object A") — never as a visible object.
- Forbidden: describing Object A positively, then describing Object B as a replacement.
- Forbidden phrases/patterns: "Object A is shown... then Object B replaces it..."; "In the replacement vision..."; "In the replacement version..."; "Instead of Object A..."; "Where Object A used to be..."; any narrative transition from A to B.
- Forbidden: "becomes", "turns into", "replaces the", "replaced by", "switches from", or any before/after story between A and B.
- Required: describe only what is visible in the final frame (Object B + objectASecondary + scene physics).
- Wrong example: "A hand holds a microphone. In the replacement version the microphone becomes an ice cream cone."
- Correct example: "A human hand naturally grips a vanilla ice cream cone against a clean neutral background. The grip, scale, position, lighting, and composition are realistic and physically natural."

- REPLACEMENT requires replacement-grade similarity, not just general silhouette similarity.
- Score 90+ only when objectB can literally occupy objectA's exact physical role, pose, position, and objectASecondary interaction without reconfiguring the scene.
- objectASecondary interaction must still make physical sense as originally paired with objectA.
- If objectB needs a different grip, support, usage posture, or a different interaction from objectASecondary, similarity is below 90 and modeDecision must be SIDE_BY_SIDE.
- Shared cone/bell shape alone is not enough for REPLACEMENT.
- Forbidden example: megaphone + hand -> trumpet must be SIDE_BY_SIDE, not REPLACEMENT, because hand/grip/usage changes and the viewer will not read it as replacement.
- Forbidden example: laptop computer + table -> open magazine must not be REPLACEMENT merely because both are open/flat/rectangular/book-like; this should be SIDE_BY_SIDE unless true replacement-grade continuity is proven.
- Forbidden example: smartphone + hand -> business card must not be REPLACEMENT merely because both can be held upright in the same hand.
- visualSimilarityScore below 70 is not allowed. If the best pair is below 70, choose a different Object A/Object B pair.
- Score bands: 90-100 => REPLACEMENT, 70-89 => SIDE_BY_SIDE, below 70 => invalid pair (must choose a new pair).
- SIDE_BY_SIDE reasoning flow: first choose Object A from product name + product description.
- For SIDE_BY_SIDE, grasp Object A's whole general form like a painter, not only technical contour/silhouette.
- Then search for Object B by morphological similarity to Object A (painter perception; sculpture-only form comparison).
- Among candidates that already pass morphological similarity, stop only when Object B also expresses the advertising promise.
- The advertising promise is not merely a justification for the move; the moment it is discovered is what makes the move possible — but it must not raise visualSimilarityScore.
- In SIDE_BY_SIDE, Object A and Object B must be shown with partial overlap, one over the other.
- SIDE_BY_SIDE is valid only for visualSimilarityScore 70-89.
- Below 70, choose a new pair.
- 90+ is reserved for true REPLACEMENT only.
- If unsure whether replacement-grade conditions are met, score below 90 and choose SIDE_BY_SIDE.
- detectedLanguage must be "he" or "en".
- modeDecision must be "REPLACEMENT" if visualSimilarityScore >= 90; otherwise "SIDE_BY_SIDE".
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
                "Physical form only (painter/sculpture perception). Conceptual, functional, linguistic, "
                "or advertising associations must NOT inflate this score. Use 90+ only if objectB can "
                "literally replace objectA in the same role/pose/position with objectASecondary interaction "
                "unchanged. Score bands: 90-100 REPLACEMENT, 70-89 SIDE_BY_SIDE, below 70 invalid."
            ),
        },
        "modeDecision": {
            "type": "string",
            "enum": ["REPLACEMENT", "SIDE_BY_SIDE"],
            "description": (
                "Must be REPLACEMENT only when visualSimilarityScore >= 90 under replacement-grade rules; "
                "if unsure or any reconfiguration is required, choose SIDE_BY_SIDE."
            ),
        },
        "visualDescription": {
            "type": "string",
            "description": (
                "For REPLACEMENT: final-frame description only — Object B + objectASecondary visible; "
                "never narrate A-to-B transition or describe Object A as visible. "
                "For SIDE_BY_SIDE: both Object A and Object B visible with overlap."
            ),
        },
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
