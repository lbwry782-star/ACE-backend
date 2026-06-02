"""
Disconnected Builder1 visual-prompt builder scaffold.
"""
from __future__ import annotations

import logging
import re

from engine.builder1_plan_spec import Builder1Plan, MODE_REPLACEMENT, MODE_SIDE_BY_SIDE

logger = logging.getLogger(__name__)

_IMAGE_PREAMBLE = (
    "Isolated objects on a seamless pure white background. "
    "Seamless pure white background, isolated objects, no environment."
)

_NEGATIVE_CONSTRAINTS = (
    "seamless pure white background",
    "isolated objects only",
    "no environment",
    "no extra objects",
    "no props",
    "no scenery",
    "no room",
    "no studio set",
    "no furniture",
    "no people",
    "no hands",
    "no floor",
    "no wall",
    "no table",
    "no desk",
    "no surface except seamless white",
    "no decorative elements",
    "no contextual objects",
    "no background objects",
    "no cast shadows that read as separate objects",
    "no text",
    "no letters",
    "no numbers",
    "no logos",
    "no brand marks",
    "no packaging text",
    "no signage text",
    "no watermark",
)

# Phrase/substring hints (multi-word or Hebrew).
_ENVIRONMENT_PHRASE_HINTS: tuple[str, ...] = (
    "hero shot",
    "product shot",
    "minimal styling",
    "wooden floor",
    "studio set",
    "no environment",
    "background object",
    "extra object",
    "חדר",
    "רקע",
    "שולחן",
    "ידיים",
    "רצפה",
    "קיר",
    "סטודיו",
    "סביבה",
    "משטח",
    "רהיט",
    "אביזר",
    "תפאורה",
    "נוף",
)

# Single-token hints matched with word boundaries (case-insensitive).
_ENVIRONMENT_WORD_HINTS: tuple[str, ...] = (
    "environment",
    "room",
    "studio",
    "table",
    "desk",
    "floor",
    "wall",
    "background",
    "scenery",
    "prop",
    "props",
    "furniture",
    "person",
    "people",
    "surface",
    "countertop",
    "shelf",
    "shelves",
    "kitchen",
    "office",
    "outdoor",
    "outdoors",
    "landscape",
    "interior",
    "exterior",
    "setting",
    "backdrop",
    "stage",
    "contextual",
    "decorative",
    "decoration",
    "decorations",
    "styling",
    "display",
    "pedestal",
    "platform",
    "marble",
    "concrete",
    "grass",
    "sky",
    "window",
    "curtain",
    "יד",
)

# Phrases stripped from planner visualDescription before image prompt (objects/pose kept).
_BACKGROUND_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bagainst\s+a\s+clean\s+[\w\s-]+\bbackground\b",
        r"\bwith\s+a\s+clean\s+[\w\s-]+\bbackground\b",
        r"\bclean\s+studio\s+backdrop\b",
        r"\bstudio\s+backdrop\b",
        r"\bclean\s+neutral\s+background\b",
        r"\bclean\s+[\w\s-]*\bbackground\b",
        r"\bneutral\s+background\b",
        r"\bcontext\s+and\s+depth\b",
        r"\bgently\s+cast\s+shadows?\b",
        r"\bcast\s+shadows?\b",
        r"\bsoft\s+key\s+light\b",
        r"\bkey\s+light\b",
        r",?\s*\band\s+depth\b",
        r",?\s*\bwith\s+depth\b",
        r"\bagainst\s+a\s*\.?",
    )
)


def _base_constraints_text() -> str:
    return ", ".join(_NEGATIVE_CONSTRAINTS)


def _split_visual_description_sentences(text: str) -> list[str]:
    chunks = re.split(r"[.\n;]+", text)
    return [c.strip() for c in chunks if c.strip()]


def _sentence_has_environment_hint(sentence: str) -> bool:
    lower = sentence.lower()
    for phrase in _ENVIRONMENT_PHRASE_HINTS:
        if phrase in lower or phrase in sentence:
            return True
    for word in _ENVIRONMENT_WORD_HINTS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return True
        if word in sentence and any("\u0590" <= c <= "\u05ea" for c in word):
            return True
    return False


def _strip_background_phrases(text: str) -> str:
    """Remove background/lighting/scenery phrases; keep object and pose wording."""
    t = (text or "").strip()
    if not t:
        return ""
    for pattern in _BACKGROUND_PHRASE_PATTERNS:
        t = pattern.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*,\s*,", ", ", t)
    t = re.sub(r",\s*\.", ".", t)
    t = re.sub(r"\band\s+add\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+\band\b\s*$", "", t, flags=re.IGNORECASE)
    return t.strip(" ,.;-")


def _sanitize_visual_description_for_image(visual_description: str) -> str:
    """
    Strip background/scenery phrases; drop sentences that are only environment language.
    Keep pose/overlap/placement/interaction cues.
    """
    raw = (visual_description or "").strip()
    if not raw:
        return ""

    kept: list[str] = []
    dropped: list[str] = []
    for sentence in _split_visual_description_sentences(raw):
        stripped = _strip_background_phrases(sentence)
        if not stripped or len(stripped.split()) < 3:
            dropped.append(sentence)
            continue
        if _sentence_has_environment_hint(stripped):
            dropped.append(sentence)
        else:
            kept.append(stripped)

    sanitized = ". ".join(kept).strip()
    if dropped:
        logger.info(
            "BUILDER1_VISUAL_DESCRIPTION_SANITIZED dropped_count=%s kept=%r dropped=%r",
            len(dropped),
            sanitized[:300],
            ". ".join(dropped)[:300],
        )
    return sanitized


def _supporting_direction_block(sanitized_description: str) -> str:
    if not sanitized_description:
        return (
            "Use only pose, overlap, relative placement, and interaction between the allowed objects. "
            "Ignore any environment, room, studio, surface, props, or extra objects."
        )
    return (
        "Allowed supporting direction (pose, overlap, relative placement, interaction only): "
        f"{sanitized_description}. "
        "Ignore any environment, room, studio, table, desk, floor, wall, background, scenery, props, or extra objects."
    )


def build_visual_prompt(plan: Builder1Plan) -> str:
    base = f"{_IMAGE_PREAMBLE} {_base_constraints_text()}."
    supporting = _sanitize_visual_description_for_image(plan.visual_description or "")

    if plan.mode_decision == MODE_SIDE_BY_SIDE:
        core = (
            f"Show only {plan.object_a} and {plan.object_b} partially overlapping on seamless pure white. "
            f"Do not show {plan.object_a_secondary}. "
            "Do not show any Object B secondary. "
            "Forbidden: every other object, prop, scenery, environment, surface, or background element. "
            "The image must contain only Object A and Object B."
        )
    elif plan.mode_decision == MODE_REPLACEMENT:
        logger.info(
            "BUILDER1_REPLACEMENT_VISUAL_RULES "
            "object_a=%r object_a_secondary=%r object_b=%r "
            "rule_object_a_absent=true rule_object_b_replaces_object_a=true rule_secondary_remains=true",
            plan.object_a,
            plan.object_a_secondary,
            plan.object_b,
        )
        core = (
            f"Do not show {plan.object_a}. "
            f"Show only {plan.object_b} and {plan.object_a_secondary} on seamless pure white. "
            f"Object B replaces Object A in spatial position only; Object A context means Object A secondary only, not environment. "
            f"Do not recreate background, room, studio, surface, scenery, or environment. "
            f"Keep {plan.object_a_secondary} visible and interacting with {plan.object_b} as if naturally paired. "
            "Do not show any Object B secondary. "
            "Forbidden: every other object, prop, scenery, environment, surface, or background element. "
            f"Only {plan.object_b} and {plan.object_a_secondary} may be visible."
        )
    else:
        raise ValueError("unsupported_mode")

    return f"{base} {core} {_supporting_direction_block(supporting)}."
