"""
Builder2 gold prototype catalog — methodology references for tournament.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

from engine.builder2_tournament_config import (
    ALL_KNOWN_PROTOTYPE_IDS,
    DEFAULT_ACTIVE_PROTOTYPE_IDS,
    REFERENCE_ONLY_PROTOTYPE_IDS,
)


@dataclass(frozen=True)
class Builder2Prototype:
    prototype_id: str
    display_name: str
    original_problem: str
    reusable_method: str
    must_not_copy: str
    creator_guidance: str
    judge_quality_guidance: str
    active: bool


_PROTOTYPES: Dict[str, Builder2Prototype] = {
    "winning_card": Builder2Prototype(
        prototype_id="winning_card",
        display_name="Winning Card in Advertising",
        original_problem='There is no established connection between "אורי לב" and advertising.',
        reusable_method=(
            "Turn the medium, container or advertising format itself into part of the persuasive proof."
        ),
        must_not_copy="Do not copy playing cards literally.",
        creator_guidance=(
            "Use the advertising medium or container as visible proof of the advantage. "
            "The format itself should help express the relative advantage."
        ),
        judge_quality_guidance=(
            "Reward when the medium becomes persuasive evidence, not decorative metaphor."
        ),
        active=True,
    ),
    "summer_fan": Builder2Prototype(
        prototype_id="summer_fan",
        display_name="Summer Fan",
        original_problem="There is no perceived connection between refreshment and watermelon.",
        reusable_method=(
            "Create a missing association through a simple, recognizable physical behavior or motion parallel."
        ),
        must_not_copy="Do not copy a fan or watermelon literally.",
        creator_guidance=(
            "Find a physical behavior parallel that makes the missing association feel inevitable."
        ),
        judge_quality_guidance=(
            "Reward motion or behavior parallels that create association without literal copying."
        ),
        active=True,
    ),
    "greenpeace_essential_pairing": Builder2Prototype(
        prototype_id="greenpeace_essential_pairing",
        display_name="Greenpeace — Essential Pairing",
        original_problem="People do not believe in Greenpeace.",
        reusable_method=(
            "Pair two things whose relationship emerges from the deepest essence of each and is emotionally understood."
        ),
        must_not_copy=(
            "Do not reduce this to appearance similarity, ordinary replacement or wordplay."
        ),
        creator_guidance=(
            "Pair two essences so the relationship feels emotionally inevitable, not merely visual resemblance."
        ),
        judge_quality_guidance=(
            "Reject appearance-only pairing. Reward essential emotional pairing."
        ),
        active=True,
    ),
    "forgot": Builder2Prototype(
        prototype_id="forgot",
        display_name="Forgot",
        original_problem="People forget to turn on their lights.",
        reusable_method=(
            "Make an omission or neglected behavior visible through a direct, memorable consequence."
        ),
        must_not_copy="Do not merely remind the viewer verbally.",
        creator_guidance=(
            "Show the consequence of the neglected behavior visually and memorably."
        ),
        judge_quality_guidance=(
            "Reward visible consequence embodiment, not verbal reminder."
        ),
        active=True,
    ),
    "closest": Builder2Prototype(
        prototype_id="closest",
        display_name="Closest",
        original_problem='A traditional advertising agency is perceived as preferable to "אורי לב."',
        reusable_method=(
            "Accept a real competitive gap and convert closeness to the stronger alternative into the advantage."
        ),
        must_not_copy="Do not falsely erase the gap.",
        creator_guidance=(
            "Acknowledge the real gap honestly, then reframe proximity/closeness as the product advantage."
        ),
        judge_quality_guidance=(
            "Reward honest gap acceptance with persuasive reframing, not denial."
        ),
        active=True,
    ),
    "think_small": Builder2Prototype(
        prototype_id="think_small",
        display_name="Think Small",
        original_problem="The car is small.",
        reusable_method=(
            "Openly accept a real weakness, invert its meaning and transform it into the relative advantage."
        ),
        must_not_copy="Do not invent a weakness merely to use the prototype.",
        creator_guidance=(
            "Identify a real product weakness and invert it into the persuasive advantage."
        ),
        judge_quality_guidance=(
            "Reject invented weaknesses. Reward honest inversion of a real weakness."
        ),
        active=True,
    ),
    "shared_word_line_mechanism": Builder2Prototype(
        prototype_id="shared_word_line_mechanism",
        display_name="Shared Word/Line Mechanism",
        original_problem="A verbal/visual mechanism such as פס can unify disparate scenes.",
        reusable_method="Use one shared verbal or visual mechanism to connect scenes.",
        must_not_copy="Do not copy the exact historical line or graphic device.",
        creator_guidance="Reference only — not active by default.",
        judge_quality_guidance="Reference only.",
        active=False,
    ),
    "old_commercial_code_inversion": Builder2Prototype(
        prototype_id="old_commercial_code_inversion",
        display_name="Old Commercial Code Inversion",
        original_problem="Familiar commercial codes such as NEW can be inverted.",
        reusable_method="Invert a familiar commercial code to express the advantage.",
        must_not_copy="Do not copy the exact historical campaign code.",
        creator_guidance="Reference only — not active by default.",
        judge_quality_guidance="Reference only.",
        active=False,
    ),
    "context_collision": Builder2Prototype(
        prototype_id="context_collision",
        display_name="Context Collision / Extreme Visual Contrast",
        original_problem="Extreme visual contrast can create immediate meaning.",
        reusable_method="Use context collision or extreme visual contrast as a visual-parallel family.",
        must_not_copy="Do not shock without a meaningful bridge.",
        creator_guidance=(
            "Context collision remains available as visualParallelType even when not an active prototype."
        ),
        judge_quality_guidance="Reject shock without bridge. Reward meaningful collision.",
        active=False,
    ),
}


def get_prototype(prototype_id: str) -> Optional[Builder2Prototype]:
    return _PROTOTYPES.get((prototype_id or "").strip())


def require_prototype(prototype_id: str) -> Builder2Prototype:
    proto = get_prototype(prototype_id)
    if proto is None or prototype_id not in ALL_KNOWN_PROTOTYPE_IDS:
        raise KeyError(f"unknown_prototype:{prototype_id}")
    return proto


def active_prototype_ids() -> FrozenSet[str]:
    return frozenset(DEFAULT_ACTIVE_PROTOTYPE_IDS)


def reference_only_prototype_ids() -> FrozenSet[str]:
    return REFERENCE_ONLY_PROTOTYPE_IDS
