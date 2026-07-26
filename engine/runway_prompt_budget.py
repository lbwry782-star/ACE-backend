"""
Runway promptText budget — UTF-16 code units, deterministic server-owned trimming.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from engine.video_planning import RUNWAY_PHYSICS_REALISM_CONSTRAINT

DEFAULT_RUNWAY_PROMPT_MAX_UTF16 = 1000

CONCISE_VISUAL_POLICY = (
    "No readable text, logos, labels, captions, signs, packaging copy, watermarks, or brand names."
)

_VISUAL_POLICY_LONG_PREFIX = (
    "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
    "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
)
_VISUAL_POLICY_COMPACT = (
    "VISUAL POLICY: No text, logos, captions, or labels in-frame; pictorial motion only. "
)
_VISUAL_POLICY_CONCISE_PREFIX = f"VISUAL POLICY: {CONCISE_VISUAL_POLICY} "
_PHYSICS_SHORT = "REALISM: weight, contact, resistance; no frictionless sliding or gliding."
_SECONDARY_RUNWAY_STYLE_TAIL = re.compile(
    r"\s*No logos, no packaging typography, no on-screen words\.\s*Single clean commercial look\.?",
    re.IGNORECASE,
)


class RunwayPromptBudgetError(ValueError):
    """Runway prompt exceeds irreducible budget."""


@dataclass(frozen=True)
class RunwayPromptBudgetResult:
    promptText: str
    utf16Length: int
    maximumUtf16Units: int
    trimApplied: bool
    utf16LengthBefore: int
    visualPolicyPresent: bool
    physicsSuffixPresent: bool

    def to_safe_metadata(self) -> Dict[str, Any]:
        return {
            "runwayPromptUtf16Length": self.utf16Length,
            "runwayPromptMaximumUtf16Length": self.maximumUtf16Units,
            "runwayPromptAccepted": 1 <= self.utf16Length <= self.maximumUtf16Units,
            "runwayPromptTrimApplied": self.trimApplied,
            "runwayPromptUtf16LengthBefore": self.utf16LengthBefore,
            "runwayPromptVisualPolicyPresent": self.visualPolicyPresent,
            "runwayPromptPhysicsSuffixPresent": self.physicsSuffixPresent,
        }


def count_utf16_code_units(text: str) -> int:
    total = 0
    for char in text or "":
        code_point = ord(char)
        total += 2 if code_point > 0xFFFF else 1
    return total


def _join_prompt(main: str, physics: str) -> str:
    m = (main or "").strip()
    p = (physics or "").strip()
    if not m:
        return p
    if not p:
        return m
    return f"{m} {p}".strip()


def _split_physics_suffix(prompt: str, physics_suffix: str) -> tuple[str, str]:
    full = (physics_suffix or RUNWAY_PHYSICS_REALISM_CONSTRAINT).strip()
    short = _PHYSICS_SHORT.strip()
    rs = (prompt or "").rstrip()
    for candidate in (full, short):
        if candidate and rs.endswith(candidate):
            return rs[: len(rs) - len(candidate)].rstrip(), candidate
    return rs, ""


def normalize_runway_prompt_to_budget(
    *,
    core_prompt: str,
    visual_policy: str = CONCISE_VISUAL_POLICY,
    physics_suffix: str = RUNWAY_PHYSICS_REALISM_CONSTRAINT,
    maximum_utf16_units: int = DEFAULT_RUNWAY_PROMPT_MAX_UTF16,
) -> RunwayPromptBudgetResult:
    core = (core_prompt or "").strip()
    if count_utf16_code_units(core) > maximum_utf16_units:
        raise RunwayPromptBudgetError("builder2_runway_prompt_too_long")
    policy_prefix = (visual_policy or CONCISE_VISUAL_POLICY).strip()
    if policy_prefix and not policy_prefix.endswith("."):
        policy_prefix = f"{policy_prefix}."
    composed = core
    if policy_prefix and policy_prefix.lower() not in composed.lower():
        composed = f"VISUAL POLICY: {policy_prefix} {core}".strip()
    composed = re.sub(r"\s+", " ", composed).strip()
    physics_full = (physics_suffix or RUNWAY_PHYSICS_REALISM_CONSTRAINT).strip()
    composed = _join_prompt(composed, physics_full)
    before = count_utf16_code_units(composed)
    if before <= maximum_utf16_units:
        return RunwayPromptBudgetResult(
            promptText=composed,
            utf16Length=before,
            maximumUtf16Units=maximum_utf16_units,
            trimApplied=False,
            utf16LengthBefore=before,
            visualPolicyPresent=True,
            physicsSuffixPresent=bool(physics_full),
        )

    main, physics = _split_physics_suffix(composed, physics_full)
    trim_applied = True

    def _total() -> int:
        return count_utf16_code_units(_join_prompt(main, physics))

    guard = 0
    while _total() > maximum_utf16_units and guard < 48:
        guard += 1
        progressed = False
        if physics == physics_full and count_utf16_code_units(physics_full) > count_utf16_code_units(_PHYSICS_SHORT):
            physics = _PHYSICS_SHORT
            progressed = True
        if not progressed:
            m2 = _SECONDARY_RUNWAY_STYLE_TAIL.sub("", main).strip()
            m2 = re.sub(r"\s+", " ", m2)
            if m2 != main:
                main = m2
                progressed = True
        if not progressed and _VISUAL_POLICY_LONG_PREFIX in main:
            main = main.replace(_VISUAL_POLICY_LONG_PREFIX, _VISUAL_POLICY_COMPACT, 1)
            main = re.sub(r"\s+", " ", main).strip()
            progressed = True
        if not progressed and _VISUAL_POLICY_COMPACT in main:
            main = main.replace(_VISUAL_POLICY_COMPACT, _VISUAL_POLICY_CONCISE_PREFIX, 1)
            main = re.sub(r"\s+", " ", main).strip()
            progressed = True
        if not progressed:
            markers = ("Physical interaction (follow exactly):", "Physical interaction:")
            idx = -1
            for marker in markers:
                pos = main.find(marker)
                if pos >= 0:
                    idx = pos
                    break
            if idx > 0:
                prefix, tail = main[:idx], main[idx:]
                physics_units = count_utf16_code_units(physics)
                budget = maximum_utf16_units - (physics_units + (1 if physics else 0))
                tail_units = count_utf16_code_units(tail)
                if tail_units >= budget:
                    trimmed_tail = tail
                    while trimmed_tail and count_utf16_code_units(trimmed_tail) > budget:
                        trimmed_tail = trimmed_tail[1:]
                    main = trimmed_tail.strip()
                else:
                    room = budget - tail_units
                    prefix_keep = prefix
                    while prefix_keep and count_utf16_code_units(prefix_keep) > room:
                        prefix_keep = prefix_keep[1:]
                    main = f"{prefix_keep}{tail}".strip()
                progressed = True
        if not progressed:
            joined = _join_prompt(main, physics)
            while joined and count_utf16_code_units(joined) > maximum_utf16_units:
                joined = joined[1:]
            main = joined
            physics = ""
            progressed = True
        if _total() <= maximum_utf16_units:
            break
        if not progressed:
            break

    final_prompt = _join_prompt(main, physics)
    final_units = count_utf16_code_units(final_prompt)
    if count_utf16_code_units(main) > maximum_utf16_units:
        raise RunwayPromptBudgetError("builder2_runway_prompt_too_long")
    if final_units > maximum_utf16_units:
        raise RunwayPromptBudgetError("builder2_runway_prompt_too_long")
    if final_units <= 0:
        raise RunwayPromptBudgetError("builder2_runway_prompt_too_long")
    return RunwayPromptBudgetResult(
        promptText=final_prompt,
        utf16Length=final_units,
        maximumUtf16Units=maximum_utf16_units,
        trimApplied=trim_applied or (final_units != before),
        utf16LengthBefore=before,
        visualPolicyPresent=True,
        physicsSuffixPresent=bool(physics_full) and bool(physics or _PHYSICS_SHORT in final_prompt or physics_full in final_prompt),
    )


def prepare_builder2_runway_prompt_text(
    plan: Dict[str, Any],
    *,
    sanitized_core: str | None = None,
    maximum_utf16_units: int = DEFAULT_RUNWAY_PROMPT_MAX_UTF16,
) -> RunwayPromptBudgetResult:
    from engine.video_planning import build_runway_prompt_from_plan, sanitize_runway_prompt_for_video_text_policy

    core = sanitized_core if sanitized_core is not None else build_runway_prompt_from_plan(plan)
    sanitized, _ = sanitize_runway_prompt_for_video_text_policy(core)
    return normalize_runway_prompt_to_budget(
        core_prompt=sanitized,
        maximum_utf16_units=maximum_utf16_units,
    )
