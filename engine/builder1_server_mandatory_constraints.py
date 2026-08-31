"""
Builder1 server-owned mandatory user constraints.

Hard binding user instructions are extracted deterministically and preserved
separately from model-selected selectedCreativeBrief.mandatoryConstraints.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from engine.builder1_no_logo import brand_guidelines_for_prompt
from engine.builder1_plan_parser import _norm_text
from engine.builder1_product_visibility import (
    explicit_product_visibility_forbidden,
    explicit_product_visibility_requested,
)
from engine.builder1_selected_creative_brief import (
    SELECTED_CREATIVE_BRIEF_MAX_ITEM_CHARS,
    SELECTED_CREATIVE_BRIEF_MAX_ITEMS,
    SelectedCreativeBrief,
)

INSTRUCTION_OWNED_GUIDELINE_KEYS = ("instructions", "userInstructions")
MIXED_GUIDELINE_KEYS = ("creativeBrief", "brief", "notes")

_BULLET_SPLIT_RE = re.compile(r"[\n\r]+|(?:^|\n)\s*[-*•]\s+")

_EN_FACTUAL_NEGATIVE_RE = re.compile(
    r"(?i)\b(?:we|product|it|they|this|the\s+product)\b.{0,12}\b(?:"
    r"don't|do\s+not|does\s+not|doesn't|did\s+not|cannot|can't"
    r")\b.{0,20}\b(?:manufacture|make|produce|sell|offer|contain|include|use)\b",
)
_HE_FACTUAL_NEGATIVE_RE = re.compile(
    r"(?:^|[\s,.:;])"
    r"(?:המוצר|המוצרים|השירות)?\s*"
    r"אינ[oaו]\s+(?:כולל|מכיל|משתמש)",
)

_EN_DIRECTIVE_START_RE = re.compile(
    r"(?i)^(?:"
    r"(?:do\s+not|don't|never|must\s+not|cannot|can't|avoid|forbidden|prohibited)\s+\w"
    r"|(?:must|shall|required\s+to|need\s+to|have\s+to)\s+(?:show|display|include|appear|use|depict|feature|contain)\b"
    r"|(?:price|pricing)\s+must\s+(?:appear|be\s+(?:shown|visible|displayed))\b"
    r"|must\s+(?:display|show|include)\s+(?:the\s+)?price\b"
    r")",
)
_HE_DIRECTIVE_RE = re.compile(
    r"(?:^|[\s,.:;])"
    r"(?:"
    r"אסור\b"
    r"|(?:אל|לא)\s+(?:להציג|תציג|לכלול|כלול|להשתמש|תשתמש)"
    r"|(?:חובה|חייב(?:ים)?)\s+(?:להציג|להופיע|לכלול|להשתמש)"
    r"|(?:המחיר|מחיר)\s+(?:חייב|חובה)\s+(?:להופיע|להציג)"
    r"|(?:חייב|חובה)\s+(?:להופיע|להציג)\s+(?:המחיר|מחיר)"
    r")",
)


def _dedupe_key(text: str) -> str:
    return _norm_text(text).casefold()


def _bounded_items(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for raw in items:
        text = _norm_text(raw)
        if not text or len(text) > SELECTED_CREATIVE_BRIEF_MAX_ITEM_CHARS:
            continue
        key = _dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= SELECTED_CREATIVE_BRIEF_MAX_ITEMS:
            break
    return result


def validate_server_mandatory_constraints(raw: object) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    parsed: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = _norm_text(item)
        if text:
            parsed.append(text)
    return _bounded_items(parsed)


def _split_instruction_segments(text: str) -> List[str]:
    segments: List[str] = []
    for chunk in _BULLET_SPLIT_RE.split(text.strip()):
        piece = _norm_text(chunk)
        if not piece:
            continue
        for part in re.split(r"(?<=[.!?])\s+", piece):
            normalized = _norm_text(part)
            if normalized:
                segments.append(normalized)
    return segments


def _is_factual_negative_statement(text: str) -> bool:
    return bool(_EN_FACTUAL_NEGATIVE_RE.search(text) or _HE_FACTUAL_NEGATIVE_RE.search(text))


def _is_explicit_directive_sentence(text: str) -> bool:
    stripped = _norm_text(text)
    if not stripped or _is_factual_negative_statement(stripped):
        return False
    if _EN_DIRECTIVE_START_RE.match(stripped):
        return True
    return bool(_HE_DIRECTIVE_RE.search(stripped))


def _is_product_visibility_owned(*, text: str, product_name: str) -> bool:
    return explicit_product_visibility_forbidden(
        product_name=product_name,
        product_description=text,
    ) or explicit_product_visibility_requested(
        product_name=product_name,
        product_description=text,
    )


def _append_constraint(
    collected: List[str],
    text: str,
    *,
    product_name: str,
) -> None:
    normalized = _norm_text(text)
    if not normalized:
        return
    if _is_product_visibility_owned(text=normalized, product_name=product_name):
        return
    collected.append(normalized)


def _extract_from_instruction_owned_field(text: str, *, product_name: str) -> List[str]:
    results: List[str] = []
    for segment in _split_instruction_segments(text):
        _append_constraint(results, segment, product_name=product_name)
    return results


def _extract_directives_from_mixed_text(text: str, *, product_name: str) -> List[str]:
    results: List[str] = []
    for segment in _split_instruction_segments(text):
        if _is_explicit_directive_sentence(segment):
            _append_constraint(results, segment, product_name=product_name)
    return results


def extract_builder1_server_mandatory_constraints(
    *,
    product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
    product_name: str = "",
) -> List[str]:
    """Deterministic extraction of server-owned binding user instructions."""
    collected: List[str] = []
    safe_guidelines = brand_guidelines_for_prompt(brand_guidelines) or {}

    for key in INSTRUCTION_OWNED_GUIDELINE_KEYS:
        value = safe_guidelines.get(key)
        if isinstance(value, str) and value.strip():
            collected.extend(
                _extract_from_instruction_owned_field(value, product_name=product_name)
            )

    for key in MIXED_GUIDELINE_KEYS:
        value = safe_guidelines.get(key)
        if isinstance(value, str) and value.strip():
            collected.extend(
                _extract_directives_from_mixed_text(value, product_name=product_name)
            )

    if product_description.strip():
        collected.extend(
            _extract_directives_from_mixed_text(product_description, product_name=product_name)
        )

    return _bounded_items(collected)


def merge_effective_mandatory_constraints(
    server_constraints: Sequence[str],
    model_constraints: Sequence[str],
) -> List[str]:
    """Server-owned constraints first; model may add but not remove server items."""
    return _bounded_items([*server_constraints, *model_constraints])


def format_server_mandatory_constraints_block(constraints: Sequence[str]) -> str:
    if not constraints:
        return ""
    lines = [
        "Server mandatory user constraints (binding — strategy must remain compatible):",
    ]
    for item in constraints:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_effective_mandatory_constraints_block(constraints: Sequence[str]) -> str:
    if not constraints:
        return ""
    lines = ["Mandatory user constraints (binding — must obey in all creative output):"]
    for item in constraints:
        lines.append(f"- {item}")
    return "\n".join(lines)


def server_mandatory_constraints_from_planning_internals(internals: object) -> List[str]:
    if not isinstance(internals, dict):
        return []
    return validate_server_mandatory_constraints(internals.get("serverMandatoryConstraints"))


def server_mandatory_constraints_from_plan(plan: object) -> List[str]:
    if plan is None:
        return []
    internals = getattr(plan, "planning_internals", None)
    if isinstance(plan, Mapping):
        internals = plan.get("planningInternals") or plan.get("planning_internals") or internals
    return server_mandatory_constraints_from_planning_internals(internals)


def resolve_server_mandatory_constraints_for_plan(
    plan: object,
    *,
    product_description: str = "",
    brand_guidelines: Optional[Dict[str, Any]] = None,
    product_name: str = "",
) -> List[str]:
    persisted = server_mandatory_constraints_from_plan(plan)
    if persisted:
        return persisted
    if product_description or brand_guidelines:
        return extract_builder1_server_mandatory_constraints(
            product_description=product_description,
            brand_guidelines=brand_guidelines,
            product_name=product_name,
        )
    return []


def effective_mandatory_constraints_for_brief(
    brief: Optional[SelectedCreativeBrief],
    server_constraints: Sequence[str],
) -> List[str]:
    model_constraints = brief.mandatory_constraints if brief is not None else []
    return merge_effective_mandatory_constraints(server_constraints, model_constraints)


def apply_effective_mandatory_to_brief(
    brief: SelectedCreativeBrief,
    server_constraints: Sequence[str],
) -> SelectedCreativeBrief:
    """Return brief copy with mandatoryConstraints set to the effective merged set."""
    return replace(
        brief,
        mandatory_constraints=effective_mandatory_constraints_for_brief(brief, server_constraints),
    )


def effective_creative_brief_for_prompts(
    brief: Optional[SelectedCreativeBrief],
    server_constraints: Sequence[str],
) -> Optional[SelectedCreativeBrief]:
    if brief is None:
        if not server_constraints:
            return None
        return SelectedCreativeBrief(mandatory_constraints=list(server_constraints))
    return apply_effective_mandatory_to_brief(brief, server_constraints)
