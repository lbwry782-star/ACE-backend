"""
Builder1 selected creative brief — campaign-relative strategic fact selection.

Produced inside strategy_slogan_stage; persisted in planningInternals.
Full productDescription remains diagnostic-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from engine.builder1_plan_parser import _norm_text
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

SELECTED_CREATIVE_BRIEF_MAX_ITEMS = 12
SELECTED_CREATIVE_BRIEF_MAX_ITEM_CHARS = 500
SELECTED_CREATIVE_BRIEF_LIST_KEYS = (
    "essentialFacts",
    "supportingEvidence",
    "mandatoryConstraints",
)


@dataclass(frozen=True)
class SelectedCreativeBrief:
    essential_facts: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    mandatory_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "essentialFacts": list(self.essential_facts),
            "supportingEvidence": list(self.supporting_evidence),
            "mandatoryConstraints": list(self.mandatory_constraints),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SelectedCreativeBrief:
        return SelectedCreativeBrief(
            essential_facts=list(raw.get("essentialFacts") or []),
            supporting_evidence=list(raw.get("supportingEvidence") or []),
            mandatory_constraints=list(raw.get("mandatoryConstraints") or []),
        )


def default_selected_creative_brief_for_tests() -> SelectedCreativeBrief:
    return SelectedCreativeBrief(
        essential_facts=["Reinforced shell product designed for daily carry"],
        supporting_evidence=["Durable reinforced construction supports everyday protection"],
        mandatory_constraints=[],
    )


def selected_creative_brief_json_schema() -> Dict[str, Any]:
    item_schema = {"type": "string", "minLength": 1, "maxLength": SELECTED_CREATIVE_BRIEF_MAX_ITEM_CHARS}
    list_schema = {
        "type": "array",
        "items": item_schema,
        "maxItems": SELECTED_CREATIVE_BRIEF_MAX_ITEMS,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(SELECTED_CREATIVE_BRIEF_LIST_KEYS),
        "properties": {
            "essentialFacts": {**list_schema, "minItems": 1},
            "supportingEvidence": list_schema,
            "mandatoryConstraints": list_schema,
        },
    }


def _validate_brief_list(
    values: object,
    *,
    field_name: str,
    prefix: str,
    require_non_empty: bool = False,
) -> List[str]:
    reasons: List[str] = []
    if not isinstance(values, list):
        raise StageParseError("strategy_slogan_stage", [f"{prefix}:{field_name}_not_list"])
    if require_non_empty and not values:
        reasons.append(f"{prefix}:{field_name}_empty")
    if len(values) > SELECTED_CREATIVE_BRIEF_MAX_ITEMS:
        reasons.append(f"{prefix}:{field_name}_too_many_items")
    parsed: List[str] = []
    for idx, item in enumerate(values):
        if not isinstance(item, str):
            reasons.append(f"{prefix}:{field_name}_item_{idx}_not_string")
            continue
        text = _norm_text(item)
        if not text:
            reasons.append(f"{prefix}:{field_name}_item_{idx}_empty")
            continue
        if len(text) > SELECTED_CREATIVE_BRIEF_MAX_ITEM_CHARS:
            reasons.append(f"{prefix}:{field_name}_item_{idx}_too_long")
        parsed.append(text)
    if reasons:
        raise StageParseError("strategy_slogan_stage", reasons)
    return parsed


def parse_selected_creative_brief(
    raw_payload: object,
    *,
    required: bool = True,
) -> SelectedCreativeBrief:
    if raw_payload is None:
        if required:
            raise StageParseError("strategy_slogan_stage", ["strategy:missing_selectedCreativeBrief"])
        return SelectedCreativeBrief()
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("strategy_slogan_stage", ["strategy:selectedCreativeBrief_not_object"]) from exc

    prefix = "strategy:selectedCreativeBrief"
    essential = _validate_brief_list(
        obj.get("essentialFacts"),
        field_name="essentialFacts",
        prefix=prefix,
        require_non_empty=True,
    )
    supporting = _validate_brief_list(
        obj.get("supportingEvidence"),
        field_name="supportingEvidence",
        prefix=prefix,
    )
    mandatory = _validate_brief_list(
        obj.get("mandatoryConstraints"),
        field_name="mandatoryConstraints",
        prefix=prefix,
    )
    return SelectedCreativeBrief(
        essential_facts=essential,
        supporting_evidence=supporting,
        mandatory_constraints=mandatory,
    )


def selected_creative_brief_from_planning_internals(
    internals: object,
) -> Optional[SelectedCreativeBrief]:
    if not isinstance(internals, dict):
        return None
    raw = internals.get("selectedCreativeBrief")
    if not isinstance(raw, dict):
        return None
    try:
        return parse_selected_creative_brief(raw, required=True)
    except StageParseError:
        return None


def selected_creative_brief_from_plan(plan: object) -> Optional[SelectedCreativeBrief]:
    if plan is None:
        return None
    internals = getattr(plan, "planning_internals", None)
    if isinstance(plan, Mapping):
        internals = plan.get("planningInternals") or plan.get("planning_internals") or internals
    return selected_creative_brief_from_planning_internals(internals)


def format_selected_creative_brief_block(brief: SelectedCreativeBrief) -> str:
    lines = [
        "Selected creative brief (campaign-relative facts committed after strategy — not a product summary):",
    ]
    lines.append("Essential facts:")
    for fact in brief.essential_facts:
        lines.append(f"- {fact}")
    if brief.supporting_evidence:
        lines.append("Supporting evidence:")
        for fact in brief.supporting_evidence:
            lines.append(f"- {fact}")
    if brief.mandatory_constraints:
        lines.append("Mandatory constraints:")
        for fact in brief.mandatory_constraints:
            lines.append(f"- {fact}")
    return "\n".join(lines)


def format_product_identity_block(*, product_name_resolved: str, brief: SelectedCreativeBrief) -> str:
    identity_line = f"Product name (fixed): {product_name_resolved}"
    if brief.essential_facts:
        identity_line += f"\nProduct identity: {brief.essential_facts[0]}"
    return identity_line


def format_graphic_creative_brief_block(brief: Optional[SelectedCreativeBrief]) -> str:
    if brief is None or not brief.mandatory_constraints:
        return ""
    from engine.builder1_server_mandatory_constraints import format_effective_mandatory_constraints_block

    return format_effective_mandatory_constraints_block(brief.mandatory_constraints)


def format_brief_for_prompt_json(brief: SelectedCreativeBrief) -> str:
    return json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)
