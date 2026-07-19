"""
Builder1 graphic-system contract — shared structured enums and descriptive fields.

Structured layout/placement fields use closed enum lists shared by prompt, schema,
and parser. Campaign-specific visual direction fields are concise non-empty strings.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Tuple

from engine.builder1_plan_spec import (
    BACKGROUND_TREATMENT_ENUMS,
    BORDER_TREATMENT_ENUMS,
    COPY_SAFE_SIDES,
    HEADLINE_ALIGNMENTS,
    HEADLINE_PLACEMENTS,
    IMAGE_STYLE_ENUMS,
    LAYOUT_TEMPLATES,
    TEXT_SCALE_ENUMS,
    TYPOGRAPHY_STYLE_ENUMS,
)

logger = logging.getLogger(__name__)

GRAPHIC_DESCRIPTIVE_FIELDS = frozenset(
    {
        "typographyStyle",
        "imageStyle",
        "backgroundTreatment",
        "recurringGraphicDevice",
        "recurringGraphicDeviceRule",
        "shapeLanguage",
        "framingRule",
        "spacingRule",
    }
)

GRAPHIC_DESCRIPTIVE_MIN_LENGTH = 8
GRAPHIC_DESCRIPTIVE_MAX_LENGTH = 240
GRAPHIC_VALUE_PREVIEW_MAX = 80

# Legacy enum sets retained for tests and repair guidance only — not used to reject
# descriptive campaign direction at parse time.
LEGACY_TYPOGRAPHY_STYLE_SUGGESTIONS = TYPOGRAPHY_STYLE_ENUMS
LEGACY_IMAGE_STYLE_SUGGESTIONS = IMAGE_STYLE_ENUMS
LEGACY_BACKGROUND_TREATMENT_SUGGESTIONS = BACKGROUND_TREATMENT_ENUMS

STRUCTURED_FIELD_ENUMS: Dict[str, frozenset[str]] = {
    "layoutTemplate": LAYOUT_TEMPLATES,
    "headlinePlacement": HEADLINE_PLACEMENTS,
    "headlineAlignment": HEADLINE_ALIGNMENTS,
    "brandBlockPlacement": HEADLINE_PLACEMENTS,
    "sloganPlacement": HEADLINE_PLACEMENTS,
    "copySafeArea.side": COPY_SAFE_SIDES,
    "headlineScale": TEXT_SCALE_ENUMS,
    "brandScale": TEXT_SCALE_ENUMS,
    "sloganScale": TEXT_SCALE_ENUMS,
    "borderTreatment": BORDER_TREATMENT_ENUMS,
}

REASON_TO_FIELD: Dict[str, str] = {
    "graphic_generator_invalid_layout": "layoutTemplate",
    "graphic_generator_invalid_headline_placement": "headlinePlacement",
    "graphic_generator_invalid_headline_alignment": "headlineAlignment",
    "graphic_generator_invalid_brand_placement": "brandBlockPlacement",
    "graphic_generator_invalid_slogan_placement": "sloganPlacement",
    "graphic_generator_invalid_copy_safe_side": "copySafeArea.side",
    "graphic_generator_invalid_headline_scale": "headlineScale",
    "graphic_generator_invalid_brand_scale": "brandScale",
    "graphic_generator_invalid_slogan_scale": "sloganScale",
    "graphic_generator_invalid_border": "borderTreatment",
    "graphic_generator_invalid_typography_style": "typographyStyle",
    "graphic_generator_invalid_image_style": "imageStyle",
    "graphic_generator_invalid_background": "backgroundTreatment",
    "graphic_generator_empty_typography_style": "typographyStyle",
    "graphic_generator_empty_image_style": "imageStyle",
    "graphic_generator_empty_background_treatment": "backgroundTreatment",
    "graphic_generator_typography_style_too_short": "typographyStyle",
    "graphic_generator_image_style_too_short": "imageStyle",
    "graphic_generator_background_treatment_too_short": "backgroundTreatment",
    "graphic_generator_missing_recurring_device": "recurringGraphicDevice",
    "graphic_generator_missing_device_rule": "recurringGraphicDeviceRule",
    "graphic_generator_missing_framing_rule": "framingRule",
    "graphic_generator_missing_shape_language": "shapeLanguage",
    "graphic_generator_missing_spacing_rule": "spacingRule",
}

GRAPHIC_CONTRACT_MISMATCH_REASONS = frozenset(
    {
        "graphic_generator_invalid_typography_style",
        "graphic_generator_invalid_image_style",
        "graphic_generator_invalid_background",
        "graphic_generator_empty_typography_style",
        "graphic_generator_empty_image_style",
        "graphic_generator_empty_background_treatment",
        "graphic_generator_typography_style_too_short",
        "graphic_generator_image_style_too_short",
        "graphic_generator_background_treatment_too_short",
    }
)


def _preview(value: object) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= GRAPHIC_VALUE_PREVIEW_MAX:
        return text
    return text[: GRAPHIC_VALUE_PREVIEW_MAX - 3] + "..."


def log_graphic_field_rejected(*, field: str, reason: str, value: object) -> None:
    logger.error(
        "BUILDER1_GRAPHIC_FIELD_REJECTED stage=graphic_system field=%s reason=%s valuePreview=%s",
        field,
        reason,
        _preview(value),
    )


def validate_descriptive_graphic_text(value: object, *, field: str) -> Optional[str]:
    text = " ".join(str(value or "").strip().split())
    if not text:
        code = {
            "typographyStyle": "graphic_generator_empty_typography_style",
            "imageStyle": "graphic_generator_empty_image_style",
            "backgroundTreatment": "graphic_generator_empty_background_treatment",
        }.get(field, f"graphic_generator_empty_{field}")
        log_graphic_field_rejected(field=field, reason=code, value=value)
        return code
    if len(text) < GRAPHIC_DESCRIPTIVE_MIN_LENGTH:
        code = {
            "typographyStyle": "graphic_generator_typography_style_too_short",
            "imageStyle": "graphic_generator_image_style_too_short",
            "backgroundTreatment": "graphic_generator_background_treatment_too_short",
        }.get(field, f"graphic_generator_{field}_too_short")
        log_graphic_field_rejected(field=field, reason=code, value=value)
        return code
    if len(text) > GRAPHIC_DESCRIPTIVE_MAX_LENGTH:
        log_graphic_field_rejected(field=field, reason=f"graphic_generator_{field}_too_long", value=value)
        return f"graphic_generator_{field}_too_long"
    return None


def validate_structured_enum(value: object, *, field: str, allowed: frozenset[str]) -> Optional[str]:
    normalized = " ".join(str(value or "").strip().split())
    if normalized not in allowed:
        reason_map = {
            "layoutTemplate": "graphic_generator_invalid_layout",
            "headlinePlacement": "graphic_generator_invalid_headline_placement",
            "headlineAlignment": "graphic_generator_invalid_headline_alignment",
            "brandBlockPlacement": "graphic_generator_invalid_brand_placement",
            "sloganPlacement": "graphic_generator_invalid_slogan_placement",
            "copySafeArea.side": "graphic_generator_invalid_copy_safe_side",
            "headlineScale": "graphic_generator_invalid_headline_scale",
            "brandScale": "graphic_generator_invalid_brand_scale",
            "sloganScale": "graphic_generator_invalid_slogan_scale",
            "borderTreatment": "graphic_generator_invalid_border",
        }
        reason = reason_map.get(field, f"graphic_generator_invalid_{field}")
        log_graphic_field_rejected(field=field, reason=reason, value=value)
        return reason
    return None


def structured_enum_prompt_lines() -> str:
    lines = [
        "Structured enum fields (exact values only):",
        f"- layoutTemplate: {', '.join(sorted(LAYOUT_TEMPLATES))}",
        f"- headlinePlacement, brandBlockPlacement, sloganPlacement: {', '.join(sorted(HEADLINE_PLACEMENTS))}",
        f"- headlineAlignment: {', '.join(sorted(HEADLINE_ALIGNMENTS))}",
        f"- copySafeArea.side: {', '.join(sorted(COPY_SAFE_SIDES))}",
        f"- headlineScale, brandScale, sloganScale: {', '.join(sorted(TEXT_SCALE_ENUMS))}",
        f"- borderTreatment: {', '.join(sorted(BORDER_TREATMENT_ENUMS))}",
    ]
    return "\n".join(lines)


def descriptive_field_prompt_lines() -> str:
    return "\n".join(
        [
            "Descriptive campaign-direction fields (concise non-empty prose, not closed enums):",
            "- typographyStyle: how type behaves in this campaign (weight, geometry, readability).",
            "- imageStyle: photography/illustration direction for the main visual.",
            "- backgroundTreatment: how the ad background should look and behave.",
            "- recurringGraphicDevice, recurringGraphicDeviceRule, shapeLanguage, framingRule, spacingRule: concrete repeatable visual rules.",
            f"Each descriptive field must be {GRAPHIC_DESCRIPTIVE_MIN_LENGTH}-{GRAPHIC_DESCRIPTIVE_MAX_LENGTH} characters.",
        ]
    )


def repair_instructions_for_reasons(reasons: List[str]) -> List[str]:
    instructions: List[str] = []
    seen: set[str] = set()
    for reason in reasons:
        field = REASON_TO_FIELD.get(reason)
        if not field or field in seen:
            continue
        seen.add(field)
        if field in {"typographyStyle", "imageStyle", "backgroundTreatment"} or field in GRAPHIC_DESCRIPTIVE_FIELDS:
            instructions.append(
                f"The previous {field} value was empty or structurally invalid. "
                f"Return one concise concrete campaign-specific sentence ({GRAPHIC_DESCRIPTIVE_MIN_LENGTH}+ characters). "
                f"Preserve every other already-valid graphic-system field unchanged."
            )
            continue
        allowed = STRUCTURED_FIELD_ENUMS.get(field)
        if allowed:
            instructions.append(
                f"The previous {field} value was not one of the permitted values. "
                f"Return exactly one of: {', '.join(sorted(allowed))}. "
                f"Preserve every other already-valid graphic-system field unchanged."
            )
    return instructions


def is_graphic_contract_mismatch(reasons: List[str]) -> bool:
    return any(code in GRAPHIC_CONTRACT_MISMATCH_REASONS for code in reasons)


def graphic_schema_enum_properties() -> Dict[str, Dict[str, object]]:
    return {
        "layoutTemplate": {"type": "string", "enum": sorted(LAYOUT_TEMPLATES)},
        "headlinePlacement": {"type": "string", "enum": sorted(HEADLINE_PLACEMENTS)},
        "headlineAlignment": {"type": "string", "enum": sorted(HEADLINE_ALIGNMENTS)},
        "brandBlockPlacement": {"type": "string", "enum": sorted(HEADLINE_PLACEMENTS)},
        "sloganPlacement": {"type": "string", "enum": sorted(HEADLINE_PLACEMENTS)},
        "headlineScale": {"type": "string", "enum": sorted(TEXT_SCALE_ENUMS)},
        "brandScale": {"type": "string", "enum": sorted(TEXT_SCALE_ENUMS)},
        "sloganScale": {"type": "string", "enum": sorted(TEXT_SCALE_ENUMS)},
        "borderTreatment": {"type": "string", "enum": sorted(BORDER_TREATMENT_ENUMS)},
    }


def graphic_schema_descriptive_properties() -> Dict[str, Dict[str, object]]:
    spec = {"type": "string", "minLength": GRAPHIC_DESCRIPTIVE_MIN_LENGTH, "maxLength": GRAPHIC_DESCRIPTIVE_MAX_LENGTH}
    return {field: dict(spec) for field in sorted(GRAPHIC_DESCRIPTIVE_FIELDS)}
