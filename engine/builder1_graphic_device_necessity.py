"""
Builder1 recurring graphic device necessity — optional devices and redundant-overlay guard.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

NO_RECURRING_GRAPHIC_DEVICE = "NO_RECURRING_GRAPHIC_DEVICE"

REDUNDANT_EXPLANATORY_GRAPHIC_DEVICE = "redundant_explanatory_graphic_device"

_ABSENT_DEVICE_VALUES = frozenset(
    {
        "",
        "none",
        "null",
        NO_RECURRING_GRAPHIC_DEVICE.lower(),
        NO_RECURRING_GRAPHIC_DEVICE,
    }
)

_EXPLANATORY_OVERLAY_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bbounding box(?:es)?\b",
        r"\bannotation box(?:es)?\b",
        r"\bhighlight box(?:es)?\b",
        r"\bselection(?:-style)? outline(?:s)?\b",
        r"\bcallout(?: line| frame| box)?\b",
        r"\b(?:rectangle|rectangular|frame|outline|circle|bracket|box)\s+around\b",
        r"\barrow(?:s)?\s+(?:pointing to|to)\b",
        r"\bmark(?:ing)?\s+the\s+(?:first|second|third)\s+stage\b",
        r"\bhighlight(?:ing)?\s+(?:around|the)\b",
        r"מסגרת\s+סביב",
        r"מסגרת\s+מקיפ",
        r"תחימ(?:ה|ות)\s+מלבנ",
        r"מלבן\s+סביב",
        r"קו\s+מסביב",
        r"סימון\s+סביב",
        r"הדגש(?:ה|ות)\s+סביב",
        r"תיב(?:ה|ות)\s+סביב",
        r"חץ\s+אל",
        r"מסמנ(?:ות|ים)?\s+א(?:ת|ת)\s+השלב",
        r"מסגר(?:ת|ות)\s+נחושת\s+סביב",
        r"בדיוק\s+שתי\s+מסגר",
        r"תחימ(?:ות|ה)\s+ש(?:מ)?סמנ",
        r"שתי\s+תחימ",
        r"שתי\s+מסגר",
        r"מסגרות\s+נחושת",
    )
)

_LEGITIMATE_CONCEPTUAL_DEVICE_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bplaying card\b",
        r"\bthe medium becomes\b",
        r"\bmedium-as-object\b",
        r"\bmedia-as-object\b",
        r"\bad format becomes\b",
        r"\bposter becomes\b",
        r"\bbillboard becomes\b",
        r"קלף\s+(?:משחק|פרסום)",
        r"המ(?:דיום|edium)\s+הופך",
        r"ה(?:מודעה|פרסומת)\s+ה(?:יא|ופכת)\s+(?:ל)?",
        r"ה(?:מדיום|medium)\s+עצמו",
    )
)

_SELF_EVIDENT_MECHANISM_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgutter\b",
        r"\brain(?:fall| barrel|water| spout)?\b",
        r"\bdownpipe\b",
        r"\bbarrel\b",
        r"\bflow(?:s|ing)?\b",
        r"\bconcentrat",
        r"\bcollect(?:ion|ing)?\b",
        r"מרזב",
        r"חבית",
        r"גשם",
        r"צינור",
        r"מעבר(?:ים)?\s+של\s+(?:מים|גשם)",
        r"איסוף",
        r"ריכוז",
    )
)

_CAMPAIGN_BORDER_DEVICE_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcampaign border\b",
        r"\bad border\b",
        r"\bouter frame\b",
        r"\bseries frame\b",
        r"מסגר(?:ת|ות)\s+קמפיין",
        r"מסגר(?:ת|ות)\s+חיצונ",
        r"גבול\s+המודעה",
    )
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_recurring_graphic_device_value(value: object) -> str:
    text = _norm(value)
    if text.lower() in _ABSENT_DEVICE_VALUES:
        return ""
    return text


def recurring_graphic_device_is_absent(device: object, rule: object) -> bool:
    return not normalize_recurring_graphic_device_value(device) and not normalize_recurring_graphic_device_value(rule)


def has_recurring_graphic_device(device: object, rule: object) -> bool:
    return not recurring_graphic_device_is_absent(device, rule)


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _mechanism_context(*parts: object) -> str:
    return " ".join(_norm(part) for part in parts if _norm(part))


def _physical_mechanism_is_self_evident(
    *,
    physical_generator: str,
    transferred_object: str,
    transferred_object_action: str = "",
    conceptual_generator: str = "",
) -> bool:
    context = _mechanism_context(
        physical_generator,
        transferred_object,
        transferred_object_action,
        conceptual_generator,
    )
    if not context:
        return False
    hits = sum(1 for pattern in _SELF_EVIDENT_MECHANISM_PATTERNS if pattern.search(context))
    return hits >= 2 or (
        hits >= 1
        and any(token in context.lower() for token in ("flow", "stage", "collect", "concentr", "איסוף", "ריכוז", "שלב"))
    )


def _is_legitimate_conceptual_graphic_device(device: str, device_rule: str) -> bool:
    combined = f"{device} {device_rule}".strip()
    if not combined:
        return False
    if _matches_any(combined, _LEGITIMATE_CONCEPTUAL_DEVICE_PATTERNS):
        return True
    if _matches_any(combined, _EXPLANATORY_OVERLAY_PATTERNS):
        return False
    return False


def _is_intentional_campaign_border_device(device: str, device_rule: str, *, border_treatment: str) -> bool:
    border = _norm(border_treatment).lower()
    if border in {"", "none"}:
        return False
    combined = f"{device} {device_rule}".strip()
    return bool(combined) and _matches_any(combined, _CAMPAIGN_BORDER_DEVICE_PATTERNS)


def device_text_is_explanatory_overlay(device: str, device_rule: str) -> bool:
    combined = f"{device} {device_rule}".strip()
    if not combined:
        return False
    return _matches_any(combined, _EXPLANATORY_OVERLAY_PATTERNS)


def evaluate_redundant_explanatory_graphic_device(
    *,
    device: object,
    device_rule: object,
    physical_generator: str = "",
    transferred_object: str = "",
    transferred_object_action: str = "",
    conceptual_generator: str = "",
    border_treatment: str = "",
) -> Optional[str]:
    device_text = normalize_recurring_graphic_device_value(device)
    rule_text = normalize_recurring_graphic_device_value(device_rule)
    if not device_text and not rule_text:
        return None
    if _is_legitimate_conceptual_graphic_device(device_text, rule_text):
        return None
    if _is_intentional_campaign_border_device(device_text, rule_text, border_treatment=border_treatment):
        return None
    if not device_text_is_explanatory_overlay(device_text, rule_text):
        return None
    if not _physical_mechanism_is_self_evident(
        physical_generator=physical_generator,
        transferred_object=transferred_object,
        transferred_object_action=transferred_object_action,
        conceptual_generator=conceptual_generator,
    ):
        return None
    return REDUNDANT_EXPLANATORY_GRAPHIC_DEVICE


def scan_graphic_device_necessity(plan_dict: Dict[str, Any]) -> List[str]:
    graphic = plan_dict.get("graphicGenerator")
    if not isinstance(graphic, dict):
        return []
    violation = evaluate_redundant_explanatory_graphic_device(
        device=graphic.get("recurringGraphicDevice"),
        device_rule=graphic.get("recurringGraphicDeviceRule"),
        physical_generator=str(plan_dict.get("physicalGenerator") or ""),
        transferred_object=str(plan_dict.get("transferredObject") or ""),
        transferred_object_action=str(plan_dict.get("transferredObjectAction") or ""),
        conceptual_generator=str(plan_dict.get("conceptualGenerator") or ""),
        border_treatment=str(graphic.get("borderTreatment") or ""),
    )
    return [violation] if violation else []


def validate_recurring_graphic_device_pair(device: object, rule: object) -> Optional[str]:
    device_text = normalize_recurring_graphic_device_value(device)
    rule_text = normalize_recurring_graphic_device_value(rule)
    if bool(device_text) != bool(rule_text):
        return "graphic_generator_inconsistent_recurring_device"
    return None


def build_no_device_annotation_guard_block(*, border_treatment: str = "none") -> str:
    border = _norm(border_treatment).lower()
    border_clause = ""
    if border not in {"", "none"}:
        border_clause = (
            f" An approved campaign border ({border}) may appear when it belongs to the overall ad frame, "
            "not as object-highlighting overlays."
        )
    return (
        "Do not add bounding boxes, selection rectangles, callout frames, annotation outlines, arrows, labels, "
        "or object-highlighting overlays unless the approved advertising concept explicitly requires them."
        f"{border_clause}"
    )
