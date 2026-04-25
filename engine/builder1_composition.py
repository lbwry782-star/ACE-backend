"""
Builder1 composition planning via o3-pro (layout metadata only).
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from openai import OpenAI

_ALLOWED_LAYOUTS = {
    "headline_below_visual",
    "headline_left_visual_right",
    "visual_left_headline_right",
}


def _parse_json_object(raw: str) -> dict[str, Any]:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
    t = t.strip()
    if t.lower().startswith("```json"):
        t = t[7:].lstrip()
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model_output_not_object")
    return obj


def _validate_composition(
    data: dict[str, Any],
    *,
    format_value: str,
    detected_language: str,
    headline_product_name: str,
    headline_text: str,
) -> dict[str, Any]:
    layout = (data.get("compositionLayout") or "").strip()
    align = (data.get("headlineAlign") or "").strip()
    lines = data.get("headlineLines")
    rel_size = (data.get("headlineRelativeSize") or "").strip()
    visual_weight = (data.get("visualWeight") or "").strip()
    headline_weight = (data.get("headlineWeight") or "").strip()
    safe_margin = (data.get("safeMarginRule") or "").strip()
    notes = (data.get("compositionNotes") or "").strip()

    if layout not in _ALLOWED_LAYOUTS:
        raise ValueError("invalid_layout")
    if align != "center":
        raise ValueError("invalid_headline_align")
    if not isinstance(lines, dict):
        raise ValueError("invalid_headline_lines")
    line1 = (lines.get("line1") or "").strip()
    line2 = (lines.get("line2") or "").strip()
    if line1 != (headline_product_name or "").strip():
        raise ValueError("headline_line1_mismatch")
    if line2 != (headline_text or "").strip():
        raise ValueError("headline_line2_mismatch")
    if rel_size != "max_allowed_but_not_larger_than_visual":
        raise ValueError("invalid_headline_relative_size")
    if visual_weight not in {"dominant", "equal"}:
        raise ValueError("invalid_visual_weight")
    if headline_weight not in {"secondary", "equal"}:
        raise ValueError("invalid_headline_weight")
    if safe_margin != "minimum_1cm":
        raise ValueError("invalid_safe_margin_rule")
    if not notes:
        raise ValueError("missing_composition_notes")

    fmt = (format_value or "").strip().lower()
    lang = (detected_language or "").strip().lower()
    if fmt == "portrait" and layout != "headline_below_visual":
        raise ValueError("portrait_layout_invalid")
    if fmt == "landscape":
        expected = "headline_left_visual_right" if lang == "he" else "visual_left_headline_right"
        if layout != expected:
            raise ValueError("landscape_layout_invalid")
    if fmt == "square":
        if layout == "headline_left_visual_right" and lang != "he":
            raise ValueError("square_side_language_invalid")
        if layout == "visual_left_headline_right" and lang != "en":
            raise ValueError("square_side_language_invalid")

    if layout == "headline_below_visual":
        if not (visual_weight == "dominant" and headline_weight == "secondary"):
            raise ValueError("below_weight_invalid")
    else:
        if not (visual_weight == "equal" and headline_weight == "equal"):
            raise ValueError("side_weight_invalid")

    return {
        "compositionLayout": layout,
        "headlineAlign": align,
        "headlineLines": {"line1": line1, "line2": line2},
        "headlineRelativeSize": rel_size,
        "visualWeight": visual_weight,
        "headlineWeight": headline_weight,
        "safeMarginRule": safe_margin,
        "compositionNotes": notes,
    }


def generate_builder1_composition_o3(
    *,
    format: str,
    detectedLanguage: str,
    productNameResolved: str,
    headlineProductName: str,
    headlineText: str,
    headlineFull: str,
    objectA: str,
    objectASecondary: str,
    objectB: str,
    modeDecision: str,
    visualDescription: str,
    visualPrompt: str,
) -> dict[str, Any]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")

    system = (
        "Return exactly one JSON object with keys only:\n"
        '{'
        '"compositionLayout":"headline_below_visual|headline_left_visual_right|visual_left_headline_right",'
        '"headlineAlign":"center",'
        '"headlineLines":{"line1":"...","line2":"..."},'
        '"headlineRelativeSize":"max_allowed_but_not_larger_than_visual",'
        '"visualWeight":"dominant|equal",'
        '"headlineWeight":"secondary|equal",'
        '"safeMarginRule":"minimum_1cm",'
        '"compositionNotes":"..."'
        "}\n"
        "Rules:\n"
        "- Headline center-aligned only.\n"
        "- Headline must have at least two lines: line1 product name, line2 headline text.\n"
        "- Keep at least 1cm safe margin between elements and edges.\n"
        "- Use as much format area as possible.\n"
        "- Landscape: headline beside visual. Hebrew side=left, English side=right.\n"
        "- Portrait: headline below visual.\n"
        "- Square: decide by visual orientation; if beside, apply language side rule.\n"
        "- Headline never larger than visual and as large as possible within rule.\n"
        "- Below layout weights: visual dominant, headline secondary.\n"
        "- Side layout weights: visual equal, headline equal.\n"
        "- Strict JSON only, no extra keys."
    )
    user = (
        f"format: {format}\n"
        f"detectedLanguage: {detectedLanguage}\n"
        f"productNameResolved: {productNameResolved}\n"
        f"headlineProductName: {headlineProductName}\n"
        f"headlineText: {headlineText}\n"
        f"headlineFull: {headlineFull}\n"
        f"objectA: {objectA}\n"
        f"objectASecondary: {objectASecondary}\n"
        f"objectB: {objectB}\n"
        f"modeDecision: {modeDecision}\n"
        f"visualDescription: {visualDescription}\n"
        f"visualPrompt: {visualPrompt}\n"
    )
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    response = client.responses.create(
        model="o3-pro",
        input=f"{system}\n\n{user}",
        reasoning={"effort": "low"},
    )
    out_text = getattr(response, "output_text", None) or ""
    data = _parse_json_object(out_text)
    return _validate_composition(
        data,
        format_value=format,
        detected_language=detectedLanguage,
        headline_product_name=headlineProductName,
        headline_text=headlineText,
    )
