"""
Post-strategy brand visual identity filter for brand_physical model prompts.

The full sanitized brandGuidelines object may remain in server storage and other
stages. This module narrows what reaches the post-strategy brand_physical prompt
so raw brief / instruction blobs cannot bypass selectedCreativeBrief.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, FrozenSet, Optional

from engine.builder1_no_logo import brand_guidelines_for_prompt

_GUIDELINE_KEY_NORMALIZE_RE = re.compile(r"[^a-z0-9]")

# Keys confirmed in Builder1/API tests and contracts (visual identity only).
PHYSICAL_BRAND_IDENTITY_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        "primarycolor",
        "secondarycolor",
        "accentcolor",
        "backgroundcolor",
        "textcolor",
        "palette",
        "colors",
        "typography",
        "font",
        "fontfamily",
        "typestyle",
        "fonts",
        "tone",
        "visualtone",
        "visualstyle",
        "style",
        "layout",
        "compositionpreference",
    }
)

# Raw brief / instruction blobs — excluded regardless of content.
PHYSICAL_BRAND_RAW_BRIEF_DENYLIST: FrozenSet[str] = frozenset(
    {
        "creativebrief",
        "brief",
        "notes",
        "instructions",
        "userinstructions",
    }
)


def _normalize_guideline_key(key: str) -> str:
    return _GUIDELINE_KEY_NORMALIZE_RE.sub("", str(key).casefold())


def brand_identity_guidelines_for_physical_prompt(
    value: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return logo-stripped brand visual identity fields for brand_physical prompts."""
    sanitized = brand_guidelines_for_prompt(value)
    if sanitized is None:
        return None

    filtered: Dict[str, Any] = {}
    for key, raw in sanitized.items():
        norm = _normalize_guideline_key(key)
        if norm in PHYSICAL_BRAND_RAW_BRIEF_DENYLIST:
            continue
        if norm not in PHYSICAL_BRAND_IDENTITY_ALLOWLIST:
            continue
        filtered[str(key)] = copy.deepcopy(raw)

    return filtered or None


def format_brand_visual_identity_block(identity: Optional[Dict[str, Any]]) -> str:
    if not identity:
        return ""
    payload = json.dumps(identity, ensure_ascii=False, indent=2)
    return f"BRAND VISUAL IDENTITY:\n{payload}"
