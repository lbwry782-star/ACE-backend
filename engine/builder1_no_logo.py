"""
Builder1 no-logo policy — product names as plain text only, never logos or brand symbols.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGO_REJECTION_CODES = frozenset(
    {
        "invented_product_logo",
        "supplied_logo_displayed",
        "logo_like_brand_symbol",
        "packaging_contains_brand_mark",
        "campaign_device_used_as_logo",
        "product_name_not_text_only",
    }
)

BUILDER1_NO_LOGO_PLANNING_RULE = (
    "NO PRODUCT LOGO: identify the product only by its written name as plain text. "
    "Never invent, reproduce, infer, request, or design a logo, symbol, icon, emblem, monogram, badge, seal, or brand mark. "
    "A recurring campaign graphic device may appear in the ad composition but must not become product identity or packaging branding."
)

BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK = "\n".join(
    [
        "=== NO PRODUCT LOGO (HIGH PRIORITY) ===",
        "NO PRODUCT LOGO.",
        "Do not invent, reproduce, infer, or display any logo, symbol, icon, emblem, monogram, badge, trademark, or brand mark.",
        "Display the product name only as plain readable text.",
        "If packaging is visible, print only the product name without any accompanying symbol.",
        "Do not convert decorative campaign elements into packaging logos or brand marks.",
        "Do not add registered-trademark or trademark symbols.",
        "Do not place the product name inside a badge, crest, seal, shield, medallion, or logo container.",
        "Typography may be attractive and campaign-consistent, but it must remain clearly text rather than logo design.",
        "=== END NO PRODUCT LOGO ===",
    ]
)

_LOGO_GUIDELINE_KEY_RE = re.compile(
    r"(^|_)(logo|logourl|logo_url|logoimage|logo_image|logobase64|logo_base64|brandlogo|brand_logo|"
    r"productlogo|product_logo|logoasset|logo_asset|logofile|logo_file|brandicon|brand_icon|"
    r"brandmark|brand_mark|logomark|logo_mark|logodescription|logo_description|logoprompt|logo_prompt|"
    r"trademark|trademarkimage|symbolurl|iconurl|emblemurl)($|_)",
    re.I,
)

_LOGO_URL_RE = re.compile(r"https?://[^\s\"']+(?:logo|brand[-_]?mark|emblem|icon)[^\s\"']*", re.I)

_PLAN_LOGO_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        r"\b(show|display|include|print|place|add|attach|use|reproduce)\b.{0,40}\b("
        r"uploaded|provided|supplied|user[- ]supplied|brand guideline)\b.{0,20}\blogo\b",
        "supplied_logo_displayed",
    ),
    (r"\b(invent|create|design|draw|generate|introduce)\b.{0,30}\blogo\b", "invented_product_logo"),
    (r"\bmonogram\b", "logo_like_brand_symbol"),
    (r"\blettermark\b", "logo_like_brand_symbol"),
    (
        r"\b(stylized|stylised)\b.{0,20}\bletter\b.{0,20}\b(mark|logo|symbol)\b",
        "product_name_not_text_only",
    ),
    (r"\bpictorial mark\b", "logo_like_brand_symbol"),
    (r"\bbrand mark\b", "logo_like_brand_symbol"),
    (r"\btrademark symbol\b", "product_name_not_text_only"),
    (r"[®™]", "product_name_not_text_only"),
    (
        r"\b(badge|seal|medallion|crest|shield)\b.{0,25}\b(beside|next to|above|under|around|with)\b.{0,20}\b("
        r"product name|brand name|name)\b",
        "product_name_not_text_only",
    ),
    (
        r"\b(device|graphic device|recurring device)\b.{0,25}\b(on|attached to|printed on|appears on)\b.{0,25}\b("
        r"packaging|can|label|bottle|box|product name|brand name)\b",
        "campaign_device_used_as_logo",
    ),
    (
        r"\b(device|graphic device|recurring device)\b.{0,25}\b(as|become|becomes|function as|serve as)\b.{0,15}\b("
        r"logo|brand mark|brand symbol|trademark)\b",
        "campaign_device_used_as_logo",
    ),
    (
        r"\b(symbol|icon|emblem|mark)\b.{0,20}\b(beside|next to|above|on|with|printed on)\b.{0,20}\b("
        r"product name|brand name|packaging|can|label|bottle|box|bag)\b",
        "packaging_contains_brand_mark",
    ),
    (
        r"\b(packaging|can|bottle|label|box|bag|package|storefront)\b.{0,40}\b("
        r"logo|brand mark|brand symbol|emblem|icon)\b",
        "packaging_contains_brand_mark",
    ),
    (
        r"\b(lightning[- ]bolt|leaf symbol|crown symbol)\b.{0,20}\b(logo|brand mark|brand symbol)\b",
        "logo_like_brand_symbol",
    ),
    (r"(?<!no )(?<!without )(?<!prohibit )\blogo\b", "logo_like_brand_symbol"),
)


def is_no_logo_rejection(codes: Iterable[str]) -> bool:
    return any(code in LOGO_REJECTION_CODES for code in codes)


def _is_logo_prohibited_context(text: str, match_start: int) -> bool:
    window = text[max(0, match_start - 24) : match_start].lower()
    return any(token in window for token in ("no ", "without ", "prohibit ", "never ", "not ", "do not "))


def scan_text_for_logo_violation(text: object) -> Optional[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    for pattern, code in _PLAN_LOGO_PATTERNS:
        match = re.search(pattern, lowered, re.I)
        if not match:
            continue
        if _is_logo_prohibited_context(lowered, match.start()):
            continue
        return code
    return None


def _scan_nested(value: object, reasons: List[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _scan_nested(nested, reasons)
        return
    if isinstance(value, list):
        for nested in value:
            _scan_nested(nested, reasons)
        return
    if not isinstance(value, str):
        return
    code = scan_text_for_logo_violation(value)
    if code:
        reasons.append(code)


def deterministic_no_logo_checks(plan_dict: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    _scan_nested(plan_dict, reasons)
    return list(dict.fromkeys(reasons))


def _should_drop_guideline_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_")
    return bool(_LOGO_GUIDELINE_KEY_RE.search(normalized))


def _sanitize_guideline_value(value: object) -> Optional[object]:
    if isinstance(value, dict):
        cleaned = sanitize_brand_guidelines_for_builder1(value)
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            sanitized = _sanitize_guideline_value(item)
            if sanitized is not None:
                cleaned_list.append(sanitized)
        return cleaned_list or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("data:image"):
            return None
        if _LOGO_URL_RE.search(text):
            return None
        if re.search(r"\blogo\b", text, re.I) and re.search(r"https?://", text, re.I):
            return None
        return text
    return value


def sanitize_brand_guidelines_for_builder1(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    cleaned: Dict[str, Any] = {}
    for key, raw in value.items():
        if _should_drop_guideline_key(str(key)):
            continue
        sanitized = _sanitize_guideline_value(raw)
        if sanitized is None:
            continue
        if isinstance(sanitized, dict) and not sanitized:
            continue
        if isinstance(sanitized, list) and not sanitized:
            continue
        cleaned[str(key)] = sanitized
    return cleaned or None


def brand_guidelines_for_prompt(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    sanitized = sanitize_brand_guidelines_for_builder1(value)
    if sanitized is None:
        return None
    return copy.deepcopy(sanitized)


def public_payload_without_logo_assets(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove logo metadata from public ZIP/API payloads."""
    cleaned = copy.deepcopy(payload)

    def _strip(obj: object) -> None:
        if isinstance(obj, dict):
            drop_keys = [key for key in obj if _should_drop_guideline_key(str(key))]
            for key in drop_keys:
                obj.pop(key, None)
            for key, val in list(obj.items()):
                if isinstance(val, str) and val.startswith("data:image"):
                    obj.pop(key, None)
                else:
                    _strip(val)
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)

    _strip(cleaned)
    return cleaned
