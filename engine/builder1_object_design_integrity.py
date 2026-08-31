"""
Builder1 object design integrity — familiar form vs justified deviation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR = "CANONICAL_FAMILIAR"
OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION = "JUSTIFIED_DEVIATION"

OBJECT_DESIGN_MODES = frozenset(
    {
        OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR,
        OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
    }
)

OBJECT_DESIGN_REJECTION_CODES = frozenset(
    {
        "object_design_intent_missing",
        "object_design_mode_invalid",
        "object_design_description_missing",
        "object_design_deviation_unjustified",
        "object_design_deviation_reason_forbidden",
        "object_design_deviation_reason_missing",
        "object_design_salient_language_unjustified",
    }
)

BUILDER1_OBJECT_DESIGN_INTEGRITY = """
OBJECT DESIGN INTEGRITY — mandatory on every ad:
Once a physical object is selected to carry the advertising idea, specify how that object
should look — not only what it does.

Default: CANONICAL_FAMILIAR — an ordinary, immediately recognizable, context-appropriate
real-world instance of the selected object. Do not invent conspicuous unusual material,
color, futuristic styling, transparency, decorative redesign, or synthetic treatment
unless the deviation itself carries advertising meaning.

JUSTIFIED_DEVIATION is allowed only when the unusual material, color, shape, scale, age,
surface, transparency, construction, or style directly serves:
advertising concept, physical mechanism, relative advantage, visual parallel, necessary
visibility, factual product identity, deliberate brand ownership, or contextual realism.

Operational test before choosing JUSTIFIED_DEVIATION:
"If this salient design property were replaced with an ordinary familiar version of the
same object, would any advertising, functional, physical, or brand meaning be lost?"
If NO — use CANONICAL_FAMILIAR instead.

Novelty alone is not justification. Campaign-palette matching alone is not justification.
""".strip()

BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY = """
CAMPAIGN PALETTE BOUNDARY:
Campaign palette governs graphic design by default — typography, graphic accents,
backgrounds, borders, layout shapes — NOT the inherent material or color of real-world
physical objects.

Do not recolor or restyle a familiar physical object merely to harmonize it with the
campaign palette. Object recoloring is allowed only when it is an approved
JUSTIFIED_DEVIATION or factual advertised-product / brand identity.
""".strip()

_CANONICAL_FORBIDDEN_SALIENT_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bneon\b",
        r"\bfuturistic\b",
        r"\bsci[- ]?fi\b",
        r"\bcyber(?:punk)?\b",
        r"\bholographic\b",
        r"\bglow(?:ing|s)?\b",
        r"\bsurreal\b",
        r"\btransparent\b",
        r"\btranslucent\b",
        r"\bchrome[- ]?plated\b",
        r"\bfluorescent\b",
        r"\bbright pink\b",
        r"\boversized\b",
        r"\bminiature\b",
        r"\bluxury styling\b",
        r"\bdecorative redesign\b",
        r"\bעתידני\b",
        r"\bשקוף\b",
        r"\bזוהר\b",
        r"\bניאון\b",
        r"\bורוד\b",
        r"\bמוגזם\b",
    )
)

_PALETTE_ONLY_DEVIATION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpalette\b",
        r"\baccent color\b",
        r"\bcampaign color\b",
        r"\bharmoniz",
        r"\bvisual interest\b",
        r"\blooks interesting\b",
        r"\bfor novelty\b",
        r"\bmerely for\b",
        r"\bonly because.*pink\b",
        r"\bצבע המותג\b",
        r"\bפלטת\b",
        r"\bנועד להתאים\b",
    )
)

_SUBSTANTIVE_DEVIATION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmechanism\b",
        r"\bconcept\b",
        r"\badvantage\b",
        r"\bvisibility\b",
        r"\bbrand\b",
        r"\bproduct identity\b",
        r"\bscale\b",
        r"\btransformation\b",
        r"\bhistorical\b",
        r"\binternal\b",
        r"\bproof\b",
        r"\bמנגנון\b",
        r"\bיתרון\b",
        r"\bמותג\b",
        r"\bהיסטור\b",
        r"\bפנימי\b",
    )
)

_VISUAL_RENDERING_MARKERS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcamera\b",
        r"\bangle\b",
        r"\boverhead\b",
        r"\bfrontal\b",
        r"\bphotoreal",
        r"\blandscape orientation\b",
        r"\bportrait orientation\b",
        r"\blighting\b",
        r"\bbackground\b",
        r"\bframing\b",
        r"\bcrop\b",
        r"\bcomposition\b",
        r"צילום",
        r"זווית",
        r"רקע",
        r"תאור",
        r"פריים",
        r"ריאליסט",
    )
)

_MIN_DESCRIPTION_LEN = 12
_MIN_DEVIATION_REASON_LEN = 24


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_object_design_mode(value: object) -> str:
    raw = _norm(value).upper().replace("-", "_").replace(" ", "_")
    if raw in OBJECT_DESIGN_MODES:
        return raw
    return ""


def get_ad_object_design_fields(ad: Dict[str, Any]) -> Dict[str, str]:
    return {
        "objectDesignMode": normalize_object_design_mode(ad.get("objectDesignMode")),
        "objectDesignDescription": _norm(ad.get("objectDesignDescription")),
        "objectDesignDeviationReason": _norm(ad.get("objectDesignDeviationReason")),
    }


def find_canonical_salient_language(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in _CANONICAL_FORBIDDEN_SALIENT_PATTERNS:
        if pattern.search(text or ""):
            hits.append(pattern.pattern)
    return hits


def validate_ad_object_design(ad: Dict[str, Any], *, ad_index: Optional[int] = None) -> List[str]:
    """Deterministic contract validation — not world-knowledge judging."""
    reasons: List[str] = []
    fields = get_ad_object_design_fields(ad)
    mode = fields["objectDesignMode"]
    description = fields["objectDesignDescription"]
    deviation_reason = fields["objectDesignDeviationReason"]

    if not mode:
        if ad.get("objectDesignMode") is None and ad.get("objectDesignDescription") is None:
            reasons.append("object_design_intent_missing")
        elif not mode:
            reasons.append("object_design_mode_invalid")
        return reasons

    if not description or len(description) < _MIN_DESCRIPTION_LEN:
        reasons.append("object_design_description_missing")

    if mode == OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR:
        if deviation_reason:
            reasons.append("object_design_deviation_reason_forbidden")
        if description and find_canonical_salient_language(description):
            reasons.append("object_design_salient_language_unjustified")
    elif mode == OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION:
        if not deviation_reason or len(deviation_reason) < _MIN_DEVIATION_REASON_LEN:
            reasons.append("object_design_deviation_reason_missing")
        elif not _deviation_reason_is_substantive(deviation_reason):
            reasons.append("object_design_deviation_unjustified")
        if description and find_canonical_salient_language(description):
            # Allowed when deviation is justified — description may name the unusual property.
            pass
    else:
        reasons.append("object_design_mode_invalid")

    return reasons


def _deviation_reason_is_substantive(reason: str) -> bool:
    if any(pattern.search(reason) for pattern in _SUBSTANTIVE_DEVIATION_PATTERNS):
        return True
    if any(pattern.search(reason) for pattern in _PALETTE_ONLY_DEVIATION_PATTERNS):
        return False
    return len(reason.split()) >= 6


def validate_series_ads_object_design(ads: Sequence[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    for ad in ads:
        if not isinstance(ad, dict):
            continue
        try:
            idx = int(ad.get("index"))
        except (TypeError, ValueError):
            idx = None
        reasons.extend(validate_ad_object_design(ad, ad_index=idx))
    return list(dict.fromkeys(reasons))


def scan_object_design_integrity(plan_dict: Dict[str, Any]) -> List[str]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return ["object_design_intent_missing"]
    return validate_series_ads_object_design(ads)


def _visual_execution_adds_rendering_context(physical_execution: str, visual_execution: str) -> bool:
    pe = _norm(physical_execution)
    ve = _norm(visual_execution)
    if not ve:
        return False
    if pe == ve:
        return False
    if pe and pe in ve and not any(marker.search(ve) for marker in _VISUAL_RENDERING_MARKERS):
        return False
    if any(marker.search(ve) for marker in _VISUAL_RENDERING_MARKERS):
        return True
    if pe and len(set(ve.lower().split()) - set(pe.lower().split())) >= 4:
        return True
    return bool(ve and not pe)


def build_composition_execution_lines(
    *,
    physical_execution: str,
    visual_execution: str,
) -> List[str]:
    """Separate physical action from camera/composition rendering."""
    pe = _norm(physical_execution)
    ve = _norm(visual_execution)
    lines: List[str] = []
    if pe:
        lines.append(f"Physical action: {pe}.")
    if _visual_execution_adds_rendering_context(pe, ve):
        lines.append(f"Visual rendering: {ve}.")
    elif not pe and ve:
        lines.append(f"Composition execution: {ve}.")
    return lines


def build_object_design_prompt_block(
    ad_design: Dict[str, str],
    *,
    skip_for_product_led: bool = False,
) -> str:
    if skip_for_product_led:
        return (
            "=== OBJECT DESIGN (PRODUCT-LED — FACTUAL APPEARANCE AUTHORITATIVE) ===\n"
            "Render the advertised product with its factual real-world appearance.\n"
            "Generic canonical-form rules do not override actual product identity.\n"
            "=== END OBJECT DESIGN ==="
        )

    mode_raw = ad_design.get("objectDesignMode") or ""
    description = ad_design.get("objectDesignDescription") or ""
    deviation_reason = ad_design.get("objectDesignDeviationReason") or ""

    if not mode_raw and not description:
        return ""

    mode = normalize_object_design_mode(mode_raw) or OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR

    lines = [
        "=== OBJECT DESIGN (APPROVED — MANDATORY) ===",
        BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY.strip(),
    ]
    if mode == OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION:
        lines.extend(
            [
                "Approved design mode: JUSTIFIED_DEVIATION.",
                f"Approved object appearance: {description}.",
                "Render this exact approved design property — it carries advertising meaning.",
            ]
        )
        if deviation_reason:
            lines.append(
                "Do not add further salient material, color, or styling beyond this approved deviation."
            )
    else:
        lines.extend(
            [
                "Approved design mode: CANONICAL_FAMILIAR.",
                "Render the selected physical object in a familiar, immediately recognizable, "
                "context-appropriate real-world form.",
                "Do not invent unusual materials, colors, futuristic styling, transparency, "
                "or decorative redesign unless explicitly approved below.",
                f"Approved object appearance: {description}.",
            ]
        )
    lines.append("=== END OBJECT DESIGN ===")
    return "\n".join(lines)


def default_canonical_object_design(description: str) -> Dict[str, str]:
    return {
        "objectDesignMode": OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR,
        "objectDesignDescription": description,
        "objectDesignDeviationReason": "",
    }


def default_justified_object_design(description: str, reason: str) -> Dict[str, str]:
    return {
        "objectDesignMode": OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
        "objectDesignDescription": description,
        "objectDesignDeviationReason": reason,
    }


def object_design_fields_for_ad_index(series_plan: Any, ad_index: int) -> Dict[str, str]:
    internals = getattr(series_plan, "planning_internals", None) or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    raw = ad_internals.get(ad_index) or ad_internals.get(str(ad_index)) or {}
    if not isinstance(raw, dict):
        raw = {}
    return get_ad_object_design_fields(raw)
