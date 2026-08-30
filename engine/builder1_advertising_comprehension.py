"""
Builder1 advertising comprehension — deterministic planning and compliance context.

Distinguishes physical mechanism clarity from advertising bridge clarity, rejects
multi-hop proxy chains, and validates dominant-object strategic roles without
adding paid model calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Set

from engine.builder1_plan_spec import Builder1SeriesPlan

ADVERTISING_COMPREHENSION_REJECTION_CODES = frozenset(
    {
        "advertising_bridge_unclear",
        "multi_hop_symbolic_chain",
        "dominant_object_strategic_role_missing",
    }
)

EXECUTION_FIDELITY_VIOLATION_CODES = frozenset(
    {
        "planned_scene_diverged",
        "planned_mechanism_diverged",
        "central_proof_not_visible",
        "unintended_dominant_interpretation",
        "advertising_bridge_not_recoverable",
        "relative_advantage_not_expressed",
        "visual_slogan_mechanism_mismatch",
        "dominant_subject_diverged",
    }
)

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "they",
        "their",
        "there",
        "where",
        "when",
        "what",
        "which",
        "while",
        "about",
        "through",
        "because",
        "viewer",
        "visual",
        "scene",
        "shows",
        "show",
        "visible",
        "clearly",
        "instant",
        "instantly",
        "immediately",
        "understand",
        "understandable",
        "every",
        "other",
        "variant",
        "subject",
        "action",
        "state",
        "punchline",
        "proof",
        "object",
        "physical",
        "advertising",
        "product",
        "service",
        "brand",
        "campaign",
        "lesson",
        "lessons",
    }
)

_PHYSICAL_ONLY_CLARITY_MARKERS = (
    "familiar",
    "understandable",
    "everyone knows",
    "object mechanics",
    "literal scene",
    "merely familiar",
    "only familiar",
    "are familiar",
    "physically familiar",
)

_ADVANTAGE_BRIDGE_MARKERS = (
    "viewer infers",
    "viewer understands",
    "advertising meaning",
    "relative advantage",
    "therefore the",
    "so the viewer",
    "means for",
    "shows why",
    "proves the advantage",
    "connects the visual",
    "without the slogan",
    "about the product",
    "about the service",
    "infers about",
    "understands that",
    "advertised offer",
    "specifically expresses",
    "expresses the advantage",
    "not merely",
    "not just an interesting",
)

_SYMBOLIC_MAPPING_MARKERS = (
    "represents",
    "symbolizes",
    "symbolises",
    "stands for",
    "maps to",
    "equivalent to",
    "metaphor for",
    "analogy to",
    "like a",
    "as if",
    "translates to",
    "→",
    "->",
)

_COMPETING_INTERPRETATION_TERMS = frozenset(
    {
        "railway",
        "rail",
        "rails",
        "track",
        "tracks",
        "train",
        "road",
        "route",
        "routes",
        "path",
        "paths",
        "maze",
        "map",
        "navigation",
        "highway",
        "station",
    }
)

_STRATEGIC_ROLE_MARKERS = (
    "proves",
    "shows",
    "expresses",
    "role",
    "because",
    "tests",
    "test",
    "survives",
    "demonstrates",
    "means",
    "readiness",
    "advantage",
    "focus",
    "focused",
    "preparation",
    "mechanism",
    "punchline",
)

_UNEXPLAINED_DOMINANT_NOUNS = frozenset(
    {
        "clock",
        "calendar",
        "tower",
        "train",
        "railway",
        "tracks",
        "road",
        "map",
        "maze",
    }
)

_GENERIC_EXECUTION_RE = re.compile(
    r"^(subject|object|scene|action|state|punchline)\s+variant\s+\d+$",
    re.IGNORECASE,
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _significant_tokens(text: str) -> Set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", _norm(text))
        if token.casefold() not in _STOPWORDS
    }


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = _significant_tokens(left)
    right_tokens = _significant_tokens(right)
    return bool(left_tokens & right_tokens)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = _norm(text).casefold()
    return any(marker in lowered for marker in markers)


def _count_symbolic_mappings(*texts: str) -> int:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return 0
    count = 0
    for marker in _SYMBOLIC_MAPPING_MARKERS:
        count += len(re.findall(re.escape(marker), combined))
    count += len(re.findall(r"\bmeans\b", combined))
    return count


def _ad_internal_fields(plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> Dict[str, Any]:
    index = ad.get("index")
    internals = plan_dict.get("planningInternals") or plan_dict.get("planning_internals") or {}
    ad_internals = internals.get("adInternals") if isinstance(internals, dict) else {}
    merged = dict(ad)
    if isinstance(ad_internals, dict) and index is not None:
        per_ad = ad_internals.get(index) or ad_internals.get(str(index))
        if isinstance(per_ad, dict):
            merged = {**merged, **per_ad}
    return merged


def _dominant_execution_tokens(*, plan_dict: Mapping[str, Any], ad: Mapping[str, Any]) -> Set[str]:
    fields = _ad_internal_fields(plan_dict, ad)
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))
    candidates = [
        transferred,
        _norm(fields.get("executionSubject")),
        _norm(fields.get("executionAction")),
        _norm(fields.get("executionScene")),
        _norm(ad.get("physicalExecution")),
        _norm(ad.get("sceneDescription")),
    ]
    tokens: Set[str] = set()
    for text in candidates:
        if not text or _GENERIC_EXECUTION_RE.match(text):
            continue
        tokens.update(_significant_tokens(text))
    return tokens


def _negated_competing_terms(no_reuse_check: str) -> Set[str]:
    lowered = _norm(no_reuse_check).casefold()
    negated: Set[str] = set()
    if not lowered:
        return negated
    for term in _COMPETING_INTERPRETATION_TERMS:
        if re.search(rf"\b(?:no|not|without|never|avoid|excluding)\b[^.]{{0,24}}\b{re.escape(term)}\b", lowered):
            negated.add(term)
        if re.search(rf"\b{re.escape(term)}\b[^.]{{0,24}}\b(?:not|forbidden|excluded)\b", lowered):
            negated.add(term)
    return negated


def _scene_mentions_competing_term(text: str, term: str) -> bool:
    lowered = _norm(text).casefold()
    if not lowered:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", lowered))


def _strategic_role_explained(term: str, *texts: str) -> bool:
    combined = " ".join(_norm(text) for text in texts if text).casefold()
    if not combined:
        return False
    if term not in combined:
        return False
    window_pattern = rf".{{0,80}}\b{re.escape(term)}\b.{{0,80}}"
    match = re.search(window_pattern, combined)
    if not match:
        return False
    window = match.group(0)
    return _contains_any(window, _STRATEGIC_ROLE_MARKERS) or _contains_any(combined, _ADVANTAGE_BRIDGE_MARKERS)


def validate_ad_advertising_comprehension(
    *,
    plan_dict: Mapping[str, Any],
    ad: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    fields = _ad_internal_fields(plan_dict, ad)
    relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
    immediate = _norm(fields.get("immediateClarityReason"))
    bridge = _norm(fields.get("relativeAdvantageConnection"))
    slogan_connection = _norm(fields.get("sloganConnection"))
    punchline = _norm(fields.get("executionPunchline"))
    no_reuse = _norm(fields.get("noReuseCheck"))
    transferred = _norm(plan_dict.get("transferredObject") or plan_dict.get("physicalGenerator"))

    if immediate and bridge:
        physical_only = (
            _contains_any(immediate, _PHYSICAL_ONLY_CLARITY_MARKERS)
            and not _token_overlap(relative_advantage, immediate)
        )
        bridge_ok = (
            _contains_any(bridge, _ADVANTAGE_BRIDGE_MARKERS)
            or _token_overlap(relative_advantage, bridge)
            or _token_overlap(relative_advantage, slogan_connection)
            or _token_overlap(relative_advantage, punchline)
            or (
                _contains_any(bridge, ("proves", "shows", "demonstrates", "expresses", "means"))
                and (
                    _token_overlap(bridge, _norm(fields.get("singleChangedPropertyOrAction")))
                    or _token_overlap(bridge, _norm(fields.get("conceptualActionProof")))
                    or _token_overlap(bridge, _norm(ad.get("conceptualExecution")))
                )
            )
        )
        if physical_only and not bridge_ok:
            reasons.append("advertising_bridge_unclear")

    mapping_count = _count_symbolic_mappings(
        immediate,
        bridge,
        slogan_connection,
        _norm(plan_dict.get("conceptualGenerator")),
        _norm(plan_dict.get("conceptualGeneratorAction")),
        _norm(fields.get("conceptualExecution")),
    )
    if mapping_count >= 3 and not (
        _contains_any(bridge, _ADVANTAGE_BRIDGE_MARKERS) or _token_overlap(relative_advantage, bridge)
    ):
        reasons.append("multi_hop_symbolic_chain")

    scene = " ".join(
        _norm(fields.get(field))
        for field in ("executionScene", "executionSubject", "executionAction", "executionObjectState")
    )
    scene = f"{scene} {_norm(ad.get('sceneDescription'))}".strip()
    strategic_blob = " ".join(
        part for part in (bridge, slogan_connection, punchline, transferred, no_reuse) if part
    ).casefold()
    for term in _UNEXPLAINED_DOMINANT_NOUNS:
        if term in scene.casefold() and term not in strategic_blob and term not in transferred.casefold():
            reasons.append("dominant_object_strategic_role_missing")
            break

    negated = _negated_competing_terms(no_reuse)
    scene_blob = " ".join(
        _norm(fields.get(field))
        for field in ("executionScene", "executionSubject", "executionAction", "executionObjectState")
    )
    for term in negated:
        if _scene_mentions_competing_term(scene_blob, term) and not _strategic_role_explained(
            term,
            bridge,
            punchline,
            slogan_connection,
        ):
            reasons.append("dominant_object_strategic_role_missing")
            break

    return list(dict.fromkeys(reasons))


def scan_advertising_comprehension(plan_dict: Mapping[str, Any]) -> List[str]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return []
    reasons: List[str] = []
    for ad in ads:
        if isinstance(ad, dict):
            reasons.extend(validate_ad_advertising_comprehension(plan_dict=plan_dict, ad=ad))
    return list(dict.fromkeys(reasons))


def _ad_internals_for_index(series_plan: Builder1SeriesPlan, ad_index: int) -> Dict[str, Any]:
    internals = series_plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    raw = ad_internals.get(ad_index) or ad_internals.get(str(ad_index)) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _ad_plan_fields(series_plan: Builder1SeriesPlan, ad_index: int) -> Dict[str, Any]:
    ad = next((item for item in series_plan.ads if item.index == ad_index), None)
    if ad is None:
        return {}
    return {
        "index": ad.index,
        "physicalExecution": ad.physical_execution,
        "visualExecution": ad.visual_execution,
        "sceneDescription": ad.scene_description,
        "conceptualExecution": ad.conceptual_execution,
    }


def build_planned_execution_compliance_block(
    series_plan: Builder1SeriesPlan,
    *,
    ad_index: int = 1,
) -> str:
    """Authoritative planned execution context for the existing compliance review call."""
    internals = _ad_internals_for_index(series_plan, ad_index)
    ad_fields = _ad_plan_fields(series_plan, ad_index)
    lines = [
        "=== PLANNED EXECUTION CONTEXT (AUTHORITATIVE — JUDGE FIDELITY, NOT BEAUTY) ===",
        f'relativeAdvantage: "{series_plan.relative_advantage}"',
        f'brandSlogan: "{series_plan.brand_slogan}"',
        f'sloganAction: "{series_plan.slogan_action}"',
        f'physicalGenerator: "{series_plan.physical_generator}"',
        f'transferredObject: "{series_plan.transferred_object or series_plan.physical_generator}"',
        f'transferredObjectAction: "{series_plan.transferred_object_action or series_plan.physical_generator_campaign_role}"',
        f'conceptualGenerator: "{series_plan.conceptual_generator}"',
        f"executionSubject: {_norm(internals.get('executionSubject') or ad_fields.get('physicalExecution'))}",
        f"executionAction: {_norm(internals.get('executionAction'))}",
        f"executionObjectState: {_norm(internals.get('executionObjectState'))}",
        f"executionScene: {_norm(internals.get('executionScene') or ad_fields.get('sceneDescription'))}",
        f"executionPunchline: {_norm(internals.get('executionPunchline'))}",
        f"immediateClarityReason: {_norm(internals.get('immediateClarityReason'))}",
        f"relativeAdvantageConnection: {_norm(internals.get('relativeAdvantageConnection'))}",
        f"sloganConnection: {_norm(internals.get('sloganConnection'))}",
        f"noReuseCheck: {_norm(internals.get('noReuseCheck'))}",
        "",
        "Evaluate whether the generated pixels faithfully execute this approved mechanism.",
        "This is NOT aesthetic criticism.",
        "Fail when concrete evidence shows:",
        "- planned object/scene became a materially different object/context (e.g. conveyor vs railway tracks)",
        "- central proof/punchline is absent or visually ambiguous",
        "- dominant subject diverged from the planned transferred object or execution subject",
        "- an unintended dominant interpretation appears (especially one explicitly denied in noReuseCheck)",
        "- a normal viewer cannot recover the relative advantage from pixels alone (slogan may reinforce, not carry all meaning)",
        "- pixels and slogan communicate unrelated mechanisms",
        "",
        "Use hardViolations with these codes when confidence is sufficient:",
        ", ".join(sorted(EXECUTION_FIDELITY_VIOLATION_CODES)),
        "=== END PLANNED EXECUTION CONTEXT ===",
    ]
    return "\n".join(lines)


def build_execution_fidelity_correction_block(
    *,
    violations: List[str],
    series_plan: Builder1SeriesPlan,
    ad_index: int = 1,
) -> str:
    fidelity = [code for code in violations if code in EXECUTION_FIDELITY_VIOLATION_CODES]
    if not fidelity:
        return ""
    internals = _ad_internals_for_index(series_plan, ad_index)
    scene = _norm(internals.get("executionScene"))
    punchline = _norm(internals.get("executionPunchline"))
    no_reuse = _norm(internals.get("noReuseCheck"))
    transferred = series_plan.transferred_object or series_plan.physical_generator
    lines = [
        "=== EXECUTION FIDELITY CORRECTION (MANDATORY) ===",
        f"Preserve the approved campaign plan. Regenerate pixels only.",
        f"MAIN VISUAL must remain: {transferred}",
        f"Planned scene/context: {scene or '(see approved plan)'}",
        f"Central proof/punchline must be immediately visible: {punchline or '(see approved plan)'}",
    ]
    if no_reuse:
        lines.append(f"Explicit exclusions from approved plan: {no_reuse}")
    if "planned_scene_diverged" in fidelity or "planned_mechanism_diverged" in fidelity:
        lines.append(
            "Render the approved industrial/testing/conveyor context exactly — do NOT substitute railway tracks, roads, or route imagery unless explicitly planned."
        )
    if "central_proof_not_visible" in fidelity:
        lines.append(
            "Make the planned proof/punchline visually dominant and unambiguous in the final pixels."
        )
    if "unintended_dominant_interpretation" in fidelity or "dominant_subject_diverged" in fidelity:
        lines.append(
            "Remove competing unintended interpretations. The dominant subject must match the approved transferred object and strategic mechanism."
        )
    if "advertising_bridge_not_recoverable" in fidelity or "relative_advantage_not_expressed" in fidelity:
        lines.append(
            "Ensure the visible mechanism makes the relative advantage recoverable without requiring hidden symbolic translation."
        )
    lines.extend(
        [
            "Do not change Product Name, slogan, palette, or graphic system.",
            "=== END EXECUTION FIDELITY CORRECTION ===",
        ]
    )
    return "\n".join(lines)
