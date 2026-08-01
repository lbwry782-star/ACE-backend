"""
Builder2 advertising-slogan quality contract — Relative Advantage → advertising formulation → canonical slogan.

Distinguishes strategic description from advertising copy. Model-assessed quality is supported by
structured evidence and deterministic structural guards.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from engine.builder2_advertising_closure_contract import (
    GENERIC_SLOGAN_PATTERNS,
    SLOGAN_MAX_WORD_COUNT,
    validate_slogan_text_structure,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION = "builder2_advertising_slogan_quality_v1"

CREATOR_SLOGAN_FORMULATION_KEY = "advertisingSloganFormulation"
WINNER_SLOGAN_EVIDENCE_KEY = "advertisingSloganEvidence"
JUDGE_SLOGAN_ASSESSMENT_KEY = "advertisingSloganAssessment"

VALID_ADVERTISING_TRANSFORMATION_TYPES: FrozenSet[str] = frozenset(
    {
        "compression",
        "contrast",
        "inversion",
        "double_meaning",
        "rhythm",
        "identity",
        "distinctive_claim",
        "imperative",
        "direct_distillation",
        "other_valid_advertising_mechanism",
    }
)

CREATOR_SLOGAN_FORMULATION_FIELDS: Tuple[str, ...] = (
    "relativeAdvantageSource",
    "finalSloganText",
    "advertisingTransformationType",
    "whyThisIsAdvertisingCopy",
    "merelyDescriptive",
    "factualGroundingPreserved",
)

JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS: Tuple[str, ...] = (
    "derivedFromRelativeAdvantage",
    "merelyDescriptive",
    "soundsLikeAdvertising",
    "memorableAfterOneExposure",
    "naturalClosingLine",
    "visualVerbalBridge",
    "factuallyGrounded",
)

STRATEGIC_DESCRIPTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"ממקור\s"),
    re.compile(r"\bגלוי\b"),
    re.compile(r"שממבטא\s"),
    re.compile(r"המשמעות\s"),
    re.compile(r"מייצג\s"),
    re.compile(r"יתרון\s+(?:יחסי|המרכזי|העיקרי)\b"),
    re.compile(r"\bstyle of\b", re.I),
    re.compile(r"\btransparent origin\b", re.I),
    re.compile(r"\bstrategic (?:description|summary|rationale)\b", re.I),
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _raise(code: str, *, field: str) -> None:
    raise Builder2TournamentError(f"{code}:{field}")


def contract_version(*, state: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(state, dict):
        version = _clean(state.get("advertisingSloganQualityContractVersion"))
        if version:
            return version
    if isinstance(plan, dict):
        return _clean(plan.get("advertisingSloganQualityContractVersion"))
    return ""


def requires_advertising_slogan_quality(
    *,
    state: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> bool:
    return contract_version(state=state, plan=plan) == BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION


def stamp_advertising_slogan_quality_contract(state: Dict[str, Any]) -> None:
    state["advertisingSloganQualityContractVersion"] = BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION


def resolve_relative_advantage_statement(
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> str:
    for source in (
        (strategy_foundation or {}).get("relativeAdvantage"),
        (plan or {}).get("relativeAdvantage"),
        ((candidate or {}).get("creatorReport") or {}).get("relativeAdvantage")
        if isinstance((candidate or {}).get("creatorReport"), dict)
        else None,
    ):
        if isinstance(source, dict):
            text = _clean(source.get("statement"))
        else:
            text = _clean(source)
        if text:
            return text
    return ""


def _normalize_comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", _clean(text))
    normalized = re.sub(r"[^\w\s\u0590-\u05FF]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _hebrew_stem(token: str) -> str:
    value = token.strip().lower()
    if not value:
        return value
    for prefix in ("וב", "ול", "מה", "ב", "ה", "ל", "מ", "כ"):
        if value.startswith(prefix) and len(value) > len(prefix) + 1:
            return value[len(prefix) :]
    for suffix in ("ים", "ות", "ית", "י"):
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            return value[: -len(suffix)]
    return value


def _comparison_tokens(text: str) -> List[str]:
    normalized = _normalize_comparison_text(text)
    if not normalized:
        return []
    return [_hebrew_stem(part) for part in normalized.split() if part]


def slogan_restates_relative_advantage(*, slogan: str, relative_advantage: str) -> bool:
    if not _clean(slogan) or not _clean(relative_advantage):
        return False
    normalized_slogan = _normalize_comparison_text(slogan)
    normalized_advantage = _normalize_comparison_text(relative_advantage)
    if normalized_slogan == normalized_advantage:
        return True
    if normalized_slogan in normalized_advantage or normalized_advantage in normalized_slogan:
        return True
    slogan_tokens = _comparison_tokens(slogan)
    advantage_tokens = _comparison_tokens(relative_advantage)
    if not slogan_tokens or not advantage_tokens:
        return False
    slogan_set = set(slogan_tokens)
    advantage_set = set(advantage_tokens)
    overlap_ratio = len(slogan_set & advantage_set) / max(1, len(slogan_set))
    length_ratio = len(slogan_tokens) / max(1, len(advantage_tokens))
    if overlap_ratio >= 0.85 and length_ratio >= 0.65:
        return True
    return False


def slogan_matches_explanatory_rationale(*, slogan: str, rationale_texts: Sequence[str]) -> bool:
    normalized_slogan = _normalize_comparison_text(slogan)
    if not normalized_slogan:
        return False
    for raw in rationale_texts:
        text = _clean(raw)
        if not text:
            continue
        normalized_rationale = _normalize_comparison_text(text)
        if normalized_slogan == normalized_rationale:
            return True
        if normalized_slogan in normalized_rationale or normalized_rationale in normalized_slogan:
            return True
    return False


def slogan_contains_strategic_description_markers(slogan: str) -> bool:
    text = _clean(slogan)
    if not text:
        return False
    return any(pattern.search(text) for pattern in STRATEGIC_DESCRIPTION_PATTERNS)


def validate_slogan_advertising_quality_deterministic(
    *,
    slogan: str,
    product_name: str,
    relative_advantage: str = "",
    explanatory_rationales: Optional[Sequence[str]] = None,
    merely_descriptive: Optional[bool] = None,
    transformation_type: str = "",
    why_advertising: str = "",
) -> None:
    validate_slogan_text_structure(slogan=slogan, product_name=product_name)
    text = _clean(slogan)
    for pattern in GENERIC_SLOGAN_PATTERNS:
        if pattern.search(text):
            _raise("builder2_advertising_slogan_quality_failed", field="sloganText.generic")
    if merely_descriptive is True:
        _raise("builder2_advertising_slogan_quality_failed", field="merelyDescriptive")
    if transformation_type and transformation_type not in VALID_ADVERTISING_TRANSFORMATION_TYPES:
        _raise("builder2_advertising_slogan_quality_failed", field="advertisingTransformationType")
    if transformation_type and not _clean(why_advertising):
        _raise("builder2_advertising_slogan_quality_failed", field="whyThisIsAdvertisingCopy")
    if relative_advantage and slogan_restates_relative_advantage(
        slogan=slogan,
        relative_advantage=relative_advantage,
    ):
        _raise("builder2_advertising_slogan_quality_failed", field="sloganText.restates_relative_advantage")
    rationales = list(explanatory_rationales or [])
    if slogan_matches_explanatory_rationale(slogan=slogan, rationale_texts=rationales):
        _raise("builder2_advertising_slogan_quality_failed", field="sloganText.explanatory_rationale")
    if relative_advantage and slogan_contains_strategic_description_markers(slogan):
        slogan_tokens = set(_comparison_tokens(slogan))
        advantage_tokens = set(_comparison_tokens(relative_advantage))
        if slogan_tokens and advantage_tokens:
            overlap = len(slogan_tokens & advantage_tokens) / max(1, len(slogan_tokens))
            if overlap >= 0.4:
                _raise(
                    "builder2_advertising_slogan_quality_failed",
                    field="sloganText.strategic_description_markers",
                )


def _collect_explanatory_rationales(
    *,
    candidate: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> List[str]:
    texts: List[str] = []
    bridge = (candidate or {}).get("semanticBridge")
    if isinstance(bridge, dict):
        for key in ("strategicMeaning", "howTheMeaningsMeet", "sloganMeaning"):
            text = _clean(bridge.get(key))
            if text:
                texts.append(text)
    report = (candidate or {}).get("creatorReport")
    if isinstance(report, dict):
        for key in ("relativeAdvantage", "whyParallelExpressesAdvantage"):
            text = _clean(report.get(key))
            if text:
                texts.append(text)
    if isinstance(plan, dict):
        for key in ("coreCreativeMechanism", "problemPerception"):
            text = _clean(plan.get(key))
            if text:
                texts.append(text)
    return texts


def normalize_creator_slogan_formulation(raw: Any, *, canonical_slogan: str) -> Dict[str, Any]:
    payload = dict(raw) if isinstance(raw, dict) else {}
    canonical = _clean(canonical_slogan)
    return {
        "relativeAdvantageSource": _clean(payload.get("relativeAdvantageSource")),
        "finalSloganText": canonical,
        "advertisingTransformationType": _clean(payload.get("advertisingTransformationType")),
        "whyThisIsAdvertisingCopy": _clean(payload.get("whyThisIsAdvertisingCopy")),
        "merelyDescriptive": payload.get("merelyDescriptive"),
        "factualGroundingPreserved": payload.get("factualGroundingPreserved"),
    }


def validate_creator_advertising_slogan_formulation(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    product_name: str = "",
) -> None:
    raw = candidate.get(CREATOR_SLOGAN_FORMULATION_KEY)
    if not isinstance(raw, dict):
        _raise("builder2_creator_validation_failed", field=CREATOR_SLOGAN_FORMULATION_KEY)
    closure = candidate.get("advertisingClosure")
    if not isinstance(closure, dict):
        _raise("builder2_creator_validation_failed", field="advertisingClosure")
    canonical_slogan = _clean(closure.get("sloganText"))
    reported_final = _clean(raw.get("finalSloganText"))
    if reported_final != canonical_slogan:
        _raise("builder2_creator_validation_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.finalSloganText")
    formulation = normalize_creator_slogan_formulation(raw, canonical_slogan=canonical_slogan)
    for key in CREATOR_SLOGAN_FORMULATION_FIELDS:
        if key.endswith("Preserved") or key == "merelyDescriptive":
            if not isinstance(formulation.get(key), bool):
                _raise("builder2_creator_validation_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.{key}")
            continue
        if not _clean(formulation.get(key)):
            _raise("builder2_creator_validation_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.{key}")
    if formulation["merelyDescriptive"] is not False:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.merelyDescriptive")
    if formulation["factualGroundingPreserved"] is not True:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.factualGroundingPreserved")
    if formulation["advertisingTransformationType"] not in VALID_ADVERTISING_TRANSFORMATION_TYPES:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.advertisingTransformationType")
    relative_advantage = resolve_relative_advantage_statement(
        strategy_foundation=strategy_foundation,
        candidate=candidate,
    )
    if relative_advantage and formulation["relativeAdvantageSource"] != relative_advantage:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{CREATOR_SLOGAN_FORMULATION_KEY}.relativeAdvantageSource")
    authoritative_product = _clean(product_name)
    if not authoritative_product and isinstance(strategy_foundation, dict):
        authoritative_product = _clean(strategy_foundation.get("productNameResolved"))
    product_label = _clean(closure.get("productNameText")) or authoritative_product
    validate_slogan_advertising_quality_deterministic(
        slogan=canonical_slogan,
        product_name=product_label,
        relative_advantage=relative_advantage,
        explanatory_rationales=_collect_explanatory_rationales(candidate=candidate),
        merely_descriptive=formulation["merelyDescriptive"],
        transformation_type=formulation["advertisingTransformationType"],
        why_advertising=formulation["whyThisIsAdvertisingCopy"],
    )
    candidate[CREATOR_SLOGAN_FORMULATION_KEY] = formulation


def validate_judge_advertising_slogan_assessment(judgment: Dict[str, Any]) -> None:
    assessment = judgment.get(JUDGE_SLOGAN_ASSESSMENT_KEY)
    if not isinstance(assessment, dict):
        _raise("builder2_judge_validation_failed", field=JUDGE_SLOGAN_ASSESSMENT_KEY)
    for key in JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS:
        if not isinstance(assessment.get(key), bool):
            _raise("builder2_judge_validation_failed", field=f"{JUDGE_SLOGAN_ASSESSMENT_KEY}.{key}")
    if not _clean(assessment.get("notes")):
        _raise("builder2_judge_validation_failed", field=f"{JUDGE_SLOGAN_ASSESSMENT_KEY}.notes")
    if judgment.get("eligible") is True:
        if assessment.get("merelyDescriptive") is True:
            _raise("builder2_judge_coherence_violation", field=f"{JUDGE_SLOGAN_ASSESSMENT_KEY}.merelyDescriptive")
        for key in JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS:
            if key == "merelyDescriptive":
                continue
            if assessment.get(key) is not True:
                _raise("builder2_judge_coherence_violation", field=f"{JUDGE_SLOGAN_ASSESSMENT_KEY}.{key}")


def apply_advertising_slogan_eligibility_rules(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    assessment = out.get(JUDGE_SLOGAN_ASSESSMENT_KEY)
    if not isinstance(assessment, dict) or out.get("eligible") is not True:
        return out
    disqualifiers = list(out.get("disqualifiers") or [])
    if assessment.get("merelyDescriptive") is True:
        out["eligible"] = False
        if "slogan_merely_descriptive" not in disqualifiers:
            disqualifiers.append("slogan_merely_descriptive")
    elif assessment.get("soundsLikeAdvertising") is False:
        out["eligible"] = False
        if "slogan_not_advertising_copy" not in disqualifiers:
            disqualifiers.append("slogan_not_advertising_copy")
    elif assessment.get("derivedFromRelativeAdvantage") is False:
        out["eligible"] = False
        if "slogan_not_derived_from_advantage" not in disqualifiers:
            disqualifiers.append("slogan_not_derived_from_advantage")
    elif assessment.get("naturalClosingLine") is False:
        out["eligible"] = False
        if "slogan_not_natural_closing_line" not in disqualifiers:
            disqualifiers.append("slogan_not_natural_closing_line")
    out["disqualifiers"] = disqualifiers
    return out


def validate_winner_advertising_slogan_evidence(
    winner_plan: Dict[str, Any],
    *,
    winning_candidate: Dict[str, Any],
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> None:
    raw = winner_plan.get(WINNER_SLOGAN_EVIDENCE_KEY)
    if not isinstance(raw, dict):
        _raise("builder2_winner_validation_failed", field=WINNER_SLOGAN_EVIDENCE_KEY)
    closure = winner_plan.get("advertisingClosure")
    if not isinstance(closure, dict):
        closure = (winning_candidate or {}).get("advertisingClosure")
    canonical_slogan = _clean((closure or {}).get("sloganText"))
    reported_final = _clean(raw.get("finalSloganText"))
    if reported_final != canonical_slogan:
        _raise("builder2_winner_validation_failed", field=f"{WINNER_SLOGAN_EVIDENCE_KEY}.finalSloganText")
    evidence = normalize_creator_slogan_formulation(raw, canonical_slogan=canonical_slogan)
    if evidence["merelyDescriptive"] is not False:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{WINNER_SLOGAN_EVIDENCE_KEY}.merelyDescriptive")
    if evidence["factualGroundingPreserved"] is not True:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{WINNER_SLOGAN_EVIDENCE_KEY}.factualGroundingPreserved")
    relative_advantage = resolve_relative_advantage_statement(
        strategy_foundation=strategy_foundation,
        candidate=winning_candidate,
        plan=winner_plan,
    )
    if relative_advantage and evidence["relativeAdvantageSource"] != relative_advantage:
        _raise("builder2_advertising_slogan_quality_failed", field=f"{WINNER_SLOGAN_EVIDENCE_KEY}.relativeAdvantageSource")
    product_name = _clean(winner_plan.get("productNameResolved"))
    if not product_name and isinstance(closure, dict):
        product_name = _clean(closure.get("productNameText"))
    validate_slogan_advertising_quality_deterministic(
        slogan=canonical_slogan,
        product_name=product_name,
        relative_advantage=relative_advantage,
        explanatory_rationales=_collect_explanatory_rationales(candidate=winning_candidate, plan=winner_plan),
        merely_descriptive=evidence["merelyDescriptive"],
        transformation_type=evidence["advertisingTransformationType"],
        why_advertising=evidence["whyThisIsAdvertisingCopy"],
    )
    winner_plan[WINNER_SLOGAN_EVIDENCE_KEY] = evidence


def build_creator_advertising_slogan_prompt_text(*, max_words: int = SLOGAN_MAX_WORD_COUNT) -> str:
    enum_values = ", ".join(sorted(VALID_ADVERTISING_TRANSFORMATION_TYPES))
    return (
        "Advertising-slogan quality (mandatory): advertisingClosure.sloganText must be an ADVERTISING FORMULATION "
        "of the selected Relative Advantage — not a strategy report, product-description field, rationale, or "
        "near-verbatim restatement.\n"
        "Inside this same Creator response (no extra call): read the exact Relative Advantage source, generate "
        "multiple possible advertising formulations internally, reject merely descriptive variants, select one "
        "final canonical slogan for advertisingClosure.sloganText, and record structured evidence in "
        f"{CREATOR_SLOGAN_FORMULATION_KEY}{{relativeAdvantageSource,finalSloganText,advertisingTransformationType,"
        "whyThisIsAdvertisingCopy,merelyDescriptive,factualGroundingPreserved}}.\n"
        f"relativeAdvantageSource must equal the strategy Relative Advantage statement exactly.\n"
        f"finalSloganText must equal advertisingClosure.sloganText exactly.\n"
        f"advertisingTransformationType must be one of: {enum_values}.\n"
        "merelyDescriptive must be false. factualGroundingPreserved must be true.\n"
        "whyThisIsAdvertisingCopy must explain why the selected line is advertising copy rather than strategic prose.\n"
        "Direct distillation is allowed when the line is already concise, memorable, and persuasive.\n"
        "Wordplay is not required. Do not persist discarded variants.\n"
        f"The final slogan must function as the closing advertising line beneath plain-text productNameText, within "
        f"{max_words} words excluding the product name."
    )


def build_judge_advertising_slogan_prompt_text() -> str:
    fields = ", ".join(JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS)
    return (
        f"{JUDGE_SLOGAN_ASSESSMENT_KEY} must include {fields} (booleans) and notes.\n"
        "Assess whether advertisingClosure.sloganText is an advertising formulation of the Relative Advantage — "
        "not merely semantically accurate strategic prose.\n"
        "merelyDescriptive=true means the slogan explains or restates the advantage instead of transforming it "
        "into memorable advertising copy.\n"
        "Do not award full slogan quality credit to merely descriptive lines. eligible=true requires "
        "merelyDescriptive=false and all other boolean fields true."
    )


def build_winner_advertising_slogan_prompt_text() -> str:
    enum_values = ", ".join(sorted(VALID_ADVERTISING_TRANSFORMATION_TYPES))
    return (
        f"Required {WINNER_SLOGAN_EVIDENCE_KEY} object mirroring the winning Creator slogan evidence: "
        "relativeAdvantageSource, finalSloganText, advertisingTransformationType, whyThisIsAdvertisingCopy, "
        "merelyDescriptive=false, factualGroundingPreserved=true.\n"
        f"finalSloganText must equal advertisingClosure.sloganText exactly — do not replace the winning Creator slogan.\n"
        f"relativeAdvantageSource must equal the preserved Relative Advantage statement exactly.\n"
        f"advertisingTransformationType must be one of: {enum_values}.\n"
        "Reject strategic prose masquerading as final copy."
    )


def sync_creator_slogan_formulation_from_closure(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> None:
    closure = candidate.get("advertisingClosure")
    if not isinstance(closure, dict):
        return
    slogan = _clean(closure.get("sloganText"))
    if not slogan:
        return
    raw = candidate.get(CREATOR_SLOGAN_FORMULATION_KEY)
    if not isinstance(raw, dict):
        return
    formulation = dict(raw)
    formulation["finalSloganText"] = slogan
    relative_advantage = resolve_relative_advantage_statement(
        strategy_foundation=strategy_foundation,
        candidate=candidate,
    )
    if relative_advantage:
        formulation["relativeAdvantageSource"] = relative_advantage
    candidate[CREATOR_SLOGAN_FORMULATION_KEY] = formulation


def sync_winner_slogan_evidence_from_closure(
    winner_plan: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    winning_candidate: Optional[Dict[str, Any]] = None,
) -> None:
    raw = winner_plan.get(WINNER_SLOGAN_EVIDENCE_KEY)
    if not isinstance(raw, dict):
        return
    closure = winner_plan.get("advertisingClosure")
    if not isinstance(closure, dict) and isinstance((winning_candidate or {}).get("advertisingClosure"), dict):
        closure = winning_candidate.get("advertisingClosure")
    slogan = _clean((closure or {}).get("sloganText"))
    if not slogan:
        return
    evidence = dict(raw)
    evidence["finalSloganText"] = slogan
    relative_advantage = resolve_relative_advantage_statement(
        strategy_foundation=strategy_foundation,
        candidate=winning_candidate,
        plan=winner_plan,
    )
    if relative_advantage:
        evidence["relativeAdvantageSource"] = relative_advantage
    winner_plan[WINNER_SLOGAN_EVIDENCE_KEY] = evidence


def build_default_creator_slogan_formulation(
    *,
    relative_advantage_source: str,
    final_slogan_text: str,
    transformation_type: str = "direct_distillation",
    why_advertising: str = "The line compresses the relative advantage into a memorable closing claim.",
) -> Dict[str, Any]:
    return {
        "relativeAdvantageSource": _clean(relative_advantage_source),
        "finalSloganText": _clean(final_slogan_text),
        "advertisingTransformationType": transformation_type,
        "whyThisIsAdvertisingCopy": _clean(why_advertising),
        "merelyDescriptive": False,
        "factualGroundingPreserved": True,
    }


def build_default_judge_slogan_assessment(*, notes: str = "") -> Dict[str, Any]:
    return {
        "derivedFromRelativeAdvantage": True,
        "merelyDescriptive": False,
        "soundsLikeAdvertising": True,
        "memorableAfterOneExposure": True,
        "naturalClosingLine": True,
        "visualVerbalBridge": True,
        "factuallyGrounded": True,
        "notes": notes
        or "The slogan transforms the relative advantage into concise advertising copy rather than explanatory prose.",
    }
