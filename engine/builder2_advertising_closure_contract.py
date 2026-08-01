"""
Builder2 Advertising Closure contract — mandatory final delivery identification and promise closure.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_is_omit,
    headline_decision_requires_headline,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SLOGAN_MAX_WORD_COUNT = 7

from engine.builder2_closure_duration_contract import BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS

DEFAULT_CLOSURE_DURATION_SECONDS = BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS
DEFAULT_CLOSURE_PRESENTATION_MODE = "end_card"
VALID_CLOSURE_PRESENTATION_MODES = frozenset({"end_card", "final_overlay"})
VALID_CLOSURE_HEADLINE_SOURCES = frozenset(
    {"winner_development", "advertising_closure_role", "persisted", "approved_proposal", "creator_candidate"}
)
VALID_ADVERTISING_CLOSURE_STATUSES = frozenset(
    {"missing", "proposed", "approved", "rendering", "completed", "failed"}
)

GENERIC_SLOGAN_PATTERNS = (
    re.compile(r"^\s*quality\s+you\s+can\s+trust\s*$", re.I),
    re.compile(r"^\s*better\s+every\s+day\s*$", re.I),
    re.compile(r"^\s*the\s+best\s+choice\s*$", re.I),
    re.compile(r"^\s*experience\s+the\s+difference\s*$", re.I),
    re.compile(r"^\s*video\s+delivery\s*\.?\s*$", re.I),
    re.compile(r"^\s*made\s+for\s+you\s*$", re.I),
    re.compile(r"^\s*part\s+of\s+the\s+journey\s*$", re.I),
    re.compile(r"^\s*חלק\s+מהדרך\s*$"),
    re.compile(r"^\s*איכות\s+שאפשר\s+לסמוך\s+עליה\s*$"),
    re.compile(r"^\s*הבחירה\s+המושלמת\s*$"),
    re.compile(r"^\s*במיוחד\s+בשבילך\s*$"),
)

NEW_PROMISE_PATTERNS = (
    re.compile(r"\b(best|fastest|cheapest|number\s*one|#1)\b", re.I),
    re.compile(r"\b(guaranteed|always|never fails)\b", re.I),
)


def _raise(code: str, *, field: str) -> None:
    raise Builder2TournamentError(f"{code}:{field}")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def count_slogan_words_excluding_product(text: str, product_name: str) -> int:
    """Count whitespace-separated tokens; subtract one contiguous product-name match."""
    words = re.findall(r"\S+", _clean(text))
    product_words = re.findall(r"\S+", _clean(product_name))
    if not product_words:
        return len(words)
    lowered = [w.lower() for w in words]
    product_lower = [w.lower() for w in product_words]
    count = len(words)
    for idx in range(len(lowered) - len(product_lower) + 1):
        if lowered[idx : idx + len(product_lower)] == product_lower:
            count = max(0, count - len(product_lower))
            break
    return count


def _word_count_excluding_product(text: str, product_name: str) -> int:
    return count_slogan_words_excluding_product(text, product_name)


def build_slogan_word_limit_prompt_text(*, max_words: int = SLOGAN_MAX_WORD_COUNT) -> str:
    return (
        f"Single-slogan word limit (server-enforced): advertisingClosure.sloganText may contain at most "
        f"{max_words} words after excluding the plain product/brand identification.\n"
        "Count words exactly as the server does: split on whitespace into non-empty tokens (\\S+). "
        "Hyphens and punctuation stay attached to their token and do not split a word.\n"
        "advertisingClosure.productNameText is rendered separately as plain identification on the closure card; "
        "do NOT repeat the product/brand name inside sloganText.\n"
        "sloganText is the only advertising sentence — no in-video headline, no second slogan, no extra copy layer.\n"
        "Before returning JSON, count the final sloganText with the same rule and confirm it is within the limit. "
        "If it is too long, shorten the slogan itself while preserving its bridge to the central visible detail "
        "and to the relative advantage.\n"
        "Exceeding the limit invalidates the candidate."
    )


def normalize_advertising_closure(raw: Any) -> Dict[str, Any]:
    payload = dict(raw) if isinstance(raw, dict) else {}
    product_name = _clean(payload.get("productNameText"))
    slogan = _clean(payload.get("sloganText"))
    language = _clean(payload.get("language")).lower() or "en"
    presentation_mode = _clean(payload.get("presentationMode")) or DEFAULT_CLOSURE_PRESENTATION_MODE
    if presentation_mode not in VALID_CLOSURE_PRESENTATION_MODES:
        presentation_mode = DEFAULT_CLOSURE_PRESENTATION_MODE
    duration_raw = payload.get("durationSeconds", DEFAULT_CLOSURE_DURATION_SECONDS)
    try:
        duration_seconds = float(duration_raw)
    except (TypeError, ValueError):
        duration_seconds = DEFAULT_CLOSURE_DURATION_SECONDS
    if duration_seconds <= 0:
        duration_seconds = DEFAULT_CLOSURE_DURATION_SECONDS
    headline_source = _clean(payload.get("headlineSource")) or "persisted"
    if headline_source not in VALID_CLOSURE_HEADLINE_SOURCES:
        headline_source = "persisted"
    return {
        "required": bool(payload.get("required", True)),
        "productNameText": product_name,
        "sloganText": slogan,
        "language": language,
        "presentationMode": presentation_mode,
        "durationSeconds": duration_seconds,
        "headlineSource": headline_source,
        "noLogo": True,
    }


def advertising_closure_is_required(plan: Dict[str, Any]) -> bool:
    closure = plan.get("advertisingClosure")
    if isinstance(closure, dict) and closure.get("required") is False:
        return False
    return True


def validate_slogan_text_structure(
    *,
    slogan: str,
    product_name: str,
) -> None:
    text = _clean(slogan)
    if not text:
        _raise("builder2_advertising_closure_invalid", field="sloganText")
    word_count = _word_count_excluding_product(text, product_name)
    if word_count > SLOGAN_MAX_WORD_COUNT:
        _raise("builder2_advertising_closure_invalid", field="sloganText.word_limit")


def validate_slogan_text_quality(
    *,
    slogan: str,
    product_name: str,
    relative_advantage: str = "",
    core_mechanism: str = "",
) -> None:
    text = _clean(slogan)
    validate_slogan_text_structure(slogan=text, product_name=product_name)
    for pattern in GENERIC_SLOGAN_PATTERNS:
        if pattern.search(text):
            _raise("builder2_advertising_closure_invalid", field="sloganText.generic")
    for pattern in NEW_PROMISE_PATTERNS:
        if pattern.search(text):
            _raise("builder2_advertising_closure_invalid", field="sloganText.unsupported_claim")
    if core_mechanism and text.lower() == _clean(core_mechanism).lower():
        _raise("builder2_advertising_closure_invalid", field="sloganText.describes_action_only")
    if relative_advantage:
        from engine.builder2_advertising_slogan_quality_contract import (
            validate_slogan_advertising_quality_deterministic,
        )

        validate_slogan_advertising_quality_deterministic(
            slogan=text,
            product_name=product_name,
            relative_advantage=relative_advantage,
        )


def validate_slogan_text(
    *,
    slogan: str,
    product_name: str,
    relative_advantage: str = "",
    core_mechanism: str = "",
    quality_checks: bool = True,
) -> None:
    validate_slogan_text_structure(slogan=slogan, product_name=product_name)
    if quality_checks:
        validate_slogan_text_quality(
            slogan=slogan,
            product_name=product_name,
            relative_advantage=relative_advantage,
            core_mechanism=core_mechanism,
        )


def build_closure_from_winner_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    product_name = _clean(plan.get("productNameResolved"))
    language = _clean(plan.get("language")) or "en"
    decision = get_normalized_headline_decision(plan)
    headline_source = "winner_development"
    slogan = ""
    if headline_decision_requires_headline(decision):
        slogan = _clean(plan.get("headlineTextRemainder") or plan.get("headline") or plan.get("headlineText"))
        headline_source = "winner_development"
    existing = plan.get("advertisingClosure")
    if isinstance(existing, dict):
        normalized = normalize_advertising_closure(existing)
        if normalized.get("productNameText"):
            product_name = normalized["productNameText"]
        if normalized.get("sloganText"):
            slogan = normalized["sloganText"]
        if normalized.get("language"):
            language = normalized["language"]
        if normalized.get("headlineSource"):
            headline_source = normalized["headlineSource"]
    return normalize_advertising_closure(
        {
            "required": True,
            "productNameText": product_name,
            "sloganText": slogan,
            "language": language,
            "presentationMode": DEFAULT_CLOSURE_PRESENTATION_MODE,
            "durationSeconds": DEFAULT_CLOSURE_DURATION_SECONDS,
            "headlineSource": headline_source,
            "noLogo": True,
        }
    )


def validate_advertising_closure_object(
    closure: Dict[str, Any],
    *,
    plan: Optional[Dict[str, Any]] = None,
    structural_only: bool = False,
) -> None:
    normalized = normalize_advertising_closure(closure)
    if not normalized.get("required"):
        _raise("builder2_advertising_closure_invalid", field="required")
    if not _clean(normalized.get("productNameText")):
        _raise("builder2_advertising_closure_invalid", field="productNameText")
    relative_advantage = ""
    core_mechanism = ""
    if isinstance(plan, dict):
        adv = plan.get("relativeAdvantage")
        if isinstance(adv, dict):
            relative_advantage = _clean(adv.get("statement"))
        elif isinstance(adv, str):
            relative_advantage = _clean(adv)
        core_mechanism = _clean(plan.get("coreCreativeMechanism"))
    validate_slogan_text(
        slogan=_clean(normalized.get("sloganText")),
        product_name=_clean(normalized.get("productNameText")),
        relative_advantage=relative_advantage,
        core_mechanism=core_mechanism,
        quality_checks=not structural_only,
    )


def validate_advertising_closure_methodology(
    winner_plan: Dict[str, Any],
    *,
    require_present: bool = True,
) -> None:
    if headline_decision_is_omit(get_normalized_headline_decision(winner_plan)):
        closure = winner_plan.get("advertisingClosure")
        if not isinstance(closure, dict):
            if require_present:
                _raise("builder2_winner_validation_failed", field="advertisingClosure")
            return
        validate_advertising_closure_object(closure, plan=winner_plan, structural_only=True)
        winner_plan["advertisingClosure"] = normalize_advertising_closure(closure)
        return
    closure = winner_plan.get("advertisingClosure")
    if isinstance(closure, dict):
        validate_advertising_closure_object(closure, plan=winner_plan, structural_only=True)
        winner_plan["advertisingClosure"] = normalize_advertising_closure(closure)
    elif require_present:
        built = build_closure_from_winner_plan(winner_plan)
        if not _clean(built.get("sloganText")):
            _raise("builder2_winner_validation_failed", field="advertisingClosure.sloganText")
        validate_advertising_closure_object(built, plan=winner_plan)
        winner_plan["advertisingClosure"] = built


def validate_silent_visual_understanding(
    *,
    winner_plan: Dict[str, Any],
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> bool:
    if isinstance(winning_judgment, dict):
        silent_score = int((winning_judgment.get("scores") or {}).get("silentVisualClarity") or 0)
        if silent_score >= 10:
            return True
        headline = winning_judgment.get("headlineNecessityAssessment") or {}
        if headline.get("visualWouldWorkWithoutHeadline") is True:
            return True
    return bool(_clean(winner_plan.get("coreVisualIdea")) and _clean(winner_plan.get("videoPrompt")))


def validate_strategic_understanding(
    *,
    winner_plan: Dict[str, Any],
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> bool:
    adv = winner_plan.get("relativeAdvantage")
    statement = _clean(adv.get("statement") if isinstance(adv, dict) else adv)
    mechanism = _clean(winner_plan.get("coreCreativeMechanism"))
    if not statement or not mechanism:
        return False
    if isinstance(winning_judgment, dict):
        verbal = winning_judgment.get("verbalLayerAssessment") or {}
        if verbal.get("strategicMeaningIsClear") is True:
            return True
        score = int((winning_judgment.get("scores") or {}).get("problemAdvantageIntegrity") or 0)
        if score >= 12:
            return True
    return True


def validate_advertising_closure_delivery(
    *,
    winner_plan: Dict[str, Any],
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    status = get_advertising_closure_status(tournament_state or winner_plan)
    closure = winner_plan.get("advertisingClosure")
    if not isinstance(closure, dict):
        missing.append("winnerDevelopmentPlan.advertisingClosure")
    else:
        try:
            validate_advertising_closure_object(closure, plan=winner_plan)
        except Builder2TournamentError as exc:
            missing.append(str(exc.args[0] if exc.args else "advertisingClosure"))
    if status not in {"approved", "completed"}:
        missing.append("advertisingClosureStatus.approved_or_completed")
    media = (tournament_state or {}).get("mediaResume") if isinstance(tournament_state, dict) else {}
    if isinstance(media, dict):
        if status == "completed" and not _clean(media.get("finalVideoWithClosureUrl")):
            missing.append("mediaResume.finalVideoWithClosureUrl")
    return not missing, missing


def get_advertising_closure_status(state: Dict[str, Any]) -> str:
    status = _clean(state.get("advertisingClosureStatus")).lower()
    if status in VALID_ADVERTISING_CLOSURE_STATUSES:
        return status
    media = state.get("mediaResume")
    if isinstance(media, dict):
        status = _clean(media.get("advertisingClosureStatus")).lower()
        if status in VALID_ADVERTISING_CLOSURE_STATUSES:
            return status
    closure = state.get("advertisingClosure")
    if isinstance(closure, dict) and _clean(closure.get("sloganText")):
        return "proposed"
    return "missing"


def set_advertising_closure_status(state: Dict[str, Any], status: str) -> None:
    normalized = _clean(status).lower()
    if normalized not in VALID_ADVERTISING_CLOSURE_STATUSES:
        _raise("builder2_advertising_closure_invalid", field="advertisingClosureStatus")
    state["advertisingClosureStatus"] = normalized
    media = state.setdefault("mediaResume", {})
    if isinstance(media, dict):
        media["advertisingClosureStatus"] = normalized


def validate_judge_advertising_completion_assessment(judgment: Dict[str, Any]) -> None:
    assessment = judgment.get("advertisingCompletionAssessment")
    if not isinstance(assessment, dict):
        _raise("builder2_judge_validation_failed", field="advertisingCompletionAssessment")
    required_bools = (
        "advertiserIdentifiable",
        "productNamePresent",
        "relativeAdvantageClosed",
        "sloganSpecificToIdea",
        "functionsAsAdvertisement",
    )
    for key in required_bools:
        if not isinstance(assessment.get(key), bool):
            _raise("builder2_judge_validation_failed", field=f"advertisingCompletionAssessment.{key}")
    if not str(assessment.get("notes") or "").strip():
        _raise("builder2_judge_validation_failed", field="advertisingCompletionAssessment.notes")
    if judgment.get("eligible") is True:
        if not all(assessment.get(key) is True for key in required_bools):
            _raise("builder2_judge_coherence_violation", field="advertisingCompletionAssessment.eligible_without_completion")


def judge_advertising_completion_passes(judgment: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(judgment, dict):
        return False
    assessment = judgment.get("advertisingCompletionAssessment")
    if not isinstance(assessment, dict):
        return False
    return all(
        assessment.get(key) is True
        for key in (
            "advertiserIdentifiable",
            "productNamePresent",
            "relativeAdvantageClosed",
            "sloganSpecificToIdea",
            "functionsAsAdvertisement",
        )
    )


def headline_decision_allows_runway_scene_text(plan: Dict[str, Any]) -> bool:
    return headline_decision_requires_headline(get_normalized_headline_decision(plan))
