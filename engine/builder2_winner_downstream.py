"""
Builder2 winner-plan downstream adapter — type-safe extraction for Runway, start image, overlay.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION
from engine.video_planning import (
    _headline_remainder_word_count,
    _MAX_HEADLINE_REMAINDER_WORDS,
)

logger = logging.getLogger(__name__)

DOWNSTREAM_COMPATIBILITY_VERSION = "builder2_tournament_winner_v1"


class Builder2WinnerDownstreamError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def is_builder2_tournament_plan(plan: Dict[str, Any]) -> bool:
    return str(plan.get("planInferenceMode") or "").startswith("builder2_tournament")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _invalid(field_path: str) -> None:
    raise Builder2WinnerDownstreamError(f"builder2_winner_downstream_invalid:{field_path}")


def _headline_invalid(field_path: str) -> None:
    raise Builder2WinnerDownstreamError(f"builder2_headline_composition_invalid:{field_path}")


def _pre_runway_invalid(field_path: str) -> None:
    raise Builder2WinnerDownstreamError(f"builder2_pre_runway_validation_failed:{field_path}")


def _require_non_empty_text(value: Any, field_path: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    _invalid(field_path)


def _optional_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def get_headline_decision(plan: Dict[str, Any]) -> str:
    return get_normalized_headline_decision(plan)


def get_visual_anchor_description(plan: Dict[str, Any]) -> str:
    anchor = plan.get("visualAnchor")
    if isinstance(anchor, str):
        return _require_non_empty_text(anchor, "visualAnchor")
    if isinstance(anchor, dict):
        return _require_non_empty_text(anchor.get("description"), "visualAnchor.description")
    _invalid("visualAnchor")


def get_visual_anchor_why_essential(plan: Dict[str, Any]) -> str:
    anchor = plan.get("visualAnchor")
    if isinstance(anchor, str):
        return ""
    if isinstance(anchor, dict):
        return _optional_text(anchor.get("whyEssential"))
    _invalid("visualAnchor")


def get_sequence_beginning(plan: Dict[str, Any]) -> str:
    sequence = plan.get("sequence")
    if not isinstance(sequence, dict):
        _invalid("sequence")
    return _require_non_empty_text(sequence.get("beginning"), "sequence.beginning")


def get_sequence_development(plan: Dict[str, Any]) -> str:
    sequence = plan.get("sequence")
    if not isinstance(sequence, dict):
        _invalid("sequence")
    return _require_non_empty_text(sequence.get("development"), "sequence.development")


def get_sequence_resolution(plan: Dict[str, Any]) -> str:
    sequence = plan.get("sequence")
    if not isinstance(sequence, dict):
        _invalid("sequence")
    return _require_non_empty_text(sequence.get("resolution"), "sequence.resolution")


def get_visual_family_definition(plan: Dict[str, Any]) -> str:
    family = plan.get("visualFamily")
    if isinstance(family, str):
        return _require_non_empty_text(family, "visualFamily")
    if isinstance(family, dict):
        return _require_non_empty_text(family.get("familyDefinition"), "visualFamily.familyDefinition")
    _invalid("visualFamily")


def get_opening_frame_description(plan: Dict[str, Any]) -> str:
    opening = _optional_text(plan.get("openingFrameDescription"))
    if opening:
        return opening
    feasibility = plan.get("runwayFeasibility")
    if isinstance(feasibility, dict):
        opening = _optional_text(feasibility.get("openingFrame"))
        if opening:
            return opening
    if (plan.get("structureType") or "").strip() == "continuous_event":
        try:
            return get_sequence_beginning(plan)
        except Builder2WinnerDownstreamError:
            pass
    _invalid("openingFrameDescription")


def get_runway_main_subject(plan: Dict[str, Any]) -> str:
    feasibility = plan.get("runwayFeasibility")
    if isinstance(feasibility, dict):
        subject = _optional_text(feasibility.get("mainSubject"))
        if subject:
            return subject
    return ""


def get_runway_main_action(plan: Dict[str, Any]) -> str:
    feasibility = plan.get("runwayFeasibility")
    if isinstance(feasibility, dict):
        action = _optional_text(feasibility.get("mainAction"))
        if action:
            return action
    return ""


def get_runway_location(plan: Dict[str, Any]) -> str:
    feasibility = plan.get("runwayFeasibility")
    if isinstance(feasibility, dict):
        location = _optional_text(feasibility.get("location"))
        if location:
            return location
    return ""


def get_scene_variation_descriptions(plan: Dict[str, Any]) -> List[str]:
    raw = plan.get("sceneVariations")
    if raw is None:
        return []
    if not isinstance(raw, list):
        _invalid("sceneVariations")
    out: List[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            text = _optional_text(item.get("description"))
            if text:
                out.append(text)
                continue
            _invalid(f"sceneVariations[{index}].description")
        if item is not None and str(item).strip():
            _invalid(f"sceneVariations[{index}]")
    return out


def get_recurring_motif(plan: Dict[str, Any]) -> str:
    consistency = plan.get("visualFamilyConsistency")
    if isinstance(consistency, dict):
        motif = _optional_text(consistency.get("recurringMotif"))
        if motif:
            return motif
    family = plan.get("visualFamily")
    if isinstance(family, dict):
        motif = _optional_text(family.get("recurringMotif"))
        if motif:
            return motif
    return get_visual_family_definition(plan)


def _normalize_product_token(text: str) -> str:
    return re.sub(r"[\s.,!?;:\"'()\[\]]+", "", _nfc(text).casefold())


def _remainder_starts_with_product_name(product_name: str, remainder: str) -> bool:
    pn = _nfc((product_name or "").strip())
    rem = _nfc((remainder or "").strip())
    if not pn or not rem:
        return False
    if rem.startswith(pn):
        tail = rem[len(pn) :].lstrip(" .,:;!?-–—")
        return True if tail or rem == pn else True
    pn_norm = _normalize_product_token(pn)
    rem_norm = _normalize_product_token(rem)
    if pn_norm and rem_norm.startswith(pn_norm):
        return True
    first_token = rem.split(maxsplit=1)[0] if rem.split() else rem
    if _normalize_product_token(first_token) == pn_norm:
        return True
    return False


def _strip_leading_product_name(product_name: str, remainder: str) -> str:
    pn = _nfc((product_name or "").strip())
    rem = _nfc((remainder or "").strip())
    if not pn or not rem:
        return rem
    if rem.startswith(pn):
        return rem[len(pn) :].lstrip(" .,:;!?-–—")
    tokens = rem.split()
    if tokens and _normalize_product_token(tokens[0]) == _normalize_product_token(pn):
        return " ".join(tokens[1:]).lstrip(" .,:;!?-–—")
    return rem


def compose_builder2_headline_text(
    product_name: str,
    headline_remainder: str,
) -> Tuple[str, str]:
    """
    Authoritative headline assembler.
    Returns (headlineText, headlineTextRemainder) with product name included exactly once.
    """
    pn = _nfc((product_name or "").strip())
    rem = " ".join(_nfc(headline_remainder or "").split())
    if not pn:
        if not rem:
            _headline_invalid("empty")
        return rem, rem
    if not rem:
        return pn, ""
    cleaned_rem = rem
    if _remainder_starts_with_product_name(pn, rem):
        cleaned_rem = _strip_leading_product_name(pn, rem)
        cleaned_rem = " ".join(cleaned_rem.split())
    if not cleaned_rem:
        return pn, ""
    headline_text = f"{pn} {cleaned_rem}".strip()
    if _headline_remainder_word_count(cleaned_rem) > _MAX_HEADLINE_REMAINDER_WORDS:
        _headline_invalid("remainder_word_count")
    assembled_twice = f"{pn} {pn}"
    if headline_text.startswith(assembled_twice) or _remainder_starts_with_product_name(pn, headline_text[len(pn) + 1 :]):
        _headline_invalid("product_name_duplicated")
    return headline_text, cleaned_rem


def apply_builder2_headline_composition(plan: Dict[str, Any]) -> None:
    decision = get_headline_decision(plan)
    if not headline_decision_requires_headline(decision):
        plan["headline"] = ""
        plan["headlineText"] = ""
        plan["headlineTextRemainder"] = ""
        plan["headlineCoreKeyword"] = ""
        plan["advertisingPromise"] = ""
        logger.info("BUILDER2_HEADLINE_COMPOSITION_OK decision=omit")
        return
    pn = _require_non_empty_text(plan.get("productNameResolved"), "productNameResolved")
    remainder = _require_non_empty_text(plan.get("headline"), "headline")
    headline_text, cleaned_rem = compose_builder2_headline_text(pn, remainder)
    plan["headline"] = cleaned_rem
    plan["headlineText"] = headline_text
    plan["headlineTextRemainder"] = cleaned_rem
    plan["advertisingPromise"] = cleaned_rem
    logger.info("BUILDER2_HEADLINE_COMPOSITION_OK decision=use")


def ensure_builder2_schema_metadata(plan: Dict[str, Any], *, compatibility_mode: bool = False) -> None:
    plan["schemaVersion"] = WINNER_PLAN_SCHEMA_VERSION
    if plan.get("methodologyVersion"):
        plan["methodologyVersion"] = plan.get("methodologyVersion")
    elif compatibility_mode:
        plan.setdefault("methodologyVersion", "")
    else:
        plan["methodologyVersion"] = METHODOLOGY_VERSION
    plan["downstreamCompatibilityVersion"] = DOWNSTREAM_COMPATIBILITY_VERSION
    logger.info(
        "BUILDER2_WINNER_SCHEMA_METADATA schemaVersion=%s methodologyVersion=%s structureType=%s compatibilityVersion=%s",
        plan.get("schemaVersion"),
        plan.get("methodologyVersion") or "(none)",
        plan.get("structureType") or "(none)",
        plan.get("downstreamCompatibilityVersion"),
    )


def log_builder2_winner_schema_metadata(plan: Dict[str, Any]) -> None:
    logger.info(
        "BUILDER2_WINNER_SCHEMA_METADATA schemaVersion=%s methodologyVersion=%s structureType=%s sceneSequenceSemantics=%s compatibilityVersion=%s",
        plan.get("schemaVersion") or WINNER_PLAN_SCHEMA_VERSION,
        plan.get("methodologyVersion") or "(none)",
        plan.get("structureType") or "(none)",
        plan.get("sceneSequenceSemantics") or "(none)",
        plan.get("downstreamCompatibilityVersion") or DOWNSTREAM_COMPATIBILITY_VERSION,
    )


def normalize_builder2_winner_downstream(
    plan: Dict[str, Any],
    *,
    job_id: str = "",
    tournament_id: str = "",
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    logger.info(
        "BUILDER2_WINNER_DOWNSTREAM_NORMALIZATION_START jobId=%s tournamentId=%s structureType=%s",
        job_id or "(none)",
        tournament_id or "(none)",
        plan.get("structureType") or "(none)",
    )
    out = dict(plan)
    try:
        get_visual_anchor_description(out)
        logger.info("BUILDER2_VISUAL_ANCHOR_NORMALIZED structureType=%s", out.get("structureType"))
        apply_builder2_headline_composition(out)
        ensure_builder2_schema_metadata(out, compatibility_mode=compatibility_mode)
        out["sceneVariations"] = get_scene_variation_descriptions(out)
        structure = (out.get("structureType") or "").strip()
        if structure == "continuous_event":
            out["sceneSequenceSemantics"] = "temporal_beats"
        elif structure == "variation_montage":
            out["sceneSequenceSemantics"] = out.get("sceneSequenceSemantics") or "montage_variations"
        validate_builder2_pre_runway(out)
    except Builder2WinnerDownstreamError as exc:
        logger.error(
            "BUILDER2_WINNER_DOWNSTREAM_NORMALIZATION_FAILED jobId=%s code=%s",
            job_id or "(none)",
            exc.code,
        )
        raise
    logger.info(
        "BUILDER2_WINNER_DOWNSTREAM_NORMALIZATION_OK jobId=%s tournamentId=%s headlineDecision=%s compatibilityMode=%s",
        job_id or "(none)",
        tournament_id or "(none)",
        get_headline_decision(out),
        str(compatibility_mode).lower(),
    )
    return out


def validate_builder2_winner_headline_composition_pure(plan: Dict[str, Any]) -> None:
    """
    Pure Builder2 headline composition and keyword validation — no media side effects.
    """
    apply_builder2_headline_composition(plan)
    decision = get_headline_decision(plan)
    if not headline_decision_requires_headline(decision):
        return
    headline_rem = (plan.get("headline") or "").strip()
    core_kw = (plan.get("headlineCoreKeyword") or "").strip()
    if not headline_rem or not core_kw:
        _headline_invalid("headline_or_keyword")
    from engine.video_planning import (
        _headline_contains_core_keyword,
        _headline_depends_on_fixed_phrase,
        _headline_remainder_word_count,
        _is_weak_industry_keyword,
        _keyword_depends_on_headline_phrase,
        _MAX_HEADLINE_REMAINDER_WORDS,
    )

    if _headline_remainder_word_count(headline_rem) > _MAX_HEADLINE_REMAINDER_WORDS:
        _headline_invalid("remainder_word_count")
    if _headline_depends_on_fixed_phrase(headline_rem):
        _headline_invalid("phrase_dependent_headline")
    if len(core_kw.split()) != 1 or _is_weak_industry_keyword(core_kw):
        _headline_invalid("invalid_keyword")
    if _keyword_depends_on_headline_phrase(headline_rem, core_kw):
        _headline_invalid("phrase_dependent_keyword")
    if not _headline_contains_core_keyword(headline_rem, core_kw):
        _headline_invalid("keyword_not_in_headline")


@dataclass(frozen=True)
class AcceptedWinnerHeadlineResolution:
    headline_required: bool
    headline_text: str = ""
    headline_text_remainder: str = ""
    product_name_resolved: str = ""
    language: str = "en"
    canonical_headline_source: str = ""
    accepted_headline_decision: str = ""
    accepted_headline_field_present: bool = False
    accepted_headline_keyword_present: bool = False
    persisted_headline_text_present: bool = False
    canonical_headline_resolution_attempted: bool = False
    canonical_headline_resolution_accepted: bool = False
    canonical_headline_character_count: int = 0
    canonical_headline_word_count: int = 0
    failure_code: str = ""
    failure_stage: str = ""


def apply_accepted_headline_resolution_observability(
    report: Dict[str, Any],
    resolution: AcceptedWinnerHeadlineResolution,
) -> None:
    report["acceptedHeadlineDecision"] = resolution.accepted_headline_decision or None
    report["acceptedHeadlineFieldPresent"] = resolution.accepted_headline_field_present
    report["acceptedHeadlineKeywordPresent"] = resolution.accepted_headline_keyword_present
    report["persistedHeadlineTextPresent"] = resolution.persisted_headline_text_present
    report["canonicalHeadlineResolutionAttempted"] = resolution.canonical_headline_resolution_attempted
    report["canonicalHeadlineResolutionAccepted"] = resolution.canonical_headline_resolution_accepted
    report["canonicalHeadlineSource"] = resolution.canonical_headline_source or None
    report["canonicalHeadlineCharacterCount"] = resolution.canonical_headline_character_count
    report["canonicalHeadlineWordCount"] = resolution.canonical_headline_word_count
    report["localHeadlineInputPresent"] = bool(resolution.headline_text)
    if resolution.failure_code:
        report["localHeadlineFailureStage"] = resolution.failure_stage or "canonical_headline_resolution"
        report["localHeadlineFailureCode"] = resolution.failure_code


def resolve_accepted_winner_headline_for_media(plan: Dict[str, Any]) -> AcceptedWinnerHeadlineResolution:
    """
    Resolve the accepted Winner headline for media/finalization without mutating persisted state.
    Uses the same canonical composition helpers as normalize_builder2_winner_downstream().
    """
    work = dict(plan)
    decision = get_headline_decision(work)
    field_present = bool(_optional_text(work.get("headline")))
    keyword_present = bool(_optional_text(work.get("headlineCoreKeyword")))
    persisted_present = bool(_optional_text(work.get("headlineText")))
    language = str(work.get("language") or "en")
    base = AcceptedWinnerHeadlineResolution(
        headline_required=headline_decision_requires_headline(decision),
        accepted_headline_decision=decision,
        accepted_headline_field_present=field_present,
        accepted_headline_keyword_present=keyword_present,
        persisted_headline_text_present=persisted_present,
        language=language,
        product_name_resolved=str(work.get("productNameResolved") or "").strip(),
    )
    if not headline_decision_requires_headline(decision):
        return replace(base, canonical_headline_source="omitted_by_decision")

    attempted = replace(base, canonical_headline_resolution_attempted=True)
    if not field_present:
        return replace(
            attempted,
            failure_code="accepted_headline_missing",
            failure_stage="canonical_headline_resolution",
        )
    if not keyword_present:
        return replace(
            attempted,
            failure_code="accepted_keyword_missing",
            failure_stage="canonical_headline_resolution",
        )
    if not str(work.get("productNameResolved") or "").strip():
        return replace(
            attempted,
            failure_code="accepted_headline_missing",
            failure_stage="canonical_headline_resolution",
        )

    try:
        persisted_text = _optional_text(work.get("headlineText"))
        source = "derived_from_accepted_winner"
        if persisted_text:
            pn = _require_non_empty_text(work.get("productNameResolved"), "productNameResolved")
            rem_source = _optional_text(work.get("headlineTextRemainder")) or _optional_text(work.get("headline"))
            composed, cleaned = compose_builder2_headline_text(pn, rem_source or _optional_text(work.get("headline")))
            if composed == persisted_text:
                verify = dict(work)
                verify["headlineText"] = persisted_text
                verify["headlineTextRemainder"] = cleaned or _optional_text(verify.get("headlineTextRemainder"))
                validate_builder2_winner_headline_composition_pure(verify)
                headline_text = persisted_text
                headline_rem = str(verify.get("headlineTextRemainder") or cleaned or "").strip()
                source = "persisted_verified"
            else:
                composed_plan = dict(work)
                apply_builder2_headline_composition(composed_plan)
                validate_builder2_winner_headline_composition_pure(composed_plan)
                headline_text = str(composed_plan.get("headlineText") or "").strip()
                headline_rem = str(composed_plan.get("headlineTextRemainder") or "").strip()
        else:
            composed_plan = dict(work)
            apply_builder2_headline_composition(composed_plan)
            validate_builder2_winner_headline_composition_pure(composed_plan)
            headline_text = str(composed_plan.get("headlineText") or "").strip()
            headline_rem = str(composed_plan.get("headlineTextRemainder") or "").strip()

        if not headline_text:
            return replace(
                attempted,
                failure_code="canonical_headline_composition_failed",
                failure_stage="canonical_headline_resolution",
            )
        return AcceptedWinnerHeadlineResolution(
            headline_required=True,
            headline_text=headline_text,
            headline_text_remainder=headline_rem,
            product_name_resolved=str(work.get("productNameResolved") or "").strip(),
            language=language,
            canonical_headline_source=source,
            accepted_headline_decision=decision,
            accepted_headline_field_present=field_present,
            accepted_headline_keyword_present=keyword_present,
            persisted_headline_text_present=persisted_present,
            canonical_headline_resolution_attempted=True,
            canonical_headline_resolution_accepted=True,
            canonical_headline_character_count=len(headline_text),
            canonical_headline_word_count=len(headline_text.split()),
        )
    except Builder2WinnerDownstreamError:
        return replace(
            attempted,
            failure_code="canonical_headline_composition_failed",
            failure_stage="canonical_headline_resolution",
        )


def validate_builder2_pre_runway(plan: Dict[str, Any]) -> None:
    structure = (plan.get("structureType") or "").strip()
    if structure not in {"continuous_event", "variation_montage"}:
        _pre_runway_invalid("structureType")

    get_opening_frame_description(plan)
    _require_non_empty_text(plan.get("coreVisualIdea"), "coreVisualIdea")
    _require_non_empty_text(plan.get("videoPrompt") or plan.get("videoPromptCore"), "videoPrompt")

    if structure == "continuous_event":
        get_sequence_beginning(plan)
        get_sequence_development(plan)
        get_sequence_resolution(plan)
        get_visual_anchor_description(plan)
        semantics = (plan.get("sceneSequenceSemantics") or "").strip()
        if semantics != "temporal_beats":
            _pre_runway_invalid("sceneSequenceSemantics")
    elif structure == "variation_montage":
        variations = get_scene_variation_descriptions(plan)
        if len(variations) < 2 or len(variations) > 4:
            _pre_runway_invalid("sceneVariations")
        get_visual_family_definition(plan)
        get_recurring_motif(plan)

    decision = get_headline_decision(plan)
    if headline_decision_requires_headline(decision):
        _require_non_empty_text(plan.get("headlineText"), "headlineText")
        _require_non_empty_text(plan.get("headlineTextRemainder"), "headlineTextRemainder")
        pn = _require_non_empty_text(plan.get("productNameResolved"), "productNameResolved")
        rem = _require_non_empty_text(plan.get("headlineTextRemainder"), "headlineTextRemainder")
        _, cleaned = compose_builder2_headline_text(pn, rem)
        if cleaned != rem:
            _pre_runway_invalid("headlineTextRemainder")
        if _remainder_starts_with_product_name(pn, rem):
            _pre_runway_invalid("headlineTextRemainder")
    elif decision == "omit":
        if _optional_text(plan.get("headlineText")) or _optional_text(plan.get("headline")):
            _pre_runway_invalid("headlineDecision.omit_with_headline")
    else:
        _pre_runway_invalid("headlineDecision")

    logger.info(
        "BUILDER2_PRE_RUNWAY_VALIDATION_OK structureType=%s headlineDecision=%s",
        structure,
        decision,
    )


def build_continuous_event_runway_prompt(plan: Dict[str, Any], *, duration_seconds: int) -> str:
    from engine.video_planning import (
        RUNWAY_PHYSICS_REALISM_CONSTRAINT,
        _RUNWAY_STYLE_TAIL,
        _finalize_runway_prompt,
        _runway_language_visual_constraints,
    )

    scene_prompt = _require_non_empty_text(plan.get("videoPrompt") or plan.get("videoPromptCore"), "videoPrompt")
    lang_vis = _runway_language_visual_constraints(plan)
    opening = get_opening_frame_description(plan)
    beginning = get_sequence_beginning(plan)
    development = get_sequence_development(plan)
    resolution = get_sequence_resolution(plan)
    anchor = get_visual_anchor_description(plan)
    subject = get_runway_main_subject(plan)
    action = get_runway_main_action(plan)
    location = get_runway_location(plan)

    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"MANDATORY: one continuous {duration_seconds}-second realistic event in a single location with one primary action. "
        "Natural pacing from opening physical state through development to a clear visual resolution; "
        "no montage, no multiple clips, no unrelated cuts, no dead seconds at the end. "
        f"Opening state: {opening}. "
        f"Beginning: {beginning}. "
    )
    if subject:
        body += f"Main subject: {subject}. "
    if action:
        body += f"Main action: {action}. "
    if location:
        body += f"Location: {location}. "
    if anchor:
        body += f"Visual anchor: {anchor}. "
    body += (
        f"Development: {development}. "
        f"Resolution: {resolution}. "
        f"Continuous event (follow exactly): {scene_prompt}. {_RUNWAY_STYLE_TAIL}"
    )
    out, _ = _finalize_runway_prompt("", body)
    if not out.strip():
        _pre_runway_invalid("runwayPrompt")
    logger.info("RUNWAY_PROMPT path=continuous_event")
    return out


def build_variation_montage_runway_prompt(plan: Dict[str, Any]) -> str:
    from engine.video_planning import (
        _RUNWAY_STYLE_TAIL,
        _finalize_runway_prompt,
        _runway_language_visual_constraints,
        _runway_variation_montage_camera_focus,
    )

    scene_prompt = _require_non_empty_text(plan.get("videoPrompt") or plan.get("videoPromptCore"), "videoPrompt")
    motion, _ = _runway_variation_montage_camera_focus()
    lang_vis = _runway_language_visual_constraints(plan)
    core_visual = _require_non_empty_text(plan.get("coreVisualIdea"), "coreVisualIdea")
    variations = get_scene_variation_descriptions(plan)
    family = get_visual_family_definition(plan)
    motif = get_recurring_motif(plan)
    numbered = "; ".join(f"({i + 1}) {v}" for i, v in enumerate(variations))
    montage_body = (
        f"Core visual idea: {core_visual}. "
        f"Visual family: {family}. "
        f"Recurring motif: {motif}. "
        f"Variation moments: {numbered}. "
        f"Montage direction: {scene_prompt}"
    )
    body = (
        "VISUAL POLICY: No readable text, letters, words, captions, labels, signage, packaging typography, "
        "title cards, watermarks, or brand names in-frame; purely pictorial motion. "
        f"{lang_vis} "
        f"{motion} "
        f"Montage (follow exactly): {montage_body}. "
        f"{_RUNWAY_STYLE_TAIL}"
    )
    out, _ = _finalize_runway_prompt("", body)
    if not out.strip():
        _pre_runway_invalid("runwayPrompt")
    logger.info("RUNWAY_PROMPT path=variation_montage")
    return out


def build_builder2_start_frame_image_prompt(plan: Dict[str, Any], *, duration_seconds: int) -> str:
    no_text = (
        "No text, letters, words, numbers as graphics, captions, labels, signage, packaging typography, "
        "title cards, watermarks, headline, UI, or brand names in the image — blank/generic surfaces only."
    )
    product = _optional_text(plan.get("productNameResolved"))
    core_visual = _require_non_empty_text(plan.get("coreVisualIdea"), "coreVisualIdea")
    opening = get_opening_frame_description(plan)
    beginning = ""
    if (plan.get("structureType") or "").strip() == "continuous_event":
        beginning = get_sequence_beginning(plan)
    anchor = ""
    try:
        anchor = get_visual_anchor_description(plan)
    except Builder2WinnerDownstreamError:
        anchor = ""
    resolution = ""
    try:
        resolution = get_sequence_resolution(plan)
    except Builder2WinnerDownstreamError:
        resolution = ""

    scene_focus = opening or beginning or core_visual
    if resolution and resolution.lower() in scene_focus.lower() and beginning:
        scene_focus = beginning

    product_clause = f"Product context (do not show as readable text): {product}. " if product else ""
    is_continuous = (plan.get("structureType") or "").strip() == "continuous_event"
    shot_kind = "continuous event" if is_continuous else "commercial montage"
    anchor_clause = f"Visible anchor at opening when present: {anchor}. " if anchor and anchor.lower() in scene_focus.lower() else ""
    safe_area = (
        "Compose the scene for a central 16:9 safe area. "
        "Keep the subject, action, and visual anchor away from the extreme top and bottom edges. "
        "Do not place critical objects or text in the crop margins. "
    )
    brief = (
        f"Single photorealistic still frame, opening shot for a silent {duration_seconds}-second {shot_kind}. "
        f"The still must be the opening moment from which action can develop naturally over {duration_seconds} seconds — "
        "not the final resolution. "
        f"{product_clause}"
        f"Core visual idea: {core_visual}. "
        f"Opening moment to animate: {scene_focus}. "
        f"{anchor_clause}"
        f"{safe_area}"
        "Realistic human scene when applicable; clear composition; soft natural lighting; realistic materials. "
        f"{no_text}"
    )
    return brief
