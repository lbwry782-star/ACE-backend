"""
Builder2 single-slogan copy contract — one advertising sentence on the closure card only.

Legacy jobs may retain headlineText + closure slogan fields for read compatibility.
New jobs use copyContractVersion=builder2_single_slogan_v1.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_advertising_closure_contract import normalize_advertising_closure
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
    normalize_headline_decision_object,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION = "builder2_single_slogan_v1"

SLOGAN_DECISION_USE = "use"
SLOGAN_DECISION_OMIT = "omit"
VALID_SLOGAN_DECISIONS = frozenset({SLOGAN_DECISION_USE, SLOGAN_DECISION_OMIT})

LITERAL_DOMAIN_SYMBOL_PATTERNS = (
    re.compile(r"\b(graph|chart|dashboard|report|spreadsheet|crm|interface|screen)\b", re.I),
    re.compile(r"\b(form|counter|metric|kpi|arrow.*growth|growth arrow)\b", re.I),
    re.compile(r"\b(lead report|printed report|numerical counter)\b", re.I),
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _word_count(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def copy_contract_version(*, state: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(state, dict):
        version = _clean(state.get("copyContractVersion"))
        if version:
            return version
    if isinstance(plan, dict):
        return _clean(plan.get("copyContractVersion"))
    return ""


def is_single_slogan_contract(*, state: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> bool:
    version = copy_contract_version(state=state, plan=plan)
    if version == BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION:
        return True
    from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION

    if isinstance(plan, dict) and _clean(plan.get("builder2NewFormatVersion")) == BUILDER2_NEW_FORMAT_VERSION:
        return True
    if isinstance(state, dict) and _clean(state.get("builder2NewFormatVersion")) == BUILDER2_NEW_FORMAT_VERSION:
        return True
    return False


def stamp_single_slogan_contract(state: Dict[str, Any]) -> None:
    state["copyContractVersion"] = BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION
    from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION

    state.setdefault("builder2NewFormatVersion", BUILDER2_NEW_FORMAT_VERSION)


def builder2_requires_headline_overlay(*, plan: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    if is_single_slogan_contract(state=state, plan=plan):
        return False
    return headline_decision_requires_headline(get_normalized_headline_decision(plan))


def single_slogan_forces_headline_omit(*, plan: Optional[Dict[str, Any]] = None, state: Optional[Dict[str, Any]] = None) -> bool:
    return is_single_slogan_contract(state=state, plan=plan)


def resolve_canonical_slogan_text(*, plan: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
    explicit = _clean(plan.get("sloganText"))
    if explicit:
        return explicit
    closure = normalize_advertising_closure(plan.get("advertisingClosure"))
    closure_text = _clean(closure.get("sloganText"))
    if closure_text:
        return closure_text
    if isinstance(state, dict):
        state_closure = normalize_advertising_closure(state.get("advertisingClosure"))
        state_text = _clean(state_closure.get("sloganText"))
        if state_text:
            return state_text
    if not is_single_slogan_contract(state=state, plan=plan):
        legacy = normalize_legacy_dual_copy(plan=plan, state=state)
        return _clean(legacy.get("canonicalSloganText"))
    return ""


def normalize_legacy_dual_copy(
    *,
    plan: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    closure = normalize_advertising_closure(plan.get("advertisingClosure"))
    if not closure and isinstance(state, dict):
        closure = normalize_advertising_closure(state.get("advertisingClosure"))
    closure_slogan = _clean(closure.get("sloganText"))
    headline_text = _clean(plan.get("headlineText") or plan.get("headline"))
    source = ""
    canonical = ""
    if closure_slogan:
        canonical = closure_slogan
        source = "advertising_closure_slogan"
    elif headline_text:
        canonical = headline_text
        source = "legacy_headline_text"
    return {
        "canonicalSloganText": canonical,
        "legacyCopyNormalized": bool(closure_slogan and headline_text and closure_slogan != headline_text),
        "legacyCopySource": source,
        "legacyHeadlineText": headline_text,
        "legacyClosureSlogan": closure_slogan,
    }


def apply_single_slogan_winner_normalization(
    winner_plan: Dict[str, Any],
    *,
    winning_candidate: Dict[str, Any],
    winning_judgment: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    if not is_single_slogan_contract(state=state, plan=winner_plan):
        return

    winner_plan["copyContractVersion"] = BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION
    creator_closure = normalize_advertising_closure((winning_candidate or {}).get("advertisingClosure"))
    winner_plan["advertisingClosure"] = creator_closure

    canonical = _clean(creator_closure.get("sloganText"))
    if not canonical:
        bridge = winning_candidate.get("singleSloganContract") if isinstance(winning_candidate.get("singleSloganContract"), dict) else {}
        metaphor = winning_candidate.get("metaphoricalEmbodiment") if isinstance(winning_candidate.get("metaphoricalEmbodiment"), dict) else {}
        canonical = _clean(
            bridge.get("sloganBridgeToBusinessMeaning")
            or bridge.get("canonicalSloganBridge")
            or metaphor.get("sloganBridgeToBusinessMeaning")
            or metaphor.get("canonicalSloganBridge")
            or bridge.get("sloganText")
        )

    verbal = winning_candidate.get("verbalPotential") if isinstance(winning_candidate.get("verbalPotential"), dict) else {}
    core_keyword = _clean(verbal.get("keywordOrKeyPhrase") or winner_plan.get("sloganCoreKeyword"))

    winner_plan["sloganDecision"] = SLOGAN_DECISION_USE if canonical else SLOGAN_DECISION_OMIT
    winner_plan["sloganText"] = canonical
    winner_plan["sloganCoreKeyword"] = core_keyword
    winner_plan["sloganSource"] = "creator_candidate_closure"
    winner_plan["sloganVisualBridge"] = _extract_visual_bridge(winning_candidate, winning_judgment)
    winner_plan["sloganUnderstandsWithoutPriorCopy"] = True
    winner_plan["sloganRenderedExactlyOnce"] = False

    bridge = winning_candidate.get("visualBridgeAssessment") if isinstance(winning_candidate.get("visualBridgeAssessment"), dict) else {}
    if bridge:
        winner_plan["centralVisibleDetail"] = _clean(bridge.get("centralVisibleDetail"))
        winner_plan["sloganConnectionToVisibleDetail"] = _clean(bridge.get("sloganConnectionToVisibleDetail"))
        winner_plan["sloganConnectionToRelativeAdvantage"] = _clean(bridge.get("sloganConnectionToRelativeAdvantage"))
        winner_plan["dependsOnEarlierCopy"] = bridge.get("dependsOnEarlierCopy") is True

    creator_closure["sloganText"] = canonical
    winner_plan["advertisingClosure"] = creator_closure

    omit_decision = normalize_headline_decision_object(
        {"decision": "omit", "reason": "single_slogan_contract", "reasonSource": "server_derived"},
        winning_judgment=winning_judgment,
    )
    winner_plan["headlineDecision"] = omit_decision
    winner_plan["headlineForm"] = "none"
    winner_plan["headline"] = ""
    winner_plan["headlineText"] = canonical
    winner_plan["headlineTextRemainder"] = canonical
    winner_plan["headlineCoreKeyword"] = core_keyword
    winner_plan["headlineCompatibilityAlias"] = True
    winner_plan["headlineOverlaySkipped"] = True


def _extract_visual_bridge(
    winning_candidate: Dict[str, Any],
    winning_judgment: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    bridge = winning_candidate.get("visualBridgeAssessment")
    if isinstance(bridge, dict) and bridge:
        return bridge
    if isinstance(winning_judgment, dict):
        assessment = winning_judgment.get("visualBridgeAssessment")
        if isinstance(assessment, dict):
            return assessment
        metaphor = winning_judgment.get("metaphoricalEmbodimentAssessment")
        if isinstance(metaphor, dict):
            return {
                "centralVisibleDetail": _clean(metaphor.get("embodimentSubjectOrWorld")),
                "sloganConnectionToVisibleDetail": _clean(metaphor.get("viewerDiscoveryPresent")),
                "sloganConnectionToRelativeAdvantage": _clean(metaphor.get("physicalEmbodimentMatchesStrategicRelationship")),
            }
    return {}


def validate_single_slogan_plan_contract(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if not is_single_slogan_contract(plan=plan):
        return True, failures
    if _clean(plan.get("copyContractVersion")) != BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION:
        failures.append("copy_contract_version_missing")
    slogan = resolve_canonical_slogan_text(plan=plan)
    if not slogan:
        failures.append("canonical_slogan_missing")
    if builder2_requires_headline_overlay(plan=plan):
        failures.append("headline_overlay_requested_under_single_slogan_contract")
    headline_raw = _clean(plan.get("headline"))
    if headline_raw and headline_raw != slogan and not plan.get("headlineCompatibilityAlias"):
        failures.append("separate_headline_message_present")
    closure = normalize_advertising_closure(plan.get("advertisingClosure"))
    closure_slogan = _clean(closure.get("sloganText"))
    if closure_slogan and slogan and closure_slogan != slogan:
        failures.append("competing_closure_slogan")
    if plan.get("dependsOnEarlierCopy") is True:
        failures.append("slogan_depends_on_earlier_copy")
    if plan.get("sloganUnderstandsWithoutPriorCopy") is False:
        failures.append("slogan_requires_prior_copy")
    return not failures, failures


def validate_single_slogan_completion(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    media: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not is_single_slogan_contract(state=state, plan=plan):
        return []
    failures: List[str] = []
    media = media if isinstance(media, dict) else (state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {})
    ok, plan_failures = validate_single_slogan_plan_contract(plan)
    failures.extend(plan_failures)
    if media.get("headlineOverlaySkipped") is not True and media.get("headlinePostprocessStatus") in {"completed", "reused"}:
        failures.append("headline_overlay_rendered_under_single_slogan_contract")
    if _clean(media.get("headlineArtifactUrl")) and media.get("headlineOverlaySkipped") is not True:
        failures.append("headline_artifact_present_under_single_slogan_contract")
    if media.get("sloganRenderedExactlyOnce") is not True:
        failures.append("slogan_not_rendered_exactly_once")
    if media.get("advertisingCopyRenderStages", 0) not in (None, 0, 1) and int(media.get("advertisingCopyRenderStages") or 0) > 1:
        failures.append("multiple_advertising_copy_render_stages")
    return failures


def mark_single_slogan_media_fields(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    headline_overlay_skipped: bool,
    closure_render_completed: bool,
) -> None:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    media["copyContractVersion"] = BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION
    media["headlineOverlaySkipped"] = bool(headline_overlay_skipped)
    media["legacyCopyNormalized"] = bool(state.get("legacyCopyNormalized"))
    if closure_render_completed:
        media["sloganRenderedExactlyOnce"] = True
        media["advertisingCopyRenderStages"] = 1
        plan["sloganRenderedExactlyOnce"] = True


def sync_closure_slogan_from_canonical(*, plan: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> None:
    canonical = resolve_canonical_slogan_text(plan=plan, state=state)
    if not canonical:
        return
    closure = normalize_advertising_closure(plan.get("advertisingClosure"))
    closure["sloganText"] = canonical
    closure["required"] = True
    plan["advertisingClosure"] = closure
    if isinstance(state, dict):
        state["advertisingClosure"] = dict(closure)


def log_single_slogan_safe_metadata(
    *,
    plan: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    media: Optional[Dict[str, Any]] = None,
    job_id: str = "",
) -> None:
    plan = plan if isinstance(plan, dict) else {}
    media = media if isinstance(media, dict) else {}
    slogan = resolve_canonical_slogan_text(plan=plan, state=state)
    logger.info(
        "BUILDER2_SINGLE_SLOGAN_METADATA jobId=%s copyContractVersion=%s sloganDecision=%s sloganPresent=%s "
        "sloganCharacterCount=%s sloganWordCount=%s sloganVisualBridgeAccepted=%s sloganRenderedExactlyOnce=%s "
        "headlineOverlaySkipped=%s legacyCopyNormalized=%s",
        (job_id or "").strip() or "(none)",
        copy_contract_version(state=state, plan=plan) or BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
        _clean(plan.get("sloganDecision")) or (SLOGAN_DECISION_USE if slogan else SLOGAN_DECISION_OMIT),
        str(bool(slogan)).lower(),
        len(slogan),
        _word_count(slogan),
        str(plan.get("sloganUnderstandsWithoutPriorCopy") is True).lower(),
        str((media.get("sloganRenderedExactlyOnce") or plan.get("sloganRenderedExactlyOnce")) is True).lower(),
        str((media.get("headlineOverlaySkipped") or plan.get("headlineOverlaySkipped")) is True).lower(),
        str((media.get("legacyCopyNormalized") or state.get("legacyCopyNormalized") if isinstance(state, dict) else False) is True).lower(),
    )


def classify_literal_domain_symbols(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for pattern in LITERAL_DOMAIN_SYMBOL_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def raise_single_slogan_contract_error(code: str, *, field: str = "") -> None:
    suffix = f":{field}" if field else ""
    raise Builder2TournamentError(f"{code}{suffix}")
