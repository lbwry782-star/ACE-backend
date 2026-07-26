"""
Builder2 Creator normalization — alias mapping and server-derived methodology metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_creator_core_contract import (
    PROTOTYPE_APPLICATION_ALIASES,
    PROTOTYPE_APPLICATION_CHILD_ALIASES,
    PROTOTYPE_APPLICATION_FIELDS,
    VALID_VERBAL_DECISIONS,
    VALID_VISUAL_ANCHOR_TIMING,
)
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import expected_strategy_foundation_id

logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_prototype_application_child_aliases(
    out: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    resolved: List[str],
) -> None:
    canonical = PROTOTYPE_APPLICATION_FIELDS.get(assigned_prototype_id)
    alias_map = PROTOTYPE_APPLICATION_CHILD_ALIASES.get(assigned_prototype_id) or {}
    if not canonical:
        return
    app_raw = out.get(canonical)
    if not isinstance(app_raw, dict):
        return
    app = dict(app_raw)
    changed = False
    for canonical_key, aliases in alias_map.items():
        if _text(app.get(canonical_key)):
            continue
        for alias in aliases:
            if alias == canonical_key:
                continue
            alias_value = app.get(alias)
            if _text(alias_value):
                app[canonical_key] = alias_value
                resolved.append(f"{canonical}.{canonical_key}")
                changed = True
                break
    if changed:
        out[canonical] = app


def _normalize_prototype_application_aliases(out: Dict[str, Any], *, assigned_prototype_id: str) -> None:
    aliases = PROTOTYPE_APPLICATION_ALIASES.get(assigned_prototype_id, ())
    canonical = PROTOTYPE_APPLICATION_FIELDS.get(assigned_prototype_id)
    if not canonical:
        return
    generic = out.get("prototypeApplication")
    if isinstance(generic, dict) and not isinstance(out.get(canonical), dict):
        out[canonical] = dict(generic)
    for alias in aliases:
        if alias == canonical:
            continue
        payload = out.get(alias)
        if isinstance(payload, dict) and not isinstance(out.get(canonical), dict):
            out[canonical] = dict(payload)


def _normalize_video_execution(out: Dict[str, Any], resolved: List[str]) -> None:
    video = out.get("videoExecution")
    if isinstance(video, dict):
        seven = out.setdefault("sevenSecondStructure", {})
        if isinstance(seven, dict):
            for key in ("beginning", "development", "resolution"):
                if _text(video.get(key)) and not _text(seven.get(key)):
                    seven[key] = video[key]
                    resolved.append(f"sevenSecondStructure.{key}")
        runway = out.setdefault("runwayFeasibility", {})
        if isinstance(runway, dict):
            for key in ("mainSubject", "mainAction", "location", "openingFrame"):
                if _text(video.get(key)) and not _text(runway.get(key)):
                    runway[key] = video[key]
                    resolved.append(f"runwayFeasibility.{key}")
        return

    seven = out.get("sevenSecondStructure")
    runway = out.get("runwayFeasibility")
    if isinstance(seven, dict) and isinstance(runway, dict):
        out["videoExecution"] = {
            "mainSubject": runway.get("mainSubject"),
            "mainAction": runway.get("mainAction"),
            "location": runway.get("location"),
            "openingFrame": runway.get("openingFrame"),
            "beginning": seven.get("beginning"),
            "development": seven.get("development"),
            "resolution": seven.get("resolution"),
        }
        resolved.append("videoExecution")


def _normalize_silent_verification(out: Dict[str, Any], resolved: List[str]) -> None:
    report = out.setdefault("creatorReport", {})
    if not isinstance(report, dict):
        return
    silent_top = out.get("silentVerification")
    if isinstance(silent_top, dict):
        explanation = _text(silent_top.get("explanation"))
        if explanation and not _text(report.get("silentVerification")):
            report["silentVerification"] = explanation
            resolved.append("creatorReport.silentVerification")
    elif _text(silent_top) and not _text(report.get("silentVerification")):
        report["silentVerification"] = _text(silent_top)
        resolved.append("creatorReport.silentVerification")


def _normalize_visual_anchor_timing(out: Dict[str, Any], resolved: List[str]) -> None:
    anchor = out.get("visualAnchor")
    if not isinstance(anchor, dict):
        return
    anchor = dict(anchor)
    timing = _text(anchor.get("visualAnchorTiming") or anchor.get("timing")).lower()
    if timing in VALID_VISUAL_ANCHOR_TIMING:
        anchor["visualAnchorTiming"] = timing
        appears = timing in {"opening", "development"}
        if anchor.get("appearsBeforeOrDuringResolution") is not True and appears:
            anchor["appearsBeforeOrDuringResolution"] = True
            resolved.append("visualAnchor.appearsBeforeOrDuringResolution")
    elif anchor.get("appearsBeforeOrDuringResolution") is True:
        anchor["visualAnchorTiming"] = "development"
        resolved.append("visualAnchor.visualAnchorTiming")
    elif _text(anchor.get("description")):
        anchor.setdefault("visualAnchorTiming", "development")
        anchor.setdefault("appearsBeforeOrDuringResolution", True)
        resolved.append("visualAnchor.appearsBeforeOrDuringResolution")
    out["visualAnchor"] = anchor


def _derive_concept_summary(out: Dict[str, Any], resolved: List[str]) -> None:
    if _text(out.get("conceptSummary")):
        return
    mechanism = _text(out.get("coreCreativeMechanism"))
    if mechanism:
        out["conceptSummary"] = mechanism
        resolved.append("conceptSummary")


def _derive_visual_family(out: Dict[str, Any], resolved: List[str]) -> None:
    if _text(out.get("visualFamily")):
        return
    mechanism = _text(out.get("visualMechanism") or out.get("coreCreativeMechanism"))
    if mechanism:
        out["visualFamily"] = mechanism[:120]
        resolved.append("visualFamily")


def _derive_prototype_method_applied(out: Dict[str, Any], *, assigned_prototype_id: str, resolved: List[str]) -> None:
    if _text(out.get("prototypeMethodApplied")):
        return
    out["prototypeMethodApplied"] = f"Applied assigned prototype method: {assigned_prototype_id}"
    resolved.append("prototypeMethodApplied")


def _derive_essence_extreme(
    out: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]],
    resolved: List[str],
) -> None:
    if isinstance(out.get("essenceExtreme"), dict) and out["essenceExtreme"].get("derivedByServer"):
        return
    advantage = ""
    if strategy_foundation:
        rel = strategy_foundation.get("relativeAdvantage") or {}
        advantage = _text(rel.get("statement"))
    report = out.get("creatorReport") or {}
    derived = {
        "advantageEssence": advantage or _text(report.get("relativeAdvantage")),
        "extremePhysicalExpression": _text(out.get("visualMechanism")),
        "whyChosenObjectsFollowFromTheEssence": _text(report.get("whyParallelExpressesAdvantage")),
        "derivedByServer": True,
    }
    if all(_text(derived[k]) for k in ("advantageEssence", "extremePhysicalExpression", "whyChosenObjectsFollowFromTheEssence")):
        out["essenceExtreme"] = derived
        resolved.append("essenceExtreme")


def _derive_participation_mechanism(out: Dict[str, Any], resolved: List[str]) -> None:
    if isinstance(out.get("participationMechanism"), dict) and out["participationMechanism"].get("derivedByServer"):
        return
    runway = out.get("runwayFeasibility") or {}
    seven = out.get("sevenSecondStructure") or {}
    who = _text(runway.get("mainSubject"))
    action = _text(runway.get("mainAction"))
    effect = _text(seven.get("resolution") or seven.get("development"))
    if not (who and action and effect):
        return
    out["participationMechanism"] = {
        "whoOrWhatParticipates": who,
        "visibleAction": action,
        "visibleCauseAndEffect": effect,
        "notMerelyAReadyMadeResult": True,
        "derivedByServer": True,
    }
    resolved.append("participationMechanism")


def _derive_anchor_punchline_separation(out: Dict[str, Any], resolved: List[str]) -> None:
    if isinstance(out.get("anchorPunchlineSeparation"), dict) and out["anchorPunchlineSeparation"].get("derivedByServer"):
        return
    anchor = out.get("visualAnchor") or {}
    seven = out.get("sevenSecondStructure") or {}
    anchor_text = _text(anchor.get("description"))
    resolution = _text(seven.get("resolution"))
    if not (anchor_text and resolution):
        return
    out["anchorPunchlineSeparation"] = {
        "anchor": anchor_text,
        "resolutionOrPunchline": resolution,
        "whyTheyAreNotTheSameThing": "Server-derived separation from anchor and resolution.",
        "derivedByServer": True,
    }
    resolved.append("anchorPunchlineSeparation")


def _derive_visual_family_consistency(out: Dict[str, Any], resolved: List[str]) -> None:
    structure = _text(out.get("structureType"))
    if structure == "variation_montage":
        family_id = _text(out.get("visualFamilyId"))
        family_def = _text(out.get("visualFamilyDefinition"))
        motif = _text(out.get("recurringMotif"))
        if not (family_def and motif):
            return
        out["visualFamilyConsistency"] = {
            "familyDefinition": family_def,
            "recurringMotif": motif,
            "visualFamilyId": family_id or "montage_family",
            "whyAllVariationsBelongTogether": _text(out.get("visualFamily")) or family_def,
            "sideBySideFrameTest": "Shared motif across montage variations.",
            "structureType": "variation_montage",
            "derivedByServer": True,
        }
        resolved.append("visualFamilyConsistency")
        return

    mechanism = _text(out.get("visualMechanism") or out.get("coreCreativeMechanism"))
    if not mechanism:
        return
    out["visualFamilyConsistency"] = {
        "familyDefinition": "single continuous visual world",
        "recurringMotif": mechanism[:120],
        "whyAllVariationsBelongTogether": "One continuous event in one coherent world.",
        "sideBySideFrameTest": "Not applicable for continuous_event.",
        "structureType": "continuous_event",
        "derivedByServer": True,
    }
    resolved.append("visualFamilyConsistency")


def _normalize_verbal_potential(out: Dict[str, Any], resolved: List[str]) -> None:
    verbal = out.get("verbalPotential")
    if verbal is None:
        out["verbalPotential"] = {
            "decision": "not_needed",
            "reason": "Visual mechanism communicates without a forced keyword.",
        }
        resolved.append("verbalPotential.decision")
        return
    if not isinstance(verbal, dict):
        return
    verbal = dict(verbal)
    decision = _text(verbal.get("decision")).lower()
    keyword = _text(verbal.get("keywordOrKeyPhrase"))
    if decision not in VALID_VERBAL_DECISIONS:
        if keyword:
            decision = "available"
        elif verbal.get("headlineMayBeUnnecessary") is True:
            decision = "not_needed"
        else:
            decision = "not_needed"
        verbal["decision"] = decision
        resolved.append("verbalPotential.decision")
    if decision == "available":
        return
    if decision in {"not_needed", "not_found"} and not _text(verbal.get("reason")):
        verbal["reason"] = (
            "Visual mechanism is sufficient without a headline keyword."
            if decision == "not_needed"
            else "No genuine verbal parallel found without forcing language."
        )
        resolved.append("verbalPotential.reason")
    out["verbalPotential"] = verbal


def _normalize_source_concept(out: Dict[str, Any], resolved: List[str]) -> None:
    source = out.get("sourceConcept")
    if isinstance(source, dict) and _text(source.get("type")):
        return
    out["sourceConcept"] = {"type": "native_builder2", "derivedByServer": True}
    resolved.append("sourceConcept")


def _normalize_montage_aliases(out: Dict[str, Any], resolved: List[str]) -> None:
    consistency = out.get("visualFamilyConsistency")
    if isinstance(consistency, dict):
        if not _text(out.get("visualFamilyDefinition")) and _text(consistency.get("familyDefinition")):
            out["visualFamilyDefinition"] = consistency["familyDefinition"]
            resolved.append("visualFamilyDefinition")
        if not _text(out.get("recurringMotif")) and _text(consistency.get("recurringMotif")):
            out["recurringMotif"] = consistency["recurringMotif"]
            resolved.append("recurringMotif")
        if not _text(out.get("visualFamilyId")) and _text(consistency.get("visualFamilyId")):
            out["visualFamilyId"] = consistency["visualFamilyId"]
            resolved.append("visualFamilyId")


def _normalize_strategy_identity(
    out: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    resolved: List[str],
) -> None:
    if strategy_foundation is None:
        return
    expected = expected_strategy_foundation_id(strategy_foundation)
    if expected and not _text(out.get("strategyFoundationId")):
        out["strategyFoundationId"] = expected
        resolved.append("strategyFoundationId")


def normalize_creator_candidate(
    raw: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    base_normalizer: Any = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Full Creator normalization pipeline. Returns (candidate, resolved_field_paths).
    """
    if base_normalizer is not None:
        out = base_normalizer(
            raw,
            assigned_prototype_id=assigned_prototype_id,
            prototype_display_name=prototype_display_name,
            compatibility_mode=compatibility_mode,
        )
    else:
        from engine.builder2_creator import normalize_creator_raw

        out = normalize_creator_raw(
            raw,
            assigned_prototype_id=assigned_prototype_id,
            prototype_display_name=prototype_display_name,
            compatibility_mode=compatibility_mode,
        )

    resolved: List[str] = []
    out.setdefault("methodologyVersion", METHODOLOGY_VERSION)

    _normalize_prototype_application_aliases(out, assigned_prototype_id=assigned_prototype_id)
    _normalize_prototype_application_child_aliases(out, assigned_prototype_id=assigned_prototype_id, resolved=resolved)
    _normalize_video_execution(out, resolved)
    _normalize_silent_verification(out, resolved)
    _normalize_visual_anchor_timing(out, resolved)
    _normalize_verbal_potential(out, resolved)
    _normalize_montage_aliases(out, resolved)
    _normalize_strategy_identity(out, strategy_foundation=strategy_foundation, resolved=resolved)

    _derive_concept_summary(out, resolved)
    _derive_visual_family(out, resolved)
    _derive_prototype_method_applied(out, assigned_prototype_id=assigned_prototype_id, resolved=resolved)
    _derive_essence_extreme(out, strategy_foundation=strategy_foundation, resolved=resolved)
    _derive_participation_mechanism(out, resolved)
    _derive_anchor_punchline_separation(out, resolved)
    _derive_visual_family_consistency(out, resolved)
    _normalize_source_concept(out, resolved)

    if resolved:
        logger.info(
            "BUILDER2_CREATOR_NORMALIZATION_RESOLVED_FIELDS prototypeId=%s count=%s paths=%s",
            assigned_prototype_id,
            len(resolved),
            ",".join(resolved[:20]),
        )
    return out, list(dict.fromkeys(resolved))
