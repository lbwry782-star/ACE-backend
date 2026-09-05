"""
Builder2 methodology validation — deterministic enforcement helpers.

Prototype validators separate deterministic structure (this module) from semantic
quality (Judge assessments). Substring heuristics are narrow safeguards only.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from engine.builder2_methodology_contract import (
    GENERIC_BUSINESS_GOAL_PATTERNS,
    METHODOLOGY_VERSION,
    PROCESS_FAILURE_TAGS,
    STRATEGY_FORBIDDEN_TOP_LEVEL_KEYS,
    VALID_HEADLINE_DECISIONS,
    VALID_HEADLINE_FORMS,
    VALID_PROVOCATION_RISK,
    VALID_REPLACEMENT_CARRIERS,
    VALID_SOURCE_CONCEPT_TYPES,
    VALID_STRUCTURE_TYPES,
)
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import (
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)

logger = logging.getLogger(__name__)

from engine.builder2_creator_core_contract import (
    PROTOTYPE_APPLICATION_FIELDS as _PROTOTYPE_APPLICATION_FIELDS,
    VALID_VERBAL_DECISIONS,
)

_PLAYING_CARD_SURFACE_MARKERS = ("playing card", "card symbol", "playing-card")


def uses_full_methodology(record: Dict[str, Any]) -> bool:
    return record.get("methodologyVersion") == METHODOLOGY_VERSION


def normalize_strategy_methodology_defaults(raw: Dict[str, Any], *, compatibility_mode: bool) -> Dict[str, Any]:
    out = dict(raw)
    if compatibility_mode and not uses_full_methodology(out):
        return out
    out["methodologyVersion"] = METHODOLOGY_VERSION
    ra = out.get("relativeAdvantage")
    if isinstance(ra, dict):
        ra = dict(ra)
        if ra.get("admitsRelevantGap") is None and not compatibility_mode:
            ra["admitsRelevantGap"] = False
        out["relativeAdvantage"] = ra
    ms = out.get("mechanismScan")
    if isinstance(ms, dict):
        ms = dict(ms)
        out["mechanismScan"] = ms
    return out


def normalize_candidate_methodology_defaults(raw: Dict[str, Any], *, compatibility_mode: bool) -> Dict[str, Any]:
    out = dict(raw)
    if compatibility_mode and not uses_full_methodology(out):
        return out
    out["methodologyVersion"] = METHODOLOGY_VERSION
    source = out.get("sourceConcept")
    if source is None and not compatibility_mode:
        out["sourceConcept"] = {"type": "native_builder2"}
    elif isinstance(source, dict) and not source.get("type"):
        source = dict(source)
        source["type"] = "native_builder2"
        out["sourceConcept"] = source
    return out


def build_winning_candidate_preservation_snapshot(
    *,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    candidate_id: str = "",
) -> Dict[str, Any]:
    from engine.builder2_winner_preservation_contract import (
        build_winning_candidate_preservation_snapshot as _build_snapshot,
    )

    return _build_snapshot(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )


def _raise(code: str, *, field: str | None = None) -> None:
    if field:
        raise Builder2TournamentError(f"{code}:{field}")
    raise Builder2TournamentError(code)


def _field_from_error(exc: Builder2TournamentError) -> str:
    msg = str(exc.args[0] if exc.args else "")
    if msg.startswith("builder2_tournament_invalid_field:"):
        return msg.split(":", 1)[1]
    return "unknown"


def _require_bool(value: Any, *, field: str, code: str) -> bool:
    if not isinstance(value, bool):
        _raise(code, field=field)
    return value


def _require_dict(value: Any, *, field: str, code: str) -> Dict[str, Any]:
    try:
        return require_dict(value, field=field)
    except Builder2TournamentError as exc:
        _raise(code, field=_field_from_error(exc))


def _require_text(value: Any, *, field: str, code: str) -> str:
    try:
        return require_non_empty_str(value, field=field)
    except Builder2TournamentError as exc:
        _raise(code, field=_field_from_error(exc))


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def validate_strategy_methodology(
    strategy: Dict[str, Any],
    *,
    compatibility_mode: bool = False,
) -> None:
    if compatibility_mode and not uses_full_methodology(strategy):
        logger.info("BUILDER2_METHODOLOGY_COMPATIBILITY_MODE role=strategy")
        return

    for key in STRATEGY_FORBIDDEN_TOP_LEVEL_KEYS:
        if key in strategy and strategy.get(key) not in (None, "", [], {}):
            _raise("builder2_strategy_validation_failed", field=key)

    pp = _require_dict(strategy.get("problemPerception"), field="problemPerception", code="builder2_strategy_schema_invalid")
    statement = _require_text(pp.get("statement"), field="problemPerception.statement", code="builder2_strategy_schema_invalid")
    blob = statement.lower()
    for pattern in GENERIC_BUSINESS_GOAL_PATTERNS:
        if pattern in blob:
            _raise("builder2_strategy_validation_failed", field="problemPerception.statement")

    ra = _require_dict(strategy.get("relativeAdvantage"), field="relativeAdvantage", code="builder2_strategy_schema_invalid")
    _require_text(ra.get("derivationFromProblem"), field="relativeAdvantage.derivationFromProblem", code="builder2_strategy_validation_failed")
    _require_text(ra.get("truthBoundary"), field="relativeAdvantage.truthBoundary", code="builder2_strategy_validation_failed")
    _require_bool(ra.get("admitsRelevantGap"), field="relativeAdvantage.admitsRelevantGap", code="builder2_strategy_schema_invalid")

    ms = _require_dict(strategy.get("mechanismScan"), field="mechanismScan", code="builder2_strategy_schema_invalid")
    _require_text(ms.get("depthEvidence"), field="mechanismScan.depthEvidence", code="builder2_strategy_validation_failed")

    logger.info("BUILDER2_STRATEGY_METHODOLOGY_VALIDATED methodologyVersion=%s", strategy.get("methodologyVersion"))


def validate_strategy_identity(
    *,
    expected_strategy_foundation_id: str,
    candidate: Dict[str, Any],
) -> None:
    """Deterministic identity check — Creator Report paraphrase is allowed."""
    expected = str(expected_strategy_foundation_id or "").strip()
    if not expected:
        return
    actual = str(candidate.get("strategyFoundationId") or "").strip()
    if actual != expected:
        _raise("builder2_creator_validation_failed", field="strategyFoundationId")


def _validate_montage_visual_family(candidate: Dict[str, Any]) -> None:
    variations = candidate.get("sceneVariations")
    if not isinstance(variations, list):
        _raise("builder2_creator_validation_failed", field="sceneVariations")
    cleaned: List[str] = []
    family_ids: List[str] = []
    for item in variations:
        if isinstance(item, dict):
            text = str(item.get("description") or item.get("variation") or "").strip()
            family_ids.append(_normalize_text(item.get("familyId") or item.get("familyLabel")))
        else:
            text = str(item or "").strip()
            family_ids.append("")
        if text:
            cleaned.append(text)
    if len(cleaned) < 2 or len(cleaned) > 4:
        _raise("builder2_creator_validation_failed", field="sceneVariations.count")
    declared_ids = {fid for fid in family_ids if fid}
    if len(declared_ids) > 1:
        _raise("builder2_creator_validation_failed", field="sceneVariations.familyId")


def _validate_visual_family_for_structure(candidate: Dict[str, Any]) -> None:
    structure = str(candidate.get("structureType") or "")
    if structure not in VALID_STRUCTURE_TYPES:
        return
    if structure == "variation_montage":
        _validate_montage_visual_family(candidate)
        consistency = candidate.get("visualFamilyConsistency")
        if isinstance(consistency, dict) and consistency.get("derivedByServer"):
            return
    consistency = candidate.get("visualFamilyConsistency")
    if isinstance(consistency, dict) and consistency.get("derivedByServer"):
        logger.info("BUILDER2_VISUAL_FAMILY_VALIDATED prototypeId=%s derived=true", candidate.get("prototypeId"))
        return
    family = _require_dict(
        candidate.get("visualFamilyConsistency"),
        field="visualFamilyConsistency",
        code="builder2_creator_validation_failed",
    )
    for key in ("familyDefinition", "recurringMotif"):
        _require_text(family.get(key), field=f"visualFamilyConsistency.{key}", code="builder2_creator_validation_failed")
    if structure == "variation_montage":
        for key in ("whyAllVariationsBelongTogether", "sideBySideFrameTest"):
            _require_text(family.get(key), field=f"visualFamilyConsistency.{key}", code="builder2_creator_validation_failed")
    logger.info("BUILDER2_VISUAL_FAMILY_VALIDATED prototypeId=%s", candidate.get("prototypeId"))


def validate_creator_methodology(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> None:
    if compatibility_mode and not uses_full_methodology(candidate):
        logger.info(
            "BUILDER2_METHODOLOGY_COMPATIBILITY_MODE role=creator candidatePrototype=%s",
            assigned_prototype_id,
        )
        return

    report = _require_dict(candidate.get("creatorReport"), field="creatorReport", code="builder2_creator_schema_invalid")
    _require_text(report.get("mechanismScanSummary"), field="creatorReport.mechanismScanSummary", code="builder2_creator_validation_failed")
    _require_text(candidate.get("visualMechanism"), field="visualMechanism", code="builder2_creator_validation_failed")
    if candidate.get("visualMechanism") == report.get("mechanismScanSummary"):
        _raise("builder2_creator_validation_failed", field="visualMechanism")

    essence = candidate.get("essenceExtreme")
    if not (isinstance(essence, dict) and essence.get("derivedByServer")):
        essence_obj = _require_dict(essence, field="essenceExtreme", code="builder2_creator_validation_failed")
        for key in ("advantageEssence", "extremePhysicalExpression", "whyChosenObjectsFollowFromTheEssence"):
            _require_text(essence_obj.get(key), field=f"essenceExtreme.{key}", code="builder2_creator_validation_failed")

    _validate_visual_family_for_structure(candidate)

    participation = candidate.get("participationMechanism")
    if not (isinstance(participation, dict) and participation.get("derivedByServer")):
        participation_obj = _require_dict(
            participation,
            field="participationMechanism",
            code="builder2_creator_validation_failed",
        )
        for key in ("whoOrWhatParticipates", "visibleAction", "visibleCauseAndEffect"):
            _require_text(participation_obj.get(key), field=f"participationMechanism.{key}", code="builder2_creator_validation_failed")
        _require_bool(
            participation_obj.get("notMerelyAReadyMadeResult"),
            field="participationMechanism.notMerelyAReadyMadeResult",
            code="builder2_creator_validation_failed",
        )
    logger.info("BUILDER2_PARTICIPATION_VALIDATED prototypeId=%s", assigned_prototype_id)

    from engine.builder2_complete_ad_contract import validate_creator_complete_ad_fields

    if not compatibility_mode:
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=strategy_foundation,
            assigned_prototype_id=assigned_prototype_id,
            product_name=str((strategy_foundation or {}).get("productNameResolved") or "").strip(),
        )
        from engine.builder2_metaphorical_embodiment_contract import validate_creator_metaphorical_embodiment
        from engine.builder2_single_slogan_contract import BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION

        requires_metaphor = str((tournament_state or {}).get("copyContractVersion") or "").strip() == BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION
        if requires_metaphor or isinstance(candidate.get("metaphoricalEmbodiment"), dict):
            validate_creator_metaphorical_embodiment(
                candidate,
                assigned_prototype_id=assigned_prototype_id,
                single_slogan_required=requires_metaphor,
                tournament_state=tournament_state,
            )
        from engine.builder2_no_logo_contract import BUILDER2_NO_LOGO_POLICY_VERSION, validate_creator_logo_policy

        requires_no_logo = str((tournament_state or {}).get("logoPolicyVersion") or "").strip() == BUILDER2_NO_LOGO_POLICY_VERSION
        if requires_no_logo or isinstance(candidate.get("logoPolicyReport"), dict):
            validate_creator_logo_policy(
                candidate,
                assigned_prototype_id=assigned_prototype_id,
                product_name=str((strategy_foundation or {}).get("productNameResolved") or "").strip(),
                no_logo_required=requires_no_logo,
            )
        from engine.builder2_advertising_slogan_quality_contract import (
            BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
            validate_creator_advertising_slogan_formulation,
        )

        requires_slogan_quality = (
            str((tournament_state or {}).get("advertisingSloganQualityContractVersion") or "").strip()
            == BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION
        )
        if requires_slogan_quality or isinstance(candidate.get("advertisingSloganFormulation"), dict):
            validate_creator_advertising_slogan_formulation(
                candidate,
                strategy_foundation=strategy_foundation,
                product_name=str((strategy_foundation or {}).get("productNameResolved") or "").strip(),
            )
        from engine.builder2_strategy_evidence_grounding_contract import validate_creator_evidence_grounding

        validate_creator_evidence_grounding(
            candidate,
            strategy_foundation=strategy_foundation if isinstance(strategy_foundation, dict) else {},
            compatibility_mode=compatibility_mode,
        )

    anchor = _require_dict(candidate.get("visualAnchor"), field="visualAnchor", code="builder2_creator_schema_invalid")
    if anchor.get("appearsBeforeOrDuringResolution") is not True and not _normalize_text(anchor.get("visualAnchorTiming")):
        _require_bool(
            anchor.get("appearsBeforeOrDuringResolution"),
            field="visualAnchor.appearsBeforeOrDuringResolution",
            code="builder2_creator_schema_invalid",
        )
    separation = candidate.get("anchorPunchlineSeparation")
    if not (isinstance(separation, dict) and separation.get("derivedByServer")):
        separation_obj = _require_dict(
            separation,
            field="anchorPunchlineSeparation",
            code="builder2_creator_validation_failed",
        )
        for key in ("anchor", "resolutionOrPunchline"):
            _require_text(separation_obj.get(key), field=f"anchorPunchlineSeparation.{key}", code="builder2_creator_validation_failed")

    runway = _require_dict(candidate.get("runwayFeasibility"), field="runwayFeasibility", code="builder2_creator_schema_invalid")
    _require_bool(runway.get("fitsSevenSeconds"), field="runwayFeasibility.fitsSevenSeconds", code="builder2_creator_schema_invalid")
    _require_bool(
        runway.get("requiresImpossibleMorphing"),
        field="runwayFeasibility.requiresImpossibleMorphing",
        code="builder2_creator_schema_invalid",
    )
    _require_bool(
        runway.get("requiresSubtleUnseenInference"),
        field="runwayFeasibility.requiresSubtleUnseenInference",
        code="builder2_creator_schema_invalid",
    )
    if runway.get("requiresImpossibleMorphing") is True:
        _raise("builder2_creator_validation_failed", field="runwayFeasibility.requiresImpossibleMorphing")
    if runway.get("requiresSubtleUnseenInference") is True:
        _raise("builder2_creator_validation_failed", field="runwayFeasibility.requiresSubtleUnseenInference")

    verbal = candidate.get("verbalPotential")
    if verbal is None:
        _raise("builder2_creator_validation_failed", field="verbalPotential")
    verbal_obj = _require_dict(verbal, field="verbalPotential", code="builder2_creator_validation_failed")
    decision = _normalize_text(verbal_obj.get("decision")).lower()
    if decision not in VALID_VERBAL_DECISIONS:
        if _normalize_text(verbal_obj.get("keywordOrKeyPhrase")):
            decision = "available"
        else:
            decision = "not_needed"
        verbal_obj = dict(verbal_obj)
        verbal_obj["decision"] = decision
    if decision == "available":
        for key in ("keywordOrKeyPhrase", "visualMeaning", "strategicMeaning"):
            _require_text(verbal_obj.get(key), field=f"verbalPotential.{key}", code="builder2_creator_validation_failed")
        if _require_bool(
            verbal_obj.get("bornFromVisibleMechanism"),
            field="verbalPotential.bornFromVisibleMechanism",
            code="builder2_creator_validation_failed",
        ) is not True:
            _raise("builder2_creator_validation_failed", field="verbalPotential.bornFromVisibleMechanism")
    else:
        _require_text(verbal_obj.get("reason"), field="verbalPotential.reason", code="builder2_creator_validation_failed")
    if verbal_obj.get("headlineMayBeUnnecessary") is not None:
        _require_bool(
            verbal_obj.get("headlineMayBeUnnecessary"),
            field="verbalPotential.headlineMayBeUnnecessary",
            code="builder2_creator_validation_failed",
        )
    logger.info("BUILDER2_VERBAL_POTENTIAL_VALIDATED prototypeId=%s decision=%s", assigned_prototype_id, decision)

    source = candidate.get("sourceConcept") or {"type": "native_builder2"}
    source_obj = _require_dict(source, field="sourceConcept", code="builder2_creator_schema_invalid")
    source_type = _require_text(source_obj.get("type"), field="sourceConcept.type", code="builder2_creator_schema_invalid")
    if source_type not in VALID_SOURCE_CONCEPT_TYPES:
        _raise("builder2_creator_schema_invalid", field="sourceConcept.type")
    if source_type == "builder1_adaptation":
        _require_text(source_obj.get("originalVisualParallel"), field="sourceConcept.originalVisualParallel", code="builder2_creator_validation_failed")
        _require_text(source_obj.get("preservedMechanism"), field="sourceConcept.preservedMechanism", code="builder2_creator_validation_failed")

    vpt = str(candidate.get("visualParallelType") or "")
    if vpt == "replacement":
        _validate_replacement_check(candidate)
    if vpt == "context_collision":
        _validate_context_collision_safeguard(candidate)

    _validate_prototype_application(candidate, assigned_prototype_id=assigned_prototype_id)

    report_self = report.get("creatorPuritySelfCheck")
    if report_self is not None:
        _require_text(report_self, field="creatorReport.creatorPuritySelfCheck", code="builder2_creator_validation_failed")

    if strategy_foundation is not None:
        validate_strategy_identity(
            expected_strategy_foundation_id=expected_strategy_foundation_id(strategy_foundation),
            candidate=candidate,
        )

    logger.info("BUILDER2_CREATOR_METHODOLOGY_VALIDATED prototypeId=%s", assigned_prototype_id)
    logger.info("BUILDER2_PROTOTYPE_METHOD_VALIDATED prototypeId=%s", assigned_prototype_id)


def _validate_replacement_check(candidate: Dict[str, Any]) -> None:
    rep = _require_dict(candidate.get("replacementCheck"), field="replacementCheck", code="builder2_creator_validation_failed")
    for key in ("visibleObject", "absentObject", "whyInferenceIsImmediate"):
        _require_text(rep.get(key), field=f"replacementCheck.{key}", code="builder2_creator_validation_failed")
    carrier = _require_text(rep.get("replacementCarrier"), field="replacementCheck.replacementCarrier", code="builder2_creator_validation_failed")
    if carrier not in VALID_REPLACEMENT_CARRIERS:
        _raise("builder2_creator_schema_invalid", field="replacementCheck.replacementCarrier")
    _require_bool(rep.get("viewerCanInferAbsentObject"), field="replacementCheck.viewerCanInferAbsentObject", code="builder2_creator_validation_failed")


def _validate_context_collision_safeguard(candidate: Dict[str, Any]) -> None:
    safeguard = _require_dict(
        candidate.get("contextCollisionSafeguard"),
        field="contextCollisionSafeguard",
        code="builder2_creator_validation_failed",
    )
    for key in ("contextA", "contextB", "naturalBridge", "howCollisionServesTheAdvantage"):
        _require_text(safeguard.get(key), field=f"contextCollisionSafeguard.{key}", code="builder2_creator_validation_failed")
    risk = _require_text(safeguard.get("provocationRisk"), field="contextCollisionSafeguard.provocationRisk", code="builder2_creator_schema_invalid")
    if risk not in VALID_PROVOCATION_RISK:
        _raise("builder2_creator_schema_invalid", field="contextCollisionSafeguard.provocationRisk")
    _require_bool(
        safeguard.get("notProvocativeForItsOwnSake"),
        field="contextCollisionSafeguard.notProvocativeForItsOwnSake",
        code="builder2_creator_validation_failed",
    )


def _append_methodology_structural(errors: List[str], code: str, field: str) -> None:
    msg = f"{code}:{field}"
    if msg not in errors:
        errors.append(msg)


def _validate_prototype_application(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    structural_errors: Optional[List[str]] = None,
) -> None:
    """Deterministic structural prototype checks — semantic truth remains Judge responsibility."""
    field_name = _PROTOTYPE_APPLICATION_FIELDS.get(assigned_prototype_id)
    if not field_name:
        return
    app_raw = candidate.get(field_name)
    if not isinstance(app_raw, dict):
        if structural_errors is not None:
            _append_methodology_structural(
                structural_errors,
                "builder2_creator_validation_failed",
                field_name,
            )
            return
        _raise("builder2_creator_validation_failed", field=field_name)
    app = app_raw

    def require_field(key: str) -> str:
        try:
            return _require_text(app.get(key), field=f"{field_name}.{key}", code="builder2_creator_validation_failed")
        except Builder2TournamentError:
            if structural_errors is not None:
                _append_methodology_structural(
                    structural_errors,
                    "builder2_creator_validation_failed",
                    f"{field_name}.{key}",
                )
                return ""
            raise

    if assigned_prototype_id == "winning_card":
        medium = require_field("mediumOrContainerIdentified")
        require_field("whatItBecomes")
        require_field("whyTheTransformationProvesTheAdvantage")
        if structural_errors is None and medium:
            medium_lower = medium.lower()
            if any(marker in medium_lower for marker in _PLAYING_CARD_SURFACE_MARKERS):
                if not str(app.get("whatItBecomes") or "").strip():
                    _raise("builder2_creator_validation_failed", field="winning_card.literal_card_imitation")

    elif assigned_prototype_id == "summer_fan":
        for key in ("visibleBehavior", "inferredAbsentObject", "whyTheViewerInfersItWithoutExplanation"):
            require_field(key)

    elif assigned_prototype_id == "forgot":
        for key in ("omittedOrForgottenAction", "visibleConsequence", "whyTheViewerSolvesIt"):
            require_field(key)
        if structural_errors is None:
            contradiction = str(app.get("plannedContradiction") or "").strip()
            consequence = str(app.get("visibleConsequence") or "").strip()
            if not contradiction and not consequence:
                _raise("builder2_creator_validation_failed", field="forgotApplication.plannedContradiction")

    elif assigned_prototype_id == "greenpeace_essential_pairing":
        for key in (
            "elementA",
            "elementB",
            "essentialRelationship",
            "notMerelyAppearance",
            "notMerelyFunction",
            "notMerelyWordplay",
            "emotionalRecognition",
        ):
            require_field(key)
        if structural_errors is None:
            appearance_decl = _normalize_text(app.get("notMerelyAppearance"))
            if "shape only" in appearance_decl or appearance_decl == "appearance only":
                _raise("builder2_creator_validation_failed", field="essential_pairing.shape_only")

    elif assigned_prototype_id == "closest":
        for key in (
            "admittedGap",
            "relativeNearness",
            "physicalOrVisualExpressionOfNearness",
            "whyThisIsHonestRatherThanInferior",
        ):
            require_field(key)

    elif assigned_prototype_id == "think_small":
        for key in (
            "realWeakness",
            "evidenceTheWeaknessIsReal",
            "acceptanceRatherThanDenial",
            "reframing",
            "relativeAdvantageCreated",
        ):
            require_field(key)
        if structural_errors is None:
            denial_patterns = ("not small", "not a weakness", "denies weakness", "denial of weakness")
            blob = _normalize_text(json.dumps(app, ensure_ascii=False))
            if any(p in blob for p in denial_patterns):
                _raise("builder2_creator_validation_failed", field="think_small.denial_of_weakness")
            if "invented weakness" in blob or "invented cosmetic weakness" in blob:
                _raise("builder2_creator_validation_failed", field="think_small.invented_weakness")


def collect_creator_methodology_structural_errors(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> List[str]:
    if compatibility_mode and not uses_full_methodology(candidate):
        return []

    errors: List[str] = []

    def require_dict_field(value: Any, field: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            _append_methodology_structural(errors, "builder2_creator_validation_failed", field)
            return {}
        return value

    def require_text_field(value: Any, field: str) -> None:
        try:
            _require_text(value, field=field, code="builder2_creator_validation_failed")
        except Builder2TournamentError:
            _append_methodology_structural(errors, "builder2_creator_validation_failed", field)

    report = require_dict_field(candidate.get("creatorReport"), "creatorReport")
    require_text_field(report.get("mechanismScanSummary"), "creatorReport.mechanismScanSummary")
    require_text_field(candidate.get("visualMechanism"), "visualMechanism")

    runway = require_dict_field(candidate.get("runwayFeasibility"), "runwayFeasibility")
    for bool_key in ("fitsSevenSeconds", "requiresImpossibleMorphing", "requiresSubtleUnseenInference"):
        if bool_key not in runway:
            _append_methodology_structural(errors, "builder2_creator_schema_invalid", f"runwayFeasibility.{bool_key}")

    verbal = candidate.get("verbalPotential")
    if verbal is None:
        _append_methodology_structural(errors, "builder2_creator_validation_failed", "verbalPotential")
    elif not isinstance(verbal, dict):
        _append_methodology_structural(errors, "builder2_creator_validation_failed", "verbalPotential")
    else:
        decision = _normalize_text(verbal.get("decision")).lower()
        if decision not in VALID_VERBAL_DECISIONS:
            decision = "available" if _normalize_text(verbal.get("keywordOrKeyPhrase")) else "not_needed"
        if decision == "available":
            for key in ("keywordOrKeyPhrase", "visualMeaning", "strategicMeaning"):
                require_text_field(verbal.get(key), f"verbalPotential.{key}")
        else:
            require_text_field(verbal.get("reason"), "verbalPotential.reason")

    structure = str(candidate.get("structureType") or "")
    if structure == "variation_montage":
        for key in ("visualFamilyId", "visualFamilyDefinition", "recurringMotif", "sceneVariations"):
            if key not in candidate or candidate.get(key) in (None, "", []):
                _append_methodology_structural(errors, "builder2_creator_validation_failed", key)

    _validate_prototype_application(
        candidate,
        assigned_prototype_id=assigned_prototype_id,
        structural_errors=errors,
    )

    if strategy_foundation is not None:
        try:
            validate_strategy_identity(
                expected_strategy_foundation_id=expected_strategy_foundation_id(strategy_foundation),
                candidate=candidate,
            )
        except Builder2TournamentError as exc:
            msg = str(exc.args[0] if exc.args else "builder2_creator_validation_failed:strategyFoundationId")
            if msg not in errors:
                errors.append(msg)

        from engine.builder2_essential_fact_fusion import collect_creator_fusion_structural_errors

        errors.extend(
            collect_creator_fusion_structural_errors(
                strategy_foundation=strategy_foundation,
                candidate=candidate,
                compatibility_mode=compatibility_mode,
            )
        )

    return list(dict.fromkeys(errors))


def _validate_judge_verbal_layer(
    judgment: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> None:
    from engine.builder2_judge_core_contract import (
        VERBAL_ASSESSMENT_BOOLEAN_FIELDS,
        VALID_VERBAL_APPLICABILITY,
        resolve_creator_verbal_decision,
        resolve_verbal_applicability,
    )

    def fail(field: str, code: str = "builder2_judge_validation_failed") -> None:
        if errors is not None:
            msg = f"{code}:{field}"
            if msg not in errors:
                errors.append(msg)
            return
        _raise(code, field=field)

    verbal_raw = judgment.get("verbalLayerAssessment")
    if not isinstance(verbal_raw, dict):
        fail("verbalLayerAssessment", code="builder2_judge_schema_invalid")
        return

    creator_decision = resolve_creator_verbal_decision(candidate)
    applicability = resolve_verbal_applicability(verbal_raw, creator_verbal_decision=creator_decision)
    if applicability not in VALID_VERBAL_APPLICABILITY:
        fail("verbalLayerAssessment.applicability")
        return

    verbal = dict(verbal_raw)
    if verbal.get("applicability") != applicability:
        verbal["applicability"] = applicability

    if applicability == "available":
        for key in VERBAL_ASSESSMENT_BOOLEAN_FIELDS:
            value = verbal.get(key)
            if not isinstance(value, bool):
                fail(f"verbalLayerAssessment.{key}", code="builder2_judge_schema_invalid")
        if any(verbal.get(key) is False for key in VERBAL_ASSESSMENT_BOOLEAN_FIELDS):
            if not str(verbal.get("notes") or "").strip():
                fail("verbalLayerAssessment.notes")
        elif not str(verbal.get("notes") or "").strip():
            fail("verbalLayerAssessment.notes")
    else:
        for key in VERBAL_ASSESSMENT_BOOLEAN_FIELDS:
            value = verbal.get(key)
            if value is not None and not isinstance(value, bool):
                fail(f"verbalLayerAssessment.{key}", code="builder2_judge_schema_invalid")
        if not str(verbal.get("notes") or "").strip():
            fail("verbalLayerAssessment.notes")

    judgment["verbalLayerAssessment"] = verbal


def _validate_judge_methodology_coherence(judgment: Dict[str, Any]) -> None:
    verbal = judgment.get("verbalLayerAssessment") or {}
    headline = judgment.get("headlineNecessityAssessment") or {}
    eligible = judgment.get("eligible") is True

    if eligible and any(
        verbal.get(key) is False
        for key in ("keywordBornFromVisual", "visualMeaningIsClear", "strategicMeaningIsClear", "twoMeaningsReinforceEachOther")
    ):
        if not str(verbal.get("notes") or "").strip():
            _raise("builder2_judge_coherence_violation", field="verbalLayerAssessment.notes")

    silent_score = int((judgment.get("scores") or {}).get("silentVisualClarity") or 0)
    if headline.get("visualWouldWorkWithoutHeadline") is False and silent_score >= 12 and eligible:
        if not str(headline.get("notes") or "").strip():
            _raise("builder2_judge_coherence_violation", field="headlineNecessityAssessment.notes")


def collect_judge_methodology_structural_errors(
    judgment: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]] = None,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> List[str]:
    errors: List[str] = []
    if compatibility_mode and not uses_full_methodology(judgment):
        return errors

    for key in (
        "problemAdvantageAssessment",
        "mechanismDepthAssessment",
        "prototypeMethodAssessment",
        "visualMechanismAssessment",
        "participationAssessment",
        "visualFamilyAssessment",
        "silentMovieAssessment",
    ):
        try:
            _require_text(judgment.get(key), field=key, code="builder2_judge_validation_failed")
        except Builder2TournamentError as exc:
            msg = str(exc.args[0] if exc.args else f"builder2_judge_validation_failed:{key}")
            if msg not in errors:
                errors.append(msg)

    _validate_judge_verbal_layer(judgment, candidate=candidate, errors=errors)

    headline = judgment.get("headlineNecessityAssessment")
    if not isinstance(headline, dict):
        errors.append("builder2_judge_validation_failed:headlineNecessityAssessment")
    else:
        headline_needed = headline.get("headlineNeeded")
        if headline_needed is None and isinstance(headline.get("headlineRecommended"), bool):
            headline_needed = headline.get("headlineRecommended")
        if not isinstance(headline_needed, bool):
            errors.append("builder2_judge_validation_failed:headlineNecessityAssessment.headlineNeeded")
        if not isinstance(headline.get("visualWouldWorkWithoutHeadline"), bool):
            errors.append("builder2_judge_validation_failed:headlineNecessityAssessment.visualWouldWorkWithoutHeadline")
        if not str(headline.get("notes") or "").strip():
            errors.append("builder2_judge_validation_failed:headlineNecessityAssessment.notes")

    advertising = judgment.get("advertisingCompletionAssessment")
    if not isinstance(advertising, dict):
        errors.append("builder2_judge_validation_failed:advertisingCompletionAssessment")
    else:
        for key in (
            "advertiserIdentifiable",
            "productNamePresent",
            "relativeAdvantageClosed",
            "sloganSpecificToIdea",
            "functionsAsAdvertisement",
        ):
            if not isinstance(advertising.get(key), bool):
                errors.append(f"builder2_judge_validation_failed:advertisingCompletionAssessment.{key}")
        if not str(advertising.get("notes") or "").strip():
            errors.append("builder2_judge_validation_failed:advertisingCompletionAssessment.notes")

    slogan = judgment.get("advertisingSloganAssessment")
    if not isinstance(slogan, dict):
        errors.append("builder2_judge_validation_failed:advertisingSloganAssessment")
    else:
        from engine.builder2_advertising_slogan_quality_contract import JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS

        for key in JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS:
            if not isinstance(slogan.get(key), bool):
                errors.append(f"builder2_judge_validation_failed:advertisingSloganAssessment.{key}")
        if not str(slogan.get("notes") or "").strip():
            errors.append("builder2_judge_validation_failed:advertisingSloganAssessment.notes")

    semantic = judgment.get("semanticAlignmentAssessment")
    if not isinstance(semantic, dict):
        errors.append("builder2_judge_validation_failed:semanticAlignmentAssessment")
    else:
        for key in (
            "visualMeaning",
            "sloganMeaning",
            "combinedAdvertisingMeaning",
            "sameStrategicPromise",
            "sloganCompletesRatherThanChangesVisual",
            "understandableWithoutCreatorReport",
            "keyWordMeaningsConnected",
            "semanticAlignment",
        ):
            value = semantic.get(key)
            if key == "semanticAlignment":
                if not isinstance(value, bool):
                    errors.append(f"builder2_judge_validation_failed:semanticAlignmentAssessment.{key}")
            elif not isinstance(value, bool) and not str(value or "").strip():
                errors.append(f"builder2_judge_validation_failed:semanticAlignmentAssessment.{key}")

    prototype_assessment = judgment.get("prototypeApplicationAssessment")
    if not isinstance(prototype_assessment, dict):
        errors.append("builder2_judge_validation_failed:prototypeApplicationAssessment")
    else:
        for key in (
            "assignedPrototypeId",
            "prototypeMethodVisibleInFilm",
            "prototypeMethodReinforcedBySlogan",
            "applicationFeelsIntrinsic",
            "applicationRequiresRetrospectiveExplanation",
        ):
            value = prototype_assessment.get(key)
            if key == "assignedPrototypeId":
                if not str(value or "").strip():
                    errors.append(f"builder2_judge_validation_failed:prototypeApplicationAssessment.{key}")
            elif not isinstance(value, bool):
                errors.append(f"builder2_judge_validation_failed:prototypeApplicationAssessment.{key}")
        fit_score = prototype_assessment.get("prototypeFitScore")
        if fit_score is None:
            fit_score = (judgment.get("scores") or {}).get("prototypeMethodApplication")
        if not isinstance(fit_score, int) or fit_score < 0 or fit_score > 15:
            errors.append("builder2_judge_validation_failed:prototypeApplicationAssessment.prototypeFitScore")

    from engine.builder2_strategy_evidence_grounding_contract import collect_judge_factual_grounding_structural_errors

    errors.extend(
        collect_judge_factual_grounding_structural_errors(
            judgment,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
        )
    )

    if strategy_foundation is not None:
        from engine.builder2_essential_fact_fusion import collect_judge_fusion_structural_errors

        errors.extend(
            collect_judge_fusion_structural_errors(
                strategy_foundation=strategy_foundation,
                judgment=judgment,
                compatibility_mode=compatibility_mode,
            )
        )

    return list(dict.fromkeys(errors))


def validate_judge_methodology(
    judgment: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]] = None,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    product_input: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> None:
    if compatibility_mode and not uses_full_methodology(judgment):
        logger.info("BUILDER2_METHODOLOGY_COMPATIBILITY_MODE role=judge")
        return

    errors = collect_judge_methodology_structural_errors(
        judgment,
        candidate=candidate,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    if errors:
        first = errors[0]
        if ":" in first:
            code, field = first.split(":", 1)
            _raise(code, field=field)
        _raise("builder2_judge_validation_failed", field=first)

    _validate_judge_methodology_coherence(judgment)
    from engine.builder2_advertising_closure_contract import validate_judge_advertising_completion_assessment

    validate_judge_advertising_completion_assessment(judgment)
    from engine.builder2_advertising_slogan_quality_contract import validate_judge_advertising_slogan_assessment

    validate_judge_advertising_slogan_assessment(judgment)
    from engine.builder2_metaphorical_embodiment_contract import validate_judge_metaphorical_embodiment

    if isinstance(judgment.get("metaphoricalEmbodimentAssessment"), dict):
        validate_judge_metaphorical_embodiment(judgment, candidate=candidate)
    from engine.builder2_no_logo_contract import validate_judge_logo_policy

    if isinstance(judgment.get("logoPolicyAssessment"), dict):
        validate_judge_logo_policy(judgment, candidate=candidate)
    from engine.builder2_strategy_evidence_grounding_contract import (
        validate_judge_factual_grounding_assessment,
    )

    strategy_foundation = strategy_foundation if isinstance(strategy_foundation, dict) else None
    validate_judge_factual_grounding_assessment(
        judgment,
        candidate=candidate,
        strategy_foundation=strategy_foundation,
        product_input=product_input,
        compatibility_mode=compatibility_mode,
    )
    logger.info("BUILDER2_JUDGE_METHODOLOGY_VALIDATED")


def _validate_winner_preservation_deterministic(
    winner_plan: Dict[str, Any],
    *,
    preservation_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY

    owned = winner_plan.get(SERVER_OWNED_WINNER_SOURCE_KEY)
    if not isinstance(owned, dict) or not owned:
        _raise("builder2_winner_preservation_contract_missing", field=SERVER_OWNED_WINNER_SOURCE_KEY)

    if preservation_snapshot:
        identity_fields = ("strategyFoundationId", "prototypeId", "structureType", "visualParallelType")
        for key in identity_fields:
            expected = preservation_snapshot.get(key)
            if key == "prototypeId":
                actual = owned.get("sourcePrototypeId") or winner_plan.get("prototypeId")
            else:
                actual = owned.get(key) if key in owned else winner_plan.get(key)
            if str(actual or "") != str(expected or ""):
                _raise("builder2_winner_source_identity_mismatch", field=key)

    server_check = winner_plan.get("serverPreservationCheck")
    if not isinstance(server_check, dict):
        _raise("builder2_winner_preservation_contract_missing", field="serverPreservationCheck")
    if str(server_check.get("source") or "") != "server_owned_contract":
        _raise("builder2_winner_preservation_contract_missing", field="serverPreservationCheck.source")

    preserved_mechanism = str(owned.get("coreCreativeMechanism") or winner_plan.get("coreCreativeMechanism") or "").strip()
    if not preserved_mechanism:
        _raise("builder2_winner_preservation_contract_missing", field="coreCreativeMechanism")


def validate_winner_methodology(
    winner_plan: Dict[str, Any],
    *,
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> None:
    if compatibility_mode and not uses_full_methodology(winner_plan):
        logger.info("BUILDER2_METHODOLOGY_COMPATIBILITY_MODE role=winner")
        return

    decision_obj = _require_dict(
        winner_plan.get("headlineDecision"),
        field="headlineDecision",
        code="builder2_winner_validation_failed",
    )
    from engine.builder2_headline_decision_contract import validate_headline_decision_methodology

    validate_headline_decision_methodology(
        winner_plan,
        winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else None,
        winning_candidate=winning_candidate,
        tournament_state=tournament_state,
    )
    from engine.builder2_advertising_closure_contract import validate_advertising_closure_methodology

    validate_advertising_closure_methodology(winner_plan, require_present=False)
    from engine.builder2_single_slogan_contract import is_single_slogan_contract, validate_single_slogan_plan_contract

    if is_single_slogan_contract(state=tournament_state, plan=winner_plan):
        ok, failures = validate_single_slogan_plan_contract(winner_plan, state=tournament_state)
        if not ok and failures:
            _raise("builder2_winner_validation_failed", field=failures[0])
    from engine.builder2_advertising_slogan_quality_contract import (
        BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
        validate_winner_advertising_slogan_evidence,
    )

    requires_slogan_quality = (
        str((tournament_state or {}).get("advertisingSloganQualityContractVersion") or "").strip()
        == BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION
    )
    if requires_slogan_quality or isinstance(winner_plan.get("advertisingSloganEvidence"), dict):
        strategy_foundation = None
        if isinstance(tournament_state, dict):
            strategy_foundation = tournament_state.get("strategyFoundation")
        validate_winner_advertising_slogan_evidence(
            winner_plan,
            winning_candidate=winning_candidate,
            strategy_foundation=strategy_foundation if isinstance(strategy_foundation, dict) else None,
        )
    strategy_foundation = tournament_state.get("strategyFoundation") if isinstance(tournament_state, dict) else None
    if isinstance(strategy_foundation, dict):
        from engine.builder2_strategy_evidence_grounding_contract import validate_winner_evidence_grounding

        product_input = None
        grounding = strategy_foundation.get("strategyEvidenceGrounding")
        if isinstance(grounding, dict):
            product_input = grounding.get("productInputAudit")
        validate_winner_evidence_grounding(
            winner_plan,
            strategy_foundation=strategy_foundation,
            winning_candidate=winning_candidate,
            product_input=product_input if isinstance(product_input, dict) else None,
            compatibility_mode=compatibility_mode,
        )
    decision = str((winner_plan.get("headlineDecision") or {}).get("decision") or "")

    _validate_winner_preservation_deterministic(
        winner_plan,
        preservation_snapshot=preservation_snapshot,
    )
    logger.info("BUILDER2_WINNER_MECHANISM_PRESERVED prototypeId=%s", winner_plan.get("prototypeId"))


def infer_process_failure_tag(failure_reason: Optional[str]) -> Optional[str]:
    if not failure_reason:
        return None
    reason = failure_reason.lower()
    if "strategyfoundationid" in reason or "strategy_identity" in reason:
        return "strategy_identity_mismatch"
    if "winner_validation_failed" in reason and "mechanism" in reason:
        return "winner_mechanism_changed"
    if "scenevariations" in reason or "visual_family" in reason or "montage" in reason:
        return "visual_family_incoherent"
    if "not_grounded" in reason or "problemperception" in reason:
        return "problem_not_grounded"
    if "derivationfromproblem" in reason or "advantage_not" in reason:
        return "advantage_not_derived"
    if "depth" in reason or "mechanism_too" in reason:
        return "mechanism_too_surface"
    if "surface" in reason or "literal" in reason or "imitation" in reason:
        return "prototype_surface_copy"
    if "silent" in reason or "audio" in reason:
        return "visual_not_silent"
    if "runway" in reason or "morph" in reason:
        return "runway_infeasible"
    if "headline" in reason:
        return "headline_rescuing_visual"
    if "headline_composition_invalid" in reason:
        return "headline_composition_invalid"
    if "pre_runway_validation_failed" in reason or "pre_runway_contract_invalid" in reason:
        return "pre_runway_contract_invalid"
    if "winner_downstream_invalid" in reason or "downstream_type_mismatch" in reason:
        return "winner_downstream_type_mismatch"
    return None
