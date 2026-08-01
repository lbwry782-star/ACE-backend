"""
Builder2 creative embodiment contract — tournament Creator/Judge enforcement.

Permits external metaphors and transformed product/domain embodiment. The viewer
must experience the strategic perception physically — not via literal diagrams.
Quantity/quality and example objects remain teaching examples only.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from engine.builder2_single_slogan_contract import (
    classify_literal_domain_symbols,
    is_single_slogan_contract,
    raise_single_slogan_contract_error,
)

logger = logging.getLogger(__name__)

BUILDER2_LITERAL_SYMBOL_DISPOSITION_CONTRACT_VERSION = "builder2_literal_symbol_disposition_v1"

VALID_LITERAL_SYMBOL_DISPOSITIONS: FrozenSet[str] = frozenset(
    {
        "not_present",
        "rejected",
        "transformed",
        "untransformed",
    }
)

LITERAL_SYMBOL_DISPOSITION_FIELD = "literalSymbolDisposition"

TRANSFORMATION_EVIDENCE_KEYWORDS: Tuple[str, ...] = (
    "reject",
    "rejected",
    "transform",
    "transformed",
    "instead",
    "rather than",
    "replaced",
    "converted",
    "physical",
    "embod",
    "without using",
    "not shown",
    "excluded",
    "avoid",
    "דח",
    "דחי",
    "המר",
    "המרה",
    "פיזי",
    "במקום",
    "ללא",
)

CREATOR_METAPHOR_FIELDS = (
    "strategicPerception",
    "obviousLiteralVisualSymbols",
    LITERAL_SYMBOL_DISPOSITION_FIELD,
    "literalSymbolsRejectedOrTransformed",
    "creativeEmbodimentMode",
    "embodimentSubjectOrWorld",
    "physicalEmbodiment",
    "embodiedStrategicRelationship",
    "visiblePhysicalRelationship",
    "transformationMechanism",
    "whyTheVisualIsNotLiteralExplanation",
    "understandableBeforeSlogan",
    "sloganBridgeToBusinessMeaning",
)

CREATOR_VISUAL_BRIDGE_FIELDS = (
    "centralVisibleDetail",
    "sloganConnectionToVisibleDetail",
    "sloganConnectionToRelativeAdvantage",
    "dependsOnEarlierCopy",
    "singleSloganContractSatisfied",
)

JUDGE_METAPHOR_FIELDS = (
    "literalExecutionDetected",
    "literalPresentationMeaningfullyTransformed",
    "creativeEmbodimentModeAccepted",
    "physicalEmbodimentMatchesStrategicRelationship",
    "viewerDiscoveryPresent",
    "sloganOnlyBridgesNotExplains",
    "creativeEmbodimentAccepted",
    "rejectionReason",
)

JUDGE_VISUAL_BRIDGE_FIELDS = CREATOR_VISUAL_BRIDGE_FIELDS

SUGGESTED_CREATIVE_EMBODIMENT_MODES: FrozenSet[str] = frozenset(
    {
        "external_metaphor",
        "transformed_product",
        "transformed_medium",
        "transformed_context",
        "transformed_action_or_motion",
        "absence_or_planned_absurdity",
        "essential_pairing",
    }
)

PROTOTYPE_EMBODIMENT_MODE_HINTS: Dict[str, str] = {
    "winning_card": "transformed_medium",
    "summer_fan": "transformed_action_or_motion",
    "greenpeace_essential_pairing": "essential_pairing",
    "forgot": "absence_or_planned_absurdity",
    "closest": "transformed_action_or_motion",
    "think_small": "transformed_product",
    "context_collision": "transformed_context",
}

DEPRECATED_EXAMPLE_ONLY_CREATOR_FIELDS = frozenset(
    {
        "quantityEmbodiment",
        "qualityOrAdvantageEmbodiment",
        "qualityEmbodiment",
        "sameOrParallelObjectFamily",
        "embodiedRelationship",
        "canonicalSloganBridge",
        "metaphoricalWorld",
        "metaphoricalPhysicalFamily",
    }
)

DEPRECATED_EXAMPLE_ONLY_JUDGE_FIELDS = frozenset(
    {
        "sameOrParallelFamilyAccepted",
        "metaphoricalWorldDistinctFromBusinessDomain",
        "metaphoricalEmbodimentAccepted",
        "embodiedStrategicRelationshipAccepted",
        "physicalRelationshipMatchesStrategicRelationship",
        "literalSymbolTransformationAccepted",
        "silentMetaphorClarity",
    }
)

THINK_SMALL_INVALID_PATTERNS = (
    re.compile(r"size comparison diagram", re.I),
    re.compile(r"measurement label", re.I),
    re.compile(r"size chart", re.I),
    re.compile(r"spec(?:ification)? sheet", re.I),
    re.compile(r"ordinary product (?:photo|shot|image)", re.I),
    re.compile(r"ruler (?:showing|measuring)", re.I),
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _require_text(value: Any, *, field: str) -> str:
    text = _clean(value)
    if not text:
        raise_single_slogan_contract_error("builder2_metaphor_validation_failed", field=field)
    return text


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise_single_slogan_contract_error("builder2_metaphor_validation_failed", field=field)
    return value


def requires_literal_symbol_disposition(*, tournament_state: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(tournament_state, dict):
        return False
    return (
        _clean(tournament_state.get("metaphoricalEmbodimentContractVersion"))
        == BUILDER2_LITERAL_SYMBOL_DISPOSITION_CONTRACT_VERSION
    )


def stamp_literal_symbol_disposition_contract(state: Dict[str, Any]) -> None:
    state["metaphoricalEmbodimentContractVersion"] = BUILDER2_LITERAL_SYMBOL_DISPOSITION_CONTRACT_VERSION


def _execution_text_parts(candidate: Dict[str, Any]) -> List[str]:
    parts = [
        _clean(candidate.get("coreVisualIdea")),
        _clean(candidate.get("visualMechanism")),
        _clean(candidate.get("openingFrameDescription")),
    ]
    sequence = candidate.get("sequence")
    if isinstance(sequence, dict):
        parts.extend(_clean(sequence.get(key)) for key in ("beginning", "development", "resolution"))
    report = candidate.get("creatorReport")
    if isinstance(report, dict):
        parts.append(_clean(report.get("mechanismScanSummary")))
    metaphor = candidate.get("metaphoricalEmbodiment")
    if isinstance(metaphor, dict):
        for key in (
            "physicalEmbodiment",
            "embodimentSubjectOrWorld",
            "visiblePhysicalRelationship",
        ):
            parts.append(_clean(metaphor.get(key)))
    return [part for part in parts if part]


def _declared_literal_symbol_parts(metaphor: Dict[str, Any]) -> List[str]:
    raw = metaphor.get("obviousLiteralVisualSymbols")
    if isinstance(raw, list):
        return [_clean(item) for item in raw if _clean(item)]
    if raw:
        return [_clean(raw)]
    return []


def _collect_execution_literal_symbols(candidate: Dict[str, Any]) -> List[str]:
    hits: List[str] = []
    for part in _execution_text_parts(candidate):
        hits.extend(classify_literal_domain_symbols(part))
    return list(dict.fromkeys(hits))


def _collect_declared_literal_symbols(metaphor: Dict[str, Any]) -> List[str]:
    hits: List[str] = []
    for part in _declared_literal_symbol_parts(metaphor):
        hits.extend(classify_literal_domain_symbols(part))
    return list(dict.fromkeys(hits))


def _collect_literal_symbols(candidate: Dict[str, Any]) -> List[str]:
    """Backward-compatible alias: execution hits only (declared symbols are not execution)."""
    return _collect_execution_literal_symbols(candidate)


def _literal_transformation_evidence_text(metaphor: Dict[str, Any]) -> str:
    rejected = metaphor.get("literalSymbolsRejectedOrTransformed")
    rejected_text = " ".join(rejected) if isinstance(rejected, list) else _clean(rejected)
    return " ".join(
        part
        for part in (
            rejected_text,
            _clean(metaphor.get("transformationMechanism")),
            _clean(metaphor.get("whyTheVisualIsNotLiteralExplanation")),
            _clean(metaphor.get("physicalEmbodiment")),
            _clean(metaphor.get("visiblePhysicalRelationship")),
        )
        if part
    )


def _has_literal_transformation_evidence(text: str) -> bool:
    lowered = _clean(text).lower()
    if not lowered:
        return False
    return any(keyword in lowered for keyword in TRANSFORMATION_EVIDENCE_KEYWORDS)


def _infer_legacy_literal_symbol_disposition(
    *,
    metaphor: Dict[str, Any],
    execution_hits: List[str],
    declared_hits: List[str],
) -> str:
    evidence = _literal_transformation_evidence_text(metaphor)
    if execution_hits:
        if _has_literal_transformation_evidence(evidence):
            return "transformed"
        return "untransformed"
    if declared_hits:
        if _has_literal_transformation_evidence(evidence):
            return "rejected"
        return "untransformed"
    return "not_present"


def _validate_literal_symbol_disposition(
    candidate: Dict[str, Any],
    *,
    metaphor: Dict[str, Any],
    tournament_state: Optional[Dict[str, Any]] = None,
) -> None:
    execution_hits = _collect_execution_literal_symbols(candidate)
    declared_hits = _collect_declared_literal_symbols(metaphor)
    evidence = _literal_transformation_evidence_text(metaphor)
    rejected = metaphor.get("literalSymbolsRejectedOrTransformed")
    rejected_text = " ".join(rejected) if isinstance(rejected, list) else _clean(rejected)
    disposition = _clean(metaphor.get(LITERAL_SYMBOL_DISPOSITION_FIELD))
    requires_disposition = requires_literal_symbol_disposition(tournament_state=tournament_state)

    if requires_disposition:
        if disposition not in VALID_LITERAL_SYMBOL_DISPOSITIONS:
            raise_single_slogan_contract_error(
                "builder2_creator_metaphor_validation_failed",
                field=f"metaphoricalEmbodiment.{LITERAL_SYMBOL_DISPOSITION_FIELD}",
            )
    elif not disposition:
        if execution_hits:
            if not _has_literal_transformation_evidence(evidence):
                raise_single_slogan_contract_error(
                    "builder2_creator_literal_execution_without_transformation",
                    field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
                )
            return
        if declared_hits and not rejected_text:
            raise_single_slogan_contract_error(
                "builder2_creator_metaphor_validation_failed",
                field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
            )
        return

    if disposition == "untransformed":
        raise_single_slogan_contract_error(
            "builder2_creator_literal_execution_without_transformation",
            field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
        )

    if disposition == "not_present":
        if _declared_literal_symbol_parts(metaphor):
            raise_single_slogan_contract_error(
                "builder2_creator_literal_execution_without_transformation",
                field="metaphoricalEmbodiment.obviousLiteralVisualSymbols",
            )
        if execution_hits:
            raise_single_slogan_contract_error(
                "builder2_creator_literal_execution_without_transformation",
                field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
            )
        return

    if disposition == "rejected":
        if not _declared_literal_symbol_parts(metaphor):
            raise_single_slogan_contract_error(
                "builder2_creator_metaphor_validation_failed",
                field="metaphoricalEmbodiment.obviousLiteralVisualSymbols",
            )
        if execution_hits:
            raise_single_slogan_contract_error(
                "builder2_creator_literal_execution_without_transformation",
                field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
            )
        if not _has_literal_transformation_evidence(evidence):
            raise_single_slogan_contract_error(
                "builder2_creator_literal_execution_without_transformation",
                field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
            )
        return

    if disposition == "transformed":
        if execution_hits and not _has_literal_transformation_evidence(evidence):
            raise_single_slogan_contract_error(
                "builder2_creator_literal_execution_without_transformation",
                field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
            )
        if not _has_literal_transformation_evidence(evidence):
            raise_single_slogan_contract_error(
                "builder2_creator_metaphor_validation_failed",
                field="metaphoricalEmbodiment.transformationMechanism",
            )
        return

    raise_single_slogan_contract_error(
        "builder2_creator_literal_execution_without_transformation",
        field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
    )


def inspect_literal_symbol_disposition(
    candidate: Dict[str, Any],
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metaphor = candidate.get("metaphoricalEmbodiment")
    if not isinstance(metaphor, dict):
        return {
            "present": False,
            "executionLiteralHits": [],
            "declaredLiteralHits": [],
            "wouldPassCorrectedContract": False,
        }
    execution_hits = _collect_execution_literal_symbols(candidate)
    declared_hits = _collect_declared_literal_symbols(metaphor)
    raw_value = metaphor.get("literalSymbolsRejectedOrTransformed")
    disposition = _clean(metaphor.get(LITERAL_SYMBOL_DISPOSITION_FIELD))
    inferred = _infer_legacy_literal_symbol_disposition(
        metaphor=metaphor,
        execution_hits=execution_hits,
        declared_hits=declared_hits,
    )
    would_pass = False
    failure = ""
    try:
        _validate_literal_symbol_disposition(
            candidate,
            metaphor=metaphor,
            tournament_state=tournament_state,
        )
        would_pass = True
    except Exception as exc:
        failure = str(exc.args[0] if getattr(exc, "args", None) else exc)
    return {
        "present": True,
        "literalSymbolDisposition": disposition or None,
        "inferredLegacyDisposition": inferred,
        "literalSymbolsRejectedOrTransformed": raw_value,
        "literalSymbolsRejectedOrTransformedType": type(raw_value).__name__,
        "executionLiteralHits": execution_hits,
        "declaredLiteralHits": declared_hits,
        "transformationEvidencePresent": _has_literal_transformation_evidence(
            _literal_transformation_evidence_text(metaphor)
        ),
        "wouldPassCorrectedContract": would_pass,
        "validationFailure": failure or None,
    }


def _collect_think_small_execution_blob(candidate: Dict[str, Any]) -> str:
    parts = [
        _clean(candidate.get("coreVisualIdea")),
        _clean(candidate.get("visualMechanism")),
        _clean(candidate.get("openingFrameDescription")),
    ]
    metaphor = candidate.get("metaphoricalEmbodiment")
    if isinstance(metaphor, dict):
        for key in (
            "physicalEmbodiment",
            "transformationMechanism",
            "embodimentSubjectOrWorld",
            "visiblePhysicalRelationship",
            "centralVisibleDetail",
        ):
            parts.append(_clean(metaphor.get(key)))
    bridge = candidate.get("visualBridgeAssessment")
    if isinstance(bridge, dict):
        parts.append(_clean(bridge.get("centralVisibleDetail")))
    return " ".join(part for part in parts if part)


def _validate_think_small_embodiment(candidate: Dict[str, Any], *, assigned_prototype_id: str) -> None:
    if assigned_prototype_id != "think_small":
        return
    blob = _collect_think_small_execution_blob(candidate)
    for pattern in THINK_SMALL_INVALID_PATTERNS:
        if pattern.search(blob):
            raise_single_slogan_contract_error(
                "builder2_creator_think_small_untransformed_execution",
                field="metaphoricalEmbodiment.transformationMechanism",
            )


def validate_creator_metaphorical_embodiment(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    single_slogan_required: bool = True,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> None:
    if not single_slogan_required and not is_single_slogan_contract(plan=candidate):
        return

    metaphor = candidate.get("metaphoricalEmbodiment")
    if not isinstance(metaphor, dict):
        raise_single_slogan_contract_error("builder2_creator_metaphor_missing", field="metaphoricalEmbodiment")

    for deprecated in DEPRECATED_EXAMPLE_ONLY_CREATOR_FIELDS:
        if deprecated in metaphor and _clean(metaphor.get(deprecated)):
            raise_single_slogan_contract_error(
                "builder2_creator_deprecated_metaphor_field",
                field=f"metaphoricalEmbodiment.{deprecated}",
            )

    for field in CREATOR_METAPHOR_FIELDS:
        if field == LITERAL_SYMBOL_DISPOSITION_FIELD:
            continue
        if field in {"obviousLiteralVisualSymbols", "literalSymbolsRejectedOrTransformed"}:
            value = metaphor.get(field)
            if isinstance(value, list):
                if not value:
                    raise_single_slogan_contract_error("builder2_creator_metaphor_validation_failed", field=f"metaphoricalEmbodiment.{field}")
            else:
                _require_text(value, field=f"metaphoricalEmbodiment.{field}")
        elif field == "understandableBeforeSlogan":
            _require_bool(metaphor.get(field), field=f"metaphoricalEmbodiment.{field}")
            if metaphor.get(field) is not True:
                raise_single_slogan_contract_error("builder2_creator_metaphor_not_silent", field=field)
        else:
            _require_text(metaphor.get(field), field=f"metaphoricalEmbodiment.{field}")

    bridge = candidate.get("visualBridgeAssessment")
    if not isinstance(bridge, dict):
        raise_single_slogan_contract_error("builder2_creator_visual_bridge_missing", field="visualBridgeAssessment")

    for field in CREATOR_VISUAL_BRIDGE_FIELDS:
        if field == "dependsOnEarlierCopy":
            _require_bool(bridge.get(field), field=f"visualBridgeAssessment.{field}")
            if bridge.get(field) is not False:
                raise_single_slogan_contract_error("builder2_creator_depends_on_earlier_copy", field=field)
        elif field == "singleSloganContractSatisfied":
            _require_bool(bridge.get(field), field=f"visualBridgeAssessment.{field}")
            if bridge.get(field) is not True:
                raise_single_slogan_contract_error("builder2_creator_single_slogan_contract_unsatisfied", field=field)
        else:
            _require_text(bridge.get(field), field=f"visualBridgeAssessment.{field}")

    if requires_literal_symbol_disposition(tournament_state=tournament_state):
        disposition = _clean(metaphor.get(LITERAL_SYMBOL_DISPOSITION_FIELD))
        if disposition not in VALID_LITERAL_SYMBOL_DISPOSITIONS:
            raise_single_slogan_contract_error(
                "builder2_creator_metaphor_validation_failed",
                field=f"metaphoricalEmbodiment.{LITERAL_SYMBOL_DISPOSITION_FIELD}",
            )

    _validate_literal_symbol_disposition(
        candidate,
        metaphor=metaphor,
        tournament_state=tournament_state,
    )

    _validate_think_small_embodiment(candidate, assigned_prototype_id=assigned_prototype_id)

    logger.info(
        "BUILDER2_CREATOR_CREATIVE_EMBODIMENT_VALIDATED prototypeId=%s mode=%s perception=%s silent=%s",
        assigned_prototype_id,
        _clean(metaphor.get("creativeEmbodimentMode"))[:32],
        _clean(metaphor.get("strategicPerception"))[:40],
        str(metaphor.get("understandableBeforeSlogan") is True).lower(),
    )


def validate_judge_metaphorical_embodiment(judgment: Dict[str, Any], *, candidate: Optional[Dict[str, Any]] = None) -> None:
    metaphor = judgment.get("metaphoricalEmbodimentAssessment")
    if not isinstance(metaphor, dict):
        return

    for deprecated in DEPRECATED_EXAMPLE_ONLY_JUDGE_FIELDS:
        if deprecated in metaphor:
            raise_single_slogan_contract_error(
                "builder2_judge_deprecated_metaphor_field",
                field=f"metaphoricalEmbodimentAssessment.{deprecated}",
            )

    for field in JUDGE_METAPHOR_FIELDS:
        if field == "rejectionReason":
            continue
        value = metaphor.get(field)
        if field in {
            "literalExecutionDetected",
            "literalPresentationMeaningfullyTransformed",
            "creativeEmbodimentModeAccepted",
            "physicalEmbodimentMatchesStrategicRelationship",
            "viewerDiscoveryPresent",
            "sloganOnlyBridgesNotExplains",
            "creativeEmbodimentAccepted",
        }:
            if not isinstance(value, bool):
                raise_single_slogan_contract_error("builder2_judge_metaphor_validation_failed", field=f"metaphoricalEmbodimentAssessment.{field}")

    bridge = judgment.get("visualBridgeAssessment")
    if isinstance(bridge, dict):
        for field in JUDGE_VISUAL_BRIDGE_FIELDS:
            if field in {"dependsOnEarlierCopy", "singleSloganContractSatisfied"}:
                if not isinstance(bridge.get(field), bool):
                    raise_single_slogan_contract_error("builder2_judge_visual_bridge_validation_failed", field=f"visualBridgeAssessment.{field}")
            elif field != "dependsOnEarlierCopy":
                if not _clean(bridge.get(field)):
                    raise_single_slogan_contract_error("builder2_judge_visual_bridge_validation_failed", field=f"visualBridgeAssessment.{field}")
        if bridge.get("dependsOnEarlierCopy") is True:
            raise_single_slogan_contract_error("builder2_judge_depends_on_earlier_copy", field="visualBridgeAssessment.dependsOnEarlierCopy")


def apply_metaphorical_eligibility_rules(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    metaphor = out.get("metaphoricalEmbodimentAssessment")
    if not isinstance(metaphor, dict):
        return out

    literal = metaphor.get("literalExecutionDetected") is True
    transformed = metaphor.get("literalPresentationMeaningfullyTransformed") is True
    accepted = metaphor.get("creativeEmbodimentAccepted") is True
    mode_accepted = metaphor.get("creativeEmbodimentModeAccepted") is True
    embodiment_matches = metaphor.get("physicalEmbodimentMatchesStrategicRelationship") is True
    bridges_only = metaphor.get("sloganOnlyBridgesNotExplains") is True

    if literal and not transformed:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("literal_execution_without_transformation")
    elif not accepted:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("creative_embodiment_rejected")
    elif not mode_accepted:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("creative_embodiment_mode_rejected")
    elif not embodiment_matches:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("physical_embodiment_mismatch_rejected")
    elif not bridges_only:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("slogan_creates_meaning_instead_of_bridging")

    bridge = out.get("visualBridgeAssessment")
    if isinstance(bridge, dict) and out.get("eligible") is True:
        if bridge.get("dependsOnEarlierCopy") is True:
            out["eligible"] = False
            out.setdefault("disqualifiers", []).append("depends_on_earlier_copy")
        if bridge.get("singleSloganContractSatisfied") is False:
            out["eligible"] = False
            out.setdefault("disqualifiers", []).append("single_slogan_contract_unsatisfied")

    return out


def candidate_literal_execution_detected(candidate: Dict[str, Any]) -> bool:
    metaphor = candidate.get("metaphoricalEmbodiment")
    if isinstance(metaphor, dict) and isinstance(metaphor.get("literalExecutionDetected"), bool):
        return metaphor.get("literalExecutionDetected") is True
    if not isinstance(metaphor, dict):
        return bool(_collect_execution_literal_symbols(candidate))
    disposition = _clean(metaphor.get(LITERAL_SYMBOL_DISPOSITION_FIELD))
    if disposition == "untransformed":
        return True
    if disposition in {"not_present", "rejected", "transformed"}:
        return False
    execution_hits = _collect_execution_literal_symbols(candidate)
    if not execution_hits:
        return False
    return not _has_literal_transformation_evidence(_literal_transformation_evidence_text(metaphor))


def judgment_rejects_literal_execution(judgment: Dict[str, Any]) -> bool:
    metaphor = judgment.get("metaphoricalEmbodimentAssessment")
    if not isinstance(metaphor, dict):
        return False
    if metaphor.get("creativeEmbodimentAccepted") is False:
        return True
    if metaphor.get("literalExecutionDetected") is True and metaphor.get("literalPresentationMeaningfullyTransformed") is not True:
        return True
    return False
