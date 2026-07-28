"""
Builder2 creative embodiment contract — tournament Creator/Judge enforcement.

Permits external metaphors and transformed product/domain embodiment. The viewer
must experience the strategic perception physically — not via literal diagrams.
Quantity/quality and example objects remain teaching examples only.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional

from engine.builder2_single_slogan_contract import (
    classify_literal_domain_symbols,
    is_single_slogan_contract,
    raise_single_slogan_contract_error,
)

logger = logging.getLogger(__name__)

CREATOR_METAPHOR_FIELDS = (
    "strategicPerception",
    "obviousLiteralVisualSymbols",
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


def _collect_literal_symbols(candidate: Dict[str, Any]) -> List[str]:
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
        raw = metaphor.get("obviousLiteralVisualSymbols")
        if isinstance(raw, list):
            parts.extend(_clean(item) for item in raw)
        elif raw:
            parts.append(_clean(raw))
    hits: List[str] = []
    for part in parts:
        hits.extend(classify_literal_domain_symbols(part))
    return list(dict.fromkeys(hits))



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

    literal_hits = _collect_literal_symbols(candidate)
    rejected = metaphor.get("literalSymbolsRejectedOrTransformed")
    rejected_text = " ".join(rejected) if isinstance(rejected, list) else _clean(rejected)
    transformation_accepted = "transform" in rejected_text.lower() or "reject" in rejected_text.lower()
    if literal_hits and not transformation_accepted:
        raise_single_slogan_contract_error(
            "builder2_creator_literal_execution_without_transformation",
            field="metaphoricalEmbodiment.literalSymbolsRejectedOrTransformed",
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
    hits = _collect_literal_symbols(candidate)
    if not hits:
        return False
    rejected = metaphor.get("literalSymbolsRejectedOrTransformed") if isinstance(metaphor, dict) else ""
    rejected_text = " ".join(rejected) if isinstance(rejected, list) else _clean(rejected)
    return "transform" not in rejected_text.lower()


def judgment_rejects_literal_execution(judgment: Dict[str, Any]) -> bool:
    metaphor = judgment.get("metaphoricalEmbodimentAssessment")
    if not isinstance(metaphor, dict):
        return False
    if metaphor.get("creativeEmbodimentAccepted") is False:
        return True
    if metaphor.get("literalExecutionDetected") is True and metaphor.get("literalPresentationMeaningfullyTransformed") is not True:
        return True
    return False
