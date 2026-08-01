"""
Builder2 Judge structural repair classifier — structural defects vs substantive negatives.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.builder2_judge_core_contract import (
    JUDGE_FACTUAL_GROUNDING_GATE_FIELDS,
    is_judge_factual_grounding_gate_field,
    is_judge_structural_repair_field,
)
BUILDER2_JUDGE_STRUCTURAL_REPAIR_CLASSIFIER_VERSION = "builder2_judge_structural_repair_classifier_v1"


def _assessment(parsed: Optional[Dict[str, Any]]) -> Any:
    return (parsed or {}).get("factualGroundingAssessment")


def is_factual_grounding_object_structurally_defective(parsed: Optional[Dict[str, Any]]) -> bool:
    assessment = _assessment(parsed)
    if not isinstance(assessment, dict):
        return True
    if not assessment:
        return True
    for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
        value = assessment.get(key)
        if not isinstance(value, bool):
            return True
    if not str(assessment.get("notes") or "").strip():
        return True
    return False


def is_substantive_factual_grounding_negative(field: Optional[str], parsed: Optional[Dict[str, Any]]) -> bool:
    if not field or not is_judge_factual_grounding_gate_field(field):
        return False
    assessment = _assessment(parsed)
    if not isinstance(assessment, dict):
        return False
    leaf = field.split(".")[-1]
    return isinstance(assessment.get(leaf), bool)


def classify_judge_structural_repair(
    code: str,
    field: Optional[str],
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decision = "non_repairable_code"
    repairable = False
    if code in {"builder2_judge_schema_invalid", "builder2_judge_score_invalid"}:
        repairable = True
        decision = "structural_defect"
    elif code == "builder2_judge_validation_failed":
        if field == "factualGroundingAssessment" or is_factual_grounding_object_structurally_defective(parsed):
            if is_substantive_factual_grounding_negative(field, parsed):
                repairable = False
                decision = "substantive_negative"
            else:
                repairable = True
                decision = "structural_defect"
        elif is_judge_factual_grounding_gate_field(field):
            if is_substantive_factual_grounding_negative(field, parsed):
                repairable = False
                decision = "substantive_negative"
            else:
                repairable = True
                decision = "structural_defect"
        elif is_judge_structural_repair_field(field):
            repairable = True
            decision = "structural_defect"
        else:
            repairable = False
            decision = "non_repairable_field"
    return {
        "repairable": repairable,
        "decision": decision,
        "structuralRepairClassifierVersion": BUILDER2_JUDGE_STRUCTURAL_REPAIR_CLASSIFIER_VERSION,
        "field": field,
        "code": code,
    }


def is_judge_structural_repairable(
    code: str,
    field: Optional[str],
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> bool:
    return bool(classify_judge_structural_repair(code, field, parsed=parsed).get("repairable"))


def structural_errors_are_repairable(
    structural_errors: List[str],
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> bool:
    for item in structural_errors or []:
        code = item.split(":", 1)[0] if ":" in item else "builder2_judge_validation_failed"
        field = item.split(":", 1)[1] if ":" in item else item
        if is_judge_structural_repairable(code, field, parsed=parsed):
            return True
    if is_factual_grounding_object_structurally_defective(parsed):
        return True
    return False


def collect_repairable_structural_failures(
    structural_errors: List[str],
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> List[str]:
    repairable = [item for item in (structural_errors or []) if ":" in item and is_judge_structural_repairable(
        item.split(":", 1)[0],
        item.split(":", 1)[1],
        parsed=parsed,
    )]
    if not repairable and is_factual_grounding_object_structurally_defective(parsed):
        repairable = ["builder2_judge_validation_failed:factualGroundingAssessment"]
    return list(dict.fromkeys(repairable))
