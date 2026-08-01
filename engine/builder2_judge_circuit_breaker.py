"""
Builder2 Judge contract circuit breaker — stop expensive Judge calls on systemic schema failure.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from engine.builder2_judge_core_contract import (
    VERBAL_ASSESSMENT_BOOLEAN_FIELDS,
    filter_judge_structural_errors,
    is_judge_conclusion_boolean_field,
    is_judge_factual_grounding_gate_field,
)
from engine.builder2_judge_structural_repair_classifier import (
    BUILDER2_JUDGE_STRUCTURAL_REPAIR_CLASSIFIER_VERSION,
    is_factual_grounding_object_structurally_defective,
    is_substantive_factual_grounding_negative,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SYSTEMIC_FAILURE_CODE = "builder2_judge_contract_systemic_failure"
JUDGE_BREAKER_CONTRACT_VERSION = BUILDER2_JUDGE_STRUCTURAL_REPAIR_CLASSIFIER_VERSION


def _failure_field(error: str) -> str:
    if ":" in error:
        return error.split(":", 1)[1]
    return error


def _common_object(path: str) -> str:
    if "." in path:
        return path.split(".", 1)[0]
    return path


def _normalize_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    breaker = state.setdefault("judgeContractCircuitBreaker", {})
    if not isinstance(breaker, dict):
        breaker = {}
        state["judgeContractCircuitBreaker"] = breaker

    breaker.setdefault("postRepairFailures", [])
    breaker.setdefault("repeatedFieldPaths", [])

    if breaker.get("contractVersion") != JUDGE_BREAKER_CONTRACT_VERSION:
        legacy_paths = dict(breaker.get("candidateFailurePaths") or {})
        if legacy_paths or breaker.get("tripped"):
            breaker["legacyTripped"] = bool(breaker.get("tripped"))
            breaker["legacyTrippedReason"] = breaker.get("trippedReason") or ""
            breaker["legacyRepeatedFieldPaths"] = list(breaker.get("repeatedFieldPaths") or [])
            breaker["legacyCandidateFailurePaths"] = legacy_paths
        breaker["contractVersion"] = JUDGE_BREAKER_CONTRACT_VERSION
        breaker["currentCandidateFailurePaths"] = {}
        breaker["currentContractTripped"] = False
        breaker["currentTrippedReason"] = ""
        breaker["currentRepeatedFieldPaths"] = []

    breaker.setdefault("legacyTripped", False)
    breaker.setdefault("legacyTrippedReason", "")
    breaker.setdefault("legacyRepeatedFieldPaths", [])
    breaker.setdefault("legacyCandidateFailurePaths", {})
    breaker.setdefault("currentCandidateFailurePaths", {})
    breaker.setdefault("currentContractTripped", False)
    breaker.setdefault("currentTrippedReason", "")
    breaker.setdefault("currentRepeatedFieldPaths", [])

    # Preserve historical visibility on legacy fields used by older inspectors.
    if breaker.get("legacyTripped"):
        breaker["tripped"] = True
        breaker["trippedReason"] = breaker.get("legacyTrippedReason") or breaker.get("trippedReason") or ""
        breaker["repeatedFieldPaths"] = list(breaker.get("legacyRepeatedFieldPaths") or breaker.get("repeatedFieldPaths") or [])
        breaker["candidateFailurePaths"] = dict(breaker.get("legacyCandidateFailurePaths") or breaker.get("candidateFailurePaths") or {})
    return breaker


def _state_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_breaker(state)


def _trip_current_breaker(breaker: Dict[str, Any], *, reason: str, paths: List[str]) -> None:
    if breaker.get("currentContractTripped"):
        return
    breaker["currentContractTripped"] = True
    breaker["currentTrippedReason"] = reason
    breaker["currentRepeatedFieldPaths"] = paths
    logger.error(
        "BUILDER2_JUDGE_CONTRACT_CIRCUIT_BREAKER_CURRENT reason=%s paths=%s",
        reason,
        ",".join(paths[:12]),
    )


def _shared_paths_across_candidates(candidate_paths: Dict[str, List[str]]) -> Set[str]:
    path_sets = [set(paths) for paths in candidate_paths.values() if paths]
    if len(path_sets) < 2:
        return set()
    shared = path_sets[0]
    for item in path_sets[1:]:
        shared &= item
    return shared


def _current_contract_structural_paths(
    error_paths: List[str],
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> List[str]:
    filtered: List[str] = []
    for path in error_paths:
        if is_substantive_factual_grounding_negative(path, parsed):
            continue
        if is_judge_factual_grounding_gate_field(path):
            continue
        if path == "eligible":
            continue
        filtered.append(path)
    if parsed is not None and is_factual_grounding_object_structurally_defective(parsed):
        if "factualGroundingAssessment" not in filtered:
            filtered.append("factualGroundingAssessment")
    return list(dict.fromkeys(filtered))


def record_judge_contract_failure(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    error_paths: List[str],
    after_repair: bool = False,
    false_boolean_misclassified: bool = False,
    parsed: Optional[Dict[str, Any]] = None,
) -> None:
    breaker = _state_breaker(state)
    owned = filter_judge_structural_errors(
        [f"builder2_judge_validation_failed:{p}" for p in error_paths]
        + [f"builder2_judge_schema_invalid:{p}" for p in error_paths]
    )
    paths = _current_contract_structural_paths([_failure_field(item) for item in owned], parsed=parsed)

    if false_boolean_misclassified:
        _trip_current_breaker(
            breaker,
            reason="false_boolean_classified_as_malformed",
            paths=sorted(set(paths))[:8],
        )
        return

    if not paths:
        return

    by_candidate = breaker.setdefault("currentCandidateFailurePaths", {})
    existing = set(by_candidate.get(candidate_id) or [])
    merged = sorted(existing | set(paths))
    by_candidate[candidate_id] = merged

    shared = _shared_paths_across_candidates(by_candidate)
    if shared:
        _trip_current_breaker(
            breaker,
            reason="shared_structural_contract_field",
            paths=sorted(shared)[:12],
        )
        return

    if after_repair:
        breaker["postRepairFailures"].append({"candidateId": candidate_id, "paths": paths[:20], "contractVersion": JUDGE_BREAKER_CONTRACT_VERSION})

    post_repair = [
        item
        for item in (breaker.get("postRepairFailures") or [])
        if isinstance(item, dict) and item.get("contractVersion") == JUDGE_BREAKER_CONTRACT_VERSION
    ]
    if len(post_repair) >= 2:
        first_objects = {_common_object(p) for p in (post_repair[0].get("paths") or [])}
        second_objects = {_common_object(p) for p in (post_repair[1].get("paths") or [])}
        shared_objects = first_objects & second_objects
        if shared_objects:
            shared_paths = sorted(
                p
                for entry in post_repair[:2]
                for p in (entry.get("paths") or [])
                if _common_object(p) in shared_objects
            )
            _trip_current_breaker(
                breaker,
                reason="shared_post_repair_contract_object",
                paths=shared_paths[:12],
            )


def detect_false_boolean_misclassification(error_paths: List[str]) -> bool:
    for path in error_paths:
        if is_judge_factual_grounding_gate_field(path):
            continue
        if not is_judge_conclusion_boolean_field(path):
            continue
        leaf = path.split(".")[-1]
        if leaf in VERBAL_ASSESSMENT_BOOLEAN_FIELDS or leaf in {
            "visualWouldWorkWithoutHeadline",
            "headlineNeeded",
            "headlineRecommended",
            "methodActuallyApplied",
        }:
            return True
    return False


def is_current_judge_contract_circuit_breaker_tripped(state: Dict[str, Any]) -> bool:
    breaker = _state_breaker(state)
    return bool(breaker.get("currentContractTripped"))


def is_judge_contract_circuit_breaker_tripped(state: Dict[str, Any]) -> bool:
    return is_current_judge_contract_circuit_breaker_tripped(state)


def legacy_breaker_evidence_excluded_count(state: Dict[str, Any]) -> int:
    breaker = _state_breaker(state)
    legacy_paths = breaker.get("legacyCandidateFailurePaths") or {}
    return sum(len(paths or []) for paths in legacy_paths.values() if isinstance(paths, list))


def current_breaker_evidence_count(state: Dict[str, Any]) -> int:
    breaker = _state_breaker(state)
    current_paths = breaker.get("currentCandidateFailurePaths") or {}
    return sum(len(paths or []) for paths in current_paths.values() if isinstance(paths, list))


def assert_judge_contract_available(state: Dict[str, Any]) -> None:
    breaker = _state_breaker(state)
    if not breaker.get("currentContractTripped"):
        return
    paths = breaker.get("currentRepeatedFieldPaths") or []
    reason = breaker.get("currentTrippedReason") or "contract_failure"
    raise Builder2TournamentError(f"{SYSTEMIC_FAILURE_CODE}:{reason}:{','.join(paths[:8])}")


def record_judge_process_contract_failure(state: Dict[str, Any], exc: Builder2TournamentError) -> None:
    from engine.builder2_tournament_store import record_process_failure_tag

    msg = str(exc.args[0] if exc.args else SYSTEMIC_FAILURE_CODE)
    record_process_failure_tag(state, SYSTEMIC_FAILURE_CODE)
    state["status"] = "failed"
    state["error"] = msg
    state["failureCategory"] = "infrastructure"
    state["completionReason"] = "judge_contract_systemic_failure"
