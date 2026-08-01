"""
Builder2 Judge contract circuit breaker — stop expensive Judge calls on systemic schema failure.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from engine.builder2_judge_core_contract import (
    VERBAL_ASSESSMENT_BOOLEAN_FIELDS,
    filter_judge_structural_errors,
    is_judge_conclusion_boolean_field,
    is_judge_factual_grounding_gate_field,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SYSTEMIC_FAILURE_CODE = "builder2_judge_contract_systemic_failure"


def _failure_field(error: str) -> str:
    if ":" in error:
        return error.split(":", 1)[1]
    return error


def _common_object(path: str) -> str:
    if "." in path:
        return path.split(".", 1)[0]
    return path


def _state_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    breaker = state.setdefault("judgeContractCircuitBreaker", {})
    breaker.setdefault("postRepairFailures", [])
    breaker.setdefault("candidateFailurePaths", {})
    breaker.setdefault("tripped", False)
    breaker.setdefault("trippedReason", "")
    breaker.setdefault("repeatedFieldPaths", [])
    return breaker


def _trip_breaker(breaker: Dict[str, Any], *, reason: str, paths: List[str]) -> None:
    if breaker.get("tripped"):
        return
    breaker["tripped"] = True
    breaker["trippedReason"] = reason
    breaker["repeatedFieldPaths"] = paths
    logger.error(
        "BUILDER2_JUDGE_CONTRACT_CIRCUIT_BREAKER reason=%s paths=%s",
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


def _structural_failure_paths(error_paths: List[str]) -> List[str]:
    filtered: List[str] = []
    for path in error_paths:
        if is_judge_factual_grounding_gate_field(path):
            continue
        if path == "eligible":
            continue
        filtered.append(path)
    return filtered


def record_judge_contract_failure(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    error_paths: List[str],
    after_repair: bool = False,
    false_boolean_misclassified: bool = False,
) -> None:
    breaker = _state_breaker(state)
    owned = filter_judge_structural_errors(
        [f"builder2_judge_validation_failed:{p}" for p in error_paths]
        + [f"builder2_judge_schema_invalid:{p}" for p in error_paths]
    )
    paths = _structural_failure_paths([_failure_field(item) for item in owned])

    if false_boolean_misclassified:
        _trip_breaker(
            breaker,
            reason="false_boolean_classified_as_malformed",
            paths=sorted(set(paths))[:8],
        )
        return

    if not paths:
        return

    by_candidate = breaker.setdefault("candidateFailurePaths", {})
    existing = set(by_candidate.get(candidate_id) or [])
    merged = sorted(existing | set(paths))
    by_candidate[candidate_id] = merged

    shared = _shared_paths_across_candidates(by_candidate)
    if shared:
        _trip_breaker(
            breaker,
            reason="shared_structural_contract_field",
            paths=sorted(shared)[:12],
        )
        return

    if after_repair:
        breaker["postRepairFailures"].append({"candidateId": candidate_id, "paths": paths[:20]})

    post_repair = breaker.get("postRepairFailures") or []
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
            _trip_breaker(
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


def is_judge_contract_circuit_breaker_tripped(state: Dict[str, Any]) -> bool:
    breaker = state.get("judgeContractCircuitBreaker") or {}
    return bool(breaker.get("tripped"))


def assert_judge_contract_available(state: Dict[str, Any]) -> None:
    breaker = state.get("judgeContractCircuitBreaker") or {}
    if not breaker.get("tripped"):
        return
    paths = breaker.get("repeatedFieldPaths") or []
    reason = breaker.get("trippedReason") or "contract_failure"
    raise Builder2TournamentError(f"{SYSTEMIC_FAILURE_CODE}:{reason}:{','.join(paths[:8])}")


def record_judge_process_contract_failure(state: Dict[str, Any], exc: Builder2TournamentError) -> None:
    from engine.builder2_tournament_store import record_process_failure_tag

    msg = str(exc.args[0] if exc.args else SYSTEMIC_FAILURE_CODE)
    record_process_failure_tag(state, SYSTEMIC_FAILURE_CODE)
    state["status"] = "failed"
    state["error"] = msg
    state["failureCategory"] = "infrastructure"
    state["completionReason"] = "judge_contract_systemic_failure"
