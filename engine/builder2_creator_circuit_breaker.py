"""
Builder2 Creator contract circuit breaker — stop expensive Creator calls on systemic schema failure.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from engine.builder2_creator_core_contract import (
    SERVER_DERIVED_FIELD_PATHS,
    filter_creator_owned_structural_errors,
    is_server_derived_field,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SYSTEMIC_FAILURE_CODE = "builder2_creator_contract_systemic_failure"
COMMON_PATH_THRESHOLD = 10


def _failure_field(error: str) -> str:
    if ":" in error:
        return error.split(":", 1)[1]
    return error


def _common_contract_fields(paths: List[str]) -> Set[str]:
    common: Set[str] = set()
    for path in paths:
        if is_server_derived_field(path):
            common.add(path)
        if path.startswith("visualFamilyConsistency"):
            common.add("visualFamilyConsistency")
        if path.startswith("essenceExtreme"):
            common.add("essenceExtreme")
        if path.startswith("participationMechanism"):
            common.add("participationMechanism")
        if path.startswith("anchorPunchlineSeparation"):
            common.add("anchorPunchlineSeparation")
    return common


def _state_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    breaker = state.setdefault("creatorContractCircuitBreaker", {})
    breaker.setdefault("postRepairFailures", [])
    breaker.setdefault("structuralFailureCounts", {})
    breaker.setdefault("tripped", False)
    breaker.setdefault("trippedReason", "")
    breaker.setdefault("repeatedFieldPaths", [])
    return breaker


def record_creator_contract_failure(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    error_paths: List[str],
    after_repair: bool = False,
) -> None:
    breaker = _state_breaker(state)
    owned_paths = [_failure_field(item) for item in filter_creator_owned_structural_errors(
        [f"builder2_creator_validation_failed:{p}" for p in error_paths]
    )]

    counts = breaker["structuralFailureCounts"]
    for field in owned_paths:
        counts[field] = int(counts.get(field) or 0) + 1

    if after_repair:
        breaker["postRepairFailures"].append(
            {"prototypeId": prototype_id, "paths": owned_paths[:20]}
        )

    if any(is_server_derived_field(p) for p in owned_paths):
        _trip_breaker(
            breaker,
            reason="server_derived_field_hard_gate",
            paths=sorted(set(error_paths)),
        )
        return

    post_repair = breaker.get("postRepairFailures") or []
    if len(post_repair) >= 2:
        first_paths = set(post_repair[0].get("paths") or [])
        second_paths = set(post_repair[1].get("paths") or [])
        shared = first_paths & second_paths
        if shared:
            _trip_breaker(
                breaker,
                reason="shared_post_repair_contract_field",
                paths=sorted(shared),
            )
            return

    heavy = [entry for entry in post_repair if len(entry.get("paths") or []) >= COMMON_PATH_THRESHOLD]
    if len(heavy) >= 2:
        shared = set(heavy[0].get("paths") or []) & set(heavy[1].get("paths") or [])
        if len(shared) >= COMMON_PATH_THRESHOLD:
            _trip_breaker(
                breaker,
                reason="mass_structural_contract_failure",
                paths=sorted(shared)[:20],
            )


def _trip_breaker(breaker: Dict[str, Any], *, reason: str, paths: List[str]) -> None:
    if breaker.get("tripped"):
        return
    breaker["tripped"] = True
    breaker["trippedReason"] = reason
    breaker["repeatedFieldPaths"] = paths
    logger.error(
        "BUILDER2_CREATOR_CONTRACT_CIRCUIT_BREAKER reason=%s paths=%s",
        reason,
        ",".join(paths[:12]),
    )


def is_creator_contract_circuit_breaker_tripped(state: Dict[str, Any]) -> bool:
    breaker = state.get("creatorContractCircuitBreaker") or {}
    return bool(breaker.get("tripped"))


def assert_creator_contract_available(state: Dict[str, Any]) -> None:
    breaker = state.get("creatorContractCircuitBreaker") or {}
    if not breaker.get("tripped"):
        return
    paths = breaker.get("repeatedFieldPaths") or []
    reason = breaker.get("trippedReason") or "contract_failure"
    raise Builder2TournamentError(f"{SYSTEMIC_FAILURE_CODE}:{reason}:{','.join(paths[:8])}")


def record_process_contract_failure(state: Dict[str, Any], exc: Builder2TournamentError) -> None:
    from engine.builder2_tournament_store import record_process_failure_tag

    msg = str(exc.args[0] if exc.args else SYSTEMIC_FAILURE_CODE)
    record_process_failure_tag(state, SYSTEMIC_FAILURE_CODE)
    state["status"] = "failed"
    state["error"] = msg
    state["failureCategory"] = "infrastructure"
    state["completionReason"] = "creator_contract_systemic_failure"
