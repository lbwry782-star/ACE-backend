"""
Builder2 Creator semantic-bridge repair — one bounded paid call after slogan repair.
"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from engine.builder2_complete_ad_contract import DUAL_MEANING_FIELDS
from engine.builder2_creator import (
    collect_creator_structural_errors,
    validate_creator_candidate,
)
from engine.builder2_creator_slogan_repair_patch import (
    SLOGAN_REPAIR_CALL_LEDGER_KEY,
    _apply_selected_patch_paths,
    _flatten_leaf_paths,
    _get_nested,
    _set_nested,
    extract_repaired_slogan_text,
    prototype_display_name_for_id,
    reconcile_slogan_repair_call_ledger,
)
from engine.builder2_slogan_repair_provenance import resolve_slogan_repair_base_and_source
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import MetricsTimer, record_model_call

logger = logging.getLogger(__name__)

SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT = "semanticBridgeRepairPatch"
SEMANTIC_BRIDGE_REPAIR_CALL_LEDGER_KEY = "semanticBridgeRepairCallLedger"
SEMANTIC_BRIDGE_REPAIR_CALL_KIND = "creatorSemanticBridgeRepair"
SEMANTIC_BRIDGE_REPAIR_ENV_FLAG = "BUILDER2_ALLOW_ONE_ADDITIONAL_SEMANTIC_BRIDGE_REPAIR"
SEMANTIC_BRIDGE_REPAIR_ROLE = "builder2_creator_semantic_bridge_repair"

LIFECYCLE_AUTHORIZED = "authorized"
LIFECYCLE_RESERVED = "reserved"
LIFECYCLE_DISPATCHED = "dispatched"
LIFECYCLE_RESPONSE_RECEIVED = "response_received"
LIFECYCLE_ACCEPTED = "accepted"
LIFECYCLE_REJECTED = "rejected"
LIFECYCLE_FAILED_PRE_DISPATCH = "failed_pre_dispatch"
LIFECYCLE_FAILED_POST_DISPATCH_UNKNOWN = "failed_post_dispatch_unknown"

PRE_DISPATCH_FAILURE_CODES = frozenset(
    {
        "builder2_semantic_bridge_repair_client_missing",
        "builder2_semantic_bridge_repair_model_missing",
        "builder2_semantic_bridge_repair_preflight_failed",
        "builder2_semantic_bridge_repair_not_authorized",
        "builder2_semantic_bridge_repair_not_required",
    }
)

SEMANTIC_BRIDGE_REPAIR_ALLOWLIST_PATHS: Tuple[str, ...] = (
    "semanticBridge.keyWordOrConcept",
    "semanticBridge.visualMeaning",
    "semanticBridge.strategicMeaning",
    "semanticBridge.sloganMeaning",
    "semanticBridge.howTheMeaningsMeet",
    "semanticBridge.understandableWithoutCreatorReport",
    "semanticBridge.dualMeaningUsed",
    "semanticBridge.physicalMeaningActivatedByVisual",
    "semanticBridge.strategicMeaningActivatedBySlogan",
    "semanticBridge.meaningsConverge",
    "visualBridgeAssessment.sloganConnectionToVisibleDetail",
    "visualBridgeAssessment.sloganConnectionToRelativeAdvantage",
    "metaphoricalEmbodiment.sloganBridgeToBusinessMeaning",
    "verbalPotential.keywordOrKeyPhrase",
    "verbalPotential.strategicMeaning",
)

SEMANTIC_BRIDGE_REPAIR_IMMUTABLE_PATHS: Tuple[str, ...] = (
    "advertisingClosure.sloganText",
    "advertisingClosure.productNameText",
    "advertisingClosure.presentationMode",
    "advertisingClosure.durationSeconds",
    "advertisingClosure.noLogo",
    "visualMechanism",
    "coreCreativeMechanism",
    "runwayFeasibility",
    "visualAnchor",
    "sevenSecondStructure",
    "prototypeId",
    "prototypeMethodApplication",
    "creativeEmbodiment",
    "physicalEmbodiment",
    "participationMechanism",
    "scene",
    "openingFrame",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _path_allowed(path: str) -> bool:
    return path in SEMANTIC_BRIDGE_REPAIR_ALLOWLIST_PATHS or any(
        path.startswith(f"{allowed}.") for allowed in SEMANTIC_BRIDGE_REPAIR_ALLOWLIST_PATHS
    )


def _path_immutable(path: str) -> bool:
    return path in SEMANTIC_BRIDGE_REPAIR_IMMUTABLE_PATHS or any(
        path.startswith(f"{immutable}.") for immutable in SEMANTIC_BRIDGE_REPAIR_IMMUTABLE_PATHS
    )


def extract_semantic_bridge_repair_patch(repair_response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(repair_response, dict):
        return {}
    patch_root = repair_response.get(SEMANTIC_BRIDGE_REPAIR_PATCH_ROOT)
    if isinstance(patch_root, dict) and patch_root:
        return deepcopy(patch_root)
    extracted: Dict[str, Any] = {}
    for path in SEMANTIC_BRIDGE_REPAIR_ALLOWLIST_PATHS:
        value = _get_nested(repair_response, path)
        if value is not None and value != "":
            _set_nested(extracted, path, deepcopy(value))
    return extracted


def apply_persisted_slogan_to_base(
    original_parsed: Dict[str, Any],
    repair_parsed: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    base = deepcopy(original_parsed)
    repaired_slogan = extract_repaired_slogan_text(repair_parsed)
    if not repaired_slogan:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:repaired_slogan_missing")
    _set_nested(base, "advertisingClosure.sloganText", repaired_slogan)
    if isinstance(base.get("advertisingSloganFormulation"), dict):
        from engine.builder2_advertising_slogan_quality_contract import sync_creator_slogan_formulation_from_closure

        sync_creator_slogan_formulation_from_closure(base, strategy_foundation=strategy_foundation)
    return base, repaired_slogan


def structural_failure_field_paths(errors: Sequence[str]) -> List[str]:
    return [item.split(":", 1)[-1] for item in errors if item]


def semantic_bridge_repair_required(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    product_name: str = "",
) -> Tuple[bool, List[str]]:
    errors = collect_creator_structural_errors(
        candidate,
        assigned_prototype_id=assigned_prototype_id,
        prototype_display_name=prototype_display_name,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    paths = structural_failure_field_paths(errors)
    semantic_paths = [path for path in paths if path.startswith("semanticBridge.")]
    word_limit_paths = [path for path in paths if path.endswith("sloganText.word_limit")]
    if word_limit_paths:
        return False, paths
    if not semantic_paths:
        return False, paths
    return True, paths


def semantic_bridge_repair_env_authorized() -> bool:
    return _clean(os.environ.get(SEMANTIC_BRIDGE_REPAIR_ENV_FLAG)).lower() in {"1", "true", "yes", "on"}


def _ledger_bucket(state: Dict[str, Any], prototype_id: str) -> Dict[str, Any]:
    root = state.setdefault(SEMANTIC_BRIDGE_REPAIR_CALL_LEDGER_KEY, {})
    if not isinstance(root, dict):
        root = {}
        state[SEMANTIC_BRIDGE_REPAIR_CALL_LEDGER_KEY] = root
    bucket = root.setdefault(prototype_id, {})
    if not isinstance(bucket, dict):
        bucket = {}
        root[prototype_id] = bucket
    return bucket


def reconcile_semantic_bridge_repair_call_ledger(state: Dict[str, Any], *, prototype_id: str) -> Dict[str, Any]:
    bucket = _ledger_bucket(state, prototype_id)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    metric_count = int(metrics.get("creatorSemanticBridgeRepairCalls") or 0)
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    if bucket.get("dispatchedAt") and paid_dispatch < 1:
        paid_dispatch = 1
    if metric_count > paid_dispatch:
        paid_dispatch = min(metric_count, 1)
    if paid_dispatch > 1:
        paid_dispatch = 1
    bucket["paidDispatchCount"] = paid_dispatch
    bucket["persistedSemanticBridgeRepairCalls"] = paid_dispatch
    bucket["canonicalSemanticBridgeRepairCalls"] = paid_dispatch
    if paid_dispatch >= 1 and not _clean(bucket.get("lifecycleState")):
        bucket["lifecycleState"] = LIFECYCLE_DISPATCHED
    return bucket


def get_semantic_bridge_repair_ledger_snapshot(
    state: Dict[str, Any],
    *,
    prototype_id: str,
) -> Dict[str, Any]:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    lifecycle = _clean(bucket.get("lifecycleState"))
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    return {
        "lifecycleState": lifecycle or (LIFECYCLE_AUTHORIZED if semantic_bridge_repair_env_authorized() else ""),
        "reservationPresent": bool(bucket.get("reservedAt")),
        "dispatchRecorded": bool(bucket.get("dispatchedAt")) or paid_dispatch >= 1,
        "responseReceived": bool(bucket.get("responseReceivedAt")),
        "accepted": lifecycle == LIFECYCLE_ACCEPTED or bool(bucket.get("acceptedAt")),
        "paidDispatchCount": paid_dispatch,
        "failureCode": _clean(bucket.get("failureCode")),
        "preDispatchFailureRecoverable": lifecycle == LIFECYCLE_FAILED_PRE_DISPATCH and paid_dispatch < 1,
    }


def is_pre_dispatch_failure_code(reason: str) -> bool:
    code = _clean(reason)
    if code in PRE_DISPATCH_FAILURE_CODES:
        return True
    return code.startswith("builder2_semantic_bridge_repair_invalid:") and code.endswith("_missing")


def additional_semantic_bridge_repair_allowed(state: Dict[str, Any], prototype_id: str) -> bool:
    if not semantic_bridge_repair_env_authorized():
        return False
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    lifecycle = _clean(bucket.get("lifecycleState"))
    if paid_dispatch >= 1 or lifecycle in {LIFECYCLE_DISPATCHED, LIFECYCLE_RESPONSE_RECEIVED, LIFECYCLE_ACCEPTED}:
        return False
    if lifecycle == LIFECYCLE_FAILED_PRE_DISPATCH and paid_dispatch < 1:
        return True
    if lifecycle in {LIFECYCLE_RESERVED, LIFECYCLE_AUTHORIZED, ""}:
        pass
    elif lifecycle in {LIFECYCLE_REJECTED, LIFECYCLE_FAILED_POST_DISPATCH_UNKNOWN}:
        return False
    reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)
    slogan_bucket = state.get(SLOGAN_REPAIR_CALL_LEDGER_KEY, {}).get(prototype_id, {})
    if not isinstance(slogan_bucket, dict):
        return False
    if int(slogan_bucket.get("canonicalCreatorRepairCalls") or 0) < 1:
        return False
    return True


def reclaim_or_reserve_semantic_bridge_repair_call(state: Dict[str, Any], *, prototype_id: str) -> bool:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    lifecycle = _clean(bucket.get("lifecycleState"))
    if paid_dispatch >= 1 or lifecycle in {LIFECYCLE_DISPATCHED, LIFECYCLE_RESPONSE_RECEIVED, LIFECYCLE_ACCEPTED}:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_call_already_used")
    reused = False
    if lifecycle == LIFECYCLE_FAILED_PRE_DISPATCH and bucket.get("reservedAt"):
        reused = True
        bucket["preDispatchFailureRecovered"] = True
        logger.info(
            "BUILDER2_SEMANTIC_BRIDGE_REPAIR_RESERVATION_REUSED prototypeId=%s lifecycleState=%s paidDispatchCount=%s",
            prototype_id,
            lifecycle,
            paid_dispatch,
        )
    bucket["lifecycleState"] = LIFECYCLE_RESERVED
    bucket["reservedAt"] = bucket.get("reservedAt") or _utc_now_iso()
    bucket["callKind"] = SEMANTIC_BRIDGE_REPAIR_CALL_KIND
    bucket["authorizedAt"] = bucket.get("authorizedAt") or _utc_now_iso()
    bucket.pop("failureCode", None)
    bucket.pop("failedPreDispatchAt", None)
    return reused


def mark_semantic_bridge_repair_failed_pre_dispatch(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    failure_code: str,
) -> None:
    bucket = _ledger_bucket(state, prototype_id)
    if int(bucket.get("paidDispatchCount") or 0) >= 1:
        return
    bucket["lifecycleState"] = LIFECYCLE_FAILED_PRE_DISPATCH
    bucket["failureCode"] = _clean(failure_code)
    bucket["failedPreDispatchAt"] = _utc_now_iso()
    if not bucket.get("reservedAt"):
        bucket["reservedAt"] = _utc_now_iso()
        bucket["callKind"] = SEMANTIC_BRIDGE_REPAIR_CALL_KIND


def mark_semantic_bridge_repair_dispatched(state: Dict[str, Any], *, prototype_id: str) -> None:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    bucket["lifecycleState"] = LIFECYCLE_DISPATCHED
    bucket["dispatchedAt"] = _utc_now_iso()
    bucket["paidDispatchCount"] = 1
    bucket["persistedSemanticBridgeRepairCalls"] = 1
    bucket["canonicalSemanticBridgeRepairCalls"] = 1


def mark_semantic_bridge_repair_response_received(state: Dict[str, Any], *, prototype_id: str) -> None:
    bucket = _ledger_bucket(state, prototype_id)
    bucket["lifecycleState"] = LIFECYCLE_RESPONSE_RECEIVED
    bucket["responseReceivedAt"] = _utc_now_iso()


def mark_semantic_bridge_repair_accepted(state: Dict[str, Any], *, prototype_id: str) -> None:
    bucket = _ledger_bucket(state, prototype_id)
    bucket["lifecycleState"] = LIFECYCLE_ACCEPTED
    bucket["acceptedAt"] = _utc_now_iso()


def reserve_semantic_bridge_repair_call(state: Dict[str, Any], *, prototype_id: str) -> None:
    reclaim_or_reserve_semantic_bridge_repair_call(state, prototype_id=prototype_id)


def revert_forbidden_semantic_bridge_paths(base_candidate: Dict[str, Any], merged: Dict[str, Any]) -> List[str]:
    reverted: List[str] = []
    base_paths = _flatten_leaf_paths(base_candidate)
    merged_paths = _flatten_leaf_paths(merged)
    for path in sorted(base_paths | merged_paths):
        if _get_nested(base_candidate, path) == _get_nested(merged, path):
            continue
        if _path_allowed(path):
            continue
        base_value = _get_nested(base_candidate, path)
        if base_value is None:
            from engine.builder2_creator_slogan_repair_patch import _delete_nested

            _delete_nested(merged, path)
        else:
            _set_nested(merged, path, deepcopy(base_value))
        reverted.append(path)
    if reverted:
        logger.info(
            "BUILDER2_SEMANTIC_BRIDGE_REPAIR_FORBIDDEN_CHANGE_REVERTED pathCount=%s paths=%s",
            len(reverted),
            ",".join(reverted[:12]),
        )
    return reverted


def validate_semantic_bridge_establishes_convergence(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    bridge = candidate.get("semanticBridge") if isinstance(candidate.get("semanticBridge"), dict) else {}
    for key in (
        "keyWordOrConcept",
        "visualMeaning",
        "sloganMeaning",
        "strategicMeaning",
        "howTheMeaningsMeet",
    ):
        if not _clean(bridge.get(key)):
            return False, f"semanticBridge.{key}"
    if bridge.get("understandableWithoutCreatorReport") is not True:
        return False, "semanticBridge.understandableWithoutCreatorReport"
    if bridge.get("dualMeaningUsed") is True:
        for key in DUAL_MEANING_FIELDS[1:]:
            if bridge.get(key) is not True:
                return False, f"semanticBridge.{key}"
    elif bridge.get("meaningsConverge") is True and not _clean(bridge.get("howTheMeaningsMeet")):
        return False, "semanticBridge.howTheMeaningsMeet"
    return True, None


def merge_semantic_bridge_repair_patch(
    base_candidate: Dict[str, Any],
    repair_response: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    patch = extract_semantic_bridge_repair_patch(repair_response)
    if not patch:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:patch_missing")
    merged = deepcopy(base_candidate)
    applied: List[str] = []
    for path in SEMANTIC_BRIDGE_REPAIR_ALLOWLIST_PATHS:
        value = _get_nested(patch, path)
        if value is None:
            value = _get_nested(repair_response, path)
        if value is None or value == "":
            continue
        _set_nested(merged, path, deepcopy(value))
        applied.append(path)
    reverted = revert_forbidden_semantic_bridge_paths(base_candidate, merged)
    if not applied:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:patch_empty")
    base_bridge = base_candidate.get("semanticBridge") if isinstance(base_candidate.get("semanticBridge"), dict) else {}
    if base_bridge.get("meaningsConverge") is not True:
        substantive_paths = {
            "semanticBridge.keyWordOrConcept",
            "semanticBridge.visualMeaning",
            "semanticBridge.strategicMeaning",
            "semanticBridge.sloganMeaning",
            "semanticBridge.howTheMeaningsMeet",
        }
        if not substantive_paths.intersection(applied):
            raise Builder2TournamentError("builder2_semantic_bridge_repair_incomplete:meaningsConverge.only_boolean")
    ok, field = validate_semantic_bridge_establishes_convergence(merged)
    if not ok:
        raise Builder2TournamentError(f"builder2_semantic_bridge_repair_incomplete:{field}")
    meta = {"appliedPaths": applied, "revertedPaths": reverted}
    return merged, meta


def detect_semantic_bridge_repair_context(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    product_name: str = "",
    compatibility_mode: bool = False,
    original_candidate_id: str = "",
    repair_candidate_id: str = "",
) -> Dict[str, Any]:
    original_payload, repair_payload = resolve_slogan_repair_base_and_source(
        state,
        prototype_id,
        original_candidate_id=original_candidate_id,
        repair_candidate_id=repair_candidate_id,
        product_name=product_name,
    )
    display_name = prototype_display_name_for_id(prototype_id)
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    base_parsed, repaired_slogan = apply_persisted_slogan_to_base(
        original_payload.get("parsed") or {},
        repair_payload.get("parsed") or {},
        strategy_foundation=strategy,
    )
    required, paths = semantic_bridge_repair_required(
        base_parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        product_name=product_name,
    )
    return {
        "originalPayload": original_payload,
        "repairPayload": repair_payload,
        "baseParsed": base_parsed,
        "repairedSlogan": repaired_slogan,
        "required": required,
        "failurePaths": paths,
        "acceptCandidateId": _clean(repair_payload.get("candidateId")) or _clean(original_payload.get("candidateId")),
    }


def preflight_semantic_bridge_repair(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    product_name: str = "",
    product_description: str = "",
    language: str = "he",
    compatibility_mode: bool = False,
    llm_client: Optional[Callable[..., Any]] = None,
    original_candidate_id: str = "",
    repair_candidate_id: str = "",
    accept_candidate_id: str = "",
) -> Dict[str, Any]:
    from engine.builder2_prototypes import require_prototype
    from engine.builder2_reasoning_config import resolve_builder2_reasoning_model
    from engine.builder2_tournament_prompts import build_semantic_bridge_repair_prompt

    job_id = _clean(state.get("jobId"))
    if not semantic_bridge_repair_env_authorized():
        raise Builder2TournamentError("builder2_semantic_bridge_repair_not_authorized")

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_PREFLIGHT_START jobId=%s prototypeId=%s role=%s",
        job_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_ROLE,
    )

    model = resolve_builder2_reasoning_model()
    if not _clean(model):
        raise Builder2TournamentError("builder2_semantic_bridge_repair_model_missing")

    if llm_client is None:
        api_key = _clean(os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            raise Builder2TournamentError("builder2_semantic_bridge_repair_client_missing")

    try:
        context = detect_semantic_bridge_repair_context(
            state,
            prototype_id=prototype_id,
            product_name=product_name,
            compatibility_mode=compatibility_mode,
            original_candidate_id=original_candidate_id,
            repair_candidate_id=repair_candidate_id,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_semantic_bridge_repair_preflight_failed")
        raise Builder2TournamentError(f"builder2_semantic_bridge_repair_preflight_failed:{reason}") from exc

    if not context["required"]:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_not_required")

    failure_paths = list(context.get("failurePaths") or [])
    if any(path.endswith("sloganText.word_limit") for path in failure_paths):
        raise Builder2TournamentError("builder2_semantic_bridge_repair_preflight_failed:word_limit_remaining")
    semantic_paths = [path for path in failure_paths if path.startswith("semanticBridge.")]
    if not semantic_paths:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_preflight_failed:no_semantic_paths")
    non_converge = [path for path in semantic_paths if path != "semanticBridge.meaningsConverge"]
    if non_converge:
        raise Builder2TournamentError(
            "builder2_semantic_bridge_repair_preflight_failed:non_converge_paths:" + ",".join(non_converge[:8])
        )

    candidate_id = accept_candidate_id or context["acceptCandidateId"]
    if not candidate_id:
        candidate_id = f"cand-1-{prototype_id}-1-semantic-bridge-repair"

    prototype = require_prototype(prototype_id)
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    try:
        prompt = build_semantic_bridge_repair_prompt(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype=prototype,
            candidate_id=candidate_id,
            base_candidate=context["baseParsed"],
            validation_failures=failure_paths,
        )
    except Exception as exc:
        raise Builder2TournamentError(
            f"builder2_semantic_bridge_repair_preflight_failed:prompt:{exc.__class__.__name__}"
        ) from exc
    if not _clean(prompt):
        raise Builder2TournamentError("builder2_semantic_bridge_repair_preflight_failed:prompt_empty")

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_CLIENT_RESOLVED jobId=%s prototypeId=%s role=%s model=%s llmClientProvided=%s",
        job_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_ROLE,
        model,
        str(llm_client is not None).lower(),
    )
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_PREFLIGHT_OK jobId=%s prototypeId=%s role=%s model=%s",
        job_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_ROLE,
        model,
    )
    return {
        "context": context,
        "model": model,
        "prompt": prompt,
        "candidateId": candidate_id,
        "prototype": prototype,
        "displayName": prototype.display_name,
        "strategy": strategy,
    }


def inspect_semantic_bridge_repair_lifecycle(
    state: Dict[str, Any],
    *,
    prototype_id: str,
) -> Dict[str, Any]:
    snapshot = get_semantic_bridge_repair_ledger_snapshot(state, prototype_id=prototype_id)
    paid_dispatch = int(snapshot.get("paidDispatchCount") or 0)
    lifecycle = _clean(snapshot.get("lifecycleState"))
    authorized = semantic_bridge_repair_env_authorized()
    another_allowed = additional_semantic_bridge_repair_allowed(state, prototype_id) and paid_dispatch < 1
    return {
        "jobId": _clean(state.get("jobId")),
        "prototypeId": prototype_id,
        "authorizationPresent": authorized,
        "semanticBridgeRepairLedgerFound": bool(state.get(SEMANTIC_BRIDGE_REPAIR_CALL_LEDGER_KEY)),
        "lifecycleState": lifecycle,
        "reservationPresent": bool(snapshot.get("reservationPresent")),
        "dispatchRecorded": bool(snapshot.get("dispatchRecorded")),
        "responseReceived": bool(snapshot.get("responseReceived")),
        "accepted": bool(snapshot.get("accepted")),
        "paidDispatchCount": paid_dispatch,
        "preDispatchFailureRecoverable": bool(snapshot.get("preDispatchFailureRecoverable")),
        "anotherPaidDispatchAllowed": another_allowed,
        "stateMutated": False,
        "paidCalls": paid_dispatch,
    }


def populate_semantic_bridge_repair_call_report(
    state: Dict[str, Any],
    report: Dict[str, Any],
    *,
    prototype_id: str,
    invocation_semantic_bridge_repair_calls: int = 0,
    semantic_bridge_repair_accepted: bool = False,
    pre_dispatch_failure_recovered: bool = False,
) -> None:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    snapshot = get_semantic_bridge_repair_ledger_snapshot(state, prototype_id=prototype_id)
    paid_dispatch = int(snapshot.get("paidDispatchCount") or 0)
    lifecycle = _clean(snapshot.get("lifecycleState"))
    report["invocationSemanticBridgeRepairCalls"] = int(invocation_semantic_bridge_repair_calls)
    report["persistedSemanticBridgeRepairCalls"] = paid_dispatch
    report["totalSemanticBridgeRepairCalls"] = paid_dispatch
    report["semanticBridgeRepairAuthorized"] = semantic_bridge_repair_env_authorized()
    report["semanticBridgeRepairReserved"] = bool(snapshot.get("reservationPresent"))
    report["semanticBridgeRepairDispatched"] = bool(snapshot.get("dispatchRecorded"))
    report["semanticBridgeRepairResponseReceived"] = bool(snapshot.get("responseReceived"))
    report["semanticBridgeRepairAccepted"] = bool(semantic_bridge_repair_accepted or snapshot.get("accepted"))
    report["semanticBridgeRepairLifecycleState"] = lifecycle or (
        LIFECYCLE_AUTHORIZED if report["semanticBridgeRepairAuthorized"] else ""
    )
    report["semanticBridgeRepairPreDispatchFailureRecovered"] = bool(
        pre_dispatch_failure_recovered or bucket.get("preDispatchFailureRecovered")
    )


def execute_semantic_bridge_repair_call(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    product_name: str = "",
    product_description: str = "",
    language: str = "he",
    compatibility_mode: bool = False,
    llm_client: Optional[Callable[..., Any]] = None,
    original_candidate_id: str = "",
    repair_candidate_id: str = "",
    accept_candidate_id: str = "",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    from engine.builder2_accepted_creator_store import persist_accepted_creator_candidate
    from engine.builder2_tournament_llm import call_builder2_role_json

    job_id = _clean(state.get("jobId"))
    if not additional_semantic_bridge_repair_allowed(state, prototype_id):
        raise Builder2TournamentError("builder2_semantic_bridge_repair_not_authorized")

    try:
        preflight = preflight_semantic_bridge_repair(
            state,
            prototype_id=prototype_id,
            product_name=product_name,
            product_description=product_description,
            language=language,
            compatibility_mode=compatibility_mode,
            llm_client=llm_client,
            original_candidate_id=original_candidate_id,
            repair_candidate_id=repair_candidate_id,
            accept_candidate_id=accept_candidate_id,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_semantic_bridge_repair_preflight_failed")
        if is_pre_dispatch_failure_code(reason) or reason.startswith("builder2_semantic_bridge_repair_preflight_failed"):
            mark_semantic_bridge_repair_failed_pre_dispatch(
                state,
                prototype_id=prototype_id,
                failure_code=reason,
            )
        raise

    context = preflight["context"]
    base_parsed = context["baseParsed"]
    original_payload = context["originalPayload"]
    failure_paths = context["failurePaths"]
    candidate_id = preflight["candidateId"]
    prototype = preflight["prototype"]
    display_name = preflight["displayName"]
    strategy = preflight["strategy"]
    model = preflight["model"]
    prompt = preflight["prompt"]

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_REQUIRED candidateId=%s prototypeId=%s failurePaths=%s",
        candidate_id,
        prototype_id,
        ",".join(failure_paths[:12]),
    )
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_AUTHORIZED candidateId=%s prototypeId=%s envFlag=%s",
        candidate_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_ENV_FLAG,
    )

    reservation_reused = reclaim_or_reserve_semantic_bridge_repair_call(state, prototype_id=prototype_id)
    if reservation_reused:
        logger.info(
            "BUILDER2_SEMANTIC_BRIDGE_REPAIR_PRE_DISPATCH_FAILURE_RECOVERED jobId=%s prototypeId=%s candidateId=%s",
            job_id,
            prototype_id,
            candidate_id,
        )

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_DISPATCH_START jobId=%s candidateId=%s prototypeId=%s role=%s model=%s callKind=%s",
        job_id,
        candidate_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_ROLE,
        model,
        SEMANTIC_BRIDGE_REPAIR_CALL_KIND,
    )

    timer = MetricsTimer()
    dispatched = {"done": False}

    def _on_paid_request_submitted() -> None:
        if dispatched["done"]:
            return
        dispatched["done"] = True
        mark_semantic_bridge_repair_dispatched(state, prototype_id=prototype_id)
        record_model_call(
            state,
            role=SEMANTIC_BRIDGE_REPAIR_ROLE,
            elapsed_ms=timer.elapsed_ms(),
        )

    parsed = call_builder2_role_json(
        role=SEMANTIC_BRIDGE_REPAIR_ROLE,
        model=model,
        prompt=prompt,
        call_type="repair",
        llm_client=llm_client,
        on_paid_request_submitted=_on_paid_request_submitted,
    )
    mark_semantic_bridge_repair_response_received(state, prototype_id=prototype_id)
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_RESPONSE_RECEIVED jobId=%s candidateId=%s prototypeId=%s lifecycleState=%s",
        job_id,
        candidate_id,
        prototype_id,
        LIFECYCLE_RESPONSE_RECEIVED,
    )

    merged, meta = merge_semantic_bridge_repair_patch(base_parsed, parsed)
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_PATCH_RECEIVED candidateId=%s prototypeId=%s appliedPathCount=%s",
        candidate_id,
        prototype_id,
        len(meta.get("appliedPaths") or []),
    )

    candidate = validate_creator_candidate(
        merged,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        job_id=job_id,
        tournament_id=_clean(state.get("tournamentId")),
        candidate_id=candidate_id,
        tournament_state=state,
    )
    ok, field = validate_semantic_bridge_establishes_convergence(candidate)
    if not ok:
        bucket = _ledger_bucket(state, prototype_id)
        bucket["lifecycleState"] = LIFECYCLE_REJECTED
        bucket["failureCode"] = f"builder2_semantic_bridge_repair_incomplete:{field}"
        raise Builder2TournamentError(f"builder2_semantic_bridge_repair_incomplete:{field}")

    round_index = int(original_payload.get("roundIndex") or base_parsed.get("roundIndex") or 1)
    attempt_number = int(original_payload.get("attemptNumber") or base_parsed.get("attemptNumber") or 1)
    persist_accepted_creator_candidate(
        state,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        round_index=round_index,
        attempt_number=attempt_number,
        creator_output=candidate,
        strategy_foundation=strategy,
    )
    rec = state.setdefault("candidates", {}).setdefault(candidate_id, {})
    rec.update(
        {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": round_index,
            "attemptNumber": attempt_number,
            "creatorOutput": deepcopy(candidate),
            "creatorSnapshot": deepcopy(candidate),
            "validationStatus": "accepted",
            "creatorAcceptanceStatus": "accepted",
            "status": "accepted",
            "judgeStatus": "pending",
            "failureReason": None,
            "semanticBridgeRepairAcceptedAt": _utc_now_iso(),
        }
    )
    mark_semantic_bridge_repair_accepted(state, prototype_id=prototype_id)

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_MERGED candidateId=%s prototypeId=%s appliedPathCount=%s",
        candidate_id,
        prototype_id,
        len(meta.get("appliedPaths") or []),
    )
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_ACCEPTED jobId=%s candidateId=%s prototypeId=%s lifecycleState=%s",
        job_id,
        candidate_id,
        prototype_id,
        LIFECYCLE_ACCEPTED,
    )
    return candidate_id, candidate, meta
