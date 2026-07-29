"""
Builder2 Creator semantic-bridge repair — one bounded paid call after slogan repair.
"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
) -> Tuple[Dict[str, Any], str]:
    base = deepcopy(original_parsed)
    repaired_slogan = extract_repaired_slogan_text(repair_parsed)
    if not repaired_slogan:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:repaired_slogan_missing")
    _set_nested(base, "advertisingClosure.sloganText", repaired_slogan)
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
    persisted = int(bucket.get("persistedSemanticBridgeRepairCalls") or 0)
    metric_count = int(metrics.get("creatorSemanticBridgeRepairCalls") or 0)
    canonical = max(persisted, metric_count)
    if canonical > 1:
        canonical = 1
    bucket["persistedSemanticBridgeRepairCalls"] = canonical
    bucket["canonicalSemanticBridgeRepairCalls"] = canonical
    return bucket


def additional_semantic_bridge_repair_allowed(state: Dict[str, Any], prototype_id: str) -> bool:
    if not semantic_bridge_repair_env_authorized():
        return False
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    if int(bucket.get("canonicalSemanticBridgeRepairCalls") or 0) >= 1:
        return False
    reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)
    slogan_bucket = state.get(SLOGAN_REPAIR_CALL_LEDGER_KEY, {}).get(prototype_id, {})
    if not isinstance(slogan_bucket, dict):
        return False
    if int(slogan_bucket.get("canonicalCreatorRepairCalls") or 0) < 1:
        return False
    return True


def reserve_semantic_bridge_repair_call(state: Dict[str, Any], *, prototype_id: str) -> None:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    if int(bucket.get("canonicalSemanticBridgeRepairCalls") or 0) >= 1:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_call_already_used")
    bucket["persistedSemanticBridgeRepairCalls"] = 1
    bucket["canonicalSemanticBridgeRepairCalls"] = 1
    bucket["reservedAt"] = _utc_now_iso()
    bucket["callKind"] = SEMANTIC_BRIDGE_REPAIR_CALL_KIND


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


def populate_semantic_bridge_repair_call_report(
    state: Dict[str, Any],
    report: Dict[str, Any],
    *,
    prototype_id: str,
    invocation_semantic_bridge_repair_calls: int = 0,
    semantic_bridge_repair_accepted: bool = False,
) -> None:
    bucket = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id=prototype_id)
    persisted = int(bucket.get("canonicalSemanticBridgeRepairCalls") or 0)
    report["invocationSemanticBridgeRepairCalls"] = int(invocation_semantic_bridge_repair_calls)
    report["persistedSemanticBridgeRepairCalls"] = persisted
    report["totalSemanticBridgeRepairCalls"] = persisted
    report["semanticBridgeRepairAuthorized"] = semantic_bridge_repair_env_authorized()
    report["semanticBridgeRepairAccepted"] = bool(semantic_bridge_repair_accepted)


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
    from engine.builder2_prototypes import require_prototype
    from engine.builder2_tournament_prompts import build_semantic_bridge_repair_prompt

    if llm_client is None:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:llm_client_missing")
    if not additional_semantic_bridge_repair_allowed(state, prototype_id):
        raise Builder2TournamentError("builder2_semantic_bridge_repair_not_authorized")

    context = detect_semantic_bridge_repair_context(
        state,
        prototype_id=prototype_id,
        product_name=product_name,
        compatibility_mode=compatibility_mode,
        original_candidate_id=original_candidate_id,
        repair_candidate_id=repair_candidate_id,
    )
    if not context["required"]:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_not_required")

    base_parsed = context["baseParsed"]
    original_payload = context["originalPayload"]
    failure_paths = context["failurePaths"]
    candidate_id = accept_candidate_id or context["acceptCandidateId"]
    if not candidate_id:
        candidate_id = f"cand-1-{prototype_id}-1-semantic-bridge-repair"

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

    reserve_semantic_bridge_repair_call(state, prototype_id=prototype_id)

    prototype = require_prototype(prototype_id)
    display_name = prototype.display_name
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    prompt = build_semantic_bridge_repair_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy,
        prototype=prototype,
        candidate_id=candidate_id,
        base_candidate=base_parsed,
        validation_failures=failure_paths,
    )

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_START candidateId=%s prototypeId=%s callKind=%s",
        candidate_id,
        prototype_id,
        SEMANTIC_BRIDGE_REPAIR_CALL_KIND,
    )

    timer = MetricsTimer()
    response = llm_client(prompt=prompt, model_role="builder2_creator_semantic_bridge_repair")
    record_model_call(
        state,
        role="builder2_creator_semantic_bridge_repair",
        elapsed_ms=timer.elapsed_ms(),
    )

    if isinstance(response, str):
        parsed = json.loads(response)
    elif isinstance(response, dict):
        parsed = response
    else:
        raise Builder2TournamentError("builder2_semantic_bridge_repair_invalid:response_type")

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
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        candidate_id=candidate_id,
        tournament_state=state,
    )
    ok, field = validate_semantic_bridge_establishes_convergence(candidate)
    if not ok:
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

    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_MERGED candidateId=%s prototypeId=%s appliedPathCount=%s",
        candidate_id,
        prototype_id,
        len(meta.get("appliedPaths") or []),
    )
    logger.info(
        "BUILDER2_SEMANTIC_BRIDGE_REPAIR_ACCEPTED candidateId=%s prototypeId=%s meaningsConverge=%s",
        candidate_id,
        prototype_id,
        str(candidate.get("semanticBridge", {}).get("meaningsConverge") is True).lower(),
    )
    return candidate_id, candidate, meta
