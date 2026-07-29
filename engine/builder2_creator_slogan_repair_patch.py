"""
Builder2 Creator slogan word-limit repair — bounded patch merge and offline salvage.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from engine.builder2_advertising_closure_contract import (
    SLOGAN_MAX_WORD_COUNT,
    count_slogan_words_excluding_product,
    validate_slogan_text_structure,
)
from engine.builder2_creator import collect_creator_structural_errors, is_slogan_word_limit_failure, validate_creator_candidate
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SLOGAN_REPAIR_PATCH_ROOT = "sloganRepairPatch"
SLOGAN_REPAIR_PARSED_INDEX_KEY = "sloganRepairParsedResponses"
SLOGAN_REPAIR_CALL_LEDGER_KEY = "sloganRepairCallLedger"

ALLOWLIST_PATHS: Tuple[str, ...] = (
    "advertisingClosure.sloganText",
    "semanticBridge.sloganMeaning",
    "semanticBridge.howTheMeaningsMeet",
    "metaphoricalEmbodiment.sloganBridgeToBusinessMeaning",
    "visualBridgeAssessment.sloganConnectionToVisibleDetail",
    "visualBridgeAssessment.sloganConnectionToRelativeAdvantage",
    "verbalPotential.keywordOrKeyPhrase",
    "verbalPotential.strategicMeaning",
)

SEMANTIC_BRIDGE_PRESERVED_KEYS: Tuple[str, ...] = (
    "keyWordOrConcept",
    "visualMeaning",
    "strategicMeaning",
    "understandableWithoutCreatorReport",
    "dualMeaningUsed",
    "physicalMeaningActivatedByVisual",
    "strategicMeaningActivatedBySlogan",
    "meaningsConverge",
)

SEMANTIC_BRIDGE_SLOGAN_DEPENDENT_KEYS: Tuple[str, ...] = (
    "sloganMeaning",
    "howTheMeaningsMeet",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _get_nested(obj: Dict[str, Any], path: str) -> Any:
    current: Any = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def _delete_nested(obj: Dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: Any = obj
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(part)
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _flatten_leaf_paths(obj: Any, *, prefix: str = "") -> Set[str]:
    paths: Set[str] = set()
    if isinstance(obj, dict):
        if not obj:
            if prefix:
                paths.add(prefix)
            return paths
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_flatten_leaf_paths(value, prefix=child))
    elif isinstance(obj, list):
        if prefix:
            paths.add(prefix)
    else:
        if prefix:
            paths.add(prefix)
    return paths


def _path_allowed(path: str) -> bool:
    return path in ALLOWLIST_PATHS or any(path.startswith(f"{allowed}.") for allowed in ALLOWLIST_PATHS)


def extract_slogan_repair_patch(repair_response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(repair_response, dict):
        return {}
    patch_root = repair_response.get(SLOGAN_REPAIR_PATCH_ROOT)
    if isinstance(patch_root, dict) and patch_root:
        return deepcopy(patch_root)
    extracted: Dict[str, Any] = {}
    for path in ALLOWLIST_PATHS:
        value = _get_nested(repair_response, path)
        if value is not None and value != "":
            _set_nested(extracted, path, deepcopy(value))
    return extracted


def _apply_allowlisted_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    merged = deepcopy(base)
    applied: List[str] = []
    for path in ALLOWLIST_PATHS:
        value = _get_nested(patch, path)
        if value is None or value == "":
            continue
        _set_nested(merged, path, deepcopy(value))
        applied.append(path)
    return merged, applied


def _preserve_semantic_bridge_basis(base: Dict[str, Any], merged: Dict[str, Any], patch: Dict[str, Any]) -> List[str]:
    reverted: List[str] = []
    base_bridge = base.get("semanticBridge") if isinstance(base.get("semanticBridge"), dict) else {}
    merged_bridge = merged.setdefault("semanticBridge", {})
    if not isinstance(merged_bridge, dict):
        merged_bridge = {}
        merged["semanticBridge"] = merged_bridge
    for key in SEMANTIC_BRIDGE_PRESERVED_KEYS:
        if key not in base_bridge:
            continue
        before = merged_bridge.get(key)
        merged_bridge[key] = deepcopy(base_bridge[key])
        if before != merged_bridge.get(key):
            reverted.append(f"semanticBridge.{key}")
    for key in SEMANTIC_BRIDGE_SLOGAN_DEPENDENT_KEYS:
        patch_value = _get_nested(patch, f"semanticBridge.{key}")
        if patch_value is not None and _clean(patch_value):
            merged_bridge[key] = patch_value
    base_converge = base_bridge.get("meaningsConverge")
    if base_converge is True and merged_bridge.get("meaningsConverge") is not True:
        merged_bridge["meaningsConverge"] = True
        reverted.append("semanticBridge.meaningsConverge")
    return reverted


def diff_changed_paths(base: Dict[str, Any], merged: Dict[str, Any]) -> Set[str]:
    base_paths = _flatten_leaf_paths(base)
    merged_paths = _flatten_leaf_paths(merged)
    changed: Set[str] = set()
    for path in base_paths | merged_paths:
        if _get_nested(base, path) != _get_nested(merged, path):
            changed.add(path)
    return changed


def merge_slogan_repair_patch_response(
    base_candidate: Dict[str, Any],
    repair_response: Dict[str, Any],
    *,
    product_name: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    patch = extract_slogan_repair_patch(repair_response)
    if not patch and not _clean(_get_nested(repair_response, "advertisingClosure.sloganText")):
        raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:patch_missing")

    merged, applied_paths = _apply_allowlisted_patch(base_candidate, patch or repair_response)
    reverted_paths = _preserve_semantic_bridge_basis(base_candidate, merged, patch or repair_response)

    changed_paths = sorted(diff_changed_paths(base_candidate, merged))
    forbidden = [path for path in changed_paths if not _path_allowed(path)]
    if forbidden:
        for path in forbidden:
            base_value = _get_nested(base_candidate, path)
            if base_value is None:
                _delete_nested(merged, path)
            else:
                _set_nested(merged, path, deepcopy(base_value))
            if path not in reverted_paths:
                reverted_paths.append(path)
        logger.info(
            "BUILDER2_SLOGAN_REPAIR_FORBIDDEN_CHANGE_REVERTED pathCount=%s paths=%s",
            len(forbidden),
            ",".join(forbidden[:12]),
        )

    closure = merged.get("advertisingClosure") if isinstance(merged.get("advertisingClosure"), dict) else {}
    product_label = _clean(closure.get("productNameText") or product_name)
    slogan = _clean(closure.get("sloganText"))
    if not slogan:
        raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:advertisingClosure.sloganText")
    validate_slogan_text_structure(slogan=slogan, product_name=product_label)

    logger.info(
        "BUILDER2_SLOGAN_REPAIR_PATCH_RECEIVED appliedPathCount=%s revertedPathCount=%s configuredWordLimit=%s actualWordCount=%s",
        len(applied_paths),
        len(reverted_paths),
        SLOGAN_MAX_WORD_COUNT,
        count_slogan_words_excluding_product(slogan, product_label),
    )
    if applied_paths:
        logger.info(
            "BUILDER2_SLOGAN_REPAIR_PATCH_PATHS_VALIDATED paths=%s",
            ",".join(applied_paths),
        )

    meta = {
        "appliedPaths": applied_paths,
        "revertedPaths": reverted_paths,
        "changedPaths": sorted(diff_changed_paths(base_candidate, merged)),
        "actualWordCount": count_slogan_words_excluding_product(slogan, product_label),
    }
    return merged, meta


def original_candidate_slogan_only_structural_errors(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> List[str]:
    errors = collect_creator_structural_errors(
        candidate,
        assigned_prototype_id=assigned_prototype_id,
        prototype_display_name=prototype_display_name,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    return [item for item in errors if item.endswith(":sloganText.word_limit") or item == "builder2_advertising_closure_invalid:sloganText.word_limit"]


def candidate_fails_only_slogan_word_limit(
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
    word_limit_only = [
        item
        for item in errors
        if is_slogan_word_limit_failure(item) or item.endswith(":sloganText.word_limit")
    ]
    if not word_limit_only:
        return False, errors
    non_word_limit = [
        item
        for item in errors
        if not (is_slogan_word_limit_failure(item) or item.endswith(":sloganText.word_limit"))
    ]
    if non_word_limit:
        return False, errors
    try:
        validate_creator_candidate(
            candidate,
            assigned_prototype_id=assigned_prototype_id,
            prototype_display_name=prototype_display_name,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "")
        if is_slogan_word_limit_failure(reason) or reason.endswith(":sloganText.word_limit"):
            return True, [reason]
        return False, [reason]
    return False, []


def persist_slogan_repair_parsed_response(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    parsed: Dict[str, Any],
    failure_reason: str = "",
    source: str = "creator_repair",
) -> None:
    if not isinstance(parsed, dict) or not parsed:
        return
    index = state.setdefault(SLOGAN_REPAIR_PARSED_INDEX_KEY, {})
    if not isinstance(index, dict):
        index = {}
        state[SLOGAN_REPAIR_PARSED_INDEX_KEY] = index
    index[candidate_id] = {
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "parsed": deepcopy(parsed),
        "failureReason": failure_reason,
        "source": source,
        "storedAt": _utc_now_iso(),
    }


def load_slogan_repair_parsed_response(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    index = state.get(SLOGAN_REPAIR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        return None
    payload = index.get(candidate_id)
    if not isinstance(payload, dict):
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict) or not parsed:
        return None
    return deepcopy(payload)


def find_slogan_repair_patch_source(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    preferred_candidate_id: str = "",
) -> Optional[Dict[str, Any]]:
    if preferred_candidate_id:
        loaded = load_slogan_repair_parsed_response(state, preferred_candidate_id)
        if loaded:
            return loaded
    index = state.get(SLOGAN_REPAIR_PARSED_INDEX_KEY)
    latest: Optional[Dict[str, Any]] = None
    if isinstance(index, dict):
        for payload in index.values():
            if isinstance(payload, dict) and _clean(payload.get("prototypeId")) == prototype_id:
                latest = deepcopy(payload)
    if latest:
        return latest
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY

    rejected_index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(rejected_index, dict):
        return None
    for candidate_id, payload in rejected_index.items():
        if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
            continue
        if candidate_id == preferred_candidate_id and preferred_candidate_id:
            return deepcopy(payload)
        failure = _clean(payload.get("failureReason"))
        if failure and not is_slogan_word_limit_failure(failure):
            latest = deepcopy(payload)
    return latest


def find_original_slogan_word_limit_rejection(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    preferred_candidate_id: str = "",
) -> Optional[Dict[str, Any]]:
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY

    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        return None
    if preferred_candidate_id and preferred_candidate_id in index:
        payload = index.get(preferred_candidate_id)
        if isinstance(payload, dict) and is_slogan_word_limit_failure(_clean(payload.get("failureReason"))):
            return deepcopy(payload)
    for payload in index.values():
        if not isinstance(payload, dict):
            continue
        if _clean(payload.get("prototypeId")) != prototype_id:
            continue
        if is_slogan_word_limit_failure(_clean(payload.get("failureReason"))):
            return deepcopy(payload)
    return None


def prototype_display_name_for_id(prototype_id: str) -> str:
    from engine.builder2_prototypes import require_prototype

    return require_prototype(prototype_id).display_name


def _ledger_bucket(state: Dict[str, Any], prototype_id: str) -> Dict[str, Any]:
    root = state.setdefault(SLOGAN_REPAIR_CALL_LEDGER_KEY, {})
    if not isinstance(root, dict):
        root = {}
        state[SLOGAN_REPAIR_CALL_LEDGER_KEY] = root
    bucket = root.setdefault(prototype_id, {})
    if not isinstance(bucket, dict):
        bucket = {}
        root[prototype_id] = bucket
    return bucket


def sync_slogan_repair_call_ledger_from_metrics(state: Dict[str, Any], *, prototype_id: str) -> Dict[str, Any]:
    bucket = _ledger_bucket(state, prototype_id)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    if find_original_slogan_word_limit_rejection(state, prototype_id) or _clean(
        (find_rejected_creator_for_prototype(state, prototype_id) or {}).get("failureReason")
    ):
        bucket.setdefault("persistedCreatorNormalCalls", 1)
    bucket["persistedCreatorRepairCalls"] = max(
        int(bucket.get("persistedCreatorRepairCalls") or 0),
        int(metrics.get("creatorRepairCalls") or 0),
    )
    bucket.setdefault("persistedJudgeCalls", int(metrics.get("judgeCalls") or 0))
    bucket.setdefault("persistedWinnerCalls", int(metrics.get("winnerDevelopmentCalls") or 0))
    bucket.setdefault(
        "persistedRunwaySubmissionCalls",
        int((state.get("mediaResume") or {}).get("runwaySubmissionCalls") or 0),
    )
    return bucket


def find_rejected_creator_for_prototype(state: Dict[str, Any], prototype_id: str) -> Optional[Dict[str, Any]]:
    from engine.builder2_complete_ad_creator_recovery import find_rejected_creator_for_prototype as _find

    return _find(state, prototype_id)


def populate_slogan_repair_call_report(
    state: Dict[str, Any],
    report: Dict[str, Any],
    *,
    prototype_id: str,
    invocation_creator_normal_calls: int = 0,
    invocation_creator_repair_calls: int = 0,
) -> None:
    bucket = sync_slogan_repair_call_ledger_from_metrics(state, prototype_id=prototype_id)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    persisted_normal = int(bucket.get("persistedCreatorNormalCalls") or 0)
    persisted_repair = int(bucket.get("persistedCreatorRepairCalls") or 0)
    bucket["persistedCreatorNormalCalls"] = persisted_normal
    bucket["persistedCreatorRepairCalls"] = persisted_repair
    report["invocationCreatorNormalCalls"] = int(invocation_creator_normal_calls)
    report["invocationCreatorRepairCalls"] = int(invocation_creator_repair_calls)
    report["persistedCreatorNormalCalls"] = persisted_normal
    report["persistedCreatorRepairCalls"] = persisted_repair
    report["totalCreatorNormalCalls"] = persisted_normal
    report["totalCreatorRepairCalls"] = persisted_repair
    report["additionalPaidRepairAllowed"] = additional_paid_slogan_repair_allowed(state, prototype_id)
    report["offlineSalvageAttempted"] = bool(bucket.get("offlineSalvageAttempted"))
    report["offlineSalvageAccepted"] = bool(bucket.get("offlineSalvageAccepted"))


def additional_paid_slogan_repair_allowed(state: Dict[str, Any], prototype_id: str) -> bool:
    bucket = sync_slogan_repair_call_ledger_from_metrics(state, prototype_id=prototype_id)
    repair_calls = int(bucket.get("persistedCreatorRepairCalls") or 0)
    if repair_calls >= 1 and bucket.get("offlineSalvageAccepted"):
        return False
    if repair_calls >= 1 and find_slogan_repair_patch_source(state, prototype_id):
        return False
    return repair_calls < 1


def record_offline_slogan_salvage_attempt(state: Dict[str, Any], *, prototype_id: str, accepted: bool) -> None:
    bucket = _ledger_bucket(state, prototype_id)
    bucket["offlineSalvageAttempted"] = True
    bucket["offlineSalvageAccepted"] = accepted


def validate_and_merge_slogan_repair_candidate(
    base_candidate: Dict[str, Any],
    repair_response: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    prototype_display_name: str,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    product_name: str = "",
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    merged, meta = merge_slogan_repair_patch_response(
        base_candidate,
        repair_response,
        product_name=product_name,
    )
    try:
        candidate = validate_creator_candidate(
            merged,
            assigned_prototype_id=assigned_prototype_id,
            prototype_display_name=prototype_display_name,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            tournament_state=tournament_state,
        )
    except Builder2TournamentError as exc:
        field = str(exc.args[0] if exc.args else "")
        if "semanticBridge.meaningsConverge" in field:
            raise Builder2TournamentError("builder2_slogan_repair_patch_regressed_valid_field:semanticBridge.meaningsConverge") from exc
        raise Builder2TournamentError(f"builder2_slogan_repair_offline_merge_invalid:{field.split(':')[-1]}") from exc
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_MERGED_CANDIDATE_VALIDATED candidateId=%s prototypeId=%s appliedPathCount=%s",
        candidate_id or "(none)",
        assigned_prototype_id,
        len(meta.get("appliedPaths") or []),
    )
    return candidate, meta


def try_offline_slogan_repair_salvage_for_prototype(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    product_name: str = "",
    compatibility_mode: bool = False,
    original_candidate_id: str = "",
    patch_candidate_id: str = "",
    accept_candidate_id: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str], List[str]]:
    """
    Returns (accepted, candidate_id, failure_code, failing_paths).
    Makes zero OpenAI/Judge/Winner/media calls.
    """
    from engine.builder2_accepted_creator_store import persist_accepted_creator_candidate
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY

    display_name = prototype_display_name_for_id(prototype_id)
    record_offline_slogan_salvage_attempt(state, prototype_id=prototype_id, accepted=False)
    original_payload = find_original_slogan_word_limit_rejection(
        state,
        prototype_id,
        preferred_candidate_id=original_candidate_id,
    )
    patch_payload = find_slogan_repair_patch_source(
        state,
        prototype_id,
        preferred_candidate_id=patch_candidate_id,
    )
    failing_paths: List[str] = []
    if original_payload is None:
        return False, None, "builder2_slogan_repair_offline_merge_invalid:original_missing", ["original_candidate"]
    if patch_payload is None:
        return False, None, "builder2_slogan_repair_offline_merge_invalid:patch_missing", ["repair_patch_source"]
    base_parsed = dict(original_payload.get("parsed") or {})
    patch_parsed = dict(patch_payload.get("parsed") or {})
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    only_word_limit, structural_errors = candidate_fails_only_slogan_word_limit(
        base_parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        product_name=product_name,
    )
    if not only_word_limit and structural_errors:
        failing_paths = [item.split(":", 1)[-1] for item in structural_errors]
        return False, None, "builder2_slogan_repair_offline_merge_invalid:original_not_word_limit_only", failing_paths

    resolved_candidate_id = accept_candidate_id or _clean(patch_payload.get("candidateId")) or _clean(original_payload.get("candidateId"))
    if not resolved_candidate_id:
        resolved_candidate_id = f"cand-1-{prototype_id}-1-offline-salvage"
    round_index = int(original_payload.get("roundIndex") or base_parsed.get("roundIndex") or 1)
    attempt_number = int(original_payload.get("attemptNumber") or base_parsed.get("attemptNumber") or 1)
    try:
        candidate, _meta = validate_and_merge_slogan_repair_candidate(
            base_parsed,
            patch_parsed,
            assigned_prototype_id=prototype_id,
            prototype_display_name=display_name,
            strategy_foundation=strategy,
            compatibility_mode=compatibility_mode,
            product_name=product_name or _clean(strategy.get("productNameResolved")),
            job_id=_clean(state.get("jobId")),
            tournament_id=_clean(state.get("tournamentId")),
            candidate_id=resolved_candidate_id,
            tournament_state=state,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_slogan_repair_offline_merge_invalid")
        failing_paths = [reason.split(":")[-1]]
        return False, resolved_candidate_id, reason, failing_paths

    persist_accepted_creator_candidate(
        state,
        candidate_id=resolved_candidate_id,
        prototype_id=prototype_id,
        round_index=round_index,
        attempt_number=attempt_number,
        creator_output=candidate,
        strategy_foundation=strategy,
    )
    rec = state.setdefault("candidates", {}).setdefault(resolved_candidate_id, {})
    rec.update(
        {
            "candidateId": resolved_candidate_id,
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
            "offlineSloganRepairSalvageAt": _utc_now_iso(),
        }
    )
    rejected_index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if isinstance(rejected_index, dict):
        for cid in list(rejected_index.keys()):
            payload = rejected_index.get(cid)
            if isinstance(payload, dict) and _clean(payload.get("prototypeId")) == prototype_id:
                if is_slogan_word_limit_failure(_clean(payload.get("failureReason"))):
                    rejected_index.pop(cid, None)
    record_offline_slogan_salvage_attempt(state, prototype_id=prototype_id, accepted=True)
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_OFFLINE_SALVAGE_ACCEPTED jobId=%s prototypeId=%s candidateId=%s",
        _clean(state.get("jobId")) or "(none)",
        prototype_id,
        resolved_candidate_id,
    )
    return True, resolved_candidate_id, None, []
