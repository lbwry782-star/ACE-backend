"""
Builder2 Creator slogan word-limit repair — bounded patch merge and offline salvage.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from engine.builder2_advertising_closure_contract import (
    SLOGAN_MAX_WORD_COUNT,
    count_slogan_words_excluding_product,
    validate_slogan_text_structure,
)
from engine.builder2_creator import collect_creator_structural_errors, is_slogan_word_limit_failure, validate_creator_candidate
from engine.builder2_slogan_repair_provenance import (
    SOURCE_ROLE_ORIGINAL,
    SOURCE_ROLE_REPAIR,
    assert_semantic_basis_unchanged,
    log_semantic_basis_fingerprint,
    resolve_slogan_repair_base_and_source,
    semantic_basis_fingerprint,
    validate_semantic_basis_with_repaired_slogan,
)
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

DEPENDENCY_GROUP_A = "A"
DEPENDENCY_GROUP_B = "B"
DEPENDENCY_GROUP_C = "C"
DEPENDENCY_GROUP_D = "D"
DEPENDENCY_GROUP_E = "E"

DEPENDENCY_GROUPS: Dict[str, Tuple[str, ...]] = {
    DEPENDENCY_GROUP_A: ("advertisingClosure.sloganText",),
    DEPENDENCY_GROUP_B: ("semanticBridge.sloganMeaning", "semanticBridge.howTheMeaningsMeet"),
    DEPENDENCY_GROUP_C: (
        "visualBridgeAssessment.sloganConnectionToVisibleDetail",
        "visualBridgeAssessment.sloganConnectionToRelativeAdvantage",
    ),
    DEPENDENCY_GROUP_D: ("metaphoricalEmbodiment.sloganBridgeToBusinessMeaning",),
    DEPENDENCY_GROUP_E: ("verbalPotential.keywordOrKeyPhrase", "verbalPotential.strategicMeaning"),
}

OPTIONAL_DEPENDENCY_GROUP_ORDER: Tuple[str, ...] = (
    DEPENDENCY_GROUP_B,
    DEPENDENCY_GROUP_C,
    DEPENDENCY_GROUP_D,
    DEPENDENCY_GROUP_E,
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


def extract_repaired_slogan_text(repair_response: Dict[str, Any]) -> str:
    patch = extract_slogan_repair_patch(repair_response)
    slogan = _clean(_get_nested(patch, "advertisingClosure.sloganText"))
    if not slogan:
        slogan = _clean(_get_nested(repair_response, "advertisingClosure.sloganText"))
    return slogan


def optional_patch_paths_available(repair_response: Dict[str, Any]) -> Dict[str, List[str]]:
    patch = extract_slogan_repair_patch(repair_response)
    available: Dict[str, List[str]] = {}
    for group_id in OPTIONAL_DEPENDENCY_GROUP_ORDER:
        paths: List[str] = []
        for path in DEPENDENCY_GROUPS[group_id]:
            value = _get_nested(patch, path)
            if value is None:
                value = _get_nested(repair_response, path)
            if value is not None and value != "":
                paths.append(path)
        if paths:
            available[group_id] = paths
    return available


def _iter_optional_group_subsets() -> List[Tuple[str, ...]]:
    subsets: List[Tuple[str, ...]] = [()]
    for size in range(1, len(OPTIONAL_DEPENDENCY_GROUP_ORDER) + 1):
        for combo in combinations(OPTIONAL_DEPENDENCY_GROUP_ORDER, size):
            subsets.append(combo)
    return subsets


def _paths_for_groups(
    groups: Sequence[str],
    *,
    repaired_slogan: str,
    optional_available: Dict[str, List[str]],
) -> List[str]:
    paths: List[str] = []
    if repaired_slogan:
        paths.append("advertisingClosure.sloganText")
    for group_id in groups:
        paths.extend(optional_available.get(group_id, []))
    return paths


def _apply_selected_patch_paths(
    base_candidate: Dict[str, Any],
    repair_response: Dict[str, Any],
    *,
    paths: Sequence[str],
    repaired_slogan: str,
) -> Tuple[Dict[str, Any], List[str]]:
    merged = deepcopy(base_candidate)
    patch = extract_slogan_repair_patch(repair_response)
    applied: List[str] = []
    for path in paths:
        if path == "advertisingClosure.sloganText":
            if not repaired_slogan:
                continue
            _set_nested(merged, path, repaired_slogan)
            applied.append(path)
            continue
        value = _get_nested(patch, path)
        if value is None:
            value = _get_nested(repair_response, path)
        if value is None or value == "":
            continue
        _set_nested(merged, path, deepcopy(value))
        applied.append(path)
    return merged, applied


def preserve_semantic_bridge_basis(base: Dict[str, Any], merged: Dict[str, Any]) -> List[str]:
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
    return reverted


def revert_forbidden_paths(base_candidate: Dict[str, Any], merged: Dict[str, Any]) -> List[str]:
    reverted: List[str] = []
    changed_paths = sorted(diff_changed_paths(base_candidate, merged))
    forbidden = [path for path in changed_paths if not _path_allowed(path)]
    if forbidden:
        for path in forbidden:
            base_value = _get_nested(base_candidate, path)
            if base_value is None:
                _delete_nested(merged, path)
            else:
                _set_nested(merged, path, deepcopy(base_value))
            reverted.append(path)
        logger.info(
            "BUILDER2_SLOGAN_REPAIR_FORBIDDEN_CHANGE_REVERTED pathCount=%s paths=%s",
            len(forbidden),
            ",".join(forbidden[:12]),
        )
    return reverted


def diff_changed_paths(base: Dict[str, Any], merged: Dict[str, Any]) -> Set[str]:
    base_paths = _flatten_leaf_paths(base)
    merged_paths = _flatten_leaf_paths(merged)
    changed: Set[str] = set()
    for path in base_paths | merged_paths:
        if _get_nested(base, path) != _get_nested(merged, path):
            changed.add(path)
    return changed


def _validation_failure_field(exc: Builder2TournamentError) -> str:
    reason = str(exc.args[0] if exc.args else "")
    if ":" in reason:
        return reason.split(":", 1)[-1]
    return reason


def _classify_slogan_repair_validation_failure(field: str, *, forbidden_reverted: Sequence[str]) -> str:
    if field in forbidden_reverted:
        return f"builder2_slogan_repair_forbidden_path_attempted:{field}"
    if field.startswith("semanticBridge.") or field.startswith("visualBridgeAssessment.") or field.startswith(
        "metaphoricalEmbodiment."
    ) or field.startswith("verbalPotential."):
        return f"builder2_slogan_repair_allowed_patch_semantic_incompatibility:{field}"
    return f"builder2_slogan_repair_offline_merge_invalid:{field}"


def select_minimal_validating_patch(
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
    from engine.builder2_creator_normalization import normalize_creator_candidate
    from engine.builder2_creator import normalize_creator_raw

    base_candidate = deepcopy(base_candidate)
    base_fingerprint = semantic_basis_fingerprint(base_candidate)
    log_semantic_basis_fingerprint(
        candidate_id=candidate_id,
        source_role=SOURCE_ROLE_ORIGINAL,
        candidate=base_candidate,
        log_event="BUILDER2_SLOGAN_REPAIR_BASE_FINGERPRINT",
    )
    repaired_slogan = extract_repaired_slogan_text(repair_response)
    if not repaired_slogan:
        raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:patch_missing")

    closure = base_candidate.get("advertisingClosure") if isinstance(base_candidate.get("advertisingClosure"), dict) else {}
    product_label = _clean(closure.get("productNameText") or product_name)
    validate_slogan_text_structure(slogan=repaired_slogan, product_name=product_label)

    optional_available = optional_patch_paths_available(repair_response)
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_MINIMAL_PATCH_START candidateId=%s prototypeId=%s optionalGroupCount=%s configuredWordLimit=%s actualWordCount=%s",
        candidate_id or "(none)",
        assigned_prototype_id,
        len(optional_available),
        SLOGAN_MAX_WORD_COUNT,
        count_slogan_words_excluding_product(repaired_slogan, product_label),
    )

    passing: List[Tuple[Dict[str, Any], List[str], Tuple[str, ...], str]] = []
    last_failure_field = ""

    for groups in _iter_optional_group_subsets():
        paths = _paths_for_groups(groups, repaired_slogan=repaired_slogan, optional_available=optional_available)
        merged, applied_paths = _apply_selected_patch_paths(
            base_candidate,
            repair_response,
            paths=paths,
            repaired_slogan=repaired_slogan,
        )
        reverted_paths = preserve_semantic_bridge_basis(base_candidate, merged)
        forbidden_reverted = revert_forbidden_paths(base_candidate, merged)
        reverted_paths = list(reverted_paths) + list(forbidden_reverted)
        if "advertisingClosure.sloganText" in applied_paths:
            from engine.builder2_advertising_slogan_quality_contract import sync_creator_slogan_formulation_from_closure

            sync_creator_slogan_formulation_from_closure(merged, strategy_foundation=strategy_foundation)
        group_label = ",".join(groups) if groups else "A"
        merged_fingerprint = semantic_basis_fingerprint(merged)
        if merged_fingerprint != base_fingerprint:
            raise Builder2TournamentError("builder2_slogan_repair_normalization_mutated_protected_basis")
        log_semantic_basis_fingerprint(
            candidate_id=candidate_id,
            source_role=SOURCE_ROLE_ORIGINAL,
            candidate=merged,
            log_event="BUILDER2_SLOGAN_REPAIR_MERGED_PRE_NORMALIZE_FINGERPRINT",
        )
        validation_input = deepcopy(merged)
        try:
            normalized, _resolved = normalize_creator_candidate(
                validation_input,
                assigned_prototype_id=assigned_prototype_id,
                prototype_display_name=prototype_display_name,
                strategy_foundation=strategy_foundation,
                compatibility_mode=compatibility_mode,
                base_normalizer=normalize_creator_raw,
                job_id=job_id,
                candidate_id=candidate_id,
            )
        except Builder2TournamentError as exc:
            last_failure_field = _validation_failure_field(exc)
            logger.info(
                "BUILDER2_SLOGAN_REPAIR_MINIMAL_PATCH_VARIANT_TESTED candidateId=%s prototypeId=%s groups=%s "
                "appliedPathCount=%s validationResult=failed failureField=%s",
                candidate_id or "(none)",
                assigned_prototype_id,
                group_label,
                len(applied_paths),
                last_failure_field,
            )
            continue
        assert_semantic_basis_unchanged(
            before=merged,
            after=normalized,
            failure_code="builder2_slogan_repair_normalization_mutated_protected_basis",
        )
        log_semantic_basis_fingerprint(
            candidate_id=candidate_id,
            source_role=SOURCE_ROLE_ORIGINAL,
            candidate=normalized,
            log_event="BUILDER2_SLOGAN_REPAIR_MERGED_POST_NORMALIZE_FINGERPRINT",
        )
        validation_snapshot = deepcopy(normalized)
        validation_fingerprint = semantic_basis_fingerprint(validation_snapshot)
        log_semantic_basis_fingerprint(
            candidate_id=candidate_id,
            source_role=SOURCE_ROLE_ORIGINAL,
            candidate=validation_snapshot,
            log_event="BUILDER2_SLOGAN_REPAIR_VALIDATION_INPUT_FINGERPRINT",
        )
        try:
            candidate = validate_creator_candidate(
                validation_snapshot,
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
            last_failure_field = _validation_failure_field(exc)
            logger.info(
                "BUILDER2_SLOGAN_REPAIR_MINIMAL_PATCH_VARIANT_TESTED candidateId=%s prototypeId=%s groups=%s "
                "appliedPathCount=%s validationResult=failed failureField=%s",
                candidate_id or "(none)",
                assigned_prototype_id,
                group_label,
                len(applied_paths),
                last_failure_field,
            )
            if groups and last_failure_field.startswith("semanticBridge."):
                logger.info(
                    "BUILDER2_SLOGAN_REPAIR_ALLOWED_PATCH_INCOMPATIBLE candidateId=%s prototypeId=%s groups=%s failureField=%s",
                    candidate_id or "(none)",
                    assigned_prototype_id,
                    group_label,
                    last_failure_field,
                )
            continue

        if semantic_basis_fingerprint(validation_snapshot) != validation_fingerprint:
            raise Builder2TournamentError("builder2_slogan_repair_validation_mutated_protected_basis")

        logger.info(
            "BUILDER2_SLOGAN_REPAIR_MINIMAL_PATCH_VARIANT_TESTED candidateId=%s prototypeId=%s groups=%s "
            "appliedPathCount=%s validationResult=passed",
            candidate_id or "(none)",
            assigned_prototype_id,
            group_label,
            len(applied_paths),
        )
        passing.append((candidate, applied_paths, groups, group_label))

    if not passing:
        if last_failure_field:
            raise Builder2TournamentError(f"builder2_slogan_repair_no_valid_minimal_patch:{last_failure_field}")
        raise Builder2TournamentError("builder2_slogan_repair_no_valid_minimal_patch:validation_failed")

    passing.sort(key=lambda item: (len(item[1]), item[2]))
    candidate, applied_paths, groups, group_label = passing[0]
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_MINIMAL_PATCH_SELECTED candidateId=%s prototypeId=%s groups=%s appliedPathCount=%s paths=%s",
        candidate_id or "(none)",
        assigned_prototype_id,
        group_label,
        len(applied_paths),
        ",".join(applied_paths),
    )
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_PATCH_RECEIVED appliedPathCount=%s revertedPathCount=%s configuredWordLimit=%s actualWordCount=%s",
        len(applied_paths),
        0,
        SLOGAN_MAX_WORD_COUNT,
        count_slogan_words_excluding_product(
            _clean(_get_nested(candidate, "advertisingClosure.sloganText")),
            product_label,
        ),
    )
    if applied_paths:
        logger.info(
            "BUILDER2_SLOGAN_REPAIR_PATCH_PATHS_VALIDATED paths=%s",
            ",".join(applied_paths),
        )
    meta = {
        "appliedPaths": applied_paths,
        "revertedPaths": [],
        "changedPaths": sorted(diff_changed_paths(base_candidate, candidate)),
        "selectedGroups": list(groups),
        "actualWordCount": count_slogan_words_excluding_product(
            _clean(_get_nested(candidate, "advertisingClosure.sloganText")),
            product_label,
        ),
    }
    return candidate, meta


def merge_slogan_repair_patch_response(
    base_candidate: Dict[str, Any],
    repair_response: Dict[str, Any],
    *,
    product_name: str = "",
    apply_all_allowlisted: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if apply_all_allowlisted:
        patch = extract_slogan_repair_patch(repair_response)
        if not patch and not _clean(_get_nested(repair_response, "advertisingClosure.sloganText")):
            raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:patch_missing")
        paths = [path for path in ALLOWLIST_PATHS if _get_nested(patch or repair_response, path) not in (None, "")]
        if "advertisingClosure.sloganText" not in paths:
            slogan = extract_repaired_slogan_text(repair_response)
            if slogan:
                paths = ["advertisingClosure.sloganText", *paths]
        merged, applied_paths = _apply_selected_patch_paths(
            base_candidate,
            repair_response,
            paths=paths,
            repaired_slogan=extract_repaired_slogan_text(repair_response),
        )
        reverted_paths = preserve_semantic_bridge_basis(base_candidate, merged)
        reverted_paths.extend(revert_forbidden_paths(base_candidate, merged))
        closure = merged.get("advertisingClosure") if isinstance(merged.get("advertisingClosure"), dict) else {}
        product_label = _clean(closure.get("productNameText") or product_name)
        slogan = _clean(closure.get("sloganText"))
        if not slogan:
            raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:advertisingClosure.sloganText")
        validate_slogan_text_structure(slogan=slogan, product_name=product_label)
        meta = {
            "appliedPaths": applied_paths,
            "revertedPaths": reverted_paths,
            "changedPaths": sorted(diff_changed_paths(base_candidate, merged)),
            "actualWordCount": count_slogan_words_excluding_product(slogan, product_label),
        }
        return merged, meta
    raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:minimal_patch_required")


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
    repaired_slogan: str = "",
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
    if repaired_slogan:
        basis_ok, basis_field = validate_semantic_basis_with_repaired_slogan(
            candidate,
            repaired_slogan,
            product_name=product_name,
        )
        if not basis_ok:
            return False, [f"builder2_creator_validation_failed:{basis_field}"]
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
    call_type: str = "repair",
    source_role: str = "repair_response",
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
        "callType": _clean(call_type) or "repair",
        "sourceRole": _clean(source_role) or "repair_response",
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


def _payload_candidate_id(payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    return _clean(payload.get("candidateId")) or _clean((payload.get("parsed") or {}).get("candidateId"))


def _find_repair_source_only(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    preferred_candidate_id: str = "",
    product_name: str = "",
) -> Optional[Dict[str, Any]]:
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY
    from engine.builder2_slogan_repair_provenance import _payload_is_repair_source

    if preferred_candidate_id:
        loaded = load_slogan_repair_parsed_response(state, preferred_candidate_id)
        if loaded:
            return loaded
    repair_index = state.get(SLOGAN_REPAIR_PARSED_INDEX_KEY)
    repair_candidates: List[Dict[str, Any]] = []
    if isinstance(repair_index, dict):
        for payload in repair_index.values():
            if isinstance(payload, dict) and _clean(payload.get("prototypeId")) == prototype_id:
                repair_candidates.append(deepcopy(payload))
    rejected_index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if isinstance(rejected_index, dict):
        for payload in rejected_index.values():
            if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
                continue
            if _payload_is_repair_source(payload, product_name=product_name):
                repair_candidates.append(deepcopy(payload))
    repair_candidates.sort(key=lambda item: (_clean(item.get("storedAt")), _payload_candidate_id(item)), reverse=True)
    return repair_candidates[0] if repair_candidates else None


def find_slogan_repair_patch_source(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    preferred_candidate_id: str = "",
) -> Optional[Dict[str, Any]]:
    return _find_repair_source_only(
        state,
        prototype_id,
        preferred_candidate_id=preferred_candidate_id,
    )


def find_original_slogan_word_limit_rejection(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    preferred_candidate_id: str = "",
) -> Optional[Dict[str, Any]]:
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY
    from engine.builder2_slogan_repair_provenance import CALL_TYPE_NORMAL, _payload_is_original_base, infer_rejected_call_type

    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        return None
    matches: List[Dict[str, Any]] = []
    for payload in index.values():
        if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
            continue
        if preferred_candidate_id and _payload_candidate_id(payload) != preferred_candidate_id:
            continue
        if _payload_is_original_base(payload):
            matches.append(deepcopy(payload))
        elif is_slogan_word_limit_failure(_clean(payload.get("failureReason"))) and infer_rejected_call_type(payload) == CALL_TYPE_NORMAL:
            matches.append(deepcopy(payload))
    if preferred_candidate_id:
        for payload in matches:
            if _payload_candidate_id(payload) == preferred_candidate_id:
                return payload
    matches.sort(key=lambda item: _clean(item.get("storedAt")))
    return matches[0] if matches else None


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


def find_rejected_creator_for_prototype(state: Dict[str, Any], prototype_id: str) -> Optional[Dict[str, Any]]:
    from engine.builder2_complete_ad_creator_recovery import find_rejected_creator_for_prototype as _find

    return _find(state, prototype_id)


def reconcile_slogan_repair_call_ledger(state: Dict[str, Any], *, prototype_id: str) -> Dict[str, Any]:
    bucket = _ledger_bucket(state, prototype_id)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    legacy_normal = int(metrics.get("creatorCalls") or 0)
    legacy_repair = int(metrics.get("creatorRepairCalls") or 0)
    ledger_normal = int(bucket.get("persistedCreatorNormalCalls") or 0)
    ledger_repair = int(bucket.get("persistedCreatorRepairCalls") or 0)

    canonical_normal = 1 if find_original_slogan_word_limit_rejection(state, prototype_id) else ledger_normal
    if canonical_normal <= 0 and legacy_normal > 0 and find_original_slogan_word_limit_rejection(state, prototype_id):
        canonical_normal = 1

    has_patch_source = bool(find_slogan_repair_patch_source(state, prototype_id))
    canonical_repair = legacy_repair
    if has_patch_source and legacy_repair > 0:
        canonical_repair = min(legacy_repair, 1)
    duplicate_suppressed = max(0, ledger_repair - canonical_repair) if ledger_repair > canonical_repair else 0
    if ledger_repair != canonical_repair or ledger_normal != canonical_normal:
        logger.info(
            "BUILDER2_REASONING_CALL_LEDGER_RECONCILED role=builder2_creator prototypeId=%s legacyCount=%s "
            "ledgerCount=%s canonicalUniqueCount=%s duplicateSuppressed=%s callKind=creatorRepair",
            prototype_id,
            legacy_repair,
            ledger_repair,
            canonical_repair,
            duplicate_suppressed,
        )

    bucket["persistedCreatorNormalCalls"] = canonical_normal
    bucket["persistedCreatorRepairCalls"] = canonical_repair
    bucket["canonicalCreatorNormalCalls"] = canonical_normal
    bucket["canonicalCreatorRepairCalls"] = canonical_repair
    bucket.setdefault("persistedJudgeCalls", int(metrics.get("judgeCalls") or 0))
    bucket.setdefault("persistedWinnerCalls", int(metrics.get("winnerDevelopmentCalls") or 0))
    bucket.setdefault(
        "persistedRunwaySubmissionCalls",
        int((state.get("mediaResume") or {}).get("runwaySubmissionCalls") or 0),
    )
    return bucket


def sync_slogan_repair_call_ledger_from_metrics(state: Dict[str, Any], *, prototype_id: str) -> Dict[str, Any]:
    return reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)


def populate_slogan_repair_call_report(
    state: Dict[str, Any],
    report: Dict[str, Any],
    *,
    prototype_id: str,
    invocation_creator_normal_calls: int = 0,
    invocation_creator_repair_calls: int = 0,
) -> None:
    bucket = reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)
    persisted_normal = int(bucket.get("canonicalCreatorNormalCalls") or bucket.get("persistedCreatorNormalCalls") or 0)
    persisted_repair = int(bucket.get("canonicalCreatorRepairCalls") or bucket.get("persistedCreatorRepairCalls") or 0)
    report["invocationCreatorNormalCalls"] = int(invocation_creator_normal_calls)
    report["invocationCreatorRepairCalls"] = int(invocation_creator_repair_calls)
    report["persistedCreatorNormalCalls"] = persisted_normal
    report["persistedCreatorRepairCalls"] = persisted_repair
    report["totalCreatorNormalCalls"] = persisted_normal + int(invocation_creator_normal_calls)
    report["totalCreatorRepairCalls"] = persisted_repair + int(invocation_creator_repair_calls)
    report["additionalPaidRepairAllowed"] = additional_paid_slogan_repair_allowed(state, prototype_id)
    report["offlineSalvageAttempted"] = bool(bucket.get("offlineSalvageAttempted"))
    report["offlineSalvageAccepted"] = bool(bucket.get("offlineSalvageAccepted"))


def additional_paid_slogan_repair_allowed(state: Dict[str, Any], prototype_id: str) -> bool:
    bucket = reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)
    repair_calls = int(bucket.get("canonicalCreatorRepairCalls") or bucket.get("persistedCreatorRepairCalls") or 0)
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
    try:
        candidate, meta = select_minimal_validating_patch(
            base_candidate,
            repair_response,
            assigned_prototype_id=assigned_prototype_id,
            prototype_display_name=prototype_display_name,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
            product_name=product_name,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            tournament_state=tournament_state,
        )
    except Builder2TournamentError:
        raise
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
    try:
        original_payload, patch_payload = resolve_slogan_repair_base_and_source(
            state,
            prototype_id,
            original_candidate_id=original_candidate_id,
            repair_candidate_id=patch_candidate_id,
            product_name=product_name or _clean((state.get("strategyFoundation") or {}).get("productNameResolved")),
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_slogan_repair_offline_merge_invalid")
        paths = [reason.split(":")[-1]]
        return False, None, reason, paths
    failing_paths: List[str] = []
    base_parsed = deepcopy(original_payload.get("parsed") or {})
    patch_parsed = deepcopy(patch_payload.get("parsed") or {})
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    repaired_slogan = extract_repaired_slogan_text(patch_parsed)
    only_word_limit, structural_errors = candidate_fails_only_slogan_word_limit(
        base_parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        product_name=product_name,
        repaired_slogan=repaired_slogan,
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
    reconcile_slogan_repair_call_ledger(state, prototype_id=prototype_id)
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_OFFLINE_SALVAGE_ACCEPTED jobId=%s prototypeId=%s candidateId=%s",
        _clean(state.get("jobId")) or "(none)",
        prototype_id,
        resolved_candidate_id,
    )
    return True, resolved_candidate_id, None, []
