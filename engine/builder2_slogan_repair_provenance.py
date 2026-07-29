"""
Builder2 slogan repair provenance — role resolution and semantic-basis fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_advertising_closure_contract import (
    SLOGAN_MAX_WORD_COUNT,
    count_slogan_words_excluding_product,
)
from engine.builder2_creator import is_slogan_word_limit_failure
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

SEMANTIC_BASIS_FINGERPRINT_KEYS: Tuple[str, ...] = (
    "keyWordOrConcept",
    "visualMeaning",
    "strategicMeaning",
    "understandableWithoutCreatorReport",
    "dualMeaningUsed",
    "physicalMeaningActivatedByVisual",
    "strategicMeaningActivatedBySlogan",
    "meaningsConverge",
)

CALL_TYPE_NORMAL = "normal"
CALL_TYPE_REPAIR = "repair"

SOURCE_ROLE_ORIGINAL = "original_rejection"
SOURCE_ROLE_REPAIR = "repair_response"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _payload_candidate_id(payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    return _clean(payload.get("candidateId")) or _clean((payload.get("parsed") or {}).get("candidateId"))


def _payload_failure_reason(payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    return _clean(payload.get("failureReason"))


def infer_rejected_call_type(payload: Dict[str, Any]) -> str:
    explicit = _clean(payload.get("callType"))
    if explicit in {CALL_TYPE_NORMAL, CALL_TYPE_REPAIR}:
        return explicit
    source = _clean(payload.get("source"))
    if source in {"creator_repair", "repair_response"}:
        return CALL_TYPE_REPAIR
    if is_slogan_word_limit_failure(_payload_failure_reason(payload)):
        return CALL_TYPE_NORMAL
    return CALL_TYPE_REPAIR


def infer_source_role(payload: Dict[str, Any]) -> str:
    explicit = _clean(payload.get("sourceRole"))
    if explicit in {SOURCE_ROLE_ORIGINAL, SOURCE_ROLE_REPAIR}:
        return explicit
    if infer_rejected_call_type(payload) == CALL_TYPE_REPAIR:
        return SOURCE_ROLE_REPAIR
    return SOURCE_ROLE_ORIGINAL


def semantic_basis_subset(candidate: Dict[str, Any]) -> Dict[str, Any]:
    bridge = candidate.get("semanticBridge") if isinstance(candidate.get("semanticBridge"), dict) else {}
    return {key: bridge.get(key) for key in SEMANTIC_BASIS_FINGERPRINT_KEYS if key in bridge}


def semantic_basis_fingerprint(candidate: Dict[str, Any]) -> str:
    payload = semantic_basis_subset(candidate)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_basis_meanings_converge(candidate: Dict[str, Any]) -> Optional[bool]:
    bridge = candidate.get("semanticBridge") if isinstance(candidate.get("semanticBridge"), dict) else {}
    value = bridge.get("meaningsConverge")
    return value if isinstance(value, bool) else None


def log_semantic_basis_fingerprint(
    *,
    candidate_id: str,
    source_role: str,
    candidate: Dict[str, Any],
    log_event: str,
) -> str:
    fingerprint = semantic_basis_fingerprint(candidate)
    converge = semantic_basis_meanings_converge(candidate)
    logger.info(
        "%s candidateId=%s sourceRole=%s meaningsConverge=%s fingerprintPrefix=%s objectIdentity=%s",
        log_event,
        candidate_id or "(none)",
        source_role,
        str(converge).lower() if converge is not None else "(none)",
        fingerprint[:16],
        hex(id(candidate)),
    )
    return fingerprint


def assert_semantic_basis_unchanged(
    *,
    before: Dict[str, Any],
    after: Dict[str, Any],
    failure_code: str,
) -> None:
    if semantic_basis_fingerprint(before) == semantic_basis_fingerprint(after):
        return
    raise Builder2TournamentError(failure_code)


def validate_semantic_basis_with_repaired_slogan(
    base_candidate: Dict[str, Any],
    repaired_slogan: str,
    *,
    product_name: str = "",
) -> Tuple[bool, Optional[str]]:
    from engine.builder2_creator_slogan_repair_patch import preserve_semantic_bridge_basis

    merged = deepcopy(base_candidate)
    closure = merged.get("advertisingClosure") if isinstance(merged.get("advertisingClosure"), dict) else {}
    product_label = _clean(closure.get("productNameText") or product_name)
    if not isinstance(closure, dict):
        closure = {}
        merged["advertisingClosure"] = closure
    closure["sloganText"] = repaired_slogan
    preserve_semantic_bridge_basis(base_candidate, merged)
    bridge = merged.get("semanticBridge") if isinstance(merged.get("semanticBridge"), dict) else {}
    if bridge.get("dualMeaningUsed") is True:
        for key in ("physicalMeaningActivatedByVisual", "strategicMeaningActivatedBySlogan", "meaningsConverge"):
            if bridge.get(key) is not True:
                return False, f"semanticBridge.{key}"
    word_count = count_slogan_words_excluding_product(repaired_slogan, product_label)
    if word_count > SLOGAN_MAX_WORD_COUNT:
        return False, "advertisingClosure.sloganText.word_limit"
    return True, None


def _stored_at(payload: Dict[str, Any]) -> str:
    return _clean(payload.get("storedAt"))


def _reject_index(state: Dict[str, Any]) -> Dict[str, Any]:
    from engine.builder2_complete_ad_creator_recovery import REJECTED_CREATOR_PARSED_INDEX_KEY

    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    return index if isinstance(index, dict) else {}


def _repair_index(state: Dict[str, Any]) -> Dict[str, Any]:
    from engine.builder2_creator_slogan_repair_patch import SLOGAN_REPAIR_PARSED_INDEX_KEY

    index = state.get(SLOGAN_REPAIR_PARSED_INDEX_KEY)
    return index if isinstance(index, dict) else {}


def _payload_is_original_base(payload: Dict[str, Any]) -> bool:
    if infer_rejected_call_type(payload) != CALL_TYPE_NORMAL:
        return False
    if not is_slogan_word_limit_failure(_payload_failure_reason(payload)):
        return False
    if infer_source_role(payload) == SOURCE_ROLE_REPAIR:
        return False
    return True


def _payload_is_repair_source(payload: Dict[str, Any], *, repaired_slogan: str = "", product_name: str = "") -> bool:
    from engine.builder2_creator_slogan_repair_patch import extract_repaired_slogan_text

    if infer_rejected_call_type(payload) == CALL_TYPE_REPAIR:
        pass
    elif infer_source_role(payload) == SOURCE_ROLE_REPAIR:
        pass
    elif not is_slogan_word_limit_failure(_payload_failure_reason(payload)):
        pass
    else:
        return False
    parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
    slogan = repaired_slogan or extract_repaired_slogan_text(parsed)
    if not slogan:
        return False
    closure = parsed.get("advertisingClosure") if isinstance(parsed.get("advertisingClosure"), dict) else {}
    product_label = _clean(closure.get("productNameText") or product_name)
    return count_slogan_words_excluding_product(slogan, product_label) <= SLOGAN_MAX_WORD_COUNT


def resolve_slogan_repair_base_and_source(
    state: Dict[str, Any],
    prototype_id: str,
    *,
    original_candidate_id: str = "",
    repair_candidate_id: str = "",
    product_name: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rejected_index = _reject_index(state)
    repair_index = _repair_index(state)

    original_candidates: List[Dict[str, Any]] = []
    repair_candidates: List[Dict[str, Any]] = []

    if original_candidate_id and original_candidate_id in rejected_index:
        payload = rejected_index.get(original_candidate_id)
        if isinstance(payload, dict) and _payload_is_original_base(payload):
            original_candidates.append(deepcopy(payload))

    if repair_candidate_id:
        if repair_candidate_id in repair_index:
            payload = repair_index.get(repair_candidate_id)
            if isinstance(payload, dict):
                repair_candidates.append(deepcopy(payload))
        if repair_candidate_id in rejected_index:
            payload = rejected_index.get(repair_candidate_id)
            if isinstance(payload, dict) and _payload_is_repair_source(payload, product_name=product_name):
                repair_candidates.append(deepcopy(payload))

    for payload in rejected_index.values():
        if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
            continue
        if _payload_is_original_base(payload):
            original_candidates.append(deepcopy(payload))
        elif _payload_is_repair_source(payload, product_name=product_name):
            repair_candidates.append(deepcopy(payload))

    for payload in repair_index.values():
        if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
            continue
        repair_candidates.append(deepcopy(payload))

    original_candidates.sort(key=lambda item: (_stored_at(item), _payload_candidate_id(item)))
    repair_candidates.sort(key=lambda item: (_stored_at(item), _payload_candidate_id(item)), reverse=True)

    if original_candidate_id:
        preferred_original = [item for item in original_candidates if _payload_candidate_id(item) == original_candidate_id]
        if preferred_original:
            original_candidates = preferred_original
    if repair_candidate_id:
        preferred_repair = [item for item in repair_candidates if _payload_candidate_id(item) == repair_candidate_id]
        if preferred_repair:
            repair_candidates = preferred_repair

    original_payload = original_candidates[0] if original_candidates else None

    if original_payload is None:
        raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:original_missing")

    base_id = _payload_candidate_id(original_payload)
    for payload in repair_index.values():
        if not isinstance(payload, dict) or _clean(payload.get("prototypeId")) != prototype_id:
            continue
        if base_id and _payload_candidate_id(payload) == base_id:
            raise Builder2TournamentError("builder2_slogan_repair_base_source_collision")

    repair_candidates = [
        item for item in repair_candidates if _payload_candidate_id(item) != base_id
    ]
    if repair_candidate_id:
        preferred_repair = [item for item in repair_candidates if _payload_candidate_id(item) == repair_candidate_id]
        repair_payload = preferred_repair[0] if preferred_repair else (repair_candidates[0] if repair_candidates else None)
    else:
        repair_payload = repair_candidates[0] if repair_candidates else None

    if repair_payload is None:
        raise Builder2TournamentError("builder2_slogan_repair_offline_merge_invalid:patch_missing")

    repair_id = _payload_candidate_id(repair_payload)
    if base_id and repair_id and base_id == repair_id:
        raise Builder2TournamentError("builder2_slogan_repair_base_source_collision")

    logger.info(
        "BUILDER2_SLOGAN_REPAIR_BASE_SELECTED candidateId=%s prototypeId=%s callType=%s failureCode=%s",
        base_id or "(none)",
        prototype_id,
        infer_rejected_call_type(original_payload),
        _payload_failure_reason(original_payload).split(":")[-1] or "(none)",
    )
    logger.info(
        "BUILDER2_SLOGAN_REPAIR_SOURCE_SELECTED candidateId=%s prototypeId=%s callType=%s failureCode=%s",
        repair_id or "(none)",
        prototype_id,
        infer_rejected_call_type(repair_payload),
        _payload_failure_reason(repair_payload).split(":")[-1] or "(none)",
    )

    base_parsed = original_payload.get("parsed") if isinstance(original_payload.get("parsed"), dict) else {}
    log_semantic_basis_fingerprint(
        candidate_id=base_id,
        source_role=SOURCE_ROLE_ORIGINAL,
        candidate=base_parsed,
        log_event="BUILDER2_SLOGAN_REPAIR_BASE_FINGERPRINT",
    )
    repair_parsed = repair_payload.get("parsed") if isinstance(repair_payload.get("parsed"), dict) else {}
    log_semantic_basis_fingerprint(
        candidate_id=repair_id,
        source_role=SOURCE_ROLE_REPAIR,
        candidate=repair_parsed,
        log_event="BUILDER2_SLOGAN_REPAIR_SOURCE_FINGERPRINT",
    )

    return original_payload, repair_payload


def naive_latest_rejected_for_prototype(state: Dict[str, Any], prototype_id: str) -> Optional[Dict[str, Any]]:
    """Legacy naive lookup — first rejected payload for prototype in index iteration order."""
    index = _reject_index(state)
    for payload in index.values():
        if isinstance(payload, dict) and _clean(payload.get("prototypeId")) == prototype_id:
            return deepcopy(payload)
    return None
