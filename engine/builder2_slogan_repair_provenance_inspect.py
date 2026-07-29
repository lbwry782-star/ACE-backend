"""
Builder2 slogan repair provenance inspector — read-only, zero side effects.

Run:
  BUILDER2_SLOGAN_REPAIR_PROVENANCE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_slogan_repair_provenance_inspect
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

from engine.builder2_advertising_closure_contract import count_slogan_words_excluding_product
from engine.builder2_creator import collect_creator_structural_errors, validate_creator_candidate
from engine.builder2_creator_semantic_bridge_repair_patch import (
    apply_persisted_slogan_to_base,
    inspect_semantic_bridge_repair_lifecycle,
    semantic_bridge_repair_required,
    structural_failure_field_paths,
)
from engine.builder2_creator_normalization import normalize_creator_candidate
from engine.builder2_creator_slogan_repair_patch import (
    _apply_selected_patch_paths,
    candidate_fails_only_slogan_word_limit,
    extract_repaired_slogan_text,
    preserve_semantic_bridge_basis,
    prototype_display_name_for_id,
    revert_forbidden_paths,
)
from engine.builder2_slogan_repair_provenance import (
    infer_rejected_call_type,
    resolve_slogan_repair_base_and_source,
    semantic_basis_fingerprint,
    semantic_basis_meanings_converge,
    semantic_basis_meanings_converge_normalized,
    semantic_basis_meanings_converge_presence,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _payload_candidate_id(payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    return _clean(payload.get("candidateId")) or _clean((payload.get("parsed") or {}).get("candidateId"))


def inspect_slogan_repair_provenance(
    state: Dict[str, Any],
    *,
    prototype_id: str = "think_small",
) -> Dict[str, Any]:
    job_id = _clean(state.get("jobId"))
    tournament_id = _clean(state.get("tournamentId"))
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    product_name = _clean(strategy.get("productNameResolved"))
    display_name = prototype_display_name_for_id(prototype_id)

    report: Dict[str, Any] = {
        "jobId": job_id,
        "tournamentId": tournament_id,
        "prototypeId": prototype_id,
        "originalCandidateFound": False,
        "originalCandidateId": "",
        "originalFailureCode": "",
        "originalCallType": "",
        "originalMeaningsConverge": None,
        "originalMeaningsConvergePresence": "",
        "originalMeaningsConvergeRaw": None,
        "originalMeaningsConvergeNormalized": None,
        "originalSemanticBasisFingerprint": "",
        "repairSourceFound": False,
        "repairSourceCandidateId": "",
        "repairSourceCallType": "",
        "repairSourceMeaningsConverge": None,
        "repairSourceMeaningsConvergePresence": "",
        "repairSourceMeaningsConvergeRaw": None,
        "repairSourceSemanticBasisFingerprint": "",
        "baseSourceCollision": False,
        "sloganWordCountBefore": 0,
        "repairedSloganWordCount": 0,
        "completeStructuralFailurePaths": [],
        "semanticBridgeRepairRequired": False,
        "sloganOnlyMergeAttempted": False,
        "mergedPreNormalizeMeaningsConverge": None,
        "mergedPreNormalizeFingerprint": "",
        "mergedPostNormalizeMeaningsConverge": None,
        "mergedPostNormalizeFingerprint": "",
        "validationInputMeaningsConverge": None,
        "validationInputFingerprint": "",
        "validationPassed": False,
        "failureField": "",
        "stateMutated": False,
        "paidCalls": 0,
    }

    try:
        original_payload, repair_payload = resolve_slogan_repair_base_and_source(
            state,
            prototype_id,
            product_name=product_name,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "")
        if reason == "builder2_slogan_repair_base_source_collision":
            report["baseSourceCollision"] = True
        report["failureField"] = reason.split(":")[-1] if reason else ""
        return report

    original_parsed = deepcopy(original_payload.get("parsed") or {})
    repair_parsed = deepcopy(repair_payload.get("parsed") or {})
    original_id = _payload_candidate_id(original_payload)
    repair_id = _payload_candidate_id(repair_payload)

    report["originalCandidateFound"] = True
    report["originalCandidateId"] = original_id
    report["originalFailureCode"] = _clean(original_payload.get("failureReason")).split(":")[-1]
    report["originalCallType"] = infer_rejected_call_type(original_payload)
    report["originalMeaningsConverge"] = semantic_basis_meanings_converge(original_parsed)
    report["originalMeaningsConvergePresence"] = semantic_basis_meanings_converge_presence(original_parsed)
    report["originalMeaningsConvergeRaw"] = semantic_basis_meanings_converge(original_parsed)
    report["originalMeaningsConvergeNormalized"] = semantic_basis_meanings_converge_normalized(
        original_parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        job_id=job_id,
        candidate_id=original_id,
    )
    report["originalSemanticBasisFingerprint"] = semantic_basis_fingerprint(original_parsed)

    report["repairSourceFound"] = True
    report["repairSourceCandidateId"] = repair_id
    report["repairSourceCallType"] = infer_rejected_call_type(repair_payload)
    report["repairSourceMeaningsConverge"] = semantic_basis_meanings_converge(repair_parsed)
    report["repairSourceMeaningsConvergePresence"] = semantic_basis_meanings_converge_presence(repair_parsed)
    report["repairSourceMeaningsConvergeRaw"] = semantic_basis_meanings_converge(repair_parsed)
    report["repairSourceSemanticBasisFingerprint"] = semantic_basis_fingerprint(repair_parsed)
    report["baseSourceCollision"] = bool(original_id and repair_id and original_id == repair_id)

    closure = original_parsed.get("advertisingClosure") if isinstance(original_parsed.get("advertisingClosure"), dict) else {}
    product_label = _clean(closure.get("productNameText") or product_name)
    report["sloganWordCountBefore"] = count_slogan_words_excluding_product(
        _clean(closure.get("sloganText")),
        product_label,
    )
    repaired_slogan = extract_repaired_slogan_text(repair_parsed)
    report["repairedSloganWordCount"] = count_slogan_words_excluding_product(repaired_slogan, product_label)

    try:
        slogan_applied_base, _ = apply_persisted_slogan_to_base(original_parsed, repair_parsed)
    except Builder2TournamentError as exc:
        report["failureField"] = str(exc.args[0] if exc.args else "").split(":")[-1]
        return report

    complete_errors = collect_creator_structural_errors(
        slogan_applied_base,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        job_id=job_id,
        candidate_id=repair_id or original_id,
        prototype_id=prototype_id,
    )
    report["completeStructuralFailurePaths"] = structural_failure_field_paths(complete_errors)
    required, _paths = semantic_bridge_repair_required(
        slogan_applied_base,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        product_name=product_name,
    )
    report["semanticBridgeRepairRequired"] = required

    only_word_limit, _errors = candidate_fails_only_slogan_word_limit(
        original_parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=display_name,
        strategy_foundation=strategy,
        product_name=product_name,
        repaired_slogan=repaired_slogan,
    )
    if not only_word_limit:
        report["failureField"] = "original_not_word_limit_only"
        return report

    report["sloganOnlyMergeAttempted"] = True
    merged, _applied = _apply_selected_patch_paths(
        original_parsed,
        repair_parsed,
        paths=["advertisingClosure.sloganText"],
        repaired_slogan=repaired_slogan,
    )
    preserve_semantic_bridge_basis(original_parsed, merged)
    revert_forbidden_paths(original_parsed, merged)
    report["mergedPreNormalizeMeaningsConverge"] = semantic_basis_meanings_converge(merged)
    report["mergedPreNormalizeFingerprint"] = semantic_basis_fingerprint(merged)

    try:
        from engine.builder2_creator import normalize_creator_raw

        normalized, _resolved = normalize_creator_candidate(
            deepcopy(merged),
            assigned_prototype_id=prototype_id,
            prototype_display_name=display_name,
            strategy_foundation=strategy,
            compatibility_mode=False,
            base_normalizer=normalize_creator_raw,
            job_id=job_id,
            candidate_id=repair_id or original_id,
        )
    except Builder2TournamentError as exc:
        report["failureField"] = str(exc.args[0] if exc.args else "").split(":")[-1]
        return report

    report["mergedPostNormalizeMeaningsConverge"] = semantic_basis_meanings_converge(normalized)
    report["mergedPostNormalizeFingerprint"] = semantic_basis_fingerprint(normalized)
    validation_input = deepcopy(normalized)
    report["validationInputMeaningsConverge"] = semantic_basis_meanings_converge(validation_input)
    report["validationInputFingerprint"] = semantic_basis_fingerprint(validation_input)

    try:
        validate_creator_candidate(
            validation_input,
            assigned_prototype_id=prototype_id,
            prototype_display_name=display_name,
            strategy_foundation=strategy,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=repair_id or original_id,
            tournament_state=state,
        )
        report["validationPassed"] = True
    except Builder2TournamentError as exc:
        report["failureField"] = str(exc.args[0] if exc.args else "").split(":")[-1]

    report["semanticBridgeRepairLifecycle"] = inspect_semantic_bridge_repair_lifecycle(
        state,
        prototype_id=prototype_id,
    )
    return report


def main() -> int:
    job_id = _env("BUILDER2_SLOGAN_REPAIR_PROVENANCE_INSPECT_JOB_ID")
    if not job_id:
        print("BUILDER2_SLOGAN_REPAIR_PROVENANCE_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    prototype_id = _env("BUILDER2_SLOGAN_REPAIR_PROVENANCE_INSPECT_PROTOTYPE_ID", "think_small")
    state = load_tournament_state(job_id)
    if not isinstance(state, dict) or not state:
        print(json.dumps({"jobId": job_id, "error": "tournament_state_missing"}))
        return 1
    report = inspect_slogan_repair_provenance(state, prototype_id=prototype_id)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
