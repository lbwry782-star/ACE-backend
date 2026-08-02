"""
Builder2 Judge factual-grounding failure inspector — read-only production audit.

Run:
  BUILDER2_JUDGE_GROUNDING_FAILURE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_judge_grounding_failure_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_judge import (
    _is_structural_repairable,
    collect_judge_structural_errors,
    validate_judge_response,
)
from engine.builder2_judge_circuit_breaker import (
    current_breaker_evidence_count,
    is_current_judge_contract_circuit_breaker_tripped,
    legacy_breaker_evidence_excluded_count,
)
from engine.builder2_judge_pending_repair import (
    normal_judge_call_must_not_repeat,
    repair_dispatch_blocked_reason,
    resolve_pending_judge_repair,
)
from engine.builder2_judge_structural_repair_classifier import classify_judge_structural_repair
from engine.builder2_judge_core_contract import is_judge_factual_grounding_gate_field
from engine.builder2_judge_factual_grounding_output_schema import (
    actual_factual_grounding_field_names,
    factual_grounding_object_empty,
    judge_schema_contract_metadata,
    schema_contract_mismatch_detected,
)
from engine.builder2_judge_response_ledger import resolve_parsed_response_fingerprint
from engine.builder2_strategy_evidence_grounding_contract import (
    JUDGE_FACTUAL_GROUNDING_GATE_FIELDS,
    apply_factual_grounding_eligibility_rules,
    collect_failed_factual_grounding_gates,
    collect_judge_factual_grounding_structural_errors,
    requires_strategy_evidence_grounding,
    stamp_creator_evidence_inheritance,
)
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _creator_payload(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    creator = record.get("creatorOutput") or record.get("creatorSnapshot") or {}
    return creator if isinstance(creator, dict) else {}


def _ledger_entries(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    ledger = state.get("judgeResponseLedgerByCandidate") or {}
    entries = ledger.get(candidate_id) or []
    return [item for item in entries if isinstance(item, dict)]


def _diagnostics_entry(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    by_candidate = state.get("judgeDiagnosticsByCandidate") or {}
    entry = by_candidate.get(candidate_id) or _candidate_record(state, candidate_id).get("judgeDiagnostics") or {}
    return entry if isinstance(entry, dict) else {}


def _schema_and_fingerprint_inspection(
    *,
    entry: Dict[str, Any],
    parsed: Dict[str, Any],
    strategy_foundation: Dict[str, Any],
    compatibility_mode: bool,
    pending_repair: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    factual_grounding_required = requires_strategy_evidence_grounding(
        strategy=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    fingerprint = resolve_parsed_response_fingerprint(entry)
    pending = pending_repair or {}
    source_fp = _clean(pending.get("sourceParsedResponseFingerprint")) or _clean(fingerprint.get("effective"))
    source_fp_complete = bool(source_fp)
    if not source_fp_complete and isinstance(parsed, dict) and parsed:
        source_fp = _clean(fingerprint.get("derived"))
        source_fp_complete = bool(source_fp)
    return {
        **judge_schema_contract_metadata(factual_grounding_required=factual_grounding_required),
        "actualFactualGroundingFieldNames": actual_factual_grounding_field_names(parsed),
        "factualGroundingObjectEmpty": factual_grounding_object_empty(parsed),
        "parsedFingerprintStored": bool(fingerprint.get("storedPresent")),
        "parsedFingerprintDerived": _clean(fingerprint.get("derived")) or None,
        "parsedFingerprintDerivationPossible": bool(fingerprint.get("derivationPossible")),
        "pendingRepairSourceFingerprintsComplete": bool(
            _clean(pending.get("sourceResponseFingerprint")) and source_fp_complete
        ),
        "pendingRepairBlockedByMissingFingerprint": bool(pending.get("repairRequired"))
        and not source_fp_complete,
        "schemaContractMismatchDetected": schema_contract_mismatch_detected(
            parsed,
            factual_grounding_required=factual_grounding_required,
        ),
    }


def _semantic_negative_fields(assessment: Dict[str, Any]) -> List[str]:
    return [key for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS if assessment.get(key) is False]


def _gate_rationales(assessment: Dict[str, Any]) -> Dict[str, Any]:
    notes = _clean(assessment.get("notes"))
    failed = _semantic_negative_fields(assessment)
    return {
        "notes": notes,
        "failedGates": failed,
        "perGateValues": {key: assessment.get(key) for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS},
        "perGateTypes": {key: type(assessment.get(key)).__name__ for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS},
    }


def _repair_dispatched_from_diagnostics(state: Dict[str, Any], candidate_id: str, entry: Dict[str, Any]) -> Optional[bool]:
    if _clean(entry.get("callType")) == "repair":
        return True
    if any(_clean(item.get("callType")) == "repair" for item in _ledger_entries(state, candidate_id)):
        return True
    diagnostics = _diagnostics_entry(state, candidate_id)
    if diagnostics.get("repairAttempted") is True:
        return True
    if diagnostics.get("repairAttempted") is False:
        return False
    return None


def analyze_judge_response_attempt(
    *,
    state: Dict[str, Any],
    candidate_id: str,
    entry: Dict[str, Any],
    strategy_foundation: Dict[str, Any],
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    candidate = _creator_payload(state, candidate_id)
    parsed_available = bool(entry.get("parsedResponseAvailable")) and isinstance(entry.get("parsedResponse"), dict) and bool(entry.get("parsedResponse"))
    response_available = entry.get("responseAvailable")
    legacy_not_persisted = bool(response_available) and not parsed_available
    parsed = entry.get("parsedResponse") if isinstance(entry.get("parsedResponse"), dict) else {}
    assessment = parsed.get("factualGroundingAssessment") if isinstance(parsed.get("factualGroundingAssessment"), dict) else {}
    validation_failure_field = _clean(entry.get("validationFailureField")) or None
    diagnostics = _diagnostics_entry(state, candidate_id)
    breaker = state.get("judgeContractCircuitBreaker") or {}
    repair_dispatched = _repair_dispatched_from_diagnostics(state, candidate_id, entry)

    if legacy_not_persisted or (not parsed_available and not parsed):
        candidate_paths = (breaker.get("candidateFailurePaths") or {}).get(candidate_id) or []
        return {
            "candidateId": candidate_id,
            "prototypeId": _clean(_candidate_record(state, candidate_id).get("prototypeId")),
            "judgmentId": _clean(entry.get("judgmentId")),
            "callType": _clean(entry.get("callType")) or "normal",
            "responseAvailable": response_available,
            "parsedResponseAvailable": False,
            "factualGroundingAssessment": {},
            "factualGroundingGateRationales": {},
            "reportedEligible": None,
            "deterministicEligible": None,
            "structuralErrors": [],
            "semanticNegativeAssessmentFields": [],
            "validationFailureField": validation_failure_field,
            "validationFailureReason": _clean(entry.get("validationFailureReason")) or _clean(diagnostics.get("failureReason")) or None,
            "repairDispatched": repair_dispatched,
            "repairNecessaryUnderCorrectedContract": None,
            "circuitBreakerCountedAsStructural": bool(candidate_paths),
            "shouldCountAsStructuralUnderCorrectedContract": None,
            "candidateCreatorFactuallyGrounded": candidate.get("creatorFactuallyGrounded"),
            "candidateUnsupportedProductClaims": list(candidate.get("newProductClaimsIntroduced") or []),
            "candidateNewProductClaimsIntroduced": list(candidate.get("newProductClaimsIntroduced") or []),
            "responseStructureAssessment": "unknown_parsed_response_not_persisted",
            "responseStructurallyValidUnderCorrectedContract": None,
            "judgmentWouldBeEligibleUnderCorrectedContract": None,
            "offlineRevalidationPossible": False,
            "offlinePersistencePossible": False,
            "falseBooleanMisclassifiedAsValidationFailure": None,
            "falseBooleanMisclassifiedAsStructuralFailure": None,
            "structuralValidationAttempted": False,
            "structuralValidationNotRunReason": "parsed_response_unavailable",
            "legacyResponseNotPersisted": True,
            "historicalCircuitBreakerTriggered": bool(breaker.get("tripped")),
            "historicalReason": _clean(breaker.get("trippedReason")) or None,
            "correctnessUnderCurrentContract": None,
            "responseFingerprint": _clean(entry.get("responseFingerprint")) or None,
            "parsedResponseFingerprint": None,
            **_schema_and_fingerprint_inspection(
                entry=entry,
                parsed={},
                strategy_foundation=strategy_foundation,
                compatibility_mode=compatibility_mode,
            ),
        }

    structural_errors = collect_judge_structural_errors(
        parsed,
        candidate_id=candidate_id,
        candidate=candidate,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
    )
    if not structural_errors and parsed:
        structural_errors = collect_judge_factual_grounding_structural_errors(
            parsed,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
        )
    validation_failure_field = _clean(entry.get("validationFailureField"))
    structurally_valid = not structural_errors
    false_boolean_misclassified: Optional[bool] = None
    if validation_failure_field and is_judge_factual_grounding_gate_field(validation_failure_field):
        leaf = validation_failure_field.split(".")[-1]
        if isinstance(assessment.get(leaf), bool):
            false_boolean_misclassified = True
    repair_necessary = bool(
        structural_errors
        and any(
            _is_structural_repairable(
                item.split(":", 1)[0] if ":" in item else "builder2_judge_validation_failed",
                item.split(":", 1)[1] if ":" in item else item,
                parsed=parsed,
            )
            for item in structural_errors
        )
    )
    reported_eligible = parsed.get("eligible")
    deterministic_eligible: Optional[bool] = None
    offline_revalidation_possible = False
    offline_persistence_possible = False
    if structurally_valid and parsed:
        try:
            product_input = None
            grounding = strategy_foundation.get("strategyEvidenceGrounding")
            if isinstance(grounding, dict):
                audit = grounding.get("productInputAudit")
                if isinstance(audit, dict):
                    product_input = audit
            trial = dict(parsed)
            judgment, _, _ = validate_judge_response(
                trial,
                candidate_id=candidate_id,
                candidate=candidate,
                strategy_foundation=strategy_foundation,
                product_input=product_input,
                compatibility_mode=compatibility_mode,
            )
            deterministic_eligible = bool(judgment.get("eligible"))
            offline_revalidation_possible = True
            offline_persistence_possible = not _clean(_candidate_record(state, candidate_id).get("judgmentId"))
        except Builder2TournamentError:
            offline_revalidation_possible = False
    elif parsed and assessment:
        trial = dict(parsed)
        apply_factual_grounding_eligibility_rules(trial)
        deterministic_eligible = bool(trial.get("eligible")) if isinstance(trial.get("eligible"), bool) else None

    breaker = state.get("judgeContractCircuitBreaker") or {}
    repair_dispatched = _repair_dispatched_from_diagnostics(state, candidate_id, entry)
    pending_repair = resolve_pending_judge_repair(state, candidate_id)
    classifier = classify_judge_structural_repair(
        "builder2_judge_validation_failed",
        validation_failure_field or (structural_errors[0].split(":", 1)[-1] if structural_errors else None),
        parsed=parsed,
    )
    blocked_reason = repair_dispatch_blocked_reason(state, candidate_id=candidate_id, pending=pending_repair)
    if repair_necessary and repair_dispatched is False and not blocked_reason:
        blocked_reason = "legacy_circuit_breaker_or_single_attempt_only"
    stamp_creator_evidence_inheritance(candidate, strategy_foundation=strategy_foundation)
    candidate_paths = (breaker.get("candidateFailurePaths") or {}).get(candidate_id) or []
    current_paths = (breaker.get("currentCandidateFailurePaths") or {}).get(candidate_id) or []
    return {
        "candidateId": candidate_id,
        "prototypeId": _clean(_candidate_record(state, candidate_id).get("prototypeId")),
        "judgmentId": _clean(entry.get("judgmentId")),
        "callType": _clean(entry.get("callType")),
        "responseAvailable": entry.get("responseAvailable"),
        "parsedResponseAvailable": True,
        "factualGroundingAssessment": assessment,
        "factualGroundingGateRationales": _gate_rationales(assessment) if assessment else {},
        "reportedEligible": reported_eligible,
        "deterministicEligible": deterministic_eligible,
        "structuralErrors": structural_errors or list(entry.get("structuralErrors") or []),
        "semanticNegativeAssessmentFields": _semantic_negative_fields(assessment) if assessment else [],
        "validationFailureField": validation_failure_field or None,
        "validationFailureReason": _clean(entry.get("validationFailureReason")) or None,
        "repairDispatched": repair_dispatched,
        "repairNecessaryUnderCorrectedContract": repair_necessary,
        "repairDispatchBlockedReason": blocked_reason,
        "structuralRepairClassifierDecision": classifier.get("decision"),
        "pendingRepairEligible": bool(pending_repair),
        "normalCallMustNotRepeat": normal_judge_call_must_not_repeat(state, candidate_id),
        "currentBreakerEvidenceCount": current_breaker_evidence_count(state),
        "legacyBreakerEvidenceExcludedCount": legacy_breaker_evidence_excluded_count(state),
        "currentContractBreakerTripped": is_current_judge_contract_circuit_breaker_tripped(state),
        "circuitBreakerCountedAsStructural": bool(current_paths) or any(
            validation_failure_field.split(".")[-1] in str(path)
            for path in candidate_paths
            if validation_failure_field
        ),
        "shouldCountAsStructuralUnderCorrectedContract": bool(structural_errors),
        "candidateCreatorFactuallyGrounded": candidate.get("creatorFactuallyGrounded"),
        "candidateUnsupportedProductClaims": list(candidate.get("newProductClaimsIntroduced") or []),
        "candidateNewProductClaimsIntroduced": list(candidate.get("newProductClaimsIntroduced") or []),
        "responseStructureAssessment": "evaluated",
        "responseStructurallyValidUnderCorrectedContract": structurally_valid,
        "judgmentWouldBeEligibleUnderCorrectedContract": deterministic_eligible,
        "offlineRevalidationPossible": offline_revalidation_possible,
        "offlinePersistencePossible": offline_persistence_possible,
        "falseBooleanMisclassifiedAsValidationFailure": false_boolean_misclassified,
        "falseBooleanMisclassifiedAsStructuralFailure": false_boolean_misclassified,
        "structuralValidationAttempted": True,
        "structuralValidationNotRunReason": None,
        "legacyResponseNotPersisted": False,
        "historicalCircuitBreakerTriggered": bool(breaker.get("tripped")),
        "historicalReason": _clean(breaker.get("trippedReason")) or None,
        "correctnessUnderCurrentContract": structurally_valid if structurally_valid is not None else None,
        "responseFingerprint": _clean(entry.get("responseFingerprint")) or None,
        "parsedResponseFingerprint": _clean(resolve_parsed_response_fingerprint(entry).get("effective")) or None,
        **_schema_and_fingerprint_inspection(
            entry=entry,
            parsed=parsed,
            strategy_foundation=strategy_foundation,
            compatibility_mode=compatibility_mode,
            pending_repair=pending_repair,
        ),
    }


def inspect_judge_grounding_failures(
    state: Dict[str, Any],
    *,
    job_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    compatibility_mode = not requires_strategy_evidence_grounding(strategy=strategy)
    attempts: List[Dict[str, Any]] = []
    candidate_ids = sorted(
        {
            cid
            for cid, record in (state.get("candidates") or {}).items()
            if isinstance(record, dict)
            and (
                record.get("creatorAcceptanceStatus") == "accepted"
                or record.get("validationStatus") == "accepted"
                or record.get("judgeStatus") in {"unavailable", "pending", "accepted"}
            )
        }
    )
    for candidate_id in candidate_ids:
        entries = _ledger_entries(state, candidate_id)
        if entries:
            for entry in entries:
                attempts.append(
                    analyze_judge_response_attempt(
                        state=state,
                        candidate_id=candidate_id,
                        entry=entry,
                        strategy_foundation=strategy,
                        compatibility_mode=compatibility_mode,
                    )
                )
            continue
        diagnostics = _diagnostics_entry(state, candidate_id)
        synthetic = {
            "judgmentId": _clean(_candidate_record(state, candidate_id).get("judgmentId")),
            "callType": "normal",
            "responseAvailable": diagnostics.get("responseReceived"),
            "parsedResponseAvailable": False,
            "parsedResponse": {},
            "validationFailureField": (diagnostics.get("failureFieldPaths") or [None])[0],
            "validationFailureReason": diagnostics.get("failureReason"),
            "structuralErrors": [],
            "responseFingerprint": "",
            "parsedResponseFingerprint": "",
        }
        attempts.append(
            analyze_judge_response_attempt(
                state=state,
                candidate_id=candidate_id,
                entry=synthetic,
                strategy_foundation=strategy,
                compatibility_mode=compatibility_mode,
            )
        )

    normal_count = sum(1 for item in attempts if _clean(item.get("callType") or "normal") == "normal")
    repair_count = sum(1 for item in attempts if _clean(item.get("callType")) == "repair")
    repair_dispatched_count = sum(
        1
        for cid in candidate_ids
        if bool((_candidate_record(state, cid).get("pendingJudgeRepair") or {}).get("repairDispatched"))
        or int((state.get("metrics") or {}).get("judgeRepairCalls") or 0) > 0
    )
    repair_response_received_count = sum(
        1 for cid in candidate_ids for entry in _ledger_entries(state, cid) if _clean(entry.get("callType")) == "repair" and entry.get("responseReceived")
    )
    repair_response_persisted_count = repair_count
    repair_parsed_persisted_count = sum(
        1
        for cid in candidate_ids
        for entry in _ledger_entries(state, cid)
        if _clean(entry.get("callType")) == "repair" and entry.get("parsedResponseAvailable")
    )
    repair_validation_failure_count = sum(
        1
        for cid in candidate_ids
        for entry in _ledger_entries(state, cid)
        if _clean(entry.get("callType")) == "repair" and _clean(entry.get("validationFailureReason"))
    )
    dispatched_without_persisted = [
        cid
        for cid in candidate_ids
        if bool((_candidate_record(state, cid).get("pendingJudgeRepair") or {}).get("repairDispatched"))
        and not any(_clean(item.get("callType")) == "repair" for item in _ledger_entries(state, cid))
    ]
    unresolved_repair_candidate_ids = dispatched_without_persisted
    structurally_invalid = sum(
        1 for item in attempts if item.get("responseStructurallyValidUnderCorrectedContract") is False
    )
    valid_negative = sum(
        1
        for item in attempts
        if item.get("responseStructurallyValidUnderCorrectedContract")
        and item.get("judgmentWouldBeEligibleUnderCorrectedContract") is False
    )
    false_misclassified = sum(
        1 for item in attempts if item.get("falseBooleanMisclassifiedAsValidationFailure") is True
    )
    legacy_unpersisted = sum(1 for item in attempts if item.get("legacyResponseNotPersisted"))
    breaker = state.get("judgeContractCircuitBreaker") or {}
    missing_judgments = [
        cid
        for cid in candidate_ids
        if not _clean(_candidate_record(state, cid).get("judgmentId"))
        and _candidate_record(state, cid).get("judgeStatus") != "accepted"
    ]
    recoverable = [item for item in attempts if item.get("offlinePersistencePossible")]
    paid_min = 0 if recoverable else len(missing_judgments)
    paid_max = len(missing_judgments)
    schema_meta = judge_schema_contract_metadata(factual_grounding_required=not compatibility_mode)
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "strategyEvidenceGroundingContractVersion": _clean((strategy.get("strategyEvidenceGrounding") or {}).get("contractVersion"))
        or "legacy_unknown",
        **schema_meta,
        "attempts": attempts,
        "attemptedJudgeCount": len(attempts),
        "normalJudgeResponseCount": normal_count,
        "repairJudgeResponseCount": repair_count,
        "repairDispatchedCount": repair_dispatched_count,
        "repairResponseReceivedCount": repair_response_received_count,
        "repairResponsePersistedCount": repair_response_persisted_count,
        "repairParsedResponsePersistedCount": repair_parsed_persisted_count,
        "repairValidationFailureCount": repair_validation_failure_count,
        "dispatchedRepairWithoutPersistedResponseCount": len(dispatched_without_persisted),
        "unresolvedRepairCandidateIds": unresolved_repair_candidate_ids,
        "repairResponseRecoverableFromAlternateState": bool(recoverable),
        "repairResponseLedgerComplete": repair_count >= repair_dispatched_count or not dispatched_without_persisted,
        "structurallyInvalidResponseCount": structurally_invalid,
        "structurallyValidNegativeJudgmentCount": valid_negative,
        "falseBooleanMisclassifiedAsValidationFailure": false_misclassified,
        "falseBooleanMisclassifiedAsStructuralFailure": false_misclassified,
        "circuitBreakerTriggeredIncorrectly": bool(
            breaker.get("tripped") and legacy_unpersisted and false_misclassified
        ),
        "historicalCircuitBreakerTriggered": bool(breaker.get("tripped")),
        "historicalCircuitBreakerReason": _clean(breaker.get("trippedReason")) or None,
        "currentContractBreakerTripped": is_current_judge_contract_circuit_breaker_tripped(state),
        "currentBreakerEvidenceCount": current_breaker_evidence_count(state),
        "legacyBreakerEvidenceExcludedCount": legacy_breaker_evidence_excluded_count(state),
        "legacyUnpersistedJudgeResponseCount": legacy_unpersisted,
        "acceptedCreatorCount": accepted_creator_count(state),
        "acceptedJudgmentCount": accepted_judgment_count(state),
        "missingJudgmentCandidateIds": missing_judgments,
        "cheapestSafeResumeStage": "judge_offline_recovery" if recoverable else "judge",
        "additionalPaidCallsRequiredMinimum": paid_min,
        "additionalPaidCallsRequiredMaximum": paid_max,
        "judgeContractCircuitBreaker": breaker,
        "paidCalls": 0,
        "openAICalls": 0,
        "stateMutated": False,
    }


def inspect_judge_grounding_failures_for_job(
    job_id: str,
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if tournament_state is None:
        if not redis_configured():
            return {"ok": False, "failureReason": "builder2_judge_grounding_failure_inspect_redis_unconfigured", "jobId": job_id}
        state = load_tournament_state(job_id)
        job_record = video_job_get(job_id)
    else:
        state = tournament_state
        job_record = None
    if not isinstance(state, dict) or not state:
        return {"ok": False, "failureReason": "builder2_judge_grounding_failure_inspect_job_not_found", "jobId": job_id}
    report = inspect_judge_grounding_failures(state, job_record=job_record if isinstance(job_record, dict) else None)
    report["ok"] = True
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_GROUNDING_FAILURE_INSPECT_JOB_ID"))
    if not job_id:
        print("BUILDER2_JUDGE_GROUNDING_FAILURE_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    logger.info("BUILDER2_JUDGE_GROUNDING_FAILURE_INSPECT_START jobId=%s", job_id)
    report = inspect_judge_grounding_failures_for_job(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
