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
from engine.builder2_judge_circuit_breaker import is_judge_contract_circuit_breaker_tripped
from engine.builder2_judge_core_contract import is_judge_factual_grounding_gate_field
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


def analyze_judge_response_attempt(
    *,
    state: Dict[str, Any],
    candidate_id: str,
    entry: Dict[str, Any],
    strategy_foundation: Dict[str, Any],
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    candidate = _creator_payload(state, candidate_id)
    parsed = entry.get("parsedResponse") if isinstance(entry.get("parsedResponse"), dict) else {}
    assessment = parsed.get("factualGroundingAssessment") if isinstance(parsed.get("factualGroundingAssessment"), dict) else {}
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
    false_boolean_misclassified = bool(
        validation_failure_field
        and is_judge_factual_grounding_gate_field(validation_failure_field)
        and isinstance(assessment.get(validation_failure_field.split(".")[-1]), bool)
    )
    repair_necessary = bool(
        validation_failure_field
        and _is_structural_repairable(
            "builder2_judge_validation_failed",
            validation_failure_field,
        )
        and not is_judge_factual_grounding_gate_field(validation_failure_field)
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

    stamp_creator_evidence_inheritance(candidate, strategy_foundation=strategy_foundation)
    breaker = state.get("judgeContractCircuitBreaker") or {}
    candidate_paths = (breaker.get("candidateFailurePaths") or {}).get(candidate_id) or []
    return {
        "candidateId": candidate_id,
        "prototypeId": _clean(_candidate_record(state, candidate_id).get("prototypeId")),
        "judgmentId": _clean(entry.get("judgmentId")),
        "callType": _clean(entry.get("callType")),
        "responseAvailable": entry.get("responseAvailable"),
        "parsedResponseAvailable": entry.get("parsedResponseAvailable"),
        "factualGroundingAssessment": assessment,
        "factualGroundingGateRationales": _gate_rationales(assessment) if assessment else {},
        "reportedEligible": reported_eligible,
        "deterministicEligible": deterministic_eligible,
        "structuralErrors": structural_errors or list(entry.get("structuralErrors") or []),
        "semanticNegativeAssessmentFields": _semantic_negative_fields(assessment) if assessment else [],
        "validationFailureField": validation_failure_field or None,
        "validationFailureReason": _clean(entry.get("validationFailureReason")) or None,
        "repairDispatched": _clean(entry.get("callType")) == "repair",
        "repairNecessaryUnderCorrectedContract": repair_necessary,
        "circuitBreakerCountedAsStructural": any(
            validation_failure_field.split(".")[-1] in str(path)
            for path in candidate_paths
            if validation_failure_field
        ),
        "shouldCountAsStructuralUnderCorrectedContract": bool(structural_errors),
        "candidateCreatorFactuallyGrounded": candidate.get("creatorFactuallyGrounded"),
        "candidateUnsupportedProductClaims": list(candidate.get("newProductClaimsIntroduced") or []),
        "candidateNewProductClaimsIntroduced": list(candidate.get("newProductClaimsIntroduced") or []),
        "responseStructurallyValidUnderCorrectedContract": structurally_valid,
        "judgmentWouldBeEligibleUnderCorrectedContract": deterministic_eligible,
        "offlineRevalidationPossible": offline_revalidation_possible,
        "offlinePersistencePossible": offline_persistence_possible,
        "falseBooleanMisclassifiedAsValidationFailure": false_boolean_misclassified,
        "falseBooleanMisclassifiedAsStructuralFailure": false_boolean_misclassified,
        "responseFingerprint": _clean(entry.get("responseFingerprint")),
        "parsedResponseFingerprint": _clean(entry.get("parsedResponseFingerprint")),
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
            for cid in (state.get("candidates") or {})
            if _candidate_record(state, cid).get("creatorAcceptanceStatus") == "accepted"
            or _candidate_record(state, cid).get("validationStatus") == "accepted"
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

    normal_count = sum(1 for item in attempts if item.get("callType") == "normal")
    repair_count = sum(1 for item in attempts if item.get("callType") == "repair")
    structurally_invalid = sum(1 for item in attempts if item.get("responseStructurallyValidUnderCorrectedContract") is False)
    valid_negative = sum(
        1
        for item in attempts
        if item.get("responseStructurallyValidUnderCorrectedContract")
        and item.get("judgmentWouldBeEligibleUnderCorrectedContract") is False
    )
    false_misclassified = sum(1 for item in attempts if item.get("falseBooleanMisclassifiedAsValidationFailure"))
    missing_judgments = [
        cid
        for cid in candidate_ids
        if not _clean(_candidate_record(state, cid).get("judgmentId"))
        and _candidate_record(state, cid).get("judgeStatus") != "accepted"
    ]
    recoverable = [item for item in attempts if item.get("offlinePersistencePossible")]
    paid_min = 0 if recoverable else len(missing_judgments)
    paid_max = len(missing_judgments)
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "strategyEvidenceGroundingContractVersion": _clean((strategy.get("strategyEvidenceGrounding") or {}).get("contractVersion"))
        or "legacy_unknown",
        "attempts": attempts,
        "attemptedJudgeCount": len(attempts),
        "normalJudgeResponseCount": normal_count,
        "repairJudgeResponseCount": repair_count,
        "structurallyInvalidResponseCount": structurally_invalid,
        "structurallyValidNegativeJudgmentCount": valid_negative,
        "falseBooleanMisclassifiedAsValidationFailure": false_misclassified,
        "falseBooleanMisclassifiedAsStructuralFailure": false_misclassified,
        "circuitBreakerTriggeredIncorrectly": bool(
            is_judge_contract_circuit_breaker_tripped(state) and false_misclassified
        ),
        "acceptedCreatorCount": accepted_creator_count(state),
        "acceptedJudgmentCount": accepted_judgment_count(state),
        "missingJudgmentCandidateIds": missing_judgments,
        "cheapestSafeResumeStage": "judge_offline_recovery" if recoverable else "judge",
        "additionalPaidCallsRequiredMinimum": paid_min,
        "additionalPaidCallsRequiredMaximum": paid_max,
        "judgeContractCircuitBreaker": state.get("judgeContractCircuitBreaker") or {},
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
