"""
Builder2 Judge repair response inspector — read-only audit of repair lifecycle.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_judge_pending_repair import (
    REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE,
    resolve_judge_repair_resume_context,
    resolve_unresolved_judge_repair,
)
from engine.builder2_judge_repair_offline_salvage import assess_repair_attempt_salvage
from engine.builder2_judge_response_ledger import (
    find_latest_attempt,
    ledger_entries,
    normal_attempts,
    repair_attempts,
)
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_strategy_evidence_grounding_contract import JUDGE_FACTUAL_GROUNDING_GATE_FIELDS
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _gate_details(assessment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "values": {key: assessment.get(key) for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS},
        "types": {key: type(assessment.get(key)).__name__ for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS},
        "notes": _clean(assessment.get("notes")),
    }


def _search_locations(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    locations: List[Dict[str, Any]] = []
    for entry in ledger_entries(state, candidate_id):
        locations.append(
            {
                "location": "judgeResponseLedgerByCandidate",
                "callType": _clean(entry.get("callType")),
                "attemptId": _clean(entry.get("attemptId")),
                "judgmentId": _clean(entry.get("judgmentId")),
                "parsedResponseAvailable": bool(entry.get("parsedResponseAvailable")),
            }
        )
    pending = _candidate_record(state, candidate_id).get("pendingJudgeRepair")
    if isinstance(pending, dict) and pending:
        locations.append({"location": "pendingJudgeRepair", "lifecycleStage": pending.get("lifecycleStage")})
    diagnostics = (state.get("judgeDiagnosticsByCandidate") or {}).get(candidate_id) or _candidate_record(state, candidate_id).get("judgeDiagnostics") or {}
    if diagnostics:
        locations.append({"location": "judgeDiagnosticsByCandidate", "repairAttempted": diagnostics.get("repairAttempted")})
    metrics = state.get("metrics") or {}
    if int(metrics.get("judgeRepairCalls") or 0) > 0:
        locations.append({"location": "metrics", "judgeRepairCalls": metrics.get("judgeRepairCalls")})
    return locations


def inspect_judge_repair_response(
    state: Dict[str, Any],
    *,
    candidate_id: str = "",
    job_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = _clean(candidate_id) or _clean(os.environ.get("BUILDER2_JUDGE_REPAIR_RESPONSE_INSPECT_CANDIDATE_ID"))
    if not target:
        repairs = []
        for cid in sorted((state.get("candidates") or {}).keys()):
            if repair_attempts(state, str(cid)) or isinstance((_candidate_record(state, str(cid)).get("pendingJudgeRepair")), dict):
                repairs.append(str(cid))
        if len(repairs) == 1:
            target = repairs[0]
    report: Dict[str, Any] = {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "candidateId": target or None,
        "repairDispatchRecorded": False,
        "repairHTTPResponseRecorded": False,
        "rawRepairResponseAvailable": False,
        "parsedRepairResponseAvailable": False,
        "repairResponseLocation": None,
        "repairResponseFingerprint": None,
        "parsedRepairResponseFingerprint": None,
        "factualGroundingAssessment": {},
        "factualGroundingGateDetails": {},
        "structuralValidationUnderCurrentContract": None,
        "substantiveEligibilityUnderCurrentContract": None,
        "offlineSalvagePossible": False,
        "unavailableReason": None,
        "normalAttempt": None,
        "repairAttempt": None,
        "pendingJudgeRepair": None,
        "searchedLocations": [],
        "paidCalls": 0,
        "stateMutated": False,
    }
    if not target:
        report["unavailableReason"] = "builder2_judge_repair_response_inspect_candidate_missing"
        return report

    report["searchedLocations"] = _search_locations(state, target)
    pending = (_candidate_record(state, target).get("pendingJudgeRepair") if isinstance(_candidate_record(state, target).get("pendingJudgeRepair"), dict) else {})
    unresolved = resolve_unresolved_judge_repair(state, target)
    ctx = resolve_judge_repair_resume_context(state, target)
    report["pendingJudgeRepair"] = pending or unresolved or None
    report["repairResumeContext"] = ctx

    normal = find_latest_attempt(state, target, call_type="normal")
    repair = find_latest_attempt(state, target, call_type="repair")
    if normal:
        report["normalAttempt"] = {
            "judgmentId": normal.get("judgmentId"),
            "responseFingerprint": normal.get("responseFingerprint"),
            "parsedResponseFingerprint": normal.get("parsedResponseFingerprint"),
            "validationFailureField": normal.get("validationFailureField"),
        }
    if repair:
        report["repairAttempt"] = dict(repair)
        report["repairResponseLocation"] = "judgeResponseLedgerByCandidate"
        report["repairResponseFingerprint"] = repair.get("responseFingerprint")
        report["parsedRepairResponseFingerprint"] = repair.get("parsedResponseFingerprint")
        report["parsedRepairResponseAvailable"] = bool(repair.get("parsedResponseAvailable"))
        report["rawRepairResponseAvailable"] = bool(repair.get("rawResponseAvailable") or repair.get("responseAvailable"))
        report["repairHTTPResponseRecorded"] = bool(repair.get("responseReceived"))
        assessment = repair.get("parsedResponse", {}).get("factualGroundingAssessment") if isinstance(repair.get("parsedResponse"), dict) else {}
        if isinstance(assessment, dict):
            report["factualGroundingAssessment"] = assessment
            report["factualGroundingGateDetails"] = _gate_details(assessment)
        salvage = assess_repair_attempt_salvage(state, candidate_id=target, entry=repair)
        report["structuralValidationUnderCurrentContract"] = salvage.get("structuralValidationUnderCurrentContract")
        report["substantiveEligibilityUnderCurrentContract"] = salvage.get("substantiveEligibilityUnderCurrentContract")
        report["offlineSalvagePossible"] = bool(salvage.get("offlineSalvagePossible"))
        report["falseBooleanMisclassified"] = salvage.get("falseBooleanMisclassified")
    metrics = state.get("metrics") or {}
    report["repairDispatchRecorded"] = bool(pending.get("repairDispatched")) or int(metrics.get("judgeRepairCalls") or 0) > 0
    if report["repairDispatchRecorded"] and not repair:
        report["unavailableReason"] = "builder2_judge_repair_response_unavailable"
        report["repairOutcomeUnrecoverable"] = True
        if pending.get("lifecycleStage") != REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE:
            report["repairResponseRecoverableFromAlternateState"] = False
    elif repair and not report["parsedRepairResponseAvailable"]:
        report["unavailableReason"] = "builder2_judge_repair_parsed_response_missing"
    return report


def inspect_judge_repair_response_for_job(job_id: str, *, candidate_id: str = "") -> Dict[str, Any]:
    if not redis_configured():
        return {"ok": False, "failureReason": "builder2_judge_repair_response_inspect_redis_unconfigured", "jobId": job_id}
    with read_only_builder2_inspection() as counter:
        state = load_tournament_state(job_id)
        if not isinstance(state, dict) or not state:
            return {"ok": False, "failureReason": "builder2_judge_repair_response_inspect_job_not_found", "jobId": job_id}
        report = inspect_judge_repair_response(state, candidate_id=candidate_id, job_record=video_job_get(job_id))
        report["ok"] = True
        report["redisMutations"] = counter.redis_mutations
        report["paidCalls"] = 0
        report["stateMutated"] = False
        return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_REPAIR_RESPONSE_INSPECT_JOB_ID"))
    if not job_id:
        print("BUILDER2_JUDGE_REPAIR_RESPONSE_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    report = inspect_judge_repair_response_for_job(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
