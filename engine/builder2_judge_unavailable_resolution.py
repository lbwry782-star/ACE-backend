"""
Builder2 Judge unavailable resolution — operator-authorized Policy A (exclude and continue).

Run dry-run:
  BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_JOB_ID=<jobId> \\
  BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CANDIDATE_ID=<candidateId> \\
  python -m engine.builder2_judge_unavailable_resolution

Apply:
  BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_APPLY=true (same env vars)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_accepted_creator_store import update_candidate_judge_state
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_judge_repair_offline_salvage import assess_repair_attempt_salvage
from engine.builder2_judge_response_ledger import find_latest_attempt
from engine.builder2_judge_unavailable_resolution_contract import (
    BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CONTRACT_VERSION,
    OPERATOR_DECISION_EXCLUDE_AND_CONTINUE,
    OPERATOR_RESOLVED_JUDGMENT_UNAVAILABLE_STAGE,
    RESOLUTION_REASON_REPAIR_UNAVAILABLE,
    accepted_judgment_exists_for_candidate,
    build_judgment_unavailable_resolution_record,
    get_candidate_judgment_resolution,
    has_operator_judgment_unavailable_resolution,
    is_creator_accepted_for_candidate,
    is_reasoning_recovery_paused_state,
    media_started,
    persist_candidate_judgment_resolution,
    pending_repair_is_unrecoverable,
    winner_or_media_started,
)
from engine.builder2_tournament_recovery import acquire_job_lease, new_worker_token, release_job_lease
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _write_pending_operator_resolution(state: Dict[str, Any], *, candidate_id: str) -> None:
    record = _candidate_record(state, candidate_id)
    pending = record.get("pendingJudgeRepair")
    if not isinstance(pending, dict):
        pending = {}
    pending = dict(pending)
    pending["lifecycleStage"] = OPERATOR_RESOLVED_JUDGMENT_UNAVAILABLE_STAGE
    pending["repairOutcomeUnrecoverable"] = True
    pending["repairResponseAvailable"] = False
    pending["repairRequired"] = False
    pending["operatorResolutionApplied"] = True
    pending["operatorDecision"] = OPERATOR_DECISION_EXCLUDE_AND_CONTINUE
    record["pendingJudgeRepair"] = pending


def _expected_fingerprints(state: Dict[str, Any], candidate_id: str) -> tuple[str, str, str]:
    normal = find_latest_attempt(state, candidate_id, call_type="normal")
    pending = (_candidate_record(state, candidate_id).get("pendingJudgeRepair") or {}) if isinstance(
        _candidate_record(state, candidate_id).get("pendingJudgeRepair"), dict
    ) else {}
    source_judgment_id = _clean((normal or {}).get("judgmentId")) or _clean(pending.get("sourceJudgmentId"))
    source_response_fp = _clean((normal or {}).get("responseFingerprint")) or _clean(pending.get("sourceResponseFingerprint"))
    source_parsed_fp = _clean((normal or {}).get("parsedResponseFingerprint")) or _clean(
        pending.get("sourceParsedResponseFingerprint")
    )
    return source_judgment_id, source_response_fp, source_parsed_fp


def assess_judge_unavailable_resolution(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    expected_job_id: str = "",
    expected_tournament_id: str = "",
    expected_source_judgment_id: str = "",
    expected_source_response_fingerprint: str = "",
    expected_source_parsed_response_fingerprint: str = "",
) -> Dict[str, Any]:
    from engine.builder2_judge_repair_response_inspect import inspect_judge_repair_response

    record = _candidate_record(state, candidate_id)
    prototype_id = _clean(record.get("prototypeId"))
    repair_report = inspect_judge_repair_response(state, candidate_id=candidate_id)
    source_judgment_id, source_response_fp, source_parsed_fp = _expected_fingerprints(state, candidate_id)
    repair = find_latest_attempt(state, candidate_id, call_type="repair")
    salvage = assess_repair_attempt_salvage(state, candidate_id=candidate_id, entry=repair) if repair else {}
    metrics = state.get("metrics") or {}
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add_check("job_id_present", bool(_clean(state.get("jobId"))))
    if expected_job_id:
        add_check("job_id_matches", _clean(state.get("jobId")) == _clean(expected_job_id))
    if expected_tournament_id:
        add_check(
            "tournament_id_matches",
            _clean(state.get("tournamentId")) == _clean(expected_tournament_id),
        )
    add_check("candidate_exists", bool(record))
    add_check("prototype_id_present", bool(prototype_id))
    add_check("creator_accepted", is_creator_accepted_for_candidate(state, candidate_id))
    add_check("no_accepted_judgment", not accepted_judgment_exists_for_candidate(state, candidate_id))
    add_check("no_winner_or_media", not winner_or_media_started(state))
    add_check("paused_reasoning_recovery_state", is_reasoning_recovery_paused_state(state))
    add_check("repair_dispatch_recorded", bool(repair_report.get("repairDispatchRecorded")))
    add_check("repair_outcome_unrecoverable", bool(repair_report.get("repairOutcomeUnrecoverable")))
    add_check("raw_repair_unavailable", not bool(repair_report.get("rawRepairResponseAvailable")))
    add_check("parsed_repair_unavailable", not bool(repair_report.get("parsedRepairResponseAvailable")))
    add_check("offline_salvage_impossible", not bool(repair_report.get("offlineSalvagePossible")))
    add_check("salvage_assessment_impossible", not bool(salvage.get("offlineSalvagePossible")))
    add_check("pending_repair_unrecoverable", pending_repair_is_unrecoverable(state, candidate_id))
    add_check("source_judgment_id_present", bool(source_judgment_id))
    if expected_source_judgment_id:
        add_check("source_judgment_id_matches", source_judgment_id == _clean(expected_source_judgment_id))
    if expected_source_response_fingerprint:
        add_check(
            "source_response_fingerprint_matches",
            source_response_fp == _clean(expected_source_response_fingerprint),
        )
    if expected_source_parsed_response_fingerprint:
        add_check(
            "source_parsed_response_fingerprint_matches",
            source_parsed_fp == _clean(expected_source_parsed_response_fingerprint),
        )
    existing = get_candidate_judgment_resolution(state, candidate_id)
    if existing:
        add_check(
            "not_conflicting_policy",
            _clean(existing.get("operatorDecision")) == OPERATOR_DECISION_EXCLUDE_AND_CONTINUE,
            detail=_clean(existing.get("operatorDecision")),
        )
        if _clean(existing.get("additionalPaidCallAuthorized")).lower() in {"1", "true", "yes"}:
            add_check("no_paid_replacement_authorized", False)

    resolution_eligible = all(item.get("ok") for item in checks if item.get("name") != "not_conflicting_policy")
    already_resolved = has_operator_judgment_unavailable_resolution(state, candidate_id)
    if already_resolved:
        resolution_eligible = True

    expected_resolution = None
    if resolution_eligible and not already_resolved:
        expected_resolution = build_judgment_unavailable_resolution_record(
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            source_judgment_id=source_judgment_id,
            source_response_fingerprint=source_response_fp,
            source_parsed_response_fingerprint=source_parsed_fp,
            repair_dispatch_recorded=bool(repair_report.get("repairDispatchRecorded")),
            repair_response_available=bool(repair_report.get("rawRepairResponseAvailable")),
            repair_outcome_unrecoverable=True,
            repair_call_count_observed=int(metrics.get("judgeRepairCalls") or 0),
        )

    post_plan = None
    if resolution_eligible:
        preview = deepcopy(state)
        if not already_resolved and expected_resolution:
            persist_candidate_judgment_resolution(preview, candidate_id=candidate_id, resolution=expected_resolution)
            update_candidate_judge_state(
                preview,
                candidate_id=candidate_id,
                judge_status="unavailable",
                failure_reason=RESOLUTION_REASON_REPAIR_UNAVAILABLE,
            )
            _write_pending_operator_resolution(preview, candidate_id=candidate_id)
        post_plan = resolve_complete_ad_canonical_resume_plan(preview, read_only=True)

    return {
        "candidateId": candidate_id,
        "prototypeId": prototype_id or None,
        "resolutionEligible": bool(resolution_eligible and not already_resolved),
        "alreadyResolved": already_resolved,
        "consistencyChecks": checks,
        "expectedResolution": expected_resolution,
        "expectedResumePlanAfterResolution": post_plan,
        "repairInspectSummary": {
            "repairDispatchRecorded": repair_report.get("repairDispatchRecorded"),
            "repairOutcomeUnrecoverable": repair_report.get("repairOutcomeUnrecoverable"),
            "unavailableReason": repair_report.get("unavailableReason"),
            "offlineSalvagePossible": repair_report.get("offlineSalvagePossible"),
        },
        "paidCalls": 0,
        "stateMutated": False,
    }


def apply_judge_unavailable_resolution(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    expected_job_id: str = "",
    expected_tournament_id: str = "",
    expected_source_judgment_id: str = "",
    expected_source_response_fingerprint: str = "",
    expected_source_parsed_response_fingerprint: str = "",
) -> Dict[str, Any]:
    assessment = assess_judge_unavailable_resolution(
        state,
        candidate_id=candidate_id,
        expected_job_id=expected_job_id,
        expected_tournament_id=expected_tournament_id,
        expected_source_judgment_id=expected_source_judgment_id,
        expected_source_response_fingerprint=expected_source_response_fingerprint,
        expected_source_parsed_response_fingerprint=expected_source_parsed_response_fingerprint,
    )
    report: Dict[str, Any] = {
        "candidateId": candidate_id,
        "resolutionApplied": False,
        "alreadyResolved": bool(assessment.get("alreadyResolved")),
        "contractVersion": BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CONTRACT_VERSION,
        "paidCalls": 0,
        "stateMutated": False,
        "reason": "",
        "nextSafeAction": "",
    }
    if assessment.get("alreadyResolved"):
        report["reason"] = "builder2_judge_unavailable_resolution_already_applied"
        report["nextSafeAction"] = "builder2_complete_ad_reasoning_resume"
        existing = get_candidate_judgment_resolution(state, candidate_id)
        report["resolution"] = existing
        return report
    if not assessment.get("resolutionEligible"):
        failed = [item for item in assessment.get("consistencyChecks") or [] if not item.get("ok")]
        report["reason"] = "builder2_judge_unavailable_resolution_guard_failed"
        report["failedChecks"] = failed
        report["nextSafeAction"] = "builder2_judge_repair_response_inspect"
        return report

    expected = assessment.get("expectedResolution")
    if not isinstance(expected, dict):
        report["reason"] = "builder2_judge_unavailable_resolution_expected_record_missing"
        return report

    persist_candidate_judgment_resolution(state, candidate_id=candidate_id, resolution=expected)
    update_candidate_judge_state(
        state,
        candidate_id=candidate_id,
        judge_status="unavailable",
        failure_reason=RESOLUTION_REASON_REPAIR_UNAVAILABLE,
    )
    _write_pending_operator_resolution(state, candidate_id=candidate_id)
    report["resolutionApplied"] = True
    report["stateMutated"] = True
    report["resolution"] = get_candidate_judgment_resolution(state, candidate_id)
    report["reason"] = "builder2_judge_unavailable_resolution_applied"
    report["nextSafeAction"] = "builder2_complete_ad_reasoning_resume"
    return report


def run_judge_unavailable_resolution(
    *,
    job_id: str,
    candidate_id: str = "",
    apply: bool = False,
    expected_source_judgment_id: str = "",
    expected_source_response_fingerprint: str = "",
    expected_source_parsed_response_fingerprint: str = "",
) -> Dict[str, Any]:
    if not redis_configured():
        return {
            "ok": False,
            "failureReason": "builder2_judge_unavailable_resolution_redis_unconfigured",
            "jobId": job_id,
            "paidCalls": 0,
            "stateMutated": False,
        }
    state = load_tournament_state(job_id)
    if not isinstance(state, dict) or not state:
        return {
            "ok": False,
            "failureReason": "builder2_judge_unavailable_resolution_job_not_found",
            "jobId": job_id,
            "paidCalls": 0,
            "stateMutated": False,
        }

    target = _clean(candidate_id)
    if not target:
        unresolved = []
        for cid in sorted((state.get("candidates") or {}).keys()):
            if pending_repair_is_unrecoverable(state, str(cid)):
                unresolved.append(str(cid))
        if len(unresolved) != 1:
            return {
                "ok": False,
                "failureReason": "builder2_judge_unavailable_resolution_candidate_required",
                "jobId": job_id,
                "unresolvedCandidateIds": unresolved,
                "paidCalls": 0,
                "stateMutated": False,
            }
        target = unresolved[0]

    assessment = assess_judge_unavailable_resolution(
        state,
        candidate_id=target,
        expected_job_id=job_id,
        expected_tournament_id=_clean(state.get("tournamentId")),
        expected_source_judgment_id=expected_source_judgment_id,
        expected_source_response_fingerprint=expected_source_response_fingerprint,
        expected_source_parsed_response_fingerprint=expected_source_parsed_response_fingerprint,
    )
    if not apply:
        return {
            "ok": True,
            "jobId": job_id,
            "tournamentId": _clean(state.get("tournamentId")) or None,
            "dryRun": True,
            "assessment": assessment,
            "paidCalls": 0,
            "stateMutated": False,
        }

    worker_token = new_worker_token()
    lease_acquired = acquire_job_lease(job_id, worker_token)
    if not lease_acquired:
        return {
            "ok": False,
            "failureReason": "builder2_judge_unavailable_resolution_lease_not_acquired",
            "jobId": job_id,
            "candidateId": target,
            "paidCalls": 0,
            "stateMutated": False,
        }
    try:
        result = apply_judge_unavailable_resolution(
            state,
            candidate_id=target,
            expected_job_id=job_id,
            expected_tournament_id=_clean(state.get("tournamentId")),
            expected_source_judgment_id=expected_source_judgment_id,
            expected_source_response_fingerprint=expected_source_response_fingerprint,
            expected_source_parsed_response_fingerprint=expected_source_parsed_response_fingerprint,
        )
        if result.get("stateMutated"):
            save_tournament_state(job_id, state)
        return {
            "ok": True,
            "jobId": job_id,
            "tournamentId": _clean(state.get("tournamentId")) or None,
            "dryRun": False,
            "apply": result,
            "paidCalls": 0,
            "stateMutated": bool(result.get("stateMutated")),
        }
    finally:
        release_job_lease(job_id, worker_token)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_JOB_ID"))
    if not job_id:
        print("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_JOB_ID is required", file=sys.stderr)
        return 2
    candidate_id = _clean(os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CANDIDATE_ID"))
    apply = _clean(os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_APPLY")).lower() in {"1", "true", "yes"}
    expected_source_judgment_id = _clean(os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_SOURCE_JUDGMENT_ID"))
    expected_source_response_fingerprint = _clean(
        os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_SOURCE_RESPONSE_FINGERPRINT")
    )
    expected_source_parsed_response_fingerprint = _clean(
        os.environ.get("BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_SOURCE_PARSED_RESPONSE_FINGERPRINT")
    )
    report = run_judge_unavailable_resolution(
        job_id=job_id,
        candidate_id=candidate_id,
        apply=apply,
        expected_source_judgment_id=expected_source_judgment_id,
        expected_source_response_fingerprint=expected_source_response_fingerprint,
        expected_source_parsed_response_fingerprint=expected_source_parsed_response_fingerprint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
