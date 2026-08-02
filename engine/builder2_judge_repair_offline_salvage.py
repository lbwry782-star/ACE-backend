"""
Builder2 Judge repair offline salvage — zero paid calls when repair response is valid.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_accepted_judgment_store import persist_accepted_judgment, update_candidate_judge_state
from engine.builder2_judge import collect_judge_structural_errors, validate_judge_response
from engine.builder2_judge_core_contract import is_judge_factual_grounding_gate_field
from engine.builder2_judge_pending_repair import mark_pending_repair_accepted
from engine.builder2_judge_response_ledger import (
    find_latest_attempt,
    finalize_judge_response_validation,
    ledger_entries,
    parsed_response_fingerprint,
    repair_attempts,
)
from engine.builder2_judge_structural_repair_classifier import is_substantive_factual_grounding_negative
from engine.builder2_strategy_evidence_grounding_contract import requires_strategy_evidence_grounding
from engine.builder2_tournament_metrics import record_judge_valid
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

JUDGE_REPAIR_OFFLINE_SALVAGE_LEDGER_KEY = "judgeRepairOfflineSalvage"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _creator_payload(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    creator = record.get("creatorOutput") or record.get("creatorSnapshot") or {}
    return creator if isinstance(creator, dict) else {}


def assess_repair_attempt_salvage(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    compatibility_mode = not requires_strategy_evidence_grounding(strategy=strategy)
    entry = entry or find_latest_attempt(state, candidate_id=candidate_id, call_type="repair")
    report: Dict[str, Any] = {
        "candidateId": candidate_id,
        "repairOutcome": "unavailable",
        "offlineSalvagePossible": False,
        "structuralValidationUnderCurrentContract": None,
        "substantiveEligibilityUnderCurrentContract": None,
        "falseBooleanMisclassified": None,
        "validationFailureField": None,
        "validationFailureReason": None,
    }
    if not entry:
        return report
    parsed = entry.get("parsedResponse") if isinstance(entry.get("parsedResponse"), dict) else {}
    if not parsed:
        report["repairOutcome"] = "unavailable"
        return report
    candidate = _creator_payload(state, candidate_id)
    structural_errors = collect_judge_structural_errors(
        parsed,
        candidate_id=candidate_id,
        candidate=candidate,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
    )
    field = _clean(entry.get("validationFailureField"))
    if field and is_substantive_factual_grounding_negative(field, parsed):
        report["falseBooleanMisclassified"] = True
        report["structuralValidationUnderCurrentContract"] = True
        report["substantiveEligibilityUnderCurrentContract"] = False
        report["offlineSalvagePossible"] = True
        report["repairOutcome"] = "salvageable"
        return report
    if structural_errors:
        report["structuralValidationUnderCurrentContract"] = False
        report["repairOutcome"] = "failed_structurally"
        report["validationFailureField"] = field or (structural_errors[0].split(":", 1)[-1] if structural_errors else None)
        report["validationFailureReason"] = _clean(entry.get("validationFailureReason"))
        return report
    try:
        product_input = None
        grounding = strategy.get("strategyEvidenceGrounding")
        if isinstance(grounding, dict):
            audit = grounding.get("productInputAudit")
            if isinstance(audit, dict):
                product_input = audit
        judgment, _, _ = validate_judge_response(
            dict(parsed),
            candidate_id=candidate_id,
            candidate=candidate,
            strategy_foundation=strategy,
            product_input=product_input,
            compatibility_mode=compatibility_mode,
        )
        report["structuralValidationUnderCurrentContract"] = True
        report["substantiveEligibilityUnderCurrentContract"] = bool(judgment.get("eligible"))
        report["offlineSalvagePossible"] = True
        report["repairOutcome"] = "salvageable"
        return report
    except Exception as exc:
        from engine.builder2_tournament_contracts import Builder2TournamentError

        if isinstance(exc, Builder2TournamentError):
            reason = str(exc.args[0] if exc.args else "builder2_judge_validation_failed")
            field = reason.split(":", 1)[1] if ":" in reason else ""
            report["validationFailureField"] = field or None
            report["validationFailureReason"] = reason
            if field and is_substantive_factual_grounding_negative(field, parsed):
                report["falseBooleanMisclassified"] = True
                report["structuralValidationUnderCurrentContract"] = True
                report["offlineSalvagePossible"] = True
                report["repairOutcome"] = "salvageable"
                return report
        report["structuralValidationUnderCurrentContract"] = False
        report["repairOutcome"] = "failed_structurally"
        return report


def salvage_repair_judgment_offline(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if _clean(_candidate_record(state, candidate_id).get("judgmentId")):
        return {
            "candidateId": candidate_id,
            "salvaged": False,
            "dryRun": dry_run,
            "reason": "builder2_judge_repair_salvage_judgment_already_accepted",
            "paidCalls": 0,
        }
    entry = find_latest_attempt(state, candidate_id=candidate_id, call_type="repair")
    assessment = assess_repair_attempt_salvage(state, candidate_id=candidate_id, entry=entry)
    report: Dict[str, Any] = {
        "candidateId": candidate_id,
        "salvaged": False,
        "dryRun": dry_run,
        "reason": "",
        "judgmentId": None,
        "parsedResponseFingerprint": assessment.get("parsedResponseFingerprint"),
        "eligible": None,
        "paidCalls": 0,
    }
    if not entry:
        report["reason"] = "builder2_judge_repair_salvage_missing_repair_response"
        return report
    if not assessment.get("offlineSalvagePossible"):
        report["reason"] = f"builder2_judge_repair_salvage_not_possible:{assessment.get('repairOutcome')}"
        return report
    parsed = entry.get("parsedResponse") if isinstance(entry.get("parsedResponse"), dict) else {}
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    candidate = _creator_payload(state, candidate_id)
    compatibility_mode = not requires_strategy_evidence_grounding(strategy=strategy)
    product_input = None
    grounding = strategy.get("strategyEvidenceGrounding")
    if isinstance(grounding, dict):
        audit = grounding.get("productInputAudit")
        if isinstance(audit, dict):
            product_input = audit
    judgment, total, scores = validate_judge_response(
        dict(parsed),
        candidate_id=candidate_id,
        candidate=candidate,
        strategy_foundation=strategy,
        product_input=product_input,
        compatibility_mode=compatibility_mode,
    )
    report["eligible"] = bool(judgment.get("eligible"))
    report["parsedResponseFingerprint"] = parsed_response_fingerprint(judgment)
    if dry_run:
        report["reason"] = "builder2_judge_repair_salvage_dry_run_ok"
        return report
    judgment_id = _clean(entry.get("judgmentId")) or f"judge-{candidate_id}-salvage"
    prototype_id = _clean(_candidate_record(state, candidate_id).get("prototypeId"))
    persist_accepted_judgment(
        state,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        judgment_id=judgment_id,
        judgment=judgment,
        total=total,
        scores=scores,
    )
    record_judge_valid(state, eligible=bool(judgment.get("eligible")))
    finalize_judge_response_validation(
        state,
        candidate_id=candidate_id,
        attempt_id=_clean(entry.get("attemptId")),
        judgment=judgment,
        deterministic_eligible=bool(judgment.get("eligible")),
        accepted=True,
    )
    mark_pending_repair_accepted(state, candidate_id=candidate_id)
    ledger = state.setdefault(JUDGE_REPAIR_OFFLINE_SALVAGE_LEDGER_KEY, {})
    ledger[candidate_id] = {
        "candidateId": candidate_id,
        "judgmentId": judgment_id,
        "parsedResponseFingerprint": report["parsedResponseFingerprint"],
        "eligible": report["eligible"],
    }
    report["salvaged"] = True
    report["judgmentId"] = judgment_id
    report["reason"] = "builder2_judge_repair_salvage_accepted"
    return report


def run_judge_repair_offline_salvage(*, job_id: str, dry_run: bool = False, candidate_id: str = "") -> Dict[str, Any]:
    if not redis_configured():
        return {"ok": False, "failureReason": "builder2_judge_repair_offline_salvage_redis_unconfigured", "jobId": job_id}
    state = load_tournament_state(job_id)
    if not isinstance(state, dict) or not state:
        return {"ok": False, "failureReason": "builder2_judge_repair_offline_salvage_job_not_found", "jobId": job_id}
    target = _clean(candidate_id) or _clean(os.environ.get("BUILDER2_JUDGE_REPAIR_OFFLINE_SALVAGE_CANDIDATE_ID"))
    if not target:
        repairs = []
        for cid in sorted((state.get("candidates") or {}).keys()):
            if repair_attempts(state, str(cid)):
                repairs.append(str(cid))
        if len(repairs) != 1:
            return {
                "ok": False,
                "failureReason": "builder2_judge_repair_offline_salvage_candidate_required",
                "jobId": job_id,
                "repairCandidateIds": repairs,
            }
        target = repairs[0]
    result = salvage_repair_judgment_offline(state, candidate_id=target, dry_run=dry_run)
    if result.get("salvaged") and not dry_run:
        save_tournament_state(job_id, state)
    return {"ok": True, "jobId": job_id, "result": result, "paidCalls": 0, "stateMutated": bool(result.get("salvaged") and not dry_run)}


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_REPAIR_OFFLINE_SALVAGE_JOB_ID"))
    if not job_id:
        print("BUILDER2_JUDGE_REPAIR_OFFLINE_SALVAGE_JOB_ID is required", file=sys.stderr)
        return 2
    dry_run = _clean(os.environ.get("BUILDER2_JUDGE_REPAIR_OFFLINE_SALVAGE_DRY_RUN")).lower() in {"1", "true", "yes"}
    report = run_judge_repair_offline_salvage(job_id=job_id, dry_run=dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
