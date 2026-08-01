"""
Builder2 Judge factual-grounding offline recovery — zero paid calls.

Run:
  BUILDER2_JUDGE_GROUNDING_OFFLINE_RECOVERY_JOB_ID=<jobId> python -m engine.builder2_judge_grounding_offline_recovery
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
from engine.builder2_judge import validate_judge_response
from engine.builder2_judge_grounding_failure_inspect import analyze_judge_response_attempt
from engine.builder2_tournament_metrics import record_judge_valid
from engine.builder2_tournament_store import load_tournament_state, register_judgment, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

JUDGE_OFFLINE_RECOVERY_LEDGER_KEY = "judgeGroundingOfflineRecovery"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _creator_payload(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    creator = record.get("creatorOutput") or record.get("creatorSnapshot") or {}
    return creator if isinstance(creator, dict) else {}


def _recovery_ledger(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.setdefault(JUDGE_OFFLINE_RECOVERY_LEDGER_KEY, {})
    if not isinstance(ledger, dict):
        ledger = {}
        state[JUDGE_OFFLINE_RECOVERY_LEDGER_KEY] = ledger
    ledger.setdefault("recoveredCandidates", {})
    return ledger


def _select_recoverable_entry(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    if _clean(_candidate_record(state, candidate_id).get("judgmentId")):
        return None
    ledger = state.get("judgeResponseLedgerByCandidate") or {}
    entries = [item for item in (ledger.get(candidate_id) or []) if isinstance(item, dict)]
    if not entries:
        return None
    for entry in reversed(entries):
        parsed = entry.get("parsedResponse")
        if isinstance(parsed, dict) and parsed:
            return entry
    return None


def recover_candidate_judgment_offline(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    compatibility_mode: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    entry = _select_recoverable_entry(state, candidate_id)
    report: Dict[str, Any] = {
        "candidateId": candidate_id,
        "recovered": False,
        "dryRun": dry_run,
        "reason": "",
        "judgmentId": None,
        "parsedResponseFingerprint": "",
        "eligible": None,
        "paidCalls": 0,
    }
    if not entry:
        report["reason"] = "builder2_judge_offline_recovery_missing_parsed_response"
        return report
    analysis = analyze_judge_response_attempt(
        state=state,
        candidate_id=candidate_id,
        entry=entry,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
    )
    report["parsedResponseFingerprint"] = analysis.get("parsedResponseFingerprint")
    if not analysis.get("offlineRevalidationPossible"):
        report["reason"] = "builder2_judge_offline_recovery_not_structurally_valid"
        return report
    if not analysis.get("offlinePersistencePossible"):
        report["reason"] = "builder2_judge_offline_recovery_judgment_already_accepted"
        return report

    parsed = deepcopy(entry.get("parsedResponse") or {})
    fingerprint = _clean(analysis.get("parsedResponseFingerprint"))
    recovery_ledger = _recovery_ledger(state)
    recovered = recovery_ledger.setdefault("recoveredCandidates", {})
    prior = recovered.get(candidate_id) if isinstance(recovered.get(candidate_id), dict) else {}
    if prior.get("parsedResponseFingerprint") == fingerprint and prior.get("accepted"):
        report["reason"] = "builder2_judge_offline_recovery_already_applied"
        report["judgmentId"] = prior.get("judgmentId")
        report["eligible"] = prior.get("eligible")
        report["recovered"] = True
        return report

    candidate = _creator_payload(state, candidate_id)
    product_input = None
    grounding = strategy.get("strategyEvidenceGrounding")
    if isinstance(grounding, dict):
        audit = grounding.get("productInputAudit")
        if isinstance(audit, dict):
            product_input = audit
    judgment, total, scores = validate_judge_response(
        parsed,
        candidate_id=candidate_id,
        candidate=candidate,
        strategy_foundation=strategy,
        product_input=product_input,
        compatibility_mode=compatibility_mode,
    )
    judgment_id = _clean(entry.get("judgmentId")) or f"judge-{candidate_id}-{uuid.uuid4().hex[:8]}"
    report["judgmentId"] = judgment_id
    report["eligible"] = bool(judgment.get("eligible"))
    if dry_run:
        report["reason"] = "builder2_judge_offline_recovery_dry_run_ok"
        report["recovered"] = True
        return report

    register_judgment(
        state,
        {
            "judgmentId": judgment_id,
            "candidateId": candidate_id,
            "judgment": judgment,
            "totalScore": total,
            "scores": scores,
            "eligible": judgment.get("eligible"),
            "completedAt": entry.get("recordedAt"),
            "source": "judge_grounding_offline_recovery",
            "parsedResponseFingerprint": fingerprint,
        },
    )
    cand_rec = _candidate_record(state, candidate_id)
    cand_rec["judgmentId"] = judgment_id
    cand_rec["eligible"] = bool(judgment.get("eligible"))
    cand_rec["totalScore"] = total
    cand_rec["tieScores"] = scores
    update_candidate_judge_state(
        state,
        candidate_id=candidate_id,
        judge_status="accepted",
        judgment_id=judgment_id,
        judgment_snapshot=judgment,
    )
    record_judge_valid(state, eligible=bool(judgment.get("eligible")))
    recovered[candidate_id] = {
        "judgmentId": judgment_id,
        "parsedResponseFingerprint": fingerprint,
        "eligible": bool(judgment.get("eligible")),
        "accepted": True,
    }
    report["recovered"] = True
    report["reason"] = "builder2_judge_offline_recovery_applied"
    return report


def recover_judge_grounding_offline_for_job(
    job_id: str,
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    persist: bool = True,
) -> Dict[str, Any]:
    if tournament_state is None:
        if not redis_configured():
            return {"ok": False, "failureReason": "builder2_judge_offline_recovery_redis_unconfigured", "jobId": job_id}
        state = load_tournament_state(job_id)
    else:
        state = tournament_state
    if not isinstance(state, dict) or not state:
        return {"ok": False, "failureReason": "builder2_judge_offline_recovery_job_not_found", "jobId": job_id}

    from engine.builder2_strategy_evidence_grounding_contract import requires_strategy_evidence_grounding

    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    compatibility_mode = not requires_strategy_evidence_grounding(strategy=strategy)
    candidate_ids = sorted(
        cid
        for cid, rec in (state.get("candidates") or {}).items()
        if isinstance(rec, dict)
        and (rec.get("creatorAcceptanceStatus") == "accepted" or rec.get("validationStatus") == "accepted")
        and not _clean(rec.get("judgmentId"))
    )
    results: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        results.append(
            recover_candidate_judgment_offline(
                state,
                candidate_id=candidate_id,
                compatibility_mode=compatibility_mode,
                dry_run=dry_run,
            )
        )
    if persist and not dry_run and any(item.get("recovered") for item in results):
        save_tournament_state(job_id, state)
    return {
        "ok": True,
        "jobId": job_id,
        "dryRun": dry_run,
        "persisted": bool(persist and not dry_run and any(item.get("recovered") for item in results)),
        "candidateResults": results,
        "recoveredCount": sum(1 for item in results if item.get("recovered")),
        "paidCalls": 0,
        "openAICalls": 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_JUDGE_GROUNDING_OFFLINE_RECOVERY_JOB_ID"))
    if not job_id:
        print("BUILDER2_JUDGE_GROUNDING_OFFLINE_RECOVERY_JOB_ID is required", file=sys.stderr)
        return 2
    dry_run = _clean(os.environ.get("BUILDER2_JUDGE_GROUNDING_OFFLINE_RECOVERY_DRY_RUN")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logger.info(
        "BUILDER2_JUDGE_GROUNDING_OFFLINE_RECOVERY_START jobId=%s dryRun=%s",
        job_id,
        dry_run,
    )
    report = recover_judge_grounding_offline_for_job(job_id, dry_run=dry_run, persist=not dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
