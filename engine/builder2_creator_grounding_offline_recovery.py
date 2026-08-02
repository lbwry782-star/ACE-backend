"""
Builder2 Creator grounding offline recovery — fingerprint-guarded, zero paid calls.

Run:
  BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_JOB_ID=<jobId> \\
  BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_CANDIDATE_ID=<candidateId> \\
  BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_DRY_RUN=1 \\
  python -m engine.builder2_creator_grounding_offline_recovery
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

from engine.builder2_complete_ad_creator_recovery import (
    CREATOR_GROUNDING_OFFLINE_RECOVERY_LEDGER_KEY,
    can_offline_revalidate_rejected_creator,
    load_rejected_creator_parsed_response,
    offline_revalidate_and_accept_rejected_creator,
)
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

PRODUCTION_WINNING_CARD_PARSED_FINGERPRINT = (
    "6104c552d86dfbe82f150171bac2173c52333fff0a581b0b6bb910e37fbd5723"
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _parsed_fingerprint(parsed: Dict[str, Any]) -> str:
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recovery_ledger(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.setdefault(CREATOR_GROUNDING_OFFLINE_RECOVERY_LEDGER_KEY, {})
    if not isinstance(ledger, dict):
        ledger = {}
        state[CREATOR_GROUNDING_OFFLINE_RECOVERY_LEDGER_KEY] = ledger
    ledger.setdefault("recoveredCandidates", {})
    return ledger


def recover_creator_grounding_offline(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    expected_fingerprint: str = "",
    compatibility_mode: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    from engine.builder2_complete_ad_creator_recovery import _candidate_already_accepted

    payload = load_rejected_creator_parsed_response(state, candidate_id)
    report: Dict[str, Any] = {
        "candidateId": candidate_id,
        "recovered": False,
        "dryRun": dry_run,
        "reason": "",
        "parsedResponseFingerprint": None,
        "nextSafeAction": "inspect_or_dispatch_creator",
        "paidCalls": 0,
        "openAICalls": 0,
    }
    if payload is None:
        report["reason"] = "builder2_creator_grounding_offline_recovery_missing_parsed_response"
        return report
    parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
    fingerprint = _parsed_fingerprint(parsed if isinstance(parsed, dict) else {})
    report["parsedResponseFingerprint"] = fingerprint
    expected = _clean(expected_fingerprint) or PRODUCTION_WINNING_CARD_PARSED_FINGERPRINT
    if expected and fingerprint != expected:
        report["reason"] = "builder2_creator_grounding_offline_recovery_fingerprint_mismatch"
        return report
    if _candidate_already_accepted(state, candidate_id):
        report["reason"] = "builder2_creator_grounding_offline_recovery_already_accepted"
        report["recovered"] = True
        report["nextSafeAction"] = "dispatch_judge_only"
        return report

    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    product_name = _clean(strategy.get("productNameResolved"))
    ok, blocked = can_offline_revalidate_rejected_creator(
        state,
        candidate_id=candidate_id,
        product_name=product_name,
        compatibility_mode=compatibility_mode,
    )
    if not ok:
        report["reason"] = blocked or "builder2_creator_grounding_offline_recovery_revalidation_failed"
        return report

    ledger = _recovery_ledger(state)
    recovered = ledger.setdefault("recoveredCandidates", {})
    prior = recovered.get(candidate_id) if isinstance(recovered.get(candidate_id), dict) else {}
    if prior.get("parsedResponseFingerprint") == fingerprint and prior.get("accepted"):
        report["reason"] = "builder2_creator_grounding_offline_recovery_already_applied"
        report["recovered"] = True
        report["nextSafeAction"] = "dispatch_judge_only"
        return report

    if dry_run:
        report["reason"] = "builder2_creator_grounding_offline_recovery_dry_run_ok"
        report["recovered"] = True
        report["nextSafeAction"] = "apply_offline_recovery_then_dispatch_judge"
        return report

    offline_revalidate_and_accept_rejected_creator(
        state,
        candidate_id=candidate_id,
        product_name=product_name,
        compatibility_mode=compatibility_mode,
        log_events=True,
    )
    recovered[candidate_id] = {
        "parsedResponseFingerprint": fingerprint,
        "accepted": True,
        "originalFailureReason": _clean(payload.get("failureReason")),
    }
    report["recovered"] = True
    report["reason"] = "builder2_creator_grounding_offline_recovery_applied"
    report["nextSafeAction"] = "dispatch_judge_only"
    return report


def recover_creator_grounding_offline_for_job(
    job_id: str,
    *,
    candidate_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
    expected_fingerprint: str = "",
    dry_run: bool = False,
    persist: bool = True,
) -> Dict[str, Any]:
    from engine.builder2_strategy_evidence_grounding_contract import requires_strategy_evidence_grounding

    if tournament_state is None:
        state = load_tournament_state(job_id)
        if not isinstance(state, dict) or not state:
            if not redis_configured():
                return {
                    "ok": False,
                    "failureReason": "builder2_creator_grounding_offline_recovery_redis_unconfigured",
                    "jobId": job_id,
                    "candidateId": candidate_id,
                }
            return {
                "ok": False,
                "failureReason": "builder2_creator_grounding_offline_recovery_job_not_found",
                "jobId": job_id,
                "candidateId": candidate_id,
            }
    else:
        state = tournament_state

    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    compatibility_mode = not requires_strategy_evidence_grounding(strategy=strategy)
    result = recover_creator_grounding_offline(
        state,
        candidate_id=candidate_id,
        expected_fingerprint=expected_fingerprint,
        compatibility_mode=compatibility_mode,
        dry_run=dry_run,
    )
    if persist and not dry_run and result.get("recovered") and result.get("reason") == (
        "builder2_creator_grounding_offline_recovery_applied"
    ):
        save_tournament_state(job_id, state)
    return {
        "ok": True,
        "jobId": job_id,
        "candidateId": candidate_id,
        "dryRun": dry_run,
        "persisted": bool(
            persist
            and not dry_run
            and result.get("recovered")
            and result.get("reason") == "builder2_creator_grounding_offline_recovery_applied"
        ),
        "recovery": result,
        "paidCalls": 0,
        "openAICalls": 0,
    }


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_JOB_ID"))
    candidate_id = _clean(os.environ.get("BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_CANDIDATE_ID"))
    if not job_id or not candidate_id:
        print(
            "BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_JOB_ID and "
            "BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_CANDIDATE_ID are required",
            file=sys.stderr,
        )
        return 2
    dry_run = _clean(os.environ.get("BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_DRY_RUN")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logger.info(
        "BUILDER2_CREATOR_GROUNDING_OFFLINE_RECOVERY_START jobId=%s candidateId=%s dryRun=%s",
        job_id,
        candidate_id,
        dry_run,
    )
    report = recover_creator_grounding_offline_for_job(job_id, candidate_id=candidate_id, dry_run=dry_run, persist=not dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
