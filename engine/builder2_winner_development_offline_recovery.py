"""
Builder2 Winner development offline recovery — fingerprint-guarded, zero paid calls.

Run dry-run:
  BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_JOB_ID=<jobId> \\
  BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_DRY_RUN=1 \\
  python -m engine.builder2_winner_development_offline_recovery

Run apply:
  BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_JOB_ID=<jobId> \\
  python -m engine.builder2_winner_development_offline_recovery
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import (
    WINNER_DEVELOPMENT_SOURCE_OFFLINE_RECOVERY,
    is_valid_persisted_winner_development,
    persist_accepted_winner_development_for_media,
)
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
)
from engine.builder2_winner_response_ledger import (
    backfill_winner_parsed_response_fingerprints,
    record_winner_validation_outcome,
    resolve_winner_parsed_response_fingerprint,
)
from engine.builder2_winner_validation_replay import replay_prepare_and_validate
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

PRODUCTION_WINNING_CARD_CANDIDATE_ID = "cand-1-winning_card-1-577b91f2"
WINNER_OFFLINE_RECOVERY_LEDGER_KEY = "winnerDevelopmentOfflineRecoveryLedger"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _clean(os.environ.get(name)).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _recovery_ledger(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.setdefault(WINNER_OFFLINE_RECOVERY_LEDGER_KEY, {})
    if not isinstance(ledger, dict):
        ledger = {}
        state[WINNER_OFFLINE_RECOVERY_LEDGER_KEY] = ledger
    return ledger


def recover_winner_development_offline(
    state: Dict[str, Any],
    *,
    expected_candidate_id: str = "",
    expected_parsed_fingerprint: str = "",
    compatibility_mode: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((payload or {}).get("parsed") or {})
    report: Dict[str, Any] = {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "candidateId": winner_id,
        "recoveryEligible": False,
        "normalizationRequired": False,
        "normalizationPaths": [],
        "exactValidationFailureBefore": None,
        "validationAcceptedAfter": False,
        "sourceParsedResponseFingerprint": None,
        "persisted": False,
        "winnerDevelopmentAccepted": is_valid_persisted_winner_development(state),
        "finalWinnerCandidateId": winner_id or None,
        "nextSafeAction": "inspect",
        "stateMutated": False,
        "paidCalls": 0,
        "openAICalls": 0,
        "dryRun": dry_run,
        "reason": "",
    }
    if not payload or not parsed:
        report["reason"] = "builder2_winner_offline_recovery_missing_parsed_response"
        return report
    if is_valid_persisted_winner_development(state):
        report["reason"] = "builder2_winner_offline_recovery_already_accepted"
        report["recoveryEligible"] = False
        report["nextSafeAction"] = "media_resume_or_inspect"
        return report

    expected_candidate = _clean(expected_candidate_id) or PRODUCTION_WINNING_CARD_CANDIDATE_ID
    if expected_candidate and _clean(payload.get("candidateId")) != expected_candidate:
        report["reason"] = "builder2_winner_offline_recovery_candidate_mismatch"
        return report

    backfill_winner_parsed_response_fingerprints(payload)
    parsed_fp = resolve_winner_parsed_response_fingerprint(payload)
    effective_fp = _clean(parsed_fp.get("effective"))
    report["sourceParsedResponseFingerprint"] = effective_fp or None
    expected_fp = _clean(expected_parsed_fingerprint)
    if expected_fp and effective_fp != expected_fp:
        report["reason"] = "builder2_winner_offline_recovery_fingerprint_mismatch"
        return report

    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    judgment_id = _clean(winner_rec.get("judgmentId"))
    winning_judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
        candidate_id=winner_id,
    )

    replay_before = replay_prepare_and_validate(
        deepcopy(parsed),
        source_reference=source_reference,
        winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
        winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else None,
        compatibility_mode=compatibility_mode,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        tournament_state=state,
    )
    first_fail = replay_before.get("firstFailure")
    report["exactValidationFailureBefore"] = first_fail
    report["validationAcceptedAfter"] = bool(replay_before.get("accepted"))
    report["recoveryEligible"] = bool(replay_before.get("accepted"))

    normalization_paths: list[str] = []
    if isinstance(parsed.get("videoPrompt"), dict):
        normalization_paths.append("videoPrompt")
    report["normalizationRequired"] = bool(normalization_paths)
    report["normalizationPaths"] = normalization_paths

    if not replay_before.get("accepted"):
        report["reason"] = "builder2_winner_offline_recovery_validation_failed"
        report["nextSafeAction"] = "inspect_failure"
        return report

    ledger = _recovery_ledger(state)
    prior = ledger.get("lastRecovery") if isinstance(ledger.get("lastRecovery"), dict) else {}
    if (
        prior.get("parsedResponseFingerprint") == effective_fp
        and prior.get("accepted")
        and is_valid_persisted_winner_development(state)
    ):
        report["reason"] = "builder2_winner_offline_recovery_already_applied"
        report["winnerDevelopmentAccepted"] = True
        report["nextSafeAction"] = "media_resume_or_inspect"
        return report

    if dry_run:
        report["reason"] = "builder2_winner_offline_recovery_dry_run_ok"
        report["nextSafeAction"] = "apply_offline_recovery"
        return report

    validated_plan = replay_before.get("validatedPlan")
    if not isinstance(validated_plan, dict):
        report["reason"] = "builder2_winner_offline_recovery_validated_plan_missing"
        return report

    persist_accepted_winner_development_for_media(
        state,
        candidate_id=winner_id,
        prototype_id=_clean(winner_rec.get("prototypeId") or payload.get("prototypeId")),
        winner_plan=validated_plan,
        winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
        winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else None,
        preservation_snapshot=validated_plan.get("winningCandidatePreservationSnapshot"),
        compatibility_mode=compatibility_mode,
        source=WINNER_DEVELOPMENT_SOURCE_OFFLINE_RECOVERY,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        save=False,
    )
    record_winner_validation_outcome(
        state,
        candidate_id=winner_id,
        accepted=True,
        failure_stage=None,
        failure_field_path=None,
        failure_reason=None,
        exception_class=None,
    )
    ledger["lastRecovery"] = {
        "candidateId": winner_id,
        "parsedResponseFingerprint": effective_fp,
        "accepted": True,
        "source": WINNER_DEVELOPMENT_SOURCE_OFFLINE_RECOVERY,
    }
    report["persisted"] = True
    report["winnerDevelopmentAccepted"] = True
    report["stateMutated"] = True
    report["reason"] = "builder2_winner_offline_recovery_applied"
    report["nextSafeAction"] = "media_resume_or_inspect"
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    job_id = _clean(os.environ.get("BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_JOB_ID"))
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_JOB_ID_missing"}, indent=2))
        return 2
    if not redis_configured():
        print(json.dumps({"ok": False, "failureReason": "redis_unconfigured"}, indent=2))
        return 2
    dry_run = _env_bool("BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_DRY_RUN", False)
    expected_fp = _clean(os.environ.get("BUILDER2_WINNER_DEVELOPMENT_OFFLINE_RECOVERY_PARSED_FINGERPRINT"))
    state = load_tournament_state(job_id)
    if not state:
        print(json.dumps({"ok": False, "failureReason": "job_not_found", "jobId": job_id}, indent=2))
        return 2
    report = recover_winner_development_offline(
        state,
        expected_parsed_fingerprint=expected_fp,
        dry_run=dry_run,
    )
    if report.get("stateMutated"):
        save_tournament_state(job_id, state)
    print(json.dumps({"ok": bool(report.get("persisted") or report.get("dryRun")), **report}, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("persisted") or report.get("dryRun") or report.get("recoveryEligible") else 1


if __name__ == "__main__":
    raise SystemExit(main())
