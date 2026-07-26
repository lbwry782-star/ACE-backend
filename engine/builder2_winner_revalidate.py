"""
Builder2 Winner offline revalidation — zero-cost reprocess of a stored parsed Winner response.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from engine.builder2_accepted_creator_store import backfill_accepted_creator_index
from engine.builder2_accepted_judgment_store import backfill_accepted_judgment_index
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    load_revalidatable_parsed_winner_response,
    process_winner_development_response,
)

logger = logging.getLogger(__name__)

DEFAULT_REVALIDATE_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def run_one_winner_revalidate(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "parsedResponseAvailable": False,
        "topLevelKeyCount": 0,
        "winnerDevelopmentAccepted": False,
        "winnerReused": False,
        "winnerNormalCalls": 0,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "judgeCalls": 0,
        "startImageCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "mediaContinuationRequired": False,
        "failureReason": None,
        "ok": False,
    }

    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_winner_revalidate_job_not_found"
        return report

    if is_valid_persisted_winner_development(state):
        report["winnerDevelopmentAccepted"] = True
        report["winnerReused"] = True
        report["mediaContinuationRequired"] = True
        report["ok"] = True
        return report

    payload = load_revalidatable_parsed_winner_response(state)
    if payload is None:
        report["failureReason"] = "builder2_winner_revalidate_parsed_response_missing"
        return report

    report["parsedResponseAvailable"] = True
    report["topLevelKeyCount"] = int(payload.get("topLevelKeyCount") or len(payload.get("topLevelKeys") or []))

    backfill_accepted_creator_index(state)
    backfill_accepted_judgment_index(state)
    candidate_id = str(payload.get("candidateId") or state.get("winnerCandidateId") or "").strip()
    winner_rec = (state.get("candidates") or {}).get(candidate_id) or {}
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    strategy = state.get("strategyFoundation") or {}
    prototype_id = str(payload.get("prototypeId") or winner_rec.get("prototypeId") or "").strip()

    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    preservation_snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    parsed = payload.get("parsed") or {}
    if not isinstance(parsed, dict):
        report["failureReason"] = "builder2_winner_revalidate_parsed_response_invalid"
        return report

    try:
        winner_plan = process_winner_development_response(
            parsed,
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
            job_id=job_id,
            tournament_id=str(state.get("tournamentId") or ""),
        )
        persist_winner_development_atomically(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            winner_plan=winner_plan,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
        )
        state["mediaContinuationRequired"] = True
        save_tournament_state(job_id, state)
        report["winnerDevelopmentAccepted"] = True
        report["mediaContinuationRequired"] = True
        report["ok"] = True
    except Exception as exc:
        report["failureReason"] = str(getattr(exc, "args", [str(exc)])[0])

    return report


def print_revalidate_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "jobId",
            "parsedResponseAvailable",
            "topLevelKeyCount",
            "winnerDevelopmentAccepted",
            "winnerReused",
            "winnerNormalCalls",
            "strategyCalls",
            "creatorCalls",
            "judgeCalls",
            "startImageCalls",
            "runwayCalls",
            "ffmpegCalls",
            "mediaContinuationRequired",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_WINNER_REVALIDATE_JOB_ID", DEFAULT_REVALIDATE_JOB_ID)
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_winner_revalidate_job_id_missing"}, indent=2))
        return 1
    logger.info("BUILDER2_WINNER_REVALIDATE_START jobId=%s", job_id)
    report = run_one_winner_revalidate(job_id=job_id)
    print_revalidate_report(report)
    logger.info(
        "BUILDER2_WINNER_REVALIDATE_DONE jobId=%s ok=%s parsedResponseAvailable=%s",
        job_id,
        report.get("ok"),
        report.get("parsedResponseAvailable"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
