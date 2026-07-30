"""
Builder2 Winner-development recovery inspector — read-only, zero side effects.

Run:
  BUILDER2_WINNER_DEVELOPMENT_RESUME_INSPECT_JOB_ID=<jobId> python -m engine.builder2_winner_development_resume_inspect
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state
from engine.builder2_winner_offline_salvage import inspect_winner_development_recovery_state
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
    prepare_and_validate_persisted_winner_offline,
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def inspect_winner_development_resume(
    state: Dict[str, Any],
    *,
    attempt_offline_validation: bool = True,
) -> Dict[str, Any]:
    report = inspect_winner_development_recovery_state(state)
    if not attempt_offline_validation:
        return report

    winner_id = _clean(report.get("winnerCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    judgment_id = _clean(winner_rec.get("judgmentId"))
    winning_judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    parsed_payload = load_revalidatable_parsed_winner_response(state)
    if not winner_id or not parsed_payload:
        return report

    report["offlineSalvageAttempted"] = True
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=winner_id,
    )
    try:
        prepare_and_validate_persisted_winner_offline(
            dict(parsed_payload.get("parsed") or {}),
            source_reference=source,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else None,
            tournament_state=state,
            job_id=_clean(state.get("jobId")),
            tournament_id=_clean(state.get("tournamentId")),
        )
        report["offlineSalvageValidationPassed"] = True
        report["offlineSalvageFailureField"] = ""
        refreshed = inspect_winner_development_recovery_state(
            state,
            offline_salvage_attempted=True,
            offline_salvage_validation_passed=True,
        )
        report.update(
            {
                "judgeRequiresSeparateHeadline": refreshed.get("judgeRequiresSeparateHeadline"),
                "compatibilityHeadlineMirrorsSlogan": refreshed.get("compatibilityHeadlineMirrorsSlogan"),
                "singleSloganContractSatisfied": refreshed.get("singleSloganContractSatisfied"),
                "headlineDecision": refreshed.get("headlineDecision"),
                "canonicalCopySatisfiedBy": refreshed.get("canonicalCopySatisfiedBy"),
            }
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "")
        report["offlineSalvageFailureField"] = reason.split(":", 1)[-1] if reason else ""
    return report


def main() -> int:
    job_id = _env("BUILDER2_WINNER_DEVELOPMENT_RESUME_INSPECT_JOB_ID")
    if not job_id:
        print("BUILDER2_WINNER_DEVELOPMENT_RESUME_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    state = load_tournament_state(job_id)
    if not isinstance(state, dict) or not state:
        print(json.dumps({"jobId": job_id, "error": "tournament_state_missing"}))
        return 1
    report = inspect_winner_development_resume(state)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
