"""
Builder2 media-resume contract inspector — read-only Winner→media handoff audit.

Run:
  BUILDER2_MEDIA_RESUME_CONTRACT_INSPECT_JOB_ID=<jobId> python -m engine.builder2_media_resume_contract_inspect
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from engine.builder2_final_output_diagnostics import (
    build_builder2_media_diagnostic_fields,
    collect_media_resume_contract_missing_fields,
    durable_final_url_present,
)
from engine.builder2_media_finalization_contract import resolve_raw_runway_artifact_url
from engine.builder2_tournament_completion_gate import (
    accepted_creator_count,
    accepted_judgment_count,
    missing_creator_prototype_ids,
)
from engine.builder2_tournament_store import load_tournament_state


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def inspect_media_resume_contract(state: Dict[str, Any]) -> Dict[str, Any]:
    media = _media_bucket(state)
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    prototype_id = _clean(state.get("winnerDevelopmentPrototypeId") or winner_rec.get("prototypeId"))
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    failure = state.get("winnerDevelopmentFailure") if isinstance(state.get("winnerDevelopmentFailure"), dict) else {}
    diagnostic = build_builder2_media_diagnostic_fields(state)
    missing = collect_media_resume_contract_missing_fields(state)
    raw_url = resolve_raw_runway_artifact_url(state)
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "selectedWinnerFound": bool(winner_id),
        "selectedWinnerCandidateId": winner_id or "",
        "selectedWinnerPrototypeId": prototype_id or "",
        "winnerDevelopmentPlanPresent": bool(plan),
        "winnerDevelopmentCandidateId": _clean(state.get("winnerDevelopmentCandidateId")),
        "winnerDevelopmentPrototypeId": prototype_id or "",
        "winnerDevelopmentAccepted": state.get("winnerDevelopmentAccepted") is True,
        "winnerDevelopmentSource": _clean(state.get("winnerDevelopmentSource")),
        "winnerDevelopmentPlanFingerprint": _clean(state.get("winnerDevelopmentPlanFingerprint")),
        "mediaContinuationRequired": state.get("mediaContinuationRequired") is True,
        "historicalWinnerFailureFound": bool(failure),
        "historicalWinnerFailureResolved": state.get("winnerDevelopmentFailureResolved") is True,
        "acceptedCreatorsCount": accepted_creator_count(state),
        "acceptedJudgmentsCount": accepted_judgment_count(state),
        "missingPrototypeIds": list(missing_creator_prototype_ids(state)),
        "startImagePresent": bool(_clean(media.get("startImageDataUri") or media.get("startImageArtifactUrl"))),
        "runwayTaskIdPresent": bool(_clean(media.get("runwayTaskId"))),
        "rawRunwayVideoPresent": bool(raw_url),
        "closureVideoPresent": diagnostic["closureVideoPresent"],
        "durableFinalUrlPresent": durable_final_url_present(state),
        "mediaCompleted": diagnostic["mediaCompleted"],
        "mediaResumeNeeded": diagnostic["mediaResumeNeeded"],
        "mediaDiagnosticPhase": diagnostic["mediaDiagnosticPhase"],
        "finalOutputAvailable": diagnostic["finalOutputAvailable"],
        "mediaResumeReady": diagnostic["mediaResumeReady"],
        "mediaResumeBlockedReason": diagnostic["mediaResumeBlockedReason"],
        "mediaResumeMissingFields": missing,
        "stateStoreAgreement": diagnostic["stateStoreAgreement"],
        "stateMutated": False,
        "paidCalls": 0,
    }


def main() -> int:
    job_id = _env("BUILDER2_MEDIA_RESUME_CONTRACT_INSPECT_JOB_ID")
    if not job_id:
        print("BUILDER2_MEDIA_RESUME_CONTRACT_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    state = load_tournament_state(job_id, read_only=True)
    if not isinstance(state, dict) or not state:
        print(json.dumps({"jobId": job_id, "error": "tournament_state_missing"}))
        return 1
    report = inspect_media_resume_contract(state)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
