"""
Builder2 Winner development failure inspector — read-only offline diagnosis.

Run:
  BUILDER2_WINNER_DEVELOPMENT_FAILURE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_winner_development_failure_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_resume_plan import resolve_winner_development_action
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_store import load_tournament_state
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
)
from engine.builder2_winner_response_ledger import (
    find_latest_winner_attempt,
    resolve_winner_parsed_response_fingerprint,
    resolve_winner_response_fingerprint,
)
from engine.builder2_winner_validation_replay import replay_prepare_and_validate
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

_NESTED_FIELD_KEYS = (
    "sequence",
    "sceneVariations",
    "visualAnchor",
    "advertisingClosure",
    "headlineDecision",
    "advertisingSloganEvidence",
    "videoPrompt",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _nested_type_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _type_name(value.get(key)) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_type_name(item) for item in value[:8]]
    return _type_name(value)


def _safe_preview(value: Any, *, limit: int = 120) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        text = "dict:" + ",".join(sorted(str(key) for key in value.keys())[:10])
    elif isinstance(value, list):
        text = f"list[{len(value)}]"
    else:
        text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def inspect_winner_development_failure(
    state: Dict[str, Any],
    *,
    read_only: bool = True,
) -> Dict[str, Any]:
    job_id = _clean(state.get("jobId"))
    tournament_id = _clean(state.get("tournamentId"))
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((payload or {}).get("parsed") or {})
    failure = state.get("winnerDevelopmentFailure") if isinstance(state.get("winnerDevelopmentFailure"), dict) else {}
    attempt = find_latest_winner_attempt(state, winner_id) if winner_id else None
    raw_fp = resolve_winner_response_fingerprint(payload or {})
    parsed_fp = resolve_winner_parsed_response_fingerprint(payload or {})
    winner_action = resolve_winner_development_action(state, winner_candidate_id=winner_id)

    replay_report: Dict[str, Any] = {}
    if payload and winner_id:
        winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        judgment_id = _clean(winner_rec.get("judgmentId"))
        winning_judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
        strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
        source_reference = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=winning_candidate,
            candidate_id=winner_id,
        )
        replay_report = replay_prepare_and_validate(
            parsed,
            source_reference=source_reference,
            winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
            winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else None,
            tournament_state=state,
            job_id=job_id,
            tournament_id=tournament_id,
        )

    top_level_types = {key: _type_name(parsed.get(key)) for key in sorted(parsed.keys())}
    nested_types = {field: _nested_type_report(parsed.get(field)) for field in _NESTED_FIELD_KEYS if field in parsed}

    return {
        "jobId": job_id or None,
        "tournamentId": tournament_id or None,
        "candidateId": winner_id or None,
        "prototypeId": _clean((payload or {}).get("prototypeId") or state.get("winnerDevelopmentPrototypeId")) or None,
        "callType": _clean((attempt or {}).get("callType") or "normal") or "normal",
        "attemptId": _clean((payload or {}).get("attemptId") or (attempt or {}).get("attemptId")) or None,
        "responseAvailable": bool(payload),
        "rawResponseAvailable": bool((payload or {}).get("rawResponseAvailable")) or bool(_clean((payload or {}).get("rawResponseText"))),
        "parsedResponseAvailable": bool(parsed),
        "responseLocation": _clean((payload or {}).get("responseLocation")) or (PARSED_WINNER_RESPONSE_KEY if payload else None),
        "responseCharacterCount": int((payload or {}).get("responseCharCount") or len(_clean((payload or {}).get("rawResponseText"))) or 0),
        "responseFingerprint": raw_fp.get("effective"),
        "parsedResponseFingerprint": parsed_fp.get("effective"),
        "parsedResponseFingerprintDerived": parsed_fp.get("derived"),
        "responseFingerprintStored": raw_fp.get("storedPresent"),
        "parsedResponseFingerprintStored": parsed_fp.get("storedPresent"),
        "fingerprintDerivationPossible": parsed_fp.get("derivationPossible"),
        "topLevelKeys": sorted(parsed.keys()),
        "topLevelKeyCount": len(parsed),
        "topLevelTypes": top_level_types,
        "nestedFieldTypes": nested_types,
        "normalizationHistory": parsed.get("continuousEventSceneVariationsNormalization")
        or state.get("winnerDevelopmentNormalizationHistory"),
        "winnerDevelopmentFailure": failure or None,
        "winnerAction": winner_action.get("winnerAction"),
        "winnerOfflineRevalidationPossible": bool(replay_report.get("accepted")),
        "validationReplayAccepted": bool(replay_report.get("accepted")),
        "validationReplayFirstFailure": replay_report.get("firstFailure"),
        "validationReplayStages": replay_report.get("stages") or [],
        "paidCalls": 0,
        "stateMutated": False,
        "readOnly": read_only,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    job_id = _clean(os.environ.get("BUILDER2_WINNER_DEVELOPMENT_FAILURE_INSPECT_JOB_ID"))
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "BUILDER2_WINNER_DEVELOPMENT_FAILURE_INSPECT_JOB_ID_missing"}, indent=2))
        return 2
    if not redis_configured():
        print(json.dumps({"ok": False, "failureReason": "redis_unconfigured"}, indent=2))
        return 2
    with read_only_builder2_inspection():
        state = load_tournament_state(job_id)
        if not state:
            print(json.dumps({"ok": False, "failureReason": "job_not_found", "jobId": job_id}, indent=2))
            return 2
        report = inspect_winner_development_failure(state, read_only=True)
    print(json.dumps({"ok": True, **report}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
