"""
Builder2 Winner-development resume inspector — read-only safe diagnostics.

Run:
  BUILDER2_WINNER_RESUME_INSPECT_JOB_ID=<jobId> python -m engine.builder2_winner_resume_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from typing import Any, Dict, Optional

from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

_WINNER_METRIC_KEYS = (
    "winnerDevelopmentCalls",
    "winnerNormalCalls",
    "winnerRepairCalls",
    "winnerRetryCalls",
    "totalReasoningCalls",
)

_HEADLINE_NECESSITY_BOOLEAN_KEYS = (
    "headlineNeeded",
    "visualWouldWorkWithoutHeadline",
    "headlineRecommended",
)

_MAX_HEADLINE_DECISION_REASON_CHARS = 80
_MAX_HEADLINE_NECESSITY_NOTES_CHARS = 120
_MAX_HEADLINE_FORM_CHARS = 32


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_text_field(value: Any, *, allow_short_enum: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "keyExists": value is not None,
        "valuePresent": False,
        "valueType": type(value).__name__ if value is not None else "missing",
    }
    if value is None:
        report["value"] = None
        report["characterCount"] = 0
        return report
    if isinstance(value, bool):
        report["valuePresent"] = True
        report["value"] = value
        return report
    if isinstance(value, (int, float)):
        report["valuePresent"] = True
        report["value"] = value
        return report
    if isinstance(value, str):
        report["characterCount"] = len(value)
        report["valuePresent"] = bool(value.strip())
        if not value.strip():
            report["value"] = value
        elif allow_short_enum and len(value.strip()) <= _MAX_HEADLINE_FORM_CHARS:
            report["value"] = value.strip()
        return report
    report["valuePresent"] = True
    return report


def _safe_headline_decision(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"keyExists": False, "valuePresent": False, "valueType": "missing"}
    if isinstance(value, str):
        decision = _clean(value)
        return {
            "keyExists": True,
            "valuePresent": bool(decision),
            "valueType": "str",
            "decision": decision or None,
            "reasonSource": None,
            "reasonPresent": False,
        }
    if not isinstance(value, dict):
        return {
            "keyExists": True,
            "valuePresent": True,
            "valueType": type(value).__name__,
        }
    reason = value.get("reason")
    reason_text = _clean(reason) if isinstance(reason, str) else ""
    report: Dict[str, Any] = {
        "keyExists": True,
        "valuePresent": True,
        "valueType": "dict",
        "decision": _clean(value.get("decision")) or None,
        "reasonSource": _clean(value.get("reasonSource")) or None,
        "reasonPresent": bool(reason_text),
    }
    if reason_text and len(reason_text) <= _MAX_HEADLINE_DECISION_REASON_CHARS:
        report["reason"] = reason_text
    elif reason is not None:
        report["reasonRedacted"] = True
        report["reasonCharacterCount"] = len(str(reason))
    return report


def _safe_headline_necessity_assessment(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "keyExists": value is not None,
            "valuePresent": False,
            "valueType": type(value).__name__ if value is not None else "missing",
        }
    report: Dict[str, Any] = {
        "keyExists": True,
        "valuePresent": True,
        "valueType": "dict",
    }
    for key in _HEADLINE_NECESSITY_BOOLEAN_KEYS:
        if key in value and isinstance(value[key], bool):
            report[key] = value[key]
    notes = value.get("notes")
    if isinstance(notes, str) and notes.strip():
        report["notesPresent"] = True
        report["notesCharacterCount"] = len(notes)
        if len(notes.strip()) <= _MAX_HEADLINE_NECESSITY_NOTES_CHARS:
            report["notes"] = notes.strip()
        else:
            report["notesRedacted"] = True
    else:
        report["notesPresent"] = False
    return report


def _resolve_winner_identity(state: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId")) or None
    winner_rec = (state.get("candidates") or {}).get(winner_id or "") or {}
    judgment_id = _clean(winner_rec.get("judgmentId")) or None
    judgment_rec = (state.get("judgments") or {}).get(judgment_id or "") if judgment_id else None
    return winner_id, winner_rec, judgment_id, judgment_rec if isinstance(judgment_rec, dict) else None


def inspect_builder2_winner_resume(job_id: str = "") -> Dict[str, Any]:
    jid = _clean(job_id)
    if not jid:
        return {"ok": False, "error": "builder2_winner_resume_inspect_job_id_missing", "jobId": None}
    if not redis_configured():
        return {"ok": False, "error": "builder2_winner_resume_inspect_redis_unconfigured", "jobId": jid}

    with read_only_builder2_inspection() as mutation_counter:
        raw = _read_raw(jid)
        if raw is None:
            return {
                "ok": False,
                "error": "builder2_winner_resume_inspect_tournament_not_found",
                "jobId": jid,
                "redisMutations": mutation_counter.redis_mutations,
            }
        state = deepcopy(raw)

        winner_id, winner_rec, _judgment_id, judgment_rec = _resolve_winner_identity(state)
        judgment = (judgment_rec or {}).get("judgment") if isinstance((judgment_rec or {}).get("judgment"), dict) else {}

        parsed_bucket = state.get(PARSED_WINNER_RESPONSE_KEY)
        parsed_exists = (
            isinstance(parsed_bucket, dict)
            and isinstance(parsed_bucket.get("parsed"), dict)
            and bool(parsed_bucket.get("parsed"))
        )
        parsed = dict(parsed_bucket.get("parsed") or {}) if parsed_exists else {}
        parsed_candidate_id = _clean(parsed_bucket.get("candidateId")) or None if isinstance(parsed_bucket, dict) else None

        metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
        paid_raw = state.get("winnerDevelopmentPaidCallRecorded")
        paid_exists = "winnerDevelopmentPaidCallRecorded" in state

        winner_score = winner_rec.get("totalScore")
        if winner_score is None and isinstance(judgment_rec, dict):
            winner_score = judgment_rec.get("totalScore")

        report: Dict[str, Any] = {
            "ok": True,
            "jobId": jid,
            "tournamentStatus": state.get("status"),
            "failureStage": state.get("failureStage"),
            "failureReason": state.get("failureReason"),
            "canResume": state.get("canResume"),
            "winnerCandidateId": winner_id,
            "winnerPrototypeId": _clean(winner_rec.get("prototypeId") or state.get("winnerDevelopmentPrototypeId")) or None,
            "winnerScore": winner_score,
            "parsedWinnerCandidateId": parsed_candidate_id,
            "parsedWinnerExists": parsed_exists,
            "parsedWinnerHeadline": _safe_text_field(parsed.get("headline")),
            "parsedWinnerHeadlineText": _safe_text_field(parsed.get("headlineText")),
            "parsedWinnerHeadlineCoreKeyword": _safe_text_field(parsed.get("headlineCoreKeyword")),
            "parsedWinnerHeadlineForm": _safe_text_field(parsed.get("headlineForm"), allow_short_enum=True),
            "parsedWinnerHeadlineDecision": _safe_headline_decision(parsed.get("headlineDecision")),
            "winningJudgmentHeadlineNecessityAssessment": _safe_headline_necessity_assessment(
                judgment.get("headlineNecessityAssessment")
            ),
            "winnerDevelopmentPaidCallRecorded": {
                "keyExists": paid_exists,
                "valueType": type(paid_raw).__name__ if paid_exists else "missing",
                "value": paid_raw if isinstance(paid_raw, bool) else None,
            },
            "winnerMetrics": {key: int(metrics.get(key) or 0) for key in _WINNER_METRIC_KEYS},
            "winnerDevelopmentPlanExists": isinstance(state.get("winnerDevelopmentPlan"), dict)
            and bool(state.get("winnerDevelopmentPlan")),
            "winnerDevelopmentAccepted": is_valid_persisted_winner_development(state),
            "reasoningComplete": bool(state.get("reasoningComplete")),
            "mediaStarted": bool(state.get("mediaStarted")),
            "redisMutations": mutation_counter.redis_mutations,
        }
        return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_WINNER_RESUME_INSPECT_JOB_ID"))
    try:
        report = inspect_builder2_winner_resume(job_id)
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return 0 if report.get("ok") else 1
    except Exception:
        logger.exception("BUILDER2_WINNER_RESUME_INSPECT_FAILED jobId=%s", job_id or "(none)")
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "builder2_winner_resume_inspect_unhandled_exception",
                    "jobId": job_id or None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
