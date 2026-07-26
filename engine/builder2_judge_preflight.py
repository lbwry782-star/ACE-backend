"""
Builder2 Judge preflight — isolated one-shot Judge contract verification.

Run: python -m engine.builder2_judge_preflight

Environment:
  BUILDER2_JUDGE_PREFLIGHT_JOB_ID
  BUILDER2_JUDGE_PREFLIGHT_CANDIDATE_ID
  BUILDER2_JUDGE_PREFLIGHT_ALLOW_ALTERNATE (optional, default false)

Does not use Strategy, Creator, Winner Development, Runway, FFmpeg, recovery, or the video queue.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from engine.builder2_accepted_creator_store import (
    find_any_persisted_accepted_candidate,
    list_accepted_creator_candidate_ids,
    load_accepted_creator_candidate,
)
from engine.builder2_judge import judge_candidate
from engine.builder2_judge_preflight_guard import JudgePreflightIsolationGuard
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import load_tournament_state

logger = logging.getLogger(__name__)

DEFAULT_PREFLIGHT_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"
DEFAULT_PREFLIGHT_CANDIDATE_ID = "cand-1-summer_fan-1-57f415ca"


def _preflight_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _preflight_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _empty_metrics() -> Dict[str, Any]:
    return {
        "strategyCalls": 0,
        "creatorCalls": 0,
        "judgeCalls": 0,
        "judgeRepairCalls": 0,
        "judgeRetryCalls": 0,
        "winnerDevelopmentCalls": 0,
    }


def run_one_isolated_judge_preflight(
    *,
    job_id: str,
    candidate_id: Optional[str] = None,
    allow_alternate: bool = False,
    llm_client: Optional[Any] = None,
    preflight_id: Optional[str] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    JudgePreflightIsolationGuard.begin()
    preflight_id = preflight_id or f"judge-preflight-{uuid.uuid4().hex[:12]}"
    requested_candidate_id = (
        candidate_id or _preflight_env("BUILDER2_JUDGE_PREFLIGHT_CANDIDATE_ID", DEFAULT_PREFLIGHT_CANDIDATE_ID)
    ).strip()
    if allow_alternate is False:
        allow_alternate = _preflight_bool("BUILDER2_JUDGE_PREFLIGHT_ALLOW_ALTERNATE", False)

    report: Dict[str, Any] = {
        "preflightId": preflight_id,
        "jobId": job_id,
        "requestedCandidateId": requested_candidate_id or None,
        "resolvedCandidateId": None,
        "candidateSource": "missing",
        "judgeAccepted": False,
        "judgeNormalCalls": 0,
        "judgeRepairCalls": 0,
        "judgeRetryCalls": 0,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "winnerCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "validationFailurePaths": [],
        "topLevelKeys": [],
        "verbalLayerAssessmentType": None,
        "verbalLayerChildKeys": [],
        "verbalBooleanValues": {},
        "eligible": None,
        "scoreFieldCount": 0,
        "persistedCandidateIds": [],
    }

    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_judge_preflight_job_not_found"
        JudgePreflightIsolationGuard.end()
        report["ok"] = False
        return report

    report["persistedCandidateIds"] = list_accepted_creator_candidate_ids(job_id=job_id, tournament_state=state)

    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        report["failureReason"] = "builder2_judge_preflight_strategy_missing"
        JudgePreflightIsolationGuard.end()
        report["ok"] = False
        return report

    snapshot: Optional[Dict[str, Any]] = None
    if requested_candidate_id:
        try:
            snapshot = load_accepted_creator_candidate(
                job_id=job_id,
                candidate_id=requested_candidate_id,
                tournament_state=state,
            )
            report["resolvedCandidateId"] = requested_candidate_id
            report["candidateSource"] = "requested_persisted_candidate"
        except Builder2TournamentError:
            snapshot = None

    if snapshot is None and allow_alternate:
        snapshot = find_any_persisted_accepted_candidate(job_id=job_id, tournament_state=state)
        if snapshot is not None:
            report["resolvedCandidateId"] = snapshot.get("candidateId")
            report["candidateSource"] = "alternate_persisted_accepted_candidate"

    if snapshot is None:
        report["failureReason"] = "builder2_judge_preflight_candidate_not_persisted"
        JudgePreflightIsolationGuard.end()
        report["ok"] = False
        return report

    resolved_candidate_id = str(snapshot.get("candidateId") or "")
    candidate = snapshot.get("creatorOutput")
    prototype_id = str(snapshot.get("prototypeId") or "")
    if not isinstance(candidate, dict) or not prototype_id or not resolved_candidate_id:
        report["failureReason"] = "builder2_judge_preflight_candidate_invalid"
        JudgePreflightIsolationGuard.end()
        report["ok"] = False
        return report

    report["resolvedCandidateId"] = resolved_candidate_id
    report["prototypeId"] = prototype_id

    product_name = str(state.get("productName") or state.get("productNameResolved") or "Preflight Product")
    product_description = str(state.get("productDescription") or "Preflight description")
    language = str(state.get("contentLanguage") or state.get("language") or "he")

    local_state: Dict[str, Any] = {
        "jobId": preflight_id,
        "tournamentId": f"judge-preflight-{preflight_id}",
        "preflightMode": True,
        "preflightLocalOnly": True,
        "metrics": _empty_metrics(),
    }
    ensure_metrics(local_state)

    try:
        JudgePreflightIsolationGuard.assert_safe_before_paid_call()
        require_prototype(prototype_id)
        judgment_id, judgment, total, scores = judge_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype_id=prototype_id,
            candidate_id=resolved_candidate_id,
            candidate=candidate,
            llm_client=llm_client,
            state=local_state,
            judgment_id=f"judge-preflight-{resolved_candidate_id}",
        )
        report["judgeAccepted"] = True
        report["judgmentId"] = judgment_id
        report["eligible"] = bool(judgment.get("eligible"))
        report["totalScore"] = total
        report["scoreFieldCount"] = len(scores)
        report["topLevelKeys"] = sorted(judgment.keys())
        verbal = judgment.get("verbalLayerAssessment")
        report["verbalLayerAssessmentType"] = type(verbal).__name__
        if isinstance(verbal, dict):
            report["verbalLayerChildKeys"] = sorted(verbal.keys())
            report["verbalBooleanValues"] = {
                key: verbal.get(key)
                for key in (
                    "keywordBornFromVisual",
                    "visualMeaningIsClear",
                    "strategicMeaningIsClear",
                    "twoMeaningsReinforceEachOther",
                )
                if key in verbal
            }
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_judge_preflight_failed")
        if ":" in reason:
            report["validationFailurePaths"] = [reason.split(":", 1)[-1]]
        report["failureReason"] = reason
        diagnostics = (local_state.get("judgeDiagnosticsByCandidate") or {}).get(resolved_candidate_id, {})
        report["topLevelKeys"] = diagnostics.get("topLevelKeys") or []
    finally:
        metrics = local_state.get("metrics") or {}
        report["judgeNormalCalls"] = int(metrics.get("judgeCalls") or 0)
        report["judgeRepairCalls"] = int(metrics.get("judgeRepairCalls") or 0)
        report["judgeRetryCalls"] = int(metrics.get("judgeRetryCalls") or 0)
        report["strategyCalls"] = int(metrics.get("strategyCalls") or 0)
        report["creatorCalls"] = int(metrics.get("creatorCalls") or 0)
        report["winnerCalls"] = int(metrics.get("winnerDevelopmentCalls") or 0)
        report["runwayCalls"] = 0
        report["ffmpegCalls"] = 0
        JudgePreflightIsolationGuard.end()

    report["ok"] = bool(report.get("judgeAccepted"))
    return report


def print_preflight_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "preflightId",
            "jobId",
            "requestedCandidateId",
            "resolvedCandidateId",
            "candidateSource",
            "prototypeId",
            "persistedCandidateIds",
            "judgeAccepted",
            "judgeNormalCalls",
            "judgeRepairCalls",
            "judgeRetryCalls",
            "strategyCalls",
            "creatorCalls",
            "winnerCalls",
            "runwayCalls",
            "ffmpegCalls",
            "validationFailurePaths",
            "topLevelKeys",
            "verbalLayerAssessmentType",
            "verbalLayerChildKeys",
            "verbalBooleanValues",
            "eligible",
            "scoreFieldCount",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _preflight_env("BUILDER2_JUDGE_PREFLIGHT_JOB_ID", DEFAULT_PREFLIGHT_JOB_ID)
    candidate_id = _preflight_env("BUILDER2_JUDGE_PREFLIGHT_CANDIDATE_ID", DEFAULT_PREFLIGHT_CANDIDATE_ID) or None

    logger.info(
        "BUILDER2_JUDGE_PREFLIGHT_START jobId=%s requestedCandidateId=%s",
        job_id,
        candidate_id,
    )
    report = run_one_isolated_judge_preflight(job_id=job_id, candidate_id=candidate_id)
    print_preflight_report(report)
    logger.info(
        "BUILDER2_JUDGE_PREFLIGHT_DONE preflightId=%s ok=%s judgeAccepted=%s candidateSource=%s resolvedCandidateId=%s",
        report.get("preflightId"),
        report.get("ok"),
        report.get("judgeAccepted"),
        report.get("candidateSource"),
        report.get("resolvedCandidateId"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
