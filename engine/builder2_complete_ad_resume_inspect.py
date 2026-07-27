"""
Builder2 complete-ad resume inspector — read-only zero paid calls.

Run:
  BUILDER2_COMPLETE_AD_RESUME_INSPECT_JOB_ID=<jobId> python -m engine.builder2_complete_ad_resume_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    can_offline_revalidate_rejected_creator,
    find_rejected_creator_for_prototype,
    load_rejected_creator_parsed_response,
)
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_completion_gate import (
    assigned_prototype_ids,
    missing_creator_prototype_ids,
    missing_judge_prototype_ids,
    tournament_resolution_summary,
)
from engine.builder2_tournament_store import load_tournament_state
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY, load_revalidatable_parsed_winner_response
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _expected_next_roles(state: Dict[str, Any]) -> List[str]:
    summary = tournament_resolution_summary(state)
    roles: List[str] = []
    rejected = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY) if isinstance(state.get(REJECTED_CREATOR_PARSED_INDEX_KEY), dict) else {}
    for prototype_id in summary.get("missingCreatorPrototypeIds") or []:
        payload = find_rejected_creator_for_prototype(state, prototype_id)
        if payload and can_offline_revalidate_rejected_creator(state, candidate_id=_clean(payload.get("candidateId")))[0]:
            roles.append("offline_creator_revalidation")
        else:
            roles.append("builder2_creator")
    for prototype_id in summary.get("missingJudgePrototypeIds") or []:
        if prototype_id not in (summary.get("missingCreatorPrototypeIds") or []):
            roles.append("builder2_judge")
    if summary.get("readyForAuthoritativeWinnerSelection") and not _clean(state.get("winnerCandidateId")):
        roles.append("winner_selection")
    if _clean(state.get("winnerCandidateId")) and not state.get("winnerDevelopmentPlan"):
        roles.append("builder2_winner")
    return roles


def inspect_builder2_complete_ad_resume(job_id: str = "", *, raw_job_reader: Optional[Any] = None) -> Dict[str, Any]:
    read_raw = raw_job_reader or video_job_get_raw
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "ok": False,
        "strategyReusable": False,
        "acceptedCreatorCount": 0,
        "acceptedJudgmentCount": 0,
        "missingPrototypeIds": [],
        "rejectedCreatorResponseAvailable": False,
        "rejectedCreatorOfflineRevalidatable": False,
        "provisionalWinnerPresent": False,
        "provisionalWinnerCandidateId": None,
        "finalWinnerReady": False,
        "parsedWinnerResponseAvailable": False,
        "parsedWinnerCandidateId": None,
        "resolvedResumeStage": None,
        "expectedNextReasoningRoles": [],
        "minimumAdditionalReasoningCalls": 0,
        "maximumAdditionalReasoningCalls": 0,
        "imageCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "redisMutations": 0,
        "openAICalls": 0,
        "builder2ResumeContractVersion": BUILDER2_RESUME_CONTRACT_VERSION,
        "builder2NewFormatVersion": BUILDER2_NEW_FORMAT_VERSION,
    }
    if not jid:
        report["failureReason"] = "builder2_complete_ad_resume_inspect_job_id_missing"
        return report
    if not redis_configured():
        report["failureReason"] = "builder2_complete_ad_resume_inspect_redis_unconfigured"
        return report

    state = load_tournament_state(jid)
    if state is None:
        report["failureReason"] = "builder2_complete_ad_resume_inspect_job_not_found"
        return report

    job_raw = read_raw(jid) or {}
    resolver = resolve_builder2_resume_stage(job_raw, state)
    summary = tournament_resolution_summary(state)
    report["strategyReusable"] = isinstance(state.get("strategyFoundation"), dict) and bool(state.get("strategyFoundation"))
    report["acceptedCreatorCount"] = summary["acceptedCreatorCount"]
    report["acceptedJudgmentCount"] = summary["acceptedJudgmentCount"]
    report["missingPrototypeIds"] = sorted(
        set(summary.get("missingCreatorPrototypeIds") or []) | set(summary.get("missingJudgePrototypeIds") or [])
    )
    report["resolvedResumeStage"] = resolver.get("resumeFromStage")
    report["provisionalWinnerPresent"] = bool(_clean(state.get("provisionalWinnerCandidateId")))
    report["provisionalWinnerCandidateId"] = _clean(state.get("provisionalWinnerCandidateId")) or None
    report["finalWinnerReady"] = bool(summary.get("readyForAuthoritativeWinnerSelection")) and bool(
        _clean(state.get("winnerCandidateId"))
    )

    rejected_index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if isinstance(rejected_index, dict) and rejected_index:
        report["rejectedCreatorResponseAvailable"] = True
        for candidate_id in rejected_index:
            ok, _reason = can_offline_revalidate_rejected_creator(state, candidate_id=str(candidate_id))
            if ok:
                report["rejectedCreatorOfflineRevalidatable"] = True
                break

    parsed_winner = load_revalidatable_parsed_winner_response(state)
    if parsed_winner:
        report["parsedWinnerResponseAvailable"] = True
        report["parsedWinnerCandidateId"] = _clean(parsed_winner.get("candidateId")) or None

    roles = _expected_next_roles(state)
    report["expectedNextReasoningRoles"] = roles
    report["minimumAdditionalReasoningCalls"] = len([role for role in roles if role != "offline_creator_revalidation"])
    report["maximumAdditionalReasoningCalls"] = report["minimumAdditionalReasoningCalls"]
    if any(role == "builder2_creator" for role in roles):
        report["maximumAdditionalReasoningCalls"] += len([pid for pid in missing_creator_prototype_ids(state)])
    report["ok"] = True
    return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = (os.environ.get("BUILDER2_COMPLETE_AD_RESUME_INSPECT_JOB_ID") or "").strip()
    report = inspect_builder2_complete_ad_resume(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
