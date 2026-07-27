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
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    can_offline_revalidate_rejected_creator,
)
from engine.builder2_complete_ad_resume_plan import plan_complete_ad_reasoning_roles, resolve_complete_ad_resume_stage
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_completion_gate import tournament_resolution_summary
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_preservation_contract import load_revalidatable_parsed_winner_response
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


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
        "requiredNextReasoningRoles": [],
        "conditionalNextReasoningRoles": [],
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

    with read_only_builder2_inspection() as mutation_counter:
        state = _read_raw(jid)
        if state is None:
            report["failureReason"] = "builder2_complete_ad_resume_inspect_job_not_found"
            return report
        state = deepcopy(state)

        job_raw = read_raw(jid) or {}
        resolver = resolve_builder2_resume_stage(job_raw, state, read_only=True)
        summary = tournament_resolution_summary(state, read_only=True)
        role_plan = plan_complete_ad_reasoning_roles(state, read_only=True)
        resolved_stage = resolve_complete_ad_resume_stage(state, read_only=True)

        report["strategyReusable"] = isinstance(state.get("strategyFoundation"), dict) and bool(state.get("strategyFoundation"))
        report["acceptedCreatorCount"] = summary["acceptedCreatorCount"]
        report["acceptedJudgmentCount"] = summary["acceptedJudgmentCount"]
        report["missingPrototypeIds"] = sorted(
            set(summary.get("missingCreatorPrototypeIds") or []) | set(summary.get("missingJudgePrototypeIds") or [])
        )
        report["resolvedResumeStage"] = resolved_stage or resolver.get("resumeFromStage")
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

        report["requiredNextReasoningRoles"] = list(role_plan.get("requiredNextReasoningRoles") or [])
        report["conditionalNextReasoningRoles"] = list(role_plan.get("conditionalNextReasoningRoles") or [])
        report["expectedNextReasoningRoles"] = list(role_plan.get("expectedNextReasoningRoles") or [])
        report["minimumAdditionalReasoningCalls"] = int(role_plan.get("minimumAdditionalReasoningCalls") or 0)
        report["maximumAdditionalReasoningCalls"] = int(role_plan.get("maximumAdditionalReasoningCalls") or 0)
        report["redisMutations"] = mutation_counter.redis_mutations
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
