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
from engine.builder2_complete_ad_resume_plan import (
    evaluate_complete_ad_reasoning_executor_preconditions,
    plan_complete_ad_reasoning_roles,
    resolve_complete_ad_canonical_resume_plan,
    resolve_complete_ad_resume_stage,
)
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_completion_gate import tournament_resolution_summary
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
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
        "reasoningComplete": False,
        "mediaStarted": False,
        "finalWinnerCandidateId": None,
        "finalWinnerPrototypeId": None,
        "finalWinnerScore": None,
        "semanticAlignmentAccepted": False,
        "prototypeFitScore": None,
        "advertisingClosurePresent": False,
        "productNamePresent": False,
        "sloganPresent": False,
        "sloganWordCount": 0,
        "winnerDevelopmentAccepted": False,
        "winnerDevelopmentReused": False,
        "resolvedNextStage": None,
        "startImageCalls": 0,
        "runwaySubmissionCount": 0,
        "finalVideoAvailable": False,
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
        canonical_plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True, job_raw=job_raw)
        _executor_ok, executor_reason, _executor_plan = evaluate_complete_ad_reasoning_executor_preconditions(
            state,
            job_raw,
        )

        report["strategyReusable"] = isinstance(state.get("strategyFoundation"), dict) and bool(state.get("strategyFoundation"))
        report["acceptedCreatorCount"] = summary["acceptedCreatorCount"]
        report["acceptedJudgmentCount"] = summary["acceptedJudgmentCount"]
        report["missingCreatorPrototypeIds"] = list(canonical_plan.get("missingCreatorPrototypeIds") or [])
        report["missingJudgmentPrototypeIds"] = list(canonical_plan.get("missingJudgmentPrototypeIds") or [])
        report["incompletePrototypeIds"] = list(canonical_plan.get("incompletePrototypeIds") or [])
        report["missingPrototypeIds"] = list(canonical_plan.get("incompletePrototypeIds") or [])
        report["resumePlanByPrototype"] = dict(canonical_plan.get("resumePlanByPrototype") or {})
        report["resolvedResumeStage"] = canonical_plan.get("resolvedResumeStage") or resolved_stage or resolver.get("resumeFromStage")
        report["resumeEligible"] = bool(canonical_plan.get("resumeEligible"))
        report["executorWouldAcceptState"] = bool(canonical_plan.get("executorWouldAcceptState"))
        report["executorRejectionReason"] = canonical_plan.get("executorRejectionReason")
        report["judgeCallsPlanned"] = int(canonical_plan.get("judgeCallsPlanned") or 0)
        report["creatorCallsPlanned"] = int(canonical_plan.get("creatorCallsPlanned") or 0)
        report["remainingCreatorNormalCalls"] = int(canonical_plan.get("remainingCreatorNormalCalls") or 0)
        report["remainingJudgeNormalCalls"] = int(canonical_plan.get("remainingJudgeNormalCalls") or 0)
        report["normalCallsBeforeWinner"] = int(canonical_plan.get("normalCallsBeforeWinner") or 0)
        report["conditionalWinnerCalls"] = int(canonical_plan.get("conditionalWinnerCalls") or 0)
        report["winnerNormalCallConditional"] = bool(canonical_plan.get("winnerNormalCallConditional"))
        report["possibleRepairCallsNotIncluded"] = bool(canonical_plan.get("possibleRepairCallsNotIncluded"))
        report["minimumAdditionalNormalReasoningCalls"] = int(canonical_plan.get("minimumAdditionalNormalReasoningCalls") or 0)
        report["maximumAdditionalNormalReasoningCalls"] = int(canonical_plan.get("maximumAdditionalNormalReasoningCalls") or 0)
        report["minimumAdditionalReasoningCallsWithoutRepairs"] = int(
            canonical_plan.get("minimumAdditionalReasoningCallsWithoutRepairs") or 0
        )
        report["maximumAdditionalReasoningCallsWithoutRepairs"] = int(
            canonical_plan.get("maximumAdditionalReasoningCallsWithoutRepairs") or 0
        )
        report["strategyWouldDispatch"] = bool(canonical_plan.get("strategyWouldDispatch"))
        report["strategyFingerprint"] = canonical_plan.get("strategyFingerprint")
        report["creatorsWouldDispatch"] = bool(canonical_plan.get("creatorsWouldDispatch"))
        report["winnerWouldDispatch"] = bool(canonical_plan.get("winnerWouldDispatch"))
        report["mediaWouldDispatch"] = bool(canonical_plan.get("mediaWouldDispatch"))
        report["jobStatus"] = canonical_plan.get("jobStatus")
        report["pauseReason"] = canonical_plan.get("pauseReason")
        report["progressStage"] = canonical_plan.get("progressStage")
        report["tournamentId"] = canonical_plan.get("tournamentId")
        report["readyForWinnerDevelopment"] = bool(canonical_plan.get("readyForWinnerDevelopment"))
        report["paidCalls"] = 0
        report["stateMutated"] = False
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

        report["reasoningComplete"] = bool(state.get("reasoningComplete"))
        report["mediaStarted"] = bool(state.get("mediaStarted"))
        report["resolvedNextStage"] = resolve_complete_ad_resume_stage(state, read_only=True)
        winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId")) or None
        report["finalWinnerCandidateId"] = winner_id
        winner_rec = (state.get("candidates") or {}).get(winner_id or "") or {}
        report["finalWinnerPrototypeId"] = _clean(winner_rec.get("prototypeId")) or None
        report["finalWinnerScore"] = winner_rec.get("totalScore")
        judgment_rec = (state.get("judgments") or {}).get(winner_rec.get("judgmentId") or "")
        winning_judgment = (judgment_rec or {}).get("judgment") or {}
        semantic = winning_judgment.get("semanticAlignmentAssessment") or {}
        report["semanticAlignmentAccepted"] = bool(semantic.get("semanticAlignment"))
        scores = winning_judgment.get("scores") if isinstance(winning_judgment.get("scores"), dict) else {}
        report["prototypeFitScore"] = scores.get("prototypeMethodApplication")
        closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
        if not closure and isinstance(winner_rec.get("creatorOutput"), dict):
            closure = (winner_rec["creatorOutput"].get("advertisingClosure") or {})
        report["advertisingClosurePresent"] = bool(closure.get("sloganText"))
        report["productNamePresent"] = bool(closure.get("productNameText"))
        slogan = _clean(closure.get("sloganText"))
        report["sloganPresent"] = bool(slogan)
        report["sloganWordCount"] = len(slogan.split()) if slogan else 0
        report["winnerDevelopmentAccepted"] = is_valid_persisted_winner_development(state)
        parsed_winner_id = _clean((parsed_winner or {}).get("candidateId"))
        report["winnerDevelopmentReused"] = bool(
            report["winnerDevelopmentAccepted"]
            and parsed_winner_id
            and parsed_winner_id == _clean(winner_id)
        )
        media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
        report["startImageCalls"] = int((state.get("metrics") or {}).get("startImageCalls") or 0)
        report["runwaySubmissionCount"] = int(
            (state.get("metrics") or {}).get("runwaySubmissionCalls")
            or (state.get("metrics") or {}).get("runwayCalls")
            or 0
        )
        report["finalVideoAvailable"] = bool(
            _clean(state.get("finalVideoUrl"))
            or _clean(media.get("finalVideoUrl"))
            or _clean(media.get("runwayVideoUrl"))
        )

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
