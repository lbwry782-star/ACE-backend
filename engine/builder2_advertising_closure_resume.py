"""
Builder2 Advertising Closure resume — proposal-only and render-only commands.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional

from engine.builder2_advertising_closure_contract import (
    get_advertising_closure_status,
    normalize_advertising_closure,
    set_advertising_closure_status,
    validate_advertising_closure_object,
)
from engine.builder2_advertising_closure_pipeline import render_advertising_closure_for_state
from engine.builder2_advertising_closure_proposal import generate_advertising_closure_proposal
from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.video_jobs_redis import redis_configured, video_job_get, video_job_mark_done

logger = logging.getLogger(__name__)

DEFAULT_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _truthy(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _initial_report(*, job_id: str) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "winnerLoaded": False,
        "winnerCandidateId": None,
        "winnerPrototypeId": None,
        "advertisingClosureStatus": "missing",
        "proposalOnly": False,
        "renderOnly": False,
        "approved": False,
        "advertisingClosureCalls": 0,
        "advertisingClosureRepairCalls": 0,
        "advertisingClosureRetryCalls": 0,
        "closureFfmpegCalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "totalReasoningCalls": 0,
        "mediaReused": False,
        "resolvedVideoUrl": None,
        "proposedProductNameText": None,
        "proposedSloganText": None,
        "ok": False,
    }


def approve_persisted_proposal(state: Dict[str, Any]) -> None:
    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict):
        raise RuntimeError("builder2_advertising_closure_proposal_missing")
    set_advertising_closure_status(state, "approved")
    state["advertisingClosure"] = normalize_advertising_closure(closure)


def run_one_advertising_closure_resume(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
    proposal_only: Optional[bool] = None,
    render_only: Optional[bool] = None,
    approve: Optional[bool] = None,
    llm_client: Optional[Callable[..., Any]] = None,
    render_endcard: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    jid = _env("BUILDER2_ADVERTISING_CLOSURE_JOB_ID", job_id or DEFAULT_JOB_ID)
    proposal = _truthy("BUILDER2_ADVERTISING_CLOSURE_PROPOSAL_ONLY") if proposal_only is None else proposal_only
    render = _truthy("BUILDER2_ADVERTISING_CLOSURE_RENDER") if render_only is None else render_only
    approved = _truthy("BUILDER2_ADVERTISING_CLOSURE_APPROVE") if approve is None else approve
    if proposal and render:
        raise RuntimeError("builder2_advertising_closure_mode_conflict")
    report = _initial_report(job_id=jid)
    report["proposalOnly"] = proposal
    report["renderOnly"] = render
    report["approved"] = approved
    AdvertisingClosureResumeGuard.begin(proposal_mode=proposal, render_mode=render)
    try:
        state = tournament_state if tournament_state is not None else load_tournament_state(jid)
        if state is None:
            report["failureReason"] = "builder2_advertising_closure_job_not_found"
            return report
        if not is_valid_persisted_winner_development(state):
            report["failureReason"] = "builder2_advertising_closure_missing_winner"
            return report
        plan = state.get("winnerDevelopmentPlan") or {}
        report["winnerLoaded"] = True
        report["winnerCandidateId"] = state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId")
        report["winnerPrototypeId"] = state.get("winnerDevelopmentPrototypeId")
        report["advertisingClosureStatus"] = get_advertising_closure_status(state)

        if approved and not proposal and not render:
            approve_persisted_proposal(state)
            save_tournament_state(jid, state)
            report["advertisingClosureStatus"] = "approved"
            report["ok"] = True
            return report

        if proposal:
            existing = state.get("advertisingClosure")
            if isinstance(existing, dict) and str(existing.get("sloganText") or "").strip():
                proposal_obj = normalize_advertising_closure(existing)
                report["mediaReused"] = True
            else:
                proposal_obj = generate_advertising_closure_proposal(plan, llm_client=llm_client)
                state["advertisingClosure"] = proposal_obj
                set_advertising_closure_status(state, "proposed")
                save_tournament_state(jid, state)
            report["proposedProductNameText"] = proposal_obj.get("productNameText")
            report["proposedSloganText"] = proposal_obj.get("sloganText")
            report["advertisingClosureStatus"] = get_advertising_closure_status(state)
            report["ok"] = True
            return report

        if render:
            status = get_advertising_closure_status(state)
            if status not in {"approved", "completed"}:
                report["failureReason"] = "builder2_advertising_closure_not_approved"
                return report
            closure = state.get("advertisingClosure")
            if not isinstance(closure, dict):
                report["failureReason"] = "builder2_advertising_closure_proposal_missing"
                return report
            validate_advertising_closure_object(closure, plan=plan)
            media = state.setdefault("mediaResume", {})
            public_base_url = str(media.get("publicBaseUrl") or "")
            if not public_base_url:
                job_data = video_job_get(jid) if redis_configured() else None
                public_base_url = str((job_data or {}).get("publicBaseUrl") or _env("ACE_PUBLIC_BASE_URL"))
            from engine.video_endcard_postprocess import append_advertising_closure_endcard

            renderer = render_endcard or append_advertising_closure_endcard
            AdvertisingClosureResumeGuard.enable_closure_ffmpeg()
            updated, counters = render_advertising_closure_for_state(
                job_id=jid,
                state=state,
                plan=plan,
                closure=closure,
                public_base_url=public_base_url,
                render_endcard=renderer,
            )
            state.update(updated)
            save_tournament_state(jid, state)
            final_url = str((state.get("mediaResume") or {}).get("finalVideoWithClosureUrl") or "")
            if redis_configured() and final_url:
                marketing_text = str((state.get("mediaResume") or {}).get("marketingText") or "")
                video_job_mark_done(jid, final_url, marketing_text, overlay_headline="")
            report["closureFfmpegCalls"] = counters.closure_ffmpeg_calls
            report["mediaReused"] = counters.media_reused
            report["resolvedVideoUrl"] = final_url or None
            report["advertisingClosureStatus"] = get_advertising_closure_status(state)
            report["ok"] = bool(final_url)
            return report

        report["failureReason"] = "builder2_advertising_closure_mode_unspecified"
        return report
    finally:
        report.update(AdvertisingClosureResumeGuard.reasoning_report())
        AdvertisingClosureResumeGuard.end()
    return report


def print_advertising_closure_report(report: Dict[str, Any]) -> None:
    safe_keys = (
        "jobId",
        "winnerLoaded",
        "winnerCandidateId",
        "winnerPrototypeId",
        "advertisingClosureStatus",
        "proposalOnly",
        "renderOnly",
        "approved",
        "advertisingClosureCalls",
        "advertisingClosureRepairCalls",
        "advertisingClosureRetryCalls",
        "closureFfmpegCalls",
        "runwayCalls",
        "imageCalls",
        "totalReasoningCalls",
        "mediaReused",
        "resolvedVideoUrl",
        "proposedProductNameText",
        "proposedSloganText",
        "failureReason",
        "ok",
    )
    print(json.dumps({key: report.get(key) for key in safe_keys}, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    report = run_one_advertising_closure_resume(job_id=_env("BUILDER2_ADVERTISING_CLOSURE_JOB_ID", DEFAULT_JOB_ID))
    print_advertising_closure_report(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
