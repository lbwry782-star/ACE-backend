"""
Builder2 media-only resume — continue from persisted Winner Development plan.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_headline_decision_contract import get_normalized_headline_decision, headline_decision_requires_headline
from engine.builder2_media_pipeline import MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_resume_guard import MEDIA_RESUME_ISOLATION_ERROR, MediaResumeIsolationGuard
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_runway_config import (
    BUILDER2_RUNWAY_VIDEO_RATIO,
    builder2_runway_requires_start_image,
    resolve_builder2_runway_video_model,
    resolve_builder2_video_duration_seconds,
)
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION, Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.builder2_winner_downstream import Builder2WinnerDownstreamError, normalize_builder2_winner_downstream
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY
from engine.video_jobs_redis import redis_configured, video_job_get, video_job_mark_done, video_job_mark_error

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_RESUME_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _truthy(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _initial_report(*, job_id: str, dry_run: bool = False) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "winnerLoaded": False,
        "winnerCandidateId": None,
        "winnerPrototypeId": None,
        "winnerReused": True,
        "headlineDecision": None,
        "downstreamValidationAccepted": False,
        "startImageRequired": False,
        "runwayRequired": True,
        "runwayModel": None,
        "durationSeconds": None,
        "strategyCalls": 0,
        "creatorCalls": 0,
        "judgeCalls": 0,
        "winnerCalls": 0,
        "startImageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingResumed": False,
        "ffmpegCalls": 0,
        "dryRun": dry_run,
        "readyForMediaResume": False,
        "mediaReused": False,
        "finalVideoAvailable": False,
        "jobCompleted": False,
        "failureStage": None,
        "failureReason": None,
        "ok": False,
    }


def collect_media_resume_missing_paths(state: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not is_valid_persisted_winner_development(state):
        missing.append("winnerDevelopmentPlan")
    if not state.get("mediaContinuationRequired"):
        missing.append("mediaContinuationRequired")
    candidate_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or "").strip()
    if not candidate_id:
        missing.append("winnerDevelopmentCandidateId")
    prototype_id = str(state.get("winnerDevelopmentPrototypeId") or "").strip()
    if not prototype_id:
        missing.append("winnerDevelopmentPrototypeId")
    plan = state.get("winnerDevelopmentPlan")
    if isinstance(plan, dict):
        if plan.get("schemaVersion") != WINNER_PLAN_SCHEMA_VERSION:
            missing.append("winnerDevelopmentPlan.schemaVersion")
        if plan.get("methodologyVersion") and plan.get("methodologyVersion") != METHODOLOGY_VERSION:
            missing.append("winnerDevelopmentPlan.methodologyVersion")
        if SERVER_OWNED_WINNER_SOURCE_KEY not in plan:
            missing.append(f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}")
        if str(plan.get("prototypeId") or "") != prototype_id and prototype_id:
            missing.append("winnerDevelopmentPlan.prototypeId")
    if candidate_id and prototype_id:
        cand = (state.get("candidates") or {}).get(candidate_id) or {}
        if str(cand.get("prototypeId") or "") != prototype_id:
            missing.append("candidates.prototypeId")
    return missing


def verify_media_configuration(*, start_image_required: bool, ffmpeg_required: bool) -> Dict[str, Any]:
    runway_key = bool(_env("RUNWAY_API_KEY"))
    openai_key = bool(_env("OPENAI_API_KEY"))
    return {
        "runwayApiKeyConfigured": runway_key,
        "openaiApiKeyConfigured": openai_key if start_image_required else True,
        "redisConfigured": redis_configured(),
        "publicBaseUrlConfigured": False,
        "storageConfigured": redis_configured(),
    }


def _resolve_public_base_url(job_id: str) -> str:
    if not redis_configured():
        return _env("PUBLIC_BASE_URL")
    job = video_job_get(job_id) or {}
    return str(job.get("public_base_url") or _env("PUBLIC_BASE_URL") or "").strip()


def _load_and_normalize_winner(
    *,
    job_id: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict):
        raise Builder2TournamentError("builder2_media_resume_missing:winnerDevelopmentPlan")
    product_name = str(state.get("productName") or state.get("productNameResolved") or plan.get("productNameResolved") or "")
    product_description = str(state.get("productDescription") or "")
    language = str(state.get("contentLanguage") or state.get("language") or plan.get("language") or "en")
    normalized = dict(plan)
    normalized.setdefault("productNameResolved", product_name)
    normalized.setdefault("language", language)
    normalized["planInferenceMode"] = normalized.get("planInferenceMode") or "builder2_tournament_winner_v1"
    return normalize_builder2_winner_downstream(
        normalized,
        job_id=job_id,
        tournament_id=str(state.get("tournamentId") or ""),
        compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
    )


def run_one_media_resume(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
    dry_run: Optional[bool] = None,
    pipeline_deps: Optional[MediaPipelineDeps] = None,
) -> Dict[str, Any]:
    dry = _truthy("BUILDER2_MEDIA_RESUME_DRY_RUN") if dry_run is None else dry_run
    report = _initial_report(job_id=job_id, dry_run=dry)
    MediaResumeIsolationGuard.begin()
    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureReason"] = "builder2_media_resume_job_not_found"
        MediaResumeIsolationGuard.end()
        return report

    media = state.get("mediaResume")
    if isinstance(media, dict) and media.get("mediaResumeStatus") == "completed" and media.get("finalPublicUrl"):
        candidate_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or "").strip()
        prototype_id = str(state.get("winnerDevelopmentPrototypeId") or "").strip()
        report["winnerCandidateId"] = candidate_id or None
        report["winnerPrototypeId"] = prototype_id or None
        report["winnerLoaded"] = is_valid_persisted_winner_development(state)
        report["headlineDecision"] = get_normalized_headline_decision(state.get("winnerDevelopmentPlan") or {})
        report["downstreamValidationAccepted"] = True
        report["mediaReused"] = True
        report["finalVideoAvailable"] = True
        report["jobCompleted"] = True
        report["readyForMediaResume"] = True
        report["ok"] = True
        MediaResumeIsolationGuard.end()
        return report

    missing = collect_media_resume_missing_paths(state)
    if missing:
        report["failureReason"] = f"builder2_media_resume_missing:{','.join(missing)}"
        report["missingPaths"] = missing
        MediaResumeIsolationGuard.end()
        return report

    candidate_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or "").strip()
    prototype_id = str(state.get("winnerDevelopmentPrototypeId") or "").strip()
    report["winnerCandidateId"] = candidate_id
    report["winnerPrototypeId"] = prototype_id
    report["winnerLoaded"] = True

    try:
        MediaResumeIsolationGuard.assert_reasoning_isolated()
        plan = _load_and_normalize_winner(job_id=job_id, state=state)
        report["downstreamValidationAccepted"] = True
        headline_decision = get_normalized_headline_decision(plan)
        report["headlineDecision"] = headline_decision
        runway_model = resolve_builder2_runway_video_model()
        duration_seconds = resolve_builder2_video_duration_seconds()
        start_image_required = builder2_runway_requires_start_image(runway_model)
        ffmpeg_required = headline_decision_requires_headline(headline_decision)
        report["startImageRequired"] = start_image_required
        report["runwayRequired"] = True
        report["runwayModel"] = runway_model
        report["durationSeconds"] = duration_seconds

        config = verify_media_configuration(start_image_required=start_image_required, ffmpeg_required=ffmpeg_required)
        config["publicBaseUrlConfigured"] = bool(_resolve_public_base_url(job_id))
        if not config.get("runwayApiKeyConfigured"):
            raise Builder2TournamentError("builder2_media_resume_not_configured:runwayApiKey")
        if start_image_required and not config.get("openaiApiKeyConfigured"):
            raise Builder2TournamentError("builder2_media_resume_not_configured:openaiApiKey")
        if not dry and not config.get("publicBaseUrlConfigured"):
            raise Builder2TournamentError("builder2_media_resume_not_configured:publicBaseUrl")

        if dry:
            execute_builder2_media_pipeline(
                job_id=job_id,
                state=state,
                plan=plan,
                public_base_url=_resolve_public_base_url(job_id),
                product_description=str(state.get("productDescription") or ""),
                dry_run=True,
            )
            report["readyForMediaResume"] = True
            report["ok"] = True
            MediaResumeIsolationGuard.end()
            return report

        state["status"] = "media_continuing"
        if redis_configured():
            job = video_job_get(job_id)
            if job and job.get("status") in {"error", "interrupted", "failed"}:
                pass
        save_tournament_state(job_id, state)

        public_base_url = _resolve_public_base_url(job_id)
        if start_image_required:
            MediaResumeIsolationGuard.enable_start_image()
            MediaResumeIsolationGuard.assert_safe_before_start_image()
        MediaResumeIsolationGuard.enable_runway()
        if ffmpeg_required:
            MediaResumeIsolationGuard.enable_ffmpeg()

        updated_state, counters = execute_builder2_media_pipeline(
            job_id=job_id,
            state=state,
            plan=plan,
            public_base_url=public_base_url,
            product_description=str(state.get("productDescription") or ""),
            dry_run=False,
            deps=pipeline_deps,
        )
        state.update(updated_state)
        report["startImageCalls"] = counters.start_image_calls
        report["runwaySubmissionCalls"] = counters.runway_submission_calls
        report["runwayPollingResumed"] = counters.runway_polling_resumed
        report["ffmpegCalls"] = counters.ffmpeg_calls
        report["mediaReused"] = counters.media_reused

        final_url = str((state.get("mediaResume") or {}).get("finalPublicUrl") or "")
        marketing_text = str((state.get("mediaResume") or {}).get("marketingText") or "")
        overlay_headline = "" if not headline_decision_requires_headline(headline_decision) else (plan.get("headlineText") or "")
        if redis_configured() and final_url:
            video_job_mark_done(job_id, final_url, marketing_text, overlay_headline=str(overlay_headline or ""))
        save_tournament_state(job_id, state)

        report["finalVideoAvailable"] = bool(final_url)
        report["jobCompleted"] = bool(final_url)
        report["ok"] = bool(final_url)
    except Builder2WinnerDownstreamError as exc:
        report["failureStage"] = "downstream_normalization"
        report["failureReason"] = exc.code
        _persist_media_failure(state, stage="downstream_normalization", reason=exc.code)
        if tournament_state is None:
            save_tournament_state(job_id, state)
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else MEDIA_RESUME_ISOLATION_ERROR)
        report["failureReason"] = reason
        if reason.startswith(MEDIA_RESUME_ISOLATION_ERROR):
            report["failureStage"] = "isolation"
        _persist_media_failure(state, stage=report.get("failureStage") or "media", reason=reason)
        if tournament_state is None:
            save_tournament_state(job_id, state)
    except Exception as exc:
        report["failureStage"] = "media"
        report["failureReason"] = str(getattr(exc, "args", [str(exc)])[0])
        _persist_media_failure(
            state,
            stage="media",
            reason=report["failureReason"],
            exception_class=type(exc).__name__,
        )
        if tournament_state is None and redis_configured():
            try:
                video_job_mark_error(job_id, "builder2_media_resume_failed")
            except Exception:
                pass
        if tournament_state is None:
            save_tournament_state(job_id, state)
    finally:
        MediaResumeIsolationGuard.end()

    return report


def _persist_media_failure(
    state: Dict[str, Any],
    *,
    stage: str,
    reason: str,
    exception_class: str = "Builder2TournamentError",
    task_id: Optional[str] = None,
    paid_step_submitted: Optional[bool] = None,
) -> None:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    media["mediaResumeStatus"] = "failed"
    media["mediaFailure"] = {
        "stage": stage,
        "reason": reason,
        "exceptionClass": exception_class,
        "runwayTaskId": task_id or media.get("runwayTaskId"),
        "paidStepSubmitted": paid_step_submitted,
        "reuseExistingWork": True,
    }


def print_media_resume_report(report: Dict[str, Any]) -> None:
    safe_keys = (
        "jobId",
        "winnerLoaded",
        "winnerCandidateId",
        "winnerPrototypeId",
        "winnerReused",
        "headlineDecision",
        "downstreamValidationAccepted",
        "startImageRequired",
        "runwayRequired",
        "runwayModel",
        "durationSeconds",
        "strategyCalls",
        "creatorCalls",
        "judgeCalls",
        "winnerCalls",
        "startImageCalls",
        "runwaySubmissionCalls",
        "runwayPollingResumed",
        "ffmpegCalls",
        "dryRun",
        "readyForMediaResume",
        "mediaReused",
        "finalVideoAvailable",
        "jobCompleted",
        "failureStage",
        "failureReason",
        "missingPaths",
        "ok",
    )
    safe = {key: report.get(key) for key in safe_keys if key in report or report.get(key) is not None}
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_MEDIA_RESUME_JOB_ID", DEFAULT_MEDIA_RESUME_JOB_ID)
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_media_resume_job_id_missing"}, indent=2))
        return 1
    dry = _truthy("BUILDER2_MEDIA_RESUME_DRY_RUN")
    logger.info("BUILDER2_MEDIA_RESUME_START jobId=%s dryRun=%s", job_id, dry)
    report = run_one_media_resume(job_id=job_id, dry_run=dry)
    print_media_resume_report(report)
    logger.info(
        "BUILDER2_MEDIA_RESUME_DONE jobId=%s ok=%s dryRun=%s startImageCalls=%s runwaySubmissionCalls=%s ffmpegCalls=%s",
        job_id,
        report.get("ok"),
        report.get("dryRun"),
        report.get("startImageCalls"),
        report.get("runwaySubmissionCalls"),
        report.get("ffmpegCalls"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
