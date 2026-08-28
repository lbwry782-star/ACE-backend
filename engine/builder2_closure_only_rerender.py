"""
Builder2 closure-only re-render — typography upgrade without reasoning/Runway/image.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.builder2_closure_copy import (
    apply_closure_only_rerender_copy_override,
    closure_only_rerender_force_requested,
    resolve_closure_only_rerender_slogan_override,
    resolve_trusted_closure_copy,
)
from engine.builder2_closure_render import Builder2ClosureRenderError, render_builder2_advertising_closure_endcard
from engine.builder2_closure_rerender_inspect import inspect_builder2_closure_rerender
from engine.builder2_closure_typography import (
    BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    verify_closure_typography_metadata,
)
from engine.builder2_durable_finalization import (
    apply_builder2_durable_publication_fields,
    publish_builder2_durable_final_video,
    require_builder2_web_storage_capability,
)
from engine.builder2_final_video_publication import Builder2FinalPublicationError
from engine.builder2_media_finalization_contract import resolve_raw_runway_artifact_url
from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds
from engine.builder2_closure_duration_contract import enforce_v3_closure_duration_contract
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get, video_job_mark_done

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    return media


def _resolve_public_base_url(state: Dict[str, Any]) -> str:
    explicit = _clean(os.environ.get("BUILDER2_CLOSURE_ONLY_RERENDER_PUBLIC_BASE_URL"))
    if explicit:
        return explicit.rstrip("/")
    job_id = _clean(state.get("jobId"))
    if redis_configured() and job_id:
        job = video_job_get(job_id) or {}
        base = _clean(job.get("publicBaseUrl") or job.get("public_base_url"))
        if base:
            return base.rstrip("/")
    from engine.public_base_url import resolve_public_base_url

    resolution = resolve_public_base_url()
    return resolution.value.rstrip("/") if resolution.configured else ""


def _append_final_output_history(media: Dict[str, Any], *, reason: str) -> None:
    history = media.get("finalOutputHistory")
    if not isinstance(history, list):
        history = []
        media["finalOutputHistory"] = history
    entry = {
        "preservedAt": _utc_now_iso(),
        "reason": reason,
        "finalPublicUrlPresent": bool(_clean(media.get("finalPublicUrl"))),
        "closureTypographyContractVersion": _clean(media.get("closureTypographyContractVersion")),
        "finalVideoToken": _clean(media.get("finalVideoToken") or media.get("finalPublicationOutputToken")),
    }
    history.append(entry)


def run_builder2_closure_only_rerender(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
    expected_typography_version: str = BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    public_base_url: str = "",
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "expectedTypographyVersion": expected_typography_version,
        "closureOnlyRerenderAttempted": False,
        "closureOnlyRerenderAccepted": False,
        "typographyUpgradeApplied": False,
        "rawRunwayReused": False,
        "previousFinalPreserved": False,
        "newFinalPromoted": False,
        "runwaySubmissionCalls": 0,
        "startImageCalls": 0,
        "openAICalls": 0,
        "reasoningCalls": 0,
        "ffmpegCalls": 0,
        "publicationCalls": 0,
        "stateMutated": False,
        "failureStage": None,
        "failureReason": None,
    }
    state = deepcopy(tournament_state) if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        report["failureStage"] = "startup"
        report["failureReason"] = "builder2_closure_rerender_job_not_found"
        return report

    media = _media_bucket(state)
    force = closure_only_rerender_force_requested()
    if (
        not force
        and _clean(media.get("closureOnlyRerenderCompletedForVersion")) == expected_typography_version
    ):
        report["ok"] = True
        report["closureOnlyRerenderAccepted"] = True
        report["idempotentReuse"] = True
        return report

    preflight = inspect_builder2_closure_rerender(
        state,
        requested_typography_version=expected_typography_version,
        force=force,
    )
    report["preflight"] = preflight
    if not preflight.get("closureOnlyRerenderEligible"):
        report["failureStage"] = "preflight"
        report["failureReason"] = (
            "builder2_closure_rerender_ineligible:"
            + ",".join(preflight.get("closureOnlyRerenderMissingFields") or [])
        )
        return report
    if not preflight.get("closureDurationContractSatisfied"):
        report["failureStage"] = "preflight"
        report["failureReason"] = "builder2_closure_duration_contract_mismatch"
        return report

    raw_url = resolve_raw_runway_artifact_url(state)
    if not raw_url:
        report["failureStage"] = "preflight"
        report["failureReason"] = "builder2_closure_rerender_missing:rawRunwayVideo"
        return report

    try:
        slogan_override = resolve_closure_only_rerender_slogan_override(state=state)
        product_name, slogan, language = resolve_trusted_closure_copy(
            state,
            slogan_override=slogan_override,
        )
        override_applied = bool(_clean(slogan_override))
    except Builder2TournamentError as exc:
        report["failureStage"] = "copy"
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_closure_rerender_missing_copy")
        return report

    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict):
        closure = (state.get("winnerDevelopmentPlan") or {}).get("advertisingClosure") or {}
    duration_seconds = resolve_builder2_effective_closure_segment_duration_seconds(
        float(closure.get("durationSeconds")) if isinstance(closure, dict) and closure.get("durationSeconds") is not None else None,
        typography_contract_version=expected_typography_version,
    )
    enforce_v3_closure_duration_contract()

    base_url = _clean(public_base_url or _resolve_public_base_url(state))
    if not base_url:
        report["failureStage"] = "configuration"
        report["failureReason"] = "builder2_closure_rerender_missing_public_base_url"
        return report

    report["closureOnlyRerenderAttempted"] = True
    _append_final_output_history(media, reason="pre_closure_typography_rerender")
    report["previousFinalPreserved"] = True

    tmp = Path(tempfile.mkdtemp(prefix="ace_closure_rerender_"))
    output_path = tmp / "builder2_closure_rerender_final.mp4"
    try:
        require_builder2_web_storage_capability(base_url)
        from engine.builder2_lyria_artifact import job_requires_lyria_soundtrack, resolve_lyria_audio_for_render

        lyria_audio_path = ""
        if job_requires_lyria_soundtrack(state):
            lyria_audio_path = resolve_lyria_audio_for_render(
                job_id=job_id,
                state=state,
                public_base_url=base_url,
            )

        render_result = render_builder2_advertising_closure_endcard(
            raw_url,
            product_name=product_name,
            slogan=slogan,
            output_path=output_path,
            language=language,
            duration_seconds=duration_seconds,
            job_id=job_id,
            lyria_audio_path=lyria_audio_path,
        )
        report["ffmpegCalls"] = 1
        report["rawRunwayReused"] = True
        if isinstance(render_result.typography_metadata, dict):
            verify_closure_typography_metadata(render_result.typography_metadata)
            media.update(render_result.typography_metadata)
            media["closureTypographyContractVersion"] = expected_typography_version
        publication = publish_builder2_durable_final_video(
            output_path,
            base_url,
            job_id=job_id,
            output_token=render_result.output_token,
        )
        report["publicationCalls"] = 1
        final_url = apply_builder2_durable_publication_fields(state, publication)
        media["finalVideoWithClosureUrl"] = final_url
        media["advertisingClosureRendered"] = True
        media["advertisingClosureStatus"] = "completed"
        state["advertisingClosureStatus"] = "completed"
        media["actualFinalVideoDurationSeconds"] = render_result.measured_duration_seconds
        media["closureOnlyRerenderCompletedForVersion"] = expected_typography_version
        media["closureOnlyRerenderCompletedAt"] = _utc_now_iso()
        media["closureOnlyRerenderSource"] = "builder2_closure_only_rerender"
        if override_applied:
            apply_closure_only_rerender_copy_override(
                state,
                product_name=product_name,
                slogan=slogan,
                language=language,
                override_applied=True,
            )
            report["closureSloganOverrideApplied"] = True
            report["renderedClosureSloganText"] = slogan
        media["mediaResumeStatus"] = "completed"
        state["status"] = "completed"
        state["mediaContinuationRequired"] = False
        save_tournament_state(job_id, state)
        if redis_configured():
            from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
            from engine.builder2_packaging_marketing_text import ensure_builder2_packaging_marketing_text

            plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
            MediaResumeIsolationGuard.end()
            marketing_text, marketing_source = ensure_builder2_packaging_marketing_text(
                existing_text=_clean(media.get("marketingText")),
                existing_source=str(media.get("marketingCopySource") or ""),
                product_name=str(state.get("productName") or plan.get("productNameResolved") or product_name or ""),
                product_description=str(state.get("productDescription") or ""),
                plan=plan,
                content_language=str(state.get("contentLanguage") or plan.get("language") or language or ""),
                headline_text=str(plan.get("headlineText") or ""),
            )
            media["marketingText"] = marketing_text
            media["marketingCopySource"] = marketing_source
            save_tournament_state(job_id, state)
            video_job_mark_done(job_id, final_url, marketing_text)
        report["stateMutated"] = True
        report["newFinalPromoted"] = True
        report["typographyUpgradeApplied"] = True
        report["closureOnlyRerenderAccepted"] = True
        report["ok"] = True
        logger.info(
            "BUILDER2_CLOSURE_ONLY_RERENDER_ACCEPTED jobId=%s typographyVersion=%s",
            job_id,
            expected_typography_version,
        )
        return report
    except (Builder2ClosureRenderError, Builder2FinalPublicationError, Builder2TournamentError) as exc:
        report["failureStage"] = "render_or_publication"
        report["failureReason"] = str(exc.args[0] if exc.args else type(exc).__name__)
        return report
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass


def main() -> int:
    job_id = _clean(os.environ.get("BUILDER2_CLOSURE_ONLY_RERENDER_JOB_ID"))
    expected = _clean(os.environ.get("BUILDER2_CLOSURE_ONLY_RERENDER_EXPECTED_VERSION")) or BUILDER2_CLOSURE_TYPOGRAPHY_VERSION
    if not job_id:
        print("BUILDER2_CLOSURE_ONLY_RERENDER_JOB_ID is required", file=sys.stderr)
        return 2
    report = run_builder2_closure_only_rerender(job_id=job_id, expected_typography_version=expected)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
