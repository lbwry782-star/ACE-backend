"""
Flask entrypoint: health, Builder1 campaign-series, Builder2 video pipeline routes.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
import requests
from flask import Flask, Response, jsonify, request, send_file
from openai import OpenAI

from engine.video_headline_postprocess import (
    get_headline_video_path,
    hard_test_video_path,
    log_video_headline_delivery_startup,
    write_headline_video_bytes,
)
from engine.video_jobs_redis import (
    redis_configured,
    video_job_create,
    video_job_get,
    video_job_try_finalize_stale_running,
)
from engine.video_web_postprocess import ensure_video_postprocessed_for_poll
from engine.builder1_campaign_store import (
    CampaignStoreError,
    apply_repaired_campaign_plan,
    begin_physical_repair,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    mark_image_retry_required,
    mark_physical_repair_required,
    release_generation_lock,
    reserve_next_ad_index,
    validate_next_ad_request,
)
from engine.builder1_retry_state import (
    RETRY_MODE_IMAGE_ONLY,
    RETRY_MODE_NONE,
    RETRY_MODE_REPAIR_FROM_PHYSICAL,
    public_retry_fields,
    resolve_authoritative_retry_mode,
)
from engine.builder1_jobs_store import (
    create_builder1_job,
    finalize_builder1_job,
    get_builder1_job,
    update_builder1_job,
)
from engine.builder2_zip import build_builder2_video_zip_bytes
from engine.builder1_composition import build_builder1_series_composition_metadata
from engine.builder1_failure_classification import (
    Builder1FailureClass,
    PlanContradictionComplianceError,
    PlanProductVisibilityConflictError,
    classify_compliance_failure,
)
from engine.builder1_image_generator import (
    ImageRateLimitError,
    generate_builder1_ad_image,
    image_bytes_to_base64,
)
from engine.builder1_physical_repair import repair_builder1_campaign_from_physical
from engine.builder1_image_compliance import (
    ImageComplianceError,
    ImageComplianceUnavailableError,
    log_builder1_image_compliance_config,
)
from engine.builder1_planning_profile import (
    log_builder1_planning_profile_config,
    resolve_stage_routing,
)
from engine.builder1_input_normalizer import Builder1InputError, normalize_ad_count
from engine.builder1_plan_spec import (
    ad_to_public_api_dict,
    campaign_identity_to_dict,
)
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_planning_metrics import (
    get_planning_metrics,
    log_builder1_initial_ad_timing,
    log_builder1_next_ad_timing,
)
from engine.builder1_zip import build_builder1_zip_bytes, build_builder1_zip_from_request

app = Flask(__name__)

# Configure logging early (handlers use logger at request time)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_builder1_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="builder1_job")

try:
    log_video_headline_delivery_startup("web")
except Exception as e:
    logger.warning("VIDEO_HEADLINE_UPLOAD_CONFIG web startup failed err=%s", e)

try:
    log_builder1_image_compliance_config()
except Exception as e:
    logger.warning("BUILDER1_IMAGE_COMPLIANCE_CONFIG startup failed err=%s", e)

try:
    log_builder1_planning_profile_config()
except Exception as e:
    logger.warning("BUILDER1_PLANNING_PROFILE_CONFIG startup failed err=%s", e)


@app.before_request
def _video_headline_artifact_incoming_trace():
    """Log any request to upload paths so we can see traffic even when routing returns 404."""
    try:
        p = request.path or ""
    except Exception:
        return
    if "video-headline-artifact" not in p:
        return
    logger.info(
        "VIDEO_HEADLINE_UPLOAD_INCOMING method=%s path=%s url_rule=%s",
        request.method,
        p,
        getattr(request.url_rule, "rule", None) if request.url_rule else None,
    )


# ============================================================================
# CORS Configuration (same behavior as main app for /api/* video clients)
# ============================================================================

ALLOWED_ORIGINS = [
    "https://ace-advertising.agency",
    "https://www.ace-advertising.agency",
]


def is_security_enabled() -> bool:
    """Read ACE_SECURITY_ENABLED from env. Only 'false' disables; missing or 'true' = enabled. Default True."""
    raw = (os.environ.get("ACE_SECURITY_ENABLED", "") or "").strip().lower()
    if raw == "false":
        return False
    return True


def is_origin_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return False
    return origin in ALLOWED_ORIGINS


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        allow_origin = is_origin_allowed(origin) if is_security_enabled() else bool(origin)
        if allow_origin and origin:
            response = Response("", status=200)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-ACE-Batch-State, X-ACE-Admin-Secret"
            )
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        return Response("", status=200)


@app.after_request
def add_cors_headers(response):
    if request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        allow_origin = is_origin_allowed(origin) if is_security_enabled() else bool(origin)
        if allow_origin and origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-ACE-Batch-State, X-ACE-Admin-Secret"
            )
            if "X-ACE-Batch-State" in response.headers:
                response.headers["Access-Control-Expose-Headers"] = "X-ACE-Batch-State"
    return response


if not is_security_enabled():
    logger.info("ACE_SECURITY_ENABLED=false security_checks_bypassed=true")


# -----------------------------------------------------------------------------
# Runway video MVP (isolated): one text-to-video only — not part of image ad flow.
# -----------------------------------------------------------------------------


@app.route("/api/test-video/<job_id>", methods=["GET"])
def serve_test_video(job_id):
    """Serve processed MP4 saved by postprocess_video_headline on this web service host."""
    path = hard_test_video_path(job_id or "")
    if not path or not path.is_file():
        return jsonify({"ok": False, "error": "not_found"}), 404
    logger.info("VIDEO_TEST_SERVE path=%s", str(path))
    return send_file(
        str(path),
        mimetype="video/mp4",
        as_attachment=False,
        download_name="ace-video-test.mp4",
    )


@app.route("/api/video-headline-artifact", methods=["POST"], strict_slashes=False)
@app.route("/video-headline-artifact", methods=["POST"], strict_slashes=False)
@app.route("/api/internal/video-headline-artifact", methods=["POST"], strict_slashes=False)
@app.route("/internal/video-headline-artifact", methods=["POST"], strict_slashes=False)
def internal_video_headline_artifact():
    """
    Background worker POSTs the processed MP4 here so GET /api/video-headline/<token> can read it
    from this process's disk (split web + worker on Render).
    """
    logger.info("VIDEO_HEADLINE_UPLOAD_ENDPOINT_HIT")
    logger.info(
        "VIDEO_HEADLINE_UPLOAD_ENDPOINT_HIT path=%s content_length=%s remote_addr=%s user_agent=%s",
        request.path,
        request.content_length,
        request.remote_addr,
        (request.headers.get("User-Agent") or "")[:120],
    )
    secret = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
    if not secret:
        return jsonify({"ok": False, "error": "not_configured"}), 503
    if (request.headers.get("X-ACE-Video-Headline-Upload-Secret") or "").strip() != secret:
        logger.warning("VIDEO_HEADLINE_UPLOAD_REJECT reason=bad_secret")
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    _max_raw = (os.environ.get("VIDEO_HEADLINE_MAX_UPLOAD_BYTES") or "").strip()
    max_bytes = int(_max_raw) if _max_raw else 250 * 1024 * 1024
    if request.content_length is not None and request.content_length > max_bytes:
        return jsonify({"ok": False, "error": "too_large"}), 413
    token = (request.form.get("token") or "").strip()
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "missing_file"}), 400
    data = upload.stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        return jsonify({"ok": False, "error": "too_large"}), 413
    if not write_headline_video_bytes(token, data):
        logger.warning(
            "VIDEO_HEADLINE_UPLOAD_REJECT reason=invalid_token_or_write token_prefix=%s",
            (token[:8] if len(token) >= 8 else ""),
        )
        return jsonify({"ok": False, "error": "invalid_token_or_write"}), 400
    logger.info("VIDEO_HEADLINE_UPLOAD_STORED")
    logger.info(
        "VIDEO_HEADLINE_UPLOAD_STORED bytes=%s token_prefix=%s",
        len(data),
        token[:8] if len(token) >= 8 else token,
    )
    return jsonify({"ok": True}), 200


@app.route("/api/video-headline/<token>", methods=["GET"], strict_slashes=False)
@app.route("/video-headline/<token>", methods=["GET"], strict_slashes=False)
def serve_video_headline(token):
    """Serve ffmpeg-processed MP4 from disk-backed store (token from videoUrl); survives worker restart."""
    path = get_headline_video_path((token or "").strip())
    if not path or not path.is_file():
        logger.info("VIDEO_HEADLINE_SERVE miss lookup=disk")
        return jsonify({"ok": False, "error": "not_found"}), 404
    logger.info("VIDEO_HEADLINE_SERVE hit lookup=disk")
    return send_file(
        str(path),
        mimetype="video/mp4",
        as_attachment=False,
        download_name="ace-video.mp4",
    )


@app.route("/api/video-headline-debug", methods=["GET"])
def video_headline_debug():
    """Diagnosis: which video-headline routes exist in this process + ACE_PUBLIC_BASE_URL (no secrets)."""
    routes = []
    for rule in app.url_map.iter_rules():
        r = str(rule.rule)
        if "video-headline" not in r:
            continue
        routes.append(
            {
                "rule": r,
                "methods": sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}),
                "endpoint": rule.endpoint,
            }
        )
    return jsonify(
        {
            "ok": True,
            "ace_public_base_url": (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip(),
            "video_headline_routes": sorted(routes, key=lambda x: x["rule"]),
            "hint": "POST uploads must hit THIS service's hostname; set ACE_PUBLIC_BASE_URL (web + worker) to the live Render URL for this deploy.",
        }
    ), 200


@app.route("/api/generate-video", methods=["POST"])
def generate_video():
    """Enqueue ACE video job in Redis; worker runs pipeline. Poll GET /api/video-status?jobId=."""
    import time as _time

    t_req0 = _time.monotonic()
    try:
        if not redis_configured():
            logger.error("VIDEO_JOB_REDIS_MISSING REDIS_URL not set")
            return jsonify({"ok": False, "error": "video_jobs_unconfigured"}), 503
        if not request.is_json:
            return jsonify({"ok": False, "error": "video_generation_failed"}), 200
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "video_generation_failed"}), 200
        product_description = (payload.get("productDescription") or "").strip()
        if not product_description:
            return jsonify({"ok": False, "error": "video_generation_failed"}), 200
        product_name = (payload.get("productName") or "").strip()
        base = (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip().rstrip("/") or (request.url_root or "").rstrip("/")
        job_id = str(uuid.uuid4())
        logger.info("VIDEO_TIMING_STAGE_START stage=request_received jobId=%s", job_id)
        try:
            video_job_create(job_id, product_name, product_description, base)
        except Exception as e:
            logger.error("VIDEO_JOB_REDIS_ENQUEUE_FAILED jobId=%s err=%s", job_id, e, exc_info=True)
            return jsonify({"ok": False, "error": "video_generation_failed"}), 200
        logger.info("VIDEO_JOB_CREATED jobId=%s", job_id)
        logger.info(
            "VIDEO_TIMING_STAGE_END stage=job_created jobId=%s elapsed_ms=%.1f",
            job_id,
            (_time.monotonic() - t_req0) * 1000.0,
        )
        return jsonify(
            {
                "ok": True,
                "jobId": job_id,
                "status": "running",
            }
        ), 200
    except Exception as e:
        logger.error("generate_video enqueue failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": "video_generation_failed"}), 200


@app.route("/api/video-status", methods=["GET"])
def video_status():
    """Poll async /api/generate-video job from Redis: running | done | error | interrupted."""
    if not redis_configured():
        return jsonify({"ok": False, "error": "video_jobs_unconfigured"}), 503
    job_id = request.args.get("jobId", "").strip()
    if not job_id:
        return jsonify({"ok": False, "error": "missing_param", "message": "jobId is required"}), 400
    try:
        job = video_job_get(job_id)
    except Exception as e:
        logger.error("VIDEO_JOB_POLL_REDIS_ERR jobId=%s err=%s", job_id, e, exc_info=True)
        return jsonify({"ok": False, "error": "video_status_failed"}), 500
    if not job:
        return jsonify({"ok": False, "error": "not_found"}), 404
    status = (job.get("status") or "running").strip()
    if status == "running":
        try:
            if video_job_try_finalize_stale_running(job_id):
                job = video_job_get(job_id)
                if not job:
                    return jsonify({"ok": False, "error": "not_found"}), 404
                status = (job.get("status") or "error").strip()
        except Exception as e:
            logger.error("VIDEO_JOB_STALE_CHECK_ERR jobId=%s err=%s", job_id, e, exc_info=True)
    logger.info("VIDEO_JOB_POLL jobId=%s status=%s", job_id, status)
    out = {"ok": True, "status": status}
    if status == "interrupted":
        err = (job.get("error") or "").strip() or "worker_shutdown_during_job"
        icode = (job.get("interruptCode") or "").strip() or "interrupted_worker_shutdown"
        logger.info(
            "VIDEO_JOB_POLL infrastructure_interruption jobId=%s interrupt_code=%s error=%s",
            job_id,
            icode,
            err,
        )
        out["error"] = err
        out["interruptCode"] = icode
        out["infrastructureInterrupted"] = bool(job.get("infrastructureFailure"))
        out["retryable"] = True
        return jsonify(out), 200
    if status == "done":
        ensure_video_postprocessed_for_poll(job_id, job)
        job = video_job_get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "not_found"}), 404
        vu = job.get("videoUrl") or ""
        out["videoUrl"] = vu
        out["marketingText"] = job.get("marketingText") or ""
        rp = (job.get("productNameResolved") or job.get("resolvedProductName") or "").strip()
        out["productNameResolved"] = rp
        out["product_name_resolved"] = rp
        logger.info(
            "VIDEO_PRODUCT_NAME_RESOLVED_RETURNED value=%s",
            json.dumps(rp, ensure_ascii=False),
        )
        logger.info("VIDEO_JOB_RESULT jobId=%s video_url=%s", job_id, vu)
        logger.info("VIDEO_TIMING_STAGE_END stage=frontend_poll_done jobId=%s", job_id)
    if status == "error":
        err = job.get("error") or "video_generation_failed"
        logger.info("VIDEO_JOB_POLL terminal_error jobId=%s reason=%s", job_id, err)
        out["error"] = err
    return jsonify(out), 200


@app.route("/api/builder1-demo", methods=["GET"])
def builder1_demo():
    return (
        jsonify(
            {
                "ok": False,
                "error": "demo_disabled",
                "message": "Use POST /api/builder1-generate for campaign-series generation.",
            }
        ),
        200,
    )


def _parse_builder1_o3_json_text(raw: str) -> dict[str, Any]:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
    t = t.strip()
    if t.lower().startswith("```json"):
        t = t[7:].lstrip()
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model_output_not_object")
    return obj


def _o3_pro_planning_model_caller(
    system_prompt: str,
    user_prompt: str,
    *,
    stage: str | None = None,
) -> object:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(150.0),
        max_retries=0,
    )
    from engine.builder1_planning_model import call_planning_model

    routing = resolve_stage_routing(stage)
    if stage:
        logger.info(
            "BUILDER1_STAGE_MODEL stage=%s model=%s reasoningEffort=%s",
            stage,
            routing.model,
            routing.reasoning_effort or "none",
        )

    return call_planning_model(
        client,
        model=routing.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage=stage,
        reasoning_effort=routing.reasoning_effort,
        parse_json_text=_parse_builder1_o3_json_text,
    )


@app.route("/api/builder1-plan-demo", methods=["GET"])
def builder1_plan_demo():
    return (
        jsonify(
            {
                "ok": False,
                "error": "demo_disabled",
                "message": "Use POST /api/builder1-generate for campaign-series generation.",
            }
        ),
        200,
    )


def _builder1_gpt_image_size_for_format(format_value: str) -> str:
    f = (format_value or "").strip().lower()
    if f == "landscape":
        return "1536x1024"
    if f == "portrait":
        return "1024x1536"
    if f == "square":
        return "1024x1024"
    return "1024x1536"


def _builder1_image_caller(prompt: str, format_value: str) -> bytes:
    if (os.environ.get("BUILDER1_IMAGE_MODEL") or "").strip():
        model = os.environ["BUILDER1_IMAGE_MODEL"].strip()
        source = "env"
    else:
        model = os.getenv("BUILDER1_IMAGE_MODEL", "gpt-image-1.5")
        source = "default"
    logger.info("BUILDER1_IMAGE_MODEL_SELECTED model=%s source=%s", model, source)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    r = client.images.generate(
        model=model,
        prompt=prompt,
        size=_builder1_gpt_image_size_for_format(format_value),
        quality="low",
    )
    if not r.data:
        raise ValueError("image_generation_empty")
    b64 = r.data[0].b64_json
    if not b64:
        raise ValueError("image_generation_empty")
    return base64.b64decode(b64)


def _builder1_update_job(job_id: str, **fields: Any) -> None:
    update_builder1_job(job_id, **fields)


def _builder1_build_incremental_result(
    *,
    campaign_id: str,
    session,
    ad_index: int,
    visual_prompt: str,
    image_base64: str,
) -> dict[str, Any]:
    ad_plan = next(a for a in session.plan.ads if a.index == ad_index)
    composition = build_builder1_series_composition_metadata(session.plan)
    generated_count = session.generated_count
    can_next = not session.complete and session.next_ad_index is not None
    return {
        "ok": True,
        "campaignId": campaign_id,
        "campaign": campaign_identity_to_dict(session.plan),
        "composition": composition,
        "ad": ad_to_public_api_dict(
            ad_plan,
            visual_prompt=visual_prompt,
            image_base64=image_base64,
        ),
        "generatedCount": generated_count,
        "totalAds": session.target_ad_count,
        "targetAdCount": session.target_ad_count,
        "nextAdIndex": session.next_ad_index,
        "canGenerateNext": can_next,
    }


BUILDER1_PUBLIC_IMAGE_RETRY_MESSAGE = (
    "התמונה לא עמדה בדרישות. ניתן לנסות ליצור את אותה מודעה שוב."
)
BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE = (
    "נמצאה סתירה בתכנון החזותי. המערכת תתקן את המודעה בלי להתחיל את הקמפיין מחדש."
)


def _builder1_retry_error_response(
    *,
    campaign_id: str,
    ad_index: int,
    session,
    error_code: str,
    retry_mode: str,
    user_message: str,
    violations: Optional[list[str]] = None,
) -> dict[str, Any]:
    payload = {
        "ok": False,
        "error": error_code,
        "retryable": True,
        "planningComplete": True,
        "retryMode": retry_mode,
        "retryAdIndex": ad_index,
        "campaignId": campaign_id,
        "nextAdIndex": ad_index,
        "generatedCount": session.generated_count,
        "targetAdCount": session.target_ad_count,
        "status": getattr(session, "status", None) or "image_retry_required",
        "preservedThroughStage": getattr(session, "preserved_through_stage", None),
        "userMessage": user_message,
        "lastImageViolations": violations or getattr(session, "last_image_violations", None) or [],
        "planRevision": getattr(session, "plan_revision", 1),
    }
    payload.update(
        public_retry_fields(
            session=session,
            retry_ad_index=ad_index,
        )
    )
    payload["retryMode"] = retry_mode
    payload["retryable"] = True
    return payload


def _builder1_assert_image_job_allowed(
    *,
    campaign_id: str,
    job_id: str,
    ad_index: int,
) -> Any:
    session = get_campaign_session(campaign_id)
    job = get_builder1_job(job_id) or {}
    job_revision = job.get("planRevision")
    if job_revision is not None and int(job_revision) != session.plan_revision:
        raise CampaignStoreError("stale_plan_revision")
    job_ad_index = job.get("retryAdIndex")
    if job_ad_index is not None and int(job_ad_index) != ad_index:
        raise CampaignStoreError("campaign_index_conflict")
    mode = resolve_authoritative_retry_mode(status=session.status, retry_mode=session.retry_mode)
    if session.repair_in_progress or mode == RETRY_MODE_REPAIR_FROM_PHYSICAL:
        raise CampaignStoreError("physical_repair_not_completed")
    return session


def _builder1_image_compliance_error_response(
    *,
    campaign_id: str,
    ad_index: int,
    session,
    error_code: str,
    violations: Optional[list[str]] = None,
    retry_mode: str = "image_only",
) -> dict[str, Any]:
    return _builder1_retry_error_response(
        campaign_id=campaign_id,
        ad_index=ad_index,
        session=session,
        error_code=error_code,
        retry_mode=retry_mode,
        user_message=BUILDER1_PUBLIC_IMAGE_RETRY_MESSAGE,
        violations=violations,
    )


def _builder1_generate_single_ad(
    *,
    job_id: str,
    campaign_id: str,
    ad_index: int,
    already_reserved: bool,
    log_next_ad_timing: bool = False,
    initial_request_started_at: float | None = None,
    planning_duration_ms: int = 0,
    campaign_persistence_duration_ms: int = 0,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    session = None
    reservation_held = already_reserved
    lock_token = ""
    reservation_started = time.perf_counter()
    reservation_duration_ms = 0
    try:
        if already_reserved:
            session = get_campaign_session(campaign_id)
            if session.generating_index != ad_index:
                raise CampaignStoreError("campaign_index_conflict")
            if session.generating_lock_owner_job_id and session.generating_lock_owner_job_id != job_id:
                raise CampaignStoreError("campaign_generation_in_progress")
            lock_token = session.generating_lock_token or ""
            logger.info(
                "BUILDER1_GENERATION_LOCK_CONTINUED campaignId=%s jobId=%s reservedAdIndex=%s",
                campaign_id,
                job_id,
                ad_index,
            )
        else:
            session = reserve_next_ad_index(campaign_id, ad_index, job_id=job_id)
            lock_token = session.generating_lock_token or ""
            reservation_held = True

        session = _builder1_assert_image_job_allowed(
            campaign_id=campaign_id,
            job_id=job_id,
            ad_index=ad_index,
        )

        reservation_duration_ms = int((time.perf_counter() - reservation_started) * 1000)

        logger.info(
            "BUILDER1_NEXT_AD_INDEX jobId=%s campaignId=%s adIndex=%s targetAdCount=%s",
            job_id,
            campaign_id,
            ad_index,
            session.target_ad_count,
        )
        _builder1_update_job(
            job_id,
            stage="generating_images",
            completedAds=session.generated_count,
            totalAds=session.target_ad_count,
            targetAdCount=session.target_ad_count,
        )

        image_result = generate_builder1_ad_image(
            session.plan,
            ad_index,
            _builder1_image_caller,
            campaign_id=campaign_id,
            job_id=job_id,
        )
        session = mark_ad_generated(campaign_id, ad_index)
        reservation_held = False

        logger.info("BUILDER1_AD_GENERATED campaignId=%s adIndex=%s", campaign_id, ad_index)
        if log_next_ad_timing:
            log_builder1_next_ad_timing(
                campaign_id=campaign_id,
                job_id=job_id,
                ad_index=ad_index,
                image_generation_duration_ms=getattr(image_result, "image_generation_duration_ms", 0),
                compliance_review_duration_ms=getattr(image_result, "compliance_review_duration_ms", 0),
                compliance_regeneration_count=getattr(image_result, "compliance_regeneration_count", 0),
                total_next_ad_duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
        elif initial_request_started_at is not None:
            log_builder1_initial_ad_timing(
                campaign_id=campaign_id,
                job_id=job_id,
                planning_duration_ms=planning_duration_ms,
                campaign_persistence_duration_ms=campaign_persistence_duration_ms,
                reservation_duration_ms=reservation_duration_ms,
                image_generation_duration_ms=getattr(image_result, "image_generation_duration_ms", 0),
                compliance_review_duration_ms=getattr(image_result, "compliance_review_duration_ms", 0),
                compliance_regeneration_count=getattr(image_result, "compliance_regeneration_count", 0),
                total_initial_request_duration_ms=int((time.perf_counter() - initial_request_started_at) * 1000),
            )
        return _builder1_build_incremental_result(
            campaign_id=campaign_id,
            session=session,
            ad_index=ad_index,
            visual_prompt=image_result.visual_prompt,
            image_base64=image_bytes_to_base64(image_result.image_bytes),
        )
    except ImageComplianceError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        session = get_campaign_session(campaign_id)
        failure_class, _action, _details, _evidence = classify_compliance_failure(
            violations=list(e.violations),
            series_plan=session.plan,
        )
        if failure_class == Builder1FailureClass.PLAN_CONTRADICTION:
            session = mark_physical_repair_required(
                campaign_id,
                failed_ad_index=ad_index,
                violations=list(e.violations),
            )
            logger.error(
                "BUILDER1_PLAN_CONTRADICTION campaignId=%s jobId=%s adIndex=%s violations=%s",
                campaign_id,
                job_id,
                ad_index,
                e.violations,
            )
            return _builder1_retry_error_response(
                campaign_id=campaign_id,
                ad_index=ad_index,
                session=session,
                error_code="physical_plan_conflict",
                retry_mode="repair_from_physical",
                user_message=BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE,
                violations=list(e.violations),
            )
        session = mark_image_retry_required(
            campaign_id,
            failed_ad_index=ad_index,
            violations=list(e.violations),
        )
        logger.error(
            "BUILDER1_IMAGE_COMPLIANCE_FAILED campaignId=%s jobId=%s adIndex=%s violations=%s",
            campaign_id,
            job_id,
            ad_index,
            e.violations,
        )
        return _builder1_image_compliance_error_response(
            campaign_id=campaign_id,
            ad_index=ad_index,
            session=session,
            error_code="image_compliance_failed",
            violations=list(e.violations),
            retry_mode="image_only",
        )
    except PlanContradictionComplianceError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        session = mark_physical_repair_required(
            campaign_id,
            failed_ad_index=ad_index,
            violations=list(e.violations),
        )
        return _builder1_retry_error_response(
            campaign_id=campaign_id,
            ad_index=ad_index,
            session=session,
            error_code="physical_plan_conflict",
            retry_mode="repair_from_physical",
            user_message=BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE,
            violations=list(e.violations),
        )
    except PlanProductVisibilityConflictError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        session = mark_physical_repair_required(
            campaign_id,
            failed_ad_index=ad_index,
            violations=list(e.reasons),
        )
        return _builder1_retry_error_response(
            campaign_id=campaign_id,
            ad_index=ad_index,
            session=session,
            error_code="plan_product_visibility_conflict",
            retry_mode="repair_from_physical",
            user_message=BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE,
            violations=list(e.reasons),
        )
    except ImageComplianceUnavailableError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        session = get_campaign_session(campaign_id)
        return _builder1_image_compliance_error_response(
            campaign_id=campaign_id,
            ad_index=ad_index,
            session=session,
            error_code="image_compliance_unavailable",
        )
    except ImageRateLimitError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        retry_after = getattr(e, "retry_after_seconds", None)
        logger.error(
            "BUILDER1_IMAGE_RATE_LIMITED campaignId=%s adIndex=%s retryAfterSeconds=%s",
            campaign_id,
            ad_index,
            retry_after,
        )
        out: dict[str, Any] = {
            "ok": False,
            "error": "image_rate_limited",
            "retryable": True,
            "campaignId": campaign_id,
            "nextAdIndex": ad_index,
        }
        if retry_after is not None:
            out["retryAfterSeconds"] = retry_after
        return out
    except CampaignStoreError as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        if e.code in {"physical_repair_not_completed", "stale_plan_revision"}:
            session = get_campaign_session(campaign_id)
            logger.error(
                "BUILDER1_IMAGE_JOB_REJECTED campaignId=%s jobId=%s adIndex=%s code=%s planRevision=%s",
                campaign_id,
                job_id,
                ad_index,
                e.code,
                session.plan_revision,
            )
            return {
                "ok": False,
                "error": e.code,
                "message": e.message,
                "campaignId": campaign_id,
                "retryable": True,
                **public_retry_fields(session=session, retry_ad_index=ad_index),
            }
        return {"ok": False, "error": e.code, "message": e.message, "campaignId": campaign_id}
    except Exception as e:
        if reservation_held:
            release_generation_lock(
                campaign_id,
                job_id=job_id,
                lock_token=lock_token,
            )
        logger.error("BUILDER1_SERIES_IMAGE_FAILED campaignId=%s adIndex=%s err=%s", campaign_id, ad_index, e)
        return {"ok": False, "error": "image_generation_failed", "message": str(e), "campaignId": campaign_id}


def _builder1_generate_initial(
    job_id: str,
    campaign_id: str,
    product_name: str,
    product_description: str,
    format_val: str,
    ad_count: int,
    brand_guidelines: Optional[dict[str, Any]],
) -> dict[str, Any]:
    initial_started_at = time.perf_counter()
    logger.info(
        "BUILDER1_SERIES_REQUEST jobId=%s campaignId=%s adCount=%s format=%s",
        job_id,
        campaign_id,
        ad_count,
        format_val,
    )
    _builder1_update_job(job_id, stage="planning", completedAds=0, totalAds=ad_count)
    planning_started_at = time.perf_counter()
    try:
        series_plan = plan_builder1(
            product_name=product_name,
            product_description=product_description,
            format_value=format_val,
            model_caller=_o3_pro_planning_model_caller,
            ad_count=ad_count,
            brand_guidelines=brand_guidelines,
            campaign_id=campaign_id,
            job_id=job_id,
        )
    except Builder1PlannerError as e:
        err_message = str(e) if e else "planning_failed"
        if err_message.startswith("product_name_generation_failed"):
            logger.error("BUILDER1_PRODUCT_NAME_GENERATION_FAILED jobId=%s err=%s", job_id, err_message)
            return {"ok": False, "error": "product_name_generation_failed", "message": err_message}
        logger.error("BUILDER1_SERIES_PLAN_REJECTED jobId=%s err=%s", job_id, err_message)
        return {"ok": False, "error": "planning_failed", "message": err_message}
    except Exception as e:
        err_message = str(e) if e else "planning_failed"
        logger.error("BUILDER1_SERIES_PLAN_REJECTED jobId=%s err=%s", job_id, err_message)
        return {"ok": False, "error": "planning_failed", "message": err_message}

    planning_duration_ms = int((time.perf_counter() - planning_started_at) * 1000)
    metrics = get_planning_metrics()
    if metrics is not None and metrics.total_planning_duration_ms:
        planning_duration_ms = metrics.total_planning_duration_ms

    persistence_started_at = time.perf_counter()
    create_campaign_session(
        campaign_id=campaign_id,
        plan=series_plan,
        target_ad_count=ad_count,
    )
    campaign_persistence_duration_ms = int((time.perf_counter() - persistence_started_at) * 1000)
    _builder1_update_job(job_id, stage="building_prompts", targetAdCount=ad_count, totalAds=ad_count)
    return _builder1_generate_single_ad(
        job_id=job_id,
        campaign_id=campaign_id,
        ad_index=1,
        already_reserved=False,
        initial_request_started_at=initial_started_at,
        planning_duration_ms=planning_duration_ms,
        campaign_persistence_duration_ms=campaign_persistence_duration_ms,
    )


def _builder1_run_initial_job(
    job_id: str,
    campaign_id: str,
    product_name: str,
    product_description: str,
    format_val: str,
    ad_count: int,
    brand_guidelines: Optional[dict[str, Any]],
) -> None:
    logger.info("BUILDER1_JOB_STARTED jobId=%s campaignId=%s adCount=%s", job_id, campaign_id, ad_count)
    try:
        result = _builder1_generate_initial(
            job_id, campaign_id, product_name, product_description, format_val, ad_count, brand_guidelines
        )
        _builder1_finalize_job(job_id, result, target_ad_count=ad_count)
    except Exception as e:
        update_builder1_job(job_id, status="error", error="builder1_generation_failed")
        logger.error("BUILDER1_JOB_ERROR jobId=%s err=%s", job_id, e, exc_info=True)


def _builder1_run_next_job(job_id: str, campaign_id: str, expected_next_index: int) -> None:
    logger.info(
        "BUILDER1_NEXT_AD_REQUEST jobId=%s campaignId=%s expectedNextIndex=%s",
        job_id,
        campaign_id,
        expected_next_index,
    )
    try:
        result = _builder1_generate_single_ad(
            job_id=job_id,
            campaign_id=campaign_id,
            ad_index=expected_next_index,
            already_reserved=True,
            log_next_ad_timing=True,
        )
        target_ad_count = get_campaign_session(campaign_id).target_ad_count
        _builder1_finalize_job(job_id, result, target_ad_count=target_ad_count)
    except Exception as e:
        update_builder1_job(job_id, status="error", error="builder1_generation_failed")
        logger.error("BUILDER1_JOB_ERROR jobId=%s err=%s", job_id, e, exc_info=True)


def _builder1_run_physical_repair_job(job_id: str, campaign_id: str, retry_ad_index: int) -> None:
    logger.info(
        "BUILDER1_PHYSICAL_REPAIR_REQUEST jobId=%s campaignId=%s retryAdIndex=%s",
        job_id,
        campaign_id,
        retry_ad_index,
    )
    try:
        session = get_campaign_session(campaign_id)
        if not session.repair_in_progress:
            session = begin_physical_repair(campaign_id, job_id=job_id)
        original_plan = session.plan
        generated_indexes = list(session.generated_indexes)
        _builder1_update_job(job_id, stage="repairing_physical", completedAds=session.generated_count)
        repaired_plan = repair_builder1_campaign_from_physical(
            session.plan,
            model_caller=_o3_pro_planning_model_caller,
        )
        from engine.builder1_physical_repair import validate_repaired_plan_preserves_generated_ads

        consistency_reasons = validate_repaired_plan_preserves_generated_ads(
            original_plan=original_plan,
            repaired_plan=repaired_plan,
            generated_indexes=generated_indexes,
        )
        if consistency_reasons:
            logger.error(
                "BUILDER1_SERIES_CONSISTENCY_CONFLICT campaignId=%s reasons=%s generatedIndexes=%s",
                campaign_id,
                consistency_reasons,
                generated_indexes,
            )
            finalize_builder1_job(
                job_id,
                {
                    "ok": False,
                    "error": "campaign_series_consistency_conflict",
                    "campaignId": campaign_id,
                    "reasons": consistency_reasons,
                    "generatedCount": session.generated_count,
                    **public_retry_fields(session=session, retry_ad_index=retry_ad_index),
                },
                target_ad_count=session.target_ad_count,
            )
            return
        session = apply_repaired_campaign_plan(campaign_id, repaired_plan)
        update_builder1_job(
            job_id,
            planRevision=session.plan_revision,
            retryMode=RETRY_MODE_IMAGE_ONLY,
            retryAdIndex=retry_ad_index,
        )
        reserve_next_ad_index(campaign_id, retry_ad_index, job_id=job_id)
        result = _builder1_generate_single_ad(
            job_id=job_id,
            campaign_id=campaign_id,
            ad_index=retry_ad_index,
            already_reserved=True,
            log_next_ad_timing=True,
        )
        target_ad_count = get_campaign_session(campaign_id).target_ad_count
        _builder1_finalize_job(job_id, result, target_ad_count=target_ad_count)
    except Exception as e:
        from engine.builder1_campaign_store import cancel_physical_repair_in_progress

        try:
            cancel_physical_repair_in_progress(campaign_id)
        except Exception:
            pass
        update_builder1_job(job_id, status="error", error="builder1_generation_failed")
        logger.error("BUILDER1_JOB_ERROR jobId=%s err=%s", job_id, e, exc_info=True)


def _builder1_finalize_job(job_id: str, result: dict[str, Any], *, target_ad_count: int) -> None:
    finalize_builder1_job(job_id, result, target_ad_count=target_ad_count)


@app.route("/api/builder1-generate", methods=["POST"])
def builder1_generate():
    if not request.is_json:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "expected JSON body",
                }
            ),
            200,
        )
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "expected JSON object",
                }
            ),
            200,
        )
    product_name = (body.get("productName") or "").strip()
    product_description = (body.get("productDescription") or "").strip()
    format_val = (body.get("format") or "portrait").strip() or "portrait"
    brand_guidelines = body.get("brandGuidelines")
    if brand_guidelines is not None and not isinstance(brand_guidelines, dict):
        brand_guidelines = None
    if not product_description:
        logger.info("BUILDER1_INPUT_REJECTED field=productDescription reason=missing")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "productDescription is required",
                }
            ),
            400,
        )
    raw_ad_count = body.get("adCount")
    logger.info("BUILDER1_AD_COUNT_RAW value=%r", raw_ad_count)
    try:
        ad_count = normalize_ad_count(raw_ad_count)
    except Builder1InputError:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "invalid_ad_count",
                    "message": "adCount must be an integer from 2 through 4",
                }
            ),
            200,
        )
    logger.info("BUILDER1_AD_COUNT_NORMALIZED value=%s", ad_count)
    job_id = str(uuid.uuid4())
    campaign_id = str(uuid.uuid4())
    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=ad_count,
        stage="planning",
    )
    logger.info("BUILDER1_JOB_CREATED jobId=%s campaignId=%s targetAdCount=%s", job_id, campaign_id, ad_count)
    _builder1_executor.submit(
        _builder1_run_initial_job,
        job_id,
        campaign_id,
        product_name,
        product_description,
        format_val,
        ad_count,
        brand_guidelines,
    )
    return (
        jsonify(
            {
                "status": "running",
                "jobId": job_id,
                "stage": "planning",
                "completedAds": 0,
                "totalAds": ad_count,
                "targetAdCount": ad_count,
                "pollUrl": f"/api/builder1-status?jobId={job_id}",
                "campaignId": campaign_id,
            }
        ),
        202,
    )


@app.route("/api/builder1-retry-image", methods=["POST"])
def builder1_retry_image():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON body"}), 200
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON object"}), 200

    campaign_id = (body.get("campaignId") or "").strip()
    if not campaign_id:
        return jsonify({"ok": False, "error": "invalid_input", "message": "campaignId is required"}), 200

    try:
        retry_ad_index = int(body.get("retryAdIndex") or body.get("adIndex") or body.get("expectedNextIndex"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_input", "message": "retryAdIndex must be an integer"}), 200

    try:
        session = get_campaign_session(campaign_id)
    except CampaignStoreError as e:
        return jsonify({"ok": False, "error": e.code, "message": e.message}), 200

    retry_mode = resolve_authoritative_retry_mode(status=session.status, retry_mode=session.retry_mode)
    if retry_mode == RETRY_MODE_REPAIR_FROM_PHYSICAL:
        return jsonify(
            {
                "ok": False,
                "error": "physical_repair_not_completed",
                "campaignId": campaign_id,
                **public_retry_fields(session=session, retry_ad_index=retry_ad_index),
            }
        ), 200

    if not session.planning_complete:
        return jsonify({"ok": False, "error": "planning_incomplete", "campaignId": campaign_id}), 200
    if retry_ad_index in session.generated_indexes:
        return jsonify({"ok": False, "error": "campaign_index_conflict", "campaignId": campaign_id}), 200
    if retry_ad_index < 1 or retry_ad_index > session.target_ad_count:
        return jsonify({"ok": False, "error": "campaign_index_conflict", "campaignId": campaign_id}), 200

    job_id = str(uuid.uuid4())
    try:
        session = reserve_next_ad_index(campaign_id, retry_ad_index, job_id=job_id)
    except CampaignStoreError as e:
        return jsonify({"ok": False, "error": e.code, "message": e.message, "campaignId": campaign_id}), 200

    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=session.target_ad_count,
        stage="generating_images",
    )
    update_builder1_job(
        job_id,
        completedAds=session.generated_count,
        totalAds=session.target_ad_count,
    )
    _builder1_executor.submit(
        _builder1_run_next_job,
        job_id,
        campaign_id,
        retry_ad_index,
    )
    return (
        jsonify(
            {
                "status": "running",
                "jobId": job_id,
                "campaignId": campaign_id,
                "stage": "generating_images",
                "completedAds": session.generated_count,
                "totalAds": session.target_ad_count,
                "targetAdCount": session.target_ad_count,
                "retryAdIndex": retry_ad_index,
                "retryMode": "image_only",
                "planningComplete": True,
                "pollUrl": f"/api/builder1-status?jobId={job_id}",
            }
        ),
        202,
    )


@app.route("/api/builder1-repair-physical", methods=["POST"])
def builder1_repair_physical():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON body"}), 200
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON object"}), 200

    campaign_id = (body.get("campaignId") or "").strip()
    if not campaign_id:
        return jsonify({"ok": False, "error": "invalid_input", "message": "campaignId is required"}), 200

    try:
        retry_ad_index = int(body.get("retryAdIndex") or body.get("adIndex") or body.get("expectedNextIndex"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_input", "message": "retryAdIndex must be an integer"}), 200

    try:
        session = get_campaign_session(campaign_id)
    except CampaignStoreError as e:
        return jsonify({"ok": False, "error": e.code, "message": e.message}), 200

    if not session.planning_complete:
        return jsonify({"ok": False, "error": "planning_incomplete", "campaignId": campaign_id}), 200

    job_id = str(uuid.uuid4())
    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=session.target_ad_count,
        stage="repairing_physical",
    )
    update_builder1_job(
        job_id,
        completedAds=session.generated_count,
        totalAds=session.target_ad_count,
    )
    _builder1_executor.submit(
        _builder1_run_physical_repair_job,
        job_id,
        campaign_id,
        retry_ad_index,
    )
    return (
        jsonify(
            {
                "status": "running",
                "jobId": job_id,
                "campaignId": campaign_id,
                "stage": "repairing_physical",
                "completedAds": session.generated_count,
                "totalAds": session.target_ad_count,
                "targetAdCount": session.target_ad_count,
                "retryAdIndex": retry_ad_index,
                "retryMode": "repair_from_physical",
                "planningComplete": True,
                "pollUrl": f"/api/builder1-status?jobId={job_id}",
            }
        ),
        202,
    )


@app.route("/api/builder1-generate-next", methods=["POST"])
def builder1_generate_next():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON body"}), 200
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input", "message": "expected JSON object"}), 200

    campaign_id = (body.get("campaignId") or "").strip()
    if not campaign_id:
        return jsonify({"ok": False, "error": "invalid_input", "message": "campaignId is required"}), 200

    try:
        expected_next_index = int(body.get("expectedNextIndex"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_input", "message": "expectedNextIndex must be an integer"}), 200

    try:
        session = validate_next_ad_request(campaign_id, expected_next_index)
    except CampaignStoreError as e:
        return jsonify({"ok": False, "error": e.code, "message": e.message, "campaignId": campaign_id}), 200

    retry_mode = resolve_authoritative_retry_mode(
        status=session.status,
        retry_mode=session.retry_mode,
    )
    job_id = str(uuid.uuid4())

    if retry_mode == RETRY_MODE_REPAIR_FROM_PHYSICAL:
        try:
            session = begin_physical_repair(campaign_id, job_id=job_id)
        except CampaignStoreError as e:
            return jsonify({"ok": False, "error": e.code, "message": e.message, "campaignId": campaign_id}), 200
        create_builder1_job(
            job_id=job_id,
            campaign_id=campaign_id,
            target_ad_count=session.target_ad_count,
            stage="repairing_physical",
        )
        update_builder1_job(
            job_id,
            completedAds=session.generated_count,
            totalAds=session.target_ad_count,
            planRevision=session.plan_revision,
            retryMode=retry_mode,
            retryAdIndex=expected_next_index,
        )
        _builder1_executor.submit(
            _builder1_run_physical_repair_job,
            job_id,
            campaign_id,
            expected_next_index,
        )
        return (
            jsonify(
                {
                    "status": "running",
                    "jobId": job_id,
                    "campaignId": campaign_id,
                    "stage": "repairing_physical",
                    "completedAds": session.generated_count,
                    "totalAds": session.target_ad_count,
                    "targetAdCount": session.target_ad_count,
                    "pollUrl": f"/api/builder1-status?jobId={job_id}",
                    **public_retry_fields(
                        session=session,
                        retry_ad_index=expected_next_index,
                        repair_in_progress=True,
                    ),
                }
            ),
            202,
        )

    try:
        session = reserve_next_ad_index(campaign_id, expected_next_index, job_id=job_id)
    except CampaignStoreError as e:
        return jsonify({"ok": False, "error": e.code, "message": e.message, "campaignId": campaign_id}), 200

    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=session.target_ad_count,
        stage="generating_images",
    )
    update_builder1_job(
        job_id,
        completedAds=session.generated_count,
        totalAds=session.target_ad_count,
        planRevision=session.plan_revision,
        retryMode=retry_mode if retry_mode != RETRY_MODE_NONE else RETRY_MODE_IMAGE_ONLY
        if session.failed_ad_index == expected_next_index
        else RETRY_MODE_NONE,
        retryAdIndex=expected_next_index,
    )
    _builder1_executor.submit(_builder1_run_next_job, job_id, campaign_id, expected_next_index)
    return (
        jsonify(
            {
                "status": "running",
                "jobId": job_id,
                "campaignId": campaign_id,
                "stage": "generating_images",
                "completedAds": session.generated_count,
                "totalAds": session.target_ad_count,
                "targetAdCount": session.target_ad_count,
                "pollUrl": f"/api/builder1-status?jobId={job_id}",
                **public_retry_fields(session=session, retry_ad_index=expected_next_index),
            }
        ),
        202,
    )


@app.route("/api/builder1-status", methods=["GET"])
def builder1_status():
    job_id = (request.args.get("jobId") or "").strip()
    if not job_id:
        return jsonify({"status": "error", "error": "missing_job_id"}), 400
    job = get_builder1_job(job_id)
    if not job:
        return jsonify({"status": "error", "jobId": job_id, "error": "not_found"}), 404
    status = (job.get("status") or "running").strip()
    logger.info("BUILDER1_JOB_STATUS jobId=%s status=%s", job_id, status)
    if status == "running":
        out: dict[str, Any] = {"status": "running", "jobId": job_id}
        if job.get("stage"):
            out["stage"] = job.get("stage")
        if job.get("totalAds") is not None:
            out["totalAds"] = job.get("totalAds")
        if job.get("completedAds") is not None:
            out["completedAds"] = job.get("completedAds")
        return jsonify(out), 200
    if status == "done":
        return jsonify({"status": "done", "jobId": job_id, "result": job.get("result") or {}}), 200
    if status == "error":
        out_err: dict[str, Any] = {
            "status": "error",
            "jobId": job_id,
            "error": job.get("error") or "builder1_generation_failed",
        }
        if job.get("retryable"):
            out_err["retryable"] = True
        if job.get("result"):
            out_err["result"] = job.get("result")
        return jsonify(out_err), 200
    return jsonify({"status": "error", "jobId": job_id, "error": job.get("error") or "builder1_generation_failed"}), 200


@app.route("/api/builder1-real-image-demo", methods=["GET"])
def builder1_real_image_demo():
    return (
        jsonify(
            {
                "ok": False,
                "error": "demo_disabled",
                "message": "Use POST /api/builder1-generate for campaign-series generation.",
            }
        ),
        200,
    )


@app.route("/api/builder1-real-image-view", methods=["GET"])
def builder1_real_image_view():
    raw = builder1_real_image_demo()
    resp = raw[0] if isinstance(raw, tuple) else raw
    data = resp.get_json(silent=True) or {}
    if data.get("ok") is False or not data.get("imageBytesBase64"):
        err = data.get("error", "error")
        if data.get("details"):
            err = f"{err}: {data['details']}"
        return Response(
            f"<html><body><p>{html.escape(err)}</p></body></html>",
            mimetype="text/html",
        )
    b64 = data["imageBytesBase64"]
    return Response(
        '<html>\n'
        '  <body style="background:white; display:flex; justify-content:center; align-items:center; height:100vh;">\n'
        f'    <img src="data:image/png;base64,{b64}" style="max-width:90%; max-height:90%;" />\n'
        "  </body>\n"
        "</html>",
        mimetype="text/html",
    )


@app.route("/api/builder2-download-zip", methods=["POST"])
def builder2_download_zip():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input"}), 400

    video_url = (body.get("videoUrl") or "").strip()
    marketing_text = (body.get("marketingText") or "").strip()
    if not video_url:
        return jsonify({"ok": False, "error": "missing_video_url"}), 400

    try:
        resp = httpx.get(video_url, timeout=httpx.Timeout(120.0), follow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            return jsonify({"ok": False, "error": "video_download_failed"}), 400
        zip_bytes = build_builder2_video_zip_bytes(resp.content, marketing_text)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "video_download_failed"}), 400

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name="builder2-video.zip",
    )


@app.route("/api/download-video-zip", methods=["GET"])
def download_video_zip():
    video_url = (request.args.get("videoUrl") or "").strip()
    marketing_text = request.args.get("text") or ""
    logger.info("DOWNLOAD_VIDEO_ZIP_START")
    logger.info("DOWNLOAD_VIDEO_ZIP_START videoUrl=%s", video_url[:300])
    if not video_url:
        logger.info("DOWNLOAD_VIDEO_ZIP_MISSING_VIDEO_URL")
        return jsonify({"ok": False, "error": "missing_video_url"}), 400

    video_bytes = b""
    local_match = re.search(r"/api/video-headline/([a-f0-9]{32})(?:[/?#]|$)", video_url)
    if local_match:
        token = (local_match.group(1) or "").strip()
        logger.info("DOWNLOAD_VIDEO_ZIP_LOCAL_TOKEN_DETECTED token_prefix=%s", token[:8])
        path = get_headline_video_path(token)
        if not path or not path.is_file():
            logger.warning("DOWNLOAD_VIDEO_ZIP_LOCAL_READ_FAILED reason=not_found_or_invalid_token")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "video_download_failed",
                        "status": None,
                        "contentType": "",
                        "urlPrefix": video_url[:120],
                    }
                ),
                500,
            )
        try:
            video_bytes = path.read_bytes()
            if not video_bytes:
                logger.warning("DOWNLOAD_VIDEO_ZIP_LOCAL_READ_FAILED reason=empty_file")
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "video_download_failed",
                            "status": None,
                            "contentType": "",
                            "urlPrefix": video_url[:120],
                        }
                    ),
                    500,
                )
            logger.info("DOWNLOAD_VIDEO_ZIP_LOCAL_READ_OK bytes=%s", len(video_bytes))
        except Exception as e:
            logger.warning("DOWNLOAD_VIDEO_ZIP_LOCAL_READ_FAILED reason=%s", type(e).__name__)
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "video_download_failed",
                        "status": None,
                        "contentType": "",
                        "urlPrefix": video_url[:120],
                    }
                ),
                500,
            )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "video/mp4,video/*,*/*",
    }
    status_code: int | None = None
    content_type = ""

    def _response_text_preview(resp: requests.Response) -> str:
        ct = (resp.headers.get("Content-Type") or "").lower()
        if any(k in ct for k in ("text/", "application/json", "application/xml", "application/javascript")):
            try:
                return (resp.text or "")[:300]
            except Exception:
                return ""
        return ""

    if not video_bytes:
        try:
            resp = requests.get(
                video_url,
                headers=headers,
                timeout=120,
                stream=True,
                allow_redirects=True,
            )
            status_code = resp.status_code
            content_type = (resp.headers.get("Content-Type") or "").strip()
            chunks: list[bytes] = []
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                chunks.append(chunk)
            video_bytes = b"".join(chunks)
            if status_code != 200 or not video_bytes:
                preview = _response_text_preview(resp)
                logger.warning(
                    "DOWNLOAD_VIDEO_ZIP_FETCH_FAILED status=%s content_type=%s has_content=%s body_preview=%s",
                    status_code,
                    content_type,
                    bool(video_bytes),
                    json.dumps(preview, ensure_ascii=False),
                )
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "video_download_failed",
                            "status": status_code,
                            "contentType": content_type,
                            "urlPrefix": video_url[:120],
                        }
                    ),
                    500,
                )
            logger.info("DOWNLOAD_VIDEO_ZIP_FETCH_OK")
        except requests.RequestException as e:
            resp = getattr(e, "response", None)
            status_code = getattr(resp, "status_code", None)
            content_type = (resp.headers.get("Content-Type") or "").strip() if resp is not None else ""
            preview = _response_text_preview(resp) if resp is not None else ""
            logger.warning(
                "DOWNLOAD_VIDEO_ZIP_FETCH_FAILED exc_type=%s err=%s status=%s content_type=%s body_preview=%s",
                type(e).__name__,
                str(e),
                status_code,
                content_type,
                json.dumps(preview, ensure_ascii=False),
            )
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "video_download_failed",
                        "status": status_code,
                        "contentType": content_type,
                        "urlPrefix": video_url[:120],
                    }
                ),
                500,
            )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.mp4", video_bytes)
        zf.writestr("marketing_text.txt", marketing_text)
    zip_size = zip_buffer.tell()
    zip_buffer.seek(0)

    logger.info("DOWNLOAD_VIDEO_ZIP_RETURN_OK bytes=%s", zip_size)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="ace-video-ad.zip",
    )


@app.route("/api/builder1-zip", methods=["POST"])
@app.route("/api/builder1-download-zip", methods=["POST"])
def builder1_download_zip():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input"}), 400

    try:
        zip_bytes, download_name = build_builder1_zip_from_request(body)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    response = Response(zip_bytes, mimetype="application/zip")
    response.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return response


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint - minimal, no heavy imports, returns plain text.
    This endpoint must NOT trigger any ACE/OpenAI initialization.
    """
    return "ok", 200


def _log_video_headline_upload_routes_registered() -> None:
    """Confirm POST upload routes exist on this process (gunicorn import)."""
    needle = ("video-headline-artifact", "video-headline/<")
    for rule in app.url_map.iter_rules():
        r = str(rule.rule)
        if not any(n in r for n in needle):
            continue
        methods = sorted((rule.methods or set()) - {"OPTIONS", "HEAD"})
        logger.info(
            "VIDEO_HEADLINE_UPLOAD_ROUTE_REGISTERED rule=%s endpoint=%s methods=%s",
            r,
            rule.endpoint,
            methods,
        )
    try:
        with app.test_client() as client:
            tc = client.post("/api/video-headline-artifact")
            st = tc.status_code
        if st == 404:
            logger.error(
                "VIDEO_HEADLINE_UPLOAD_ROUTE_SELFTEST_FAIL path=/api/video-headline-artifact "
                "http_status=404 (route not registered in this process)"
            )
        else:
            logger.info(
                "VIDEO_HEADLINE_UPLOAD_ROUTE_READY path=/api/video-headline-artifact "
                "self_test_http_status=%s (404 means route missing; 401/503/400 means route exists)",
                st,
            )
    except Exception as ex:
        logger.error("VIDEO_HEADLINE_UPLOAD_ROUTE_SELFTEST_ERR err=%s", ex, exc_info=True)

    upload_reg = 0
    serve_reg = 0
    for rule in app.url_map.iter_rules():
        r = str(rule.rule)
        methods = rule.methods or set()
        if "video-headline-artifact" in r and "POST" in methods:
            upload_reg = 1
        if "<token>" in r and "video-headline" in r and "artifact" not in r and "GET" in methods:
            serve_reg = 1
    pub_base = (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    logger.info(
        "VIDEO_HEADLINE_ROUTE_BOOT upload_routes_registered=%s serve_routes_registered=%s public_base_url=%s",
        upload_reg,
        serve_reg,
        pub_base or "(unset)",
    )


_log_video_headline_upload_routes_registered()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("HEALTH_READY /health returns 200 immediately")
    app.run(host="0.0.0.0", port=port, debug=False)
