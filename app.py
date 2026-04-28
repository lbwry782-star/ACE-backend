"""
Minimal Flask entrypoint: health + Runway/ACE video pipeline routes only.
Image ad generation, preview jobs, ZIP download, Builder1, and payment routes are omitted.
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
from engine.builder1_generate_demo import build_demo_ad
from engine.builder2_zip import build_builder2_video_zip_bytes
from engine.builder1_composition import generate_builder1_composition_o3
from engine.builder1_headline import generate_builder1_headline_o3
from engine.builder1_image_generator import generate_builder1_image
from engine.builder1_marketing_text import generate_builder1_marketing_text_o3
from engine.builder1_planner import plan_builder1
from engine.builder1_zip import build_builder1_zip_bytes

app = Flask(__name__)

# Configure logging early (handlers use logger at request time)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_builder1_jobs_lock = threading.Lock()
_builder1_jobs: dict[str, dict[str, Any]] = {}
_builder1_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="builder1_job")

try:
    log_video_headline_delivery_startup("web")
except Exception as e:
    logger.warning("VIDEO_HEADLINE_UPLOAD_CONFIG web startup failed err=%s", e)


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
        try:
            video_job_create(job_id, product_name, product_description, base)
        except Exception as e:
            logger.error("VIDEO_JOB_REDIS_ENQUEUE_FAILED jobId=%s err=%s", job_id, e, exc_info=True)
            return jsonify({"ok": False, "error": "video_generation_failed"}), 200
        logger.info("VIDEO_JOB_CREATED jobId=%s", job_id)
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
    if status == "error":
        err = job.get("error") or "video_generation_failed"
        logger.info("VIDEO_JOB_POLL terminal_error jobId=%s reason=%s", job_id, err)
        out["error"] = err
    return jsonify(out), 200


@app.route("/api/builder1-demo", methods=["GET"])
def builder1_demo():
    try:
        result = build_demo_ad()
        p = result.plan
        return (
            jsonify(
                {
                    "productNameResolved": p.product_name_resolved,
                    "advertisingPromise": p.advertising_promise,
                    "objectA": p.object_a,
                    "objectB": p.object_b,
                    "modeDecision": p.mode_decision,
                    "visualPrompt": result.visual_prompt,
                    "imageBytesBase64": base64.b64encode(result.image_bytes).decode("ascii"),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 200


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


def _o3_pro_planning_model_caller(system_prompt: str, user_prompt: str) -> object:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(150.0),
        max_retries=0,
    )
    combined = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    response = client.responses.create(
        model="o3-pro",
        input=combined,
        reasoning={"effort": "low"},
    )
    out_text = getattr(response, "output_text", None) or ""
    return _parse_builder1_o3_json_text(out_text)


def _fake_image_bytes_caller(_prompt: str, _format_value: str) -> bytes:
    return b"demo-image-bytes"


@app.route("/api/builder1-plan-demo", methods=["GET"])
def builder1_plan_demo():
    try:
        product_name = (request.args.get("productName") or "AeroSip Bottle").strip()
        product_description = (
            request.args.get("productDescription")
            or "A lightweight bottle that feels effortless to carry all day."
        ).strip()
        format_val = (request.args.get("format") or "portrait").strip()
        p = plan_builder1(
            product_name=product_name,
            product_description=product_description,
            format_value=format_val,
            model_caller=_o3_pro_planning_model_caller,
        )
        image_result = generate_builder1_image(p, _fake_image_bytes_caller)
        return (
            jsonify(
                {
                    "productNameResolved": p.product_name_resolved,
                    "detectedLanguage": p.detected_language,
                    "advertisingPromise": p.advertising_promise,
                    "objectA": p.object_a,
                    "objectASecondary": p.object_a_secondary,
                    "objectB": p.object_b,
                    "visualSimilarityScore": p.visual_similarity_score,
                    "modeDecision": p.mode_decision,
                    "visualDescription": p.visual_description,
                    "visualPrompt": image_result.visual_prompt,
                    "imageBytesBase64": base64.b64encode(image_result.image_bytes).decode("ascii"),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200


def _builder1_gpt_image_size_for_format(format_value: str) -> str:
    f = (format_value or "").strip().lower()
    if f == "landscape":
        return "1536x1024"
    if f == "portrait":
        return "1024x1536"
    if f == "square":
        return "1024x1024"
    return "1024x1536"


def _gpt_image_15_caller(prompt: str, format_value: str) -> bytes:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    r = client.images.generate(
        model="gpt-image-1.5",
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


def _o3_pro_marketing_text_model_caller(system_prompt: str, user_prompt: str) -> str:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    combined = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    response = client.responses.create(
        model="o3-pro",
        input=combined,
        reasoning={"effort": "low"},
    )
    return (getattr(response, "output_text", None) or "").strip()


def _builder1_json_real_generate(
    product_name: str, product_description: str, format_val: str
) -> dict[str, Any]:
    try:
        p = plan_builder1(
            product_name=product_name,
            product_description=product_description,
            format_value=format_val,
            model_caller=_o3_pro_planning_model_caller,
        )
    except Exception as e:
        return {"ok": False, "error": "planning_failed", "message": str(e)}
    try:
        image_result = generate_builder1_image(p, _gpt_image_15_caller)
    except Exception as e:
        return {
            "ok": False,
            "error": "builder1_image_call_failed",
            "message": str(e),
        }
    try:
        headline = generate_builder1_headline_o3(
            product_name_resolved=p.product_name_resolved,
            detected_language=p.detected_language,
            advertising_promise=p.advertising_promise,
            object_a=p.object_a,
            object_a_secondary=p.object_a_secondary,
            object_b=p.object_b,
            mode_decision=p.mode_decision,
            visual_description=p.visual_description,
            visual_prompt=image_result.visual_prompt,
        )
    except Exception as e:
        return {"ok": False, "error": "headline_failed", "message": str(e)}
    try:
        composition = generate_builder1_composition_o3(
            format=p.format,
            detectedLanguage=p.detected_language,
            productNameResolved=p.product_name_resolved,
            headlineProductName=headline["headlineProductName"],
            headlineText=headline["headlineText"],
            headlineFull=headline["headlineFull"],
            objectA=p.object_a,
            objectASecondary=p.object_a_secondary,
            objectB=p.object_b,
            modeDecision=p.mode_decision,
            visualDescription=p.visual_description,
            visualPrompt=image_result.visual_prompt,
        )
    except Exception as e:
        return {"ok": False, "error": "composition_failed", "message": str(e)}
    marketing_text = ""
    try:
        marketing_text = generate_builder1_marketing_text_o3(
            product_name_resolved=p.product_name_resolved,
            product_description=p.product_description,
            detected_language=p.detected_language,
            advertising_promise=p.advertising_promise,
            object_a=p.object_a,
            object_a_secondary=p.object_a_secondary,
            object_b=p.object_b,
            mode_decision=p.mode_decision,
            visual_description=p.visual_description,
            visual_prompt=image_result.visual_prompt,
            headline_product_name=headline["headlineProductName"],
            headline_text=headline["headlineText"],
            headline_full=headline["headlineFull"],
            model_caller=_o3_pro_marketing_text_model_caller,
        )
    except Exception as e:
        logger.warning("BUILDER1_MARKETING_TEXT_FAILED error=%r", str(e))
    return {
        "ok": True,
        "productNameResolved": p.product_name_resolved,
        "detectedLanguage": p.detected_language,
        "advertisingPromise": p.advertising_promise,
        "objectA": p.object_a,
        "objectASecondary": p.object_a_secondary,
        "objectB": p.object_b,
        "visualSimilarityScore": p.visual_similarity_score,
        "modeDecision": p.mode_decision,
        "visualDescription": p.visual_description,
        "visualPrompt": image_result.visual_prompt,
        "imageBase64": base64.b64encode(image_result.image_bytes).decode("ascii"),
        "marketingText": marketing_text,
        "headlineProductName": headline["headlineProductName"],
        "headlineText": headline["headlineText"],
        "headlineFull": headline["headlineFull"],
        "compositionLayout": composition["compositionLayout"],
        "headlineAlign": composition["headlineAlign"],
        "headlineLines": composition["headlineLines"],
        "headlineRelativeSize": composition["headlineRelativeSize"],
        "visualWeight": composition["visualWeight"],
        "headlineWeight": composition["headlineWeight"],
        "safeMarginRule": composition["safeMarginRule"],
        "safeMarginCss": composition["safeMarginCss"],
        "headlineSizeRule": composition["headlineSizeRule"],
        "productNameScale": composition["productNameScale"],
        "headlineTextScale": composition["headlineTextScale"],
        "compositionNotes": composition["compositionNotes"],
    }


def _builder1_run_job(job_id: str, product_name: str, product_description: str, format_val: str) -> None:
    logger.info("BUILDER1_JOB_STARTED jobId=%s", job_id)
    try:
        result = _builder1_json_real_generate(product_name, product_description, format_val)
        with _builder1_jobs_lock:
            if result.get("ok") is True:
                _builder1_jobs[job_id] = {"status": "done", "result": result}
                logger.info("BUILDER1_JOB_DONE jobId=%s", job_id)
            else:
                err = (result.get("error") or "builder1_generation_failed")
                _builder1_jobs[job_id] = {"status": "error", "error": err}
                logger.error("BUILDER1_JOB_ERROR jobId=%s err=%s", job_id, err)
    except Exception as e:
        with _builder1_jobs_lock:
            _builder1_jobs[job_id] = {"status": "error", "error": "builder1_generation_failed"}
        logger.error("BUILDER1_JOB_ERROR jobId=%s err=%s", job_id, e, exc_info=True)


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
    if not product_description:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "invalid_input",
                    "message": "productDescription is required",
                }
            ),
            200,
        )
    job_id = str(uuid.uuid4())
    with _builder1_jobs_lock:
        _builder1_jobs[job_id] = {"status": "running"}
    logger.info("BUILDER1_JOB_CREATED jobId=%s", job_id)
    _builder1_executor.submit(_builder1_run_job, job_id, product_name, product_description, format_val)
    return (
        jsonify(
            {
                "status": "running",
                "jobId": job_id,
                "pollUrl": f"/api/builder1-status?jobId={job_id}",
            }
        ),
        202,
    )


@app.route("/api/builder1-status", methods=["GET"])
def builder1_status():
    job_id = (request.args.get("jobId") or "").strip()
    if not job_id:
        return jsonify({"status": "error", "error": "missing_job_id"}), 400
    with _builder1_jobs_lock:
        job = _builder1_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "jobId": job_id, "error": "not_found"}), 404
    status = (job.get("status") or "running").strip()
    logger.info("BUILDER1_JOB_STATUS jobId=%s status=%s", job_id, status)
    if status == "running":
        return jsonify({"status": "running", "jobId": job_id}), 200
    if status == "done":
        return jsonify({"status": "done", "jobId": job_id, "result": job.get("result") or {}}), 200
    return jsonify({"status": "error", "jobId": job_id, "error": job.get("error") or "builder1_generation_failed"}), 200


@app.route("/api/builder1-real-image-demo", methods=["GET"])
def builder1_real_image_demo():
    try:
        product_name = (request.args.get("productName") or "AeroSip Bottle").strip()
        product_description = (
            request.args.get("productDescription")
            or "A lightweight bottle that feels effortless to carry all day."
        ).strip()
        format_val = (request.args.get("format") or "portrait").strip()
        p = plan_builder1(
            product_name=product_name,
            product_description=product_description,
            format_value=format_val,
            model_caller=_o3_pro_planning_model_caller,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
    try:
        image_result = generate_builder1_image(p, _gpt_image_15_caller)
    except Exception as e:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "builder1_image_call_failed",
                    "details": str(e),
                }
            ),
            200,
        )
    return (
        jsonify(
            {
                "productNameResolved": p.product_name_resolved,
                "detectedLanguage": p.detected_language,
                "advertisingPromise": p.advertising_promise,
                "objectA": p.object_a,
                "objectASecondary": p.object_a_secondary,
                "objectB": p.object_b,
                "visualSimilarityScore": p.visual_similarity_score,
                "modeDecision": p.mode_decision,
                "visualDescription": p.visual_description,
                "visualPrompt": image_result.visual_prompt,
                "imageBytesBase64": base64.b64encode(image_result.image_bytes).decode("ascii"),
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


@app.route("/api/builder1-download-zip", methods=["POST"])
def builder1_download_zip():
    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input"}), 400

    image_base64 = body.get("imageBase64") or ""
    marketing_text = body.get("marketingText") or ""
    try:
        zip_bytes = build_builder1_zip_bytes(image_base64, marketing_text)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    response = Response(zip_bytes, mimetype="application/zip")
    response.headers["Content-Disposition"] = 'attachment; filename="builder1-ad.zip"'
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
