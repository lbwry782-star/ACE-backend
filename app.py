"""
Minimal Flask entrypoint: health + Runway/ACE video pipeline routes only.
Image ad generation, preview jobs, ZIP download, Builder1, and payment routes are omitted.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Optional

from flask import Flask, Response, jsonify, request, send_file

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

app = Flask(__name__)

# Configure logging early (handlers use logger at request time)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
