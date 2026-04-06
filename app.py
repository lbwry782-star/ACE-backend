from flask import Flask, request, jsonify, send_file, Response
import uuid
import logging
import time
import io
import zipfile
import json
import base64
import os
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from engine.side_by_side_v1 import (
    generate_preview_data,
    Step0BundleTimeoutError,
    Step0BundleOpenAIError,
    create_goal_pair_background,
    poll_goal_pair_response,
    cancel_goal_pair_response,
    GOAL_PAIR_BG_MAX_WAIT_SECONDS,
    GOAL_PAIR_BG_POLL_INTERVAL_SECONDS,
    GOAL_PAIR_MIN_SIMILARITY_ACCEPT,
    GOAL_PAIR_RETRY_INSTRUCTION,
)
from engine.openai_retry import OpenAIRateLimitError
from engine.video_headline_postprocess import (
    get_headline_video_path,
    hard_test_video_path,
    write_headline_video_bytes,
    log_video_headline_delivery_startup,
)
from engine.video_jobs_redis import redis_configured, video_job_create, video_job_get
from engine.video_web_postprocess import ensure_video_postprocessed_for_poll
import db_session

app = Flask(__name__)


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
# CORS Configuration (Manual Implementation)
# ============================================================================

# Allowed origins (exact match)
ALLOWED_ORIGINS = [
    "https://ace-advertising.agency",
    "https://www.ace-advertising.agency"
]

def is_origin_allowed(origin: Optional[str]) -> bool:
    """Check if origin is in allowed list."""
    if not origin:
        return False
    return origin in ALLOWED_ORIGINS

@app.before_request
def handle_preflight():
    """Handle OPTIONS preflight requests for CORS."""
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        allow_origin = is_origin_allowed(origin) if is_security_enabled() else bool(origin)
        if allow_origin and origin:
            response = Response('', status=200)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-ACE-Batch-State"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        return Response('', status=200)

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to all /api/* responses.
    When ACE_SECURITY_ENABLED=false, any provided Origin is allowed (bypass origin check).
    """
    if request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        allow_origin = is_origin_allowed(origin) if is_security_enabled() else bool(origin)
        if allow_origin and origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-ACE-Batch-State"
            if "X-ACE-Batch-State" in response.headers:
                response.headers["Access-Control-Expose-Headers"] = "X-ACE-Batch-State"
    return response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    log_video_headline_delivery_startup("web")
except Exception as e:
    logger.warning("VIDEO_HEADLINE_UPLOAD_CONFIG web startup failed err=%s", e)

# ACE_TEST_MODE: "1" or "true" = no OpenAI, return demo result immediately
ACE_TEST_MODE = (os.environ.get("ACE_TEST_MODE", "") or "").strip().lower() in ("1", "true")
# ACE_IMAGE_ONLY: "1" or "true" = gpt-image-1.5 only, no o3-pro, placeholder copy
ACE_IMAGE_ONLY = (os.environ.get("ACE_IMAGE_ONLY", "") or "").strip().lower() in ("1", "true")
# ACE_PHASE2_GOAL_PAIRS: "1" or "true" or "yes" = when IMAGE_ONLY, call o3-pro for goal + 3 pairs
ACE_PHASE2_GOAL_PAIRS = (os.environ.get("ACE_PHASE2_GOAL_PAIRS", "") or "").strip().lower() in ("1", "true", "yes")
# ACE_FALLBACK_RETURN_ERROR: when Stage 2 fails (FALLBACK_USED), return error instead of generating 1 fallback ad (cost control)
ACE_FALLBACK_RETURN_ERROR = (os.environ.get("ACE_FALLBACK_RETURN_ERROR", "") or "").strip().lower() in ("1", "true", "yes")

# ACE_SECURITY_ENABLED (Render ENV): controls Builder security. Payment/IPN unchanged.
#   "true" or missing: security on — direct Builder, refresh, extra tab, incognito redirect to Preview; only post-payment may enter Builder.
#   "false": security off — allow Builder access without enforcing those redirects.
# Wrapped: CORS origin check, GET /api/security-status, GET /api/entitlement/latest-paid. IPN is never bypassed.
def is_security_enabled() -> bool:
    """Read ACE_SECURITY_ENABLED from env. Only 'false' disables; missing or 'true' = enabled. Default True."""
    raw = (os.environ.get("ACE_SECURITY_ENABLED", "") or "").strip().lower()
    if raw == "false":
        return False
    return True

if ACE_TEST_MODE:
    logger.info("TEST_MODE_ACTIVE=true")
if not is_security_enabled():
    logger.info("ACE_SECURITY_ENABLED=false security_checks_bypassed=true")

# Small placeholder PNG (1x1 transparent) as base64 - no OpenAI image call
_TEST_PLACEHOLDER_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def _get_test_mode_demo_result():
    """Demo result when ACE_TEST_MODE is on. No engine/OpenAI calls."""
    return {
        "imageBase64": _TEST_PLACEHOLDER_PNG_BASE64,
        "image_base64": _TEST_PLACEHOLDER_PNG_BASE64,
        "headline": "ACE TEST MODE DEMO",
        "bodyText50": "This is a placeholder body text for ACE test mode. No OpenAI calls were made. You can develop and test the app with images and copy visible immediately. About fifty words of sample content to fill the body text area for layout and UI checks.",
        "body_text": "This is a placeholder body text for ACE test mode. No OpenAI calls were made. You can develop and test the app with images and copy visible immediately. About fifty words of sample content to fill the body text area for layout and UI checks.",
        "image_url": None,
        "ad_goal": "Demo ad goal for test mode",
        "object_a": "object_a",
        "object_b": "object_b",
        "marketing_copy_50_words": "This is a placeholder body text for ACE test mode. No OpenAI calls were made. About fifty words of sample content.",
    }

# ENGINE REMOVED: get_engine_functions() deleted - engine module removed

# Allowed image sizes
ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]

# Per-session lock: only one ad generation at a time per sessionId (serial execution)
_session_locks: Dict[str, Lock] = {}
_session_locks_guard = Lock()

# Artifact store for generate: (sessionId, adIndex) -> { image_bytes, bodyText50, headline } for ZIP download without regenerating
# TTL 45 minutes
_GENERATE_ARTIFACT_TTL_SECONDS = 45 * 60
_generate_artifacts: Dict[tuple, dict] = {}  # (session_id, ad_index) -> { "image_bytes", "bodyText50", "headline", "timestamp" }
_generate_artifacts_guard = Lock()

# SessionId alias: request-provided sessionId -> internal sessionId used for storage (so download works when frontend sends a different id)
_sid_alias_map: Dict[str, str] = {}
_sid_alias_lock = Lock()


def _set_sid_alias(requested_sid: str, resolved_sid: str) -> None:
    """Map request-provided sessionId to the internal sessionId used when storing artifacts (for download fallback lookup)."""
    if not (requested_sid and str(requested_sid).strip()):
        return
    with _sid_alias_lock:
        _sid_alias_map[str(requested_sid).strip()] = resolved_sid


def _get_sid_alias(requested_sid: str) -> Optional[str]:
    """Resolve request-provided sessionId to internal sessionId if we have a mapping."""
    if not (requested_sid and str(requested_sid).strip()):
        return None
    with _sid_alias_lock:
        return _sid_alias_map.get(str(requested_sid).strip())

# In-memory async job store for /api/preview (background execution + polling)
_JOB_TTL_SECONDS = 30 * 60
_jobs: Dict[str, dict] = {}  # jobId -> { status, created_at, finished_at, session_id, ad_index, result, error, error_message }
_jobs_lock = Lock()
_preview_executor = ThreadPoolExecutor(max_workers=1)

def _get_artifact(session_id: str, ad_index: int):
    """Get stored artifact for (session_id, ad_index). Returns None if missing or expired."""
    key = (session_id, ad_index)
    with _generate_artifacts_guard:
        if key not in _generate_artifacts:
            return None
        entry = _generate_artifacts[key]
        if time.time() - entry["timestamp"] > _GENERATE_ARTIFACT_TTL_SECONDS:
            del _generate_artifacts[key]
            return None
        return entry


def _set_artifact(session_id: str, ad_index: int, image_bytes: bytes, body_text_50: str, headline: str) -> None:
    """Store artifact for later ZIP download."""
    key = (session_id, ad_index)
    with _generate_artifacts_guard:
        _generate_artifacts[key] = {
            "image_bytes": image_bytes,
            "bodyText50": body_text_50,
            "headline": headline,
            "timestamp": time.time(),
        }


def _cleanup_jobs() -> None:
    """Remove expired jobs from in-memory store."""
    now = time.time()
    with _jobs_lock:
        expired_ids = [jid for jid, job in _jobs.items() if now - job.get("created_at", now) > _JOB_TTL_SECONDS]
        for jid in expired_ids:
            _jobs.pop(jid, None)


def _get_job(job_id: str) -> Optional[dict]:
    _cleanup_jobs()
    with _jobs_lock:
        return _jobs.get(job_id)


def _set_job(job_id: str, data: dict) -> None:
    with _jobs_lock:
        _jobs[job_id] = data


def _acquire_session_lock(session_id: str, ad_index: int) -> bool:
    """Try to acquire the lock for this session. Returns True if acquired, False if already held."""
    with _session_locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = Lock()
        lock = _session_locks[session_id]
    acquired = lock.acquire(blocking=False)
    if acquired:
        logger.info(f"SESSION_LOCK acquired=true sid={session_id} adIndex={ad_index}")
    return acquired


def _release_session_lock(session_id: str, ad_index: int) -> None:
    """Release the lock for this session."""
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
    if lock is not None:
        try:
            lock.release()
        except RuntimeError:
            pass
    logger.info(f"SESSION_LOCK released=true sid={session_id} adIndex={ad_index}")

# SESSION SYSTEM REMOVED: All entitlement/session functions deleted


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate a single ad: sketch image + headline + 50-word body. Returns JSON for display;
    stores artifacts for later ZIP download (no OpenAI on download).
    
    Request JSON:
    {
        "productName": string (optional; if empty/missing, backend invents one in the reasoning flow),
        "productDescription": string (required),
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536" (optional),
        "adIndex": int (optional, 1-3),
        "sessionId": string (optional)
    }
    
    Returns: 200 JSON { ok: true, sessionId, adIndex, imageBase64, headline, bodyText50 }
    Or 409 if generation already in progress for this session.
    """
    request_id = str(uuid.uuid4())
    try:
        if not request.is_json:
            return jsonify({'ok': False, 'error': 'invalid_request', 'message': 'Request must be JSON'}), 400
        payload = request.get_json()
        if not payload.get("productDescription"):
            return jsonify({'ok': False, 'error': 'missing_field', 'message': 'productDescription is required'}), 400
        # Use client sessionId as canonical everywhere; only generate new UUID when request truly has no sessionId
        client_sid = (payload.get("sessionId") or "").strip()
        if client_sid:
            session_id = client_sid
            sid_source = "client"
        else:
            session_id = str(uuid.uuid4())
            sid_source = "generated"
        logger.info(f"GEN_REQUEST_SID_SOURCE source={sid_source} request_id={request_id}")
        logger.info(f"GEN_REQUEST_SID_VALUE sessionId={session_id} request_id={request_id}")
        logger.info(f"SESSION_ID_RESOLVED sessionId={session_id} source={sid_source} request_id={request_id}")
        ad_index = int(payload.get("adIndex", 1))
        ad_index = max(1, min(3, ad_index))

        # ACE_TEST_MODE: bypass engine; return demo and store artifact for download-zip
        if ACE_TEST_MODE:
            logger.info("TEST_MODE_RETURNING_DEMO_RESULT=true")
            demo = _get_test_mode_demo_result()
            image_bytes = base64.b64decode(demo["imageBase64"])
            _set_artifact(session_id, ad_index, image_bytes, demo["bodyText50"], demo["headline"])
            if client_sid:
                _set_sid_alias(client_sid, session_id)
            logger.info(f"RESULT_STORED sessionId={session_id} adIndex={ad_index} bytes={len(image_bytes)} request_id={request_id}")
            return jsonify({
                "ok": True,
                "sessionId": session_id,
                "adIndex": ad_index,
                "imageBase64": demo["imageBase64"],
                "headline": demo["headline"],
                "bodyText50": demo["bodyText50"],
            }), 200

        logger.info(f"GENERATE_FLAGS TEST_MODE=false IMAGE_ONLY={ACE_IMAGE_ONLY} size={payload.get('imageSize', '')} adIndex={ad_index} sessionId={session_id}")
        if not _acquire_session_lock(session_id, ad_index):
            return jsonify({'ok': False, 'error': 'busy', 'message': 'Generation already in progress'}), 409
        logger.info(f"GENERATE_START sid={session_id} ad={ad_index}")
        try:
            # Build payload for preview with image (same flow as generate: image + headline + body)
            gen_payload = {
                "productName": payload.get("productName") or "",
                "productDescription": payload["productDescription"],
                "imageSize": payload.get("imageSize", "1536x1024"),
                "adIndex": ad_index,
                "sessionId": session_id,
                "includeImage": True,
            }
            result = generate_preview_data(gen_payload)
            image_base64 = result["imageBase64"]
            headline = result.get("headline", "")
            body_text_50 = result.get("bodyText50", "")
            image_bytes = base64.b64decode(image_base64)
            _set_artifact(session_id, ad_index, image_bytes, body_text_50, headline)
            if client_sid:
                _set_sid_alias(client_sid, session_id)
            logger.info(f"RESULT_STORED sessionId={session_id} adIndex={ad_index} bytes={len(image_bytes)} request_id={request_id}")
            logger.info(f"GENERATE_DONE sid={session_id} ad={ad_index} stored=true")
            resolved_pn_gen = result.get("resolvedProductName", "")
            logger.info(f"PRODUCT_NAME_RESPONSE_STAGE=generate resolvedProductName=\"{resolved_pn_gen}\"")
            logger.info("PRODUCT_NAME_RESPONSE_SOURCE=canonical_final_name")
            return jsonify({
                "ok": True,
                "sessionId": session_id,
                "adIndex": ad_index,
                "imageBase64": image_base64,
                "headline": headline,
                "bodyText50": body_text_50,
                "resolvedProductName": resolved_pn_gen,
            }), 200
        finally:
            _release_session_lock(session_id, ad_index)
    except OpenAIRateLimitError as e:
        return jsonify({'ok': False, 'error': 'rate_limited', 'message': e.message}), 503
    except Step0BundleTimeoutError:
        return jsonify({'ok': False, 'error': 'timeout', 'step': 'step0_bundle', 'message': 'Step0 bundle timed out'}), 504
    except Step0BundleOpenAIError as e:
        return jsonify({'ok': False, 'error': 'openai_error', 'step': 'step0_bundle', 'message': str(e)}), 500
    except ValueError as e:
        return jsonify({'ok': False, 'error': 'validation_error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"[{request_id}] Generate failed: {e}", exc_info=True)
        if "rate_limited" in str(e):
            return jsonify({'ok': False, 'error': 'rate_limited', 'message': 'Temporarily rate limited. Please retry.'}), 503
        return jsonify({'ok': False, 'error': 'generation_failed', 'message': str(e)}), 500
    


@app.route('/api/preview', methods=['POST'])
def preview():
    """
    Generate a single ad preview and return as JSON.
    
    By default returns text-only (no image) for low cost.
    Set "includeImage": true in the request body to generate an image.
    
    Request JSON:
    {
        "productName": string (optional; if empty/missing, backend invents one in the reasoning flow),
        "productDescription": string (required),
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536" (optional),
        "adIndex": int (optional, 1-3),
        "sessionId": string (optional),
        "includeImage": boolean (optional, default false = text-only)
    }
    
    Returns: 200 JSON with ad_goal, object_a, object_b, headline, marketing_copy_50_words, etc.
    """
    request_id = str(uuid.uuid4())
    try:
        if not request.is_json:
            logger.warning(f"[{request_id}] Preview request is not JSON")
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': 'Request must be JSON'
            }), 400
        payload = request.get_json()
        if not payload.get("productDescription"):
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productDescription is required'
            }), 400
        # Use client sessionId as canonical everywhere; only generate new UUID when request truly has no sessionId
        client_sid = (payload.get("sessionId") or "").strip()
        if client_sid:
            session_id = client_sid
            sid_source = "client"
        else:
            session_id = str(uuid.uuid4())
            sid_source = "generated"
        logger.info(f"GEN_REQUEST_SID_SOURCE source={sid_source} request_id={request_id}")
        logger.info(f"GEN_REQUEST_SID_VALUE sessionId={session_id} request_id={request_id}")
        logger.info(f"SESSION_ID_RESOLVED sessionId={session_id} source={sid_source} request_id={request_id}")
        ad_index = int(payload.get("adIndex", 1) or 1)
        ad_index = max(1, min(3, ad_index))

        # ACE_TEST_MODE: bypass engine and job; return finished demo result immediately
        if ACE_TEST_MODE:
            logger.info("TEST_MODE_RETURNING_DEMO_RESULT=true")
            demo = _get_test_mode_demo_result()
            return jsonify({
                "ok": True,
                "status": "done",
                "result": demo,
            }), 200

        logger.info(f"PREVIEW_FLAGS TEST_MODE=false IMAGE_ONLY={ACE_IMAGE_ONLY} PHASE2_GOAL_PAIRS={ACE_PHASE2_GOAL_PAIRS} FALLBACK_RETURN_ERROR={ACE_FALLBACK_RETURN_ERROR} size={payload.get('imageSize', '')} adIndex={ad_index} sessionId={session_id}")
        # Enforce serial execution per session: acquire lock before scheduling job
        if not _acquire_session_lock(session_id, ad_index):
            return jsonify({
                'ok': False,
                'error': 'busy',
                'message': 'Generation already in progress'
            }), 409

        # Create async job
        job_id = str(uuid.uuid4())
        created_at = time.time()
        job_record = {
            "jobId": job_id,
            "status": "pending",
            "created_at": created_at,
            "finished_at": None,
            "session_id": session_id,
            "requested_session_id": client_sid or session_id,
            "ad_index": ad_index,
            "result": None,
            "error": None,
            "error_message": None,
            # Phase 2D: background GOAL_PAIR state (one request_id per job for all logs)
            "request_id": request_id,
            "openai_response_id": None,
            "openai_goal_pair_response_id": None,
            "goal_pair_created_at": None,
            "goal_pairs_data": None,
            "goal_pair_fallback": False,
            "goal_pair_poll_attempt": 0,
            "goal_pair_next_poll_time_ms": 0,
            "goal_pair_retry_done": False,
            "resolved_product_name": (payload.get("productName") or "").strip(),
        }
        _cleanup_jobs()
        _set_job(job_id, job_record)
        logger.info(f"JOB_CREATED jobId={job_id} sid={session_id} ad={ad_index} request_id={request_id}")

        # Schedule background execution
        def _run_preview_job(jid: str, payload_data: dict, sid: str, ad_idx: int, req_id: str, req_sid: str = "") -> None:
            start = time.time()
            try:
                with _jobs_lock:
                    job = _jobs.get(jid)
                    if not job:
                        return
                    job["status"] = "running"
                    job_request_id = job.get("request_id") or req_id
                flags = f"ACE_PHASE2_GOAL_PAIRS={1 if ACE_PHASE2_GOAL_PAIRS else 0}, ACE_IMAGE_ONLY={1 if ACE_IMAGE_ONLY else 0}, ACE_FALLBACK_RETURN_ERROR={1 if ACE_FALLBACK_RETURN_ERROR else 0}"
                logger.info(
                    f'INPUT_SNAPSHOT productName="{payload_data.get("productName", "") or ""}" '
                    f'productDescription="{str(payload_data.get("productDescription", "") or "")[:200]}" '
                    f'imageSize="{payload_data.get("imageSize", "") or ""}" flags="{flags}" request_id={job_request_id}'
                )
                try:
                    if ACE_PHASE2_GOAL_PAIRS and ACE_IMAGE_ONLY:
                        product_name = payload_data.get("productName", "") or "product"
                        product_description = payload_data.get("productDescription", "") or "description"
                        response_id = create_goal_pair_background(product_name, product_description, job_request_id)
                        if not response_id:
                            with _jobs_lock:
                                j = _jobs.get(jid)
                                if j is not None:
                                    j["goal_pair_fallback"] = True
                        elif response_id:
                            with _jobs_lock:
                                j = _jobs.get(jid)
                                if j is not None:
                                    j["openai_response_id"] = response_id
                                    j["openai_goal_pair_response_id"] = response_id
                                    j["goal_pair_created_at"] = time.time()
                                    j["goal_pair_poll_attempt"] = 0
                                    j["goal_pair_next_poll_time_ms"] = 0
                            while True:
                                with _jobs_lock:
                                    j = _jobs.get(jid)
                                    if j is None:
                                        break
                                    rid = j.get("openai_goal_pair_response_id") or j.get("openai_response_id")
                                    created_at_ts = j.get("goal_pair_created_at") or 0
                                    poll_attempt = j.get("goal_pair_poll_attempt", 0)
                                    next_poll_time_ms = j.get("goal_pair_next_poll_time_ms", 0)
                                    job_request_id = j.get("request_id") or job_request_id
                                if not rid:
                                    break
                                if time.time() - created_at_ts > GOAL_PAIR_BG_MAX_WAIT_SECONDS:
                                    total_wait_s = int(time.time() - created_at_ts)
                                    logger.info(f"GOAL_PAIR_BG_FAIL status=timeout total_wait_s={total_wait_s} max_wait_s={GOAL_PAIR_BG_MAX_WAIT_SECONDS} FALLBACK_USED=true request_id={job_request_id}")
                                    cancel_goal_pair_response(rid, job_request_id)
                                    j["goal_pair_fallback"] = True
                                    break
                                now_ms = int(time.time() * 1000)
                                if next_poll_time_ms and now_ms < next_poll_time_ms:
                                    wait_ms = next_poll_time_ms - now_ms
                                    logger.info(f"GOAL_PAIR_BG_POLL_SKIPPED wait_ms={wait_ms} request_id={job_request_id}")
                                    time.sleep(wait_ms / 1000.0)
                                    continue
                                logger.info(f"GOAL_PAIR_BG_POLL_CALL attempt={poll_attempt + 1} request_id={job_request_id}")
                                goal_data, status = poll_goal_pair_response(rid, job_request_id, created_at_ts)
                                with _jobs_lock:
                                    j = _jobs.get(jid)
                                    if j is None:
                                        break
                                    new_attempt = poll_attempt + 1
                                    # Progressive backoff 2s → 3s → 5s → 8s → 10s (cap 10s)
                                    if new_attempt == 1:
                                        delay_ms = 2000
                                    elif new_attempt == 2:
                                        delay_ms = 3000
                                    elif new_attempt == 3:
                                        delay_ms = 5000
                                    elif new_attempt == 4:
                                        delay_ms = 8000
                                    else:
                                        delay_ms = 10000
                                    j["goal_pair_poll_attempt"] = new_attempt
                                    j["goal_pair_next_poll_time_ms"] = int(time.time() * 1000) + delay_ms
                                    if status == "completed" and goal_data:
                                        sim = goal_data.get("pairs", [{}])[0].get("silhouette_similarity", 0)
                                        if sim < GOAL_PAIR_MIN_SIMILARITY_ACCEPT and not j.get("goal_pair_retry_done"):
                                            logger.info(f"GOAL_PAIR_REJECTED similarity={sim} reason=too_low retrying=true request_id={job_request_id}")
                                            retry_response_id = create_goal_pair_background(
                                                product_name, product_description, job_request_id,
                                                retry_instruction=GOAL_PAIR_RETRY_INSTRUCTION,
                                            )
                                            if retry_response_id:
                                                logger.info(f"GOAL_PAIR_RETRY_STARTED attempt=2 request_id={job_request_id}")
                                                j["openai_response_id"] = retry_response_id
                                                j["openai_goal_pair_response_id"] = retry_response_id
                                                j["goal_pair_created_at"] = time.time()
                                                j["goal_pair_poll_attempt"] = 0
                                                j["goal_pair_next_poll_time_ms"] = 0
                                                j["goal_pair_retry_done"] = True
                                            else:
                                                j["goal_pairs_data"] = goal_data
                                                break
                                        elif sim < GOAL_PAIR_MIN_SIMILARITY_ACCEPT and j.get("goal_pair_retry_done"):
                                            logger.info(f"GOAL_PAIR_RETRY_RESULT similarity={sim} request_id={job_request_id}")
                                            j["goal_pairs_data"] = goal_data
                                            break
                                        else:
                                            j["goal_pairs_data"] = goal_data
                                            break
                                    if status == "failed":
                                        j["goal_pair_fallback"] = True
                                        break
                    with _jobs_lock:
                        j = _jobs.get(jid)
                        goal_data = j.get("goal_pairs_data") if j else None
                        use_fallback = j.get("goal_pair_fallback", False) if j else False
                    # Resolved product name: only from engine result (canonical). Never set from goal_data or description-derived fallback.
                    if goal_data:
                        result = generate_preview_data(payload_data, goal_pairs_data_override=goal_data, request_id=job_request_id)
                    elif use_fallback:
                        logger.info(f"FALLBACK_ABORTED remaining_ads_skipped=true request_id={job_request_id} jobId={jid} adIndex={ad_idx}")
                        if ACE_FALLBACK_RETURN_ERROR:
                            with _jobs_lock:
                                job = _jobs.get(jid)
                                if job is not None:
                                    job["status"] = "error"
                                    job["finished_at"] = time.time()
                                    job["result"] = None
                                    job["error"] = "stage2_fallback"
                                    job["error_message"] = "Stage 2 failed. Please retry."
                            logger.info(f"JOB_DONE jobId={jid} sid={sid} ad={ad_idx} error=stage2_fallback (no fallback ad generated)")
                        else:
                            result = generate_preview_data(payload_data, goal_pair_skip_fetch=True, request_id=job_request_id)
                            with _jobs_lock:
                                job = _jobs.get(jid)
                                if job is not None:
                                    job["status"] = "done"
                                    job["finished_at"] = time.time()
                                    job["result"] = result
                                    job["resolved_product_name"] = (result or {}).get("resolvedProductName", "") or ""
                                    job["error"] = None
                                    job["error_message"] = None
                            logger.info(f"JOB_DONE jobId={jid} sid={sid} ad={ad_idx} elapsed_ms={int((time.time() - start) * 1000)}")
                    else:
                        result = generate_preview_data(payload_data, request_id=job_request_id)
                    if not (use_fallback and ACE_FALLBACK_RETURN_ERROR):
                        elapsed_ms = int((time.time() - start) * 1000)
                        with _jobs_lock:
                            job = _jobs.get(jid)
                            if job is not None:
                                job["status"] = "done"
                                job["finished_at"] = time.time()
                                job["result"] = result
                                job["resolved_product_name"] = (result or {}).get("resolvedProductName", "") or ""
                                job["error"] = None
                                job["error_message"] = None
                        try:
                            img_b64 = (result or {}).get("imageBase64") or (result or {}).get("image_base64")
                            if img_b64:
                                img_bytes = base64.b64decode(img_b64)
                                body_50 = (result or {}).get("bodyText50") or (result or {}).get("body_text") or ""
                                headline = (result or {}).get("headline") or ""
                                _set_artifact(sid, ad_idx, img_bytes, body_50, headline)
                                req_sid = (job.get("requested_session_id") or "").strip() if job else ""
                                if req_sid and req_sid != sid:
                                    _set_sid_alias(req_sid, sid)
                                logger.info(f"RESULT_STORED sessionId={sid} adIndex={ad_idx} bytes={len(img_bytes)} request_id={job_request_id}")
                        except Exception:
                            pass
                        logger.info(f"JOB_DONE jobId={jid} sid={sid} ad={ad_idx} elapsed_ms={elapsed_ms}")
                except OpenAIRateLimitError as e:
                    with _jobs_lock:
                        job = _jobs.get(jid)
                        if job is not None:
                            job["status"] = "error"
                            job["finished_at"] = time.time()
                            job["error"] = "rate_limited"
                            job["error_message"] = e.message
                    logger.error(f"JOB_ERROR jobId={jid} sid={sid} ad={ad_idx} err=rate_limited")
                except Step0BundleTimeoutError as e:
                    with _jobs_lock:
                        job = _jobs.get(jid)
                        if job is not None:
                            job["status"] = "error"
                            job["finished_at"] = time.time()
                            job["error"] = "timeout"
                            job["error_message"] = "Step0 bundle timed out"
                    logger.error(f"JOB_ERROR jobId={jid} sid={sid} ad={ad_idx} err=timeout")
                except Step0BundleOpenAIError as e:
                    with _jobs_lock:
                        job = _jobs.get(jid)
                        if job is not None:
                            job["status"] = "error"
                            job["finished_at"] = time.time()
                            job["error"] = "openai_error"
                            job["error_message"] = str(e)
                    logger.error(f"JOB_ERROR jobId={jid} sid={sid} ad={ad_idx} err=openai_error")
                except Exception as e:
                    with _jobs_lock:
                        job = _jobs.get(jid)
                        if job is not None:
                            job["status"] = "error"
                            job["finished_at"] = time.time()
                            job["error"] = "generation_failed"
                            job["error_message"] = str(e)
                    logger.error(f"JOB_ERROR jobId={jid} sid={sid} ad={ad_idx} err={e}")
            finally:
                # Always release session lock so it never gets stuck
                _release_session_lock(sid, ad_idx)

        try:
            _preview_executor.submit(_run_preview_job, job_id, payload, session_id, ad_index, request_id)
        except Exception as e:
            # Failed to schedule job: release lock and surface error
            _release_session_lock(session_id, ad_index)
            logger.error(f"JOB_ERROR jobId={job_id} sid={session_id} ad={ad_index} err=schedule_failed: {e}")
            return jsonify({
                'ok': False,
                'error': 'schedule_failed',
                'message': 'Failed to schedule preview job'
            }), 500

        # Return immediately with jobId for polling (sessionId so frontend can call download-zip)
        return jsonify({
            'ok': True,
            'jobId': job_id,
            'sessionId': session_id,
            'adIndex': ad_index,
            'status': 'pending'
        }), 202
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Preview job creation failed: {error_msg}", exc_info=True)
        return jsonify({
            'ok': False,
            'error': 'preview_init_failed',
            'message': error_msg
        }), 500


def _download_zip_impl():
    """
    Shared impl for ZIP download (sessionId + adIndex). In-memory artifact store.
    Tries requested sessionId first; if not found, resolves via sid alias map (request-provided -> storage id).
    """
    request_id = str(uuid.uuid4())
    requested_sid = request.args.get("sessionId", "").strip()
    ad_index_str = request.args.get("adIndex", "")
    if not requested_sid:
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'sessionId is required'}), 400
    if not ad_index_str:
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'adIndex is required'}), 400
    try:
        ad_index = int(ad_index_str)
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'adIndex must be 1, 2, or 3'}), 400
    ad_index = max(1, min(3, ad_index))

    logger.info(f"ZIP_DOWNLOAD_REQUEST requested_sid={requested_sid} adIndex={ad_index} request_id={request_id}")

    artifact = _get_artifact(requested_sid, ad_index)
    resolved_sid = None
    if not artifact:
        resolved_sid = _get_sid_alias(requested_sid)
        if resolved_sid:
            artifact = _get_artifact(resolved_sid, ad_index)
    logger.info(f"ZIP_DOWNLOAD_RESOLVED requested_sid={requested_sid} resolved_sid={resolved_sid or 'none'} request_id={request_id}")

    final_sid = resolved_sid if (resolved_sid and artifact) else requested_sid
    if not artifact:
        logger.info(f"ZIP_DOWNLOAD_EXISTS found=false requested_sid={requested_sid} final_lookup_key=({final_sid},{ad_index}) request_id={request_id}")
        logger.info(f"DOWNLOAD_ZIP_LOOKUP sessionId={requested_sid} adIndex={ad_index} found=false request_id={request_id}")
        return jsonify({
            'error': 'not_found',
            'reason': 'missing_session_or_ad',
            'sessionId': requested_sid,
            'adIndex': ad_index,
        }), 404

    img_size = len(artifact.get("image_bytes") or b"")
    logger.info(f"ZIP_DOWNLOAD_EXISTS found=true final_lookup_key=({final_sid},{ad_index}) size_bytes={img_size} request_id={request_id}")
    logger.info(f"DOWNLOAD_ZIP_LOOKUP sessionId={final_sid} adIndex={ad_index} found=true request_id={request_id}")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("image.jpg", artifact["image_bytes"])
        zf.writestr("text.txt", artifact["bodyText50"].encode("utf-8"))
    zip_buffer.seek(0)
    zip_filename = f"ACE_ad-{ad_index}.zip"
    logger.info(f"ZIP_SENT 200 sessionId={final_sid} adIndex={ad_index} request_id={request_id}")
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename,
    )


@app.route('/api/download-zip', methods=['GET'])
def download_zip():
    """Download ZIP for a previously generated ad (sessionId + adIndex). No OpenAI calls."""
    return _download_zip_impl()


@app.route('/api/download', methods=['GET'])
def download():
    """Alias for /api/download-zip so both paths work for backward compatibility."""
    return _download_zip_impl()


@app.route('/api/job-status', methods=['GET'])
def job_status():
    """Poll job status for /api/preview async jobs."""
    job_id = request.args.get("jobId", "").strip()
    if not job_id:
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'jobId is required'}), 400
    job = _get_job(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'not_found', 'status': 'error', 'message': 'Job not found or expired'}), 404
    status = job.get("status", "pending")
    resolved_pn = job.get("resolved_product_name", "")
    logger.info(f"JOB_POLL jobId={job_id} status={status}")
    if status in ("pending", "running"):
        logger.info(f"PRODUCT_NAME_RESPONSE_STAGE=running resolvedProductName=\"{resolved_pn}\"")
        logger.info("PRODUCT_NAME_RESPONSE_SOURCE=canonical_final_name")
        return jsonify({'ok': True, 'status': status, 'resolvedProductName': resolved_pn}), 200
    if status == "done":
        sid = job.get("session_id", "")
        ad_idx = job.get("ad_index", 1)
        logger.info(f"JOB_STATUS_RESPONSE status=done sessionId={sid} adIndex={ad_idx}")
        logger.info(f"PRODUCT_NAME_RESPONSE_STAGE=done resolvedProductName=\"{resolved_pn}\"")
        logger.info("PRODUCT_NAME_RESPONSE_SOURCE=canonical_final_name")
        return jsonify({
            'ok': True,
            'status': 'done',
            'sessionId': sid,
            'adIndex': ad_idx,
            'resolvedProductName': resolved_pn,
            'result': job.get("result"),
        }), 200
    # error
    return jsonify({
        'ok': False,
        'status': 'error',
        'error': job.get("error"),
        'message': job.get("error_message"),
    }), 200


# -----------------------------------------------------------------------------
# Runway video MVP (isolated): one text-to-video only — not part of /api/generate or /api/preview.
# Future: dedicated ACE video engine may add a second output and richer prompting.
# Register POST upload before GET /api/video-headline/<token>. Primary path first for workers.
# Processed MP4 from worker: /tmp/ace_video_test_<jobId>.mp4 (no worker→web upload).
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


@app.route('/api/video-headline/<token>', methods=['GET'], strict_slashes=False)
@app.route('/video-headline/<token>', methods=['GET'], strict_slashes=False)
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


@app.route('/api/generate-video', methods=['POST'])
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
        return jsonify({
            "ok": True,
            "jobId": job_id,
            "status": "running",
        }), 200
    except Exception as e:
        logger.error("generate_video enqueue failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": "video_generation_failed"}), 200


@app.route('/api/video-status', methods=['GET'])
def video_status():
    """Poll async /api/generate-video job from Redis: running | done | error."""
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
    logger.info("VIDEO_JOB_POLL jobId=%s status=%s", job_id, status)
    out = {"ok": True, "status": status}
    if status == "done":
        ensure_video_postprocessed_for_poll(job_id, job)
        job = video_job_get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "not_found"}), 404
        vu = job.get("videoUrl") or ""
        out["videoUrl"] = vu
        out["marketingText"] = job.get("marketingText") or ""
        logger.info("VIDEO_JOB_RESULT jobId=%s video_url=%s", job_id, vu)
    if status == "error":
        out["error"] = job.get("error") or "video_generation_failed"
    return jsonify(out), 200


# Security status: frontend uses this to decide whether to enforce redirect/Builder checks
@app.route('/api/security-status', methods=['GET'])
def security_status():
    """Return whether security is enabled. When false, frontend may allow direct Builder access and skip redirect to Preview."""
    return jsonify({"securityEnabled": is_security_enabled()}), 200


@app.route('/api/security/config', methods=['GET'])
def security_config():
    """Backend-controlled security flag for frontend (e.g. GitHub Pages). Reads ACE_SECURITY_ENABLED; default true if missing."""
    enabled = is_security_enabled()
    return jsonify({"securityEnabled": enabled}), 200


def _under_construction_password_expected() -> str:
    """UNDER_CONSTRUCTION_PASSWORD from env; empty if missing or blank (fail closed in caller)."""
    return (os.environ.get("UNDER_CONSTRUCTION_PASSWORD", "") or "").strip()


@app.route('/api/check-under-construction-password', methods=['POST'])
def check_under_construction_password():
    """
    Owner gate for Under Construction / Preview 2: compare body password to UNDER_CONSTRUCTION_PASSWORD.
    No sessions, cookies, or JWT. If env is unset/empty, always returns ok: false.
    """
    logger.info("UNDER_CONSTRUCTION_PASSWORD_CHECK attempt")
    expected = _under_construction_password_expected()
    if not expected:
        return jsonify({"ok": False}), 200
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False}), 200
    raw = payload.get("password")
    if raw is None:
        submitted = ""
    elif isinstance(raw, str):
        submitted = raw
    else:
        submitted = str(raw)
    if submitted == expected:
        return jsonify({"ok": True}), 200
    return jsonify({"ok": False}), 200


@app.route('/api/entitlement/latest-paid', methods=['GET'])
def entitlement_latest_paid():
    """
    Return whether the current request has a paid entitlement (for Builder access).
    Resolves session from query payment_session/session, cookie payment_session/ace_payment_session, or header X-Payment-Session / X-Fingerprint.
    When ACE_SECURITY_ENABLED=false, returns paid: true so Builder is allowed without payment check.
    Response: { "paid": true } or { "paid": false }.
    """
    if not is_security_enabled():
        return jsonify({"paid": True}), 200
    try:
        db_session.init_db()
    except Exception as e:
        logger.error(f"entitlement_latest_paid init_db error: {e}", exc_info=True)
        return jsonify({"paid": False, "error": "db_error"}), 500
    payment_session = (
        (request.args.get("payment_session") or request.args.get("session") or "").strip()
        or (request.cookies.get("payment_session") or request.cookies.get("ace_payment_session") or "").strip()
        or (request.headers.get("X-Payment-Session") or request.headers.get("X-ACE-Payment-Session") or "").strip()
    )
    if not payment_session:
        fingerprint = (request.headers.get("X-Fingerprint") or "").strip()
        if fingerprint:
            payment_session = db_session.get_payment_session_from_fingerprint(fingerprint) or ""
        if not payment_session:
            cookie_id = (request.cookies.get("ace_cookie_id") or request.cookies.get("cookie_id") or "").strip()
            if cookie_id:
                payment_session = db_session.get_payment_session_from_cookie(cookie_id) or ""
    if not payment_session:
        return jsonify({"paid": False}), 200
    paid = db_session.is_payment_paid(payment_session)
    return jsonify({"paid": paid}), 200


# IPN: payment confirmation — never bypassed by ACE_SECURITY_ENABLED (always runs)
IPN_TOKEN = "ace_icount_7f3a9"

def _ipn_get(data: dict, *keys: str) -> str:
    """Get first non-empty string from data for any of the keys (case-insensitive)."""
    if not isinstance(data, dict):
        return ""
    key_lower = {str(k).lower(): k for k in data.keys()}
    for key in keys:
        k = key_lower.get(str(key).lower()) or (key if key in data else None)
        if k is not None:
            v = data.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    return ""


@app.route(f'/api/ipn/{IPN_TOKEN}', methods=['POST'])
def ipn_ace_icount():
    """
    iCount IPN endpoint: receive payment confirmation and mark session paid.
    iCount may send JSON array or form with flat keys: cp, doctype, docnum, etc. (no payment_session).
    Session id is accepted from cp (and fallbacks) so mark_payment_paid uses the same id the checkout stored.
    """
    try:
        db_session.init_db()
    except Exception as e:
        logger.error(f"IPN init_db error: {e}", exc_info=True)
        return jsonify({"ok": False, "error": "db_error"}), 500
    raw = request.get_json(silent=True)
    if raw is None and request.get_data():
        try:
            raw = json.loads(request.get_data(as_text=True))
        except (json.JSONDecodeError, TypeError):
            pass
    if raw is None or raw == {}:
        raw = request.form or {}
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        data = raw[0]
    elif isinstance(raw, dict):
        data = raw
    elif raw is not None and hasattr(raw, "get"):
        # Form MultiDict / flat key-value (e.g. cp, doctype, docnum) — normalize to dict
        data = dict(raw) if hasattr(raw, "keys") else {}
    else:
        data = {}
    client_obj = data.get("client") if isinstance(data.get("client"), dict) else {}
    # iCount IPN sends flat form keys: cp (payment/session reference), doctype, docnum, customer_id, confirmation_code, etc.
    # No payment_session key — use cp as the session id our checkout passes through the payment link.
    payment_session = (
        _ipn_get(data, "payment_session", "payment_session_id", "session_id", "session")
        or (request.args.get("payment_session") or "").strip()
        or _ipn_get(data, "cp", "confirmation_code", "customer_id")
        or _ipn_get(data, "comment", "Comment", "invoice_po_number", "invoice_po", "based_on_order", "order_id", "ref", "reference")
        or _ipn_get(client_obj, "custom_client_id", "Custom_Client_Id")
    )
    if not payment_session:
        keys_seen = list(data.keys())[:30] if isinstance(data, dict) else []
        logger.warning("IPN missing payment_session keys_seen=%s", keys_seen)
        return jsonify({"ok": False, "error": "missing_payment_session"}), 400
    docnum = (data.get("docnum") or data.get("docnum_id") or "").strip()
    doctype = (data.get("doctype") or "").strip()
    try:
        db_session.mark_payment_paid(payment_session, docnum=docnum, doctype=doctype)
        logger.info(f"IPN_OK payment_session={payment_session}")
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.error(f"IPN mark_payment_paid error: {e}", exc_info=True)
        return jsonify({"ok": False, "error": "payment_update_failed"}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - minimal, no heavy imports, returns plain text.
    This endpoint must NOT trigger any ACE/OpenAI initialization.
    """
    return 'ok', 200




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
    # Self-test: POST must hit the view (401/403/503), not Flask 404 (route missing).
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


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"HEALTH_READY /health returns 200 immediately")
    app.run(host='0.0.0.0', port=port, debug=False)

