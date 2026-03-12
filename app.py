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
import db_session

app = Flask(__name__)

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

# ACE_TEST_MODE: "1" or "true" = no OpenAI, return demo result immediately
ACE_TEST_MODE = (os.environ.get("ACE_TEST_MODE", "") or "").strip().lower() in ("1", "true")
# ACE_IMAGE_ONLY: "1" or "true" = gpt-image-1.5 only, no o3-pro, placeholder copy
ACE_IMAGE_ONLY = (os.environ.get("ACE_IMAGE_ONLY", "") or "").strip().lower() in ("1", "true")
# ACE_PHASE2_GOAL_PAIRS: "1" or "true" or "yes" = when IMAGE_ONLY, call o3-pro for goal + 3 pairs
ACE_PHASE2_GOAL_PAIRS = (os.environ.get("ACE_PHASE2_GOAL_PAIRS", "") or "").strip().lower() in ("1", "true", "yes")
# ACE_FALLBACK_RETURN_ERROR: when Stage 2 fails (FALLBACK_USED), return error instead of generating 1 fallback ad (cost control)
ACE_FALLBACK_RETURN_ERROR = (os.environ.get("ACE_FALLBACK_RETURN_ERROR", "") or "").strip().lower() in ("1", "true", "yes")

# ACE_SECURITY_ENABLED: "true" = run security (redirect to Preview after payment, Builder/session validation).
# "false" = bypass: allow direct Builder access, refresh without redirect, no tab/session validation.
# IPN endpoint is never bypassed — payment confirmation always runs.
def is_security_enabled() -> bool:
    """Read ACE_SECURITY_ENABLED from env. If 'false', security checks (redirect, Builder/session) are bypassed. Default True."""
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


# Security status: frontend uses this to decide whether to enforce redirect/Builder checks
@app.route('/api/security-status', methods=['GET'])
def security_status():
    """Return whether security is enabled. When false, frontend may allow direct Builder access and skip redirect to Preview."""
    return jsonify({"securityEnabled": is_security_enabled()}), 200


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
    iCount sends JSON array with one document; document has doctype, docnum, comment, etc. (no payment_session).
    We accept payment_session from: our field names, or iCount fields (comment, invoice_po_number, based_on_order, client.custom_client_id).
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
    else:
        data = {}
    client_obj = data.get("client") if isinstance(data.get("client"), dict) else {}
    payment_session = (
        _ipn_get(data, "payment_session", "payment_session_id", "session_id", "session")
        or (request.args.get("payment_session") or "").strip()
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




if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"HEALTH_READY /health returns 200 immediately")
    app.run(host='0.0.0.0', port=port, debug=False)

