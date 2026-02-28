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
from engine.side_by_side_v1 import generate_preview_data, Step0BundleTimeoutError, Step0BundleOpenAIError
from engine.openai_retry import OpenAIRateLimitError

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
        
        if is_origin_allowed(origin):
            response = Response('', status=200)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-ACE-Batch-State"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        else:
            # Return 200 without Allow-Origin if origin not allowed (don't reveal info)
            return Response('', status=200)

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to all /api/* responses.
    
    This applies to all responses (200, 404, 500, etc.) for endpoints under /api/.
    """
    if request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        
        if is_origin_allowed(origin):
            # Add CORS headers for allowed origin
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-ACE-Batch-State"
            
            # Expose X-ACE-Batch-State header for frontend (if needed)
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
if ACE_TEST_MODE:
    logger.info("TEST_MODE_ACTIVE=true")

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
        "productName": string (required),
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
        if not payload.get("productName"):
            return jsonify({'ok': False, 'error': 'missing_field', 'message': 'productName is required'}), 400
        if not payload.get("productDescription"):
            return jsonify({'ok': False, 'error': 'missing_field', 'message': 'productDescription is required'}), 400
        session_id = payload.get("sessionId") or "no_session"
        ad_index = int(payload.get("adIndex", 1))
        ad_index = max(1, min(3, ad_index))

        # ACE_TEST_MODE: bypass engine; return demo and store artifact for download-zip
        if ACE_TEST_MODE:
            logger.info("TEST_MODE_RETURNING_DEMO_RESULT=true")
            demo = _get_test_mode_demo_result()
            image_bytes = base64.b64decode(demo["imageBase64"])
            _set_artifact(session_id, ad_index, image_bytes, demo["bodyText50"], demo["headline"])
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
                "productName": payload["productName"],
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
            logger.info(f"GENERATE_DONE sid={session_id} ad={ad_index} stored=true")
            return jsonify({
                "ok": True,
                "sessionId": session_id,
                "adIndex": ad_index,
                "imageBase64": image_base64,
                "headline": headline,
                "bodyText50": body_text_50,
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
        "productName": string (required),
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
        if not payload.get("productName"):
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productName is required'
            }), 400
        if not payload.get("productDescription"):
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productDescription is required'
            }), 400
        session_id = payload.get("sessionId") or "no_session"
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

        logger.info(f"PREVIEW_FLAGS TEST_MODE=false IMAGE_ONLY={ACE_IMAGE_ONLY} PHASE2_GOAL_PAIRS={ACE_PHASE2_GOAL_PAIRS} size={payload.get('imageSize', '')} adIndex={ad_index} sessionId={session_id}")
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
            "ad_index": ad_index,
            "result": None,
            "error": None,
            "error_message": None,
        }
        _cleanup_jobs()
        _set_job(job_id, job_record)
        logger.info(f"JOB_CREATED jobId={job_id} sid={session_id} ad={ad_index}")

        # Schedule background execution
        def _run_preview_job(jid: str, payload_data: dict, sid: str, ad_idx: int, req_id: str) -> None:
            start = time.time()
            try:
                with _jobs_lock:
                    job = _jobs.get(jid)
                    if not job:
                        return
                    job["status"] = "running"
                try:
                    result = generate_preview_data(payload_data)
                    elapsed_ms = int((time.time() - start) * 1000)
                    with _jobs_lock:
                        job = _jobs.get(jid)
                        if job is not None:
                            job["status"] = "done"
                            job["finished_at"] = time.time()
                            job["result"] = result
                            job["error"] = None
                            job["error_message"] = None
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

        # Return immediately with jobId for polling
        return jsonify({
            'ok': True,
            'jobId': job_id,
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


@app.route('/api/download-zip', methods=['GET'])
def download_zip():
    """
    Download ZIP for a previously generated ad (sessionId + adIndex). No OpenAI calls.
    ZIP contains: image.jpg (sketch), text.txt (50-word body text).
    """
    session_id = request.args.get("sessionId", "").strip()
    ad_index_str = request.args.get("adIndex", "")
    if not session_id:
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'sessionId is required'}), 400
    try:
        ad_index = int(ad_index_str)
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'missing_param', 'message': 'adIndex must be 1, 2, or 3'}), 400
    ad_index = max(1, min(3, ad_index))
    artifact = _get_artifact(session_id, ad_index)
    if not artifact:
        logger.info(f"ZIP_DOWNLOAD sid={session_id} ad={ad_index} hit=false")
        return jsonify({'ok': False, 'error': 'not_found', 'message': 'No generated ad found for this session and ad index. Generate first or link may have expired.'}), 404
    logger.info(f"ZIP_DOWNLOAD sid={session_id} ad={ad_index} hit=true")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("image.jpg", artifact["image_bytes"])
        zf.writestr("text.txt", artifact["bodyText50"].encode("utf-8"))
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'ad-{ad_index}.zip'
    )


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
    logger.info(f"JOB_POLL jobId={job_id} status={status}")
    if status in ("pending", "running"):
        return jsonify({'ok': True, 'status': status}), 200
    if status == "done":
        return jsonify({'ok': True, 'status': 'done', 'result': job.get("result")}), 200
    # error
    return jsonify({
        'ok': False,
        'status': 'error',
        'error': job.get("error"),
        'message': job.get("error_message"),
    }), 200


# SESSION SYSTEM REMOVED: All entitlement endpoints deleted
# - /api/entitlement/create
# - /api/entitlement/<sid>
# - /api/entitlement/latest-paid
# - /api/ipn/<token>


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

