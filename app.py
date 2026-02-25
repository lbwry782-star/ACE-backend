from flask import Flask, request, jsonify, send_file, Response
import uuid
import logging
import io
import zipfile
import json
import base64
import os
from typing import Dict, Optional
from engine.side_by_side_v1 import generate_preview_data

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


# ENGINE REMOVED: get_engine_functions() deleted - engine module removed

# Allowed image sizes
ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]

# SESSION SYSTEM REMOVED: All entitlement/session functions deleted


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate a single ad and return as ZIP file.
    
    ENGINE DISABLED: Returns 501 stub response (no image generation, no OpenAI calls).
    
    Request JSON:
    {
        "productName": string,
        "productDescription": string,
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536",
        "adIndex": int (optional, default 0),
        "batchState": {  // optional
            "material_analogy_used": boolean,
            "structural_morphology_used": boolean,
            "structural_exception_used": boolean
        }
    }
    
    Returns: 501 with engine_disabled error (no credits consumed, no OpenAI calls)
    """
    # ENGINE DISABLED: Short-circuit immediately before any engine logic
    # No OpenAI calls, no image generation, no credit consumption
    logger.info("ENGINE_DISABLED /api/generate called")
    return jsonify({
        'ok': False,
        'error': 'engine_disabled',
        'message': 'ACE engine is disabled (rebuild mode).'
    }), 501
    


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
        result = generate_preview_data(payload)
        return jsonify(result), 200
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Preview failed (400): {error_msg}", exc_info=True)
        if "invalid_request" in error_msg.lower():
            actual_error = error_msg.split("invalid_request:", 1)[-1].strip() if "invalid_request:" in error_msg else error_msg
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': f'Invalid request: {actual_error}'
            }), 400
        return jsonify({
            'ok': False,
            'error': 'validation_error',
            'message': error_msg
        }), 400
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Preview failed: {error_msg}", exc_info=True)
        if "rate_limited" in error_msg:
            return jsonify({
                'ok': False,
                'error': 'rate_limited',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 503
        return jsonify({
            'ok': False,
            'error': 'generation_failed',
            'message': str(error_msg)
        }), 500


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

