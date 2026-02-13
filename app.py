from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import uuid
import logging
import io
import zipfile
import json
import base64
import os
from typing import Dict, Optional
from engine.side_by_side_v1 import generate_zip

app = Flask(__name__)

# ============================================================================
# CORS Configuration (Flask-CORS)
# ============================================================================

# Allowed origins for production and development
ALLOWED_ORIGINS = [
    "https://lbwry782-star.github.io",
    "https://ace-advertising.agency",
    "http://localhost:5173"
]

# Configure CORS for /api/* routes only
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-ACE-Batch-State"],
            "expose_headers": ["X-ACE-Batch-State"],
            "supports_credentials": False,
            "max_age": 86400
        }
    }
)

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
    
    SideBySide Engine v1: Generates one ad with SIDE_BY_SIDE layout.
    
    Request JSON:
    {
        "productName": string (required),
        "productDescription": string (required),
        "imageSize": string (e.g. "1536x1024", default: "1536x1024"),
        "language": string ("he" or "en", default: "he"),
        "adIndex": int (1-3, default: 1, 0 is treated as 1),
        "sessionId": string uuid (optional),
        "history": array of previous ads (optional),
        "objectList": array of strings (optional, uses default if missing)
    }
    
    Returns: application/zip with image.jpg and text.txt
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Get JSON payload
        if not request.is_json:
            logger.warning(f"[{request_id}] Request is not JSON")
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': 'Request must be JSON'
            }), 400
        
        payload = request.get_json()
        
        # Validate required fields
        if not payload.get("productName"):
            logger.warning(f"[{request_id}] Missing productName")
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productName is required'
            }), 400
        
        if not payload.get("productDescription"):
            logger.warning(f"[{request_id}] Missing productDescription")
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productDescription is required'
            }), 400
        
        # Generate ZIP using SideBySide Engine v1
        logger.info(f"[{request_id}] Starting generation: productName={payload.get('productName')[:50]}, "
                   f"adIndex={payload.get('adIndex', 0)}, sessionId={payload.get('sessionId')}")
        
        zip_bytes = generate_zip(payload_dict=payload, is_preview=False)
        
        logger.info(f"[{request_id}] Generation successful, returning ZIP ({len(zip_bytes)} bytes)")
        
        # Return ZIP file
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name='ad.zip'
        )
        
    except ValueError as e:
        # Handle 400 errors (invalid_request from OpenAI)
        error_msg = str(e)
        logger.error(f"[{request_id}] Generation failed (400): {error_msg}", exc_info=True)
        
        if "invalid_request" in error_msg.lower():
            # Extract the actual error message (after "invalid_request: ")
            actual_error = error_msg.split("invalid_request:", 1)[-1].strip() if "invalid_request:" in error_msg else error_msg
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': f'Invalid request to OpenAI: {actual_error}'
            }), 400
        else:
            # Other ValueError - still 400
            return jsonify({
                'ok': False,
                'error': 'validation_error',
                'message': error_msg
            }), 400
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Generation failed: {error_msg}", exc_info=True)
        
        # Handle rate limit specifically
        if "rate_limited" in error_msg:
            return jsonify({
                'ok': False,
                'error': 'rate_limited',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 503
        
        # Generic error
        return jsonify({
            'ok': False,
            'error': 'generation_failed',
            'message': f'Failed to generate ad: {error_msg}'
        }), 500
    


@app.route('/api/preview', methods=['POST'])
def preview():
    """
    Generate a single ad preview and return as ZIP file.
    
    SideBySide Engine v1: Same logic as /api/generate, returns ZIP for preview.
    
    Request JSON:
    {
        "productName": string (required),
        "productDescription": string (required),
        "imageSize": string (e.g. "1536x1024", default: "1536x1024"),
        "language": string ("he" or "en", default: "he"),
        "adIndex": int (1-3, default: 1, 0 is treated as 1),
        "sessionId": string uuid (optional),
        "history": array of previous ads (optional),
        "objectList": array of strings (optional, uses default if missing)
    }
    
    Returns: application/zip with image.jpg and text.txt (same as /api/generate)
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Get JSON payload
        if not request.is_json:
            logger.warning(f"[{request_id}] Preview request is not JSON")
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': 'Request must be JSON'
            }), 400
        
        payload = request.get_json()
        
        # Validate required fields
        if not payload.get("productName"):
            logger.warning(f"[{request_id}] Preview missing productName")
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productName is required'
            }), 400
        
        if not payload.get("productDescription"):
            logger.warning(f"[{request_id}] Preview missing productDescription")
            return jsonify({
                'ok': False,
                'error': 'missing_field',
                'message': 'productDescription is required'
            }), 400
        
        # Generate ZIP using SideBySide Engine v1 (preview mode)
        logger.info(f"[{request_id}] Starting preview: productName={payload.get('productName')[:50]}, "
                   f"adIndex={payload.get('adIndex', 0)}, sessionId={payload.get('sessionId')}")
        
        zip_bytes = generate_zip(payload_dict=payload, is_preview=True)
        
        logger.info(f"[{request_id}] Preview generation successful, returning ZIP ({len(zip_bytes)} bytes)")
        
        # Return ZIP file (same format as generate)
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name='preview.zip'
        )
        
    except ValueError as e:
        # Handle 400 errors (invalid_request from OpenAI)
        error_msg = str(e)
        logger.error(f"[{request_id}] Preview generation failed (400): {error_msg}", exc_info=True)
        
        if "invalid_request" in error_msg.lower():
            # Extract the actual error message (after "invalid_request: ")
            actual_error = error_msg.split("invalid_request:", 1)[-1].strip() if "invalid_request:" in error_msg else error_msg
            return jsonify({
                'ok': False,
                'error': 'invalid_request',
                'message': f'Invalid request to OpenAI: {actual_error}'
            }), 400
        else:
            # Other ValueError - still 400
            return jsonify({
                'ok': False,
                'error': 'validation_error',
                'message': error_msg
            }), 400
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Preview generation failed: {error_msg}", exc_info=True)
        
        # Handle rate limit specifically
        if "rate_limited" in error_msg:
            return jsonify({
                'ok': False,
                'error': 'rate_limited',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 503
        
        # Generic error
        return jsonify({
            'ok': False,
            'error': 'generation_failed',
            'message': f'Failed to generate preview: {error_msg}'
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

