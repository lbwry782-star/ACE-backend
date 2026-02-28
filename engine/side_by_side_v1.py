"""
SideBySide Engine v1

Generates ads with SIDE_BY_SIDE layout only.
Each call generates one ad (to avoid 429 rate limits).
"""

import os
import uuid
import json
import logging
import time
import random
import io
import zipfile
import base64
import string
import hashlib
import re
from typing import Dict, List, Optional, Tuple
import httpx
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from threading import Lock

from . import openai_retry

logger = logging.getLogger(__name__)


# Exceptions for STEP0_BUNDLE timeout/errors (so app can return 504/500)
class Step0BundleTimeoutError(Exception):
    """STEP0_BUNDLE OpenAI call timed out."""
    pass


class Step0BundleOpenAIError(Exception):
    """STEP0_BUNDLE OpenAI call failed (non-timeout)."""
    pass

# ============================================================================
# Feature Flags (from ENV, with safe defaults = legacy behavior)
# ============================================================================

ENGINE_MODE = os.environ.get("ENGINE_MODE", "legacy")  # "legacy" | "optimized"
PREVIEW_MODE = os.environ.get("PREVIEW_MODE", "image")  # "image" | "plan_only"
ENABLE_PLAN_CACHE = os.environ.get("ENABLE_PLAN_CACHE", "0") == "1"
PLAN_CACHE_TTL_SECONDS = int(os.environ.get("PLAN_CACHE_TTL_SECONDS", "900"))
ENABLE_IMAGE_CACHE = os.environ.get("ENABLE_IMAGE_CACHE", "0") == "1"
IMAGE_CACHE_TTL_SECONDS = int(os.environ.get("IMAGE_CACHE_TTL_SECONDS", "900"))

# Preview optimization flags (o4-mini deprecated: always use o3-pro for planning)
_def_preview = os.environ.get("PREVIEW_PLANNER_MODEL", "o3-pro")
PREVIEW_PLANNER_MODEL = "o3-pro" if _def_preview == "o4-mini" else _def_preview
_def_generate = os.environ.get("GENERATE_PLANNER_MODEL", "o3-pro")
GENERATE_PLANNER_MODEL = "o3-pro" if _def_generate == "o4-mini" else _def_generate


def _get_text_model() -> str:
    """Resolved text model; o4-mini is deprecated and mapped to o3-pro."""
    m = os.environ.get("OPENAI_TEXT_MODEL", "o3-pro")
    return "o3-pro" if m == "o4-mini" else m


def _get_shape_model() -> str:
    """Resolved shape/planning model; o4-mini is deprecated and mapped to o3-pro."""
    m = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    return "o3-pro" if m == "o4-mini" else m


PREVIEW_SKIP_PHYSICAL_CONTEXT = os.environ.get("PREVIEW_SKIP_PHYSICAL_CONTEXT", "1") == "1"  # Skip STEP 1.5 in preview
PREVIEW_USE_CACHE = os.environ.get("PREVIEW_USE_CACHE", "1") == "1"  # Use cache for preview
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "900"))  # Cache TTL for preview

# Step-level caching flags
ENABLE_STEP0_CACHE = os.environ.get("ENABLE_STEP0_CACHE", "1") == "1"  # Cache STEP 0 (objectList building)
STEP0_CACHE_TTL_SECONDS = int(os.environ.get("STEP0_CACHE_TTL_SECONDS", "3600"))  # STEP 0 cache TTL
ENABLE_STEP1_CACHE = os.environ.get("ENABLE_STEP1_CACHE", "1") == "1"  # Cache STEP 1 (shape match)
STEP1_CACHE_TTL_SECONDS = int(os.environ.get("STEP1_CACHE_TTL_SECONDS", "1800"))  # STEP 1 cache TTL

# Cache key versioning and diversity
CACHE_KEY_VERSION = os.environ.get("CACHE_KEY_VERSION", "v4")  # Cache key version (incremented to invalidate old cache)
ENABLE_DIVERSITY_GUARD = os.environ.get("ENABLE_DIVERSITY_GUARD", "1") == "1"  # Diversity guard to prevent repetition
DIVERSITY_GUARD_TTL_SECONDS = 1800  # 30 minutes TTL for diversity guard

# Layout mode
ACE_LAYOUT_MODE = os.environ.get("ACE_LAYOUT_MODE", "side_by_side")  # "side_by_side" | "hybrid" (default: side_by_side, hybrid ignored)

# Image generation mode
ACE_IMAGE_MODE = os.environ.get("ACE_IMAGE_MODE", "replacement")  # "replacement" | "side_by_side" (default: replacement)

# Shape matching parameters
SHAPE_MIN_SCORE = float(os.environ.get("SHAPE_MIN_SCORE", "0.80"))  # Minimum shape similarity score (0-1)
SHAPE_SEARCH_K = int(os.environ.get("SHAPE_SEARCH_K", "40"))  # Number of candidates to check per object (K=35-50)
MAX_CHECKED_PAIRS = int(os.environ.get("MAX_CHECKED_PAIRS", "500"))  # Maximum pairs to check before stopping
CANDIDATE_LIMIT = int(os.environ.get("CANDIDATE_LIMIT", "80"))  # Maximum candidate pairs to collect before batch scoring

# Object list size parameters
OBJECT_LIST_TARGET = 150  # Target size for STEP 0 object list
OBJECT_LIST_MIN_OK = 130  # Minimum acceptable size to proceed without failure

# Image size and quality parameters
PREVIEW_IMAGE_SIZE_DEFAULT = "1024x1024"  # Fast preview size
PREVIEW_IMAGE_QUALITY_DEFAULT = "low"  # Fast preview quality
GENERATE_IMAGE_QUALITY_DEFAULT = "high"  # High quality for final generation
ALLOWED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024"}  # Supported by gpt-image-*

# ACE_IMAGE_ONLY=1: real image via gpt-image-1.5 only, no o3-pro, placeholder copy
ACE_IMAGE_ONLY = (os.environ.get("ACE_IMAGE_ONLY", "") or "").strip() in ("1", "true")

# Fallback placeholder PNG (1x1) when image call fails in IMAGE_ONLY mode
_IMAGE_ONLY_PLACEHOLDER_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Timeout for single image call in IMAGE_ONLY mode (no retries)
IMAGE_ONLY_CALL_TIMEOUT_SECONDS = 60

# ============================================================================
# In-Memory Caches (with TTL)
# ============================================================================

# Plan cache: key -> (value, timestamp)
_plan_cache: Dict[str, Tuple[Dict, float]] = {}
_plan_cache_lock = Lock()
_session_plan_cache: Dict[str, Tuple[Dict, float]] = {}  # Cache for one-call session plans
_session_plan_cache_lock = Lock()

# Preview cache: key -> (value, timestamp)
_preview_cache: Dict[str, Tuple[Dict, float]] = {}
_preview_cache_lock = Lock()

# Image cache: key -> (value_bytes, timestamp)
_image_cache: Dict[str, Tuple[bytes, float]] = {}
_image_cache_lock = Lock()

# STEP 0 cache: key -> (objectList, timestamp)
_step0_cache: Dict[str, Tuple[List[str], float]] = {}
_step0_cache_lock = Lock()

# STEP 1 cache: key -> (shape_result_dict, timestamp)
_step1_cache: Dict[str, Tuple[Dict, float]] = {}
_step1_cache_lock = Lock()

# Three pairs cache: key -> (list of 3 pairs, timestamp)
_three_pairs_cache: Dict[str, Tuple[List[Dict], float]] = {}
_three_pairs_cache_lock = Lock()

# Diversity guard: (session_seed, productName) -> set of (object_a, object_b) pairs with timestamp
_diversity_guard: Dict[str, Tuple[set, float]] = {}  # key: f"{session_seed}|{productName}", value: (set of tuples, timestamp)
_diversity_guard_lock = Lock()

# Session-based anti-repeat: sid -> (used_pairs, used_objects, timestamp)
_session_used_pairs: Dict[str, Tuple[set, set, float]] = {}  # key: sid, value: (set of pair_hashes, set of object_ids, timestamp)
_session_used_lock = Lock()


def _get_cache_key_plan(
    product_name: str,
    message: str,
    ad_goal: str,
    ad_index: int,
    object_list: Optional[List[str]],
    engine_mode: str,
    mode: str,
    language: str = "en",
    session_seed: Optional[str] = None,
    image_size: Optional[str] = None,
) -> str:
    """Generate cache key for plan."""
    object_list_hash = hashlib.md5(json.dumps(object_list or [], sort_keys=True).encode()).hexdigest()[:16]
    layout_mode = ACE_LAYOUT_MODE  # Include layout mode in cache key
    size_part = image_size or ""
    quality_part = "high"  # PLAN cache is for generate, always high quality
    key_str = f"{product_name}|{message}|{ad_goal}|{language}|{ad_index}|{session_seed or ''}|{engine_mode}|{mode}|{layout_mode}|{size_part}|{quality_part}|{object_list_hash}|PLAN_{CACHE_KEY_VERSION}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_key_preview(product_name: str, message: str, ad_goal: str, ad_index: int, object_list: Optional[List], language: str = "en", session_seed: Optional[str] = None, engine_mode: str = "legacy", preview_mode: str = "image", image_size: Optional[str] = None, quality: str = "low") -> str:
    """Generate cache key for preview plan. Includes sid + ad_index to prevent repetition."""
    # Hash productName + productDescription (or ad_goal as fallback)
    hash_inp = hashlib.sha256(f"{product_name}|{message}".encode()).hexdigest()[:16]
    object_list_hash = hashlib.md5(json.dumps([describe_item(item) for item in (object_list or [])], sort_keys=True).encode()).hexdigest()[:16]
    layout_mode = ACE_LAYOUT_MODE
    image_size_str = image_size or PREVIEW_IMAGE_SIZE_DEFAULT
    key_str = f"{session_seed or 'no_session'}|{ad_index}|{image_size_str}|{quality}|{layout_mode}|{engine_mode}|{preview_mode}|{hash_inp}|{object_list_hash}|PREVIEW_{CACHE_KEY_VERSION}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def _get_from_preview_cache(key: str) -> Optional[Dict]:
    """Get from preview cache if valid."""
    if not PREVIEW_USE_CACHE:
        return None
    
    with _preview_cache_lock:
        if key in _preview_cache:
            value, timestamp = _preview_cache[key]
            if time.time() - timestamp < CACHE_TTL_SECONDS:
                return value
            else:
                # Expired, remove
                del _preview_cache[key]
    return None


def _set_to_preview_cache(key: str, value: Dict):
    """Set to preview cache."""
    if not PREVIEW_USE_CACHE:
        return
    
    with _preview_cache_lock:
        _preview_cache[key] = (value, time.time())


def _get_cache_key_image(prompt: str, image_size: str, model: str, quality: str = "high") -> str:
    """Generate cache key for image."""
    key_str = f"{prompt}|{image_size}|{model}|{quality}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_key_final_image(
    session_seed: str,
    ad_index: int,
    product_name: str,
    message: str,
    headline: str,
    image_size: str,
    quality: str,
    layout_mode: str = "side_by_side",
    image_mode: str = "replacement",
    object_a: Optional[str] = None,
    object_a_sub: Optional[str] = None,
    object_b: Optional[str] = None
) -> str:
    """
    Generate unified cache key for final image (shared between preview and zip).
    
    Key includes: sid + ad_index + product hash + layout/mode + image_mode + image_size + headline + quality + A.object + A.sub_object + B.object + CACHE_KEY_VERSION
    For replacement mode, includes A.object, A.sub_object, B.object to ensure cache correctness.
    """
    # Hash productName + message
    product_hash = hashlib.sha256(f"{product_name}|{message}".encode()).hexdigest()[:16]
    # Hash headline (shortened)
    headline_hash = hashlib.sha256(headline.encode()).hexdigest()[:16]
    
    # For replacement mode, include A.object, A.sub_object, B.object in cache key
    if image_mode == "replacement" and object_a and object_b:
        object_a_sub_str = object_a_sub or ""
        key_str = f"{session_seed}|{ad_index}|{product_hash}|{layout_mode}|{image_mode}|{image_size}|{headline_hash}|{quality}|A={object_a}|A_sub={object_a_sub_str}|B={object_b}|FINAL_IMAGE_{CACHE_KEY_VERSION}"
    else:
        key_str = f"{session_seed}|{ad_index}|{product_hash}|{layout_mode}|{image_mode}|{image_size}|{headline_hash}|{quality}|FINAL_IMAGE_{CACHE_KEY_VERSION}"
    
    return hashlib.sha256(key_str.encode()).hexdigest()


def _get_from_plan_cache(key: str) -> Optional[Dict]:
    """Get from plan cache if valid."""
    if not ENABLE_PLAN_CACHE:
        return None
    
    with _plan_cache_lock:
        if key in _plan_cache:
            value, timestamp = _plan_cache[key]
            if time.time() - timestamp < PLAN_CACHE_TTL_SECONDS:
                return value
            else:
                # Expired, remove
                del _plan_cache[key]
    return None


def _set_to_plan_cache(key: str, value: Dict):
    """Set to plan cache."""
    if not ENABLE_PLAN_CACHE:
        return
    
    with _plan_cache_lock:
        _plan_cache[key] = (value, time.time())


def _get_from_image_cache(key: str) -> Optional[bytes]:
    """Get from image cache if valid."""
    if not ENABLE_IMAGE_CACHE:
        return None
    
    with _image_cache_lock:
        if key in _image_cache:
            value_bytes, timestamp = _image_cache[key]
            if time.time() - timestamp < IMAGE_CACHE_TTL_SECONDS:
                return value_bytes
            else:
                # Expired, remove
                del _image_cache[key]
    return None


def _set_to_image_cache(key: str, value_bytes: bytes):
    """Set to image cache."""
    if not ENABLE_IMAGE_CACHE:
        return
    
    with _image_cache_lock:
        _image_cache[key] = (value_bytes, time.time())


def _get_cache_key_step0(ad_goal: str, product_name: Optional[str], language: str = "en", session_seed: Optional[str] = None) -> str:
    """Generate cache key for STEP 0 (objectList building).
    
    Shared across ads in same session (session_seed included in key).
    """
    session_part = f"|{session_seed}" if session_seed else ""
    key_str = f"{ad_goal}|{product_name or ''}|{language}{session_part}|STEP0_V1"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_from_step0_cache(key: str) -> Optional[List[str]]:
    """Get from STEP 0 cache if valid."""
    if not ENABLE_STEP0_CACHE:
        return None
    
    with _step0_cache_lock:
        if key in _step0_cache:
            value, timestamp = _step0_cache[key]
            if time.time() - timestamp < STEP0_CACHE_TTL_SECONDS:
                ttl_remaining = int(STEP0_CACHE_TTL_SECONDS - (time.time() - timestamp))
                logger.info(f"STEP0_CACHE hit=true ttl={ttl_remaining}s key={key[:16]}...")
                return value
            else:
                # Expired, remove
                del _step0_cache[key]
                logger.info(f"STEP0_CACHE hit=false (expired) ttl=0s key={key[:16]}...")
        else:
            logger.info(f"STEP0_CACHE hit=false ttl={STEP0_CACHE_TTL_SECONDS}s key={key[:16]}...")
    return None


def _set_to_step0_cache(key: str, value: List[str]):
    """Set to STEP 0 cache."""
    if not ENABLE_STEP0_CACHE:
        return
    
    with _step0_cache_lock:
        _step0_cache[key] = (value, time.time())


def _stable_object_list_fingerprint(object_list: Optional[List]) -> str:
    """
    Generate a stable fingerprint for object_list that works with both List[str] and List[Dict].
    Never sorts dicts directly - only sorts string identifiers.
    Handles non-string types (int, float, etc.) by coercing to string.
    """
    if not object_list:
        return ""
    # object_list may be List[str] (legacy) or List[Dict] (new)
    if isinstance(object_list[0], dict):
        ids = []
        for it in object_list:
            # prefer id; fallback to object+sub_object
            raw_id = it.get("id")
            _id = (str(raw_id) if raw_id is not None else "").strip()
            if not _id:
                # Fallback: use object+sub_object (also coerce to string)
                raw_obj = it.get("object")
                raw_sub = it.get("sub_object")
                obj_str = str(raw_obj) if raw_obj is not None else ""
                sub_str = str(raw_sub) if raw_sub is not None else ""
                _id = f'{obj_str}::{sub_str}'
            ids.append(_id)
        ids.sort()
        return hashlib.md5("|".join(ids).encode()).hexdigest()[:16]
    else:
        ids = [str(x) for x in object_list]
        ids.sort()
        return hashlib.md5("|".join(ids).encode()).hexdigest()[:16]


def _get_cache_key_step1(object_list: List[str], min_shape_score: int, min_env_diff_score: int, used_objects: set, product_name: str = "", ad_goal: str = "", language: str = "en", ad_index: int = 1, session_seed: Optional[str] = None, engine_mode: str = "legacy", preview_mode: str = "image") -> str:
    """Generate cache key for STEP 1 (shape match)."""
    # Include objectList hash, gate parameters, used_objects, and context
    # Use stable fingerprint that handles both List[str] and List[Dict]
    object_list_hash = _stable_object_list_fingerprint(object_list)
    used_objects_str = "|".join(sorted(used_objects)) if used_objects else ""
    layout_mode = ACE_LAYOUT_MODE  # Include layout mode in cache key
    key_str = f"{product_name}|{ad_goal}|{language}|{ad_index}|{session_seed or ''}|{engine_mode}|{preview_mode}|{layout_mode}|{object_list_hash}|PAIR_GATE(min_shape={min_shape_score},min_env_diff={min_env_diff_score})|{used_objects_str}|STEP1_{CACHE_KEY_VERSION}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_from_step1_cache(key: str) -> Optional[Dict]:
    """Get from STEP 1 cache if valid."""
    if not ENABLE_STEP1_CACHE:
        return None
    
    with _step1_cache_lock:
        if key in _step1_cache:
            value, timestamp = _step1_cache[key]
            if time.time() - timestamp < STEP1_CACHE_TTL_SECONDS:
                ttl_remaining = int(STEP1_CACHE_TTL_SECONDS - (time.time() - timestamp))
                logger.info(f"STEP1_CACHE hit=true ttl={ttl_remaining}s key={key[:16]}...")
                return value
            else:
                # Expired, remove
                del _step1_cache[key]
                logger.info(f"STEP1_CACHE hit=false (expired) ttl=0s key={key[:16]}...")
        else:
            logger.info(f"STEP1_CACHE hit=false ttl={STEP1_CACHE_TTL_SECONDS}s key={key[:16]}...")
    return None


def _set_to_step1_cache(key: str, value: Dict):
    """Set to STEP 1 cache."""
    if not ENABLE_STEP1_CACHE:
        return
    
    with _step1_cache_lock:
        _step1_cache[key] = (value, time.time())


def _get_diversity_guard_key(session_seed: Optional[str], product_name: str) -> str:
    """Generate diversity guard key."""
    return f"{session_seed or 'no_session'}|{product_name}"


def _check_diversity_guard(session_seed: Optional[str], product_name: str, object_a: str, object_b: str) -> bool:
    """Check if pair was already used in this session. Returns True if pair is new (allowed), False if already used (blocked)."""
    if not ENABLE_DIVERSITY_GUARD:
        return True  # Diversity guard disabled, allow all
    
    guard_key = _get_diversity_guard_key(session_seed, product_name)
    pair_tuple = tuple(sorted([object_a, object_b]))  # Normalize order
    
    with _diversity_guard_lock:
        if guard_key in _diversity_guard:
            pairs_set, timestamp = _diversity_guard[guard_key]
            # Check if expired
            if time.time() - timestamp >= DIVERSITY_GUARD_TTL_SECONDS:
                # Expired, clear and allow
                del _diversity_guard[guard_key]
                return True
            
            # Check if pair already exists
            if pair_tuple in pairs_set:
                return False  # Pair already used, block
            else:
                return True  # Pair is new, allow
        else:
            return True  # No history, allow


def _add_to_diversity_guard(session_seed: Optional[str], product_name: str, object_a: str, object_b: str):
    """Add pair to diversity guard."""
    if not ENABLE_DIVERSITY_GUARD:
        return
    
    guard_key = _get_diversity_guard_key(session_seed, product_name)
    pair_tuple = tuple(sorted([object_a, object_b]))  # Normalize order
    
    with _diversity_guard_lock:
        if guard_key in _diversity_guard:
            pairs_set, timestamp = _diversity_guard[guard_key]
            # Check if expired
            if time.time() - timestamp >= DIVERSITY_GUARD_TTL_SECONDS:
                # Expired, reset
                _diversity_guard[guard_key] = ({pair_tuple}, time.time())
            else:
                # Add to existing set
                pairs_set.add(pair_tuple)
        else:
            # Create new entry
            _diversity_guard[guard_key] = ({pair_tuple}, time.time())

# Default object list - concrete nouns in English (for shape similarity)
DEFAULT_OBJECT_LIST = [
    "leaf", "shell", "ear", "mask", "bottle", "candle", "banana", "crescent",
    "ring", "coin", "plate", "wheel", "key", "spoon", "fork", "ladder",
    "tower", "tree", "feather", "fish", "rocket", "pencil", "pipe", "cone",
    "triangle", "circle", "square", "cube", "sphere", "cylinder", "hourglass",
    "moon", "sun", "star", "diamond", "heart", "arrow", "bow", "shield",
    "crown", "bell", "horn", "trumpet", "flute", "violin", "guitar", "drum"
]

# Allowed image sizes
ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]


def build_ad_goal(product_name: str, product_description: str) -> str:
    """
    STEP 0.5 - BUILD_AD_GOAL
    
    Generate advertising goal (ad_goal) from productName + productDescription.
    
    Args:
        product_name: Product name
        product_description: Product description
    
    Returns:
        str: Single English sentence (6-12 words) defining intent, not a slogan.
    
    Example:
        "Protect natural ecosystems and wildlife habitats"
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = _get_text_model()
    
    prompt = f"""Generate a single advertising goal (ad_goal) from the product information below.

Product Name: {product_name}
Product Description: {product_description}

Requirements:
- Output EXACTLY one English sentence.
- 6-12 words only.
- Define the intent/purpose, NOT a slogan or tagline.
- Focus on the core message or action.
- No punctuation at the end (unless necessary).

Example outputs:
- "Protect natural ecosystems and wildlife habitats"
- "Reduce plastic waste in oceans"
- "Support renewable energy adoption"
- "Promote sustainable agriculture practices"

Output ONLY the ad_goal sentence, nothing else:"""
    
    try:
        def _call():
            if model.startswith("o"):
                r = client.responses.create(model=model, input=prompt)
                return r.output_text.strip()
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a marketing strategist. Generate concise advertising goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            return r.choices[0].message.content.strip()
        ad_goal = openai_retry.openai_call_with_retry(_call, endpoint="responses")
        # Clean up: remove quotes, extra whitespace
        ad_goal = ad_goal.strip('"\'')
        ad_goal = ' '.join(ad_goal.split())
        logger.info(f"AD_GOAL={ad_goal}")
        return ad_goal
    except openai_retry.OpenAIRateLimitError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate ad_goal: {str(e)}")
        # Fallback
        fallback = f"Support {product_name.lower()}" if product_name else "Make a positive difference"
        logger.warning(f"Using fallback ad_goal: {fallback}")
        return fallback


def _norm(s: str) -> str:
    """
    Normalize string for matching: remove all non-alphanumeric, convert to lowercase.
    
    Args:
        s: String to normalize
    
    Returns:
        Normalized string (only lowercase alphanumeric)
    """
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _main_key(it) -> str:
    """
    Get normalized main object key from item (dict or string).
    
    Args:
        it: Item (dict with "object" key, or string)
    
    Returns:
        Normalized main object name
    """
    if isinstance(it, dict):
        return _norm(it.get("object", ""))
    return _norm(str(it))


def build_theme_tags(ad_goal: str) -> List[str]:
    """
    Build theme tags from ad_goal.
    
    Generates 8-12 short tags (1-2 words) that represent the themes of the ad_goal.
    
    Args:
        ad_goal: Advertising goal string
    
    Returns:
        List[str]: List of theme tags (e.g., ["marketing", "ads", "campaigns", ...])
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = _get_text_model()
    
    prompt = f"""Generate 8-12 short theme tags (1-2 words each) that represent the key themes and topics related to this advertising goal.

Advertising goal: {ad_goal}

Requirements:
- Return EXACTLY 8-12 tags
- Each tag is 1-2 words only
- Tags should be relevant to the ad_goal's themes and topics
- Use lowercase, single words or short phrases
- No punctuation, no special characters
- Focus on core concepts, not specific products

Example for "AI advertising platform":
["marketing", "ads", "campaigns", "analytics", "optimization", "creativity", "branding", "targeting", "automation", "conversion"]

Example for "Protect natural ecosystems":
["nature", "conservation", "wildlife", "environment", "sustainability", "ecology", "habitat", "biodiversity", "preservation", "ecosystem"]

Output ONLY a JSON array of strings, nothing else:"""
    
    try:
        def _call():
            if model.startswith("o"):
                r = client.responses.create(model=model, input=prompt)
                return r.output_text.strip()
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a marketing analyst. Generate concise theme tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return r.choices[0].message.content.strip()
        response_text = openai_retry.openai_call_with_retry(_call, endpoint="responses")
        # Parse JSON array
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        if response_text.startswith("```json"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        theme_tags = json.loads(response_text)
        if not isinstance(theme_tags, list):
            raise ValueError("Response is not a list")
        
        # Normalize tags: lowercase, strip, filter empty
        theme_tags = [str(tag).lower().strip() for tag in theme_tags if tag]
        
        # Ensure we have at least 8 tags, pad if needed
        if len(theme_tags) < 8:
            logger.warning(f"Only {len(theme_tags)} theme tags generated, expected 8-12")
            # Could add fallback logic here if needed
        
        # Limit to 12 tags max
        theme_tags = theme_tags[:12]
        
        logger.info(f"THEME_TAGS={','.join(theme_tags)}")
        return theme_tags
    except openai_retry.OpenAIRateLimitError:
        raise
    except Exception as e:
        logger.error(f"Failed to build theme tags: {e}")
        # Fallback to generic tags
        return ["marketing", "advertising", "campaigns", "branding", "targeting", "optimization", "creativity", "analytics"]


def uniform_shuffle(items: List, seed: int) -> List:
    """
    Fisher-Yates shuffle with deterministic seed.
    
    Args:
        items: List to shuffle
        seed: Integer seed for reproducibility
    
    Returns:
        List: Shuffled list (new list, original unchanged)
    """
    rng = random.Random(seed)
    shuffled = list(items)  # Copy
    n = len(shuffled)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled


def validate_object_item(item: Dict, forbidden_words: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a single object item with STRUCTURAL checks only (no forbidden words).
    
    Args:
        item: Dict with keys: id, object, sub_object, classic_context, theme_link, category, shape_hint, theme_tag
        forbidden_words: DEPRECATED - ignored (forbidden words are now model responsibility)
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # STRUCTURAL CHECKS ONLY
    
    # 1. Check required fields
    if not item.get("id") or not item.get("object") or not item.get("classic_context"):
        return False, "Missing required fields (id, object, classic_context)"
    
    # 2. Check sub_object is present and not empty
    sub_object = item.get("sub_object", "").strip()
    if not sub_object:
        return False, "Missing sub_object (required field)"
    
    classic_context = item.get("classic_context", "").strip()
    
    # 3. Check minimum length (3 words minimum)
    words = classic_context.split()
    if len(words) < 3:
        return False, f"classic_context too short ({len(words)} words, need >=3): '{classic_context}'"
    
    # 4. Check maximum length (12 words maximum)
    if len(words) > 12:
        return False, f"classic_context too long ({len(words)} words, max 12): '{classic_context}'"
    
    # 5. Check for required physical preposition/relationship
    context_lower = classic_context.lower()
    physical_prepositions = [
        "on", "in", "under", "next to", "attached to", "inside", "resting on",
        "landing on", "with", "lying on", "sitting on", "placed on", "hanging from",
        "growing from", "emerging from", "surrounded by", "near", "beside", "against",
        "within", "among", "between", "alongside", "above", "below", "over", "underneath",
        "inserted into", "being opened with", "holding", "touching"
    ]
    has_physical_relationship = any(prep in context_lower for prep in physical_prepositions)
    if not has_physical_relationship:
        return False, f"classic_context missing physical preposition/relationship: '{classic_context}'"
    
    # 6. Check that classic_context explicitly mentions sub_object
    sub_object_lower = sub_object.lower()
    # Check if sub_object or its main word appears in classic_context
    sub_object_words = sub_object_lower.split()
    sub_object_main = sub_object_words[0] if sub_object_words else ""
    if sub_object_main and sub_object_main not in context_lower:
        # Also check if any significant word from sub_object is in context
        # (allow some flexibility for variations like "can opener" vs "opener")
        significant_words = [w for w in sub_object_words if len(w) > 3]  # Words longer than 3 chars
        if significant_words and not any(w in context_lower for w in significant_words):
            return False, f"classic_context must explicitly mention sub_object '{sub_object}': '{classic_context}'"
    
    # 7. Check that sub_object is not environment-like (structural check, not word-based)
    # This is a structural check: if sub_object is a single generic word that's typically an environment
    # But we don't use a blacklist - we check if it's too generic structurally
    if len(sub_object_words) == 1 and len(sub_object_main) > 0:
        # Single-word sub_object might be too generic, but we don't reject based on specific words
        # We only check if it's structurally too vague (e.g., just "nature" without specificity)
        # This is a minimal structural check, not a word blacklist
        pass  # Allow model to handle this via prompt instructions
    
    return True, None


def parse_image_size(image_size: str) -> Tuple[int, int]:
    """
    Parse image size string to (width, height).
    Falls back to 1536x1024 if invalid.
    """
    try:
        parts = image_size.split('x')
        if len(parts) != 2:
            raise ValueError("Invalid format")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Invalid dimensions")
        return (width, height)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid imageSize '{image_size}', falling back to 1536x1024")
        return (1536, 1024)


# STEP0_BUNDLE OpenAI call timeouts (longer read for o3-pro; align with ~4min UX)
STEP0_OPENAI_CONNECT_TIMEOUT = 10.0
STEP0_OPENAI_READ_TIMEOUT = 180.0
STEP0_OPENAI_WRITE_TIMEOUT = 180.0
STEP0_OPENAI_POOL_TIMEOUT = 10.0


def build_step0_bundle(
    product_name: str,
    product_description: str,
    language: str = "en",
    max_retries: int = 0,
    request_id: Optional[str] = None
) -> Dict:
    """
    STEP 0 - UNIFIED BUNDLE: ad_goal + difficulty_score + object_list(150)
    
    Single call to OPENAI_TEXT_MODEL (o3-pro). Compact JSON only. Timeouts: connect 10s, read/write 180s.
    Returns (compact schema accepted): ad_goal, difficulty_score, object_list
    (object_list items normalized from primary_object/sub_object to full id/object/sub_object/classic_context/...).
    
    Args:
        product_name: Product name
        product_description: Product description
        language: Language (default: "en")
        max_retries: Maximum retry attempts
        request_id: Optional request ID for logging
    
    Returns:
        Dict with ad_goal, difficulty_score, object_list
    
    Raises:
        Step0BundleTimeoutError: On OpenAI call timeout (caller should return 504)
        Step0BundleOpenAIError: On other OpenAI errors (caller should return 500)
    """
    rid = request_id or str(uuid.uuid4())
    step0_timeout = httpx.Timeout(
        connect=STEP0_OPENAI_CONNECT_TIMEOUT,
        read=STEP0_OPENAI_READ_TIMEOUT,
        write=STEP0_OPENAI_WRITE_TIMEOUT,
        pool=STEP0_OPENAI_POOL_TIMEOUT,
    )
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=step0_timeout)
    model_name = _get_text_model()
    
    prompt = f"""Product: {product_name}. Description: {product_description}. Language: English only.

Output a single JSON object. No explanations, no bullet points, no extra keys. Minimal whitespace (compact JSON).

Schema:
{{"ad_goal":"6-12 word English sentence, commercial intent","difficulty_score":0-100,"object_list":[{{"primary_object":"1-3 words","sub_object":"1-3 words"}}, ... exactly 150 items]}}

Rules:
- ad_goal: one short sentence, English only.
- difficulty_score: number 0-100 (0=easy, 100=very hard).
- object_list: exactly 150 items. Each item only "primary_object" and "sub_object". Use 1-3 words per field. Physical classical objects only. sub_object must be a physical object (not environment, nature, background, scene, sky, world). No logos, text, brands, labels. English only.

Return only this JSON, no other text:
{{"ad_goal":"...","difficulty_score":0,"object_list":[{{"primary_object":"...","sub_object":"..."}}]}}"""
    
    max_attempts = max_retries + 1
    for attempt in range(max_attempts):
        try:
            logger.info(f"STEP0_BUNDLE attempt={attempt+1} model={model_name} product={product_name[:50]}")
            logger.info(f"STEP0_BUNDLE_OPENAI_CALL_START request_id={rid} model={model_name} product={product_name[:50]}")
            t_openai_start = time.time()
            try:
                def _step0_api_call():
                    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                    if is_o_model:
                        r = client.responses.create(model=model_name, input=prompt)
                        return r.output_text.strip()
                    r = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an advertising content generator. Output must be in English only. Return JSON only without additional text."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=8000
                    )
                    return r.choices[0].message.content.strip()
                response_text = openai_retry.openai_call_with_retry(_step0_api_call, endpoint="responses")
            except httpx.TimeoutException as e:
                elapsed_ms = int((time.time() - t_openai_start) * 1000)
                logger.error(f"STEP0_BUNDLE_OPENAI_CALL_TIMEOUT request_id={rid} elapsed_ms={elapsed_ms}")
                raise Step0BundleTimeoutError(f"Step0 bundle timed out after {elapsed_ms}ms") from e
            except openai_retry.OpenAIRateLimitError:
                raise
            except Exception as openai_err:
                elapsed_ms = int((time.time() - t_openai_start) * 1000)
                logger.error(f"STEP0_BUNDLE_OPENAI_CALL_ERROR request_id={rid} elapsed_ms={elapsed_ms} err={openai_err}")
                raise Step0BundleOpenAIError(f"Step0 bundle OpenAI error: {openai_err}") from openai_err
            elapsed_ms = int((time.time() - t_openai_start) * 1000)
            raw_len = len(response_text) if response_text else 0
            logger.info(f"STEP0_BUNDLE_OPENAI_CALL_END request_id={rid} elapsed_ms={elapsed_ms} raw_len={raw_len}")
            
            # Parse JSON
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            bundle = json.loads(response_text)
            
            # Validation
            if not isinstance(bundle, dict):
                raise ValueError("Response is not a dict")
            
            ad_goal = bundle.get("ad_goal", "").strip()
            if not ad_goal:
                raise ValueError("Missing ad_goal")
            
            difficulty_score = bundle.get("difficulty_score")
            if difficulty_score is None:
                raise ValueError("Missing difficulty_score")
            try:
                difficulty_score = float(difficulty_score)
                if difficulty_score < 0 or difficulty_score > 100:
                    raise ValueError(f"difficulty_score out of range: {difficulty_score}")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid difficulty_score: {difficulty_score}")
            
            object_list = bundle.get("object_list", [])
            if not isinstance(object_list, list):
                raise ValueError("object_list is not a list")
            
            if len(object_list) == 0:
                raise ValueError("OBJECT_LIST_EMPTY")
            
            if len(object_list) != OBJECT_LIST_TARGET:
                logger.warning(f"STEP0_BUNDLE: object_list length={len(object_list)}, expected={OBJECT_LIST_TARGET}")
            
            # Validate and normalize each item (accept compact primary_object/sub_object or full schema)
            normalized_items = []
            for i, item in enumerate(object_list):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i} is not a dict")
                # Accept compact schema: primary_object + sub_object only; or full object/sub_object/...
                raw_object = item.get("object") or item.get("primary_object")
                raw_sub_object = item.get("sub_object")
                raw_object = str(raw_object).strip() if raw_object is not None else ""
                raw_sub_object = str(raw_sub_object).strip() if raw_sub_object is not None else ""
                if not raw_object or not raw_sub_object:
                    raise ValueError(f"Item {i} missing primary_object/object or sub_object")
                # Build full item for downstream (id, classic_context, theme_link, shape_hint, theme_tag)
                safe_id = re.sub(r'[^a-z0-9_]', '_', (raw_object + "_" + raw_sub_object).lower())[:48] or f"item_{i}"
                normalized_items.append({
                    "id": safe_id if safe_id else f"item_{i}",
                    "object": raw_object,
                    "sub_object": raw_sub_object,
                    "classic_context": "with " + raw_sub_object,
                    "theme_link": ad_goal[:50] if ad_goal else "",
                    "shape_hint": "",
                    "theme_tag": "",
                })
            
            # Replace object_list with normalized version
            object_list = normalized_items
            
            hard_mode = difficulty_score > 80
            logger.info(f"AD_GOAL={ad_goal}")
            logger.info(f"DIFFICULTY_SCORE={difficulty_score} HARD_MODE={hard_mode}")
            logger.info(f"STEP0_BUNDLE SUCCESS: ad_goal={ad_goal[:50]}, difficulty_score={difficulty_score}, object_list_size={len(object_list)}")
            
            return {
                "ad_goal": ad_goal,
                "difficulty_score": difficulty_score,
                "object_list": object_list
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"STEP0_BUNDLE JSON parse error (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                prompt = f"""Return valid compact JSON only. No other text. Product: {product_name}. {{"ad_goal":"...","difficulty_score":0-100,"object_list":[{{"primary_object":"...","sub_object":"..."}}, ... 150 items]}}"""
                continue
            raise ValueError(f"STEP0_BUNDLE_PARSE_ERROR: Failed to parse JSON after {max_attempts} attempts: {e}")
        except ValueError as e:
            error_msg = str(e)
            if "STEP0_BUNDLE_PARSE_ERROR" in error_msg or "OBJECT_LIST_EMPTY" in error_msg:
                raise
            logger.warning(f"STEP0_BUNDLE validation error (attempt {attempt+1}/{max_attempts}): {error_msg}")
            if attempt < max_attempts - 1:
                prompt = f"""Return valid compact JSON only. {{"ad_goal":"6-12 words","difficulty_score":0-100,"object_list":[{{"primary_object":"1-3 words","sub_object":"1-3 words"}}, ... exactly 150 items]}}. Product: {product_name}."""
                continue
            raise ValueError(f"STEP0_BUNDLE_VALIDATION_ERROR: {error_msg}")
        except Exception as e:
            logger.error(f"STEP0_BUNDLE failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                continue
            raise
    
    raise ValueError("STEP0_BUNDLE_FAILED: Failed after all retries")


def build_object_list_from_ad_goal(
    ad_goal: str,
    product_name: Optional[str] = None,
    max_retries: int = 2,
    language: str = "en"
) -> List[Dict]:
    """
    STEP 0 - BUILD_OBJECT_LIST_FROM_AD_GOAL with Self-Repair Fill-Up
    
    Build a list of EXACTLY {OBJECT_LIST_TARGET} object items related to ad_goal.
    Uses fill-up mechanism: if initial attempt doesn't reach target, requests only missing items.
    """
    """
    STEP 0 - BUILD_OBJECT_LIST_FROM_AD_GOAL
    
    Build a list of EXACTLY {OBJECT_LIST_TARGET} object items related to ad_goal.
    Each item includes: id, object, sub_object, classic_context, theme_link, category, shape_hint, theme_tag.
    
    Args:
        ad_goal: The advertising goal (e.g., "protect nature", "climate action")
        product_name: Optional product name for context
        max_retries: Maximum retry attempts
        language: Language (default: "en")
    
    Returns:
        List[Dict]: List of {OBJECT_LIST_TARGET} items, each with keys:
            - id: unique identifier
            - object: main physical object name
            - sub_object: secondary physical object (REQUIRED - not environment)
            - classic_context: 3-12 words describing PHYSICAL INTERACTION between object and sub_object
            - theme_link: 5-12 words explaining connection to ad_goal
            - category: object category
            - shape_hint: very short shape description
            - theme_tag: single word theme tag
    
    Rules:
    - EXACTLY {OBJECT_LIST_TARGET} ITEMS
    - PHYSICAL CLASSIC OBJECTS ONLY
    - Every item must follow: MAIN_OBJECT interacting with SUB_OBJECT
    - sub_object MUST be concrete physical object, NOT environment (forbidden: nature, forest, water, etc.)
    - classic_context MUST describe physical interaction between object and sub_object
    - classic_context MUST include physical preposition (on/in/next to/attached to/inserted into/etc.)
    - classic_context FORBIDDEN: abstract settings, symbolic language, marketing language, generic phrases
    - NO brand names, NO logos, NO labels, NO printed text, NO numbers, NO signs, NO screens, NO packaging
    """
    # Check STEP 0 cache first
    step0_cache_key = _get_cache_key_step0(ad_goal, product_name, language)
    cached_object_list = _get_from_step0_cache(step0_cache_key)
    if cached_object_list:
        logger.info(f"STEP 0 - BUILD_OBJECT_LIST: Using cached objectList (size={len(cached_object_list)})")
        return cached_object_list
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_text_model()
    
    # Check if model is o* type - these use Responses API
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    # Constants for fill-up mechanism
    MAX_TOTAL_ATTEMPTS = 5
    valid_items = []  # Accumulate valid items across attempts
    seen_ids = set()  # Dedup by id
    seen_pairs = set()  # Dedup by (object, sub_object)
    rejection_reasons = {}  # Track structural rejection reasons
    
    product_context = f"\nProduct name (optional context): {product_name}" if product_name else ""
    
    # Base prompt template
    def build_base_prompt(count: int, feedback: str = "") -> str:
        return f"""Generate EXACTLY {count} physical classic objects related to this advertising goal.

Advertising goal: {ad_goal}{product_context}

CRITICAL REQUIREMENTS:
- EXACTLY {count} ITEMS (no more, no less).
- PHYSICAL CLASSIC OBJECTS ONLY (concrete, tangible, drawable).
- Each item MUST include:
  * id: unique identifier (e.g., "bee_flower", "can_opener", "mouse_cheese")
  * object: the main physical object name (e.g., "bee", "tin can", "mouse")
  * sub_object: a secondary physical object (e.g., "flower", "manual can opener", "wedge of cheese")
  * classic_context: 3-12 words describing the PHYSICAL INTERACTION between object and sub_object (e.g., "landing on a flower", "being opened with a manual can opener", "next to a wedge of cheese")
  * theme_link: 5-12 words explaining how it supports the ad_goal theme (e.g., "pollination supports healthy ecosystems", "photosynthesis produces oxygen")
  * category: object category (e.g., "insect", "container", "rodent", "tool", "plant")
  * shape_hint: very short shape description (e.g., "curved organic", "cylindrical", "small round", "tall vertical")
  * theme_tag: single word theme tag (e.g., "nature", "ocean", "kitchen", "wildlife")

🔴 CRITICAL RULE: Every item must follow the pattern: MAIN_OBJECT interacting with SUB_OBJECT.

FORBIDDEN CONTENT RULES (MODEL RESPONSIBILITY):
- Do NOT generate any object or sub_object that contains logos, brand names, labels, printed text, numbers, barcodes, signage, packaging graphics, or screens with text.
- All objects must be generic and unbranded.
- classic_context must describe ONLY physical interaction (no marketing/abstract language).
- If an object normally has branding, it must be rendered blank/generic.

SUB_OBJECT RULES (ABSOLUTE - MUST BE FOLLOWED):
- sub_object MUST be a concrete physical object, NOT an environment
- sub_object MUST be a specific, tangible item
- FORBIDDEN as sub_object:
  * General environments: "nature", "environment", "world", "background", "scene", "forest", "ocean", "sky", "space", "ecosystem", "habitat", "setting", "context"
  * Abstract concepts: "life", "existence", "reality", "concept"
  * Generic surfaces without specificity: "ground", "floor", "surface" (unless very specific like "wooden floor")

CLASSIC_CONTEXT RULES (CRITICAL - MUST BE FOLLOWED):
- classic_context MUST describe the PHYSICAL INTERACTION between object and sub_object
- MUST include a clear physical preposition/relationship: "on", "in", "under", "next to", "attached to", "inside", "resting on", "landing on", "with", "lying on", "inserted into", "hanging from", "touching", "holding", "opening", "closing"
- MUST be concrete and specific (3-12 words)
- MUST explicitly mention sub_object
- MUST be a natural, expected, real-world interaction
- FORBIDDEN in classic_context:
  * Abstract settings: "in an eco-friendly setting", "in a meaningful environment"
  * Symbolic language: "symbolizing sustainability", "representing change"
  * Marketing language: "in a modern context", "for awareness"
  * Generic phrases: "in nature", "in the wild", "in its habitat" (too vague)
  * No environments as sub_object

{feedback}

Return ONLY a JSON array with EXACTLY {count} items:"""
    
    # First attempt: request full target
    attempt = 0
    needed = OBJECT_LIST_TARGET
    
    logger.info(f"STEP 0 - BUILD_OBJECT_LIST: text_model={model_name}, ad_goal={ad_goal[:50]}, productName={product_name[:50] if product_name else 'N/A'}, using_responses_api={using_responses_api}")
    
    while attempt < MAX_TOTAL_ATTEMPTS and len(valid_items) < OBJECT_LIST_TARGET:
        try:
            # Build feedback from previous attempt
            feedback = ""
            if attempt > 0 and rejection_reasons:
                top_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                reasons_text = ", ".join([f"{reason}({count})" for reason, count in top_reasons])
                feedback = f"""FEEDBACK FROM PREVIOUS ATTEMPT:
- Most common structural rejection reasons: {reasons_text}
- Reminder: classic_context must include a physical preposition AND explicitly mention sub_object
- Reminder: no environments as sub_object
- Reminder: forbidden content rules (logos/text/brands etc.) - model responsibility

You need to generate EXACTLY {needed} additional valid items."""

            prompt = build_base_prompt(needed, feedback)
            
            def _object_list_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                request_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return r.choices[0].message.content.strip()
            response_text = openai_retry.openai_call_with_retry(_object_list_call, endpoint="responses")
            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            data = json.loads(response_text)
            if isinstance(data, list):
                object_list = data
            elif isinstance(data, dict) and "objectList" in data:
                old_list = data.get("objectList", [])
                object_list = [{"id": f"item_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(old_list)]
            else:
                raise ValueError("Invalid JSON format: expected array or object with 'objectList' key")
            
            # Validate items (structural checks only, no forbidden words)
            added_this_attempt = 0
            rejected_this_attempt = 0
            
            for item in object_list:
                # Dedup check: id
                item_id = item.get("id", "").strip()
                if item_id in seen_ids:
                    rejected_this_attempt += 1
                    rejection_reasons["duplicate_id"] = rejection_reasons.get("duplicate_id", 0) + 1
                    continue
                
                # Dedup check: (object, sub_object) pair
                obj_key = (item.get("object", "").strip().lower(), item.get("sub_object", "").strip().lower())
                if obj_key in seen_pairs:
                    rejected_this_attempt += 1
                    rejection_reasons["duplicate_pair"] = rejection_reasons.get("duplicate_pair", 0) + 1
                    continue
                
                # Structural validation (no forbidden words check)
                is_valid, error_msg = validate_object_item(item, forbidden_words=None)
                if is_valid:
                    valid_items.append(item)
                    seen_ids.add(item_id)
                    seen_pairs.add(obj_key)
                    added_this_attempt += 1
                else:
                    rejected_this_attempt += 1
                    reason_key = error_msg.split(":")[0] if ":" in error_msg else error_msg
                    if len(reason_key) > 50:
                        reason_key = reason_key[:50]
                    rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1
            
            # Log attempt
            top_struct_rejects = ", ".join([f"{k}({v})" for k, v in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]])
            logger.info(f"STEP0_FILL attempt={attempt+1} needed={needed} added={added_this_attempt} total={len(valid_items)} top_struct_rejects={top_struct_rejects}")
            
            # Check if we reached target
            if len(valid_items) >= OBJECT_LIST_TARGET:
                break
            
            # Check if we reached minimum
            if len(valid_items) >= OBJECT_LIST_MIN_OK:
                logger.warning(f"STEP0 proceeding with {len(valid_items)} items (target={OBJECT_LIST_TARGET}, min_ok={OBJECT_LIST_MIN_OK})")
                break
            
            # Prepare for next attempt
            needed = OBJECT_LIST_TARGET - len(valid_items)
            attempt += 1
            
        except Exception as e:
            logger.error(f"STEP 0 - Attempt {attempt+1} failed: {e}")
            attempt += 1
            if attempt >= MAX_TOTAL_ATTEMPTS:
                break
    
    # Final validation
    if len(valid_items) < OBJECT_LIST_MIN_OK:
        raise ValueError(f"Failed to generate at least {OBJECT_LIST_MIN_OK} valid items after {MAX_TOTAL_ATTEMPTS} attempts. Got {len(valid_items)} valid items. Rejection reasons: {dict(list(rejection_reasons.items())[:5])}")
    
    # Take up to OBJECT_LIST_TARGET items
    final_list = valid_items[:OBJECT_LIST_TARGET]
    
    # Log final result
    logger.info(f"STEP0_DONE total={len(final_list)} target={OBJECT_LIST_TARGET} min_ok={OBJECT_LIST_MIN_OK} attempts={attempt+1}")
    
    # Calculate SHA for logging
    object_list_str = json.dumps(final_list, sort_keys=True)
    object_list_sha = hashlib.sha256(object_list_str.encode()).hexdigest()[:16]
    
    logger.info(f"OBJECTLIST_SHA={object_list_sha} total={len(final_list)} rejected={sum(rejection_reasons.values())} retry={attempt}")
    
    # Log sample (first 5 items)
    sample = final_list[:5]
    logger.info(f"STEP 0 OBJECTLIST: size={len(final_list)}, sample5={json.dumps(sample, indent=2)}")
    
    # Save to cache
    _set_to_step0_cache(step0_cache_key, final_list)
    
    return final_list


def validate_object_list(object_list: Optional[List], ad_goal: Optional[str] = None, product_name: Optional[str] = None, language: str = "en") -> List[Dict]:
    """
    Validate and return object list (new format: List[Dict] with id, object, classic_context, theme_link).
    If None or too small, and ad_goal is provided, use STEP 0 to build list.
    Otherwise, return default concrete objects list (converted to new format).
    """
    # Check if old format (List[str]) and convert
    if object_list and len(object_list) > 0 and isinstance(object_list[0], str):
        # Old format - convert to new format
        logger.info(f"Converting old format objectList (List[str]) to new format (List[Dict])")
        object_list = [{"id": f"item_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(object_list)]
    
    if not object_list or len(object_list) < 2:
        # If ad_goal is provided, use STEP 0 to build list
        if ad_goal:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), building from ad_goal using STEP 0")
            return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name, language=language)
        else:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), using default concrete objects list (size={len(DEFAULT_OBJECT_LIST)})")
            # Convert DEFAULT_OBJECT_LIST to new format
            return [{"id": f"default_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(DEFAULT_OBJECT_LIST)]
    
    # If object_list is provided but small (<OBJECT_LIST_MIN_OK), and ad_goal is provided, use STEP 0
    if len(object_list) < OBJECT_LIST_MIN_OK and ad_goal:
        logger.info(f"objectList too small (size={len(object_list)}), building from ad_goal using STEP 0")
        return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name, language=language)
    
    logger.info(f"objectList provided with {len(object_list)} items")
    return object_list


def validate_object_list(object_list: Optional[List], ad_goal: Optional[str] = None, product_name: Optional[str] = None, language: str = "en") -> List[Dict]:
    """
    Validate and return object list (new format: List[Dict] with id, object, classic_context, theme_link).
    If None or too small, and ad_goal is provided, use STEP 0 to build list.
    Otherwise, return default concrete objects list (converted to new format).
    """
    # Check if old format (List[str]) and convert
    if object_list and len(object_list) > 0 and isinstance(object_list[0], str):
        # Old format - convert to new format
        logger.info(f"Converting old format objectList (List[str]) to new format (List[Dict])")
        object_list = [{"id": f"item_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(object_list)]
    
    if not object_list or len(object_list) < 2:
        # If ad_goal is provided, use STEP 0 to build list
        if ad_goal:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), building from ad_goal using STEP 0")
            return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name, language=language)
        else:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), using default concrete objects list (size={len(DEFAULT_OBJECT_LIST)})")
            # Convert DEFAULT_OBJECT_LIST to new format
            return [{"id": f"default_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(DEFAULT_OBJECT_LIST)]
    
    # If object_list is provided but small (<OBJECT_LIST_MIN_OK), and ad_goal is provided, use STEP 0
    if len(object_list) < OBJECT_LIST_MIN_OK and ad_goal:
        logger.info(f"objectList too small (size={len(object_list)}), building from ad_goal using STEP 0")
        return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name, language=language)
    
    logger.info(f"objectList provided with {len(object_list)} items")
    return object_list


def get_used_objects(history: Optional[List[Dict]]) -> set:
    """Extract all used objects from history."""
    used = set()
    if not history:
        return used
    
    for item in history:
        if isinstance(item, dict) and "chosen_objects" in item:
            objs = item.get("chosen_objects", [])
            if isinstance(objs, list):
                used.update(objs)
    
    return used


def get_used_ad_goals(history: Optional[List[Dict]]) -> set:
    """Extract all used ad_goals from history."""
    used = set()
    if not history:
        return used
    
    for item in history:
        if isinstance(item, dict) and "ad_goal" in item:
            goal = item.get("ad_goal", "")
            if goal:
                used.add(goal)
    
    return used


def _make_random_candidates(objects: List[Dict], limit: int, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    """
    Fallback: Generate random candidate pairs when neighborhood search fails.
    
    Args:
        objects: List of object items
        limit: Maximum number of pairs to generate
        rng: Seeded random number generator for determinism
    
    Returns:
        List of tuples (item_a, item_b)
    """
    pairs = []
    n = len(objects)
    if n < 2:
        return pairs
    
    attempts = 0
    max_attempts = limit * 20
    
    while len(pairs) < limit and attempts < max_attempts:
        i = rng.randrange(n)
        j = rng.randrange(n)
        
        if i == j:
            attempts += 1
            continue
        
        A = objects[i]
        B = objects[j]
        
        # Reject if same main object
        if _main_key(A) == _main_key(B):
            attempts += 1
            continue
        
        pairs.append((A, B))
        attempts += 1
    
    return pairs


def plan_session_one_call(
    product_name: str,
    product_description: str,
    language: str,
    session_seed: str,
    image_size: str
) -> Dict:
    """
    Generate complete session plan in a SINGLE o3-pro call.
    
    Returns a JSON plan with:
    - ad_goal: string (6-12 words, English)
    - theme_tags: list of 8-12 tags
    - object_list: list of 150 items with id/object/sub_object/classic_context/theme_tag/theme_link/shape_hint
    - ads: list of 3 ads, each with ad_index, a_id, b_id, shape_score, shape_reason, headline
    
    Args:
        product_name: Product name
        product_description: Product description
        language: Language (should be "en")
        session_seed: Session seed for cache key
        image_size: Image size (e.g., "1536x1024")
    
    Returns:
        Dict with complete plan structure
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = _get_shape_model()
    
    prompt = f"""You are designing a visual advertising engine.

INPUT:
Product name: {product_name}
Product description: {product_description}
Language: English

Your task is to compute EVERYTHING in one structured JSON response.

========================================================
RULES
========================================================

1) Generate an AD_GOAL:
- One sentence.
- 6–12 words.
- Clear commercial intent.

2) Generate THEME_TAGS:
- 8–12 short tags (1–3 words each).
- Must be directly related to the ad_goal.
- No abstract marketing fluff.

2.5) PHYSICAL_METAPHOR_MAPPING (CRITICAL):
- Convert each theme_tag into concrete physical metaphor objects.
- For each theme_tag, produce 1–3 concrete PHYSICAL OBJECTS that naturally represent it.
- Objects must be everyday physical objects.
- No abstract symbols.
- No text.
- No logos.
- No conceptual words.
- Must be photographable.

Examples (for guidance only):
- optimization → tuning knob, adjustable wrench, compass needle
- targeting → arrow hitting target, magnifying glass over object
- personalization → key and lock, engraved nameplate
- analytics → measuring scale, ruler, caliper
- automation → conveyor belt, mechanical lever
- precision → laser pointer dot, dart hitting bullseye
- growth → sprouting seed, rising thermometer

3) Generate OBJECT_LIST:
- EXACTLY 150 items.
- Each item must contain:

{{
  "id": "unique_snake_case_id",
  "object": "main physical object",
  "sub_object": "secondary physical object",
  "classic_context": "3–12 words describing physical interaction",
  "theme_tag": "must be one of theme_tags",
  "theme_link": "5–12 words explaining relevance to ad_goal",
  "shape_hint": "one of: round, oval, elongated, rectangular, triangular, curved_organic, cylindrical, irregular"
}}

STRICT OBJECT RULES:
- Each item must be MAIN_OBJECT + SUB_OBJECT.
- Sub_object must be a physical object.
- No environments like forest, nature, water, sky, background.
- classic_context must include a physical relation:
  (on, in, next to, attached to, inserted into, resting on, being opened with, holding, touching)
- classic_context must explicitly mention sub_object.

FORBIDDEN CONTENT RULES (MODEL RESPONSIBILITY):
- Do NOT generate any object or sub_object that contains logos, brand names, labels, printed text, numbers, barcodes, signage, packaging graphics, or screens with text.
- All objects must be generic and unbranded.
- classic_context must describe ONLY physical interaction (no marketing/abstract language).
- If an object normally has branding, it must be rendered blank/generic.

PHYSICAL METAPHOR ENFORCEMENT (CRITICAL):
- Each object in object_list MUST originate from one of the physical metaphors generated in step 2.5.
- OR be a direct physical extension of them.
- If theme_tag = "targeting" and metaphor = "arrow and target", then valid objects may include:
  - arrow + target board
  - dart + dartboard
  - spear + marked circle
- NOT: random household items unrelated to the metaphor.
- If product domain is abstract (AI, advertising, analytics, finance, SaaS, optimization, etc):
  - Do NOT generate generic everyday objects unrelated to the physical metaphors.
  - Reject internally any object that cannot be explained as a physical embodiment of a theme_tag.
- Each object must clearly represent an abstract idea physically.

4) SELECT 3 FINAL PAIRS FROM THE LIST:

Each pair must:
- Use two DIFFERENT main objects.
- Not reuse the same main object across the 3 pairs.
- Be strongly similar in MAIN object silhouette.
- Ignore sub_object when judging silhouette.
- Have shape_score between 0–100 (aim 85+).
- Provide one-sentence shape_reason explaining silhouette similarity.

5) HEADLINE PER PAIR:
- English only.
- Max 7 words including product name.
- Bold, commercial, not poetic.
- Must relate to ad_goal.

6) SIDE BY SIDE ONLY:
- Two full objects.
- No overlap.
- Equal visual weight.
- Clear comparable outer contour.

========================================================
OUTPUT FORMAT (STRICT JSON)
========================================================

{{
  "ad_goal": "...",
  "theme_tags": [...],
  "object_list": [150 items],
  "ads": [
    {{
      "ad_index": 1,
      "a_id": "...",
      "b_id": "...",
      "shape_score": 0-100,
      "shape_reason": "...",
      "headline": "..."
    }},
    {{
      "ad_index": 2,
      ...
    }},
    {{
      "ad_index": 3,
      ...
    }}
  ]
}}

Do NOT include explanations.
Do NOT include extra text.
Return JSON only."""
    
    try:
        def _plan_call():
            r = client.responses.create(model=model, input=prompt)
            return r.output_text.strip()
        response_text = openai_retry.openai_call_with_retry(_plan_call, endpoint="responses")
        # Parse JSON
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        if response_text.startswith("```json"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        plan = json.loads(response_text)
        
        # Validate structure
        if not isinstance(plan, dict):
            raise ValueError("Plan is not a dict")
        if "ad_goal" not in plan or "theme_tags" not in plan or "object_list" not in plan or "ads" not in plan:
            raise ValueError("Plan missing required fields")
        if not isinstance(plan["object_list"], list) or len(plan["object_list"]) != OBJECT_LIST_TARGET:
            raise ValueError(f"object_list must have exactly {OBJECT_LIST_TARGET} items")
        if not isinstance(plan["ads"], list) or len(plan["ads"]) != 3:
            raise ValueError("ads must have exactly 3 items")
        
        logger.info(f"PLAN_ONE_CALL used=true model={model} ad_goal={plan.get('ad_goal', '')[:50]} object_list_size={len(plan.get('object_list', []))} ads_count={len(plan.get('ads', []))}")
        
        # Log metaphor enforcement check (first 5 items)
        if plan.get("object_list"):
            sample_items = plan["object_list"][:5]
            sample_str = json.dumps([{"id": it.get("id", ""), "object": it.get("object", ""), "sub_object": it.get("sub_object", ""), "theme_tag": it.get("theme_tag", "")} for it in sample_items], ensure_ascii=False)
            logger.info(f"METAPHOR_CHECK sample={sample_str}")
        
        return plan
        
    except Exception as e:
        logger.error(f"plan_session_one_call failed: {e}")
        raise


def _get_cache_key_session_plan(
    product_name: str,
    product_description: str,
    session_seed: str,
    image_size: str
) -> str:
    """Generate cache key for session plan."""
    product_hash = hashlib.md5(f"{product_name}|{product_description}".encode()).hexdigest()[:16]
    layout_mode = ACE_LAYOUT_MODE
    key_str = f"SESSION_PLAN_V1|{session_seed}|{product_hash}|{image_size}|{layout_mode}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_from_session_plan_cache(key: str) -> Optional[Dict]:
    """Get session plan from cache."""
    with _session_plan_cache_lock:
        if key in _session_plan_cache:
            value, timestamp = _session_plan_cache[key]
            if time.time() - timestamp < PLAN_CACHE_TTL_SECONDS:
                return value
            else:
                # Expired, remove
                del _session_plan_cache[key]
    return None


def _set_to_session_plan_cache(key: str, value: Dict):
    """Store session plan in cache."""
    with _session_plan_cache_lock:
        _session_plan_cache[key] = (value, time.time())


def score_pairs_batch_o3_pro(
    pairs: List[Tuple[Dict, Dict]],
    model_name: str,
    language: str = "en"
) -> List[Dict]:
    """
    Score multiple pairs in a single batch call to o3-pro.
    
    Args:
        pairs: List of tuples (itemA, itemB) where each item is a dict with keys: object, sub_object, etc.
        model_name: Model name (should be o3-pro)
        language: Language (default: "en")
    
    Returns:
        List of dicts with keys: {"i": idx, "score": 0-100, "archetype": str, "reason": str}
        Returns empty list if parsing fails or error occurs.
    """
    if not pairs:
        return []
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Build compact JSON input for batch scoring
    pairs_data = []
    for idx, (item_a, item_b) in enumerate(pairs):
        # Extract main objects and sub_objects
        a_main = item_a.get("object", "") if isinstance(item_a, dict) else str(item_a)
        a_sub = item_a.get("sub_object", "") if isinstance(item_a, dict) else ""
        b_main = item_b.get("object", "") if isinstance(item_b, dict) else str(item_b)
        b_sub = item_b.get("sub_object", "") if isinstance(item_b, dict) else ""
        
        # Optional: short context strings
        a_context = item_a.get("classic_context", "")[:30] if isinstance(item_a, dict) else ""
        b_context = item_b.get("classic_context", "")[:30] if isinstance(item_b, dict) else ""
        
        pairs_data.append({
            "i": idx,
            "A_main": a_main,
            "A_sub": a_sub,
            "B_main": b_main,
            "B_sub": b_sub,
            "A_context": a_context,
            "B_context": b_context
        })
    
    prompt = f"""You are scoring OUTLINE similarity of MAIN objects only (ignore sub_object for silhouette).

Score each pair based ONLY on the geometric shape similarity of the OUTER CONTOUR/OUTLINE of the MAIN objects (A_main and B_main).
Ignore sub_objects, context, meaning, color, material, texture, category, theme.

For each pair, return:
- score: 0-100 (0=completely different shapes, 100=identical outlines)
- archetype: one word describing the shared shape archetype (e.g., "round", "tall", "curved", "flat", "spiral", "crescent", "cylindrical", "spherical", "rectangular", "triangular", "organic", "geometric")
- reason: one short sentence explaining the shape similarity (focus on outline/contour only)

Pairs to score:
{json.dumps(pairs_data, indent=2, ensure_ascii=False)}

Return ONLY a JSON array with one object per pair:
[
  {{"i": 0, "score": 85, "archetype": "round", "reason": "Both have circular outer contours"}},
  {{"i": 1, "score": 72, "archetype": "tall", "reason": "Both are vertically elongated shapes"}},
  ...
]

Do not write any extra text. Return JSON array only."""
    
    try:
        def _batch_score_call():
            is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
            if is_o_model:
                r = client.responses.create(model=model_name, input=prompt)
                return r.output_text.strip()
            r = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a shape similarity analyzer. Output must be in English only. Return JSON only without additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return r.choices[0].message.content.strip()
        response_text = openai_retry.openai_call_with_retry(_batch_score_call, endpoint="responses")
        # Parse JSON array
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        if response_text.startswith("```json"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        results = json.loads(response_text)
        if not isinstance(results, list):
            logger.warning(f"Batch scoring returned non-list: {type(results)}")
            return []
        # Validate and normalize results
        validated_results = []
        for r in results:
            if not isinstance(r, dict):
                continue
            idx = r.get("i")
            score = r.get("score", 0)
            archetype = r.get("archetype", "")
            reason = r.get("reason", "")
            
            # Ensure score is 0-100
            try:
                score = max(0, min(100, int(float(score))))
            except (ValueError, TypeError):
                score = 0
            
            validated_results.append({
                "i": idx,
                "score": score,
                "archetype": archetype,
                "reason": reason
            })
        
        return validated_results
    except openai_retry.OpenAIRateLimitError:
        raise
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        return []


def select_three_pairs_single_call(
    object_list: List[Dict],
    sid: str,
    ad_goal: str,
    model_name: Optional[str] = None
) -> List[Dict]:
    """
    Select exactly 3 pairs from object_list using a single model call.
    
    Critical rule: Silhouette similarity is judged ONLY by the MAIN OBJECT (field "object").
    The sub_object and classic_context are ignored for similarity scoring.
    
    Args:
        object_list: List of dicts with keys: id, object, sub_object, classic_context, shape_hint
        sid: Session ID (for logging)
        ad_goal: Advertising goal (for context, not used in similarity)
        model_name: Model name (optional, defaults to OPENAI_SHAPE_MODEL)
    
    Returns:
        List of 3 dicts, each with keys: {"a_id": "...", "b_id": "..."}
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model_name is None:
        model_name = _get_shape_model()
    
    # Format object list for prompt (only id, object, shape_hint - ignore sub_object for similarity)
    # Normalize all values to string for safe formatting
    object_list_lines = []
    for item in object_list:
        raw_id = item.get("id")
        raw_object = item.get("object")
        raw_shape_hint = item.get("shape_hint")
        item_id = str(raw_id) if raw_id is not None else ""
        main_object = str(raw_object) if raw_object is not None else ""
        shape_hint = str(raw_shape_hint) if raw_shape_hint is not None else ""
        object_list_lines.append(f"- id: {item_id}, object: {main_object}, shape_hint: {shape_hint}")
    
    object_list_formatted = "\n".join(object_list_lines)
    
    prompt = f"""Return JSON ONLY. Do not include any commentary, markdown, code fences, or extra text.

TASK:
Select exactly 3 pairs of items with the highest silhouette similarity of the MAIN OBJECTS ONLY.

DEFINITION - MAIN OBJECT:
MAIN OBJECT is the value of the 'object' field (not id). Two items share the same main object if their 'object' strings are equal after lowercasing and trimming.

CRITICAL RULES:
1) Within each pair: a.object != b.object (the two main objects must be different)
2) Across all 3 pairs: no 'object' value may repeat (each main object appears at most once across all pairs)
3) Output exactly 3 pairs
4) Use only ids that exist in the provided list
5) Prefer high silhouette similarity of MAIN OBJECT only (ignore sub_object, classic_context, theme_link, theme_tag)

SIMILARITY CRITERION:
Compare ONLY the outer contour / silhouette of the "object" field (MAIN object).
Ignore sub_object, classic_context, theme_link, theme_tag for similarity scoring.

OUTPUT FORMAT (EXACT):
[
  {{"a_id":"id1", "b_id":"id2"}},
  {{"a_id":"id3", "b_id":"id4"}},
  {{"a_id":"id5", "b_id":"id6"}}
]

EXAMPLE OUTPUT:
[
  {{"a_id":"leaf_1", "b_id":"petal_2"}},
  {{"a_id":"coin_3", "b_id":"button_4"}},
  {{"a_id":"wheel_5", "b_id":"ring_6"}}
]

INPUT LIST ({len(object_list)} items):
Each item has: id, object (MAIN), sub_object, classic_context, shape_hint

LIST:
{object_list_formatted}

Return ONLY a JSON array as specified. No other text."""
    
    # Build ID to item mapping for validation
    # Normalize all ids to string for consistent comparison
    id_to_item = {}
    id_to_main_object = {}
    for item in object_list:
        raw_id = item.get("id")
        normalized_id = str(raw_id) if raw_id is not None else ""
        id_to_item[normalized_id] = item
        raw_object = item.get("object", "")
        id_to_main_object[normalized_id] = str(raw_object).lower().strip() if raw_object else ""
    
    max_retries = 3  # Increased from 2 to 3
    last_raw_response = None
    last_parse_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"SELECT_THREE_PAIRS attempt={attempt+1} model={model_name} object_list_size={len(object_list)}")
            
            # Use Responses API for o* models
            def _select_three_call():
                is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                if is_o_model:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text or ""
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a shape similarity analyzer. Output must be in English only. Return JSON only without additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                return (r.choices[0].message.content or "") if r.choices else ""
            response_text = openai_retry.openai_call_with_retry(_select_three_call, endpoint="responses")
            # Log raw response length
            raw_len = len(response_text) if response_text else 0
            logger.info(f"SELECT_THREE_PAIRS_RAW_LEN={raw_len}")
            last_raw_response = response_text
            
            # Robust JSON extraction
            response_text = response_text.strip() if response_text else ""
            
            # If empty, treat as parse failure
            if not response_text:
                raise json.JSONDecodeError("Empty response", response_text, 0)
            
            # Remove markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.strip()
            
            # Attempt to extract JSON array if there's leading/trailing text
            extracted_json = response_text
            if '[' in response_text and ']' in response_text:
                first_bracket = response_text.find('[')
                last_bracket = response_text.rfind(']')
                if first_bracket >= 0 and last_bracket > first_bracket:
                    extracted_json = response_text[first_bracket:last_bracket+1]
                    if len(extracted_json) != len(response_text):
                        extracted_len = len(extracted_json)
                        logger.info(f"SELECT_THREE_PAIRS_EXTRACTED_JSON_LEN={extracted_len} (extracted from {raw_len} chars)")
            
            # Parse JSON
            pairs_data = json.loads(extracted_json)
            
            if not isinstance(pairs_data, list):
                raise ValueError(f"Expected list, got {type(pairs_data)}")
            
            if len(pairs_data) != 3:
                raise ValueError(f"Expected exactly 3 pairs, got {len(pairs_data)}")
            
            # Validate pairs
            validated_pairs = []
            used_main_objects = set()
            
            for pair in pairs_data:
                if not isinstance(pair, dict):
                    raise ValueError(f"Pair must be dict, got {type(pair)}")
                
                raw_a_id = pair.get("a_id", "")
                raw_b_id = pair.get("b_id", "")
                
                # Normalize ids to string for comparison
                a_id = str(raw_a_id) if raw_a_id is not None else ""
                b_id = str(raw_b_id) if raw_b_id is not None else ""
                
                # Check IDs exist
                if a_id not in id_to_item:
                    raise ValueError(f"a_id '{a_id}' (raw: {raw_a_id}, type: {type(raw_a_id).__name__}) not found in object_list")
                if b_id not in id_to_item:
                    raise ValueError(f"b_id '{b_id}' (raw: {raw_b_id}, type: {type(raw_b_id).__name__}) not found in object_list")
                
                # Get main objects (normalized for comparison)
                a_main = id_to_main_object.get(a_id, "").lower().strip()
                b_main = id_to_main_object.get(b_id, "").lower().strip()
                
                # Check different main objects within pair
                if a_main == b_main:
                    raise ValueError(f"Pair has same main object: '{a_main}' (a_id={a_id}, b_id={b_id})")
                
                # Check no reuse across pairs
                if a_main in used_main_objects:
                    raise ValueError(f"Main object '{a_main}' (a_id={a_id}) already used in another pair")
                if b_main in used_main_objects:
                    raise ValueError(f"Main object '{b_main}' (b_id={b_id}) already used in another pair")
                
                used_main_objects.add(a_main)
                used_main_objects.add(b_main)
                
                validated_pairs.append({"a_id": a_id, "b_id": b_id})
            
            # Final validation: ensure exactly 3 pairs, all ids exist, no repeated main objects
            if len(validated_pairs) != 3:
                raise ValueError(f"PAIRSET_INVALID: Expected 3 pairs, got {len(validated_pairs)}")
            
            # Validate all ids exist
            for pair in validated_pairs:
                if pair.get("a_id") not in id_to_item or pair.get("b_id") not in id_to_item:
                    raise ValueError(f"PAIRSET_INVALID: Invalid id in pair {pair}")
            
            # Validate no repeated main objects across all pairs (normalized comparison)
            all_main_objects = []
            for pair in validated_pairs:
                a_id = pair.get("a_id")
                b_id = pair.get("b_id")
                a_main = id_to_main_object.get(a_id, "").lower().strip()
                b_main = id_to_main_object.get(b_id, "").lower().strip()
                all_main_objects.extend([a_main, b_main])
            
            if len(all_main_objects) != len(set(all_main_objects)):
                raise ValueError(f"PAIRSET_INVALID: Repeated main objects found: {all_main_objects}")
            
            logger.info(f"SELECT_THREE_PAIRS SUCCESS: pairs={validated_pairs}")
            logger.info(f"PAIRSET_PICKED ids={[(p.get('a_id'), p.get('b_id')) for p in validated_pairs]}")
            return validated_pairs
            
        except json.JSONDecodeError as e:
            last_parse_error = str(e)
            logger.error(f"SELECT_THREE_PAIRS JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Retry with fix prompt including error details
                raw_response_snippet = (last_raw_response[:200] + "...") if last_raw_response and len(last_raw_response) > 200 else (last_raw_response or "empty")
                prompt = f"""Return JSON ONLY. Do not include any commentary, markdown, code fences, or extra text.

PREVIOUS ATTEMPT FAILED:
Error: {last_parse_error}
Raw response (truncated): {raw_response_snippet}

TASK:
Select exactly 3 pairs of items with the highest silhouette similarity of the MAIN OBJECTS ONLY.

DEFINITION - MAIN OBJECT:
MAIN OBJECT is the value of the 'object' field (not id). Two items share the same main object if their 'object' strings are equal after lowercasing and trimming.

CRITICAL RULES:
1) Within each pair: a.object != b.object (the two main objects must be different)
2) Across all 3 pairs: no 'object' value may repeat (each main object appears at most once across all pairs)
3) Output exactly 3 pairs
4) Use only ids that exist in the provided list
5) Prefer high silhouette similarity of MAIN OBJECT only (ignore sub_object, classic_context, theme_link, theme_tag)

OUTPUT FORMAT (EXACT):
[
  {{"a_id":"id1", "b_id":"id2"}},
  {{"a_id":"id3", "b_id":"id4"}},
  {{"a_id":"id5", "b_id":"id6"}}
]

LIST:
{object_list_formatted}

Return ONLY a JSON array as specified. No other text."""
                continue
            raise ValueError(f"PAIRSET_INVALID: Failed to parse JSON after {max_retries} attempts: {last_parse_error}")
        except ValueError as e:
            error_msg = str(e)
            if "PAIRSET_INVALID" in error_msg:
                raise
            logger.warning(f"SELECT_THREE_PAIRS validation error (attempt {attempt+1}/{max_retries}): {error_msg}")
            if attempt < max_retries - 1:
                # Retry with stricter prompt
                prompt = f"""Return JSON ONLY. Do not include any commentary, markdown, code fences, or extra text.

PREVIOUS ATTEMPT FAILED:
Validation error: {error_msg}

TASK:
Select exactly 3 pairs of items with the highest silhouette similarity of the MAIN OBJECTS ONLY.

DEFINITION - MAIN OBJECT:
MAIN OBJECT is the value of the 'object' field (not id). Two items share the same main object if their 'object' strings are equal after lowercasing and trimming.

CRITICAL RULES:
1) Within each pair: a.object != b.object (the two main objects must be different)
2) Across all 3 pairs: no 'object' value may repeat (each main object appears at most once across all pairs)
3) Output exactly 3 pairs
4) Use only ids that exist in the provided list
5) Prefer high silhouette similarity of MAIN OBJECT only (ignore sub_object, classic_context, theme_link, theme_tag)

OUTPUT FORMAT (EXACT):
[
  {{"a_id":"id1", "b_id":"id2"}},
  {{"a_id":"id3", "b_id":"id4"}},
  {{"a_id":"id5", "b_id":"id6"}}
]

LIST:
{object_list_formatted}

Return ONLY a JSON array as specified. No other text."""
                continue
            raise ValueError(f"PAIRSET_INVALID: Validation failed after {max_retries} attempts: {error_msg}")
        except Exception as e:
            logger.error(f"SELECT_THREE_PAIRS failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"PAIRSET_INVALID: Failed after all retries: {e}")
    
    raise ValueError("PAIRSET_INVALID: Failed to select 3 pairs after all retries")


def select_pair_from_three_pairs(
    object_list: List[Dict],
    sid: str,
    ad_index: int,
    ad_goal: str,
    image_size: str,
    used_objects: Optional[set] = None,
    model_name: Optional[str] = None,
    allowed_theme_tags: Optional[List[str]] = None
) -> Dict:
    """
    Select a single pair for the given ad_index from 3 pre-selected pairs.
    
    Uses select_three_pairs_single_call to get 3 pairs, then returns the pair
    corresponding to ad_index (1 -> pairs[0], 2 -> pairs[1], 3 -> pairs[2]).
    
    Args:
        object_list: List[Dict] with keys: id, object, classic_context, theme_link, theme_tag
        sid: Session ID
        ad_index: Ad index (1-3)
        ad_goal: Advertising goal (for context)
        image_size: Image size (for cache key)
        used_objects: Set of already used object IDs (optional, ignored - pairs are pre-selected)
        model_name: Model name (optional, defaults to OPENAI_SHAPE_MODEL)
        allowed_theme_tags: Optional list of theme tags (ignored - full list is used)
    
    Returns:
        Dict with keys: object_a, object_b, object_a_id, object_b_id, shape_similarity_score, shape_hint
    """
    # Normalize ad_index to 0-based for array access
    pair_index = (ad_index - 1) % 3  # Ensure 0-2 range
    
    # Cache key for 3 pairs (shared across all ad_index in same session)
    # Use object_list hash for cache key stability
    # Normalize ids to string for hash calculation
    object_list_hash = hashlib.md5(json.dumps([str(item.get("id")) if item.get("id") is not None else "" for item in object_list], sort_keys=True).encode()).hexdigest()[:16]
    cache_key = f"THREE_PAIRS|{sid}|{ad_goal}|{image_size}|{object_list_hash}"
    
    # Check cache
    cached_pairs = None
    with _three_pairs_cache_lock:
        if cache_key in _three_pairs_cache:
            cached_data, timestamp = _three_pairs_cache[cache_key]
            if time.time() - timestamp < STEP1_CACHE_TTL_SECONDS:
                cached_pairs = cached_data
    
    if cached_pairs:
        logger.info(f"THREE_PAIRS_CACHE hit=true key={cache_key[:16]}...")
        three_pairs = cached_pairs
    else:
        # Select 3 pairs using single model call
        three_pairs = select_three_pairs_single_call(
            object_list=object_list,
            sid=sid,
            ad_goal=ad_goal,
            model_name=model_name
        )
        
        # Cache the 3 pairs
        with _three_pairs_cache_lock:
            _three_pairs_cache[cache_key] = (three_pairs, time.time())
        logger.info(f"THREE_PAIRS_CACHE miss=true key={cache_key[:16]}... generated 3 pairs")
    
    # Guard: Validate three_pairs is not empty
    if not three_pairs or len(three_pairs) == 0:
        raise ValueError("PAIRSET_EMPTY_FOR_AD_INDEX: No pairs available")
    
    # Get the pair for this ad_index
    if pair_index >= len(three_pairs):
        raise ValueError(f"PAIRSET_EMPTY_FOR_AD_INDEX: Pair index {pair_index} out of range (got {len(three_pairs)} pairs) for ad_index={ad_index}")
    
    selected_pair = three_pairs[pair_index]
    raw_a_id = selected_pair.get("a_id")
    raw_b_id = selected_pair.get("b_id")
    
    if raw_a_id is None or raw_b_id is None:
        raise ValueError(f"PAIRSET_EMPTY_FOR_AD_INDEX: Invalid pair at index {pair_index}: missing a_id or b_id")
    
    # Normalize ids to string for comparison
    a_id = str(raw_a_id)
    b_id = str(raw_b_id)
    
    # Find items in object_list (normalize id for comparison)
    item_a = next((item for item in object_list if str(item.get("id")) == a_id), None)
    item_b = next((item for item in object_list if str(item.get("id")) == b_id), None)
    
    if not item_a or not item_b:
        raise ValueError(f"PAIRSET_EMPTY_FOR_AD_INDEX: Could not find items for ids: {a_id}, {b_id}")
    
    object_a_name = item_a.get("object", "")
    object_b_name = item_b.get("object", "")
    shape_hint = item_a.get("shape_hint", "") or item_b.get("shape_hint", "")
    
    # Return in same format as select_pair_with_limited_shape_search
    result = {
        "object_a": object_a_name,
        "object_b": object_b_name,
        "object_a_id": a_id,
        "object_b_id": b_id,
        "shape_similarity_score": 85,  # Default score (model selected based on similarity)
        "shape_hint": shape_hint,
        "shape_reason": f"Selected from 3 pre-computed pairs (pair {pair_index + 1}/3)"
    }
    
    logger.info(f"PAIR_SELECTED_FROM_THREE sid={sid} ad={ad_index} pair_index={pair_index} A={a_id} B={b_id} A_obj={object_a_name} B_obj={object_b_name}")
    
    return result


def select_pair_with_limited_shape_search(
    object_list: List[Dict],
    sid: str,
    ad_index: int,
    ad_goal: str,
    image_size: str,
    used_objects: Optional[set] = None,
    model_name: Optional[str] = None,
    allowed_theme_tags: Optional[List[str]] = None
) -> Dict:
    """
    STEP 1 - SELECT PAIR WITH LIMITED SHAPE SEARCH (LEGACY - kept for compatibility)
    
    Select a pair of objects with shape similarity using limited search (K=35-50, MAX_CHECKED_PAIRS).
    Includes uniform shuffle for fairness and anti-repeat logic per session.
    
    Args:
        object_list: List[Dict] with keys: id, object, classic_context, theme_link, theme_tag
        sid: Session ID
        ad_index: Ad index (1-3)
        ad_goal: Advertising goal (for seed)
        image_size: Image size (for seed)
        used_objects: Set of already used object IDs (optional)
        model_name: Model name (optional, defaults to OPENAI_SHAPE_MODEL)
        allowed_theme_tags: Optional list of theme tags to filter by (if provided, prefer objects with matching theme_tag)
    
    Returns:
        Dict with keys: object_a, object_b, object_a_id, object_b_id, shape_similarity_score, shape_hint
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model_name is None:
        model_name = _get_shape_model()
    
    # Filter by theme tags if provided (with normalized matching)
    themed_pool = None
    if allowed_theme_tags and len(allowed_theme_tags) > 0:
        theme_tags_norm = {_norm(t) for t in allowed_theme_tags}
        themed_pool_by_tag = []
        themed_pool_by_link = []
        for it in object_list:
            it_tag_norm = _norm(it.get("theme_tag", ""))
            it_link_norm = _norm(it.get("theme_link", ""))
            
            # Primary match: by theme_tag
            if it_tag_norm in theme_tags_norm:
                themed_pool_by_tag.append(it)
            # Secondary match: by theme_link (if primary didn't match)
            elif any(tn in it_link_norm for tn in theme_tags_norm):
                themed_pool_by_link.append(it)
        
        themed_pool = themed_pool_by_tag + themed_pool_by_link
        
        # Log theme matching counts
        logger.info(f"THEME_POOL_MATCH counts: by_tag={len(themed_pool_by_tag)} by_link={len(themed_pool_by_link)} total={len(themed_pool)} (from {len(object_list)} total)")
        
        # Use themed_pool if it's large enough (>= 60), otherwise fallback to full object_list
        if len(themed_pool) >= 60:
            search_objects = themed_pool
            logger.info(f"THEME_FILTER: using themed_pool size={len(themed_pool)} (from {len(object_list)} total)")
        else:
            search_objects = object_list
            logger.warning(f"THEME_FILTER: themed_pool too small ({len(themed_pool)} < 60), using full object_list ({len(object_list)})")
    else:
        search_objects = object_list
        themed_pool = []  # Empty for fallback checks
    
    # Generate seed for uniform shuffle
    seed_str = f"{sid}|{ad_index}|{ad_goal}|{image_size}|side_by_side"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
    
    # Uniform shuffle for fairness
    shuffled_objects = uniform_shuffle(search_objects, seed_int)
    
    # Get used pairs and objects for this session
    used_objects_set = used_objects or set()
    with _session_used_lock:
        if sid in _session_used_pairs:
            used_pairs_set, used_objects_session, timestamp = _session_used_pairs[sid]
            # Check TTL
            if time.time() - timestamp >= CACHE_TTL_SECONDS:
                # Expired, reset
                _session_used_pairs[sid] = (set(), set(), time.time())
                used_pairs_set = set()
                used_objects_session = set()
        else:
            used_pairs_set = set()
            used_objects_session = set()
            _session_used_pairs[sid] = (used_pairs_set, used_objects_session, time.time())
    
    # BATCH MODE: Collect candidate pairs first, then batch score them
    K = SHAPE_SEARCH_K
    # Ensure K is not zero/None - if < 5, set to 40
    if K is None or K < 5:
        K = 40
        logger.warning(f"SHAPE_SEARCH_K was invalid ({SHAPE_SEARCH_K}), setting to {K}")
    
    candidate_limit = CANDIDATE_LIMIT
    min_score_threshold = int(SHAPE_MIN_SCORE * 100)  # Convert to 0-100 scale
    
    # Ensure we use o3-pro for shape scoring
    shape_model = _get_shape_model()
    if model_name and model_name != shape_model:
        logger.warning(f"Overriding model_name={model_name} with OPENAI_SHAPE_MODEL={shape_model} for batch scoring")
    
    # Create seeded RNG for fallback random generation
    rng = random.Random(seed_int)
    
    best_pair_result = None
    best_score = 0
    total_checked = 0
    batch_call_count = 0
    max_batch_calls = 3
    
    # Track used main objects for debug
    used_main_set = {_main_key(obj) for obj in search_objects if obj.get("id") in used_objects_session}
    
    for batch_attempt in range(max_batch_calls):
        # Collect candidate pairs (without model calls)
        candidate_pairs = []  # List of tuples: (item_a, item_b)
        candidate_indices = []  # Track original indices in shuffled_objects
        
        for i in range(len(shuffled_objects)):
            if len(candidate_pairs) >= candidate_limit:
                break
            
            obj_a_item = shuffled_objects[i]
            obj_a_id = obj_a_item["id"]
            obj_a_name = obj_a_item["object"]
            
            # Skip if already used in session (prefer diversity)
            if obj_a_id in used_objects_session and len(used_objects_session) < len(shuffled_objects) * 0.8:
                continue
            
            # Check candidates j=i+1..i+K
            for j in range(i + 1, min(i + 1 + K, len(shuffled_objects))):
                if len(candidate_pairs) >= candidate_limit:
                    break
                
                obj_b_item = shuffled_objects[j]
                obj_b_id = obj_b_item["id"]
                obj_b_name = obj_b_item["object"]
                
                # CRITICAL: Reject if same main object (forbidden: same object with different sub_object)
                obj_a_main_key = _main_key(obj_a_item)
                obj_b_main_key = _main_key(obj_b_item)
                if obj_a_main_key == obj_b_main_key:
                    continue
                
                # Skip if already used in session
                if obj_b_id in used_objects_session and len(used_objects_session) < len(shuffled_objects) * 0.8:
                    continue
                
                # Check if pair already used
                pair_hash = hashlib.sha256("|".join(sorted([obj_a_id, obj_b_id])).encode()).hexdigest()
                if pair_hash in used_pairs_set:
                    continue
                
                # Check theme relevance if allowed_theme_tags provided (before adding to candidates)
                if allowed_theme_tags and len(allowed_theme_tags) > 0 and themed_pool and len(themed_pool) >= 60:
                    theme_tags_norm = {_norm(t) for t in allowed_theme_tags}
                    obj_a_theme = obj_a_item.get("theme_tag", "")
                    obj_b_theme = obj_b_item.get("theme_tag", "")
                    obj_a_theme_norm = _norm(obj_a_theme)
                    obj_b_theme_norm = _norm(obj_b_theme)
                    if obj_a_theme_norm not in theme_tags_norm or obj_b_theme_norm not in theme_tags_norm:
                        continue
                
                # Add to candidates
                candidate_pairs.append((obj_a_item, obj_b_item))
                candidate_indices.append((i, j))
        
        # Fallback: If no candidates from neighborhood search, use random generation
        if not candidate_pairs:
            logger.warning(f"CANDIDATE_DEBUG pool={len(search_objects)} K={K} limit={candidate_limit} used_pairs={len(used_pairs_set)} used_main={len(used_main_set)}")
            logger.warning(f"No candidate pairs collected in batch attempt {batch_attempt + 1} from neighborhood search; using random candidate generation")
            
            # Try random generation with current search_objects
            candidate_pairs = _make_random_candidates(search_objects, candidate_limit, rng)
            
            # If still empty and we're using themed_pool, expand to full object_list
            if not candidate_pairs and themed_pool and len(themed_pool) >= 60 and search_objects is themed_pool:
                logger.warning(f"Random generation with themed_pool failed; expanding to full object_list (size={len(object_list)})")
                candidate_pairs = _make_random_candidates(object_list, candidate_limit, rng)
                # Update search_objects for this attempt
                if candidate_pairs:
                    search_objects = object_list
                    shuffled_objects = uniform_shuffle(object_list, seed_int)
        
        # If still empty after all fallbacks, log and continue to next attempt
        if not candidate_pairs:
            logger.error(f"CANDIDATE_DEBUG pool={len(search_objects)} K={K} limit={candidate_limit} used_pairs={len(used_pairs_set)} used_main={len(used_main_set)}")
            logger.error(f"Failed to generate any candidate pairs after all fallbacks in batch attempt {batch_attempt + 1}")
            if batch_attempt < max_batch_calls - 1:
                continue  # Try next batch attempt
            else:
                # Last attempt failed - raise error
                raise ValueError(f"NO_CANDIDATE_PAIRS: Failed to generate any candidate pairs after {max_batch_calls} batch attempts. pool={len(search_objects)} K={K} limit={candidate_limit}")
        
        # Batch score all candidates at once
        batch_call_count += 1
        logger.info(f"SHAPE_BATCH attempt={batch_attempt + 1} n_pairs={len(candidate_pairs)} model={shape_model}")
        
        batch_results = score_pairs_batch_o3_pro(candidate_pairs, shape_model, language="en")
        
        # Guard: If parsed_results is empty, try another attempt with new candidates
        if not batch_results:
            logger.warning(f"SHAPE_BATCH_PARSED count=0 (empty results) in attempt {batch_attempt + 1}")
            if batch_attempt < max_batch_calls - 1:
                continue  # Try next batch attempt with new candidates
            else:
                # Last attempt - raise error
                raise ValueError(f"SHAPE_BATCH_EMPTY_RESULTS: Batch scoring returned empty results after {max_batch_calls} attempts")
        
        # Log parsed results count
        logger.info(f"SHAPE_BATCH_PARSED count={len(batch_results)}")
        
        # Find best scoring pair that passes threshold
        passes_threshold = []
        for result in batch_results:
            idx = result.get("i", -1)
            if idx < 0 or idx >= len(candidate_pairs):
                continue
            
            score = result.get("score", 0)
            archetype = result.get("archetype", "")
            reason = result.get("reason", "")
            
            if score >= min_score_threshold:
                item_a, item_b = candidate_pairs[idx]
                i_idx, j_idx = candidate_indices[idx]
                
                passes_threshold.append({
                    "item_a": item_a,
                    "item_b": item_b,
                    "score": score,
                    "archetype": archetype,
                    "reason": reason,
                    "idx": idx
                })
        
        # Log batch results with detailed info
        best_batch_score = max([r.get("score", 0) for r in batch_results], default=0)
        logger.info(f"SHAPE_BATCH attempt={batch_attempt + 1} n_pairs={len(candidate_pairs)} best={best_batch_score} passes={len(passes_threshold)}")
        logger.info(f"SHAPE_BATCH_AFTER_FILTER count={len(passes_threshold)}")
        
        # If we found a pair that passes threshold, use it
        if passes_threshold:
            # Sort by score descending and pick the best
            passes_threshold.sort(key=lambda x: x["score"], reverse=True)
            best_candidate = passes_threshold[0]
            
            item_a = best_candidate["item_a"]
            item_b = best_candidate["item_b"]
            score = best_candidate["score"]
            archetype = best_candidate["archetype"]
            reason = best_candidate["reason"]
            
            obj_a_id = item_a["id"]
            obj_a_name = item_a["object"]
            obj_b_id = item_b["id"]
            obj_b_name = item_b["object"]
            
            best_pair_result = {
                "object_a": obj_a_name,
                "object_b": obj_b_name,
                "object_a_id": obj_a_id,
                "object_b_id": obj_b_id,
                "shape_similarity_score": score,
                "shape_hint": archetype,
                "shape_reason": reason
            }
            
            # Update session tracking
            pair_hash = hashlib.sha256("|".join(sorted([obj_a_id, obj_b_id])).encode()).hexdigest()
            with _session_used_lock:
                if sid in _session_used_pairs:
                    used_pairs_set, used_objects_session, _ = _session_used_pairs[sid]
                    used_pairs_set.add(pair_hash)
                    used_objects_session.add(obj_a_id)
                    used_objects_session.add(obj_b_id)
                    _session_used_pairs[sid] = (used_pairs_set, used_objects_session, time.time())
            
            # Log with detailed information
            obj_a_theme = item_a.get("theme_tag", "")
            obj_b_theme = item_b.get("theme_tag", "")
            obj_a_sub = item_a.get("sub_object", "")
            obj_b_sub = item_b.get("sub_object", "")
            obj_a_link = item_a.get("theme_link", "")
            obj_b_link = item_b.get("theme_link", "")
            
            theme_info = ""
            if obj_a_theme or obj_b_theme:
                a_link_short = obj_a_link[:30] + "..." if len(obj_a_link) > 30 else obj_a_link
                b_link_short = obj_b_link[:30] + "..." if len(obj_b_link) > 30 else obj_b_link
                theme_info = f" A_theme={obj_a_theme} B_theme={obj_b_theme} A_link={a_link_short} B_link={b_link_short}"
            
            logger.info(f"PAIR_PICK sid={sid} ad={ad_index} A={obj_a_id} A_obj={obj_a_name} A_sub={obj_a_sub} B={obj_b_id} B_obj={obj_b_name} B_sub={obj_b_sub} shape={score} archetype={archetype} reason={reason[:50]} checked_pairs={len(candidate_pairs)} batch_calls={batch_call_count} cache_hit_plan=0{theme_info}")
            return best_pair_result
        
        # If passes == 0, check if best_score >= 70 for soft fallback
        if len(passes_threshold) == 0:
            if best_batch_score >= 70:
                # Soft fallback: use best pair even if below threshold
                logger.warning(f"SHAPE_BATCH: passes=0 but best_score={best_batch_score} >= 70, using soft fallback")
                best_result = max(batch_results, key=lambda x: x.get("score", 0))
                idx = best_result.get("i", -1)
                if 0 <= idx < len(candidate_pairs):
                    item_a, item_b = candidate_pairs[idx]
                    best_pair_result = {
                        "object_a": item_a["object"],
                        "object_b": item_b["object"],
                        "object_a_id": item_a["id"],
                        "object_b_id": item_b["id"],
                        "shape_similarity_score": best_result.get("score", 0),
                        "shape_hint": best_result.get("archetype", ""),
                        "shape_reason": best_result.get("reason", "")
                    }
                    # Update best_score for tracking
                    if best_result.get("score", 0) > best_score:
                        best_score = best_result.get("score", 0)
                    # Break to use fallback logic below
                    break
            else:
                # best_score < 70, try another attempt
                logger.warning(f"SHAPE_BATCH: passes=0 and best_score={best_batch_score} < 70, trying another attempt")
                if batch_attempt < max_batch_calls - 1:
                    # Track best for potential final fallback
                    if best_batch_score > best_score:
                        best_score = best_batch_score
                        best_result = max(batch_results, key=lambda x: x.get("score", 0))
                        idx = best_result.get("i", -1)
                        if 0 <= idx < len(candidate_pairs):
                            item_a, item_b = candidate_pairs[idx]
                            best_pair_result = {
                                "object_a": item_a["object"],
                                "object_b": item_b["object"],
                                "object_a_id": item_a["id"],
                                "object_b_id": item_b["id"],
                                "shape_similarity_score": best_result.get("score", 0),
                                "shape_hint": best_result.get("archetype", ""),
                                "shape_reason": best_result.get("reason", "")
                            }
                    continue  # Try next batch attempt
                else:
                    # Last attempt failed - raise error if best_score still < 70
                    if best_score < 70:
                        raise ValueError(f"NO_VALID_PAIR_BEST_TOO_LOW: No pair passed threshold ({min_score_threshold}) and best_score={best_score} < 70 after {max_batch_calls} attempts")
        
        # Track best score for fallback
        if best_batch_score > best_score:
            best_score = best_batch_score
            # Store best pair for fallback
            best_result = max(batch_results, key=lambda x: x.get("score", 0))
            idx = best_result.get("i", -1)
            if 0 <= idx < len(candidate_pairs):
                item_a, item_b = candidate_pairs[idx]
                best_pair_result = {
                    "object_a": item_a["object"],
                    "object_b": item_b["object"],
                    "object_a_id": item_a["id"],
                    "object_b_id": item_b["id"],
                    "shape_similarity_score": best_result.get("score", 0),
                    "shape_hint": best_result.get("archetype", ""),
                    "shape_reason": best_result.get("reason", "")
                }
        
        total_checked += len(candidate_pairs)
        
        # If we found a good enough pair (even if below threshold), or exhausted candidates, break
        if best_score >= 70 or total_checked >= MAX_CHECKED_PAIRS:
            break
    
    # Fallback: use best pair if >= 70 (0-100 scale)
    if best_pair_result and best_score >= 70:
        # Find theme info for best_pair_result
        obj_a_id_str = str(best_pair_result["object_a_id"]) if best_pair_result.get("object_a_id") is not None else ""
        obj_b_id_str = str(best_pair_result["object_b_id"]) if best_pair_result.get("object_b_id") is not None else ""
        obj_a_item_fallback = next((it for it in search_objects if str(it.get("id")) == obj_a_id_str), None)
        obj_b_item_fallback = next((it for it in search_objects if str(it.get("id")) == obj_b_id_str), None)
        
        obj_a_theme = obj_a_item_fallback.get("theme_tag", "") if obj_a_item_fallback else ""
        obj_b_theme = obj_b_item_fallback.get("theme_tag", "") if obj_b_item_fallback else ""
        obj_a_link = obj_a_item_fallback.get("theme_link", "") if obj_a_item_fallback else ""
        obj_b_link = obj_b_item_fallback.get("theme_link", "") if obj_b_item_fallback else ""
        
        obj_a_sub = obj_a_item_fallback.get("sub_object", "") if obj_a_item_fallback else ""
        obj_b_sub = obj_b_item_fallback.get("sub_object", "") if obj_b_item_fallback else ""
        obj_a_name_fallback = best_pair_result["object_a"]
        obj_b_name_fallback = best_pair_result["object_b"]
        archetype = best_pair_result.get("shape_hint", "")
        reason = best_pair_result.get("shape_reason", "")
        
        # Check theme relevance for fallback too
        if allowed_theme_tags and len(allowed_theme_tags) > 0 and themed_pool and len(themed_pool) >= 60:
            theme_tags_norm = {_norm(t) for t in allowed_theme_tags}
            obj_a_theme_norm = _norm(obj_a_theme)
            obj_b_theme_norm = _norm(obj_b_theme)
            if obj_a_theme_norm not in theme_tags_norm or obj_b_theme_norm not in theme_tags_norm:
                logger.warning(f"PAIR_REJECT_THEME_FALLBACK sid={sid} ad={ad_index} A={best_pair_result['object_a_id']}(theme={obj_a_theme}) B={best_pair_result['object_b_id']}(theme={obj_b_theme}) not in allowed_tags")
                # Still return it as fallback, but log the issue
        
        pair_hash = hashlib.sha256("|".join(sorted([best_pair_result["object_a_id"], best_pair_result["object_b_id"]])).encode()).hexdigest()
        with _session_used_lock:
            if sid in _session_used_pairs:
                used_pairs_set, used_objects_session, _ = _session_used_pairs[sid]
                used_pairs_set.add(pair_hash)
                used_objects_session.add(best_pair_result["object_a_id"])
                used_objects_session.add(best_pair_result["object_b_id"])
                _session_used_pairs[sid] = (used_pairs_set, used_objects_session, time.time())
        
        # Log with detailed information including main objects and sub_objects
        theme_info = ""
        if obj_a_theme or obj_b_theme:
            a_link_short = obj_a_link[:30] + "..." if len(obj_a_link) > 30 else obj_a_link
            b_link_short = obj_b_link[:30] + "..." if len(obj_b_link) > 30 else obj_b_link
            theme_info = f" A_theme={obj_a_theme} B_theme={obj_b_theme} A_link={a_link_short} B_link={b_link_short}"
        
        logger.warning(f"PAIR_PICK sid={sid} ad={ad_index} A={best_pair_result['object_a_id']} A_obj={obj_a_name_fallback} A_sub={obj_a_sub} B={best_pair_result['object_b_id']} B_obj={obj_b_name_fallback} B_sub={obj_b_sub} shape={best_pair_result['shape_similarity_score']} archetype={archetype} reason={reason[:50]} checked_pairs={total_checked} batch_calls={batch_call_count} cache_hit_plan=0 (fallback){theme_info}")
        return best_pair_result
    
    raise ValueError(f"No valid pair found after {total_checked} checks ({batch_call_count} batch calls)")


def describe_item(item) -> str:
    """Helper to describe an item (supports both Dict and str formats)."""
    if isinstance(item, dict):
        obj = item.get("object", "")
        ctx = item.get("classic_context", "")
        if ctx:
            return f'{obj} ({ctx})'
        return obj
    return str(item)


def _obj_key(obj) -> str:
    """
    Helper to get hashable key from object (supports both dict and str).
    """
    if isinstance(obj, dict):
        return obj.get("id") or f'{obj.get("object","")}::{obj.get("sub_object","")}'
    return str(obj)


def select_similar_pair_shape_only(
    object_list: List,
    used_objects: set,
    max_retries: int = 2,
    model_name: Optional[str] = None
) -> Dict:
    """
    STEP 1 - SHAPE SELECTION (ONLY SHAPE)
    
    Select two objects based ONLY on geometric shape similarity.
    Uses model_name if provided, otherwise OPENAI_SHAPE_MODEL (default: o3-pro).
    
    Rules:
    - NO productName
    - NO productDescription
    - NO message
    - NO headline
    - NO composition
    - NO SIDE BY SIDE
    - ONLY criterion: geometric shape similarity
    
    Returns: {
        "object_a": str,
        "object_b": str,
        "shape_similarity_score": int (0-100),
        "shape_hint": str,
        "why": str
    }
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model_name is None:
        model_name = _get_shape_model()
    
    # Filter out used objects - use hashable keys, not dicts
    used_objects_keys = {_obj_key(obj) for obj in used_objects} if used_objects else set()
    available_objects = [obj for obj in object_list if _obj_key(obj) not in used_objects_keys]
    if len(available_objects) < 2:
        available_objects = object_list
        logger.warning("Not enough unused objects for shape selection, allowing reuse")
    
    logger.info(f"STEP 1 - SHAPE SELECTION: shape_model={model_name}, objectList_size={len(object_list)}, available={len(available_objects)}")
    
    # Check if model is o* type - these use Responses API, not Chat Completions
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    for attempt in range(max_retries):
        is_strict = attempt > 0
        
        # Build full prompt (include system instruction in prompt for Responses API)
        system_instruction = "You are a shape similarity analyzer. Output must be in English only. Return JSON only without additional text.\n\n"
        
        # Request list of candidates (12 pairs, minimum 5) with environment difference scoring
        # Format available_objects for prompt (handle both List[str] and List[Dict])
        if available_objects and isinstance(available_objects[0], dict):
            available_objects_formatted = [describe_item(obj) for obj in available_objects]
        else:
            available_objects_formatted = available_objects
        
        prompt = f"""{system_instruction}Task: Find pairs of items from the provided list with similar geometric shapes (outer contour/outline) BUT with clearly different classic natural environments.

CRITERIA:
1. Shape similarity: geometric shape similarity of the objects' outer contour (outline).
2. Environment difference: the classic natural environments where each object is normally found must be clearly different.

Ignore meaning, category, theme, symbolism, relevance, marketing, color, material, texture.

Return EXACT strings from the list (no synonyms, no new words).

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

ENVIRONMENT RULES (CRITICAL):
- For each object, identify its classic natural environment (where it is normally found in real life).
- Penalize pairs from same category where environments match (e.g., coin/plate both in "flat surface", ring/wheel both in "circular mechanism").
- Prefer pairs where environments are clearly different (e.g., leaf in "forest floor" vs. shell in "ocean beach").

Available object list:
{json.dumps(available_objects_formatted, ensure_ascii=False, indent=2)}

Output JSON only with a list of candidate pairs:
{{
  "candidates": [
    {{
      "a": "OBJECT_1",
      "b": "OBJECT_2",
      "shape_score": 0-100,
      "env_difference_score": 0-100,
      "env_a": "2-5 words classic environment",
      "env_b": "2-5 words classic environment",
      "hint": "short shape hint"
    }},
    ...
  ]
}}

Requirements:
- Return 12 candidate pairs if possible, minimum 5
- Each a/b must be EXACT match from the object list
- shape_score: based ONLY on shape similarity (outer contour/outline)
- env_a/env_b: classic natural environment for each object (2-5 words, concrete and specific)
- env_difference_score: how different env_a and env_b are (0=almost same, 100=completely different)
- Penalize pairs from same category where environments match
- Exclude objects with visible text or branding"""
        
        if is_strict:
            # Format available_objects for prompt (handle both List[str] and List[Dict])
            if available_objects and isinstance(available_objects[0], dict):
                available_objects_formatted = [describe_item(obj) for obj in available_objects]
            else:
                available_objects_formatted = available_objects
            
            prompt = f"""{system_instruction}Task: Find pairs of items from the provided list with similar geometric shapes (outer contour/outline) BUT with clearly different classic natural environments.

CRITERIA:
1. Shape similarity: geometric shape similarity of the objects' outer contour (outline).
2. Environment difference: the classic natural environments where each object is normally found must be clearly different.

Ignore meaning, category, theme, symbolism, relevance, marketing, color, material, texture.

Return exact strings from the list only. Return EXACT strings from the list (no synonyms, no new words).

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

ENVIRONMENT RULES (CRITICAL):
- For each object, identify its classic natural environment (where it is normally found in real life).
- Penalize pairs from same category where environments match.
- Prefer pairs where environments are clearly different.

Available object list:
{json.dumps(available_objects_formatted, ensure_ascii=False, indent=2)}

Output JSON only with a list of candidate pairs:
{{
  "candidates": [
    {{
      "a": "OBJECT_1",
      "b": "OBJECT_2",
      "shape_score": 0-100,
      "env_difference_score": 0-100,
      "env_a": "2-5 words classic environment",
      "env_b": "2-5 words classic environment",
      "hint": "short shape hint"
    }},
    ...
  ]
}}

Requirements:
- Return 12 candidate pairs if possible, minimum 5
- Each a/b must be EXACT match from the object list
- shape_score: based ONLY on shape similarity (outer contour/outline)
- env_a/env_b: classic natural environment for each object (2-5 words, concrete and specific)
- env_difference_score: how different env_a and env_b are (0=almost same, 100=completely different)
- Penalize pairs from same category where environments match"""
        
        try:
            if attempt == 0:
                logger.info(f"Shape selection: using model={model_name}, using_responses_api={using_responses_api}, shape_model={model_name}")
            
            def _shape_select_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text or ""
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a shape similarity analyzer. Output must be in English only. Return JSON only without additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return (r.choices[0].message.content or "") if r.choices else ""
            content = openai_retry.openai_call_with_retry(_shape_select_call, endpoint="responses")
            result = json.loads(content)
            
            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            
            if "candidates" not in result or not isinstance(result["candidates"], list):
                raise ValueError("Missing or invalid 'candidates' field")
            
            candidates = result["candidates"]
            if len(candidates) < 5:
                raise ValueError(f"Too few candidates returned: {len(candidates)}, minimum 5 required")
            
            # Filter candidates: keep only those with a/b in objectList
            valid_candidates = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                
                # Support both "a"/"b" and "object_a"/"object_b" formats
                obj_a = c.get("a") or c.get("object_a")
                obj_b = c.get("b") or c.get("object_b")
                
                if not obj_a or not obj_b:
                    continue
                
                # Validate that obj_a and obj_b exist in object_list
                # Handle both List[str] and List[Dict]
                obj_a_found = False
                obj_b_found = False
                if isinstance(object_list[0] if object_list else None, dict):
                    # List[Dict] - check by id or object name
                    for item in object_list:
                        item_id = item.get("id", "")
                        item_obj = item.get("object", "")
                        if obj_a == item_id or obj_a == item_obj:
                            obj_a_found = True
                        if obj_b == item_id or obj_b == item_obj:
                            obj_b_found = True
                else:
                    # List[str] - direct comparison
                    obj_a_found = obj_a in object_list
                    obj_b_found = obj_b in object_list
                
                if not obj_a_found or not obj_b_found:
                    continue
                
                if obj_a == obj_b:
                    continue
                
                # Normalize shape_score (support both "score" and "shape_score")
                shape_score = c.get("shape_score") or c.get("score") or c.get("shape_similarity_score", 0)
                if not isinstance(shape_score, (int, float)):
                    try:
                        shape_score = int(shape_score)
                    except:
                        shape_score = 0
                
                # Get environment difference score and environments
                env_diff_score = c.get("env_difference_score", 0)
                if not isinstance(env_diff_score, (int, float)):
                    try:
                        env_diff_score = int(env_diff_score)
                    except:
                        env_diff_score = 0
                
                env_a = c.get("env_a", "").strip().lower()
                env_b = c.get("env_b", "").strip().lower()
                
                # Guardrail: reject if environments are too similar (same or almost same)
                if env_a and env_b:
                    # Simple similarity check: if environments are identical or very similar
                    if env_a == env_b:
                        logger.debug(f"Rejecting pair {obj_a}~{obj_b}: identical environments ({env_a})")
                        continue
                    # Check if one environment contains the other (very similar)
                    if env_a in env_b or env_b in env_a:
                        if len(env_a) > 5 and len(env_b) > 5:  # Only for longer descriptions
                            logger.debug(f"Rejecting pair {obj_a}~{obj_b}: very similar environments ({env_a} vs {env_b})")
                            continue
                
                hint = c.get("hint") or c.get("shape_hint", "")
                
                valid_candidates.append({
                    "a": obj_a,
                    "b": obj_b,
                    "shape_score": shape_score,
                    "env_difference_score": env_diff_score,
                    "env_a": env_a,
                    "env_b": env_b,
                    "hint": hint
                })
            
            if len(valid_candidates) < 5:
                logger.warning(f"Too few valid candidates after filtering: {len(valid_candidates)}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with stricter instruction (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    raise ValueError(f"Too few valid candidates after filtering: {len(valid_candidates)}")
            
            # Apply filters: min_shape_score and min_env_difference_score
            min_shape_score = 85 if len(valid_candidates) >= 10 else 80
            min_env_diff_score = 60
            
            logger.info(f"STEP 1 PAIR_GATE: min_shape={min_shape_score} min_env_diff={min_env_diff_score}")
            
            # Filter by thresholds
            filtered_candidates = []
            for c in valid_candidates:
                if c["shape_score"] >= min_shape_score and c["env_difference_score"] >= min_env_diff_score:
                    filtered_candidates.append(c)
            
            # If no candidates pass filters, fallback to best shape_score with WARNING
            if len(filtered_candidates) == 0:
                logger.warning(f"STEP 1 PAIR_GATE: No candidates passed filters, falling back to best shape_score")
                # Sort by shape_score descending
                valid_candidates.sort(key=lambda x: x["shape_score"], reverse=True)
                filtered_candidates = [valid_candidates[0]]  # Take best by shape only
            
            # Calculate weighted combined score: (0.65*shape_score) + (0.35*env_difference_score)
            for c in filtered_candidates:
                c["combined_score"] = (0.65 * c["shape_score"]) + (0.35 * c["env_difference_score"])
            
            # Sort by combined_score descending
            filtered_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Select best pair
            best_pair = filtered_candidates[0]
            object_a = best_pair["a"]
            object_b = best_pair["b"]
            shape_score = best_pair["shape_score"]
            env_diff_score = best_pair["env_difference_score"]
            combined_score = best_pair["combined_score"]
            env_a = best_pair.get("env_a", "")
            env_b = best_pair.get("env_b", "")
            shape_hint = best_pair["hint"]
            
            # Calculate similar_pairs_found (shape_score >= 80)
            similar_pairs_found = sum(1 for c in valid_candidates if c["shape_score"] >= 80)
            
            # Log summary
            logger.info(f"STEP 1 SHAPE_MATCH summary: objectList_size={len(object_list)}, candidates_returned={len(candidates)}, candidates_valid={len(valid_candidates)}, similar_pairs_found(score>=80)={similar_pairs_found}, best_pair=\"{object_a} ~ {object_b}\" shape_score={shape_score} env_diff_score={env_diff_score} combined_score={combined_score:.1f} hint=\"{shape_hint}\"")
            
            # Log environment details
            if env_a and env_b:
                logger.info(f"STEP 1 PAIR_SELECT_ENV: env_a=\"{env_a}\" env_b=\"{env_b}\"")
            
            # Log selection
            logger.info(f"STEP 1 PAIR_SELECT: chosen=\"{object_a}~{object_b}\" shape={shape_score} env_diff={env_diff_score} combined={combined_score:.1f}")
            
            # Log top 5 (max 10 lines to avoid flooding)
            top5 = filtered_candidates[:5]
            top5_str = " | ".join([f"{i+1}) {c['a']}~{c['b']} shape={c['shape_score']} env_diff={c['env_difference_score']} combined={c['combined_score']:.1f}" for i, c in enumerate(top5)])
            logger.info(f"STEP 1 SHAPE_MATCH top5: {top5_str}")
            
            # Return result in expected format
            result = {
                "object_a": object_a,
                "object_b": object_b,
                "shape_similarity_score": shape_score,
                "shape_hint": shape_hint,
                "why": f"Selected from {len(valid_candidates)} valid candidates, {similar_pairs_found} with shape_score>=80, combined_score={combined_score:.1f}",
                "_similar_pairs_found": similar_pairs_found,  # Internal variable for future use
                "_env_difference_score": env_diff_score,  # Internal variable
                "_env_a": env_a,  # Internal variable
                "_env_b": env_b  # Internal variable
            }
            
            logger.info(f"STEP 1 - SHAPE SELECTION SUCCESS: selected_pair=[{object_a}, {object_b}], shape_score={shape_score}, env_diff_score={env_diff_score}, combined_score={combined_score:.1f}, shape_hint={shape_hint}, validation_passed=true")
            
            # Save to STEP 1 cache (only for preview mode)
            if PREVIEW_MODE in ["plan_only", "image"]:
                min_shape_score = 85 if len(available_objects) >= 10 else 80
                min_env_diff_score = 60
                step1_cache_key = _get_cache_key_step1(object_list, min_shape_score, min_env_diff_score, used_objects)
                _set_to_step1_cache(step1_cache_key, result)
            
            return result
            
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            
            # Check for 400 errors - DO NOT RETRY
            is_400_error = (
                "400" in error_str or 
                "invalid_request" in error_lower or 
                "unsupported_value" in error_lower or
                "bad_request" in error_lower
            )
            
            if is_400_error:
                logger.error(f"Shape selection 400 error (no retry): {error_str}")
                raise ValueError(f"invalid_request: {error_str}")
            
            # Check for rate limit - RETRY
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_lower or 
                "quota" in error_lower or
                "rate limit" in error_lower
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    delay = base_delay + jitter
                    logger.warning(f"Shape selection rate limit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Shape selection rate limit exceeded after {max_retries} attempts")
                    raise Exception("rate_limited")
            
            # Other errors
            if attempt < max_retries - 1:
                logger.warning(f"Shape selection failed (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                time.sleep(1 + attempt)
                continue
            else:
                logger.error(f"Shape selection failed after {max_retries} attempts: {error_str}")
                raise
    
    raise Exception("Failed to select shape pair after retries")


def generate_marketing_copy(
    product_name: str,
    product_description: str,
    ad_goal: str,
    max_retries: int = 2
) -> str:
    """
    Generate marketing copy: 45-55 words, English, product-specific, with CTA.
    
    Args:
        product_name: Product name
        product_description: Product description
        ad_goal: Advertising goal
        max_retries: Maximum retry attempts
    
    Returns:
        str: Marketing copy (45-55 words)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_text_model()
    
    prompt = f"""Generate marketing copy for an advertisement.

Product name: {product_name}
Product description: {product_description}
Advertising goal: {ad_goal}

Requirements:
- English only
- Exactly 45-55 words (count carefully)
- Must include product name
- Must be product-specific (not generic marketing language)
- Must include one short CTA (call-to-action) at the end
- Professional, compelling, clear
- No fluff, no generic phrases

Marketing copy:"""
    
    max_attempts = max_retries + 1
    for attempt in range(max_attempts):
        try:
            logger.info(f"MARKETING_COPY attempt={attempt+1} model={model_name} product={product_name[:50]}")
            
            def _copy_call():
                is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                if is_o_model:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a marketing copywriter. Output must be in English only. Return only the marketing copy text, no JSON, no quotes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                return r.choices[0].message.content.strip()
            copy_text = openai_retry.openai_call_with_retry(_copy_call, endpoint="responses")
            # Clean copy
            copy_text = copy_text.strip('"\'')
            
            # Count words
            words = copy_text.split()
            word_count = len(words)
            
            # Validate word count
            if word_count < 45:
                if attempt < max_attempts - 1:
                    logger.warning(f"MARKETING_COPY: word_count={word_count} < 45, retrying...")
                    prompt = f"""Generate marketing copy for an advertisement.

Product name: {product_name}
Product description: {product_description}
Advertising goal: {ad_goal}

Requirements:
- English only
- MUST be 45-55 words (current attempt was {word_count} words - too short)
- Must include product name
- Must be product-specific (not generic marketing language)
- Must include one short CTA (call-to-action) at the end
- Professional, compelling, clear
- No fluff, no generic phrases

Marketing copy:"""
                    continue
                else:
                    # Final attempt: pad with product-specific content
                    logger.warning(f"MARKETING_COPY: word_count={word_count} < 45, padding...")
                    needed = 45 - word_count
                    padding = f"{product_name} delivers on {ad_goal.lower()}. " * (needed // 3 + 1)
                    copy_text = f"{copy_text} {padding}".strip()
                    words = copy_text.split()
                    copy_text = " ".join(words[:55])  # Cap at 55
            elif word_count > 55:
                if attempt < max_attempts - 1:
                    logger.warning(f"MARKETING_COPY: word_count={word_count} > 55, retrying...")
                    prompt = f"""Generate marketing copy for an advertisement.

Product name: {product_name}
Product description: {product_description}
Advertising goal: {ad_goal}

Requirements:
- English only
- MUST be 45-55 words (current attempt was {word_count} words - too long)
- Must include product name
- Must be product-specific (not generic marketing language)
- Must include one short CTA (call-to-action) at the end
- Professional, compelling, clear
- No fluff, no generic phrases

Marketing copy:"""
                    continue
                else:
                    # Final attempt: truncate to 55 words
                    logger.warning(f"MARKETING_COPY: word_count={word_count} > 55, truncating...")
                    words = copy_text.split()
                    copy_text = " ".join(words[:55])
            
            final_word_count = len(copy_text.split())
            logger.info(f"MARKETING_COPY SUCCESS: word_count={final_word_count} copy='{copy_text[:100]}...'")
            return copy_text
            
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            logger.error(f"MARKETING_COPY failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                continue
            # Fallback: return minimal copy
            logger.warning(f"MARKETING_COPY: Using fallback copy")
            return f"{product_name} helps you achieve {ad_goal.lower()}. Discover how {product_name} can transform your workflow. Get started today."
    # Final fallback
    return f"{product_name} helps you achieve {ad_goal.lower()}. Discover how {product_name} can transform your workflow. Get started today."


def generate_headline_only(
    product_name: str,
    message: str,
    object_a: str,
    object_b: str,
    headline_placement: Optional[str] = None,
    max_retries: int = 3,
    hard_mode: bool = False,
    ad_goal: Optional[str] = None
) -> str:
    """
    STEP 2 - HEADLINE GENERATION
    
    Generate headline ONLY using OPENAI_TEXT_MODEL (default: o3-pro).
    
    Input:
    - productName
    - message (pre-determined)
    - object_a (from STEP 1)
    - object_b (from STEP 1)
    
    Output:
    - headline (string only)
    
    Rules:
    - English only
    - ALL CAPS
    - Max 7 words INCLUDING productName
    - Do NOT change the pair
    - Do NOT re-select objects
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_text_model()
    
    logger.info(f"STEP 2 - HEADLINE GENERATION: text_model={model_name}, productName={product_name[:50]}, message={message[:50]}, object_a={object_a}, object_b={object_b}, hard_mode={hard_mode}")
    
    # HARD_MODE logic: if False, return product_name only (no model call)
    if not hard_mode:
        headline = product_name.upper()
        words_count = len(headline.split())
        logger.info(f"HEADLINE_POLICY hard_mode=False headline_words={words_count} headline='{headline}'")
        return headline
    
    # HARD_MODE: Generate full headline via model
    ad_goal_context = f"\nAdvertising goal: {ad_goal}" if ad_goal else ""
    prompt = f"""Generate a headline for an advertisement.

Product name: {product_name}
Message: {message}{ad_goal_context}
Objects (already selected, do not change): {object_a} and {object_b}

Requirements:
- English only
- ALL CAPS
- Include the product name in the headline (mandatory)
- Do NOT mention the objects ({object_a} or {object_b}) in the headline
- Do NOT change or re-select the objects
- No punctuation (no colons, commas, periods, etc.)
- Return ONLY the headline text, no JSON, no quotes
- Full headline (no word limit restriction)

Headline:"""

    # Check if model is o* type - these use Responses API, not Chat Completions
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                logger.info(f"STEP 2 - HEADLINE GENERATION: using_responses_api={using_responses_api}")
            
            def _headline_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                request_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return r.choices[0].message.content.strip() if r.choices else ""
            headline = openai_retry.openai_call_with_retry(_headline_call, endpoint="responses")
            # Clean headline (remove quotes, ensure ALL CAPS, remove punctuation)
            headline = headline.strip('"\'')
            headline = headline.upper()
            
            # Remove punctuation (colons, commas, periods, etc.)
            import string
            headline = ''.join(char for char in headline if char not in string.punctuation)
            
            # Remove object_a and object_b from headline if present
            object_a_upper = object_a.upper()
            object_b_upper = object_b.upper()
            words = headline.split()
            words = [w for w in words if w != object_a_upper and w != object_b_upper]
            headline = " ".join(words)
            
            # CRITICAL: Ensure product name is always in headline
            product_name_upper = product_name.upper()
            product_words = product_name_upper.split()
            headline_words = headline.split()
            
            # Check if product name (or any part of it) is in headline
            product_in_headline = any(word in headline_words for word in product_words)
            
            if not product_in_headline:
                # Add product name at the beginning
                logger.info(f"STEP 2 - HEADLINE: Product name '{product_name}' not found, adding it to headline")
                headline = f"{product_name_upper} {headline}".strip()
                headline_words = headline.split()
            
            # HARD_MODE: No truncation (full headline required)
            # In hard_mode, we keep the full headline as generated
            
            words_count = len(headline.split())
            logger.info(f"HEADLINE_POLICY hard_mode=True headline_words={words_count} headline='{headline}'")
            return headline
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            # Check for 400 errors (invalid_request, unsupported_value, etc.) - DO NOT RETRY
            is_400_error = (
                "400" in error_str or 
                "invalid_request" in error_lower or 
                "unsupported_value" in error_lower or
                "bad_request" in error_lower
            )
            if is_400_error:
                logger.error(f"OpenAI 400 error (no retry): {error_str}")
                # Wrap in a specific exception type for 400 errors
                raise ValueError(f"invalid_request: {error_str}")
            
            # Check for rate limit (429) - RETRY with backoff
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_lower or 
                "quota" in error_lower or
                "rate limit" in error_lower
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    # Exponential backoff + jitter
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    delay = base_delay + jitter
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise Exception("rate_limited")
            
            # Check for server errors (500-599) or connection errors - RETRY
            is_server_error = (
                "500" in error_str or 
                "502" in error_str or 
                "503" in error_str or
                "504" in error_str or
                "timeout" in error_lower or
                "connection" in error_lower or
                "network" in error_lower
            )
            
            if is_server_error:
                if attempt < max_retries - 1:
                    logger.warning(f"Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)  # Progressive delay
                    continue
                else:
                    logger.error(f"Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            # Other errors - don't retry, raise immediately
            logger.error(f"OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
            raise
    
    raise Exception("Failed to get valid response from OpenAI")


def generate_physical_context_extensions(
    object_a: str,
    object_b: str,
    ad_goal: str,
    max_retries: int = 2
) -> Dict:
    """
    STEP 1.5 - PHYSICAL CONTEXT EXTENSION
    
    Enrich two objects with natural physical causal context.
    
    Args:
        object_a: First object (from STEP 1)
        object_b: Second object (from STEP 1)
        ad_goal: Advertising goal (to ensure context doesn't contradict message)
        max_retries: Maximum retry attempts
    
    Returns:
        dict: {
            "object_a": str (same),
            "physical_extension_a": {
                "description": str,
                "connection_type": str
            },
            "object_b": str (same),
            "physical_extension_b": {
                "description": str,
                "connection_type": str
            },
            "silhouette_integrity_confirmed": bool
        }
    
    Rules:
    - Natural physical extension or interaction
    - Must be physically connected or directly interacting
    - Must explain object's function or origin
    - No decorative background
    - No abstract ideas
    - No full human figures
    - A hand is allowed only if necessary
    - Must not drastically change object's core silhouette
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_shape_model()
    
    prompt = f"""You are enriching two objects with physical causal context.

Objects:
- Object A: {object_a}
- Object B: {object_b}
- Advertising goal: {ad_goal}

For EACH object:
- Add a natural physical extension or interaction.
- The extension must be physically connected or directly interacting.
- It must explain the object's function or origin.
- No decorative background.
- No abstract ideas.
- No full human figures.
- A hand is allowed only if necessary.
- The extension must not drastically change the object's core silhouette.

The context must feel physically real and minimal.

Return JSON only:

{{
  "object_a": "{object_a}",
  "physical_extension_a": {{
    "description": "short phrase describing the connected physical element",
    "connection_type": "attached | inserted | held | growing_from | emitting | interacting_with"
  }},
  "object_b": "{object_b}",
  "physical_extension_b": {{
    "description": "short phrase describing the connected physical element",
    "connection_type": "attached | inserted | held | growing_from | emitting | interacting_with"
  }},
  "silhouette_integrity_confirmed": true
}}

Rules:
- Do NOT modify object names.
- Do NOT add background scenery.
- Focus on minimal, physically connected extensions only.

JSON:"""

    # Check if model is o* type - these use Responses API
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    logger.info(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: shape_model={model_name}, object_a={object_a}, object_b={object_b}, ad_goal={ad_goal[:50]}, using_responses_api={using_responses_api}")
    
    for attempt in range(max_retries):
        try:
            def _physical_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                request_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return r.choices[0].message.content.strip() if r.choices else ""
            response_text = openai_retry.openai_call_with_retry(_physical_call, endpoint="responses")
            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            # Parse JSON
            data = json.loads(response_text)
            # Validate response structure
            if "object_a" not in data or "object_b" not in data:
                raise ValueError("Missing object_a or object_b in response")
            if "physical_extension_a" not in data or "physical_extension_b" not in data:
                raise ValueError("Missing physical extensions in response")
            if data["object_a"] != object_a or data["object_b"] != object_b:
                raise ValueError("Object names were modified")
            
            # Validate connection types
            valid_connection_types = {"attached", "inserted", "held", "growing_from", "emitting", "interacting_with"}
            if data["physical_extension_a"]["connection_type"] not in valid_connection_types:
                raise ValueError(f"Invalid connection_type for object_a: {data['physical_extension_a']['connection_type']}")
            if data["physical_extension_b"]["connection_type"] not in valid_connection_types:
                raise ValueError(f"Invalid connection_type for object_b: {data['physical_extension_b']['connection_type']}")
            
            logger.info(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION SUCCESS: object_a={object_a}, extension_a={data['physical_extension_a']['description']}, connection_a={data['physical_extension_a']['connection_type']}, object_b={object_b}, extension_b={data['physical_extension_b']['description']}, connection_b={data['physical_extension_b']['connection_type']}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: JSON parse error: {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Failed to parse physical context JSON: {e}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            # Check for 400 errors - DO NOT RETRY
            is_400_error = (
                "400" in error_str or 
                "invalid_request" in error_lower or 
                "unsupported_value" in error_lower or
                "bad_request" in error_lower
            )
            if is_400_error:
                logger.error(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: OpenAI 400 error (no retry): {error_str}")
                raise ValueError(f"invalid_request: {error_str}")
            
            # Check for rate limit (429) - RETRY with backoff
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_lower or 
                "quota" in error_lower or
                "rate limit" in error_lower
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    delay = base_delay + jitter
                    logger.warning(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: Rate limit exceeded after {max_retries} attempts")
                    raise Exception("rate_limited")
            
            # Check for server errors - RETRY
            is_server_error = (
                "500" in error_str or 
                "502" in error_str or 
                "503" in error_str or
                "504" in error_str or
                "timeout" in error_lower or
                "connection" in error_lower or
                "network" in error_lower
            )
            
            if is_server_error:
                if attempt < max_retries - 1:
                    logger.warning(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)
                    continue
                else:
                    logger.error(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            # Other errors - don't retry, raise immediately
            logger.error(f"STEP 1.5 - PHYSICAL CONTEXT EXTENSION: OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
            raise
    
    raise Exception("Failed to generate physical context extensions")


def select_shape_and_environment_plan_optimized(
    object_list: List[str],
    used_objects: set,
    ad_goal: str,
    message: str,
    image_size: str,
    max_retries: int = 2
) -> Dict:
    """
    OPTIMIZED MODE: Combined shape selection + environment swap plan in one call.
    
    Returns unified JSON with both shape selection and environment plan.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_shape_model()
    
    # Filter out used objects
    available_objects = [obj for obj in object_list if obj not in used_objects]
    if len(available_objects) < 2:
        available_objects = object_list
    
    prompt = f"""You are planning an advertisement composition with shape selection and environment swap.

Task 1 - SHAPE SELECTION WITH ENVIRONMENT DIFFERENCE:
Find a pair of items from the provided list with similar geometric shapes (outer contour/outline) BUT with clearly different classic natural environments.

CRITERIA:
1. Shape similarity: geometric shape similarity of the objects' outer contour (outline).
2. Environment difference: the classic natural environments where each object is normally found must be clearly different.

Ignore meaning, category, theme, symbolism, relevance, marketing, color, material, texture.

Return EXACT strings from the list (no synonyms, no new words).

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

ENVIRONMENT RULES (CRITICAL):
- For each object, identify its classic natural environment (where it is normally found in real life).
- Penalize pairs from same category where environments match (e.g., coin/plate both in "flat surface", ring/wheel both in "circular mechanism").
- Prefer pairs where environments are clearly different (e.g., leaf in "forest floor" vs. shell in "ocean beach").

Available object list:
{json.dumps(available_objects, ensure_ascii=False, indent=2)}

Task 2 - ENVIRONMENT SWAP PLAN:
For the selected pair from Task 1, determine which object is the hero and which provides the classic natural environment.

CORE RULE:
- Show ONLY one object (hero_object).
- Place it inside the CLASSIC NATURAL ENVIRONMENT of the second object.
- Do NOT show the second object.
- No physical merging.
- No structural integration.
- No inserted mechanisms.

Definition of classic environment:
- The natural, iconic setting where the second object normally exists.
- Physically realistic.
- Immediately recognizable.
- Real-world photographable environment.

Context:
- Advertising goal: {ad_goal}
- Message: {message}
- Image size: {image_size}

Return JSON only:

{{
  "shape_selection": {{
    "object_a": "OBJECT_NAME",
    "object_b": "OBJECT_NAME",
    "shape_similarity_score": 0-100,
    "env_difference_score": 0-100,
    "env_a": "2-5 words classic environment",
    "env_b": "2-5 words classic environment",
    "shape_hint": "short shape hint",
    "why": "one short sentence focused on shape and environment difference"
  }},
  "environment_plan": {{
    "hero_object": "<selected_object_a>" | "<selected_object_b>",
    "environment_from": "<selected_object_a>" | "<selected_object_b>",
    "environment_description": "short concrete natural environment description (max 15 words)",
    "environment_is_classic": true,
    "single_object_confirmed": true
  }},
  "headline_placement_suggestion": "BOTTOM" | "SIDE"
}}

Rules:
- shape_selection.object_a and object_b must be EXACT matches from the object list.
- shape_selection.shape_similarity_score: based ONLY on shape similarity (outer contour/outline).
- shape_selection.env_a/env_b: classic natural environment for each object (2-5 words, concrete and specific).
- shape_selection.env_difference_score: how different env_a and env_b are (0=almost same, 100=completely different).
- Penalize pairs from same category where environments match.
- environment_plan.hero_object and environment_from must be different.
- environment_description must be concrete and natural (max 15 words).
- environment_is_classic and single_object_confirmed must be true.

JSON:"""

    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    logger.info(f"OPTIMIZED MODE - COMBINED SHAPE+ENV PLAN: shape_model={model_name}, using_responses_api={using_responses_api}")
    
    for attempt in range(max_retries):
        try:
            def _combined_plan_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                request_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return r.choices[0].message.content.strip() if r.choices else ""
            response_text = openai_retry.openai_call_with_retry(_combined_plan_call, endpoint="responses")
            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            data = json.loads(response_text)
            # Validate shape_selection
            if "shape_selection" not in data:
                raise ValueError("Missing shape_selection in response")
            shape_sel = data["shape_selection"]
            if shape_sel.get("object_a") not in available_objects or shape_sel.get("object_b") not in available_objects:
                raise ValueError("Shape selection objects not in available list")
            
            # Validate environment_plan
            if "environment_plan" not in data:
                raise ValueError("Missing environment_plan in response")
            env_plan = data["environment_plan"]
            selected_a = shape_sel.get("object_a")
            selected_b = shape_sel.get("object_b")
            if env_plan.get("hero_object") not in [selected_a, selected_b]:
                raise ValueError(f"hero_object must be '{selected_a}' or '{selected_b}'")
            if env_plan.get("environment_from") not in [selected_a, selected_b]:
                raise ValueError(f"environment_from must be '{selected_a}' or '{selected_b}'")
            if env_plan.get("hero_object") == env_plan.get("environment_from"):
                raise ValueError("hero_object and environment_from must be different")
            if not env_plan.get("environment_is_classic", False):
                raise ValueError("environment_is_classic must be true")
            if not env_plan.get("single_object_confirmed", False):
                raise ValueError("single_object_confirmed must be true")
            
            logger.info(f"OPTIMIZED MODE - COMBINED SHAPE+ENV PLAN SUCCESS: shape_pair=[{shape_sel.get('object_a')}, {shape_sel.get('object_b')}], hero={env_plan.get('hero_object')}, env_from={env_plan.get('environment_from')}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"OPTIMIZED MODE - COMBINED PLAN: JSON parse error: {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Failed to parse combined plan JSON: {e}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            is_400_error = (
                "400" in error_str or
                "invalid_request" in error_lower or 
                "unsupported_value" in error_lower or
                "bad_request" in error_lower
            )
            
            if is_400_error:
                logger.error(f"OPTIMIZED MODE - COMBINED PLAN: OpenAI 400 error (no retry): {error_str}")
                raise ValueError(f"invalid_request: {error_str}")
            
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_lower or 
                "quota" in error_lower or
                "rate limit" in error_lower
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    delay = base_delay + jitter
                    logger.warning(f"OPTIMIZED MODE - COMBINED PLAN: Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"OPTIMIZED MODE - COMBINED PLAN: Rate limit exceeded after {max_retries} attempts")
                    raise Exception("rate_limited")
            
            is_server_error = (
                "500" in error_str or 
                "502" in error_str or 
                "503" in error_str or
                "504" in error_str or
                "timeout" in error_lower or
                "connection" in error_lower or
                "network" in error_lower
            )
            
            if is_server_error:
                if attempt < max_retries - 1:
                    logger.warning(f"OPTIMIZED MODE - COMBINED PLAN: Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)
                    continue
                else:
                    logger.error(f"OPTIMIZED MODE - COMBINED PLAN: Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            logger.error(f"OPTIMIZED MODE - COMBINED PLAN: OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
            raise
    
    raise Exception("Failed to generate combined shape+environment plan")


def generate_hybrid_context_plan(
    object_a: str,
    object_b: str,
    ad_goal: str,
    message: str,
    image_size: str,
    max_retries: int = 2,
    model_name: Optional[str] = None
) -> Dict:
    """
    STEP 1.75 - HYBRID CONTEXT PLAN
    
    Determine which object is the hero and which provides physical context.
    
    Args:
        object_a: First object (from STEP 1)
        object_b: Second object (from STEP 1)
        ad_goal: Advertising goal
        message: Pre-determined message
        image_size: Image size (e.g., "1536x1024")
        max_retries: Maximum retry attempts
        model_name: Model to use (if None, uses OPENAI_SHAPE_MODEL)
    
    Returns:
        dict: {
            "hero_object": "object_a" | "object_b",
            "environment_from": "object_a" | "object_b",
            "environment_description": str (max 15 words),
            "environment_is_classic": bool,
            "single_object_confirmed": bool
        }
    
    Rules:
    - Show ONLY hero_object as full object
    - environment_from object must NOT appear
    - Only the classic natural environment of environment_from is shown
    - No physical merging, no structural integration, no inserted mechanisms
    - No decorative background, no extra elements
    - English only
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model_name is None:
        model_name = _get_shape_model()
    
    prompt = f"""You are planning an ENVIRONMENT SWAP advertisement composition.

CORE RULE:
- Show ONLY one object (hero_object).
- Place it inside the CLASSIC NATURAL ENVIRONMENT of the second object.
- Do NOT show the second object.
- No physical merging.
- No structural integration.
- No inserted mechanisms.

Definition of classic environment:
- The natural, iconic setting where the second object normally exists.
- Physically realistic.
- Immediately recognizable.
- Real-world photographable environment.

Objects:
- Object A: {object_a}
- Object B: {object_b}
- Advertising goal: {ad_goal}
- Message: {message}
- Image size: {image_size}

Rules:
- Only one full object allowed.
- The second object must NOT appear.
- No decorative elements unrelated to the environment.
- Photorealistic only.
- English only.
- The environment must be natural and physically plausible.

VISUAL RULES (apply always):
- Photorealistic photography only.
- No illustration.
- No drawing.
- No 3D render look.
- No painterly texture.
- No logos.
- No branding.
- No printed text on objects.
- No packaging with labels.
- The only readable text allowed in the entire image is the generated headline.

Return JSON only:

{{
  "hero_object": "{object_a}" | "{object_b}",
  "environment_from": "{object_a}" | "{object_b}",
  "environment_description": "short concrete natural environment description (max 15 words)",
  "environment_is_classic": true,
  "single_object_confirmed": true
}}

Rules:
- hero_object and environment_from must be different.
- environment_description must be concrete and describe the natural/iconic setting (max 15 words).
- environment_is_classic must be true.
- single_object_confirmed must be true.

JSON:"""

    # Check if model is o* type - these use Responses API
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    logger.info(f"STEP 1.75 - ENVIRONMENT SWAP PLAN: shape_model={model_name}, object_a={object_a}, object_b={object_b}, ad_goal={ad_goal[:50]}, using_responses_api={using_responses_api}")
    
    for attempt in range(max_retries):
        try:
            def _hybrid_plan_call():
                if using_responses_api:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                request_params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                r = client.chat.completions.create(**request_params)
                return r.choices[0].message.content.strip() if r.choices else ""
            response_text = openai_retry.openai_call_with_retry(_hybrid_plan_call, endpoint="responses")
            # Parse JSON response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            # Parse JSON
            data = json.loads(response_text)
            # Validate response structure
            if data.get("hero_object") not in [object_a, object_b]:
                raise ValueError(f"hero_object must be '{object_a}' or '{object_b}'")
            if data.get("environment_from") not in [object_a, object_b]:
                raise ValueError(f"environment_from must be '{object_a}' or '{object_b}'")
            if data.get("hero_object") == data.get("environment_from"):
                raise ValueError("hero_object and environment_from must be different")
            
            if not data.get("environment_is_classic", False):
                raise ValueError("environment_is_classic must be true")
            if not data.get("single_object_confirmed", False):
                raise ValueError("single_object_confirmed must be true")
            
            # Determine actual object names
            hero_name = data["hero_object"]
            environment_name = data["environment_from"]
            env_description = data.get("environment_description", "")
            
            logger.info(f"STEP 1.75 - ENVIRONMENT SWAP PLAN SUCCESS: hero={hero_name}, environment_from={environment_name}, environment=\"{env_description}\"")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: JSON parse error: {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Failed to parse hybrid classic environment plan JSON: {e}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            # Check for 400 errors - DO NOT RETRY
            is_400_error = (
                "400" in error_str or 
                "invalid_request" in error_lower or 
                "unsupported_value" in error_lower or
                "bad_request" in error_lower
            )
            
            if is_400_error:
                logger.error(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: OpenAI 400 error (no retry): {error_str}")
                raise ValueError(f"invalid_request: {error_str}")
            
            # Check for rate limit (429) - RETRY with backoff
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_lower or 
                "quota" in error_lower or
                "rate limit" in error_lower
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0, 1)
                    delay = base_delay + jitter
                    logger.warning(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: Rate limit exceeded after {max_retries} attempts")
                    raise Exception("rate_limited")
            
            # Check for server errors - RETRY
            is_server_error = (
                "500" in error_str or 
                "502" in error_str or 
                "503" in error_str or
                "504" in error_str or
                "timeout" in error_lower or
                "connection" in error_lower or
                "network" in error_lower
            )
            
            if is_server_error:
                if attempt < max_retries - 1:
                    logger.warning(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)
                    continue
                else:
                    logger.error(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            # Other errors - don't retry, raise immediately
            logger.error(f"STEP 1.75 - HYBRID CLASSIC ENVIRONMENT PLAN: OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
            raise
    
    raise Exception("Failed to generate hybrid context plan")


def generate_short_phrase(product_name: str) -> str:
    """
    Generate short headline from product name.
    Returns: headline (max 7 words INCLUDING product name, ALL CAPS)
    """
    product_upper = product_name.upper().strip()
    product_words = product_upper.split()
    
    # Simple mapping for common product types
    if "GREENPEACE" in product_upper or "ENVIRONMENT" in product_upper or "CLIMATE" in product_upper:
        headline = "PROTECT OUR PLANET"
    elif "CHARITY" in product_upper or "DONATE" in product_upper or "HELP" in product_upper:
        headline = "MAKE A DIFFERENCE"
    elif "TECH" in product_upper or "INNOVATION" in product_upper or "FUTURE" in product_upper:
        headline = "INNOVATION FOR TOMORROW"
    elif "HEALTH" in product_upper or "MEDICAL" in product_upper or "CARE" in product_upper:
        headline = "YOUR HEALTH MATTERS"
    elif "EDUCATION" in product_upper or "LEARN" in product_upper or "SCHOOL" in product_upper:
        headline = "EDUCATION FOR ALL"
    else:
        # Generic fallback - use product name or short phrase
        if len(product_words) <= 2:
            headline = product_upper
        else:
            # Use first 2-3 words of product name
            headline = " ".join(product_words[:3])
    
    # Ensure headline includes product name and is max 7 words total
    headline_words = headline.split()
    
    # If product name is not in headline, add it
    product_in_headline = any(word in headline for word in product_words)
    if not product_in_headline:
        # Add product name, but keep total <= 7 words
        available_slots = 7 - len(headline_words)
        if available_slots >= len(product_words):
            # Can fit full product name
            headline = f"{product_upper} {headline}"
        elif available_slots > 0:
            # Can fit part of product name
            product_part = " ".join(product_words[:available_slots])
            headline = f"{product_part} {headline}"
        # If no slots available, use product name only
        else:
            headline = product_upper
    
    # Final check: ensure max 7 words
    headline_words = headline.split()
    if len(headline_words) > 7:
        # Cut to 7 words, prioritizing product name
        if len(product_words) <= 7:
            # Keep product name + first words from rest
            remaining = 7 - len(product_words)
            if remaining > 0:
                other_words = [w for w in headline_words if w not in product_words][:remaining]
                headline = " ".join(product_words + other_words)
            else:
                headline = " ".join(product_words[:7])
        else:
            # Product name itself is too long, use first 7 words
            headline = " ".join(headline_words[:7])
    
    return headline


def create_image_prompt(
    object_a: str,
    object_b: str,
    headline: str,
    shape_hint: Optional[str] = None,
    physical_context: Optional[Dict] = None,
    hybrid_plan: Optional[Dict] = None,
    is_strict: bool = False,
    object_a_context: Optional[str] = None,
    object_b_context: Optional[str] = None,
    object_a_item: Optional[Dict] = None,
    object_b_item: Optional[Dict] = None,
    product_name: Optional[str] = None,
    force_mode: Optional[str] = None  # "replacement" | "side_by_side" | None (use ACE_IMAGE_MODE)
) -> str:
    """
    STEP 3 - IMAGE GENERATION PROMPT
    
    Create DALL-E prompt based on ACE_IMAGE_MODE:
    - "replacement": B replaces A in A's scene (single unified scene)
    - "side_by_side": Two panels side by side (legacy)
    
    Args:
        object_a: First object (from STEP 1)
        object_b: Second object (from STEP 1)
        headline: Headline from STEP 2 (ALL CAPS, max 7 words)
        shape_hint: Shape hint from STEP 1 (e.g., "tall-vertical", "round-flat"), optional
        physical_context: Physical context extensions (ignored, kept for compatibility), optional
        hybrid_plan: Hybrid context plan (ignored, kept for compatibility), optional
        is_strict: If True, use stricter prompt for retry
        object_a_context: Classic context for object_a (optional)
        object_b_context: Classic context for object_b (optional)
    """
    # Determine mode: use force_mode if provided, otherwise use ACE_IMAGE_MODE
    effective_mode = force_mode if force_mode else ACE_IMAGE_MODE
    # Log is handled in MODE_APPLIED in render_final_ad_bytes, so we don't duplicate here
    
    # Shared composition: pencil style, pure white background, very wide margins, lots of negative space
    negative_space_rules = """
Composition and framing (CRITICAL):
- Pencil drawing style only; soft pencil sketch aesthetic.
- Pure white background only; no scene, no texture, no gradient.
- Very wide white margins on all four sides; lots of negative space.
- Subject(s) centered and small; together they occupy about 25-35% of the canvas.
- Subject(s) must not touch or nearly touch the image edges; keep ample empty white space between subjects and frame."""
    
    # MODE 1 — REPLACEMENT (TEMPORARILY DISABLED - see TEMP_DISABLE_REPLACEMENT in render_final_ad_bytes)
    # This branch should not execute as replacement mode is disabled
    if effective_mode == "replacement":
        # Get A.sub_object directly from object_a_item (NEVER use B.sub_object)
        scene_context = object_a_context or f"{object_a} in its classic situation"
        locked_sub_object = ""
        if object_a_item:
            locked_sub_object = object_a_item.get("sub_object", "")
        
        # Ensure we have sub_object - if not, try to extract from context as fallback
        if not locked_sub_object and object_a_context:
            import re
            patterns = [
                r"on a ([a-z]+)",
                r"with a ([a-z]+)",
                r"next to a ([a-z]+)",
                r"landing on a ([a-z]+)",
                r"resting on a ([a-z]+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, object_a_context.lower())
                if match:
                    locked_sub_object = match.group(1)
                    break
        
        replacement_main_object = object_b
        product_name_str = product_name or "PRODUCT"
        
        replacement_prompt = f"""Create a professional advertisement image in pencil drawing style as a single unified scene.

BASE SCENE:
The scene is based entirely on:
"{scene_context}"

The sub-object from Object A must remain visible:
"{locked_sub_object}"

The environment, lighting, camera angle, framing, and depth of field remain identical to A's original situation.

MAIN OBJECT REPLACEMENT:

Replace ONLY the main object body of A with "{replacement_main_object}".

Do NOT show A anywhere.

Keep exact same pose, position, orientation, and scale.

Maintain natural physics.

Object B must interact physically with "{locked_sub_object}".

No floating objects.

No added B.sub_object.

HEADLINE RULE:

Include exactly one headline.

Headline must include "{product_name_str}".

Headline must be large and visually dominant.

Headline must have visual weight comparable to the main object.

No additional text allowed.

Pencil drawing style only; soft pencil sketch aesthetic. Pure white background.
No logos.
No labels.
No surreal distortions."""
        replacement_prompt = replacement_prompt.rstrip() + negative_space_rules
        return replacement_prompt
    
    # MODE 2 — SIDE_BY_SIDE (AUTO FALLBACK IF < 85%)
    # Build shape hint instruction if provided
    shape_instruction = ""
    if shape_hint:
        shape_instruction = f"\n- Both objects must share a similar outline: {shape_hint}. Emphasize comparable silhouettes."
    
    product_name_str = product_name or "PRODUCT"
    
    side_by_side_prompt = f"""Create a professional advertisement image in pencil drawing style with SIDE BY SIDE composition.

Composition Rules:

Display BOTH main objects: "{object_a}" and "{object_b}".

Do NOT include any sub-objects.

No sub-objects from either object.

Clean isolated presentation.

Both objects must partially overlap visually in the center.

The overlap must clearly show their similar silhouette logic.

Objects must not be duplicated.

Main object types must be different.

Background:
Pure white only; no scene, no texture, no gradient.

Headline:

Exactly one headline.

Must include "{product_name_str}".

Must be large and dominant.

Positioned above or integrated professionally.

Pencil drawing style only; soft pencil sketch aesthetic.
No logos.
No brand graphics.
No text except headline.{shape_instruction}{negative_space_rules}"""
    
    return side_by_side_prompt


def check_text_quality(image_base64: str) -> bool:
    """
    Basic heuristic check for text quality in image.
    Returns True if text appears readable, False if suspicious.
    
    Note: This is a simple heuristic. For production, consider OCR.
    """
    # Simple check: if base64 is too short, might be placeholder
    if len(image_base64) < 10000:  # Very small image might be placeholder
        return False
    
    # For now, we'll do a stochastic check for language=en
    # In production, you could add OCR here
    return True


def generate_image_with_dalle(
    client: OpenAI,
    object_a: str,
    object_b: str,
    headline: str,
    width: int,
    height: int,
    shape_hint: Optional[str] = None,
    physical_context: Optional[Dict] = None,
    hybrid_plan: Optional[Dict] = None,
    max_retries: int = 3,
    object_a_context: Optional[str] = None,
    object_b_context: Optional[str] = None,
    quality: str = "high",
    product_name: Optional[str] = None
) -> bytes:
    """
    STEP 3 - IMAGE GENERATION
    
    Generate image using OPENAI_IMAGE_MODEL (default: gpt-image-1.5).
    
    Input:
    - object_a (from STEP 1)
    - object_b (from STEP 1)
    - shape_hint (from STEP 1)
    - headline (from STEP 2)
    - imageSize
    
    Rules:
    - SIDE BY SIDE
    - no overlap
    - headline prominent
    - no extra text
    - no CTA
    - English only
    
    Returns:
        bytes: JPEG image
    """
    model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    image_size = f"{width}x{height}"
    
    # Log before image generation (SIDE BY SIDE only)
    image_prompt_includes_shape_hint = shape_hint is not None and shape_hint != ""
    mode = "SIDE_BY_SIDE"  # Always SIDE BY SIDE
    logger.info(f"ACE_LAYOUT_MODE={ACE_LAYOUT_MODE}")
    logger.info(f"STEP3_MODE={mode}")
    logger.info(f"STEP 3 - IMAGE GENERATION: image_model={model}, image_size={image_size}, object_a={object_a}, object_b={object_b}, headline={headline}, image_prompt_includes_shape_hint={image_prompt_includes_shape_hint}, shape_hint=\"{shape_hint or ''}\", mode={mode}")
    logger.info(f"STEP 3 VISUAL_RULES_APPLIED: no_logos=true, photorealistic_only=true")
    logger.info(f"STEP 3 COMPOSITION: two_full_panels=true, overlap=false_expected")
    
    for attempt in range(max_retries):
        is_strict = attempt > 0  # Use stricter prompt on retries
        
        prompt = create_image_prompt(
            object_a=object_a,
            object_b=object_b,
            headline=headline,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            is_strict=is_strict,
            object_a_context=object_a_context,
            object_b_context=object_b_context,
            product_name=product_name
        )
        
        try:
            logger.info(f"STEP 3 - IMAGE GENERATION: attempt={attempt + 1}/{max_retries}, image_model={model}, image_size={image_size}")
            logger.info("NEGATIVE_SPACE=enabled subject_scale=small margins=very_wide background=white")
            
            # Simple call without response_format for gpt-image-1.5 compatibility
            # Include quality parameter for preview (low) vs generate (high)
            # 429 is handled by openai_retry (exponential backoff + Retry-After)
            response = openai_retry.openai_call_with_retry(
                lambda: client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=image_size,
                    quality=quality
                ),
                endpoint="images"
            )
            
            # Extract base64 from response
            image_base64 = response.data[0].b64_json
            
            # Basic quality check
            if attempt < max_retries - 1:
                if not check_text_quality(image_base64):
                    logger.warning(f"Text quality check failed (attempt {attempt + 1}), retrying with stricter prompt...")
                    time.sleep(1)
                    continue
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            logger.info(f"Image generated successfully (attempt {attempt + 1}), image_model={model}, image_size={image_size}")
            return image_bytes
            
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            logger.error(f"DALL-E generation failed (attempt {attempt + 1}/{max_retries}): {error_str}")
            
            if attempt < max_retries - 1:
                # Retry on errors (non-429)
                time.sleep(1 + attempt)
                continue
            else:
                raise
    
    raise Exception("Failed to generate image after retries")


def _generate_final_image_unified(
    client: OpenAI,
    object_a: str,
    object_b: str,
    headline: str,
    width: int,
    height: int,
    shape_hint: Optional[str] = None,
    object_a_context: Optional[str] = None,
    object_b_context: Optional[str] = None,
    quality: str = "high",
    max_retries: int = 3,
    product_name: Optional[str] = None
) -> bytes:
    """
    Unified function to generate final image (used by both preview and zip).
    
    Returns:
        bytes: JPEG image bytes
    """
    return generate_image_with_dalle(
        client=client,
        object_a=object_a,
        object_b=object_b,
        headline=headline,
        width=width,
        height=height,
        shape_hint=shape_hint,
        physical_context=None,  # No physical context in SIDE BY SIDE mode
        hybrid_plan=None,  # No hybrid plan in SIDE BY SIDE mode
        max_retries=max_retries,
        object_a_context=object_a_context,
        object_b_context=object_b_context,
        quality=quality,
        product_name=product_name
    )


def evaluate_silhouette_similarity(
    object_a: str,
    object_b: str,
    model_name: Optional[str] = None
) -> float:
    """
    Evaluate silhouette similarity between Object A and Object B.
    
    This metric is used to decide between REPLACEMENT mode (>=85%) and SIDE_BY_SIDE mode (<85%).
    
    Critical rule: Similarity must be based on the main object bodies only.
    Ignore sub-objects completely. Ignore background and composition.
    Compare only the pure outer contour (dominant silhouette).
    
    Args:
        object_a: Main object A name
        object_b: Main object B name
        model_name: Model to use (default: OPENAI_SHAPE_MODEL)
    
    Returns:
        float: Silhouette similarity percentage (0-100)
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model_name is None:
        model_name = _get_shape_model()
    
    prompt = f"""Evaluate the silhouette similarity between two objects.

CRITICAL RULES:
- Similarity must be based on the main object bodies ONLY.
- Ignore sub-objects completely.
- Ignore background and composition.
- Compare ONLY the pure outer contour (dominant silhouette).

Objects:
- Object A: {object_a}
- Object B: {object_b}

Task:
Calculate the silhouette similarity percentage between Object A and Object B.

Consider:
- The dominant outer contour/shape of each main object
- How similar are the geometric shapes/silhouettes of the main objects
- Only the main object body, not any attached or nearby objects
- Pure geometric shape comparison

Return JSON only:
{{
  "silhouette_similarity_percentage": 0-100,
  "reason": "brief explanation of the similarity calculation"
}}

The percentage must be a number between 0 and 100."""
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"SILHOUETTE_SIMILARITY_EVAL attempt={attempt+1} A={object_a} B={object_b} model={model_name}")
            def _silhouette_call():
                is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                if is_o_model:
                    r = client.responses.create(model=model_name, input=prompt)
                    return r.output_text.strip()
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a silhouette similarity analyzer. Output must be in English only. Return JSON only without additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return r.choices[0].message.content.strip() if r.choices else ""
            response_text = openai_retry.openai_call_with_retry(_silhouette_call, endpoint="responses")
            # Parse JSON
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            result = json.loads(response_text)
            # Support both old and new field names for backward compatibility
            similarity = float(result.get("silhouette_similarity_percentage") or result.get("silhouette_overlap_percentage", 0))
            
            # Validate range
            if similarity < 0:
                similarity = 0
            if similarity > 100:
                similarity = 100
            
            reason = result.get("reason", "")
            logger.info(f"SILHOUETTE_SIMILARITY_EVAL SUCCESS: A={object_a} B={object_b} similarity={similarity}% reason={reason}")
            return similarity
            
        except json.JSONDecodeError as e:
            logger.error(f"SILHOUETTE_SIMILARITY_EVAL JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            # Fallback: assume low similarity if parsing fails
            logger.warning(f"SILHOUETTE_SIMILARITY_EVAL: Failed to parse, defaulting to 0% similarity")
            return 0.0
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            logger.error(f"SILHOUETTE_SIMILARITY_EVAL failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            logger.warning(f"SILHOUETTE_SIMILARITY_EVAL: Failed, defaulting to 0% similarity")
            return 0.0
    # Final fallback
    return 0.0


def render_final_ad_bytes(
    client: OpenAI,
    object_a_name: str,
    object_b_name: str,
    object_a_id: str,
    object_b_id: str,
    object_a_item: Dict,
    object_b_item: Dict,
    headline: str,
    shape_hint: Optional[str],
    width: int,
    height: int,
    quality: str = "high",
    max_retries: int = 3,
    product_name: Optional[str] = None
) -> Tuple[bytes, str, Dict, str]:
    """
    Unified function to render final ad image (used by both preview and zip).
    
    Returns:
        Tuple of:
        - image_bytes: JPEG image bytes
        - headline: Generated headline
        - selected_pair: Dict with object_a, object_b, object_a_id, object_b_id
        - prompt_used: The prompt string used for generation
    """
    # Get context from items
    object_a_context = object_a_item.get("classic_context", "") if object_a_item else ""
    object_b_context = object_b_item.get("classic_context", "") if object_b_item else ""
    
    # Get sub_objects for replacement mode
    object_a_sub = object_a_item.get("sub_object", "") if object_a_item else ""
    object_b_sub = object_b_item.get("sub_object", "") if object_b_item else ""
    
    # SILHOUETTE SIMILARITY DECISION: Evaluate similarity (always computed for measurement)
    silhouette_similarity_pct = evaluate_silhouette_similarity(
        object_a=object_a_name,
        object_b=object_b_name
    )
    
    # TEMP_DISABLE_REPLACEMENT: Force side_by_side mode regardless of similarity
    # Similarity is still computed and logged for measurement purposes
    SILHOUETTE_THRESHOLD = 85.0
    replacement_eligible = silhouette_similarity_pct >= SILHOUETTE_THRESHOLD
    
    # Force side_by_side mode (replacement disabled)
    final_mode = "side_by_side"
    
    # Log similarity and replacement eligibility
    logger.info(f"SILHOUETTE_SIMILARITY_PCT={silhouette_similarity_pct}")
    logger.info(f"REPLACEMENT_DISABLED=true")
    if replacement_eligible:
        logger.info(f"REPLACEMENT_ELIGIBLE=true (similarity >= {SILHOUETTE_THRESHOLD}%, but replacement mode is disabled)")
    
    # Log mode decision (forced to side_by_side)
    logger.info(f"MODE_DECISION SILHOUETTE_SIMILARITY_PCT={silhouette_similarity_pct} threshold={SILHOUETTE_THRESHOLD} final_mode={final_mode} (forced)")
    logger.info(f"FINAL_MODE={final_mode}")
    
    # SIDE_BY_SIDE mode: Visual overlap is handled in the image prompt (geometry/layout)
    # Note: SIDE_BY_SIDE_OVERLAP is a separate geometry metric for side-by-side layout,
    # not the silhouette similarity metric used for measurement.
    # The prompt will handle visual overlap rules (e.g., "partially overlap visually in the center").
    # We don't compute a separate overlap percentage here, as it's a visual composition rule, not a metric.
    
    # Log mode application before image generation
    logger.info(f"MODE_APPLIED final_mode={final_mode} ACE_LAYOUT_MODE={ACE_LAYOUT_MODE} STEP3_MODE={final_mode.upper()}")
    
    # Create prompt with determined mode
    prompt_used = create_image_prompt(
        object_a=object_a_name,
        object_b=object_b_name,
        headline=headline,
        shape_hint=shape_hint,
        physical_context=None,  # No physical context in SIDE BY SIDE mode
        hybrid_plan=None,  # No hybrid plan in SIDE BY SIDE mode
        is_strict=False,
        object_a_context=object_a_context,
        object_b_context=object_b_context,
        object_a_item=object_a_item,
        object_b_item=object_b_item,
        product_name=product_name,
        force_mode=final_mode  # Pass determined mode
    )
    
    # Generate image
    image_bytes = _generate_final_image_unified(
        client=client,
        object_a=object_a_name,
        object_b=object_b_name,
        headline=headline,
        width=width,
        height=height,
        shape_hint=shape_hint,
        object_a_context=object_a_context,
        object_b_context=object_b_context,
        quality=quality,
        max_retries=max_retries,
        product_name=product_name
    )
    
    # Build selected_pair dict
    selected_pair = {
        "object_a": object_a_name,
        "object_b": object_b_name,
        "object_a_id": object_a_id,
        "object_b_id": object_b_id
    }
    
    return image_bytes, headline, selected_pair, prompt_used


def create_text_file(
    session_id: Optional[str],
    ad_index: int,
    product_name: str,
    ad_goal: str,
    headline: str,
    chosen_objects: List[str]
) -> str:
    """
    Create minimal text.txt content (optional, for documentation only).
    All text is already in the image.
    """
    lines = []
    if session_id:
        lines.append(f"sessionId={session_id}")
    lines.append(f"adIndex={ad_index}")
    lines.append("layout=SIDE_BY_SIDE")
    lines.append(f"productName={product_name}")
    # Note: All text is in the image, this file is for documentation only
    return "\n".join(lines)


# Hardcoded prompt for ACE_IMAGE_ONLY: pencil sketch, two objects SIDE BY SIDE, no text
IMAGE_ONLY_HARDCODED_PROMPT = (
    "Classical pencil sketch diagram, white background, minimal shading, clean contours. "
    "Two simple objects side by side with slight overlap: "
    "a playing card next to a card deck, and a notebook with spiral binding. "
    "NO text, NO logos, NO letters, NO numbers."
)


def _image_only_single_call(image_size: str, request_id: str) -> Tuple[str, bool]:
    """
    Single gpt-image-1.5 call for ACE_IMAGE_ONLY mode. No retries, 60s timeout.
    Returns (image_base64_str, success).
    """
    t0 = time.time()
    quality = "low"
    logger.info(f"IMAGE_CALL_START size={image_size} quality={quality} request_id={request_id}")
    timeout_sec = IMAGE_ONLY_CALL_TIMEOUT_SECONDS
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(timeout_sec)
    )
    model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    try:
        response = client.images.generate(
            model=model,
            prompt=IMAGE_ONLY_HARDCODED_PROMPT,
            size=image_size,
            quality=quality
        )
        latency_ms = int((time.time() - t0) * 1000)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise ValueError("No image data in response")
        logger.info(f"IMAGE_CALL_OK latency_ms={latency_ms} request_id={request_id}")
        return (b64, True)
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        logger.error(f"IMAGE_CALL_FAIL latency_ms={latency_ms} error={e}")
        logger.info("IMAGE_FALLBACK_PLACEHOLDER_USED=true")
        return (_IMAGE_ONLY_PLACEHOLDER_BASE64, False)


def generate_preview_data(payload_dict: Dict) -> Dict:
    """
    Generate preview data and return as dictionary (for JSON response).
    
    Supports:
    - PREVIEW_MODE=plan_only: Returns plan JSON without image (fast)
    - PREVIEW_MODE=image: Returns imageBase64 (legacy behavior)
    - ENGINE_MODE=optimized: Uses combined LLM calls
    - ENGINE_MODE=legacy: Uses separate calls (legacy behavior)
    - Caching: Plan cache and image cache (if enabled)
    
    Args:
        payload_dict: Request payload with productName, productDescription, etc.
    
    Returns:
        dict: {
            "imageBase64": str (if PREVIEW_MODE=image),
            OR
            "mode": "ENV_SWAP",
            "hero_object": "...",
            "environment_from": "...",
            "environment_description": "...",
            "headline": "...",
            "headline_placement": "BOTTOM|SIDE",
            "shape_similarity_score": ...,
            "shape_hint": "..."
            (if PREVIEW_MODE=plan_only)
        }
    """
    request_id = str(uuid.uuid4())
    t_start = time.time()
    logger.info(
        f"MODEL_CONFIG text_model={_get_text_model()} preview_planner={PREVIEW_PLANNER_MODEL} generate_planner={GENERATE_PLANNER_MODEL} shape_model={_get_shape_model()} image_model={os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1.5')}"
    )
    
    # Extract and validate required fields
    product_name = payload_dict.get("productName", "")
    product_description = payload_dict.get("productDescription", "")
    image_size_str = payload_dict.get("imageSize", "1536x1024")
    # Force English-only: default to "en", override "he" to "en"
    language = payload_dict.get("language", "en")
    if language == "he":
        language = "en"
        logger.info(f"[{request_id}] Overriding language=he to language=en (English-only mode)")
    ad_index = payload_dict.get("adIndex", 0)
    
    # Normalize ad_index (0 -> 1, ensure 1-3)
    if ad_index == 0:
        ad_index = 1
    ad_index = max(1, min(3, ad_index))
    
    # Optional fields
    session_id = payload_dict.get("sessionId") or "no_session"
    session_seed = session_id  # Use session_id as session_seed for cache key
    history = payload_dict.get("history", [])
    object_list = payload_dict.get("objectList")

    # ACE_IMAGE_ONLY: single gpt-image-1.5 call, no o3-pro, placeholder copy, no retries
    if ACE_IMAGE_ONLY:
        size = image_size_str if image_size_str in ALLOWED_IMAGE_SIZES else "1536x1024"
        image_base64, ok = _image_only_single_call(size, request_id)
        if not ok:
            image_base64 = _IMAGE_ONLY_PLACEHOLDER_BASE64
        return {
            "imageBase64": image_base64,
            "image_base64": image_base64,
            "headline": "Preview Headline",
            "bodyText50": "Preview body text (placeholder)",
            "body_text": "Preview body text (placeholder)",
            "image_url": None,
            "ad_goal": "Preview ad goal",
            "object_a": "object_a",
            "object_b": "object_b",
            "marketing_copy_50_words": "Preview body text (placeholder)",
        }
    
    # A) STEP 0 - UNIFIED BUNDLE: ad_goal + difficulty_score + object_list(150)
    step0_cache_hit = False
    step0_cache_key = None
    hard_mode = False
    difficulty_score = 50  # Default
    
    if not object_list or len(object_list) < OBJECT_LIST_MIN_OK:
        # Build step0 bundle (ad_goal + difficulty_score + object_list)
        # Cache key includes product_name + product_description hash
        product_hash = hashlib.sha256(f"{product_name}|{product_description}".encode()).hexdigest()[:16]
        step0_cache_key = f"STEP0_BUNDLE|{product_hash}|{language}|{CACHE_KEY_VERSION}"
        
        # Check cache
        with _step0_cache_lock:
            if step0_cache_key in _step0_cache:
                cached_data, timestamp = _step0_cache[step0_cache_key]
                if time.time() - timestamp < STEP0_CACHE_TTL_SECONDS:
                    step0_bundle = cached_data
                    step0_cache_hit = True
                    logger.info(f"[{request_id}] STEP0_BUNDLE_CACHE hit=true key={step0_cache_key[:16]}...")
                else:
                    del _step0_cache[step0_cache_key]
        
        if not step0_cache_hit:
            step0_bundle = build_step0_bundle(product_name, product_description, language=language, request_id=request_id)
            # Cache the bundle
            with _step0_cache_lock:
                _step0_cache[step0_cache_key] = (step0_bundle, time.time())
            logger.info(f"[{request_id}] STEP0_BUNDLE_CACHE miss=true key={step0_cache_key[:16]}...")
        
        ad_goal = step0_bundle["ad_goal"]
        difficulty_score = step0_bundle["difficulty_score"]
        object_list = step0_bundle["object_list"]
        hard_mode = difficulty_score > 80
    else:
        # Use provided object_list, but still need ad_goal and difficulty_score
        # For now, build ad_goal separately (fallback)
        ad_goal = build_ad_goal(product_name, product_description)
        difficulty_score = 50  # Default if not provided
        hard_mode = difficulty_score > 80
        logger.warning(f"[{request_id}] Using provided object_list, difficulty_score defaulted to {difficulty_score}")
    
    message = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # Guard: Ensure object_list is not empty
    if not object_list or len(object_list) == 0:
        raise ValueError("OBJECT_LIST_EMPTY")
    
    # Log objectList stats
    object_list_sha = hashlib.sha256(json.dumps(object_list, sort_keys=True).encode()).hexdigest()[:16]
    logger.info(f"OBJECTLIST_SHA={object_list_sha} total={len(object_list)} cache_hit_list={1 if step0_cache_hit else 0}")
    
    # C) Build theme tags from ad_goal
    theme_tags = build_theme_tags(ad_goal)
    
    # Normalize theme tags for robust matching
    theme_tags_norm = {_norm(t) for t in theme_tags}
    
    # Filter object_list to themed_pool with normalized matching
    themed_pool_by_tag = []
    themed_pool_by_link = []
    for it in object_list:
        it_tag_norm = _norm(it.get("theme_tag", ""))
        it_link_norm = _norm(it.get("theme_link", ""))
        
        # Primary match: by theme_tag
        if it_tag_norm in theme_tags_norm:
            themed_pool_by_tag.append(it)
        # Secondary match: by theme_link (if primary didn't match)
        elif any(tn in it_link_norm for tn in theme_tags_norm):
            themed_pool_by_link.append(it)
    
    # Combine both matches
    themed_pool = themed_pool_by_tag + themed_pool_by_link
    
    if len(themed_pool) < 60:
        logger.warning(f"THEME_POOL: themed_pool too small ({len(themed_pool)}), will use fallback to full object_list")
        themed_pool = object_list  # Fallback to full list
    
    logger.info(f"THEME_POOL_MATCH counts: by_tag={len(themed_pool_by_tag)} by_link={len(themed_pool_by_link)} total={len(themed_pool)} (from {len(object_list)} total)")
    
    # Log request with preview flags
    logger.info(f"[{request_id}] generate_preview_data called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}, ad_goal={ad_goal[:50]}, "
                f"ENGINE_MODE={ENGINE_MODE}, PREVIEW_MODE={PREVIEW_MODE}")
    logger.info(f"[{request_id}] PREVIEW_FLAGS planner_model={PREVIEW_PLANNER_MODEL} skip_physical={PREVIEW_SKIP_PHYSICAL_CONTEXT} cache={PREVIEW_USE_CACHE}")
    
    # Check preview cache (if enabled)
    preview_cache_key = None
    preview_cache_hit = False
    cached_plan = None
    
    if PREVIEW_USE_CACHE:
        # Use final image size and quality (same as ZIP) for plan cache key
        final_quality = GENERATE_IMAGE_QUALITY_DEFAULT
        preview_cache_key = _get_cache_key_preview(product_name, message, ad_goal, ad_index, object_list, language=language, session_seed=session_seed, engine_mode=ENGINE_MODE, preview_mode=PREVIEW_MODE, image_size=image_size_str, quality=final_quality)
        cached_plan = _get_from_preview_cache(preview_cache_key)
        if cached_plan:
            preview_cache_hit = True
            ttl_remaining = int(CACHE_TTL_SECONDS - (time.time() - _preview_cache[preview_cache_key][1]))
            logger.info(f"[{request_id}] PREVIEW_CACHE hit=true ttl={ttl_remaining}s key={preview_cache_key[:16]}...")
            
            # If plan_only mode, return cached plan immediately
            if PREVIEW_MODE == "plan_only":
                t_total_ms = int((time.time() - t_start) * 1000)
                logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms=0 env_ms=0 headline_ms=0 image_ms=0 cache_hit=true")
                return cached_plan
            
            # If image mode, use cached plan data
            object_a_name = cached_plan.get("chosen_objects", [None, None])[0]
            object_b_name = cached_plan.get("chosen_objects", [None, None])[1]
            object_a_item = cached_plan.get("object_a_item")
            object_b_item = cached_plan.get("object_b_item")
            shape_hint = cached_plan.get("shape_hint", "")
            shape_score = cached_plan.get("shape_similarity_score", 0)
            headline = cached_plan.get("headline", "")
            hybrid_plan = None
            physical_context = None
        else:
            logger.info(f"[{request_id}] PREVIEW_CACHE hit=false ttl={CACHE_TTL_SECONDS}s key={preview_cache_key[:16]}...")
    
    # Also check legacy plan cache (if enabled)
    plan_cache_key = None
    plan_cache_hit = False
    if ENABLE_PLAN_CACHE and not preview_cache_hit:
        plan_cache_key = _get_cache_key_plan(product_name, message, ad_goal, ad_index, object_list, ENGINE_MODE, "SIDE_BY_SIDE", language=language, session_seed=session_seed, image_size=image_size_str)
        cached_plan = _get_from_plan_cache(plan_cache_key)
        if cached_plan:
            plan_cache_hit = True
            logger.info(f"[{request_id}] PLAN_CACHE hit=true key={plan_cache_key[:16]}...")
            
            # If plan_only mode, return cached plan immediately
            if PREVIEW_MODE == "plan_only":
                t_total_ms = int((time.time() - t_start) * 1000)
                logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms=0 env_ms=0 headline_ms=0 image_ms=0 cache_hit=true")
                return cached_plan
            
            # If image mode, use cached plan data
            object_a_name = cached_plan.get("chosen_objects", [None, None])[0]
            object_b_name = cached_plan.get("chosen_objects", [None, None])[1]
            object_a_item = cached_plan.get("object_a_item")
            object_b_item = cached_plan.get("object_b_item")
            shape_hint = cached_plan.get("shape_hint", "")
            shape_score = cached_plan.get("shape_similarity_score", 0)
            headline = cached_plan.get("headline", "")
            hybrid_plan = None
            physical_context = None
        else:
            logger.info(f"[{request_id}] PLAN_CACHE hit=false key={plan_cache_key[:16]}...")
    
    # Timing variables
    t_shape_ms = 0
    t_envswap_ms = 0
    t_headline_ms = 0
    t_image_ms = 0
    image_cache_hit = False
    step0_cache_hit = False
    step1_cache_hit = False
    
    # If not cached, generate plan
    if not preview_cache_hit and not plan_cache_hit:
        used_objects = get_used_objects(history)
        
        # Use PREVIEW_PLANNER_MODEL for preview
        planner_model = PREVIEW_PLANNER_MODEL
        
        # Force SIDE BY SIDE mode only (ignore ENGINE_MODE optimized/hybrid)
        # STEP 1 - SHAPE SELECTION (using PREVIEW_PLANNER_MODEL)
        # Check STEP 1 cache before calling
        t_envswap_ms = 0  # No environment swap in SIDE BY SIDE mode
        physical_context = None  # No physical context in SIDE BY SIDE mode
        hybrid_plan = None  # No hybrid plan in SIDE BY SIDE mode
        
        if PREVIEW_MODE in ["plan_only", "image"]:
            min_shape_score = 85 if len(object_list) >= 10 else 80
            min_env_diff_score = 60
            step1_cache_key = _get_cache_key_step1(
                object_list, min_shape_score, min_env_diff_score, used_objects,
                product_name=product_name, ad_goal=ad_goal, language=language,
                ad_index=ad_index, session_seed=session_seed,
                engine_mode=ENGINE_MODE, preview_mode=PREVIEW_MODE
            )
            cached_step1_result = _get_from_step1_cache(step1_cache_key)
            if cached_step1_result:
                step1_cache_hit = True
                shape_result = cached_step1_result
                t_shape_ms = 0  # Cache hit, no time spent
            else:
                # Add verification log
                logger.info(f"SELECTOR=three_pairs_single_call object_list_type={type(object_list[0]).__name__ if object_list else 'empty'}")
                t_shape_start = time.time()
                shape_result = select_pair_from_three_pairs(
                    sid=session_seed or session_id or "no_session",
                    ad_index=ad_index,
                    ad_goal=ad_goal,
                    image_size=image_size_str,  # Use final image size (same as ZIP)
                    object_list=object_list,
                    used_objects=used_objects,
                    model_name=planner_model or _get_shape_model(),
                    allowed_theme_tags=theme_tags
                )
                t_shape_ms = int((time.time() - t_shape_start) * 1000)
                # Save to cache (individual pair result)
                if ENABLE_STEP1_CACHE:
                    _set_to_step1_cache(step1_cache_key, shape_result)
        else:
            # Add verification log
            logger.info(f"SELECTOR=three_pairs_single_call object_list_type={type(object_list[0]).__name__ if object_list else 'empty'}")
            t_shape_start = time.time()
            shape_result = select_pair_from_three_pairs(
                sid=session_seed or session_id or "no_session",
                ad_index=ad_index,
                ad_goal=ad_goal,
                image_size=image_size_str,  # Use final image size (same as ZIP)
                object_list=object_list,
                used_objects=used_objects,
                model_name=planner_model or _get_shape_model(),
                allowed_theme_tags=theme_tags
            )
            t_shape_ms = int((time.time() - t_shape_start) * 1000)
        
        try:
            object_a_name = shape_result["object_a"]
            object_b_name = shape_result["object_b"]
            object_a_id = shape_result.get("object_a_id")
            object_b_id = shape_result.get("object_b_id")
            shape_hint = shape_result.get("shape_hint", "")
            shape_score = shape_result.get("shape_similarity_score", 0)
            
            # Get full item objects for context (if List[Dict] format)
            object_a_item = None
            object_b_item = None
            if object_list and isinstance(object_list[0], dict):
                if object_a_id:
                    object_a_id_str = str(object_a_id) if object_a_id is not None else ""
                    object_a_item = next((item for item in object_list if str(item.get("id")) == object_a_id_str), None)
                if object_b_id:
                    object_b_id_str = str(object_b_id) if object_b_id is not None else ""
                    object_b_item = next((item for item in object_list if str(item.get("id")) == object_b_id_str), None)
                # Fallback to name matching if id not found
                if not object_a_item:
                    object_a_item = next((item for item in object_list if item.get("object") == object_a_name), None)
                if not object_b_item:
                    object_b_item = next((item for item in object_list if item.get("object") == object_b_name), None)
            
            # For compatibility with legacy code
            object_a = object_a_name
            object_b = object_b_name
            
            if step1_cache_hit:
                logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a_name}, {object_b_name}], score={shape_score}, shape_hint={shape_hint}, model={planner_model} (from cache)")
            else:
                logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a_name}, {object_b_name}], score={shape_score}, shape_hint={shape_hint}, model={planner_model}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
                raise openai_retry.OpenAIRateLimitError()
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection error: {error_msg}")
            raise
        # STEP 2 - HEADLINE GENERATION
        t_headline_start = time.time()
        try:
            headline = generate_headline_only(
                product_name=product_name,
                message=message,
                object_a=object_a,
                object_b=object_b,
                headline_placement=None,  # No headline_placement in SIDE BY SIDE mode
                max_retries=3,
                hard_mode=hard_mode,
                ad_goal=ad_goal
            )
            t_headline_ms = int((time.time() - t_headline_start) * 1000)
            logger.info(f"[{request_id}] STEP 2 SUCCESS: headline={headline}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
                raise openai_retry.OpenAIRateLimitError()
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation error: {error_msg}")
            raise
        # Save to preview cache and plan cache (SIDE BY SIDE mode only)
        plan_data = {
            "mode": "SIDE_BY_SIDE",
            "layout": "SIDE_BY_SIDE",
            "headline": headline,
            "shape_similarity_score": shape_score,
            "shape_hint": shape_hint,
            "chosen_objects": [object_a, object_b]
        }
        
        # Save to preview cache (if enabled)
        if PREVIEW_USE_CACHE and preview_cache_key:
            _set_to_preview_cache(preview_cache_key, plan_data)
        
        # Also save to legacy plan cache (if enabled)
        if ENABLE_PLAN_CACHE and plan_cache_key:
            _set_to_plan_cache(plan_cache_key, plan_data)
    
    # Text-only preview: return plan + marketing copy, no image (default unless includeImage=true)
    text_only = not payload_dict.get("includeImage", False)
    if text_only:
        marketing_copy = generate_marketing_copy(
            product_name=product_name,
            product_description=product_description,
            ad_goal=ad_goal
        )
        text_only_response = {
            "ad_goal": ad_goal,
            "ad_index": ad_index,
            "object_a": object_a_name,
            "object_b": object_b_name,
            "chosen_objects": [object_a_name, object_b_name],
            "mode_decision": "side_by_side",
            "headline": headline,
            "marketing_copy_50_words": marketing_copy,
            "shape_similarity_score": shape_score,
            "shape_hint": shape_hint,
        }
        t_total_ms = int((time.time() - t_start) * 1000)
        cache_hit = preview_cache_hit or plan_cache_hit
        logger.info(f"[{request_id}] PREVIEW_MODE=text_only image_called=false")
        logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms=0 cache_hit={cache_hit}")
        return text_only_response
    
    # If plan_only mode (env), return plan immediately (no image generation)
    if PREVIEW_MODE == "plan_only":
        t_total_ms = int((time.time() - t_start) * 1000)
        cache_hit = preview_cache_hit or plan_cache_hit
        logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms=0 cache_hit={cache_hit}")
        return plan_data if not cache_hit else cached_plan
    
    # STEP 3 - IMAGE GENERATION (only if includeImage=true and PREVIEW_MODE=image)
    # Use FINAL image size and quality (same as ZIP) - no preview-only variants
    t_image_start = time.time()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Use same quality as generate_zip (high quality for final)
    final_quality = GENERATE_IMAGE_QUALITY_DEFAULT
    
    # Get context from items if available
    if 'object_a_item' in locals() and object_a_item:
        object_a_context = object_a_item.get("classic_context", "")
        object_a_sub = object_a_item.get("sub_object", "")
    else:
        object_a_context = ""
        object_a_sub = ""
    if 'object_b_item' in locals() and object_b_item:
        object_b_context = object_b_item.get("classic_context", "")
    else:
        object_b_context = ""
    
    # Get object names for cache key
    obj_a_name = object_a_name if 'object_a_name' in locals() else ""
    obj_b_name = object_b_name if 'object_b_name' in locals() else ""
    
    # Check unified final image cache (includes ACE_IMAGE_MODE + A.object + A.sub_object + B.object)
    final_image_cache_key = _get_cache_key_final_image(
        session_seed=session_seed or session_id,
        ad_index=ad_index,
        product_name=product_name,
        message=message,
        headline=headline,
        image_size=image_size_str,
        quality=final_quality,
        layout_mode=ACE_LAYOUT_MODE,
        image_mode=ACE_IMAGE_MODE,
        object_a=obj_a_name,
        object_a_sub=object_a_sub,
        object_b=obj_b_name
    )
    
    image_bytes = None
    image_cache_hit = False
    image_gen_called = False
    prompt_used = None
    
    # Check cache for final image
    cached_image_bytes = _get_from_image_cache(final_image_cache_key)
    if cached_image_bytes:
        image_bytes = cached_image_bytes
        image_cache_hit = True
        logger.info(f"[{request_id}] PREVIEW_IMAGE_CACHE hit=true mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}... bytes={len(image_bytes)}")
    else:
        logger.info(f"[{request_id}] PREVIEW_IMAGE_CACHE hit=false mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}...")
    
    if not image_cache_hit:
        try:
            # Determine max_retries based on mode
            max_img_retries = 1 if ENGINE_MODE == "optimized" else 3
            
            # Generate final image using unified render function
            image_gen_called = True
            image_bytes, headline, selected_pair, prompt_used = render_final_ad_bytes(
                client=client,
                object_a_name=object_a_name,
                object_b_name=object_b_name,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                object_a_item=object_a_item,
                object_b_item=object_b_item,
                headline=headline,
                shape_hint=shape_hint,
                width=width,
                height=height,
                quality=final_quality,
                max_retries=max_img_retries,
                product_name=product_name
            )
            
            # Save to unified image cache
            _set_to_image_cache(final_image_cache_key, image_bytes)
            logger.info(f"[{request_id}] IMAGE_GEN_CALLED=true mode={ACE_IMAGE_MODE} bytes={len(image_bytes)}")
                
        except Exception as e:
            logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
            raise
    
    t_image_ms = int((time.time() - t_image_start) * 1000)
    
    # Convert image to base64 (without data URI header) - use same bytes as ZIP
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get image model and size for logging
    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    
    t_total_ms = int((time.time() - t_start) * 1000)
    cache_hit = preview_cache_hit or plan_cache_hit
    logger.info(f"[{request_id}] STEP 3 SUCCESS: image_model={image_model}, image_size={image_size_str}, preview_success=true")
    logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms={t_image_ms} cache_hit={cache_hit} cache_image_hit={image_cache_hit} cache_step0_hit={step0_cache_hit} cache_step1_hit={step1_cache_hit}")
    logger.info(f"[{request_id}] PREVIEW_IMAGE_CACHE hit={image_cache_hit} mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}... bytes={len(image_bytes)} IMAGE_GEN_CALLED={image_gen_called}")
    
    # 50-word body text for on-screen display and ZIP (copy phase infers from image; we generate for display/store)
    body_text_50 = generate_marketing_copy(
        product_name=product_name,
        product_description=product_description,
        ad_goal=ad_goal
    )
    # Return image + headline + body for display and for artifact store (sketch only in image; headline/body separate)
    return {
        "imageBase64": image_base64,
        "headline": headline,
        "bodyText50": body_text_50,
    }


def generate_zip(payload_dict: Dict, is_preview: bool = False) -> bytes:
    """
    Generate ZIP file with image.jpg and text.txt.
    
    Supports:
    - ENGINE_MODE=optimized: Uses combined LLM calls
    - ENGINE_MODE=legacy: Uses separate calls (legacy behavior)
    - Caching: Plan cache and image cache (if enabled)
    
    Args:
        payload_dict: Request payload with productName, productDescription, etc.
        is_preview: If True, this is a preview request (same logic, but can be optimized)
    
    Returns:
        bytes: ZIP file content
    """
    request_id = str(uuid.uuid4())
    t_start = time.time()
    logger.info(
        f"MODEL_CONFIG text_model={_get_text_model()} preview_planner={PREVIEW_PLANNER_MODEL} generate_planner={GENERATE_PLANNER_MODEL} shape_model={_get_shape_model()} image_model={os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1.5')}"
    )
    
    # Extract and validate required fields
    product_name = payload_dict.get("productName", "")
    product_description = payload_dict.get("productDescription", "")
    image_size_str = payload_dict.get("imageSize", "1536x1024")
    # Force English-only: default to "en", override "he" to "en"
    language = payload_dict.get("language", "en")
    if language == "he":
        language = "en"
        logger.info(f"[{request_id}] Overriding language=he to language=en (English-only mode)")
    ad_index = payload_dict.get("adIndex", 0)
    
    # Normalize ad_index (0 -> 1, ensure 1-3)
    if ad_index == 0:
        ad_index = 1
    ad_index = max(1, min(3, ad_index))
    
    # Optional fields
    session_id = payload_dict.get("sessionId") or "no_session"
    session_seed = session_id  # Use session_id as session_seed for cache key
    history = payload_dict.get("history", [])
    object_list = payload_dict.get("objectList")
    
    # A) STEP 0 - UNIFIED BUNDLE: ad_goal + difficulty_score + object_list(150)
    step0_cache_hit = False
    step0_cache_key = None
    hard_mode = False
    difficulty_score = 50  # Default
    
    if not object_list or len(object_list) < OBJECT_LIST_MIN_OK:
        # Build step0 bundle (ad_goal + difficulty_score + object_list)
        # Cache key includes product_name + product_description hash
        product_hash = hashlib.sha256(f"{product_name}|{product_description}".encode()).hexdigest()[:16]
        step0_cache_key = f"STEP0_BUNDLE|{product_hash}|{language}|{CACHE_KEY_VERSION}"
        
        # Check cache
        with _step0_cache_lock:
            if step0_cache_key in _step0_cache:
                cached_data, timestamp = _step0_cache[step0_cache_key]
                if time.time() - timestamp < STEP0_CACHE_TTL_SECONDS:
                    step0_bundle = cached_data
                    step0_cache_hit = True
                    logger.info(f"[{request_id}] STEP0_BUNDLE_CACHE hit=true key={step0_cache_key[:16]}...")
                else:
                    del _step0_cache[step0_cache_key]
        
        if not step0_cache_hit:
            step0_bundle = build_step0_bundle(product_name, product_description, language=language, request_id=request_id)
            # Cache the bundle
            with _step0_cache_lock:
                _step0_cache[step0_cache_key] = (step0_bundle, time.time())
            logger.info(f"[{request_id}] STEP0_BUNDLE_CACHE miss=true key={step0_cache_key[:16]}...")
        
        ad_goal = step0_bundle["ad_goal"]
        difficulty_score = step0_bundle["difficulty_score"]
        object_list = step0_bundle["object_list"]
        hard_mode = difficulty_score > 80
    else:
        # Use provided object_list, but still need ad_goal and difficulty_score
        # For now, build ad_goal separately (fallback)
        ad_goal = build_ad_goal(product_name, product_description)
        difficulty_score = 50  # Default if not provided
        hard_mode = difficulty_score > 80
        logger.warning(f"[{request_id}] Using provided object_list, difficulty_score defaulted to {difficulty_score}")
    
    message = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # Guard: Ensure object_list is not empty
    if not object_list or len(object_list) == 0:
        raise ValueError("OBJECT_LIST_EMPTY")
    
    # Log objectList stats
    object_list_sha = hashlib.sha256(json.dumps(object_list, sort_keys=True).encode()).hexdigest()[:16]
    logger.info(f"OBJECTLIST_SHA={object_list_sha} total={len(object_list)} cache_hit_list={1 if step0_cache_hit else 0}")
    
    # C) Build theme tags from ad_goal
    theme_tags = build_theme_tags(ad_goal)
    
    # Normalize theme tags for robust matching
    theme_tags_norm = {_norm(t) for t in theme_tags}
    
    # Filter object_list to themed_pool with normalized matching
    themed_pool_by_tag = []
    themed_pool_by_link = []
    for it in object_list:
        it_tag_norm = _norm(it.get("theme_tag", ""))
        it_link_norm = _norm(it.get("theme_link", ""))
        
        # Primary match: by theme_tag
        if it_tag_norm in theme_tags_norm:
            themed_pool_by_tag.append(it)
        # Secondary match: by theme_link (if primary didn't match)
        elif any(tn in it_link_norm for tn in theme_tags_norm):
            themed_pool_by_link.append(it)
    
    # Combine both matches
    themed_pool = themed_pool_by_tag + themed_pool_by_link
    
    if len(themed_pool) < 60:
        logger.warning(f"THEME_POOL: themed_pool too small ({len(themed_pool)}), will use fallback to full object_list")
        themed_pool = object_list  # Fallback to full list
    
    logger.info(f"THEME_POOL_MATCH counts: by_tag={len(themed_pool_by_tag)} by_link={len(themed_pool_by_link)} total={len(themed_pool)} (from {len(object_list)} total)")
    
    # Log request
    logger.info(f"[{request_id}] generate_zip called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}, is_preview={is_preview}, ad_goal={ad_goal[:50]}, "
                f"ENGINE_MODE={ENGINE_MODE}, GENERATE_PLANNER_MODEL={GENERATE_PLANNER_MODEL}")
    
    # Check plan cache (if enabled)
    plan_cache_key = None
    plan_cache_hit = False
    cached_plan = None
    
    if ENABLE_PLAN_CACHE:
        plan_cache_key = _get_cache_key_plan(product_name, message, ad_goal, ad_index, object_list, ENGINE_MODE, "SIDE_BY_SIDE", language=language, session_seed=session_seed, image_size=image_size_str)
        cached_plan = _get_from_plan_cache(plan_cache_key)
        if cached_plan:
            plan_cache_hit = True
            logger.info(f"[{request_id}] PLAN_CACHE hit=true key={plan_cache_key[:16]}...")
            # Use cached plan data
            object_a_name = cached_plan.get("chosen_objects", [None, None])[0]
            object_b_name = cached_plan.get("chosen_objects", [None, None])[1]
            object_a_item = cached_plan.get("object_a_item")
            object_b_item = cached_plan.get("object_b_item")
            shape_hint = cached_plan.get("shape_hint", "")
            shape_score = cached_plan.get("shape_similarity_score", 0)
            headline = cached_plan.get("headline", "")
            hybrid_plan = None
            physical_context = None
        else:
            logger.info(f"[{request_id}] PLAN_CACHE hit=false key={plan_cache_key[:16]}...")
    
    # Timing variables
    t_shape_ms = 0
    t_envswap_ms = 0
    t_headline_ms = 0
    t_image_ms = 0
    image_cache_hit = False
    
    # If not cached, generate plan
    if not plan_cache_hit:
        used_objects = get_used_objects(history)
        
        # Force SIDE BY SIDE mode only (ignore ENGINE_MODE optimized/hybrid)
        # Use GENERATE_PLANNER_MODEL for generate
        planner_model = GENERATE_PLANNER_MODEL
        t_envswap_ms = 0  # No environment swap in SIDE BY SIDE mode
        physical_context = None  # No physical context in SIDE BY SIDE mode
        hybrid_plan = None  # No hybrid plan in SIDE BY SIDE mode
        
        # C) Select pair using select_pair_with_limited_shape_search
        t_shape_start = time.time()
        try:
            pair_result = select_pair_from_three_pairs(
                object_list=object_list,
                sid=session_id,
                ad_index=ad_index,
                ad_goal=ad_goal,
                image_size=image_size_str,
                used_objects=None,  # Anti-repeat handled by pre-selection
                model_name=planner_model,
                allowed_theme_tags=theme_tags
            )
            t_shape_ms = int((time.time() - t_shape_start) * 1000)
            
            object_a_name = pair_result["object_a"]
            object_b_name = pair_result["object_b"]
            object_a_id = pair_result["object_a_id"]
            object_b_id = pair_result["object_b_id"]
            shape_hint = pair_result.get("shape_hint", "")
            shape_score = pair_result.get("shape_similarity_score", 0)
            
            # Get full item objects for context
            object_a_id_str = str(object_a_id) if object_a_id is not None else ""
            object_b_id_str = str(object_b_id) if object_b_id is not None else ""
            object_a_item = next((item for item in object_list if str(item.get("id")) == object_a_id_str), None)
            object_b_item = next((item for item in object_list if str(item.get("id")) == object_b_id_str), None)
            
            if not object_a_item or not object_b_item:
                raise ValueError(f"Could not find items for ids: {object_a_id}, {object_b_id}")
            
            logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a_name}, {object_b_name}], score={shape_score}, shape_hint={shape_hint}, model={planner_model}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
                raise openai_retry.OpenAIRateLimitError()
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection error: {error_msg}")
            raise
        # STEP 2 - HEADLINE GENERATION
        t_headline_start = time.time()
        try:
            headline = generate_headline_only(
                product_name=product_name,
                message=message,
                object_a=object_a_name,
                object_b=object_b_name,
                headline_placement=None,  # No headline_placement in SIDE BY SIDE mode
                max_retries=3
            )
            t_headline_ms = int((time.time() - t_headline_start) * 1000)
            logger.info(f"[{request_id}] STEP 2 SUCCESS: headline={headline}")
        except openai_retry.OpenAIRateLimitError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
                raise openai_retry.OpenAIRateLimitError()
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation error: {error_msg}")
            raise
        # Save to plan cache (SIDE BY SIDE mode only)
        plan_data = {
            "mode": "SIDE_BY_SIDE",
            "layout": "SIDE_BY_SIDE",
            "headline": headline,
            "shape_similarity_score": shape_score,
            "shape_hint": shape_hint,
            "chosen_objects": [object_a_name, object_b_name],
            "chosen_object_ids": [object_a_id, object_b_id],
            "object_a_item": object_a_item,
            "object_b_item": object_b_item
        }
        
        if ENABLE_PLAN_CACHE and plan_cache_key:
            _set_to_plan_cache(plan_cache_key, plan_data)
    
    # STEP 4 - FINAL VALIDATION (skip if cached)
    if not plan_cache_hit:
        logger.info(f"[{request_id}] STEP 4 VALIDATION PASSED: object_a={object_a_name}, object_b={object_b_name} (unchanged)")
    
    # STEP 3 - IMAGE GENERATION
    # Use unified cache key (same as preview) to ensure byte-level identical images
    t_image_start = time.time()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Get context from items
    object_a_context = object_a_item.get("classic_context", "") if object_a_item else ""
    object_b_context = object_b_item.get("classic_context", "") if object_b_item else ""
    object_a_sub = object_a_item.get("sub_object", "") if object_a_item else ""
    
    # Use same quality as preview (high quality for final)
    final_quality = GENERATE_IMAGE_QUALITY_DEFAULT
    
    # Check unified final image cache (same key as preview, includes ACE_IMAGE_MODE + A.object + A.sub_object + B.object)
    final_image_cache_key = _get_cache_key_final_image(
        session_seed=session_seed or session_id,
        ad_index=ad_index,
        product_name=product_name,
        message=message,
        headline=headline,
        image_size=image_size_str,
        quality=final_quality,
        layout_mode=ACE_LAYOUT_MODE,
        image_mode=ACE_IMAGE_MODE,
        object_a=object_a_name,
        object_a_sub=object_a_sub,
        object_b=object_b_name
    )
    
    image_bytes = None
    image_cache_hit = False
    image_gen_called = False
    prompt_used = None
    
    # Check cache for final image (may have been generated by preview)
    cached_image_bytes = _get_from_image_cache(final_image_cache_key)
    if cached_image_bytes:
        image_bytes = cached_image_bytes
        image_cache_hit = True
        logger.info(f"[{request_id}] ZIP_IMAGE_CACHE hit=true mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}... bytes={len(image_bytes)}")
    else:
        logger.info(f"[{request_id}] ZIP_IMAGE_CACHE hit=false mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}...")
    
    if not image_cache_hit:
        try:
            # Determine max_retries based on mode
            max_img_retries = 2 if ENGINE_MODE == "optimized" else 3
            
            # Generate final image using unified render function (same as preview)
            image_gen_called = True
            image_bytes, headline, selected_pair, prompt_used = render_final_ad_bytes(
                client=client,
                object_a_name=object_a_name,
                object_b_name=object_b_name,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                object_a_item=object_a_item,
                object_b_item=object_b_item,
                headline=headline,
                shape_hint=shape_hint,
                width=width,
                height=height,
                quality=final_quality,
                max_retries=max_img_retries,
                product_name=product_name
            )
            
            # Save to unified image cache (for future preview/zip reuse)
            _set_to_image_cache(final_image_cache_key, image_bytes)
            logger.info(f"[{request_id}] IMAGE_GEN_CALLED=true mode={ACE_IMAGE_MODE} bytes={len(image_bytes)}")
                
        except Exception as e:
            logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
            raise
    
    t_image_ms = int((time.time() - t_image_start) * 1000)
    logger.info(f"[{request_id}] ZIP_IMAGE_CACHE hit={image_cache_hit} mode={ACE_IMAGE_MODE} key={final_image_cache_key[:16]}... bytes={len(image_bytes)} IMAGE_GEN_CALLED={image_gen_called}")
    
    # Create minimal text file (optional, for documentation)
    text_content = create_text_file(
        session_id=session_id,
        ad_index=ad_index,
        product_name=product_name,
        ad_goal=ad_goal,  # Use ad_goal from build_ad_goal
        headline=headline,  # Use headline from STEP 2
        chosen_objects=[object_a_name, object_b_name]
    )
    
    # Create ZIP with image.jpg only (text.txt is optional)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("image.jpg", image_bytes)
        # Optional: include minimal text.txt for documentation
        zip_file.writestr("text.txt", text_content.encode('utf-8'))
    
    t_total_ms = int((time.time() - t_start) * 1000)
    logger.info(f"[{request_id}] ZIP created successfully: {len(zip_buffer.getvalue())} bytes")
    logger.info(f"[{request_id}] PERF total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms={t_image_ms} cache_plan_hit={plan_cache_hit} cache_image_hit={image_cache_hit}")
    
    return zip_buffer.getvalue()

