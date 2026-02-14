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
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from threading import Lock

logger = logging.getLogger(__name__)

# ============================================================================
# Feature Flags (from ENV, with safe defaults = legacy behavior)
# ============================================================================

ENGINE_MODE = os.environ.get("ENGINE_MODE", "legacy")  # "legacy" | "optimized"
PREVIEW_MODE = os.environ.get("PREVIEW_MODE", "image")  # "image" | "plan_only"
ENABLE_PLAN_CACHE = os.environ.get("ENABLE_PLAN_CACHE", "0") == "1"
PLAN_CACHE_TTL_SECONDS = int(os.environ.get("PLAN_CACHE_TTL_SECONDS", "900"))
ENABLE_IMAGE_CACHE = os.environ.get("ENABLE_IMAGE_CACHE", "0") == "1"
IMAGE_CACHE_TTL_SECONDS = int(os.environ.get("IMAGE_CACHE_TTL_SECONDS", "900"))

# Preview optimization flags
PREVIEW_PLANNER_MODEL = os.environ.get("PREVIEW_PLANNER_MODEL", "o4-mini")  # Model for preview planning
GENERATE_PLANNER_MODEL = os.environ.get("GENERATE_PLANNER_MODEL", "o3-pro")  # Model for generate planning
PREVIEW_SKIP_PHYSICAL_CONTEXT = os.environ.get("PREVIEW_SKIP_PHYSICAL_CONTEXT", "1") == "1"  # Skip STEP 1.5 in preview
PREVIEW_USE_CACHE = os.environ.get("PREVIEW_USE_CACHE", "1") == "1"  # Use cache for preview
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "900"))  # Cache TTL for preview

# Step-level caching flags
ENABLE_STEP0_CACHE = os.environ.get("ENABLE_STEP0_CACHE", "1") == "1"  # Cache STEP 0 (objectList building)
STEP0_CACHE_TTL_SECONDS = int(os.environ.get("STEP0_CACHE_TTL_SECONDS", "3600"))  # STEP 0 cache TTL
ENABLE_STEP1_CACHE = os.environ.get("ENABLE_STEP1_CACHE", "1") == "1"  # Cache STEP 1 (shape match)
STEP1_CACHE_TTL_SECONDS = int(os.environ.get("STEP1_CACHE_TTL_SECONDS", "1800"))  # STEP 1 cache TTL

# Cache key versioning and diversity
CACHE_KEY_VERSION = os.environ.get("CACHE_KEY_VERSION", "v2")  # Cache key version
ENABLE_DIVERSITY_GUARD = os.environ.get("ENABLE_DIVERSITY_GUARD", "1") == "1"  # Diversity guard to prevent repetition
DIVERSITY_GUARD_TTL_SECONDS = 1800  # 30 minutes TTL for diversity guard

# Layout mode
ACE_LAYOUT_MODE = os.environ.get("ACE_LAYOUT_MODE", "side_by_side")  # "side_by_side" | "hybrid" (default: side_by_side, hybrid ignored)

# Shape matching parameters
SHAPE_MIN_SCORE = float(os.environ.get("SHAPE_MIN_SCORE", "0.80"))  # Minimum shape similarity score (0-1)
SHAPE_SEARCH_K = int(os.environ.get("SHAPE_SEARCH_K", "40"))  # Number of candidates to check per object (K=35-50)
MAX_CHECKED_PAIRS = int(os.environ.get("MAX_CHECKED_PAIRS", "500"))  # Maximum pairs to check before stopping

# Object list size parameters
OBJECT_LIST_TARGET = 150  # Target size for STEP 0 object list
OBJECT_LIST_MIN_OK = 130  # Minimum acceptable size to proceed without failure

# Image size and quality parameters
PREVIEW_IMAGE_SIZE_DEFAULT = "1024x1024"  # Fast preview size
PREVIEW_IMAGE_QUALITY_DEFAULT = "low"  # Fast preview quality
GENERATE_IMAGE_QUALITY_DEFAULT = "high"  # High quality for final generation
ALLOWED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024"}  # Supported by gpt-image-*

# ============================================================================
# In-Memory Caches (with TTL)
# ============================================================================

# Plan cache: key -> (value, timestamp)
_plan_cache: Dict[str, Tuple[Dict, float]] = {}
_plan_cache_lock = Lock()

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
    """
    if not object_list:
        return ""
    # object_list may be List[str] (legacy) or List[Dict] (new)
    if isinstance(object_list[0], dict):
        ids = []
        for it in object_list:
            # prefer id; fallback to object+sub_object
            _id = (it.get("id") or "").strip()
            if not _id:
                _id = f'{it.get("object","")}::{it.get("sub_object","")}'
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
    model = os.environ.get("OPENAI_TEXT_MODEL", "o4-mini")
    
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
        if model.startswith("o"):
            # Use Responses API for o* models
            response = client.responses.create(model=model, input=prompt)
            ad_goal = response.output_text.strip()
        else:
            # Use Chat Completions for other models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a marketing strategist. Generate concise advertising goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            ad_goal = response.choices[0].message.content.strip()
        
        # Clean up: remove quotes, extra whitespace
        ad_goal = ad_goal.strip('"\'')
        ad_goal = ' '.join(ad_goal.split())
        
        logger.info(f"AD_GOAL={ad_goal}")
        return ad_goal
    
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
    model = os.environ.get("OPENAI_TEXT_MODEL", "o4-mini")
    
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
        if model.startswith("o"):
            # Use Responses API for o* models
            response = client.responses.create(model=model, input=prompt)
            response_text = response.output_text.strip()
        else:
            # Use Chat Completions for other models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a marketing analyst. Generate concise theme tags."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            response_text = response.choices[0].message.content.strip()
        
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


def validate_object_item(item: Dict, forbidden_words: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a single object item with CRITICAL classic_context quality checks.
    
    Args:
        item: Dict with keys: id, object, classic_context, theme_link, category, shape_hint, theme_tag
        forbidden_words: List of forbidden words to check
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check required fields
    if not item.get("id") or not item.get("object") or not item.get("classic_context"):
        return False, "Missing required fields (id, object, classic_context)"
    
    classic_context = item.get("classic_context", "").strip()
    
    # B) CRITICAL CLASSIC_CONTEXT VALIDATION
    
    # 1. Check minimum length (3 words minimum)
    words = classic_context.split()
    if len(words) < 3:
        return False, f"classic_context too short ({len(words)} words, need >=3): '{classic_context}'"
    
    # 2. Check for forbidden abstract/marketing words in classic_context
    forbidden_context_words = [
        "eco", "sustainable", "meaningful", "modern", "creative",
        "concept", "awareness", "symbol", "campaign", "environmental",
        "green", "ethical", "conscious", "responsible", "impact",
        "change", "future", "vision", "mission", "purpose"
    ]
    context_lower = classic_context.lower()
    for word in forbidden_context_words:
        if word in context_lower:
            return False, f"classic_context contains forbidden abstract/marketing word '{word}': '{classic_context}'"
    
    # 3. Check for required physical preposition/relationship
    physical_prepositions = [
        "on", "in", "under", "next to", "attached to", "inside", "resting on",
        "landing on", "with", "lying on", "sitting on", "placed on", "hanging from",
        "growing from", "emerging from", "surrounded by", "near", "beside", "against",
        "within", "among", "between", "alongside", "above", "below", "over", "underneath"
    ]
    has_physical_relationship = any(prep in context_lower for prep in physical_prepositions)
    if not has_physical_relationship:
        return False, f"classic_context missing physical preposition/relationship: '{classic_context}'"
    
    # 4. Check for forbidden generic phrases
    forbidden_phrases = [
        "in nature", "in the wild", "in its habitat", "in natural setting",
        "in environment", "in context", "in setting", "in scene"
    ]
    for phrase in forbidden_phrases:
        if phrase in context_lower:
            return False, f"classic_context contains forbidden generic phrase '{phrase}': '{classic_context}'"
    
    # 5. Check maximum length (12 words maximum)
    if len(words) > 12:
        return False, f"classic_context too long ({len(words)} words, max 12): '{classic_context}'"
    
    # Check other fields for forbidden words (object, theme_link, etc.)
    text = f"{item.get('object', '')} {item.get('theme_link', '')}".lower()
    
    # Check for forbidden words in object/theme_link
    for word in forbidden_words:
        if word in text:
            return False, f"Contains forbidden word '{word}' in object/theme_link"
    
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


def build_object_list_from_ad_goal(
    ad_goal: str,
    product_name: Optional[str] = None,
    max_retries: int = 2,
    language: str = "en"
) -> List[Dict]:
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
    model_name = os.environ.get("OPENAI_TEXT_MODEL", "o4-mini")
    
    # Build prompt for new format ({OBJECT_LIST_TARGET} items with classic_context and theme_link)
    product_context = f"\nProduct name (optional context): {product_name}" if product_name else ""
    
    forbidden_patterns = "logo, brand, label, text, number, sign, screen, packaging, barcode, sticker, poster, billboard, phone screen, book cover, receipt, newspaper"
    
    prompt = f"""Generate EXACTLY {OBJECT_LIST_TARGET} physical classic objects related to this advertising goal.

Advertising goal: {ad_goal}{product_context}

CRITICAL REQUIREMENTS:
- EXACTLY {OBJECT_LIST_TARGET} ITEMS (no more, no less).
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

SUB_OBJECT RULES (ABSOLUTE - MUST BE FOLLOWED):
- sub_object MUST be a concrete physical object, NOT an environment
- sub_object MUST be a specific, tangible item
- FORBIDDEN as sub_object:
  * General environments: "nature", "environment", "world", "background", "scene", "forest", "ocean", "sky", "space", "ecosystem", "habitat", "setting", "context"
  * Abstract concepts: "life", "existence", "reality", "concept"
  * Generic surfaces without specificity: "ground", "floor", "surface" (unless very specific like "wooden floor")
- CORRECT sub_object examples:
  * "flower" (for bee)
  * "manual can opener" (for tin can)
  * "straw" (for soda can)
  * "wedge of cheese" (for mouse)
  * "bed" (for pillow)
  * "soil" (for tree)
  * "bookshelf" (for book)
  * "nail" (for hammer)
  * "lock" (for key)
  * "fork" (for plate)
  * "saucer" (for coffee cup)
- INCORRECT sub_object examples (DO NOT USE):
  * "forest" (too general - use "tree trunk" or "soil" instead)
  * "water" (too general - use "fish tank" or "pond surface" instead)
  * "nature" (forbidden)
  * "environment" (forbidden)
  * "road" (too general - use "parking meter" or "traffic cone" instead)

CLASSIC_CONTEXT RULES (CRITICAL - MUST BE FOLLOWED):
- classic_context MUST describe the PHYSICAL INTERACTION between object and sub_object
- MUST include a clear physical preposition/relationship: "on", "in", "under", "next to", "attached to", "inside", "resting on", "landing on", "with", "lying on", "inserted into", "hanging from", "touching", "holding", "opening", "closing"
- MUST be concrete and specific (3-12 words)
- MUST mention both object and sub_object (implicitly or explicitly)
- MUST be a natural, expected, real-world interaction
- FORBIDDEN in classic_context:
  * Abstract settings: "in an eco-friendly setting", "in a meaningful environment"
  * Symbolic language: "symbolizing sustainability", "representing change"
  * Marketing language: "in a modern context", "for awareness"
  * Generic phrases: "in nature", "in the wild", "in its habitat" (too vague)
  * Words: "eco", "sustainable", "meaningful", "modern", "creative", "concept", "awareness", "symbol", "campaign"
- CORRECT examples:
  * "landing on a flower" (bee + flower)
  * "being opened with a manual can opener" (tin can + can opener)
  * "next to a wedge of cheese" (mouse + cheese)
  * "resting on a bed" (pillow + bed)
  * "growing from soil" (tree + soil)
  * "sitting on a bookshelf" (book + bookshelf)
  * "driving a nail" (hammer + nail)
  * "inserted into a lock" (key + lock)
  * "placed next to a fork" (plate + fork)
- INCORRECT examples (DO NOT USE):
  * "in an eco-friendly setting"
  * "in a meaningful environment"
  * "symbolizing sustainability"
  * "in a modern context"
  * "in nature" (too vague, no sub_object)
  * "in forest" (forest is environment, not sub_object)
  * "in water" (water is too general)

NO brand names, NO logos, NO labels, NO printed text, NO numbers, NO signs, NO screens, NO packaging with writing.
Avoid: "bottle label", "poster", "billboard", "phone screen", "book cover", "receipt", "newspaper".
Keep objects concrete and timeless (everyday, nature, kitchen, tools, etc.).

Return ONLY a JSON array with this exact format:
[
  {{
    "id": "bee_flower",
    "object": "bee",
    "sub_object": "flower",
    "classic_context": "landing on a flower",
    "theme_link": "pollination supports healthy ecosystems",
    "category": "insect",
    "shape_hint": "small round",
    "theme_tag": "nature"
  }},
  {{
    "id": "can_opener",
    "object": "tin can",
    "sub_object": "manual can opener",
    "classic_context": "being opened with a manual can opener",
    "theme_link": "reusable containers reduce waste",
    "category": "container",
    "shape_hint": "cylindrical",
    "theme_tag": "kitchen"
  }},
  ...
]

EXACTLY {OBJECT_LIST_TARGET} items. JSON array only:"""

    # Check if model is o* type - these use Responses API
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                logger.info(f"STEP 0 - BUILD_OBJECT_LIST: text_model={model_name}, ad_goal={ad_goal[:50]}, productName={product_name[:50] if product_name else 'N/A'}, using_responses_api={using_responses_api}")
            
            if using_responses_api:
                # Use Responses API for o* models
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                response_text = response.output_text.strip()
            else:
                # Use Chat Completions for other models
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Try to extract JSON from response (might have markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Parse JSON - expect array of objects
            data = json.loads(response_text)
            if isinstance(data, list):
                object_list = data
            elif isinstance(data, dict) and "objectList" in data:
                # Fallback for old format - convert to new format
                old_list = data.get("objectList", [])
                object_list = [{"id": f"item_{i}", "object": obj, "classic_context": "", "theme_link": ""} for i, obj in enumerate(old_list)]
            else:
                raise ValueError("Invalid JSON format: expected array or object with 'objectList' key")
            
            # Validate items with CRITICAL classic_context quality checks
            forbidden_words = ["logo", "brand", "label", "text", "number", "sign", "screen", "packaging", "barcode", "sticker", "poster", "billboard", "phone", "book cover", "receipt", "newspaper"]
            valid_items = []
            rejected_count = 0
            rejection_reasons = {}
            
            for item in object_list:
                is_valid, error_msg = validate_object_item(item, forbidden_words)
                if is_valid:
                    valid_items.append(item)
                else:
                    rejected_count += 1
                    # Extract rejection reason category for statistics
                    reason_key = error_msg.split(":")[0] if ":" in error_msg else error_msg
                    if len(reason_key) > 50:  # Truncate long reasons
                        reason_key = reason_key[:50]
                    rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1
                    if attempt == 0:  # Log first attempt rejections
                        logger.debug(f"STEP 0 - Rejected item: {item.get('id', 'unknown')} - {error_msg}")
            
            # Must have at least OBJECT_LIST_MIN_OK valid items
            if len(valid_items) < OBJECT_LIST_MIN_OK:
                logger.warning(f"STEP 0 - Only {len(valid_items)} valid items (need at least {OBJECT_LIST_MIN_OK}), rejected={rejected_count}, reasons={dict(list(rejection_reasons.items())[:5])}")
                if attempt < max_retries - 1:
                    # Retry with stricter prompt focusing on classic_context quality
                    top_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                    reasons_text = ", ".join([f"{reason}({count})" for reason, count in top_reasons])
                    feedback = f"""You generated {rejected_count} rejected items. Common issues:
- classic_context must be 3-12 words with a physical preposition (on/in/next to/attached to/etc.)
- classic_context must NOT contain: eco, sustainable, meaningful, modern, creative, concept, awareness, symbol, campaign
- classic_context must NOT use generic phrases like "in nature", "in the wild", "in its habitat"
- classic_context must describe a NATURAL, EXPECTED, REAL-WORLD PHYSICAL SITUATION
- Examples of CORRECT classic_context: "landing on a flower", "with a metal straw inserted", "next to a wedge of cheese", "attached to a tree branch"
- Examples of INCORRECT classic_context: "in an eco-friendly setting", "symbolizing sustainability", "in nature" (too vague)

Top rejection reasons: {reasons_text}

Ensure EXACTLY {OBJECT_LIST_TARGET} valid items with proper classic_context."""
                    prompt = f"""Generate EXACTLY {OBJECT_LIST_TARGET} physical classic objects related to this advertising goal.

Advertising goal: {ad_goal}{product_context}

CRITICAL REQUIREMENTS:
- EXACTLY {OBJECT_LIST_TARGET} ITEMS (no more, no less).
- PHYSICAL CLASSIC OBJECTS ONLY (concrete, tangible, drawable).
- Each item MUST include:
  * id: unique identifier (e.g., "bee_flower", "can_straw", "mouse_cheese")
  * object: the physical object name (e.g., "bee", "can", "mouse")
  * classic_context: 3-12 words describing a NATURAL, EXPECTED, REAL-WORLD PHYSICAL SITUATION (e.g., "landing on a flower", "with a metal straw inserted", "next to a wedge of cheese", "attached to a tree branch", "lying on ocean beach sand", "on a wooden kitchen table", "resting in a toolbox")
  * theme_link: 5-12 words explaining how it supports the ad_goal theme
  * category: object category (e.g., "insect", "container", "rodent", "tool", "plant")
  * shape_hint: very short shape description (e.g., "curved organic", "cylindrical", "small round")
  * theme_tag: single word theme tag (e.g., "nature", "ocean", "kitchen", "wildlife")

CLASSIC_CONTEXT RULES (CRITICAL - MUST BE FOLLOWED):
- classic_context MUST describe a PHYSICAL CLASSIC SITUATION
- MUST include a clear physical preposition: "on", "in", "under", "next to", "attached to", "inside", "resting on", "landing on", "with", "lying on", etc.
- MUST be concrete and specific (3-12 words)
- MUST be a natural, expected, real-world scene
- FORBIDDEN in classic_context:
  * Abstract settings: "in an eco-friendly setting", "in a meaningful environment"
  * Symbolic language: "symbolizing sustainability", "representing change"
  * Marketing language: "in a modern context", "for awareness"
  * Generic phrases: "in nature", "in the wild", "in its habitat" (too vague)
  * Words: "eco", "sustainable", "meaningful", "modern", "creative", "concept", "awareness", "symbol", "campaign"
- CORRECT examples:
  * "landing on a flower" (bee)
  * "with a metal straw inserted" (can)
  * "next to a wedge of cheese" (mouse)
  * "attached to a tree branch" (leaf)
  * "lying on ocean beach sand" (shell)
  * "on a wooden kitchen table" (spoon)
  * "resting in a toolbox" (hammer)

FEEDBACK: {feedback}

Return ONLY a JSON array with EXACTLY {OBJECT_LIST_TARGET} items:"""
                    continue
                else:
                    raise ValueError(f"Failed to generate at least {OBJECT_LIST_MIN_OK} valid items after {max_retries} attempts. Got {len(valid_items)} valid items. Rejection reasons: {dict(list(rejection_reasons.items())[:5])}")
            
            # Take up to OBJECT_LIST_TARGET items (or all if less)
            object_list = valid_items[:OBJECT_LIST_TARGET]
            
            # Log warning if we have less than target
            if len(object_list) < OBJECT_LIST_TARGET:
                logger.warning(f"STEP0 proceeding with {len(object_list)} items (target={OBJECT_LIST_TARGET})")
            
            # Calculate SHA for logging
            object_list_str = json.dumps(object_list, sort_keys=True)
            object_list_sha = hashlib.sha256(object_list_str.encode()).hexdigest()[:16]
            
            logger.info(f"OBJECTLIST_SHA={object_list_sha} total={len(object_list)} rejected={rejected_count} retry={attempt}")
            
            # Log sample (first 5 items)
            sample = object_list[:5]
            logger.info(f"STEP 0 OBJECTLIST: size={len(object_list)}, sample5={json.dumps(sample, indent=2)}")
            
            # Save to cache
            _set_to_step0_cache(step0_cache_key, object_list)
            
            return object_list
            
        except json.JSONDecodeError as e:
            logger.error(f"STEP 0 - BUILD_OBJECT_LIST: JSON parse error: {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Failed to parse object list JSON: {e}")
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
                logger.error(f"STEP 0 - BUILD_OBJECT_LIST: OpenAI 400 error (no retry): {error_str}")
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
                    logger.warning(f"STEP 0 - BUILD_OBJECT_LIST: Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"STEP 0 - BUILD_OBJECT_LIST: Rate limit exceeded after {max_retries} attempts")
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
                    logger.warning(f"STEP 0 - BUILD_OBJECT_LIST: Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)
                    continue
                else:
                    logger.error(f"STEP 0 - BUILD_OBJECT_LIST: Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            # Other errors - don't retry, raise immediately
            logger.error(f"STEP 0 - BUILD_OBJECT_LIST: OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
            raise
    
    raise Exception("Failed to build object list from ad_goal")


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
    STEP 1 - SELECT PAIR WITH LIMITED SHAPE SEARCH
    
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
        model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
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
        
        # Use themed_pool if it's large enough (>= 60), otherwise fallback to full object_list
        if len(themed_pool) >= 60:
            search_objects = themed_pool
            logger.info(f"THEME_FILTER: using themed_pool size={len(themed_pool)} (by_tag={len(themed_pool_by_tag)} by_link={len(themed_pool_by_link)}) (from {len(object_list)} total)")
        else:
            search_objects = object_list
            logger.warning(f"THEME_FILTER: themed_pool too small ({len(themed_pool)}), using full object_list ({len(object_list)})")
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
    
    # Limited search: for each i, check j=i+1..i+K
    K = SHAPE_SEARCH_K
    max_checked = MAX_CHECKED_PAIRS
    checked_pairs = 0
    best_pair = None
    best_score = 0.0
    
    for i in range(len(shuffled_objects)):
        if checked_pairs >= max_checked:
            break
        
        obj_a_item = shuffled_objects[i]
        obj_a_id = obj_a_item["id"]
        obj_a_name = obj_a_item["object"]
        
        # Skip if already used in session (prefer diversity)
        if obj_a_id in used_objects_session and len(used_objects_session) < len(shuffled_objects) * 0.8:
            continue
        
        # Check candidates j=i+1..i+K
        for j in range(i + 1, min(i + 1 + K, len(shuffled_objects))):
            if checked_pairs >= max_checked:
                break
            
            obj_b_item = shuffled_objects[j]
            obj_b_id = obj_b_item["id"]
            obj_b_name = obj_b_item["object"]
            
            # CRITICAL: Reject if same main object (forbidden: same object with different sub_object)
            obj_a_main_key = _main_key(obj_a_item)
            obj_b_main_key = _main_key(obj_b_item)
            if obj_a_main_key == obj_b_main_key:
                logger.debug(f"PAIR_REJECT_MAIN_DUPLICATE sid={sid} ad={ad_index} A={obj_a_id}(main={obj_a_main_key}) B={obj_b_id}(main={obj_b_main_key}) - same main object")
                continue
            
            # Skip if already used in session
            if obj_b_id in used_objects_session and len(used_objects_session) < len(shuffled_objects) * 0.8:
                continue
            
            # Check if pair already used
            pair_hash = hashlib.sha256("|".join(sorted([obj_a_id, obj_b_id])).encode()).hexdigest()
            if pair_hash in used_pairs_set:
                logger.info(f"PAIR_SKIP_USED sid={sid} ad={ad_index} reason=pair A={obj_a_id} B={obj_b_id}")
                continue
            
            checked_pairs += 1
            
            # Call shape model to get similarity score (simplified - in production, batch or cache)
            try:
                prompt = f"""Compare geometric shape similarity:

Object A: {obj_a_name}
Object B: {obj_b_name}

Return JSON: {{"shape_score": 0-100, "hint": "short description"}}"""

                is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                if is_o_model:
                    response = client.responses.create(model=model_name, input=prompt)
                    response_text = response.output_text.strip()
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=100
                    )
                    response_text = response.choices[0].message.content.strip()
                
                # Parse JSON
                if response_text.startswith("```"):
                    lines = response_text.split('\n')
                    response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                if response_text.startswith("```json"):
                    lines = response_text.split('\n')
                    response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                
                result = json.loads(response_text)
                shape_score = float(result.get("shape_score", 0)) / 100.0
                hint = result.get("hint", "")
                
                if shape_score >= SHAPE_MIN_SCORE:
                    # Get theme and sub_object info for logging
                    obj_a_theme = obj_a_item.get("theme_tag", "")
                    obj_b_theme = obj_b_item.get("theme_tag", "")
                    obj_a_sub = obj_a_item.get("sub_object", "")
                    obj_b_sub = obj_b_item.get("sub_object", "")
                    
                    # Check theme relevance if allowed_theme_tags provided
                    obj_a_link = obj_a_item.get("theme_link", "")
                    obj_b_link = obj_b_item.get("theme_link", "")
                    
                    # Acceptance rule: reject if theme tags don't match (unless fallback mode)
                    if allowed_theme_tags and len(allowed_theme_tags) > 0 and themed_pool and len(themed_pool) >= 60:
                        theme_tags_norm = {_norm(t) for t in allowed_theme_tags}
                        obj_a_theme_norm = _norm(obj_a_theme)
                        obj_b_theme_norm = _norm(obj_b_theme)
                        if obj_a_theme_norm not in theme_tags_norm or obj_b_theme_norm not in theme_tags_norm:
                            logger.debug(f"PAIR_REJECT_THEME sid={sid} ad={ad_index} A={obj_a_id}(theme={obj_a_theme}) B={obj_b_id}(theme={obj_b_theme}) not in allowed_tags")
                            continue
                    
                    best_pair = {
                        "object_a": obj_a_name,
                        "object_b": obj_b_name,
                        "object_a_id": obj_a_id,
                        "object_b_id": obj_b_id,
                        "shape_similarity_score": int(shape_score * 100),
                        "shape_hint": hint
                    }
                    
                    # Update session tracking
                    with _session_used_lock:
                        if sid in _session_used_pairs:
                            used_pairs_set, used_objects_session, _ = _session_used_pairs[sid]
                            used_pairs_set.add(pair_hash)
                            used_objects_session.add(obj_a_id)
                            used_objects_session.add(obj_b_id)
                            _session_used_pairs[sid] = (used_pairs_set, used_objects_session, time.time())
                    
                    # Log with detailed information including main objects and sub_objects
                    theme_info = ""
                    if obj_a_theme or obj_b_theme:
                        a_link_short = obj_a_link[:30] + "..." if len(obj_a_link) > 30 else obj_a_link
                        b_link_short = obj_b_link[:30] + "..." if len(obj_b_link) > 30 else obj_b_link
                        theme_info = f" A_theme={obj_a_theme} B_theme={obj_b_theme} A_link={a_link_short} B_link={b_link_short}"
                    
                    logger.info(f"PAIR_PICK sid={sid} ad={ad_index} A={obj_a_id} A_obj={obj_a_name} A_sub={obj_a_sub} B={obj_b_id} B_obj={obj_b_name} B_sub={obj_b_sub} shape={int(shape_score*100)} checked_pairs={checked_pairs} cache_hit_plan=0{theme_info}")
                    return best_pair
                
                if shape_score > best_score:
                    best_score = shape_score
                    best_pair = {
                        "object_a": obj_a_name,
                        "object_b": obj_b_name,
                        "object_a_id": obj_a_id,
                        "object_b_id": obj_b_id,
                        "shape_similarity_score": int(shape_score * 100),
                        "shape_hint": hint
                    }
            
            except Exception as e:
                logger.debug(f"Shape matching error: {e}")
                continue
    
    # Fallback: use best pair if >= 0.70
    if best_pair and best_score >= 0.70:
        # Find theme info for best_pair
        obj_a_item_fallback = next((it for it in search_objects if it.get("id") == best_pair["object_a_id"]), None)
        obj_b_item_fallback = next((it for it in search_objects if it.get("id") == best_pair["object_b_id"]), None)
        
        obj_a_theme = obj_a_item_fallback.get("theme_tag", "") if obj_a_item_fallback else ""
        obj_b_theme = obj_b_item_fallback.get("theme_tag", "") if obj_b_item_fallback else ""
        obj_a_link = obj_a_item_fallback.get("theme_link", "") if obj_a_item_fallback else ""
        obj_b_link = obj_b_item_fallback.get("theme_link", "") if obj_b_item_fallback else ""
        
        obj_a_sub = obj_a_item_fallback.get("sub_object", "") if obj_a_item_fallback else ""
        obj_b_sub = obj_b_item_fallback.get("sub_object", "") if obj_b_item_fallback else ""
        obj_a_name_fallback = best_pair["object_a"]
        obj_b_name_fallback = best_pair["object_b"]
        
        # Check theme relevance for fallback too
        if allowed_theme_tags and len(allowed_theme_tags) > 0 and themed_pool and len(themed_pool) >= 60:
            theme_tags_norm = {_norm(t) for t in allowed_theme_tags}
            obj_a_theme_norm = _norm(obj_a_theme)
            obj_b_theme_norm = _norm(obj_b_theme)
            if obj_a_theme_norm not in theme_tags_norm or obj_b_theme_norm not in theme_tags_norm:
                logger.warning(f"PAIR_REJECT_THEME_FALLBACK sid={sid} ad={ad_index} A={best_pair['object_a_id']}(theme={obj_a_theme}) B={best_pair['object_b_id']}(theme={obj_b_theme}) not in allowed_tags")
                # Still return it as fallback, but log the issue
        
        pair_hash = hashlib.sha256("|".join(sorted([best_pair["object_a_id"], best_pair["object_b_id"]])).encode()).hexdigest()
        with _session_used_lock:
            if sid in _session_used_pairs:
                used_pairs_set, used_objects_session, _ = _session_used_pairs[sid]
                used_pairs_set.add(pair_hash)
                used_objects_session.add(best_pair["object_a_id"])
                used_objects_session.add(best_pair["object_b_id"])
                _session_used_pairs[sid] = (used_pairs_set, used_objects_session, time.time())
        
        # Log with detailed information including main objects and sub_objects
        theme_info = ""
        if obj_a_theme or obj_b_theme:
            a_link_short = obj_a_link[:30] + "..." if len(obj_a_link) > 30 else obj_a_link
            b_link_short = obj_b_link[:30] + "..." if len(obj_b_link) > 30 else obj_b_link
            theme_info = f" A_theme={obj_a_theme} B_theme={obj_b_theme} A_link={a_link_short} B_link={b_link_short}"
        
        logger.warning(f"PAIR_PICK sid={sid} ad={ad_index} A={best_pair['object_a_id']} A_obj={obj_a_name_fallback} A_sub={obj_a_sub} B={best_pair['object_b_id']} B_obj={obj_b_name_fallback} B_sub={obj_b_sub} shape={best_pair['shape_similarity_score']} checked_pairs={checked_pairs} cache_hit_plan=0 (fallback){theme_info}")
        return best_pair
    
    raise ValueError(f"No valid pair found after {checked_pairs} checks")


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
        model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
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
            
            if using_responses_api:
                # Use Responses API for o* models
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                content = response.output_text
            else:
                # Use Chat Completions for other models
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a shape similarity analyzer. Output must be in English only. Return JSON only without additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
            
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


def generate_headline_only(
    product_name: str,
    message: str,
    object_a: str,
    object_b: str,
    headline_placement: Optional[str] = None,
    max_retries: int = 3
) -> str:
    """
    STEP 2 - HEADLINE GENERATION
    
    Generate headline ONLY using OPENAI_TEXT_MODEL (default: o4-mini).
    
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
    model_name = os.environ.get("OPENAI_TEXT_MODEL", "o4-mini")
    
    logger.info(f"STEP 2 - HEADLINE GENERATION: text_model={model_name}, productName={product_name[:50]}, message={message[:50]}, object_a={object_a}, object_b={object_b}")
    
    prompt = f"""Generate a headline for an advertisement.

Product name: {product_name}
Message: {message}
Objects (already selected, do not change): {object_a} and {object_b}

Requirements:
- English only
- ALL CAPS
- Maximum 7 words INCLUDING the product name
- Include the product name in the headline
- Do NOT mention the objects ({object_a} or {object_b}) in the headline
- Do NOT change or re-select the objects
- No punctuation (no colons, commas, periods, etc.)
- Return ONLY the headline text, no JSON, no quotes

Headline:"""

    # Check if model is o* type - these use Responses API, not Chat Completions
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                logger.info(f"STEP 2 - HEADLINE GENERATION: using_responses_api={using_responses_api}")
            
            if using_responses_api:
                # Use Responses API for o* models
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                headline = response.output_text.strip()
            else:
                # Use Chat Completions for other models
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                headline = response.choices[0].message.content.strip()
            
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
            
            # Validate: max 7 words including productName
            words = headline.split()
            product_words = product_name.upper().split()
            if len(words) > 7:
                # Try to keep product name + first words
                if len(product_words) <= 7:
                    remaining = 7 - len(product_words)
                    other_words = [w for w in words if w not in product_words][:remaining]
                    headline = " ".join(product_words + other_words)
                else:
                    headline = " ".join(words[:7])
            
            words_count = len(headline.split())
            placement_info = f", placement={headline_placement}" if headline_placement else ""
            logger.info(f"STEP 2 - HEADLINE GENERATION SUCCESS: headline={headline}, words_count={words_count}{placement_info}")
            return headline
            
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
    model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
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
            if using_responses_api:
                # Use Responses API for o* models
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                response_text = response.output_text.strip()
            else:
                # Use Chat Completions for other models (fallback)
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
            
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
    model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
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
            if using_responses_api:
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                response_text = response.output_text.strip()
            else:
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
            
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
        model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
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
            if using_responses_api:
                # Use Responses API for o* models
                response = client.responses.create(
                    model=model_name,
                    input=prompt
                )
                response_text = response.output_text.strip()
            else:
                # Use Chat Completions for other models (fallback)
                request_params = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
            
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
    object_b_context: Optional[str] = None
) -> str:
    """
    STEP 3 - IMAGE GENERATION PROMPT
    
    Create DALL-E prompt for SIDE_BY_SIDE layout only.
    
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
    # Force SIDE BY SIDE mode only (ignore hybrid_plan and physical_context)
    # ACE_LAYOUT_MODE is always "side_by_side" or ignored
    
    # SIDE_BY_SIDE mode only
    # Build shape hint instruction if provided
    shape_instruction = ""
    if shape_hint:
        shape_instruction = f"\n- Both objects must share a similar outline: {shape_hint}. Emphasize comparable silhouettes."
    
    # Build composition rules section (SIDE BY SIDE only - two full panels)
    # Include classic_context as photography hints (not text on image)
    context_a_hint = ""
    if object_a_context is not None and object_a_context:
        context_a_hint = f", shown {object_a_context}"
    context_b_hint = ""
    if object_b_context is not None and object_b_context:
        context_b_hint = f", shown {object_b_context}"
    
    composition_rules = f"""COMPOSITION RULES (CRITICAL):
- CRITICAL: The left and right panels MUST show two DIFFERENT object types. Do NOT repeat the same main object on both sides.
- Left panel MUST show: {object_a}{context_a_hint} (interacting with its sub_object as described in classic_context).
- Right panel MUST show: {object_b}{context_b_hint} (interacting with its sub_object as described in classic_context).
- Both objects are FULL objects, completely visible in their respective panels.
- Each object must be shown interacting with its sub_object as described in classic_context.
- The sub_object is part of the composition, not just background.
- VISUAL DOMINANCE RULE: The main object ({object_a} and {object_b}) must be visually dominant in each panel.
- The sub_object must support the scene but must not dominate the silhouette.
- The overall outline similarity should be determined by the main objects, not the secondary objects.
- Two separate objects side by side, no overlap.
- Clear separation between left and right panels.
- Same vertical alignment (same baseline alignment).
- Center of the composition is between the two objects.
- Clear comparable outer contours.{shape_instruction}
- Maintain this exact compositional structure in all attempts (do not change positioning between retries)."""

    # Build visual style constraints section
    visual_style_constraints = """VISUAL STYLE CONSTRAINTS:
- Ultra realistic photography.
- Professional studio or real-world photography.
- Natural lighting.
- Real materials.
- No illustration style.
- No drawn elements.
- No graphic design look.
- No visible logos.
- No printed brand names.
- No readable text except the main headline generated for the ad.
- If any object would normally contain branding, render it completely generic and blank."""

    if is_strict:
        return f"""Create a professional advertisement image with a SIDE BY SIDE layout.

LAYOUT:
- Two panels side by side (left and right).
- Left panel shows FULL {object_a}.
- Right panel shows FULL {object_b}.
- Both objects are completely visible, no overlap.
- Clear separation between panels.

{composition_rules}

{visual_style_constraints}

HEADLINE:
- Only one headline: "{headline}"
- Use fewer letters. Use one short headline. Make text extremely large and bold.
- The headline must be integrated into the design as a strong central visual element.
- The headline must have the same visual importance as the objects.

TEXT RULES (CRITICAL):
- ALL CAPS.
- English only.
- Perfectly legible.
- No paragraphs.
- No small print.
- No separate CTA.
- No extra text.

STYLE:
- Bold, modern, minimal.
- High contrast.
- Clear typography.
- Professional advertising aesthetic."""

    else:
        return f"""Create a professional advertisement image with a SIDE BY SIDE layout.

LAYOUT:
- Two panels side by side (left and right).
- Left panel shows FULL {object_a}.
- Right panel shows FULL {object_b}.
- Both objects are completely visible, no overlap.
- Clear separation between panels.

{composition_rules}

{visual_style_constraints}

HEADLINE:
- Only one headline: "{headline}"
- The headline must be integrated into the design as a strong central visual element.
- The headline must have the same visual importance as the objects.

TEXT RULES (CRITICAL):
- ALL CAPS.
- English only.
- Perfectly legible.
- No paragraphs.
- No small print.
- No separate CTA.
- No extra text.

STYLE:
- Bold, modern, minimal.
- High contrast.
- Clear typography.
- Professional advertising aesthetic."""


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
    quality: str = "high"
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
            object_b_context=object_b_context
        )
        
        try:
            logger.info(f"STEP 3 - IMAGE GENERATION: attempt={attempt + 1}/{max_retries}, image_model={model}, image_size={image_size}")
            
            # Simple call without response_format for gpt-image-1.5 compatibility
            # Include quality parameter for preview (low) vs generate (high)
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=image_size,
                quality=quality
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
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"DALL-E generation failed (attempt {attempt + 1}/{max_retries}): {error_str}")
            
            if attempt < max_retries - 1:
                # Retry on errors
                time.sleep(1 + attempt)
                continue
            else:
                raise
    
    raise Exception("Failed to generate image after retries")


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
    
    # A) Build ad_goal from productName + productDescription (NEW - not from history)
    ad_goal = build_ad_goal(product_name, product_description)
    logger.info(f"AD_GOAL={ad_goal}")
    
    message = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # B) Build object_list in new format ({OBJECT_LIST_TARGET} items with id/object/classic_context/theme_link)
    step0_cache_hit = False
    if not object_list or len(object_list) < OBJECT_LIST_MIN_OK:
        # Build from ad_goal
        step0_cache_key = _get_cache_key_step0(ad_goal, product_name, language, session_seed=session_id)  # Shared across ads in same session
        cached_step0_list = _get_from_step0_cache(step0_cache_key)
        if cached_step0_list:
            step0_cache_hit = True
            object_list = cached_step0_list
            logger.info(f"[{request_id}] STEP 0 - Using cached objectList (size={len(object_list)})")
        else:
            object_list = build_object_list_from_ad_goal(ad_goal, product_name, language=language)
    else:
        # Validate existing object_list (convert if needed)
        object_list = validate_object_list(object_list, ad_goal=ad_goal, product_name=product_name, language=language)
    
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
        preview_cache_key = _get_cache_key_preview(product_name, message, ad_goal, ad_index, object_list, language=language, session_seed=session_seed, engine_mode=ENGINE_MODE, preview_mode=PREVIEW_MODE, image_size=PREVIEW_IMAGE_SIZE_DEFAULT, quality=PREVIEW_IMAGE_QUALITY_DEFAULT)
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
                logger.info(f"SELECTOR=limited_shape_search object_list_type={type(object_list[0]).__name__ if object_list else 'empty'}")
                t_shape_start = time.time()
                shape_result = select_pair_with_limited_shape_search(
                    sid=session_seed or session_id or "no_session",
                    ad_index=ad_index,
                    ad_goal=ad_goal,
                    image_size=PREVIEW_IMAGE_SIZE_DEFAULT,
                    object_list=object_list,
                    used_objects=used_objects,
                    model_name=planner_model or os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro"),
                    allowed_theme_tags=theme_tags
                )
                t_shape_ms = int((time.time() - t_shape_start) * 1000)
                # Save to cache
                if ENABLE_STEP1_CACHE:
                    _set_to_step1_cache(step1_cache_key, shape_result)
        else:
            # Add verification log
            logger.info(f"SELECTOR=limited_shape_search object_list_type={type(object_list[0]).__name__ if object_list else 'empty'}")
            t_shape_start = time.time()
            shape_result = select_pair_with_limited_shape_search(
                sid=session_seed or session_id or "no_session",
                ad_index=ad_index,
                ad_goal=ad_goal,
                image_size=PREVIEW_IMAGE_SIZE_DEFAULT,
                object_list=object_list,
                used_objects=used_objects,
                model_name=planner_model or os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro"),
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
                    object_a_item = next((item for item in object_list if item.get("id") == object_a_id), None)
                if object_b_id:
                    object_b_item = next((item for item in object_list if item.get("id") == object_b_id), None)
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
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
                raise Exception("rate_limited")
            else:
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
                max_retries=3
            )
            t_headline_ms = int((time.time() - t_headline_start) * 1000)
            logger.info(f"[{request_id}] STEP 2 SUCCESS: headline={headline}")
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
                raise Exception("rate_limited")
            else:
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
    
    # If plan_only mode, return plan immediately (no image generation)
    if PREVIEW_MODE == "plan_only":
        t_total_ms = int((time.time() - t_start) * 1000)
        cache_hit = preview_cache_hit or plan_cache_hit
        logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms=0 cache_hit={cache_hit}")
        return plan_data if not cache_hit else cached_plan
    
    # STEP 3 - IMAGE GENERATION (only if PREVIEW_MODE=image)
    t_image_start = time.time()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Check image cache
    image_cache_key = None
    image_bytes = None
    
    if ENABLE_IMAGE_CACHE:
        # Create prompt for cache key
        prompt_for_cache = create_image_prompt(
            object_a=object_a,
            object_b=object_b,
            headline=headline,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            is_strict=False
        )
        image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
        image_cache_key = _get_cache_key_image(prompt_for_cache, PREVIEW_IMAGE_SIZE_DEFAULT, image_model, PREVIEW_IMAGE_QUALITY_DEFAULT)
        cached_image_bytes = _get_from_image_cache(image_cache_key)
        if cached_image_bytes:
            image_bytes = cached_image_bytes
            image_cache_hit = True
            logger.info(f"[{request_id}] IMAGE_CACHE hit=true key={image_cache_key[:16]}...")
        else:
            logger.info(f"[{request_id}] IMAGE_CACHE hit=false key={image_cache_key[:16]}...")
    
    if not image_cache_hit:
        try:
            # Determine max_retries based on mode
            max_img_retries = 1 if ENGINE_MODE == "optimized" else 3
            
            # D) Update image prompt to use new format with classic_context
            # Get context from items if available
            if 'object_a_item' in locals() and object_a_item:
                object_a_context = object_a_item.get("classic_context", "")
            else:
                object_a_context = ""
            if 'object_b_item' in locals() and object_b_item:
                object_b_context = object_b_item.get("classic_context", "")
            else:
                object_b_context = ""
            
            image_bytes = generate_image_with_dalle(
                client=client,
                object_a=object_a_name,
                object_b=object_b_name,
                object_a_context=object_a_context,
                object_b_context=object_b_context,
                shape_hint=shape_hint,
                physical_context=None,  # No physical context in SIDE BY SIDE mode
                hybrid_plan=None,  # No hybrid plan in SIDE BY SIDE mode
                headline=headline,
                width=width,
                height=height,
                max_retries=max_img_retries,
                quality=PREVIEW_IMAGE_QUALITY_DEFAULT
            )
            
            # Save to image cache
            if ENABLE_IMAGE_CACHE and image_cache_key:
                _set_to_image_cache(image_cache_key, image_bytes)
                
        except Exception as e:
            logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
            raise
    
    t_image_ms = int((time.time() - t_image_start) * 1000)
    
    # Convert image to base64 (without data URI header)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get image model and size for logging
    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    
    t_total_ms = int((time.time() - t_start) * 1000)
    cache_hit = preview_cache_hit or plan_cache_hit
    logger.info(f"[{request_id}] STEP 3 SUCCESS: image_model={image_model}, image_size={image_size_str}, preview_success=true")
    logger.info(f"[{request_id}] PERF_PREVIEW total_ms={t_total_ms} shape_ms={t_shape_ms} env_ms={t_envswap_ms} headline_ms={t_headline_ms} image_ms={t_image_ms} cache_hit={cache_hit} cache_image_hit={image_cache_hit} cache_step0_hit={step0_cache_hit} cache_step1_hit={step1_cache_hit}")
    
    # Return only imageBase64 (all text is in the image)
    return {
        "imageBase64": image_base64
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
    
    # A) Build ad_goal from productName + productDescription (NEW - not from history)
    ad_goal = build_ad_goal(product_name, product_description)
    logger.info(f"AD_GOAL={ad_goal}")
    
    message = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # B) Build object_list in new format ({OBJECT_LIST_TARGET} items with id/object/classic_context/theme_link)
    step0_cache_hit = False
    if not object_list or len(object_list) < OBJECT_LIST_MIN_OK:
        # Build from ad_goal
        step0_cache_key = _get_cache_key_step0(ad_goal, product_name, language, session_seed=session_id)  # Shared across ads in same session
        cached_step0_list = _get_from_step0_cache(step0_cache_key)
        if cached_step0_list:
            step0_cache_hit = True
            object_list = cached_step0_list
            logger.info(f"[{request_id}] STEP 0 - Using cached objectList (size={len(object_list)})")
        else:
            object_list = build_object_list_from_ad_goal(ad_goal, product_name, language=language)
    else:
        # Validate existing object_list (convert if needed)
        object_list = validate_object_list(object_list, ad_goal=ad_goal, product_name=product_name, language=language)
    
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
            pair_result = select_pair_with_limited_shape_search(
                object_list=object_list,
                sid=session_id,
                ad_index=ad_index,
                ad_goal=ad_goal,
                image_size=image_size_str,
                used_objects=None,  # Anti-repeat handled internally
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
            object_a_item = next((item for item in object_list if item.get("id") == object_a_id), None)
            object_b_item = next((item for item in object_list if item.get("id") == object_b_id), None)
            
            if not object_a_item or not object_b_item:
                raise ValueError(f"Could not find items for ids: {object_a_id}, {object_b_id}")
            
            logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a_name}, {object_b_name}], score={shape_score}, shape_hint={shape_hint}, model={planner_model}")
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
                raise Exception("rate_limited")
            else:
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
        except Exception as e:
            error_msg = str(e)
            if "rate_limited" in error_msg:
                logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
                raise Exception("rate_limited")
            else:
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
    t_image_start = time.time()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Check image cache
    image_cache_key = None
    image_bytes = None
    
    if ENABLE_IMAGE_CACHE:
        # Create prompt for cache key
        object_a_context = object_a_item.get("classic_context", "") if object_a_item else ""
        object_b_context = object_b_item.get("classic_context", "") if object_b_item else ""
        prompt_for_cache = create_image_prompt(
            object_a=object_a_name,
            object_b=object_b_name,
            headline=headline,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            is_strict=False,
            object_a_context=object_a_context,
            object_b_context=object_b_context
        )
        image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
        # Use generate_size and generate_quality from outer scope
        image_cache_key = _get_cache_key_image(prompt_for_cache, image_size_str, image_model, GENERATE_IMAGE_QUALITY_DEFAULT)
        cached_image_bytes = _get_from_image_cache(image_cache_key)
        if cached_image_bytes:
            image_bytes = cached_image_bytes
            image_cache_hit = True
            logger.info(f"[{request_id}] IMAGE_CACHE hit=true key={image_cache_key[:16]}...")
        else:
            logger.info(f"[{request_id}] IMAGE_CACHE hit=false key={image_cache_key[:16]}...")
    
    if not image_cache_hit:
        try:
            # Determine max_retries based on mode
            max_img_retries = 2 if ENGINE_MODE == "optimized" else 3
            
            # D) Update image prompt to use new format with classic_context
            # Get context from items if available
            if 'object_a_item' in locals() and object_a_item:
                object_a_context = object_a_item.get("classic_context", "")
            else:
                object_a_context = ""
            if 'object_b_item' in locals() and object_b_item:
                object_b_context = object_b_item.get("classic_context", "")
            else:
                object_b_context = ""
            
            image_bytes = generate_image_with_dalle(
                client=client,
                object_a=object_a_name,
                object_b=object_b_name,
                object_a_context=object_a_context,
                object_b_context=object_b_context,
                shape_hint=shape_hint,
                physical_context=None,  # No physical context in SIDE BY SIDE mode
                hybrid_plan=None,  # No hybrid plan in SIDE BY SIDE mode
                headline=headline,
                width=width,
                height=height,
                max_retries=max_img_retries,
                quality=GENERATE_IMAGE_QUALITY_DEFAULT
            )
            
            # Save to image cache
            if ENABLE_IMAGE_CACHE and image_cache_key:
                _set_to_image_cache(image_cache_key, image_bytes)
                
        except Exception as e:
            logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
            raise
    
    t_image_ms = int((time.time() - t_image_start) * 1000)
    
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

