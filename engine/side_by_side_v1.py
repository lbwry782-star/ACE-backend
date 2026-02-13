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
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

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
    max_retries: int = 2
) -> List[str]:
    """
    STEP 0 - BUILD_OBJECT_LIST_FROM_AD_GOAL
    
    Build a list of 200 concrete visual objects related to ad_goal.
    
    Args:
        ad_goal: The advertising goal (e.g., "protect nature", "climate action")
        product_name: Optional product name for context
        max_retries: Maximum retry attempts
    
    Returns:
        List[str]: List of 200 concrete nouns (visual objects)
    
    Rules:
    - Each item is a concrete noun that can be drawn (visual object)
    - Must be directly related to ad_goal
    - NO generic shapes: circle, cylinder, square, triangle, sphere, etc.
    - English only, single words or short noun phrases
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = os.environ.get("OPENAI_TEXT_MODEL", "o4-mini")
    
    # Build prompt
    product_context = f"\nProduct name (optional context): {product_name}" if product_name else ""
    
    prompt = f"""Generate a list of 200 concrete visual objects (nouns) that are directly related to this advertising goal.

Advertising goal: {ad_goal}{product_context}

Requirements:
- Each item must be a concrete noun that can be drawn/visualized (visual object)
- All items must be directly related to the advertising goal
- English only
- Single words or short noun phrases (max 2-3 words)
- NO generic geometric shapes: circle, cylinder, square, triangle, sphere, cube, rectangle, oval, etc.
- NO abstract concepts
- Focus on tangible, drawable objects

Return ONLY a JSON object with this exact format:
{{"objectList": ["item1", "item2", ..., "item200"]}}

JSON:"""

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
            
            # Parse JSON
            data = json.loads(response_text)
            object_list = data.get("objectList", [])
            
            # Validate: must have at least 120 items
            if len(object_list) < 120:
                if attempt < max_retries - 1:
                    logger.warning(f"STEP 0 - BUILD_OBJECT_LIST: Only {len(object_list)} items returned (need >=120), retrying with stricter prompt...")
                    # Stricter prompt for retry
                    prompt = f"""Generate a list of 200 concrete visual objects (nouns) that are directly related to this advertising goal.

Advertising goal: {ad_goal}{product_context}

CRITICAL REQUIREMENTS:
- Return EXACTLY 200 items (no fewer)
- Each item must be a concrete noun that can be drawn/visualized
- All items must be directly related to the advertising goal
- English only
- Single words or short noun phrases (max 2-3 words)
- NO generic geometric shapes: circle, cylinder, square, triangle, sphere, cube, rectangle, oval, etc.
- NO abstract concepts
- Focus on tangible, drawable objects

Return ONLY a JSON object with this exact format:
{{"objectList": ["item1", "item2", ..., "item200"]}}

JSON:"""
                    continue
                else:
                    logger.error(f"STEP 0 - BUILD_OBJECT_LIST: Only {len(object_list)} items returned after {max_retries} attempts (need >=120)")
                    raise ValueError(f"Failed to generate sufficient object list: got {len(object_list)} items, need >=120")
            
            # Filter out generic shapes
            generic_shapes = {"circle", "cylinder", "square", "triangle", "sphere", "cube", "rectangle", "oval", "cone", "pyramid", "hexagon", "pentagon", "octagon", "diamond", "trapezoid"}
            filtered_list = [obj for obj in object_list if obj.lower().strip() not in generic_shapes]
            
            # Log sample
            sample_size = min(10, len(filtered_list))
            sample = filtered_list[:sample_size]
            logger.info(f"STEP 0 OBJECTLIST: size={len(filtered_list)}, sample10={sample}")
            
            return filtered_list
            
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


def validate_object_list(object_list: Optional[List[str]], ad_goal: Optional[str] = None, product_name: Optional[str] = None) -> List[str]:
    """
    Validate and return object list.
    If None or too small, and ad_goal is provided, use STEP 0 to build list.
    Otherwise, return default concrete objects list.
    """
    if not object_list or len(object_list) < 2:
        # If ad_goal is provided, use STEP 0 to build list
        if ad_goal:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), building from ad_goal using STEP 0")
            return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name)
        else:
            logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), using default concrete objects list (size={len(DEFAULT_OBJECT_LIST)})")
            return DEFAULT_OBJECT_LIST
    
    # If object_list is provided but small (<120), and ad_goal is provided, use STEP 0
    if len(object_list) < 120 and ad_goal:
        logger.info(f"objectList too small (size={len(object_list)}), building from ad_goal using STEP 0")
        return build_object_list_from_ad_goal(ad_goal=ad_goal, product_name=product_name)
    
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


def select_similar_pair_shape_only(
    object_list: List[str],
    used_objects: set,
    max_retries: int = 2
) -> Dict:
    """
    STEP 1 - SHAPE SELECTION (ONLY SHAPE)
    
    Select two objects based ONLY on geometric shape similarity.
    Uses OPENAI_SHAPE_MODEL (default: o3-pro).
    
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
    model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
    # Filter out used objects
    available_objects = [obj for obj in object_list if obj not in used_objects]
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
        
        # Request list of candidates (12 pairs, minimum 5)
        prompt = f"""{system_instruction}Task: Find pairs of items from the provided list with similar geometric shapes (outer contour/outline).

ONLY criterion: geometric shape similarity of the objects' outer contour (outline).

Ignore meaning, category, theme, symbolism, relevance, marketing, color, material, texture.

Return EXACT strings from the list (no synonyms, no new words).

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

Available object list:
{json.dumps(available_objects, ensure_ascii=False, indent=2)}

Output JSON only with a list of candidate pairs:
{{
  "candidates": [
    {{"a": "OBJECT_1", "b": "OBJECT_2", "score": 0-100, "hint": "short shape hint"}},
    {{"a": "OBJECT_3", "b": "OBJECT_4", "score": 0-100, "hint": "short shape hint"}},
    ...
  ]
}}

Requirements:
- Return 12 candidate pairs if possible, minimum 5
- Each a/b must be EXACT match from the object list
- Score based ONLY on shape similarity (outer contour/outline)
- No meaning, no concept, only geometric shape
- Exclude objects with visible text or branding"""
        
        if is_strict:
            prompt = f"""{system_instruction}Task: Find pairs of items from the provided list with similar geometric shapes (outer contour/outline).

ONLY criterion: geometric shape similarity of the objects' outer contour (outline).

Ignore meaning, category, theme, symbolism, relevance, marketing, color, material, texture.

Return exact strings from the list only. Return EXACT strings from the list (no synonyms, no new words).

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

Available object list:
{json.dumps(available_objects, ensure_ascii=False, indent=2)}

Output JSON only with a list of candidate pairs:
{{
  "candidates": [
    {{"a": "OBJECT_1", "b": "OBJECT_2", "score": 0-100, "hint": "short shape hint"}},
    {{"a": "OBJECT_3", "b": "OBJECT_4", "score": 0-100, "hint": "short shape hint"}},
    ...
  ]
}}

Requirements:
- Return 12 candidate pairs if possible, minimum 5
- Each a/b must be EXACT match from the object list
- Score based ONLY on shape similarity (outer contour/outline)
- No meaning, no concept, only geometric shape"""
        
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
                
                if obj_a not in object_list or obj_b not in object_list:
                    continue
                
                if obj_a == obj_b:
                    continue
                
                # Normalize score
                score = c.get("score") or c.get("shape_similarity_score", 0)
                if not isinstance(score, (int, float)):
                    try:
                        score = int(score)
                    except:
                        score = 0
                
                hint = c.get("hint") or c.get("shape_hint", "")
                
                valid_candidates.append({
                    "a": obj_a,
                    "b": obj_b,
                    "score": score,
                    "hint": hint
                })
            
            if len(valid_candidates) < 5:
                logger.warning(f"Too few valid candidates after filtering: {len(valid_candidates)}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with stricter instruction (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    raise ValueError(f"Too few valid candidates after filtering: {len(valid_candidates)}")
            
            # Sort by score descending
            valid_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Calculate similar_pairs_found (score >= 80)
            similar_pairs_found = sum(1 for c in valid_candidates if c["score"] >= 80)
            
            # Select best pair (highest score)
            best_pair = valid_candidates[0]
            object_a = best_pair["a"]
            object_b = best_pair["b"]
            score = best_pair["score"]
            shape_hint = best_pair["hint"]
            
            # Log summary
            logger.info(f"STEP 1 SHAPE_MATCH summary: objectList_size={len(object_list)}, candidates_returned={len(candidates)}, candidates_valid={len(valid_candidates)}, similar_pairs_found(score>=80)={similar_pairs_found}, best_pair=\"{object_a} ~ {object_b}\" score={score} hint=\"{shape_hint}\"")
            
            # Log top 5 (max 10 lines to avoid flooding)
            top5 = valid_candidates[:5]
            top5_str = " | ".join([f"{i+1}) {c['a']}~{c['b']} score={c['score']} hint={c['hint']}" for i, c in enumerate(top5)])
            logger.info(f"STEP 1 SHAPE_MATCH top5: {top5_str}")
            
            # Return result in expected format
            result = {
                "object_a": object_a,
                "object_b": object_b,
                "shape_similarity_score": score,
                "shape_hint": shape_hint,
                "why": f"Selected from {len(valid_candidates)} valid candidates, {similar_pairs_found} with score>=80",
                "_similar_pairs_found": similar_pairs_found  # Internal variable for future use
            }
            
            logger.info(f"STEP 1 - SHAPE SELECTION SUCCESS: selected_pair=[{object_a}, {object_b}], score={score}, shape_hint={shape_hint}, validation_passed=true")
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


def generate_hybrid_context_plan(
    object_a: str,
    object_b: str,
    ad_goal: str,
    message: str,
    image_size: str,
    max_retries: int = 2
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
    
    Returns:
        dict: {
            "mode": "HYBRID_SINGLE_OBJECT",
            "hero_object": "object_a" | "object_b",
            "context_object": "object_a" | "object_b",
            "context_mechanic": "attached|inserted|replaced|held|growing_from|wrapped_by|filled_with",
            "context_description": str (max 10 words),
            "do_not_show_full_context_object": true,
            "headline_placement": "BOTTOM" | "SIDE"
        }
    
    Rules:
    - Show ONLY hero_object as full object
    - context_object appears only as physical context/mechanism, NOT as full object
    - Physical and tangible: connected/touching/stuck/replaced/wrapped/filled
    - No decorative background, no extra elements
    - English only
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = os.environ.get("OPENAI_SHAPE_MODEL", "o3-pro")
    
    prompt = f"""You are planning a HYBRID_SINGLE_OBJECT advertisement composition.

Objects:
- Object A: {object_a}
- Object B: {object_b}
- Advertising goal: {ad_goal}
- Message: {message}
- Image size: {image_size}

Task: Show ONE object in the PHYSICAL CONTEXT of the other object.

GLOBAL VISUAL RULES (MANDATORY):
- Do NOT select objects that inherently contain printed text or branding.
- Avoid packaging, labels, posters, signs, billboards.
- Only physical objects without visible text surfaces.
- Must be photographable in real life.
- Exception: objects where text is an integral structural part (e.g., playing cards, compass dial, measuring scale) are allowed.

Rules:
- In the image, we see ONLY the hero_object as a full object.
- The context_object does NOT appear as a full object in any situation.
- The context must be physical and tangible: connected/touching/stuck/replaced/wrapped/filled.
- No decorative background, no extra elements.
- English only.

Examples:
- "A soda can WITH a can-opener lodged in the lid" (inserted)
- "A globe WHERE the sphere is a real Earth photo texture" (replaced)
- "A tree GROWING FROM soil with visible roots" (growing_from)
- "A plastic bottle WRAPPED BY a leaf label" (wrapped_by)

Return JSON only:

{{
  "mode": "HYBRID_SINGLE_OBJECT",
  "hero_object": "object_a" | "object_b",
  "context_object": "object_a" | "object_b",
  "context_mechanic": "attached" | "inserted" | "replaced" | "held" | "growing_from" | "wrapped_by" | "filled_with",
  "context_description": "short concrete physical description (max 10 words)",
  "do_not_show_full_context_object": true,
  "headline_placement": "BOTTOM" | "SIDE"
}}

Rules:
- hero_object and context_object must be different (one is object_a, the other is object_b).
- context_description must be concrete and physical (max 10 words).
- headline_placement: BOTTOM for vertical/portrait, SIDE for landscape/wide images.

JSON:"""

    # Check if model is o* type - these use Responses API
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    using_responses_api = is_o_model
    
    logger.info(f"STEP 1.75 - HYBRID CONTEXT PLAN: shape_model={model_name}, object_a={object_a}, object_b={object_b}, ad_goal={ad_goal[:50]}, using_responses_api={using_responses_api}")
    
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
            if data.get("mode") != "HYBRID_SINGLE_OBJECT":
                raise ValueError("Mode must be HYBRID_SINGLE_OBJECT")
            if data.get("hero_object") not in ["object_a", "object_b"]:
                raise ValueError("hero_object must be 'object_a' or 'object_b'")
            if data.get("context_object") not in ["object_a", "object_b"]:
                raise ValueError("context_object must be 'object_a' or 'object_b'")
            if data.get("hero_object") == data.get("context_object"):
                raise ValueError("hero_object and context_object must be different")
            
            valid_mechanics = {"attached", "inserted", "replaced", "held", "growing_from", "wrapped_by", "filled_with"}
            if data.get("context_mechanic") not in valid_mechanics:
                raise ValueError(f"Invalid context_mechanic: {data.get('context_mechanic')}")
            
            if data.get("headline_placement") not in ["BOTTOM", "SIDE"]:
                raise ValueError("headline_placement must be 'BOTTOM' or 'SIDE'")
            
            # Determine actual object names
            hero_name = object_a if data["hero_object"] == "object_a" else object_b
            context_name = object_a if data["context_object"] == "object_a" else object_b
            
            logger.info(f"STEP 1.75 - HYBRID CONTEXT PLAN SUCCESS: hero={hero_name}, context={context_name}, mechanic={data.get('context_mechanic')}, desc=\"{data.get('context_description', '')}\", placement={data.get('headline_placement')}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"STEP 1.75 - HYBRID CONTEXT PLAN: JSON parse error: {e}")
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Failed to parse hybrid context plan JSON: {e}")
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
                logger.error(f"STEP 1.75 - HYBRID CONTEXT PLAN: OpenAI 400 error (no retry): {error_str}")
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
                    logger.warning(f"STEP 1.75 - HYBRID CONTEXT PLAN: Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"STEP 1.75 - HYBRID CONTEXT PLAN: Rate limit exceeded after {max_retries} attempts")
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
                    logger.warning(f"STEP 1.75 - HYBRID CONTEXT PLAN: Server/connection error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                    time.sleep(1 + attempt)
                    continue
                else:
                    logger.error(f"STEP 1.75 - HYBRID CONTEXT PLAN: Server/connection error after {max_retries} attempts: {error_str}")
                    raise
            
            # Other errors - don't retry, raise immediately
            logger.error(f"STEP 1.75 - HYBRID CONTEXT PLAN: OpenAI call failed (non-retryable, attempt {attempt + 1}): {error_str}")
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
    is_strict: bool = False
) -> str:
    """
    STEP 3 - IMAGE GENERATION PROMPT
    
    Create DALL-E prompt for SIDE_BY_SIDE or HYBRID_SINGLE_OBJECT image with text.
    
    Args:
        object_a: First object (from STEP 1)
        object_b: Second object (from STEP 1)
        headline: Headline from STEP 2 (ALL CAPS, max 7 words)
        shape_hint: Shape hint from STEP 1 (e.g., "tall-vertical", "round-flat"), optional
        physical_context: Physical context extensions (for SIDE_BY_SIDE), optional
        hybrid_plan: Hybrid context plan (for HYBRID_SINGLE_OBJECT), optional
        is_strict: If True, use stricter prompt for retry
    """
    # Check if we're in HYBRID_SINGLE_OBJECT mode
    if hybrid_plan and hybrid_plan.get("mode") == "HYBRID_SINGLE_OBJECT":
        # HYBRID_SINGLE_OBJECT mode
        hero_object = object_a if hybrid_plan["hero_object"] == "object_a" else object_b
        context_object = object_a if hybrid_plan["context_object"] == "object_a" else object_b
        context_description = hybrid_plan.get("context_description", "")
        context_mechanic = hybrid_plan.get("context_mechanic", "")
        headline_placement = hybrid_plan.get("headline_placement", "BOTTOM")
        
        # Build headline placement instruction
        if headline_placement == "BOTTOM":
            headline_instruction = "Place the headline under the object on clean space."
        else:  # SIDE
            headline_instruction = "Place the headline beside the object on clean space (landscape layout)."
        
        return f"""Create a professional advertisement image in HYBRID_SINGLE_OBJECT mode.

COMPOSITION:
- Show ONLY the hero object: {hero_object}
- Show it IN the physical context of {context_object}: {context_description}
- Do NOT show {context_object} as a full object—only the physical context/mechanism.
- The context mechanic is: {context_mechanic}
- Minimal background, clean, physically realistic.

VISUAL STYLE CONSTRAINTS:
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
- If any object would normally contain branding, render it completely generic and blank.

HEADLINE:
- Only one headline: "{headline}"
- {headline_instruction}
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
    
    # SIDE_BY_SIDE mode (existing logic)
    # Build shape hint instruction if provided
    shape_instruction = ""
    if shape_hint:
        shape_instruction = f"\n- Both objects must share a similar outline: {shape_hint}. Emphasize comparable silhouettes."
    
    # Build physical context extensions section
    physical_context_section = ""
    if physical_context:
        ext_a = physical_context.get("physical_extension_a", {})
        ext_b = physical_context.get("physical_extension_b", {})
        if ext_a and ext_b:
            physical_context_section = f"""
PHYSICAL CONTEXT EXTENSIONS:
- Object A ({object_a}): {ext_a.get("description", "")} (connection: {ext_a.get("connection_type", "")})
- Object B ({object_b}): {ext_b.get("description", "")} (connection: {ext_b.get("connection_type", "")})
- These extensions are physically connected to their objects and explain function/origin.
- Extensions must stay attached and NOT cross the central gap between objects."""
    
    # Build composition rules section
    composition_rules = f"""COMPOSITION RULES (CRITICAL):
- Place both objects ({object_a} and {object_b}) extremely close together with minimal space between them.
- The objects must be almost touching with a very small gap only.
- No overlap between the objects.
- Same vertical alignment (same baseline alignment).
- Center of the composition is between the two objects.
- Clear comparable outer contours.{shape_instruction}
- Physical context extensions (if any) must:
  * Stay attached to each object.
  * NOT extend toward the center beyond the main shape boundary.
  * NOT create overlap with the other object.
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

OBJECTS:
- Left object: {object_a}
- Right object: {object_b}
{physical_context_section}

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

OBJECTS:
- Left object: {object_a}
- Right object: {object_b}
{physical_context_section}

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
    max_retries: int = 3
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
    
    # Log before image generation
    image_prompt_includes_shape_hint = shape_hint is not None and shape_hint != ""
    mode = "HYBRID_SINGLE_OBJECT" if hybrid_plan and hybrid_plan.get("mode") == "HYBRID_SINGLE_OBJECT" else "SIDE_BY_SIDE"
    logger.info(f"STEP 3 - IMAGE GENERATION: image_model={model}, image_size={image_size}, object_a={object_a}, object_b={object_b}, headline={headline}, image_prompt_includes_shape_hint={image_prompt_includes_shape_hint}, shape_hint=\"{shape_hint or ''}\", mode={mode}")
    logger.info(f"STEP 3 VISUAL_RULES_APPLIED: no_logos=true, photorealistic_only=true")
    if mode == "SIDE_BY_SIDE":
        logger.info(f"STEP 3 COMPOSITION: near_touching=true, overlap=false_expected")
    else:
        hero_name = object_a if hybrid_plan["hero_object"] == "object_a" else object_b
        context_name = object_a if hybrid_plan["context_object"] == "object_a" else object_b
        logger.info(f"STEP 3 IMAGE: mode=HYBRID_SINGLE_OBJECT, hero={hero_name}, context={context_name}")
    
    for attempt in range(max_retries):
        is_strict = attempt > 0  # Use stricter prompt on retries
        
        prompt = create_image_prompt(
            object_a=object_a,
            object_b=object_b,
            headline=headline,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            is_strict=is_strict
        )
        
        try:
            logger.info(f"STEP 3 - IMAGE GENERATION: attempt={attempt + 1}/{max_retries}, image_model={model}, image_size={image_size}")
            
            # Simple call without response_format for gpt-image-1.5 compatibility
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=image_size
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
    
    Args:
        payload_dict: Request payload with productName, productDescription, etc.
    
    Returns:
        dict: {
            "imageBase64": str (base64 encoded JPG),
            "ad_goal": str,
            "headline": str,
            "chosen_objects": [str, str],
            "layout": "SIDE_BY_SIDE"
        }
    """
    request_id = str(uuid.uuid4())
    
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
    session_id = payload_dict.get("sessionId")
    history = payload_dict.get("history", [])
    object_list = payload_dict.get("objectList")
    
    # Determine ad_goal (from history or from productDescription)
    ad_goal = None
    if history:
        # Try to get ad_goal from most recent history item
        for item in reversed(history):
            if isinstance(item, dict) and "ad_goal" in item:
                ad_goal = item.get("ad_goal", "")
                if ad_goal:
                    break
    
    # If no ad_goal from history, use productDescription as ad_goal
    if not ad_goal:
        ad_goal = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # STEP 0 - BUILD_OBJECT_LIST_FROM_AD_GOAL (if needed)
    # Use validate_object_list which will call STEP 0 if object_list is missing/small and ad_goal exists
    object_list = validate_object_list(object_list, ad_goal=ad_goal, product_name=product_name)
    
    # Log request
    logger.info(f"[{request_id}] generate_preview_data called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}, ad_goal={ad_goal[:50]}")
    
    # STEP 1 - SHAPE SELECTION (ONLY SHAPE)
    used_objects = get_used_objects(history)
    try:
        shape_result = select_similar_pair_shape_only(
            object_list=object_list,
            used_objects=used_objects,
            max_retries=2
        )
        object_a = shape_result["object_a"]
        object_b = shape_result["object_b"]
        shape_hint = shape_result.get("shape_hint", "")
        shape_score = shape_result.get("shape_similarity_score", 0)
        
        logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a}, {object_b}], score={shape_score}, shape_hint={shape_hint}")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection error: {error_msg}")
            raise
    
    # STEP 1.5 - PHYSICAL CONTEXT EXTENSION
    try:
        physical_context = generate_physical_context_extensions(
            object_a=object_a,
            object_b=object_b,
            ad_goal=ad_goal,
            max_retries=2
        )
        logger.info(f"[{request_id}] STEP 1.5 SUCCESS: physical_context_extensions generated")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1.5 FAILED: Physical context extension rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1.5 FAILED: Physical context extension error: {error_msg}")
            raise
    
    # STEP 1.75 - HYBRID CONTEXT PLAN
    message = product_description if product_description else "Make a difference"
    try:
        hybrid_plan = generate_hybrid_context_plan(
            object_a=object_a,
            object_b=object_b,
            ad_goal=ad_goal,
            message=message,
            image_size=image_size_str,
            max_retries=2
        )
        logger.info(f"[{request_id}] STEP 1.75 SUCCESS: hybrid_context_plan generated")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1.75 FAILED: Hybrid context plan rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1.75 FAILED: Hybrid context plan error: {error_msg}")
            raise
    
    # STEP 2 - HEADLINE GENERATION
    # Use headline_placement from hybrid_plan
    headline_placement = hybrid_plan.get("headline_placement") if hybrid_plan else None
    try:
        headline = generate_headline_only(
            product_name=product_name,
            message=message,
            object_a=object_a,
            object_b=object_b,
            headline_placement=headline_placement,
            max_retries=3
        )
        logger.info(f"[{request_id}] STEP 2 SUCCESS: headline={headline}")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation error: {error_msg}")
            raise
    
    # STEP 4 - FINAL VALIDATION
    # Ensure objects haven't changed
    if object_a != shape_result["object_a"] or object_b != shape_result["object_b"]:
        logger.error(f"[{request_id}] STEP 4 VALIDATION FAILED: Objects changed after STEP 1")
        raise ValueError("Objects changed after shape selection")
    
    logger.info(f"[{request_id}] STEP 4 VALIDATION PASSED: object_a={object_a}, object_b={object_b} (unchanged)")
    
    # STEP 3 - IMAGE GENERATION
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        image_bytes = generate_image_with_dalle(
            client=client,
            object_a=object_a,
            object_b=object_b,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            headline=headline,
            width=width,
            height=height,
            max_retries=3
        )
    except Exception as e:
        logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
        raise
    
    # Convert image to base64 (without data URI header)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get image model and size for logging
    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    image_size_str = payload_dict.get("imageSize", "1536x1024")
    
    logger.info(f"[{request_id}] STEP 3 SUCCESS: image_model={image_model}, image_size={image_size_str}, preview_success=true")
    
    # Return only imageBase64 (all text is in the image)
    return {
        "imageBase64": image_base64
    }


def generate_zip(payload_dict: Dict, is_preview: bool = False) -> bytes:
    """
    Generate ZIP file with image.jpg and text.txt.
    
    Args:
        payload_dict: Request payload with productName, productDescription, etc.
        is_preview: If True, this is a preview request (same logic, but can be optimized)
    
    Returns:
        bytes: ZIP file content
    """
    request_id = str(uuid.uuid4())
    
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
    session_id = payload_dict.get("sessionId")
    history = payload_dict.get("history", [])
    object_list = payload_dict.get("objectList")
    
    # Determine ad_goal (from history or from productDescription)
    ad_goal = None
    if history:
        # Try to get ad_goal from most recent history item
        for item in reversed(history):
            if isinstance(item, dict) and "ad_goal" in item:
                ad_goal = item.get("ad_goal", "")
                if ad_goal:
                    break
    
    # If no ad_goal from history, use productDescription as ad_goal
    if not ad_goal:
        ad_goal = product_description if product_description else "Make a difference"
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    
    # STEP 0 - BUILD_OBJECT_LIST_FROM_AD_GOAL (if needed)
    # Use validate_object_list which will call STEP 0 if object_list is missing/small and ad_goal exists
    object_list = validate_object_list(object_list, ad_goal=ad_goal, product_name=product_name)
    
    # Log request
    logger.info(f"[{request_id}] generate_zip called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}, is_preview={is_preview}, ad_goal={ad_goal[:50]}")
    
    # STEP 1 - SHAPE SELECTION (ONLY SHAPE)
    used_objects = get_used_objects(history)
    try:
        shape_result = select_similar_pair_shape_only(
            object_list=object_list,
            used_objects=used_objects,
            max_retries=2
        )
        object_a = shape_result["object_a"]
        object_b = shape_result["object_b"]
        shape_hint = shape_result.get("shape_hint", "")
        shape_score = shape_result.get("shape_similarity_score", 0)
        
        logger.info(f"[{request_id}] STEP 1 SUCCESS: selected_pair=[{object_a}, {object_b}], score={shape_score}, shape_hint={shape_hint}")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1 FAILED: Shape selection error: {error_msg}")
            raise
    
    # STEP 1.5 - PHYSICAL CONTEXT EXTENSION
    try:
        physical_context = generate_physical_context_extensions(
            object_a=object_a,
            object_b=object_b,
            ad_goal=ad_goal,
            max_retries=2
        )
        logger.info(f"[{request_id}] STEP 1.5 SUCCESS: physical_context_extensions generated")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1.5 FAILED: Physical context extension rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1.5 FAILED: Physical context extension error: {error_msg}")
            raise
    
    # STEP 1.75 - HYBRID CONTEXT PLAN
    message = product_description if product_description else "Make a difference"
    try:
        hybrid_plan = generate_hybrid_context_plan(
            object_a=object_a,
            object_b=object_b,
            ad_goal=ad_goal,
            message=message,
            image_size=image_size_str,
            max_retries=2
        )
        logger.info(f"[{request_id}] STEP 1.75 SUCCESS: hybrid_context_plan generated")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 1.75 FAILED: Hybrid context plan rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 1.75 FAILED: Hybrid context plan error: {error_msg}")
            raise
    
    # STEP 2 - HEADLINE GENERATION
    # Use headline_placement from hybrid_plan
    headline_placement = hybrid_plan.get("headline_placement") if hybrid_plan else None
    try:
        headline = generate_headline_only(
            product_name=product_name,
            message=message,
            object_a=object_a,
            object_b=object_b,
            headline_placement=headline_placement,
            max_retries=3
        )
        logger.info(f"[{request_id}] STEP 2 SUCCESS: headline={headline}")
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation rate limited")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] STEP 2 FAILED: Headline generation error: {error_msg}")
            raise
    
    # STEP 4 - FINAL VALIDATION
    # Ensure objects haven't changed
    if object_a != shape_result["object_a"] or object_b != shape_result["object_b"]:
        logger.error(f"[{request_id}] STEP 4 VALIDATION FAILED: Objects changed after STEP 1")
        raise ValueError("Objects changed after shape selection")
    
    logger.info(f"[{request_id}] STEP 4 VALIDATION PASSED: object_a={object_a}, object_b={object_b} (unchanged)")
    
    # STEP 3 - IMAGE GENERATION
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        image_bytes = generate_image_with_dalle(
            client=client,
            object_a=object_a,
            object_b=object_b,
            shape_hint=shape_hint,
            physical_context=physical_context,
            hybrid_plan=hybrid_plan,
            headline=headline,
            width=width,
            height=height,
            max_retries=3
        )
    except Exception as e:
        logger.error(f"[{request_id}] STEP 3 FAILED: Image generation error: {str(e)}")
        raise
    
    # Create minimal text file (optional, for documentation)
    text_content = create_text_file(
        session_id=session_id,
        ad_index=ad_index,
        product_name=product_name,
        ad_goal="",  # No ad_goal in new architecture
        headline=headline,  # Use headline from STEP 2
        chosen_objects=[object_a, object_b]
    )
    
    # Create ZIP with image.jpg only (text.txt is optional)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("image.jpg", image_bytes)
        # Optional: include minimal text.txt for documentation
        zip_file.writestr("text.txt", text_content.encode('utf-8'))
    
    logger.info(f"[{request_id}] ZIP created successfully: {len(zip_buffer.getvalue())} bytes")
    
    return zip_buffer.getvalue()

