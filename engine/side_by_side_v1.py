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


def validate_object_list(object_list: Optional[List[str]]) -> List[str]:
    """
    Validate and return object list.
    If None or too small, return default concrete objects list.
    """
    if not object_list or len(object_list) < 2:
        logger.info(f"objectList missing or too small (size={len(object_list) if object_list else 0}), using default concrete objects list (size={len(DEFAULT_OBJECT_LIST)})")
        return DEFAULT_OBJECT_LIST
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


def call_openai_model(
    product_name: str,
    product_description: str,
    language: str,
    object_list: List[str],
    history: Optional[List[Dict]],
    model_name: str,
    max_retries: int = 3
) -> Dict:
    """
    Call OpenAI model to generate ad content.
    Returns: {"ad_goal": str, "headline": str, "chosen_objects": [str, str]}
    
    Implements retry logic with exponential backoff + jitter for 429 errors.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    used_objects = get_used_objects(history)
    used_goals = get_used_ad_goals(history)
    
    # Filter out used objects
    available_objects = [obj for obj in object_list if obj not in used_objects]
    if len(available_objects) < 2:
        # If not enough unused objects, allow reuse but prefer unused
        available_objects = object_list
        logger.warning("Not enough unused objects, allowing reuse")
    
    # Build prompt in English only
    history_context = ""
    if history:
        history_context = f"\n\nPrevious ads:\n"
        for i, item in enumerate(history, 1):
            goal = item.get("ad_goal", "")
            objs = item.get("chosen_objects", [])
            history_context += f"{i}. ad_goal: {goal}, chosen_objects: {', '.join(objs) if isinstance(objs, list) else str(objs)}\n"
        history_context += "\nImportant: Do not choose objects that already appeared, and give a different ad_goal from previous ones."
    
    prompt = f"""You are creating an advertisement for a product.

Product: {product_name}
Description: {product_description}
Language: English only (output must be in English only)

Available object list (choose EXACTLY 2 different items):
{json.dumps(available_objects, ensure_ascii=False, indent=2)}
{history_context}

Requirements:
1. Choose EXACTLY 2 different objects from the list above. Return EXACTLY two items from the provided list. No new words. No categories.
2. Choose two objects with similar silhouette/shape (e.g., ear~shell, leaf~tree, bottle~candle). Do NOT choose conceptual categories.
3. Write a headline in English (5-8 words) that includes the product name or brand name.
4. Give a clear and compelling ad_goal.
5. Layout is always SIDE_BY_SIDE (do not mention layout in response).

Return JSON only in this format:
{{
  "ad_goal": "...",
  "headline": "...",
  "chosen_objects": ["object1", "object2"]
}}"""

    # Check if model is o* type (o4-mini, o3, o1-mini, etc.) - these don't support temperature
    # Pattern: starts with "o" followed by a digit
    is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
    use_temperature = not is_o_model
    
    logger.info(f"objectList_size={len(object_list)}, selected objects will be validated against this list")
    
    for attempt in range(max_retries):
        is_strict_validation = attempt > 0  # Use stricter prompt on retries
        
        # Build prompt (stricter on retries)
        current_prompt = prompt
        if is_strict_validation:
            current_prompt = f"""You are creating an advertisement for a product.

Product: {product_name}
Description: {product_description}
Language: English only (output must be in English only)

Available object list (choose EXACTLY 2 different items):
{json.dumps(available_objects, ensure_ascii=False, indent=2)}
{history_context}

Requirements (STRICT):
1. Return EXACTLY two items from the provided list. No new words. No categories.
2. Choose two objects with similar silhouette/shape (e.g., ear~shell, leaf~tree, bottle~candle). Do NOT choose conceptual categories.
3. Write a headline in English (5-8 words) that includes the product name or brand name.
4. Give a clear and compelling ad_goal.
5. Layout is always SIDE_BY_SIDE (do not mention layout in response).

Return JSON only in this format:
{{
  "ad_goal": "...",
  "headline": "...",
  "chosen_objects": ["object1", "object2"]
}}"""
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an assistant for creating advertisements. Output must be in English only. Return JSON only without additional text."},
                {"role": "user", "content": current_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        # Only add temperature for non-o* models
        if use_temperature:
            request_params["temperature"] = 0.7
        
        try:
            if attempt == 0:
                logger.info(f"Using temperature={'0.7' if use_temperature else 'omitted'} for model {model_name}")
            
            response = client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate result
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            
            if "chosen_objects" not in result or not isinstance(result["chosen_objects"], list):
                raise ValueError("Missing or invalid chosen_objects")
            
            if len(result["chosen_objects"]) != 2:
                raise ValueError("chosen_objects must contain exactly 2 items")
            
            # STRICT VALIDATION: objects must be EXACTLY in the list
            chosen = result["chosen_objects"]
            validation_passed = all(obj in object_list for obj in chosen)
            
            if not validation_passed:
                invalid = [obj for obj in chosen if obj not in object_list]
                logger.warning(f"Validation failed: objects not in list: {invalid}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Retry with stricter prompt
                    logger.info(f"Retrying with stricter prompt (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    raise ValueError(f"Objects not in list after {max_retries} attempts: {invalid}")
            
            if result["chosen_objects"][0] == result["chosen_objects"][1]:
                raise ValueError("chosen_objects must be different")
            
            if not result.get("headline") or not result.get("ad_goal"):
                raise ValueError("Missing headline or ad_goal")
            
            logger.info(f"OpenAI call succeeded on attempt {attempt + 1}, selected objects={chosen}, validation_passed=true")
            return result
            
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
    product_name: str,
    object_a: str,
    object_b: str,
    headline: str,
    is_strict: bool = False
) -> str:
    """
    Create DALL-E prompt for SIDE_BY_SIDE image with text.
    
    Args:
        product_name: Product name
        object_a: Left panel object/concept
        object_b: Right panel object/concept
        headline: Single headline (max 7 words INCLUDING product name, ALL CAPS)
        is_strict: If True, use stricter prompt for retry
    """
    if is_strict:
        # Stricter prompt for retry
        return f"""Create a professional advertisement image with a SIDE BY SIDE layout.

LAYOUT:
- Two distinct objects side by side (left and right).
- No overlap.
- Clean composition.
- The headline must be integrated into the design as a strong central visual element.
- The headline must have the same visual importance as the objects.

TEXT RULES (CRITICAL):
- Only one headline: "{headline}"
- Use fewer letters. Use one short headline. Make text extremely large and bold.
- Maximum 7 words INCLUDING the product name.
- ALL CAPS.
- English only.
- Perfectly legible.
- No paragraphs.
- No small print.
- No separate CTA.

STYLE:
- Bold, modern, minimal.
- High contrast.
- Clear typography.
- Professional advertising aesthetic."""

    else:
        # Standard prompt
        return f"""Create a professional advertisement image with a SIDE BY SIDE layout.

LAYOUT:
- Two distinct objects side by side (left and right).
- No overlap.
- Clean composition.
- The headline must be integrated into the design as a strong central visual element.
- The headline must have the same visual importance as the objects.

TEXT RULES (CRITICAL):
- Only one headline: "{headline}"
- Maximum 7 words INCLUDING the product name.
- ALL CAPS.
- English only.
- Perfectly legible.
- No paragraphs.
- No small print.
- No separate CTA.

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
    product_name: str,
    object_a: str,
    object_b: str,
    headline: str,
    width: int,
    height: int,
    max_retries: int = 3
) -> bytes:
    """
    Generate image using DALL-E with retry logic for text quality.
    
    Returns:
        bytes: JPEG image
    """
    model = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
    image_size = f"{width}x{height}"
    
    for attempt in range(max_retries):
        is_strict = attempt > 0  # Use stricter prompt on retries
        
        prompt = create_image_prompt(
            product_name=product_name,
            object_a=object_a,
            object_b=object_b,
            headline=headline,
            is_strict=is_strict
        )
        
        try:
            logger.info(f"Generating image (attempt {attempt + 1}/{max_retries}), image_model={model}, image_size={image_size}, strict={is_strict}")
            
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
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    object_list = validate_object_list(object_list)
    
    # Log request
    logger.info(f"[{request_id}] generate_preview_data called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}")
    
    # Get model name
    model_name = os.environ.get("OPENAI_TEXT_MODEL", "o1-mini")
    
    # Call OpenAI model
    try:
        result = call_openai_model(
            product_name=product_name,
            product_description=product_description,
            language=language,
            object_list=object_list,
            history=history,
            model_name=model_name,
            max_retries=3
        )
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] Rate limited after retries")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] OpenAI call failed: {error_msg}")
            raise
    
    ad_goal = result["ad_goal"]
    headline = result["headline"]
    chosen_objects = result["chosen_objects"]
    
    # Log result
    logger.info(f"[{request_id}] Model response: ad_goal={ad_goal[:50]}, "
                f"headline={headline[:50]}, chosen_objects={chosen_objects}, "
                f"model={model_name}")
    
    # Generate short headline from product name (for image text)
    product_name = payload_dict.get("productName", "")
    image_headline = generate_short_phrase(product_name)
    
    # Create image using DALL-E
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        image_bytes = generate_image_with_dalle(
            client=client,
            product_name=product_name,
            object_a=chosen_objects[0],
            object_b=chosen_objects[1],
            headline=image_headline,
            width=width,
            height=height,
            max_retries=3
        )
    except Exception as e:
        logger.error(f"[{request_id}] Image generation failed: {str(e)}")
        raise
    
    # Convert image to base64 (without data URI header)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get image model and size for logging
    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
    image_size_str = payload_dict.get("imageSize", "1536x1024")
    
    logger.info(f"[{request_id}] Preview data created successfully: imageBase64 length={len(image_base64)}, "
               f"image_model={image_model}, image_size={image_size_str}, preview_success=true")
    
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
    
    # Validate and normalize
    width, height = parse_image_size(image_size_str)
    object_list = validate_object_list(object_list)
    
    # Log request
    logger.info(f"[{request_id}] generate_zip called: sessionId={session_id}, adIndex={ad_index}, "
                f"productName={product_name[:50]}, language={language}, is_preview={is_preview}")
    
    # Get model name
    model_name = os.environ.get("OPENAI_TEXT_MODEL", "o1-mini")
    # Use o4-mini if specified, otherwise fallback to o1-mini
    # OpenAI will handle model validation
    
    # Call OpenAI model
    try:
        result = call_openai_model(
            product_name=product_name,
            product_description=product_description,
            language=language,
            object_list=object_list,
            history=history,
            model_name=model_name,
            max_retries=3
        )
    except Exception as e:
        error_msg = str(e)
        if "rate_limited" in error_msg:
            logger.error(f"[{request_id}] Rate limited after retries")
            raise Exception("rate_limited")
        else:
            logger.error(f"[{request_id}] OpenAI call failed: {error_msg}")
            raise
    
    ad_goal = result["ad_goal"]
    headline = result["headline"]
    chosen_objects = result["chosen_objects"]
    
    # Log result (retries are logged inside call_openai_model)
    logger.info(f"[{request_id}] Model response: ad_goal={ad_goal[:50]}, "
                f"headline={headline[:50]}, chosen_objects={chosen_objects}, "
                f"model={model_name}")
    
    # Generate short headline from product name (for image text)
    image_headline = generate_short_phrase(product_name)
    
    # Create image using DALL-E
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        image_bytes = generate_image_with_dalle(
            client=client,
            product_name=product_name,
            object_a=chosen_objects[0],
            object_b=chosen_objects[1],
            headline=image_headline,
            width=width,
            height=height,
            max_retries=3
        )
    except Exception as e:
        logger.error(f"[{request_id}] Image generation failed: {str(e)}")
        raise
    
    # Create minimal text file (optional, for documentation)
    text_content = create_text_file(
        session_id=session_id,
        ad_index=ad_index,
        product_name=product_name,
        ad_goal=ad_goal,
        headline=image_headline,  # Use image headline
        chosen_objects=chosen_objects
    )
    
    # Create ZIP with image.jpg only (text.txt is optional)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("image.jpg", image_bytes)
        # Optional: include minimal text.txt for documentation
        zip_file.writestr("text.txt", text_content.encode('utf-8'))
    
    logger.info(f"[{request_id}] ZIP created successfully: {len(zip_buffer.getvalue())} bytes")
    
    return zip_buffer.getvalue()

