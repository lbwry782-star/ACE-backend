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
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Default object list (if not provided)
DEFAULT_OBJECT_LIST = [
    "ספורט", "מוזיקה", "טכנולוגיה", "אוכל", "טבע", "עיצוב", "אופנה", "מכוניות",
    "אמנות", "ספרים", "סרטים", "משחקים", "חיות", "ים", "הרים", "ערים",
    "צבעים", "אור", "צללים", "מרקם", "צורה", "קו", "נקודה", "משטח"
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
    If None or too small, return default list.
    """
    if not object_list or len(object_list) < 2:
        logger.info("objectList missing or too small, using default list")
        return DEFAULT_OBJECT_LIST
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
    
    # Build prompt
    lang_instruction = "עברית" if language == "he" else "English"
    headline_lang = "בעברית" if language == "he" else "in English"
    
    history_context = ""
    if history:
        history_context = f"\n\nמודעות קודמות:\n"
        for i, item in enumerate(history, 1):
            goal = item.get("ad_goal", "")
            objs = item.get("chosen_objects", [])
            history_context += f"{i}. ad_goal: {goal}, chosen_objects: {', '.join(objs) if isinstance(objs, list) else str(objs)}\n"
        history_context += "\nחשוב: אל תבחר אובייקטים שכבר הופיעו, ותן ad_goal שונה ממה שכבר הופיע."
    
    prompt = f"""אתה יוצר מודעה פרסומית למוצר.

מוצר: {product_name}
תיאור: {product_description}
שפה: {lang_instruction}

רשימת אובייקטים זמינים (בחר בדיוק 2 שונים):
{json.dumps(available_objects, ensure_ascii=False, indent=2)}
{history_context}

דרישות:
1. בחר בדיוק 2 אובייקטים שונים מתוך הרשימה בלבד.
2. כתוב headline {headline_lang} (5-8 מילים) שכולל את שם המוצר או שם המותג.
3. תן ad_goal ברור ומשכנע.
4. Layout תמיד SIDE_BY_SIDE (אל תציין layout בתשובה).

החזר JSON בלבד בפורמט:
{{
  "ad_goal": "...",
  "headline": "...",
  "chosen_objects": ["אובייקט1", "אובייקט2"]
}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "אתה עוזר ליצירת מודעות פרסומיות. החזר JSON בלבד ללא טקסט נוסף."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate result
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            
            if "chosen_objects" not in result or not isinstance(result["chosen_objects"], list):
                raise ValueError("Missing or invalid chosen_objects")
            
            if len(result["chosen_objects"]) != 2:
                raise ValueError("chosen_objects must contain exactly 2 items")
            
            # Validate objects are in the list
            chosen = result["chosen_objects"]
            if not all(obj in object_list for obj in chosen):
                invalid = [obj for obj in chosen if obj not in object_list]
                raise ValueError(f"Objects not in list: {invalid}")
            
            if result["chosen_objects"][0] == result["chosen_objects"][1]:
                raise ValueError("chosen_objects must be different")
            
            if not result.get("headline") or not result.get("ad_goal"):
                raise ValueError("Missing headline or ad_goal")
            
            logger.info(f"OpenAI call succeeded on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit (429)
            if "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower():
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
            
            # Other errors - if last attempt, raise; otherwise retry once
            if attempt < max_retries - 1:
                logger.warning(f"OpenAI call failed (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...")
                time.sleep(1)
                continue
            else:
                logger.error(f"OpenAI call failed after {max_retries} attempts: {error_str}")
                raise
    
    raise Exception("Failed to get valid response from OpenAI")


def create_side_by_side_image(
    width: int,
    height: int,
    headline: str,
    object_a: str,
    object_b: str
) -> bytes:
    """
    Create a SIDE_BY_SIDE image with two objects and headline.
    Returns image as bytes (JPEG).
    """
    # Create image with light background
    img = Image.new('RGB', (width, height), color='#F5F5F5')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    font = None
    headline_font = None
    font_size = max(40, width // 30)
    
    # Try common font paths (Windows, Linux, Mac)
    font_paths = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            headline_font = ImageFont.truetype(font_path, font_size + 10)
            break
        except:
            continue
    
    # Fallback to default font if no truetype font found
    if font is None:
        try:
            font = ImageFont.load_default()
            headline_font = ImageFont.load_default()
        except:
            pass
    
    # Calculate layout
    padding = width // 20
    card_width = (width - 3 * padding) // 2
    card_height = height - 4 * padding - 80  # Space for headline
    card_y = 80 + padding  # Start below headline area
    
    # Draw headline at top
    headline_y = padding + 20
    if headline_font:
        draw.text((width // 2, headline_y), headline, fill='#333333', 
                 font=headline_font, anchor='mm')
    else:
        draw.text((width // 2, headline_y), headline, fill='#333333', anchor='mm')
    
    # Draw left card (Object A)
    left_x = padding
    left_rect = [left_x, card_y, left_x + card_width, card_y + card_height]
    draw.rectangle(left_rect, fill='#E8F4F8', outline='#B0D4E3', width=3)
    
    # Object A text
    obj_a_y = card_y + card_height // 2
    if font:
        draw.text((left_x + card_width // 2, obj_a_y), object_a, 
                 fill='#1A5F7A', font=font, anchor='mm')
    else:
        draw.text((left_x + card_width // 2, obj_a_y), object_a, 
                 fill='#1A5F7A', anchor='mm')
    
    # Draw right card (Object B)
    right_x = padding * 2 + card_width
    right_rect = [right_x, card_y, right_x + card_width, card_y + card_height]
    draw.rectangle(right_rect, fill='#F8E8F4', outline='#E3B0D4', width=3)
    
    # Object B text
    obj_b_y = card_y + card_height // 2
    if font:
        draw.text((right_x + card_width // 2, obj_b_y), object_b, 
                 fill='#7A1A5F', font=font, anchor='mm')
    else:
        draw.text((right_x + card_width // 2, obj_b_y), object_b, 
                 fill='#7A1A5F', anchor='mm')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    return img_bytes.getvalue()


def create_text_file(
    session_id: Optional[str],
    ad_index: int,
    product_name: str,
    ad_goal: str,
    headline: str,
    chosen_objects: List[str]
) -> str:
    """
    Create text.txt content with proper format.
    """
    lines = []
    if session_id:
        lines.append(f"sessionId={session_id}")
    lines.append(f"adIndex={ad_index}")
    lines.append("layout=SIDE_BY_SIDE")
    lines.append(f"productName={product_name}")
    lines.append(f"ad_goal={ad_goal}")
    lines.append(f"headline={headline}")
    lines.append(f"chosen_objects={' | '.join(chosen_objects)}")
    
    return "\n".join(lines)


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
    language = payload_dict.get("language", "he")
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
    
    # Create image
    try:
        image_bytes = create_side_by_side_image(
            width=width,
            height=height,
            headline=headline,
            object_a=chosen_objects[0],
            object_b=chosen_objects[1]
        )
    except Exception as e:
        logger.error(f"[{request_id}] Image creation failed: {str(e)}")
        raise
    
    # Create text file
    text_content = create_text_file(
        session_id=session_id,
        ad_index=ad_index,
        product_name=product_name,
        ad_goal=ad_goal,
        headline=headline,
        chosen_objects=chosen_objects
    )
    
    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("image.jpg", image_bytes)
        zip_file.writestr("text.txt", text_content.encode('utf-8'))
    
    logger.info(f"[{request_id}] ZIP created successfully: {len(zip_buffer.getvalue())} bytes")
    
    return zip_buffer.getvalue()

