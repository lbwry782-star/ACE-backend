"""Builder1 side-by-side pipeline: o3-pro concept/copy + gpt-image still generation."""

from __future__ import annotations

import base64
import io
import json
import os
from threading import Lock
from typing import Dict, List, Optional

import httpx
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

_memory_lock = Lock()
# Per sessionId: all prior runs (no TTL; new sessionId starts empty).
_session_memory: Dict[str, List[dict]] = {}
# Cross-session recent concepts (bounded list, no TTL).
_global_concepts: List[dict] = []
_GLOBAL_LIST_MAX = 100
_SESSION_PROMPT_TAIL = 3
_GLOBAL_PROMPT_TAIL = 4


def _memory_triplet_line(rec: dict) -> str:
    a = ((rec.get("objectA") or "").replace("\n", " ").strip())[:64]
    b = ((rec.get("objectB") or "").replace("\n", " ").strip())[:64]
    p = ((rec.get("advertisingPromise") or "").replace("\n", " ").strip())[:96]
    return f"{a} | {b} | {p}"


def mode_from_similarity(morphological_similarity: int) -> str:
    """REPLACEMENT if morphological similarity >= 85, otherwise SIDE_BY_SIDE."""
    return "REPLACEMENT" if morphological_similarity >= 85 else "SIDE_BY_SIDE"


def _memory_instruction_appendix(session_id: Optional[str]) -> str:
    """Tiny o3 addendum: last few triplets only; morphology stays primary."""
    with _memory_lock:
        sess_rows = list(_session_memory.get(session_id or "", []))[-_SESSION_PROMPT_TAIL:] if session_id else []
        glob_rows = list(_global_concepts)[-_GLOBAL_PROMPT_TAIL:]
    blocks: list[str] = []
    if sess_rows:
        lines = [_memory_triplet_line(r) for r in sess_rows]
        blocks.append(
            "Session memory (vary; morphology rules above always win): "
            + " ; ".join(lines)
        )
    if glob_rows:
        lines = [_memory_triplet_line(r) for r in glob_rows]
        blocks.append(
            "Global memory (avoid clichés; morphology first): "
            + " ; ".join(lines)
        )
    return "\n\n".join(blocks) if blocks else ""


def _record_builder1_memories(
    session_id: Optional[str],
    object_a: str,
    object_b: str,
    advertising_promise: str,
    mode: str,
    headline: str,
) -> None:
    sess_rec = {
        "objectA": object_a,
        "objectB": object_b,
        "advertisingPromise": advertising_promise,
        "mode": mode,
        "headline": headline,
    }
    glob_rec = {
        "objectA": object_a,
        "objectB": object_b,
        "advertisingPromise": advertising_promise,
    }
    with _memory_lock:
        if session_id:
            _session_memory.setdefault(session_id, []).append(sess_rec)
        _global_concepts.append(glob_rec)
        while len(_global_concepts) > _GLOBAL_LIST_MAX:
            _global_concepts.pop(0)


def get_concept_from_o3(
    product_name: str, product_description: str, session_id: Optional[str] = None
) -> dict:
    """One o3-pro call: structured concept JSON for Builder1. Raises ValueError if JSON is invalid."""
    pn = (product_name or "").strip() or "Product"
    desc = (product_description or "").strip() or "No description provided."
    user_input = f"""Product name: {pn}
Product description: {desc}

Return ONLY valid JSON with exactly these keys and no others (no markdown fences):
{{ "objectA": "<string>", "objectB": "<string>", "advertisingPromise": "<string>", "morphologicalSimilarity": <integer 0-100>, "reasoning": "<string>" }}

Rules for the model:
- Choose Object A from the product name and description by grasping its overall physical form intuitively, like a painter, not as a technical contour only.
- Then find Object B with the strongest possible morphological similarity to A.
- Stop only when B also introduces an additional conceptual reason, which is the advertising promise (the advertisingPromise field).
- Do not choose a weaker B just to make the advertising promise more obvious.
- Trust viewer intuition.
- objectA and objectB must be physical, simple, everyday, clearly defined. No text, logos, or brands. No vague environments or non-physical situations.
- Keep reasoning short.
"""
    appendix = _memory_instruction_appendix(session_id)
    if appendix:
        user_input = user_input.rstrip() + "\n\n" + appendix

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    response = client.responses.create(
        model="o3-pro",
        input=user_input,
        reasoning={"effort": "low"},
    )
    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw:
        raise ValueError("BUILDER1_CONCEPT_O3: empty output_text from o3-pro")

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    if cleaned.startswith("```json"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"BUILDER1_CONCEPT_O3: invalid JSON from o3-pro: {e}") from e

    required = (
        "objectA",
        "objectB",
        "advertisingPromise",
        "morphologicalSimilarity",
        "reasoning",
    )
    if not isinstance(data, dict):
        raise ValueError("BUILDER1_CONCEPT_O3: parsed JSON root is not an object")
    if set(data.keys()) != set(required):
        raise ValueError(
            f"BUILDER1_CONCEPT_O3: JSON keys must be exactly {list(required)}, got {list(data.keys())}"
        )

    for key in ("objectA", "objectB", "advertisingPromise", "reasoning"):
        val = data[key]
        if not isinstance(val, str):
            raise ValueError(
                f"BUILDER1_CONCEPT_O3: {key} must be a string, got {type(val).__name__}"
            )

    sim = data["morphologicalSimilarity"]
    try:
        sim_i = int(sim)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"BUILDER1_CONCEPT_O3: morphologicalSimilarity must be an integer 0-100, got {sim!r}"
        ) from e
    if sim_i < 0 or sim_i > 100:
        raise ValueError(f"BUILDER1_CONCEPT_O3: morphologicalSimilarity out of range: {sim_i}")

    return {
        "objectA": data["objectA"],
        "objectB": data["objectB"],
        "advertisingPromise": data["advertisingPromise"],
        "morphologicalSimilarity": sim_i,
        "reasoning": data["reasoning"],
    }


def build_image_prompt(objectA: str, objectB: str, mode: str) -> str:
    """Build a single gpt-image-1.5 photorealistic prompt from objects and layout mode."""
    a = (objectA or "").strip()
    b = (objectB or "").strip()
    if mode == "SIDE_BY_SIDE":
        return f"""Photorealistic product-ad image, studio quality, pure white seamless background.
No text, letters, numbers, logos, signage, labels, or brands anywhere in the image.
Clean studio lighting, sharp focus, clear silhouettes for both subjects.

Show these two physical objects together: Object A is "{a}". Object B is "{b}".
Place them side by side with partial overlap so both remain clearly visible; the overlap should emphasize their morphological similarity.
Do not add any secondary object, prop, or extra item beyond A and B.
Composition is minimal and centered; only the two objects matter."""
    if mode == "REPLACEMENT":
        return f"""Photorealistic product-ad image, studio quality, pure white seamless background.
No text, letters, numbers, logos, signage, labels, or brands anywhere in the image.
Clean studio lighting, sharp focus, clear readable silhouette.

Object A is "{a}". Object B is "{b}".
Show Object B replacing Object A in Object A's role and original spatial logic (same position/scale relationship as A would occupy).
Include only Object A's classic, minimal secondary object (one simple everyday companion object strongly associated with A) — do not add any secondary object for B.
The replacement must read instantly: B clearly occupies A's role; composition is minimal and centered."""
    raise ValueError(f"BUILDER1_IMAGE_PROMPT: unknown mode {mode!r} (expected SIDE_BY_SIDE or REPLACEMENT)")


def generate_image_bytes(image_prompt: str) -> bytes:
    """One gpt-image call from prompt; returns raw image bytes (PNG)."""
    prompt = (image_prompt or "").strip()
    if not prompt:
        raise ValueError("BUILDER1_IMAGE: image_prompt is empty")

    model = (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()
    size = (os.environ.get("OPENAI_BUILDER1_IMAGE_SIZE") or "1024x1024").strip()
    quality = (os.environ.get("OPENAI_BUILDER1_IMAGE_QUALITY") or "low").strip()
    timeout_s = float((os.environ.get("OPENAI_BUILDER1_IMAGE_TIMEOUT_SECONDS") or "120").strip() or "120")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(timeout_s),
        max_retries=0,
    )
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
    )
    if not response.data:
        raise ValueError("BUILDER1_IMAGE: empty response.data from images.generate")
    b64 = getattr(response.data[0], "b64_json", None)
    if not b64:
        raise ValueError("BUILDER1_IMAGE: no b64_json in image response")
    try:
        return base64.b64decode(b64)
    except Exception as e:
        raise ValueError(f"BUILDER1_IMAGE: failed to decode image bytes: {e}") from e


def validate_image_bytes(image_bytes: bytes) -> None:
    """Non-empty bytes must load as a raster image or raise ValueError."""
    if not image_bytes:
        raise ValueError("BUILDER1_IMAGE_VALIDATE: image_bytes is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
    except Exception as e:
        raise ValueError(f"BUILDER1_IMAGE_VALIDATE: not a decodable image: {e}") from e


def _builder1_headline_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    env_path = (os.environ.get("BUILDER1_HEADLINE_FONT") or "").strip()
    candidates = [
        p for p in (
            env_path,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        )
        if p
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def _split_headline_product_rest(headline: str, product_name: str) -> tuple[str, str]:
    """Product token(s) in ALL CAPS first, then remainder; matches generate_headline layout."""
    pname = " ".join((product_name or "").strip().split())
    if not pname:
        return "", (headline or "").strip()
    pu = pname.upper()
    hw = (headline or "").strip().split()
    pw = pu.split()
    if len(hw) >= len(pw) and " ".join(hw[: len(pw)]).upper() == pu:
        rest = " ".join(hw[len(pw) :]).strip()
        return pu, rest
    return "", (headline or "").strip()


def _wrap_words_to_width(
    draw: ImageDraw.ImageDraw, words: list[str], font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int
) -> list[str]:
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w]) if cur else w
        if _text_width(draw, trial, font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def composite_headline_on_image(image_bytes: bytes, headline: str, product_name: str) -> bytes:
    """Compose final ad: orientation-driven layout; product name larger and first. Returns PNG bytes."""
    text = (headline or "").strip()
    if not text:
        return image_bytes

    with Image.open(io.BytesIO(image_bytes)) as src:
        src = src.convert("RGB")
        w, h = src.size
        margin = max(10, min(w, h) // 48)
        is_vertical = h > w
        pu, rest = _split_headline_product_rest(headline, product_name)
        stroke = max(1, min(w, h) // 256)

        if is_vertical:
            out = Image.new("RGB", (w, h), (255, 255, 255))
            top_h = int(h * 0.72)
            bottom_h = h - top_h
            tw, th = w - 2 * margin, top_h - 2 * margin
            scale = min(tw / max(1, src.width), th / max(1, src.height))
            nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
            scaled = src.resize((nw, nh), Image.Resampling.LANCZOS)
            ox = margin + max(0, (tw - nw) // 2)
            oy = margin + max(0, (th - nh) // 2)
            out.paste(scaled, (ox, oy))
            draw = ImageDraw.Draw(out)
            line = f"{pu} {rest}".strip() if pu else text
            max_w = w - 2 * margin
            base_sz = max(14, min(44, w // 18))
            name_sz = int(base_sz * 1.35)
            for _ in range(40):
                font_name = _builder1_headline_font(name_sz)
                font_rest = _builder1_headline_font(base_sz)
                if pu and rest:
                    gap_w = _text_width(draw, " ", font_rest)
                    total = _text_width(draw, pu, font_name) + gap_w + _text_width(draw, rest, font_rest)
                elif pu:
                    total = _text_width(draw, pu, font_name)
                else:
                    total = _text_width(draw, line, font_rest)
                if total <= max_w:
                    break
                name_sz = max(12, name_sz - 2)
                base_sz = max(10, base_sz - 2)
            font_name = _builder1_headline_font(name_sz)
            font_rest = _builder1_headline_font(base_sz)
            mid_y = top_h + bottom_h // 2
            if pu and rest:
                gap_w = _text_width(draw, " ", font_rest)
                total = _text_width(draw, pu, font_name) + gap_w + _text_width(draw, rest, font_rest)
                x_cursor = margin + max(0, (w - 2 * margin - total) // 2)
                kw = dict(
                    fill=(16, 16, 16),
                    stroke_width=stroke,
                    stroke_fill=(248, 248, 248),
                )
                try:
                    draw.text((x_cursor, mid_y), pu, font=font_name, anchor="lm", **kw)
                    x_cursor += int(draw.textlength(pu, font=font_name)) + gap_w
                    draw.text((x_cursor, mid_y), rest, font=font_rest, anchor="lm", **kw)
                except TypeError:
                    draw.text((x_cursor, mid_y), pu, font=font_name, **kw)
                    x_cursor += _text_width(draw, pu, font_name) + gap_w
                    draw.text((x_cursor, mid_y), rest, font=font_rest, **kw)
            else:
                font_one = _builder1_headline_font(base_sz)
                twl = _text_width(draw, line, font_one)
                x = margin + max(0, (w - 2 * margin - twl) // 2)
                kw = dict(
                    fill=(16, 16, 16),
                    stroke_width=stroke,
                    stroke_fill=(248, 248, 248),
                )
                try:
                    draw.text((x, mid_y), line, font=font_one, anchor="lm", **kw)
                except TypeError:
                    draw.text((x, mid_y), line, font=font_one, **kw)
        else:
            out = Image.new("RGB", (w, h), (250, 250, 250))
            split_x = w // 2
            left_w = split_x - 2 * margin
            right_w = w - split_x - 2 * margin
            lh, rh = h - 2 * margin, h - 2 * margin
            scale = min(left_w / max(1, src.width), lh / max(1, src.height))
            nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
            scaled = src.resize((nw, nh), Image.Resampling.LANCZOS)
            ox = margin + max(0, (left_w - nw) // 2)
            oy = margin + max(0, (lh - nh) // 2)
            out.paste(scaled, (ox, oy))
            draw = ImageDraw.Draw(out)
            tx0 = split_x + margin
            max_tw = right_w - margin
            name_sz = max(16, min(52, right_w // 8))
            body_sz = max(14, int(name_sz * 0.72))
            font_name = _builder1_headline_font(name_sz)
            font_body = _builder1_headline_font(body_sz)
            if pu:
                lines_r = _wrap_words_to_width(draw, rest.split(), font_body, max_tw) if rest else []
                block: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]] = [(pu, font_name)]
                for ln in lines_r:
                    block.append((ln, font_body))
            else:
                block = [(text, font_body)]
            heights = []
            for s, fnt in block:
                bb = draw.textbbox((0, 0), s, font=fnt)
                heights.append(bb[3] - bb[1] + 8)
            total_block = sum(heights)
            y_start = margin + max(0, (lh - total_block) // 2)
            y = y_start
            for s, fnt in block:
                twl = _text_width(draw, s, fnt)
                x = tx0 + max(0, (max_tw - twl) // 2)
                draw.text(
                    (x, y),
                    s,
                    font=fnt,
                    fill=(16, 16, 16),
                    stroke_width=stroke,
                    stroke_fill=(248, 248, 248),
                )
                bb = draw.textbbox((0, 0), s, font=fnt)
                y += bb[3] - bb[1] + 10

        out_buf = io.BytesIO()
        out.save(out_buf, format="PNG", optimize=True)
        return out_buf.getvalue()


def generate_headline(
    objectA: str, objectB: str, advertisingPromise: str, product_name: str
) -> str:
    """Single o3-pro call: one-line headline; product name (ALL CAPS) then existing expression (≤7 words)."""
    a = (objectA or "").strip()
    b = (objectB or "").strip()
    promise = (advertisingPromise or "").strip()
    pname = " ".join((product_name or "").strip().split())
    if not pname:
        raise ValueError("BUILDER1_HEADLINE_O3: product_name is empty")
    pname_upper = pname.upper()

    user_input = f"""Write exactly one print headline for this Builder1 ad visual.

Context (do not paste these labels into the headline):
Object A: {a}
Object B: {b}
Advertising promise (conceptual, for tone only): {promise}
Product name (canonical spelling): {pname}

Rules (all mandatory):
1) The headline must hinge on an EXISTING expression people already know (idiom, proverb, common saying, stock phrase, or familiar slogan fragment). Do not coin a brand-new phrase.
2) After the product name, the expression part must be at most 7 words.
3) That expression part must describe what the viewer literally sees (objects, layout, relation) — not interpret hidden meaning in explanatory prose.
4) Any double meaning should arise only because the photographed visual naturally fits the existing expression; do not spell out the second reading.
5) The headline must begin with the product name in ALL CAPS exactly as: {pname_upper} then one normal space, then the expression part.
6) Keep the product name at the start in ALL CAPS as given; do not lowercase or alter those characters.
7) Assume the product name and the expression sit on one line on the same baseline in the layout.
8) You choose light punctuation or spacing between name and expression only; keep it one readable line.

Output: return ONLY the headline characters — no quotes, no JSON, no numbering, no explanation, no line breaks."""

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    response = client.responses.create(
        model="o3-pro",
        input=user_input,
        reasoning={"effort": "low"},
    )
    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw:
        raise ValueError("BUILDER1_HEADLINE_O3: empty output_text from o3-pro")

    out = raw.strip()
    if (out.startswith('"') and out.endswith('"')) or (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()
    out = " ".join(out.split())
    if not out:
        raise ValueError("BUILDER1_HEADLINE_O3: empty headline after normalizing whitespace")

    if not out.startswith(pname_upper + " "):
        raise ValueError(
            f"BUILDER1_HEADLINE_O3: headline must start with {pname_upper!r} followed by a space, got {out!r}"
        )

    expression = out[len(pname_upper):].strip()
    if not expression:
        raise ValueError("BUILDER1_HEADLINE_O3: missing expression after product name")
    if len(expression.split()) > 7:
        raise ValueError(
            f"BUILDER1_HEADLINE_O3: expression part exceeds 7 words: {expression!r}"
        )

    return out


def generate_marketing_text(headline: str, advertisingPromise: str) -> str:
    """Single o3-pro call: ~50 words of body copy under the ad (Download ZIP area)."""
    h = (headline or "").strip()
    p = (advertisingPromise or "").strip()
    user_input = f"""Write the short marketing body copy that appears under the ad (near a Download ZIP control).

Headline (fixed): {h}
Advertising promise (conceptual anchor): {p}

Rules:
- About 50 words (45–55 is acceptable).
- Ground the copy in the headline and the advertising promise; do not contradict them.
- Concise, polished, advertising-style prose.
- No bullet points or lists.
- No quotation marks in the output.
- Return ONLY the final body text: no titles, no labels, no JSON, no numbering, no preamble."""

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    response = client.responses.create(
        model="o3-pro",
        input=user_input,
        reasoning={"effort": "low"},
    )
    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw:
        raise ValueError("BUILDER1_MARKETING_O3: empty output_text from o3-pro")
    out = " ".join(raw.split())
    if not out:
        raise ValueError("BUILDER1_MARKETING_O3: empty marketing text after normalizing whitespace")
    return out


def generate_builder1_ad(
    product_name: str, product_description: str, session_id: Optional[str] = None
) -> dict:
    """Orchestrate Builder1: concept → mode → prompts → headline → marketing text."""
    concept = get_concept_from_o3(product_name, product_description, session_id=session_id)
    required = (
        "objectA",
        "objectB",
        "advertisingPromise",
        "morphologicalSimilarity",
    )
    for k in required:
        if k not in concept:
            raise KeyError(f"BUILDER1_AD: concept missing required key {k!r}")

    object_a = concept["objectA"]
    object_b = concept["objectB"]
    advertising_promise = concept["advertisingPromise"]
    sim = concept["morphologicalSimilarity"]

    mode = mode_from_similarity(sim)
    image_prompt = build_image_prompt(object_a, object_b, mode)
    headline = generate_headline(
        object_a, object_b, advertising_promise, product_name
    )
    marketing_text = generate_marketing_text(headline, advertising_promise)
    image_bytes = generate_image_bytes(image_prompt)
    validate_image_bytes(image_bytes)
    image_bytes = composite_headline_on_image(image_bytes, headline, product_name)
    validate_image_bytes(image_bytes)

    _record_builder1_memories(
        session_id, object_a, object_b, advertising_promise, mode, headline
    )

    return {
        "objectA": object_a,
        "objectB": object_b,
        "advertisingPromise": advertising_promise,
        "mode": mode,
        "headline": headline,
        "imagePrompt": image_prompt,
        "imageBytes": image_bytes,
        "marketingText": marketing_text,
    }
