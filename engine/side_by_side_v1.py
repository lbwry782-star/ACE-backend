"""Builder1 side-by-side pipeline: o3-pro concept/copy + gpt-image still generation."""

from __future__ import annotations

import base64
import contextvars
import inspect
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
            "Session memory (morphology wins; do not repeat/near-duplicate advertisingPromise vs lines; A|B|promise): "
            + " ; ".join(lines)
        )
    if glob_rows:
        lines = [_memory_triplet_line(r) for r in glob_rows]
        blocks.append(
            "Global memory (morphology first; novel advertisingPromise vs lines; A|B|promise): "
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


_BUILDER1_ALLOWED_IMAGE_SIZES = frozenset({"1024x1024", "1536x1024", "1024x1536"})
# Optional per-request override (e.g. host sets from client `imageSize` before `generate_builder1_ad`).
_builder1_request_image_size: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "builder1_request_image_size", default=None
)


def set_builder1_request_image_size(size: Optional[str]) -> contextvars.Token:
    """Set OpenAI images.generate size for the current context (e.g. ``1024x1536``). Reset with ``reset_builder1_request_image_size``."""
    normalized = _normalize_builder1_image_size(size)
    return _builder1_request_image_size.set(normalized)


def reset_builder1_request_image_size(token: contextvars.Token) -> None:
    _builder1_request_image_size.reset(token)


def _normalize_builder1_image_size(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().replace("×", "x")
    if s in _BUILDER1_ALLOWED_IMAGE_SIZES:
        return s
    return None


def _image_size_from_call_stack() -> Optional[str]:
    """If a caller frame holds preview ``payload_data`` / ``payload`` with ``imageSize``, use it (preview job wiring)."""
    try:
        for fr in inspect.stack()[2:28]:
            loc = fr.frame.f_locals
            for key in ("payload_data", "payload", "data"):
                pd = loc.get(key)
                if isinstance(pd, dict):
                    n = _normalize_builder1_image_size(pd.get("imageSize") or pd.get("image_size"))
                    if n:
                        return n
    except Exception:
        return None
    return None


def _resolve_builder1_generate_size(explicit: Optional[str]) -> str:
    """Pick images.generate size: explicit → contextvar → call-stack preview dict → env → default."""
    for candidate in (
        explicit,
        _builder1_request_image_size.get(),
        _image_size_from_call_stack(),
        os.environ.get("ACE_BUILDER1_IMAGE_SIZE"),
        os.environ.get("OPENAI_BUILDER1_IMAGE_SIZE"),
    ):
        norm = _normalize_builder1_image_size(candidate)
        if norm:
            return norm
    return "1024x1024"


def generate_image_bytes(image_prompt: str, image_size: Optional[str] = None) -> bytes:
    """One gpt-image call from prompt; returns raw image bytes (PNG)."""
    prompt = (image_prompt or "").strip()
    if not prompt:
        raise ValueError("BUILDER1_IMAGE: image_prompt is empty")

    model = (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()
    size = _resolve_builder1_generate_size(image_size)
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


def _safe_margins_xy(w: int, h: int) -> tuple[int, int]:
    """~10% horizontal and vertical padding (within 8–12%)."""
    mx = max(6, int(round(w * 0.10)))
    my = max(6, int(round(h * 0.10)))
    return mx, my


def _corner_mean_rgb(im: Image.Image) -> tuple[int, int, int]:
    s = max(2, min(im.width, im.height) // 32)
    samples: list[tuple[int, int, int]] = []
    for box in (
        (0, 0, s, s),
        (im.width - s, 0, im.width, s),
        (0, im.height - s, s, im.height),
        (im.width - s, im.height - s, im.width, im.height),
    ):
        crop = im.crop(box)
        samples.append(tuple(int(sum(c) / len(c)) for c in zip(*list(crop.getdata()))))
    r = sum(p[0] for p in samples) // len(samples)
    g = sum(p[1] for p in samples) // len(samples)
    b = sum(p[2] for p in samples) // len(samples)
    return (r, g, b)


def _estimate_foreground_bbox_full(im: Image.Image) -> tuple[int, int, int, int]:
    """Rough subject bbox on full image via downsampled mask (no new deps)."""
    sw = min(200, im.width)
    sh = min(200, im.height)
    if sw < 2 or sh < 2:
        return 0, 0, im.width, im.height
    mini = im.resize((sw, sh), Image.Resampling.BILINEAR).convert("L")
    edge: list[int] = []
    for x in range(0, sw, max(1, sw // 32)):
        edge.append(mini.getpixel((x, 0)))
        edge.append(mini.getpixel((x, sh - 1)))
    for y in range(0, sh, max(1, sh // 32)):
        edge.append(mini.getpixel((0, y)))
        edge.append(mini.getpixel((sw - 1, y)))
    bg = sum(edge) // max(1, len(edge))
    xs: list[int] = []
    ys: list[int] = []
    for y in range(sh):
        for x in range(sw):
            if abs(mini.getpixel((x, y)) - bg) > 22:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0, 0, im.width, im.height
    sx = im.width / sw
    sy = im.height / sh
    x0 = int(min(xs) * sx)
    y0 = int(min(ys) * sy)
    x1 = min(im.width, int((max(xs) + 1) * sx))
    y1 = min(im.height, int((max(ys) + 1) * sy))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, im.width, im.height
    return x0, y0, x1, y1


def _square_allow_side_by_side(w: int, h: int, bbox: tuple[int, int, int, int]) -> bool:
    """Prefer below; side-by-side only for compact, centered subject mass."""
    bx0, by0, bx1, by1 = bbox
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 8 or bh < 8:
        return False
    area_f = (bw * bh) / max(1, w * h)
    ar = bw / max(bh, 1)
    cx = (bx0 + bx1) / 2.0 / max(1, w)
    cy = (by0 + by1) / 2.0 / max(1, h)
    compact = 0.20 <= area_f <= 0.45 and 0.80 <= ar <= 1.25
    centered = 0.28 <= cx <= 0.72 and 0.28 <= cy <= 0.72
    return compact and centered


def _shrink_fonts_to_width(
    draw: ImageDraw.ImageDraw,
    pu: str,
    rest: str,
    line: str,
    text: str,
    max_w: int,
    name_sz0: int,
    base_sz0: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont | None, ImageFont.FreeTypeFont | ImageFont.ImageFont, bool]:
    """Return (font_name, font_rest_or_none, font_one, two_fonts) sized so one-line headline fits max_w."""
    name_sz, base_sz = name_sz0, base_sz0
    for _ in range(48):
        font_name = _builder1_headline_font(name_sz)
        font_rest = _builder1_headline_font(base_sz)
        font_one = _builder1_headline_font(base_sz)
        if pu and rest:
            gap_w = _text_width(draw, " ", font_rest)
            total = _text_width(draw, pu, font_name) + gap_w + _text_width(draw, rest, font_rest)
            if total <= max_w:
                return font_name, font_rest, font_one, True
        elif pu:
            total = _text_width(draw, pu, font_name)
            if total <= max_w:
                return font_name, None, font_one, True
        else:
            total = _text_width(draw, line, font_one)
            if total <= max_w:
                return font_name, None, font_one, False
        name_sz = max(8, name_sz - 2)
        base_sz = max(7, base_sz - 1)
    font_one = _builder1_headline_font(base_sz)
    return font_name, font_rest if pu and rest else None, font_one, bool(pu and rest)


def _headline_one_line_union_bbox(
    draw: ImageDraw.ImageDraw,
    mid_y: int,
    w: int,
    mx: int,
    pu: str,
    rest: str,
    line: str,
    stroke: int,
    font_name: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_rest: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    font_one: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    two_fonts: bool,
) -> tuple[int, int, int, int]:
    max_w = w - 2 * mx
    sw = stroke
    if two_fonts and font_rest is not None:
        gap_w = _text_width(draw, " ", font_rest)
        total = _text_width(draw, pu, font_name) + gap_w + _text_width(draw, rest, font_rest)
        x = mx + max(0, (max_w - total) // 2)
        try:
            b1 = draw.textbbox((x, mid_y), pu, font=font_name, anchor="lm", stroke_width=sw)
            x2 = x + int(draw.textlength(pu, font=font_name)) + gap_w
            b2 = draw.textbbox((x2, mid_y), rest, font=font_rest, anchor="lm", stroke_width=sw)
        except TypeError:
            b1 = draw.textbbox((x, mid_y), pu, font=font_name, stroke_width=sw)
            x2 = x + _text_width(draw, pu, font_name) + gap_w
            b2 = draw.textbbox((x2, mid_y), rest, font=font_rest, stroke_width=sw)
        return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
    twl = _text_width(draw, line, font_one)
    x = mx + max(0, (max_w - twl) // 2)
    try:
        return draw.textbbox((x, mid_y), line, font=font_one, anchor="lm", stroke_width=sw)
    except TypeError:
        return draw.textbbox((x, mid_y), line, font=font_one, stroke_width=sw)


def _draw_headline_one_line_centered(
    draw: ImageDraw.ImageDraw,
    mid_y: int,
    w: int,
    mx: int,
    pu: str,
    rest: str,
    line: str,
    text: str,
    stroke: int,
    font_name: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_rest: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    font_one: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    two_fonts: bool,
) -> None:
    max_w = w - 2 * mx
    kw = dict(fill=(16, 16, 16), stroke_width=stroke, stroke_fill=(248, 248, 248))
    if two_fonts and font_rest is not None:
        gap_w = _text_width(draw, " ", font_rest)
        total = _text_width(draw, pu, font_name) + gap_w + _text_width(draw, rest, font_rest)
        x_cursor = mx + max(0, (max_w - total) // 2)
        try:
            draw.text((x_cursor, mid_y), pu, font=font_name, anchor="lm", **kw)
            x_cursor += int(draw.textlength(pu, font=font_name)) + gap_w
            draw.text((x_cursor, mid_y), rest, font=font_rest, anchor="lm", **kw)
        except TypeError:
            draw.text((x_cursor, mid_y), pu, font=font_name, **kw)
            x_cursor += _text_width(draw, pu, font_name) + gap_w
            draw.text((x_cursor, mid_y), rest, font=font_rest, **kw)
    else:
        twl = _text_width(draw, line, font_one)
        x = mx + max(0, (max_w - twl) // 2)
        try:
            draw.text((x, mid_y), line, font=font_one, anchor="lm", **kw)
        except TypeError:
            draw.text((x, mid_y), line, font=font_one, **kw)


def composite_headline_on_image(image_bytes: bytes, headline: str, product_name: str) -> bytes:
    """Compose onto same WxH as generation; safe margins; square uses bbox heuristic."""
    text = (headline or "").strip()
    if not text:
        return image_bytes

    with Image.open(io.BytesIO(image_bytes)) as src:
        src = src.convert("RGB")
        w, h = src.size
        mx, my = _safe_margins_xy(w, h)
        bg = _corner_mean_rgb(src)
        out = Image.new("RGB", (w, h), bg)
        pu, rest = _split_headline_product_rest(headline, product_name)
        stroke = max(1, min(w, h) // 256)
        line = f"{pu} {rest}".strip() if pu else text

        if w > h:
            # Landscape: balanced split, visual ~65% of left band
            split_x = int(w * 0.50)
            left_inner_w = split_x - mx - mx // 2
            zone_h = h - 2 * my
            scale = min(left_inner_w / max(1, src.width), zone_h / max(1, src.height)) * 0.97
            nw = max(1, int(src.width * scale))
            nh = max(1, int(src.height * scale))
            scaled = src.resize((nw, nh), Image.Resampling.LANCZOS)
            ox = mx + max(0, (left_inner_w - nw) // 2)
            oy = my + max(0, (zone_h - nh) // 2)
            out.paste(scaled, (ox, oy))
            draw = ImageDraw.Draw(out)
            tx0 = split_x + mx // 2
            max_tw = w - mx - tx0
            name_sz = max(14, min(48, max_tw // 7))
            body_sz = max(12, int(name_sz * 0.72))
            for shrink in range(30):
                font_name = _builder1_headline_font(name_sz)
                font_body = _builder1_headline_font(body_sz)
                if pu:
                    lines_r = _wrap_words_to_width(draw, rest.split(), font_body, max_tw) if rest else []
                    block: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]] = [(pu, font_name)]
                    for ln in lines_r:
                        block.append((ln, font_body))
                else:
                    block = [(text, font_body)]
                heights = [draw.textbbox((0, 0), s, font=f)[3] - draw.textbbox((0, 0), s, font=f)[1] + 6 for s, f in block]
                total_block = sum(heights)
                if total_block <= zone_h - 4:
                    break
                name_sz = max(11, name_sz - 2)
                body_sz = max(10, int(name_sz * 0.72))
            y = my + max(0, (zone_h - total_block) // 2)
            for s, fnt in block:
                twl = _text_width(draw, s, fnt)
                x = tx0 + max(0, (max_tw - twl) // 2)
                draw.text(
                    (min(x, w - mx - 1), min(y, h - my - 1)),
                    s,
                    font=fnt,
                    fill=(16, 16, 16),
                    stroke_width=stroke,
                    stroke_fill=(248, 248, 248),
                )
                bb = draw.textbbox((0, 0), s, font=fnt)
                y += bb[3] - bb[1] + 8
        else:
            bbox = _estimate_foreground_bbox_full(src)
            use_side = w == h and _square_allow_side_by_side(w, h, bbox)
            if use_side:
                split_x = int(w * 0.50)
                left_inner_w = split_x - mx - mx // 2
                zone_h = h - 2 * my
                scale = min(left_inner_w / max(1, src.width), zone_h / max(1, src.height)) * 0.97
                nw = max(1, int(src.width * scale))
                nh = max(1, int(src.height * scale))
                scaled = src.resize((nw, nh), Image.Resampling.LANCZOS)
                ox = mx + max(0, (left_inner_w - nw) // 2)
                oy = my + max(0, (zone_h - nh) // 2)
                out.paste(scaled, (ox, oy))
                draw = ImageDraw.Draw(out)
                tx0 = split_x + mx // 2
                max_tw = w - mx - tx0
                name_sz = max(14, min(48, max_tw // 7))
                body_sz = max(12, int(name_sz * 0.72))
                for shrink in range(30):
                    font_name = _builder1_headline_font(name_sz)
                    font_body = _builder1_headline_font(body_sz)
                    if pu:
                        lines_r = _wrap_words_to_width(draw, rest.split(), font_body, max_tw) if rest else []
                        block = [(pu, font_name)] + [(ln, font_body) for ln in lines_r]
                    else:
                        block = [(text, font_body)]
                    heights = [draw.textbbox((0, 0), s, font=f)[3] - draw.textbbox((0, 0), s, font=f)[1] + 6 for s, f in block]
                    total_block = sum(heights)
                    if total_block <= zone_h - 4:
                        break
                    name_sz = max(11, name_sz - 2)
                    body_sz = max(10, int(name_sz * 0.72))
                y = my + max(0, (zone_h - total_block) // 2)
                for s, fnt in block:
                    twl = _text_width(draw, s, fnt)
                    x = tx0 + max(0, (max_tw - twl) // 2)
                    draw.text(
                        (min(x, w - mx - 1), min(y, h - my - 1)),
                        s,
                        font=fnt,
                        fill=(16, 16, 16),
                        stroke_width=stroke,
                        stroke_fill=(248, 248, 248),
                    )
                    bb = draw.textbbox((0, 0), s, font=fnt)
                    y += bb[3] - bb[1] + 8
            else:
                # Portrait headline-below: larger visual, tighter gutters; square-below unchanged.
                is_portrait = h > w
                if is_portrait:
                    mx_u = max(4, int(round(w * 0.065)))
                    my_u = max(4, int(round(h * 0.07)))
                    visual_frac = 0.89
                    min_text = int(h * 0.086)
                    scale_mul = 1.0
                else:
                    mx_u, my_u = mx, my
                    visual_frac = 0.66
                    min_text = int(h * 0.12)
                    scale_mul = 0.98
                top_h = int(h * visual_frac)
                text_band = h - top_h
                if text_band < min_text:
                    top_h = h - min_text
                    text_band = h - top_h
                tw = w - 2 * mx_u
                th = top_h - my_u
                scale = min(tw / max(1, src.width), th / max(1, src.height)) * scale_mul
                nw = max(1, int(src.width * scale))
                nh = max(1, int(src.height * scale))
                scaled = src.resize((nw, nh), Image.Resampling.LANCZOS)
                ox = mx_u + max(0, (tw - nw) // 2)
                oy = my_u + max(0, (th - nh) // 2)
                out.paste(scaled, (ox, oy))
                draw = ImageDraw.Draw(out)
                if is_portrait:
                    gap_top = max(2, int(round(h * 0.005)))
                    mid_y = top_h + gap_top + max(0, (text_band - gap_top) // 2)
                    y_text_min = top_h + gap_top
                else:
                    mid_y = top_h + text_band // 2
                    y_text_min = top_h + 2
                max_w = w - 2 * mx_u
                y_text_max = h - my_u - 2
                name_sz = max(14, min(46, max_w // 8))
                base_sz = max(11, int(name_sz // 1.35))
                for _ in range(56):
                    fn, fr, fo, two = _shrink_fonts_to_width(draw, pu, rest, line, text, max_w, name_sz, base_sz)
                    bb = _headline_one_line_union_bbox(
                        draw, mid_y, w, mx_u, pu, rest, line, stroke, fn, fr, fo, two
                    )
                    if bb[1] >= y_text_min and bb[3] <= y_text_max:
                        break
                    name_sz = max(8, name_sz - 2)
                    base_sz = max(7, base_sz - 1)
                _draw_headline_one_line_centered(
                    draw, mid_y, w, mx_u, pu, rest, line, text, stroke, fn, fr, fo, two
                )

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
    product_name: str,
    product_description: str,
    session_id: Optional[str] = None,
    *,
    image_size: Optional[str] = None,
) -> dict:
    """Orchestrate Builder1: concept → mode → prompts → headline → marketing text.

    ``image_size`` must be one of 1024x1024, 1536x1024, 1024x1536 when set; otherwise
    resolution follows context (``set_builder1_request_image_size``) then env vars.
    """
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
    image_bytes = generate_image_bytes(image_prompt, image_size=image_size)
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
