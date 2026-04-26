"""
Builder1 ZIP helper (image + marketing text).
"""
from __future__ import annotations

import base64
import io
import zipfile

from PIL import Image


def build_builder1_zip_bytes(image_base64: str, marketing_text: str) -> bytes:
    raw = (image_base64 or "").strip()
    if not raw:
        raise ValueError("missing_image_base64")
    if raw.lower().startswith("data:image/") and "," in raw:
        raw = raw.split(",", 1)[1].strip()

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("invalid_image_base64") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            jpg_buffer = io.BytesIO()
            rgb.save(jpg_buffer, format="JPEG", quality=92)
            jpg_bytes = jpg_buffer.getvalue()
    except Exception as exc:
        raise ValueError("invalid_image_data") from exc

    text_value = marketing_text if isinstance(marketing_text, str) else str(marketing_text or "")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.jpg", jpg_bytes)
        zf.writestr("text.txt", text_value.encode("utf-8"))
    return zip_buffer.getvalue()

