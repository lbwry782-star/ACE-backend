"""
Builder2 video ZIP helper (video + marketing text).
"""
from __future__ import annotations

import io
import zipfile


def build_builder2_video_zip_bytes(video_bytes: bytes, marketing_text: str) -> bytes:
    if not video_bytes:
        raise ValueError("missing_video_bytes")

    text_value = marketing_text if isinstance(marketing_text, str) else str(marketing_text or "")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.mp4", video_bytes)
        zf.writestr("text.txt", text_value.encode("utf-8"))
    return zip_buffer.getvalue()

