"""
Bidirectional text: stabilize embedded Latin in Hebrew (RTL) for display / ffmpeg overlay.

Uses U+200E LEFT-TO-RIGHT MARK as invisible anchors around Latin segments (no visible change).
"""

from __future__ import annotations

import re
from typing import Tuple

from engine.video_language import normalize_video_content_language

_LRM = "\u200e"

# Latin/English token: starts with ASCII letter; continues with letters, digits, hyphen, apostrophe
# (e.g. Air-Flex, SaaS, McDonald's). Does not match pure-digit runs.
_LATIN_SEGMENT_RE = re.compile(r"[A-Za-z](?:[A-Za-z0-9\-']*)")


def _contains_hebrew_letter(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x0590 <= o <= 0x05FF:
            return True
    return False


def stabilize_hebrew_embedded_latin_bidi(text: str, *, content_language: str) -> Tuple[str, bool]:
    """
    When content language is Hebrew and the string mixes Hebrew with Latin segments,
    wrap each Latin segment with LRM so RTL layout keeps English tokens visually stable.

    Does not change: pure Hebrew, pure Latin (no Hebrew letters), or non-Hebrew jobs.
    Returns (possibly_updated_text, whether any LRM was inserted).
    """
    if normalize_video_content_language(content_language) != "he":
        return text, False
    raw = text or ""
    if not raw.strip():
        return text, False
    if not _contains_hebrew_letter(raw):
        return text, False

    out: list[str] = []
    last_end = 0
    applied = False
    for m in _LATIN_SEGMENT_RE.finditer(raw):
        start, end = m.span()
        if start > 0 and raw[start - 1] == _LRM and end < len(raw) and raw[end] == _LRM:
            continue
        out.append(raw[last_end:start])
        seg = m.group(0)
        out.append(_LRM)
        out.append(seg)
        out.append(_LRM)
        last_end = end
        applied = True
    if not applied:
        return text, False
    out.append(raw[last_end:])
    return "".join(out), True
