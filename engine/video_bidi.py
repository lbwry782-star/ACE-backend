"""
Bidirectional text: stabilize embedded Latin in Hebrew (RTL) for display / ffmpeg overlay.

Uses U+2066 LEFT-TO-RIGHT ISOLATE + U+2069 POP DIRECTIONAL ISOLATE around Latin runs
(Unicode TR9 embedding); marks are non-printing. LRM-only was insufficient for
multi-word English (e.g. product names) inside RTL paragraphs.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from engine.video_language import normalize_video_content_language

_LRI = "\u2066"  # LEFT-TO-RIGHT ISOLATE
_PDI = "\u2069"  # POP DIRECTIONAL ISOLATE

# Strip prior invisible directional embeddings so re-stabilization is idempotent.
_BIDI_MARKS_TO_RESET = frozenset(
    {
        "\u200e",  # LRM
        "\u200f",  # RLM
        "\u2066",  # LRI
        "\u2067",  # RLI
        "\u2068",  # FSI
        "\u2069",  # PDI
    }
)

# One English/Latin word; then allow space-separated Latin words in one LTR island (e.g. "Shoe Haven", "Air-Flex Pro").
_LATIN_WORD = r"[A-Za-z][A-Za-z0-9\-']*"
_LATIN_ISLAND_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_LATIN_WORD}(?:\s+{_LATIN_WORD})*)(?![A-Za-z0-9])"
)


def _strip_bidi_embedding_marks(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch not in _BIDI_MARKS_TO_RESET)


def _contains_hebrew_letter(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x0590 <= o <= 0x05FF:
            return True
    return False


def finalize_hebrew_mixed_bidi_for_display(
    text: str,
    *,
    content_language: str,
    protected_phrases: Tuple[str, ...] = (),
) -> Tuple[str, bool, List[str]]:
    """
    On Hebrew jobs, wrap embedded Latin in LRI...PDI so word order stays stable (incl. multi-word names).

    - Skips pure Latin strings (no Hebrew letters).
    - Protected phrases (e.g. resolved product name) use word-boundary-safe matches, longest first.
    - Strips existing LRM/LRI/PDI/RLM/FSI so applying twice does not nest isolates.

    Returns (new_text, whether wrapping was applied, list of raw Latin segments wrapped, for logs).
    """
    if normalize_video_content_language(content_language) != "he":
        return text, False, []

    raw = _strip_bidi_embedding_marks(text or "")
    if not raw.strip():
        return text, False, []
    if not _contains_hebrew_letter(raw):
        return raw, False, []

    spans: List[Tuple[int, int]] = []
    used: List[Tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        for us, ue in used:
            if s < ue and us < e:
                return True
        return False

    def add_span(s: int, e: int) -> None:
        if s >= e or overlaps(s, e):
            return
        spans.append((s, e))
        used.append((s, e))

    uniq_phrases = list(dict.fromkeys(p.strip() for p in protected_phrases if (p or "").strip()))
    for p in sorted(uniq_phrases, key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(p) + r"(?![A-Za-z0-9])", re.I)
        for m in pat.finditer(raw):
            add_span(m.start(), m.end())

    for m in _LATIN_ISLAND_RE.finditer(raw):
        add_span(m.start(), m.end())

    if not spans:
        return raw, False, []

    spans.sort(key=lambda x: x[0])
    segments_logged = [raw[s:e] for s, e in spans]

    out: List[str] = []
    pos = 0
    for s, e in spans:
        out.append(raw[pos:s])
        out.append(_LRI)
        out.append(raw[s:e])
        out.append(_PDI)
        pos = e
    out.append(raw[pos:])
    return "".join(out), True, segments_logged


def format_bidi_segments_for_log(segments: List[str]) -> str:
    """JSON list for VIDEO_BIDI_LATIN_SEGMENTS_* logs."""
    return json.dumps(segments, ensure_ascii=False)
