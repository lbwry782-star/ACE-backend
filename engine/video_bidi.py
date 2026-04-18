"""
Bidirectional text for video outputs.

- Marketing copy (API / UI): LRI+PDI around Latin islands (finalize_hebrew_mixed_bidi_for_display).
- ffmpeg drawtext headline: visible text only in headline.txt (no bidi isolate/control chars).
  English + Hebrew uses dual drawtext positioning in video_headline_postprocess (see OverlayHeadlinePrep).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import List, NamedTuple, Tuple

logger = logging.getLogger(__name__)

from engine.video_language import (
    is_english_only_product_name_script,
    normalize_video_content_language,
)

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


class OverlayHeadlinePrep(NamedTuple):
    """Planner/API headline stays plain; overlay path may split for dual drawtext (no bidi controls in files)."""

    text_plain: str
    strategy: str
    render_mode: str  # plain_text | dual_drawtext
    dual_latin: str
    dual_hebrew: str


def _strip_planner_separators_for_overlay(s: str) -> str:
    """
    Remove stray planner joiners before recomposing the overlay.
    Replaces -, –, —, ·, |, : with spaces, then collapses whitespace.
    Product + remainder use a single normal space in the visible line (no comma or other separator).
    """
    if not s:
        return ""
    t = s
    for ch in ("-", "\u2013", "\u2014", "\u00b7", "|", ":"):
        t = t.replace(ch, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def prepare_ffmpeg_overlay_headline(
    headline: str,
    *,
    content_language: str,
    canonical_name: str,
) -> OverlayHeadlinePrep:
    """
    Normalize overlay headline: visible characters only (no LRI/RLI/PDI or other bidi controls in output).

    Product name + remainder → dual_drawtext when two parts are needed (English or Hebrew product name);
    postprocess draws product larger. No comma or punctuation between parts in stored text.
    Returns OverlayHeadlinePrep with text_plain equal to the full visible line for logs/API consistency.
    """
    raw_in = headline or ""
    h = unicodedata.normalize("NFC", _strip_bidi_embedding_marks(raw_in)).strip()
    cn = unicodedata.normalize("NFC", (canonical_name or "").strip())
    lang = normalize_video_content_language(content_language)

    if lang != "he":
        out = _strip_planner_separators_for_overlay(h) if h else raw_in.strip()
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_STRIPPED_INPUT=%s",
            json.dumps(out, ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_REMAINDER=%s", json.dumps("", ensure_ascii=False))
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
            json.dumps(out, ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
        tp = out if out else raw_in.strip()
        return OverlayHeadlinePrep(tp, "overlay_non_hebrew_strip_marks", "plain_text", "", "")

    if not h:
        logger.info("VIDEO_HEADLINE_OVERLAY_STRIPPED_INPUT=%s", json.dumps("", ensure_ascii=False))
        logger.info("VIDEO_HEADLINE_OVERLAY_REMAINDER=%s", json.dumps("", ensure_ascii=False))
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
            json.dumps(raw_in.strip(), ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
        return OverlayHeadlinePrep(raw_in.strip(), "overlay_empty", "plain_text", "", "")

    h_clean = _strip_planner_separators_for_overlay(h)
    logger.info(
        "VIDEO_HEADLINE_OVERLAY_STRIPPED_INPUT=%s",
        json.dumps(h_clean, ensure_ascii=False),
    )

    if not cn:
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_REMAINDER=%s",
            json.dumps(h_clean, ensure_ascii=False),
        )
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
            json.dumps(h_clean, ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
        return OverlayHeadlinePrep(
            h_clean, "overlay_no_canonical_name", "plain_text", "", ""
        )

    def _tail_after_prefix(full: str, prefix: str) -> str:
        hs = full.strip()
        pr = prefix.strip()
        if len(hs) < len(pr) or hs[: len(pr)].lower() != pr.lower():
            return ""
        return hs[len(pr) :].lstrip()

    # Latin-only product name: remainder is Hebrew (dual overlay: product right, larger).
    if is_english_only_product_name_script(cn):
        if not _contains_hebrew_letter(h_clean):
            logger.info(
                "VIDEO_HEADLINE_OVERLAY_REMAINDER=%s",
                json.dumps("", ensure_ascii=False),
            )
            logger.info(
                "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
                json.dumps(h_clean, ensure_ascii=False),
            )
            logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
            return OverlayHeadlinePrep(
                h_clean, "overlay_latin_headline_strip_only", "plain_text", "", ""
            )

        remainder = _tail_after_prefix(h_clean, cn)
        remainder = re.sub(r"^[\s,·\u00b7\u2022•:;|–—−\-]+", "", remainder)
        remainder = _strip_planner_separators_for_overlay(remainder)
        remainder = re.sub(r"\s+", " ", remainder).strip()

        logger.info(
            "VIDEO_HEADLINE_OVERLAY_REMAINDER=%s",
            json.dumps(remainder, ensure_ascii=False),
        )

        if not remainder:
            logger.info(
                "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
                json.dumps(cn, ensure_ascii=False),
            )
            logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
            return OverlayHeadlinePrep(cn, "overlay_latin_canonical_only", "plain_text", "", "")

        plain = f"{cn} {remainder}"
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
            json.dumps(plain, ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=dual_drawtext")
        return OverlayHeadlinePrep(
            plain,
            "overlay_latin_space_hebrew_remainder",
            "dual_drawtext",
            cn,
            remainder,
        )

    # Hebrew script product name: remainder is Hebrew (dual overlay).
    if _contains_hebrew_letter(cn):
        tail = _tail_after_prefix(h_clean, cn)
        tail = re.sub(r"^[\s,·\u00b7\u2022•:;|–—−\-]+", "", tail)
        tail = _strip_planner_separators_for_overlay(tail)
        tail = re.sub(r"\s+", " ", tail).strip()

        logger.info(
            "VIDEO_HEADLINE_OVERLAY_REMAINDER=%s",
            json.dumps(tail, ensure_ascii=False),
        )

        if not tail or not _contains_hebrew_letter(tail):
            logger.info(
                "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
                json.dumps(h_clean, ensure_ascii=False),
            )
            logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
            return OverlayHeadlinePrep(
                h_clean,
                "overlay_hebrew_product_plain_or_tail_mismatch",
                "plain_text",
                "",
                "",
            )

        plain = f"{cn} {tail}"
        logger.info(
            "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
            json.dumps(plain, ensure_ascii=False),
        )
        logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=dual_drawtext")
        return OverlayHeadlinePrep(
            plain,
            "overlay_hebrew_product_space_remainder",
            "dual_drawtext",
            cn,
            tail,
        )

    logger.info(
        "VIDEO_HEADLINE_OVERLAY_REMAINDER=%s",
        json.dumps(h_clean, ensure_ascii=False),
    )
    logger.info(
        "VIDEO_HEADLINE_OVERLAY_FINAL_TEXT=%s",
        json.dumps(h_clean, ensure_ascii=False),
    )
    logger.info("VIDEO_HEADLINE_OVERLAY_RENDER_MODE=plain_text")
    return OverlayHeadlinePrep(
        h_clean, "overlay_hebrew_or_mixed_canonical_strip_only", "plain_text", "", ""
    )
