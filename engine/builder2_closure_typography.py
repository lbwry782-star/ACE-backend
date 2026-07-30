"""
Builder2 closure-card typography contract — Ogen fonts, adaptive sizing, metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.video_language import normalize_video_content_language

BUILDER2_CLOSURE_TYPOGRAPHY_VERSION = "builder2_closure_typography_v3"
BUILDER2_CLOSURE_TYPOGRAPHY_V2 = "builder2_closure_typography_v2"
BUILDER2_CLOSURE_TYPOGRAPHY_V1 = "builder2_closure_typography_v1"

PRODUCT_FONT_RELATIVE = Path("assets") / "fonts" / "OgenBlack.ttf"
SLOGAN_FONT_RELATIVE = Path("assets") / "fonts" / "OgenBold.ttf"

TARGET_PRODUCT_SLOGAN_SIZE_RATIO = 1.55
MIN_ACCEPTABLE_DOMINANCE_RATIO = 1.25

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
HORIZONTAL_SAFE_MARGIN_PX = 80
VERTICAL_SAFE_MARGIN_PX = 60
PRODUCT_SLOGAN_BLOCK_GAP_PX = 9
PREVIOUS_PRODUCT_SLOGAN_BLOCK_GAP_PX = 18
LEGACY_PRODUCT_SLOGAN_BLOCK_GAP_PX = 36
MAX_PRODUCT_LINES = 2
MAX_SLOGAN_LINES = 2
BASE_SLOGAN_FONT_SIZE = 44
MIN_SLOGAN_FONT_SIZE = 28
MIN_PRODUCT_FONT_SIZE = 36
LINE_HEIGHT_FACTOR = 1.18

CLOSURE_BACKGROUND_STYLE_VERSION = "builder2_closure_background_black_purple_v1"
# Dark near-black with a subtle purple cast (FFmpeg 0xRRGGBB).
CLOSURE_BACKGROUND_FFMPEG_COLOR = "0x0E0014"

CLOSURE_TEXT_REVEAL_VERSION = "builder2_closure_text_reveal_upward_v1"
CLOSURE_TEXT_REVEAL_TRAVEL_PX = 52
CLOSURE_PRODUCT_REVEAL_START_S = 0.20
CLOSURE_PRODUCT_REVEAL_DURATION_S = 0.65
CLOSURE_SLOGAN_REVEAL_START_S = 0.78
CLOSURE_SLOGAN_REVEAL_DURATION_S = 0.65
CLOSURE_PRODUCT_LINE_STAGGER_S = 0.08

_CLOSURE_PUNCTUATION_PATTERN = re.compile(
    r"[\.\,\:\;\!\?\"\'`\(\)\[\]\{\}\/\\|\-–—―…·•«»„”“‘’׳״]+"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_builder2_closure_product_font_path() -> Path:
    path = _repo_root() / PRODUCT_FONT_RELATIVE
    if not path.is_file():
        raise Builder2TournamentError("builder2_closure_product_font_missing")
    return path


def resolve_builder2_closure_slogan_font_path() -> Path:
    path = _repo_root() / SLOGAN_FONT_RELATIVE
    if not path.is_file():
        raise Builder2TournamentError("builder2_closure_slogan_font_missing")
    return path


def validate_builder2_closure_font_assets() -> Tuple[Path, Path]:
    product_path = resolve_builder2_closure_product_font_path()
    slogan_path = resolve_builder2_closure_slogan_font_path()
    if not _font_readable(product_path):
        raise Builder2TournamentError("builder2_closure_font_unreadable:product")
    if not _font_readable(slogan_path):
        raise Builder2TournamentError("builder2_closure_font_unreadable:slogan")
    return product_path, slogan_path


def _font_readable(font_path: Path) -> bool:
    try:
        from PIL import ImageFont

        ImageFont.truetype(str(font_path), 24)
        return True
    except Exception:
        return False


def font_supports_hebrew_glyphs(font_path: Path, *, size: int = 24) -> bool:
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(font_path), size)
        sample = "\u05d0\u05d1"
        if hasattr(font, "getlength"):
            width = font.getlength(sample)
            return width > 0
        bbox = font.getbbox(sample)
        return bbox is not None and (bbox[2] - bbox[0]) > 0
    except Exception:
        return False


def _has_hebrew_letter(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", text or ""))


def _sanitize_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sanitize_closure_render_text(text: str) -> str:
    """Remove punctuation for visible closure copy; preserve letters, digits, and spaces."""
    cleaned = _CLOSURE_PUNCTUATION_PATTERN.sub(" ", text or "")
    return _sanitize_line(cleaned)


def closure_punctuation_removed(original: str, rendered: str) -> bool:
    return sanitize_closure_render_text(original) == _sanitize_line(rendered)


def closure_card_lavfi_background(*, width: int, height: int, duration: float) -> str:
    return f"color=c={CLOSURE_BACKGROUND_FFMPEG_COLOR}:s={width}x{height}:d={duration:.6f}"


def _text_width_px(font_path: Path, text: str, fontsize: int) -> int:
    from PIL import ImageFont

    font = ImageFont.truetype(str(font_path), fontsize)
    if hasattr(font, "getlength"):
        return max(1, int(round(font.getlength(text))))
    bbox = font.getbbox(text)
    return max(1, bbox[2] - bbox[0])


def _line_height_px(fontsize: int) -> int:
    return max(1, int(round(fontsize * LINE_HEIGHT_FACTOR)))


def _wrap_text_lines(
    text: str,
    *,
    font_path: Path,
    fontsize: int,
    max_width_px: int,
    max_lines: int,
) -> List[str]:
    cleaned = _sanitize_line(text)
    if not cleaned:
        return []
    words = cleaned.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_width_px(font_path, candidate, fontsize) <= max_width_px:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            # Single word wider than max — hard split by characters deterministically.
            chunk = ""
            for ch in word:
                probe = chunk + ch
                if _text_width_px(font_path, probe, fontsize) <= max_width_px or not chunk:
                    chunk = probe
                else:
                    lines.append(chunk)
                    chunk = ch
                    if len(lines) >= max_lines:
                        break
            current = chunk
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


@dataclass(frozen=True)
class ClosureTypographyLineSpec:
    text: str
    font_path: Path
    fontsize: int
    y_px: int
    use_text_shaping: bool
    role: str
    reveal_start_seconds: float = 0.0
    reveal_duration_seconds: float = CLOSURE_PRODUCT_REVEAL_DURATION_S
    reveal_travel_px: int = CLOSURE_TEXT_REVEAL_TRAVEL_PX


@dataclass(frozen=True)
class ClosureTypographyLayout:
    product_lines: Tuple[str, ...]
    slogan_lines: Tuple[str, ...]
    product_font_path: Path
    slogan_font_path: Path
    requested_product_font_size: int
    requested_slogan_font_size: int
    effective_product_font_size: int
    effective_slogan_font_size: int
    effective_dominance_ratio: float
    product_line_count: int
    slogan_line_count: int
    safe_margins_satisfied: bool
    original_product_text: str = ""
    original_slogan_text: str = ""
    rendered_product_text: str = ""
    rendered_slogan_text: str = ""
    product_slogan_gap_px: int = PRODUCT_SLOGAN_BLOCK_GAP_PX
    configured_closure_segment_duration_seconds: float = 0.0
    line_specs: Tuple[ClosureTypographyLineSpec, ...] = field(default_factory=tuple)

    def metadata(self, *, measured_final_duration_seconds: float | None = None) -> Dict[str, Any]:
        from engine.builder2_new_format_config import (
            resolve_builder2_effective_closure_segment_duration_seconds,
            resolve_builder2_final_video_duration_seconds,
        )

        configured_closure = float(
            self.configured_closure_segment_duration_seconds
            or resolve_builder2_effective_closure_segment_duration_seconds()
        )
        product_specs = [spec for spec in self.line_specs if spec.role == "product"]
        slogan_specs = [spec for spec in self.line_specs if spec.role == "slogan"]
        return {
            "typographyContractVersion": BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            "productFontAsset": PRODUCT_FONT_RELATIVE.as_posix(),
            "sloganFontAsset": SLOGAN_FONT_RELATIVE.as_posix(),
            "requestedProductFontSize": self.requested_product_font_size,
            "effectiveProductFontSize": self.effective_product_font_size,
            "requestedSloganFontSize": self.requested_slogan_font_size,
            "effectiveSloganFontSize": self.effective_slogan_font_size,
            "effectiveDominanceRatio": round(self.effective_dominance_ratio, 4),
            "effectiveProductSloganGapPx": self.product_slogan_gap_px,
            "productLineCount": self.product_line_count,
            "sloganLineCount": self.slogan_line_count,
            "safeMarginsSatisfied": self.safe_margins_satisfied,
            "canonicalProductNameText": self.original_product_text,
            "canonicalSloganText": self.original_slogan_text,
            "renderedClosureProductText": self.rendered_product_text,
            "renderedClosureSloganText": self.rendered_slogan_text,
            "closurePunctuationSanitizationApplied": True,
            "closureBackgroundStyleVersion": CLOSURE_BACKGROUND_STYLE_VERSION,
            "closureBackgroundFfmpegColor": CLOSURE_BACKGROUND_FFMPEG_COLOR,
            "configuredClosureSegmentDurationSeconds": configured_closure,
            "configuredFinalVideoDurationSeconds": resolve_builder2_final_video_duration_seconds(),
            "measuredFinalDurationSeconds": measured_final_duration_seconds,
            "closureTextRevealVersion": CLOSURE_TEXT_REVEAL_VERSION,
            "closureTextRevealEnabled": True,
            "productTextRevealApplied": bool(product_specs),
            "sloganTextRevealApplied": bool(slogan_specs),
            "closureTextRevealTravelPx": CLOSURE_TEXT_REVEAL_TRAVEL_PX,
            "productNameRenderedAsPlainText": True,
            "productNameFontRole": "primary",
            "sloganFontRole": "secondary",
            "sloganRenderedExactlyOnce": True,
            "separateHeadlineRendered": False,
            "brandNameDominanceSatisfied": self.effective_dominance_ratio >= MIN_ACCEPTABLE_DOMINANCE_RATIO,
        }


def fit_builder2_closure_typography(
    *,
    product_name: str,
    slogan: str,
    language: str = "he",
    closure_segment_duration_seconds: float | None = None,
) -> ClosureTypographyLayout:
    product_font, slogan_font = validate_builder2_closure_font_assets()
    lang = normalize_video_content_language(language)
    original_product = _sanitize_line(product_name)
    original_slogan = _sanitize_line(slogan)
    product_text = sanitize_closure_render_text(original_product)
    slogan_text = sanitize_closure_render_text(original_slogan)
    if not product_text or not slogan_text:
        raise Builder2TournamentError("builder2_closure_missing_text")

    gap_px = PRODUCT_SLOGAN_BLOCK_GAP_PX

    safe_width = CANVAS_WIDTH - (2 * HORIZONTAL_SAFE_MARGIN_PX)
    safe_height = CANVAS_HEIGHT - (2 * VERTICAL_SAFE_MARGIN_PX)

    best: ClosureTypographyLayout | None = None
    for step in range(40):
        scale = 1.0 - (step * 0.025)
        slogan_size = max(MIN_SLOGAN_FONT_SIZE, int(round(BASE_SLOGAN_FONT_SIZE * scale)))
        product_size = max(MIN_PRODUCT_FONT_SIZE, int(round(slogan_size * TARGET_PRODUCT_SLOGAN_SIZE_RATIO)))
        ratio = product_size / max(1, slogan_size)
        if ratio < MIN_ACCEPTABLE_DOMINANCE_RATIO:
            product_size = max(MIN_PRODUCT_FONT_SIZE, int(round(slogan_size * MIN_ACCEPTABLE_DOMINANCE_RATIO)))
            ratio = product_size / max(1, slogan_size)

        product_lines = _wrap_text_lines(
            product_text,
            font_path=product_font,
            fontsize=product_size,
            max_width_px=safe_width,
            max_lines=MAX_PRODUCT_LINES,
        )
        slogan_lines = _wrap_text_lines(
            slogan_text,
            font_path=slogan_font,
            fontsize=slogan_size,
            max_width_px=safe_width,
            max_lines=MAX_SLOGAN_LINES,
        )
        if not product_lines or not slogan_lines:
            continue

        product_block_h = sum(_line_height_px(product_size) for _ in product_lines)
        if len(product_lines) > 1:
            product_block_h += int((len(product_lines) - 1) * product_size * 0.08)
        slogan_block_h = sum(_line_height_px(slogan_size) for _ in slogan_lines)
        if len(slogan_lines) > 1:
            slogan_block_h += int((len(slogan_lines) - 1) * slogan_size * 0.08)
        total_h = product_block_h + gap_px + slogan_block_h
        if total_h > safe_height:
            continue

        width_ok = True
        for line in product_lines:
            if _text_width_px(product_font, line, product_size) > safe_width:
                width_ok = False
                break
        if width_ok:
            for line in slogan_lines:
                if _text_width_px(slogan_font, line, slogan_size) > safe_width:
                    width_ok = False
                    break
        if not width_ok:
            continue

        start_y = int(round((CANVAS_HEIGHT - total_h) / 2))
        if start_y < VERTICAL_SAFE_MARGIN_PX:
            continue
        if start_y + total_h > CANVAS_HEIGHT - VERTICAL_SAFE_MARGIN_PX:
            continue

        line_specs: List[ClosureTypographyLineSpec] = []
        y_cursor = start_y
        product_shaping = lang == "he" or _has_hebrew_letter(product_text)
        for index, line in enumerate(product_lines):
            line_specs.append(
                ClosureTypographyLineSpec(
                    text=line,
                    font_path=product_font,
                    fontsize=product_size,
                    y_px=y_cursor,
                    use_text_shaping=product_shaping,
                    role="product",
                    reveal_start_seconds=CLOSURE_PRODUCT_REVEAL_START_S + (index * CLOSURE_PRODUCT_LINE_STAGGER_S),
                    reveal_duration_seconds=CLOSURE_PRODUCT_REVEAL_DURATION_S,
                    reveal_travel_px=CLOSURE_TEXT_REVEAL_TRAVEL_PX,
                )
            )
            y_cursor += _line_height_px(product_size)
        y_cursor += gap_px
        slogan_shaping = lang == "he" or _has_hebrew_letter(slogan_text)
        for index, line in enumerate(slogan_lines):
            line_specs.append(
                ClosureTypographyLineSpec(
                    text=line,
                    font_path=slogan_font,
                    fontsize=slogan_size,
                    y_px=y_cursor,
                    use_text_shaping=slogan_shaping,
                    role="slogan",
                    reveal_start_seconds=CLOSURE_SLOGAN_REVEAL_START_S + (index * CLOSURE_PRODUCT_LINE_STAGGER_S),
                    reveal_duration_seconds=CLOSURE_SLOGAN_REVEAL_DURATION_S,
                    reveal_travel_px=CLOSURE_TEXT_REVEAL_TRAVEL_PX,
                )
            )
            y_cursor += _line_height_px(slogan_size)

        from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds

        configured_closure = resolve_builder2_effective_closure_segment_duration_seconds(
            closure_segment_duration_seconds
        )

        best = ClosureTypographyLayout(
            product_lines=tuple(product_lines),
            slogan_lines=tuple(slogan_lines),
            product_font_path=product_font,
            slogan_font_path=slogan_font,
            requested_product_font_size=int(round(BASE_SLOGAN_FONT_SIZE * TARGET_PRODUCT_SLOGAN_SIZE_RATIO)),
            requested_slogan_font_size=BASE_SLOGAN_FONT_SIZE,
            effective_product_font_size=product_size,
            effective_slogan_font_size=slogan_size,
            effective_dominance_ratio=ratio,
            product_line_count=len(product_lines),
            slogan_line_count=len(slogan_lines),
            safe_margins_satisfied=True,
            original_product_text=original_product,
            original_slogan_text=original_slogan,
            rendered_product_text=" ".join(product_lines),
            rendered_slogan_text=" ".join(slogan_lines),
            product_slogan_gap_px=gap_px,
            configured_closure_segment_duration_seconds=configured_closure,
            line_specs=tuple(line_specs),
        )
        break

    if best is None:
        raise Builder2TournamentError("builder2_closure_text_overflow")
    if best.effective_dominance_ratio < MIN_ACCEPTABLE_DOMINANCE_RATIO:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied")
    return best


def _drawtext_reveal_y_expression(spec: ClosureTypographyLineSpec) -> str:
    final_y = spec.y_px
    travel = spec.reveal_travel_px
    start = spec.reveal_start_seconds
    duration = max(0.05, spec.reveal_duration_seconds)
    return f"{final_y}+({travel})*max(0\\,1-min(1\\,(t-{start:.3f})/{duration:.3f}))"


def build_closure_card_drawtext_filter(
    layout: ClosureTypographyLayout,
    *,
    textfile_paths: Sequence[Path],
    ffmpeg_path_filter: Any,
) -> str:
    if len(textfile_paths) != len(layout.line_specs):
        raise Builder2TournamentError("builder2_closure_text_overflow")
    parts: List[str] = []
    for spec, text_path in zip(layout.line_specs, textfile_paths):
        font_e = ffmpeg_path_filter(spec.font_path)
        text_e = ffmpeg_path_filter(text_path)
        shaping = ":text_shaping=1" if spec.use_text_shaping else ""
        y_expr = _drawtext_reveal_y_expression(spec)
        parts.append(
            f"drawtext=fontfile='{font_e}':textfile='{text_e}':"
            f"fontcolor=white:fontsize={spec.fontsize}:x=(w-text_w)/2:y='{y_expr}'"
            f":borderw=2:bordercolor=black@0.35{shaping}"
        )
    return ",".join(parts)


def stamp_closure_typography_metadata(
    target: Dict[str, Any],
    layout: ClosureTypographyLayout,
    *,
    measured_final_duration_seconds: float | None = None,
) -> None:
    target.update(layout.metadata(measured_final_duration_seconds=measured_final_duration_seconds))


def current_closure_typography_version(media: Dict[str, Any]) -> str:
    return str(media.get("closureTypographyContractVersion") or media.get("typographyContractVersion") or "").strip()


def closure_typography_upgrade_needed(
    media: Dict[str, Any],
    *,
    requested_version: str = BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
) -> bool:
    current = current_closure_typography_version(media)
    if not current:
        return True
    return current != requested_version


def verify_closure_typography_metadata(metadata: Dict[str, Any]) -> None:
    if metadata.get("typographyContractVersion") != BUILDER2_CLOSURE_TYPOGRAPHY_VERSION:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:typography_version")
    if metadata.get("closurePunctuationSanitizationApplied") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:punctuation")
    if metadata.get("closureBackgroundStyleVersion") != CLOSURE_BACKGROUND_STYLE_VERSION:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:background")
    if metadata.get("closureTextRevealEnabled") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:animation")
    if metadata.get("closureTextRevealVersion") != CLOSURE_TEXT_REVEAL_VERSION:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:animation_version")
    if metadata.get("brandNameDominanceSatisfied") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied")
    if metadata.get("separateHeadlineRendered") is True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:headline_layer")
    if metadata.get("sloganRenderedExactlyOnce") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:slogan_count")
