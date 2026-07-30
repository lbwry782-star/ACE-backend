"""
Builder2 closure-card typography contract — Ogen fonts, adaptive sizing, masked reveal.
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
PRODUCT_SLOGAN_BLOCK_GAP_PX = 16
TARGET_VISIBLE_INK_GAP_PX = 16
MIN_VISIBLE_INK_GAP_PX = 15
MAX_VISIBLE_INK_GAP_PX = 17
PREVIOUS_PRODUCT_SLOGAN_BLOCK_GAP_PX = 9
LEGACY_PRODUCT_SLOGAN_BLOCK_GAP_PX = 36
MAX_PRODUCT_LINES = 2
MAX_SLOGAN_LINES = 2
BASE_SLOGAN_FONT_SIZE = 44
MIN_SLOGAN_FONT_SIZE = 28
MIN_PRODUCT_FONT_SIZE = 36
LINE_HEIGHT_FACTOR = 1.18

CLOSURE_BACKGROUND_STYLE_VERSION = "builder2_closure_background_black_purple_v1"
CLOSURE_BACKGROUND_FFMPEG_COLOR = "0x0E0014"

CLOSURE_TEXT_REVEAL_VERSION = "builder2_closure_text_masked_reveal_upward_ease_out_cubic_v1"
CLOSURE_TEXT_REVEAL_EASING = "ease_out_cubic"
REVEAL_PROGRESS_FUNCTION_VERSION = "builder2_closure_reveal_ease_out_cubic_v1"
REVEAL_WINDOW_HORIZONTAL_PAD_PX = 12
REVEAL_WINDOW_TOP_PAD_PX = 2
REVEAL_WINDOW_BOTTOM_PAD_PX = 4
REVEAL_DRAWTEXT_BORDER_PX = 2
REVEAL_HIDDEN_EXTRA_PAD_PX = REVEAL_DRAWTEXT_BORDER_PX + 8
# FFmpeg rendered ink-to-ink gap is smaller than PIL layout gap by this amount.
CLOSURE_FFMPEG_RENDERED_INK_GAP_ADJUSTMENT_PX = 6
CLOSURE_PRODUCT_REVEAL_START_S = 0.20
CLOSURE_PRODUCT_REVEAL_DURATION_S = 0.65
CLOSURE_SLOGAN_REVEAL_DURATION_S = 0.65
CLOSURE_PRODUCT_LINE_STAGGER_S = 0.08
SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO = 0.50
REVEAL_STAGGER_RULE_VERSION = "builder2_closure_reveal_stagger_half_product_travel_v1"

def closure_reveal_linear_progress_for_eased_travel_ratio(travel_ratio: float) -> float:
    """Inverse ease-out cubic: linear p where eased spatial travel equals travel_ratio."""
    target = max(0.0, min(1.0, travel_ratio))
    return 1.0 - (1.0 - target) ** (1.0 / 3.0)


def closure_reveal_derived_slogan_start_seconds(
    *,
    product_start_seconds: float = CLOSURE_PRODUCT_REVEAL_START_S,
    product_duration_seconds: float = CLOSURE_PRODUCT_REVEAL_DURATION_S,
    travel_ratio: float = SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO,
) -> float:
    linear_progress = closure_reveal_linear_progress_for_eased_travel_ratio(travel_ratio)
    return product_start_seconds + (product_duration_seconds * linear_progress)


CLOSURE_SLOGAN_REVEAL_START_S = closure_reveal_derived_slogan_start_seconds()


def closure_reveal_product_travel_ratio_at_timestamp(
    *,
    timestamp_seconds: float,
    product_start_seconds: float = CLOSURE_PRODUCT_REVEAL_START_S,
    product_duration_seconds: float = CLOSURE_PRODUCT_REVEAL_DURATION_S,
) -> float:
    start = product_start_seconds
    duration = max(0.05, product_duration_seconds)
    if timestamp_seconds <= start:
        return 0.0
    if timestamp_seconds >= start + duration:
        return 1.0
    linear_progress = max(0.0, min(1.0, (timestamp_seconds - start) / duration))
    return closure_reveal_eased_progress(linear_progress)


def closure_reveal_product_still_moving_at_timestamp(
    *,
    timestamp_seconds: float,
    product_start_seconds: float = CLOSURE_PRODUCT_REVEAL_START_S,
    product_duration_seconds: float = CLOSURE_PRODUCT_REVEAL_DURATION_S,
) -> bool:
    return timestamp_seconds < (product_start_seconds + product_duration_seconds)


def closure_reveal_product_slogan_overlap_seconds(
    *,
    product_start_seconds: float = CLOSURE_PRODUCT_REVEAL_START_S,
    product_duration_seconds: float = CLOSURE_PRODUCT_REVEAL_DURATION_S,
    slogan_start_seconds: float = CLOSURE_SLOGAN_REVEAL_START_S,
) -> float:
    product_end = product_start_seconds + product_duration_seconds
    if slogan_start_seconds >= product_end:
        return 0.0
    return product_end - slogan_start_seconds


def closure_reveal_stable_reading_hold_seconds(
    *,
    closure_duration_seconds: float,
    slogan_start_seconds: float = CLOSURE_SLOGAN_REVEAL_START_S,
    slogan_duration_seconds: float = CLOSURE_SLOGAN_REVEAL_DURATION_S,
) -> float:
    return closure_duration_seconds - (slogan_start_seconds + slogan_duration_seconds)


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
    cleaned = _CLOSURE_PUNCTUATION_PATTERN.sub(" ", text or "")
    return _sanitize_line(cleaned)


def closure_punctuation_removed(original: str, rendered: str) -> bool:
    return sanitize_closure_render_text(original) == _sanitize_line(rendered)


def closure_card_lavfi_background(*, width: int, height: int, duration: float) -> str:
    return f"color=c={CLOSURE_BACKGROUND_FFMPEG_COLOR}:s={width}x{height}:d={duration:.6f}"


def _text_font(font_path: Path, fontsize: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(font_path), fontsize)


def _text_ink_bbox(font_path: Path, text: str, fontsize: int) -> Tuple[int, int, int, int]:
    bbox = _text_font(font_path, fontsize).getbbox(text or " ")
    if not bbox:
        return (0, 0, 1, fontsize)
    return tuple(int(v) for v in bbox)


def _text_width_px(font_path: Path, text: str, fontsize: int) -> int:
    bbox = _text_ink_bbox(font_path, text, fontsize)
    return max(1, bbox[2] - bbox[0])


def _text_ink_height(font_path: Path, text: str, fontsize: int) -> int:
    bbox = _text_ink_bbox(font_path, text, fontsize)
    return max(1, bbox[3] - bbox[1])


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
    reveal_travel_px: int = 0
    reveal_window_width: int = 0
    reveal_window_height: int = 0
    overlay_x_px: int = 0
    overlay_y_px: int = 0
    final_y_local_px: int = 0
    hidden_y_local_px: int = 0
    ink_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


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
    effective_logical_product_slogan_gap_px: int = 0
    effective_visible_ink_gap_px: int = 0
    visible_ink_gap_satisfied: bool = False
    configured_closure_segment_duration_seconds: float = 0.0
    line_specs: Tuple[ClosureTypographyLineSpec, ...] = field(default_factory=tuple)

    def metadata(
        self,
        *,
        measured_final_duration_seconds: float | None = None,
        measured_raw_video_duration_seconds: float | None = None,
        measured_closure_duration_seconds: float | None = None,
        expected_final_video_duration_from_components: float | None = None,
        final_duration_verified: bool | None = None,
    ) -> Dict[str, Any]:
        from engine.builder2_new_format_config import (
            resolve_builder2_effective_closure_segment_duration_seconds,
        )
        from engine.builder2_closure_duration_contract import (
            resolve_expected_final_video_duration_seconds,
        )

        configured_closure = float(
            self.configured_closure_segment_duration_seconds
            or resolve_builder2_effective_closure_segment_duration_seconds()
        )
        product_specs = [spec for spec in self.line_specs if spec.role == "product"]
        slogan_specs = [spec for spec in self.line_specs if spec.role == "slogan"]
        max_reveal_travel = max((spec.reveal_travel_px for spec in self.line_specs), default=0)
        return {
            "typographyContractVersion": BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            "productFontAsset": PRODUCT_FONT_RELATIVE.as_posix(),
            "sloganFontAsset": SLOGAN_FONT_RELATIVE.as_posix(),
            "requestedProductFontSize": self.requested_product_font_size,
            "effectiveProductFontSize": self.effective_product_font_size,
            "requestedSloganFontSize": self.requested_slogan_font_size,
            "effectiveSloganFontSize": self.effective_slogan_font_size,
            "effectiveDominanceRatio": round(self.effective_dominance_ratio, 4),
            "configuredProductSloganGapPx": PRODUCT_SLOGAN_BLOCK_GAP_PX,
            "effectiveLogicalProductSloganGapPx": self.effective_logical_product_slogan_gap_px,
            "effectiveVisibleInkGapPx": self.effective_visible_ink_gap_px,
            "visibleInkGapSatisfied": self.visible_ink_gap_satisfied,
            "effectiveProductSloganGapPx": self.effective_visible_ink_gap_px,
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
            "configuredFinalVideoDurationSeconds": resolve_expected_final_video_duration_seconds(
                raw_video_duration_seconds=measured_raw_video_duration_seconds,
            ),
            "measuredRawVideoDurationSeconds": measured_raw_video_duration_seconds,
            "measuredClosureDurationSeconds": measured_closure_duration_seconds,
            "measuredFinalDurationSeconds": measured_final_duration_seconds,
            "expectedFinalVideoDurationFromComponents": (
                expected_final_video_duration_from_components
                if expected_final_video_duration_from_components is not None
                else (
                    (float(measured_raw_video_duration_seconds) + configured_closure)
                    if measured_raw_video_duration_seconds is not None
                    else resolve_expected_final_video_duration_seconds()
                )
            ),
            "finalDurationVerified": final_duration_verified,
            "closureTextRevealVersion": CLOSURE_TEXT_REVEAL_VERSION,
            "closureTextRevealEasing": CLOSURE_TEXT_REVEAL_EASING,
            "revealProgressFunctionVersion": REVEAL_PROGRESS_FUNCTION_VERSION,
            "productRevealStartSeconds": CLOSURE_PRODUCT_REVEAL_START_S,
            "productRevealDurationSeconds": CLOSURE_PRODUCT_REVEAL_DURATION_S,
            "sloganRevealStartSeconds": CLOSURE_SLOGAN_REVEAL_START_S,
            "sloganRevealDurationSeconds": CLOSURE_SLOGAN_REVEAL_DURATION_S,
            "sloganStartAfterProductTravelRatio": SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO,
            "productTravelRatioAtSloganStart": closure_reveal_product_travel_ratio_at_timestamp(
                timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S,
            ),
            "productStillMovingAtSloganStart": closure_reveal_product_still_moving_at_timestamp(
                timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S,
            ),
            "revealStaggerRuleVersion": REVEAL_STAGGER_RULE_VERSION,
            "revealUsesFixedMask": True,
            "revealUsesFade": False,
            "revealUsesScale": False,
            "revealUsesOvershoot": False,
            "closureTextRevealEnabled": True,
            "closureTextRevealMasked": True,
            "closureTextRevealUsesBoundedOverlay": True,
            "productTextRevealApplied": bool(product_specs),
            "sloganTextRevealApplied": bool(slogan_specs),
            "closureTextRevealTravelPx": max_reveal_travel,
            "productNameRenderedAsPlainText": True,
            "productNameFontRole": "primary",
            "sloganFontRole": "secondary",
            "sloganRenderedExactlyOnce": True,
            "separateHeadlineRendered": False,
            "brandNameDominanceSatisfied": self.effective_dominance_ratio >= MIN_ACCEPTABLE_DOMINANCE_RATIO,
        }


def _reveal_window_for_line(
    *,
    font_path: Path,
    text: str,
    fontsize: int,
    safe_width: int,
) -> Tuple[int, int, int, int, int]:
    bbox = _text_ink_bbox(font_path, text, fontsize)
    ink_w = max(1, bbox[2] - bbox[0])
    ink_h = max(1, bbox[3] - bbox[1])
    window_w = min(safe_width, ink_w + (2 * REVEAL_WINDOW_HORIZONTAL_PAD_PX))
    # Fixed clip slot: final visible ink area only. Travel happens below this boundary.
    window_h = REVEAL_WINDOW_TOP_PAD_PX + ink_h + REVEAL_WINDOW_BOTTOM_PAD_PX
    final_y_local = REVEAL_WINDOW_TOP_PAD_PX - bbox[1]
    hidden_y_local = window_h - bbox[1] + REVEAL_HIDDEN_EXTRA_PAD_PX
    reveal_travel_px = hidden_y_local - final_y_local
    return window_w, window_h, final_y_local, hidden_y_local, reveal_travel_px


def closure_reveal_linear_progress_at_timestamp(spec: ClosureTypographyLineSpec, timestamp_seconds: float) -> float:
    start = spec.reveal_start_seconds
    duration = max(0.05, spec.reveal_duration_seconds)
    if timestamp_seconds <= start:
        return 0.0
    if timestamp_seconds >= start + duration:
        return 1.0
    return max(0.0, min(1.0, (timestamp_seconds - start) / duration))


def closure_reveal_progress_at_timestamp(spec: ClosureTypographyLineSpec, timestamp_seconds: float) -> float:
    """Normalized linear progress p in [0, 1] for the reveal interval."""
    return closure_reveal_linear_progress_at_timestamp(spec, timestamp_seconds)


def closure_reveal_eased_progress(linear_progress: float) -> float:
    p = max(0.0, min(1.0, linear_progress))
    return 1.0 - (1.0 - p) ** 3


def closure_reveal_eased_progress_at_timestamp(spec: ClosureTypographyLineSpec, timestamp_seconds: float) -> float:
    return closure_reveal_eased_progress(closure_reveal_linear_progress_at_timestamp(spec, timestamp_seconds))


def closure_reveal_y_local_at_progress(spec: ClosureTypographyLineSpec, linear_progress: float) -> float:
    eased = closure_reveal_eased_progress(linear_progress)
    return float(spec.hidden_y_local_px) - (float(spec.reveal_travel_px) * eased)


def expected_visible_ink_height_at_progress(spec: ClosureTypographyLineSpec, linear_progress: float) -> int:
    y_local = closure_reveal_y_local_at_progress(spec, linear_progress)
    ink_top = y_local + spec.ink_bbox[1]
    ink_bottom = y_local + spec.ink_bbox[3]
    visible_top = max(0.0, ink_top)
    visible_bottom = min(float(spec.reveal_window_height), ink_bottom)
    if visible_bottom <= visible_top:
        return 0
    return max(0, int(round(visible_bottom - visible_top)))


def closure_reveal_ffmpeg_linear_progress_expression(start_seconds: float, duration_seconds: float) -> str:
    duration = max(0.05, duration_seconds)
    return f"max(0\\,min(1\\,(t-{start_seconds:.3f})/{duration:.3f}))"


def closure_reveal_ffmpeg_ease_out_cubic_expression(linear_progress_expression: str) -> str:
    return f"(1-pow(1-({linear_progress_expression})\\,3))"


def closure_reveal_ffmpeg_y_local_expression(spec: ClosureTypographyLineSpec) -> str:
    linear_p = closure_reveal_ffmpeg_linear_progress_expression(
        spec.reveal_start_seconds,
        spec.reveal_duration_seconds,
    )
    eased = closure_reveal_ffmpeg_ease_out_cubic_expression(linear_p)
    return f"{spec.hidden_y_local_px}-({spec.reveal_travel_px})*({eased})"


def closure_reveal_geometry_report(
    spec: ClosureTypographyLineSpec,
    *,
    timestamp_seconds: float,
) -> Dict[str, int | float]:
    linear_progress = closure_reveal_linear_progress_at_timestamp(spec, timestamp_seconds)
    eased_progress = closure_reveal_eased_progress(linear_progress)
    y_local = closure_reveal_y_local_at_progress(spec, linear_progress)
    bbox = spec.ink_bbox
    return {
        "revealWindowWidthPx": spec.reveal_window_width,
        "revealWindowHeightPx": spec.reveal_window_height,
        "glyphInkTopBoundPx": bbox[1],
        "glyphInkBottomBoundPx": bbox[3],
        "fullFinalVisibleInkHeightPx": max(1, bbox[3] - bbox[1]),
        "finalYLocalPx": spec.final_y_local_px,
        "hiddenYLocalPx": spec.hidden_y_local_px,
        "revealTravelPx": spec.reveal_travel_px,
        "revealStartSeconds": spec.reveal_start_seconds,
        "revealDurationSeconds": spec.reveal_duration_seconds,
        "calculatedLinearProgress": round(linear_progress, 6),
        "calculatedEasedProgress": round(eased_progress, 6),
        "calculatedProgress": round(linear_progress, 4),
        "calculatedYLocalPx": round(y_local, 2),
        "expectedVisibleInkHeightPx": expected_visible_ink_height_at_progress(spec, linear_progress),
    }


def _canvas_ink_bottom(y_px: int, bbox: Tuple[int, int, int, int]) -> int:
    return y_px + bbox[3]


def _canvas_ink_top(y_px: int, bbox: Tuple[int, int, int, int]) -> int:
    return y_px + bbox[1]


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

        rel_product: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        y_cursor = 0
        for line in product_lines:
            bbox = _text_ink_bbox(product_font, line, product_size)
            rel_product.append((line, y_cursor, bbox))
            y_cursor += _line_height_px(product_size)

        last_product_y, last_product_bbox = rel_product[-1][1], rel_product[-1][2]
        last_product_bottom_ink = _canvas_ink_bottom(last_product_y, last_product_bbox)

        first_slogan_line = slogan_lines[0]
        first_slogan_bbox = _text_ink_bbox(slogan_font, first_slogan_line, slogan_size)
        first_slogan_y = (
            last_product_bottom_ink
            + TARGET_VISIBLE_INK_GAP_PX
            - CLOSURE_FFMPEG_RENDERED_INK_GAP_ADJUSTMENT_PX
            - first_slogan_bbox[1]
        )

        rel_slogan: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        y_cursor = first_slogan_y
        for line in slogan_lines:
            bbox = _text_ink_bbox(slogan_font, line, slogan_size)
            rel_slogan.append((line, y_cursor, bbox))
            y_cursor += _line_height_px(slogan_size)

        block_top = _canvas_ink_top(rel_product[0][1], rel_product[0][2])
        block_bottom = _canvas_ink_bottom(rel_slogan[-1][1], rel_slogan[-1][2])
        block_h = block_bottom - block_top
        if block_h > safe_height:
            continue

        start_y_offset = int(round((CANVAS_HEIGHT - block_h) / 2)) - block_top

        product_specs_abs: List[Tuple[str, int, Tuple[int, int, int, int]]] = [
            (line, y + start_y_offset, bbox) for line, y, bbox in rel_product
        ]
        slogan_specs_abs: List[Tuple[str, int, Tuple[int, int, int, int]]] = [
            (line, y + start_y_offset, bbox) for line, y, bbox in rel_slogan
        ]

        last_prod_y, last_prod_bbox = product_specs_abs[-1][1], product_specs_abs[-1][2]
        first_slog_y, first_slog_bbox = slogan_specs_abs[0][1], slogan_specs_abs[0][2]
        pil_visible_ink_gap = _canvas_ink_top(first_slog_y, first_slog_bbox) - _canvas_ink_bottom(
            last_prod_y, last_prod_bbox
        )
        visible_ink_gap = pil_visible_ink_gap + CLOSURE_FFMPEG_RENDERED_INK_GAP_ADJUSTMENT_PX
        logical_gap = first_slog_y - (last_prod_y + _line_height_px(product_size))
        if visible_ink_gap < MIN_VISIBLE_INK_GAP_PX or visible_ink_gap > MAX_VISIBLE_INK_GAP_PX:
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

        top_ink = _canvas_ink_top(product_specs_abs[0][1], product_specs_abs[0][2])
        bottom_ink = _canvas_ink_bottom(slogan_specs_abs[-1][1], slogan_specs_abs[-1][2])
        if top_ink < VERTICAL_SAFE_MARGIN_PX:
            continue
        if bottom_ink > CANVAS_HEIGHT - VERTICAL_SAFE_MARGIN_PX:
            continue

        product_shaping = lang == "he" or _has_hebrew_letter(product_text)
        slogan_shaping = lang == "he" or _has_hebrew_letter(slogan_text)
        line_specs: List[ClosureTypographyLineSpec] = []

        for index, (line, y_px, bbox) in enumerate(product_specs_abs):
            window_w, window_h, final_y_local, hidden_y_local, reveal_travel = _reveal_window_for_line(
                font_path=product_font,
                text=line,
                fontsize=product_size,
                safe_width=safe_width,
            )
            line_specs.append(
                ClosureTypographyLineSpec(
                    text=line,
                    font_path=product_font,
                    fontsize=product_size,
                    y_px=y_px,
                    use_text_shaping=product_shaping,
                    role="product",
                    reveal_start_seconds=CLOSURE_PRODUCT_REVEAL_START_S + (index * CLOSURE_PRODUCT_LINE_STAGGER_S),
                    reveal_duration_seconds=CLOSURE_PRODUCT_REVEAL_DURATION_S,
                    reveal_travel_px=reveal_travel,
                    reveal_window_width=window_w,
                    reveal_window_height=window_h,
                    overlay_x_px=(CANVAS_WIDTH - window_w) // 2,
                    overlay_y_px=y_px,
                    final_y_local_px=final_y_local,
                    hidden_y_local_px=hidden_y_local,
                    ink_bbox=bbox,
                )
            )

        for index, (line, y_px, bbox) in enumerate(slogan_specs_abs):
            window_w, window_h, final_y_local, hidden_y_local, reveal_travel = _reveal_window_for_line(
                font_path=slogan_font,
                text=line,
                fontsize=slogan_size,
                safe_width=safe_width,
            )
            line_specs.append(
                ClosureTypographyLineSpec(
                    text=line,
                    font_path=slogan_font,
                    fontsize=slogan_size,
                    y_px=y_px,
                    use_text_shaping=slogan_shaping,
                    role="slogan",
                    reveal_start_seconds=CLOSURE_SLOGAN_REVEAL_START_S + (index * CLOSURE_PRODUCT_LINE_STAGGER_S),
                    reveal_duration_seconds=CLOSURE_SLOGAN_REVEAL_DURATION_S,
                    reveal_travel_px=reveal_travel,
                    reveal_window_width=window_w,
                    reveal_window_height=window_h,
                    overlay_x_px=(CANVAS_WIDTH - window_w) // 2,
                    overlay_y_px=y_px,
                    final_y_local_px=final_y_local,
                    hidden_y_local_px=hidden_y_local,
                    ink_bbox=bbox,
                )
            )

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
            product_slogan_gap_px=PRODUCT_SLOGAN_BLOCK_GAP_PX,
            effective_logical_product_slogan_gap_px=logical_gap,
            effective_visible_ink_gap_px=visible_ink_gap,
            visible_ink_gap_satisfied=MIN_VISIBLE_INK_GAP_PX <= visible_ink_gap <= MAX_VISIBLE_INK_GAP_PX,
            configured_closure_segment_duration_seconds=configured_closure,
            line_specs=tuple(line_specs),
        )
        break

    if best is None:
        raise Builder2TournamentError("builder2_closure_text_overflow")
    if best.effective_dominance_ratio < MIN_ACCEPTABLE_DOMINANCE_RATIO:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied")
    if not best.visible_ink_gap_satisfied:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:visible_gap")
    return best


def _masked_reveal_y_local_expression(spec: ClosureTypographyLineSpec) -> str:
    return closure_reveal_ffmpeg_y_local_expression(spec)


def build_closure_card_masked_reveal_filter_complex(
    layout: ClosureTypographyLayout,
    *,
    textfile_paths: Sequence[Path],
    duration_seconds: float,
    ffmpeg_path_filter: Any,
    font_paths: Sequence[Path] | None = None,
) -> Tuple[str, str]:
    if len(textfile_paths) != len(layout.line_specs):
        raise Builder2TournamentError("builder2_closure_text_overflow")
    if font_paths is not None and len(font_paths) != len(layout.line_specs):
        raise Builder2TournamentError("builder2_closure_text_overflow")

    parts: List[str] = ["[0:v]format=rgba[closure_base0]"]
    current_base = "closure_base0"

    for index, (spec, text_path) in enumerate(zip(layout.line_specs, textfile_paths)):
        font_path = font_paths[index] if font_paths is not None else spec.font_path
        font_e = ffmpeg_path_filter(Path(font_path))
        text_e = ffmpeg_path_filter(text_path)
        shaping = ":text_shaping=1:expansion=none" if spec.use_text_shaping else ""
        y_expr = _masked_reveal_y_local_expression(spec)
        win_label = f"closure_win{index}"
        next_base = f"closure_base{index + 1}"
        parts.append(
            f"color=c=0x00000000:s={spec.reveal_window_width}x{spec.reveal_window_height}:"
            f"d={duration_seconds:.6f},format=rgba,"
            f"drawtext=fontfile='{font_e}':textfile='{text_e}':"
            f"fontcolor=white:fontsize={spec.fontsize}:x=(w-text_w)/2:y='{y_expr}'"
            f":borderw=2:bordercolor=black@0.35{shaping}[{win_label}]"
        )
        parts.append(
            f"[{current_base}][{win_label}]overlay=x={spec.overlay_x_px}:y={spec.overlay_y_px}:format=auto[{next_base}]"
        )
        current_base = next_base

    out_label = "closure_outv"
    parts.append(f"[{current_base}]format=yuv420p[{out_label}]")
    return ";".join(parts), out_label


def closure_filter_uses_masked_bounded_overlays(filter_complex: str) -> bool:
    token = filter_complex or ""
    if "overlay=" not in token:
        return False
    if "color=c=0x00000000" not in token:
        return False
    if "format=rgba" not in token:
        return False
    return closure_filter_rejects_full_frame_slide(token)


def closure_filter_rejects_full_frame_slide(filter_complex: str) -> bool:
    token = filter_complex or ""
    if "[0:v]drawtext" in token.replace(" ", ""):
        return False
    if "drawtext=" not in token:
        return False
    return "overlay=" in token


def build_closure_card_drawtext_filter(
    layout: ClosureTypographyLayout,
    *,
    textfile_paths: Sequence[Path],
    ffmpeg_path_filter: Any,
) -> str:
    raise Builder2TournamentError("builder2_closure_full_frame_slide_rejected")


def stamp_closure_typography_metadata(
    target: Dict[str, Any],
    layout: ClosureTypographyLayout,
    *,
    measured_final_duration_seconds: float | None = None,
    measured_raw_video_duration_seconds: float | None = None,
    measured_closure_duration_seconds: float | None = None,
    expected_final_video_duration_from_components: float | None = None,
    final_duration_verified: bool | None = None,
) -> None:
    target.update(
        layout.metadata(
            measured_final_duration_seconds=measured_final_duration_seconds,
            measured_raw_video_duration_seconds=measured_raw_video_duration_seconds,
            measured_closure_duration_seconds=measured_closure_duration_seconds,
            expected_final_video_duration_from_components=expected_final_video_duration_from_components,
            final_duration_verified=final_duration_verified,
        )
    )


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
    if metadata.get("closureTextRevealEasing") != CLOSURE_TEXT_REVEAL_EASING:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:animation_easing")
    if metadata.get("revealProgressFunctionVersion") != REVEAL_PROGRESS_FUNCTION_VERSION:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:progress_function")
    if metadata.get("revealUsesFixedMask") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:fixed_mask")
    if metadata.get("revealUsesFade") is not False:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:fade")
    if metadata.get("revealUsesScale") is not False:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:scale")
    if metadata.get("revealUsesOvershoot") is not False:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:overshoot")
    if metadata.get("closureTextRevealMasked") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:animation_mask")
    if metadata.get("closureTextRevealUsesBoundedOverlay") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:animation_overlay")
    if metadata.get("visibleInkGapSatisfied") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:visible_gap")
    if metadata.get("brandNameDominanceSatisfied") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied")
    if metadata.get("separateHeadlineRendered") is True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:headline_layer")
    if metadata.get("sloganRenderedExactlyOnce") is not True:
        raise Builder2TournamentError("builder2_closure_brand_dominance_unsatisfied:slogan_count")
