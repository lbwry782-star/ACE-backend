"""
Builder2 closure Hebrew glyph integrity verification — rejects FFmpeg tofu rectangles.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from engine.builder2_closure_typography import (
    ClosureTypographyLayout,
    ClosureTypographyLineSpec,
    REVEAL_DRAWTEXT_BORDER_PX,
    REVEAL_SOURCE_LAYER_AA_PAD_PX,
    REVEAL_WINDOW_BORDER_INSET_PX,
    REVEAL_WINDOW_BOTTOM_PAD_PX,
    REVEAL_WINDOW_HORIZONTAL_PAD_PX,
    REVEAL_WINDOW_TOP_PAD_PX,
    closure_reveal_canvas_ink_top,
    closure_reveal_settled_ink_bottom_in_window,
    closure_reveal_settled_ink_top_in_window,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

_TEXT_LUMA_THRESHOLD = 165


@dataclass(frozen=True)
class GlyphRowWidthStats:
    bright_rows: int
    width_unique: int
    dominant_width: int
    dominant_row_fraction: float


def _is_background_pixel(r: int, g: int, b: int) -> bool:
    return r <= 30 and g <= 12 and b <= 35


def _is_text_pixel(r: int, g: int, b: int) -> bool:
    if _is_background_pixel(r, g, b):
        return False
    luma = (r * 299 + g * 587 + b * 114) // 1000
    return luma >= _TEXT_LUMA_THRESHOLD


def glyph_row_width_stats(image_path: Path, region: Tuple[int, int, int, int]) -> GlyphRowWidthStats:
    from PIL import Image

    left, top, right, bottom = region
    image = Image.open(image_path).convert("RGB")
    row_widths: list[int] = []
    for y in range(max(0, top), min(bottom, image.height)):
        count = 0
        for x in range(max(0, left), min(right, image.width)):
            if _is_text_pixel(*image.getpixel((x, y))):
                count += 1
        if count:
            row_widths.append(count)
    if not row_widths:
        return GlyphRowWidthStats(0, 0, 0, 0.0)
    dominant_width, dominant_count = Counter(row_widths).most_common(1)[0]
    return GlyphRowWidthStats(
        bright_rows=len(row_widths),
        width_unique=len(set(row_widths)),
        dominant_width=dominant_width,
        dominant_row_fraction=dominant_count / len(row_widths),
    )


def detect_missing_glyph_rectangle_pattern(stats: GlyphRowWidthStats) -> bool:
    if stats.bright_rows < 5:
        return False
    if stats.width_unique > 3:
        return False
    return stats.dominant_row_fraction >= 0.55


def render_independent_settled_reference_mask(
    *,
    font_path: Path,
    text: str,
    fontsize: int,
    width: int,
    height: int,
) -> "object":
    """
    Reference mask built only from PIL bbox measurement and positive draw coordinates.

    Does not reuse the reveal-window helper; settled anchor is placed so ink top lands
    at the same border-safe row used by the production source-layer model.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path.resolve()), fontsize)
    bbox = font.getbbox(text or " ")
    if not bbox:
        bbox = (0, 0, 1, fontsize)
    left, top, right, bottom = (int(v) for v in bbox)
    ink_w = max(1, right - left)
    ink_h = max(1, bottom - top)
    top_pad = REVEAL_WINDOW_TOP_PAD_PX + REVEAL_WINDOW_BORDER_INSET_PX
    bottom_pad = REVEAL_WINDOW_BOTTOM_PAD_PX + REVEAL_WINDOW_BORDER_INSET_PX
    border = REVEAL_DRAWTEXT_BORDER_PX
    aa = REVEAL_SOURCE_LAYER_AA_PAD_PX
    anchor_settled = top_pad - top
    ink_bottom_settled = top_pad + ink_h
    source_origin_y = min(0, anchor_settled - border - aa)
    source_end_y = ink_bottom_settled + bottom_pad + border + aa
    expected_h = source_end_y - source_origin_y
    expected_w = min(
        width,
        ink_w + (2 * REVEAL_WINDOW_HORIZONTAL_PAD_PX) + (2 * REVEAL_WINDOW_BORDER_INSET_PX),
    )
    if width != expected_w or height != expected_h:
        raise AssertionError(
            f"independent_reference_dimensions_mismatch expected={expected_w}x{expected_h} actual={width}x{height}"
        )
    final_y_local = anchor_settled - source_origin_y
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    text_w = max(1, ink_w)
    x = max(0, (width - text_w) // 2)
    draw.text((x, final_y_local), text, font=font, fill=255)
    return image.point(lambda v: 255 if v >= 128 else 0)


def extract_binary_text_mask(image_path: Path, region: Tuple[int, int, int, int]) -> "object":
    from PIL import Image

    left, top, right, bottom = region
    image = Image.open(image_path).convert("RGB")
    mask = Image.new("L", (right - left, bottom - top), 0)
    pixels = mask.load()
    for y in range(max(0, top), min(bottom, image.height)):
        for x in range(max(0, left), min(right, image.width)):
            if _is_text_pixel(*image.getpixel((x, y))):
                pixels[x - left, y - top] = 255
    return mask


def mask_iou(reference_mask: object, rendered_mask: object) -> float:
    ref_data = reference_mask.getdata()
    ren_data = rendered_mask.getdata()
    intersection = 0
    union = 0
    for ref_v, ren_v in zip(ref_data, ren_data):
        ref_on = ref_v >= 128
        ren_on = ren_v >= 128
        if ref_on and ren_on:
            intersection += 1
        if ref_on or ren_on:
            union += 1
    if union <= 0:
        return 0.0
    return intersection / union


def reference_mask_top_coverage(reference_mask: object) -> Tuple[int, int]:
    width, height = reference_mask.size
    top_row = -1
    covered_rows = 0
    for y in range(height):
        row_has_pixel = any(reference_mask.getpixel((x, y)) >= 128 for x in range(width))
        if row_has_pixel:
            if top_row < 0:
                top_row = y
            covered_rows += 1
    return top_row, covered_rows


def rendered_mask_top_coverage(rendered_mask: object) -> Tuple[int, int]:
    return reference_mask_top_coverage(rendered_mask)


def assert_reference_top_rows_preserved(
    reference_mask: object,
    rendered_mask: object,
    *,
    min_top_row_coverage_ratio: float = 0.85,
) -> None:
    ref_top, ref_rows = reference_mask_top_coverage(reference_mask)
    ren_top, ren_rows = rendered_mask_top_coverage(rendered_mask)
    if ref_top < 0 or ren_top < 0:
        raise AssertionError("reference_or_rendered_mask_empty")
    if ren_top > ref_top + REVEAL_DRAWTEXT_BORDER_PX + REVEAL_SOURCE_LAYER_AA_PAD_PX + 1:
        raise AssertionError(
            f"product_glyph_top_clipped ref_top={ref_top} rendered_top={ren_top}"
        )
    if ref_rows <= 0:
        raise AssertionError("reference_mask_has_no_rows")
    if ren_rows / ref_rows < min_top_row_coverage_ratio:
        raise AssertionError(
            f"product_glyph_vertical_coverage_too_low ref_rows={ref_rows} rendered_rows={ren_rows}"
        )


def render_pil_reference_mask(
    *,
    font_path: Path,
    text: str,
    fontsize: int,
    width: int,
    height: int,
    y_local: int,
) -> "object":
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path.resolve()), fontsize)
    bbox = font.getbbox(text or " ")
    text_w = max(1, bbox[2] - bbox[0])
    x = max(0, (width - text_w) // 2)
    draw.text((x, y_local), text, font=font, fill=255)
    return image.point(lambda v: 255 if v >= 128 else 0)


def _measure_canvas_region(image_path: Path, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    from PIL import Image

    left, top, right, bottom = region
    image = Image.open(image_path).convert("RGB")
    bright = 0
    min_y = bottom
    max_y = top - 1
    for y in range(max(0, top), min(bottom, image.height)):
        for x in range(max(0, left), min(right, image.width)):
            if _is_text_pixel(*image.getpixel((x, y))):
                bright += 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    if bright == 0:
        return 0, 0, -1, -1
    return bright, max_y - min_y + 1, min_y, max_y


def assert_settled_product_source_layer_coverage(
    image_path: Path,
    spec: ClosureTypographyLineSpec,
    *,
    min_reference_iou: float = 0.15,
    min_top_row_coverage_ratio: float = 0.85,
    min_settled_ink_height_ratio: float = 0.85,
) -> None:
    region = (
        spec.overlay_x_px,
        spec.overlay_y_px,
        spec.overlay_x_px + spec.reveal_window_width,
        spec.overlay_y_px + spec.reveal_window_height,
    )
    reference = render_independent_settled_reference_mask(
        font_path=spec.font_path,
        text=spec.text,
        fontsize=spec.fontsize,
        width=spec.reveal_window_width,
        height=spec.reveal_window_height,
    )
    rendered = extract_binary_text_mask(image_path, region)
    assert_reference_top_rows_preserved(
        reference,
        rendered,
        min_top_row_coverage_ratio=min_top_row_coverage_ratio,
    )
    iou = mask_iou(reference, rendered)
    if iou < min_reference_iou:
        raise AssertionError(f"hebrew_glyph_reference_iou_too_low iou={iou:.4f}")

    expected_ink_height = max(1, spec.ink_bbox[3] - spec.ink_bbox[1])
    ink_top = closure_reveal_settled_ink_top_in_window(spec)
    ink_bottom = closure_reveal_settled_ink_bottom_in_window(spec)
    visible_top = max(0.0, ink_top)
    visible_bottom = min(float(spec.reveal_window_height), ink_bottom)
    if visible_bottom <= visible_top:
        raise AssertionError("settled_product_visible_ink_missing")
    settled_visible_height = visible_bottom - visible_top
    if settled_visible_height < expected_ink_height * min_settled_ink_height_ratio:
        raise AssertionError(
            "settled_product_ink_height_clipped "
            f"expected={expected_ink_height:.1f} visible={settled_visible_height:.1f}"
        )

    canvas_top = closure_reveal_canvas_ink_top(spec)
    ink_h = max(1, spec.ink_bbox[3] - spec.ink_bbox[1])
    canvas_region = (
        spec.overlay_x_px,
        canvas_top - REVEAL_DRAWTEXT_BORDER_PX - REVEAL_SOURCE_LAYER_AA_PAD_PX,
        spec.overlay_x_px + spec.reveal_window_width,
        canvas_top + ink_h + REVEAL_DRAWTEXT_BORDER_PX,
    )
    _bright, measured_height, measured_top, _measured_bottom = _measure_canvas_region(image_path, canvas_region)
    if measured_height < expected_ink_height * min_settled_ink_height_ratio:
        raise AssertionError(
            "settled_product_canvas_ink_height_clipped "
            f"expected={expected_ink_height} measured={measured_height} canvas_top={canvas_top}"
        )
    if measured_top >= 0 and measured_top > canvas_top + REVEAL_DRAWTEXT_BORDER_PX + 1:
        raise AssertionError(
            f"settled_product_canvas_top_clipped expected_top={canvas_top} measured_top={measured_top}"
        )


def assert_hebrew_line_glyph_integrity(
    image_path: Path,
    spec: ClosureTypographyLineSpec,
    *,
    min_row_width_variants: int = 4,
    min_reference_iou: float = 0.18,
) -> None:
    region = (
        spec.overlay_x_px,
        spec.overlay_y_px,
        spec.overlay_x_px + spec.reveal_window_width,
        spec.overlay_y_px + spec.reveal_window_height,
    )
    stats = glyph_row_width_stats(image_path, region)
    if detect_missing_glyph_rectangle_pattern(stats):
        raise AssertionError("hebrew_missing_glyph_rectangle_pattern_detected")
    if stats.width_unique < min_row_width_variants and len(spec.text.strip()) >= 4:
        raise AssertionError(
            f"hebrew_glyph_contour_variation_too_low unique={stats.width_unique}"
        )

    reference = render_independent_settled_reference_mask(
        font_path=spec.font_path,
        text=spec.text,
        fontsize=spec.fontsize,
        width=spec.reveal_window_width,
        height=spec.reveal_window_height,
    )
    rendered = extract_binary_text_mask(image_path, region)
    assert_reference_top_rows_preserved(reference, rendered)
    iou = mask_iou(reference, rendered)
    if iou < min_reference_iou:
        raise AssertionError(f"hebrew_glyph_reference_iou_too_low iou={iou:.4f}")


def assert_layout_hebrew_glyph_integrity(
    image_path: Path,
    layout: ClosureTypographyLayout,
) -> None:
    for spec in layout.line_specs:
        if not spec.use_text_shaping:
            continue
        assert_hebrew_line_glyph_integrity(image_path, spec)


def write_glyph_diagnostic_crops(
    image_path: Path,
    layout: ClosureTypographyLayout,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    full = Image.open(image_path).convert("RGB")
    stable_path = output_dir / "stable_final.png"
    full.save(stable_path)
    paths["stable_final"] = stable_path

    for index, spec in enumerate(layout.line_specs):
        region = (
            spec.overlay_x_px,
            spec.overlay_y_px,
            spec.overlay_x_px + spec.reveal_window_width,
            spec.overlay_y_px + spec.reveal_window_height,
        )
        crop = full.crop(region)
        crop_path = output_dir / f"{spec.role}_{index}_crop.png"
        crop.save(crop_path)
        paths[f"{spec.role}_{index}_crop"] = crop_path

        if spec.use_text_shaping:
            ref = render_independent_settled_reference_mask(
                font_path=spec.font_path,
                text=spec.text,
                fontsize=spec.fontsize,
                width=spec.reveal_window_width,
                height=spec.reveal_window_height,
            )
            ref_path = output_dir / f"{spec.role}_{index}_reference_mask.png"
            ref.save(ref_path)
            paths[f"{spec.role}_{index}_reference_mask"] = ref_path

    return paths


__all__ = [
    "GlyphRowWidthStats",
    "assert_hebrew_line_glyph_integrity",
    "assert_layout_hebrew_glyph_integrity",
    "assert_reference_top_rows_preserved",
    "assert_settled_product_source_layer_coverage",
    "detect_missing_glyph_rectangle_pattern",
    "extract_binary_text_mask",
    "glyph_row_width_stats",
    "mask_iou",
    "render_independent_settled_reference_mask",
    "render_pil_reference_mask",
    "rendered_mask_top_coverage",
    "reference_mask_top_coverage",
    "write_glyph_diagnostic_crops",
]
