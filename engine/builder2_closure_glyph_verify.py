"""
Builder2 closure Hebrew glyph integrity verification — rejects FFmpeg tofu rectangles.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from engine.builder2_closure_typography import ClosureTypographyLayout, ClosureTypographyLineSpec
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

    reference = render_pil_reference_mask(
        font_path=spec.font_path,
        text=spec.text,
        fontsize=spec.fontsize,
        width=spec.reveal_window_width,
        height=spec.reveal_window_height,
        y_local=spec.final_y_local_px,
    )
    rendered = extract_binary_text_mask(image_path, region)
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
            ref = render_pil_reference_mask(
                font_path=spec.font_path,
                text=spec.text,
                fontsize=spec.fontsize,
                width=spec.reveal_window_width,
                height=spec.reveal_window_height,
                y_local=spec.final_y_local_px,
            )
            ref_path = output_dir / f"{spec.role}_{index}_reference_mask.png"
            ref.save(ref_path)
            paths[f"{spec.role}_{index}_reference_mask"] = ref_path

    return paths


__all__ = [
    "GlyphRowWidthStats",
    "assert_hebrew_line_glyph_integrity",
    "assert_layout_hebrew_glyph_integrity",
    "detect_missing_glyph_rectangle_pattern",
    "extract_binary_text_mask",
    "glyph_row_width_stats",
    "mask_iou",
    "render_pil_reference_mask",
    "write_glyph_diagnostic_crops",
]
