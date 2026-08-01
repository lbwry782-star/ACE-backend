"""
Builder2 closure masked-reveal render verification — local FFmpeg frame probes.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from engine.builder2_closure_typography import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CLOSURE_PRODUCT_REVEAL_DURATION_S,
    CLOSURE_PRODUCT_REVEAL_START_S,
    ClosureTypographyLayout,
    ClosureTypographyLineSpec,
    REVEAL_DRAWTEXT_BORDER_PX,
    build_closure_card_masked_reveal_filter_complex,
    closure_card_lavfi_background,
    closure_filter_rejects_full_frame_slide,
    closure_filter_uses_masked_bounded_overlays,
    closure_reveal_canvas_ink_bottom,
    closure_reveal_canvas_ink_top,
    closure_reveal_eased_progress,
    closure_reveal_eased_progress_at_timestamp,
    closure_reveal_geometry_report,
    closure_reveal_linear_progress_at_timestamp,
    closure_reveal_y_local_at_progress,
    expected_visible_ink_height_at_progress,
    fit_builder2_closure_typography,
    REVEAL_WINDOW_BORDER_INSET_PX,
)
from engine.builder2_closure_ffmpeg_paths import (
    ClosureFfmpegAssetSession,
    assert_closure_ffmpeg_stderr_font_health,
)
from engine.video_headline_postprocess import _ffmpeg_bin

_FFMPEG_TIMEOUT = 180.0
_TARGET_FPS = 30
_TEXT_LUMA_THRESHOLD = 165


@dataclass(frozen=True)
class ClosureFrameProbe:
    timestamp_seconds: float
    image_path: Path
    bright_pixel_count: int
    role_bright_counts: Dict[str, int]
    role_visible_heights: Dict[str, int]
    role_visible_ink_heights: Dict[str, int]
    role_outside_window_pixels: Dict[str, int]


@dataclass(frozen=True)
class ClosureRevealDiagnosticFrames:
    before_reveal: Path
    midpoint_product: Path
    product_complete: Path
    both_complete: Path
    stable_hold: Path
    diagnostic_dir: Path


def closure_reveal_diagnostic_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "ace_builder2_closure_reveal_diag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_background_pixel(r: int, g: int, b: int) -> bool:
    return r <= 30 and g <= 12 and b <= 35


def _is_text_pixel(r: int, g: int, b: int) -> bool:
    if _is_background_pixel(r, g, b):
        return False
    luma = (r * 299 + g * 587 + b * 114) // 1000
    return luma >= _TEXT_LUMA_THRESHOLD


def _line_window_on_canvas(spec: ClosureTypographyLineSpec) -> Tuple[int, int, int, int]:
    return (
        spec.overlay_x_px,
        spec.overlay_y_px,
        spec.overlay_x_px + spec.reveal_window_width,
        spec.overlay_y_px + spec.reveal_window_height,
    )


def _line_window_with_border_on_canvas(spec: ClosureTypographyLineSpec) -> Tuple[int, int, int, int]:
    inset = REVEAL_WINDOW_BORDER_INSET_PX
    return (
        spec.overlay_x_px - inset,
        spec.overlay_y_px - inset,
        spec.overlay_x_px + spec.reveal_window_width + inset,
        spec.overlay_y_px + spec.reveal_window_height + inset,
    )


def count_text_pixels_in_region(image_path: Path, region: Tuple[int, int, int, int]) -> Tuple[int, int]:
    bright, ink_height, _min_y, _max_y = measure_text_pixels_in_region(image_path, region)
    return bright, ink_height


def measure_text_pixels_in_region(
    image_path: Path,
    region: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    from PIL import Image

    left, top, right, bottom = region
    image = Image.open(image_path).convert("RGB")
    bright = 0
    min_y = bottom
    max_y = top - 1
    for y in range(max(0, top), min(bottom, image.height)):
        for x in range(max(0, left), min(right, image.width)):
            r, g, b = image.getpixel((x, y))
            if _is_text_pixel(r, g, b):
                bright += 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    if bright == 0:
        return 0, 0, -1, -1
    return bright, max_y - min_y + 1, min_y, max_y


def count_text_pixels_outside_window(image_path: Path, spec: ClosureTypographyLineSpec) -> int:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    left, top, right, bottom = _line_window_with_border_on_canvas(spec)
    ink_left = left
    ink_right = right
    ink_top = top
    ink_bottom = bottom
    outside = 0
    for y in range(max(0, ink_top), min(ink_bottom, image.height)):
        for x in range(max(0, ink_left), min(ink_right, image.width)):
            in_window = left <= x < right and top <= y < bottom
            if in_window:
                continue
            r, g, b = image.getpixel((x, y))
            if _is_text_pixel(r, g, b):
                outside += 1
    return outside


def _line_ink_window_on_canvas(spec: ClosureTypographyLineSpec) -> Tuple[int, int, int, int]:
    left, _top, right, bottom = _line_window_on_canvas(spec)
    ink_top = closure_reveal_canvas_ink_top(spec) - REVEAL_WINDOW_BORDER_INSET_PX
    ink_bottom = closure_reveal_canvas_ink_bottom(spec) + REVEAL_DRAWTEXT_BORDER_PX
    return (
        left,
        ink_top,
        right,
        min(bottom, ink_bottom),
    )


def measure_visible_ink_gap_px(image_path: Path, layout: ClosureTypographyLayout) -> int:
    """Distance in px from lowest product ink row to highest slogan ink row on canvas."""
    product_specs = [spec for spec in layout.line_specs if spec.role == "product"]
    slogan_specs = [spec for spec in layout.line_specs if spec.role == "slogan"]
    if not product_specs or not slogan_specs:
        return 0

    product_bottom_ink_y = -1
    for spec in product_specs:
        _bright, _ink_height, _min_y, max_y = measure_text_pixels_in_region(
            image_path,
            _line_ink_window_on_canvas(spec),
        )
        if max_y >= 0:
            product_bottom_ink_y = max(product_bottom_ink_y, max_y)

    slogan_top_ink_y = -1
    for spec in slogan_specs:
        _bright, _ink_height, min_y, _max_y = measure_text_pixels_in_region(
            image_path,
            _line_ink_window_on_canvas(spec),
        )
        if min_y >= 0:
            slogan_top_ink_y = min(slogan_top_ink_y, min_y) if slogan_top_ink_y >= 0 else min_y

    if product_bottom_ink_y < 0 or slogan_top_ink_y < 0:
        return 0
    return max(0, slogan_top_ink_y - product_bottom_ink_y)


def render_closure_card_artifact(
    *,
    product_name: str,
    slogan: str,
    language: str = "he",
    output_path: Path,
    duration_seconds: float = 3.5,
) -> Tuple[ClosureTypographyLayout, str]:
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_found")

    layout = fit_builder2_closure_typography(
        product_name=product_name,
        slogan=slogan,
        language=language,
        closure_segment_duration_seconds=duration_seconds,
    )
    session = ClosureFfmpegAssetSession.create()
    try:
        line_files, font_files = session.prepare_line_assets(layout.line_specs)
        filter_complex, out_label = build_closure_card_masked_reveal_filter_complex(
            layout,
            textfile_paths=line_files,
            font_paths=font_files,
            duration_seconds=duration_seconds,
            ffmpeg_path_filter=session.filter_path,
        )
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            closure_card_lavfi_background(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, duration=duration_seconds),
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{out_label}]",
            "-r",
            str(_TARGET_FPS),
            "-t",
            f"{duration_seconds:.6f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        completed = subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT, capture_output=True)
        assert_closure_ffmpeg_stderr_font_health(completed.stderr)
        return layout, filter_complex
    finally:
        session.cleanup()


def extract_closure_frame(video_path: Path, *, timestamp_seconds: float, output_path: Path) -> None:
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_found")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT, capture_output=True)


def probe_closure_frame(
    image_path: Path,
    layout: ClosureTypographyLayout,
    *,
    timestamp_seconds: float,
) -> ClosureFrameProbe:
    role_bright: Dict[str, int] = {"product": 0, "slogan": 0}
    role_heights: Dict[str, int] = {"product": 0, "slogan": 0}
    role_ink_heights: Dict[str, int] = {"product": 0, "slogan": 0}
    role_outside: Dict[str, int] = {"product": 0, "slogan": 0}
    total = 0
    for spec in layout.line_specs:
        region = _line_window_on_canvas(spec)
        bright, ink_height, _min_y, _max_y = measure_text_pixels_in_region(image_path, region)
        total += bright
        role_bright[spec.role] = role_bright.get(spec.role, 0) + bright
        role_heights[spec.role] = max(role_heights.get(spec.role, 0), ink_height)
        role_ink_heights[spec.role] = max(role_ink_heights.get(spec.role, 0), ink_height)
        role_outside[spec.role] = role_outside.get(spec.role, 0) + count_text_pixels_outside_window(image_path, spec)
    return ClosureFrameProbe(
        timestamp_seconds=timestamp_seconds,
        image_path=image_path,
        bright_pixel_count=total,
        role_bright_counts=role_bright,
        role_visible_heights=role_heights,
        role_visible_ink_heights=role_ink_heights,
        role_outside_window_pixels=role_outside,
    )


def primary_product_spec(layout: ClosureTypographyLayout) -> ClosureTypographyLineSpec:
    for spec in layout.line_specs:
        if spec.role == "product":
            return spec
    raise ValueError("product_spec_missing")


def assert_eased_reveal_visible_height(
    layout: ClosureTypographyLayout,
    *,
    timestamp_seconds: float,
    measured_visible_height: int,
    stable_visible_height: int,
) -> None:
    product = primary_product_spec(layout)
    linear_progress = closure_reveal_linear_progress_at_timestamp(product, timestamp_seconds)
    expected = expected_visible_ink_height_at_progress(product, linear_progress)
    if stable_visible_height <= 0:
        raise AssertionError("stable_visible_height_missing")
    tolerance = max(5, int(round(stable_visible_height * 0.18)))
    if abs(measured_visible_height - expected) > tolerance:
        raise AssertionError(
            f"eased_height_geometry_mismatch measured={measured_visible_height} expected={expected}"
        )


def assert_ease_out_early_velocity_exceeds_late(
    *,
    early_start: float,
    early_end: float,
    late_start: float,
    late_end: float,
    early_delta: int,
    late_delta: int,
) -> None:
    if early_delta <= 0:
        raise AssertionError("early_reveal_interval_no_visible_gain")
    if late_delta < 0:
        raise AssertionError("late_reveal_interval_overshoot")
    if early_delta <= late_delta:
        raise AssertionError(
            f"ease_out_velocity_not_decreasing earlyDelta={early_delta} lateDelta={late_delta}"
        )


def assert_ease_out_near_complete_at_linear_midpoint(
    layout: ClosureTypographyLayout,
    *,
    timestamp_seconds: float,
    measured_visible_height: int,
    stable_visible_height: int,
) -> None:
    product = primary_product_spec(layout)
    linear_progress = closure_reveal_linear_progress_at_timestamp(product, timestamp_seconds)
    eased = closure_reveal_eased_progress(linear_progress)
    if eased < 0.85:
        raise AssertionError(f"linear_midpoint_eased_too_low eased={eased}")
    stable_reference = expected_visible_ink_height_at_progress(product, 1.0)
    minimum = max(1, int(stable_reference * 0.75))
    if measured_visible_height < minimum:
        raise AssertionError(
            f"linear_midpoint_not_near_complete measured={measured_visible_height} minimum={minimum}"
        )


def write_local_closure_preview_artifact(
    *,
    product_name: str = "שם מוצר לדוגמה",
    slogan: str = "סלוגן קצר לדוגמה",
    language: str = "he",
) -> Path:
    from engine.builder2_closure_glyph_verify import write_glyph_diagnostic_crops

    preview_dir = Path(r"D:\Temp\ace_builder2_closure_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / "builder2_closure_masked_reveal_preview.mp4"
    layout, _filter_complex = render_closure_card_artifact(
        product_name=product_name,
        slogan=slogan,
        language=language,
        output_path=output_path,
        duration_seconds=3.5,
    )
    stable_png = preview_dir / "builder2_closure_stable_final.png"
    extract_closure_frame(output_path, timestamp_seconds=3.0, output_path=stable_png)
    write_glyph_diagnostic_crops(stable_png, layout, output_dir=preview_dir / "glyph_diag")
    return output_path


def assert_masked_filter_contract(filter_complex: str) -> None:
    if not closure_filter_uses_masked_bounded_overlays(filter_complex):
        raise AssertionError("closure_filter_missing_masked_bounded_overlay")
    if not closure_filter_rejects_full_frame_slide(filter_complex):
        raise AssertionError("closure_filter_allows_full_frame_slide")


def probe_timestamps_for_layout(_layout: ClosureTypographyLayout) -> Sequence[float]:
    return (0.10, 0.525, 0.90, 1.50, 3.00)


def extract_reveal_diagnostic_frames(
    video_path: Path,
    *,
    layout: ClosureTypographyLayout,
) -> ClosureRevealDiagnosticFrames:
    diag_dir = closure_reveal_diagnostic_dir()
    timestamps = {
        "before_reveal": 0.10,
        "midpoint_product": 0.525,
        "product_complete": 0.90,
        "both_complete": 1.50,
        "stable_hold": 3.00,
    }
    paths: Dict[str, Path] = {}
    for name, timestamp in timestamps.items():
        frame_path = diag_dir / f"{name}_{timestamp:.3f}.png"
        extract_closure_frame(video_path, timestamp_seconds=timestamp, output_path=frame_path)
        paths[name] = frame_path
    return ClosureRevealDiagnosticFrames(
        before_reveal=paths["before_reveal"],
        midpoint_product=paths["midpoint_product"],
        product_complete=paths["product_complete"],
        both_complete=paths["both_complete"],
        stable_hold=paths["stable_hold"],
        diagnostic_dir=diag_dir,
    )


__all__ = [
    "ClosureFrameProbe",
    "ClosureRevealDiagnosticFrames",
    "assert_eased_reveal_visible_height",
    "assert_ease_out_early_velocity_exceeds_late",
    "assert_ease_out_near_complete_at_linear_midpoint",
    "assert_masked_filter_contract",
    "closure_reveal_diagnostic_dir",
    "closure_reveal_geometry_report",
    "count_text_pixels_in_region",
    "extract_closure_frame",
    "extract_reveal_diagnostic_frames",
    "measure_text_pixels_in_region",
    "measure_visible_ink_gap_px",
    "primary_product_spec",
    "probe_closure_frame",
    "probe_timestamps_for_layout",
    "render_closure_card_artifact",
    "write_local_closure_preview_artifact",
]
