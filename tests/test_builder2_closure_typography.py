"""
Builder2 closure typography and masked-reveal contract tests.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_closure_glyph_verify import (
    assert_layout_hebrew_glyph_integrity,
    detect_missing_glyph_rectangle_pattern,
    glyph_row_width_stats,
)
from engine.builder2_closure_ffmpeg_paths import (
    ClosureFfmpegAssetSession,
    closure_ffmpeg_filter_escape_path,
    closure_path_requires_ffmpeg_staging,
    read_closure_utf8_textfile,
    verify_closure_hebrew_font_cmap,
    write_closure_utf8_textfile,
)
from engine.builder2_closure_copy import (
    resolve_closure_only_rerender_slogan_override,
    resolve_trusted_closure_copy,
)
from engine.builder2_closure_only_rerender import run_builder2_closure_only_rerender
from engine.builder2_closure_render import ClosureRenderResult, render_builder2_advertising_closure_endcard
from engine.builder2_closure_render_verify import (
    assert_eased_reveal_visible_height,
    assert_ease_out_early_velocity_exceeds_late,
    assert_ease_out_near_complete_at_linear_midpoint,
    assert_masked_filter_contract,
    closure_reveal_geometry_report,
    extract_closure_frame,
    extract_reveal_diagnostic_frames,
    measure_visible_ink_gap_px,
    primary_product_spec,
    probe_closure_frame,
    render_closure_card_artifact,
    write_local_closure_preview_artifact,
)
from engine.builder2_closure_rerender_inspect import inspect_builder2_closure_rerender
from engine.builder2_closure_typography import (
    BUILDER2_CLOSURE_TYPOGRAPHY_V1,
    BUILDER2_CLOSURE_TYPOGRAPHY_V2,
    BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    CLOSURE_BACKGROUND_FFMPEG_COLOR,
    CLOSURE_BACKGROUND_STYLE_VERSION,
    CLOSURE_PRODUCT_REVEAL_DURATION_S,
    CLOSURE_PRODUCT_REVEAL_START_S,
    CLOSURE_SLOGAN_REVEAL_DURATION_S,
    CLOSURE_SLOGAN_REVEAL_START_S,
    CLOSURE_TEXT_REVEAL_EASING,
    CLOSURE_TEXT_REVEAL_VERSION,
    REVEAL_STAGGER_RULE_VERSION,
    SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO,
    closure_reveal_derived_slogan_start_seconds,
    closure_reveal_eased_progress,
    closure_reveal_linear_progress_for_eased_travel_ratio,
    closure_reveal_product_slogan_overlap_seconds,
    closure_reveal_product_still_moving_at_timestamp,
    closure_reveal_product_travel_ratio_at_timestamp,
    closure_reveal_stable_reading_hold_seconds,
    MAX_VISIBLE_INK_GAP_PX,
    MIN_ACCEPTABLE_DOMINANCE_RATIO,
    MIN_VISIBLE_INK_GAP_PX,
    TARGET_VISIBLE_INK_GAP_PX,
    PRODUCT_FONT_RELATIVE,
    PRODUCT_SLOGAN_BLOCK_GAP_PX,
    REVEAL_PROGRESS_FUNCTION_VERSION,
    SLOGAN_FONT_RELATIVE,
    build_closure_card_masked_reveal_filter_complex,
    closure_card_lavfi_background,
    closure_filter_rejects_full_frame_slide,
    closure_filter_uses_masked_bounded_overlays,
    closure_reveal_eased_progress,
    closure_reveal_ffmpeg_y_local_expression,
    closure_reveal_settled_ink_top_in_window,
    closure_reveal_y_local_at_progress,
    assert_closure_reveal_settled_fits_window,
    expected_visible_ink_height_at_progress,
    fit_builder2_closure_typography,
    font_supports_hebrew_glyphs,
    resolve_builder2_closure_product_font_path,
    resolve_builder2_closure_slogan_font_path,
    sanitize_closure_render_text,
    validate_builder2_closure_font_assets,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.video_headline_postprocess import _ffmpeg_bin
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL, _false_completion_state


def _completed_state_for_rerender(*, typography_version: str = "") -> Dict[str, Any]:
    state = deepcopy(_false_completion_state(with_valid_closure=True))
    state["jobId"] = "8b34c172-2b8b-404a-885d-ca41a07513a7"
    state["tournamentId"] = "f5b5c684-5500-4b96-826d-df690e634c83"
    state["mediaContinuationRequired"] = False
    state["copyContractVersion"] = "builder2_single_slogan_v1"
    plan = state["winnerDevelopmentPlan"]
    plan["copyContractVersion"] = "builder2_single_slogan_v1"
    plan["canonicalCopySatisfiedBy"] = "slogan"
    plan["headlineOverlaySkipped"] = True
    plan["productNameResolved"] = plan["advertisingClosure"]["productNameText"]
    state["advertisingClosure"] = dict(plan["advertisingClosure"])
    media = state["mediaResume"]
    media["finalPublicationOutputToken"] = "3f2715c460b9494a92940bd44829cbe8"
    if typography_version:
        media["closureTypographyContractVersion"] = typography_version
        media["closureOnlyRerenderCompletedForVersion"] = typography_version
    return state


class TestBuilder2ClosureTypographyContract(unittest.TestCase):
    def test_exact_font_assets_resolve(self) -> None:
        product = resolve_builder2_closure_product_font_path()
        slogan = resolve_builder2_closure_slogan_font_path()
        self.assertTrue(product.is_file())
        self.assertTrue(slogan.is_file())
        self.assertEqual(product.name, "OgenBlack.ttf")
        self.assertEqual(slogan.name, "OgenBold.ttf")

    def test_product_uses_ogen_black_slogan_ogen_bold(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Brand Name Example",
            slogan="Short slogan here",
            language="en",
        )
        self.assertIn("OgenBlack.ttf", layout.product_font_path.name)
        self.assertIn("OgenBold.ttf", layout.slogan_font_path.name)

    def test_product_font_larger_than_slogan(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Canonical slogan text", language="he")
        self.assertGreater(layout.effective_product_font_size, layout.effective_slogan_font_size)
        self.assertGreaterEqual(layout.effective_dominance_ratio, MIN_ACCEPTABLE_DOMINANCE_RATIO)

    def test_hebrew_layout_enables_text_shaping(self) -> None:
        layout = fit_builder2_closure_typography(product_name="שם מוצר", slogan="סלוגן קצר", language="he")
        self.assertTrue(any(spec.use_text_shaping for spec in layout.line_specs))

    def test_metadata_contract_flags(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Product", slogan="Slogan once", language="en")
        meta = layout.metadata()
        self.assertEqual(meta["typographyContractVersion"], "builder2_closure_typography_v3")
        self.assertEqual(meta["closureTextRevealVersion"], CLOSURE_TEXT_REVEAL_VERSION)
        self.assertEqual(meta["closureTextRevealEasing"], CLOSURE_TEXT_REVEAL_EASING)
        self.assertEqual(meta["revealProgressFunctionVersion"], REVEAL_PROGRESS_FUNCTION_VERSION)
        self.assertTrue(meta["revealUsesFixedMask"])
        self.assertFalse(meta["revealUsesFade"])
        self.assertFalse(meta["revealUsesScale"])
        self.assertFalse(meta["revealUsesOvershoot"])
        self.assertEqual(meta["revealStaggerRuleVersion"], REVEAL_STAGGER_RULE_VERSION)
        self.assertEqual(meta["sloganStartAfterProductTravelRatio"], SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO)
        self.assertAlmostEqual(meta["productTravelRatioAtSloganStart"], 0.50, places=3)
        self.assertTrue(meta["productStillMovingAtSloganStart"])
        self.assertTrue(meta["closureTextRevealMasked"])
        self.assertTrue(meta["closureTextRevealUsesBoundedOverlay"])
        self.assertTrue(meta["visibleInkGapSatisfied"])
        self.assertEqual(meta["configuredProductSloganGapPx"], PRODUCT_SLOGAN_BLOCK_GAP_PX)
        self.assertGreaterEqual(meta["effectiveVisibleInkGapPx"], MIN_VISIBLE_INK_GAP_PX)
        self.assertLessEqual(meta["effectiveVisibleInkGapPx"], MAX_VISIBLE_INK_GAP_PX)

    def test_visible_ink_gap_not_line_box_gap(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Slogan line", language="en")
        self.assertEqual(layout.product_slogan_gap_px, PRODUCT_SLOGAN_BLOCK_GAP_PX)
        self.assertEqual(layout.product_slogan_gap_px, TARGET_VISIBLE_INK_GAP_PX)
        self.assertGreaterEqual(layout.effective_visible_ink_gap_px, MIN_VISIBLE_INK_GAP_PX)
        self.assertLessEqual(layout.effective_visible_ink_gap_px, MAX_VISIBLE_INK_GAP_PX)
        self.assertNotEqual(layout.effective_logical_product_slogan_gap_px, layout.effective_visible_ink_gap_px)

    def test_punctuation_removed_from_rendered_copy(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name='Brand, Inc. — "Premium"',
            slogan="Hello, world! Really?",
            language="en",
        )
        self.assertNotIn(",", layout.rendered_product_text)
        self.assertNotIn("!", layout.rendered_slogan_text)

    def test_masked_filtergraph_uses_bounded_overlays(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Product", slogan="Slogan line", language="en")
        text_paths = [Path(f"/tmp/line_{index}.txt") for index in range(len(layout.line_specs))]
        filter_complex, out_label = build_closure_card_masked_reveal_filter_complex(
            layout,
            textfile_paths=text_paths,
            duration_seconds=3.5,
            ffmpeg_path_filter=lambda path: str(path),
        )
        self.assertEqual(out_label, "closure_outv")
        assert_masked_filter_contract(filter_complex)
        self.assertIn("overlay=", filter_complex)
        self.assertIn("color=c=0x00000000", filter_complex)
        self.assertIn("pow(1-(", filter_complex)
        for spec in layout.line_specs:
            self.assertEqual(
                closure_reveal_ffmpeg_y_local_expression(spec),
                closure_reveal_ffmpeg_y_local_expression(spec),
            )
            self.assertGreater(spec.reveal_window_width, 0)
            self.assertGreater(spec.reveal_window_height, 0)
            self.assertIn(f"overlay=x={spec.overlay_x_px}:y={spec.overlay_y_px}", filter_complex)

    def test_hebrew_filtergraph_uses_shaping_and_expansion_none(self) -> None:
        layout = fit_builder2_closure_typography(product_name="שם מוצר", slogan="סלוגן קצר", language="he")
        session = ClosureFfmpegAssetSession.create()
        try:
            text_paths, font_paths = session.prepare_line_assets(layout.line_specs)
            filter_complex, _out = build_closure_card_masked_reveal_filter_complex(
                layout,
                textfile_paths=text_paths,
                font_paths=font_paths,
                duration_seconds=3.5,
                ffmpeg_path_filter=session.filter_path,
            )
        finally:
            session.cleanup()
        self.assertIn("text_shaping=1", filter_complex)
        self.assertIn("expansion=none", filter_complex)
        self.assertIn("OgenBlack.ttf", filter_complex)
        self.assertIn("OgenBold.ttf", filter_complex)

    def test_ease_out_cubic_progress_values(self) -> None:
        self.assertAlmostEqual(closure_reveal_eased_progress(0.0), 0.0, places=6)
        self.assertAlmostEqual(closure_reveal_eased_progress(1.0), 1.0, places=6)
        self.assertAlmostEqual(closure_reveal_eased_progress(0.25), 0.578125, places=6)
        self.assertAlmostEqual(closure_reveal_eased_progress(0.50), 0.875, places=6)
        self.assertAlmostEqual(closure_reveal_eased_progress(0.75), 0.984375, places=6)

    def test_ease_out_y_exact_at_start_and_end(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Slogan", language="en")
        spec = primary_product_spec(layout)
        self.assertAlmostEqual(closure_reveal_y_local_at_progress(spec, 0.0), spec.hidden_y_local_px, places=3)
        self.assertAlmostEqual(closure_reveal_y_local_at_progress(spec, 1.0), spec.final_y_local_px, places=3)
        assert_closure_reveal_settled_fits_window(spec)

    def test_hebrew_product_settled_ink_has_border_safe_top_pad(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="דובי",
            slogan="שוקולד דובאי תוצרת ישראל",
            language="he",
        )
        spec = primary_product_spec(layout)
        ink_top = closure_reveal_settled_ink_top_in_window(spec)
        self.assertGreaterEqual(ink_top, float(spec.reveal_window_top_pad_px))
        self.assertGreaterEqual(ink_top, 4.0)
        assert_closure_reveal_settled_fits_window(spec)

    def test_product_overlay_aligns_canvas_ink_top(self) -> None:
        layout = fit_builder2_closure_typography(product_name="דובי", slogan="סלוגן קצר", language="he")
        spec = primary_product_spec(layout)
        canvas_ink_top = spec.overlay_y_px + spec.reveal_window_top_pad_px
        self.assertEqual(canvas_ink_top, spec.y_px + spec.ink_bbox[1])

    def test_visible_ink_gap_does_not_steal_product_top_pad(self) -> None:
        layout = fit_builder2_closure_typography(product_name="דובי", slogan="שוקולד דובאי תוצרת ישראל", language="he")
        product = primary_product_spec(layout)
        slogan = next(spec for spec in layout.line_specs if spec.role == "slogan")
        self.assertGreaterEqual(layout.effective_visible_ink_gap_px, MIN_VISIBLE_INK_GAP_PX)
        assert_closure_reveal_settled_fits_window(product)
        assert_closure_reveal_settled_fits_window(slogan)

    def test_ease_out_monotonic_and_non_linear(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Slogan", language="en")
        spec = primary_product_spec(layout)
        samples = [closure_reveal_y_local_at_progress(spec, p / 20.0) for p in range(21)]
        for earlier, later in zip(samples, samples[1:]):
            self.assertLessEqual(later, earlier)
        linear_mid = closure_reveal_y_local_at_progress(spec, 0.5)
        linear_end = spec.final_y_local_px
        linear_start = spec.hidden_y_local_px
        linear_mid_travel = linear_start + (linear_end - linear_start) * 0.5
        self.assertNotAlmostEqual(linear_mid, linear_mid_travel, places=2)

    def test_ease_out_midpoint_height_exceeds_linear_midpoint(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Slogan", language="en")
        spec = primary_product_spec(layout)
        eased_height = expected_visible_ink_height_at_progress(spec, 0.5)
        linear_y = spec.hidden_y_local_px + (spec.final_y_local_px - spec.hidden_y_local_px) * 0.5
        ink_top = linear_y + spec.ink_bbox[1]
        ink_bottom = linear_y + spec.ink_bbox[3]
        visible_top = max(0.0, ink_top)
        visible_bottom = min(float(spec.reveal_window_height), ink_bottom)
        linear_height = max(0, int(round(visible_bottom - visible_top)))
        self.assertGreater(eased_height, linear_height)

    def test_product_and_slogan_share_canonical_easing_helper(self) -> None:
        layout = fit_builder2_closure_typography(product_name="Brand", slogan="Slogan", language="en")
        product_expr = closure_reveal_ffmpeg_y_local_expression(primary_product_spec(layout))
        slogan_expr = closure_reveal_ffmpeg_y_local_expression(
            next(spec for spec in layout.line_specs if spec.role == "slogan")
        )
        self.assertIn("pow(1-(", product_expr)
        self.assertIn("pow(1-(", slogan_expr)
        self.assertNotIn("max(0\\,1-min(1\\,", product_expr)

    def test_regression_rejects_unmasked_full_canvas_drawtext(self) -> None:
        bad = "[0:v]drawtext=fontfile='f.ttf':text='x':x=0:y='100+t*10'[outv]"
        self.assertFalse(closure_filter_uses_masked_bounded_overlays(bad))
        self.assertFalse(closure_filter_rejects_full_frame_slide(bad))

    def test_black_purple_background_lavfi(self) -> None:
        spec = closure_card_lavfi_background(width=1280, height=720, duration=3.5)
        self.assertIn(CLOSURE_BACKGROUND_FFMPEG_COLOR, spec)


class TestBuilder2ClosureFfmpegPaths(unittest.TestCase):
    def test_linux_font_path_ffmpeg_safe(self) -> None:
        from pathlib import PurePosixPath

        escaped = closure_ffmpeg_filter_escape_path(
            PurePosixPath("/opt/render/project/src/assets/fonts/OgenBlack.ttf")
        )
        self.assertEqual(escaped, "/opt/render/project/src/assets/fonts/OgenBlack.ttf")

    def test_windows_font_path_ffmpeg_safe(self) -> None:
        escaped = closure_ffmpeg_filter_escape_path(Path(r"D:\ACE-Backend\assets\fonts\OgenBlack.ttf"))
        self.assertEqual(escaped, "D\\:/ACE-Backend/assets/fonts/OgenBlack.ttf")

    def test_unicode_directory_requires_staging(self) -> None:
        path = Path("D:/אס2/ACE-Backend/assets/fonts/OgenBlack.ttf")
        self.assertTrue(closure_path_requires_ffmpeg_staging(path))

    def test_textfile_utf8_without_bom(self) -> None:
        target = Path(self._tmpdir()) / "line.txt"
        write_closure_utf8_textfile(target, "שם מוצר\n")
        raw = target.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(read_closure_utf8_textfile(target), "שם מוצר\n")

    def test_hebrew_cmap_maps_real_glyphs(self) -> None:
        from engine.builder2_closure_typography import resolve_builder2_closure_product_font_path

        font = resolve_builder2_closure_product_font_path()
        verify_closure_hebrew_font_cmap(font, "שם מוצר לדוגמה")

    def test_staged_fonts_use_ogen_assets(self) -> None:
        layout = fit_builder2_closure_typography(product_name="שם מוצר", slogan="סלוגן קצר", language="he")
        session = ClosureFfmpegAssetSession.create()
        try:
            _text_paths, font_paths = session.prepare_line_assets(layout.line_specs)
            self.assertIn("OgenBlack.ttf", font_paths[0].name)
            self.assertIn("OgenBold.ttf", font_paths[1].name)
            if closure_path_requires_ffmpeg_staging(layout.product_font_path):
                self.assertTrue(str(font_paths[0]).startswith(str(session.work_dir)))
        finally:
            session.cleanup()

    def _tmpdir(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="ace_closure_path_test_")


class TestBuilder2ClosureRevealStagger(unittest.TestCase):
    def test_inverse_easing_half_travel_linear_progress(self) -> None:
        p = closure_reveal_linear_progress_for_eased_travel_ratio(0.50)
        self.assertAlmostEqual(p, 1.0 - (0.5 ** (1.0 / 3.0)), places=6)
        self.assertAlmostEqual(p, 0.206299, places=5)
        self.assertAlmostEqual(closure_reveal_eased_progress(p), 0.50, places=6)

    def test_derived_slogan_start_from_product_timing(self) -> None:
        derived = closure_reveal_derived_slogan_start_seconds()
        self.assertAlmostEqual(
            derived,
            CLOSURE_PRODUCT_REVEAL_START_S
            + (CLOSURE_PRODUCT_REVEAL_DURATION_S * closure_reveal_linear_progress_for_eased_travel_ratio(0.50)),
            places=6,
        )
        self.assertAlmostEqual(derived, 0.334, places=2)
        self.assertEqual(CLOSURE_SLOGAN_REVEAL_START_S, derived)

    def test_product_travel_at_slogan_start_is_half(self) -> None:
        ratio = closure_reveal_product_travel_ratio_at_timestamp(
            timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S,
        )
        self.assertAlmostEqual(ratio, SLOGAN_START_AFTER_PRODUCT_TRAVEL_RATIO, places=3)

    def test_reveal_timings_preserve_stagger_and_hold(self) -> None:
        self.assertEqual(CLOSURE_PRODUCT_REVEAL_START_S, 0.20)
        self.assertLess(CLOSURE_PRODUCT_REVEAL_START_S, CLOSURE_SLOGAN_REVEAL_START_S)
        product_end = CLOSURE_PRODUCT_REVEAL_START_S + CLOSURE_PRODUCT_REVEAL_DURATION_S
        self.assertLess(CLOSURE_SLOGAN_REVEAL_START_S, product_end)
        self.assertTrue(
            closure_reveal_product_still_moving_at_timestamp(
                timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S,
            )
        )
        overlap = closure_reveal_product_slogan_overlap_seconds()
        self.assertGreater(overlap, 0.0)
        slogan_end = CLOSURE_SLOGAN_REVEAL_START_S + CLOSURE_SLOGAN_REVEAL_DURATION_S
        hold = closure_reveal_stable_reading_hold_seconds(closure_duration_seconds=3.5)
        self.assertGreaterEqual(hold, 1.5)
        self.assertGreaterEqual(3.5 - slogan_end, 1.5)


class TestBuilder2ClosureEaseOutTiming(unittest.TestCase):
    def test_reveal_timings_delegate_to_stagger_rule(self) -> None:
        self.assertEqual(REVEAL_STAGGER_RULE_VERSION, "builder2_closure_reveal_stagger_half_product_travel_v1")
        self.assertAlmostEqual(CLOSURE_SLOGAN_REVEAL_START_S, 0.334, places=2)


@unittest.skipUnless(_ffmpeg_bin(), "ffmpeg not available")
class TestBuilder2ClosureMaskedRevealRender(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="ace_closure_render_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_render_level_masked_reveal_frames(self) -> None:
        card_path = self._tmpdir / "closure_card.mp4"
        layout, filter_complex = render_closure_card_artifact(
            product_name="שם מוצר לדוגמה",
            slogan="סלוגן קצר לדוגמה",
            language="he",
            output_path=card_path,
            duration_seconds=3.5,
        )
        assert_masked_filter_contract(filter_complex)
        product_spec = primary_product_spec(layout)
        geometry = closure_reveal_geometry_report(product_spec, timestamp_seconds=0.525)
        self.assertGreaterEqual(
            geometry["hiddenYLocalPx"] + geometry["glyphInkTopBoundPx"],
            geometry["revealWindowHeightPx"],
        )
        self.assertGreaterEqual(geometry["finalYLocalPx"] + geometry["glyphInkTopBoundPx"], 0)
        self.assertLessEqual(
            geometry["finalYLocalPx"] + geometry["glyphInkBottomBoundPx"],
            geometry["revealWindowHeightPx"],
        )
        self.assertEqual(
            geometry["revealTravelPx"],
            geometry["hiddenYLocalPx"] - geometry["finalYLocalPx"],
        )

        self.assertAlmostEqual(geometry["calculatedLinearProgress"], 0.5, places=2)
        self.assertAlmostEqual(geometry["calculatedEasedProgress"], 0.875, places=3)

        from engine.builder2_closure_render import _ffprobe_duration_seconds, _FFPROBE_TIMEOUT

        measured_closure = _ffprobe_duration_seconds(card_path, _FFPROBE_TIMEOUT)
        self.assertAlmostEqual(measured_closure, 3.5, delta=0.2)

        diagnostics = extract_reveal_diagnostic_frames(card_path, layout=layout)
        early_start_path = diagnostics.diagnostic_dir / "early_reveal_0.280.png"
        early_end_path = diagnostics.diagnostic_dir / "early_reveal_0.320.png"
        late_start_t = 0.720
        late_end_t = 0.760
        late_start_path = diagnostics.diagnostic_dir / f"late_reveal_{late_start_t:.3f}.png"
        late_end_path = diagnostics.diagnostic_dir / f"late_reveal_{late_end_t:.3f}.png"
        slogan_pre_path = diagnostics.diagnostic_dir / "slogan_pre_start.png"
        slogan_post_path = diagnostics.diagnostic_dir / "slogan_post_start.png"
        extract_closure_frame(card_path, timestamp_seconds=0.280, output_path=early_start_path)
        extract_closure_frame(card_path, timestamp_seconds=0.320, output_path=early_end_path)
        extract_closure_frame(card_path, timestamp_seconds=late_start_t, output_path=late_start_path)
        extract_closure_frame(card_path, timestamp_seconds=late_end_t, output_path=late_end_path)
        extract_closure_frame(
            card_path,
            timestamp_seconds=max(0.0, CLOSURE_SLOGAN_REVEAL_START_S - 0.02),
            output_path=slogan_pre_path,
        )
        extract_closure_frame(
            card_path,
            timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S + 0.05,
            output_path=slogan_post_path,
        )

        probes = {
            0.10: probe_closure_frame(diagnostics.before_reveal, layout, timestamp_seconds=0.10),
            0.280: probe_closure_frame(early_start_path, layout, timestamp_seconds=0.280),
            0.320: probe_closure_frame(early_end_path, layout, timestamp_seconds=0.320),
            0.525: probe_closure_frame(diagnostics.midpoint_product, layout, timestamp_seconds=0.525),
            late_start_t: probe_closure_frame(late_start_path, layout, timestamp_seconds=late_start_t),
            late_end_t: probe_closure_frame(late_end_path, layout, timestamp_seconds=late_end_t),
            0.90: probe_closure_frame(diagnostics.product_complete, layout, timestamp_seconds=0.90),
            1.50: probe_closure_frame(diagnostics.both_complete, layout, timestamp_seconds=1.50),
            3.00: probe_closure_frame(diagnostics.stable_hold, layout, timestamp_seconds=3.00),
            "slogan_pre": probe_closure_frame(
                slogan_pre_path,
                layout,
                timestamp_seconds=max(0.0, CLOSURE_SLOGAN_REVEAL_START_S - 0.02),
            ),
            "slogan_post": probe_closure_frame(
                slogan_post_path,
                layout,
                timestamp_seconds=CLOSURE_SLOGAN_REVEAL_START_S + 0.05,
            ),
        }

        before = probes[0.10]
        early_start = probes[0.280]
        early_end = probes[0.320]
        mid_product = probes[0.525]
        late_start = probes[late_start_t]
        late_end = probes[late_end_t]
        product_done = probes[0.90]
        both_done = probes[1.50]
        stable = probes[3.00]
        slogan_pre = probes["slogan_pre"]
        slogan_post = probes["slogan_post"]

        self.assertLessEqual(before.role_visible_ink_heights.get("product", 99), 4)
        self.assertLessEqual(before.role_visible_ink_heights.get("slogan", 99), 4)
        self.assertEqual(before.role_bright_counts.get("product", 0), 0)
        self.assertEqual(before.role_bright_counts.get("slogan", 0), 0)
        self.assertEqual(before.role_outside_window_pixels.get("product", 0), 0)
        self.assertGreater(early_end.role_visible_ink_heights.get("product", 0), 0)

        stable_product_height = stable.role_visible_ink_heights.get("product", 0)
        self.assertGreater(stable_product_height, 0)
        assert_ease_out_near_complete_at_linear_midpoint(
            layout,
            timestamp_seconds=0.525,
            measured_visible_height=mid_product.role_visible_ink_heights.get("product", 0),
            stable_visible_height=stable_product_height,
        )
        assert_eased_reveal_visible_height(
            layout,
            timestamp_seconds=0.525,
            measured_visible_height=mid_product.role_visible_ink_heights.get("product", 0),
            stable_visible_height=stable_product_height,
        )
        early_delta = (
            early_end.role_visible_ink_heights.get("product", 0)
            - early_start.role_visible_ink_heights.get("product", 0)
        )
        late_delta = (
            late_end.role_visible_ink_heights.get("product", 0)
            - late_start.role_visible_ink_heights.get("product", 0)
        )
        assert_ease_out_early_velocity_exceeds_late(
            early_start=0.280,
            early_end=0.320,
            late_start=late_start_t,
            late_end=late_end_t,
            early_delta=early_delta,
            late_delta=late_delta,
        )
        self.assertEqual(slogan_pre.role_bright_counts.get("slogan", 0), 0)
        self.assertGreater(slogan_post.role_visible_ink_heights.get("slogan", 0), 0)
        self.assertLess(
            slogan_post.role_visible_ink_heights.get("product", 0),
            stable_product_height,
        )
        self.assertGreater(slogan_post.role_visible_ink_heights.get("product", 0), 0)
        self.assertEqual(mid_product.role_outside_window_pixels.get("product", 0), 0)
        self.assertAlmostEqual(
            product_done.role_visible_ink_heights.get("product", 0),
            stable_product_height,
            delta=max(4, int(round(stable_product_height * 0.12))),
        )
        self.assertGreater(both_done.role_bright_counts.get("slogan", 0), 0)
        self.assertEqual(before.role_bright_counts.get("slogan", 0), 0)
        self.assertEqual(stable.role_outside_window_pixels.get("product", 0), 0)
        self.assertEqual(stable.role_outside_window_pixels.get("slogan", 0), 0)
        self.assertIn("pow(1-(", filter_complex)

        visible_gap = measure_visible_ink_gap_px(diagnostics.stable_hold, layout)
        self.assertGreaterEqual(visible_gap, MIN_VISIBLE_INK_GAP_PX - 2)
        self.assertLessEqual(visible_gap, MAX_VISIBLE_INK_GAP_PX + 2)
        assert_layout_hebrew_glyph_integrity(diagnostics.stable_hold, layout)
        self.assertIn("expansion=none", filter_complex)
        self.assertIn("text_shaping=1", filter_complex)

        self._diagnostic_dir = diagnostics.diagnostic_dir

    def test_local_preview_artifact_path_reported(self) -> None:
        preview_path = write_local_closure_preview_artifact()
        self.assertTrue(preview_path.is_file())
        self.assertIn("builder2_closure_masked_reveal_preview.mp4", preview_path.name)


class TestBuilder2ClosureRerenderInspect(unittest.TestCase):
    def test_completed_job_needs_upgrade_when_version_missing(self) -> None:
        state = _completed_state_for_rerender()
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["typographyUpgradeNeeded"])

    def test_force_allows_rerender_when_typography_current(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        blocked = inspect_builder2_closure_rerender(state)
        self.assertIn("typographyAlreadyCurrent", blocked["closureOnlyRerenderMissingFields"])
        forced = inspect_builder2_closure_rerender(state, force=True)
        self.assertNotIn("typographyAlreadyCurrent", forced["closureOnlyRerenderMissingFields"])
        self.assertTrue(forced["closureOnlyRerenderEligible"])

    def test_v2_job_needs_v3_upgrade(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_V2)
        state["mediaResume"].pop("closureOnlyRerenderCompletedForVersion", None)
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["typographyUpgradeNeeded"])

    def test_v1_job_needs_v3_upgrade(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_V1)
        state["mediaResume"].pop("closureOnlyRerenderCompletedForVersion", None)
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["typographyUpgradeNeeded"])


class TestBuilder2ClosureOnlyRerender(unittest.TestCase):
    @patch("engine.builder2_closure_only_rerender.save_tournament_state")
    @patch("engine.builder2_closure_only_rerender.video_job_mark_done")
    @patch("engine.builder2_closure_only_rerender.publish_builder2_durable_final_video")
    @patch("engine.builder2_closure_only_rerender.require_builder2_web_storage_capability")
    @patch("engine.builder2_closure_only_rerender.render_builder2_advertising_closure_endcard")
    def test_rerender_reuses_raw_runway_zero_external_calls(
        self,
        render_mock,
        _capability,
        publish_mock,
        _mark_done,
        save_mock,
    ) -> None:
        state = _completed_state_for_rerender()
        typography_meta = fit_builder2_closure_typography(
            product_name=state["advertisingClosure"]["productNameText"],
            slogan=state["advertisingClosure"]["sloganText"],
            language="he",
        ).metadata()
        from tests.builder2_durable_finalization_test_helpers import durable_publication_result

        render_mock.return_value = ClosureRenderResult(
            public_url="",
            local_path="/tmp/out.mp4",
            measured_duration_seconds=13.51,
            output_token="newtoken0123456789abcdef01234567",
            input_fingerprint="abc",
            typography_metadata=typography_meta,
        )
        publish_mock.return_value = durable_publication_result(CLOSURE_URL)
        report = run_builder2_closure_only_rerender(
            job_id=state["jobId"],
            tournament_state=state,
            expected_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            public_base_url="https://ace.example.com",
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["runwaySubmissionCalls"], 0)

    @patch("engine.builder2_closure_only_rerender.save_tournament_state")
    @patch("engine.builder2_closure_only_rerender.video_job_mark_done")
    @patch("engine.builder2_closure_only_rerender.publish_builder2_durable_final_video")
    @patch("engine.builder2_closure_only_rerender.require_builder2_web_storage_capability")
    @patch("engine.builder2_closure_only_rerender.render_builder2_advertising_closure_endcard")
    def test_rerender_uses_slogan_override_without_openai(
        self,
        render_mock,
        _capability,
        publish_mock,
        _mark_done,
        save_mock,
    ) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        corrected = "שוקולד דובאי תוצרת ישראל"
        with patch.dict(
            os.environ,
            {
                "BUILDER2_CLOSURE_ONLY_RERENDER_FORCE": "1",
                "BUILDER2_CLOSURE_ONLY_RERENDER_SLOGAN_TEXT": corrected,
            },
            clear=False,
        ):
            product, slogan, _lang = resolve_trusted_closure_copy(state)
            self.assertEqual(product, state["advertisingClosure"]["productNameText"])
            self.assertEqual(slogan, corrected)

            typography_meta = fit_builder2_closure_typography(
                product_name=product,
                slogan=slogan,
                language="he",
            ).metadata()
            from tests.builder2_durable_finalization_test_helpers import durable_publication_result

            render_mock.return_value = ClosureRenderResult(
                public_url="",
                local_path="/tmp/out.mp4",
                measured_duration_seconds=13.51,
                output_token="newtoken0123456789abcdef01234567",
                input_fingerprint="abc",
                typography_metadata=typography_meta,
            )
            publish_mock.return_value = durable_publication_result(CLOSURE_URL)
            report = run_builder2_closure_only_rerender(
                job_id=state["jobId"],
                tournament_state=state,
                expected_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
                public_base_url="https://ace.example.com",
            )
        self.assertTrue(report["ok"])
        self.assertTrue(report.get("closureSloganOverrideApplied"))
        self.assertEqual(report.get("renderedClosureSloganText"), corrected)
        self.assertEqual(render_mock.call_args.kwargs["slogan"], corrected)
        saved_state = save_mock.call_args[0][1]
        self.assertEqual(saved_state["advertisingClosure"]["sloganText"], corrected)
        self.assertEqual(saved_state["mediaResume"]["closureSloganOverride"], corrected)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 1)
        self.assertTrue(report["newFinalPromoted"])

    def test_persisted_media_override_preferred_when_env_missing(self) -> None:
        state = _completed_state_for_rerender()
        state["mediaResume"]["closureSloganOverride"] = "שוקולד דובאי תוצרת ישראל"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BUILDER2_CLOSURE_ONLY_RERENDER_SLOGAN_TEXT", None)
            override = resolve_closure_only_rerender_slogan_override(state=state)
        self.assertEqual(override, "שוקולד דובאי תוצרת ישראל")


class TestBuilder2ClosureTypographyBuilder1Isolation(unittest.TestCase):
    def test_builder1_unchanged(self) -> None:
        import glob

        root = os.path.dirname(os.path.dirname(__file__))
        for path in glob.glob(os.path.join(root, "engine", "builder1*.py")):
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("OgenBlack.ttf", source)
            self.assertNotIn("builder2_closure_typography", source)


if __name__ == "__main__":
    unittest.main()
