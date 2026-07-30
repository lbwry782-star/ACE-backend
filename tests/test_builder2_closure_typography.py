"""
Builder2 closure typography tests — mocks only except local font assets.
"""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_closure_copy import resolve_trusted_closure_copy
from engine.builder2_closure_only_rerender import run_builder2_closure_only_rerender
from engine.builder2_closure_render import ClosureRenderResult, render_builder2_advertising_closure_endcard
from engine.builder2_closure_rerender_inspect import inspect_builder2_closure_rerender, main as rerender_inspect_main
from engine.builder2_closure_typography import (
    BUILDER2_CLOSURE_TYPOGRAPHY_V1,
    BUILDER2_CLOSURE_TYPOGRAPHY_V2,
    BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    CLOSURE_BACKGROUND_FFMPEG_COLOR,
    CLOSURE_BACKGROUND_STYLE_VERSION,
    CLOSURE_PRODUCT_REVEAL_START_S,
    CLOSURE_SLOGAN_REVEAL_START_S,
    CLOSURE_TEXT_REVEAL_VERSION,
    MIN_ACCEPTABLE_DOMINANCE_RATIO,
    PREVIOUS_PRODUCT_SLOGAN_BLOCK_GAP_PX,
    PRODUCT_FONT_RELATIVE,
    PRODUCT_SLOGAN_BLOCK_GAP_PX,
    SLOGAN_FONT_RELATIVE,
    TARGET_PRODUCT_SLOGAN_SIZE_RATIO,
    build_closure_card_drawtext_filter,
    closure_card_lavfi_background,
    fit_builder2_closure_typography,
    font_supports_hebrew_glyphs,
    resolve_builder2_closure_product_font_path,
    resolve_builder2_closure_slogan_font_path,
    sanitize_closure_render_text,
    validate_builder2_closure_font_assets,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
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
        self.assertEqual(product, product.parent / "OgenBlack.ttf")

    def test_linux_case_sensitive_paths(self) -> None:
        root = Path(__file__).resolve().parent.parent
        product = root / PRODUCT_FONT_RELATIVE
        slogan = root / SLOGAN_FONT_RELATIVE
        self.assertTrue(product.is_file())
        self.assertTrue(slogan.is_file())
        self.assertEqual(product.name, "OgenBlack.ttf")
        self.assertEqual(slogan.name, "OgenBold.ttf")
        resolved_product = resolve_builder2_closure_product_font_path()
        self.assertEqual(resolved_product.name, "OgenBlack.ttf")
        self.assertIn("OgenBlack.ttf", resolved_product.as_posix())
        self.assertNotIn("ogenblack.ttf", resolved_product.as_posix())

    def test_hebrew_glyphs_supported(self) -> None:
        product, slogan = validate_builder2_closure_font_assets()
        self.assertTrue(font_supports_hebrew_glyphs(product))
        self.assertTrue(font_supports_hebrew_glyphs(slogan))

    def test_product_uses_ogen_black_slogan_ogen_bold(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Brand Name Example",
            slogan="Short slogan here",
            language="en",
        )
        self.assertIn("OgenBlack.ttf", layout.product_font_path.name)
        self.assertIn("OgenBold.ttf", layout.slogan_font_path.name)

    def test_product_font_larger_than_slogan(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Brand",
            slogan="Canonical slogan text",
            language="he",
        )
        self.assertGreater(layout.effective_product_font_size, layout.effective_slogan_font_size)
        self.assertGreaterEqual(layout.effective_dominance_ratio, MIN_ACCEPTABLE_DOMINANCE_RATIO)

    def test_target_ratio_near_configured_value(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="ACE",
            slogan="Short line",
            language="en",
        )
        self.assertAlmostEqual(
            layout.effective_product_font_size / layout.effective_slogan_font_size,
            TARGET_PRODUCT_SLOGAN_SIZE_RATIO,
            delta=0.35,
        )

    def test_short_product_name_remains_prominent(self) -> None:
        layout = fit_builder2_closure_typography(product_name="X", slogan="Short slogan", language="en")
        self.assertGreaterEqual(layout.effective_product_font_size, 50)

    def test_long_product_name_fits_without_overflow_error(self) -> None:
        long_name = "Very Long Product Brand Name For Adaptive Fitting Example"
        layout = fit_builder2_closure_typography(
            product_name=long_name,
            slogan="Short slogan",
            language="en",
        )
        self.assertGreaterEqual(layout.product_line_count, 1)
        self.assertLessEqual(layout.product_line_count, 2)

    def test_hebrew_layout_enables_text_shaping(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="שם מוצר",
            slogan="סלוגן קצר",
            language="he",
        )
        self.assertTrue(any(spec.use_text_shaping for spec in layout.line_specs))

    def test_distinct_product_and_slogan_blocks(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Product Block",
            slogan="Slogan Block",
            language="en",
        )
        roles = [spec.role for spec in layout.line_specs]
        self.assertIn("product", roles)
        self.assertIn("slogan", roles)

    def test_metadata_contract_flags(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Product",
            slogan="Slogan once",
            language="en",
        )
        meta = layout.metadata()
        self.assertEqual(meta["typographyContractVersion"], BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        self.assertEqual(meta["typographyContractVersion"], "builder2_closure_typography_v3")
        self.assertTrue(meta["productNameRenderedAsPlainText"])
        self.assertTrue(meta["sloganRenderedExactlyOnce"])
        self.assertFalse(meta["separateHeadlineRendered"])
        self.assertTrue(meta["brandNameDominanceSatisfied"])
        self.assertTrue(meta["closurePunctuationSanitizationApplied"])
        self.assertEqual(meta["closureBackgroundStyleVersion"], CLOSURE_BACKGROUND_STYLE_VERSION)
        self.assertEqual(meta["effectiveProductSloganGapPx"], PRODUCT_SLOGAN_BLOCK_GAP_PX)
        self.assertTrue(meta["closureTextRevealEnabled"])
        self.assertEqual(meta["closureTextRevealVersion"], CLOSURE_TEXT_REVEAL_VERSION)
        self.assertTrue(meta["productTextRevealApplied"])
        self.assertTrue(meta["sloganTextRevealApplied"])
        self.assertAlmostEqual(meta["configuredClosureSegmentDurationSeconds"], 3.5, places=2)
        self.assertAlmostEqual(meta["configuredFinalVideoDurationSeconds"], 13.5, places=2)

    def test_vertical_spacing_tighter_than_previous_version(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Brand",
            slogan="Slogan line",
            language="en",
        )
        self.assertLess(layout.product_slogan_gap_px, PREVIOUS_PRODUCT_SLOGAN_BLOCK_GAP_PX)
        self.assertEqual(layout.product_slogan_gap_px, 9)
        product_specs = [spec for spec in layout.line_specs if spec.role == "product"]
        slogan_specs = [spec for spec in layout.line_specs if spec.role == "slogan"]
        self.assertTrue(product_specs)
        self.assertTrue(slogan_specs)
        last_product_y = product_specs[-1].y_px
        first_slogan_y = slogan_specs[0].y_px
        self.assertGreaterEqual(first_slogan_y - last_product_y, layout.product_slogan_gap_px)

    def test_punctuation_removed_from_rendered_copy(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name='Brand, Inc. — "Premium"',
            slogan="Hello, world! Really?",
            language="en",
        )
        self.assertNotIn(",", layout.rendered_product_text)
        self.assertNotIn('"', layout.rendered_product_text)
        self.assertNotIn("—", layout.rendered_product_text)
        self.assertNotIn("!", layout.rendered_slogan_text)
        self.assertNotIn("?", layout.rendered_slogan_text)
        meta = layout.metadata()
        self.assertEqual(meta["renderedClosureProductText"], layout.rendered_product_text)
        self.assertTrue(meta["canonicalProductNameText"])

    def test_sanitize_collapses_spaces(self) -> None:
        self.assertEqual(sanitize_closure_render_text("Hello,   world."), "Hello world")

    def test_hebrew_punctuation_removed(self) -> None:
        rendered = sanitize_closure_render_text('שם "מוצר", טוב!')
        self.assertNotIn('"', rendered)
        self.assertNotIn(",", rendered)
        self.assertNotIn("!", rendered)

    def test_drawtext_filter_uses_upward_reveal_y_expression(self) -> None:
        layout = fit_builder2_closure_typography(
            product_name="Product",
            slogan="Slogan line",
            language="en",
        )
        from pathlib import Path as PathType

        text_paths = [PathType(f"/tmp/line_{index}.txt") for index in range(len(layout.line_specs))]
        filter_chain = build_closure_card_drawtext_filter(
            layout,
            textfile_paths=text_paths,
            ffmpeg_path_filter=lambda path: str(path),
        )
        self.assertIn("y='", filter_chain)
        self.assertNotIn(f"y={layout.line_specs[0].y_px}", filter_chain)
        self.assertIn(f"t-{CLOSURE_PRODUCT_REVEAL_START_S:.3f}", filter_chain)
        self.assertIn(f"t-{CLOSURE_SLOGAN_REVEAL_START_S:.3f}", filter_chain)

    def test_black_purple_background_lavfi(self) -> None:
        spec = closure_card_lavfi_background(width=1280, height=720, duration=3.5)
        self.assertIn(CLOSURE_BACKGROUND_FFMPEG_COLOR, spec)
        self.assertNotIn("c=black", spec)

    def test_missing_font_fails_before_render(self) -> None:
        with patch(
            "engine.builder2_closure_typography.resolve_builder2_closure_product_font_path",
            side_effect=Builder2TournamentError("builder2_closure_product_font_missing"),
        ):
            with self.assertRaises(Builder2TournamentError):
                validate_builder2_closure_font_assets()


class TestBuilder2ClosureRerenderInspect(unittest.TestCase):
    def test_completed_job_needs_upgrade_when_version_missing(self) -> None:
        state = _completed_state_for_rerender()
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["mediaCompleted"])
        self.assertTrue(report["typographyUpgradeNeeded"])
        self.assertTrue(report["canonicalProductNamePresent"])
        self.assertTrue(report["canonicalSloganPresent"])
        self.assertEqual(report["requestedTypographyContractVersion"], BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)

    def test_v2_job_needs_v3_upgrade(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_V2)
        state["mediaResume"].pop("closureOnlyRerenderCompletedForVersion", None)
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["typographyUpgradeNeeded"])
        self.assertTrue(report["closureOnlyRerenderEligible"])

    def test_v1_job_needs_v3_upgrade(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_V1)
        state["mediaResume"].pop("closureOnlyRerenderCompletedForVersion", None)
        report = inspect_builder2_closure_rerender(state)
        self.assertTrue(report["typographyUpgradeNeeded"])
        self.assertTrue(report["closureOnlyRerenderEligible"])

    def test_current_version_not_eligible(self) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        report = inspect_builder2_closure_rerender(state)
        self.assertFalse(report["typographyUpgradeNeeded"])
        self.assertFalse(report["closureOnlyRerenderEligible"])
        self.assertIn("typographyAlreadyCurrent", report["closureOnlyRerenderMissingFields"])

    @patch("engine.builder2_closure_rerender_inspect.load_tournament_state")
    def test_inspector_read_only(self, load_state) -> None:
        state = _completed_state_for_rerender()
        load_state.return_value = state
        buffer = StringIO()
        with patch.dict("os.environ", {"BUILDER2_CLOSURE_RERENDER_INSPECT_JOB_ID": state["jobId"]}, clear=False), patch(
            "sys.stdout", buffer
        ):
            code = rerender_inspect_main()
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["stateMutated"])
        self.assertEqual(payload["paidCalls"], 0)
        self.assertNotIn("productName", payload)
        self.assertNotIn("slogan", payload)
        self.assertEqual(payload["closureBackgroundStyleVersion"], CLOSURE_BACKGROUND_STYLE_VERSION)


class TestBuilder2ClosureOnlyRerender(unittest.TestCase):
    def test_resolves_trusted_copy_without_logging(self) -> None:
        state = _completed_state_for_rerender()
        product, slogan, language = resolve_trusted_closure_copy(state)
        self.assertTrue(product)
        self.assertTrue(slogan)
        self.assertEqual(language, "he")

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
        self.assertTrue(report["rawRunwayReused"])
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["openAICalls"], 0)
        self.assertTrue(report["previousFinalPreserved"])
        self.assertTrue(report["newFinalPromoted"])
        render_mock.assert_called_once()
        args, kwargs = render_mock.call_args
        self.assertTrue(str(args[0]).startswith("https://runway") or "runway" in str(args[0]))

    @patch("engine.builder2_closure_only_rerender.render_builder2_advertising_closure_endcard")
    def test_idempotent_when_version_already_applied(self, render_mock) -> None:
        state = _completed_state_for_rerender(typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        state["mediaResume"]["closureOnlyRerenderCompletedForVersion"] = BUILDER2_CLOSURE_TYPOGRAPHY_VERSION
        report = run_builder2_closure_only_rerender(
            job_id=state["jobId"],
            tournament_state=state,
            expected_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            public_base_url="https://ace.example.com",
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report.get("idempotentReuse"))
        render_mock.assert_not_called()


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
