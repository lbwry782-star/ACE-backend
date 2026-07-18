"""
Builder1 no-logo policy tests.

Run: python -m unittest tests.test_builder1_no_logo -v
"""
from __future__ import annotations

import base64
import copy
import io
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_no_logo import (
    BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK,
    BUILDER1_NO_LOGO_PLANNING_RULE,
    brand_guidelines_for_prompt,
    deterministic_no_logo_checks,
    public_payload_without_logo_assets,
    sanitize_brand_guidelines_for_builder1,
    scan_text_for_logo_violation,
)
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    build_brand_physical_user_prompt,
    build_product_name_resolution_user_prompt,
)
from engine.builder1_creative_methodology import methodology_repair_stage
from engine.builder1_strategy_judge import (
    BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT,
    deterministic_judge_checks,
    finalize_judge_result,
)
from engine.builder1_visual_prompt import build_text_to_render_block, build_visual_prompt
from engine.builder1_zip import build_builder1_single_ad_zip_bytes
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_series import _base_campaign, _parse


def _plan_dict(**overrides: Any) -> Dict[str, Any]:
    plan = copy.deepcopy(_base_campaign(2))
    plan.update(overrides)
    return plan


class TestNoLogoDeterministicChecks(unittest.TestCase):
    def test_plain_product_name_text_allowed(self) -> None:
        plan = _plan_dict()
        plan["ads"][0]["sceneDescription"] = "Energy drink can with product name printed as plain text"
        self.assertEqual(deterministic_no_logo_checks(plan), [])

    def test_lightning_bolt_logo_rejected(self) -> None:
        plan = _plan_dict()
        plan["graphicGenerator"]["recurringGraphicDevice"] = (
            "Lightning-bolt logo beside the energy drink name"
        )
        reasons = deterministic_no_logo_checks(plan)
        self.assertTrue(any(code in reasons for code in ("logo_like_brand_symbol", "packaging_contains_brand_mark")))

    def test_leaf_product_logo_rejected(self) -> None:
        text = "Place a leaf symbol as the product logo on the package"
        self.assertEqual(scan_text_for_logo_violation(text), "logo_like_brand_symbol")

    def test_monogram_rejected(self) -> None:
        self.assertEqual(scan_text_for_logo_violation("Use TG monogram above the product name"), "logo_like_brand_symbol")

    def test_badge_beside_product_name_rejected(self) -> None:
        text = "Render a badge beside the product name on the label"
        self.assertEqual(scan_text_for_logo_violation(text), "product_name_not_text_only")

    def test_symbol_on_can_as_branding_rejected(self) -> None:
        text = "Print a symbol on the can beside the product name"
        self.assertEqual(scan_text_for_logo_violation(text), "packaging_contains_brand_mark")

    def test_campaign_device_allowed_when_not_product_identity(self) -> None:
        plan = _plan_dict()
        plan["graphicGenerator"]["recurringGraphicDevice"] = "Orange corner bracket in ad composition"
        plan["graphicGenerator"]["recurringGraphicDeviceRule"] = (
            "Identical bracket appears in the layout margin, not on packaging"
        )
        self.assertEqual(deterministic_no_logo_checks(plan), [])

    def test_campaign_device_on_packaging_as_logo_rejected(self) -> None:
        text = "The recurring device printed on packaging as a logo"
        self.assertEqual(scan_text_for_logo_violation(text), "campaign_device_used_as_logo")

    def test_stylized_letter_mark_rejected(self) -> None:
        text = "Use a stylized letter as a pictorial mark beside the name"
        self.assertEqual(scan_text_for_logo_violation(text), "product_name_not_text_only")

    def test_negative_logo_instruction_not_flagged(self) -> None:
        text = "Do not design a logo on the packaging"
        self.assertIsNone(scan_text_for_logo_violation(text))


class TestSuppliedLogoSanitization(unittest.TestCase):
    def test_supplied_logo_removed_from_brand_guidelines(self) -> None:
        guidelines = {
            "primaryColor": "#FF0000",
            "logoUrl": "https://cdn.example.com/brand-logo.png",
            "logoImageBase64": "data:image/png;base64,abc",
            "tone": "bold",
        }
        sanitized = sanitize_brand_guidelines_for_builder1(guidelines)
        self.assertIsNotNone(sanitized)
        self.assertEqual(sanitized, {"primaryColor": "#FF0000", "tone": "bold"})

    def test_logo_field_not_in_brand_physical_prompt(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="VoltRush",
            product_description="Energy drink with clean citrus taste",
            detected_language="en",
            format_value="portrait",
            strategic_problem="Buyers doubt energy claims",
            relative_advantage="Clean sustained energy",
            brand_slogan="Clean Sustained Energy",
            slogan_derivation="From clean sustained energy advantage",
            implied_action="Show clean sustained energy visually",
            conceptual={"action": "Prove clean energy"},
            brand_guidelines={
                "logoUrl": "https://cdn.example.com/voltrush-logo.png",
                "accentColor": "#00AAFF",
            },
        )
        self.assertNotIn("logo", prompt.lower())
        self.assertIn("accentColor", prompt)

    def test_logo_field_not_in_product_name_resolution_prompt(self) -> None:
        prompt = build_product_name_resolution_user_prompt(
            product_description="Energy drink with clean citrus taste",
            detected_language="en",
            brand_guidelines={"logoUrl": "https://cdn.example.com/logo.png", "tone": "energetic"},
        )
        self.assertNotIn("logo", prompt.lower())
        self.assertIn("tone", prompt)

    def test_normalized_input_strips_logo_fields(self) -> None:
        normalized = normalize_builder1_input(
            product_name="VoltRush",
            product_description="Energy drink",
            format_value="portrait",
            brand_guidelines={"logoUrl": "https://cdn.example.com/logo.png", "palette": ["#111111"]},
        )
        self.assertIsNotNone(normalized.brand_guidelines)
        self.assertNotIn("logoUrl", normalized.brand_guidelines or {})


class TestVisualPromptNoLogoBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _parse(_base_campaign(2), 2)

    def test_every_image_prompt_includes_explicit_no_logo_block(self) -> None:
        for ad in self.plan.ads:
            prompt = build_visual_prompt(self.plan, ad)
            self.assertIn("=== NO PRODUCT LOGO (HIGH PRIORITY) ===", prompt)
            self.assertIn("NO PRODUCT LOGO.", prompt)
            self.assertIn("Display the product name only as plain readable text.", prompt)
            self.assertIn(
                "Do not convert decorative campaign elements into packaging logos or brand marks.",
                prompt,
            )
            self.assertLess(
                prompt.index("=== NO PRODUCT LOGO (HIGH PRIORITY) ==="),
                prompt.index("=== CAMPAIGN GRAPHIC IDENTITY"),
            )

    def test_product_name_rendered_as_plain_text_only(self) -> None:
        block = build_text_to_render_block(self.plan, self.plan.ads[0])
        self.assertIn('"TestBrand"', block)
        self.assertIn("plain readable text only", block)

    def test_visual_prompt_excludes_supplied_logo_urls(self) -> None:
        prompt = build_visual_prompt(self.plan, self.plan.ads[0])
        self.assertNotIn("logoUrl", prompt)
        self.assertNotIn("https://", prompt)


class TestFinalJudgeNoLogo(unittest.TestCase):
    def test_judge_rejects_plan_requesting_invented_logo(self) -> None:
        plan = _plan_dict()
        plan["campaignRationale"] = "Introduce a custom logo for the product on every ad"
        result = finalize_judge_result(
            model_pass=True,
            model_codes=[],
            plan_dict=plan,
            unsupported=False,
            raw={"pass": True, "rejectionReasonCodes": []},
        )
        self.assertFalse(result.passed)
        self.assertIn("invented_product_logo", result.rejection_reason_codes)

    def test_judge_system_prompt_documents_logo_fields(self) -> None:
        self.assertIn("productNameShownAsTextOnly", BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT)
        self.assertIn("invented_product_logo", BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT)
        self.assertIn("campaign_device_used_as_logo", BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT)

    def test_logo_violations_are_not_methodology_restart_codes(self) -> None:
        self.assertIsNone(methodology_repair_stage(["invented_product_logo"]))
        self.assertIsNone(methodology_repair_stage(["packaging_contains_brand_mark"]))
        self.assertIsNone(methodology_repair_stage(["supplied_logo_displayed"]))


class TestPlanningStagesNoLogoRule(unittest.TestCase):
    def test_brand_physical_stage_includes_no_logo_rule(self) -> None:
        self.assertIn(BUILDER1_NO_LOGO_PLANNING_RULE, STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_graphic_system_stage_includes_no_logo_rule(self) -> None:
        self.assertIn(BUILDER1_NO_LOGO_PLANNING_RULE, STAGE_GRAPHIC_SYSTEM_SYSTEM)

    def test_series_ads_stage_includes_no_logo_rule(self) -> None:
        self.assertIn(BUILDER1_NO_LOGO_PLANNING_RULE, STAGE_SERIES_ADS_SYSTEM)


class TestGeneratedNameAndIncrementalGeneration(unittest.TestCase):
    def test_generated_product_name_remains_plain_text_in_prompt(self) -> None:
        from dataclasses import replace

        plan = replace(_parse(_base_campaign(2), 2), product_name_resolved="AutoName")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn('"AutoName"', prompt)
        self.assertIn(BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK, prompt)

    def test_generate_again_prompt_preserves_no_logo_block(self) -> None:
        from dataclasses import replace

        from engine.builder1_campaign_store import (
            clear_memory_store_for_tests,
            create_campaign_session,
            mark_ad_generated,
            try_acquire_generation_lock,
            validate_next_ad_request,
        )

        plan = replace(_parse(_base_campaign(2), 2), product_name_resolved="AutoName")
        clear_memory_store_for_tests()
        create_campaign_session(campaign_id="logo-next", plan=plan)
        try_acquire_generation_lock("logo-next", 1)
        mark_ad_generated("logo-next", 1)
        session = validate_next_ad_request("logo-next", 2)
        prompt = build_visual_prompt(session.plan, session.plan.ads[1])
        self.assertIn(BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK, prompt)


class TestZipNoLogoMetadata(unittest.TestCase):
    def test_single_ad_zip_contains_no_logo_asset_or_metadata(self) -> None:
        image_b64 = base64.b64encode(b"fakejpeg").decode("ascii")
        payload = {
            "scope": "single_ad",
            "campaign": {
                "productNameResolved": "VoltRush",
                "brandSlogan": "Clean Energy",
                "logoUrl": "https://cdn.example.com/logo.png",
                "logoMetadata": {"description": "Bolt mark"},
            },
            "ad": {
                "index": 1,
                "imageBase64": image_b64,
                "headline": None,
                "marketingText": marketing_text_words(50),
            },
        }
        zip_bytes, _ = build_builder1_single_ad_zip_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            contents = {name: zf.read(name).decode("utf-8", errors="ignore") for name in names}
        self.assertEqual(names, ["ad-01.jpg", "ad-01.txt"])
        joined = "\n".join(contents.values()).lower()
        self.assertNotIn("logo", joined)
        self.assertNotIn("https://", joined)

    def test_public_payload_strips_logo_assets(self) -> None:
        cleaned = public_payload_without_logo_assets(
            {"productNameResolved": "VoltRush", "logoUrl": "https://x/logo.png", "brandSlogan": "Go"}
        )
        self.assertEqual(cleaned, {"productNameResolved": "VoltRush", "brandSlogan": "Go"})


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_no_logo_module(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_no_logo", text)
            self.assertNotIn("NO PRODUCT LOGO", text)


if __name__ == "__main__":
    unittest.main()
