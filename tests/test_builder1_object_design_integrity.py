"""
Builder1 object design integrity tests.

Run: python -m unittest tests.test_builder1_object_design_integrity -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_advertising_comprehension import build_planned_execution_compliance_block
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from engine.builder1_final_stages import parse_series_ads_output
from engine.builder1_image_compliance import IMAGE_COMPLIANCE_SYSTEM_PROMPT
from engine.builder1_image_compliance_contract import IMAGE_COMPLIANCE_VIOLATION_CODES
from engine.builder1_object_design_integrity import (
    OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR,
    OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
    build_composition_execution_lines,
    build_object_design_prompt_block,
    default_canonical_object_design,
    default_justified_object_design,
    validate_ad_object_design,
    validate_series_ads_object_design,
)
from engine.builder1_planning_contract import STAGE_SERIES_ADS_SYSTEM
from engine.builder1_plan_spec import series_plan_from_store_dict
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_graphic_device_necessity import _production_shaped_rain_gutter_campaign
from tests.test_builder1_staged_planning import _internal_ad_fields, _series_ads


def _ad_with_design(
    index: int,
    *,
    mode: str = OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR,
    description: str,
    deviation_reason: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    ad = {
        "index": index,
        "variationLabel": f"var-{index}",
        "newContribution": f"Contribution {index}",
        "conceptualExecution": f"Concept {index}",
        "conceptualActionProof": f"Proof {index}",
        "physicalExecution": f"Water flows into storage object {index}",
        "visualExecution": f"Photoreal front angle showing flow {index}",
        "sceneDescription": f"Exterior scene {index}",
        "headline": None,
        "headlineNeededReason": "Self-explanatory",
        "marketingText": "word " * 50,
        **_internal_ad_fields(headline=None, ad_index=index),
        "objectDesignMode": mode,
        "objectDesignDescription": description,
        "objectDesignDeviationReason": deviation_reason,
    }
    ad.update(extra)
    return ad


class TestObjectDesignValidation(unittest.TestCase):
    def test_canonical_padlock_valid(self) -> None:
        ad = _ad_with_design(
            1,
            description="Ordinary steel padlock with familiar shackle and body proportions",
        )
        self.assertEqual(validate_ad_object_design(ad), [])

    def test_transparent_padlock_justified_valid(self) -> None:
        ad = _ad_with_design(
            1,
            mode=OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
            description="Transparent padlock body so internal pins remain visible",
            deviation_reason=(
                "Transparent material is required so viewers can see the internal locking mechanism "
                "that proves the relative advantage"
            ),
        )
        self.assertEqual(validate_ad_object_design(ad), [])

    def test_pink_suitcase_palette_only_rejected(self) -> None:
        ad = _ad_with_design(
            1,
            mode=OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
            description="Bright pink hard-shell suitcase",
            deviation_reason="Matches campaign accent color for visual harmony",
        )
        self.assertIn("object_design_deviation_unjustified", validate_ad_object_design(ad))

    def test_pink_suitcase_brand_identity_justified(self) -> None:
        ad = _ad_with_design(
            1,
            mode=OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
            description="Iconic pink hard-shell suitcase matching the famous advertised product",
            deviation_reason=(
                "Pink is the factual advertised product identity and brand ownership depends on that color"
            ),
        )
        self.assertEqual(validate_ad_object_design(ad), [])

    def test_oversized_fork_justified(self) -> None:
        ad = _ad_with_design(
            1,
            mode=OBJECT_DESIGN_MODE_JUSTIFIED_DEVIATION,
            description="Giant fork towering over a plate",
            deviation_reason="Scale distortion is the advertising idea and proves the relative advantage visually",
        )
        self.assertEqual(validate_ad_object_design(ad), [])

    def test_futuristic_chair_canonical_with_neon_rejected(self) -> None:
        ad = _ad_with_design(
            1,
            description="Neon futuristic chair with glowing chrome surfaces",
        )
        self.assertIn("object_design_salient_language_unjustified", validate_ad_object_design(ad))

    def test_canonical_missing_description_rejected(self) -> None:
        ad = _ad_with_design(1, description="short")
        self.assertIn("object_design_description_missing", validate_ad_object_design(ad))

    def test_canonical_with_deviation_reason_rejected(self) -> None:
        ad = _ad_with_design(
            1,
            description="Ordinary familiar steel padlock with standard proportions",
            deviation_reason="Should be empty",
        )
        self.assertIn("object_design_deviation_reason_forbidden", validate_ad_object_design(ad))


class TestVisualPromptIntegration(unittest.TestCase):
    def test_rain_barrel_prompt_includes_canonical_design_not_arbitrary_styling(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        for i, ad in enumerate(raw["ads"], start=1):
            ad.update(
                default_canonical_object_design(
                    "Familiar immediately recognizable rain-collection barrel appropriate to a home exterior scene"
                )
            )
            ad["physicalExecution"] = (
                "Rain flows from roof gutter through downpipe into one storage barrel."
                if i == 1
                else "Water from both roof slopes converges through the pipe elbow into one storage barrel."
            )
            ad["visualExecution"] = (
                "Photoreal front angle; gutter, pipe, and splash into barrel visible."
                if i == 1
                else "Photoreal front-high angle; roof valley, pipe elbow, and splash moment visible."
            )
        raw["planningInternals"] = raw.get("planningInternals") or {}
        raw["planningInternals"]["adInternals"] = {
            idx: {
                **_internal_ad_fields(headline=None, ad_index=idx),
                **default_canonical_object_design(
                    "Familiar immediately recognizable rain-collection barrel appropriate to a home exterior scene"
                ),
            }
            for idx in (1, 2)
        }
        plan = series_plan_from_store_dict(raw)
        prompt = build_visual_prompt(plan, plan.ads[1])
        self.assertIn("OBJECT DESIGN", prompt)
        self.assertIn("CANONICAL_FAMILIAR", prompt)
        self.assertIn("Familiar immediately recognizable rain-collection barrel", prompt)
        self.assertIn("Physical action:", prompt)
        self.assertIn("Visual rendering:", prompt)
        self.assertIn("do not invent unusual materials, colors, futuristic styling", prompt.lower())
        self.assertIn("Campaign palette governs graphic design by default", prompt)
        lowered = prompt.lower()
        self.assertFalse(any(token in lowered for token in ("blue plastic", "polyethylene", "dark blue barrel")))

    def test_physical_and_visual_execution_both_reach_prompt(self) -> None:
        lines = build_composition_execution_lines(
            physical_execution="Water enters one barrel through the pipe.",
            visual_execution="Photoreal front-high angle with bright winter exterior background.",
        )
        joined = "\n".join(lines)
        self.assertIn("Physical action:", joined)
        self.assertIn("Visual rendering:", joined)

    def test_legacy_plan_without_object_design_omits_block(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        plan = series_plan_from_store_dict(raw)
        prompt = build_visual_prompt(plan, plan.ads[1])
        self.assertNotIn("=== OBJECT DESIGN (APPROVED — MANDATORY) ===", prompt)


class TestComplianceAndPlanning(unittest.TestCase):
    def test_compliance_prompt_documents_object_design_violation(self) -> None:
        self.assertIn("unplanned_object_design_deviation", IMAGE_COMPLIANCE_VIOLATION_CODES)
        self.assertIn("unplanned_object_design_deviation", IMAGE_COMPLIANCE_SYSTEM_PROMPT)
        self.assertIn("objectDesignMode", IMAGE_COMPLIANCE_SYSTEM_PROMPT)

    def test_series_ads_prompt_documents_object_design(self) -> None:
        self.assertIn("OBJECT DESIGN INTEGRITY", STAGE_SERIES_ADS_SYSTEM)
        self.assertIn("objectDesignMode", STAGE_SERIES_ADS_SYSTEM)

    def test_compliance_block_includes_object_design_fields(self) -> None:
        raw = copy.deepcopy(_production_shaped_rain_gutter_campaign())
        raw["planningInternals"]["adInternals"][2] = {
            **_internal_ad_fields(headline=None, ad_index=2),
            **default_canonical_object_design("Familiar rain-collection barrel in ordinary home-exterior form"),
        }
        plan = series_plan_from_store_dict(raw)
        block = build_planned_execution_compliance_block(plan, ad_index=2)
        self.assertIn("objectDesignMode:", block)
        self.assertIn("objectDesignDescription:", block)
        self.assertIn("unplanned_object_design_deviation", block)

    def test_parse_series_ads_requires_object_design(self) -> None:
        payload = _series_ads(2)
        for ad in payload["ads"]:
            ad.update(
                default_canonical_object_design(
                    "Ordinary immediately recognizable object appearance for this execution"
                )
            )
        parsed = parse_series_ads_output(payload, expected_ad_count=2)
        self.assertEqual(len(parsed.ads), 2)

    def test_parse_series_ads_rejects_missing_object_design(self) -> None:
        payload = _series_ads(2)
        for ad in payload["ads"]:
            ad.pop("objectDesignMode", None)
            ad.pop("objectDesignDescription", None)
            ad.pop("objectDesignDeviationReason", None)
        from engine.builder1_final_stages import StageParseError

        with self.assertRaises(StageParseError):
            parse_series_ads_output(payload, expected_ad_count=2)

    def test_integrity_scan_flags_missing_design_on_new_schema_plans(self) -> None:
        plan = copy.deepcopy(_production_shaped_rain_gutter_campaign())
        plan["ads"][1]["objectDesignMode"] = OBJECT_DESIGN_MODE_CANONICAL_FAMILIAR
        plan["ads"][1]["objectDesignDescription"] = ""
        reasons = deterministic_builder1_integrity_checks(plan)
        self.assertIn("object_design_description_missing", reasons)


class TestPaletteBoundaryPrompt(unittest.TestCase):
    def test_palette_boundary_in_object_design_block(self) -> None:
        block = build_object_design_prompt_block(
            default_canonical_object_design("Ordinary steel padlock with familiar proportions")
        )
        self.assertIn("Campaign palette governs graphic design by default", block)


if __name__ == "__main__":
    unittest.main()
