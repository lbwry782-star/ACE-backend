"""
Builder1 staged planning pipeline tests.

Run: python -m unittest tests.test_builder1_staged_planning -v
"""
from __future__ import annotations

from tests.builder1_test_helpers import marketing_text_words

import copy
import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from engine.builder1_final_stages import (
    BrandPhysicalOutput,
    assemble_builder1_campaign,
    parse_brand_physical_output,
    parse_graphic_system_output,
    parse_series_ads_output,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_plan_spec import campaign_identity_to_dict, series_plan_to_store_dict
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
)
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_staged_parsers import (
    StageParseError,
    assemble_builder1_series_plan,
    detect_brief_language,
    parse_conceptual_scan,
    parse_conceptual_selection,
    parse_strategy_scan,
    parse_strategy_selection,
)
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT


def _graphic(*, missing_palette: bool = False, missing_color: bool = False, snake_case: bool = False) -> Dict[str, Any]:
    g = {
        "palette": {
            "dominant": "#111111",
            "secondary": "#EEEEEE",
            "accent": "#FF5500",
            "background": "#F5F5F5",
            "text": "#222222",
        },
        "layoutTemplate": "visual_right_copy_left",
        "headlinePlacement": "top_left",
        "headlineAlignment": "right",
        "headlineMaxWidthPercent": 34,
        "brandBlockPlacement": "bottom_left",
        "sloganPlacement": "bottom_left",
        "copySafeArea": {"side": "left", "widthPercent": 38},
        "typographyStyle": "bold_geometric_sans",
        "headlineScale": "large",
        "brandScale": "small",
        "sloganScale": "medium",
        "imageStyle": "editorial_photography",
        "backgroundTreatment": "solid",
        "borderTreatment": "none",
        "recurringGraphicDevice": "Orange corner bracket",
        "recurringGraphicDeviceRule": "Identical bracket on every ad",
        "shapeLanguage": "Angular geometric frames",
        "framingRule": "Subject with copy-side negative space",
        "spacingRule": "Wide outer margins",
    }
    if missing_palette:
        del g["palette"]
    if missing_color:
        del g["palette"]["text"]
    if snake_case:
        return {
            "palette": g["palette"],
            "layout_template": g["layoutTemplate"],
            "headline_placement": g["headlinePlacement"],
            "headline_alignment": g["headlineAlignment"],
            "headline_max_width_percent": g["headlineMaxWidthPercent"],
            "brand_block_placement": g["brandBlockPlacement"],
            "slogan_placement": g["sloganPlacement"],
            "copy_safe_area": g["copySafeArea"],
            "typography_style": g["typographyStyle"],
            "headline_scale": g["headlineScale"],
            "brand_scale": g["brandScale"],
            "slogan_scale": g["sloganScale"],
            "image_style": g["imageStyle"],
            "background_treatment": g["backgroundTreatment"],
            "border_treatment": g["borderTreatment"],
            "recurring_graphic_device": g["recurringGraphicDevice"],
            "recurring_graphic_device_rule": g["recurringGraphicDeviceRule"],
            "shape_language": g["shapeLanguage"],
            "framing_rule": g["framingRule"],
            "spacing_rule": g["spacingRule"],
        }
    return g


def _brand_physical(*, missing_natural: bool = False, missing_role: bool = False) -> Dict[str, Any]:
    payload = {
        "productNameResolved": "TestBrand",
        "brandSlogan": "Built To Last",
        "sloganDerivation": "From durability advantage",
        "sloganAction": "Trust everyday use",
        "physicalGenerator": "Rubber ball family",
        "physicalGeneratorNaturalPurpose": "Absorb impact",
        "physicalGeneratorCampaignRole": "Durability metaphor",
        "mediumParticipates": False,
        "mediumRole": "",
        "campaignRationale": "Ownable durability story",
    }
    if missing_natural:
        del payload["physicalGeneratorNaturalPurpose"]
    if missing_role:
        del payload["physicalGeneratorCampaignRole"]
    return payload


def _series_ads(ad_count: int = 2, *, series_string: bool = False, incomplete_series: bool = False,
                omit_indexes: bool = False, string_indexes: bool = False, wrong_indexes: bool = False,
                too_few_ads: bool = False) -> Dict[str, Any]:
    ads = []
    for i in range(1, ad_count + 1 if not too_few_ads else max(1, ad_count - 1)):
        ad = {
            "variationLabel": f"var-{i}",
            "newContribution": f"Contribution {i}",
            "conceptualExecution": f"Perform drop proof variant {i}",
            "conceptualActionProof": f"Proof {i}",
            "physicalExecution": f"Object variant {i}",
            "visualExecution": f"Visual variant {i}",
            "sceneDescription": f"Scene {i}",
            "headline": None if i == 1 else f"Line {i}",
            "headlineNeededReason": "Needed" if i > 1 else "Self-explanatory",
                "marketingText": marketing_text_words(50, prefix=f"m{i}"),
        }
        if not omit_indexes:
            if string_indexes:
                ad["index"] = str(i)
            elif wrong_indexes:
                ad["index"] = i + 10
            else:
                ad["index"] = i
        ads.append(ad)
    if series_string:
        series = "situations progression"
    elif incomplete_series:
        series = {"type": "situations", "principle": "", "progression": "Escalating"}
    else:
        series = {"type": "situations", "principle": "Different drop contexts", "progression": "Escalating severity"}
    return {"seriesGenerator": series, "ads": ads}


LENSES = [
    "economic", "perceptual", "emotional", "operational", "time", "accessibility",
    "expertise", "challenger_positioning", "participation", "simplicity", "specialization", "category_convention",
]


def _strategy_scan_payload(*, string_candidate: bool = False) -> Dict[str, Any]:
    candidates: List[Any] = []
    for i in range(1, 13):
        candidates.append(
            {
                "id": f"S{i:02d}",
                "lens": LENSES[i - 1],
                "strategicProblem": f"Distinct buyer problem {i}",
                "relativeAdvantage": f"Distinct advantage {i}",
                "briefSupport": "Follows from brief reinforced shell mention",
                "advantageSource": "observable_product_mechanism",
                "claimRisk": "low",
                "campaignExecutableNow": True,
                "requiresClientConsultation": False,
                "clientActionLevel": "none",
                "implementationCostLevel": "none",
                "simpleStrategicAction": None,
            }
        )
    if string_candidate:
        candidates[0] = "bad string candidate"
    return {"candidates": candidates}


def _conceptual_scan_payload(*, incomplete: bool = False) -> Dict[str, Any]:
    candidates = []
    for i in range(1, 7):
        c = {
            "id": f"C{i:02d}",
            "generator": f"Stress-test mechanism {i}",
            "action": f"Drop and survive variant {i}",
            "input": "Everyday carry item",
            "transformation": f"Impact absorbed step {i}",
            "result": f"Visible durability proof {i}",
            "whyItExpressesAdvantage": "Shows advantage through action",
            "seriesPotential": "Escalating drop contexts",
        }
        if incomplete and i == 1:
            c["action"] = ""
        candidates.append(c)
    return {"candidates": candidates}


def _selected_strategy():
    return parse_strategy_scan(
        _strategy_scan_payload(),
        product_description="Reinforced shell product for daily carry",
    )[0]


def _selected_conceptual():
    return parse_conceptual_scan(_conceptual_scan_payload())[0]


def _early_stage_responses(ad_count: int = 2) -> Dict[str, Any]:
    return {
        STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM: {"productNameResolved": "TestBrand"},
        STAGE_STRATEGY_SCAN_SYSTEM: _strategy_scan_payload(),
        STAGE_STRATEGY_SELECT_SYSTEM: {
            "selectedCandidateId": "S01",
            "selectionReason": "Strongest brief fit",
            "strategyFamily": "durability",
            "scores": {
                "truth": 9, "briefSupport": 9, "relevance": 8, "distinctiveness": 8,
                "brandOwnership": 8, "persuasiveStrength": 8, "seriesPotential": 9,
                "conceptualActionPotential": 9,
            },
        },
        STAGE_CONCEPTUAL_SCAN_SYSTEM: _conceptual_scan_payload(),
        STAGE_CONCEPTUAL_SELECT_SYSTEM: {
            "selectedCandidateId": "C01",
            "selectionReason": "Clearest action",
            "scores": {
                "advantageConnection": 9, "actionClarity": 9, "visualPower": 8,
                "seriesPotential": 9, "distinctiveness": 8, "physicalIndependence": 8,
            },
        },
        BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT: {
            "pass": True,
            "rejectionReasonCodes": [],
            "unsupportedClaimDetected": False,
        },
    }


def _full_final_responses(ad_count: int = 2) -> Dict[str, Any]:
    responses = _early_stage_responses(ad_count)
    responses[STAGE_BRAND_PHYSICAL_SYSTEM] = _brand_physical()
    responses[STAGE_GRAPHIC_SYSTEM_SYSTEM] = _graphic()
    responses[STAGE_SERIES_ADS_SYSTEM] = _series_ads(ad_count)
    return responses


class TestFinalSubstageParsers(unittest.TestCase):
    def test_5a_missing_natural_purpose_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(_brand_physical(missing_natural=True))
        self.assertIn("missing_physicalGeneratorNaturalPurpose", ctx.exception.reasons)

    def test_5a_missing_campaign_role_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(_brand_physical(missing_role=True))
        self.assertIn("missing_physicalGeneratorCampaignRole", ctx.exception.reasons)

    def test_5a_medium_participates_string_false_normalized(self) -> None:
        payload = _brand_physical()
        payload["mediumParticipates"] = "false"
        result = parse_brand_physical_output(payload)
        self.assertFalse(result.medium_participates)
        self.assertEqual(result.medium_role, "")

    def test_5b_missing_palette_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_graphic_system_output(_graphic(missing_palette=True))
        self.assertIn("graphic_generator_missing_palette", ctx.exception.reasons)

    def test_5b_missing_palette_color_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_graphic_system_output(_graphic(missing_color=True))
        self.assertIn("graphic_generator_incomplete_palette", ctx.exception.reasons)

    def test_5b_snake_case_keys_accepted(self) -> None:
        graphic = parse_graphic_system_output(_graphic(snake_case=True))
        self.assertEqual(graphic.layout_template, "visual_right_copy_left")

    def test_5c_series_generator_string_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_series_ads_output(_series_ads(2, series_string=True), expected_ad_count=2)
        self.assertIn("series_generator_not_object", ctx.exception.reasons)

    def test_5c_incomplete_series_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_series_ads_output(_series_ads(2, incomplete_series=True), expected_ad_count=2)
        self.assertIn("series_generator_incomplete", ctx.exception.reasons)

    def test_5c_omitted_indexes_injected(self) -> None:
        result = parse_series_ads_output(_series_ads(2, omit_indexes=True), expected_ad_count=2)
        self.assertEqual([a["index"] for a in result.ads], [1, 2])

    def test_5c_string_indexes_coerced(self) -> None:
        result = parse_series_ads_output(_series_ads(2, string_indexes=True), expected_ad_count=2)
        self.assertEqual([a["index"] for a in result.ads], [1, 2])

    def test_5c_wrong_indexes_normalized_to_position(self) -> None:
        result = parse_series_ads_output(_series_ads(2, wrong_indexes=True), expected_ad_count=2)
        self.assertEqual([a["index"] for a in result.ads], [1, 2])

    def test_5c_too_few_ads_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_series_ads_output(_series_ads(4, too_few_ads=True), expected_ad_count=4)
        self.assertIn("ads_length_mismatch", ctx.exception.reasons)


class TestDeterministicAssembly(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def _assemble(self, ad_count: int) -> Any:
        strategy = _selected_strategy()
        conceptual = _selected_conceptual()
        brand = parse_brand_physical_output(_brand_physical())
        graphic = parse_graphic_system_output(_graphic())
        series = parse_series_ads_output(_series_ads(ad_count), expected_ad_count=ad_count)
        return assemble_builder1_campaign(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            ad_count=ad_count,
            detected_language=detect_brief_language(self.BRIEF),
            exploration_seed="seed-test",
            product_name_resolved="TestBrand",
            strategy=strategy,
            strategy_selection=parse_strategy_selection(
                {
                    "selectedCandidateId": strategy.id,
                    "selectionReason": "Best",
                    "strategyFamily": "durability",
                    "scores": {"truth": 8, "briefSupport": 8, "relevance": 8, "distinctiveness": 7,
                               "brandOwnership": 8, "persuasiveStrength": 8, "seriesPotential": 8,
                               "conceptualActionPotential": 8},
                },
                parse_strategy_scan(_strategy_scan_payload(), product_description=self.BRIEF),
            )[0],
            conceptual=conceptual,
            brand_physical=brand,
            graphic=graphic,
            series_ads=series,
        )

    def test_explicit_ad_count_four(self) -> None:
        plan = self._assemble(4)
        self.assertEqual(plan.ad_count, 4)
        self.assertEqual(len(plan.ads), 4)

    def test_no_internal_scans_in_public_plan(self) -> None:
        public = campaign_identity_to_dict(self._assemble(2))
        self.assertNotIn("strategyCandidateScan", public)
        self.assertNotIn("conceptualGeneratorScan", public)


class TestProductionShapedPlanner(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def _run_with_sequence(
        self,
        *,
        ad_count: int = 2,
        stage_sequences: Optional[Dict[str, List[Any]]] = None,
    ) -> Any:
        sequences = stage_sequences or {}
        counters: Dict[str, int] = {}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            key = system
            if key in sequences:
                idx = counters.get(key, 0)
                counters[key] = idx + 1
                seq = sequences[key]
                return copy.deepcopy(seq[min(idx, len(seq) - 1)])
            return copy.deepcopy(_full_final_responses(ad_count).get(key, {"pass": True, "rejectionReasonCodes": []}))

        return plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=ad_count,
        ), counters

    def test_5b_failure_does_not_rerun_early_stages(self) -> None:
        brand_calls: List[int] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_BRAND_PHYSICAL_SYSTEM:
                brand_calls.append(1)
            if system == STAGE_GRAPHIC_SYSTEM_SYSTEM:
                if "Repair ONLY" in user:
                    return _graphic()
                if len(brand_calls) == 0:
                    pass
                return _graphic(missing_palette=True)
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(len(brand_calls), 1)

    def test_5c_failure_does_not_rerun_5a_or_5b(self) -> None:
        counters = {"brand": 0, "graphic": 0, "series": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_BRAND_PHYSICAL_SYSTEM:
                counters["brand"] += 1
                return _brand_physical()
            if system == STAGE_GRAPHIC_SYSTEM_SYSTEM:
                counters["graphic"] += 1
                return _graphic()
            if system == STAGE_SERIES_ADS_SYSTEM:
                counters["series"] += 1
                if "Repair ONLY" in user:
                    return _series_ads(2)
                return _series_ads(2, series_string=True)
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True, "rejectionReasonCodes": []}))

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(counters["brand"], 1)
        self.assertEqual(counters["graphic"], 1)
        self.assertGreaterEqual(counters["series"], 2)

    def test_focused_repair_success_reaches_assembly(self) -> None:
        plan, _ = self._run_with_sequence(
            stage_sequences={
                STAGE_BRAND_PHYSICAL_SYSTEM: [
                    _brand_physical(missing_role=True),
                    _brand_physical(),
                ]
            }
        )
        self.assertEqual(plan.ad_count, 2)

    def test_repeated_malformed_returns_substage_failure(self) -> None:
        with self.assertRaises(Builder1PlannerError) as ctx:
            self._run_with_sequence(
                stage_sequences={
                    STAGE_GRAPHIC_SYSTEM_SYSTEM: [
                        _graphic(missing_palette=True),
                        _graphic(missing_palette=True),
                        _graphic(missing_palette=True),
                    ]
                }
            )
        self.assertIn("graphic_system_failed", str(ctx.exception))

    def test_ad_count_four_end_to_end(self) -> None:
        plan, _ = self._run_with_sequence(ad_count=4)
        self.assertEqual(plan.ad_count, 4)

    def test_successful_assembly_reaches_image_generation(self) -> None:
        plan, _ = self._run_with_sequence(ad_count=2)
        calls: List[int] = []

        def image_caller(_p: str, _f: str) -> bytes:
            calls.append(1)
            return b"jpeg"

        generate_builder1_ad_image(plan, 1, image_caller)
        self.assertEqual(len(calls), 1)

    def test_next_ad_does_not_run_planning(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            from engine.builder1_campaign_store import (
                clear_memory_store_for_tests,
                create_campaign_session,
                mark_ad_generated,
                try_acquire_generation_lock,
                validate_next_ad_request,
            )
            from tests.test_builder1_series import _parse, _base_campaign

            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="staged-next", plan=plan)
            try_acquire_generation_lock("staged-next", 1)
            mark_ad_generated("staged-next", 1)
            validate_next_ad_request("staged-next", 2)


class TestStagedParsers(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_strategy_scan_accepts_twelve_objects(self) -> None:
        result = parse_strategy_scan(_strategy_scan_payload(), product_description=self.BRIEF)
        self.assertEqual(len(result), 12)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_no_builder2_final_stage_modules(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_final_stages", text)


class TestPlanningModelStrictSchema(unittest.TestCase):
    def test_strict_schema_enabled_when_text_param_supported(self) -> None:
        import engine.builder1_planning_model as pm

        pm._strict_schema_probe_done = False
        pm._strict_schema_available = False
        with patch.object(pm, "_responses_create_supports_text_parameter", return_value=True):
            self.assertTrue(pm.strict_json_schema_available())

    def test_strict_schema_disabled_when_text_param_missing(self) -> None:
        import engine.builder1_planning_model as pm

        pm._strict_schema_probe_done = False
        pm._strict_schema_available = False
        with patch.object(pm, "_responses_create_supports_text_parameter", return_value=False):
            self.assertFalse(pm.strict_json_schema_available())


if __name__ == "__main__":
    unittest.main()
