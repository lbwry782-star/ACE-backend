"""
Builder1 staged planning pipeline tests.

Run: python -m unittest tests.test_builder1_staged_planning -v
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_plan_spec import campaign_identity_to_dict, series_plan_to_store_dict
from engine.builder1_planning_contract import (
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_CONCEPTUAL_SELECT_SYSTEM,
    STAGE_FINAL_CAMPAIGN_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_staged_parsers import (
    StageParseError,
    assemble_builder1_series_plan,
    detect_brief_language,
    parse_conceptual_scan,
    parse_conceptual_selection,
    parse_final_campaign_output,
    parse_strategy_scan,
    parse_strategy_selection,
)
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT


def _graphic() -> Dict[str, Any]:
    return {
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


LENSES = [
    "economic",
    "perceptual",
    "emotional",
    "operational",
    "time",
    "accessibility",
    "expertise",
    "challenger_positioning",
    "participation",
    "simplicity",
    "specialization",
    "category_convention",
]


def _strategy_scan_payload(*, string_candidate: bool = False) -> Dict[str, Any]:
    candidates: List[Any] = []
    for i in range(1, 13):
        item: Dict[str, Any] = {
            "id": f"S{i:02d}",
            "lens": LENSES[i - 1],
            "strategicProblem": f"Distinct buyer problem {i}",
            "relativeAdvantage": f"Distinct advantage {i}",
            "briefSupport": "Follows from brief reinforced shell mention",
            "advantageSource": "observable_product_mechanism",
            "claimRisk": "low",
        }
        candidates.append(item)
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


def _final_creative(ad_count: int = 2, *, bad_graphic: bool = False) -> Dict[str, Any]:
    ads = []
    for i in range(1, ad_count + 1):
        ads.append(
            {
                "index": i,
                "variationLabel": f"var-{i}",
                "newContribution": f"Contribution {i}",
                "conceptualExecution": f"Perform drop proof variant {i}",
                "conceptualActionProof": f"Proof {i}",
                "physicalExecution": f"Object variant {i}",
                "visualExecution": f"Visual variant {i}",
                "sceneDescription": f"Scene {i}",
                "headline": None if i == 1 else f"Line {i}",
                "headlineNeededReason": "Needed" if i > 1 else "Self-explanatory",
                "marketingText": f"Short {i}",
            }
        )
    graphic = _graphic()
    if bad_graphic:
        del graphic["palette"]
    return {
        "productNameResolved": "TestBrand",
        "brandSlogan": "Built To Last",
        "sloganDerivation": "From durability advantage",
        "sloganAction": "Trust everyday use",
        "physicalGenerator": "Rubber ball family",
        "physicalGeneratorNaturalPurpose": "Absorb impact",
        "physicalGeneratorCampaignRole": "Durability metaphor",
        "graphicGenerator": graphic,
        "seriesGenerator": {
            "type": "situations",
            "principle": "Different drop contexts",
            "progression": "Escalating severity",
        },
        "mediumParticipates": False,
        "mediumRole": "",
        "campaignRationale": "Ownable durability story",
        "ads": ads,
    }


def _selected_strategy():
    scan = parse_strategy_scan(
        _strategy_scan_payload(),
        product_description="Reinforced shell product for daily carry",
    )
    return scan[0]


def _selected_conceptual():
    return parse_conceptual_scan(_conceptual_scan_payload())[0]


class TestStagedParsers(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_strategy_scan_accepts_twelve_objects(self) -> None:
        result = parse_strategy_scan(_strategy_scan_payload(), product_description=self.BRIEF)
        self.assertEqual(len(result), 12)
        self.assertEqual(result[0].id, "S01")

    def test_strategy_scan_string_candidate_triggers_parse_error(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_scan(
                _strategy_scan_payload(string_candidate=True),
                product_description=self.BRIEF,
            )
        self.assertIn("strategy_scan_string_candidate", ctx.exception.reasons)

    def test_strategy_selector_uses_lookup_not_rewrite(self) -> None:
        candidates = parse_strategy_scan(_strategy_scan_payload(), product_description=self.BRIEF)
        selection, selected = parse_strategy_selection(
            {
                "selectedCandidateId": "S03",
                "selectionReason": "Best fit",
                "strategyFamily": "durability",
                "scores": {"truth": 8, "briefSupport": 8, "relevance": 8, "distinctiveness": 7,
                           "brandOwnership": 8, "persuasiveStrength": 8, "seriesPotential": 8,
                           "conceptualActionPotential": 8},
            },
            candidates,
        )
        self.assertEqual(selection.selected_candidate_id, "S03")
        self.assertEqual(selected.relative_advantage, "Distinct advantage 3")

    def test_conceptual_scan_accepts_six_objects(self) -> None:
        result = parse_conceptual_scan(_conceptual_scan_payload())
        self.assertEqual(len(result), 6)

    def test_incomplete_conceptual_triggers_stage_error(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_conceptual_scan(_conceptual_scan_payload(incomplete=True))
        self.assertIn("conceptual_scan_candidate_incomplete", ctx.exception.reasons)

    def test_final_output_rejects_internal_scans(self) -> None:
        payload = _final_creative(2)
        payload["strategyCandidateScan"] = {"candidates": []}
        _, reasons = parse_final_campaign_output(payload, expected_ad_count=2)
        self.assertTrue(any("final_campaign_forbidden_field" in r for r in reasons))

    def test_final_output_rejects_request_controlled_fields(self) -> None:
        payload = _final_creative(2)
        payload["format"] = "landscape"
        payload["adCount"] = "3"
        _, reasons = parse_final_campaign_output(payload, expected_ad_count=2)
        self.assertTrue(any("format" in r for r in reasons))
        self.assertTrue(any("adCount" in r for r in reasons))

    def test_malformed_graphic_rejected_in_final_stage(self) -> None:
        _, reasons = parse_final_campaign_output(
            _final_creative(2, bad_graphic=True),
            expected_ad_count=2,
        )
        self.assertIn("graphic_generator_missing_palette", reasons)

    def test_unsupported_statistics_rejected_in_strategy_scan(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0]["briefSupport"] = "According to a 2024 survey, 87% agree"
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_scan(payload, product_description=self.BRIEF)
        self.assertIn("unsupported_evidence_claim", ctx.exception.reasons)

    def test_fabricated_study_rejected(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][1]["strategicProblem"] = "Market report shows widespread breakage"
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_scan(payload, product_description=self.BRIEF)
        self.assertIn("unsupported_evidence_claim", ctx.exception.reasons)


class TestServerAuthoritativeAssembly(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def _assemble(self, ad_count: int) -> Any:
        strategy = _selected_strategy()
        conceptual = _selected_conceptual()
        final = _final_creative(ad_count)
        return assemble_builder1_series_plan(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            ad_count=ad_count,
            detected_language=detect_brief_language(self.BRIEF),
            exploration_seed="seed-test",
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
            final_creative=final,
        )

    def test_injects_format(self) -> None:
        plan = self._assemble(2)
        self.assertEqual(plan.format, "portrait")

    def test_injects_ad_count_two(self) -> None:
        plan = self._assemble(2)
        self.assertEqual(plan.ad_count, 2)

    def test_explicit_ad_count_three(self) -> None:
        plan = self._assemble(3)
        self.assertEqual(plan.ad_count, 3)
        self.assertEqual(len(plan.ads), 3)

    def test_explicit_ad_count_four(self) -> None:
        plan = self._assemble(4)
        self.assertEqual(plan.ad_count, 4)
        self.assertEqual(len(plan.ads), 4)

    def test_injects_detected_language(self) -> None:
        plan = self._assemble(2)
        self.assertEqual(plan.detected_language, "en")

    def test_model_cannot_override_ad_count_via_final(self) -> None:
        final = _final_creative(2)
        final["adCount"] = 4
        _, reasons = parse_final_campaign_output(final, expected_ad_count=2)
        self.assertTrue(any("adCount" in r for r in reasons))
        plan = self._assemble(2)
        self.assertEqual(plan.ad_count, 2)

    def test_final_parser_does_not_require_scans(self) -> None:
        from engine.builder1_plan_parser import parse_builder1_series_plan

        plan = self._assemble(2)
        reparsed = parse_builder1_series_plan(
            series_plan_to_store_dict(plan),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="",
            product_description=self.BRIEF,
            require_internal_scans=False,
        )
        self.assertEqual(reparsed.ad_count, 2)

    def test_assembled_plan_has_no_internal_scans(self) -> None:
        plan = self._assemble(2)
        store = series_plan_to_store_dict(plan)
        self.assertNotIn("strategyCandidateScan", store)
        self.assertNotIn("conceptualGeneratorScan", store)
        public = campaign_identity_to_dict(plan)
        self.assertNotIn("strategyCandidateScan", public)
        self.assertNotIn("conceptualGeneratorScan", public)
        plan = self._assemble(2)
        store = series_plan_to_store_dict(plan)
        self.assertNotIn("strategyCandidateScan", store)
        self.assertNotIn("conceptualGeneratorScan", store)
        public = campaign_identity_to_dict(plan)
        self.assertNotIn("strategyCandidateScan", public)


class TestStagedPlannerIntegration(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def _mock_responses(self, ad_count: int = 2) -> Dict[str, Any]:
        return {
            STAGE_STRATEGY_SCAN_SYSTEM: _strategy_scan_payload(),
            STAGE_STRATEGY_SELECT_SYSTEM: {
                "selectedCandidateId": "S01",
                "selectionReason": "Strongest brief fit",
                "strategyFamily": "durability",
                "scores": {
                    "truth": 9,
                    "briefSupport": 9,
                    "relevance": 8,
                    "distinctiveness": 8,
                    "brandOwnership": 8,
                    "persuasiveStrength": 8,
                    "seriesPotential": 9,
                    "conceptualActionPotential": 9,
                },
            },
            STAGE_CONCEPTUAL_SCAN_SYSTEM: _conceptual_scan_payload(),
            STAGE_CONCEPTUAL_SELECT_SYSTEM: {
                "selectedCandidateId": "C01",
                "selectionReason": "Clearest action",
                "scores": {
                    "advantageConnection": 9,
                    "actionClarity": 9,
                    "visualPower": 8,
                    "seriesPotential": 9,
                    "distinctiveness": 8,
                    "physicalIndependence": 8,
                },
            },
            STAGE_FINAL_CAMPAIGN_SYSTEM: _final_creative(ad_count),
            BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT: {
                "pass": True,
                "rejectionReasonCodes": [],
                "unsupportedClaimDetected": False,
            },
        }

    def test_successful_planning_reaches_image_generation(self) -> None:
        responses = self._mock_responses(2)
        calls: List[str] = []

        def model_caller(system: str, user: str) -> object:
            calls.append(system[:40])
            if "Repair ONLY the candidates array" in user:
                return _strategy_scan_payload()
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        img_calls: List[int] = []

        def image_caller(_prompt: str, _fmt: str) -> bytes:
            img_calls.append(1)
            return b"jpeg"

        generate_builder1_ad_image(plan, 1, image_caller)
        self.assertEqual(len(img_calls), 1)

    def test_ad_count_three_survives_pipeline(self) -> None:
        responses = self._mock_responses(3)

        def model_caller(system: str, _user: str) -> object:
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=3,
        )
        self.assertEqual(plan.ad_count, 3)

    def test_ad_count_four_survives_pipeline(self) -> None:
        responses = self._mock_responses(4)

        def model_caller(system: str, _user: str) -> object:
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=4,
        )
        self.assertEqual(plan.ad_count, 4)

    def test_string_candidate_triggers_repair_not_total_failure(self) -> None:
        responses = self._mock_responses(2)
        scan_calls = {"n": 0}

        def model_caller(system: str, user: str) -> object:
            if system == STAGE_STRATEGY_SCAN_SYSTEM:
                scan_calls["n"] += 1
                if "Repair ONLY" in user:
                    return _strategy_scan_payload()
                if scan_calls["n"] == 1:
                    return _strategy_scan_payload(string_candidate=True)
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertGreaterEqual(scan_calls["n"], 2)

    def test_judge_receives_normalized_plan(self) -> None:
        seen: Dict[str, Any] = {}

        def model_caller(system: str, user: str) -> object:
            if system == BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT:
                seen["judge_user"] = user
                return {"pass": True, "rejectionReasonCodes": []}
            if system == STAGE_STRATEGY_SCAN_SYSTEM and "Repair ONLY" in user:
                return _strategy_scan_payload()
            return self._mock_responses(2).get(system, {"pass": True})

        plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertIn("judge_user", seen)
        self.assertIn('"format": "portrait"', seen["judge_user"])
        self.assertNotIn("strategyCandidateScan", seen["judge_user"])

    def test_next_ad_does_not_rerun_planner(self) -> None:
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


class TestBuilder2Unchanged(unittest.TestCase):
    def test_no_builder2_staged_modules(self) -> None:
        import importlib.util
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        builder2_files = list(root.glob("builder2*.py"))
        for path in builder2_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_staged_parsers", text)
            self.assertNotIn("STAGE_STRATEGY_SCAN_SYSTEM", text)


if __name__ == "__main__":
    unittest.main()
