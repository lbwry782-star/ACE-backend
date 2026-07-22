"""
Builder1 concept-first / anti-literal embodiment tests.

Run: python -m unittest tests.test_builder1_literal_embodiment -v
"""
from __future__ import annotations

import copy
import unittest

from engine.builder1_creative_methodology import (
    deterministic_methodology_checks,
    earliest_methodology_repair_stage,
)
from engine.builder1_image_prompt_preflight import classify_image_prompt_plan
from engine.builder1_literal_embodiment import (
    BUILDER1_CONCEPT_FIRST_RULE,
    BUILDER1_SLOGAN_COMPLEMENTARITY_RULE,
    BUILDER1_SLOGAN_LITERALNESS_SCAN,
    contains_literal_route_family,
    scan_literal_embodiment_bias,
    scan_series_plan_literal_embodiment,
    validate_visual_prompt_expressive_object,
    validate_visual_prompt_slogan_noun_reintroduction,
)
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
)
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_series import _base_campaign, _parse


def _campaign_from_overrides(**overrides: object) -> dict:
    data = _base_campaign(2)
    ads = overrides.pop("ads", None)
    data.update({key: value for key, value in overrides.items() if key != "ads"})
    if isinstance(ads, list):
        data["ads"] = ads
    return data


def _literal_trap_plan_dict() -> dict:
    return _campaign_from_overrides(
        brandSlogan="We shorten your way",
        sloganAction="Make distances shorter",
        productNameResolved="RoutePro",
        physicalGenerator="City road network",
        transferredObject="Highway maze",
        transferredObjectAction="Routes collapse into a shorter path",
        conceptualGenerator="Distances shrink",
        conceptualGeneratorAction="Shorten familiar long things",
        ads=[
            {
                "index": 1,
                "variationLabel": "v1",
                "newContribution": "Road trap one",
                "physicalExecution": "Aerial view of a car inside a maze",
                "visualExecution": "Driver follows a winding road",
                "sceneDescription": "Urban highway intersection with navigation map",
                "conceptualExecution": "Roads become shorter",
                "conceptualActionProof": "Route compression proof",
                "headline": None,
                "headlineNeededReason": "Self-explanatory",
                "marketingText": "word " * 50,
            },
            {
                "index": 2,
                "variationLabel": "v2",
                "newContribution": "Road trap two",
                "physicalExecution": "GPS map over city streets",
                "visualExecution": "Route lines shrink on a road map",
                "sceneDescription": "Traffic on a highway",
                "conceptualExecution": "Navigation path shortens",
                "conceptualActionProof": "Second route proof",
                "headline": None,
                "headlineNeededReason": "Self-explanatory",
                "marketingText": "word " * 50,
            },
        ],
    )


def _external_proxy_plan_dict() -> dict:
    return _campaign_from_overrides(
        brandSlogan="We shorten your way",
        sloganAction="Make distances shorter",
        productNameResolved="RoutePro",
        physicalGenerator="Short-neck giraffe",
        transferredObject="Short-neck giraffe",
        transferredObjectAction="Its famously long neck becomes visibly shorter",
        conceptualGenerator="Familiar long things become short",
        conceptualGeneratorAction="Shorten recognizable long objects",
        ads=[
            {
                "index": 1,
                "variationLabel": "v1",
                "newContribution": "Giraffe proof",
                "physicalExecution": "A giraffe with a shortened neck beside a normal giraffe silhouette",
                "visualExecution": "Studio shot of the shortened-neck giraffe",
                "sceneDescription": "Clean background, no roads or cars",
                "conceptualExecution": "Long neck becomes short",
                "conceptualActionProof": "Neck shortening proof",
                "headline": None,
                "headlineNeededReason": "Self-explanatory",
                "marketingText": "word " * 50,
            },
            {
                "index": 2,
                "variationLabel": "v2",
                "newContribution": "Rope proof",
                "physicalExecution": "A shortened rope next to a normal rope",
                "visualExecution": "Two ropes compared for length",
                "sceneDescription": "Plain tabletop, no vehicles",
                "conceptualExecution": "Another long object becomes short",
                "conceptualActionProof": "Rope shortening proof",
                "headline": None,
                "headlineNeededReason": "Self-explanatory",
                "marketingText": "word " * 50,
            },
        ],
    )


class TestLiteralEmbodimentPrompts(unittest.TestCase):
    def test_conceptual_stage_contains_concept_first_rule(self) -> None:
        self.assertIn("CONCEPT FIRST", STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn(BUILDER1_CONCEPT_FIRST_RULE.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn("literal slogan nouns", STAGE_CONCEPTUAL_STAGE_SYSTEM.lower())

    def test_conceptual_stage_contains_slogan_complementarity(self) -> None:
        self.assertIn(BUILDER1_SLOGAN_COMPLEMENTARITY_RULE.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn(BUILDER1_SLOGAN_LITERALNESS_SCAN.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn("complement each other", STAGE_CONCEPTUAL_STAGE_SYSTEM.lower())

    def test_brand_physical_prefers_external_expressive_object(self) -> None:
        self.assertIn("strongest expressive object", STAGE_BRAND_PHYSICAL_SYSTEM.lower())
        self.assertIn("literal slogan nouns", STAGE_BRAND_PHYSICAL_SYSTEM.lower())
        self.assertIn("SLOGAN / VISUAL COMPLEMENTARITY", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_series_stage_rejects_literal_family_trap(self) -> None:
        self.assertIn("road/maze/car", STAGE_SERIES_ADS_SYSTEM.lower())
        self.assertIn("external expressive embodiment", STAGE_SERIES_ADS_SYSTEM.lower())
        self.assertIn("MANDATORY SLOGAN-LITERALNESS SCAN", STAGE_SERIES_ADS_SYSTEM)


class TestLiteralEmbodimentDetection(unittest.TestCase):
    def test_literal_route_family_detector(self) -> None:
        self.assertTrue(contains_literal_route_family("A car drives through a maze on a highway"))
        self.assertFalse(contains_literal_route_family("A short-neck giraffe in a studio"))

    def test_literal_slogan_road_trap_is_rejected(self) -> None:
        reasons = scan_literal_embodiment_bias(_literal_trap_plan_dict())
        self.assertIn("literal_slogan_illustration", reasons)
        self.assertIn("series_literal_category_trap", reasons)

    def test_shorter_way_does_not_force_road_route_maze_car_train(self) -> None:
        reasons = scan_literal_embodiment_bias(_external_proxy_plan_dict())
        blob = " ".join(
            [
                _external_proxy_plan_dict()["ads"][0]["physicalExecution"],
                _external_proxy_plan_dict()["ads"][1]["sceneDescription"],
            ]
        ).lower()
        for forbidden in ("road", "route", "maze", "car", "train"):
            self.assertNotIn(forbidden, blob)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_external_proxy_passes_literal_guard(self) -> None:
        reasons = scan_literal_embodiment_bias(_external_proxy_plan_dict())
        self.assertNotIn("slogan_word_illustration", reasons)
        self.assertNotIn("series_literal_category_trap", reasons)
        self.assertNotIn("literal_slogan_object_depiction", reasons)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_product_visibility_not_required(self) -> None:
        plan = _external_proxy_plan_dict()
        plan["productVisibilityPolicy"] = "FORBIDDEN"
        for ad in plan["ads"]:
            ad["productVisible"] = False
            ad["productIsMainVisual"] = False
        reasons = deterministic_methodology_checks(plan)
        self.assertNotIn("unauthorized_product_visibility", reasons)

    def test_literal_slogan_object_not_required(self) -> None:
        reasons = scan_literal_embodiment_bias(_external_proxy_plan_dict())
        self.assertEqual(reasons, [])

    def test_shortening_concept_can_use_non_road_objects(self) -> None:
        reasons = scan_literal_embodiment_bias(_external_proxy_plan_dict())
        self.assertNotIn("literal_slogan_object_depiction", reasons)

    def test_methodology_repair_routes_literal_physical_issues(self) -> None:
        stage = earliest_methodology_repair_stage(["literal_slogan_illustration"])
        self.assertEqual(stage, "brand_physical")

    def test_caption_only_literal_illustration_is_rejected(self) -> None:
        plan = _campaign_from_overrides(
            brandSlogan="Opens every door",
            sloganAction="Remove access barriers",
            transferredObject="Door",
            physicalGenerator="Door",
            whyClearerThanShowingProduct="Shows a door because the slogan mentions opening doors",
            ads=[
                {
                    "index": 1,
                    "variationLabel": "v1",
                    "newContribution": "Literal door",
                    "physicalExecution": "A door opening",
                    "visualExecution": "Literal door illustration",
                    "sceneDescription": "Door centered in frame",
                    "conceptualExecution": "Door opens",
                    "conceptualActionProof": "Shows the slogan noun",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": "word " * 50,
                    "sloganConnection": "Shows a door because the slogan mentions opening doors",
                }
            ],
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertIn("literal_slogan_illustration", reasons)

    def test_strong_literal_transformation_may_pass(self) -> None:
        plan = _campaign_from_overrides(
            brandSlogan="Break through every barrier",
            sloganAction="Create visible breakthrough",
            transferredObject="Glass wall shattering into light",
            physicalGenerator="Glass wall shattering into light",
            whyClearerThanShowingProduct=(
                "The unexpected shattering transformation proves breakthrough more vividly than repeating the word barrier"
            ),
            ads=[
                {
                    "index": 1,
                    "variationLabel": "v1",
                    "newContribution": "Breakthrough proof",
                    "physicalExecution": "Glass wall shatters into open passage",
                    "visualExecution": "Barrier state transforms into breakthrough",
                    "sceneDescription": "Clean studio with shattering glass wall",
                    "conceptualExecution": "Barrier becomes passage",
                    "conceptualActionProof": "Physical breakthrough transformation proof",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": "word " * 50,
                    "sloganConnection": (
                        "The fixed slogan promise is proven by an unexpected physical breakthrough transformation, "
                        "not by repeating the word barrier"
                    ),
                    "singleChangedPropertyOrAction": "Solid barrier state changes to open passage",
                    "executionPunchline": "Breakthrough made visible through transformation",
                }
            ],
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_methodology_repair_routes_series_literal_trap(self) -> None:
        stage = earliest_methodology_repair_stage(["series_literal_category_trap"])
        self.assertEqual(stage, "series_ads")


class TestVisualPromptEmbodiment(unittest.TestCase):
    def _plan_with_transferred(self, transferred: str):
        raw = _base_campaign(2)
        raw["transferredObject"] = transferred
        raw["physicalGenerator"] = transferred
        raw["transferredObjectAction"] = "Its length becomes visibly shorter"
        raw["conceptualGenerator"] = "Familiar long things become short"
        raw["sloganAction"] = "Make distances shorter"
        raw["brandSlogan"] = "We shorten your way"
        for ad in raw["ads"]:
            ad["physicalExecution"] = f"Studio depiction of {transferred} variant {ad['index']}"
            ad["visualExecution"] = f"The {transferred} performs action variant {ad['index']}"
            ad["sceneDescription"] = f"Clean studio background without roads or cars for ad {ad['index']}"
        return _parse(raw, 2)

    def test_image_prompt_preserves_external_embodiment(self) -> None:
        plan = self._plan_with_transferred("Short-neck giraffe")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("Short-neck giraffe", prompt)
        self.assertIn("external expressive object", prompt.lower())
        main_visual = prompt.split("=== END MAIN VISUAL ===")[0]
        self.assertNotIn("road", main_visual.lower())
        self.assertNotIn("maze", main_visual.lower())

    def test_image_prompt_preflight_rejects_literal_collapse(self) -> None:
        plan = self._plan_with_transferred("Short-neck giraffe")
        bad_prompt = build_visual_prompt(plan, plan.ads[0]).replace(
            "Short-neck giraffe",
            "Car driving through a highway maze",
            1,
        )
        reasons = validate_visual_prompt_expressive_object(bad_prompt, series_plan=plan)
        self.assertIn("expressive_object_weakened", reasons)
        self.assertIn("literal_slogan_illustration", reasons)

    def test_image_prompt_does_not_reintroduce_discarded_slogan_nouns(self) -> None:
        plan = self._plan_with_transferred("Short-neck giraffe")
        prompt = build_visual_prompt(plan, plan.ads[0])
        injected = prompt.replace(
            "ACTION: Its length becomes visibly shorter",
            "ACTION: A road, train, and city route appear behind the giraffe while its length becomes visibly shorter",
            1,
        )
        reasons = validate_visual_prompt_slogan_noun_reintroduction(injected, series_plan=plan)
        self.assertIn("literal_slogan_illustration", reasons)

    def test_preflight_accepts_external_embodiment_prompt(self) -> None:
        plan = self._plan_with_transferred("Short ruler")
        prompt = build_visual_prompt(plan, plan.ads[0])
        result = classify_image_prompt_plan(plan, plan.ads[0], prompt=prompt)
        self.assertTrue(result.ok)

    def test_clarity_requirement_remains_in_prompt(self) -> None:
        plan = self._plan_with_transferred("Short rope")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("MAIN VISUAL:", prompt)
        self.assertIn("Short rope", prompt)


class TestSeriesLiteralEmbodiment(unittest.TestCase):
    def test_series_scan_detects_literal_family_repetition(self) -> None:
        plan = _parse(_literal_trap_plan_dict(), 2)
        reasons = scan_series_plan_literal_embodiment(plan)
        self.assertIn("series_literal_category_trap", reasons)
        self.assertIn("literal_slogan_illustration", reasons)

    def test_series_shares_conceptual_law_without_literal_object_family(self) -> None:
        plan = _parse(_external_proxy_plan_dict(), 2)
        reasons = scan_series_plan_literal_embodiment(plan)
        self.assertNotIn("series_literal_category_trap", reasons)
        self.assertNotIn("literal_slogan_illustration", reasons)
        self.assertEqual(plan.conceptual_generator, "Familiar long things become short")
        subjects = {ad.physical_execution for ad in plan.ads}
        self.assertEqual(len(subjects), 2)


class TestPlanningBudgetRegression(unittest.TestCase):
    def test_normal_planning_calls_remain_five(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)


if __name__ == "__main__":
    unittest.main()
