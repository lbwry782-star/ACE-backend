"""
Builder1 authoritative fixed campaign slogan tests.

Run: python -m unittest tests.test_builder1_authoritative_slogan -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder1_campaign_integrity import (
    make_upstream_snapshot,
    validate_builder1_campaign_integrity,
)
from engine.builder1_final_stages import (
    assemble_builder1_campaign,
    build_series_ad_internals,
    inject_fixed_campaign_slogan_into_series_ads,
    parse_brand_physical_output,
    parse_graphic_system_output,
    parse_series_ads_output,
    strip_model_slogan_fields_from_series_ads,
)
from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_plan_spec import HEADLINE_MAX_WORDS
from engine.builder1_slogan_stage import parse_slogan_scan, parse_slogan_selection
from engine.builder1_staged_parsers import detect_brief_language
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _graphic,
    _internal_ad_fields,
    _selected_conceptual,
    _selected_strategy,
    _series_ads,
    _slogan_scan_payload,
    _strategy_scan_payload,
    _strategy_selection_payload,
)
from engine.builder1_staged_parsers import parse_strategy_scan, parse_strategy_selection


FIXED_SLOGAN = "Built To Last"
BRIEF = "Reinforced shell product for daily carry"


def _selected_slogan():
    candidates = parse_slogan_scan(_slogan_scan_payload())
    _, selected = parse_slogan_selection(
        {
            "selectedCandidateId": "L01",
            "selectionReason": "Best",
            "scores": {
                "directAdvantageExpression": 9,
                "naturalness": 8,
                "memorability": 8,
                "credibility": 9,
                "brandOwnership": 8,
                "competitorTransferResistance": 8,
                "actionClarity": 9,
                "campaignGenerativePower": 9,
            },
        },
        candidates,
    )
    return selected


def _assemble(
    ad_count: int,
    *,
    series_payload: Dict[str, Any] | None = None,
):
    strategy = _selected_strategy()
    series = parse_series_ads_output(
        series_payload or _series_ads(ad_count),
        expected_ad_count=ad_count,
    )
    return assemble_builder1_campaign(
        product_name="",
        product_description=BRIEF,
        format_value="portrait",
        ad_count=ad_count,
        detected_language=detect_brief_language(BRIEF),
        exploration_seed="seed-test",
        product_name_resolved="TestBrand",
        strategy=strategy,
        strategy_selection=parse_strategy_selection(
            _strategy_selection_payload(selected_id=strategy.id),
            parse_strategy_scan(_strategy_scan_payload(), product_description=BRIEF),
        )[0],
        selected_slogan=_selected_slogan(),
        conceptual=_selected_conceptual(),
        brand_physical=parse_brand_physical_output(_brand_physical()),
        graphic=parse_graphic_system_output(_graphic()),
        series_ads=series,
    )


def _series_ads_with_modified_slogan_connection(*, ad_count: int = 2) -> Dict[str, Any]:
    payload = _series_ads(ad_count)
    payload["ads"][0]["sloganConnection"] = "Shows Built To Last through visible survival"
    payload["ads"][1]["sloganConnection"] = "Expresses durability through repeated impact proof"
    return payload


def _series_ads_with_model_slogan_fields(*, ad_count: int = 2) -> Dict[str, Any]:
    payload = _series_ads(ad_count)
    payload["ads"][0]["brandSlogan"] = FIXED_SLOGAN
    payload["ads"][1]["brandSlogan"] = "Different Slogan Text"
    payload["ads"][1]["slogan"] = "Also Wrong"
    return payload


class TestAuthoritativeSloganInjection(unittest.TestCase):
    def test_assembly_injects_fixed_slogan_into_ad_internals_for_all_ads(self) -> None:
        for ad_count in (2, 3, 4):
            with self.subTest(ad_count=ad_count):
                plan = _assemble(ad_count)
                for ad in plan.ads:
                    internals = plan.planning_internals["adInternals"][ad.index]
                    self.assertEqual(internals["brandSlogan"], FIXED_SLOGAN)

    def test_model_returned_per_ad_slogan_is_stripped_from_assembled_ads(self) -> None:
        series = parse_series_ads_output(
            _series_ads_with_model_slogan_fields(ad_count=2),
            expected_ad_count=2,
        )
        cleaned = inject_fixed_campaign_slogan_into_series_ads(series.ads, fixed_slogan=FIXED_SLOGAN)
        for ad in cleaned:
            self.assertNotIn("brandSlogan", ad)
            self.assertNotIn("slogan", ad)

    def test_model_modified_slogan_value_is_discarded(self) -> None:
        plan = _assemble(2, series_payload=_series_ads_with_model_slogan_fields(ad_count=2))
        self.assertEqual(plan.brand_slogan, FIXED_SLOGAN)
        for ad in plan.ads:
            internals = plan.planning_internals["adInternals"][ad.index]
            self.assertEqual(internals["brandSlogan"], FIXED_SLOGAN)
            self.assertNotEqual(internals["brandSlogan"], "Different Slogan Text")

    def test_parse_strips_deprecated_model_slogan_fields(self) -> None:
        raw = _series_ads_with_model_slogan_fields(ad_count=2)
        parsed = parse_series_ads_output(raw, expected_ad_count=2)
        for ad in parsed.ads:
            self.assertNotIn("brandSlogan", ad)
            self.assertNotIn("slogan", ad)


class TestIntegrityValidation(unittest.TestCase):
    def test_integrity_passes_when_slogan_connection_does_not_echo_exact_slogan(self) -> None:
        plan = _assemble(2, series_payload=_series_ads_with_modified_slogan_connection(ad_count=2))
        upstream = make_upstream_snapshot(
            product_name_resolved=plan.product_name_resolved,
            selected_strategy=_selected_strategy(),
            selected_slogan=_selected_slogan(),
            selected_conceptual=_selected_conceptual(),
            brand_physical=parse_brand_physical_output(_brand_physical()),
            graphic=parse_graphic_system_output(_graphic()),
        )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=upstream,
            detected_language=plan.detected_language,
        )
        self.assertTrue(result.ok, msg=str(result.reasons))
        self.assertNotIn("ad_2_slogan_inconsistent", result.reasons)

    def test_integrity_still_detects_corrupted_assembled_ad_slogan(self) -> None:
        plan = _assemble(2)
        plan.planning_internals["adInternals"][2]["brandSlogan"] = "Mutated Slogan"
        upstream = make_upstream_snapshot(
            product_name_resolved=plan.product_name_resolved,
            selected_strategy=_selected_strategy(),
            selected_slogan=_selected_slogan(),
            selected_conceptual=_selected_conceptual(),
            brand_physical=parse_brand_physical_output(_brand_physical()),
            graphic=parse_graphic_system_output(_graphic()),
        )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=upstream,
            detected_language=plan.detected_language,
        )
        self.assertIn("ad_2_slogan_inconsistent", result.reasons)

    def test_integrity_still_detects_upstream_brand_slogan_mutation(self) -> None:
        plan = _assemble(2)
        plan = copy.deepcopy(plan)
        object.__setattr__(plan, "brand_slogan", "Changed Campaign Slogan")
        upstream = make_upstream_snapshot(
            product_name_resolved=plan.product_name_resolved,
            selected_strategy=_selected_strategy(),
            selected_slogan=_selected_slogan(),
            selected_conceptual=_selected_conceptual(),
            brand_physical=parse_brand_physical_output(_brand_physical()),
            graphic=parse_graphic_system_output(_graphic()),
        )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=upstream,
            detected_language=plan.detected_language,
        )
        self.assertIn("upstream_brand_slogan_mutated", result.reasons)


class TestProductionShapedRegression(unittest.TestCase):
    def test_observed_failure_sequence_now_passes_integrity(self) -> None:
        selected = _selected_slogan()
        self.assertEqual(selected.brand_slogan, FIXED_SLOGAN)
        series_raw = _series_ads_with_modified_slogan_connection(ad_count=2)
        series = parse_series_ads_output(series_raw, expected_ad_count=2)
        plan = assemble_builder1_campaign(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            ad_count=2,
            detected_language="en",
            exploration_seed="prod",
            product_name_resolved="CarryShell",
            strategy=_selected_strategy(),
            strategy_selection=parse_strategy_selection(
                _strategy_selection_payload(selected_id="S01"),
                parse_strategy_scan(_strategy_scan_payload(), product_description=BRIEF),
            )[0],
            selected_slogan=selected,
            conceptual=_selected_conceptual(),
            brand_physical=parse_brand_physical_output(_brand_physical()),
            graphic=parse_graphic_system_output(_graphic()),
            series_ads=series,
        )
        upstream = make_upstream_snapshot(
            product_name_resolved="CarryShell",
            selected_strategy=_selected_strategy(),
            selected_slogan=selected,
            selected_conceptual=_selected_conceptual(),
            brand_physical=parse_brand_physical_output(_brand_physical()),
            graphic=parse_graphic_system_output(_graphic()),
        )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=upstream,
            detected_language="en",
        )
        self.assertTrue(result.ok, msg=str(result.reasons))
        self.assertEqual(plan.planning_internals["adInternals"][1]["brandSlogan"], FIXED_SLOGAN)
        self.assertEqual(plan.planning_internals["adInternals"][2]["brandSlogan"], FIXED_SLOGAN)

    def test_schema_without_per_ad_slogan_field_still_assembles(self) -> None:
        payload = _series_ads(2)
        for ad in payload["ads"]:
            ad.pop("brandSlogan", None)
        plan = _assemble(2, series_payload=payload)
        self.assertEqual(plan.brand_slogan, FIXED_SLOGAN)


class TestImagePromptAndHeadlineRegression(unittest.TestCase):
    def test_image_prompt_uses_campaign_level_fixed_slogan(self) -> None:
        plan = _assemble(2, series_payload=_series_ads_with_model_slogan_fields(ad_count=2))
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(FIXED_SLOGAN, prompt)
        self.assertNotIn("Different Slogan Text", prompt)

    def test_headline_seven_word_rule_unchanged(self) -> None:
        headline = " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 1))
        obj = {
            "productNameResolved": "TestBrand",
            "strategicProblem": "Daily carry damage",
            "relativeAdvantage": "Survives daily drops",
            "relativeAdvantageSource": "explicit_brief",
            "brandSlogan": FIXED_SLOGAN,
            "conceptualGenerator": "Stress-test mechanism",
            "conceptualGeneratorAction": "Show everyday impact survival visually",
            "conceptualGeneratorInput": "Everyday carry item",
            "conceptualGeneratorTransformation": "Impact absorbed",
            "conceptualGeneratorResult": "Visible durability proof",
            "physicalGenerator": "Reinforced shell",
            "ads": [
                {
                    "index": 1,
                    "headline": headline,
                    "marketingText": "word " * 50,
                    "physicalExecution": "Drop test",
                    "visualExecution": "Impact frame",
                    "sceneDescription": "Concrete sidewalk drop",
                    "newContribution": "Escalating drop height",
                    "conceptualExecution": "Stress reveal",
                    "conceptualActionProof": "Shell stays intact",
                },
                {
                    "index": 2,
                    "headline": "",
                    "marketingText": "word " * 50,
                    "physicalExecution": "Corner impact",
                    "visualExecution": "Close impact",
                    "sceneDescription": "Corner strike scene",
                    "newContribution": "Different impact angle",
                    "conceptualExecution": "Stress reveal two",
                    "conceptualActionProof": "Shell stays intact again",
                },
            ],
        }
        _, reasons = validate_series_plan_structure(
            obj,
            expected_format="portrait",
            expected_ad_count=2,
            product_name="TestBrand",
            product_description=BRIEF,
        )
        self.assertNotIn("headline_too_long", reasons)


class TestGenerateAgainUsesStoredSlogan(unittest.TestCase):
    def test_stored_plan_keeps_single_campaign_slogan_for_later_ads(self) -> None:
        plan = _assemble(4)
        self.assertEqual(plan.brand_slogan, FIXED_SLOGAN)
        for ad in plan.ads:
            self.assertEqual(plan.planning_internals["adInternals"][ad.index]["brandSlogan"], FIXED_SLOGAN)

    def test_campaign_store_round_trip_preserves_fixed_slogan_in_internals(self) -> None:
        from engine.builder1_campaign_store import (
            clear_memory_store_for_tests,
            create_campaign_session,
            get_campaign_session,
        )
        from engine.builder1_plan_spec import series_plan_from_store_dict, series_plan_to_store_dict

        clear_memory_store_for_tests()
        plan = _assemble(2, series_payload=_series_ads_with_modified_slogan_connection(ad_count=2))
        create_campaign_session(campaign_id="slogan-store", plan=plan, target_ad_count=2)
        stored = series_plan_to_store_dict(get_campaign_session("slogan-store").plan)
        for ad in stored["ads"]:
            self.assertNotIn("brandSlogan", ad)
        self.assertIn("planningInternals", stored)
        reloaded = series_plan_from_store_dict(stored)
        self.assertEqual(reloaded.brand_slogan, FIXED_SLOGAN)
        self.assertEqual(reloaded.planning_internals["adInternals"][2]["brandSlogan"], FIXED_SLOGAN)


class TestSeriesAdsPromptContract(unittest.TestCase):
    def test_series_ads_system_prompt_forbids_slogan_generation(self) -> None:
        from engine.builder1_planning_contract import STAGE_SERIES_ADS_SYSTEM

        self.assertIn("immutable", STAGE_SERIES_ADS_SYSTEM.lower())
        self.assertIn("sloganconnection", STAGE_SERIES_ADS_SYSTEM.lower())
        self.assertIn("server", STAGE_SERIES_ADS_SYSTEM.lower())


if __name__ == "__main__":
    unittest.main()
