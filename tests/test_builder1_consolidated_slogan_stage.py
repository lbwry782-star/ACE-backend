"""
Builder1 final-path slogan validation tests.

Run: python -m unittest tests.test_builder1_consolidated_slogan_stage -v
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from engine.builder1_plan_parser import validate_series_plan_structure, _word_count
from engine.builder1_plan_spec import BRAND_SLOGAN_MAX_WORDS, HEADLINE_MAX_WORDS
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_strategy_slogan_final import FINAL_SLOGAN_ID, parse_slogan_final_section
from tests.test_builder1_staged_planning import _slogan_final_payload

BRIEF = "Reinforced shell product for daily carry"


class TestFinalSloganPath(unittest.TestCase):
    def test_final_slogan_payload_parses(self) -> None:
        selection, selected = parse_slogan_final_section(
            _slogan_final_payload(),
            relative_advantage="Distinct advantage 1",
            product_description=BRIEF,
            detected_language="en",
        )
        self.assertEqual(selection.selected_candidate_id, FINAL_SLOGAN_ID)
        self.assertEqual(selected.brand_slogan, "Built To Last")

    def test_candidate_arrays_rejected(self) -> None:
        payload = {
            "candidates": [{"id": "L01", "brandSlogan": "Built To Last"}],
            "selectedCandidateId": "L01",
            "selectionReason": "Legacy",
        }
        with self.assertRaises(StageParseError) as ctx:
            parse_slogan_final_section(
                payload,
                relative_advantage="Distinct advantage 1",
                product_description=BRIEF,
                detected_language="en",
            )
        self.assertTrue(any("forbidden_candidate_field" in reason for reason in ctx.exception.reasons))

    def test_longer_natural_slogan_passes(self) -> None:
        slogan = "Built To Last Through Daily Drops"
        selection, selected = parse_slogan_final_section(
            _slogan_final_payload(brandSlogan=slogan),
            relative_advantage="Distinct advantage 1",
            product_description=BRIEF,
            detected_language="en",
        )
        self.assertEqual(selected.brand_slogan, slogan)
        self.assertEqual(selection.selected_candidate_id, FINAL_SLOGAN_ID)


class TestHeadlineRegression(unittest.TestCase):
    def _minimal_plan(self, *, headline: str) -> Dict[str, Any]:
        return {
            "productNameResolved": "CarryShell",
            "strategicProblem": "Daily carry damage",
            "relativeAdvantage": "Survives daily drops",
            "relativeAdvantageSource": "explicit_brief",
            "brandSlogan": "Built To Last Through Every Daily Drop And Impact",
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

    def test_seven_word_headline_remains_valid(self) -> None:
        headline = " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 1))
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertNotIn("headline_too_long", reasons)

    def test_eight_word_headline_remains_invalid(self) -> None:
        headline = " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 2))
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertIn("headline_too_long", reasons)

    def test_product_name_excluded_from_headline_count(self) -> None:
        headline = "CarryShell " + " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 1))
        self.assertGreater(_word_count(headline), HEADLINE_MAX_WORDS)
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertIn("headline_too_long", reasons)

    def test_headline_remains_optional(self) -> None:
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=""),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertNotIn("headline_too_long", reasons)


if __name__ == "__main__":
    unittest.main()
