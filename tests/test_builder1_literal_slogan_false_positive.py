"""
Builder1 literal-slogan false-positive regression tests (production צעד צעד / שונות).

Run: python -m unittest tests.test_builder1_literal_slogan_false_positive -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_literal_embodiment import (
    extract_internal_slogan_action_tokens,
    extract_public_slogan_content_tokens,
    scan_literal_embodiment_bias,
    _detect_literal_slogan_illustration,
    _has_independent_visual_proof,
    _slogan_object_overlap_tokens,
)
from tests.test_builder1_series import _base_campaign, _graphic


def _marketing_block() -> str:
    return "word " * 50


def _tsaad_tsaad_production_rejected_plan() -> Dict[str, Any]:
    """Sanitized production-shaped plan rejected on matchedTerms=['שונות']."""
    transferred = "מגש הגשה עם מסגרת מחיצות נשלפת ומנות מזון שונות"
    plan = _base_campaign(2)
    plan.update(
        {
            "productName": "צעד צעד",
            "productNameResolved": "צעד צעד",
            "detectedLanguage": "he",
            "brandSlogan": "צעד צעד. כל צעד, בחנות אחת.",
            "sloganAction": "לרכז אפשרויות שונות למסלול קנייה אחד.",
            "relativeAdvantage": "כל סוגי הנעליים במקום אחד במקום חיפוש מפוצל",
            "strategicProblem": "קונים מחפשים סגנונות שונים בחנויות נפרדות",
            "conceptualGenerator": "ריכוז סוגים שונים במרחב אחד",
            "conceptualGeneratorAction": "מחיצות נעלמות ומשאירות סוגים שונים באותו מגש",
            "physicalGenerator": transferred,
            "transferredObject": transferred,
            "transferredObjectAction": "מסגרת המחיצות נשלפת ומשאירה קבוצות מזון שונות באותו מגש",
            "campaignRationale": (
                "היעלמות המחיצות יוצרת ביטוי עצמאי לריכוז סוגים שונים במקום אחד "
                "בלי לצייר מילות הסלוגן מילולית"
            ),
            "graphicGenerator": _graphic(),
            "seriesGenerator": {
                "type": "partition_removal",
                "principle": "Different groups share one tray after partitions leave",
                "progression": "Two distinct food-group proofs",
            },
            "planningInternals": {
                "whyClearerThanShowingProduct": (
                    "מגש עם מחיצות נעלמות מראה ריכוז סוגים שונים בצורה ברורה יותר מצילום חנות נעליים"
                ),
                "conceptualGeneratorWhyItExpressesSlogan": (
                    "המחיצות הנעלמות מבטאות את רעיון 'בחנות אחת' כמרחב משותף לסוגים שונים"
                ),
                "conceptualLineage": {
                    "selectedConceptCandidateId": "C01",
                    "sourceSloganCandidateId": "S01",
                    "fixedBrandSlogan": "צעד צעד. כל צעד, בחנות אחת.",
                    "fixedImpliedAction": "לרכז אפשרויות שונות למסלול קנייה אחד.",
                },
                "adInternals": {
                    1: {
                        "conceptualActionProof": "רק מסגרת המחיצות נשארת; שלוש קבוצות מזון שונות נשארות מובחנות",
                        "categoryRelevanceReason": (
                            "המזון הוא אובייקט חיצוני ולא המחשה מילולית של צעדים, מסלולים או נעליים"
                        ),
                        "relativeAdvantageConnection": "סוגים שונים חולקים מרחב אחד כמו קטגוריות נעליים בחנות אחת",
                        "immediateClarityReason": "הצופה רואה מחיצות שנעלמות ומשאירות סוגים שונים באותו מגש",
                        "singleChangedPropertyOrAction": "מסגרת המחיצות נעלמת",
                    },
                    2: {
                        "conceptualActionProof": "במגש השני נשארות שתי קבוצות מזון שונות ללא מחיצה",
                        "categoryRelevanceReason": (
                            "המנגנון מראה ריכוז סוגים שונים בלי לצייר את מילת הסלוגן"
                        ),
                        "relativeAdvantageConnection": "מרחב אחד מכיל סוגים שונים במקום חיפוש מפוצל",
                        "immediateClarityReason": "שתי קבוצות מזון שונות חולקות מגש אחד",
                        "singleChangedPropertyOrAction": "אין מחיצה בין הקבוצות",
                    },
                },
            },
            "ads": [
                {
                    "index": 1,
                    "variationLabel": "v1",
                    "newContribution": "Partition frame leaves",
                    "physicalExecution": "מגש הגשה; מסגרת מחיצות נשלפת",
                    "visualExecution": "מגש עם קבוצות מזון שונות",
                    "sceneDescription": "סטודיו נקי עם מגש הגשה",
                    "conceptualExecution": "ריכוז סוגים שונים",
                    "conceptualActionProof": "רק מסגרת המחיצות נשארת; שלוש קבוצות מזון שונות נשארות מובחנות",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
                {
                    "index": 2,
                    "variationLabel": "v2",
                    "newContribution": "Two groups one tray",
                    "physicalExecution": "מגש שני ללא מחיצות; שתי קבוצות מזון",
                    "visualExecution": "שתי קבוצות מזון שונות באותו מגש",
                    "sceneDescription": "רקע נקי",
                    "conceptualExecution": "סוגים שונים באותו מרחב",
                    "conceptualActionProof": "במגש השני נשארות שתי קבוצות מזון שונות ללא מחיצה",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
            ],
        }
    )
    return plan


class TestProductionShonutFalsePositive(unittest.TestCase):
    def test_action_only_shonut_overlap_does_not_reject(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        public = extract_public_slogan_content_tokens(slogan=plan["brandSlogan"])
        action = extract_internal_slogan_action_tokens(implied_action=plan["sloganAction"])
        transferred = plan["transferredObject"]
        self.assertIn("שונות", action)
        self.assertNotIn("שונות", public)
        self.assertIn(
            "שונות",
            _slogan_object_overlap_tokens(slogan_tokens=action, object_text=transferred),
        )
        self.assertFalse(
            _slogan_object_overlap_tokens(slogan_tokens=public, object_text=transferred),
        )

    def test_production_plan_passes_literal_guard(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        evidence: list[dict] = []
        reasons = scan_literal_embodiment_bias(plan, evidence)
        self.assertNotIn("literal_slogan_illustration", reasons)
        self.assertFalse(_detect_literal_slogan_illustration(plan, evidence))

    def test_production_plan_has_independent_proof(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        self.assertTrue(_has_independent_visual_proof(plan))


class TestLiteralGuardPreservation(unittest.TestCase):
    def test_genuine_literal_door_still_rejects(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "brandSlogan": "Opens every door",
                "sloganAction": "Remove access barriers",
                "transferredObject": "Door",
                "physicalGenerator": "Door",
                "whyClearerThanShowingProduct": "Shows a door because the slogan mentions opening doors",
                "ads": [
                    {
                        "index": 1,
                        "variationLabel": "v1",
                        "newContribution": "Literal door",
                        "physicalExecution": "Door opening",
                        "visualExecution": "Door",
                        "sceneDescription": "Door centered",
                        "conceptualExecution": "Door opens",
                        "conceptualActionProof": "Shows the slogan noun",
                        "headline": None,
                        "headlineNeededReason": "x",
                        "marketingText": _marketing_block(),
                        "sloganConnection": "Shows a door because the slogan mentions opening doors",
                    },
                    copy.deepcopy(
                        {
                            "index": 2,
                            "variationLabel": "v2",
                            "newContribution": "Literal door two",
                            "physicalExecution": "Door handle",
                            "visualExecution": "Door",
                            "sceneDescription": "Door",
                            "conceptualExecution": "Door",
                            "conceptualActionProof": "Door",
                            "headline": None,
                            "headlineNeededReason": "x",
                            "marketingText": _marketing_block(),
                        }
                    ),
                ],
            }
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertIn("literal_slogan_illustration", reasons)

    def test_internal_action_overlap_only_passes_with_mechanism(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_single_generic_public_overlap_does_not_auto_reject(self) -> None:
        plan = _tsaad_tsaad_production_rejected_plan()
        plan["brandSlogan"] = "צעד צעד. אפשרויות שונות בחנות אחת."
        plan["planningInternals"]["whyClearerThanShowingProduct"] = (
            "The unexpected tray mechanism proves variety in one place without repeating slogan words"
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_english_independent_proof_still_counts(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "brandSlogan": "Opens every door",
                "sloganAction": "Remove barriers",
                "transferredObject": "Door",
                "physicalGenerator": "Door",
                "whyClearerThanShowingProduct": (
                    "The unexpected shattering transformation proves breakthrough without repeating the word door"
                ),
                "campaignRationale": "Physical proof through transformation",
                "ads": plan["ads"],
            }
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_slogan_illustration", reasons)

    def test_literal_product_embodiment_unchanged(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "productNameResolved": "צעד צעד",
                "brandSlogan": "ביטוח שמלווה אותך",
                "transferredObject": "צעד צעד",
                "physicalGenerator": "צעד צעד",
            }
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertIn("literal_product_embodiment", reasons)


if __name__ == "__main__":
    unittest.main()
