"""
Builder1 series ad execution distinctness tests.

Run: python -m unittest tests.test_builder1_series_distinctness -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from engine.builder1_series_distinctness import (
    execution_fingerprint,
    normalize_execution_text,
    validate_ad_execution_distinctness,
    VISUAL_EXECUTION_FIELDS,
    CONCEPTUAL_EXECUTION_FIELDS,
)
from engine.builder1_series_execution_repair import merge_repaired_ads
from tests.test_builder1_series import _graphic
from tests.test_builder1_staged_planning import _series_ads


def _ad(
    index: int,
    *,
    subject: str,
    action: str,
    state: str,
    scene: str,
    punchline: str,
    conceptual: str | None = None,
    visual: str | None = None,
    physical: str | None = None,
    scene_description: str | None = None,
    headline: str | None = None,
) -> Dict[str, Any]:
    return {
        "index": index,
        "variationLabel": f"var-{index}",
        "newContribution": f"Contribution {index}",
        "conceptualExecution": conceptual or f"Conceptual realization {index} for {subject}",
        "conceptualActionProof": f"Proof {index}",
        "physicalExecution": physical or f"Physical setup {index}",
        "visualExecution": visual or f"Visual setup {index}",
        "sceneDescription": scene_description or scene,
        "headline": headline,
        "headlineNeededReason": "Needed" if headline else "Self-explanatory",
        "marketingText": "word " * 50,
        "familiarExpectation": "Shared campaign expectation",
        "singleChangedPropertyOrAction": action,
        "immediateClarityReason": "Clear at a glance",
        "sloganConnection": "Expresses fixed slogan",
        "relativeAdvantageConnection": "Shows advantage",
        "brandOwnershipReason": "Ownable to brand",
        "categoryRelevanceReason": "Category relevant",
        "headlineRequired": headline is not None,
        "headlineReason": "Needed" if headline else "Self-explanatory",
        "sameVisualLawProof": "Same campaign visual law",
        "distinctFromOtherAdsReason": f"Distinct from other ads {index}",
        "noReuseCheck": f"No reuse {index}",
        "executionSubject": subject,
        "executionAction": action,
        "executionObjectState": state,
        "executionScene": scene,
        "executionPunchline": punchline,
    }


class TestSeriesDistinctnessPasses(unittest.TestCase):
    def test_same_conceptual_generator_different_executions_pass(self) -> None:
        ads = [
            _ad(1, subject="Domino chain A", action="First tile falls", state="Starting tilt", scene="Studio table", punchline="Chain begins"),
            _ad(2, subject="Domino chain B", action="Final tile lands", state="Completed row", scene="Wide studio", punchline="Chain completes"),
        ]
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])

    def test_same_physical_generator_different_states_pass(self) -> None:
        ads = [
            _ad(1, subject="Rubber ball", action="Drop from low height", state="Compressed on impact", scene="Concrete floor", punchline="Absorbs shock"),
            _ad(
                2,
                subject="Rubber ball",
                action="Drop from high shelf",
                state="Rebounds upward",
                scene="Concrete floor",
                punchline="Returns energy",
                physical="Rubber ball family",
                visual="Rubber ball family",
            ),
        ]
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])

    def test_identical_visual_execution_but_distinct_scene_passes(self) -> None:
        ads = [
            _ad(
                1,
                subject="Domino row",
                action="First tile tips",
                state="Initial motion",
                scene="Close tabletop",
                punchline="Start of chain",
                visual="Domino tiles in controlled chain reaction",
            ),
            _ad(
                2,
                subject="Domino row",
                action="Last tile strikes marker",
                state="Final impact",
                scene="Wide studio floor",
                punchline="End of chain",
                visual="Domino tiles in controlled chain reaction",
            ),
        ]
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])


class TestSeriesDistinctnessFailures(unittest.TestCase):
    def test_same_subject_and_action_with_rewording_fails_conceptual(self) -> None:
        ads = [
            _ad(1, subject="Domino row", action="Tiles fall in sequence", state="Mid cascade", scene="Studio", punchline="Sequential collapse"),
            _ad(
                2,
                subject="Domino row",
                action="Tiles fall in sequence",
                state="Mid cascade",
                scene="Studio",
                punchline="Sequential collapse",
                conceptual="Domino tiles fall one after another in a controlled chain",
            ),
        ]
        ads[0]["conceptualExecution"] = "Domino tiles fall one after another in a controlled chain"
        ads[1]["conceptualExecution"] = "A controlled chain makes domino tiles fall one after another"
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertIn("duplicate_conceptual_execution", reasons)

    def test_same_scene_with_camera_angle_change_fails_visual(self) -> None:
        ads = [
            _ad(1, subject="Domino row", action="Cascade", state="Moving", scene="Studio table", punchline="Chain reaction"),
            _ad(
                2,
                subject="Domino row",
                action="Cascade",
                state="Moving",
                scene="Studio table",
                punchline="Chain reaction",
                visual="Top-down view of domino cascade on studio table",
            ),
        ]
        ads[0]["visualExecution"] = "Top-down view of domino cascade on studio table"
        ads[0]["physicalExecution"] = "Domino cascade on studio table"
        ads[1]["physicalExecution"] = "Domino cascade on studio table from low angle"
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertIn("duplicate_visual_execution", reasons)

    def test_cosmetic_background_change_fails(self) -> None:
        ads = [
            _ad(1, subject="Ball", action="Bounce", state="Mid-air", scene="Studio backdrop", punchline="Returns up"),
            _ad(2, subject="Ball", action="Bounce", state="Mid-air", scene="Studio backdrop", punchline="Returns up"),
        ]
        ads[0]["visualExecution"] = "Rubber ball bouncing against gray background"
        ads[1]["visualExecution"] = "Rubber ball bouncing against blue background"
        ads[0]["physicalExecution"] = "Rubber ball bouncing"
        ads[1]["physicalExecution"] = "Rubber ball bouncing"
        ads[0]["sceneDescription"] = "Studio backdrop"
        ads[1]["sceneDescription"] = "Studio backdrop"
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertIn("duplicate_visual_execution", reasons)


class TestDistinctnessEdgeCases(unittest.TestCase):
    def test_missing_optional_fields_do_not_false_duplicate(self) -> None:
        ads = [
            _ad(1, subject="A", action="Act one", state="State one", scene="Scene one", punchline="Punch one"),
            _ad(2, subject="B", action="Act two", state="State two", scene="Scene two", punchline="Punch two"),
        ]
        ads[0]["executionScene"] = ""
        ads[1]["executionScene"] = ""
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])

    def test_hebrew_punctuation_variants_still_duplicate(self) -> None:
        ads = [
            _ad(1, subject="שורת domino", action="נפילה", state="בתנועה", scene="שולחן", punchline="שרשרת"),
            _ad(2, subject="שורת domino", action="נפילה", state="בתנועה", scene="שולחן", punchline="שרשרת"),
        ]
        ads[0]["conceptualExecution"] = "אריחי דומינו נופלים בשרשרת."
        ads[1]["conceptualExecution"] = "אריחי דומינו, נופלים בשרשרת"
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertIn("duplicate_conceptual_execution", reasons)

    def test_distinct_hebrew_executions_not_collapsed(self) -> None:
        ads = [
            _ad(1, subject="Domino A", action="Action A", state="State A", scene="Scene A", punchline="Punch A"),
            _ad(2, subject="Domino B", action="Action B", state="State B", scene="Scene B", punchline="Punch B"),
        ]
        ads[0]["conceptualExecution"] = "אריח ראשון מתחיל שרשרת"
        ads[1]["conceptualExecution"] = "אריח אחרון סוגר את השרשרת"
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])

    def test_campaign_fields_excluded_from_fingerprint(self) -> None:
        ad = _ad(1, subject="A", action="Act", state="State", scene="Scene", punchline="Punch")
        fp = execution_fingerprint(ad, VISUAL_EXECUTION_FIELDS)
        field_names = {name for name, _ in fp}
        self.assertNotIn("sameVisualLawProof", field_names)
        self.assertNotIn("sloganConnection", field_names)

    def test_ad_specific_fields_included(self) -> None:
        ad = _ad(1, subject="Domino", action="Fall", state="Moving", scene="Table", punchline="Chain")
        fp = execution_fingerprint(ad, CONCEPTUAL_EXECUTION_FIELDS)
        field_names = {name for name, _ in fp}
        self.assertIn("executionSubject", field_names)
        self.assertIn("executionAction", field_names)


class TestAssemblyIntegration(unittest.TestCase):
    def _assembled(self, ads: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = _series_ads(len(ads))
        base["ads"] = ads
        return {
            "productName": "TestBrand",
            "productDescription": "Brief",
            "format": "portrait",
            "adCount": len(ads),
            "detectedLanguage": "en",
            "productNameResolved": "TestBrand",
            "strategicProblem": "Problem",
            "strategicProblemEvidence": "From brief",
            "relativeAdvantage": "Advantage",
            "relativeAdvantageSource": "explicit_brief",
            "relativeAdvantageBriefSupport": "Support",
            "relativeAdvantageClaimRisk": "low",
            "problemAdvantageLink": "Link",
            "brandSlogan": "Built To Last",
            "sloganDerivation": "Derivation",
            "sloganAction": "Show durability",
            "conceptualGenerator": "Drop proof",
            "conceptualGeneratorAction": "Show impact survival",
            "conceptualGeneratorInput": "Everyday object",
            "conceptualGeneratorTransformation": "Impact absorbed",
            "conceptualGeneratorResult": "Object survives",
            "conceptualGeneratorWhyItExpressesSlogan": "Shows durability",
            "conceptualGeneratorWhyItExpressesAdvantage": "Shows advantage",
            "physicalGenerator": "Rubber ball family",
            "physicalGeneratorNaturalPurpose": "Bounce",
            "physicalGeneratorCampaignRole": "Prove durability",
            "transferredObject": "Rubber ball family",
            "transferredObjectAction": "Bounces",
            "productVisibilityPolicy": "FORBIDDEN",
            "graphicGenerator": _graphic(),
            "seriesGenerator": base["seriesGenerator"],
            "mediumParticipates": False,
            "mediumRole": "",
            "campaignRationale": "Rationale",
            "ads": ads,
        }

    def test_two_ad_campaign_passes_with_two_dimensions(self) -> None:
        ads = [
            _ad(1, subject="A", action="Act A", state="State A", scene="Scene A", punchline="Punch A"),
            _ad(2, subject="B", action="Act B", state="State B", scene="Scene B", punchline="Punch B"),
        ]
        plan, reasons = validate_series_plan_structure(
            self._assembled(ads),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="TestBrand",
            product_description="Brief",
            require_internal_scans=False,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(reasons, [])

    def test_genuine_duplicate_blocks_assembly(self) -> None:
        ads = [
            _ad(1, subject="Same", action="Same", state="Same", scene="Same", punchline="Same"),
            _ad(2, subject="Same", action="Same", state="Same", scene="Same", punchline="Same"),
        ]
        ads[0]["visualExecution"] = "Identical visual execution"
        ads[1]["visualExecution"] = "Identical visual execution"
        ads[0]["conceptualExecution"] = "Identical conceptual execution"
        ads[1]["conceptualExecution"] = "Identical conceptual execution"
        plan, reasons = validate_series_plan_structure(
            self._assembled(ads),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="TestBrand",
            product_description="Brief",
            require_internal_scans=False,
        )
        self.assertIsNone(plan)
        self.assertTrue(any(reason.startswith("duplicate_") for reason in reasons))


class TestTargetedRepair(unittest.TestCase):
    def test_merge_repaired_ads_preserves_valid_records(self) -> None:
        original = [
            _ad(1, subject="Keep", action="Keep", state="Keep", scene="Keep", punchline="Keep"),
            _ad(2, subject="Dup", action="Dup", state="Dup", scene="Dup", punchline="Dup"),
        ]
        repaired = [
            _ad(2, subject="New", action="New act", state="New state", scene="New scene", punchline="New punch"),
        ]
        merged = merge_repaired_ads(original_ads=original, repaired_ads=repaired, repair_indexes=[2])
        self.assertEqual(merged[0]["executionSubject"], "Keep")
        self.assertEqual(merged[1]["executionSubject"], "New")


class TestPlanningCallBudget(unittest.TestCase):
    def test_normal_expected_calls_remain_five(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)


class TestNormalization(unittest.TestCase):
    def test_normalize_execution_text_preserves_hebrew_words(self) -> None:
        left = normalize_execution_text("אריחי דומינו נופלים")
        right = normalize_execution_text("אריחי  דומינו, נופלים!")
        self.assertEqual(left, right)

    def test_normalize_execution_text_lowercases_ascii(self) -> None:
        self.assertEqual(normalize_execution_text("Domino TILE"), normalize_execution_text("domino tile"))


if __name__ == "__main__":
    unittest.main()
