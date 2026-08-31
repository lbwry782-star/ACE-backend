"""
Builder1 Public Simplicity Hebrew + overlap-safe familiarity tests.

Run: python -m unittest tests.test_builder1_public_simplicity_hebrew -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_advertising_comprehension import (
    _bridge_connects_advantage,
    _contains_observable_causal_action,
    _contains_physical_action,
    _count_symbolic_mappings,
    _passes_two_sentence_test,
    assess_everyday_familiarity,
    count_technical_familiarity_occurrences,
    detect_public_analogy_too_complex,
    scan_advertising_comprehension,
    validate_ad_advertising_comprehension,
)
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from tests.test_builder1_staged_planning import _internal_ad_fields


def _amir_seesaw_production_plan() -> Dict[str, Any]:
    """Production-shaped Hebrew tutor / seesaw regression (job f699d25b semantics)."""
    rel = "זמן ההוראה והתשומת לב של אמיר מוקדשים לתלמיד אחד ולהכנה לבגרות בהיסטוריה"
    ad1 = {
        "index": 1,
        "physicalExecution": "מאזניים עם משקולות שנים; מחוון מוכנות עולה",
        "visualExecution": "מאזניים ומחוון מוכנות",
        "sceneDescription": "סטודיו גרפי נקי",
        "conceptualExecution": "מוכנות עולה לבגרות",
        "conceptualActionProof": "המחוון עוקב אחרי ההתקדמות",
    }
    in1 = {
        "immediateClarityReason": (
            "הצופה רואה מאזניים עם משקולות של שנות היסטוריה "
            "ומחוון מוכנות שעולה לקראת יעד הבגרות"
        ),
        "relativeAdvantageConnection": (
            "עליית המחוון מראה שההכנה לבגרות בנויה סביב לתלמיד אחד "
            "ולא מתחלקת בין רבים"
        ),
        "sloganConnection": "המאזניים מבטאים שיעור שלא מתחלק",
        "executionScene": "סטודיו גרפי",
        "executionSubject": "מאזניים עם משקולות שנים",
        "executionAction": "מחוון המוכנות עוקבת ועולה ככל שהמשקולות מתייצבות",
        "executionObjectState": "משקולות שנים; מחוון עולה",
    }
    ad2 = {
        "index": 2,
        "physicalExecution": "מאזניים עם משקולת כבדה בצד התלמיד",
        "visualExecution": "מאזניים",
        "sceneDescription": "רקע נקי",
        "conceptualExecution": "משקל מרוכז",
        "conceptualActionProof": "צד אחד כבד",
    }
    in2 = {
        "immediateClarityReason": "הצופה רואה משקולת כבדה בצד התלמיד על המאזניים",
        "relativeAdvantageConnection": (
            "התשומת לב המלאה לתלמיד אחד מוצגת כמשקולת מרוכזת בצד שלו"
        ),
        "sloganConnection": "שיעור שלא מתחלק כאיזון לטובת תלמיד אחד",
        "executionScene": "מאזניים",
        "executionSubject": "מאזניים",
        "executionAction": "משקולת אחת מורידה צד",
    }
    return {
        "relativeAdvantage": rel,
        "brandSlogan": "אמיר גוטליב. שיעור שלא מתחלק.",
        "productName": "אמיר גוטליב",
        "productDescription": "מורה פרטי להיסטוריה. מכין תלמידים לבגרות בהיסטוריה.",
        "physicalGenerator": "מאזניים עם משקולות של שנות היסטוריה",
        "transferredObject": "מאזניים עם משקולות של שנות היסטוריה",
        "transferredObjectAction": "מחוון מוכנות עולה ככל שהמשקולות מתייצבות לקראת הבגרות",
        "physicalGeneratorNaturalPurpose": "איזון משקולות",
        "physicalGeneratorCampaignRole": "מנגנון מדידה מדורג שמראה התקדמות לקראת הבגרות",
        "conceptualGenerator": "הכנה ממוקדת לבגרות",
        "conceptualGeneratorAction": "הצגת מוכנות עולה לתלמיד אחד",
        "planningInternals": {
            "visualExecutionRoute": "ANALOGY_LED",
            "productModality": "PHYSICAL_PRODUCT",
            "adInternals": {1: in1, 2: in2},
        },
        "ads": [ad1, ad2],
    }


class TestTechnicalOverlapSafeCounting(unittest.TestCase):
    def test_ovet_counts_once_not_twice(self) -> None:
        count, matches = count_technical_familiarity_occurrences("מחוון המוכנות עוקבת ועולה")
        self.assertEqual(count, 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["marker"], "עוקבת")

    def test_distinct_ovet_and_ovav_count_separately(self) -> None:
        count, matches = count_technical_familiarity_occurrences("מערכת עוקב ועוקבת")
        self.assertEqual(count, 2)
        markers = {item["marker"] for item in matches}
        self.assertIn("עוקב", markers)
        self.assertIn("עוקבת", markers)

    def test_mikud_autofocus_overlapping_counts_once(self) -> None:
        count, matches = count_technical_familiarity_occurrences("מיקוד אוטומטי על התלמיד")
        self.assertEqual(count, 1)
        self.assertEqual(matches[0]["marker"], "מיקוד אוטומטי")

    def test_ovet_alone_is_specialized_not_technical(self) -> None:
        level = assess_everyday_familiarity("מחוון המוכנות עוקבת ועולה")
        self.assertEqual(level, "specialized")
        self.assertNotEqual(level, "technical")


class TestHebrewPhysicalActionCoverage(unittest.TestCase):
    def test_roeh_visible_action(self) -> None:
        self.assertTrue(_contains_physical_action("הצופה רואה מאזניים"))

    def test_roim_still_recognized(self) -> None:
        self.assertTrue(_contains_physical_action("הצופה רואים מאזניים"))

    def test_oleh_causal(self) -> None:
        self.assertTrue(_contains_observable_causal_action("מחוון מוכנות עולה"))

    def test_yored_inflections(self) -> None:
        self.assertTrue(_contains_observable_causal_action("משקולת יורדת בצד"))
        self.assertTrue(_contains_observable_causal_action("משקולות יורדים"))

    def test_raise_lower_place(self) -> None:
        self.assertTrue(_contains_observable_causal_action("מעלה משקולת"))
        self.assertTrue(_contains_observable_causal_action("מוריד משקולת"))
        self.assertTrue(_contains_observable_causal_action("מניח משקולות"))
        self.assertTrue(_contains_observable_causal_action("מניחים משקולות"))

    def test_seesaw_balance_simple_physical_event(self) -> None:
        text = "הצופה רואה מאזניים ומחוון מוכנות עולה"
        self.assertTrue(_contains_observable_causal_action(text))
        self.assertTrue(_contains_physical_action(text))


class TestTwoSentenceHebrewBridge(unittest.TestCase):
    def test_hebrew_bridge_overlap_reached_when_physical_simple(self) -> None:
        rel = "זמן ההוראה לתלמיד אחד ולהכנה לבגרות"
        imm = "הצופה רואה מאזניים"
        br = "ההכנה לבגרות לתלמיד אחד"
        ex = "מחוון עולה"
        self.assertTrue(
            _passes_two_sentence_test(
                immediate=imm,
                bridge=br,
                relative_advantage=rel,
                execution_blob=ex,
            )
        )

    def test_hebrew_weak_bridge_still_fails(self) -> None:
        rel = "זמן ההוראה לתלמיד אחד ולהכנה לבגרות"
        imm = "הצופה רואה מאזניים"
        br = "זה נראה מעניין"
        ex = "מחוון עולה"
        self.assertFalse(
            _passes_two_sentence_test(
                immediate=imm,
                bridge=br,
                relative_advantage=rel,
                execution_blob=ex,
            )
        )


class TestProductionSeesawRegression(unittest.TestCase):
    def test_production_ad1_no_longer_false_positive(self) -> None:
        plan = _amir_seesaw_production_plan()
        ad = plan["ads"][0]
        fields = plan["planningInternals"]["adInternals"][1]
        rel = plan["relativeAdvantage"]
        imm = fields["immediateClarityReason"]
        br = fields["relativeAdvantageConnection"]
        ex = " ".join(
            fields.get(k, "")
            for k in (
                "executionScene",
                "executionSubject",
                "executionAction",
                "executionObjectState",
            )
        )

        mapping = _count_symbolic_mappings(imm, br, fields.get("sloganConnection", ""), "", "", ad.get("conceptualExecution", ""))
        bridge_ok = _bridge_connects_advantage(
            bridge=br,
            relative_advantage=rel,
            slogan_connection=fields.get("sloganConnection", ""),
            punchline="",
            fields=fields,
            ad=ad,
        )
        fam = assess_everyday_familiarity(
            " ".join([imm, br, fields.get("sloganConnection", ""), plan["physicalGenerator"]]),
            ex,
        )
        physical_blob = f"{ex} {imm}"
        observable = _contains_observable_causal_action(physical_blob)
        physical = _contains_physical_action(physical_blob)
        simple = observable and physical
        two_sentence = _passes_two_sentence_test(
            immediate=imm,
            bridge=br,
            relative_advantage=rel,
            execution_blob=ex,
        )
        detect = detect_public_analogy_too_complex(plan_dict=plan, ad=ad, fields=fields)
        validate = validate_ad_advertising_comprehension(plan_dict=plan, ad=ad)

        self.assertEqual(mapping, 0)
        self.assertTrue(bridge_ok)
        self.assertTrue(simple)
        self.assertNotEqual(fam, "technical")
        self.assertTrue(two_sentence)
        self.assertFalse(detect)
        self.assertNotIn("public_analogy_too_complex", validate)

    def test_production_ad2_still_passes(self) -> None:
        plan = _amir_seesaw_production_plan()
        ad = plan["ads"][1]
        self.assertFalse(detect_public_analogy_too_complex(plan_dict=plan, ad=ad))
        self.assertNotIn(
            "public_analogy_too_complex",
            validate_ad_advertising_comprehension(plan_dict=plan, ad=ad),
        )


class TestComplexityStillRejected(unittest.TestCase):
    def test_autofocus_technical_chain_still_rejected(self) -> None:
        from tests.test_builder1_advertising_comprehension import _tutor_plan_dict

        plan = _tutor_plan_dict(
            immediateClarityReason=(
                "Autofocus plane tracks one gymnast through sensor feedback calibration loop"
            ),
            relativeAdvantageConnection=(
                "Optical tracking represents personalized adaptation through dynamic correction"
            ),
            sloganConnection="Gymnast maps to autofocus which maps to tutoring",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("public_analogy_too_complex", reasons)

    def test_three_plus_mappings_without_bridge_still_rejected(self) -> None:
        from tests.test_builder1_advertising_comprehension import _tutor_plan_dict

        plan = _tutor_plan_dict(
            immediateClarityReason="Gymnast represents autofocus represents sensor loop represents tutoring",
            relativeAdvantageConnection="Each step symbolizes the next mapping in the chain",
            sloganConnection="Scene maps gymnastics to camera to education",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("public_analogy_too_complex", reasons)
        self.assertIn("multi_hop_symbolic_chain", reasons)

    def test_distinct_technical_concepts_still_technical(self) -> None:
        level = assess_everyday_familiarity(
            "Autofocus plane with sensor feedback calibration optical tracking"
        )
        self.assertEqual(level, "technical")

    def test_caption_only_explanation_still_fails(self) -> None:
        from tests.test_builder1_advertising_comprehension import _tutor_plan_dict

        plan = _tutor_plan_dict(
            immediateClarityReason="Sharp landing against blurred movement is a familiar sports photography effect",
            relativeAdvantageConnection="The photo looks dynamic",
        )
        reasons = validate_ad_advertising_comprehension(plan_dict=plan, ad=plan["ads"][0])
        self.assertIn("advertising_bridge_unclear", reasons)


class TestPublicSimplicityEvidence(unittest.TestCase):
    def test_rejection_records_branch_and_metrics(self) -> None:
        from tests.test_builder1_advertising_comprehension import _tutor_plan_dict

        plan = _tutor_plan_dict(
            immediateClarityReason="Gymnast represents autofocus represents sensor loop represents tutoring",
            relativeAdvantageConnection="Each step symbolizes the next mapping in the chain",
            sloganConnection="Scene maps gymnastics to camera to education",
        )
        evidence: list[dict] = []
        baseline = scan_advertising_comprehension(copy.deepcopy(plan))
        with_evidence = scan_advertising_comprehension(copy.deepcopy(plan), evidence)
        self.assertEqual(baseline, with_evidence)
        public_entries = [item for item in evidence if item.get("code") == "public_analogy_too_complex"]
        self.assertTrue(public_entries)
        entry = public_entries[0]
        self.assertEqual(entry.get("detector"), "advertising_comprehension")
        self.assertTrue(entry.get("branch"))
        self.assertIn("mappingCount", entry)
        self.assertIn("bridgeOk", entry)
        self.assertIn("everydayFamiliarity", entry)

    def test_integrity_checks_identical_with_and_without_evidence(self) -> None:
        from tests.test_builder1_advertising_comprehension import _tutor_plan_dict

        plan = _tutor_plan_dict(
            immediateClarityReason="Gymnast represents autofocus represents sensor loop represents tutoring",
            relativeAdvantageConnection="Each step symbolizes the next mapping in the chain",
        )
        baseline = deterministic_builder1_integrity_checks(copy.deepcopy(plan))
        evidence: list[dict] = []
        with_evidence = deterministic_builder1_integrity_checks(copy.deepcopy(plan), evidence)
        self.assertEqual(baseline, with_evidence)


if __name__ == "__main__":
    unittest.main()
