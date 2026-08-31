"""
Builder1 recurring graphic device necessity and campaign repair tests.

Run: python -m unittest tests.test_builder1_graphic_device_necessity -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_final_stages import parse_graphic_system_output
from engine.builder1_graphic_device_campaign_repair import (
    TARGET_RAIN_GUTTER_CAMPAIGN_ID,
    _AD2_CONCEPTUAL_EXECUTION_CANONICAL,
    _AD2_SLOGAN_CONNECTION_CANONICAL,
    _AD2_VISUAL_EXECUTION_CANONICAL,
    find_ad_entry,
    list_ad_list_position,
    plan_matches_rain_gutter_frame_repair,
    repair_redundant_frame_references_in_plan,
    run_graphic_device_campaign_cleanup,
    scan_ad_overlay_semantics,
)
from engine.builder1_graphic_device_necessity import (
    NO_RECURRING_GRAPHIC_DEVICE,
    REDUNDANT_EXPLANATORY_GRAPHIC_DEVICE,
    build_no_device_annotation_guard_block,
    evaluate_redundant_explanatory_graphic_device,
    recurring_graphic_device_is_absent,
    scan_graphic_device_necessity,
)
from engine.builder1_image_compliance import IMAGE_COMPLIANCE_SYSTEM_PROMPT
from engine.builder1_image_compliance_contract import IMAGE_COMPLIANCE_VIOLATION_CODES
from engine.builder1_planning_contract import STAGE_GRAPHIC_SYSTEM_SYSTEM
from engine.builder1_planning_model import GRAPHIC_SYSTEM_JSON_SCHEMA
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_visual_prompt import build_campaign_graphic_identity_block, build_visual_prompt
from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    reserve_next_ad_index,
    validate_next_ad_request,
)
from tests.test_builder1_graphic_contract import _gpt41_shaped_graphic
from tests.test_builder1_series import _base_campaign, _graphic, _parse


def _rain_gutter_campaign() -> Dict[str, Any]:
    plan = copy.deepcopy(_base_campaign(2))
    plan.update(
        {
            "detectedLanguage": "he",
            "physicalGenerator": "גשם → מרזב → צינור → חבית אחת",
            "transferredObject": "מערכת איסוף גשם",
            "transferredObjectAction": "הגשם נאסף במרזב, עובר בצינור ונשמר בחבית אחת",
            "conceptualGenerator": "ריכוז במקום פיזור",
        }
    )
    plan["graphicGenerator"] = _graphic()
    plan["graphicGenerator"]["recurringGraphicDevice"] = (
        "שתי תחימות מלבניות דקות בגוון נחושת: אחת סביב אזור המרזב ואחת סביב חבית האגירה."
    )
    plan["graphicGenerator"]["recurringGraphicDeviceRule"] = (
        "בכל מודעה מופיעות בדיוק שתי התחימות, באותו עובי ובאותו צבע; הן מסמנות שני שלבי איסוף."
    )
    plan["ads"][0]["visualExecution"] = (
        "גשם זורם אל מרזב גג; שתי מסגרות נחושת דקות וזהות מקיפות בנפרד את המרזב ואת החבית."
    )
    plan["ads"][1]["visualExecution"] = (
        "בדיוק שתי מסגרות נחושת מסמנות את שלבי הריכוז סביב המרזב והחבית."
    )
    plan["planningInternals"] = {
        "adInternals": {
            1: {"sameVisualLawProof": "שתי תחימות מסמנות את השלבים"},
            2: {"sameVisualLawProof": "בדיוק שתי מסגרות נחושת חוזרות"},
        }
    }
    return plan


def _production_shaped_rain_gutter_campaign() -> Dict[str, Any]:
    plan = _rain_gutter_campaign()
    ad2 = plan["ads"][1]
    ad2["index"] = 2
    ad2["visualExecution"] = (
        "צילום ריאליסטי לרוחב מזווית חזיתית־מעט־עליונה. "
        "עמק הגג, ברך הצינור ורגע ההתזה בחבית גלויים במרכז־ימין. "
        "מסגרת נחושת אחת מקיפה את מרזב העמק והשנייה את החבית. "
        "אותו רקע חורפי בהיר ואותו אזור טקסט שמאלי נשמרים."
    )
    ad2["conceptualExecution"] = (
        "גם כאשר המשאב מגיע בבת אחת משני כיוונים, "
        "שתי מסגרות נחושת מסמנות את שני שלבי הריכוז סביב המרזב והחבית."
    )
    ad2["sloganConnection"] = (
        "הסיסמה מספקת את ההבחנה המילולית בין עזרה כללית לבין מקצוע ובחינה מסוימים; "
        "החזות מוסיפה הוכחה עצמאית לכך ששתי מסגרות נחושת לוכדות גם משאב שמגיע מכמה כיוונים."
    )
    plan["graphicGenerator"]["palette"]["accent"] = "#D98245"
    plan["planningInternals"]["adInternals"][2] = {
        "sameVisualLawProof": "בדיוק שתי מסגרות נחושת מסמנות את שלבי הריכוז.",
        "visualExecution": ad2["visualExecution"],
    }
    return plan


class TestOptionalRecurringGraphicDevice(unittest.TestCase):
    def test_empty_recurring_device_parses(self) -> None:
        graphic = parse_graphic_system_output(
            _gpt41_shaped_graphic(recurringGraphicDevice="", recurringGraphicDeviceRule="")
        )
        self.assertTrue(recurring_graphic_device_is_absent(graphic.recurring_graphic_device, graphic.recurring_graphic_device_rule))

    def test_none_sentinel_recurring_device_parses(self) -> None:
        graphic = parse_graphic_system_output(
            _gpt41_shaped_graphic(
                recurringGraphicDevice=NO_RECURRING_GRAPHIC_DEVICE,
                recurringGraphicDeviceRule=NO_RECURRING_GRAPHIC_DEVICE,
            )
        )
        self.assertEqual(graphic.recurring_graphic_device, "")
        self.assertEqual(graphic.recurring_graphic_device_rule, "")

    def test_schema_allows_zero_length_recurring_fields(self) -> None:
        props = GRAPHIC_SYSTEM_JSON_SCHEMA["properties"]
        self.assertEqual(props["recurringGraphicDevice"]["minLength"], 0)
        self.assertEqual(props["recurringGraphicDeviceRule"]["minLength"], 0)

    def test_prompt_documents_optional_device(self) -> None:
        self.assertIn("OPTIONAL", STAGE_GRAPHIC_SYSTEM_SYSTEM)
        self.assertIn("NO_RECURRING_GRAPHIC_DEVICE", STAGE_GRAPHIC_SYSTEM_SYSTEM)


class TestVisualPromptWithoutDevice(unittest.TestCase):
    def test_image_prompt_omits_mandatory_device_language_when_absent(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.graphic_generator.recurring_graphic_device = ""
        plan.graphic_generator.recurring_graphic_device_rule = ""
        block = build_campaign_graphic_identity_block(plan)
        self.assertNotIn("Recurring graphic device:", block)
        self.assertNotIn("Render the recurring graphic device prominently", block)
        self.assertNotIn("Do not omit it", block)

    def test_image_prompt_keeps_mandatory_device_language_when_present(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        block = build_campaign_graphic_identity_block(plan)
        self.assertIn("Render the recurring graphic device prominently", block)

    def test_no_device_prompt_includes_annotation_guard(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.graphic_generator.recurring_graphic_device = ""
        plan.graphic_generator.recurring_graphic_device_rule = ""
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("Do not add bounding boxes", prompt)
        self.assertNotIn("Render prominently", prompt)


class TestRedundantExplanatoryGuard(unittest.TestCase):
    def test_rain_gutter_rectangle_device_is_redundant(self) -> None:
        code = evaluate_redundant_explanatory_graphic_device(
            device="שתי תחימות מלבניות דקות סביב המרזב והחבית",
            device_rule="בכל מודעה מופיעות בדיוק שתי התחימות ומסמנות את השלבים",
            physical_generator="גשם → מרזב → צינור → חבית",
            transferred_object="מערכת איסוף גשם",
        )
        self.assertEqual(code, REDUNDANT_EXPLANATORY_GRAPHIC_DEVICE)

    def test_playing_card_device_remains_allowed(self) -> None:
        code = evaluate_redundant_explanatory_graphic_device(
            device="The advertising medium becomes a playing card",
            device_rule="Identical card frame appears on every ad as the conceptual generator",
            physical_generator="Magnet attracting one nail",
            transferred_object="Magnet",
        )
        self.assertIsNone(code)

    def test_campaign_border_with_border_treatment_allowed(self) -> None:
        code = evaluate_redundant_explanatory_graphic_device(
            device="Thin outer campaign border framing the entire ad",
            device_rule="Identical campaign border on every execution",
            physical_generator="Umbrella blocking rain",
            transferred_object="Umbrella",
            border_treatment="thin_frame",
        )
        self.assertIsNone(code)

    def test_scan_flags_redundant_device_in_plan(self) -> None:
        reasons = scan_graphic_device_necessity(_rain_gutter_campaign())
        self.assertIn(REDUNDANT_EXPLANATORY_GRAPHIC_DEVICE, reasons)


class TestComplianceAnnotationOverlay(unittest.TestCase):
    def test_unplanned_annotation_overlay_is_allowed_code(self) -> None:
        self.assertIn("unplanned_annotation_overlay", IMAGE_COMPLIANCE_VIOLATION_CODES)

    def test_compliance_prompt_documents_annotation_overlay(self) -> None:
        self.assertIn("unplanned_annotation_overlay", IMAGE_COMPLIANCE_SYSTEM_PROMPT)


class TestRainGutterCampaignRepair(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def test_list_index_one_is_semantic_ad_two(self) -> None:
        plan = _production_shaped_rain_gutter_campaign()
        self.assertEqual(list_ad_list_position(plan, 2), 1)
        ad2 = find_ad_entry(plan, 2)
        self.assertIsNotNone(ad2)
        self.assertEqual(ad2.get("index"), 2)
        self.assertIn("מסגרת נחושת אחת", ad2["visualExecution"])

    def test_production_ad2_visual_execution_is_rewritten(self) -> None:
        repaired, changes = repair_redundant_frame_references_in_plan(_production_shaped_rain_gutter_campaign())
        ad2 = find_ad_entry(repaired, 2)
        self.assertIsNotNone(ad2)
        self.assertEqual(ad2["visualExecution"], _AD2_VISUAL_EXECUTION_CANONICAL)
        self.assertTrue(any("visualExecution" in c["path"] and c["path"].startswith("ads[2]") for c in changes))

    def test_production_ad2_conceptual_and_slogan_rewrites_are_grammatical(self) -> None:
        repaired, _ = repair_redundant_frame_references_in_plan(_production_shaped_rain_gutter_campaign())
        ad2 = find_ad_entry(repaired, 2)
        self.assertEqual(ad2["conceptualExecution"], _AD2_CONCEPTUAL_EXECUTION_CANONICAL)
        self.assertEqual(ad2["sloganConnection"], _AD2_SLOGAN_CONNECTION_CANONICAL)
        combined = f"{ad2['conceptualExecution']} {ad2['sloganConnection']}"
        self.assertNotIn("הזרימה הפיזית עצמאיות", combined)
        self.assertNotIn("הזרימה הפיזית לוכדות", combined)

    def test_complete_ad2_overlay_scan_is_clean(self) -> None:
        repaired, _ = repair_redundant_frame_references_in_plan(_production_shaped_rain_gutter_campaign())
        hits = scan_ad_overlay_semantics(repaired, semantic_index=2)
        self.assertEqual(hits, [])

    def test_accent_palette_preserved(self) -> None:
        repaired, _ = repair_redundant_frame_references_in_plan(_production_shaped_rain_gutter_campaign())
        self.assertEqual(repaired["graphicGenerator"]["palette"]["accent"], "#D98245")

    def test_dry_run_reports_clean_ad2_prompt(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        plan = _parse(raw, 2)
        cid = TARGET_RAIN_GUTTER_CAMPAIGN_ID
        create_campaign_session(campaign_id=cid, plan=plan, target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)

        report = run_graphic_device_campaign_cleanup(cid, dry_run=True)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["ad2PromptOverlayHits"], [])
        self.assertIn("Do not add bounding boxes", report["ad2PromptRelevantExcerpt"])
        self.assertIn("חבית", report["ad2VisualExecutionAfter"])
        self.assertIn("עמק הגג", report["ad2VisualExecutionAfter"])
        self.assertNotIn("מסגרת נחושת", report["ad2VisualExecutionAfter"])

    def test_dry_run_preserves_generated_count_and_artifact(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        plan = _parse(raw, 2)
        cid = TARGET_RAIN_GUTTER_CAMPAIGN_ID
        create_campaign_session(campaign_id=cid, plan=plan, target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)
        session = get_campaign_session(cid)
        session.ad_artifacts["1"] = {"token": "abc", "status": "succeeded", "planRevision": 1}
        from engine.builder1_campaign_store import _save_raw, get_campaign_session_raw

        raw_store = get_campaign_session_raw(cid) or {}
        raw_store["adArtifacts"] = session.ad_artifacts
        _save_raw(cid, raw_store)

        report = run_graphic_device_campaign_cleanup(cid, dry_run=True)
        self.assertTrue(report["dryRun"])
        self.assertFalse(report["applied"])
        self.assertEqual(report["before"]["generatedCount"], 1)
        self.assertEqual(report["after"]["planRevision"], report["before"]["planRevision"] + 1)
        self.assertEqual(report["before"]["nextAdIndex"], 2)

    def test_apply_bumps_revision_without_retry_mode(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        plan = _parse(raw, 2)
        cid = "cmp-apply-repair"
        create_campaign_session(campaign_id=cid, plan=plan, target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)

        report = run_graphic_device_campaign_cleanup(cid, dry_run=False)
        self.assertTrue(report["applied"])
        session = get_campaign_session(cid)
        self.assertEqual(session.generated_count, 1)
        self.assertEqual(session.next_ad_index, 2)
        self.assertEqual(session.retry_mode, "none")
        self.assertEqual(session.plan_revision, 2)
        validate_next_ad_request(cid, 2)

    def test_repair_makes_no_paid_calls(self) -> None:
        raw = _production_shaped_rain_gutter_campaign()
        plan = _parse(raw, 2)
        cid = TARGET_RAIN_GUTTER_CAMPAIGN_ID
        create_campaign_session(campaign_id=cid, plan=plan, target_ad_count=2)
        report = run_graphic_device_campaign_cleanup(cid, dry_run=True)
        self.assertEqual(report["paidCalls"], 0)
        self.assertEqual(report.get("planningCalls", 0), 0)

    def test_eligibility_fails_when_overlay_survives(self) -> None:
        from engine.builder1_graphic_device_campaign_repair import evaluate_repair_eligibility

        raw = _production_shaped_rain_gutter_campaign()
        plan = _parse(raw, 2)
        eligible, reasons = evaluate_repair_eligibility(raw, plan)
        self.assertFalse(eligible)
        self.assertTrue(any("overlay" in r or "visualExecution" in r for r in reasons))


class TestAnnotationGuardBlock(unittest.TestCase):
    def test_guard_block_text(self) -> None:
        block = build_no_device_annotation_guard_block(border_treatment="none")
        self.assertIn("bounding boxes", block.lower())


if __name__ == "__main__":
    unittest.main()
