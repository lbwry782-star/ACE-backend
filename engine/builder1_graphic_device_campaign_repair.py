"""
Deterministic zero-planning repair for redundant explanatory graphic devices.

Removes copper-frame annotation devices from the known rain-gutter campaign while
preserving the physical concentration mechanism and campaign state.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from engine.builder1_campaign_completion import evaluate_campaign_completion
from engine.builder1_campaign_store import (
    CampaignStoreError,
    apply_proactive_plan_revision,
    get_campaign_session,
    get_campaign_session_raw,
)
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from engine.builder1_graphic_device_necessity import (
    device_text_is_explanatory_overlay,
    recurring_graphic_device_is_absent,
)
from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_to_store_dict

TARGET_RAIN_GUTTER_CAMPAIGN_ID = "b59781f3-a4fa-4352-9f27-fa9ca326b1f3"

_FRAME_REFERENCE_RE = re.compile(
    r"("
    r"שתי\s+תחימ(?:ות|ה)|"
    r"שתי\s+מסגר(?:ות|ה)|"
    r"מסגר(?:ות|ה)\s+נחושת|"
    r"תחימ(?:ות|ה)\s+מלבנ|"
    r"בדיוק\s+שתי\s+מסגר|"
    r"מסמנ(?:ות|ים)\s+א(?:ת|ת)\s+השלב|"
    r"תחימ(?:ות|ה)\s+ש(?:מ)?סמנ"
    r")",
    re.IGNORECASE,
)

_REWRITE_SENTENCE_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"[^.]*?(?:שתי\s+(?:תחימ(?:ות|ה)|מסגר(?:ות|ה))[^.]*(?:מרזב|חבית)[^.]*\.)",
            re.IGNORECASE,
        ),
        "המרזב והחבית יוצרים יחד את שני שלבי הריכוז דרך הזרימה הפיזית, בלי סימונים גרפיים.",
    ),
    (
        re.compile(
            r"[^.]*?בדיוק\s+שתי\s+מסגר(?:ות|ה)[^.]*\.",
            re.IGNORECASE,
        ),
        "אותה חוקיות ויזואלית נשמרת דרך המרזב, הצינור והחבית כשלבי ריכוז נראים.",
    ),
    (
        re.compile(
            r"[^.]*?מסגר(?:ות|ה)\s+נחושת[^.]*\.",
            re.IGNORECASE,
        ),
        "הריכוז נקרא בעין דרך המסלול הפיזי מהגג אל החבית.",
    ),
    (
        re.compile(
            r"[^.]*?תחימ(?:ות|ה)\s+מלבנ[^.]*\.",
            re.IGNORECASE,
        ),
        "שלבי האיסוף נקראים מהאובייקטים הפיזיים עצמם — מרזב, צינור וחבית.",
    ),
)

_SHAPE_LANGUAGE_REWRITE = (
    "גיאומטריה נקייה עם היררכיה חזותית ברורה דרך האובייקטים הפיזיים, בלי תחימות הסבר."
)
_FRAMING_RULE_REWRITE = (
    "המרכיב המרכזי והאזור הטקסטואלי מופרדים בהיררכיה טיפוגרפית; המנגנון הפיזי נשאר קריא ללא מסגרות הסבר."
)
_DEVICE_RULE_CLEAR = ""
_DEVICE_CLEAR = ""


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def plan_matches_rain_gutter_frame_repair(plan_dict: Dict[str, Any]) -> bool:
    graphic = plan_dict.get("graphicGenerator")
    if not isinstance(graphic, dict):
        return False
    device = _norm(graphic.get("recurringGraphicDevice"))
    rule = _norm(graphic.get("recurringGraphicDeviceRule"))
    if not device_text_is_explanatory_overlay(device, rule):
        return False
    physical = _norm(plan_dict.get("physicalGenerator"))
    transferred = _norm(plan_dict.get("transferredObject"))
    context = f"{physical} {transferred} {device} {rule}".lower()
    markers = ("מרזב", "חבית", "gutter", "barrel", "rain", "גשם")
    return sum(1 for marker in markers if marker in context) >= 2


def _rewrite_text_value(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original or not _FRAME_REFERENCE_RE.search(original):
        return original, False
    updated = original
    for pattern, replacement in _REWRITE_SENTENCE_PATTERNS:
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated, count=1)
            break
    if _FRAME_REFERENCE_RE.search(updated):
        updated = _FRAME_REFERENCE_RE.sub("הזרימה הפיזית", updated)
    updated = " ".join(updated.split())
    return updated, updated != original


def _set_change(changes: List[Dict[str, str]], path: str, before: str, after: str) -> None:
    if before != after:
        changes.append({"path": path, "before": before, "after": after})


def repair_redundant_frame_references_in_plan(plan_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    repaired = copy.deepcopy(plan_dict)
    changes: List[Dict[str, str]] = []

    graphic = repaired.setdefault("graphicGenerator", {})
    if isinstance(graphic, dict):
        before_device = _norm(graphic.get("recurringGraphicDevice"))
        before_rule = _norm(graphic.get("recurringGraphicDeviceRule"))
        graphic["recurringGraphicDevice"] = _DEVICE_CLEAR
        graphic["recurringGraphicDeviceRule"] = _DEVICE_RULE_CLEAR
        _set_change(changes, "graphicGenerator.recurringGraphicDevice", before_device, _DEVICE_CLEAR)
        _set_change(changes, "graphicGenerator.recurringGraphicDeviceRule", before_rule, _DEVICE_RULE_CLEAR)

        for field, replacement in (
            ("shapeLanguage", _SHAPE_LANGUAGE_REWRITE),
            ("framingRule", _FRAMING_RULE_REWRITE),
        ):
            before = _norm(graphic.get(field))
            after, _ = _rewrite_text_value(before)
            if _FRAME_REFERENCE_RE.search(before):
                after = replacement
            graphic[field] = after
            _set_change(changes, f"graphicGenerator.{field}", before, after)

    ad_fields = (
        "visualExecution",
        "headlineNeededReason",
        "singleChangedPropertyOrAction",
        "sameVisualLawProof",
        "physicalExecution",
        "sceneDescription",
        "conceptualExecution",
        "conceptualActionProof",
        "newContribution",
        "immediateClarityReason",
        "sloganConnection",
        "relativeAdvantageConnection",
        "distinctFromOtherAdsReason",
    )
    ads = repaired.get("ads")
    if isinstance(ads, list):
        for ad in ads:
            if not isinstance(ad, dict):
                continue
            idx = ad.get("index")
            for field in ad_fields:
                before = _norm(ad.get(field))
                after, changed = _rewrite_text_value(before)
                if changed:
                    ad[field] = after
                    _set_change(changes, f"ads[{idx}].{field}", before, after)

    internals = repaired.setdefault("planningInternals", {})
    if isinstance(internals, dict):
        ad_internals = internals.get("adInternals")
        if isinstance(ad_internals, dict):
            for key, payload in ad_internals.items():
                if not isinstance(payload, dict):
                    continue
                for field in ad_fields:
                    before = _norm(payload.get(field))
                    after, changed = _rewrite_text_value(before)
                    if changed:
                        payload[field] = after
                        _set_change(changes, f"planningInternals.adInternals[{key}].{field}", before, after)

        rationale = _norm(internals.get("campaignRationale") or repaired.get("campaignRationale"))
        if rationale:
            after, changed = _rewrite_text_value(rationale)
            if changed:
                internals["campaignRationale"] = after
                repaired["campaignRationale"] = after
                _set_change(changes, "campaignRationale", rationale, after)

    return repaired, changes


def validate_repaired_plan(plan: Builder1SeriesPlan) -> List[str]:
    plan_dict = series_plan_to_store_dict(plan)
    return deterministic_builder1_integrity_checks(plan_dict)


def _collect_state_snapshot(session) -> Dict[str, Any]:
    completion = evaluate_campaign_completion(session)
    return {
        "planRevision": session.plan_revision,
        "generatedCount": session.generated_count,
        "nextAdIndex": session.next_ad_index,
        "generatedIndexes": list(session.generated_indexes),
        "retryMode": session.retry_mode,
        "status": session.status,
        "campaignReady": completion.get("campaignReady"),
        "campaignComplete": completion.get("campaignComplete"),
        "canGenerateNext": (
            not session.complete
            and session.next_ad_index is not None
            and session.generating_index is None
            and not session.repair_in_progress
        ),
        "adArtifactIndexes": sorted(str(k) for k in (session.ad_artifacts or {}).keys()),
    }


def run_graphic_device_campaign_cleanup(
    campaign_id: str,
    *,
    dry_run: bool = True,
) -> Dict[str, Any]:
    cid = (campaign_id or "").strip()
    raw = get_campaign_session_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")

    session = get_campaign_session(cid)
    before_state = _collect_state_snapshot(session)
    plan_dict = copy.deepcopy(raw.get("plan") or {})
    if not plan_matches_rain_gutter_frame_repair(plan_dict):
        raise CampaignStoreError("campaign_not_eligible_for_graphic_device_repair")

    repaired_dict, field_changes = repair_redundant_frame_references_in_plan(plan_dict)
    repaired_plan, parse_reasons = validate_series_plan_structure(
        repaired_dict,
        expected_format=str(repaired_dict.get("format") or session.format),
        expected_ad_count=int(repaired_dict.get("adCount") or session.target_ad_count),
        product_name=str(repaired_dict.get("productName") or session.plan.product_name),
        product_description=str(repaired_dict.get("productDescription") or session.plan.product_description),
        require_internal_scans=False,
    )
    validation_errors = list(parse_reasons or [])
    if repaired_plan is None:
        validation_errors = validation_errors or ["invalid_plan"]
    else:
        validation_errors.extend(validate_repaired_plan(repaired_plan))

    graphic = repaired_dict.get("graphicGenerator") or {}
    device_absent = recurring_graphic_device_is_absent(
        graphic.get("recurringGraphicDevice"),
        graphic.get("recurringGraphicDeviceRule"),
    )

    report: Dict[str, Any] = {
        "campaignId": cid,
        "dryRun": bool(dry_run),
        "eligible": True,
        "before": before_state,
        "fieldChanges": field_changes,
        "validationErrors": validation_errors,
        "deviceAbsentAfterRepair": device_absent,
        "paidCalls": 0,
        "planningCalls": 0,
    }

    if validation_errors or repaired_plan is None:
        report["applied"] = False
        report["after"] = before_state
        return report

    if dry_run:
        after_state = dict(before_state)
        after_state["planRevision"] = before_state["planRevision"] + 1
        report["applied"] = False
        report["after"] = after_state
        return report

    updated = apply_proactive_plan_revision(cid, repaired_plan)
    after_state = _collect_state_snapshot(updated)
    report["applied"] = True
    report["after"] = after_state
    return report
