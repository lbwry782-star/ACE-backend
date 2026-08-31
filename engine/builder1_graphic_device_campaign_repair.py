"""
Deterministic zero-planning repair for redundant explanatory graphic devices.

Removes copper-frame annotation devices from the known rain-gutter campaign while
preserving the physical concentration mechanism and campaign state.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.builder1_campaign_completion import evaluate_campaign_completion
from engine.builder1_campaign_store import (
    CampaignStoreError,
    apply_proactive_plan_revision,
    get_campaign_session,
    get_campaign_session_raw,
)
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from engine.builder1_graphic_device_necessity import (
    build_no_device_annotation_guard_block,
    device_text_is_explanatory_overlay,
    recurring_graphic_device_is_absent,
)
from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_to_store_dict
from engine.builder1_visual_prompt import build_visual_prompt

TARGET_RAIN_GUTTER_CAMPAIGN_ID = "b59781f3-a4fa-4352-9f27-fa9ca326b1f3"

SEMANTIC_AD2_INDEX = 2

_SHAPE_LANGUAGE_REWRITE = (
    "גיאומטריה נקייה עם היררכיה חזותית ברורה דרך האובייקטים הפיזיים, בלי תחימות הסבר."
)
_FRAMING_RULE_REWRITE = (
    "המרכיב המרכזי והאזור הטקסטואלי מופרדים בהיררכיה טיפוגרפית; המנגנון הפיזי נשאר קריא ללא מסגרות הסבר."
)

_AD2_VISUAL_EXECUTION_CANONICAL = (
    "צילום ריאליסטי לרוחב מזווית חזיתית־מעט־עליונה. "
    "עמק הגג, ברך הצינור ורגע ההתזה בחבית גלויים במרכז־ימין. "
    "המים משני שיפועי הגג מתכנסים באופן נראה לעמק המרכזי, "
    "ממשיכים דרך הצינור ונכנסים לחבית אחת. "
    "אותו רקע חורפי בהיר ואותו אזור טקסט שמאלי נשמרים."
)

_AD2_CONCEPTUAL_EXECUTION_CANONICAL = (
    "גם כאשר המשאב מגיע בבת אחת משני כיוונים, "
    "הזרימה מתכנסת פיזית למסלול אחד ונשמרת ביעד מוגדר אחד, "
    "במקום להתפצל לכמה שימושים."
)

_AD2_SLOGAN_CONNECTION_CANONICAL = (
    "הסיסמה מספקת את ההבחנה המילולית בין עזרה כללית לבין מקצוע ובחינה מסוימים; "
    "החזות מוסיפה הוכחה עצמאית לכך שגם משאב שמגיע מכמה כיוונים מתכנס ונשמר ביעד אחד."
)

_AD2_SAME_VISUAL_LAW_PROOF_CANONICAL = (
    "אותה חוקיות של ריכוז פיזי נראית דרך עמק הגג, הצינור והחבית — בלי סימון גרפי על השלבים."
)

_AD1_VISUAL_FRAME_SENTENCE_RE = re.compile(
    r"[^.]*?(?:שתי\s+(?:מסגר(?:ות|ה)|תחימ(?:ות|ה))[^.]*(?:מרזב|חבית)[^.]*\.)",
    re.IGNORECASE,
)
_AD1_VISUAL_FRAME_REPLACEMENT = (
    "הגשם זורם אל מרזב הגג; המים מתכנסים דרך הצינור ונשמרים בחבית אחת — בלי סימון גרפי."
)

_AD2_VISUAL_OVERLAY_SENTENCE_RE = re.compile(
    r"[^.]*?מסגר(?:ת|ות)\s+נחושת[^.]*\.",
    re.IGNORECASE,
)

_OVERLAY_SEMANTIC_TERMS: Tuple[str, ...] = (
    "מסגרת",
    "מסגרות",
    "תחימה",
    "תחימות",
    "מלבן",
    "מלבניות",
    "סימון",
    "מסמנות",
    "מסומנות",
    "outline",
    "frame",
    "rectangle",
    "annotation",
    "bounding",
    "highlight",
    "callout",
    "bracket",
)

_OVERLAY_AROUND_OBJECT_RES: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"מסגר(?:ת|ות)\s+נחושת",
        r"מסגר(?:ת|ות)\s+(?:אחת\s+)?מקיפ",
        r"תחימ(?:ה|ות)\s+מלבנ",
        r"שתי\s+(?:מסגר(?:ות|ה)|תחימ(?:ות|ה))",
        r"בדיוק\s+שתי\s+מסגר",
        r"מסמנ(?:ות|ים)\s+א(?:ת|ת)\s+(?:ה)?(?:שלב|מרזב|חבית)",
        r"\bframe\s+around\b",
        r"\brectangle\s+around\b",
        r"\bbounding\s+box",
        r"\bcallout\b",
        r"\bhighlight\s+box\b",
        r"נחושת\s+(?:אחת|שתיים).*?(?:מרזב|חבית|עמק)",
        r"(?:מרזב|חבית|עמק).*?מסגר(?:ת|ות)",
    )
)

_BROKEN_REWRITE_MARKERS: Tuple[str, ...] = (
    "הזרימה הפיזית עצמאיות",
    "הזרימה הפיזית לוכדות",
)

_PROMPT_OVERLAY_HIT_RES: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"Render the recurring graphic device prominently",
        r"Recurring device rule \(must be visibly present",
        r"Recurring graphic device:",
        r"מסגר(?:ת|ות)\s+נחושת",
        r"מקיפ(?:ה|ות)\s+א(?:ת|ת)\s+(?:מרזב|חבית|עמק)",
        r"תחימ(?:ה|ות)\s+מלבנ",
        r"שתי\s+(?:מסגר(?:ות|ה)|תחימ(?:ות|ה))",
        r"\bframe\s+around\b",
        r"\brectangle\s+around\b",
        r"\bbounding\s+box",
    )
)

_AD2_SCAN_FIELDS: Tuple[str, ...] = (
    "visualExecution",
    "physicalExecution",
    "sceneDescription",
    "conceptualExecution",
    "conceptualActionProof",
    "headlineNeededReason",
    "singleChangedPropertyOrAction",
    "sameVisualLawProof",
    "newContribution",
    "immediateClarityReason",
    "sloganConnection",
    "relativeAdvantageConnection",
    "distinctFromOtherAdsReason",
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _norm_key(value: object) -> str:
    return _norm(value).lower()


def find_ad_entry(plan_dict: Dict[str, Any], semantic_index: int) -> Optional[Dict[str, Any]]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return None
    for ad in ads:
        if isinstance(ad, dict) and int(ad.get("index") or 0) == semantic_index:
            return ad
    if 1 <= semantic_index <= len(ads):
        candidate = ads[semantic_index - 1]
        if isinstance(candidate, dict) and int(candidate.get("index") or semantic_index) == semantic_index:
            return candidate
    return None


def list_ad_list_position(plan_dict: Dict[str, Any], semantic_index: int) -> Optional[int]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return None
    for pos, ad in enumerate(ads):
        if isinstance(ad, dict) and int(ad.get("index") or 0) == semantic_index:
            return pos
    return None


def _ad_internals_entry(plan_dict: Dict[str, Any], semantic_index: int) -> Optional[Dict[str, Any]]:
    internals = plan_dict.get("planningInternals")
    if not isinstance(internals, dict):
        return None
    ad_internals = internals.get("adInternals")
    if not isinstance(ad_internals, dict):
        return None
    for key in (semantic_index, str(semantic_index)):
        payload = ad_internals.get(key)
        if isinstance(payload, dict):
            return payload
    return None


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


def _set_change(changes: List[Dict[str, str]], path: str, before: str, after: str) -> None:
    if before != after:
        changes.append({"path": path, "before": before, "after": after})


def _contains_overlay_around_object_semantics(text: object) -> bool:
    normalized = _norm(text)
    if not normalized:
        return False
    lowered = normalized.lower()
    if "#d98245" in lowered.replace(" ", ""):
        pass
    return any(pattern.search(normalized) for pattern in _OVERLAY_AROUND_OBJECT_RES)


def _rewrite_ad2_visual_execution(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original:
        return original, False
    if _AD2_VISUAL_OVERLAY_SENTENCE_RE.search(original) or _contains_overlay_around_object_semantics(original):
        return _AD2_VISUAL_EXECUTION_CANONICAL, original != _AD2_VISUAL_EXECUTION_CANONICAL
    if "מסגרת נחושת" in original or "מקיפה את מרזב" in original:
        return _AD2_VISUAL_EXECUTION_CANONICAL, True
    return original, False


def _rewrite_ad2_conceptual_execution(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original:
        return original, False
    if any(marker in original for marker in _BROKEN_REWRITE_MARKERS):
        return _AD2_CONCEPTUAL_EXECUTION_CANONICAL, True
    if "משני כיוונים" in original and (
        _contains_overlay_around_object_semantics(original)
        or "עצמאיות" in original
        or "מסגר" in original
        or "תחימ" in original
    ):
        return _AD2_CONCEPTUAL_EXECUTION_CANONICAL, True
    return original, False


def _rewrite_ad2_slogan_connection(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original:
        return original, False
    if any(marker in original for marker in _BROKEN_REWRITE_MARKERS):
        return _AD2_SLOGAN_CONNECTION_CANONICAL, True
    if "החזות מוסיפה הוכחה" in original and (
        _contains_overlay_around_object_semantics(original)
        or "לוכדות" in original
        or "מסגר" in original
        or "תחימ" in original
        or "הזרימה הפיזית" in original
    ):
        return _AD2_SLOGAN_CONNECTION_CANONICAL, True
    return original, False


def _rewrite_ad2_same_visual_law_proof(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original:
        return original, False
    if _contains_overlay_around_object_semantics(original):
        return _AD2_SAME_VISUAL_LAW_PROOF_CANONICAL, True
    return original, False


def _rewrite_ad1_visual_execution(text: object) -> Tuple[str, bool]:
    original = _norm(text)
    if not original:
        return original, False
    if _AD1_VISUAL_FRAME_SENTENCE_RE.search(original):
        updated = _AD1_VISUAL_FRAME_SENTENCE_RE.sub(_AD1_VISUAL_FRAME_REPLACEMENT, original, count=1)
        return updated, updated != original
    return original, False


def _apply_field_rewrite(
    *,
    target: Dict[str, Any],
    field: str,
    path: str,
    changes: List[Dict[str, str]],
    rewrite_fn,
) -> None:
    before = _norm(target.get(field))
    after, changed = rewrite_fn(before)
    if changed:
        target[field] = after
        _set_change(changes, path, before, after)


def repair_redundant_frame_references_in_plan(plan_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    repaired = copy.deepcopy(plan_dict)
    changes: List[Dict[str, str]] = []

    graphic = repaired.setdefault("graphicGenerator", {})
    if isinstance(graphic, dict):
        before_device = _norm(graphic.get("recurringGraphicDevice"))
        before_rule = _norm(graphic.get("recurringGraphicDeviceRule"))
        graphic["recurringGraphicDevice"] = ""
        graphic["recurringGraphicDeviceRule"] = ""
        _set_change(changes, "graphicGenerator.recurringGraphicDevice", before_device, "")
        _set_change(changes, "graphicGenerator.recurringGraphicDeviceRule", before_rule, "")

        for field, replacement in (
            ("shapeLanguage", _SHAPE_LANGUAGE_REWRITE),
            ("framingRule", _FRAMING_RULE_REWRITE),
        ):
            before = _norm(graphic.get(field))
            after = replacement if _contains_overlay_around_object_semantics(before) or before != replacement else before
            graphic[field] = after
            _set_change(changes, f"graphicGenerator.{field}", before, after)

    ad2 = find_ad_entry(repaired, SEMANTIC_AD2_INDEX)
    if ad2 is not None:
        list_pos = list_ad_list_position(repaired, SEMANTIC_AD2_INDEX)
        path_prefix = f"ads[{SEMANTIC_AD2_INDEX}]"
        if list_pos is not None:
            path_prefix = f"ads[{SEMANTIC_AD2_INDEX}] (listIndex={list_pos})"
        _apply_field_rewrite(
            target=ad2,
            field="visualExecution",
            path=f"{path_prefix}.visualExecution",
            changes=changes,
            rewrite_fn=_rewrite_ad2_visual_execution,
        )
        _apply_field_rewrite(
            target=ad2,
            field="conceptualExecution",
            path=f"{path_prefix}.conceptualExecution",
            changes=changes,
            rewrite_fn=_rewrite_ad2_conceptual_execution,
        )
        _apply_field_rewrite(
            target=ad2,
            field="sloganConnection",
            path=f"{path_prefix}.sloganConnection",
            changes=changes,
            rewrite_fn=_rewrite_ad2_slogan_connection,
        )
        _apply_field_rewrite(
            target=ad2,
            field="sameVisualLawProof",
            path=f"{path_prefix}.sameVisualLawProof",
            changes=changes,
            rewrite_fn=_rewrite_ad2_same_visual_law_proof,
        )
        for field in _AD2_SCAN_FIELDS:
            if field in {"visualExecution", "conceptualExecution", "sloganConnection", "sameVisualLawProof"}:
                continue
            before = _norm(ad2.get(field))
            if before and _contains_overlay_around_object_semantics(before):
                after = _AD2_SAME_VISUAL_LAW_PROOF_CANONICAL
                ad2[field] = after
                _set_change(changes, f"{path_prefix}.{field}", before, after)

    ad1 = find_ad_entry(repaired, 1)
    if ad1 is not None:
        _apply_field_rewrite(
            target=ad1,
            field="visualExecution",
            path="ads[1].visualExecution",
            changes=changes,
            rewrite_fn=_rewrite_ad1_visual_execution,
        )

    ad2_internals = _ad_internals_entry(repaired, SEMANTIC_AD2_INDEX)
    if ad2_internals is not None:
        for field in _AD2_SCAN_FIELDS:
            rewrite_fn = {
                "visualExecution": _rewrite_ad2_visual_execution,
                "conceptualExecution": _rewrite_ad2_conceptual_execution,
                "sloganConnection": _rewrite_ad2_slogan_connection,
                "sameVisualLawProof": _rewrite_ad2_same_visual_law_proof,
            }.get(field)
            if rewrite_fn:
                _apply_field_rewrite(
                    target=ad2_internals,
                    field=field,
                    path=f"planningInternals.adInternals[{SEMANTIC_AD2_INDEX}].{field}",
                    changes=changes,
                    rewrite_fn=rewrite_fn,
                )
            else:
                before = _norm(ad2_internals.get(field))
                if before and _contains_overlay_around_object_semantics(before):
                    after = _AD2_SAME_VISUAL_LAW_PROOF_CANONICAL
                    ad2_internals[field] = after
                    _set_change(
                        changes,
                        f"planningInternals.adInternals[{SEMANTIC_AD2_INDEX}].{field}",
                        before,
                        after,
                    )

    return repaired, changes


def scan_ad_overlay_semantics(plan_dict: Dict[str, Any], *, semantic_index: int) -> List[str]:
    hits: List[str] = []
    ad = find_ad_entry(plan_dict, semantic_index)
    if ad is not None:
        for field in _AD2_SCAN_FIELDS:
            text = _norm(ad.get(field))
            if not text:
                continue
            if any(marker in text for marker in _BROKEN_REWRITE_MARKERS):
                hits.append(f"ads[{semantic_index}].{field}:broken_rewrite_marker")
            if _contains_overlay_around_object_semantics(text):
                hits.append(f"ads[{semantic_index}].{field}:overlay_around_object")
            for term in _OVERLAY_SEMANTIC_TERMS:
                if term.lower() in text.lower() and _contains_overlay_around_object_semantics(text):
                    hits.append(f"ads[{semantic_index}].{field}:term:{term}")
                    break

    internals = _ad_internals_entry(plan_dict, semantic_index)
    if internals is not None:
        for field in _AD2_SCAN_FIELDS:
            text = _norm(internals.get(field))
            if not text:
                continue
            if any(marker in text for marker in _BROKEN_REWRITE_MARKERS):
                hits.append(f"planningInternals.adInternals[{semantic_index}].{field}:broken_rewrite_marker")
            if _contains_overlay_around_object_semantics(text):
                hits.append(f"planningInternals.adInternals[{semantic_index}].{field}:overlay_around_object")

    graphic = plan_dict.get("graphicGenerator")
    if isinstance(graphic, dict):
        for field in ("recurringGraphicDevice", "recurringGraphicDeviceRule", "shapeLanguage", "framingRule"):
            text = _norm(graphic.get(field))
            if text and _contains_overlay_around_object_semantics(text):
                hits.append(f"graphicGenerator.{field}:overlay_around_object")

    return list(dict.fromkeys(hits))


def scan_prompt_overlay_hits(prompt: str) -> List[str]:
    hits: List[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("do not ") or lowered.startswith("prohibit "):
            continue
        if "do not add bounding boxes" in lowered:
            continue
        for pattern in _PROMPT_OVERLAY_HIT_RES:
            if pattern.search(stripped):
                hits.append(pattern.pattern)
                break
    return list(dict.fromkeys(hits))


def build_ad2_prompt_relevant_excerpt(prompt: str) -> str:
    lines: List[str] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(token in line for token in ("מרזב", "חבית", "צינור", "גג", "גשם", "barrel", "gutter", "roof")):
            lines.append(line)
        if "bounding boxes" in lowered or "annotation outlines" in lowered or "callout frames" in lowered:
            lines.append(line)
        if line.startswith("Composition execution:") or line.startswith("MAIN VISUAL:") or line.startswith("ACTION:"):
            lines.append(line)
    if not lines:
        return _norm(prompt)[:480]
    return "\n".join(lines[:12])


def _build_projected_ad2_prompt(plan: Builder1SeriesPlan) -> str:
    ad2 = next((ad for ad in plan.ads if ad.index == SEMANTIC_AD2_INDEX), None)
    if ad2 is None:
        raise ValueError("ad2_missing")
    return build_visual_prompt(plan, ad2)


def evaluate_repair_eligibility(
    repaired_dict: Dict[str, Any],
    repaired_plan: Optional[Builder1SeriesPlan],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    graphic = repaired_dict.get("graphicGenerator") or {}
    if not recurring_graphic_device_is_absent(
        graphic.get("recurringGraphicDevice"),
        graphic.get("recurringGraphicDeviceRule"),
    ):
        reasons.append("recurring_device_not_absent")

    overlay_hits = scan_ad_overlay_semantics(repaired_dict, semantic_index=SEMANTIC_AD2_INDEX)
    if overlay_hits:
        reasons.extend(overlay_hits)

    ad2 = find_ad_entry(repaired_dict, SEMANTIC_AD2_INDEX)
    if ad2 is None:
        reasons.append("semantic_ad2_missing")
    else:
        visual = _norm(ad2.get("visualExecution"))
        if _contains_overlay_around_object_semantics(visual):
            reasons.append("ads[2].visualExecution:still_contains_overlay_instruction")
        if any(marker in visual for marker in _BROKEN_REWRITE_MARKERS):
            reasons.append("ads[2].visualExecution:broken_rewrite_marker")

    if repaired_plan is None:
        reasons.append("invalid_plan")
        return False, reasons

    try:
        prompt = _build_projected_ad2_prompt(repaired_plan)
    except ValueError as exc:
        reasons.append(str(exc))
        return False, reasons

    prompt_hits = scan_prompt_overlay_hits(prompt)
    if prompt_hits:
        reasons.extend([f"ad2_prompt:{hit}" for hit in prompt_hits])

    guard = build_no_device_annotation_guard_block(
        border_treatment=str((graphic or {}).get("borderTreatment") or "none")
    )
    if guard not in prompt:
        reasons.append("ad2_prompt:missing_annotation_guard")

    return not reasons, reasons


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

    ad2_overlay_scan = scan_ad_overlay_semantics(repaired_dict, semantic_index=SEMANTIC_AD2_INDEX)
    ad2_prompt = ""
    ad2_prompt_overlay_hits: List[str] = []
    ad2_prompt_excerpt = ""
    if repaired_plan is not None:
        try:
            ad2_prompt = _build_projected_ad2_prompt(repaired_plan)
            ad2_prompt_overlay_hits = scan_prompt_overlay_hits(ad2_prompt)
            ad2_prompt_excerpt = build_ad2_prompt_relevant_excerpt(ad2_prompt)
        except ValueError:
            ad2_prompt_overlay_hits = ["ad2_missing"]

    eligible, eligibility_reasons = evaluate_repair_eligibility(repaired_dict, repaired_plan)
    if validation_errors:
        eligible = False
        eligibility_reasons = list(dict.fromkeys(eligibility_reasons + validation_errors))

    ad2_entry = find_ad_entry(repaired_dict, SEMANTIC_AD2_INDEX)
    list_pos = list_ad_list_position(repaired_dict, SEMANTIC_AD2_INDEX)

    report: Dict[str, Any] = {
        "campaignId": cid,
        "dryRun": bool(dry_run),
        "eligible": eligible,
        "eligibilityReasons": eligibility_reasons,
        "before": before_state,
        "fieldChanges": field_changes,
        "validationErrors": validation_errors,
        "deviceAbsentAfterRepair": device_absent,
        "semanticAd2Index": SEMANTIC_AD2_INDEX,
        "semanticAd2ListIndex": list_pos,
        "ad2VisualExecutionAfter": _norm((ad2_entry or {}).get("visualExecution")),
        "ad2OverlayScan": ad2_overlay_scan,
        "ad2PromptOverlayHits": ad2_prompt_overlay_hits,
        "ad2PromptRelevantExcerpt": ad2_prompt_excerpt,
        "paidCalls": 0,
        "planningCalls": 0,
    }

    if not eligible or repaired_plan is None:
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
