"""
Builder2 no-logo policy — unbranded visuals; plain product name on closure only.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

BUILDER2_NO_LOGO_POLICY_VERSION = "builder2_no_logos_v1"

CREATOR_LOGO_POLICY_FIELDS = (
    "advertisedEntityName",
    "logoDependentConcept",
    "advertisedLogoRequested",
    "thirdPartyBrandingRisk",
    "inventedLogoRisk",
    "brandedObjectRisk",
    "logoFreeSceneDescription",
    "genericObjectSubstitutions",
    "plainTextNameReservedForClosureOnly",
    "logoPolicySatisfied",
)

JUDGE_LOGO_POLICY_FIELDS = (
    "logoDetectedInPlan",
    "logoDependentMeaning",
    "advertisedLogoRequested",
    "thirdPartyBrandingDetected",
    "inventedLogoDetected",
    "brandedObjectRiskAccepted",
    "plainTextIdentificationOnly",
    "logoFreeExecutionAccepted",
    "logoPolicySatisfied",
    "rejectionReason",
)

LOGO_POLICY_BOOL_FIELDS = frozenset(
    {
        "logoDependentConcept",
        "advertisedLogoRequested",
        "plainTextNameReservedForClosureOnly",
        "logoPolicySatisfied",
    }
)

JUDGE_LOGO_POLICY_BOOL_FIELDS = frozenset(
    {
        "logoDetectedInPlan",
        "logoDependentMeaning",
        "advertisedLogoRequested",
        "thirdPartyBrandingDetected",
        "inventedLogoDetected",
        "brandedObjectRiskAccepted",
        "plainTextIdentificationOnly",
        "logoFreeExecutionAccepted",
        "logoPolicySatisfied",
    }
)

CREATOR_LOGO_VIOLATION_PATTERNS = (
    re.compile(r"\b(?:company|brand|product)\s+logo\b", re.I),
    re.compile(r"\b(?:custom|invented|designed)\s+(?:logo|emblem|wordmark|mark)\b", re.I),
    re.compile(r"\blogo(?:type|mark|-like|-shaped)?\b", re.I),
    re.compile(r"\bwordmark\b", re.I),
    re.compile(r"\bbrand emblem\b", re.I),
    re.compile(r"\btrademark\b", re.I),
    re.compile(r"\bmonogram\b", re.I),
    re.compile(r"\bbrand mascot\b", re.I),
    re.compile(r"\bwatermark\b", re.I),
    re.compile(r"\bbranded packaging\b", re.I),
    re.compile(r"\bbranded clothing\b", re.I),
    re.compile(r"\bvehicle emblem\b", re.I),
    re.compile(r"\bapp icon\b", re.I),
    re.compile(r"\bcompany sign\b", re.I),
)

NEGATION_PREFIX = re.compile(
    r"\b(?:no|without|reject(?:ing|ed)?|avoid(?:ing)?|exclude(?:d|s)?|absent|free of|never)\s+",
    re.I,
)

COMMERCIAL_REFERENCE_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bZippo\b", re.I), "a refined rectangular brushed-metal flip-top lighter with no markings"),
    (re.compile(r"\biPhone\b", re.I), "a generic modern smartphone with a blank screen and no emblem"),
    (re.compile(r"\bCoca-Cola\b", re.I), "an unbranded red beverage can with no text or symbols"),
    (re.compile(r"\bNike\b", re.I), "a plain athletic shirt with no marks"),
    (re.compile(r"\bTesla\b", re.I), "a generic modern electric car with all emblems absent"),
    (re.compile(r"\bApple\b(?=\s+(?:logo|device|phone|screen|mark))", re.I), "generic"),
    (re.compile(r"\bSamsung\b", re.I), "generic"),
    (re.compile(r"\bGoogle\b", re.I), "generic"),
    (re.compile(r"\bMicrosoft\b", re.I), "generic"),
    (re.compile(r"\bAmazon\b", re.I), "generic"),
)

THIRD_PARTY_BRAND_NAMES = frozenset(
    {
        "zippo",
        "iphone",
        "coca-cola",
        "nike",
        "tesla",
        "apple",
        "samsung",
        "google",
        "microsoft",
        "amazon",
        "adidas",
        "pepsi",
        "mcdonald",
        "starbucks",
    }
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def logo_policy_version(*, state: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(state, dict):
        version = _clean(state.get("logoPolicyVersion"))
        if version:
            return version
    if isinstance(plan, dict):
        return _clean(plan.get("logoPolicyVersion"))
    return ""


def is_no_logo_policy(*, state: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None) -> bool:
    return logo_policy_version(state=state, plan=plan) == BUILDER2_NO_LOGO_POLICY_VERSION


def stamp_no_logo_policy(state: Dict[str, Any]) -> None:
    state["logoPolicyVersion"] = BUILDER2_NO_LOGO_POLICY_VERSION
    state["logosAllowed"] = False
    state["advertisedLogoAllowed"] = False
    state["thirdPartyLogosAllowed"] = False
    state["inventedLogoAllowed"] = False
    state["plainTextAdvertisedNameAllowed"] = True
    state["plainTextAdvertisedNameOnly"] = True
    state["inSceneBrandTextAllowed"] = False


def build_no_logo_policy_snapshot() -> Dict[str, Any]:
    return {
        "logoPolicyVersion": BUILDER2_NO_LOGO_POLICY_VERSION,
        "logosAllowed": False,
        "advertisedLogoAllowed": False,
        "thirdPartyLogosAllowed": False,
        "inventedLogoAllowed": False,
        "plainTextAdvertisedNameAllowed": True,
        "plainTextAdvertisedNameOnly": True,
        "inSceneBrandTextAllowed": False,
        "logoPolicySatisfied": True,
    }


def build_builder2_no_logo_visual_policy_block(*, compact: bool = False) -> str:
    if compact:
        return (
            "NO-LOGO: Unbranded scene only; generic unmarked objects; blank or neutral packaging; "
            "no logos, wordmarks, trademarks, emblems, monograms, badges, seals, watermarks, branded clothing, "
            "vehicle emblems, branded interfaces, company signs, invented marks, or visible commercial names."
        )
    return (
        "NO-LOGO VISUAL POLICY: Entirely unbranded scene. Generic products and objects only. "
        "Blank or neutral packaging with no labels. No logos, trademarks, brand symbols, wordmarks, "
        "monograms, emblems, badges, seals, crests, mascots, watermarks, branded clothing, vehicle emblems, "
        "branded interfaces, company signs, invented logos, fake brand identities, decorative initials used as "
        "brand marks, or visible commercial names in the visual portion. Do not name real third-party brands as "
        "visual references. Translate commonly branded object types into generic physical descriptions with no "
        "markings."
    )


def normalize_builder2_media_prompt_text(text: str) -> str:
    out = _clean(text)
    if not out:
        return out
    for pattern, replacement in COMMERCIAL_REFERENCE_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    return out


def contains_third_party_brand_reference(text: str) -> bool:
    lowered = _clean(text).lower()
    if not lowered:
        return False
    for brand in THIRD_PARTY_BRAND_NAMES:
        if re.search(rf"\b{re.escape(brand)}\b", lowered):
            return True
    return False


def _raise_logo_error(code: str, *, field: str = "") -> None:
    suffix = f":{field}" if field else ""
    raise Builder2TournamentError(f"{code}{suffix}")


def _require_text(value: Any, *, field: str) -> str:
    text = _clean(value)
    if not text:
        _raise_logo_error("builder2_logo_policy_validation_failed", field=field)
    return text


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        _raise_logo_error("builder2_logo_policy_validation_failed", field=field)
    return value


def _collect_creator_execution_blob(candidate: Dict[str, Any]) -> str:
    parts = [
        _clean(candidate.get("coreVisualIdea")),
        _clean(candidate.get("visualMechanism")),
        _clean(candidate.get("openingFrameDescription")),
        _clean(candidate.get("videoPrompt")),
    ]
    sequence = candidate.get("sequence")
    if isinstance(sequence, dict):
        parts.extend(_clean(sequence.get(key)) for key in ("beginning", "development", "resolution"))
    runway = candidate.get("runwayFeasibility")
    if isinstance(runway, dict):
        for key in ("mainSubject", "mainAction", "location", "openingFrame", "whyRunwayShouldUnderstand"):
            parts.append(_clean(runway.get(key)))
    report = candidate.get("logoPolicyReport")
    if isinstance(report, dict):
        parts.append(_clean(report.get("logoFreeSceneDescription")))
        parts.append(_clean(report.get("brandedObjectRisk")))
    return " ".join(part for part in parts if part)


def _logo_violation_in_text(text: str) -> bool:
    if not text:
        return False
    for pattern in CREATOR_LOGO_VIOLATION_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            prefix = text[max(0, start - 40) : start]
            if NEGATION_PREFIX.search(prefix):
                continue
            return True
    return False


def validate_creator_logo_policy(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
    product_name: str = "",
    no_logo_required: bool = True,
) -> None:
    if not no_logo_required and not is_no_logo_policy(plan=candidate):
        return

    report = candidate.get("logoPolicyReport")
    if not isinstance(report, dict):
        _raise_logo_error("builder2_creator_logo_policy_missing", field="logoPolicyReport")

    for field in CREATOR_LOGO_POLICY_FIELDS:
        if field in LOGO_POLICY_BOOL_FIELDS:
            value = _require_bool(report.get(field), field=f"logoPolicyReport.{field}")
            if field == "logoDependentConcept" and value is not False:
                _raise_logo_error("builder2_creator_logo_dependent_concept", field=field)
            if field == "advertisedLogoRequested" and value is not False:
                _raise_logo_error("builder2_creator_advertised_logo_requested", field=field)
            if field == "plainTextNameReservedForClosureOnly" and value is not True:
                _raise_logo_error("builder2_creator_in_scene_brand_text", field=field)
            if field == "logoPolicySatisfied" and value is not True:
                _raise_logo_error("builder2_creator_logo_policy_unsatisfied", field=field)
        else:
            _require_text(report.get(field), field=f"logoPolicyReport.{field}")

    advertised_name = _clean(report.get("advertisedEntityName"))
    if product_name and advertised_name.lower() != _clean(product_name).lower():
        _raise_logo_error("builder2_creator_logo_policy_validation_failed", field="logoPolicyReport.advertisedEntityName")

    execution_blob = _collect_creator_execution_blob(candidate)
    if _logo_violation_in_text(execution_blob):
        _raise_logo_error("builder2_creator_logo_visible_in_plan", field="logoPolicyReport.logoFreeSceneDescription")
    if contains_third_party_brand_reference(execution_blob):
        _raise_logo_error("builder2_creator_third_party_brand_reference", field="coreVisualIdea")

    logger.info(
        "BUILDER2_LOGO_FREE_CREATOR_ACCEPTED prototypeId=%s logoDependent=%s plainTextOnly=%s riskSources=%s",
        assigned_prototype_id,
        str(report.get("logoDependentConcept") is True).lower(),
        str(report.get("plainTextNameReservedForClosureOnly") is True).lower(),
        len(_clean(report.get("brandedObjectRisk")).split(",")) if _clean(report.get("brandedObjectRisk")) else 0,
    )


def validate_judge_logo_policy(judgment: Dict[str, Any], *, candidate: Optional[Dict[str, Any]] = None) -> None:
    assessment = judgment.get("logoPolicyAssessment")
    if not isinstance(assessment, dict):
        return

    for field in JUDGE_LOGO_POLICY_FIELDS:
        if field == "rejectionReason":
            continue
        value = assessment.get(field)
        if field in JUDGE_LOGO_POLICY_BOOL_FIELDS:
            if not isinstance(value, bool):
                _raise_logo_error("builder2_judge_logo_policy_validation_failed", field=f"logoPolicyAssessment.{field}")


def apply_logo_eligibility_rules(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    assessment = out.get("logoPolicyAssessment")
    if not isinstance(assessment, dict):
        return out

    if assessment.get("logoDependentMeaning") is True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("logo_dependent_meaning")
    elif assessment.get("advertisedLogoRequested") is True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("advertised_logo_requested")
    elif assessment.get("thirdPartyBrandingDetected") is True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("third_party_branding_detected")
    elif assessment.get("inventedLogoDetected") is True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("invented_logo_detected")
    elif assessment.get("logoFreeExecutionAccepted") is not True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("logo_free_execution_rejected")
    elif assessment.get("plainTextIdentificationOnly") is not True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("in_scene_brand_text_proposed")
    elif assessment.get("logoPolicySatisfied") is not True:
        out["eligible"] = False
        out.setdefault("disqualifiers", []).append("logo_policy_rejected")

    return out


def judgment_rejects_logo_policy(judgment: Dict[str, Any]) -> bool:
    assessment = judgment.get("logoPolicyAssessment")
    if not isinstance(assessment, dict):
        return False
    if assessment.get("logoPolicySatisfied") is False:
        return True
    if assessment.get("logoFreeExecutionAccepted") is False:
        return True
    if assessment.get("logoDependentMeaning") is True:
        return True
    if assessment.get("advertisedLogoRequested") is True:
        return True
    if assessment.get("thirdPartyBrandingDetected") is True:
        return True
    if assessment.get("inventedLogoDetected") is True:
        return True
    return False


def mark_logo_free_closure_render(media: Dict[str, Any]) -> None:
    media["brandNameRenderedAsPlainText"] = True
    media["brandGraphicRendered"] = False
    media["logoAssetUsed"] = False
    media["inventedLogoRendered"] = False
    media["logoPolicyVersion"] = BUILDER2_NO_LOGO_POLICY_VERSION
    media["logoPolicySatisfied"] = True
    logger.info(
        "BUILDER2_LOGO_FREE_CLOSURE_RENDERED logoPolicyVersion=%s brandGraphicRendered=false logoAssetUsed=false "
        "plainTextNameOnly=true",
        BUILDER2_NO_LOGO_POLICY_VERSION,
    )


def log_logo_policy_validated(*, job_id: str = "", role: str = "") -> None:
    logger.info(
        "BUILDER2_LOGO_POLICY_VALIDATED jobId=%s role=%s logoPolicyVersion=%s logosAllowed=false "
        "plainTextAdvertisedNameOnly=true inSceneBrandTextAllowed=false",
        (job_id or "").strip() or "(none)",
        (role or "").strip() or "(none)",
        BUILDER2_NO_LOGO_POLICY_VERSION,
    )


def log_logo_free_prompt_applied(*, prompt_kind: str = "") -> None:
    logger.info(
        "BUILDER2_LOGO_FREE_PROMPT_APPLIED promptKind=%s logoPolicyVersion=%s",
        (prompt_kind or "").strip() or "(none)",
        BUILDER2_NO_LOGO_POLICY_VERSION,
    )


def validate_no_logo_completion(
    *,
    state: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    media: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not is_no_logo_policy(state=state, plan=plan):
        return []
    failures: List[str] = []
    media = media if isinstance(media, dict) else {}
    if media.get("logoAssetUsed") is True:
        failures.append("logo_asset_used_under_no_logo_policy")
    if media.get("brandGraphicRendered") is True:
        failures.append("brand_graphic_rendered_under_no_logo_policy")
    if media.get("brandNameRenderedAsPlainText") is not True:
        failures.append("brand_name_not_rendered_as_plain_text")
    if media.get("logoPolicySatisfied") is not True:
        failures.append("logo_policy_unsatisfied_at_completion")
    if _clean(media.get("logoPolicyVersion")) != BUILDER2_NO_LOGO_POLICY_VERSION:
        failures.append("logo_policy_version_missing_at_completion")
    return failures
