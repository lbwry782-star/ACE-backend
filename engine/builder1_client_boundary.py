"""
Builder1 digital-agent client-implementation boundary.

Campaigns must be executable from the submitted brief without material client-side
strategic consulting, business transformation, or operational investment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from engine.builder1_plan_parser import _norm_text

CLIENT_ACTION_LEVELS = frozenset({"none", "simple_optional", "complex_required"})
IMPLEMENTATION_COST_LEVELS = frozenset({"none", "negligible", "material"})

SIMPLE_STRATEGIC_ACTION_MAX_WORDS = 30

CLIENT_BOUNDARY_REJECTION_CODES = frozenset(
    {
        "client_consultation_required",
        "client_implementation_too_complex",
        "material_client_investment_required",
        "business_transformation_required",
        "advantage_not_currently_true",
        "unsupported_future_capability",
    }
)

PROHIBITED_CLIENT_ACTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bnew product\b", "business_transformation_required"),
    (r"\bintroduce a new product\b", "business_transformation_required"),
    (r"\bnew service\b", "business_transformation_required"),
    (r"\bcreate a service\b", "business_transformation_required"),
    (r"\bbusiness model\b", "business_transformation_required"),
    (r"\bchange pricing\b", "business_transformation_required"),
    (r"\bpricing change\b", "business_transformation_required"),
    (r"\blower prices\b", "business_transformation_required"),
    (r"\braise prices\b", "business_transformation_required"),
    (r"\bdiscount\b", "unsupported_future_capability"),
    (r"\bprice cut\b", "unsupported_future_capability"),
    (r"\bguarantee\b", "unsupported_future_capability"),
    (r"\bmoney[- ]back\b", "unsupported_future_capability"),
    (r"\bstaff training\b", "client_implementation_too_complex"),
    (r"\btrain employees\b", "client_implementation_too_complex"),
    (r"\btrain staff\b", "client_implementation_too_complex"),
    (r"\bhire employees\b", "client_implementation_too_complex"),
    (r"\bhiring specialists\b", "client_implementation_too_complex"),
    (r"\bdashboard\b", "client_implementation_too_complex"),
    (r"\breporting system\b", "client_implementation_too_complex"),
    (r"\bnew technology\b", "client_implementation_too_complex"),
    (r"\btechnical system\b", "client_implementation_too_complex"),
    (r"\bbuild a system\b", "client_implementation_too_complex"),
    (r"\bdevelop a platform\b", "client_implementation_too_complex"),
    (r"\bcollect customer data\b", "client_implementation_too_complex"),
    (r"\bcommission research\b", "client_implementation_too_complex"),
    (r"\bpackaging redesign\b", "client_implementation_too_complex"),
    (r"\brepackage\b", "client_implementation_too_complex"),
    (r"\brenovate\b", "client_implementation_too_complex"),
    (r"\bchange production\b", "client_implementation_too_complex"),
    (r"\bloyalty program\b", "business_transformation_required"),
    (r"\bdelivery service\b", "business_transformation_required"),
    (r"\blogistics capability\b", "business_transformation_required"),
    (r"\bcommercial partnership\b", "business_transformation_required"),
    (r"\bopen a new location\b", "business_transformation_required"),
    (r"\bfull rebrand\b", "business_transformation_required"),
    (r"\brebrand\b", "business_transformation_required"),
    (r"\bdiscovery workshop\b", "client_consultation_required"),
    (r"\bstrategic workshop\b", "client_consultation_required"),
    (r"\bconsultation\b", "client_consultation_required"),
    (r"\bmanagement interview\b", "client_consultation_required"),
    (r"\bmajor investment\b", "material_client_investment_required"),
    (r"\bsubstantial cost\b", "material_client_investment_required"),
    (r"\binvest in infrastructure\b", "material_client_investment_required"),
)

FUTURE_CAPABILITY_AS_CURRENT_PATTERNS: Tuple[str, ...] = (
    r"\bwill launch\b",
    r"\bcoming soon\b",
    r"\bwe are introducing\b",
    r"\bnow offers\b",
    r"\bnow includes\b",
    r"\bnow provides\b",
)


@dataclass(frozen=True)
class StrategyBoundaryFields:
    campaign_executable_now: bool
    requires_client_consultation: bool
    client_action_level: str
    implementation_cost_level: str
    simple_strategic_action: Optional[str]


def parse_strategy_boundary_fields(item: Dict[str, Any], *, candidate_id: str) -> Tuple[Optional[StrategyBoundaryFields], List[str]]:
    reasons: List[str] = []
    prefix = f"strategy_scan_{candidate_id}"

    exec_now = item.get("campaignExecutableNow")
    if not isinstance(exec_now, bool):
        reasons.append(f"{prefix}_campaignExecutableNow_not_boolean")
        exec_now = False

    requires_consultation = item.get("requiresClientConsultation")
    if not isinstance(requires_consultation, bool):
        reasons.append(f"{prefix}_requiresClientConsultation_not_boolean")
        requires_consultation = False

    action_level = _norm_text(item.get("clientActionLevel")).lower()
    if action_level not in CLIENT_ACTION_LEVELS:
        reasons.append(f"{prefix}_invalid_clientActionLevel")
        action_level = "none"

    cost_level = _norm_text(item.get("implementationCostLevel")).lower()
    if cost_level not in IMPLEMENTATION_COST_LEVELS:
        reasons.append(f"{prefix}_invalid_implementationCostLevel")
        cost_level = "none"

    simple_raw = item.get("simpleStrategicAction")
    if simple_raw is not None and not isinstance(simple_raw, str):
        reasons.append(f"{prefix}_simpleStrategicAction_not_string_or_null")
        simple_action: Optional[str] = None
    else:
        simple_action = _norm_text(simple_raw) if simple_raw else None
        if simple_action and len(simple_action.split()) > SIMPLE_STRATEGIC_ACTION_MAX_WORDS:
            reasons.append(f"{prefix}_simpleStrategicAction_too_long")

    if action_level == "none" and simple_action:
        reasons.append(f"{prefix}_simpleStrategicAction_without_optional_action")

    if reasons:
        return None, reasons

    return (
        StrategyBoundaryFields(
            campaign_executable_now=exec_now,
            requires_client_consultation=requires_consultation,
            client_action_level=action_level,
            implementation_cost_level=cost_level,
            simple_strategic_action=simple_action,
        ),
        [],
    )


def strategy_boundary_fields_is_eligible(fields: StrategyBoundaryFields) -> bool:
    if not fields.campaign_executable_now:
        return False
    if fields.requires_client_consultation:
        return False
    if fields.client_action_level not in {"none", "simple_optional"}:
        return False
    if fields.implementation_cost_level not in {"none", "negligible"}:
        return False
    return True


def strategy_candidate_is_eligible(candidate: Any) -> bool:
    return strategy_boundary_fields_is_eligible(
        StrategyBoundaryFields(
            campaign_executable_now=candidate.campaign_executable_now,
            requires_client_consultation=candidate.requires_client_consultation,
            client_action_level=candidate.client_action_level,
            implementation_cost_level=candidate.implementation_cost_level,
            simple_strategic_action=candidate.simple_strategic_action,
        )
    )


def filter_eligible_strategy_candidates(candidates: Iterable[Any]) -> List[Any]:
    return [c for c in candidates if strategy_candidate_is_eligible(c)]


def _term_supported_by_brief(term_pattern: str, product_description: str) -> bool:
    return bool(re.search(term_pattern, product_description or "", re.I))


def scan_text_for_prohibited_client_action(
    text: object,
    *,
    product_description: str = "",
) -> Optional[str]:
    normalized = _norm_text(text)
    if not normalized:
        return None
    lowered = normalized.lower()
    for pattern, code in PROHIBITED_CLIENT_ACTION_PATTERNS:
        if not re.search(pattern, lowered, re.I):
            continue
        if code == "unsupported_future_capability" and (
            _term_supported_by_brief(pattern, product_description)
            or _term_supported_by_brief(r"\bdiscount\b", product_description)
            or _term_supported_by_brief(r"\bguarantee\b", product_description)
        ):
            continue
        return code
    return None


def scan_text_for_future_capability_claim(text: object) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    lowered = normalized.lower()
    return any(re.search(pattern, lowered, re.I) for pattern in FUTURE_CAPABILITY_AS_CURRENT_PATTERNS)


def validate_strategy_candidate_text_boundary(
    *,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    simple_strategic_action: Optional[str],
    product_description: str,
) -> List[str]:
    reasons: List[str] = []
    for field_name, value in (
        ("strategicProblem", strategic_problem),
        ("relativeAdvantage", relative_advantage),
        ("briefSupport", brief_support),
    ):
        code = scan_text_for_prohibited_client_action(value, product_description=product_description)
        if code:
            reasons.append(f"strategy_scan_{field_name}_{code}")
    if simple_strategic_action:
        code = scan_text_for_prohibited_client_action(
            simple_strategic_action,
            product_description=product_description,
        )
        if code:
            reasons.append(f"strategy_scan_simpleStrategicAction_{code}")
    return reasons


def validate_conceptual_boundary_text(
    *,
    generator: str,
    action: str,
    result: str,
    why: str,
    product_description: str,
) -> List[str]:
    reasons: List[str] = []
    for field_name, value in (
        ("generator", generator),
        ("action", action),
        ("result", result),
        ("whyItExpressesAdvantage", why),
    ):
        code = scan_text_for_prohibited_client_action(value, product_description=product_description)
        if code:
            reasons.append(f"conceptual_scan_{field_name}_{code}")
    return reasons


def validate_brand_physical_boundary_text(
    *,
    brand_slogan: str,
    slogan_action: str,
    campaign_rationale: str,
    physical_generator_campaign_role: str,
    product_description: str,
) -> List[str]:
    reasons: List[str] = []
    for field_name, value in (
        ("brandSlogan", brand_slogan),
        ("sloganAction", slogan_action),
        ("campaignRationale", campaign_rationale),
        ("physicalGeneratorCampaignRole", physical_generator_campaign_role),
    ):
        code = scan_text_for_prohibited_client_action(value, product_description=product_description)
        if code:
            reasons.append(f"brand_physical_{field_name}_{code}")
    return reasons


def validate_series_ads_boundary_text(
    ads: Iterable[Dict[str, Any]],
    *,
    product_description: str,
) -> List[str]:
    reasons: List[str] = []
    for ad in ads:
        idx = ad.get("index")
        for field_name in ("headline", "marketingText", "sceneDescription", "conceptualExecution"):
            value = ad.get(field_name)
            if value is None or value == "":
                continue
            code = scan_text_for_prohibited_client_action(value, product_description=product_description)
            if code:
                reasons.append(f"series_ads_ad{idx}_{field_name}_{code}")
            if field_name in {"headline", "marketingText"} and scan_text_for_future_capability_claim(value):
                reasons.append(f"series_ads_ad{idx}_{field_name}_unsupported_future_capability")
    return reasons


def deterministic_client_boundary_checks(plan_dict: Dict[str, Any]) -> List[str]:
    product_description = str(plan_dict.get("productDescription") or "")
    reasons: List[str] = []

    for field_name, key in (
        ("relativeAdvantage", "relativeAdvantage"),
        ("brandSlogan", "brandSlogan"),
        ("campaignRationale", "campaignRationale"),
        ("conceptualGeneratorAction", "conceptualGeneratorAction"),
    ):
        code = scan_text_for_prohibited_client_action(plan_dict.get(key), product_description=product_description)
        if code:
            reasons.append(code)

    ads = plan_dict.get("ads")
    if isinstance(ads, list):
        reasons.extend(
            validate_series_ads_boundary_text(ads, product_description=product_description)
        )

    normalized: List[str] = []
    for reason in reasons:
        if reason.startswith("series_ads_") or reason.startswith("brand_physical_") or reason.startswith("conceptual_"):
            for code in CLIENT_BOUNDARY_REJECTION_CODES:
                if code in reason:
                    normalized.append(code)
                    break
            else:
                normalized.append("business_transformation_required")
        elif reason in CLIENT_BOUNDARY_REJECTION_CODES:
            normalized.append(reason)
        else:
            normalized.append(reason)
    return list(dict.fromkeys(normalized))


def is_client_boundary_rejection(codes: List[str]) -> bool:
    return any(code in CLIENT_BOUNDARY_REJECTION_CODES for code in codes)
