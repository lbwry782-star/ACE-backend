"""
Builder1 direct-product route assessment — SIMPLE PRODUCT / DIRECT ADVANTAGE PRIORITY.

Deterministic validation only; assessment is produced inside the existing brand_physical call.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from engine.builder1_plan_parser import _norm_text


class AdditionalTranslationCost(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEANINGFUL = "MEANINGFUL"


class RecommendedVisualRoute(str, Enum):
    PRODUCT_LED = "PRODUCT_LED"
    PRODUCT_INTEGRATED_ANALOGY = "PRODUCT_INTEGRATED_ANALOGY"
    ANALOGY_LED = "ANALOGY_LED"


DIRECT_PRODUCT_ROUTE_ASSESSMENT_NESTED_FIELDS: tuple[str, ...] = (
    "productOrCategoryImmediatelyReadable",
    "relativeAdvantageDirectlyExpressibleWithProduct",
    "productLedAdvertisingMechanismAvailable",
    "productLedMechanismSummary",
    "externalAnalogyAddsUniquePersuasiveGain",
    "externalAnalogyUniqueGain",
    "additionalTranslationCost",
    "recommendedRoute",
    "routeDecisionReason",
)


def direct_product_route_assessment_json_schema() -> Dict[str, Any]:
    """Canonical strict JSON schema for directProductRouteAssessment (brand_physical)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(DIRECT_PRODUCT_ROUTE_ASSESSMENT_NESTED_FIELDS),
        "properties": {
            "productOrCategoryImmediatelyReadable": {"type": "boolean"},
            "relativeAdvantageDirectlyExpressibleWithProduct": {"type": "boolean"},
            "productLedAdvertisingMechanismAvailable": {"type": "boolean"},
            "productLedMechanismSummary": {"type": "string"},
            "externalAnalogyAddsUniquePersuasiveGain": {"type": "boolean"},
            "externalAnalogyUniqueGain": {"type": "string"},
            "additionalTranslationCost": {
                "type": "string",
                "enum": [item.value for item in AdditionalTranslationCost],
            },
            "recommendedRoute": {
                "type": "string",
                "enum": [item.value for item in RecommendedVisualRoute],
            },
            "routeDecisionReason": {"type": "string"},
        },
    }


BUILDER1_SIMPLE_PRODUCT_DIRECT_ADVANTAGE_PRIORITY = """
SIMPLE PRODUCT / DIRECT ADVANTAGE PRIORITY — mandatory pre-route gate:
Before selecting an external transferred object, compare direct product/category expression against external analogy.

WHEN the product or product category is itself simple, familiar, immediately readable, AND the relative
advantage can be expressed clearly through that product/category:
→ DIRECT product-based creative MUST be considered first.
→ Prefer PRODUCT_LED or, when genuinely useful, PRODUCT_INTEGRATED_ANALOGY before pure external ANALOGY_LED.

External ANALOGY_LED is allowed only when it contributes a meaningful advertising or persuasive mechanism
that the direct product route cannot provide equally well.

This is NOT "always show the product":
- Do NOT default to catalog photography or generic packshots.
- PRODUCT_LED requires a genuine advertising mechanism (transformation, arrangement, comparison, grouping, etc.).
- If the only product-led option is "show several items" with no mechanism, and an external analogy supplies
  a stronger causal proof, ANALOGY_LED may still win.

PUBLIC SIMPLICITY means the audience decodes the ad simply — NOT "use a simple everyday analogy."
When both product domain and analogy domain are equally simple, do not force translation unless the transfer
creates a clearly stronger advertising idea.

Structural similarity alone (different items sharing one container) is NOT sufficient analogy justification.
State the SPECIFIC persuasive capability gained — not "more creative", "interesting", "familiar", or "visually strong".

POPULAR ANALOGY FIRST applies ONLY after ANALOGY_LED is justified — choose familiar analogies over obscure ones
inside the analogy branch; it must NOT push campaigns into analogy by default.
""".strip()


@dataclass(frozen=True)
class DirectProductRouteAssessment:
    product_or_category_immediately_readable: bool
    relative_advantage_directly_expressible_with_product: bool
    product_led_advertising_mechanism_available: bool
    product_led_mechanism_summary: str
    external_analogy_adds_unique_persuasive_gain: bool
    external_analogy_unique_gain: str
    additional_translation_cost: AdditionalTranslationCost
    recommended_route: RecommendedVisualRoute
    route_decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "productOrCategoryImmediatelyReadable": self.product_or_category_immediately_readable,
            "relativeAdvantageDirectlyExpressibleWithProduct": self.relative_advantage_directly_expressible_with_product,
            "productLedAdvertisingMechanismAvailable": self.product_led_advertising_mechanism_available,
            "productLedMechanismSummary": self.product_led_mechanism_summary,
            "externalAnalogyAddsUniquePersuasiveGain": self.external_analogy_adds_unique_persuasive_gain,
            "externalAnalogyUniqueGain": self.external_analogy_unique_gain,
            "additionalTranslationCost": self.additional_translation_cost.value,
            "recommendedRoute": self.recommended_route.value,
            "routeDecisionReason": self.route_decision_reason,
        }


def _normalize_bool(value: object, *, field: str, reasons: List[str]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    reasons.append(f"direct_product_route_{field}_not_boolean")
    return False


def parse_direct_product_route_assessment(raw: object) -> tuple[Optional[DirectProductRouteAssessment], List[str]]:
    reasons: List[str] = []
    if raw is None:
        return None, reasons
    if not isinstance(raw, dict):
        reasons.append("direct_product_route_assessment_not_object")
        return None, reasons

    product_readable = _normalize_bool(
        raw.get("productOrCategoryImmediatelyReadable"),
        field="product_or_category_immediately_readable",
        reasons=reasons,
    )
    advantage_direct = _normalize_bool(
        raw.get("relativeAdvantageDirectlyExpressibleWithProduct"),
        field="relative_advantage_directly_expressible_with_product",
        reasons=reasons,
    )
    mechanism_available = _normalize_bool(
        raw.get("productLedAdvertisingMechanismAvailable"),
        field="product_led_advertising_mechanism_available",
        reasons=reasons,
    )
    unique_gain = _normalize_bool(
        raw.get("externalAnalogyAddsUniquePersuasiveGain"),
        field="external_analogy_adds_unique_persuasive_gain",
        reasons=reasons,
    )

    mechanism_summary = _norm_text(raw.get("productLedMechanismSummary"))
    external_gain = _norm_text(raw.get("externalAnalogyUniqueGain"))
    route_reason = _norm_text(raw.get("routeDecisionReason"))

    cost_raw = _norm_text(raw.get("additionalTranslationCost")).upper()
    try:
        translation_cost = AdditionalTranslationCost(cost_raw or AdditionalTranslationCost.NONE.value)
    except ValueError:
        reasons.append("direct_product_route_invalid_translation_cost")
        translation_cost = AdditionalTranslationCost.NONE

    route_raw = _norm_text(raw.get("recommendedRoute")).upper()
    try:
        recommended_route = RecommendedVisualRoute(route_raw)
    except ValueError:
        reasons.append("direct_product_route_invalid_recommended_route")
        recommended_route = RecommendedVisualRoute.ANALOGY_LED

    if not route_reason:
        reasons.append("direct_product_route_missing_route_decision_reason")
    if mechanism_available and not mechanism_summary:
        reasons.append("direct_product_route_missing_product_led_mechanism_summary")
    if recommended_route == RecommendedVisualRoute.ANALOGY_LED and unique_gain and not external_gain:
        reasons.append("direct_product_route_missing_external_analogy_unique_gain")

    if reasons:
        return None, reasons

    return (
        DirectProductRouteAssessment(
            product_or_category_immediately_readable=product_readable,
            relative_advantage_directly_expressible_with_product=advantage_direct,
            product_led_advertising_mechanism_available=mechanism_available,
            product_led_mechanism_summary=mechanism_summary,
            external_analogy_adds_unique_persuasive_gain=unique_gain,
            external_analogy_unique_gain=external_gain,
            additional_translation_cost=translation_cost,
            recommended_route=recommended_route,
            route_decision_reason=route_reason,
        ),
        [],
    )


def direct_product_route_viable(assessment: DirectProductRouteAssessment) -> bool:
    return (
        assessment.product_or_category_immediately_readable
        and assessment.relative_advantage_directly_expressible_with_product
        and assessment.product_led_advertising_mechanism_available
    )


def validate_direct_product_route_consistency(
    assessment: DirectProductRouteAssessment,
    *,
    physical_generator_is_product: bool,
    physical_generator_is_packaging: bool,
    product_evidence_required: bool,
    visibility_policy: Any,
) -> List[str]:
    from engine.builder1_product_visibility import ProductVisibilityPolicy, policy_prohibits_product_depiction

    reasons: List[str] = []
    policy = visibility_policy
    if isinstance(policy, str):
        try:
            policy = ProductVisibilityPolicy(policy.upper())
        except ValueError:
            policy = ProductVisibilityPolicy.CREATIVE_DECISION

    route = assessment.recommended_route
    viable_direct = direct_product_route_viable(assessment)

    if policy_prohibits_product_depiction(policy):
        if route != RecommendedVisualRoute.ANALOGY_LED:
            reasons.append("physical_route_assessment_inconsistent")
        if physical_generator_is_product or physical_generator_is_packaging:
            reasons.append("physical_route_assessment_inconsistent")
        return list(dict.fromkeys(reasons))

    if physical_generator_is_product and not physical_generator_is_packaging:
        if route != RecommendedVisualRoute.PRODUCT_LED:
            reasons.append("physical_route_assessment_inconsistent")
        return list(dict.fromkeys(reasons))

    if product_evidence_required:
        if route not in {
            RecommendedVisualRoute.PRODUCT_INTEGRATED_ANALOGY,
            RecommendedVisualRoute.ANALOGY_LED,
        }:
            reasons.append("physical_route_assessment_inconsistent")

    if route == RecommendedVisualRoute.PRODUCT_LED:
        if not assessment.product_led_advertising_mechanism_available:
            reasons.append("physical_route_assessment_inconsistent")
        if not physical_generator_is_product:
            reasons.append("physical_route_assessment_inconsistent")

    if route == RecommendedVisualRoute.PRODUCT_INTEGRATED_ANALOGY:
        if not assessment.product_led_advertising_mechanism_available and not product_evidence_required:
            reasons.append("physical_route_assessment_inconsistent")

    if route == RecommendedVisualRoute.ANALOGY_LED:
        if viable_direct and not assessment.external_analogy_adds_unique_persuasive_gain:
            reasons.append("physical_analogy_without_unique_gain")
        if (
            viable_direct
            and not assessment.external_analogy_adds_unique_persuasive_gain
            and assessment.additional_translation_cost == AdditionalTranslationCost.MEANINGFUL
        ):
            reasons.append("physical_unjustified_external_analogy")
        if viable_direct and assessment.additional_translation_cost == AdditionalTranslationCost.MEANINGFUL:
            if not assessment.external_analogy_adds_unique_persuasive_gain:
                reasons.append("physical_unjustified_external_analogy")
        if physical_generator_is_product:
            reasons.append("physical_route_assessment_inconsistent")

    if (
        not physical_generator_is_product
        and not product_evidence_required
        and route in {RecommendedVisualRoute.PRODUCT_LED, RecommendedVisualRoute.PRODUCT_INTEGRATED_ANALOGY}
    ):
        reasons.append("physical_route_assessment_inconsistent")

    return list(dict.fromkeys(reasons))


def resolve_visual_execution_route(
    *,
    physical_generator_is_product: bool = False,
    physical_generator_is_packaging: bool = False,
    product_evidence_required: bool = False,
    direct_product_route_assessment: Optional[DirectProductRouteAssessment] = None,
) -> "VisualExecutionRoute":
    from engine.builder1_product_visibility import VisualExecutionRoute

    if physical_generator_is_product and not physical_generator_is_packaging:
        return VisualExecutionRoute.PRODUCT_LED
    if product_evidence_required:
        return VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY
    if direct_product_route_assessment is not None:
        mapping = {
            RecommendedVisualRoute.PRODUCT_LED: VisualExecutionRoute.PRODUCT_LED,
            RecommendedVisualRoute.PRODUCT_INTEGRATED_ANALOGY: VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY,
            RecommendedVisualRoute.ANALOGY_LED: VisualExecutionRoute.ANALOGY_LED,
        }
        return mapping[direct_product_route_assessment.recommended_route]
    return VisualExecutionRoute.ANALOGY_LED


def assessment_from_planning_internals(internals: Mapping[str, Any]) -> Optional[DirectProductRouteAssessment]:
    raw = internals.get("directProductRouteAssessment")
    if not isinstance(raw, dict):
        return None
    assessment, reasons = parse_direct_product_route_assessment(raw)
    if reasons or assessment is None:
        return None
    return assessment
