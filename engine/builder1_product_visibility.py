"""
Builder1 server-owned product visibility policy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


class ProductVisibilityPolicy(str, Enum):
    FORBIDDEN = "FORBIDDEN"
    CREATIVE_DECISION = "CREATIVE_DECISION"
    PRODUCT_VISIBILITY_REQUIRED = "PRODUCT_VISIBILITY_REQUIRED"
    SECONDARY_EXPLICIT_EXCEPTION = "SECONDARY_EXPLICIT_EXCEPTION"


class ProductVisibilitySource(str, Enum):
    DEFAULT = "default"
    EXPLICIT_USER_REQUEST = "explicit_user_request"


class VisualExecutionRoute(str, Enum):
    ANALOGY_LED = "ANALOGY_LED"
    PRODUCT_LED = "PRODUCT_LED"
    PRODUCT_INTEGRATED_ANALOGY = "PRODUCT_INTEGRATED_ANALOGY"


_EXPLICIT_HIDE_PRODUCT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdo\s+not\s+show\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\b(?:don't|do not)\s+(?:show|include|display)\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\b(?:hide|exclude)\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\bno\s+product\s+(?:in\s+)?(?:the\s+)?(?:ad|advertisement|image|visual)\b", re.I),
    re.compile(r"\bproduct\s+(?:must|should)\s+not\s+(?:be\s+)?(?:visible|shown|appear|included|depicted)\b", re.I),
    re.compile(r"\b(?:without|exclude)\s+(?:the\s+)?(?:product|packaging|package)\b", re.I),
    re.compile(r"\b(?:אל|לא)\s+(?:להציג|תציג|כלול|לכלול)\s+(?:את\s+)?(?:המוצר|הבקבוק|האריזה|המכשיר)\b", re.I),
)


_EXPLICIT_SHOW_PRODUCT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bshow\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\binclude\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\bdisplay\s+(?:the\s+)?(?:actual\s+)?product\b", re.I),
    re.compile(r"\bshow\s+(?:the\s+)?(?:product\s+)?(?:bottle|package|packaging|container|can|carton|box|jar|bag|device)\b", re.I),
    re.compile(r"\binclude\s+(?:the\s+)?(?:product\s+)?(?:bottle|package|packaging|container|can|carton|box|jar|bag|device)\b", re.I),
    re.compile(r"\b(?:product|packaging|package|bottle|device)\s+(?:must|should)\s+(?:be\s+)?(?:visible|shown|appear|included)\b", re.I),
    re.compile(r"\b(?:show|include)\s+(?:the\s+)?(?:product\s+)?(?:in\s+)?(?:the\s+)?(?:ad|advertisement|image|visual)\b", re.I),
    re.compile(r"\bproduct\s+shot\b", re.I),
    re.compile(r"\bhero\s+product\b", re.I),
    re.compile(r"\b(?:הצג|להציג|כלול|לכלול)\s+(?:את\s+)?(?:המוצר|הבקבוק|האריזה|המכשיר)\b", re.I),
)


@dataclass(frozen=True)
class ProductVisibilityDecision:
    policy: ProductVisibilityPolicy
    source: ProductVisibilitySource


def resolve_product_visibility_policy(raw: object) -> ProductVisibilityPolicy:
    """Resolve stored policy text. Legacy explicit FORBIDDEN is honored; absent values default to creative decision."""
    text = str(raw or "").strip().upper()
    if text:
        try:
            return ProductVisibilityPolicy(text)
        except ValueError:
            pass
    return ProductVisibilityPolicy.CREATIVE_DECISION


def policy_prohibits_product_depiction(policy: ProductVisibilityPolicy) -> bool:
    return policy == ProductVisibilityPolicy.FORBIDDEN


def policy_allows_creative_visibility_decision(policy: ProductVisibilityPolicy) -> bool:
    return policy == ProductVisibilityPolicy.CREATIVE_DECISION


def policy_requires_product_depiction(policy: ProductVisibilityPolicy) -> bool:
    return policy in {
        ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
        ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
    }


def policy_uses_route_selection(policy: ProductVisibilityPolicy) -> bool:
    return policy in {
        ProductVisibilityPolicy.CREATIVE_DECISION,
        ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
    }


def policy_is_legacy_secondary_only(policy: ProductVisibilityPolicy) -> bool:
    """Stored legacy campaigns that forced secondary-only product presence."""
    return policy == ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION


def _collect_user_text(
    *,
    product_name: str,
    product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [product_description]
    if brand_guidelines:
        for key in ("instructions", "creativeBrief", "brief", "notes", "userInstructions"):
            value = brand_guidelines.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n".join(parts)


def explicit_product_visibility_forbidden(
    *,
    product_name: str,
    product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> bool:
    text = _collect_user_text(
        product_name=product_name,
        product_description=product_description,
        brand_guidelines=brand_guidelines,
    )
    for pattern in _EXPLICIT_HIDE_PRODUCT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def explicit_product_visibility_requested(
    *,
    product_name: str,
    product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> bool:
    text = _collect_user_text(
        product_name=product_name,
        product_description=product_description,
        brand_guidelines=brand_guidelines,
    )
    for pattern in _EXPLICIT_SHOW_PRODUCT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def derive_product_visibility_policy(
    *,
    product_name: str,
    product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> ProductVisibilityDecision:
    if explicit_product_visibility_forbidden(
        product_name=product_name,
        product_description=product_description,
        brand_guidelines=brand_guidelines,
    ):
        return ProductVisibilityDecision(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            source=ProductVisibilitySource.EXPLICIT_USER_REQUEST,
        )
    if explicit_product_visibility_requested(
        product_name=product_name,
        product_description=product_description,
        brand_guidelines=brand_guidelines,
    ):
        return ProductVisibilityDecision(
            policy=ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
            source=ProductVisibilitySource.EXPLICIT_USER_REQUEST,
        )
    return ProductVisibilityDecision(
        policy=ProductVisibilityPolicy.CREATIVE_DECISION,
        source=ProductVisibilitySource.DEFAULT,
    )


def log_builder1_product_visibility_policy(
    *,
    campaign_id: str = "",
    policy: ProductVisibilityPolicy,
    source: ProductVisibilitySource,
) -> None:
    logger.info(
        "BUILDER1_PRODUCT_VISIBILITY_POLICY campaignId=%s mode=%s source=%s",
        campaign_id or "",
        policy.value,
        source.value,
    )


def infer_visual_execution_route(
    *,
    physical_generator_is_product: bool = False,
    physical_generator_is_packaging: bool = False,
    product_evidence_required: bool = False,
) -> VisualExecutionRoute:
    if physical_generator_is_product and not physical_generator_is_packaging:
        return VisualExecutionRoute.PRODUCT_LED
    if product_evidence_required:
        return VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY
    return VisualExecutionRoute.ANALOGY_LED


def _brand_physical_route_kwargs(brand_physical: Any) -> Dict[str, bool]:
    if brand_physical is None:
        return {
            "physical_generator_is_product": False,
            "physical_generator_is_packaging": False,
            "product_evidence_required": False,
        }
    if isinstance(brand_physical, Mapping):
        return {
            "physical_generator_is_product": bool(
                brand_physical.get("physicalGeneratorIsProduct")
                or brand_physical.get("physical_generator_is_product")
            ),
            "physical_generator_is_packaging": bool(
                brand_physical.get("physicalGeneratorIsPackaging")
                or brand_physical.get("physical_generator_is_packaging")
            ),
            "product_evidence_required": bool(
                brand_physical.get("productEvidenceRequired")
                or brand_physical.get("product_evidence_required")
            ),
        }
    return {
        "physical_generator_is_product": bool(getattr(brand_physical, "physical_generator_is_product", False)),
        "physical_generator_is_packaging": bool(getattr(brand_physical, "physical_generator_is_packaging", False)),
        "product_evidence_required": bool(getattr(brand_physical, "product_evidence_required", False)),
    }


def apply_creative_visibility_fields(
    ads: List[Dict[str, Any]],
    *,
    brand_physical: Any = None,
    require_product_visible: bool = False,
) -> List[Dict[str, Any]]:
    route = infer_visual_execution_route(**_brand_physical_route_kwargs(brand_physical))
    enforced: List[Dict[str, Any]] = []
    for ad in ads:
        if not isinstance(ad, dict):
            enforced.append(ad)
            continue
        ad_copy = dict(ad)
        ad_copy.pop("productVisibilityRequired", None)
        ad_copy.pop("productVisibilityReason", None)
        ad_copy.pop("showProduct", None)
        ad_copy.pop("includePackaging", None)
        ad_copy.pop("heroProduct", None)
        ad_copy.pop("productPlacement", None)
        if route == VisualExecutionRoute.PRODUCT_LED:
            ad_copy["productVisible"] = True
            ad_copy["productIsMainVisual"] = True
            ad_copy["productIsPhysicalGenerator"] = True
            ad_copy["packagingVisible"] = False
        elif route == VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY:
            ad_copy["productVisible"] = True
            ad_copy["productIsMainVisual"] = False
            ad_copy["productIsPhysicalGenerator"] = False
            ad_copy["packagingVisible"] = False
        elif require_product_visible:
            ad_copy["productVisible"] = True
            ad_copy["packagingVisible"] = False
            ad_copy.setdefault("productIsMainVisual", False)
            ad_copy.setdefault("productIsPhysicalGenerator", False)
        else:
            ad_copy["productVisible"] = False
            ad_copy["packagingVisible"] = False
            ad_copy["productIsMainVisual"] = False
            ad_copy["productIsPhysicalGenerator"] = False
        enforced.append(ad_copy)
    return enforced


def enforce_series_ad_visibility_fields(
    ads: List[Dict[str, Any]],
    *,
    policy: ProductVisibilityPolicy,
    brand_physical: Any = None,
) -> List[Dict[str, Any]]:
    if policy_uses_route_selection(policy):
        return apply_creative_visibility_fields(
            ads,
            brand_physical=brand_physical,
            require_product_visible=policy == ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
        )
    enforced: List[Dict[str, Any]] = []
    for ad in ads:
        if not isinstance(ad, dict):
            enforced.append(ad)
            continue
        ad_copy = dict(ad)
        ad_copy.pop("productVisibilityRequired", None)
        ad_copy.pop("productVisibilityReason", None)
        ad_copy.pop("showProduct", None)
        ad_copy.pop("includePackaging", None)
        ad_copy.pop("heroProduct", None)
        ad_copy.pop("productPlacement", None)
        if policy == ProductVisibilityPolicy.FORBIDDEN:
            ad_copy["productVisible"] = False
            ad_copy["packagingVisible"] = False
            ad_copy["productIsMainVisual"] = False
            ad_copy["productIsPhysicalGenerator"] = False
        elif policy_is_legacy_secondary_only(policy):
            ad_copy["productVisible"] = True
            ad_copy["packagingVisible"] = False
            ad_copy["productIsMainVisual"] = False
            ad_copy["productIsPhysicalGenerator"] = False
        else:
            ad_copy.setdefault("productVisible", True)
            ad_copy.setdefault("packagingVisible", False)
            ad_copy.setdefault("productIsMainVisual", False)
            ad_copy.setdefault("productIsPhysicalGenerator", False)
        enforced.append(ad_copy)
    return enforced


def _ad_visibility_fields(series_plan: Any, *, ad_index: int) -> Dict[str, Any]:
    internals = getattr(series_plan, "planning_internals", None) or {}
    if isinstance(series_plan, Mapping):
        internals = series_plan.get("planningInternals") or series_plan.get("planning_internals") or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    extra = ad_internals.get(ad_index) or ad_internals.get(str(ad_index)) or {}
    return dict(extra) if isinstance(extra, dict) else {}


def plan_approves_product_visibility(series_plan: Any, *, ad_index: int = 1) -> bool:
    policy = resolve_product_visibility_policy(
        getattr(series_plan, "product_visibility_policy", None)
        if not isinstance(series_plan, Mapping)
        else series_plan.get("productVisibilityPolicy") or series_plan.get("product_visibility_policy")
    )
    if policy_requires_product_depiction(policy):
        return True
    if policy == ProductVisibilityPolicy.CREATIVE_DECISION:
        fields = _ad_visibility_fields(series_plan, ad_index=ad_index)
        return fields.get("productVisible") is True
    return False


def plan_approves_product_as_main_visual(series_plan: Any, *, ad_index: int = 1) -> bool:
    if not plan_approves_product_visibility(series_plan, ad_index=ad_index):
        return False
    fields = _ad_visibility_fields(series_plan, ad_index=ad_index)
    return fields.get("productIsMainVisual") is True


def visual_route_for_plan(series_plan: Any) -> VisualExecutionRoute:
    internals = getattr(series_plan, "planning_internals", None) or {}
    if isinstance(series_plan, Mapping):
        internals = series_plan.get("planningInternals") or series_plan.get("planning_internals") or {}
    raw = str(internals.get("visualExecutionRoute") or "").strip().upper()
    if raw:
        try:
            return VisualExecutionRoute(raw)
        except ValueError:
            pass
    return infer_visual_execution_route(
        physical_generator_is_product=bool(internals.get("physicalGeneratorIsProduct")),
        product_evidence_required=bool(internals.get("productEvidenceRequired")),
    )


def plan_approves_product_as_physical_generator(series_plan: Any, *, ad_index: int = 1) -> bool:
    if not plan_approves_product_visibility(series_plan, ad_index=ad_index):
        return False
    fields = _ad_visibility_fields(series_plan, ad_index=ad_index)
    return fields.get("productIsPhysicalGenerator") is True


def build_product_visibility_image_block(
    *,
    policy: ProductVisibilityPolicy,
    transferred_object: str,
    transferred_object_action: str,
    product_name: str,
    visual_route: VisualExecutionRoute | None = None,
) -> str:
    route = visual_route or VisualExecutionRoute.ANALOGY_LED
    if policy_uses_route_selection(policy) and route == VisualExecutionRoute.PRODUCT_LED:
        return "\n".join(
            [
                "=== MAIN VISUAL (PRODUCT-LED — APPROVED CREATIVE ROUTE) ===",
                f"The advertised product itself is the hero visual and creative mechanism.",
                f"Product action/mechanism: {transferred_object_action}.",
                "Express the relative advantage through the product's own form, property, arrangement, or transformation.",
                "Product Name appears only as plain readable advertising typography — never as an invented logo or packaging mark.",
                "Do not invent logos, emblems, or trademark-like symbols on the product.",
                "=== END MAIN VISUAL ===",
            ]
        )
    if policy_uses_route_selection(policy) and route == VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY:
        return "\n".join(
            [
                "=== MAIN VISUAL (PRODUCT-INTEGRATED ANALOGY — APPROVED) ===",
                f"Primary transferred mechanism: {transferred_object}.",
                f"Primary action: {transferred_object_action}.",
                "The advertised product may appear as a participant in this mechanism — not merely as decoration.",
                "The transferred analogy remains the governing visual law; product supports the idea.",
                "Product Name as plain typography only; no invented logo or packaging brand mark.",
                "=== END MAIN VISUAL ===",
            ]
        )
    positive = "\n".join(
        [
            "=== MAIN VISUAL (TRANSFERRED PHYSICAL GENERATOR) ===",
            f"Primary object: {transferred_object}.",
            f"Primary action: {transferred_object_action}.",
            "This transferred object is the hero visual and carries the campaign idea.",
            "=== END MAIN VISUAL ===",
        ]
    )
    if policy_is_legacy_secondary_only(policy):
        secondary = "\n".join(
            [
                "=== PRODUCT VISIBILITY (LEGACY SECONDARY EXCEPTION) ===",
                "The advertised product may appear only as a small secondary contextual element.",
                "It must not dominate the composition, become the joke, or carry any logo or brand mark.",
                "The transferred physical generator remains the main visual.",
                "=== END PRODUCT VISIBILITY ===",
            ]
        )
        return f"{positive}\n{secondary}"
    if policy == ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED:
        if route == VisualExecutionRoute.ANALOGY_LED:
            required = "\n".join(
                [
                    "=== PRODUCT VISIBILITY (REQUIRED — ANALOGY-LED WITH VISIBLE PRODUCT) ===",
                    "The advertised product must appear in the image.",
                    "Visual hierarchy follows the approved route: the transferred object may remain main visual while the product appears as required participant or secondary element.",
                    f'Product Name "{product_name}" as plain readable advertising typography only.',
                    "=== END PRODUCT VISIBILITY ===",
                ]
            )
        else:
            required = "\n".join(
                [
                    "=== PRODUCT VISIBILITY (REQUIRED — APPROVED ROUTE) ===",
                    "The advertised product must appear in the image as the approved plan specifies.",
                    f'Product Name "{product_name}" as plain readable advertising typography only.',
                    "=== END PRODUCT VISIBILITY ===",
                ]
            )
        return f"{positive}\n{required}"
    if policy == ProductVisibilityPolicy.CREATIVE_DECISION:
        analogy = "\n".join(
            [
                "=== PRODUCT VISIBILITY (ANALOGY-LED — APPROVED) ===",
                "Do not depict the advertised product or its packaging unless the approved plan explicitly integrates it.",
                f'Product Name "{product_name}" as plain readable advertising typography only.',
                "Show the transferred physical generator and its visual action as the hero subject.",
                "=== END PRODUCT VISIBILITY ===",
            ]
        )
        return f"{positive}\n{analogy}"
    forbidden = "\n".join(
        [
            "=== PRODUCT VISIBILITY (FORBIDDEN) ===",
            "Do not depict the advertised product itself.",
            f'Do not depict any package, container, bottle, can, box, carton, jar, bag, device, or ordinary category unit for "{product_name}".',
            "Do not create a product shot or packaging mockup.",
            f'Do not place Product Name "{product_name}" on any object or package.',
            f'Product Name "{product_name}" must appear only as normal readable advertising typography.',
            "Show only the transferred physical generator and its visual action.",
            "Do not invent branding, logos, or packaging marks on any object.",
            "=== END PRODUCT VISIBILITY ===",
        ]
    )
    return f"{positive}\n{forbidden}"


def build_visibility_compliance_correction(violations: List[str]) -> str:
    lines = ["=== IMAGE VISIBILITY COMPLIANCE CORRECTION (MANDATORY) ==="]
    if "product_visible_without_explicit_request" in violations:
        lines.append("Remove the advertised product from the image entirely.")
    if "packaging_visible_without_explicit_request" in violations:
        lines.append("Remove all product packaging, containers, bottles, cans, boxes, cartons, jars, bags, and devices.")
    if "product_used_as_physical_generator" in violations:
        lines.append("Replace the product with the approved transferred physical generator object.")
    if "product_used_as_main_visual" in violations:
        lines.append("Demote any product depiction; the transferred physical generator must be the main visual.")
    lines.append("Preserve the approved campaign concept, scene composition, slogan, and graphic system.")
    lines.append("=== END IMAGE VISIBILITY COMPLIANCE CORRECTION ===")
    return "\n".join(lines)


BUILDER1_NO_PRODUCT_STRICT_CORRECTION_BLOCK = "\n".join(
    [
        "=== IMAGE COMPLIANCE CORRECTION — NO_PRODUCT_STRICT (MANDATORY) ===",
        "Remove the advertised product completely from the image.",
        "Remove every product unit, container, bottle, can, box, carton, jar, bag, device, garment, food item, and vehicle that matches the advertised product category.",
        "Remove all packaging and mock packaging.",
        "Do not reinterpret the advertised product as a prop, background object, label, or partial silhouette.",
        "Show ONLY the approved transferred physical generator as the hero subject.",
        "Preserve Product Name and slogan only as plain readable advertising typography.",
        "Preserve the approved graphic system, palette, and composition.",
        "=== END IMAGE COMPLIANCE CORRECTION — NO_PRODUCT_STRICT ===",
    ]
)


def build_no_product_strict_correction(*, transferred_object: str, transferred_object_action: str) -> str:
    return "\n".join(
        [
            BUILDER1_NO_PRODUCT_STRICT_CORRECTION_BLOCK,
            f"Approved transferred physical generator: {transferred_object}.",
            f"Approved transferred action: {transferred_object_action}.",
        ]
    )


def build_policy_aware_global_image_constraints(
    *,
    policy: ProductVisibilityPolicy,
    visual_route: VisualExecutionRoute | None = None,
) -> str:
    if not policy_prohibits_product_depiction(policy):
        route = visual_route or VisualExecutionRoute.ANALOGY_LED
        if route == VisualExecutionRoute.PRODUCT_LED:
            return "\n".join(
                [
                    "=== GLOBAL IMAGE CONSTRAINTS (PERMANENT — PRODUCT-LED PLAN) ===",
                    "Execute the approved product-led creative mechanism faithfully.",
                    "Product Name appears only as plain readable advertising typography.",
                    "No supplied or invented logo, emblem, badge, seal, monogram, trademark-like symbol, or brand mark.",
                    "The recurring campaign graphic device is compositional only — never a logo.",
                    "Preserve the fixed campaign slogan and approved graphic system.",
                    "=== END GLOBAL IMAGE CONSTRAINTS ===",
                ]
            )
        return "\n".join(
            [
                "=== GLOBAL IMAGE CONSTRAINTS (PERMANENT — CREATIVE VISIBILITY PLAN) ===",
                "Follow the approved plan's product-visibility decision exactly.",
                "If the plan is analogy-led, keep the transferred external object as the dominant main visual.",
                "If the plan integrates the product, show only what the approved mechanism requires.",
                "Product Name as plain typography only; no invented logo or brand mark.",
                "Preserve the fixed campaign slogan and approved graphic system.",
                "=== END GLOBAL IMAGE CONSTRAINTS ===",
            ]
        )
    return "\n".join(
        [
            "=== GLOBAL IMAGE CONSTRAINTS (PERMANENT — ALWAYS APPLY) ===",
            "Advertised product must not be depicted.",
            "Product packaging must not be depicted.",
            "Product Name appears only as plain readable advertising typography.",
            "No supplied or invented logo, emblem, badge, seal, monogram, trademark-like symbol, or brand mark.",
            "The recurring campaign graphic device may appear only as a compositional motif — never as a logo.",
            "The transferred external physical object remains the dominant main visual.",
            "The transferred-object action remains unchanged.",
            "The fixed campaign slogan remains unchanged.",
            "The approved graphic system, palette, and series coherence remain recognizable.",
            "Do not introduce a substitute product object disguised as the transferred generator.",
            "=== END GLOBAL IMAGE CONSTRAINTS ===",
        ]
    )
