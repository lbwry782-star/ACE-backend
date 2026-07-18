"""
Builder1 server-owned product visibility policy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProductVisibilityPolicy(str, Enum):
    FORBIDDEN = "FORBIDDEN"
    SECONDARY_EXPLICIT_EXCEPTION = "SECONDARY_EXPLICIT_EXCEPTION"


class ProductVisibilitySource(str, Enum):
    DEFAULT = "default"
    EXPLICIT_USER_REQUEST = "explicit_user_request"


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
    if explicit_product_visibility_requested(
        product_name=product_name,
        product_description=product_description,
        brand_guidelines=brand_guidelines,
    ):
        return ProductVisibilityDecision(
            policy=ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
            source=ProductVisibilitySource.EXPLICIT_USER_REQUEST,
        )
    return ProductVisibilityDecision(
        policy=ProductVisibilityPolicy.FORBIDDEN,
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


def enforce_series_ad_visibility_fields(
    ads: List[Dict[str, Any]],
    *,
    policy: ProductVisibilityPolicy,
) -> List[Dict[str, Any]]:
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
        else:
            ad_copy.setdefault("productVisible", True)
            ad_copy.setdefault("packagingVisible", False)
            ad_copy.setdefault("productIsMainVisual", False)
            ad_copy.setdefault("productIsPhysicalGenerator", False)
        enforced.append(ad_copy)
    return enforced


def build_product_visibility_image_block(
    *,
    policy: ProductVisibilityPolicy,
    transferred_object: str,
    transferred_object_action: str,
    product_name: str,
) -> str:
    positive = "\n".join(
        [
            "=== MAIN VISUAL (TRANSFERRED PHYSICAL GENERATOR) ===",
            f"Primary object: {transferred_object}.",
            f"Primary action: {transferred_object_action}.",
            "This transferred object is the hero visual and carries the campaign idea.",
            "=== END MAIN VISUAL ===",
        ]
    )
    if policy == ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION:
        secondary = "\n".join(
            [
                "=== PRODUCT VISIBILITY (SECONDARY EXCEPTION) ===",
                "The advertised product may appear only as a small secondary contextual element.",
                "It must not dominate the composition, become the joke, or carry any logo or brand mark.",
                "The transferred physical generator remains the main visual.",
                "=== END PRODUCT VISIBILITY ===",
            ]
        )
        return f"{positive}\n{secondary}"
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
