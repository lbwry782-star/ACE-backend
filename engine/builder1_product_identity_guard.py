"""
Deterministic product-identity guard for FORBIDDEN visibility campaigns.

Rejects cases where the selected physical generator is the advertised product
or its category unit, even when model-provided booleans claim otherwise.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from engine.builder1_product_visibility import ProductVisibilityPolicy

CATEGORY_UNIT_TERMS: frozenset[str] = frozenset(
    {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "sandal",
        "sandals",
        "bottle",
        "bottles",
        "cup",
        "cups",
        "mug",
        "mugs",
        "glass",
        "glasses",
        "jar",
        "jars",
        "can",
        "cans",
        "carton",
        "cartons",
        "box",
        "boxes",
        "bag",
        "bags",
        "backpack",
        "backpacks",
        "purse",
        "handbag",
        "device",
        "phone",
        "tablet",
        "laptop",
        "watch",
        "watches",
        "car",
        "cars",
        "vehicle",
        "vehicles",
        "truck",
        "trucks",
        "garment",
        "shirt",
        "shirts",
        "jacket",
        "jackets",
        "dress",
        "dresses",
        "pants",
        "jeans",
        "snack",
        "snacks",
        "bar",
        "bars",
        "candy",
        "food",
        "meal",
        "meals",
    }
)

COMPOUND_CATEGORY_PHRASES: tuple[str, ...] = (
    "coffee cup",
    "running shoe",
    "sports bottle",
    "water bottle",
    "protein bar",
    "energy bar",
    "smart phone",
    "cell phone",
)

IDENTITY_CONFLICT_PREFIX = "physical_generator_product_identity"


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z\u0590-\u05FF]{3,}", text.lower()))


def extract_product_category_identities(*, product_name: str, product_description: str) -> set[str]:
    """Return category-unit terms present in the advertised product brief."""
    combined = f"{product_name} {product_description}".lower()
    identities: set[str] = set()
    for phrase in COMPOUND_CATEGORY_PHRASES:
        if phrase in combined:
            identities.add(phrase)
    for term in CATEGORY_UNIT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", combined):
            identities.add(term)
    return identities


def collect_physical_identity_text(
    *,
    physical_generator: str = "",
    transferred_object: str = "",
    physical_generator_natural_purpose: str = "",
    physical_generator_campaign_role: str = "",
    transferred_object_action: str = "",
    campaign_rationale: str = "",
    why_clearer_than_showing_product: str = "",
) -> str:
    return " ".join(
        [
            physical_generator,
            transferred_object,
            physical_generator_natural_purpose,
            physical_generator_campaign_role,
            transferred_object_action,
            campaign_rationale,
            why_clearer_than_showing_product,
        ]
    )


def _identity_in_text(text: str, identity: str) -> bool:
    lowered = text.lower()
    if " " in identity:
        return identity in lowered
    return bool(re.search(rf"\b{re.escape(identity)}\b", lowered))


def detect_product_identity_conflicts(
    *,
    product_name: str,
    product_description: str,
    physical_generator: str,
    transferred_object: str,
    physical_generator_natural_purpose: str = "",
    physical_generator_campaign_role: str = "",
    transferred_object_action: str = "",
    campaign_rationale: str = "",
    why_clearer_than_showing_product: str = "",
    visibility_policy: ProductVisibilityPolicy | str = ProductVisibilityPolicy.FORBIDDEN,
) -> list[str]:
    policy = (
        visibility_policy
        if isinstance(visibility_policy, ProductVisibilityPolicy)
        else ProductVisibilityPolicy(str(visibility_policy or "FORBIDDEN").upper())
    )
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return []

    reasons: list[str] = []
    identity_blob = collect_physical_identity_text(
        physical_generator=physical_generator,
        transferred_object=transferred_object,
        physical_generator_natural_purpose=physical_generator_natural_purpose,
        physical_generator_campaign_role=physical_generator_campaign_role,
        transferred_object_action=transferred_object_action,
        campaign_rationale=campaign_rationale,
        why_clearer_than_showing_product=why_clearer_than_showing_product,
    )
    if not _norm(identity_blob):
        reasons.append(f"{IDENTITY_CONFLICT_PREFIX}:missing_physical_fields")
        return reasons

    for identity in sorted(extract_product_category_identities(product_name=product_name, product_description=product_description)):
        if _identity_in_text(identity_blob, identity):
            reasons.append(f"{IDENTITY_CONFLICT_PREFIX}:{identity}")

    resolved_name = _norm(product_name)
    if len(resolved_name) >= 3 and resolved_name.lower() in identity_blob.lower():
        reasons.append(f"{IDENTITY_CONFLICT_PREFIX}:product_name_match")

    generator = _norm(physical_generator).lower()
    transferred = _norm(transferred_object).lower()
    description = _norm(product_description).lower()
    if generator and description and generator == description:
        reasons.append(f"{IDENTITY_CONFLICT_PREFIX}:generator_equals_product_description")
    if transferred and description and transferred == description:
        reasons.append(f"{IDENTITY_CONFLICT_PREFIX}:transferred_equals_product_description")

    return list(dict.fromkeys(reasons))


def detect_product_identity_conflicts_from_mapping(
    *,
    product_name: str,
    product_description: str,
    physical: Mapping[str, Any],
    visibility_policy: ProductVisibilityPolicy | str = ProductVisibilityPolicy.FORBIDDEN,
) -> list[str]:
    return detect_product_identity_conflicts(
        product_name=product_name,
        product_description=product_description,
        physical_generator=_norm(physical.get("physicalGenerator")),
        transferred_object=_norm(physical.get("transferredObject")),
        physical_generator_natural_purpose=_norm(physical.get("physicalGeneratorNaturalPurpose")),
        physical_generator_campaign_role=_norm(physical.get("physicalGeneratorCampaignRole")),
        transferred_object_action=_norm(physical.get("transferredObjectAction")),
        campaign_rationale=_norm(physical.get("campaignRationale")),
        why_clearer_than_showing_product=_norm(physical.get("whyClearerThanShowingProduct")),
        visibility_policy=visibility_policy,
    )


def ad_execution_mentions_product_category(
    *,
    product_name: str,
    product_description: str,
    execution_text: str,
) -> list[str]:
    reasons: list[str] = []
    blob = _norm(execution_text).lower()
    if not blob:
        return reasons
    for identity in extract_product_category_identities(product_name=product_name, product_description=product_description):
        if _identity_in_text(blob, identity):
            reasons.append(f"ad_execution_product_category:{identity}")
    resolved_name = _norm(product_name)
    if len(resolved_name) >= 3 and resolved_name.lower() in blob:
        reasons.append("ad_execution_product_name_match")
    return reasons


def is_identity_conflict_reason(code: str) -> bool:
    return code.startswith(IDENTITY_CONFLICT_PREFIX)


def identity_conflict_reasons(codes: Sequence[str]) -> list[str]:
    return [code for code in codes if is_identity_conflict_reason(code)]
