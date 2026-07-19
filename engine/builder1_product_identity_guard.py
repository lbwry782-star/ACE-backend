"""
Deterministic product-identity guard for FORBIDDEN visibility campaigns.

Distinguishes Product Name as advertising typography from the advertised product
as a depicted visual object.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
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

VISUAL_SUBJECT_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "mainVisual",
        "visualSubject",
        "physicalGenerator",
        "transferredObject",
        "transformedObject",
        "heroObject",
        "foregroundObject",
        "prop",
        "productObject",
        "packagingObject",
    }
)

VISUAL_SUBJECT_AD_FIELDS: tuple[str, ...] = (
    "physical_execution",
    "visual_execution",
    "scene_description",
)

COPY_AND_TYPOGRAPHY_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "productNameResolved",
        "brandSlogan",
        "headline",
        "marketingText",
        "marketing_text",
        "typography",
        "copyLayout",
        "textPlacement",
        "displayedText",
        "sloganConnection",
        "headlineReason",
        "headlineRequired",
        "relativeAdvantageConnection",
        "brandOwnershipReason",
        "categoryRelevanceReason",
        "immediateClarityReason",
        "familiarExpectation",
        "conceptualExecution",
        "conceptual_execution",
        "conceptualActionProof",
        "conceptual_action_proof",
        "headlineNeededReason",
        "headline_needed_reason",
        "newContribution",
        "new_contribution",
        "variationLabel",
        "variation_label",
    }
)

VISUAL_CONFLICT_PREFIXES: tuple[str, ...] = (
    "physical_generator_matches_advertised_product",
    "transferred_object_matches_advertised_product",
    "main_visual_matches_advertised_product",
    "visible_prop_matches_advertised_product",
    "packaging_matches_advertised_product",
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _identity_in_text(text: str, identity: str) -> bool:
    lowered = text.lower()
    if " " in identity:
        return identity in lowered
    return bool(re.search(rf"\b{re.escape(identity)}\b", lowered))


def extract_product_category_identities(*, product_description: str) -> set[str]:
    """Return category-unit terms derived from the product description only."""
    combined = _norm(product_description).lower()
    identities: set[str] = set()
    for phrase in COMPOUND_CATEGORY_PHRASES:
        if phrase in combined:
            identities.add(phrase)
    for term in CATEGORY_UNIT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", combined):
            identities.add(term)
    return identities


def _category_conflicts_for_field(
    *,
    field_name: str,
    field_value: str,
    product_description: str,
) -> list[str]:
    text = _norm(field_value)
    if not text:
        return []
    prefix = {
        "physical_generator": "physical_generator_matches_advertised_product",
        "transferred_object": "transferred_object_matches_advertised_product",
        "physical_execution": "main_visual_matches_advertised_product",
        "visual_execution": "main_visual_matches_advertised_product",
        "scene_description": "main_visual_matches_advertised_product",
        "mainVisual": "main_visual_matches_advertised_product",
        "visualSubject": "main_visual_matches_advertised_product",
        "prop": "visible_prop_matches_advertised_product",
        "packagingObject": "packaging_matches_advertised_product",
    }.get(field_name, "main_visual_matches_advertised_product")
    reasons: list[str] = []
    for identity in sorted(extract_product_category_identities(product_description=product_description)):
        if _identity_in_text(text, identity):
            reasons.append(f"{prefix}:{identity}")
    return reasons


def _exact_visual_object_is_product_name(*, field_value: str, product_name: str) -> bool:
    name = _norm(product_name)
    value = _norm(field_value)
    if len(name) < 4:
        return False
    return value.lower() == name.lower()


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
    if not _norm(physical_generator) and not _norm(transferred_object):
        reasons.append("missing_transferred_object")
        return reasons

    reasons.extend(
        _category_conflicts_for_field(
            field_name="physical_generator",
            field_value=physical_generator,
            product_description=product_description,
        )
    )
    reasons.extend(
        _category_conflicts_for_field(
            field_name="transferred_object",
            field_value=transferred_object,
            product_description=product_description,
        )
    )

    if _exact_visual_object_is_product_name(field_value=physical_generator, product_name=product_name):
        reasons.append("physical_generator_matches_advertised_product:product_name_exact")
    if _exact_visual_object_is_product_name(field_value=transferred_object, product_name=product_name):
        reasons.append("transferred_object_matches_advertised_product:product_name_exact")

    generator = _norm(physical_generator).lower()
    transferred = _norm(transferred_object).lower()
    description = _norm(product_description).lower()
    if generator and description and generator == description:
        reasons.append("physical_generator_matches_advertised_product:equals_product_description")
    if transferred and description and transferred == description:
        reasons.append("transferred_object_matches_advertised_product:equals_product_description")

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


def detect_ad_visual_subject_identity_conflicts(
    *,
    ad: Builder1AdPlan,
    product_description: str,
    ad_internals: Mapping[str, Any] | None = None,
) -> list[str]:
    """Inspect only structured visual-subject fields — never copy/typography prose."""
    reasons: list[str] = []
    for field_name in VISUAL_SUBJECT_AD_FIELDS:
        field_value = getattr(ad, field_name, "")
        reasons.extend(
            _category_conflicts_for_field(
                field_name=field_name,
                field_value=str(field_value or ""),
                product_description=product_description,
            )
        )

    if isinstance(ad_internals, Mapping):
        for key in VISUAL_SUBJECT_FIELD_KEYS:
            if key in COPY_AND_TYPOGRAPHY_FIELD_KEYS:
                continue
            value = ad_internals.get(key)
            if isinstance(value, str) and value.strip():
                reasons.extend(
                    _category_conflicts_for_field(
                        field_name=key,
                        field_value=value,
                        product_description=product_description,
                    )
                )
    return list(dict.fromkeys(reasons))


def detect_series_plan_visual_subject_conflicts(series_plan: Builder1SeriesPlan) -> list[str]:
    policy_raw = (series_plan.product_visibility_policy or "").strip().upper() or "FORBIDDEN"
    try:
        policy = ProductVisibilityPolicy(policy_raw)
    except ValueError:
        policy = ProductVisibilityPolicy.FORBIDDEN
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return []

    reasons = detect_product_identity_conflicts(
        product_name=series_plan.product_name_resolved,
        product_description=series_plan.product_description,
        physical_generator=series_plan.physical_generator,
        transferred_object=series_plan.transferred_object,
        physical_generator_natural_purpose=series_plan.physical_generator_natural_purpose,
        physical_generator_campaign_role=series_plan.physical_generator_campaign_role,
        transferred_object_action=series_plan.transferred_object_action,
        campaign_rationale=series_plan.campaign_rationale,
        visibility_policy=policy,
    )

    internals = series_plan.planning_internals or {}
    ad_internals_map = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    for ad in series_plan.ads:
        extra = {}
        if isinstance(ad_internals_map, dict):
            extra = ad_internals_map.get(ad.index) or ad_internals_map.get(str(ad.index)) or {}
        reasons.extend(
            detect_ad_visual_subject_identity_conflicts(
                ad=ad,
                product_description=series_plan.product_description,
                ad_internals=extra if isinstance(extra, dict) else None,
            )
        )
    return list(dict.fromkeys(reasons))


def is_identity_conflict_reason(code: str) -> bool:
    return any(code.startswith(prefix) for prefix in VISUAL_CONFLICT_PREFIXES) or code == "missing_transferred_object"


def identity_conflict_reasons(codes: Sequence[str]) -> list[str]:
    return [code for code in codes if is_identity_conflict_reason(code)]
