"""
Deterministic advertised-product grounding for Builder1 image compliance.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_product_modality import ProductModality, resolve_product_modality
from engine.builder1_product_visibility import ProductVisibilityPolicy

logger = logging.getLogger(__name__)

PRODUCT_VISIBLE_CODE = "product_visible_without_explicit_request"

PRODUCT_MATCH_BASIS_VALUES = frozenset(
    {
        "exact_product_text",
        "explicit_packaging",
        "explicit_product_shape",
        "supplied_reference_image",
        "explicit_plan_identification",
        "explicit_prompt_identification",
        "none",
    }
)

RELATIONSHIP_TO_ADVERTISED_PRODUCT_VALUES = frozenset(
    {
        "actual_product",
        "explicit_representation",
        "brand_text_only",
        "creative_generator",
        "transferred_object",
        "generic_prop",
        "unrelated_object",
        "uncertain",
        "none",
    }
)

SUPPRESSED_RELATIONSHIPS = frozenset(
    {
        "creative_generator",
        "transferred_object",
        "generic_prop",
        "unrelated_object",
        "uncertain",
        "none",
    }
)

NAMED_PERSON_ALLOWED_BASIS = frozenset(
    {
        "supplied_reference_image",
        "explicit_plan_identification",
        "explicit_prompt_identification",
        "exact_product_text",
    }
)

_PHYSICAL_PATTERNS = re.compile(
    r"\b(shoe|sneaker|bottle|can|box|device|phone|watch|bag|jar|carton|"
    r"packaging|product unit|hardware|appliance|food|drink|cosmetic|tool|"
    r"package|container|tube|pill|tablet|cream|lotion|supplement)\b",
    re.IGNORECASE,
)
_SERVICE_PATTERNS = re.compile(
    r"\b(service|consulting|agency|support|subscription|membership|insurance|"
    r"banking|delivery service|maintenance|training|law firm|clinic|therapy)\b",
    re.IGNORECASE,
)
_ORG_PATTERNS = re.compile(
    r"\b(company|brand|organization|nonprofit|foundation|institution|studio|"
    r"agency|corporation|inc\.?|ltd\.?|llc)\b",
    re.IGNORECASE,
)
_DIGITAL_PATTERNS = re.compile(
    r"\b(app|application|software|platform|saas|api|cloud|digital agent|ai agent|"
    r"virtual assistant|dashboard|browser|online tool|web service)\b",
    re.IGNORECASE,
)
_PLACE_PATTERNS = re.compile(
    r"\b(hotel|restaurant|store|venue|city|resort|campus|location|museum|gallery)\b",
    re.IGNORECASE,
)
_EVENT_PATTERNS = re.compile(
    r"\b(event|festival|conference|concert|exhibition|tournament|summit|webinar)\b",
    re.IGNORECASE,
)
_PORTRAIT_REQUEST_PATTERNS = re.compile(
    r"\b(portrait|likeness|headshot|photo of|photograph of|depiction of|"
    r"image of|face of|picture of)\b",
    re.IGNORECASE,
)


class AdvertisedProductType(str, Enum):
    TANGIBLE_PRODUCT = "tangible_product"
    PACKAGED_PRODUCT = "packaged_product"
    SERVICE = "service"
    BUSINESS_OR_BRAND = "business_or_brand"
    NAMED_PERSON = "named_person"
    PLACE = "place"
    EVENT = "event"
    DIGITAL_PRODUCT = "digital_product"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComplianceProductMatch:
    advertised_product_present: bool = False
    product_match_basis: str = "none"
    matched_visual_element: str = ""
    relationship_to_advertised_product: str = "none"
    product_match_explanation: str = ""


@dataclass(frozen=True)
class ComplianceGroundingContext:
    advertised_product_name: str
    advertised_product_description: str
    advertised_product_type: AdvertisedProductType
    product_visibility_policy: str
    physical_generator: str
    transferred_object: str
    conceptual_generator: str
    intended_visual_objects: List[str]
    product_evidence_required: bool
    product_evidence_reason: str
    named_person_depiction_requested: bool


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokenize(value: str) -> List[str]:
    return [token for token in re.split(r"[\s,;/|]+", value.lower()) if token]


def _looks_like_person_name(name: str) -> bool:
    cleaned = _norm_text(name)
    if not cleaned:
        return False
    tokens = cleaned.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if _PHYSICAL_PATTERNS.search(cleaned):
        return False
    if _SERVICE_PATTERNS.search(cleaned):
        return False
    if _ORG_PATTERNS.search(cleaned):
        return False
    if _DIGITAL_PATTERNS.search(cleaned):
        return False
    if _PLACE_PATTERNS.search(cleaned):
        return False
    if _EVENT_PATTERNS.search(cleaned):
        return False
    return True


def classify_advertised_product_type(
    *,
    product_name: str = "",
    product_description: str = "",
    planning_internals: object = None,
) -> AdvertisedProductType:
    if isinstance(planning_internals, dict):
        raw_type = _norm_text(planning_internals.get("advertisedProductType")).lower()
        if raw_type:
            try:
                return AdvertisedProductType(raw_type)
            except ValueError:
                pass

    modality = resolve_product_modality(
        product_name=product_name,
        product_description=product_description,
        planning_internals=planning_internals,
    )
    combined = f"{product_name} {product_description}".strip()
    lowered = combined.lower()

    if _looks_like_person_name(product_name):
        return AdvertisedProductType.NAMED_PERSON

    if modality == ProductModality.EVENT or _EVENT_PATTERNS.search(lowered):
        return AdvertisedProductType.EVENT
    if modality == ProductModality.PLACE or _PLACE_PATTERNS.search(lowered):
        return AdvertisedProductType.PLACE
    if modality == ProductModality.SERVICE or _SERVICE_PATTERNS.search(lowered):
        return AdvertisedProductType.SERVICE
    if modality == ProductModality.DIGITAL_PRODUCT or _DIGITAL_PATTERNS.search(lowered):
        return AdvertisedProductType.DIGITAL_PRODUCT
    if modality == ProductModality.ORGANIZATION or _ORG_PATTERNS.search(lowered):
        return AdvertisedProductType.BUSINESS_OR_BRAND

    if _PHYSICAL_PATTERNS.search(lowered) and any(
        token in lowered for token in ("packaging", "package", "box", "bottle", "can", "carton", "jar", "bag")
    ):
        return AdvertisedProductType.PACKAGED_PRODUCT
    if modality == ProductModality.PHYSICAL_PRODUCT or _PHYSICAL_PATTERNS.search(lowered):
        return AdvertisedProductType.TANGIBLE_PRODUCT
    return AdvertisedProductType.UNKNOWN


def _resolve_policy(series_plan: Builder1SeriesPlan) -> ProductVisibilityPolicy:
    raw = (series_plan.product_visibility_policy or "").strip().upper()
    try:
        return ProductVisibilityPolicy(raw)
    except ValueError:
        internals = series_plan.planning_internals or {}
        raw = str(internals.get("productVisibilityPolicy") or "FORBIDDEN").strip().upper()
        try:
            return ProductVisibilityPolicy(raw)
        except ValueError:
            return ProductVisibilityPolicy.FORBIDDEN


def _ad_for_index(series_plan: Builder1SeriesPlan, ad_index: int):
    for ad in series_plan.ads or []:
        if int(ad.index) == int(ad_index):
            return ad
    return series_plan.ads[0] if series_plan.ads else None


def named_person_depiction_requested(
    series_plan: Builder1SeriesPlan,
    *,
    ad_index: int = 1,
) -> bool:
    internals = series_plan.planning_internals or {}
    if bool(internals.get("referenceImageSupplied") or internals.get("suppliedReferenceImage")):
        return True

    ad = _ad_for_index(series_plan, ad_index)
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    extra = ad_internals.get(ad.index if ad else ad_index) or ad_internals.get(str(ad.index if ad else ad_index))
    if isinstance(extra, dict) and extra.get("productVisible") is True:
        return True

    if ad is not None:
        execution_text = " ".join(
            [
                ad.physical_execution or "",
                ad.visual_execution or "",
                ad.scene_description or "",
                ad.conceptual_execution or "",
            ]
        )
        name = series_plan.product_name_resolved or series_plan.product_name
        if name and name.lower() in execution_text.lower() and _PORTRAIT_REQUEST_PATTERNS.search(execution_text):
            return True
    return False


def build_compliance_grounding_context(
    series_plan: Builder1SeriesPlan,
    *,
    ad_index: int = 1,
) -> ComplianceGroundingContext:
    internals = series_plan.planning_internals or {}
    advertised_type = classify_advertised_product_type(
        product_name=series_plan.product_name_resolved or series_plan.product_name,
        product_description=series_plan.product_description,
        planning_internals=internals,
    )
    intended: List[str] = []
    for value in (
        series_plan.transferred_object,
        series_plan.physical_generator,
        series_plan.conceptual_generator,
        series_plan.graphic_generator.recurring_graphic_device if series_plan.graphic_generator else "",
    ):
        cleaned = _norm_text(value)
        if cleaned and cleaned not in intended:
            intended.append(cleaned)

    return ComplianceGroundingContext(
        advertised_product_name=series_plan.product_name_resolved or series_plan.product_name,
        advertised_product_description=series_plan.product_description,
        advertised_product_type=advertised_type,
        product_visibility_policy=getattr(_resolve_policy(series_plan), "value", str(series_plan.product_visibility_policy)),
        physical_generator=series_plan.physical_generator,
        transferred_object=series_plan.transferred_object or series_plan.physical_generator,
        conceptual_generator=series_plan.conceptual_generator,
        intended_visual_objects=intended,
        product_evidence_required=bool(internals.get("productEvidenceRequired")),
        product_evidence_reason=_norm_text(internals.get("productEvidenceReason")),
        named_person_depiction_requested=named_person_depiction_requested(series_plan, ad_index=ad_index),
    )


def parse_product_match(raw: object) -> ComplianceProductMatch:
    if not isinstance(raw, dict):
        return ComplianceProductMatch()
    basis = _norm_text(raw.get("productMatchBasis")).lower() or "none"
    if basis not in PRODUCT_MATCH_BASIS_VALUES:
        basis = "none"
    relationship = _norm_text(raw.get("relationshipToAdvertisedProduct")).lower() or "none"
    if relationship not in RELATIONSHIP_TO_ADVERTISED_PRODUCT_VALUES:
        relationship = "none"
    return ComplianceProductMatch(
        advertised_product_present=bool(raw.get("advertisedProductPresent")),
        product_match_basis=basis,
        matched_visual_element=_norm_text(raw.get("matchedVisualElement")),
        relationship_to_advertised_product=relationship,
        product_match_explanation=_norm_text(raw.get("productMatchExplanation")),
    )


def _normalize_tokens(value: str) -> set[str]:
    return set(_tokenize(value))


def matches_creative_plan_object(
    matched_visual_element: str,
    *,
    series_plan: Optional[Builder1SeriesPlan],
) -> bool:
    if not series_plan or not matched_visual_element:
        return False
    matched_tokens = _normalize_tokens(matched_visual_element)
    if not matched_tokens:
        return False
    candidates = [
        series_plan.physical_generator,
        series_plan.transferred_object,
        series_plan.conceptual_generator,
        series_plan.conceptual_generator_result,
        series_plan.conceptual_generator_input,
    ]
    for candidate in candidates:
        cleaned = _norm_text(candidate)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        element_lower = matched_visual_element.lower()
        if lowered in element_lower or element_lower in lowered:
            return True
        candidate_tokens = _normalize_tokens(cleaned)
        if candidate_tokens and matched_tokens & candidate_tokens:
            overlap = len(matched_tokens & candidate_tokens) / max(len(matched_tokens), 1)
            if overlap >= 0.5:
                return True
    return False


def infer_product_match_from_plan(
    *,
    series_plan: Optional[Builder1SeriesPlan],
    matched_visual_element: str,
) -> Optional[ComplianceProductMatch]:
    if not series_plan or not matched_visual_element:
        return None
    if matches_creative_plan_object(matched_visual_element, series_plan=series_plan):
        if _norm_text(series_plan.transferred_object) and _norm_text(matched_visual_element).lower() in (
            series_plan.transferred_object.lower(),
            (series_plan.physical_generator or "").lower(),
        ):
            relationship = "transferred_object"
        elif _norm_text(series_plan.physical_generator) and matched_visual_element.lower() in series_plan.physical_generator.lower():
            relationship = "creative_generator"
        else:
            relationship = "creative_generator"
        return ComplianceProductMatch(
            advertised_product_present=False,
            product_match_basis="none",
            matched_visual_element=matched_visual_element,
            relationship_to_advertised_product=relationship,
            product_match_explanation="Matched element aligns with approved campaign creative object, not advertised product.",
        )
    return None


def _legacy_tangible_inference_allowed(
    *,
    confidence: str,
    advertised_type: AdvertisedProductType,
    series_plan: Optional[Builder1SeriesPlan],
    matched_visual_element: str,
) -> bool:
    if series_plan is not None:
        if matches_creative_plan_object(matched_visual_element, series_plan=series_plan):
            return False
        if advertised_type == AdvertisedProductType.NAMED_PERSON:
            return False
    if confidence not in {"high", "medium"}:
        return False
    return advertised_type in {
        AdvertisedProductType.TANGIBLE_PRODUCT,
        AdvertisedProductType.PACKAGED_PRODUCT,
        AdvertisedProductType.UNKNOWN,
    }


def evaluate_product_visible_hard_support(
    *,
    policy: ProductVisibilityPolicy,
    product_match: ComplianceProductMatch,
    advertised_type: AdvertisedProductType,
    series_plan: Optional[Builder1SeriesPlan],
    confidence: str,
    relationship_to_brand_text: str = "",
    legacy_unstructured: bool = False,
) -> Tuple[bool, str]:
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return False, "policy_allows_visibility"

    if advertised_type == AdvertisedProductType.NAMED_PERSON and series_plan is not None:
        if not named_person_depiction_requested(series_plan):
            if product_match.advertised_product_present and product_match.product_match_basis not in NAMED_PERSON_ALLOWED_BASIS:
                return False, "named_person_ungrounded"
            if product_match.relationship_to_advertised_product in {"actual_product", "explicit_representation"}:
                if product_match.product_match_basis not in NAMED_PERSON_ALLOWED_BASIS:
                    return False, "named_person_ungrounded"

    if matches_creative_plan_object(product_match.matched_visual_element, series_plan=series_plan):
        return False, "creative_generator_not_product"

    if product_match.product_match_basis == "none" or not product_match.advertised_product_present:
        if legacy_unstructured and _legacy_tangible_inference_allowed(
            confidence=confidence,
            advertised_type=advertised_type,
            series_plan=series_plan,
            matched_visual_element=product_match.matched_visual_element,
        ):
            return True, "legacy_tangible_inference"
        return False, "ungrounded_product_match"

    if product_match.relationship_to_advertised_product in SUPPRESSED_RELATIONSHIPS:
        return False, f"relationship_{product_match.relationship_to_advertised_product}"

    if product_match.relationship_to_advertised_product not in {"actual_product", "explicit_representation"}:
        return False, "not_actual_product"

    if not _norm_text(product_match.product_match_explanation):
        return False, "missing_match_explanation"

    return True, ""


def log_false_positive_suppressed(
    *,
    campaign_id: str,
    ad_index: int,
    product_match: ComplianceProductMatch,
    advertised_type: AdvertisedProductType,
    reason: str,
) -> None:
    logger.info(
        "BUILDER1_IMAGE_COMPLIANCE_FALSE_POSITIVE_SUPPRESSED code=%s reason=%s "
        "campaignId=%s adIndex=%s advertisedProductType=%s advertisedProductPresent=%s "
        "productMatchBasis=%s relationshipToAdvertisedProduct=%s matchedVisualElement=%s",
        PRODUCT_VISIBLE_CODE,
        reason,
        campaign_id or "",
        ad_index,
        advertised_type.value,
        str(product_match.advertised_product_present).lower(),
        product_match.product_match_basis,
        product_match.relationship_to_advertised_product,
        product_match.matched_visual_element,
    )


def log_product_match_decision(
    *,
    campaign_id: str,
    ad_index: int,
    product_match: ComplianceProductMatch,
    advertised_type: AdvertisedProductType,
    accepted: bool,
    reason: str = "",
) -> None:
    logger.info(
        "BUILDER1_IMAGE_COMPLIANCE_PRODUCT_MATCH campaignId=%s adIndex=%s accepted=%s reason=%s "
        "advertisedProductType=%s advertisedProductPresent=%s productMatchBasis=%s "
        "relationshipToAdvertisedProduct=%s matchedVisualElement=%s",
        campaign_id or "",
        ad_index,
        str(accepted).lower(),
        reason,
        advertised_type.value,
        str(product_match.advertised_product_present).lower(),
        product_match.product_match_basis,
        product_match.relationship_to_advertised_product,
        product_match.matched_visual_element,
    )


def build_compliance_grounding_user_block(context: ComplianceGroundingContext) -> str:
    intended = ", ".join(context.intended_visual_objects) if context.intended_visual_objects else "(none listed)"
    return "\n".join(
        [
            "=== ADVERTISED PRODUCT GROUNDING (AUTHORITATIVE) ===",
            f'advertisedProductName: "{context.advertised_product_name}"',
            f"advertisedProductDescription: {context.advertised_product_description or '(empty)'}",
            f"advertisedProductType: {context.advertised_product_type.value}",
            f"productVisibilityPolicy: {context.product_visibility_policy}",
            f'physicalGenerator: "{context.physical_generator}"',
            f'transferredObject: "{context.transferred_object}"',
            f'conceptualGenerator: "{context.conceptual_generator}"',
            f"intendedVisualObjects: {intended}",
            f"productEvidenceRequired: {str(context.product_evidence_required).lower()}",
            f"productEvidenceReason: {context.product_evidence_reason or '(none)'}",
            f"namedPersonDepictionRequested: {str(context.named_person_depiction_requested).lower()}",
            "",
            "The physical generator, transferred object, conceptual metaphor, scene prop, and dominant visual object",
            "are NOT the advertised product unless the plan explicitly identifies them as the advertised product.",
            "Before emitting product_visible_without_explicit_request, prove the visible element is specifically",
            "the advertised product using productMatch fields — not merely central, thematic, or metaphorical.",
            "For named_person advertised entities: do not infer identity from appearance alone and do not treat",
            "unrelated objects as a representation of that person unless namedPersonDepictionRequested is true.",
            "=== END ADVERTISED PRODUCT GROUNDING ===",
        ]
    )


def product_match_schema_properties() -> Dict[str, Any]:
    return {
        "advertisedProductPresent": {"type": "boolean"},
        "productMatchBasis": {
            "type": "string",
            "enum": sorted(PRODUCT_MATCH_BASIS_VALUES),
        },
        "matchedVisualElement": {"type": "string"},
        "relationshipToAdvertisedProduct": {
            "type": "string",
            "enum": sorted(RELATIONSHIP_TO_ADVERTISED_PRODUCT_VALUES),
        },
        "productMatchExplanation": {"type": "string"},
    }
