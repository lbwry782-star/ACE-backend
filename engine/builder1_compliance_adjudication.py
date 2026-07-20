"""
Server-side Builder1 image compliance adjudication — hard vs advisory findings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from engine.builder1_failure_classification import validate_forbidden_plan_visibility
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_product_modality import ProductModality, resolve_product_modality
from engine.builder1_product_visibility import ProductVisibilityPolicy

logger = logging.getLogger(__name__)

ADVISORY_CODES = frozenset(
    {
        "possible_product_resemblance",
        "possible_logo_like_shape",
        "possible_campaign_device_interpretation",
        "low_confidence_product_identification",
        "low_confidence_logo_identification",
    }
)

OBJECTIVE_LOGO_HARD_CODES = frozenset(
    {
        "invented_product_logo",
        "supplied_logo_displayed",
        "packaging_contains_brand_mark",
        "product_name_rendered_as_logo",
    }
)

LOGO_CONTEXT_CODES = frozenset({"logo_like_brand_symbol", "campaign_device_used_as_logo"})

PIXEL_PRODUCT_ROLE_CODES = frozenset(
    {
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
    }
)

PIXEL_PRODUCT_VISIBILITY_CODES = frozenset(
    {
        "product_visible_without_explicit_request",
        "packaging_visible_without_explicit_request",
    }
)

CONCRETE_DIGITAL_PRODUCT_EVIDENCE = frozenset(
    {
        "application_interface",
        "software_screen",
        "device_presenting_product",
        "literal_service_avatar",
        "direct_product_representation",
        "explicit_product_interface",
        "explicit_software_screen",
    }
)

BRAND_SIGNATURE_RELATIONSHIPS = (
    "beside",
    "above",
    "lockup",
    "signature",
    "adjacent",
    "attached",
    "paired with product name",
    "brand block",
)


@dataclass
class ComplianceEvidenceItem:
    code: str
    symbol_description: str = ""
    symbol_location: str = ""
    relationship_to_product_name: str = ""
    relationship_to_slogan: str = ""
    relationship_to_brand_text: str = ""
    compact_and_isolated: bool = False
    enclosed_as_badge_or_seal: bool = False
    repeated_as_brand_signature: bool = False
    confidence: str = "medium"
    evidence_type: str = ""
    location: str = ""


@dataclass
class AdjudicatedComplianceResult:
    passed: bool
    hard_violations: List[str] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)
    evidence: List[ComplianceEvidenceItem] = field(default_factory=list)
    overall_confidence: str = "high"
    raw_violations: List[str] = field(default_factory=list)

    @property
    def violations(self) -> List[str]:
        return list(self.hard_violations)


def _confidence_rank(value: str) -> int:
    normalized = str(value or "medium").strip().lower()
    return {"high": 3, "medium": 2, "low": 1}.get(normalized, 2)


def _normalize_confidence(value: str) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "medium"


def _evidence_for_code(
    evidence_items: Sequence[ComplianceEvidenceItem],
    code: str,
) -> Optional[ComplianceEvidenceItem]:
    for item in evidence_items:
        if item.code == code:
            return item
    return None


def _brand_signature_relationship(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in BRAND_SIGNATURE_RELATIONSHIPS)


def _logo_evidence_supports_hard(item: ComplianceEvidenceItem) -> bool:
    if _confidence_rank(item.confidence) < 2:
        return False
    if item.enclosed_as_badge_or_seal or item.repeated_as_brand_signature:
        return True
    if item.compact_and_isolated and (
        _brand_signature_relationship(item.relationship_to_product_name)
        or _brand_signature_relationship(item.relationship_to_brand_text)
        or _brand_signature_relationship(item.symbol_location)
    ):
        return True
    return False


def _concrete_product_evidence(item: Optional[ComplianceEvidenceItem]) -> bool:
    if item is None:
        return False
    evidence_type = str(item.evidence_type or "").strip().lower()
    if evidence_type in CONCRETE_DIGITAL_PRODUCT_EVIDENCE:
        return _confidence_rank(item.confidence) >= 2
    description = " ".join(
        [
            str(item.symbol_description or ""),
            str(item.location or ""),
            str(item.relationship_to_product_name or ""),
        ]
    ).lower()
    return any(token in description for token in CONCRETE_DIGITAL_PRODUCT_EVIDENCE)


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


def log_compliance_findings(
    *,
    campaign_id: str,
    ad_index: int,
    plan_revision: int,
    result: AdjudicatedComplianceResult,
) -> None:
    for code in result.hard_violations:
        item = _evidence_for_code(result.evidence, code)
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_FINDING campaignId=%s adIndex=%s code=%s severity=hard "
            "confidence=%s evidenceType=%s location=%s relationshipToBrandText=%s planRevision=%s",
            campaign_id or "",
            ad_index,
            code,
            item.confidence if item else result.overall_confidence,
            item.evidence_type if item else "",
            (item.location if item else "") or (item.symbol_location if item else ""),
            (item.relationship_to_brand_text if item else "")
            or (item.relationship_to_product_name if item else ""),
            max(1, int(plan_revision or 1)),
        )
    for code in result.advisories:
        item = _evidence_for_code(result.evidence, code)
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_FINDING campaignId=%s adIndex=%s code=%s severity=advisory "
            "confidence=%s evidenceType=%s location=%s relationshipToBrandText=%s planRevision=%s",
            campaign_id or "",
            ad_index,
            code,
            item.confidence if item else result.overall_confidence,
            item.evidence_type if item else "",
            (item.location if item else "") or (item.symbol_location if item else ""),
            (item.relationship_to_brand_text if item else "")
            or (item.relationship_to_product_name if item else ""),
            max(1, int(plan_revision or 1)),
        )


def adjudicate_compliance_review(
    *,
    raw_violations: Sequence[str],
    evidence_items: Sequence[ComplianceEvidenceItem],
    overall_confidence: str,
    series_plan: Optional[Builder1SeriesPlan] = None,
    structured_plan_conflict: bool = False,
    preflight_conflict: bool = False,
    reviewer_pass: Optional[bool] = None,
) -> AdjudicatedComplianceResult:
    plan = series_plan
    structured_conflict = structured_plan_conflict
    if plan is not None and not structured_conflict:
        structured_conflict = bool(validate_forbidden_plan_visibility(plan))

    policy = _resolve_policy(plan) if plan is not None else ProductVisibilityPolicy.FORBIDDEN
    modality = (
        resolve_product_modality(
            product_name=plan.product_name_resolved if plan else "",
            product_description=plan.product_description if plan else "",
            planning_internals=(plan.planning_internals if plan else None),
        )
        if plan is not None
        else ProductModality.PHYSICAL_PRODUCT
    )

    candidate_codes = list(
        dict.fromkeys(str(code).strip() for code in raw_violations if str(code).strip())
    )
    hard: List[str] = []
    advisories: List[str] = []

    for code in candidate_codes:
        item = _evidence_for_code(evidence_items, code)
        confidence = _normalize_confidence(item.confidence if item else overall_confidence)

        if code in OBJECTIVE_LOGO_HARD_CODES:
            if _confidence_rank(confidence) >= 2:
                hard.append(code)
            else:
                advisories.append("low_confidence_logo_identification")
            continue

        if code in LOGO_CONTEXT_CODES:
            if item and _logo_evidence_supports_hard(item):
                hard.append(code)
            elif _confidence_rank(confidence) < 2:
                advisories.append("low_confidence_logo_identification")
            else:
                advisories.append("possible_logo_like_shape")
            continue

        if code == "product_used_as_physical_generator":
            if structured_conflict or preflight_conflict:
                hard.append(code)
            else:
                advisories.append("possible_product_resemblance")
            continue

        if code == "product_used_as_main_visual":
            if structured_conflict or preflight_conflict:
                hard.append(code)
            elif modality == ProductModality.PHYSICAL_PRODUCT and policy == ProductVisibilityPolicy.FORBIDDEN:
                if _confidence_rank(confidence) >= 2:
                    hard.append(code)
                else:
                    advisories.append("possible_product_resemblance")
            elif _concrete_product_evidence(item):
                hard.append(code)
            else:
                advisories.append("possible_product_resemblance")
            continue

        if code == "product_visible_without_explicit_request":
            if policy != ProductVisibilityPolicy.FORBIDDEN:
                continue
            if modality == ProductModality.PHYSICAL_PRODUCT:
                if _confidence_rank(confidence) >= 2:
                    hard.append(code)
                else:
                    advisories.append("low_confidence_product_identification")
            elif _concrete_product_evidence(item):
                hard.append(code)
            else:
                advisories.append("possible_product_resemblance")
            continue

        if code == "packaging_visible_without_explicit_request":
            if policy != ProductVisibilityPolicy.FORBIDDEN:
                continue
            if modality == ProductModality.PHYSICAL_PRODUCT:
                if _confidence_rank(confidence) >= 2:
                    hard.append(code)
                else:
                    advisories.append("low_confidence_product_identification")
            else:
                advisories.append("possible_product_resemblance")
            continue

        if code in ADVISORY_CODES:
            advisories.append(code)
            continue

        if code in PIXEL_PRODUCT_ROLE_CODES | PIXEL_PRODUCT_VISIBILITY_CODES | LOGO_CONTEXT_CODES:
            advisories.append("possible_product_resemblance")
            continue

        if _confidence_rank(confidence) >= 2:
            hard.append(code)
        else:
            advisories.append(code)

    hard = list(dict.fromkeys(hard))
    advisories = list(dict.fromkeys(advisories))
    passed = len(hard) == 0
    if reviewer_pass is True and not hard:
        passed = True
    elif reviewer_pass is False and hard:
        passed = False

    return AdjudicatedComplianceResult(
        passed=passed,
        hard_violations=hard,
        advisories=advisories,
        evidence=list(evidence_items),
        overall_confidence=_normalize_confidence(overall_confidence),
        raw_violations=candidate_codes,
    )


def parse_compliance_evidence(raw_items: object) -> List[ComplianceEvidenceItem]:
    if not isinstance(raw_items, list):
        return []
    parsed: List[ComplianceEvidenceItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        parsed.append(
            ComplianceEvidenceItem(
                code=str(raw.get("code") or raw.get("violationCode") or "").strip(),
                symbol_description=str(raw.get("symbolDescription") or raw.get("description") or "").strip(),
                symbol_location=str(raw.get("symbolLocation") or raw.get("location") or "").strip(),
                relationship_to_product_name=str(
                    raw.get("relationshipToProductName") or raw.get("relationshipToBrandText") or ""
                ).strip(),
                relationship_to_slogan=str(raw.get("relationshipToSlogan") or "").strip(),
                relationship_to_brand_text=str(raw.get("relationshipToBrandText") or "").strip(),
                compact_and_isolated=bool(raw.get("compactAndIsolated")),
                enclosed_as_badge_or_seal=bool(raw.get("enclosedAsBadgeOrSeal")),
                repeated_as_brand_signature=bool(raw.get("repeatedAsBrandSignature")),
                confidence=_normalize_confidence(str(raw.get("confidence") or "medium")),
                evidence_type=str(raw.get("evidenceType") or raw.get("type") or "").strip(),
                location=str(raw.get("location") or raw.get("symbolLocation") or "").strip(),
            )
        )
    return [item for item in parsed if item.code]
