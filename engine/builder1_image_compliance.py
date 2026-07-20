"""
Builder1 post-generation image compliance review — no-logo visual enforcement.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypeAlias

from engine.builder1_compliance_adjudication import (
    AdjudicatedComplianceResult,
    ComplianceEvidenceItem,
    adjudicate_compliance_review,
    log_compliance_findings,
    parse_compliance_evidence,
)
from engine.builder1_methodology_reasons import IMAGE_COMPLIANCE_REASON

logger = logging.getLogger(__name__)

IMAGE_COMPLIANCE_VIOLATION_CODES = frozenset(
    {
        "invented_product_logo",
        "supplied_logo_displayed",
        "logo_like_brand_symbol",
        "packaging_contains_brand_mark",
        "campaign_device_used_as_logo",
        "product_name_rendered_as_logo",
        "product_visible_without_explicit_request",
        "packaging_visible_without_explicit_request",
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
    }
)

IMAGE_COMPLIANCE_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})

BUILDER1_IMAGE_COMPLIANCE_CORRECTION_BLOCK = "\n".join(
    [
        "=== IMAGE COMPLIANCE CORRECTION (MANDATORY) ===",
        "Remove every logo, symbol, icon, emblem, monogram, badge, or brand mark from the image.",
        "Retain only the exact product name as ordinary readable text.",
        "Do not replace the removed mark with another symbol.",
        "Preserve the approved campaign concept, scene, composition, slogan, and graphic system.",
        "=== END IMAGE COMPLIANCE CORRECTION ===",
    ]
)

IMAGE_COMPLIANCE_SYSTEM_PROMPT = f"""
You are a strict Builder1 advertisement image compliance reviewer.

{IMAGE_COMPLIANCE_REASON}

Inspect the generated advertisement image only.

Allowed:
- product or brand name as readable plain text
- ordinary non-brand scene objects
- decorative campaign graphics clearly not attached to product identity
- generic packaging decoration that cannot reasonably be interpreted as a logo
- when explicit secondary product visibility is permitted: an unbranded product may appear small and secondary only

Prohibited:
- invented product logo
- supplied logo
- lightning bolt, leaf, crown, badge, monogram, emblem, icon, or symbol used as product identity
- logo-like mark beside or above the product name
- brand symbol printed on packaging
- campaign graphic device converted into a package logo
- product name stylized into a pictorial logo mark
- when product visibility is forbidden: any depiction of the advertised product, its packaging, container, bottle, can, box, carton, jar, bag, device, or ordinary category unit
- when product visibility is forbidden: product shot, hero product, or product used as the main visual or physical generator
- when secondary visibility is permitted: product dominating the composition, product as the joke, or any logo/mark on the product

Return JSON only:
{{
  "pass": true,
  "hardViolations": [],
  "advisories": [],
  "violations": [],
  "evidence": [],
  "overallConfidence": "high"
}}

When failing, set pass to false and populate hardViolations and/or advisories.
Each evidence item should include when available:
code, symbolDescription, symbolLocation, relationshipToProductName, relationshipToSlogan,
relationshipToBrandText, compactAndIsolated, enclosedAsBadgeOrSeal, repeatedAsBrandSignature,
confidence, evidenceType, location.

Use hardViolations only for objective visible violations with sufficient confidence.
Use advisories for ambiguous resemblance, low-confidence logo/product interpretation,
or possible campaign-device interpretation away from the brand block.

Violation codes for hardViolations or evidence.code:
invented_product_logo, supplied_logo_displayed, logo_like_brand_symbol,
packaging_contains_brand_mark, campaign_device_used_as_logo, product_name_rendered_as_logo,
product_visible_without_explicit_request, packaging_visible_without_explicit_request,
product_used_as_physical_generator, product_used_as_main_visual

Advisory codes may include:
possible_product_resemblance, possible_logo_like_shape,
possible_campaign_device_interpretation, low_confidence_product_identification,
low_confidence_logo_identification

For logo_like_brand_symbol or campaign_device_used_as_logo, include evidence showing
whether the symbol is compact, isolated, emblematic, enclosed, or used as a brand signature
beside Product Name. Do not fail merely because a large campaign object or background motif
is visually distinctive.

For product_used_as_physical_generator, note that the campaign's conceptual generator role
is defined in the structured plan. Only report this when the depicted object clearly matches
the advertised product itself, not merely a campaign metaphor.

Do not use OCR as the primary mechanism. Judge the visible composition.
Fail hard only when the advertised product, product unit, package, or brand mark is actually
depicted contrary to policy with concrete supporting evidence.
""".strip()

_config_logged = False


@dataclass
class ImageComplianceResult:
    passed: bool
    violations: List[str]
    confidence: str = "high"
    hard_violations: List[str] | None = None
    advisories: List[str] | None = None
    evidence: List[object] | None = None
    overall_confidence: str = "high"
    raw_violations: List[str] | None = None

    def __post_init__(self) -> None:
        if self.hard_violations is None:
            self.hard_violations = list(self.violations)
        if self.advisories is None:
            self.advisories = []
        if self.evidence is None:
            self.evidence = []
        if self.raw_violations is None:
            self.raw_violations = list(self.violations) + list(self.advisories or [])
        if not self.overall_confidence:
            self.overall_confidence = self.confidence


class ImageComplianceError(Exception):
    """Visual compliance rejection after review (including post-regeneration)."""

    def __init__(
        self,
        violations: List[str],
        *,
        ad_index: int,
        hard_violations: List[str] | None = None,
        advisories: List[str] | None = None,
    ):
        self.hard_violations = list(hard_violations if hard_violations is not None else violations)
        self.advisories = list(advisories or [])
        self.violations = list(self.hard_violations)
        self.ad_index = ad_index
        joined = ",".join(self.hard_violations)
        super().__init__(f"image_compliance_failed:{joined}")


ComplianceReviewer: TypeAlias = Callable[..., "ImageComplianceResult"]


class ImageComplianceResponseError(ValueError):
    """Malformed reviewer JSON — must never be treated as pass."""


class ImageComplianceUnavailableError(Exception):
    """Review infrastructure unavailable — not a visual logo violation."""

    def __init__(self, reason_code: str, *, ad_index: int):
        self.reason_code = reason_code
        self.ad_index = ad_index
        super().__init__(f"image_compliance_unavailable:{reason_code}")


def compliance_model_name() -> str:
    return (os.environ.get("BUILDER1_IMAGE_COMPLIANCE_MODEL") or "gpt-4o").strip() or "gpt-4o"


def compliance_client_available() -> bool:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return False
    try:
        from openai import OpenAI  # noqa: F401
    except Exception:
        return False
    return True


def log_builder1_image_compliance_config() -> None:
    global _config_logged
    if _config_logged:
        return
    model = compliance_model_name()
    client_available = compliance_client_available()
    logger.info(
        "BUILDER1_IMAGE_COMPLIANCE_CONFIG enabled=true model=%s clientAvailable=%s",
        model,
        client_available,
    )
    _config_logged = True


def _coerce_review_dict(raw_payload: object) -> dict:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ImageComplianceResponseError("compliance_output_not_object")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ImageComplianceResponseError("compliance_output_invalid_json") from exc
        if not isinstance(obj, dict):
            raise ImageComplianceResponseError("compliance_output_not_object")
        return obj
    raise ImageComplianceResponseError("compliance_output_not_object")


def _extract_raw_compliance(data: dict) -> tuple[bool, List[str], List[ComplianceEvidenceItem], str]:
    if "pass" not in data or not isinstance(data["pass"], bool):
        raise ImageComplianceResponseError("pass_not_boolean")
    reviewer_pass = bool(data["pass"])

    hard_raw = data.get("hardViolations")
    advisory_raw = data.get("advisories")
    violations_raw = data.get("violations")
    if not isinstance(hard_raw, list):
        hard_raw = []
    if not isinstance(advisory_raw, list):
        advisory_raw = []
    if not isinstance(violations_raw, list):
        violations_raw = []

    candidate_violations = list(
        dict.fromkeys(
            str(v).strip()
            for v in list(hard_raw) + list(advisory_raw) + list(violations_raw)
            if str(v).strip()
        )
    )
    for code in candidate_violations:
        if code not in IMAGE_COMPLIANCE_VIOLATION_CODES and not code.startswith("possible_") and not code.startswith(
            "low_confidence_"
        ):
            raise ImageComplianceResponseError("invalid_violation_code")

    overall_confidence_raw = data.get("overallConfidence", data.get("confidence"))
    if overall_confidence_raw is None:
        raise ImageComplianceResponseError("confidence_missing")
    overall_confidence = str(overall_confidence_raw).strip().lower()
    if overall_confidence not in IMAGE_COMPLIANCE_CONFIDENCE_VALUES:
        raise ImageComplianceResponseError("invalid_confidence")

    evidence_items = parse_compliance_evidence(data.get("evidence"))
    return reviewer_pass, candidate_violations, evidence_items, overall_confidence


def _result_from_adjudication(
    *,
    adjudicated: AdjudicatedComplianceResult,
    reviewer_pass: bool,
    candidate_violations: List[str],
) -> ImageComplianceResult:
    if reviewer_pass and adjudicated.hard_violations:
        raise ImageComplianceResponseError("pass_true_with_violations")
    if not reviewer_pass and not candidate_violations and not adjudicated.hard_violations:
        raise ImageComplianceResponseError("pass_false_without_violations")
    return ImageComplianceResult(
        passed=adjudicated.passed,
        violations=list(adjudicated.hard_violations),
        confidence=adjudicated.overall_confidence,
        hard_violations=list(adjudicated.hard_violations),
        advisories=list(adjudicated.advisories),
        evidence=list(adjudicated.evidence),
        overall_confidence=adjudicated.overall_confidence,
        raw_violations=list(candidate_violations),
    )


def finalize_compliance_result(
    *,
    reviewer_pass: bool,
    candidate_violations: Sequence[str],
    evidence_items: Sequence[ComplianceEvidenceItem],
    overall_confidence: str,
    series_plan: Optional[object] = None,
    structured_plan_conflict: bool = False,
    preflight_conflict: bool = False,
) -> ImageComplianceResult:
    adjudicated = adjudicate_compliance_review(
        raw_violations=candidate_violations,
        evidence_items=evidence_items,
        overall_confidence=overall_confidence,
        series_plan=series_plan,
        structured_plan_conflict=structured_plan_conflict,
        preflight_conflict=preflight_conflict,
        reviewer_pass=reviewer_pass,
    )
    return _result_from_adjudication(
        adjudicated=adjudicated,
        reviewer_pass=reviewer_pass,
        candidate_violations=list(candidate_violations),
    )


def parse_image_compliance_response(raw_payload: object) -> ImageComplianceResult:
    data = _coerce_review_dict(raw_payload)
    reviewer_pass, candidate_violations, evidence_items, overall_confidence = _extract_raw_compliance(data)
    return finalize_compliance_result(
        reviewer_pass=reviewer_pass,
        candidate_violations=candidate_violations,
        evidence_items=evidence_items,
        overall_confidence=overall_confidence,
    )


def _is_transient_review_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "TimeoutError"}:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in ("429", "rate limit", "rate_limit", "timeout", "temporarily unavailable", "connection")
    )


def _is_unsupported_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in ("model_not_found", "does not exist", "unsupported model", "invalid model")
    )


def _openai_compliance_review_call(
    *,
    image_bytes: bytes,
    product_name: str,
    product_description: str = "",
    visibility_policy: str = "FORBIDDEN",
    transferred_object: str = "",
) -> object:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ImageComplianceUnavailableError("missing_api_key", ad_index=-1)
    try:
        from openai import OpenAI
        import httpx
    except Exception as exc:
        raise ImageComplianceUnavailableError("client_unavailable", ad_index=-1) from exc

    model = compliance_model_name()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(120.0), max_retries=0)
    user_text = (
        f'Product name allowed as plain text only: "{product_name}".\n'
        f"Product visibility policy: {visibility_policy}.\n"
        f"Approved transferred physical generator: {transferred_object or '(see campaign plan)'}.\n"
        f"Product description (for identifying unauthorized product depictions): {product_description}\n"
        "Review this generated Builder1 advertisement image for logo and product-visibility compliance."
    )
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"{IMAGE_COMPLIANCE_SYSTEM_PROMPT}\n\n{user_text}",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        ],
                    }
                ],
            )
            out_text = getattr(response, "output_text", None) or ""
            if not out_text and hasattr(response, "output"):
                parts: List[str] = []
                for item in response.output or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", None) == "output_text":
                            parts.append(getattr(content, "text", "") or "")
                out_text = "".join(parts)
            if not out_text:
                raise ImageComplianceResponseError("compliance_output_empty")
            return out_text
        except ImageComplianceUnavailableError:
            raise
        except ImageComplianceResponseError:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_unsupported_model_error(exc):
                raise ImageComplianceUnavailableError("unsupported_model", ad_index=-1) from exc
            if _is_transient_review_error(exc) and attempt == 1:
                logger.info("BUILDER1_IMAGE_COMPLIANCE_REVIEW_RETRY attempt=2")
                continue
            if _is_transient_review_error(exc):
                raise ImageComplianceUnavailableError("transient_review_failure", ad_index=-1) from exc
            raise ImageComplianceUnavailableError("review_service_error", ad_index=-1) from exc
    raise ImageComplianceUnavailableError("transient_review_failure", ad_index=-1) from last_exc


def review_builder1_ad_image_compliance(
    image_bytes: bytes,
    *,
    product_name: str,
    ad_index: int,
    campaign_id: Optional[str] = None,
    job_id: Optional[str] = None,
    product_description: str = "",
    visibility_policy: str = "FORBIDDEN",
    transferred_object: str = "",
    reviewer: Optional[ComplianceReviewer] = None,
    series_plan: Optional[object] = None,
    plan_revision: int = 1,
    structured_plan_conflict: bool = False,
    preflight_conflict: bool = False,
) -> ImageComplianceResult:
    logger.info(
        "BUILDER1_IMAGE_COMPLIANCE_REVIEW_START campaignId=%s jobId=%s adIndex=%s",
        campaign_id or "",
        job_id or "",
        ad_index,
    )
    try:
        if reviewer is not None:
            raw = reviewer(
                image_bytes=image_bytes,
                product_name=product_name,
                ad_index=ad_index,
                campaign_id=campaign_id,
                job_id=job_id,
            )
            if isinstance(raw, ImageComplianceResult):
                candidate_violations = list(
                    dict.fromkeys(
                        list(raw.raw_violations or [])
                        + list(raw.violations or [])
                        + list(raw.hard_violations or [])
                        + list(raw.advisories or [])
                    )
                )
                evidence_items = [
                    item
                    for item in (raw.evidence or [])
                    if isinstance(item, ComplianceEvidenceItem)
                ]
                result = finalize_compliance_result(
                    reviewer_pass=bool(raw.passed),
                    candidate_violations=candidate_violations,
                    evidence_items=evidence_items,
                    overall_confidence=raw.overall_confidence or raw.confidence,
                    series_plan=series_plan,
                    structured_plan_conflict=structured_plan_conflict,
                    preflight_conflict=preflight_conflict,
                )
            else:
                data = _coerce_review_dict(raw)
                reviewer_pass, candidate_violations, evidence_items, overall_confidence = _extract_raw_compliance(data)
                result = finalize_compliance_result(
                    reviewer_pass=reviewer_pass,
                    candidate_violations=candidate_violations,
                    evidence_items=evidence_items,
                    overall_confidence=overall_confidence,
                    series_plan=series_plan,
                    structured_plan_conflict=structured_plan_conflict,
                    preflight_conflict=preflight_conflict,
                )
        else:
            log_builder1_image_compliance_config()
            raw = _openai_compliance_review_call(
                image_bytes=image_bytes,
                product_name=product_name,
                product_description=product_description,
                visibility_policy=visibility_policy,
                transferred_object=transferred_object,
            )
            data = _coerce_review_dict(raw)
            reviewer_pass, candidate_violations, evidence_items, overall_confidence = _extract_raw_compliance(data)
            result = finalize_compliance_result(
                reviewer_pass=reviewer_pass,
                candidate_violations=candidate_violations,
                evidence_items=evidence_items,
                overall_confidence=overall_confidence,
                series_plan=series_plan,
                structured_plan_conflict=structured_plan_conflict,
                preflight_conflict=preflight_conflict,
            )
        log_compliance_findings(
            campaign_id=campaign_id or "",
            ad_index=ad_index,
            plan_revision=plan_revision,
            result=AdjudicatedComplianceResult(
                passed=result.passed,
                hard_violations=list(result.hard_violations or []),
                advisories=list(result.advisories or []),
                evidence=[item for item in (result.evidence or []) if isinstance(item, ComplianceEvidenceItem)],
                overall_confidence=result.overall_confidence or result.confidence,
                raw_violations=list(result.raw_violations or []),
            ),
        )
    except ImageComplianceUnavailableError as exc:
        reason_code = exc.reason_code
        logger.error(
            "BUILDER1_IMAGE_COMPLIANCE_UNAVAILABLE campaignId=%s jobId=%s adIndex=%s reasonCode=%s",
            campaign_id or "",
            job_id or "",
            ad_index,
            reason_code,
        )
        raise ImageComplianceUnavailableError(reason_code, ad_index=ad_index) from exc
    except ImageComplianceResponseError as exc:
        reason_code = "malformed_response"
        logger.error(
            "BUILDER1_IMAGE_COMPLIANCE_UNAVAILABLE campaignId=%s jobId=%s adIndex=%s reasonCode=%s",
            campaign_id or "",
            job_id or "",
            ad_index,
            reason_code,
        )
        raise ImageComplianceUnavailableError(reason_code, ad_index=ad_index) from exc

    if result.passed:
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_REVIEW_PASS campaignId=%s jobId=%s adIndex=%s confidence=%s",
            campaign_id or "",
            job_id or "",
            ad_index,
            result.confidence,
        )
    else:
        logger.error(
            "BUILDER1_IMAGE_COMPLIANCE_REVIEW_FAIL campaignId=%s jobId=%s adIndex=%s hardViolations=%s advisories=%s",
            campaign_id or "",
            job_id or "",
            ad_index,
            result.hard_violations or [],
            result.advisories or [],
        )
    return result
