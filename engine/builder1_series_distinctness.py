"""
Builder1 series ad execution distinctness — campaign consistency vs ad-level duplication.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CORE_EXECUTION_FIELDS: Tuple[str, ...] = (
    "executionSubject",
    "executionAction",
    "executionObjectState",
    "executionScene",
    "executionPunchline",
)

CONCEPTUAL_EXECUTION_FIELDS: Tuple[str, ...] = (
    "executionSubject",
    "executionAction",
    "executionObjectState",
    "conceptualExecution",
    "conceptualActionProof",
    "newContribution",
    "singleChangedPropertyOrAction",
)

VISUAL_EXECUTION_FIELDS: Tuple[str, ...] = (
    "executionSubject",
    "executionAction",
    "executionObjectState",
    "executionScene",
    "executionPunchline",
    "visualExecution",
    "physicalExecution",
    "sceneDescription",
    "singleChangedPropertyOrAction",
)

CAMPAIGN_LEVEL_FIELDS_EXCLUDED = frozenset(
    {
        "sameVisualLawProof",
        "sloganConnection",
        "relativeAdvantageConnection",
        "brandOwnershipReason",
        "categoryRelevanceReason",
        "familiarExpectation",
        "distinctFromOtherAdsReason",
        "noReuseCheck",
        "immediateClarityReason",
        "headlineNeededReason",
        "headlineReason",
        "variationLabel",
        "marketingText",
        "headline",
    }
)

_PUNCTUATION_RE = re.compile(r"[\.,;:!?\u05BE\u05C0\u05C3\"'`()\[\]{}«»„”“‘’\-–—/\\|]+")
_ASCII_LOWER = str.maketrans(
    {
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "E": "e",
        "F": "f",
        "G": "g",
        "H": "h",
        "I": "i",
        "J": "j",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "O": "o",
        "P": "p",
        "Q": "q",
        "R": "r",
        "S": "s",
        "T": "t",
        "U": "u",
        "V": "v",
        "W": "w",
        "X": "x",
        "Y": "y",
        "Z": "z",
    }
)


@dataclass(frozen=True)
class DuplicateExecutionFinding:
    reason: str
    ad_index_a: int
    ad_index_b: int
    duplicate_type: str
    compared_fields: Tuple[str, ...]
    normalized_values: Tuple[str, ...]
    fingerprint_hash: str
    excluded_campaign_fields: Tuple[str, ...]


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    return " ".join(s.strip().split())


def normalize_execution_text(value: object) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCTUATION_RE.sub(" ", text)
    text = " ".join(text.split())
    return text.translate(_ASCII_LOWER).casefold() if text.isascii() else text


def execution_dimension_values(ad: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for field in fields:
        if field in CAMPAIGN_LEVEL_FIELDS_EXCLUDED:
            continue
        normalized = normalize_execution_text(ad.get(field))
        if normalized:
            values[field] = normalized
    return values


def execution_fingerprint(ad: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    values = execution_dimension_values(ad, fields)
    return tuple(sorted(values.items()))


def fingerprint_hash(fingerprint: Sequence[Tuple[str, str]]) -> str:
    joined = "|".join(f"{field}={value}" for field, value in fingerprint)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def fingerprint_is_degenerate(fingerprint: Sequence[Tuple[str, str]]) -> bool:
    return len(fingerprint) < 2


def count_distinct_dimensions(
    ad_a: Mapping[str, Any],
    ad_b: Mapping[str, Any],
    fields: Sequence[str],
) -> int:
    values_a = execution_dimension_values(ad_a, fields)
    values_b = execution_dimension_values(ad_b, fields)
    shared_fields = set(values_a) | set(values_b)
    differing = 0
    for field in shared_fields:
        left = values_a.get(field, "")
        right = values_b.get(field, "")
        if left and right and left != right:
            differing += 1
        elif (left and not right) or (right and not left):
            differing += 1
    return differing


def _pairwise_indexes(count: int) -> Iterable[Tuple[int, int]]:
    for left in range(count):
        for right in range(left + 1, count):
            yield left, right


def core_execution_identity(ad: Mapping[str, Any]) -> Optional[Tuple[Tuple[str, str], ...]]:
    values = execution_dimension_values(ad, CORE_EXECUTION_FIELDS)
    if len(values) < 3:
        return None
    return tuple(sorted(values.items()))


def validate_ad_execution_distinctness(
    ads: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str = "",
    job_id: str = "",
) -> Tuple[List[str], List[DuplicateExecutionFinding]]:
    reasons: List[str] = []
    findings: List[DuplicateExecutionFinding] = []
    if len(ads) < 2:
        return reasons, findings

    indexed_ads: List[Tuple[int, Mapping[str, Any]]] = []
    for ad in ads:
        if not isinstance(ad, Mapping):
            continue
        try:
            idx = int(ad.get("index"))
        except (TypeError, ValueError):
            continue
        indexed_ads.append((idx, ad))
    indexed_ads.sort(key=lambda item: item[0])

    for left_pos, right_pos in _pairwise_indexes(len(indexed_ads)):
        idx_a, ad_a = indexed_ads[left_pos]
        idx_b, ad_b = indexed_ads[right_pos]

        core_a = core_execution_identity(ad_a)
        core_b = core_execution_identity(ad_b)
        if core_a and core_a == core_b:
            for reason, duplicate_type in (
                ("duplicate_conceptual_execution", "conceptual"),
                ("duplicate_visual_execution", "visual"),
            ):
                if reason in reasons:
                    continue
                reasons.append(reason)
                logger.info(
                    "BUILDER1_SERIES_DUPLICATE_DETECTED campaignId=%s jobId=%s adIndexA=%s adIndexB=%s "
                    "duplicateType=%s reason=%s comparedFields=%s normalizedValues=%s fingerprint=core_identity "
                    "excludedCampaignFields=%s targetedRepairUsed=false",
                    campaign_id or "",
                    job_id or "",
                    idx_a,
                    idx_b,
                    duplicate_type,
                    reason,
                    tuple(field for field, _ in core_a),
                    tuple(value for _, value in core_a),
                    tuple(sorted(CAMPAIGN_LEVEL_FIELDS_EXCLUDED)),
                )
                findings.append(
                    DuplicateExecutionFinding(
                        reason=reason,
                        ad_index_a=idx_a,
                        ad_index_b=idx_b,
                        duplicate_type=duplicate_type,
                        compared_fields=tuple(field for field, _ in core_a),
                        normalized_values=tuple(value for _, value in core_a),
                        fingerprint_hash=fingerprint_hash(core_a),
                        excluded_campaign_fields=tuple(sorted(CAMPAIGN_LEVEL_FIELDS_EXCLUDED)),
                    )
                )

        conceptual_a = execution_fingerprint(ad_a, CONCEPTUAL_EXECUTION_FIELDS)
        conceptual_b = execution_fingerprint(ad_b, CONCEPTUAL_EXECUTION_FIELDS)
        if (
            conceptual_a
            and conceptual_a == conceptual_b
            and not fingerprint_is_degenerate(conceptual_a)
        ):
            reason = "duplicate_conceptual_execution"
            if reason not in reasons:
                reasons.append(reason)
            finding = DuplicateExecutionFinding(
                reason=reason,
                ad_index_a=idx_a,
                ad_index_b=idx_b,
                duplicate_type="conceptual",
                compared_fields=tuple(field for field, _ in conceptual_a),
                normalized_values=tuple(value for _, value in conceptual_a),
                fingerprint_hash=fingerprint_hash(conceptual_a),
                excluded_campaign_fields=tuple(sorted(CAMPAIGN_LEVEL_FIELDS_EXCLUDED)),
            )
            findings.append(finding)
            logger.info(
                "BUILDER1_SERIES_DUPLICATE_DETECTED campaignId=%s jobId=%s adIndexA=%s adIndexB=%s "
                "duplicateType=conceptual reason=%s comparedFields=%s normalizedValues=%s fingerprint=%s "
                "excludedCampaignFields=%s targetedRepairUsed=false",
                campaign_id or "",
                job_id or "",
                idx_a,
                idx_b,
                reason,
                finding.compared_fields,
                finding.normalized_values,
                finding.fingerprint_hash,
                finding.excluded_campaign_fields,
            )

        visual_a = execution_fingerprint(ad_a, VISUAL_EXECUTION_FIELDS)
        visual_b = execution_fingerprint(ad_b, VISUAL_EXECUTION_FIELDS)
        if visual_a and visual_a == visual_b and not fingerprint_is_degenerate(visual_a):
            reason = "duplicate_visual_execution"
            if reason not in reasons:
                reasons.append(reason)
            finding = DuplicateExecutionFinding(
                reason=reason,
                ad_index_a=idx_a,
                ad_index_b=idx_b,
                duplicate_type="visual",
                compared_fields=tuple(field for field, _ in visual_a),
                normalized_values=tuple(value for _, value in visual_a),
                fingerprint_hash=fingerprint_hash(visual_a),
                excluded_campaign_fields=tuple(sorted(CAMPAIGN_LEVEL_FIELDS_EXCLUDED)),
            )
            findings.append(finding)
            logger.info(
                "BUILDER1_SERIES_DUPLICATE_DETECTED campaignId=%s jobId=%s adIndexA=%s adIndexB=%s "
                "duplicateType=visual reason=%s comparedFields=%s normalizedValues=%s fingerprint=%s "
                "excludedCampaignFields=%s targetedRepairUsed=false",
                campaign_id or "",
                job_id or "",
                idx_a,
                idx_b,
                reason,
                finding.compared_fields,
                finding.normalized_values,
                finding.fingerprint_hash,
                finding.excluded_campaign_fields,
            )

        headline_a = _norm_text(ad_a.get("headline")) or None
        headline_b = _norm_text(ad_b.get("headline")) or None
        if (
            conceptual_a == conceptual_b
            and visual_a == visual_b
            and not fingerprint_is_degenerate(visual_a)
            and headline_a != headline_b
            and "headline_only_variation" not in reasons
        ):
            reasons.append("headline_only_variation")

    if not reasons and len(indexed_ads) >= 2:
        pairwise = max(len(indexed_ads) - 1, 1)
        dimensions = count_distinct_dimensions(
            indexed_ads[0][1],
            indexed_ads[1][1],
            tuple(dict.fromkeys(CONCEPTUAL_EXECUTION_FIELDS + VISUAL_EXECUTION_FIELDS)),
        )
        logger.info(
            "BUILDER1_SERIES_DISTINCTNESS_OK campaignId=%s jobId=%s adCount=%s pairwiseComparisons=%s "
            "distinctnessDimensions=%s",
            campaign_id or "",
            job_id or "",
            len(indexed_ads),
            pairwise,
            dimensions,
        )

    return list(dict.fromkeys(reasons)), findings


def duplicate_assembly_reasons(reasons: Sequence[str]) -> List[str]:
    return [
        reason
        for reason in reasons
        if reason.startswith("duplicate_") or reason == "headline_only_variation"
    ]


def duplicated_ad_indexes(findings: Sequence[DuplicateExecutionFinding]) -> List[int]:
    indexes: List[int] = []
    for finding in findings:
        indexes.extend([finding.ad_index_a, finding.ad_index_b])
    return sorted(set(indexes))
