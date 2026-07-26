"""
Builder2 Judge normalization — alias mapping and conditional verbal-layer defaults.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_judge_core_contract import (
    HEADLINE_NEEDED_ALIASES,
    VALID_VERBAL_APPLICABILITY,
    VERBAL_ASSESSMENT_BOOLEAN_FIELDS,
    resolve_creator_verbal_decision,
    resolve_verbal_applicability,
)

_NON_APPLICABLE_NULL_FIELDS = VERBAL_ASSESSMENT_BOOLEAN_FIELDS


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_judge_headline_assessment(out: Dict[str, Any], resolved: List[str]) -> None:
    headline = out.get("headlineNecessityAssessment")
    if not isinstance(headline, dict):
        return
    headline = dict(headline)
    for alias in HEADLINE_NEEDED_ALIASES:
        if alias == "headlineNeeded":
            continue
        if headline.get("headlineNeeded") is None and isinstance(headline.get(alias), bool):
            headline["headlineNeeded"] = headline[alias]
            resolved.append("headlineNecessityAssessment.headlineNeeded")
            break
    out["headlineNecessityAssessment"] = headline


def normalize_judge_verbal_layer(
    out: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]],
    resolved: List[str],
) -> str:
    creator_decision = resolve_creator_verbal_decision(candidate)
    verbal = out.get("verbalLayerAssessment")
    if verbal is None:
        return creator_decision
    if not isinstance(verbal, dict):
        return creator_decision

    verbal = dict(verbal)
    applicability = resolve_verbal_applicability(verbal, creator_verbal_decision=creator_decision)
    if verbal.get("applicability") != applicability:
        verbal["applicability"] = applicability
        resolved.append("verbalLayerAssessment.applicability")

    if applicability in {"not_needed", "not_found"}:
        for key in _NON_APPLICABLE_NULL_FIELDS:
            if key in verbal and verbal.get(key) is not None and not isinstance(verbal.get(key), bool):
                verbal[key] = None
                resolved.append(f"verbalLayerAssessment.{key}")
    out["verbalLayerAssessment"] = verbal
    return applicability


def normalize_judge_candidate(
    raw: Dict[str, Any],
    *,
    candidate_id: str,
    candidate: Optional[Dict[str, Any]] = None,
    base_normalizer: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    if base_normalizer is not None:
        out = base_normalizer(raw, candidate_id=candidate_id)
    else:
        out = dict(raw)
        out["candidateId"] = str(out.get("candidateId") or candidate_id).strip() or candidate_id
        scores = out.get("scores")
        if isinstance(scores, dict):
            cleaned = dict(scores)
            cleaned.pop("total", None)
            cleaned.pop("totalScore", None)
            out["scores"] = cleaned
        out.pop("totalScore", None)
        out.pop("total", None)

    resolved: List[str] = []
    normalize_judge_headline_assessment(out, resolved)
    normalize_judge_verbal_layer(out, candidate=candidate, resolved=resolved)
    return out, resolved
