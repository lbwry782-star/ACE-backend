"""Structural parsing for Builder1 consolidated slogan_stage responses."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from engine.builder1_plan_parser import _norm_text
from engine.builder1_slogan_stage import SLOGAN_IDS, SLOGAN_TRANSFER_RISKS, SloganCandidate
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict


def _norm_id(value: object) -> str:
    return _norm_text(value).upper()


def _parse_candidate_item(item: Dict[str, Any]) -> Tuple[SloganCandidate, List[str]]:
    reasons: List[str] = []
    cid = _norm_id(item.get("id"))
    if cid not in SLOGAN_IDS:
        reasons.append("slogan_stage_candidate_ids_invalid")

    brand_slogan = _norm_text(item.get("brandSlogan"))
    derivation = _norm_text(item.get("derivationFromAdvantage"))
    implied_action = _norm_text(item.get("impliedAction"))
    why_ownable = _norm_text(item.get("whyOwnable"))
    why_natural = _norm_text(item.get("whyNaturalInLanguage"))
    transfer_risk = _norm_text(item.get("competitorTransferRisk")).lower()
    generative = _norm_text(item.get("campaignGenerativePower"))

    if transfer_risk not in SLOGAN_TRANSFER_RISKS:
        reasons.append("slogan_stage_invalid_structure")

    candidate = SloganCandidate(
        id=cid,
        brand_slogan=brand_slogan,
        derivation_from_advantage=derivation,
        implied_action=implied_action,
        why_ownable=why_ownable,
        why_natural_in_language=why_natural,
        competitor_transfer_risk=transfer_risk,
        campaign_generative_power=generative,
    )
    return candidate, reasons


def _parse_evaluations(
    raw_payload: object,
    *,
    expected_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_stage", ["slogan_stage_invalid_structure"]) from exc

    evaluations_raw = obj.get("evaluations")
    if not isinstance(evaluations_raw, list):
        raise StageParseError("slogan_stage", ["slogan_stage_evaluations_invalid"])

    expected = {_norm_id(cid) for cid in expected_ids}
    parsed: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    for item in evaluations_raw:
        if not isinstance(item, dict):
            reasons.append("slogan_stage_evaluations_invalid")
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid not in expected:
            reasons.append("slogan_stage_evaluations_invalid")
            continue
        if cid in seen:
            reasons.append("slogan_stage_evaluations_invalid")
            continue
        seen.add(cid)
        rejection_codes = [
            str(code).strip()
            for code in (item.get("rejectionCodes") or [])
            if str(code).strip()
        ]
        eligible_flag = bool(item.get("eligible"))
        if eligible_flag and rejection_codes:
            reasons.append("slogan_stage_evaluations_invalid")
        if not eligible_flag and not rejection_codes:
            reasons.append("slogan_stage_evaluations_invalid")
        parsed[cid] = {
            "eligible": eligible_flag,
            "rejection_codes": rejection_codes,
            "derived_from_advantage": bool(item.get("derivedFromAdvantage")),
            "natural_in_language": bool(item.get("naturalInLanguage")),
            "credible": bool(item.get("credible")),
            "ownable": bool(item.get("ownable")),
            "implied_action_valid": bool(item.get("impliedActionValid")),
            "campaign_generative": bool(item.get("campaignGenerative")),
        }

    if seen != expected:
        reasons.append("slogan_stage_evaluations_invalid")

    if reasons:
        raise StageParseError("slogan_stage", list(dict.fromkeys(reasons)))
    return parsed


def parse_consolidated_slogan_stage_response(
    raw_payload: object,
) -> Tuple[List[SloganCandidate], Dict[str, Dict[str, Any]], str, str]:
    """Parse consolidated slogan_stage JSON with structural validation only."""
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_stage", ["slogan_stage_invalid_structure"]) from exc

    candidates_raw = obj.get("candidates")
    if not isinstance(candidates_raw, list):
        raise StageParseError("slogan_stage", ["slogan_stage_invalid_structure"])
    if any(isinstance(item, str) for item in candidates_raw):
        reasons.append("slogan_stage_invalid_structure")

    parsed_candidates: List[SloganCandidate] = []
    seen_ids: set[str] = set()
    for item in candidates_raw:
        if not isinstance(item, dict):
            reasons.append("slogan_stage_invalid_structure")
            continue
        candidate, item_reasons = _parse_candidate_item(item)
        reasons.extend(item_reasons)
        cid = candidate.id
        if cid in seen_ids:
            reasons.append("slogan_stage_candidate_ids_invalid")
        seen_ids.add(cid)
        parsed_candidates.append(candidate)

    if len(parsed_candidates) != 6:
        reasons.append("slogan_stage_candidate_ids_invalid")
    if seen_ids != set(SLOGAN_IDS):
        reasons.append("slogan_stage_candidate_ids_invalid")

    selected_id = _norm_id(obj.get("selectedCandidateId"))
    if not selected_id:
        raise StageParseError("slogan_stage", ["slogan_stage_selected_candidate_missing"])

    by_id = {candidate.id: candidate for candidate in parsed_candidates}
    if selected_id not in by_id:
        raise StageParseError("slogan_stage", ["slogan_stage_selected_candidate_missing"])

    selected = by_id[selected_id]
    if not selected.brand_slogan.strip():
        raise StageParseError("slogan_stage", ["slogan_stage_selected_slogan_empty"])
    if not selected.implied_action.strip():
        raise StageParseError("slogan_stage", ["slogan_stage_implied_action_empty"])

    for candidate in parsed_candidates:
        if not all(
            [
                candidate.brand_slogan,
                candidate.derivation_from_advantage,
                candidate.implied_action,
                candidate.why_ownable,
                candidate.why_natural_in_language,
                candidate.campaign_generative_power,
            ]
        ):
            reasons.append("slogan_stage_invalid_structure")

    if reasons:
        raise StageParseError("slogan_stage", list(dict.fromkeys(reasons)))

    evaluations = _parse_evaluations(raw_payload, expected_ids=SLOGAN_IDS)

    selected_eval = evaluations.get(selected_id)
    if not selected_eval or not selected_eval.get("eligible"):
        raise StageParseError("slogan_stage", ["slogan_stage_selected_candidate_ineligible"])

    selection_reason = _norm_text(obj.get("selectionReason")) or "Selected by model"
    parsed_candidates.sort(key=lambda candidate: candidate.id)
    return parsed_candidates, evaluations, selected_id, selection_reason
