"""
Builder1 slogan candidate quality validation, review, repair, and selection orchestration.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.builder1_client_boundary import validate_brand_physical_boundary_text
from engine.builder1_planning_contract import (
    STAGE_SLOGAN_CANDIDATE_REPAIR_SYSTEM,
    STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM,
    STAGE_SLOGAN_SELECT_SYSTEM,
    build_slogan_candidate_repair_user_prompt,
    build_slogan_quality_review_user_prompt,
    build_slogan_select_user_prompt,
)
from engine.builder1_planner import Builder1PlannerError
from engine.builder1_slogan_stage import (
    SLOGAN_IDS,
    SLOGAN_REJECTION_CODES,
    SloganCandidate,
    SloganCandidateValidationResult,
    SloganSelection,
    parse_slogan_candidate_item,
    parse_slogan_scan,
    parse_slogan_selection,
    slogan_candidate_to_dict,
    validate_selected_slogan,
    validate_slogan_candidate,
)
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

logger = logging.getLogger(__name__)

PlanningModelCaller = Callable[..., object]


def _norm_id(value: object) -> str:
    return str(value or "").strip().upper()


def parse_slogan_quality_review(
    raw_payload: object,
    *,
    expected_ids: List[str],
) -> Dict[str, SloganCandidateValidationResult]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_quality_review", ["slogan_quality_review_not_object"]) from exc

    reviews_raw = obj.get("reviews")
    if not isinstance(reviews_raw, list):
        raise StageParseError("slogan_quality_review", ["slogan_quality_review_reviews_not_list"])

    parsed: Dict[str, SloganCandidateValidationResult] = {}
    seen: Set[str] = set()
    expected = {_norm_id(cid) for cid in expected_ids}

    for item in reviews_raw:
        if not isinstance(item, dict):
            reasons.append("slogan_quality_review_entry_not_object")
            continue
        cid = _norm_id(item.get("candidateId"))
        if cid not in expected:
            reasons.append(f"slogan_quality_review_unknown_id:{cid}")
            continue
        if cid in seen:
            reasons.append(f"slogan_quality_review_duplicate_id:{cid}")
            continue
        seen.add(cid)

        rejection_codes = [
            str(code)
            for code in (item.get("rejectionCodes") or [])
            if str(code).strip() in SLOGAN_REJECTION_CODES
        ]
        eligible_flag = bool(item.get("eligible"))
        semantic_flags = (
            bool(item.get("derivedFromAdvantage")),
            bool(item.get("naturalInLanguage")),
            bool(item.get("credible")),
            bool(item.get("ownable")),
            bool(item.get("impliedActionValid")),
            bool(item.get("campaignGenerative")),
        )
        if not all(semantic_flags) and not rejection_codes:
            if not item.get("derivedFromAdvantage"):
                rejection_codes.append("slogan_not_derived_from_advantage")
            elif not item.get("naturalInLanguage"):
                rejection_codes.append("slogan_not_credible")
            elif not item.get("credible"):
                rejection_codes.append("slogan_not_credible")
            elif not item.get("ownable"):
                rejection_codes.append("slogan_not_ownable")
            elif not item.get("impliedActionValid"):
                rejection_codes.append("slogan_no_implied_action")
            elif not item.get("campaignGenerative"):
                rejection_codes.append("slogan_not_campaign_generative")
        if eligible_flag and rejection_codes:
            reasons.append(f"slogan_quality_review_contradictory:{cid}")
        if not eligible_flag and not rejection_codes:
            reasons.append(f"slogan_quality_review_ineligible_without_codes:{cid}")
        eligible = eligible_flag and not rejection_codes
        parsed[cid] = SloganCandidateValidationResult(
            candidate_id=cid,
            eligible=eligible,
            rejection_codes=list(dict.fromkeys(rejection_codes)),
        )

    if seen != expected:
        missing = sorted(expected - seen)
        for cid in missing:
            reasons.append(f"slogan_quality_review_missing_id:{cid}")

    if reasons:
        raise StageParseError("slogan_quality_review", reasons)
    return parsed


def parse_slogan_candidate_replacements(
    raw_payload: object,
    *,
    allowed_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("slogan_candidate_repair", ["slogan_candidate_repair_not_object"]) from exc

    replacements_raw = obj.get("replacements")
    if not isinstance(replacements_raw, list):
        raise StageParseError("slogan_candidate_repair", ["slogan_candidate_repair_not_list"])

    allowed = {_norm_id(cid) for cid in allowed_ids}
    parsed: Dict[str, Dict[str, Any]] = {}
    seen: Set[str] = set()
    for item in replacements_raw:
        if not isinstance(item, dict):
            reasons.append("slogan_candidate_repair_entry_not_object")
            continue
        cid = _norm_id(item.get("id"))
        if cid not in allowed:
            reasons.append(f"slogan_candidate_repair_unexpected_id:{cid}")
            continue
        if cid in seen:
            reasons.append(f"slogan_candidate_repair_duplicate_id:{cid}")
            continue
        seen.add(cid)
        parsed[cid] = item

    for cid in allowed:
        if cid not in parsed:
            reasons.append(f"slogan_candidate_repair_missing_id:{cid}")

    if reasons:
        raise StageParseError("slogan_candidate_repair", reasons)
    return parsed


def merge_slogan_candidate_replacements(
    candidates: List[SloganCandidate],
    replacements: Dict[str, Dict[str, Any]],
    *,
    preserved: Dict[str, SloganCandidate],
) -> List[SloganCandidate]:
    merged: List[SloganCandidate] = []
    for candidate in candidates:
        cid = candidate.id
        if cid in preserved:
            merged.append(copy.deepcopy(preserved[cid]))
        elif cid in replacements:
            merged.append(parse_slogan_candidate_item(replacements[cid]))
        else:
            merged.append(copy.deepcopy(candidate))
    merged.sort(key=lambda c: c.id)
    return merged


def _combine_candidate_validations(
    deterministic: Dict[str, SloganCandidateValidationResult],
    semantic: Dict[str, SloganCandidateValidationResult],
) -> Dict[str, SloganCandidateValidationResult]:
    combined: Dict[str, SloganCandidateValidationResult] = {}
    for cid in SLOGAN_IDS:
        det = deterministic.get(cid)
        sem = semantic.get(cid)
        codes: List[str] = []
        if det:
            codes.extend(det.rejection_codes)
        if sem:
            codes.extend(sem.rejection_codes)
        codes = list(dict.fromkeys(codes))
        combined[cid] = SloganCandidateValidationResult(
            candidate_id=cid,
            eligible=not codes,
            rejection_codes=codes,
        )
    return combined


def _validate_candidates_deterministic(
    candidates: List[SloganCandidate],
    *,
    relative_advantage: str,
    product_name: str,
    product_description: str,
    detected_language: str,
) -> Dict[str, SloganCandidateValidationResult]:
    results: Dict[str, SloganCandidateValidationResult] = {}
    for candidate in candidates:
        validation = validate_slogan_candidate(
            candidate,
            relative_advantage=relative_advantage,
            product_name=product_name,
            product_description=product_description,
            detected_language=detected_language,
        )
        results[candidate.id] = validation
    return results


def _log_candidate_validation(validations: Dict[str, SloganCandidateValidationResult]) -> None:
    eligible = sum(1 for v in validations.values() if v.eligible)
    invalid = len(validations) - eligible
    logger.info(
        "BUILDER1_SLOGAN_CANDIDATE_VALIDATION total=%s eligible=%s invalid=%s",
        len(validations),
        eligible,
        invalid,
    )
    for validation in validations.values():
        if not validation.eligible:
            for code in validation.rejection_codes:
                logger.info(
                    "BUILDER1_SLOGAN_CANDIDATE_REJECTED candidateId=%s reasonCode=%s",
                    validation.candidate_id,
                    code,
                )


def _run_semantic_quality_review(
    model_caller: PlanningModelCaller,
    *,
    candidates: List[SloganCandidate],
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    product_name: str,
    detected_language: str,
) -> Dict[str, SloganCandidateValidationResult]:
    user_prompt = build_slogan_quality_review_user_prompt(
        strategic_problem=strategic_problem,
        relative_advantage=relative_advantage,
        brief_support=brief_support,
        product_name_resolved=product_name,
        detected_language=detected_language,
        candidates=[slogan_candidate_to_dict(c) for c in candidates],
    )
    raw = model_caller(
        STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM,
        user_prompt,
        stage="slogan_quality_review",
    )
    expected = [c.id for c in candidates]
    if isinstance(raw, dict):
        reviews_raw = raw.get("reviews")
        if isinstance(reviews_raw, list):
            allowed = set(expected)
            raw = {
                **raw,
                "reviews": [
                    item
                    for item in reviews_raw
                    if isinstance(item, dict) and str(item.get("candidateId", "")).strip().upper() in allowed
                ],
            }
    review = parse_slogan_quality_review(
        raw,
        expected_ids=expected,
    )
    return review


def _run_focused_candidate_repair(
    model_caller: PlanningModelCaller,
    *,
    candidates: List[SloganCandidate],
    invalid: Dict[str, List[str]],
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    product_name: str,
    detected_language: str,
) -> Dict[str, Dict[str, Any]]:
    invalid_ids = sorted(invalid.keys())
    logger.info(
        "BUILDER1_SLOGAN_CANDIDATE_REPAIR_START candidateIds=%s",
        ",".join(invalid_ids),
    )
    original_by_id = {c.id: c for c in candidates}
    user_prompt = build_slogan_candidate_repair_user_prompt(
        invalid_candidate_ids=invalid_ids,
        rejection_codes_by_id=invalid,
        strategic_problem=strategic_problem,
        relative_advantage=relative_advantage,
        brief_support=brief_support,
        product_name_resolved=product_name,
        detected_language=detected_language,
        candidates=[slogan_candidate_to_dict(original_by_id[cid]) for cid in invalid_ids if cid in original_by_id],
    )
    raw = model_caller(
        STAGE_SLOGAN_CANDIDATE_REPAIR_SYSTEM,
        user_prompt,
        stage="slogan_candidate_repair",
    )
    replacements = parse_slogan_candidate_replacements(raw, allowed_ids=invalid_ids)
    logger.info("BUILDER1_SLOGAN_CANDIDATE_REPAIR_OK repaired=%s", len(replacements))
    return replacements


def validate_and_prepare_slogan_candidates(
    candidates: List[SloganCandidate],
    *,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    product_name: str,
    product_description: str,
    detected_language: str,
    model_caller: PlanningModelCaller,
    repair_used: bool = False,
    preserved_semantic: Optional[Dict[str, SloganCandidateValidationResult]] = None,
) -> Tuple[List[SloganCandidate], Set[str], bool]:
    """Return updated candidates, eligible ids, and whether repair was attempted."""
    deterministic = _validate_candidates_deterministic(
        candidates,
        relative_advantage=relative_advantage,
        product_name=product_name,
        product_description=product_description,
        detected_language=detected_language,
    )
    if preserved_semantic is None:
        semantic = _run_semantic_quality_review(
            model_caller,
            candidates=candidates,
            strategic_problem=strategic_problem,
            relative_advantage=relative_advantage,
            brief_support=brief_support,
            product_name=product_name,
            detected_language=detected_language,
        )
    else:
        semantic = dict(preserved_semantic)
        review_candidates = [c for c in candidates if c.id not in preserved_semantic]
        if review_candidates:
            partial = _run_semantic_quality_review(
                model_caller,
                candidates=review_candidates,
                strategic_problem=strategic_problem,
                relative_advantage=relative_advantage,
                brief_support=brief_support,
                product_name=product_name,
                detected_language=detected_language,
            )
            semantic.update(partial)
    combined = _combine_candidate_validations(deterministic, semantic)
    _log_candidate_validation(combined)

    invalid = {
        cid: validation.rejection_codes
        for cid, validation in combined.items()
        if not validation.eligible
    }
    if invalid and not repair_used:
        preserved = {
            cid: candidate
            for candidate in candidates
            for cid in (candidate.id,)
            if combined.get(cid) and combined[cid].eligible
        }
        preserved_reviews = {
            cid: validation
            for cid, validation in combined.items()
            if validation.eligible
        }
        try:
            replacements = _run_focused_candidate_repair(
                model_caller,
                candidates=candidates,
                invalid=invalid,
                strategic_problem=strategic_problem,
                relative_advantage=relative_advantage,
                brief_support=brief_support,
                product_name=product_name,
                detected_language=detected_language,
            )
        except StageParseError:
            logger.warning("BUILDER1_SLOGAN_CANDIDATE_REPAIR_FAILED")
            eligible_ids = {cid for cid, validation in combined.items() if validation.eligible}
            return candidates, eligible_ids, True
        merged = merge_slogan_candidate_replacements(
            candidates,
            replacements,
            preserved=preserved,
        )
        merged, eligible_ids, _ = validate_and_prepare_slogan_candidates(
            merged,
            strategic_problem=strategic_problem,
            relative_advantage=relative_advantage,
            brief_support=brief_support,
            product_name=product_name,
            product_description=product_description,
            detected_language=detected_language,
            model_caller=model_caller,
            repair_used=True,
            preserved_semantic=preserved_reviews,
        )
        return merged, eligible_ids, True

    eligible_ids = {cid for cid, validation in combined.items() if validation.eligible}
    return candidates, eligible_ids, repair_used


def _run_slogan_selection_stage(
    run_stage: Callable[..., Any],
    model_caller: PlanningModelCaller,
    *,
    candidates: List[SloganCandidate],
    eligible_ids: Set[str],
) -> Tuple[SloganSelection, SloganCandidate]:
    slogan_dicts = [slogan_candidate_to_dict(c) for c in candidates]
    eligible_list = sorted(eligible_ids)
    logger.info(
        "BUILDER1_SLOGAN_SELECTION_ELIGIBLE candidateIds=%s",
        ",".join(eligible_list),
    )

    def _parse(raw: object):
        return parse_slogan_selection(raw, candidates, eligible_ids=eligible_ids)

    return run_stage(
        "slogan_selection",
        model_caller,
        STAGE_SLOGAN_SELECT_SYSTEM,
        build_slogan_select_user_prompt(slogan_dicts, eligible_candidate_ids=eligible_list),
        _parse,
    )


def run_slogan_selection_with_quality_gate(
    *,
    candidates: List[SloganCandidate],
    eligible_ids: Set[str],
    relative_advantage: str,
    product_description: str,
    model_caller: PlanningModelCaller,
    run_stage: Callable[..., Any],
) -> Tuple[SloganSelection, SloganCandidate, Set[str]]:
    if not eligible_ids:
        raise Builder1PlannerError("slogan_quality_gate_failed")

    remaining = set(eligible_ids)
    selection_retry_used = False
    reselection_used = False
    last_selection: Optional[Tuple[SloganSelection, SloganCandidate]] = None

    while remaining:
        try:
            last_selection = _run_slogan_selection_stage(
                run_stage,
                model_caller,
                candidates=candidates,
                eligible_ids=remaining,
            )
        except Builder1PlannerError as exc:
            if not selection_retry_used and "slogan_selection_ineligible_id" in str(exc):
                selection_retry_used = True
                continue
            raise

        _, selected = last_selection
        gate = validate_selected_slogan(
            selected,
            relative_advantage=relative_advantage,
            product_description=product_description,
        )
        if not gate:
            return last_selection[0], selected, remaining

        logger.info(
            "BUILDER1_SLOGAN_RESELECTION rejectedCandidateId=%s reasons=%s",
            selected.id,
            gate,
        )
        remaining.discard(selected.id)
        if not remaining:
            break
        if reselection_used:
            break
        reselection_used = True

    raise Builder1PlannerError("slogan_quality_gate_failed")


def execute_slogan_scan_through_selection(
    *,
    slogan_candidates: List[SloganCandidate],
    selected_strategy: Any,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    model_caller: PlanningModelCaller,
    run_stage: Callable[..., Any],
    full_rescan_used: bool = False,
) -> Tuple[SloganCandidate, List[SloganCandidate]]:
    candidates, eligible_ids, _ = validate_and_prepare_slogan_candidates(
        slogan_candidates,
        strategic_problem=selected_strategy.strategic_problem,
        relative_advantage=selected_strategy.relative_advantage,
        brief_support=selected_strategy.brief_support,
        product_name=product_name_resolved,
        product_description=product_description,
        detected_language=detected_language,
        model_caller=model_caller,
    )
    if not eligible_ids:
        if full_rescan_used:
            raise Builder1PlannerError("slogan_quality_gate_failed")
        logger.info("BUILDER1_SLOGAN_FULL_RESCAN reason=no_eligible_candidates")
        raise SloganFullRescanRequired()

    _, selected_slogan, _ = run_slogan_selection_with_quality_gate(
        candidates=candidates,
        eligible_ids=eligible_ids,
        relative_advantage=selected_strategy.relative_advantage,
        product_description=product_description,
        model_caller=model_caller,
        run_stage=run_stage,
    )
    return selected_slogan, candidates


class SloganFullRescanRequired(Exception):
    """Signal that one bounded full slogan_scan retry is required."""
