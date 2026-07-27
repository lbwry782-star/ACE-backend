"""
Builder2 complete-ad contract — Creator slogans, Judge semantic alignment, Winner preservation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_advertising_closure_contract import (
    GENERIC_SLOGAN_PATTERNS,
    NEW_PROMISE_PATTERNS,
    normalize_advertising_closure,
    validate_slogan_text,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

COMPLETE_AD_CREATOR_FIELDS = ("advertisingClosure", "semanticBridge")
FINAL_DURATION_TOLERANCE_SECONDS = 0.35
SEMANTIC_ALIGNMENT_FIELDS = (
    "visualMeaning",
    "sloganMeaning",
    "combinedAdvertisingMeaning",
    "sameStrategicPromise",
    "sloganCompletesRatherThanChangesVisual",
    "understandableWithoutCreatorReport",
    "keyWordMeaningsConnected",
    "semanticAlignment",
)
DUAL_MEANING_FIELDS = (
    "dualMeaningUsed",
    "physicalMeaningActivatedByVisual",
    "strategicMeaningActivatedBySlogan",
    "meaningsConverge",
)
PROTOTYPE_APPLICATION_ASSESSMENT_FIELDS = (
    "assignedPrototypeId",
    "prototypeMethodVisibleInFilm",
    "prototypeMethodReinforcedBySlogan",
    "applicationFeelsIntrinsic",
    "applicationRequiresRetrospectiveExplanation",
    "prototypeFitScore",
)

HEBREW_GENERIC_JOURNEY = re.compile(r"חלק\s+מהדרך", re.I)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _raise(code: str, *, field: str) -> None:
    raise Builder2TournamentError(f"{code}:{field}")


def build_default_creator_advertising_closure(
    *,
    product_name: str,
    slogan_text: str,
    language: str = "he",
) -> Dict[str, Any]:
    return normalize_advertising_closure(
        {
            "required": True,
            "productNameText": product_name,
            "sloganText": slogan_text,
            "language": language,
            "presentationMode": "end_card",
            "durationSeconds": 2,
            "noLogo": True,
            "headlineSource": "creator_candidate",
        }
    )


def build_default_creator_semantic_bridge(
    *,
    key_word: str,
    visual_meaning: str,
    slogan_meaning: str,
    strategic_meaning: str,
    how_they_meet: str,
    dual_meaning_used: bool = False,
) -> Dict[str, Any]:
    bridge = {
        "keyWordOrConcept": key_word,
        "visualMeaning": visual_meaning,
        "sloganMeaning": slogan_meaning,
        "strategicMeaning": strategic_meaning,
        "howTheMeaningsMeet": how_they_meet,
        "understandableWithoutCreatorReport": True,
    }
    if dual_meaning_used:
        bridge.update(
            {
                "dualMeaningUsed": True,
                "physicalMeaningActivatedByVisual": True,
                "strategicMeaningActivatedBySlogan": True,
                "meaningsConverge": True,
            }
        )
    return bridge


def validate_creator_complete_ad_fields(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    assigned_prototype_id: str = "",
) -> None:
    closure_raw = candidate.get("advertisingClosure")
    if not isinstance(closure_raw, dict):
        _raise("builder2_creator_validation_failed", field="advertisingClosure")
    closure = normalize_advertising_closure(closure_raw)
    if closure.get("required") is not True:
        _raise("builder2_creator_validation_failed", field="advertisingClosure.required")
    product_name = _clean(closure.get("productNameText"))
    if not product_name:
        _raise("builder2_creator_validation_failed", field="advertisingClosure.productNameText")
    slogan = _clean(closure.get("sloganText"))
    relative = ""
    if isinstance(strategy_foundation, dict):
        adv = strategy_foundation.get("relativeAdvantage") or {}
        if isinstance(adv, dict):
            relative = _clean(adv.get("statement"))
    validate_slogan_text(
        slogan=slogan,
        product_name=product_name,
        relative_advantage=relative,
        core_mechanism=_clean(candidate.get("coreCreativeMechanism")),
    )
    if HEBREW_GENERIC_JOURNEY.search(slogan):
        _raise("builder2_creator_validation_failed", field="advertisingClosure.sloganText.generic")

    bridge = candidate.get("semanticBridge")
    if not isinstance(bridge, dict):
        _raise("builder2_creator_validation_failed", field="semanticBridge")
    for key in (
        "keyWordOrConcept",
        "visualMeaning",
        "sloganMeaning",
        "strategicMeaning",
        "howTheMeaningsMeet",
    ):
        if not _clean(bridge.get(key)):
            _raise("builder2_creator_validation_failed", field=f"semanticBridge.{key}")
    if bridge.get("understandableWithoutCreatorReport") is not True:
        _raise("builder2_creator_validation_failed", field="semanticBridge.understandableWithoutCreatorReport")
    if bridge.get("dualMeaningUsed") is True:
        for key in DUAL_MEANING_FIELDS[1:]:
            if bridge.get(key) is not True:
                _raise("builder2_creator_validation_failed", field=f"semanticBridge.{key}")

    candidate["advertisingClosure"] = closure
    candidate["semanticBridge"] = bridge


def validate_judge_semantic_alignment_assessment(judgment: Dict[str, Any]) -> None:
    assessment = judgment.get("semanticAlignmentAssessment")
    if not isinstance(assessment, dict):
        _raise("builder2_judge_validation_failed", field="semanticAlignmentAssessment")
    for key in SEMANTIC_ALIGNMENT_FIELDS:
        if key == "failureReason":
            continue
        value = assessment.get(key)
        if key == "semanticAlignment":
            if not isinstance(value, bool):
                _raise("builder2_judge_validation_failed", field="semanticAlignmentAssessment.semanticAlignment")
            continue
        if key.endswith("Promise") or key.endswith("Visual") or key.endswith("Report") or key.endswith("Connected"):
            if not isinstance(value, bool):
                _raise("builder2_judge_validation_failed", field=f"semanticAlignmentAssessment.{key}")
            continue
        if not _clean(value):
            _raise("builder2_judge_validation_failed", field=f"semanticAlignmentAssessment.{key}")


def validate_judge_prototype_application_assessment(
    judgment: Dict[str, Any],
    *,
    assigned_prototype_id: str = "",
) -> None:
    assessment = judgment.get("prototypeApplicationAssessment")
    if not isinstance(assessment, dict):
        _raise("builder2_judge_validation_failed", field="prototypeApplicationAssessment")
    expected = _clean(assigned_prototype_id or assessment.get("assignedPrototypeId"))
    if expected and _clean(assessment.get("assignedPrototypeId")) != expected:
        _raise("builder2_judge_validation_failed", field="prototypeApplicationAssessment.assignedPrototypeId")
    for key in (
        "prototypeMethodVisibleInFilm",
        "prototypeMethodReinforcedBySlogan",
        "applicationFeelsIntrinsic",
        "applicationRequiresRetrospectiveExplanation",
    ):
        if not isinstance(assessment.get(key), bool):
            _raise("builder2_judge_validation_failed", field=f"prototypeApplicationAssessment.{key}")
    score = assessment.get("prototypeFitScore")
    if score is None:
        score = (judgment.get("scores") or {}).get("prototypeMethodApplication")
    if not isinstance(score, int):
        _raise("builder2_judge_validation_failed", field="prototypeApplicationAssessment.prototypeFitScore")
    if score < 0 or score > 15:
        _raise("builder2_judge_validation_failed", field="prototypeApplicationAssessment.prototypeFitScore")


def apply_semantic_eligibility_rules(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    assessment = out.get("semanticAlignmentAssessment")
    if not isinstance(assessment, dict):
        return out
    if assessment.get("semanticAlignment") is not True:
        out["eligible"] = False
        reason = _clean(assessment.get("failureReason")) or "semantic_alignment_failed"
        disqualifiers = list(out.get("disqualifiers") or [])
        if reason not in disqualifiers:
            disqualifiers.append(reason)
        out["disqualifiers"] = disqualifiers
    elif assessment.get("dualMeaningUsed") is True:
        dual_required = (
            assessment.get("physicalMeaningActivatedByVisual") is True
            and assessment.get("strategicMeaningActivatedBySlogan") is True
            and assessment.get("meaningsConverge") is True
        )
        if not dual_required:
            out["eligible"] = False
            reason = _clean(assessment.get("failureReason")) or "dual_meaning_convergence_failed"
            disqualifiers = list(out.get("disqualifiers") or [])
            if reason not in disqualifiers:
                disqualifiers.append(reason)
            out["disqualifiers"] = disqualifiers
    advertising = out.get("advertisingCompletionAssessment")
    if isinstance(advertising, dict) and out.get("eligible") is True:
        if advertising.get("functionsAsAdvertisement") is False:
            out["eligible"] = False
            out.setdefault("disqualifiers", []).append("advertising_not_complete")
    return out


def validate_winner_slogan_preservation(
    winner_plan: Dict[str, Any],
    *,
    winning_candidate: Dict[str, Any],
) -> None:
    creator_closure = normalize_advertising_closure((winning_candidate or {}).get("advertisingClosure"))
    winner_closure = normalize_advertising_closure(winner_plan.get("advertisingClosure"))
    creator_slogan = _clean(creator_closure.get("sloganText"))
    winner_slogan = _clean(winner_closure.get("sloganText"))
    if not creator_slogan:
        _raise("builder2_winner_slogan_preservation_failed", field="winningCandidate.advertisingClosure.sloganText")
    if winner_slogan != creator_slogan:
        _raise("builder2_winner_slogan_preservation_failed", field="advertisingClosure.sloganText")


def copy_winner_advertising_closure_from_candidate(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    winning_candidate: Dict[str, Any],
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    closure = normalize_advertising_closure((winning_candidate or {}).get("advertisingClosure"))
    closure["headlineSource"] = "creator_candidate"
    state["advertisingClosure"] = closure
    state["advertisingClosureStatus"] = "approved"
    state["advertisingClosureSource"] = "winner_creator_candidate"
    state["winnerSelectedSloganText"] = _clean(closure.get("sloganText"))
    state["winnerSelectedProductNameText"] = _clean(closure.get("productNameText"))
    if isinstance(winning_judgment, dict):
        state["winnerSemanticAlignmentAssessment"] = winning_judgment.get("semanticAlignmentAssessment")
        state["winnerPrototypeApplicationAssessment"] = winning_judgment.get("prototypeApplicationAssessment")
    state["winnerCandidateId"] = candidate_id
    return closure


def validate_final_video_duration(actual_seconds: float, expected_seconds: float = 12.0) -> Tuple[bool, str]:
    delta = abs(float(actual_seconds) - float(expected_seconds))
    if delta <= FINAL_DURATION_TOLERANCE_SECONDS:
        return True, ""
    return False, f"final_duration_mismatch expected={expected_seconds} actual={actual_seconds}"
