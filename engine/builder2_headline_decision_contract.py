"""
Builder2 headline decision contract — canonical use-or-omit decision with optional diagnostic reason.

omit suppresses readable text inside the Runway-generated scene only.
It does not suppress mandatory Advertising Closure after Runway.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

CANONICAL_HEADLINE_DECISIONS = frozenset({"use", "omit"})
HEADLINE_DECISION_ALIASES = {"include": "use"}
VALID_HEADLINE_DECISION_INPUTS = CANONICAL_HEADLINE_DECISIONS | frozenset(HEADLINE_DECISION_ALIASES.keys())
VALID_HEADLINE_REASON_SOURCES = frozenset({"model", "judge", "server_derived", "not_required"})

_TEXTUAL_HEADLINE_DEPENDENCY = re.compile(
    r"\b(read the headline|headline text|on-screen text|title card|caption says|text overlay)\b",
    re.IGNORECASE,
)


def _raise(code: str, *, field: str) -> None:
    raise Builder2TournamentError(f"{code}:{field}")


def normalize_headline_decision_value(raw_decision: Any) -> str:
    text = str(raw_decision or "").strip().lower()
    if not text:
        return ""
    if text in HEADLINE_DECISION_ALIASES:
        return HEADLINE_DECISION_ALIASES[text]
    if text in CANONICAL_HEADLINE_DECISIONS:
        return text
    return text


def headline_decision_requires_headline(decision: Any) -> bool:
    return normalize_headline_decision_value(decision) == "use"


def headline_decision_is_omit(decision: Any) -> bool:
    return normalize_headline_decision_value(decision) == "omit"


def _optional_reason_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def derive_reason_source(
    *,
    reason: Optional[str],
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> str:
    if reason:
        return "model"
    if isinstance(winning_judgment, dict) and isinstance(
        winning_judgment.get("headlineNecessityAssessment"), dict
    ):
        return "judge"
    return "not_required"


def normalize_headline_decision_object(
    raw: Any,
    *,
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(raw, str):
        payload: Dict[str, Any] = {"decision": raw}
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        payload = {}

    decision = normalize_headline_decision_value(payload.get("decision"))
    reason = _optional_reason_text(payload.get("reason"))
    reason_source = str(payload.get("reasonSource") or "").strip()
    if reason_source not in VALID_HEADLINE_REASON_SOURCES:
        reason_source = derive_reason_source(reason=reason, winning_judgment=winning_judgment)
    if not reason and reason_source == "model":
        reason_source = derive_reason_source(reason=reason, winning_judgment=winning_judgment)

    return {
        "decision": decision,
        "reason": reason,
        "reasonSource": reason_source,
    }


def capture_headline_decision_diagnostic(raw: Any) -> Dict[str, Any]:
    existed = raw is not None
    field_type = type(raw).__name__ if raw is not None else "missing"
    keys: list[str] = []
    decision = ""
    reason_exists = False
    reason_type = "missing"
    reason_present = False
    if isinstance(raw, dict):
        keys = sorted(raw.keys())
        decision = normalize_headline_decision_value(raw.get("decision"))
        reason_exists = "reason" in raw
        reason_value = raw.get("reason")
        reason_type = type(reason_value).__name__ if reason_value is not None else "null"
        reason_present = bool(_optional_reason_text(reason_value))
    elif isinstance(raw, str):
        decision = normalize_headline_decision_value(raw)
    return {
        "fieldExisted": existed,
        "fieldType": field_type,
        "keys": keys,
        "normalizedDecision": decision,
        "reasonExisted": reason_exists,
        "reasonType": reason_type,
        "reasonPresent": reason_present,
    }


def apply_headline_decision_execution_normalization(
    plan: Dict[str, Any],
    *,
    headline_decision: Dict[str, Any],
) -> None:
    plan["headlineDecision"] = dict(headline_decision)
    decision = headline_decision.get("decision")
    if headline_decision_is_omit(decision):
        plan["headline"] = ""
        plan["headlineText"] = ""
        plan["headlineTextRemainder"] = ""
        plan["headlineCoreKeyword"] = ""
        plan["advertisingPromise"] = ""
        if plan.get("headlineForm") not in {None, "none", "other"}:
            plan["headlineForm"] = "none"


def _judge_requires_headline(winning_judgment: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not isinstance(winning_judgment, dict):
        return None
    headline = winning_judgment.get("headlineNecessityAssessment")
    if not isinstance(headline, dict):
        return None
    needed = headline.get("headlineNeeded")
    visual_ok = headline.get("visualWouldWorkWithoutHeadline")
    if needed is True and visual_ok is False:
        return True
    if needed is False and visual_ok is True:
        return False
    if needed is False:
        return False
    if needed is True:
        return True
    return None


def judge_requires_separate_headline(winning_judgment: Optional[Dict[str, Any]]) -> Optional[bool]:
    return _judge_requires_headline(winning_judgment)


def judge_requires_verbal_copy(winning_judgment: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(winning_judgment, dict):
        return False
    headline = winning_judgment.get("headlineNecessityAssessment")
    if isinstance(headline, dict) and headline.get("headlineNeeded") is True:
        return True
    verbal = winning_judgment.get("verbalLayerAssessment")
    if isinstance(verbal, dict) and verbal.get("verbalCopyNeeded") is True:
        return True
    return False


def validate_headline_decision_methodology(
    winner_plan: Dict[str, Any],
    *,
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> str:
    raw = winner_plan.get("headlineDecision")
    if raw is None:
        _raise("builder2_winner_validation_failed", field="headlineDecision")
    normalized = normalize_headline_decision_object(raw, winning_judgment=winning_judgment)
    decision = normalized.get("decision") or ""
    if decision not in CANONICAL_HEADLINE_DECISIONS:
        _raise("builder2_winner_validation_failed", field="headlineDecision.decision")

    apply_headline_decision_execution_normalization(winner_plan, headline_decision=normalized)

    headline_form = winner_plan.get("headlineForm")
    if headline_form is not None:
        form = str(headline_form).strip()
        from engine.builder2_methodology_contract import VALID_HEADLINE_FORMS

        if form not in VALID_HEADLINE_FORMS:
            _raise("builder2_winner_validation_failed", field="headlineForm")
        if form == "none" and not headline_decision_is_omit(decision):
            _raise("builder2_winner_validation_failed", field="headlineForm.none_requires_omit")
        if headline_decision_is_omit(decision) and form not in {"none", "other"}:
            _raise("builder2_winner_validation_failed", field="headlineForm.omit_requires_none")

    if headline_decision_is_omit(decision):
        headline = str(winner_plan.get("headline") or "").strip()
        headline_text = str(winner_plan.get("headlineText") or "").strip()
        if winner_plan.get("headlineCompatibilityAlias") is not True and (headline or headline_text):
            _raise("builder2_winner_validation_failed", field="headlineDecision.omit_with_headline")
        video_prompt = str(winner_plan.get("videoPrompt") or winner_plan.get("videoPromptCore") or "")
        if _TEXTUAL_HEADLINE_DEPENDENCY.search(video_prompt):
            _raise("builder2_winner_validation_failed", field="headlineDecision.omit_with_textual_dependency")
        judge_requires = _judge_requires_headline(winning_judgment)
        if judge_requires is True:
            from engine.builder2_single_slogan_contract import (
                canonical_verbal_copy_satisfied_by_slogan,
                is_single_slogan_contract,
                stamp_canonical_copy_judge_mapping,
            )

            if is_single_slogan_contract(plan=winner_plan):
                stamp_canonical_copy_judge_mapping(
                    winner_plan,
                    winning_judgment=winning_judgment,
                )
                if canonical_verbal_copy_satisfied_by_slogan(
                    winner_plan,
                    winning_judgment=winning_judgment,
                ):
                    logger.info(
                        "BUILDER2_HEADLINE_DECISION_OMIT_SATISFIED_BY_SLOGAN decision=omit canonicalCopySatisfiedBy=slogan",
                    )
                else:
                    _raise("builder2_winner_validation_failed", field="headlineDecision.omit_contradicts_judge")
            else:
                _raise("builder2_winner_validation_failed", field="headlineDecision.omit_contradicts_judge")
    elif headline_decision_requires_headline(decision):
        from engine.builder2_tournament_contracts import require_non_empty_str

        require_non_empty_str(winner_plan.get("headline"), field="headline")
    else:
        _raise("builder2_winner_validation_failed", field="headlineDecision.decision")

    diagnostic = capture_headline_decision_diagnostic(raw)
    logger.info(
        "BUILDER2_HEADLINE_DECISION_VALIDATED decision=%s reasonPresent=%s reasonSource=%s",
        decision,
        diagnostic.get("reasonPresent"),
        normalized.get("reasonSource"),
    )
    return decision


def get_normalized_headline_decision(plan: Dict[str, Any]) -> str:
    raw = plan.get("headlineDecision")
    if isinstance(raw, dict):
        return normalize_headline_decision_value(raw.get("decision"))
    if isinstance(raw, str):
        return normalize_headline_decision_value(raw)
    return "omit"
