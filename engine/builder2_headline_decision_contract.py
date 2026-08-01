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

HEADLINE_OMIT_DEPENDENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bread the headline\b", "headline_read_requirement"),
    (r"\bheadline text\b", "in_video_headline_text"),
    (r"\bon-screen text\b", "on_screen_text"),
    (r"\btitle card\b", "title_card"),
    (r"\bcaption says\b", "caption_says"),
    (r"\btext overlay\b", "text_overlay"),
    (r"\bwritten caption\b", "written_caption"),
    (r"\b(read|reads|reading)\s+(?:the\s+)?(?:sign|label|caption|subtitle)\b", "readable_sign_or_caption"),
    (r"\b(?:sign|label|screen)\s+(?:reads|displays|shows)\b", "sign_or_screen_copy"),
    (
        r"\b(?:display|show|render|burn(?:s|-in)?|superimpose)\s+(?:an?\s+)?(?:headline|caption|subtitle|overlay)\b",
        "render_text_instruction",
    ),
)

HEADLINE_OMIT_EXCLUDED_FIELD_PREFIXES = frozenset(
    {
        "headlineDecision.reason",
        "advertisingClosure",
        "advertisingSloganEvidence",
        "serverPreservationCheck",
        "serverOwnedWinnerSource",
        "winnerPreservationCheck",
        "preservationReference",
    }
)

_PROHIBITION_BEFORE_MATCH = re.compile(
    r"(?:^|[\W_])(?:no|without|avoid|never|not|none|forbidden|prohibited|exclude|ban)\s+(?:any\s+|visible\s+|readable\s+|an?\s+)?$",
    re.IGNORECASE,
)

_REQUIREMENT_BEFORE_MATCH = re.compile(
    r"(?:^|[\W_])(?:must|needs to|required to|have to|should|viewer|audience)\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sequence_stage_text(stage: Any) -> str:
    if isinstance(stage, str):
        return stage.strip()
    if isinstance(stage, dict):
        return _clean(stage.get("description") or stage.get("text"))
    return _clean(stage)


def _visual_anchor_text(anchor: Any) -> str:
    if isinstance(anchor, str):
        return anchor.strip()
    if isinstance(anchor, dict):
        parts = [_clean(anchor.get("description")), _clean(anchor.get("whyEssential"))]
        return " ".join(part for part in parts if part)
    return ""


def collect_headline_omit_runway_execution_field_texts(plan: Dict[str, Any]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for key in ("videoPrompt", "videoPromptCore", "openingFrameDescription", "coreVisualIdea"):
        value = _clean(plan.get(key))
        if value:
            texts.append((key, value))
    sequence = plan.get("sequence")
    if isinstance(sequence, dict):
        for stage_key in ("beginning", "development", "resolution"):
            value = _sequence_stage_text(sequence.get(stage_key))
            if value:
                texts.append((f"sequence.{stage_key}", value))
    anchor = _visual_anchor_text(plan.get("visualAnchor"))
    if anchor:
        texts.append(("visualAnchor", anchor))
    return texts


def _dependency_match_is_prohibition_only(text: str, match: re.Match[str]) -> bool:
    start = match.start()
    before = text[max(0, start - 64): start]
    if _PROHIBITION_BEFORE_MATCH.search(before):
        return True
    policy_window = text[max(0, start - 160): start].lower()
    if "visual policy" in policy_window and re.search(r"\bno\b[^.]{0,40}$", before.lower()):
        return True
    phrase = match.group(0).lower()
    if phrase in {"read the headline", "headline text"}:
        req_window = text[max(0, start - 32): start]
        if _REQUIREMENT_BEFORE_MATCH.search(req_window):
            return False
    return False


def _scan_field_for_headline_omit_dependency(field_path: str, text: str) -> list[Dict[str, Any]]:
    hits: list[Dict[str, Any]] = []
    for pattern, category in HEADLINE_OMIT_DEPENDENCY_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            prohibited_only = _dependency_match_is_prohibition_only(text, match)
            hits.append(
                {
                    "fieldPath": field_path,
                    "matchedPhrase": match.group(0),
                    "safeCategory": category,
                    "prohibitionOnly": prohibited_only,
                    "countsAsPreClosureDependency": not prohibited_only,
                }
            )
    return hits


def analyze_headline_omit_textual_dependency(
    winner_plan: Dict[str, Any],
    *,
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from engine.builder2_advertising_closure_contract import validate_silent_visual_understanding

    decision = get_normalized_headline_decision(winner_plan)
    all_hits: list[Dict[str, Any]] = []
    for field_path, text in collect_headline_omit_runway_execution_field_texts(winner_plan):
        all_hits.extend(_scan_field_for_headline_omit_dependency(field_path, text))

    active_hits = [hit for hit in all_hits if hit["countsAsPreClosureDependency"]]
    source_fields = sorted({hit["fieldPath"] for hit in active_hits})
    categories = sorted({hit["safeCategory"] for hit in active_hits})
    video_prompt = _clean(winner_plan.get("videoPrompt") or winner_plan.get("videoPromptCore"))
    video_hits = [hit for hit in active_hits if hit["fieldPath"] in {"videoPrompt", "videoPromptCore"}]

    return {
        "headlineDecision": decision or None,
        "textualDependencySourceFields": source_fields,
        "textualDependencySafeCategories": categories,
        "textualDependencyMatches": active_hits,
        "dependencyBeforeClosure": bool(active_hits),
        "dependencyOnlyOnClosureSlogan": not bool(active_hits),
        "videoPromptRequestsRenderedText": bool(video_hits),
        "silentVisualUnderstandable": validate_silent_visual_understanding(
            winner_plan=winner_plan,
            winning_judgment=winning_judgment,
        ),
        "headlineFieldsRequired": headline_decision_requires_headline(decision),
    }


def winner_plan_has_pre_closure_textual_dependency(
    winner_plan: Dict[str, Any],
    *,
    winning_judgment: Optional[Dict[str, Any]] = None,
) -> bool:
    return bool(
        analyze_headline_omit_textual_dependency(
            winner_plan,
            winning_judgment=winning_judgment,
        )["dependencyBeforeClosure"]
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


def judge_requires_separate_headline(
    winning_judgment: Optional[Dict[str, Any]],
    *,
    state: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    winning_candidate: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    legacy = _judge_requires_headline(winning_judgment)
    if legacy is not True:
        return legacy
    from engine.builder2_single_slogan_contract import (
        canonical_verbal_copy_satisfied_by_slogan,
        is_single_slogan_contract,
    )

    if is_single_slogan_contract(state=state, plan=plan):
        return False
    return True


def judge_requires_separate_headline_strict(
    winning_judgment: Optional[Dict[str, Any]],
    *,
    state: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    winning_candidate: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Legacy dual-copy interpretation without single-slogan remapping."""
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
    winning_candidate: Optional[Dict[str, Any]] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
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
        if winner_plan_has_pre_closure_textual_dependency(
            winner_plan,
            winning_judgment=winning_judgment,
        ):
            _raise("builder2_winner_validation_failed", field="headlineDecision.omit_with_textual_dependency")
        judge_requires = judge_requires_separate_headline(
            winning_judgment,
            state=tournament_state,
            plan=winner_plan,
            winning_candidate=winning_candidate,
        )
        if judge_requires is True:
            from engine.builder2_single_slogan_contract import (
                canonical_verbal_copy_satisfied_by_slogan,
                is_single_slogan_contract,
                stamp_canonical_copy_judge_mapping,
            )

            if is_single_slogan_contract(state=tournament_state, plan=winner_plan):
                stamp_canonical_copy_judge_mapping(
                    winner_plan,
                    winning_judgment=winning_judgment,
                    winning_candidate=winning_candidate,
                    state=tournament_state,
                )
                if canonical_verbal_copy_satisfied_by_slogan(
                    winner_plan,
                    winning_judgment=winning_judgment,
                    winning_candidate=winning_candidate,
                    state=tournament_state,
                ):
                    logger.info(
                        "BUILDER2_HEADLINE_DECISION_OMIT_SATISFIED_BY_SLOGAN decision=omit canonicalCopySatisfiedBy=slogan",
                    )
                else:
                    _raise(
                        "builder2_winner_validation_failed",
                        field="builder2_winner_canonical_copy_does_not_satisfy_judge",
                    )
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
