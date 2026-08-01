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

HEADLINE_OMIT_DEPENDENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bread the headline\b", "headline_read_requirement"),
    (r"\bheadline text\b", "in_video_headline_text"),
    (r"\bon-screen text\b", "on_screen_text"),
    (r"\bonscreen text\b", "on_screen_text"),
    (r"\btitle cards?\b", "title_card"),
    (r"\bcaption says\b", "caption_says"),
    (r"\btext overlays?\b", "text_overlay"),
    (r"\btext overlay reveals\b", "text_overlay_reveals"),
    (r"\bwritten captions?\b", "written_caption"),
    (r"\bcaptions?\b", "captions"),
    (r"\b(read|reads|reading)\s+(?:the\s+)?(?:sign|label|caption|subtitle)\b", "readable_sign_or_caption"),
    (r"\b(?:the\s+)?viewer\s+(?:must\s+)?(?:read|reads|reading)\b", "viewer_reads"),
    (r"\b(?:sign|label|screen)\s+(?:reads|displays|shows)\b", "sign_or_screen_copy"),
    (
        r"\b(?:display|show|render|burn(?:s|-in)?|superimpose)\s+(?:the\s+|an?\s+)?(?:headline|caption|subtitle|overlay)\b",
        "render_text_instruction",
    ),
    (r"\bsuperimpose(?:s)?\s+(?:the\s+)?words\b", "superimpose_words"),
    (r"\btitle card appears\b", "title_card_appears"),
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

_NEGATION_WINDOW_CHARS = 96
_CONTEXT_SNIPPET_CHARS = 48

_INHERENTLY_POSITIVE_CATEGORIES = frozenset(
    {
        "headline_read_requirement",
        "caption_says",
        "sign_or_screen_copy",
        "render_text_instruction",
        "text_overlay_reveals",
        "superimpose_words",
        "title_card_appears",
        "viewer_reads",
    }
)

_NEGATION_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)(?:^|[\W_])(?:do not|don't|don`t)\s+"
            r"(?:show|display|render|include|add|superimpose|use|feature|present|write|place|put)"
            r"(?:\s+(?:any|visible|readable|written|an?|the|a))*$"
        ),
        "do_not_action",
    ),
    (
        re.compile(
            r"(?i)(?:^|[\W_])(?:without|avoid|exclude|never|ban|forbidden|prohibited)"
            r"(?:\s+(?:any|visible|readable|written|an?|the|a))*$"
        ),
        "without",
    ),
    (re.compile(r"(?i)(?:^|[\W_])no(?:\s+(?:any|visible|readable|written|an?|the|a))*$"), "no"),
    (re.compile(r"(?i)(?:^|[\W_])not(?:\s+(?:any|visible|readable|written|an?|the|a))*$"), "not"),
    (re.compile(r"(?i)(?:^|[\W_])none(?:\s+(?:of|the))?$"), "none"),
    (
        re.compile(
            r"(?:^|[\W_])(?:ללא|בלי|אין)(?:\s+[\u0590-\u05FF\s.,!?-]+)*$"
        ),
        "hebrew_without",
    ),
    (
        re.compile(
            r"(?:^|[\W_])(?:אל|לא)\s+(?:להציג|להוסיף|לכתוב|להציגו)(?:\s+[\u0590-\u05FF\s.,!?-]+)*$"
        ),
        "hebrew_do_not_show",
    ),
)

_NEGATION_PHRASE_ANYWHERE = (
    (re.compile(r"(?i)\btext-free\b"), "text_free"),
    (re.compile(r"(?i)\bpurely visual(?:,\s*no text)?\b"), "purely_visual"),
)

_POSITIVE_ACTION_BEFORE_MATCH = re.compile(
    r"(?i)(?:^|[\W_])"
    r"(?:must|needs to|required to|have to|should|viewer(?:s)?\s+(?:must|need to|should)|"
    r"display|show|render|superimpose|burn(?:-in)?|include|add|feature|present|place|put|write)"
    r"(?:\s+\w+){0,4}$"
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


def _safe_context_snippet(text: str, start: int, end: int) -> tuple[str, str]:
    before = text[max(0, start - _CONTEXT_SNIPPET_CHARS): start]
    after = text[end: min(len(text), end + _CONTEXT_SNIPPET_CHARS)]
    return before, after


def _detect_negation_before_match(text: str, start: int) -> tuple[bool, str, Optional[int]]:
    before = text[max(0, start - _NEGATION_WINDOW_CHARS): start]
    normalized_before = before.rstrip()
    for pattern, token in _NEGATION_SUFFIX_PATTERNS:
        match = pattern.search(normalized_before)
        if match:
            return True, token, len(normalized_before) - match.start()
    scan_window = text[max(0, start - _NEGATION_WINDOW_CHARS): start]
    for pattern, token in _NEGATION_PHRASE_ANYWHERE:
        if pattern.search(scan_window):
            return True, token, None
    policy_window = text[max(0, start - 160): start].lower()
    if "visual policy" in policy_window and re.search(r"\bno\b[^.]{0,40}$", before.lower()):
        return True, "visual_policy_no", len(before)
    return False, "", None


def _positive_action_verb_before_match(text: str, start: int) -> str:
    before = text[max(0, start - _NEGATION_WINDOW_CHARS): start]
    match = _POSITIVE_ACTION_BEFORE_MATCH.search(before.rstrip())
    if not match:
        return ""
    fragment = match.group(0).strip()
    tokens = fragment.split()
    return tokens[-1].lower() if tokens else ""


def _headline_requirement_overrides_negation(text: str, start: int, category: str) -> bool:
    if category not in {"headline_read_requirement", "in_video_headline_text"}:
        return False
    req_window = text[max(0, start - 32): start]
    return bool(_REQUIREMENT_BEFORE_MATCH.search(req_window))


def _classify_textual_dependency_match(
    text: str,
    match: re.Match[str],
    *,
    field_path: str,
    category: str,
) -> Dict[str, Any]:
    start, end = match.start(), match.end()
    matched_text = match.group(0)
    normalized_matched = re.sub(r"\s+", " ", matched_text.strip().lower())
    context_before, context_after = _safe_context_snippet(text, start, end)
    negated, negation_token, negation_distance = _detect_negation_before_match(text, start)
    positive_action_verb = _positive_action_verb_before_match(text, start)
    if negated and _headline_requirement_overrides_negation(text, start, category):
        negated = False
        negation_token = ""
        negation_distance = None

    if negated:
        polarity = "negative_instruction"
        requests_rendered = False
        counts_as_dependency = False
        exclusion_reason = "negative_instruction"
    elif category in _INHERENTLY_POSITIVE_CATEGORIES or positive_action_verb:
        polarity = "positive_instruction"
        requests_rendered = True
        counts_as_dependency = True
        exclusion_reason = ""
    else:
        polarity = "ambiguous"
        requests_rendered = True
        counts_as_dependency = True
        exclusion_reason = ""

    return {
        "sourceField": field_path,
        "fieldPath": field_path,
        "category": category,
        "safeCategory": category,
        "matchedText": matched_text,
        "matchedPhrase": matched_text,
        "normalizedMatchedText": normalized_matched,
        "contextBefore": context_before,
        "contextAfter": context_after,
        "characterStart": start,
        "characterEnd": end,
        "polarity": polarity,
        "negationToken": negation_token,
        "negationDistance": negation_distance,
        "positiveActionVerb": positive_action_verb,
        "requestsRenderedText": requests_rendered,
        "countsAsPreClosureDependency": counts_as_dependency,
        "prohibitionOnly": polarity == "negative_instruction",
        "exclusionReason": exclusion_reason,
    }


def _field_has_active_textual_dependency(matches: list[Dict[str, Any]]) -> bool:
    if any(match.get("polarity") == "positive_instruction" and match.get("requestsRenderedText") for match in matches):
        return True
    return any(
        match.get("polarity") == "ambiguous" and match.get("countsAsPreClosureDependency")
        for match in matches
    )


def _scan_field_for_headline_omit_dependency(field_path: str, text: str) -> list[Dict[str, Any]]:
    hits: list[Dict[str, Any]] = []
    for pattern, category in HEADLINE_OMIT_DEPENDENCY_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append(
                _classify_textual_dependency_match(
                    text,
                    match,
                    field_path=field_path,
                    category=category,
                )
            )
    return hits


def _partition_textual_dependency_matches(
    matches: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    positive = [match for match in matches if match.get("polarity") == "positive_instruction"]
    negative = [match for match in matches if match.get("polarity") == "negative_instruction"]
    ambiguous = [match for match in matches if match.get("polarity") == "ambiguous"]
    return positive, negative, ambiguous


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

    positive_matches, negative_matches, ambiguous_matches = _partition_textual_dependency_matches(all_hits)
    active_hits: list[Dict[str, Any]] = []
    for field_path in sorted({hit.get("sourceField") for hit in all_hits if hit.get("sourceField")}):
        field_matches = [hit for hit in all_hits if hit.get("sourceField") == field_path]
        if not _field_has_active_textual_dependency(field_matches):
            continue
        active_hits.extend(
            hit
            for hit in field_matches
            if hit.get("polarity") in {"positive_instruction", "ambiguous"}
        )
    source_fields = sorted(
        {
            field_path
            for field_path in {hit.get("sourceField") for hit in all_hits}
            if field_path
            and _field_has_active_textual_dependency([hit for hit in all_hits if hit.get("sourceField") == field_path])
        }
    )
    categories = sorted({hit["category"] for hit in active_hits})
    video_prompt_fields = {"videoPrompt", "videoPromptCore"}
    video_prompt_matches = [hit for hit in all_hits if hit.get("sourceField") in video_prompt_fields]
    video_positive = [
        hit
        for hit in video_prompt_matches
        if hit.get("polarity") == "positive_instruction"
        and _field_has_active_textual_dependency(
            [match for match in video_prompt_matches if match.get("sourceField") == hit.get("sourceField")]
        )
    ]
    video_negative_only = bool(video_prompt_matches) and not video_positive and all(
        hit.get("polarity") == "negative_instruction" for hit in video_prompt_matches
    )

    return {
        "headlineDecision": decision or None,
        "textualDependencySourceFields": source_fields,
        "exactDependencySourceFields": source_fields,
        "textualDependencyMatchCategories": categories,
        "textualDependencySafeCategories": categories,
        "textualDependencyMatches": active_hits,
        "allTextualDependencyMatches": all_hits,
        "positiveTextualDependencyMatches": positive_matches,
        "negativeTextualDependencyMatches": negative_matches,
        "ambiguousTextualDependencyMatches": ambiguous_matches,
        "dependencyBeforeClosure": bool(source_fields),
        "dependencyOnlyOnClosureSlogan": not bool(source_fields),
        "videoPromptPositiveRenderedTextRequest": bool(video_positive),
        "videoPromptNegativeTextInstructionOnly": video_negative_only,
        "videoPromptRequestsRenderedText": bool(video_positive),
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
