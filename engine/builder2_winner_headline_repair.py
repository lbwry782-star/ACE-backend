"""
Builder2 Winner headline repair — one bounded paid repair for missing in-scene headline fields.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_complete_ad_resume_plan import parsed_winner_reusable_for_candidate
from engine.builder2_headline_decision_contract import (
    _judge_requires_headline,
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_config import resolve_builder2_winner_model
from engine.builder2_tournament_contracts import Builder2TournamentError, require_non_empty_str
from engine.builder2_tournament_llm import call_builder2_role_json_with_text
from engine.builder2_tournament_metrics import MetricsTimer, ensure_metrics, record_winner_paid_call_submitted
from engine.builder2_tournament_prompts import build_winner_headline_repair_prompt
from engine.builder2_winner_downstream import (
    Builder2WinnerDownstreamError,
    validate_builder2_winner_headline_composition_pure,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    load_revalidatable_parsed_winner_response,
    persist_parsed_winner_response,
    process_winner_development_response,
)

logger = logging.getLogger(__name__)

HEADLINE_REPAIR_FAILURE_PREFIX = "builder2_tournament_invalid_field:"
ELIGIBLE_HEADLINE_FIELDS = frozenset({"headline", "headlineCoreKeyword", "headlineText"})
ALLOWED_PARTIAL_REPAIR_KEYS = frozenset({"headline", "headlineCoreKeyword"})
REPAIR_ALREADY_ATTEMPTED = "builder2_winner_headline_repair_already_attempted"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def classify_headline_only_offline_failure(failure_reason: str) -> Optional[str]:
    reason = _clean(failure_reason)
    if not reason.startswith(HEADLINE_REPAIR_FAILURE_PREFIX):
        return None
    field = reason.split(":", 1)[-1]
    if field not in ELIGIBLE_HEADLINE_FIELDS:
        return None
    return field


def _headline_field_missing_or_empty(plan: Dict[str, Any], field: str) -> bool:
    if field not in plan:
        return True
    value = plan.get(field)
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def assess_winner_headline_repair_eligibility(
    state: Dict[str, Any],
    *,
    winner_candidate_id: str,
    offline_failure_reason: str,
    allow_repair: bool,
    remaining_call_budget: int,
) -> Dict[str, Any]:
    metrics = ensure_metrics(state)
    prior_repair_calls = int(metrics.get("winnerRepairCalls") or 0)
    failure_field = classify_headline_only_offline_failure(offline_failure_reason)
    parsed = load_revalidatable_parsed_winner_response(state)
    parsed_plan = dict((parsed or {}).get("parsed") or {}) if parsed else {}
    decision = get_normalized_headline_decision(parsed_plan) if parsed_plan else ""
    judge_requires = _judge_requires_headline(
        _resolve_judgment(state, winner_candidate_id) if winner_candidate_id else None
    )

    report: Dict[str, Any] = {
        "eligible": False,
        "reason": "builder2_winner_headline_repair_ineligible",
        "failureField": failure_field,
        "priorWinnerRepairCalls": prior_repair_calls,
        "authorizationFlag": allow_repair,
        "remainingCallBudget": remaining_call_budget,
        "headlineDecision": decision or None,
        "judgeRequiresHeadline": judge_requires,
    }

    if accepted_creator_count(state) != 6 or accepted_judgment_count(state) != 6:
        report["reason"] = "builder2_winner_headline_repair_ineligible:incomplete_tournament"
        return report
    if not winner_candidate_id:
        report["reason"] = "builder2_winner_headline_repair_ineligible:winner_missing"
        return report
    from engine.builder2_single_slogan_contract import is_single_slogan_contract

    if is_single_slogan_contract(state=state):
        report["reason"] = "builder2_winner_headline_repair_ineligible:single_slogan_contract"
        return report
    if is_valid_persisted_winner_development(state):
        report["reason"] = "builder2_winner_headline_repair_ineligible:winner_already_accepted"
        return report
    if not parsed_winner_reusable_for_candidate(state, winner_candidate_id=winner_candidate_id):
        report["reason"] = "builder2_winner_headline_repair_ineligible:parsed_response_missing_or_mismatch"
        return report
    if failure_field is None:
        report["reason"] = "builder2_winner_headline_repair_ineligible:offline_failure_not_headline_only"
        return report
    if not headline_decision_requires_headline(decision):
        report["reason"] = "builder2_winner_headline_repair_ineligible:headline_decision_not_use"
        return report
    if judge_requires is not True:
        report["reason"] = "builder2_winner_headline_repair_ineligible:judge_does_not_require_headline"
        return report
    if prior_repair_calls >= 1:
        report["reason"] = REPAIR_ALREADY_ATTEMPTED
        return report
    if not allow_repair:
        report["reason"] = "builder2_winner_headline_repair_ineligible:authorization_disabled"
        return report
    if remaining_call_budget < 1:
        report["reason"] = "builder2_winner_headline_repair_ineligible:call_budget_exhausted"
        return report
    if not _headline_field_missing_or_empty(parsed_plan, "headline") and failure_field == "headline":
        if not _headline_field_missing_or_empty(parsed_plan, "headlineCoreKeyword"):
            report["reason"] = "builder2_winner_headline_repair_ineligible:headline_fields_present"
            return report

    report["eligible"] = True
    report["reason"] = "builder2_winner_headline_repair_eligible"
    return report


def _resolve_judgment(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    winner_rec = (state.get("candidates") or {}).get(candidate_id) or {}
    judgment_id = winner_rec.get("judgmentId")
    if not judgment_id:
        return {}
    judgment_rec = (state.get("judgments") or {}).get(str(judgment_id)) or {}
    judgment = judgment_rec.get("judgment")
    return judgment if isinstance(judgment, dict) else {}


def _log_eligibility(*, job_id: str, candidate_id: str, eligibility: Dict[str, Any]) -> None:
    logger.info(
        "BUILDER2_WINNER_HEADLINE_REPAIR_ELIGIBILITY jobId=%s candidateId=%s eligible=%s reason=%s "
        "failureField=%s priorWinnerRepairCalls=%s authorizationFlag=%s remainingCallBudget=%s",
        job_id,
        candidate_id,
        eligibility.get("eligible"),
        eligibility.get("reason"),
        eligibility.get("failureField"),
        eligibility.get("priorWinnerRepairCalls"),
        eligibility.get("authorizationFlag"),
        eligibility.get("remainingCallBudget"),
    )


def _log_skipped(*, job_id: str, reason: str) -> None:
    logger.info("BUILDER2_WINNER_HEADLINE_REPAIR_SKIPPED jobId=%s reason=%s", job_id, reason)


def parse_winner_headline_repair_partial(raw: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise Builder2TournamentError("builder2_winner_headline_repair_invalid_response:not_object")
    extra = set(raw.keys()) - ALLOWED_PARTIAL_REPAIR_KEYS
    if extra:
        raise Builder2TournamentError("builder2_winner_headline_repair_invalid_response:extra_keys")
    headline = require_non_empty_str(raw.get("headline"), field="headline")
    keyword = require_non_empty_str(raw.get("headlineCoreKeyword"), field="headlineCoreKeyword")
    if len(keyword.split()) != 1:
        raise Builder2TournamentError("builder2_winner_headline_repair_invalid_response:headlineCoreKeyword")
    return {"headline": headline.strip(), "headlineCoreKeyword": keyword.strip()}


def merge_headline_repair_into_parsed_plan(
    parsed_plan: Dict[str, Any],
    *,
    partial: Dict[str, str],
) -> Tuple[Dict[str, Any], int]:
    merged = deepcopy(parsed_plan)
    preserved_count = len(merged)
    merged["headline"] = partial["headline"]
    merged["headlineCoreKeyword"] = partial["headlineCoreKeyword"]
    for key in ("headlineText", "headlineTextRemainder", "advertisingPromise"):
        merged.pop(key, None)
    return merged, preserved_count


def _word_count(text: str) -> int:
    return len([part for part in str(text or "").split() if part.strip()])


def _log_merged(*, merged: Dict[str, Any], preserved_field_count: int) -> None:
    headline = str(merged.get("headline") or "")
    keyword = str(merged.get("headlineCoreKeyword") or "")
    logger.info(
        "BUILDER2_WINNER_HEADLINE_REPAIR_MERGED headlineCharCount=%s headlineWordCount=%s "
        "keywordCharCount=%s keywordWordCount=%s preservedFieldCount=%s",
        len(headline),
        _word_count(headline),
        len(keyword),
        _word_count(keyword),
        preserved_field_count,
    )


def validate_and_finalize_repaired_winner_plan(
    merged_parsed: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    preservation_snapshot: Dict[str, Any],
    compatibility_mode: bool,
    job_id: str,
    tournament_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    validated = process_winner_development_response(
        merged_parsed,
        source_reference=source_reference,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        job_id=job_id,
        tournament_id=tournament_id,
        tournament_state=tournament_state,
    )
    composition_plan = deepcopy(validated)
    try:
        validate_builder2_winner_headline_composition_pure(composition_plan)
    except Builder2WinnerDownstreamError as exc:
        raise Builder2TournamentError(exc.code) from exc
    for key in ("headline", "headlineText", "headlineTextRemainder", "headlineCoreKeyword", "advertisingPromise"):
        if key in composition_plan:
            validated[key] = composition_plan[key]
    return validated


def repair_builder2_winner_headline_from_parsed(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    product_name: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    validation_failures: List[str],
    compatibility_mode: bool = False,
    llm_client: Optional[Any] = None,
    job_id: str = "",
    tournament_id: str = "",
) -> Dict[str, Any]:
    payload = load_revalidatable_parsed_winner_response(state)
    if payload is None:
        raise Builder2TournamentError("builder2_winner_headline_repair_missing_parsed_response")
    parsed_plan = dict(payload.get("parsed") or {})
    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    preservation_snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=candidate_id,
    )
    model = resolve_builder2_winner_model()
    prompt = build_winner_headline_repair_prompt(
        product_name=product_name,
        language=language,
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        winning_judgment=winning_judgment,
        parsed_winner_plan=parsed_plan,
        validation_failures=validation_failures,
    )

    logger.info(
        "BUILDER2_WINNER_HEADLINE_REPAIR_START jobId=%s tournamentId=%s candidateId=%s prototypeId=%s model=%s callType=repair",
        job_id,
        tournament_id,
        candidate_id,
        prototype_id,
        model,
    )

    timer = MetricsTimer()
    response_text = ""
    repair_call_recorded = False

    def _on_paid_request_submitted() -> None:
        nonlocal repair_call_recorded
        record_winner_paid_call_submitted(state, repair=True, retry=False)
        state["winnerDevelopmentPaidCallRecorded"] = True
        repair_call_recorded = True

    try:
        raw, response_text = call_builder2_role_json_with_text(
            role="builder2_winner",
            model=model,
            prompt=prompt,
            call_type="repair",
            llm_client=llm_client,
            on_paid_request_submitted=_on_paid_request_submitted,
        )
    except Exception as exc:
        logger.error(
            "BUILDER2_WINNER_HEADLINE_REPAIR_FAILED jobId=%s failureStage=openai_call exceptionClass=%s "
            "safeErrorCode=builder2_winner_headline_repair_openai_failed validationField=(none) "
            "responsePresent=%s responseChars=%s repairCallRecorded=%s",
            job_id,
            type(exc).__name__,
            bool(response_text.strip()),
            len(response_text or ""),
            repair_call_recorded,
        )
        raise Builder2TournamentError(f"builder2_winner_headline_repair_openai_failed:{type(exc).__name__}") from exc

    logger.info(
        "BUILDER2_WINNER_HEADLINE_REPAIR_RESPONSE_RECEIVED jobId=%s elapsedMs=%.1f responseTextPresent=%s responseTextChars=%s",
        job_id,
        timer.elapsed_ms(),
        bool(response_text.strip()),
        len(response_text or ""),
    )

    try:
        partial = parse_winner_headline_repair_partial(raw)
        merged, preserved_count = merge_headline_repair_into_parsed_plan(parsed_plan, partial=partial)
        _log_merged(merged=merged, preserved_field_count=preserved_count)
        persist_parsed_winner_response(
            state,
            parsed=merged,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            top_level_keys=sorted(merged.keys()),
            response_char_count=len(response_text),
        )
        winner_plan = validate_and_finalize_repaired_winner_plan(
            merged,
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
            job_id=job_id,
            tournament_id=tournament_id,
            tournament_state=state,
        )
        persist_winner_development_atomically(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            winner_plan=winner_plan,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_winner_headline_repair_failed")
        field = reason.split(":", 1)[-1] if ":" in reason else None
        logger.error(
            "BUILDER2_WINNER_HEADLINE_REPAIR_FAILED jobId=%s failureStage=validation exceptionClass=Builder2TournamentError "
            "safeErrorCode=%s validationField=%s responsePresent=%s responseChars=%s repairCallRecorded=%s",
            job_id,
            reason,
            field or "(none)",
            bool(response_text.strip()),
            len(response_text or ""),
            repair_call_recorded,
        )
        raise
    except Exception as exc:
        logger.exception(
            "BUILDER2_WINNER_HEADLINE_REPAIR_FAILED jobId=%s failureStage=unexpected exceptionClass=%s "
            "responsePresent=%s responseChars=%s repairCallRecorded=%s",
            job_id,
            type(exc).__name__,
            bool(response_text.strip()),
            len(response_text or ""),
            repair_call_recorded,
        )
        raise Builder2TournamentError(f"builder2_winner_headline_repair_failed:{type(exc).__name__}") from exc

    logger.info(
        "BUILDER2_WINNER_HEADLINE_REPAIR_ACCEPTED jobId=%s candidateId=%s prototypeId=%s",
        job_id,
        candidate_id,
        prototype_id,
    )
    return winner_plan


def attempt_winner_headline_repair_after_offline_failure(
    state: Dict[str, Any],
    *,
    job_id: str,
    winner_candidate_id: str,
    prototype_id: str,
    product_name: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    offline_failure_reason: str,
    allow_repair: bool,
    remaining_call_budget: int,
    compatibility_mode: bool = False,
    llm_client: Optional[Any] = None,
    tournament_id: str = "",
    on_eligible_before_call: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    eligibility = assess_winner_headline_repair_eligibility(
        state,
        winner_candidate_id=winner_candidate_id,
        offline_failure_reason=offline_failure_reason,
        allow_repair=allow_repair,
        remaining_call_budget=remaining_call_budget,
    )
    _log_eligibility(job_id=job_id, candidate_id=winner_candidate_id, eligibility=eligibility)

    if not eligibility.get("eligible"):
        _log_skipped(job_id=job_id, reason=str(eligibility.get("reason")))
        return {
            "attempted": False,
            "accepted": False,
            "skipped": True,
            "skip_reason": eligibility.get("reason"),
            "failure_reason": offline_failure_reason,
            "winner_plan": None,
        }

    if on_eligible_before_call is not None:
        on_eligible_before_call()

    try:
        winner_plan = repair_builder2_winner_headline_from_parsed(
            state,
            candidate_id=winner_candidate_id,
            prototype_id=prototype_id,
            product_name=product_name,
            language=language,
            strategy_foundation=strategy_foundation,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            validation_failures=[offline_failure_reason],
            compatibility_mode=compatibility_mode,
            llm_client=llm_client,
            job_id=job_id,
            tournament_id=tournament_id,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_winner_headline_repair_failed")
        return {
            "attempted": True,
            "accepted": False,
            "skipped": False,
            "skip_reason": None,
            "failure_reason": reason,
            "winner_plan": None,
        }

    return {
        "attempted": True,
        "accepted": True,
        "skipped": False,
        "skip_reason": None,
        "failure_reason": None,
        "winner_plan": winner_plan,
    }
