"""
Builder2 Winner offline salvage — reuse persisted paid responses without additional API calls.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from engine.builder2_complete_ad_resume_plan import parsed_winner_reusable_for_candidate
from engine.builder2_headline_decision_contract import get_normalized_headline_decision
from engine.builder2_single_slogan_contract import (
    BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
    compatibility_headline_mirror_status,
    copy_contract_version,
    resolve_canonical_slogan_text,
    separate_headline_present,
    validate_single_slogan_plan_contract,
)
from engine.builder2_tournament_completion_gate import (
    accepted_creator_count,
    accepted_judgment_count,
    missing_creator_prototype_ids,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_winner_persistence import (
    has_failed_winner_attempt_after_paid_call,
    is_valid_persisted_winner_development,
    persist_winner_development_atomically,
)
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
    prepare_and_validate_persisted_winner_offline,
)

logger = logging.getLogger(__name__)

WINNER_DEVELOPMENT_CALL_LEDGER_KEY = "winnerDevelopmentCallLedger"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ledger_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    bucket = state.setdefault(WINNER_DEVELOPMENT_CALL_LEDGER_KEY, {})
    if not isinstance(bucket, dict):
        bucket = {}
        state[WINNER_DEVELOPMENT_CALL_LEDGER_KEY] = bucket
    return bucket


def reconcile_winner_development_call_ledger(state: Dict[str, Any]) -> Dict[str, Any]:
    bucket = _ledger_bucket(state)
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    metric_dispatch = int(metrics.get("winnerDevelopmentCalls") or 0)
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    if state.get("winnerDevelopmentPaidCallRecorded") and paid_dispatch < 1:
        paid_dispatch = 1
    if metric_dispatch > paid_dispatch:
        paid_dispatch = min(metric_dispatch, 1)
    if paid_dispatch > 1:
        paid_dispatch = 1
    bucket["paidDispatchCount"] = paid_dispatch
    bucket["winnerDevelopmentDispatchCalls"] = paid_dispatch
    if load_revalidatable_parsed_winner_response(state):
        bucket["responseReceived"] = True
        bucket["parsed"] = True
    elif state.get("winnerDevelopmentPaidCallRecorded") or int(metrics.get("winnerDevelopmentCalls") or 0) >= 1:
        bucket["responseReceived"] = True
        bucket["parsed"] = True
    if is_valid_persisted_winner_development(state):
        bucket["accepted"] = True
    if _clean(state.get("winnerCandidateId")):
        bucket["winnerSelected"] = True
    return bucket


def winner_development_dispatch_count(state: Dict[str, Any]) -> int:
    return int(reconcile_winner_development_call_ledger(state).get("paidDispatchCount") or 0)


def additional_paid_winner_development_allowed(state: Dict[str, Any]) -> bool:
    if is_valid_persisted_winner_development(state):
        return False
    if winner_development_dispatch_count(state) >= 1:
        return False
    if has_failed_winner_attempt_after_paid_call(state):
        return False
    return True


def populate_winner_development_call_report(state: Dict[str, Any], report: Dict[str, Any]) -> None:
    bucket = reconcile_winner_development_call_ledger(state)
    parsed = load_revalidatable_parsed_winner_response(state)
    failure = state.get("winnerDevelopmentFailure") if isinstance(state.get("winnerDevelopmentFailure"), dict) else {}
    paid_dispatch = int(bucket.get("paidDispatchCount") or 0)
    response_received = bool(parsed) or bool(bucket.get("responseReceived")) or paid_dispatch >= 1
    report["winnerDevelopmentDispatchCalls"] = paid_dispatch
    report["winnerDevelopmentResponseReceived"] = response_received
    report["winnerDevelopmentParsed"] = bool(parsed) or bool(bucket.get("parsed")) or paid_dispatch >= 1
    report["winnerDevelopmentAccepted"] = is_valid_persisted_winner_development(state)
    report["winnerDevelopmentAdditionalPaidCallAllowed"] = additional_paid_winner_development_allowed(state)
    report["acceptedCreatorsCount"] = accepted_creator_count(state)
    report["acceptedJudgmentsCount"] = accepted_judgment_count(state)
    report["missingPrototypeIds"] = list(missing_creator_prototype_ids(state))
    if failure:
        report["winnerDevelopmentFailureField"] = _clean(failure.get("failureField") or failure.get("stage"))


def inspect_winner_development_recovery_state(
    state: Dict[str, Any],
    *,
    offline_salvage_attempted: bool = False,
    offline_salvage_validation_passed: bool = False,
    offline_salvage_failure_field: str = "",
) -> Dict[str, Any]:
    bucket = reconcile_winner_development_call_ledger(state)
    parsed_payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((parsed_payload or {}).get("parsed") or {})
    failure = state.get("winnerDevelopmentFailure") if isinstance(state.get("winnerDevelopmentFailure"), dict) else {}
    judgment = {}
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    judgment_id = _clean(winner_rec.get("judgmentId"))
    if judgment_id:
        judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    from engine.builder2_headline_decision_contract import (
        judge_requires_separate_headline,
        judge_requires_verbal_copy,
    )
    from engine.builder2_advertising_closure_contract import count_slogan_words_excluding_product

    plan_for_inspect: Dict[str, Any] = {}
    if parsed and winner_rec:
        candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        source = build_server_owned_winner_source_reference(
            strategy_foundation=state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {},
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        try:
            plan_for_inspect = prepare_and_validate_persisted_winner_offline(
                parsed,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=judgment if isinstance(judgment, dict) else None,
                tournament_state=state,
                job_id=_clean(state.get("jobId")),
                tournament_id=_clean(state.get("tournamentId")),
            )
        except Builder2TournamentError:
            plan_for_inspect = dict(parsed)
    elif parsed:
        plan_for_inspect = dict(parsed)
    slogan = resolve_canonical_slogan_text(plan=plan_for_inspect, state=state)
    product = _clean((plan_for_inspect.get("advertisingClosure") or {}).get("productNameText"))
    ok, _failures = (
        validate_single_slogan_plan_contract(plan_for_inspect, state=state) if plan_for_inspect else (False, [])
    )
    mirror_status = compatibility_headline_mirror_status(plan_for_inspect, state=state) if plan_for_inspect else "not_required"
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "winnerSelected": bool(_clean(state.get("winnerCandidateId"))),
        "winnerPrototypeId": _clean(winner_rec.get("prototypeId") or state.get("winnerDevelopmentPrototypeId")),
        "winnerCandidateId": winner_id,
        "winnerDispatchCount": int(bucket.get("paidDispatchCount") or 0),
        "winnerResponseFound": bool(parsed_payload),
        "winnerResponseReceived": bool(parsed_payload) or bool(bucket.get("responseReceived")),
        "winnerParsedResponseFound": bool(parsed),
        "winnerFailureFound": bool(failure),
        "winnerFailureField": _clean(failure.get("failureField") or failure.get("stage")),
        "copyContractVersion": copy_contract_version(state=state, plan=plan_for_inspect)
        or BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
        "judgeRequiresVerbalCopy": judge_requires_verbal_copy(judgment if isinstance(judgment, dict) else None),
        "judgeRequiresSeparateHeadline": judge_requires_separate_headline(
            judgment if isinstance(judgment, dict) else None,
            state=state,
            plan=plan_for_inspect if plan_for_inspect else None,
            winning_candidate=winner_rec.get("creatorOutput") or winner_rec.get("creatorSnapshot") if winner_rec else None,
        )
        is True,
        "canonicalSloganPresent": bool(slogan),
        "canonicalSloganWordCount": count_slogan_words_excluding_product(slogan, product) if slogan else 0,
        "headlineDecision": get_normalized_headline_decision(plan_for_inspect) if plan_for_inspect else "",
        "separateHeadlinePresent": separate_headline_present(plan_for_inspect) if plan_for_inspect else False,
        "compatibilityHeadlineMirrorsSlogan": mirror_status,
        "singleSloganContractSatisfied": ok,
        "canonicalCopySatisfiedBy": _clean(plan_for_inspect.get("canonicalCopySatisfiedBy")),
        "offlineSalvageAttempted": bool(offline_salvage_attempted),
        "offlineSalvageValidationPassed": bool(offline_salvage_validation_passed),
        "offlineSalvageFailureField": _clean(offline_salvage_failure_field),
        "additionalPaidWinnerCallAllowed": additional_paid_winner_development_allowed(state),
        "acceptedCreatorsCount": accepted_creator_count(state),
        "acceptedJudgmentsCount": accepted_judgment_count(state),
        "missingPrototypeIds": list(missing_creator_prototype_ids(state)),
        "stateMutated": False,
        "paidCalls": 0,
    }


def attempt_offline_winner_development_salvage(
    state: Dict[str, Any],
    *,
    winner_candidate_id: str,
    prototype_id: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta = {
        "attempted": True,
        "accepted": False,
        "failure_field": "",
        "failure_reason": "",
    }
    if is_valid_persisted_winner_development(state):
        meta["accepted"] = True
        meta["reusedAccepted"] = True
        return deepcopy(state.get("winnerDevelopmentPlan") or {}), meta

    if not parsed_winner_reusable_for_candidate(state, winner_candidate_id=winner_candidate_id):
        meta["failure_reason"] = "builder2_winner_response_not_persisted"
        raise Builder2TournamentError("builder2_winner_response_not_persisted")

    logger.info(
        "BUILDER2_WINNER_OFFLINE_SALVAGE_START jobId=%s candidateId=%s prototypeId=%s",
        job_id or _clean(state.get("jobId")),
        winner_candidate_id,
        prototype_id,
    )
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        candidate_id=winner_candidate_id,
    )
    try:
        winner_plan = prepare_and_validate_persisted_winner_offline(
            dict((load_revalidatable_parsed_winner_response(state) or {}).get("parsed") or {}),
            source_reference=source,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
            job_id=job_id or _clean(state.get("jobId")),
            tournament_id=tournament_id or _clean(state.get("tournamentId")),
            tournament_state=state,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_winner_offline_salvage_invalid")
        meta["failure_reason"] = reason
        meta["failure_field"] = reason.split(":", 1)[-1] if reason else ""
        logger.warning(
            "BUILDER2_WINNER_OFFLINE_SALVAGE_FAILED jobId=%s candidateId=%s failureField=%s",
            job_id or _clean(state.get("jobId")),
            winner_candidate_id,
            meta["failure_field"],
        )
        raise

    persist_winner_development_atomically(
        state,
        candidate_id=winner_candidate_id,
        prototype_id=prototype_id,
        winner_plan=winner_plan,
        winning_candidate=winning_candidate,
        preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
        compatibility_mode=compatibility_mode,
    )
    reconcile_winner_development_call_ledger(state)
    meta["accepted"] = True
    logger.info(
        "BUILDER2_WINNER_OFFLINE_SALVAGE_ACCEPTED jobId=%s candidateId=%s prototypeId=%s",
        job_id or _clean(state.get("jobId")),
        winner_candidate_id,
        prototype_id,
    )
    return winner_plan, meta


def assert_no_duplicate_paid_winner_development(state: Dict[str, Any], *, winner_candidate_id: str) -> None:
    if is_valid_persisted_winner_development(state):
        return
    if parsed_winner_reusable_for_candidate(state, winner_candidate_id=winner_candidate_id):
        return
    if winner_development_dispatch_count(state) >= 1:
        raise Builder2TournamentError("builder2_winner_additional_paid_call_requires_approval")
    if state.get("winnerDevelopmentPaidCallRecorded"):
        raise Builder2TournamentError("builder2_winner_additional_paid_call_requires_approval")
