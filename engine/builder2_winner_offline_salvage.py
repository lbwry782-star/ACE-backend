"""
Builder2 Winner offline salvage — reuse persisted paid responses without additional API calls.

Run pre-salvage inspect:
  BUILDER2_WINNER_OFFLINE_SALVAGE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_winner_offline_salvage inspect

Run offline salvage:
  BUILDER2_WINNER_OFFLINE_SALVAGE_JOB_ID=<jobId> python -m engine.builder2_winner_offline_salvage
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from engine.builder2_complete_ad_resume_plan import parsed_winner_reusable_for_candidate
from engine.builder2_headline_decision_contract import (
    analyze_headline_omit_textual_dependency,
    get_normalized_headline_decision,
)
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
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import (
    WINNER_DEVELOPMENT_SOURCE_OFFLINE_SALVAGE,
    has_failed_winner_attempt_after_paid_call,
    is_valid_persisted_winner_development,
    persist_accepted_winner_development_for_media,
    verify_winner_media_continuation_contract,
)
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
    prepare_and_validate_persisted_winner_offline,
)
from engine.video_jobs_redis import redis_configured

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
    inspect_plan = plan_for_inspect if plan_for_inspect else parsed
    dependency_for_slogan = analyze_headline_omit_textual_dependency(
        inspect_plan,
        winning_judgment=judgment if isinstance(judgment, dict) else None,
    )
    headline_decision = get_normalized_headline_decision(inspect_plan) if inspect_plan else ""
    headline_validation_blocked = bool(
        headline_decision == "omit" and dependency_for_slogan.get("dependencyBeforeClosure")
    )
    slogan_status = _evaluate_single_slogan_inspection_status(
        inspect_plan,
        state=state,
        headline_validation_blocked=headline_validation_blocked,
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
        **slogan_status,
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

    winner_plan = persist_accepted_winner_development_for_media(
        state,
        candidate_id=winner_candidate_id,
        prototype_id=prototype_id,
        winner_plan=winner_plan,
        winning_candidate=winning_candidate,
        winning_judgment=winning_judgment,
        preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
        compatibility_mode=compatibility_mode,
        source=WINNER_DEVELOPMENT_SOURCE_OFFLINE_SALVAGE,
        job_id=job_id or _clean(state.get("jobId")),
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
        save=False,
    )
    verify_winner_media_continuation_contract(
        state,
        job_id=job_id or _clean(state.get("jobId")),
        tournament_id=tournament_id or _clean(state.get("tournamentId")),
    )
    reconcile_winner_development_call_ledger(state)
    meta["accepted"] = True
    logger.info(
        "BUILDER2_WINNER_OFFLINE_SALVAGE_ACCEPTED jobId=%s candidateId=%s prototypeId=%s",
        job_id or _clean(state.get("jobId")),
        winner_candidate_id,
        prototype_id,
    )
    return deepcopy(state.get("winnerDevelopmentPlan") or {}), meta


def _winner_parsed_response_fingerprint(parsed: Dict[str, Any]) -> str:
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _winner_response_identity_fingerprint(state: Dict[str, Any], payload: Dict[str, Any], parsed: Dict[str, Any]) -> str:
    identity = {
        "candidateId": _clean(payload.get("candidateId")),
        "prototypeId": _clean(payload.get("prototypeId")),
        "topLevelKeyCount": payload.get("topLevelKeyCount"),
        "topLevelKeys": payload.get("topLevelKeys"),
        "parsedFingerprint": _winner_parsed_response_fingerprint(parsed),
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _winner_response_character_count(payload: Dict[str, Any], parsed: Dict[str, Any]) -> int:
    stored = payload.get("responseCharCount")
    if isinstance(stored, int) and stored > 0:
        return stored
    return len(json.dumps(parsed, ensure_ascii=False, sort_keys=True))


def _response_fingerprint(state: Dict[str, Any]) -> Optional[str]:
    payload = load_revalidatable_parsed_winner_response(state)
    if not payload:
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return _winner_response_identity_fingerprint(state, payload, parsed)


def _parsed_response_fingerprint(state: Dict[str, Any]) -> Optional[str]:
    payload = load_revalidatable_parsed_winner_response(state)
    if not payload:
        return None
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return _winner_parsed_response_fingerprint(parsed)


def _evaluate_single_slogan_inspection_status(
    plan: Dict[str, Any],
    *,
    state: Dict[str, Any],
    headline_validation_blocked: bool,
) -> Dict[str, Any]:
    if not plan:
        return {
            "singleSloganContractEvaluationStatus": "not_reached",
            "singleSloganContractFailureReason": "plan_unavailable",
            "singleSloganContractSatisfied": None,
            "canonicalCopySatisfiedBy": "",
        }
    if headline_validation_blocked:
        return {
            "singleSloganContractEvaluationStatus": "not_reached",
            "singleSloganContractFailureReason": "headline_decision_validation_blocked",
            "singleSloganContractSatisfied": None,
            "canonicalCopySatisfiedBy": _clean(plan.get("canonicalCopySatisfiedBy")),
        }
    ok, failures = validate_single_slogan_plan_contract(plan, state=state)
    if ok:
        return {
            "singleSloganContractEvaluationStatus": "passed",
            "singleSloganContractFailureReason": "",
            "singleSloganContractSatisfied": True,
            "canonicalCopySatisfiedBy": _clean(plan.get("canonicalCopySatisfiedBy")),
        }
    return {
        "singleSloganContractEvaluationStatus": "failed",
        "singleSloganContractFailureReason": failures[0] if failures else "builder2_single_slogan_contract_failed",
        "singleSloganContractSatisfied": False,
        "canonicalCopySatisfiedBy": _clean(plan.get("canonicalCopySatisfiedBy")),
    }


def _prepare_inspection_plan(
    state: Dict[str, Any],
    *,
    winner_id: str,
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
) -> Dict[str, Any]:
    parsed_payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((parsed_payload or {}).get("parsed") or {})
    if not parsed:
        return {}
    source = build_server_owned_winner_source_reference(
        strategy_foundation=state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {},
        winning_candidate=winning_candidate,
        candidate_id=winner_id,
    )
    working = deepcopy(parsed)
    from engine.builder2_winner_scene_variations_normalization import (
        normalize_continuous_event_scene_variations_for_execution,
    )

    normalize_continuous_event_scene_variations_for_execution(
        working,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        candidate_id=winner_id,
        prototype_id=_clean(working.get("prototypeId") or winning_candidate.get("prototypeId")),
    )
    return prepare_and_validate_persisted_winner_offline(
        working,
        source_reference=source,
        winning_candidate=winning_candidate,
        winning_judgment=winning_judgment,
        tournament_state=state,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
    )


def inspect_offline_winner_salvage_preconditions(state: Dict[str, Any]) -> Dict[str, Any]:
    bucket = reconcile_winner_development_call_ledger(state)
    parsed_payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((parsed_payload or {}).get("parsed") or {})
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    judgment_id = _clean(winner_rec.get("judgmentId"))
    winning_judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    dependency = analyze_headline_omit_textual_dependency(parsed, winning_judgment=winning_judgment)
    parsed_payload_dict = parsed_payload if isinstance(parsed_payload, dict) else {}
    winner_response_character_count = _winner_response_character_count(parsed_payload_dict, parsed) if parsed else 0
    structure = _clean(parsed.get("structureType"))
    scene_meta = parsed.get("continuousEventSceneVariationsNormalization")
    if not isinstance(scene_meta, dict) and structure == "continuous_event":
        from engine.builder2_winner_scene_variations_normalization import describe_scene_variations_metadata

        scene_meta = describe_scene_variations_metadata(parsed)
        scene_meta = {
            **scene_meta,
            "continuousEventNormalizationRequired": bool(scene_meta.get("originalListCount")),
        }
    remaining_errors: list[str] = []
    would_pass = False
    blocked_reason = ""
    if not parsed_payload:
        blocked_reason = "builder2_winner_response_not_persisted"
    elif not winner_id:
        blocked_reason = "builder2_winner_candidate_missing"
    else:
        try:
            _prepare_inspection_plan(
                state,
                winner_id=winner_id,
                winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
                winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else {},
            )
            would_pass = True
        except Builder2TournamentError as exc:
            reason = str(exc.args[0] if exc.args else "")
            remaining_errors.append(reason.split(":", 1)[-1] if reason else reason)
            blocked_reason = reason
    report = inspect_winner_development_recovery_state(state)
    report.update(
        {
            "textualDependencySourceFields": dependency.get("textualDependencySourceFields") or [],
            "exactDependencySourceFields": dependency.get("exactDependencySourceFields") or [],
            "textualDependencyMatchCategories": dependency.get("textualDependencyMatchCategories") or [],
            "textualDependencySafeCategories": dependency.get("textualDependencyMatchCategories") or [],
            "textualDependencyMatches": dependency.get("allTextualDependencyMatches") or [],
            "positiveTextualDependencyMatches": dependency.get("positiveTextualDependencyMatches") or [],
            "negativeTextualDependencyMatches": dependency.get("negativeTextualDependencyMatches") or [],
            "ambiguousTextualDependencyMatches": dependency.get("ambiguousTextualDependencyMatches") or [],
            "dependencyBeforeClosure": bool(dependency.get("dependencyBeforeClosure")),
            "dependencyOnlyOnClosureSlogan": bool(dependency.get("dependencyOnlyOnClosureSlogan")),
            "videoPromptPositiveRenderedTextRequest": bool(
                dependency.get("videoPromptPositiveRenderedTextRequest")
            ),
            "videoPromptNegativeTextInstructionOnly": bool(
                dependency.get("videoPromptNegativeTextInstructionOnly")
            ),
            "videoPromptRequestsRenderedText": bool(dependency.get("videoPromptRequestsRenderedText")),
            "silentVisualUnderstandable": bool(dependency.get("silentVisualUnderstandable")),
            "headlineFieldsRequired": bool(dependency.get("headlineFieldsRequired")),
            "continuousEventNormalizationRequired": structure == "continuous_event"
            and bool((scene_meta or {}).get("originalListCount")),
            "remainingValidationErrorsAfterNormalization": remaining_errors,
            "wouldPassCorrectedHeadlineContract": not bool(dependency.get("dependencyBeforeClosure")),
            "offlineWinnerSalvagePossible": would_pass and not is_valid_persisted_winner_development(state),
            "offlineWinnerSalvageBlockedReason": blocked_reason or None,
            "winnerResponseCharacterCount": winner_response_character_count,
            "winnerResponseFingerprint": _response_fingerprint(state),
            "winnerParsedResponseFingerprint": _parsed_response_fingerprint(state),
            "winnerPaidCallCount": int(bucket.get("paidDispatchCount") or 0),
            "openAICalls": 0,
            "stateMutated": False,
            "paidCalls": 0,
        }
    )
    return report


def run_offline_winner_salvage_for_job(
    job_id: str,
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
    save_state: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": _clean(job_id),
        "ok": False,
        "stateMutated": False,
        "openAICalls": 0,
        "paidCalls": 0,
    }
    if tournament_state is None:
        if not redis_configured():
            report["failureReason"] = "builder2_winner_offline_salvage_redis_unconfigured"
            return report
        state = load_tournament_state(job_id)
    else:
        state = tournament_state
    if not isinstance(state, dict) or not state:
        report["failureReason"] = "builder2_winner_offline_salvage_job_not_found"
        return report

    report.update(inspect_offline_winner_salvage_preconditions(state))
    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    judgment_id = _clean(winner_rec.get("judgmentId"))
    winning_judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    prototype_id = _clean(winner_rec.get("prototypeId") or state.get("winnerDevelopmentPrototypeId"))

    if is_valid_persisted_winner_development(state):
        report["ok"] = True
        report["winnerDevelopmentAccepted"] = True
        report["reusedAccepted"] = True
        populate_winner_development_call_report(state, report)
        return report

    try:
        winner_plan, _meta = attempt_offline_winner_development_salvage(
            state,
            winner_candidate_id=winner_id,
            prototype_id=prototype_id,
            strategy_foundation=state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {},
            winning_candidate=winning_candidate if isinstance(winning_candidate, dict) else {},
            winning_judgment=winning_judgment if isinstance(winning_judgment, dict) else {},
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
            job_id=job_id,
            tournament_id=_clean(state.get("tournamentId")),
        )
    except Builder2TournamentError as exc:
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_winner_offline_salvage_invalid")
        report["offlineWinnerSalvageBlockedReason"] = report["failureReason"]
        return report

    state["offlineWinnerSalvageAt"] = _clean(state.get("offlineWinnerSalvageAt")) or state.get("winnerDevelopmentAcceptedAt")
    state["offlineWinnerSalvageVersion"] = "builder2_headline_omit_dependency_v1"
    if save_state and tournament_state is None:
        save_tournament_state(job_id, state)
    report["ok"] = True
    report["stateMutated"] = save_state and tournament_state is None
    report["winnerDevelopmentAccepted"] = is_valid_persisted_winner_development(state)
    report["stoppedBeforeMedia"] = True
    report["nextStage"] = "media_prerequisite_validation"
    populate_winner_development_call_report(state, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = list(argv or sys.argv[1:])
    mode = "salvage"
    if args and args[0] == "inspect":
        mode = "inspect"
        args = args[1:]
    job_id = _clean(
        os.environ.get("BUILDER2_WINNER_OFFLINE_SALVAGE_INSPECT_JOB_ID" if mode == "inspect" else "BUILDER2_WINNER_OFFLINE_SALVAGE_JOB_ID")
        or (args[0] if args else "")
    )
    if not job_id:
        print(
            json.dumps(
                {
                    "ok": False,
                    "failureReason": "builder2_winner_offline_salvage_job_id_missing",
                },
                indent=2,
            )
        )
        return 1
    if mode == "inspect":
        if not redis_configured():
            print(json.dumps({"ok": False, "failureReason": "builder2_winner_offline_salvage_redis_unconfigured"}, indent=2))
            return 1
        state = load_tournament_state(job_id)
        if state is None:
            print(json.dumps({"ok": False, "failureReason": "builder2_winner_offline_salvage_job_not_found"}, indent=2))
            return 1
        report = inspect_offline_winner_salvage_preconditions(state)
        report["ok"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = run_offline_winner_salvage_for_job(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())


def assert_no_duplicate_paid_winner_development(state: Dict[str, Any], *, winner_candidate_id: str) -> None:
    if is_valid_persisted_winner_development(state):
        return
    if parsed_winner_reusable_for_candidate(state, winner_candidate_id=winner_candidate_id):
        return
    if winner_development_dispatch_count(state) >= 1:
        raise Builder2TournamentError("builder2_winner_additional_paid_call_requires_approval")
    if state.get("winnerDevelopmentPaidCallRecorded"):
        raise Builder2TournamentError("builder2_winner_additional_paid_call_requires_approval")
