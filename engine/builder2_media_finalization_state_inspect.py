"""
Builder2 media finalization state inspector — read-only eligibility and persistence audit.

Run:
  BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_JOB_ID=<jobId> \\
    python -m engine.builder2_media_finalization_state_inspect

Optional final URL fields (read-only; default off):
  BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_INCLUDE_FINAL_URL=true
"""
from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_advertising_closure_contract import advertising_closure_is_required
from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_media_finalization_contract import (
    assess_false_completion,
    assess_recoverable_failed_finalization_state,
    closure_inclusive_artifact_valid,
    evaluate_finalization_recovery_eligibility,
    resolve_legacy_headline_artifact_url,
    resolve_raw_runway_artifact_url,
    validate_builder2_media_completion_contract,
)
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

_LEGACY_FALSE_COMPLETION_SIGNATURE: Dict[str, Any] = {
    "persistedTournamentStatus": "completed",
    "persistedMediaResumeStatus": "completed",
    "mediaContinuationRequired": False,
    "advertisingClosureRendered": False,
    "advertisingClosureStatusCompleted": False,
    "finalUrlRouteFamily": "api/video-headline",
    "rawRunwayReferencePresent": True,
}

_POST_FAILED_RECOVERY_SIGNATURE: Dict[str, Any] = {
    "persistedTournamentStatus": "media_finalization_incomplete",
    "persistedMediaResumeStatus": "finalization_failed",
    "mediaContinuationRequired": True,
    "advertisingClosureStatusFailed": True,
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _first_url(*values: Any) -> str:
    for value in values:
        token = _clean(value)
        if token:
            return token
    return ""


def _compare_urls(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def _url_excludes_raw_runway(url: str, raw_url: str) -> Optional[str]:
    token = _clean(url)
    if not token:
        return None
    if raw_url and _compare_urls(token, raw_url):
        return None
    return token


def _optional_final_url_fields(
    *,
    final_public: str,
    closure_url: str,
    job_video_url: str,
    raw_url: str,
    include_final_url: bool,
) -> Dict[str, Optional[str]]:
    if not include_final_url:
        return {}
    return {
        "finalPublicUrl": _url_excludes_raw_runway(final_public, raw_url),
        "finalVideoWithClosureUrl": _url_excludes_raw_runway(closure_url, raw_url),
        "jobVideoUrl": _url_excludes_raw_runway(job_video_url, raw_url),
    }


def _duration_value(media: Dict[str, Any], key: str) -> Optional[float]:
    raw = media.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _route_family(url: str) -> str:
    token = _clean(url)
    if not token:
        return "missing"
    return classify_url_route_family(token)


def _evaluate_false_completion_conditions(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
) -> Dict[str, bool]:
    media = _media_bucket(state)
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = headline_decision_requires_headline(headline_decision)
    closure_required = advertising_closure_is_required(plan) or bool(
        _clean((state.get("advertisingClosure") or {}).get("sloganText"))
    )
    raw_url = resolve_raw_runway_artifact_url(state)
    headline_url = resolve_legacy_headline_artifact_url(
        state=state,
        job_video_url=job_video_url,
        headline_required=headline_required,
    )
    closure_url = _first_url(media.get("finalVideoWithClosureUrl"))
    valid_closure = closure_inclusive_artifact_valid(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )
    persisted_status_completed = _clean(state.get("status")) == "completed"
    return {
        "persistedStatusCompleted": persisted_status_completed,
        "closureRequired": closure_required,
        "validClosureInclusiveArtifact": valid_closure,
        "closureRequiredButInvalidArtifact": closure_required and not valid_closure,
        "closureMarkedCompletedWithoutValidArtifact": (
            closure_required
            and _clean(media.get("advertisingClosureStatus")) == "completed"
            and not valid_closure
        ),
        "headlineRequired": headline_required,
        "headlineArtifactIdentified": bool(headline_url),
        "headlineRequiredButMissingArtifact": headline_required and not headline_url,
        "jobVideoUrlPointsToHeadlineOnly": bool(
            job_video_url
            and headline_url
            and _compare_urls(job_video_url, headline_url)
            and closure_required
            and not valid_closure
        ),
        "finalClosureUrlMatchesRawRunway": bool(closure_url and raw_url and _compare_urls(closure_url, raw_url)),
        "finalClosureUrlMatchesHeadlineOnly": bool(
            closure_url and headline_url and _compare_urls(closure_url, headline_url) and closure_required
        ),
        "falseCompletionProven": persisted_status_completed and bool(
            assess_false_completion(state=state, plan=plan, job_video_url=job_video_url)[0]
        ),
    }


def _evaluate_eligibility_conditions(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
) -> Dict[str, bool]:
    evaluation = evaluate_finalization_recovery_eligibility(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
    )
    failed_recovery = assess_recoverable_failed_finalization_state(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
    )
    return {
        "reasoningComplete": bool(state.get("reasoningComplete")),
        "winnerDevelopmentAccepted": is_valid_persisted_winner_development(state),
        "advertisingClosurePresent": bool(
            evaluation.get("recoverableFailedFinalizationConditionResults", {}).get("advertisingClosurePresent")
            or isinstance(state.get("advertisingClosure"), dict)
        ),
        "falseCompletionProven": bool(evaluation.get("legacyFalseCompletionConfirmed")),
        "recoverableFailedFinalizationProven": bool(evaluation.get("recoverableFailedFinalizationConfirmed")),
        "recoveryBasisProven": bool(evaluation.get("recoveryEligibilityBasis")),
        "visualOrHeadlineIntermediatePresent": not bool(evaluation.get("recoveryBlockedByMissingIntermediate")),
        "validClosureAlreadyPresent": bool(evaluation.get("recoveryBlockedByValidFinal")),
        "completedFinalPublicationPresent": bool(evaluation.get("recoveryBlockedByCompletedPublication")),
        **failed_recovery.condition_results,
    }


def _classify_publication_evidence(
    *,
    state: Dict[str, Any],
    media: Dict[str, Any],
    job_raw: Dict[str, Any],
    raw_url: str,
    headline_url: str,
    closure_url: str,
    final_public: str,
    job_video_url: str,
) -> str:
    closure_rendered = bool(media.get("advertisingClosureRendered"))
    closure_status = _clean(media.get("advertisingClosureStatus"))
    job_status = _clean(job_raw.get("status"))
    measured_final = _duration_value(media, "actualFinalVideoDurationSeconds")
    final_route = _route_family(closure_url or final_public or job_video_url)
    if closure_rendered and closure_status == "completed" and measured_final is not None:
        if final_route != "api/video-headline" or not _compare_urls(closure_url or final_public, headline_url):
            return "final_publication_persisted"
    if closure_url and not _compare_urls(closure_url, headline_url) and final_route != "api/video-headline":
        return "publication_reference_persisted"
    if closure_status == "failed" and _clean(state.get("status")) == "media_finalization_incomplete":
        return "proven_not_published"
    if job_status == "done" and _compare_urls(job_video_url, headline_url):
        return "proven_not_published"
    if not closure_rendered and not measured_final and _compare_urls(closure_url or final_public, headline_url):
        return "proven_not_published"
    return "unknown"


def _recommend_next_action(
    *,
    eligible: bool,
    false_completion: bool,
    recoverable_failed_finalization: bool,
    raw_url: str,
    contract_ok: bool,
    publication_evidence: str,
    state_changed: bool,
    recovery_failure_present: bool,
) -> str:
    if contract_ok and publication_evidence == "final_publication_persisted":
        return "use_existing_verified_final"
    if eligible and recoverable_failed_finalization:
        return "run_finalization_preflight"
    if eligible and false_completion:
        return "run_finalization_preflight"
    if raw_url and recovery_failure_present and not contract_ok and not eligible:
        return "allow_recovery_from_failed_finalization_state"
    if state_changed and raw_url and not eligible and recovery_failure_present:
        return "allow_recovery_from_failed_finalization_state"
    if state_changed and raw_url and not eligible:
        return "clear_stale_recovery_metadata_only"
    if publication_evidence == "publication_reference_persisted":
        return "inspect_publication_storage"
    if not raw_url:
        return "unrecoverable_without_regeneration"
    return "insufficient_evidence"


def _compare_signature(
    *,
    state: Dict[str, Any],
    media: Dict[str, Any],
    raw_url: str,
    final_public: str,
    signature: Dict[str, Any],
) -> Dict[str, bool]:
    final_route = _route_family(final_public)
    return {
        "persistedTournamentStatus": (_clean(state.get("status")) == signature.get("persistedTournamentStatus")),
        "persistedMediaResumeStatus": (
            _clean(media.get("mediaResumeStatus")) == signature.get("persistedMediaResumeStatus")
        ),
        "mediaContinuationRequired": bool(state.get("mediaContinuationRequired"))
        == bool(signature.get("mediaContinuationRequired")),
        "advertisingClosureRendered": bool(media.get("advertisingClosureRendered"))
        == bool(signature.get("advertisingClosureRendered")),
        "advertisingClosureStatusCompleted": (
            _clean(media.get("advertisingClosureStatus")) == "completed"
            if signature.get("advertisingClosureStatusCompleted") is False
            else _clean(media.get("advertisingClosureStatus")) == "completed"
        ),
        "advertisingClosureStatusFailed": (
            _clean(media.get("advertisingClosureStatus")) == "failed"
            if signature.get("advertisingClosureStatusFailed")
            else _clean(media.get("advertisingClosureStatus")) != "failed"
        ),
        "finalUrlRouteFamily": final_route == signature.get("finalUrlRouteFamily"),
        "rawRunwayReferencePresent": bool(raw_url) == bool(signature.get("rawRunwayReferencePresent")),
    }


def inspect_builder2_media_finalization_state(
    job_id: str,
    *,
    include_final_url: Optional[bool] = None,
) -> Dict[str, Any]:
    if include_final_url is None:
        include_final_url = _truthy("BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_INCLUDE_FINAL_URL")
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "inspectionCompleted": False,
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingCalls": 0,
        "ffmpegCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
        "leaseOperations": 0,
    }
    if not redis_configured():
        report["failureReason"] = "builder2_media_finalization_state_inspect_redis_unconfigured"
        return report

    with read_only_builder2_inspection() as mutation_counter:
        raw_state = _read_raw(job_id)
        job_raw = video_job_get_raw(job_id) or {}
        report["redisMutations"] = mutation_counter.redis_mutations
        if raw_state is None:
            report["jobFound"] = False
            report["tournamentFound"] = False
            report["inspectionCompleted"] = True
            report["ok"] = True
            return report

        state = deepcopy(raw_state)
        media = _media_bucket(state)
        plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
        job_video_url = _first_url(job_raw.get("video_url"), job_raw.get("videoUrl"))
        raw_url = resolve_raw_runway_artifact_url(state)
        headline_required = headline_decision_requires_headline(get_normalized_headline_decision(plan))
        headline_url = resolve_legacy_headline_artifact_url(
            state=state,
            job_video_url=job_video_url,
            headline_required=headline_required,
        )
        closure_url = _first_url(media.get("finalVideoWithClosureUrl"))
        final_public = _first_url(media.get("finalPublicUrl"))

        eligible_eval = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        eligible = bool(eligible_eval["eligible"])
        missing = list(eligible_eval["missing"])
        false_completion = bool(eligible_eval["legacyFalseCompletionConfirmed"])
        recoverable_failed_finalization = bool(eligible_eval["recoverableFailedFinalizationConfirmed"])
        false_completion_legacy, false_reasons = assess_false_completion(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        _ = false_completion_legacy
        contract_ok, contract_failure, contract_failures = validate_builder2_media_completion_contract(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        false_conditions = _evaluate_false_completion_conditions(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        eligibility_conditions = _evaluate_eligibility_conditions(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )

        legacy_match = _compare_signature(
            state=state,
            media=media,
            raw_url=raw_url,
            final_public=final_public or closure_url or job_video_url,
            signature=_LEGACY_FALSE_COMPLETION_SIGNATURE,
        )
        post_recovery_match = _compare_signature(
            state=state,
            media=media,
            raw_url=raw_url,
            final_public=final_public or closure_url or job_video_url,
            signature=_POST_FAILED_RECOVERY_SIGNATURE,
        )
        changed_conditions = [
            name
            for name, matched in legacy_match.items()
            if not matched and name in {"persistedTournamentStatus", "mediaContinuationRequired", "advertisingClosureStatusCompleted"}
        ]
        if _clean(media.get("advertisingClosureStatus")) == "failed":
            changed_conditions.append("advertisingClosureStatusFailed")

        publication_evidence = _classify_publication_evidence(
            state=state,
            media=media,
            job_raw=job_raw,
            raw_url=raw_url,
            headline_url=headline_url,
            closure_url=closure_url,
            final_public=final_public,
            job_video_url=job_video_url,
        )
        recovery_failure_present = (
            _clean(media.get("mediaResumeStatus")) == "finalization_failed"
            or _clean(media.get("advertisingClosureStatus")) == "failed"
            or _clean(state.get("status")) == "media_finalization_incomplete"
        )
        eligibility_reason = None
        if not eligible:
            eligibility_reason = ",".join(missing) if missing else None
        elif recoverable_failed_finalization:
            eligibility_reason = "recoverable_failed_finalization_state"
        elif false_completion:
            eligibility_reason = "legacy_false_completion"

        report.update(
            {
                "jobFound": True,
                "tournamentFound": True,
                "persistedJobStatus": _clean(job_raw.get("status")) or None,
                "persistedTournamentStatus": _clean(state.get("status")) or None,
                "persistedCompletionStatus": _clean(media.get("mediaResumeStatus")) or None,
                "effectiveCompletionStatus": "completed" if contract_ok else "incomplete",
                "currentEligibility": eligible,
                "currentEligibilityReason": eligibility_reason,
                "falseCompletionConfirmed": false_completion,
                "recoverableFailedFinalizationConfirmed": recoverable_failed_finalization,
                "recoveryEligibilityBasis": eligible_eval.get("recoveryEligibilityBasis"),
                "recoverableFailedFinalizationConditionResults": eligible_eval.get(
                    "recoverableFailedFinalizationConditionResults"
                ),
                "recoverableFailedFinalizationReasons": eligible_eval.get("recoverableFailedFinalizationReasons"),
                "recoveryBlockedByValidFinal": bool(eligible_eval.get("recoveryBlockedByValidFinal")),
                "recoveryBlockedByCompletedPublication": bool(
                    eligible_eval.get("recoveryBlockedByCompletedPublication")
                ),
                "recoveryBlockedByMissingIntermediate": bool(
                    eligible_eval.get("recoveryBlockedByMissingIntermediate")
                ),
                "falseCompletionConditionResults": false_conditions,
                "eligibilityConditionResults": eligibility_conditions,
                "falseCompletionReasons": false_reasons,
                "rawRunwayReferencePresent": bool(raw_url),
                "rawRunwayRouteFamily": _route_family(raw_url),
                "rawRunwayDurationPresent": _duration_value(media, "rawRunwayDurationSeconds") is not None,
                "rawRunwayDurationSeconds": _duration_value(media, "rawRunwayDurationSeconds"),
                "finalPublicUrlPresent": bool(final_public),
                "finalPublicUrlRouteFamily": _route_family(final_public),
                "finalVideoWithClosureUrlPresent": bool(closure_url),
                "finalVideoWithClosureRouteFamily": _route_family(closure_url),
                "jobVideoUrlPresent": bool(job_video_url),
                "jobVideoUrlRouteFamily": _route_family(job_video_url),
                "finalPublicEqualsClosureUrl": _compare_urls(final_public, closure_url),
                "finalPublicEqualsJobVideoUrl": _compare_urls(final_public, job_video_url),
                "closureUrlEqualsJobVideoUrl": _compare_urls(closure_url, job_video_url),
                "finalUrlEqualsRawRunway": _compare_urls(closure_url or final_public, raw_url),
                "finalUrlEqualsLegacyHeadlineArtifact": _compare_urls(
                    closure_url or final_public or job_video_url,
                    headline_url,
                ),
                "legacyHeadlineArtifactReferencePresent": bool(headline_url),
                "legacyHeadlineRouteFamily": _route_family(headline_url),
                "advertisingClosureRendered": bool(media.get("advertisingClosureRendered")),
                "advertisingClosureStatus": _clean(media.get("advertisingClosureStatus")) or None,
                "headlineReconstructionCompleted": bool(media.get("headlineReconstructionCompleted")),
                "headlineArtifactSource": _clean(media.get("headlineArtifactSource")) or None,
                "closureDurationMetadataPresent": _duration_value(
                    state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {},
                    "durationSeconds",
                )
                is not None,
                "finalDurationMetadataPresent": _duration_value(media, "actualFinalVideoDurationSeconds") is not None,
                "finalDurationAcceptedPersisted": (
                    _duration_value(media, "actualFinalVideoDurationSeconds") is not None
                    and contract_ok
                ),
                "recoveryAttemptMetadataPresent": recovery_failure_present,
                "recoveryFailureMetadataPresent": recovery_failure_present,
                "recoveryFailureStage": _clean(media.get("finalizationFailureStage")) or None,
                "recoveryFailureCode": _clean(media.get("finalizationFailureCode")) or None,
                "recoveryFailureClass": _clean(media.get("finalizationFailureClass")) or None,
                "recoveryStartedAtPresent": bool(media.get("finalizationRecoveryStartedAt")),
                "recoveryCompletedAtPresent": bool(media.get("finalizationRecoveryCompletedAt")),
                "publicationReferencePresent": publication_evidence
                in {"publication_reference_persisted", "final_publication_persisted"},
                "publicationRouteFamily": _route_family(closure_url or final_public),
                "publicationCompletedPersisted": publication_evidence == "final_publication_persisted",
                "publicationEvidenceClassification": publication_evidence,
                "completionContractSatisfied": contract_ok,
                "completionContractFailureReasons": contract_failures,
                "completionContractPrimaryFailure": contract_failure or None,
                "legacyFalseCompletionSignatureMatch": legacy_match,
                "postFailedRecoverySignatureMatch": post_recovery_match,
                "stateChangedFromKnownLegacyPattern": bool(changed_conditions),
                "changedConditionNames": changed_conditions,
                "likelyMutationSourceFunction": (
                    "run_one_media_finalization_resume.save_tournament_state_on_render_failure"
                    if recovery_failure_present and not legacy_match.get("persistedTournamentStatus")
                    else None
                ),
                "recommendedNextAction": _recommend_next_action(
                    eligible=eligible,
                    false_completion=false_completion,
                    recoverable_failed_finalization=recoverable_failed_finalization,
                    raw_url=raw_url,
                    contract_ok=contract_ok,
                    publication_evidence=publication_evidence,
                    state_changed=bool(changed_conditions),
                    recovery_failure_present=recovery_failure_present,
                ),
                "minimalFutureStateRepairFields": (
                    []
                    if eligible
                    else (
                        ["status", "mediaContinuationRequired", "mediaResume.mediaResumeStatus", "mediaResume.advertisingClosureStatus"]
                        if recovery_failure_present and "recoveryBasisNotProven" in missing
                        else []
                    )
                ),
                "rawRunwayRecoverable": bool(raw_url),
                "unlinkedFinalArtifactPossible": publication_evidence == "unknown" and bool(raw_url),
                "jobMarkedDone": _clean(job_raw.get("status")) == "done",
                "jobVideoUrlReplacedWithClosureFinal": bool(
                    job_video_url and closure_url and _compare_urls(job_video_url, closure_url) and not _compare_urls(job_video_url, headline_url)
                ),
                "inspectionCompleted": True,
                "ok": True,
                **_optional_final_url_fields(
                    final_public=final_public,
                    closure_url=closure_url,
                    job_video_url=job_video_url,
                    raw_url=raw_url,
                    include_final_url=include_final_url,
                ),
            }
        )
        return report


def print_builder2_media_finalization_state_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_JOB_ID"))
    if not job_id:
        print(
            json.dumps(
                {"ok": False, "failureReason": "builder2_media_finalization_state_inspect_job_id_missing"},
                indent=2,
            ),
            flush=True,
        )
        return 1
    logger.info("BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_START jobId=%s", job_id)
    report = inspect_builder2_media_finalization_state(job_id)
    print_builder2_media_finalization_state_report(report)
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_DONE jobId=%s ok=%s eligible=%s falseCompletion=%s recoverableFailedFinalization=%s",
        job_id,
        report.get("ok"),
        report.get("currentEligibility"),
        report.get("falseCompletionConfirmed"),
        report.get("recoverableFailedFinalizationConfirmed"),
    )
    return 0 if report.get("inspectionCompleted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
