"""
Builder2 media finalization contract — completion gate and legacy artifact recovery rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from engine.builder2_advertising_closure_contract import advertising_closure_is_required
from engine.builder2_closure_render import (
    classify_url_route_family,
    url_fingerprint,
    verify_builder2_final_video_duration,
)
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_single_slogan_contract import builder2_requires_headline_overlay
from engine.builder2_new_format_config import (
    FINAL_DURATION_TOLERANCE_SECONDS,
    resolve_builder2_end_card_duration_seconds,
    resolve_builder2_final_video_duration_seconds,
    resolve_builder2_video_duration_seconds,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development


def _clean(value: Any) -> str:
    return str(value or "").strip()


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


def _duration_value(media: Dict[str, Any], key: str) -> Optional[float]:
    raw = media.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def resolve_raw_runway_artifact_url(state: Dict[str, Any]) -> str:
    media = _media_bucket(state)
    runway = state.get("runway") if isinstance(state.get("runway"), dict) else {}
    return _first_url(
        media.get("rawRunwayVideoUrl"),
        media.get("rawRunwayVideoPath"),
        media.get("runwayVideoUrl"),
        media.get("downloadedVideoPath"),
        runway.get("videoUrl"),
    )


def is_recognized_headline_route(url: str) -> bool:
    return classify_url_route_family(url) == "api/video-headline"


def is_builder2_durable_final_video_route(url: str) -> bool:
    return classify_url_route_family(url) == "api/builder2-final-video"


def final_publication_metadata_valid(*, media: Dict[str, Any], closure_url: str) -> bool:
    if not closure_url:
        return False
    if not is_builder2_durable_final_video_route(closure_url):
        return False
    if media.get("finalPublicationVerificationAccepted") is not True:
        return False
    if media.get("finalPublicationDurableStorageConfirmed") is not True:
        return False
    backend = _clean(media.get("finalPublicationBackendKind"))
    if backend and backend == "ephemeral_tmp":
        return False
    return True


def resolve_legacy_headline_artifact_url(
    *,
    state: Dict[str, Any],
    job_video_url: str = "",
    headline_required: bool = False,
) -> str:
    media = _media_bucket(state)
    if media.get("headlineReconstructionCompleted"):
        return ""
    explicit = _first_url(
        media.get("headlineArtifactUrl"),
        media.get("headlineVideoUrl"),
        media.get("headlineOverlayUrl"),
        state.get("headlineArtifactUrl"),
    )
    if explicit:
        return explicit
    if not headline_required:
        return ""
    raw_url = resolve_raw_runway_artifact_url(state)
    candidate = _first_url(
        media.get("headlineArtifactUrl"),
        job_video_url,
        media.get("finalPublicUrl"),
        media.get("finalVideoWithClosureUrl"),
        media.get("finalVideoPath"),
    )
    if not candidate or not is_recognized_headline_route(candidate):
        return ""
    if raw_url and _compare_urls(candidate, raw_url):
        return ""
    measured_final = _duration_value(media, "actualFinalVideoDurationSeconds")
    configured_final = _duration_value(media, "finalVideoDurationSeconds")
    if measured_final is not None and abs(measured_final - resolve_builder2_final_video_duration_seconds()) <= FINAL_DURATION_TOLERANCE_SECONDS:
        return ""
    if (
        configured_final is not None
        and abs(configured_final - resolve_builder2_final_video_duration_seconds()) <= FINAL_DURATION_TOLERANCE_SECONDS
        and media.get("advertisingClosureRendered") is True
        and media.get("advertisingClosureStatus") == "completed"
        and not _compare_urls(candidate, raw_url)
    ):
        return ""
    return candidate


def closure_inclusive_artifact_valid(
    *,
    state: Dict[str, Any],
    closure_url: str,
    raw_url: str,
    headline_url: str,
    job_video_url: str = "",
) -> bool:
    if not closure_url:
        return False
    if raw_url and _compare_urls(closure_url, raw_url):
        return False
    if headline_url and _compare_urls(closure_url, headline_url):
        return False
    media = _media_bucket(state)
    if not media.get("advertisingClosureRendered"):
        return False
    if _clean(media.get("advertisingClosureStatus")) != "completed":
        return False
    if not final_publication_metadata_valid(media=media, closure_url=closure_url):
        return False
    measured = _duration_value(media, "actualFinalVideoDurationSeconds")
    if measured is not None:
        verify_kwargs: dict[str, float] = {}
        for media_key, param in (
            ("headlineReconstructionDurationSeconds", "visual_duration_seconds"),
            ("actualVisualDurationSeconds", "visual_duration_seconds"),
        ):
            visual = _duration_value(media, media_key)
            if visual is not None:
                verify_kwargs["visual_duration_seconds"] = visual
                break
        closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
        end_card = _duration_value(closure, "durationSeconds")
        if end_card is not None:
            verify_kwargs["end_card_duration_seconds"] = end_card
        try:
            verify_builder2_final_video_duration(measured, **verify_kwargs)
            return True
        except Exception:
            return False
    if is_recognized_headline_route(closure_url):
        if headline_url and _compare_urls(closure_url, headline_url):
            return False
        if job_video_url and _compare_urls(closure_url, job_video_url) and headline_url and _compare_urls(job_video_url, headline_url):
            return False
        return False
    return bool(closure_url) and not _compare_urls(closure_url, raw_url)


def assess_false_completion(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
) -> Tuple[bool, List[str]]:
    media = _media_bucket(state)
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = builder2_requires_headline_overlay(plan=plan, state=state)
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
    final_public = _first_url(media.get("finalPublicUrl"))
    valid_closure = closure_inclusive_artifact_valid(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )
    reasons: List[str] = []
    if closure_required and not valid_closure:
        reasons.append("closure_required_but_closure_inclusive_artifact_missing_or_invalid")
    if closure_required and media.get("advertisingClosureStatus") == "completed" and not valid_closure:
        reasons.append("advertising_closure_marked_rendered_without_distinct_closure_artifact")
    if valid_closure and final_publication_metadata_valid(media=media, closure_url=closure_url):
        return False, []
    if headline_required and not headline_url and not media.get("headlineReconstructionCompleted"):
        reasons.append("headline_required_but_headline_artifact_not_identified")
    if job_video_url and headline_url and _compare_urls(job_video_url, headline_url) and closure_required and not valid_closure:
        reasons.append("job_video_url_points_to_headline_artifact_not_closure_inclusive_final")
    if closure_url and raw_url and _compare_urls(closure_url, raw_url):
        reasons.append("finalVideoWithClosureUrl_matches_raw_runway_artifact")
    if closure_url and headline_url and _compare_urls(closure_url, headline_url) and closure_required:
        reasons.append("finalVideoWithClosureUrl_matches_headline_artifact_only")
    configured_final = _duration_value(media, "finalVideoDurationSeconds")
    if (
        closure_required
        and configured_final is not None
        and _clean(media.get("actualFinalVideoDurationSeconds")) == ""
        and is_recognized_headline_route(_first_url(final_public, closure_url, job_video_url))
    ):
        reasons.append("configured_final_duration_metadata_not_trusted_without_actual_probe")
    if _clean(state.get("status")) == "completed" and reasons:
        reasons.append("persisted_status_completed_despite_contract_violation")
    false_completion = _clean(state.get("status")) == "completed" and bool(reasons)
    return false_completion, reasons


def validate_builder2_media_completion_contract(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
    require_job_video_url_match: bool = True,
) -> Tuple[bool, str, List[str]]:
    media = _media_bucket(state)
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = builder2_requires_headline_overlay(plan=plan, state=state)
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
    final_public = _first_url(media.get("finalPublicUrl"))
    failures: List[str] = []
    if closure_required and _clean(media.get("advertisingClosureStatus")) != "completed":
        failures.append("advertising_closure_not_completed")
    if closure_required and not media.get("advertisingClosureRendered"):
        failures.append("advertising_closure_not_rendered")
    if not closure_url:
        failures.append("finalVideoWithClosureUrl_missing")
    if final_public and closure_url and not _compare_urls(final_public, closure_url):
        failures.append("finalPublicUrl_mismatch")
    if raw_url and closure_url and _compare_urls(closure_url, raw_url):
        failures.append("final_url_is_raw_runway")
    if headline_url and closure_url and _compare_urls(closure_url, headline_url) and closure_required:
        failures.append("final_url_is_headline_only")
    if require_job_video_url_match and job_video_url and closure_url and not _compare_urls(job_video_url, closure_url):
        failures.append("job_video_url_not_closure_inclusive")
    if headline_required and not headline_url and not media.get("headlineReconstructionCompleted"):
        failures.append("headline_artifact_missing")
    if closure_url and not is_builder2_durable_final_video_route(closure_url):
        failures.append("final_publication_route_not_durable")
    if not media.get("finalPublicationVerificationAccepted"):
        failures.append("final_publication_verification_failed")
    if not media.get("finalPublicationDurableStorageConfirmed"):
        failures.append("final_publication_not_durable")
    measured = _duration_value(media, "actualFinalVideoDurationSeconds")
    if measured is None:
        failures.append("actual_final_duration_missing")
    else:
        verify_kwargs: dict[str, float] = {}
        for media_key, _param in (
            ("headlineReconstructionDurationSeconds", "visual_duration_seconds"),
            ("actualVisualDurationSeconds", "visual_duration_seconds"),
        ):
            visual = _duration_value(media, media_key)
            if visual is not None:
                verify_kwargs["visual_duration_seconds"] = visual
                break
        closure_obj = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
        end_card = _duration_value(closure_obj, "durationSeconds")
        if end_card is not None:
            verify_kwargs["end_card_duration_seconds"] = end_card
        try:
            verify_builder2_final_video_duration(measured, **verify_kwargs)
        except Exception as exc:
            failures.append(str(getattr(exc, "args", ["builder2_media_final_duration_invalid"])[0]))
    if not closure_inclusive_artifact_valid(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    ):
        failures.append("closure_inclusive_artifact_invalid")

    from engine.builder2_single_slogan_contract import validate_single_slogan_completion

    failures.extend(
        validate_single_slogan_completion(
            state=state,
            plan=plan,
            media=media,
        )
    )
    from engine.builder2_no_logo_contract import validate_no_logo_completion

    failures.extend(
        validate_no_logo_completion(
            state=state,
            plan=plan,
            media=media,
        )
    )
    from engine.builder2_music_artifact_publication import durable_music_reference_present

    if durable_music_reference_present(media) or (
        str(media.get("musicGenerationStatus") or "").lower() == "succeeded"
        and _clean(media.get("musicArtifactUrl"))
    ):
        if media.get("finalVideoHasAudioStream") is not True:
            failures.append("builder2_final_video_missing_audio_stream")
    if failures:
        return False, failures[0], failures
    return True, "", []


def _completed_final_publication_present(
    *,
    state: Dict[str, Any],
    closure_url: str,
    raw_url: str,
    headline_url: str,
    job_video_url: str = "",
) -> bool:
    return closure_inclusive_artifact_valid(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )


def _conflicting_publication_evidence(
    *,
    media: Dict[str, Any],
    closure_url: str,
    final_public: str,
    headline_url: str,
    raw_url: str,
    valid_closure: bool,
) -> bool:
    if valid_closure:
        return False
    if media.get("advertisingClosureRendered") is True and _clean(media.get("advertisingClosureStatus")) == "completed":
        return True
    if (
        closure_url
        and final_public
        and not _compare_urls(closure_url, final_public)
        and not _compare_urls(closure_url, headline_url)
        and not _compare_urls(closure_url, raw_url)
        and not is_recognized_headline_route(closure_url)
    ):
        return True
    measured = _duration_value(media, "actualFinalVideoDurationSeconds")
    if measured is not None and not valid_closure and media.get("advertisingClosureRendered") is True:
        return True
    return False


_UNRECOVERABLE_STATUSES = frozenset({"canceled", "cancelled", "unrecoverable", "abandoned"})


@dataclass(frozen=True)
class RecoverableFailedFinalizationAssessment:
    recoverable: bool
    reasons: List[str] = field(default_factory=list)
    condition_results: Dict[str, bool] = field(default_factory=dict)
    recovery_basis: str = ""
    blocking_conditions: List[str] = field(default_factory=list)


def assess_recoverable_failed_finalization_state(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
    active_finalization_lease: bool = False,
) -> RecoverableFailedFinalizationAssessment:
    media = _media_bucket(state)
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = builder2_requires_headline_overlay(plan=plan, state=state)
    raw_url = resolve_raw_runway_artifact_url(state)
    headline_url = resolve_legacy_headline_artifact_url(
        state=state,
        job_video_url=job_video_url,
        headline_required=headline_required,
    )
    closure_url = _first_url(media.get("finalVideoWithClosureUrl"))
    final_public = _first_url(media.get("finalPublicUrl"))
    valid_closure = closure_inclusive_artifact_valid(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )
    completed_publication = _completed_final_publication_present(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )
    closure_status = _clean(media.get("advertisingClosureStatus"))
    tournament_status = _clean(state.get("status"))
    intermediate_present = bool(raw_url or headline_url)
    winner_valid = is_valid_persisted_winner_development(state)
    closure_obj = state.get("advertisingClosure")
    closure_present = isinstance(closure_obj, dict) and bool(_clean(closure_obj.get("sloganText")))
    conflicting_publication = _conflicting_publication_evidence(
        media=media,
        closure_url=closure_url,
        final_public=final_public,
        headline_url=headline_url,
        raw_url=raw_url,
        valid_closure=valid_closure,
    )

    condition_results = {
        "persistedStatusMediaFinalizationIncomplete": tournament_status == "media_finalization_incomplete",
        "mediaContinuationRequired": bool(state.get("mediaContinuationRequired")),
        "mediaResumeStatusFinalizationFailed": _clean(media.get("mediaResumeStatus")) == "finalization_failed",
        "advertisingClosureStatusFailedOrIncomplete": closure_status in {"failed", "incomplete"}
        or (closure_status != "completed" and not media.get("advertisingClosureRendered")),
        "rawOrHeadlineIntermediatePresent": intermediate_present,
        "validClosureInclusiveFinalAbsent": not valid_closure,
        "finalPublicationCompletedAbsent": not completed_publication,
        "acceptedWinnerStillValid": winner_valid,
        "reasoningComplete": bool(state.get("reasoningComplete")),
        "advertisingClosurePresent": closure_present,
        "recoverableFailedFinalizationProven": False,
    }

    blocking: List[str] = []
    if valid_closure:
        blocking.append("validClosureInclusiveFinalPresent")
    if completed_publication:
        blocking.append("completedFinalPublicationPresent")
    if not intermediate_present:
        blocking.append("missingRawOrHeadlineIntermediate")
    if not winner_valid:
        blocking.append("acceptedWinnerInvalid")
    if not closure_present:
        blocking.append("advertisingClosureMissing")
    if not state.get("reasoningComplete"):
        blocking.append("reasoningIncomplete")
    if conflicting_publication:
        blocking.append("conflictingPublicationEvidence")
    if tournament_status in _UNRECOVERABLE_STATUSES:
        blocking.append("stateMarkedUnrecoverable")
    if active_finalization_lease:
        blocking.append("activeFinalizationLeasePresent")

    positive = (
        condition_results["persistedStatusMediaFinalizationIncomplete"]
        and condition_results["mediaContinuationRequired"]
        and condition_results["mediaResumeStatusFinalizationFailed"]
        and condition_results["advertisingClosureStatusFailedOrIncomplete"]
        and condition_results["validClosureInclusiveFinalAbsent"]
        and condition_results["finalPublicationCompletedAbsent"]
        and condition_results["rawOrHeadlineIntermediatePresent"]
        and condition_results["acceptedWinnerStillValid"]
        and condition_results["reasoningComplete"]
        and condition_results["advertisingClosurePresent"]
    )
    recoverable = positive and not blocking
    reasons: List[str] = []
    if recoverable:
        reasons.append("recoverable_failed_finalization_state")
    else:
        if not condition_results["persistedStatusMediaFinalizationIncomplete"]:
            reasons.append("status_not_media_finalization_incomplete")
        if not condition_results["mediaContinuationRequired"]:
            reasons.append("media_continuation_not_required")
        if not condition_results["mediaResumeStatusFinalizationFailed"]:
            reasons.append("media_resume_status_not_finalization_failed")
        if not condition_results["advertisingClosureStatusFailedOrIncomplete"]:
            reasons.append("advertising_closure_status_not_failed_or_incomplete")
        reasons.extend(blocking)

    condition_results["recoverableFailedFinalizationProven"] = recoverable
    return RecoverableFailedFinalizationAssessment(
        recoverable=recoverable,
        reasons=reasons,
        condition_results=condition_results,
        recovery_basis="failed_finalization_state" if recoverable else "",
        blocking_conditions=blocking,
    )


def evaluate_finalization_recovery_eligibility(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
    active_finalization_lease: bool = False,
) -> Dict[str, Any]:
    media = _media_bucket(state)
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = builder2_requires_headline_overlay(plan=plan, state=state)
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
    completed_publication = _completed_final_publication_present(
        state=state,
        closure_url=closure_url,
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    )
    false_completion, false_reasons = assess_false_completion(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
    )
    failed_recovery = assess_recoverable_failed_finalization_state(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
        active_finalization_lease=active_finalization_lease,
    )
    recovery_basis_proven = false_completion or failed_recovery.recoverable

    missing: List[str] = []
    if not state.get("reasoningComplete"):
        missing.append("reasoningComplete")
    if not is_valid_persisted_winner_development(state):
        missing.append("winnerDevelopmentAccepted")
    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict) or not _clean(closure.get("sloganText")):
        missing.append("advertisingClosure")
    if not recovery_basis_proven:
        missing.append("recoveryBasisNotProven")
    if not raw_url and not headline_url:
        missing.append("visualOrHeadlineIntermediate")
    if valid_closure:
        missing.append("validClosureAlreadyPresent")
    if completed_publication and not valid_closure:
        missing.append("completedFinalPublicationPresent")

    if false_completion:
        recovery_basis = "legacy_false_completion"
    elif failed_recovery.recoverable:
        recovery_basis = "failed_finalization_state"
    else:
        recovery_basis = None

    return {
        "eligible": not missing,
        "missing": missing,
        "legacyFalseCompletionConfirmed": false_completion,
        "legacyFalseCompletionReasons": false_reasons,
        "recoverableFailedFinalizationConfirmed": failed_recovery.recoverable,
        "recoverableFailedFinalizationReasons": failed_recovery.reasons,
        "recoverableFailedFinalizationConditionResults": failed_recovery.condition_results,
        "recoverableFailedFinalizationBlockingConditions": failed_recovery.blocking_conditions,
        "recoveryEligibilityBasis": recovery_basis,
        "recoveryBlockedByValidFinal": valid_closure,
        "recoveryBlockedByCompletedPublication": completed_publication,
        "recoveryBlockedByMissingIntermediate": not bool(raw_url or headline_url),
        "falseCompletionConfirmed": false_completion,
    }


def finalization_recovery_eligible(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
    active_finalization_lease: bool = False,
) -> Tuple[bool, List[str]]:
    evaluation = evaluate_finalization_recovery_eligibility(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
        active_finalization_lease=active_finalization_lease,
    )
    return bool(evaluation["eligible"]), list(evaluation["missing"])


def backfill_legacy_headline_reference(state: Dict[str, Any], *, job_video_url: str = "") -> str:
    media = _media_bucket(state)
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    headline_required = builder2_requires_headline_overlay(plan=plan, state=state)
    existing = _clean(media.get("headlineArtifactUrl"))
    if existing:
        return existing
    headline_url = resolve_legacy_headline_artifact_url(
        state=state,
        job_video_url=job_video_url,
        headline_required=headline_required,
    )
    if headline_url:
        media["headlineArtifactUrl"] = headline_url
        if not _clean(media.get("headlinePostprocessStatus")):
            media["headlinePostprocessStatus"] = "completed"
    return headline_url


def artifact_fingerprint(url: str) -> str:
    return url_fingerprint(url)
