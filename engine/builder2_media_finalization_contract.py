"""
Builder2 media finalization contract — completion gate and legacy artifact recovery rules.
"""
from __future__ import annotations

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


def resolve_legacy_headline_artifact_url(
    *,
    state: Dict[str, Any],
    job_video_url: str = "",
    headline_required: bool = False,
) -> str:
    media = _media_bucket(state)
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
    measured = _duration_value(media, "actualFinalVideoDurationSeconds")
    if measured is not None:
        try:
            verify_builder2_final_video_duration(measured)
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
    if headline_required and not headline_url:
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
    if headline_required and not headline_url:
        failures.append("headline_artifact_missing")
    measured = _duration_value(media, "actualFinalVideoDurationSeconds")
    if measured is None:
        failures.append("actual_final_duration_missing")
    else:
        try:
            verify_builder2_final_video_duration(measured)
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
    if failures:
        return False, failures[0], failures
    return True, "", []


def finalization_recovery_eligible(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str = "",
) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not state.get("reasoningComplete"):
        missing.append("reasoningComplete")
    if not is_valid_persisted_winner_development(state):
        missing.append("winnerDevelopmentAccepted")
    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict) or not _clean(closure.get("sloganText")):
        missing.append("advertisingClosure")
    headline_decision = get_normalized_headline_decision(plan)
    headline_required = headline_decision_requires_headline(headline_decision)
    false_completion, _reasons = assess_false_completion(state=state, plan=plan, job_video_url=job_video_url)
    if not false_completion:
        missing.append("falseCompletionNotProven")
    raw_url = resolve_raw_runway_artifact_url(state)
    headline_url = resolve_legacy_headline_artifact_url(
        state=state,
        job_video_url=job_video_url,
        headline_required=headline_required,
    )
    if not raw_url and not headline_url:
        missing.append("visualOrHeadlineIntermediate")
    if closure_inclusive_artifact_valid(
        state=state,
        closure_url=_first_url(_media_bucket(state).get("finalVideoWithClosureUrl")),
        raw_url=raw_url,
        headline_url=headline_url,
        job_video_url=job_video_url,
    ):
        missing.append("validClosureAlreadyPresent")
    return not missing, missing


def backfill_legacy_headline_reference(state: Dict[str, Any], *, job_video_url: str = "") -> str:
    media = _media_bucket(state)
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    headline_required = headline_decision_requires_headline(get_normalized_headline_decision(plan))
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
