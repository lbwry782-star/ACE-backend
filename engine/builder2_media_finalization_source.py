"""
Builder2 media finalization source-selection contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_media_finalization_contract import (
    resolve_legacy_headline_artifact_url,
    resolve_raw_runway_artifact_url,
)
from engine.builder2_media_finalization_download import SafeDownloadDiagnostics, safe_download_to_path
from engine.builder2_closure_render import classify_url_route_family


SOURCE_PERSISTED_HEADLINE = "persisted_headline_artifact"
SOURCE_LEGACY_HEADLINE = "legacy_headline_artifact"
SOURCE_RAW_RUNWAY_LOCAL_HEADLINE = "raw_runway_requires_local_headline"
SOURCE_RAW_RUNWAY_NO_HEADLINE = "raw_runway_no_headline"


@dataclass
class FinalizationSourceDecision:
    source_kind: str = ""
    closure_input_path: Optional[Path] = None
    selected_route_family: Optional[str] = None
    legacy_headline_download_failed: bool = False
    local_headline_render_required: bool = False
    raw_runway_download_required: bool = False
    persisted_headline_url: str = ""
    legacy_headline_url: str = ""
    raw_runway_url: str = ""
    legacy_headline_diagnostics: Optional[SafeDownloadDiagnostics] = None
    raw_runway_diagnostics: Optional[SafeDownloadDiagnostics] = None
    failure_reason: Optional[str] = None
    failure_stage: Optional[str] = None

    def to_report_dict(self) -> Dict[str, Any]:
        legacy = self.legacy_headline_diagnostics
        raw = self.raw_runway_diagnostics
        return {
            "selectedFinalizationSourceKind": self.source_kind or None,
            "selectedRouteFamily": self.selected_route_family,
            "legacyHeadlineDownloadFailed": self.legacy_headline_download_failed,
            "localHeadlineRenderRequired": self.local_headline_render_required,
            "rawRunwayDownloadRequired": self.raw_runway_download_required,
            "legacyHeadlineDownloadAttempted": bool(legacy and legacy.request_attempted),
            "legacyHeadlineDownloadAccepted": bool(legacy and legacy.download_accepted),
            "legacyHeadlineHttpStatusCode": legacy.http_status_code if legacy else None,
            "legacyHeadlineDownloadFailureCategory": legacy.download_failure_category if legacy else None,
            "legacyHeadlineArtifactUnavailable": bool(legacy and legacy.legacy_headline_artifact_unavailable),
            "rawRunwayFallbackAttempted": bool(raw and raw.request_attempted),
            "rawRunwayFallbackAccepted": bool(
                raw and raw.download_accepted and self.local_headline_render_required
            ),
            "rawRunwayDownloadAccepted": bool(raw and raw.download_accepted),
        }


def _first_clean(*values: Any) -> str:
    for value in values:
        token = str(value or "").strip()
        if token:
            return token
    return ""


def _headline_candidates(
    *,
    state: Dict[str, Any],
    job_video_url: str,
    headline_required: bool,
) -> List[tuple[str, str]]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    candidates: List[tuple[str, str]] = []
    persisted = _first_clean(media.get("headlineArtifactUrl"))
    if persisted:
        candidates.append((SOURCE_PERSISTED_HEADLINE, persisted))
    legacy = resolve_legacy_headline_artifact_url(
        state=state,
        job_video_url=job_video_url,
        headline_required=headline_required,
    )
    if legacy and legacy != persisted:
        candidates.append((SOURCE_LEGACY_HEADLINE, legacy))
    elif legacy and not persisted:
        candidates.append((SOURCE_LEGACY_HEADLINE, legacy))
    return candidates


def resolve_finalization_source_decision(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
    work_dir: Path,
    download_headline: bool = True,
) -> FinalizationSourceDecision:
    headline_required = headline_decision_requires_headline(get_normalized_headline_decision(plan))
    raw_url = resolve_raw_runway_artifact_url(state)
    decision = FinalizationSourceDecision(
        raw_runway_url=raw_url,
        raw_runway_download_required=bool(raw_url),
    )

    if headline_required and download_headline:
        for kind, url in _headline_candidates(
            state=state,
            job_video_url=job_video_url,
            headline_required=headline_required,
        ):
            path = work_dir / f"headline_{kind}.mp4"
            diagnostics = safe_download_to_path(url, path, validate_video=True)
            if kind in {SOURCE_PERSISTED_HEADLINE, SOURCE_LEGACY_HEADLINE}:
                decision.legacy_headline_diagnostics = diagnostics
            if diagnostics.download_accepted:
                decision.source_kind = kind
                decision.closure_input_path = path
                decision.selected_route_family = classify_url_route_family(url) or None
                decision.persisted_headline_url = url if kind == SOURCE_PERSISTED_HEADLINE else decision.persisted_headline_url
                decision.legacy_headline_url = url if kind == SOURCE_LEGACY_HEADLINE else decision.legacy_headline_url
                decision.legacy_headline_download_failed = False
                return decision
            decision.legacy_headline_download_failed = True

    if headline_required:
        if not raw_url:
            decision.failure_stage = "source_selection"
            decision.failure_reason = "builder2_media_finalization_raw_runway_missing_for_headline_fallback"
            return decision
        decision.local_headline_render_required = True
        decision.source_kind = SOURCE_RAW_RUNWAY_LOCAL_HEADLINE
        decision.selected_route_family = classify_url_route_family(raw_url) or None
        raw_path = work_dir / "raw_runway.mp4"
        diagnostics = safe_download_to_path(raw_url, raw_path, validate_video=True)
        decision.raw_runway_diagnostics = diagnostics
        if not diagnostics.download_accepted:
            decision.failure_stage = "raw_runway_download"
            decision.failure_reason = "builder2_media_finalization_raw_runway_download_failed"
            return decision
        decision.closure_input_path = raw_path
        return decision

    if not raw_url:
        decision.failure_stage = "source_selection"
        decision.failure_reason = "builder2_media_finalization_raw_runway_missing"
        return decision
    decision.source_kind = SOURCE_RAW_RUNWAY_NO_HEADLINE
    decision.selected_route_family = classify_url_route_family(raw_url) or None
    raw_path = work_dir / "raw_runway.mp4"
    diagnostics = safe_download_to_path(raw_url, raw_path, validate_video=True)
    decision.raw_runway_diagnostics = diagnostics
    if not diagnostics.download_accepted:
        decision.failure_stage = "raw_runway_download"
        decision.failure_reason = "builder2_media_finalization_raw_runway_download_failed"
        return decision
    decision.closure_input_path = raw_path
    return decision
