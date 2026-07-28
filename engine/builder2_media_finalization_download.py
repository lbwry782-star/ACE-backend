"""
Builder2 finalization — safe HTTP download diagnostics without exposing secrets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from engine.builder2_closure_render import classify_url_route_family

_HTTP_DOWNLOAD_TIMEOUT = float(
    (__import__("os").environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180"
)


@dataclass
class SafeDownloadDiagnostics:
    request_attempted: bool = False
    request_method: str = "GET"
    original_route_family: Optional[str] = None
    redirect_count: int = 0
    final_route_family: Optional[str] = None
    http_status_code: Optional[int] = None
    response_content_type: Optional[str] = None
    response_content_length: Optional[int] = None
    download_failure_class: Optional[str] = None
    download_failure_category: Optional[str] = None
    legacy_headline_artifact_unavailable: bool = False
    download_accepted: bool = False

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "requestAttempted": self.request_attempted,
            "requestMethod": self.request_method,
            "originalRouteFamily": self.original_route_family,
            "redirectCount": self.redirect_count,
            "finalRouteFamily": self.final_route_family,
            "httpStatusCode": self.http_status_code,
            "responseContentType": self.response_content_type,
            "responseContentLength": self.response_content_length,
            "downloadFailureClass": self.download_failure_class,
            "downloadFailureCategory": self.download_failure_category,
            "legacyHeadlineArtifactUnavailable": self.legacy_headline_artifact_unavailable,
            "downloadAccepted": self.download_accepted,
        }


def _classify_http_failure(status_code: Optional[int], exc: Optional[BaseException]) -> tuple[str, str]:
    if isinstance(exc, requests.Timeout):
        return type(exc).__name__, "timeout"
    if isinstance(exc, requests.ConnectionError):
        return type(exc).__name__, "connection_error"
    if status_code == 404:
        return "HTTPError", "not_found"
    if status_code == 410:
        return "HTTPError", "expired_or_gone"
    if status_code in {401, 403}:
        return "HTTPError", "forbidden"
    if status_code is not None and status_code >= 500:
        return "HTTPError", "server_error"
    if isinstance(exc, requests.HTTPError):
        return type(exc).__name__, "other"
    if exc is not None:
        return type(exc).__name__, "other"
    return "HTTPError", "other"


def safe_download_to_path(
    url: str,
    path: Path,
    *,
    timeout: float = _HTTP_DOWNLOAD_TIMEOUT,
    validate_video: bool = True,
) -> SafeDownloadDiagnostics:
    diagnostics = SafeDownloadDiagnostics(
        original_route_family=classify_url_route_family(url) or None,
    )
    token = (url or "").strip()
    if not token:
        diagnostics.download_failure_class = "ValueError"
        diagnostics.download_failure_category = "other"
        return diagnostics

    diagnostics.request_attempted = True
    response: Optional[requests.Response] = None
    try:
        response = requests.get(token, timeout=timeout, stream=True, allow_redirects=True)
        diagnostics.redirect_count = len(response.history)
        diagnostics.final_route_family = classify_url_route_family(response.url) or None
        diagnostics.http_status_code = response.status_code
        diagnostics.response_content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip() or None
        raw_len = response.headers.get("Content-Length")
        if raw_len is not None and str(raw_len).strip().isdigit():
            diagnostics.response_content_length = int(raw_len)
        response.raise_for_status()
        with open(path, "wb") as handle:
            written = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
            if diagnostics.response_content_length is None:
                diagnostics.response_content_length = written
    except requests.RequestException as exc:
        status = response.status_code if response is not None else None
        diagnostics.http_status_code = status
        diagnostics.download_failure_class, diagnostics.download_failure_category = _classify_http_failure(status, exc)
        if diagnostics.original_route_family == "api/video-headline":
            diagnostics.legacy_headline_artifact_unavailable = True
        return diagnostics

    if validate_video:
        content_type = (diagnostics.response_content_type or "").lower()
        if content_type and "video" not in content_type and "octet-stream" not in content_type:
            diagnostics.download_failure_class = "InvalidMediaType"
            diagnostics.download_failure_category = "invalid_media"
            if diagnostics.original_route_family == "api/video-headline":
                diagnostics.legacy_headline_artifact_unavailable = True
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return diagnostics
        try:
            from engine.builder2_closure_render import _ffprobe_duration_seconds, _FFPROBE_TIMEOUT

            duration = _ffprobe_duration_seconds(path, _FFPROBE_TIMEOUT)
            if duration <= 0:
                raise ValueError("non_positive_duration")
        except Exception as exc:
            diagnostics.download_failure_class = type(exc).__name__
            diagnostics.download_failure_category = "invalid_media"
            if diagnostics.original_route_family == "api/video-headline":
                diagnostics.legacy_headline_artifact_unavailable = True
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return diagnostics

    diagnostics.download_accepted = True
    return diagnostics
