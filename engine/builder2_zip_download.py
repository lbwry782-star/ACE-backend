"""
Builder2 ZIP download helpers — resolve final video bytes without legacy GET query routes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

_BUILDER2_FINAL_VIDEO_RE = re.compile(
    r"/api/builder2-final-video/([a-f0-9]{32})(?:[/?#]|$)",
    re.IGNORECASE,
)
_VIDEO_HEADLINE_RE = re.compile(
    r"/api/video-headline/([a-f0-9]{32})(?:[/?#]|$)",
    re.IGNORECASE,
)
_DEFAULT_TIMEOUT = httpx.Timeout(120.0)


@dataclass(frozen=True)
class Builder2ZipVideoFetchError(Exception):
    code: str
    http_status: Optional[int] = None

    def __str__(self) -> str:
        if self.http_status is not None:
            return f"{self.code}:status={self.http_status}"
        return self.code


def _extract_token(url: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(unquote((url or "").strip()))
    return (match.group(1) if match else "").strip()


def _read_local_builder2_final_video(token: str) -> bytes:
    from engine.builder2_final_video_store import get_builder2_final_video_path

    path = get_builder2_final_video_path(token)
    if path is None or not path.is_file():
        raise Builder2ZipVideoFetchError("video_not_found_local")
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("BUILDER2_ZIP_LOCAL_READ_FAIL token_prefix=%s err=%s", token[:8], type(exc).__name__)
        raise Builder2ZipVideoFetchError("video_read_failed") from exc
    if not data:
        raise Builder2ZipVideoFetchError("video_empty")
    return data


def _read_local_headline_video(token: str) -> bytes:
    from engine.video_headline_postprocess import get_headline_video_path

    path = get_headline_video_path(token)
    if path is None or not path.is_file():
        raise Builder2ZipVideoFetchError("video_not_found_local")
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("BUILDER2_ZIP_HEADLINE_READ_FAIL token_prefix=%s err=%s", token[:8], type(exc).__name__)
        raise Builder2ZipVideoFetchError("video_read_failed") from exc
    if not data:
        raise Builder2ZipVideoFetchError("video_empty")
    return data


def fetch_builder2_zip_video_bytes(video_url: str, *, timeout: httpx.Timeout | None = None) -> bytes:
    """
    Load Builder2 final MP4 bytes for ZIP packaging.
    Prefers local durable artifact paths; falls back to HTTP GET for remote URLs.
    """
    url = (video_url or "").strip()
    if not url:
        raise Builder2ZipVideoFetchError("missing_video_url")

    final_token = _extract_token(url, _BUILDER2_FINAL_VIDEO_RE)
    if final_token:
        logger.info("BUILDER2_ZIP_FETCH_LOCAL builder2-final-video token_prefix=%s", final_token[:8])
        return _read_local_builder2_final_video(final_token)

    headline_token = _extract_token(url, _VIDEO_HEADLINE_RE)
    if headline_token:
        logger.info("BUILDER2_ZIP_FETCH_LOCAL video-headline token_prefix=%s", headline_token[:8])
        return _read_local_headline_video(headline_token)

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise Builder2ZipVideoFetchError("invalid_video_url")

    logger.info(
        "BUILDER2_ZIP_FETCH_HTTP host=%s path_prefix=%s",
        parsed.hostname or "",
        (parsed.path or "")[:80],
    )
    try:
        with httpx.Client(timeout=timeout or _DEFAULT_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "ACE-Builder2-Zip/1.0",
                    "Accept": "video/mp4,video/*,*/*",
                },
            )
    except httpx.TimeoutException as exc:
        raise Builder2ZipVideoFetchError("video_download_timeout") from exc
    except httpx.HTTPError as exc:
        raise Builder2ZipVideoFetchError("video_download_failed") from exc

    if response.status_code != 200 or not response.content:
        raise Builder2ZipVideoFetchError("video_download_failed", http_status=response.status_code)
    return bytes(response.content)
