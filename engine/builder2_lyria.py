"""
Builder2 Google Lyria music generation — REST generateContent, MP3 artifact persistence.

Builder2-only. Uses requests against Gemini Generate Content API.
Does not read OpenAI or global GEMINI_API_KEY env vars.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from engine.builder2_lyria_config import (
    Builder2LyriaConfigError,
    resolve_builder2_lyria_api_key,
    resolve_builder2_lyria_enabled,
    resolve_builder2_lyria_generate_content_url,
    resolve_builder2_lyria_job_artifact_path,
    resolve_builder2_lyria_model,
)
from engine.builder2_music_direction import (
    build_lyria_request_prompt,
    validate_music_direction_for_lyria_media,
)
from engine.builder2_music_artifact_publication import (
    Builder2MusicPublicationError,
    publish_builder2_music_artifact,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import patch_tournament_state
from engine.video_headline_postprocess import _ffprobe_duration_seconds, _input_has_audio

logger = logging.getLogger(__name__)

_LYRIA_HTTP_TIMEOUT = float((os.environ.get("BUILDER2_LYRIA_HTTP_TIMEOUT_SECONDS") or "300").strip() or "300")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_LYRIA_AUTO_RETRY_HTTP_STATUS = 503
_LYRIA_AUTO_RETRY_DELAY_SECONDS = float(
    (os.environ.get("BUILDER2_LYRIA_AUTO_RETRY_DELAY_SECONDS") or "2").strip() or "2"
)


class Builder2LyriaError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        failure_class: str = "Builder2LyriaError",
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(code)
        self.failure_class = failure_class
        self.http_status = http_status


def is_lyria_auto_retryable_http_failure(exc: Builder2LyriaError) -> bool:
    """Known provider HTTP failures eligible for one automatic in-production retry."""
    return getattr(exc, "http_status", None) == _LYRIA_AUTO_RETRY_HTTP_STATUS


def _sleep_lyria_auto_retry_delay() -> None:
    delay = max(0.0, _LYRIA_AUTO_RETRY_DELAY_SECONDS)
    if delay > 0:
        time.sleep(delay)


@dataclass(frozen=True)
class Builder2LyriaResult:
    artifact_path: str
    artifact_url: str
    model: str
    prompt: str
    reused: bool
    audio_duration_seconds: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    return media


def music_artifact_is_valid(path: str) -> bool:
    token = (path or "").strip()
    if not token:
        return False
    candidate = Path(token)
    if not candidate.is_file():
        return False
    try:
        return candidate.stat().st_size > 0
    except OSError:
        return False


def _persist_music_fields(state: Dict[str, Any], **fields: Any) -> None:
    media = _media_bucket(state)
    media.update(fields)


def build_lyria_generate_content_payload(*, combined_prompt: str) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {"text": combined_prompt},
                ],
            }
        ],
    }


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _part_inline_data(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inline = part.get("inlineData")
    if isinstance(inline, dict):
        return inline
    inline = part.get("inline_data")
    if isinstance(inline, dict):
        return inline
    return None


def _safe_safety_ratings_summary(raw: Any) -> Any:
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "category": item.get("category"),
                "probability": item.get("probability"),
                "blocked": item.get("blocked"),
            }
        )
    return out or None


def _summarize_part_for_missing_audio(part: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"keys": sorted(str(k) for k in part.keys())}
    if "text" in part:
        summary["type"] = "text"
        summary["textPresent"] = bool(_clean_str(part.get("text")))
    inline = _part_inline_data(part)
    if isinstance(inline, dict):
        summary["inlineDataPresent"] = True
        summary["mimeType"] = _clean_str(inline.get("mimeType") or inline.get("mime_type"))
        data_b64 = inline.get("data")
        summary["dataPresent"] = bool(data_b64)
        if data_b64:
            summary["dataLength"] = len(str(data_b64))
    return summary


def summarize_lyria_missing_audio_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Safe diagnostics when Lyria returns JSON without extractable audio — never includes audio bytes."""
    summary: Dict[str, Any] = {}
    response_id = _clean_str(payload.get("responseId"))
    if response_id:
        summary["responseId"] = response_id
    model_version = _clean_str(payload.get("modelVersion"))
    if model_version:
        summary["modelVersion"] = model_version
    usage = payload.get("usageMetadata")
    if isinstance(usage, dict):
        summary["usageMetadata"] = {
            key: usage[key]
            for key in (
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
                "cachedContentTokenCount",
            )
            if key in usage
        }
    prompt_feedback = payload.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        summary["promptFeedback"] = {
            key: prompt_feedback.get(key)
            for key in ("blockReason", "blockReasonMessage")
            if prompt_feedback.get(key) is not None
        }
        ratings = _safe_safety_ratings_summary(prompt_feedback.get("safetyRatings"))
        if ratings:
            summary["promptFeedback"]["safetyRatings"] = ratings
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        summary["candidateCount"] = len(candidates)
        candidate_summaries = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_summary: Dict[str, Any] = {}
            finish_reason = _clean_str(candidate.get("finishReason"))
            if finish_reason:
                candidate_summary["finishReason"] = finish_reason
            ratings = _safe_safety_ratings_summary(candidate.get("safetyRatings"))
            if ratings:
                candidate_summary["safetyRatings"] = ratings
            content = candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    candidate_summary["partsCount"] = len(parts)
                    candidate_summary["parts"] = [
                        _summarize_part_for_missing_audio(part)
                        for part in parts
                        if isinstance(part, dict)
                    ]
            candidate_summaries.append(candidate_summary)
        summary["candidates"] = candidate_summaries
    return summary


def _log_missing_audio_diagnostics(payload: Dict[str, Any]) -> None:
    diagnostics = summarize_lyria_missing_audio_response(payload)
    logger.error(
        "BUILDER2_LYRIA_RESPONSE_MISSING_AUDIO diagnostics=%s",
        json.dumps(diagnostics, ensure_ascii=False, default=str),
    )


def extract_audio_bytes_from_generate_content_response(payload: Dict[str, Any]) -> bytes:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise Builder2LyriaError("builder2_lyria_response_missing_candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = _part_inline_data(part)
            if not isinstance(inline, dict):
                continue
            mime = str(inline.get("mimeType") or inline.get("mime_type") or "").lower()
            data_b64 = inline.get("data")
            if not data_b64:
                continue
            if "audio" in mime or mime in {"audio/mpeg", "audio/mp3"}:
                try:
                    return base64.b64decode(str(data_b64))
                except (ValueError, TypeError) as exc:
                    raise Builder2LyriaError("builder2_lyria_response_invalid_audio_data") from exc
    _log_missing_audio_diagnostics(payload)
    raise Builder2LyriaError("builder2_lyria_response_missing_audio")


def call_lyria_generate_content(
    *,
    combined_prompt: str,
    model: str,
    api_key: str,
    session: Optional[requests.Session] = None,
) -> bytes:
    url = resolve_builder2_lyria_generate_content_url(model=model)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = build_lyria_generate_content_payload(combined_prompt=combined_prompt)
    http = session or requests.Session()
    try:
        response = http.post(url, headers=headers, json=body, timeout=_LYRIA_HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise Builder2LyriaError("builder2_lyria_http_error") from exc
    if response.status_code >= 400:
        logger.error(
            "BUILDER2_LYRIA_HTTP_ERROR status=%s model=%s",
            response.status_code,
            model,
        )
        raise Builder2LyriaError(
            "builder2_lyria_api_rejected",
            http_status=int(response.status_code),
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise Builder2LyriaError("builder2_lyria_response_not_json") from exc
    if not isinstance(payload, dict):
        raise Builder2LyriaError("builder2_lyria_response_invalid")
    return extract_audio_bytes_from_generate_content_response(payload)


def probe_mp3_duration_seconds(path: Path) -> float:
    try:
        return float(_ffprobe_duration_seconds(path, _FFPROBE_TIMEOUT))
    except RuntimeError as exc:
        raise Builder2LyriaError("builder2_lyria_audio_duration_probe_failed") from exc


def write_mp3_artifact(*, job_id: str, audio_bytes: bytes) -> Path:
    if not audio_bytes:
        raise Builder2LyriaError("builder2_lyria_empty_audio")
    dest = resolve_builder2_lyria_job_artifact_path(job_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(audio_bytes)
    if dest.stat().st_size <= 0:
        raise Builder2LyriaError("builder2_lyria_artifact_write_failed")
    return dest


def probe_local_mp3_has_audio(path: Path) -> bool:
    try:
        return bool(_input_has_audio(path, _FFPROBE_TIMEOUT))
    except RuntimeError:
        return False


def _mark_music_failure(state: Dict[str, Any], *, reason: str, failure_class: str = "Builder2LyriaError") -> None:
    status = "failed"
    if reason == "builder2_lyria_paid_call_outcome_unknown":
        status = "paid_call_outcome_unknown"
    _persist_music_fields(
        state,
        musicGenerationStatus=status,
        musicFailure={
            "failureStage": "music_generation",
            "failureReason": reason,
            "failureClass": failure_class,
            "failedAt": _utc_now_iso(),
        },
    )


def _persist_music_success(
    state: Dict[str, Any],
    *,
    job_id: str,
    dest: Path,
    duration: float,
    model: str,
    creative_prompt: str,
    next_attempt: int,
    music_artifact_url: str,
    music_artifact_token: str,
) -> None:
    generated_at = _utc_now_iso()
    _persist_music_fields(
        state,
        musicGenerationStatus="succeeded",
        musicArtifactPath=str(dest),
        musicArtifactUrl=music_artifact_url,
        musicArtifactToken=music_artifact_token,
        musicModel=model,
        musicPrompt=creative_prompt,
        musicGenerationAttempt=next_attempt,
        musicGeneratedAt=generated_at,
        musicAudioDurationSeconds=duration,
        musicFailure=None,
    )

    def _persist_success(st: Dict[str, Any]) -> None:
        bucket = _media_bucket(st)
        bucket.update(dict(state.get("mediaResume") or {}))
        bucket.pop("musicFailure", None)

    patch_tournament_state(job_id, _persist_success)


def generate_builder2_music(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    public_base_url: str = "",
    session: Optional[requests.Session] = None,
    api_caller: Optional[Callable[..., bytes]] = None,
    publisher: Optional[Callable[..., Any]] = None,
    downloader: Optional[Callable[..., Path]] = None,
) -> Builder2LyriaResult:
    """
    Generate or reuse Lyria MP3 for a Builder2 job.

    Paid generation is allowed only before documented success. After success, only
    local/durable recovery is permitted — never a second paid Lyria call.
    """
    if not resolve_builder2_lyria_enabled():
        raise Builder2LyriaError("builder2_lyria_disabled")

    from engine.builder2_lyria_artifact import resolve_existing_lyria_audio

    existing = resolve_existing_lyria_audio(
        job_id=job_id,
        state=state,
        session=session,
        downloader=downloader,
    )
    if existing is not None:
        media = _media_bucket(state)
        logger.info(
            "BUILDER2_LYRIA_REUSED jobId=%s model=%s reused=true recoveredFromDurable=%s durationSeconds=%.3f",
            job_id,
            media.get("musicModel") or resolve_builder2_lyria_model(),
            existing.recovered_from_durable,
            existing.audio_duration_seconds,
        )
        return Builder2LyriaResult(
            artifact_path=existing.local_path,
            artifact_url=str(media.get("musicArtifactUrl") or ""),
            model=existing.model or str(media.get("musicModel") or resolve_builder2_lyria_model()),
            prompt=existing.prompt,
            reused=True,
            audio_duration_seconds=existing.audio_duration_seconds,
        )

    media = _media_bucket(state)
    if int(media.get("musicGenerationAttempt") or 0) >= 2:
        raise Builder2LyriaError("builder2_lyria_generation_exhausted")

    music_direction = validate_music_direction_for_lyria_media(plan)
    creative_prompt, combined_prompt = build_lyria_request_prompt(music_direction)
    model = resolve_builder2_lyria_model()

    try:
        api_key = resolve_builder2_lyria_api_key(required=True)
    except Builder2LyriaConfigError as exc:
        raise Builder2LyriaError(str(exc)) from exc

    caller = api_caller or call_lyria_generate_content
    dest: Optional[Path] = None
    duration = 0.0
    publication = None
    succeeded_attempt = 0

    while True:
        from engine.builder2_job_cancellation import checkpoint_builder2_cancellation

        checkpoint_builder2_cancellation(job_id, stage=f"lyria_attempt_{int(media.get('musicGenerationAttempt') or 0) + 1}")
        media = _media_bucket(state)
        attempt = int(media.get("musicGenerationAttempt") or 0)
        if attempt >= 2:
            raise Builder2LyriaError("builder2_lyria_generation_exhausted")

        next_attempt = attempt + 1

        _persist_music_fields(
            state,
            musicGenerationStatus="generating",
            musicGenerationAttempt=next_attempt,
            musicModel=model,
            musicPrompt=creative_prompt,
        )

        def _persist_generating(st: Dict[str, Any]) -> None:
            bucket = _media_bucket(st)
            bucket["musicGenerationStatus"] = "generating"
            bucket["musicGenerationAttempt"] = next_attempt
            bucket["musicModel"] = model
            bucket["musicPrompt"] = creative_prompt

        patch_tournament_state(job_id, _persist_generating)

        logger.info(
            "BUILDER2_LYRIA_GENERATION_START jobId=%s model=%s attempt=%s reused=false",
            job_id,
            model,
            next_attempt,
        )

        try:
            audio_bytes = caller(combined_prompt=combined_prompt, model=model, api_key=api_key, session=session)
            checkpoint_builder2_cancellation(job_id, stage=f"lyria_after_attempt_{next_attempt}")
            dest = write_mp3_artifact(job_id=job_id, audio_bytes=audio_bytes)
            duration = probe_mp3_duration_seconds(dest)
            publish = publisher or publish_builder2_music_artifact
            publication = publish(dest, public_base_url, job_id=job_id)
            succeeded_attempt = next_attempt
            break
        except Builder2LyriaError as exc:
            from engine.builder2_job_cancellation import Builder2JobCancelledError, CANCELLED_ERROR_CODE

            if isinstance(exc, Builder2JobCancelledError) or str(exc.args[0] if exc.args else "") == CANCELLED_ERROR_CODE:
                raise
            _mark_music_failure(state, reason=str(exc.args[0] if exc.args else "builder2_lyria_failed"))
            patch_tournament_state(
                job_id,
                lambda st: _media_bucket(st).update(dict(state.get("mediaResume") or {})),
            )
            if is_lyria_auto_retryable_http_failure(exc) and next_attempt == 1:
                logger.warning(
                    "BUILDER2_LYRIA_TRANSIENT_FAILURE jobId=%s attempt=%s status=%s outcomeKnown=true retryAllowed=true",
                    job_id,
                    next_attempt,
                    exc.http_status,
                )
                checkpoint_builder2_cancellation(job_id, stage="lyria_auto_retry_before_delay")
                _sleep_lyria_auto_retry_delay()
                checkpoint_builder2_cancellation(job_id, stage="lyria_auto_retry_attempt_2")
                logger.info(
                    "BUILDER2_LYRIA_AUTO_RETRY jobId=%s fromAttempt=%s toAttempt=%s reason=http_503",
                    job_id,
                    1,
                    2,
                )
                continue
            raise
        except Builder2MusicPublicationError as exc:
            _mark_music_failure(state, reason=str(exc.args[0] if exc.args else "builder2_music_artifact_upload_failed"))
            patch_tournament_state(
                job_id,
                lambda st: _media_bucket(st).update(dict(state.get("mediaResume") or {})),
            )
            raise Builder2LyriaError(str(exc.args[0] if exc.args else "builder2_music_artifact_upload_failed")) from exc
        except Exception as exc:
            from engine.builder2_job_cancellation import Builder2JobCancelledError, CANCELLED_ERROR_CODE

            if isinstance(exc, Builder2JobCancelledError) or str(getattr(exc, "args", [""])[0]) == CANCELLED_ERROR_CODE:
                raise
            _mark_music_failure(state, reason="builder2_lyria_unexpected_failure")
            patch_tournament_state(
                job_id,
                lambda st: _media_bucket(st).update(dict(state.get("mediaResume") or {})),
            )
            raise Builder2LyriaError("builder2_lyria_unexpected_failure") from exc

    if dest is None or publication is None or succeeded_attempt <= 0:
        raise Builder2LyriaError("builder2_lyria_unexpected_failure")

    _persist_music_success(
        state,
        job_id=job_id,
        dest=dest,
        duration=duration,
        model=model,
        creative_prompt=creative_prompt,
        next_attempt=succeeded_attempt,
        music_artifact_url=publication.music_artifact_url,
        music_artifact_token=publication.output_token,
    )

    logger.info(
        "BUILDER2_LYRIA_GENERATION_DONE jobId=%s model=%s attempt=%s status=succeeded durationSeconds=%.3f durableUrl=%s",
        job_id,
        model,
        succeeded_attempt,
        duration,
        publication.music_artifact_url,
    )
    return Builder2LyriaResult(
        artifact_path=str(dest),
        artifact_url=publication.music_artifact_url,
        model=model,
        prompt=creative_prompt,
        reused=False,
        audio_duration_seconds=duration,
    )


def validate_lyria_audio_covers_final_duration(
    *,
    audio_duration_seconds: float,
    required_duration_seconds: float,
) -> None:
    if required_duration_seconds <= 0:
        return
    if audio_duration_seconds + 0.05 < required_duration_seconds:
        raise Builder2LyriaError("builder2_lyria_audio_shorter_than_final_video")
