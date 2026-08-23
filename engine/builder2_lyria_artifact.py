"""
Builder2 Lyria audio artifact resolution — local/durable reuse without duplicate paid calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import requests

from engine import builder2_lyria as _lyria_module
from engine.builder2_lyria_config import resolve_builder2_lyria_enabled
from engine.builder2_music_artifact_publication import (
    download_builder2_music_artifact_to_local,
    durable_music_reference_present,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LyriaAudioResolution:
    local_path: str
    reused: bool
    recovered_from_durable: bool
    model: str
    prompt: str
    audio_duration_seconds: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    return media


def job_requires_lyria_soundtrack(state: Mapping[str, Any]) -> bool:
    if not resolve_builder2_lyria_enabled():
        return False
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    return str(media.get("musicGenerationStatus") or "").strip().lower() == "succeeded"


def _mark_ambiguous_paid_call(state: Dict[str, Any]) -> None:
    media = _media_bucket(state)
    media["musicGenerationStatus"] = "paid_call_outcome_unknown"
    media["musicFailure"] = {
        "failureStage": "music_generation",
        "failureReason": "builder2_lyria_paid_call_outcome_unknown",
        "failureClass": "Builder2LyriaError",
        "failedAt": _utc_now_iso(),
    }


def _try_recover_local_from_durable(
    *,
    job_id: str,
    state: Dict[str, Any],
    media: Dict[str, Any],
    session: Optional[requests.Session],
    downloader: Optional[Callable[..., Path]],
) -> Optional[LyriaAudioResolution]:
    if not durable_music_reference_present(media):
        return None
    music_url = str(media.get("musicArtifactUrl") or "").strip()
    download = downloader or download_builder2_music_artifact_to_local
    try:
        dest = download(music_artifact_url=music_url, job_id=job_id, session=session)
    except Exception as exc:
        logger.warning(
            "BUILDER2_LYRIA_DURABLE_DOWNLOAD_FAIL jobId=%s reason=%s",
            job_id,
            type(exc).__name__,
        )
        return None
    if not _lyria_module.music_artifact_is_valid(str(dest)):
        return None
    duration = _lyria_module.probe_mp3_duration_seconds(dest)
    _media_bucket(state).update(
        {
            "musicGenerationStatus": "succeeded",
            "musicArtifactPath": str(dest),
            "musicGeneratedAt": media.get("musicGeneratedAt") or _utc_now_iso(),
            "musicAudioDurationSeconds": duration,
            "musicFailure": None,
        }
    )
    return LyriaAudioResolution(
        local_path=str(dest),
        reused=True,
        recovered_from_durable=True,
        model=str(media.get("musicModel") or ""),
        prompt=str(media.get("musicPrompt") or ""),
        audio_duration_seconds=duration,
    )


def resolve_lyria_audio_for_render(
    *,
    job_id: str,
    state: Dict[str, Any],
    public_base_url: str = "",
    session: Optional[requests.Session] = None,
    downloader: Optional[Callable[..., Path]] = None,
) -> str:
    """
    Resolve a local MP3 path for FFmpeg. Never performs paid Lyria generation.
    Raises when a Lyria-enabled succeeded job cannot recover its soundtrack.
    """
    if not job_requires_lyria_soundtrack(state):
        return ""
    media = _media_bucket(state)
    local_path = str(media.get("musicArtifactPath") or "").strip()
    if _lyria_module.music_artifact_is_valid(local_path):
        return local_path
    recovered = _try_recover_local_from_durable(
        job_id=job_id,
        state=state,
        media=media,
        session=session,
        downloader=downloader,
    )
    if recovered is not None:
        return recovered.local_path
    raise _lyria_module.Builder2LyriaError("builder2_lyria_succeeded_artifact_missing")


def resolve_existing_lyria_audio(
    *,
    job_id: str,
    state: Dict[str, Any],
    session: Optional[requests.Session] = None,
    downloader: Optional[Callable[..., Path]] = None,
) -> Optional[LyriaAudioResolution]:
    """
    Resolve existing Lyria audio without paid generation.
    Returns None when generation is still required (failed/never attempted).
    Raises on succeeded-but-missing or ambiguous generating states.
    """
    if not resolve_builder2_lyria_enabled():
        return None

    media = _media_bucket(state)
    existing_path = str(media.get("musicArtifactPath") or "").strip()
    existing_status = str(media.get("musicGenerationStatus") or "").strip().lower()

    if existing_status == "paid_call_outcome_unknown":
        raise _lyria_module.Builder2LyriaError("builder2_lyria_paid_call_outcome_unknown")

    if existing_status == "succeeded":
        if _lyria_module.music_artifact_is_valid(existing_path):
            duration = _lyria_module.probe_mp3_duration_seconds(Path(existing_path))
            return LyriaAudioResolution(
                local_path=existing_path,
                reused=True,
                recovered_from_durable=False,
                model=str(media.get("musicModel") or ""),
                prompt=str(media.get("musicPrompt") or ""),
                audio_duration_seconds=duration,
            )
        recovered = _try_recover_local_from_durable(
            job_id=job_id,
            state=state,
            media=media,
            session=session,
            downloader=downloader,
        )
        if recovered is not None:
            return recovered
        raise _lyria_module.Builder2LyriaError("builder2_lyria_succeeded_artifact_missing")

    if existing_status == "generating":
        if _lyria_module.music_artifact_is_valid(existing_path):
            duration = _lyria_module.probe_mp3_duration_seconds(Path(existing_path))
            media.update(
                {
                    "musicGenerationStatus": "succeeded",
                    "musicArtifactPath": existing_path,
                    "musicGeneratedAt": media.get("musicGeneratedAt") or _utc_now_iso(),
                    "musicAudioDurationSeconds": duration,
                }
            )
            media.pop("musicFailure", None)
            return LyriaAudioResolution(
                local_path=existing_path,
                reused=True,
                recovered_from_durable=False,
                model=str(media.get("musicModel") or ""),
                prompt=str(media.get("musicPrompt") or ""),
                audio_duration_seconds=duration,
            )
        recovered = _try_recover_local_from_durable(
            job_id=job_id,
            state=state,
            media=media,
            session=session,
            downloader=downloader,
        )
        if recovered is not None:
            return recovered
        _mark_ambiguous_paid_call(state)
        raise _lyria_module.Builder2LyriaError("builder2_lyria_paid_call_outcome_unknown")

    return None
