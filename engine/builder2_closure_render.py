"""
Builder2 advertising-closure FFmpeg rendering — typed failures and duration verification.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

from engine.builder2_new_format_config import (
    FINAL_DURATION_TOLERANCE_SECONDS,
    resolve_builder2_end_card_duration_seconds,
    resolve_builder2_final_video_duration_seconds,
    resolve_builder2_video_duration_seconds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.video_headline_postprocess import (
    _ffmpeg_bin,
    _filter_path_for_ffmpeg,
    _input_has_audio,
    _path_for_token,
    _storage_root,
)
from engine.video_language import normalize_video_content_language

logger = logging.getLogger(__name__)

_HTTP_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_FFMPEG_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")
_STDERR_TAIL_MAX_CHARS = 512
_TARGET_WIDTH = 1280
_TARGET_HEIGHT = 720
_TARGET_FPS = 30


class Builder2ClosureRenderError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str,
        return_code: Optional[int] = None,
        stderr_tail: str = "",
        command_category: str = "",
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.return_code = return_code
        self.stderr_tail = stderr_tail
        self.command_category = command_category


@dataclass(frozen=True)
class ClosureRenderResult:
    public_url: str
    local_path: Optional[str]
    measured_duration_seconds: float
    output_token: str
    input_fingerprint: str


def _sanitize_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _ffprobe_duration_seconds(path: Path, timeout: float) -> float:
    from engine.video_headline_postprocess import _ffprobe_duration_seconds as probe

    return probe(path, timeout)


def sanitize_ffmpeg_stderr(raw: bytes | str | None) -> str:
    text = (raw or b"").decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
    text = re.sub(r"/[^\s'\"]+", "<path>", text)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _STDERR_TAIL_MAX_CHARS:
        return text[-_STDERR_TAIL_MAX_CHARS :]
    return text


def url_fingerprint(url: str) -> str:
    token = (url or "").strip()
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def classify_url_route_family(url: str) -> str:
    path = (urlparse((url or "").strip()).path or "").lower()
    if "/api/video-headline-artifact" in path:
        return "api/video-headline-artifact"
    if "/api/video-headline/" in path:
        return "api/video-headline"
    if "/api/video-closure" in path or "/video-closure" in path:
        return "api/video-closure"
    host = (urlparse((url or "").strip()).hostname or "").lower()
    if "runway" in host or "cloudfront" in host or "amazonaws.com" in host:
        return "runway-artifact"
    return "other"


def verify_builder2_final_video_duration(
    measured_seconds: float,
    *,
    visual_duration_seconds: Optional[float] = None,
    end_card_duration_seconds: Optional[float] = None,
    expected_final_seconds: Optional[float] = None,
) -> None:
    visual = float(visual_duration_seconds if visual_duration_seconds is not None else resolve_builder2_video_duration_seconds())
    end_card = float(
        end_card_duration_seconds if end_card_duration_seconds is not None else resolve_builder2_end_card_duration_seconds()
    )
    expected = float(expected_final_seconds if expected_final_seconds is not None else resolve_builder2_final_video_duration_seconds())
    tolerance = max(FINAL_DURATION_TOLERANCE_SECONDS, 0.2)
    if measured_seconds <= visual + 0.05:
        raise Builder2ClosureRenderError(
            "builder2_media_final_duration_not_longer_than_visual",
            stage="duration_verification",
            command_category="ffprobe",
        )
    if abs(measured_seconds - expected) > tolerance:
        raise Builder2ClosureRenderError(
            "builder2_media_final_duration_out_of_tolerance",
            stage="duration_verification",
            command_category="ffprobe",
        )
    if measured_seconds < visual + end_card - tolerance:
        raise Builder2ClosureRenderError(
            "builder2_media_final_duration_missing_end_card",
            stage="duration_verification",
            command_category="ffprobe",
        )


def _run_checked(cmd: list[str], *, stage: str, category: str) -> None:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            timeout=_FFMPEG_TIMEOUT,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise Builder2ClosureRenderError(
            "builder2_closure_ffmpeg_failed",
            stage=stage,
            return_code=exc.returncode,
            stderr_tail=sanitize_ffmpeg_stderr(exc.stderr),
            command_category=category,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise Builder2ClosureRenderError(
            "builder2_closure_ffmpeg_timeout",
            stage=stage,
            command_category=category,
            stderr_tail=sanitize_ffmpeg_stderr(getattr(exc, "stderr", b"")),
        ) from exc
    if completed.stderr:
        logger.debug(
            "BUILDER2_CLOSURE_FFMPEG stage=%s category=%s stderr_len=%s",
            stage,
            category,
            len(completed.stderr or b""),
        )


def _default_font_path(language: str) -> str:
    from engine.video_headline_postprocess import _default_font_path as headline_font

    return headline_font(language)


def _closure_storage_token(job_id: str) -> str:
    _ = job_id
    return uuid.uuid4().hex


def render_builder2_advertising_closure_endcard(
    source_video_url: str,
    public_base_url: str,
    *,
    product_name: str,
    slogan: str,
    language: str = "he",
    duration_seconds: float | None = None,
    job_id: str = "",
    publish: bool = True,
    output_path: Optional[Path] = None,
    ffmpeg_runner: Optional[Callable[[list[str], str, str], None]] = None,
) -> ClosureRenderResult:
    source = (source_video_url or "").strip()
    if not source:
        raise Builder2ClosureRenderError(
            "builder2_closure_missing_source_video",
            stage="input_validation",
            command_category="validation",
        )
    product = _sanitize_line(product_name)
    slogan_text = _sanitize_line(slogan)
    if not product or not slogan_text:
        raise Builder2ClosureRenderError(
            "builder2_closure_missing_text",
            stage="input_validation",
            command_category="validation",
        )
    base = (public_base_url or "").strip().rstrip("/")
    if publish and not base:
        from engine.public_base_url import resolve_public_base_url

        resolution = resolve_public_base_url()
        if resolution.configured:
            base = resolution.value
    if publish and not base:
        raise Builder2ClosureRenderError(
            "builder2_closure_missing_public_base_url",
            stage="input_validation",
            command_category="validation",
        )

    lang = normalize_video_content_language(language)
    ffmpeg = _ffmpeg_bin()
    font = _default_font_path(lang)
    if not ffmpeg or not font:
        raise Builder2ClosureRenderError(
            "builder2_closure_missing_ffmpeg_or_font",
            stage="input_validation",
            command_category="validation",
        )

    hold = max(0.5, float(duration_seconds if duration_seconds is not None else resolve_builder2_end_card_duration_seconds()))
    token = _closure_storage_token(job_id)
    published_path = output_path or _path_for_token(token)
    if published_path is None:
        raise Builder2ClosureRenderError(
            "builder2_closure_output_path_unavailable",
            stage="input_validation",
            command_category="validation",
        )

    tmp = Path(tempfile.mkdtemp(prefix="ace_closure_endcard_"))
    inp = tmp / "in.mp4"
    card = tmp / "card.mp4"
    out_tmp = tmp / "out.mp4"
    product_file = tmp / "product.txt"
    slogan_file = tmp / "slogan.txt"
    input_fingerprint = url_fingerprint(source)

    def _runner(cmd: list[str], stage: str, category: str) -> None:
        if ffmpeg_runner is not None:
            try:
                ffmpeg_runner(cmd, stage, category)
            except subprocess.CalledProcessError as exc:
                raise Builder2ClosureRenderError(
                    "builder2_closure_ffmpeg_failed",
                    stage=stage,
                    return_code=exc.returncode,
                    stderr_tail=sanitize_ffmpeg_stderr(exc.stderr),
                    command_category=category,
                ) from exc
            return
        _run_checked(cmd, stage=stage, category=category)

    try:
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source, timeout=_HTTP_DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            with open(inp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
        else:
            local = Path(source)
            if not local.is_file():
                raise Builder2ClosureRenderError(
                    "builder2_closure_source_not_found",
                    stage="input_validation",
                    command_category="validation",
                )
            inp.write_bytes(local.read_bytes())

        source_duration = _ffprobe_duration_seconds(inp, _FFPROBE_TIMEOUT)
        has_audio = _input_has_audio(inp, _FFPROBE_TIMEOUT)
        font_path = _filter_path_for_ffmpeg(Path(font))
        product_file.write_text(product, encoding="utf-8")
        slogan_file.write_text(slogan_text, encoding="utf-8")

        card_filter = (
            f"drawtext=fontfile='{font_path}':textfile='{_filter_path_for_ffmpeg(product_file)}':"
            f"fontcolor=white:fontsize=52:x=(w-text_w)/2:y=(h/2)-70:borderw=2:bordercolor=black@0.35,"
            f"drawtext=fontfile='{font_path}':textfile='{_filter_path_for_ffmpeg(slogan_file)}':"
            f"fontcolor=white:fontsize=40:x=(w-text_w)/2:y=(h/2)+10:borderw=2:bordercolor=black@0.35"
        )
        card_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={_TARGET_WIDTH}x{_TARGET_HEIGHT}:d={hold}",
            "-vf",
            card_filter,
            "-r",
            str(_TARGET_FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(card),
        ]

        if has_audio:
            filter_complex = (
                f"[0:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:"
                f"force_original_aspect_ratio=decrease,pad={_TARGET_WIDTH}:{_TARGET_HEIGHT}:"
                f"(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[v0];"
                f"[1:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT},setsar=1,setpts=PTS-STARTPTS[v1];"
                f"[v0][v1]concat=n=2:v=1:a=0[vout];"
                f"[0:a]apad,atrim=0:{source_duration + hold}[aout]"
            )
            concat_cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(inp),
                "-i",
                str(card),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(out_tmp),
            ]
        else:
            filter_complex = (
                f"[0:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:"
                f"force_original_aspect_ratio=decrease,pad={_TARGET_WIDTH}:{_TARGET_HEIGHT}:"
                f"(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[v0];"
                f"[1:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT},setsar=1,setpts=PTS-STARTPTS[v1];"
                f"[v0][v1]concat=n=2:v=1:a=0[vout]"
            )
            concat_cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(inp),
                "-i",
                str(card),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out_tmp),
            ]

        logger.info("BUILDER2_CLOSURE_ENDCARD start jobId=%s", (job_id or "").strip() or "(none)")
        t0 = time.monotonic()
        _runner(card_cmd, "card_generation", "ffmpeg_card")
        _runner(concat_cmd, "concatenation", "ffmpeg_concat")
        measured = _ffprobe_duration_seconds(out_tmp, _FFPROBE_TIMEOUT)
        verify_builder2_final_video_duration(
            measured,
            visual_duration_seconds=source_duration,
            end_card_duration_seconds=hold,
        )
        out_tmp.replace(published_path)

        public_url = ""
        if publish:
            upload_secret = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
            if upload_secret:
                upload_endpoint = f"{base}/api/video-headline-artifact"
                with open(published_path, "rb") as handle:
                    upload = requests.post(
                        upload_endpoint,
                        headers={"X-ACE-Video-Headline-Upload-Secret": upload_secret},
                        files={"file": ("closure.mp4", handle, "video/mp4")},
                        data={"token": token},
                        timeout=_FFMPEG_TIMEOUT,
                    )
                if not upload.ok:
                    raise Builder2ClosureRenderError(
                        "builder2_closure_publication_failed",
                        stage="publication",
                        return_code=upload.status_code,
                        command_category="http_upload",
                    )
            public_url = f"{base}/api/video-headline/{token}"
            if public_url == source:
                raise Builder2ClosureRenderError(
                    "builder2_closure_output_same_as_input",
                    stage="publication",
                    command_category="validation",
                )
        else:
            public_url = published_path.as_uri()

        logger.info(
            "BUILDER2_CLOSURE_ENDCARD done jobId=%s elapsed_ms=%.1f duration_s=%.3f",
            (job_id or "").strip() or "(none)",
            (time.monotonic() - t0) * 1000.0,
            measured,
        )
        return ClosureRenderResult(
            public_url=public_url,
            local_path=str(published_path),
            measured_duration_seconds=measured,
            output_token=token,
            input_fingerprint=input_fingerprint,
        )
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass
