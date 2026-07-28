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
    resolve_builder2_effective_closure_segment_duration_seconds,
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


@dataclass(frozen=True)
class FinalDurationVerificationDiagnostics:
    measured_closure_output_duration_seconds: float
    measured_closure_source_duration_seconds: float
    configured_visual_duration_seconds: float
    configured_end_card_duration_seconds: float
    effective_closure_segment_duration_seconds: float
    configured_final_duration_seconds: float
    calculated_expected_final_duration_seconds: float
    actual_closure_gain_seconds: float
    closure_gain_accepted: bool
    accepted_final_duration_lower_bound_seconds: float
    accepted_final_duration_upper_bound_seconds: float
    final_duration_delta_seconds: float
    final_duration_verification_failure_code: str = ""
    closure_failure_substage: str = "duration_verification"

    def to_report_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "measuredClosureOutputDurationSeconds": self.measured_closure_output_duration_seconds,
            "measuredClosureSourceDurationSeconds": self.measured_closure_source_duration_seconds,
            "configuredVisualDurationSeconds": self.configured_visual_duration_seconds,
            "configuredEndCardDurationSeconds": self.configured_end_card_duration_seconds,
            "effectiveClosureSegmentDurationSeconds": self.effective_closure_segment_duration_seconds,
            "configuredFinalDurationSeconds": self.configured_final_duration_seconds,
            "calculatedExpectedFinalDurationSeconds": self.calculated_expected_final_duration_seconds,
            "actualClosureGainSeconds": self.actual_closure_gain_seconds,
            "closureGainAccepted": self.closure_gain_accepted,
            "acceptedFinalDurationLowerBoundSeconds": self.accepted_final_duration_lower_bound_seconds,
            "acceptedFinalDurationUpperBoundSeconds": self.accepted_final_duration_upper_bound_seconds,
            "finalDurationDeltaSeconds": self.final_duration_delta_seconds,
            "finalDurationVerificationFailureCode": self.final_duration_verification_failure_code or None,
            "closureFailureSubstage": self.closure_failure_substage,
            "closureDurationProbeAttempted": True,
            "closureDurationProbeAccepted": not bool(self.final_duration_verification_failure_code),
        }


class Builder2ClosureRenderError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str,
        return_code: Optional[int] = None,
        stderr_tail: str = "",
        command_category: str = "",
        duration_diagnostics: Optional[FinalDurationVerificationDiagnostics] = None,
        closure_ffmpeg_execution_accepted: Optional[bool] = None,
        closure_output_file_created: Optional[bool] = None,
        closure_output_file_size_bytes: Optional[int] = None,
        closure_ffprobe_calls: int = 0,
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.return_code = return_code
        self.stderr_tail = stderr_tail
        self.command_category = command_category
        self.duration_diagnostics = duration_diagnostics
        self.closure_ffmpeg_execution_accepted = closure_ffmpeg_execution_accepted
        self.closure_output_file_created = closure_output_file_created
        self.closure_output_file_size_bytes = closure_output_file_size_bytes
        self.closure_ffprobe_calls = closure_ffprobe_calls


@dataclass(frozen=True)
class ClosureRenderResult:
    public_url: str
    local_path: Optional[str]
    measured_duration_seconds: float
    output_token: str
    input_fingerprint: str
    closure_ffprobe_calls: int = 0
    duration_diagnostics: Optional[FinalDurationVerificationDiagnostics] = None


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


def build_final_duration_verification_diagnostics(
    measured_seconds: float,
    *,
    visual_duration_seconds: float,
    effective_closure_segment_duration_seconds: float | None = None,
    failure_code: str = "",
) -> FinalDurationVerificationDiagnostics:
    configured_visual = float(resolve_builder2_video_duration_seconds())
    configured_end_card = float(resolve_builder2_end_card_duration_seconds())
    configured_final = float(resolve_builder2_final_video_duration_seconds())
    effective_segment = float(
        effective_closure_segment_duration_seconds
        if effective_closure_segment_duration_seconds is not None
        else resolve_builder2_effective_closure_segment_duration_seconds()
    )
    calculated_expected = float(visual_duration_seconds) + effective_segment
    tolerance = max(FINAL_DURATION_TOLERANCE_SECONDS, 0.2)
    actual_gain = float(measured_seconds) - float(visual_duration_seconds)
    closure_gain_accepted = (
        not failure_code
        and actual_gain >= effective_segment - tolerance
        and actual_gain <= effective_segment + tolerance
    )
    return FinalDurationVerificationDiagnostics(
        measured_closure_output_duration_seconds=float(measured_seconds),
        measured_closure_source_duration_seconds=float(visual_duration_seconds),
        configured_visual_duration_seconds=configured_visual,
        configured_end_card_duration_seconds=configured_end_card,
        effective_closure_segment_duration_seconds=effective_segment,
        configured_final_duration_seconds=configured_final,
        calculated_expected_final_duration_seconds=calculated_expected,
        accepted_final_duration_lower_bound_seconds=calculated_expected - tolerance,
        accepted_final_duration_upper_bound_seconds=calculated_expected + tolerance,
        final_duration_delta_seconds=float(measured_seconds) - calculated_expected,
        actual_closure_gain_seconds=actual_gain,
        closure_gain_accepted=closure_gain_accepted,
        final_duration_verification_failure_code=failure_code,
    )


def verify_builder2_final_video_duration(
    measured_seconds: float,
    *,
    visual_duration_seconds: Optional[float] = None,
    end_card_duration_seconds: Optional[float] = None,
    expected_final_seconds: Optional[float] = None,
) -> FinalDurationVerificationDiagnostics:
    visual = float(
        visual_duration_seconds if visual_duration_seconds is not None else resolve_builder2_video_duration_seconds()
    )
    effective_segment = resolve_builder2_effective_closure_segment_duration_seconds(
        end_card_duration_seconds,
    )
    tolerance = max(FINAL_DURATION_TOLERANCE_SECONDS, 0.2)
    calculated_expected = visual + effective_segment

    def _fail(code: str) -> None:
        diagnostics = build_final_duration_verification_diagnostics(
            measured_seconds,
            visual_duration_seconds=visual,
            effective_closure_segment_duration_seconds=effective_segment,
            failure_code=code,
        )
        raise Builder2ClosureRenderError(
            code,
            stage="duration_verification",
            command_category="ffprobe",
            duration_diagnostics=diagnostics,
        )

    if measured_seconds <= visual + 0.05:
        _fail("builder2_media_final_duration_not_longer_than_visual")

    gained = measured_seconds - visual
    if gained < effective_segment - tolerance:
        _fail("builder2_media_final_duration_missing_end_card")

    if gained > effective_segment + tolerance:
        _fail("builder2_media_final_duration_excessive_closure_gain")

    if measured_seconds > visual + effective_segment + tolerance + max(visual * 0.75, 7.0):
        _fail("builder2_media_final_duration_duplicated_visual")

    if abs(measured_seconds - calculated_expected) > tolerance:
        _fail("builder2_media_final_duration_out_of_tolerance")

    configured_final = float(resolve_builder2_final_video_duration_seconds())
    configured_visual = float(resolve_builder2_video_duration_seconds())
    if abs(visual - configured_visual) <= tolerance:
        product_expected = configured_final + (visual - configured_visual)
        if abs(measured_seconds - product_expected) > tolerance:
            _fail("builder2_media_final_duration_outside_product_contract")

    return build_final_duration_verification_diagnostics(
        measured_seconds,
        visual_duration_seconds=visual,
        effective_closure_segment_duration_seconds=effective_segment,
    )


def _run_checked(cmd: list[str], *, stage: str, category: str) -> int:
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
    except OSError as exc:
        raise Builder2ClosureRenderError(
            "builder2_closure_ffmpeg_os_error",
            stage=stage,
            command_category=category,
        ) from exc
    if completed.stderr:
        logger.debug(
            "BUILDER2_CLOSURE_FFMPEG stage=%s category=%s stderr_len=%s",
            stage,
            category,
            len(completed.stderr or b""),
        )
    return int(completed.returncode)


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

    effective_segment = resolve_builder2_effective_closure_segment_duration_seconds(duration_seconds)
    hold = effective_segment
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

    def _runner(cmd: list[str], stage: str, category: str) -> int:
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
            return 0
        return _run_checked(cmd, stage=stage, category=category)

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
        closure_ffprobe_calls = 1
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
            "-t",
            f"{hold:.6f}",
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
                f"[1:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT},trim=duration={hold:.6f},"
                f"setsar=1,setpts=PTS-STARTPTS[v1];"
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
                f"[1:v]fps={_TARGET_FPS},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT},trim=duration={hold:.6f},"
                f"setsar=1,setpts=PTS-STARTPTS[v1];"
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
        card_rc = _runner(card_cmd, "card_generation", "ffmpeg_card")
        card_created = card.is_file()
        card_size = card.stat().st_size if card_created else 0
        logger.info(
            "BUILDER2_CLOSURE_CARD_RENDER_COMPLETED ffmpegReturnCode=%s outputCreated=%s outputSizeBytes=%s effectiveClosureDurationSeconds=%.3f",
            card_rc,
            card_created,
            card_size,
            hold,
        )
        concat_rc = _runner(concat_cmd, "concatenation", "ffmpeg_concat")
        closure_ffmpeg_execution_accepted = True
        closure_output_file_created = out_tmp.is_file()
        closure_output_file_size_bytes = out_tmp.stat().st_size if closure_output_file_created else 0
        logger.info(
            "BUILDER2_CLOSURE_CONCAT_COMPLETED ffmpegReturnCode=%s outputCreated=%s outputSizeBytes=%s measuredSourceDurationSeconds=%.3f effectiveClosureDurationSeconds=%.3f",
            concat_rc,
            closure_output_file_created,
            closure_output_file_size_bytes,
            source_duration,
            hold,
        )
        try:
            measured = _ffprobe_duration_seconds(out_tmp, _FFPROBE_TIMEOUT)
        except RuntimeError as exc:
            raise Builder2ClosureRenderError(
                "builder2_closure_ffprobe_failed",
                stage="duration_probe",
                command_category="ffprobe",
                closure_ffmpeg_execution_accepted=closure_ffmpeg_execution_accepted,
                closure_output_file_created=closure_output_file_created,
                closure_output_file_size_bytes=closure_output_file_size_bytes,
            ) from exc
        closure_ffprobe_calls += 1
        logger.info(
            "BUILDER2_CLOSURE_OUTPUT_PROBED measuredFinalDurationSeconds=%.3f measuredSourceDurationSeconds=%.3f effectiveClosureDurationSeconds=%.3f",
            measured,
            source_duration,
            hold,
        )
        try:
            duration_diagnostics = verify_builder2_final_video_duration(
                measured,
                visual_duration_seconds=source_duration,
                end_card_duration_seconds=effective_segment,
            )
        except Builder2ClosureRenderError as exc:
            if exc.stage == "duration_verification":
                raise Builder2ClosureRenderError(
                    str(exc.args[0] if exc.args else "builder2_media_final_duration_invalid"),
                    stage=exc.stage,
                    command_category=exc.command_category,
                    duration_diagnostics=exc.duration_diagnostics,
                    closure_ffmpeg_execution_accepted=closure_ffmpeg_execution_accepted,
                    closure_output_file_created=closure_output_file_created,
                    closure_output_file_size_bytes=closure_output_file_size_bytes,
                    closure_ffprobe_calls=closure_ffprobe_calls,
                ) from exc
            raise
        duration_accepted = bool(duration_diagnostics.closure_gain_accepted)
        logger.info(
            "BUILDER2_CLOSURE_DURATION_VERIFIED measuredFinalDurationSeconds=%.3f measuredSourceDurationSeconds=%.3f effectiveClosureDurationSeconds=%.3f durationAccepted=%s",
            measured,
            source_duration,
            hold,
            duration_accepted,
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
            "BUILDER2_CLOSURE_ENDCARD_DONE elapsedMs=%.1f measuredFinalDurationSeconds=%.3f durationAccepted=%s ffmpegReturnCode=%s outputCreated=%s outputSizeBytes=%s",
            (time.monotonic() - t0) * 1000.0,
            measured,
            duration_accepted,
            concat_rc,
            closure_output_file_created,
            closure_output_file_size_bytes,
        )
        return ClosureRenderResult(
            public_url=public_url,
            local_path=str(published_path),
            measured_duration_seconds=measured,
            output_token=token,
            input_fingerprint=input_fingerprint,
            closure_ffprobe_calls=closure_ffprobe_calls,
            duration_diagnostics=duration_diagnostics,
        )
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass
