"""
Builder2 Advertising Closure end-card rendering — append plain-text end card after Runway video.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

from engine.video_headline_postprocess import (
    _ffmpeg_bin,
    _filter_path_for_ffmpeg,
    _path_for_token,
    _storage_root,
)
from engine.video_language import normalize_video_content_language

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")
_DEFAULT_DURATION = 1.5
_HTTP_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_FFMPEG_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")


def _default_font_path(language: str) -> str:
    from engine.video_headline_postprocess import _default_font_path as headline_font

    return headline_font(language)


def _ffprobe_duration_seconds(path: Path, timeout: float) -> float:
    from engine.video_headline_postprocess import _ffprobe_duration_seconds as probe

    return probe(path, timeout)


def _input_has_audio(path: Path, timeout: float) -> bool:
    from engine.video_headline_postprocess import _input_has_audio as has_audio

    return has_audio(path, timeout)


def _sanitize_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _closure_storage_token(job_id: str) -> str:
    return uuid.uuid4().hex


def append_advertising_closure_endcard(
    source_video_url: str,
    public_base_url: str,
    *,
    product_name: str,
    slogan: str,
    language: str = "he",
    duration_seconds: float = _DEFAULT_DURATION,
    job_id: str = "",
    ffmpeg_runner: Optional[Callable[..., None]] = None,
) -> str:
    product = _sanitize_line(product_name)
    slogan_text = _sanitize_line(slogan)
    if not product or not slogan_text:
        logger.warning("BUILDER2_CLOSURE_ENDCARD skipped reason=missing_text")
        return source_video_url
    base = (public_base_url or "").strip().rstrip("/")
    if not base:
        from engine.public_base_url import resolve_public_base_url

        resolution = resolve_public_base_url()
        if resolution.configured:
            base = resolution.value
    if not base:
        logger.warning("BUILDER2_CLOSURE_ENDCARD skipped reason=no_public_base_url")
        return source_video_url

    lang = normalize_video_content_language(language)
    ffmpeg = _ffmpeg_bin()
    font = _default_font_path(lang)
    if not ffmpeg or not font:
        logger.warning("BUILDER2_CLOSURE_ENDCARD skipped reason=missing_ffmpeg_or_font")
        return source_video_url

    token = _closure_storage_token(job_id)
    out_path = _path_for_token(token)
    if out_path is None:
        return source_video_url

    tmp = Path(tempfile.mkdtemp(prefix="ace_closure_endcard_"))
    inp = tmp / "in.mp4"
    card = tmp / "card.mp4"
    concat_list = tmp / "concat.txt"
    out_tmp = tmp / "out.mp4"
    product_file = tmp / "product.txt"
    slogan_file = tmp / "slogan.txt"

    try:
        _storage_root().mkdir(parents=True, exist_ok=True)
        response = requests.get(source_video_url, timeout=_HTTP_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        with open(inp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)

        source_duration = _ffprobe_duration_seconds(inp, _FFPROBE_TIMEOUT)
        has_audio = _input_has_audio(inp, _FFPROBE_TIMEOUT)
        hold = max(0.5, float(duration_seconds or _DEFAULT_DURATION))
        font_path = _filter_path_for_ffmpeg(Path(font))
        product_file.write_text(product, encoding="utf-8")
        slogan_file.write_text(slogan_text, encoding="utf-8")

        card_filter = (
            f"color=c=black:s=1280x720:d={hold}[bg];"
            f"[bg]drawtext=fontfile='{font_path}':textfile='{_filter_path_for_ffmpeg(product_file)}':"
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
            f"color=c=black:s=1280x720:d={hold}",
            "-vf",
            card_filter.split("[bg]", 1)[1] if "[bg]" in card_filter else card_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(card),
        ]
        card_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=1280x720:d={hold}",
            "-vf",
            (
                f"drawtext=fontfile='{font_path}':textfile='{_filter_path_for_ffmpeg(product_file)}':"
                f"fontcolor=white:fontsize=52:x=(w-text_w)/2:y=(h/2)-70:borderw=2:bordercolor=black@0.35,"
                f"drawtext=fontfile='{font_path}':textfile='{_filter_path_for_ffmpeg(slogan_file)}':"
                f"fontcolor=white:fontsize=40:x=(w-text_w)/2:y=(h/2)+10:borderw=2:bordercolor=black@0.35"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(card),
        ]

        concat_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(inp),
            "-i",
            str(card),
            "-filter_complex",
            (
                "[0:v][1:v]concat=n=2:v=1:a=0[vout];"
                + ("[0:a]apad,atrim=0:" + f"{source_duration + hold}[aout]" if has_audio else "")
            ),
            "-map",
            "[vout]",
        ]
        if has_audio:
            concat_cmd.extend(["-map", "[aout]", "-c:a", "aac"])
        concat_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_tmp)])

        def _run(cmd: list[str]) -> None:
            subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT, capture_output=True)

        runner = ffmpeg_runner or _run
        logger.info("BUILDER2_CLOSURE_ENDCARD start jobId=%s", (job_id or "").strip() or "(none)")
        t0 = time.monotonic()
        runner(card_cmd)
        runner(concat_cmd)
        out_tmp.replace(out_path)
        upload_secret = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
        if upload_secret:
            upload_endpoint = f"{base}/api/video-headline-artifact"
            with open(out_path, "rb") as handle:
                upload = requests.post(
                    upload_endpoint,
                    headers={"X-ACE-Video-Headline-Upload-Secret": upload_secret},
                    files={"file": ("closure.mp4", handle, "video/mp4")},
                    data={"token": token},
                    timeout=_FFMPEG_TIMEOUT,
                )
            if upload.ok:
                return f"{base}/api/video-headline/{token}"
        return f"{base}/api/video-headline/{token}"
    except Exception as exc:
        logger.warning("BUILDER2_CLOSURE_ENDCARD failed reason=%s", type(exc).__name__)
        return source_video_url
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass
