"""
Post-process Runway MP4: burn in headline locally with ffmpeg (Runway receives visuals-only prompts).

Does not touch image engine routes. On any failure, returns the original video URL.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Ephemeral token -> path on disk (short-lived; process lifetime)
_video_by_token: Dict[str, Path] = {}

_HOLD_SECONDS = float((os.environ.get("VIDEO_HEADLINE_HOLD_SECONDS") or "1.5").strip() or "1.5")
_HTTP_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_FFMPEG_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "120").strip() or "120")


def _default_font_path() -> Optional[str]:
    env = (os.environ.get("VIDEO_HEADLINE_FONT") or "").strip()
    if env and Path(env).is_file():
        return env
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    return None


def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def _ffprobe_bin() -> Optional[str]:
    return shutil.which("ffprobe")


def _ffprobe_duration_seconds(path: Path, timeout_sec: float) -> float:
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr or r.stdout}")
    return float((r.stdout or "").strip() or 0.0)


def _filter_path_for_ffmpeg(p: Path) -> str:
    # Paths in ffmpeg filter args: escape special chars
    s = str(p.resolve()).replace("\\", "/")
    return s.replace(":", "\\:")


def _fontsize_for_headline(text: str) -> int:
    n = len(text)
    if n <= 28:
        return 52
    if n <= 45:
        return 40
    return 32


def _sanitize_headline(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\r\n]+", " ", t)
    return t


def get_headline_video_path(token: str) -> Optional[Path]:
    with _lock:
        return _video_by_token.get(token)


def postprocess_video_headline(
    source_video_url: str,
    headline: str,
    public_base_url: str,
    headline_decision: Optional[str] = None,
) -> str:
    """
    If plan calls for a headline (not no_headline) and headline text non-empty: download MP4,
    draw exact text on last ~1.5s, return URL to this server. Otherwise return source_video_url unchanged.
    """
    dec = (headline_decision or "").strip()
    if dec == "no_headline":
        logger.info("VIDEO_HEADLINE_POSTPROCESS skipped no_headline")
        return source_video_url

    headline_clean = _sanitize_headline(headline)
    if not headline_clean:
        logger.info("VIDEO_HEADLINE_POSTPROCESS skipped no_headline")
        return source_video_url

    base = (public_base_url or "").strip().rstrip("/")
    if not base:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=no_public_base_url"
        )
        return source_video_url

    ffmpeg = _ffmpeg_bin()
    font = _default_font_path()
    if not ffmpeg or not font:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=missing_ffmpeg_or_font "
            "ffmpeg=%s font=%s",
            bool(ffmpeg),
            bool(font),
        )
        return source_video_url

    logger.info("VIDEO_HEADLINE_POSTPROCESS start")
    tmp = Path(tempfile.mkdtemp(prefix="ace_vid_headline_"))
    inp = tmp / "in.mp4"
    out = tmp / "out.mp4"
    text_file = tmp / "headline.txt"

    def _fail(reason: str) -> str:
        logger.warning("VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=%s", reason)
        return source_video_url

    try:
        text_file.write_text(headline_clean, encoding="utf-8")
        try:
            r = requests.get(source_video_url, timeout=_HTTP_DOWNLOAD_TIMEOUT, stream=True)
            r.raise_for_status()
        except requests.Timeout:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS download_timeout")
            return _fail("download_timeout")
        except requests.RequestException as e:
            return _fail(f"download_error:{type(e).__name__}")

        with open(inp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        try:
            duration = _ffprobe_duration_seconds(inp, _FFPROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS ffprobe_timeout")
            return _fail("ffprobe_timeout")
        except Exception as e:
            return _fail(f"ffprobe_error:{type(e).__name__}")

        hold = min(_HOLD_SECONDS, max(0.4, duration * 0.35))
        start_t = max(0.0, duration - hold)
        fs = _fontsize_for_headline(headline_clean)

        font_e = _filter_path_for_ffmpeg(Path(font))
        tf_e = _filter_path_for_ffmpeg(text_file)

        vf = (
            f"drawtext=fontfile='{font_e}':textfile='{tf_e}':"
            f"enable='gte(t\\,{start_t:.4f})':"
            f"fontsize={fs}:fontcolor=white:borderw=3:bordercolor=black@0.9:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"box=1:boxcolor=black@0.45:boxborderw=12"
        )

        cmd_copy = [
            ffmpeg,
            "-y",
            "-i",
            str(inp),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(out),
        ]
        try:
            p = subprocess.run(
                cmd_copy,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS ffmpeg_timeout")
            return _fail("ffmpeg_timeout")

        if p.returncode != 0:
            cmd_aac = [
                ffmpeg,
                "-y",
                "-i",
                str(inp),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out),
            ]
            try:
                p2 = subprocess.run(
                    cmd_aac,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_FFMPEG_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.warning("VIDEO_HEADLINE_POSTPROCESS ffmpeg_timeout")
                return _fail("ffmpeg_timeout")
            if p2.returncode != 0:
                return _fail(
                    f"ffmpeg_exit:{p2.returncode}:stderr_len={len(p2.stderr or '')}"
                )

        token = uuid.uuid4().hex
        with _lock:
            _video_by_token[token] = out

        final_url = f"{base}/api/video-headline/{token}"
        preview = headline_clean[:80] + ("…" if len(headline_clean) > 80 else "")
        logger.info('VIDEO_HEADLINE_POSTPROCESS applied headline="%s"', preview)
        return final_url
    except Exception as e:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=exception:%s err=%s",
            type(e).__name__,
            e,
        )
        return source_video_url
    finally:
        try:
            if inp.exists():
                inp.unlink(missing_ok=True)
        except OSError:
            pass
