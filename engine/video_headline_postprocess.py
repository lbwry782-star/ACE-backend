"""
Post-process Runway MP4: append a black end card with headline (no text on original footage).

Processed files are stored on disk under VIDEO_HEADLINE_STORAGE_DIR as {token}.mp4 so they survive
worker restarts (lookup is disk-only, not in-memory).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Only uuid4().hex tokens (32 lowercase hex chars) map to files — no path traversal
_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")

_HOLD_SECONDS = float((os.environ.get("VIDEO_HEADLINE_HOLD_SECONDS") or "1.5").strip() or "1.5")
_HTTP_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_FFMPEG_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "120").strip() or "120")


def _storage_root() -> Path:
    raw = (os.environ.get("VIDEO_HEADLINE_STORAGE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "ace_video_headline_store"


def _path_for_token(token: str) -> Optional[Path]:
    """Safe resolved path under storage root for a valid token; None if token malformed."""
    t = (token or "").strip()
    if not _TOKEN_RE.match(t):
        return None
    try:
        root = _storage_root().resolve()
        p = (root / f"{t}.mp4").resolve()
        if p.parent != root:
            return None
        return p
    except OSError:
        return None


def get_headline_video_path(token: str) -> Optional[Path]:
    """
    Return path to stored processed MP4 if token is valid and file exists on disk.
    Survives process restart (no in-memory map).
    """
    p = _path_for_token(token)
    if p is None:
        return None
    return p if p.is_file() else None


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


def _ffprobe_streams_json(path: Path, timeout_sec: float) -> Dict[str, Any]:
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr or r.stdout}")
    return json.loads(r.stdout or "{}")


def _video_size_and_audio(
    path: Path, timeout_sec: float
) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """
    First video stream WxH; first audio (sample_rate, channels) or None.
    """
    data = _ffprobe_streams_json(path, timeout_sec)
    streams: List[Dict[str, Any]] = data.get("streams") or []
    w, h = 1280, 720
    audio: Optional[Tuple[int, int]] = None
    for s in streams:
        if s.get("codec_type") == "video":
            tw = int(s.get("width") or 0)
            th = int(s.get("height") or 0)
            if tw > 0 and th > 0:
                w, h = tw, th
            break
    for s in streams:
        if s.get("codec_type") == "audio":
            sr = int(float(s.get("sample_rate") or 48000))
            ch = int(s.get("channels") or 2)
            audio = (sr, ch)
            break
    return w, h, audio


def _filter_path_for_ffmpeg(p: Path) -> str:
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


def postprocess_video_headline(
    source_video_url: str,
    headline: str,
    public_base_url: str,
    headline_decision: Optional[str] = None,
) -> str:
    """
    Download MP4, append black end card with white headline (original video unchanged), return URL.
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
    endcard = tmp / "endcard.mp4"
    text_file = tmp / "headline.txt"

    token = uuid.uuid4().hex
    out_path = _path_for_token(token)
    if out_path is None:
        logger.warning("VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=token_internal_error")
        return source_video_url

    def _fail(reason: str) -> str:
        logger.warning("VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=%s", reason)
        try:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return source_video_url

    hold = max(0.4, _HOLD_SECONDS)

    try:
        try:
            _storage_root().mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return _fail(f"storage_mkdir:{type(e).__name__}")

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
            w, h, audio_info = _video_size_and_audio(inp, _FFPROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS ffprobe_timeout")
            return _fail("ffprobe_timeout")
        except Exception as e:
            return _fail(f"ffprobe_error:{type(e).__name__}")

        fs = _fontsize_for_headline(headline_clean)
        font_e = _filter_path_for_ffmpeg(Path(font))
        tf_e = _filter_path_for_ffmpeg(text_file)

        # Solid black frame + white text only (no overlay on main video)
        vf_end = (
            f"drawtext=fontfile='{font_e}':textfile='{tf_e}':"
            f"fontsize={fs}:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
        )

        if audio_info:
            sr, ch = audio_info
            layout = "stereo" if ch >= 2 else "mono"
            cmd_end: List[str] = [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={w}x{h}:d={hold}",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout={layout}:sample_rate={sr}",
                "-vf",
                vf_end,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                str(hold),
                "-shortest",
                str(endcard),
            ]
        else:
            cmd_end = [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={w}x{h}:d={hold}",
                "-vf",
                vf_end,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-t",
                str(hold),
                str(endcard),
            ]

        try:
            pe = subprocess.run(
                cmd_end,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS ffmpeg_timeout")
            return _fail("ffmpeg_timeout_endcard")

        if pe.returncode != 0:
            return _fail(f"ffmpeg_endcard_exit:{pe.returncode}:stderr_len={len(pe.stderr or '')}")

        # Concat: [original][endcard] — no text on original frames
        if audio_info:
            fc = (
                f"[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]"
            )
            cmd_cat = [
                ffmpeg,
                "-y",
                "-i",
                str(inp),
                "-i",
                str(endcard),
                "-filter_complex",
                fc,
                "-map",
                "[outv]",
                "-map",
                "[outa]",
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
                str(out_path),
            ]
        else:
            fc = "[0:v][1:v]concat=n=2:v=1[outv]"
            cmd_cat = [
                ffmpeg,
                "-y",
                "-i",
                str(inp),
                "-i",
                str(endcard),
                "-filter_complex",
                fc,
                "-map",
                "[outv]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(out_path),
            ]

        concat_timeout = max(_FFMPEG_TIMEOUT, 180.0)
        try:
            pc = subprocess.run(
                cmd_cat,
                capture_output=True,
                text=True,
                check=False,
                timeout=concat_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("VIDEO_HEADLINE_POSTPROCESS ffmpeg_timeout")
            return _fail("ffmpeg_timeout_concat")

        if pc.returncode != 0:
            return _fail(
                f"ffmpeg_concat_exit:{pc.returncode}:stderr_len={len(pc.stderr or '')}"
            )

        preview = headline_clean[:80] + ("…" if len(headline_clean) > 80 else "")
        logger.info(
            'VIDEO_HEADLINE_POSTPROCESS applied headline="%s" endcard=true storage=disk',
            preview,
        )
        final_url = f"{base}/api/video-headline/{token}"
        return final_url
    except Exception as e:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=exception:%s err=%s",
            type(e).__name__,
            e,
        )
        try:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return source_video_url
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass
