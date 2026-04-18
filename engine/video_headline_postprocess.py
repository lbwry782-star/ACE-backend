"""
Post-process Runway MP4: overlay planner headline on the last ~1.5s of the video (white, centered, fade-in only).

No black card, no tail extension — same duration as source; original picture stays visible underneath.

Processed files are stored on disk under VIDEO_HEADLINE_STORAGE_DIR as {token}.mp4 so they survive
worker restarts (lookup is disk-only, not in-memory).

Success path (split deploy): worker POSTs bytes to {ACE_PUBLIC_BASE_URL}/api/video-headline-artifact
with ACE_VIDEO_HEADLINE_UPLOAD_SECRET; web stores file; final URL is /api/video-headline/<token>.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Optional, Tuple

import requests

from engine.video_language import (
    is_english_only_product_name_script,
    normalize_video_content_language,
    normalize_video_overlay_text,
)

logger = logging.getLogger(__name__)

# Only uuid4().hex tokens (32 lowercase hex chars) map to files — no path traversal
_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")

# Duration (seconds) of headline overlay on the final part of the video (default last 1.5s)
_HOLD_SECONDS = float((os.environ.get("VIDEO_HEADLINE_HOLD_SECONDS") or "1.5").strip() or "1.5")
# Opacity fade-in at the start of that overlay window (seconds)
_TEXT_FADE_SECONDS = float((os.environ.get("VIDEO_HEADLINE_TEXT_FADE_SECONDS") or "0.5").strip() or "0.5")
_HTTP_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_FFPROBE_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
# Single re-encode pass is typically faster than previous concat; default allows headroom on slow hosts.
_FFMPEG_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")
_UPLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_UPLOAD_TIMEOUT_SECONDS") or "120").strip() or "120")
_HARD_TEST_JOB_RE = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")


def _safe_job_id_segment(job_id: str) -> str:
    s = (job_id or "").strip()
    if not s:
        return ""
    if _HARD_TEST_JOB_RE.fullmatch(s):
        return s
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:128] or ""


def hard_test_video_path(job_id: str) -> Optional[Path]:
    """
    Predictable path: /tmp/ace_video_test_<jobId>.mp4 (jobId passed through _safe_job_id_segment).
    Used by worker (write) and web (send_file).
    """
    seg = _safe_job_id_segment(job_id)
    if not seg:
        return None
    try:
        root = Path("/tmp").resolve()
        p = (root / f"ace_video_test_{seg}.mp4").resolve()
        if p.parent != root:
            return None
        return p
    except OSError:
        return None


def log_video_headline_delivery_startup(service_name: str) -> None:
    """
    Log once at process start: whether artifact upload can run (secret) and public base hints.
    Never logs the secret value — only booleans and lengths.
    """
    sec = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
    pub = (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip()
    logger.info(
        "ACE_PUBLIC_BASE_URL_RUNTIME service=%s value=%s",
        service_name,
        pub,
    )
    logger.info(
        "VIDEO_HEADLINE_UPLOAD_CONFIG service=%s secret_present=%s secret_len=%s "
        "public_base_env_present=%s public_base_len=%s",
        service_name,
        bool(sec),
        len(sec),
        bool(pub),
        len(pub),
    )


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


def write_headline_video_bytes(token: str, data: bytes) -> bool:
    """
    Persist processed MP4 bytes for a valid token (same path as ffmpeg output).
    Used by the web service when the worker POSTs the artifact (split deploy).
    """
    if not data:
        return False
    p = _path_for_token(token)
    if p is None:
        return False
    try:
        _storage_root().mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return True
    except OSError:
        return False


def _default_font_path(overlay_language: str = "he") -> Optional[str]:
    """Prefer fonts with Hebrew coverage when overlay_language is Hebrew."""
    env = (os.environ.get("VIDEO_HEADLINE_FONT") or "").strip()
    if env and Path(env).is_file():
        return env
    lang = normalize_video_content_language(overlay_language)
    candidates: list[str] = []
    if lang == "he":
        candidates.extend(
            [
                "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansHebrew-Regular.otf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
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


def _ffprobe_video_dimensions(path: Path, timeout_sec: float) -> Tuple[int, int]:
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe size failed: {r.stderr or r.stdout}")
    line = (r.stdout or "").strip()
    parts = [int(x) for x in line.split(",") if x.strip().isdigit()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    raise RuntimeError(f"ffprobe size parse failed: {line!r}")


def _pillow_text_width_px(font_path: str, text: str, fontsize: int) -> int:
    from PIL import ImageFont

    f = ImageFont.truetype(font_path, fontsize)
    if hasattr(f, "getlength"):
        return max(1, int(round(f.getlength(text))))
    bbox = f.getbbox(text)
    return max(1, bbox[2] - bbox[0])


def _overlay_product_remainder_fontsizes(base_fs: int) -> Tuple[int, int]:
    """Headline remainder (smaller) and product name (larger), for burn-in emphasis without punctuation."""
    fs_r = max(22, base_fs)
    fs_p = min(fs_r + max(8, fs_r // 5), 78)
    return fs_r, fs_p


def _build_dual_drawtext_vf(
    font_e: str,
    tf_remainder_e: str,
    tf_product_e: str,
    fs_remainder: int,
    fs_product: int,
    x_remainder: int,
    x_product: int,
    alpha_expr: str,
    t0_str: str,
    *,
    remainder_text_shaping: bool,
    product_text_shaping: bool,
) -> str:
    """
    Two drawtext chains: remainder on the left (smaller), product name on the right (larger).
    Same vertical centering; gap = one space width between blocks (via x positions only).
    """
    sh_r = ":text_shaping=1" if remainder_text_shaping else ""
    sh_p = ":text_shaping=1" if product_text_shaping else ""
    rows: list[Tuple[int, int, str]] = [
        (1, 0, ""),
        (0, 1, ""),
        (0, 0, ":shadowcolor=black@0.5:shadowx=2:shadowy=2"),
    ]
    parts: list[str] = []
    for ox, oy, shadow in rows:
        y_expr = "(h-text_h)/2+1" if oy else "(h-text_h)/2"
        for tf_path, xb, fs, sh in (
            (tf_remainder_e, x_remainder, fs_remainder, sh_r),
            (tf_product_e, x_product, fs_product, sh_p),
        ):
            xe = xb + ox
            parts.append(
                f"drawtext=fontfile='{font_e}':textfile='{tf_path}':fontsize={fs}"
                f":fontcolor=white{sh}:alpha='{alpha_expr}':enable='gte(t\\,{t0_str})'"
                f"{shadow}:x={xe}:y={y_expr}"
            )
    return ",".join(parts)


def _input_has_audio(path: Path, timeout_sec: float) -> bool:
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return False
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    return r.returncode == 0 and bool((r.stdout or "").strip())


def _filter_path_for_ffmpeg(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return s.replace(":", "\\:")


def _fontsize_for_headline(text: str) -> int:
    n = len(text)
    if n <= 28:
        return 60
    if n <= 45:
        return 48
    return 40


def _sanitize_headline_line(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\r\n]+", " ", t)
    return t


def _has_hebrew_letter(s: str) -> bool:
    for ch in s or "":
        if 0x0590 <= ord(ch) <= 0x05FF:
            return True
    return False


def _has_ascii_latin_letter(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _mixed_hebrew_latin_headline(s: str) -> bool:
    """Hebrew + Latin in one line — must never be sent as a single drawtext when overlay is Hebrew."""
    return _has_hebrew_letter(s) and _has_ascii_latin_letter(s)


def _split_overlay_product_remainder(
    headline: str, canonical_name: str
) -> Optional[Tuple[str, str]]:
    """
    Same composition as prepare_ffmpeg_overlay_headline: ``<product> <remainder>`` (one space; no comma).
    Returns (product, remainder). No bidi control characters.
    """
    cn = unicodedata.normalize("NFC", (canonical_name or "").strip())
    if not cn:
        return None
    h = unicodedata.normalize("NFC", (headline or "").strip())
    if not h:
        return None
    hs = h.strip()
    if len(hs) < len(cn) or hs[: len(cn)].lower() != cn.lower():
        return None
    tail = hs[len(cn) :].lstrip()
    tail = re.sub(r"^[\s,·\u00b7\u2022•:;|–—−\-]+", "", tail)
    tail = re.sub(r"\s+", " ", tail).strip()
    if not tail:
        return None
    if is_english_only_product_name_script(cn):
        if not _has_hebrew_letter(tail):
            return None
        return (cn, tail)
    if _has_hebrew_letter(cn):
        if not _has_hebrew_letter(tail):
            return None
        return (cn, tail)
    return None


def postprocess_video_headline(
    source_video_url: str,
    public_base_url: str,
    *,
    headline: str = "",
    job_id: str = "",
    overlay_language: str = "he",
    overlay_render_mode: str = "plain_text",
    overlay_dual_latin: str = "",
    overlay_dual_hebrew: str = "",
    overlay_canonical_name: str = "",
) -> str:
    """
    Download MP4, one ffmpeg pass: draw the short planner headline (headlineText, not marketing body copy)
    on top of the video for the last HOLD_SECONDS only (white, centered, opacity fade-in).
    Same length as input; no black frame, no tpad.
    """
    olang = normalize_video_content_language(overlay_language)
    headline_clean = _sanitize_headline_line(headline)
    headline_clean = normalize_video_overlay_text(headline_clean, olang)
    if not headline_clean:
        logger.info(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=empty_headline"
        )
        return source_video_url

    base = (public_base_url or "").strip().rstrip("/")
    if not base:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=no_public_base_url"
        )
        return source_video_url

    ffmpeg = _ffmpeg_bin()
    font = _default_font_path(olang)
    if not ffmpeg or not font:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=missing_ffmpeg_or_font "
            "ffmpeg=%s font=%s",
            bool(ffmpeg),
            bool(font),
        )
        return source_video_url

    t0 = time.monotonic()
    logger.info("VIDEO_HEADLINE_POSTPROCESS start")

    tmp = Path(tempfile.mkdtemp(prefix="ace_vid_headline_"))
    inp = tmp / "in.mp4"
    text_file = tmp / "headline.txt"
    text_file_latin = tmp / "overlay_latin.txt"
    text_file_hebrew = tmp / "overlay_hebrew.txt"

    token = uuid.uuid4().hex
    out_path = _path_for_token(token)
    if out_path is None:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=token_internal_error"
        )
        return source_video_url

    def _fail(reason: str) -> str:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=%s",
            reason,
        )
        try:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return source_video_url

    try:
        try:
            _storage_root().mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return _fail(f"storage_mkdir:{type(e).__name__}")

        try:
            r = requests.get(source_video_url, timeout=_HTTP_DOWNLOAD_TIMEOUT, stream=True)
            r.raise_for_status()
        except requests.Timeout:
            return _fail("download_timeout")
        except requests.RequestException as e:
            return _fail(f"download_error:{type(e).__name__}")

        with open(inp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        try:
            duration_sec = _ffprobe_duration_seconds(inp, _FFPROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _fail("ffprobe_timeout")
        except Exception as e:
            return _fail(f"ffprobe_error:{type(e).__name__}")

        if duration_sec <= 0:
            return _fail("invalid_duration")

        has_audio = _input_has_audio(inp, _FFPROBE_TIMEOUT)

        fs = _fontsize_for_headline(headline_clean)
        font_e = _filter_path_for_ffmpeg(Path(font))

        cn = (overlay_canonical_name or "").strip()
        lat_s = (overlay_dual_latin or "").strip()
        he_s = (overlay_dual_hebrew or "").strip()
        if olang == "he" and (not lat_s or not he_s):
            sp = _split_overlay_product_remainder(headline_clean, cn)
            if sp:
                lat_s, he_s = sp

        use_dual = olang == "he" and bool(lat_s and he_s)

        if olang == "he" and _mixed_hebrew_latin_headline(headline_clean) and not use_dual:
            logger.error(
                "VIDEO_HEADLINE_INVALID_MODE_SINGLE_STRING reason=mixed_he_latin_without_dual_parts "
                "headline=%s canonical=%s render_mode=%s",
                json.dumps(headline_clean, ensure_ascii=False),
                json.dumps(cn, ensure_ascii=False),
                json.dumps((overlay_render_mode or "").strip(), ensure_ascii=False),
            )
            return _fail("mixed_headline_requires_dual_drawtext")

        video_w = 1920
        x_latin = x_hebrew = 0
        if use_dual:
            try:
                video_w, _ = _ffprobe_video_dimensions(inp, _FFPROBE_TIMEOUT)
            except Exception as e:
                logger.warning(
                    "VIDEO_HEADLINE_FFPROBE_DIMS_FALLBACK use_w=1920 err=%s",
                    e,
                )
                video_w = 1920
            try:
                fs_rem, fs_prod = _overlay_product_remainder_fontsizes(fs)
                tw_he = _pillow_text_width_px(font, he_s, fs_rem)
                tw_lat = _pillow_text_width_px(font, lat_s, fs_prod)
                tw_space = _pillow_text_width_px(font, " ", fs_rem)
                # Visual: remainder (left) <space> product (right, larger).
                total = tw_he + tw_space + tw_lat
                x_hebrew = max(0, (video_w - total) // 2)
                x_latin = x_hebrew + tw_he + tw_space
                text_file_latin.write_text(lat_s, encoding="utf-8")
                text_file_hebrew.write_text(he_s, encoding="utf-8")
                logger.info("VIDEO_HEADLINE_OVERLAY_MODE=dual_only")
                logger.info("VIDEO_HEADLINE_SINGLE_STRING_DISABLED=true")
                logger.info(
                    "VIDEO_HEADLINE_PREFIX=%s",
                    json.dumps(lat_s, ensure_ascii=False),
                )
                logger.info(
                    "VIDEO_HEADLINE_REMAINDER=%s",
                    json.dumps(he_s, ensure_ascii=False),
                )
                logger.info(
                    "VIDEO_HEADLINE_POSTPROCESS dual_layout video_w=%s fs_rem=%s fs_prod=%s tw_rem=%s tw_space=%s tw_prod=%s x_rem=%s x_prod=%s",
                    video_w,
                    fs_rem,
                    fs_prod,
                    tw_he,
                    tw_space,
                    tw_lat,
                    x_hebrew,
                    x_latin,
                )
            except Exception as e:
                logger.error(
                    "VIDEO_HEADLINE_DUAL_SETUP_FAILED err=%s",
                    e,
                    exc_info=True,
                )
                return _fail("dual_drawtext_setup_failed")

        if not use_dual:
            logger.info("VIDEO_HEADLINE_OVERLAY_MODE=plain_centered")
            logger.info("VIDEO_HEADLINE_SINGLE_STRING_DISABLED=false")
            text_file.write_text(headline_clean, encoding="utf-8")

        tf_e = _filter_path_for_ffmpeg(text_file)
        tf_lat_e = _filter_path_for_ffmpeg(text_file_latin)
        tf_he_e = _filter_path_for_ffmpeg(text_file_hebrew)

        # Last N seconds of the source video (or full length if shorter than N)
        overlay_s = min(max(0.4, _HOLD_SECONDS), duration_sec)
        t_overlay_start = duration_sec - overlay_s
        t0_str = f"{t_overlay_start:.4f}"
        fade_s = max(0.05, min(_TEXT_FADE_SECONDS, max(overlay_s - 0.05, 0.05)))
        fade_end = t_overlay_start + fade_s
        fade_end_str = f"{fade_end:.4f}"
        # Opacity: 0 before overlay window; linear 0→1 during fade_s at start of window; 1 until end (no motion/scale).
        alpha_expr = (
            f"if(lt(t\\,{t0_str})\\,0\\,if(lt(t\\,{fade_end_str})\\,(t-{t0_str})/{fade_s}\\,1))"
        )
        # text_shaping=1: HarfBuzz/fribidi for Hebrew RTL shaping (ffmpeg build-dependent).
        shaping = ":text_shaping=1" if olang == "he" else ""
        logger.info(
            "VIDEO_HEADLINE_POSTPROCESS render=%s dual=%s",
            "dual_drawtext" if use_dual else "plain_text",
            str(use_dual).lower(),
        )
        if use_dual:
            fs_rem, fs_prod = _overlay_product_remainder_fontsizes(fs)
            vf = _build_dual_drawtext_vf(
                font_e,
                tf_he_e,
                tf_lat_e,
                fs_rem,
                fs_prod,
                x_hebrew,
                x_latin,
                alpha_expr,
                t0_str,
                remainder_text_shaping=_has_hebrew_letter(he_s),
                product_text_shaping=_has_hebrew_letter(lat_s),
            )
        else:
            dt = (
                f"fontfile='{font_e}':textfile='{tf_e}':fontsize={fs}:fontcolor=white{shaping}:"
                f"alpha='{alpha_expr}':enable='gte(t\\,{t0_str})'"
            )
            vf = (
                f"drawtext={dt}:x=(w-text_w)/2+1:y=(h-text_h)/2,"
                f"drawtext={dt}:x=(w-text_w)/2:y=(h-text_h)/2+1,"
                f"drawtext={dt}:shadowcolor=black@0.5:shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:y=(h-text_h)/2"
            )

        cmd: list[str] = [
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
            "-pix_fmt",
            "yuv420p",
        ]
        if has_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.append("-an")

        cmd.extend(["-movflags", "+faststart", str(out_path)])

        vf_preview = vf[:220] + ("…" if len(vf) > 220 else "")
        logger.info(
            "VIDEO_HEADLINE_POSTPROCESS_CMD overlay_last_s=%s text_fade_s=%s duration_s=%s has_audio=%s vf_preview=%r",
            overlay_s,
            fade_s,
            duration_sec,
            has_audio,
            vf_preview,
        )

        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return _fail("ffmpeg_timeout")

        if p.returncode != 0:
            return _fail(f"ffmpeg_exit:{p.returncode}:stderr_len={len(p.stderr or '')}")

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        preview = headline_clean[:200] + ("…" if len(headline_clean) > 200 else "")
        logger.info(
            'VIDEO_HEADLINE_POSTPROCESS applied headline="%s" elapsed_ms=%s',
            preview,
            elapsed_ms,
        )
        out_exists = bool(out_path.is_file())
        if not out_exists:
            logger.warning(
                "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=no_local_artifact_after_ffmpeg"
            )
            return source_video_url

        upload_secret = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
        if not upload_secret:
            logger.warning(
                "VIDEO_HEADLINE_UPLOAD failed fallback_to_original=true reason=no_upload_secret"
            )
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return source_video_url

        logger.info("VIDEO_HEADLINE_UPLOAD_RESOLVED_BASE_URL value=%s", base)
        upload_endpoint = f"{base}/api/video-headline-artifact"
        logger.info(
            "VIDEO_HEADLINE_UPLOAD start endpoint=%s token_prefix=%s",
            upload_endpoint,
            token[:8] if len(token) >= 8 else token,
        )
        try:
            with open(out_path, "rb") as fp:
                up = requests.post(
                    upload_endpoint,
                    headers={"X-ACE-Video-Headline-Upload-Secret": upload_secret},
                    files={"file": ("headline.mp4", fp, "video/mp4")},
                    data={"token": token},
                    timeout=_UPLOAD_TIMEOUT,
                )
        except requests.RequestException as e:
            logger.warning(
                "VIDEO_HEADLINE_UPLOAD failed fallback_to_original=true reason=request_exception:%s",
                type(e).__name__,
            )
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return source_video_url

        ok_body = False
        if up.status_code == 200:
            try:
                j = up.json()
                ok_body = isinstance(j, dict) and j.get("ok") is True
            except ValueError:
                ok_body = False

        if not ok_body:
            logger.warning(
                "VIDEO_HEADLINE_UPLOAD failed fallback_to_original=true http_status=%s body_len=%s",
                up.status_code,
                len(up.content or b""),
            )
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return source_video_url

        public_url = f"{base}/api/video-headline/{token}"
        logger.info("VIDEO_HEADLINE_UPLOAD success final_url=%s", public_url)
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.info("VIDEO_HEADLINE_POSTPROCESS_RESULT public_url=%s", public_url)
        return public_url
    except Exception as e:
        logger.warning(
            "VIDEO_HEADLINE_POSTPROCESS failed fallback_to_original=true reason=exception:%s err=%s",
            type(e).__name__,
            e,
            exc_info=True,
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
