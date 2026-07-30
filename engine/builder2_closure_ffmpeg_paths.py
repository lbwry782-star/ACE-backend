"""
Builder2 closure FFmpeg asset paths — cross-platform filter escaping and font staging.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

from engine.builder2_tournament_contracts import Builder2TournamentError

_FFMPEG_STDERR_FONT_FALLBACK_PATTERNS = (
    r"could not load font",
    r"fontfile[^\n]*not found",
    r"cannot find a valid font",
    r"no font filename provided",
    r"using fallback font",
    r"fontconfig error:[^\n]*fallback",
)
_FFMPEG_STDERR_FONT_FALLBACK_RE = re.compile(
    "|".join(_FFMPEG_STDERR_FONT_FALLBACK_PATTERNS),
    re.IGNORECASE,
)
_NOTDEF_GLYPH_NAMES = frozenset({".notdef", ".null"})


def closure_ffmpeg_filter_escape_path(path: Path) -> str:
    """Return an FFmpeg filter-safe absolute path (fontfile=/textfile= values)."""
    raw = str(path).replace("\\", "/")
    if raw.startswith("/") and not (len(raw) >= 2 and raw[1] == ":"):
        normalized = raw
    else:
        normalized = str(path.resolve()).replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        normalized = normalized[0] + "\\:" + normalized[2:]
    return normalized.replace("'", "\\'")


def closure_path_requires_ffmpeg_staging(path: Path) -> bool:
    """True when FFmpeg on Windows cannot reliably open the path (e.g. non-ASCII segments)."""
    return any(ord(ch) > 127 for ch in str(path.resolve()))


def write_closure_utf8_textfile(path: Path, text: str) -> Path:
    """Write closure line copy as UTF-8 without BOM."""
    payload = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    encoded = payload.encode("utf-8")
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise Builder2TournamentError("builder2_closure_textfile_encoding_invalid")
    path.write_bytes(encoded)
    return path


def read_closure_utf8_textfile(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Builder2TournamentError("builder2_closure_textfile_encoding_invalid")
    return raw.decode("utf-8")


def _hebrew_codepoints(text: str) -> Iterable[int]:
    for ch in text or "":
        if 0x0590 <= ord(ch) <= 0x05FF:
            yield ord(ch)


def verify_closure_hebrew_font_cmap(font_path: Path, text: str) -> None:
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise Builder2TournamentError("builder2_closure_missing_hebrew_glyph") from exc

    resolved = font_path.resolve()
    if not resolved.is_file():
        raise Builder2TournamentError("builder2_closure_ffmpeg_product_font_not_loaded")

    tt = TTFont(str(resolved), lazy=True)
    try:
        cmap = tt.getBestCmap() or {}
        for codepoint in _hebrew_codepoints(text):
            glyph_name = cmap.get(codepoint)
            if not glyph_name or glyph_name in _NOTDEF_GLYPH_NAMES:
                raise Builder2TournamentError("builder2_closure_missing_hebrew_glyph")
    finally:
        tt.close()


def assert_closure_ffmpeg_stderr_font_health(stderr: bytes | str | None) -> None:
    text = (stderr or b"").decode("utf-8", errors="replace") if isinstance(stderr, (bytes, bytearray)) else str(stderr or "")
    if _FFMPEG_STDERR_FONT_FALLBACK_RE.search(text):
        raise Builder2TournamentError("builder2_closure_ffmpeg_font_fallback_detected")


@dataclass
class ClosureFfmpegAssetSession:
    """Stage closure fonts/textfiles on FFmpeg-safe absolute paths for one render."""

    work_dir: Path
    _staged_fonts: Dict[str, Path] = field(default_factory=dict)

    @classmethod
    def create(cls) -> ClosureFfmpegAssetSession:
        return cls(work_dir=Path(tempfile.mkdtemp(prefix="ace_closure_ff_")))

    def cleanup(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def filter_path(self, path: Path) -> str:
        return closure_ffmpeg_filter_escape_path(path)

    def prepare_font(self, font_path: Path, *, role: str) -> Path:
        resolved = font_path.resolve()
        if not resolved.is_file():
            if role == "product":
                raise Builder2TournamentError("builder2_closure_ffmpeg_product_font_not_loaded")
            raise Builder2TournamentError("builder2_closure_ffmpeg_slogan_font_not_loaded")

        if not closure_path_requires_ffmpeg_staging(resolved):
            return resolved

        cache_key = str(resolved)
        cached = self._staged_fonts.get(cache_key)
        if cached is not None:
            return cached

        staged = self.work_dir / resolved.name
        shutil.copy2(resolved, staged)
        if not staged.is_file():
            if role == "product":
                raise Builder2TournamentError("builder2_closure_ffmpeg_product_font_not_loaded")
            raise Builder2TournamentError("builder2_closure_ffmpeg_slogan_font_not_loaded")
        self._staged_fonts[cache_key] = staged
        return staged

    def write_line_textfile(self, index: int, text: str) -> Path:
        return write_closure_utf8_textfile(self.work_dir / f"line_{index}.txt", text)

    def prepare_line_assets(
        self,
        line_specs: Sequence,
    ) -> tuple[list[Path], list[Path]]:
        text_paths: list[Path] = []
        font_paths: list[Path] = []
        for index, spec in enumerate(line_specs):
            text_paths.append(self.write_line_textfile(index, spec.text))
            font_path = self.prepare_font(spec.font_path, role=spec.role)
            font_paths.append(font_path)
            if spec.use_text_shaping:
                verify_closure_hebrew_font_cmap(font_path, spec.text)
        return text_paths, font_paths


__all__ = [
    "ClosureFfmpegAssetSession",
    "assert_closure_ffmpeg_stderr_font_health",
    "closure_ffmpeg_filter_escape_path",
    "closure_path_requires_ffmpeg_staging",
    "read_closure_utf8_textfile",
    "verify_closure_hebrew_font_cmap",
    "write_closure_utf8_textfile",
]
