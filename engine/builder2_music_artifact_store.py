"""
Builder2 Lyria music artifact store — Web Service disk-backed MP3 persistence.

Uses the same persistent storage root as headline/final-video when configured.
Builder2-only.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")


def _clean_env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def resolve_builder2_music_artifact_storage_root() -> Path:
    explicit = _clean_env("BUILDER2_MUSIC_ARTIFACT_STORAGE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    shared = _clean_env("VIDEO_HEADLINE_STORAGE_DIR")
    if shared:
        return Path(shared).expanduser() / "builder2_music"
    return Path(tempfile.gettempdir()) / "ace_builder2_music_store"


def explicit_builder2_music_storage_configured() -> bool:
    return bool(_clean_env("BUILDER2_MUSIC_ARTIFACT_STORAGE_DIR") or _clean_env("VIDEO_HEADLINE_STORAGE_DIR"))


def classify_music_artifact_backend_kind() -> str:
    if not explicit_builder2_music_storage_configured():
        return "ephemeral_tmp"
    return "persistent_disk"


def _path_for_token(token: str) -> Optional[Path]:
    t = (token or "").strip()
    if not _TOKEN_RE.match(t):
        return None
    try:
        root = resolve_builder2_music_artifact_storage_root().resolve()
        path = (root / f"{t}.mp3").resolve()
        if path.parent != root:
            return None
        return path
    except OSError:
        return None


def get_builder2_music_artifact_path(token: str) -> Optional[Path]:
    path = _path_for_token(token)
    if path is None:
        return None
    return path if path.is_file() else None


def write_builder2_music_artifact_bytes(token: str, data: bytes) -> bool:
    t = (token or "").strip()
    if not _TOKEN_RE.match(t) or not data:
        return False
    path = _path_for_token(t)
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        path.write_bytes(data)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
