"""
Builder2 final-video durable artifact store — disk-backed, distinct from headline /tmp store.
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


def resolve_builder2_final_video_storage_root() -> Path:
    explicit = _clean_env("BUILDER2_FINAL_VIDEO_STORAGE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    shared = _clean_env("VIDEO_HEADLINE_STORAGE_DIR")
    if shared:
        return Path(shared).expanduser() / "builder2_final"
    return Path(tempfile.gettempdir()) / "ace_builder2_final_video_store"


def _system_temp_roots() -> set[Path]:
    roots: set[Path] = set()
    for candidate in (tempfile.gettempdir(), "/tmp"):
        token = (candidate or "").strip()
        if not token:
            continue
        try:
            roots.add(Path(token).expanduser().resolve())
        except OSError:
            continue
    return roots


def explicit_builder2_final_storage_configured() -> bool:
    return bool(_clean_env("BUILDER2_FINAL_VIDEO_STORAGE_DIR") or _clean_env("VIDEO_HEADLINE_STORAGE_DIR"))


def classify_publication_backend_kind() -> str:
    if not explicit_builder2_final_storage_configured():
        root = resolve_builder2_final_video_storage_root()
        try:
            resolved = root.resolve()
        except OSError:
            return "unconfigured"
        for temp_root in _system_temp_roots():
            try:
                if resolved == temp_root or temp_root in resolved.parents:
                    return "ephemeral_tmp"
            except OSError:
                continue
        return "ephemeral_tmp"
    return "persistent_disk"


def is_durable_publication_backend() -> bool:
    return classify_publication_backend_kind() == "persistent_disk"


def _path_for_token(token: str) -> Optional[Path]:
    t = (token or "").strip()
    if not _TOKEN_RE.match(t):
        return None
    try:
        root = resolve_builder2_final_video_storage_root().resolve()
        path = (root / f"{t}.mp4").resolve()
        if path.parent != root:
            return None
        return path
    except OSError:
        return None


def get_builder2_final_video_path(token: str) -> Optional[Path]:
    path = _path_for_token(token)
    if path is None:
        return None
    return path if path.is_file() else None


def write_builder2_final_video_bytes(token: str, data: bytes) -> bool:
    if not data:
        return False
    path = _path_for_token(token)
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True
    except OSError:
        return False
