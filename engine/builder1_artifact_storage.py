"""
Builder1 artifact storage classification — durable vs ephemeral (Builder1-only).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

StorageBackendKind = Literal["persistent_disk", "ephemeral_tmp", "unconfigured"]

_BUILDER1_SUBDIR = "builder1_images"


def _clean_env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def explicit_builder1_image_storage_configured() -> bool:
    return bool(_clean_env("BUILDER1_IMAGE_ARTIFACT_STORAGE_DIR"))


def explicit_shared_headline_storage_configured() -> bool:
    return bool(_clean_env("VIDEO_HEADLINE_STORAGE_DIR"))


def resolve_builder1_image_storage_root() -> Path:
    dedicated = _clean_env("BUILDER1_IMAGE_ARTIFACT_STORAGE_DIR")
    if dedicated:
        return Path(dedicated).expanduser()
    headline_root = _clean_env("VIDEO_HEADLINE_STORAGE_DIR")
    if headline_root:
        return Path(headline_root).expanduser() / _BUILDER1_SUBDIR
    return Path(tempfile.gettempdir()) / "ace_builder1_image_store"


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


def classify_builder1_artifact_backend_kind() -> StorageBackendKind:
    if not explicit_builder1_image_storage_configured() and not explicit_shared_headline_storage_configured():
        return "unconfigured"
    root = resolve_builder1_image_storage_root()
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
    return "persistent_disk"


def builder1_artifact_storage_is_durable() -> bool:
    return classify_builder1_artifact_backend_kind() == "persistent_disk"


def probe_builder1_artifact_storage_writable() -> bool:
    root = resolve_builder1_image_storage_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".builder1_write_probe"
        probe.write_bytes(b"1")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def assert_builder1_durable_artifact_storage_ready() -> None:
    kind = classify_builder1_artifact_backend_kind()
    if kind != "persistent_disk":
        raise RuntimeError("builder1_production_requires_durable_artifact_storage")
    if not probe_builder1_artifact_storage_writable():
        raise RuntimeError("builder1_production_artifact_storage_not_writable")
