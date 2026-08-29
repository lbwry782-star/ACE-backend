"""
Builder1 durable final ad image artifacts — disk-backed, campaign-scoped metadata.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")
_ARTIFACT_ROUTE_PREFIX = "/api/builder1-image-artifact/"


from engine.builder1_artifact_storage import resolve_builder1_image_storage_root


def _storage_root() -> Path:
    return resolve_builder1_image_storage_root()


def artifact_token_for_ad(*, campaign_id: str, ad_index: int, plan_revision: int) -> str:
    seed = f"{campaign_id.strip()}:{int(ad_index)}:r{int(plan_revision)}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _path_for_token(token: str) -> Optional[Path]:
    t = (token or "").strip()
    if not _TOKEN_RE.match(t):
        return None
    try:
        root = _storage_root().resolve()
        p = (root / f"{t}.jpg").resolve()
        if p.parent != root:
            return None
        return p
    except OSError:
        return None


def write_builder1_image_artifact_bytes(token: str, image_bytes: bytes) -> Path:
    path = _path_for_token(token)
    if path is None:
        raise ValueError("invalid_artifact_token")
    if not image_bytes:
        raise ValueError("empty_image_bytes")
    root = _storage_root()
    root.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(image_bytes)
    tmp.replace(path)
    logger.info(
        "BUILDER1_IMAGE_ARTIFACT_WRITTEN token=%s bytes=%s path=%s",
        token,
        len(image_bytes),
        path,
    )
    return path


def get_builder1_image_artifact_path(token: str) -> Optional[Path]:
    path = _path_for_token(token)
    if path is None or not path.is_file():
        return None
    return path


def read_builder1_image_artifact_bytes(token: str) -> Optional[bytes]:
    path = get_builder1_image_artifact_path(token)
    if path is None:
        return None
    return path.read_bytes()


def builder1_image_artifact_public_url(token: str) -> str:
    return f"{_ARTIFACT_ROUTE_PREFIX}{token.strip()}"


def ad_artifact_record(
    *,
    campaign_id: str,
    ad_index: int,
    plan_revision: int,
    token: Optional[str] = None,
    status: str = "succeeded",
) -> Dict[str, Any]:
    tok = token or artifact_token_for_ad(
        campaign_id=campaign_id,
        ad_index=ad_index,
        plan_revision=plan_revision,
    )
    return {
        "token": tok,
        "adIndex": int(ad_index),
        "planRevision": int(plan_revision),
        "status": status,
        "artifactUrl": builder1_image_artifact_public_url(tok),
    }


def new_ephemeral_artifact_token() -> str:
    return uuid.uuid4().hex
