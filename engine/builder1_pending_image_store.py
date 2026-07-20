"""
Temporary pending Builder1 ad images awaiting compliance review (REVIEW_ONLY retry).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PENDING_IMAGE_TTL_SECONDS = 24 * 60 * 60
PENDING_IMAGE_KEY_PREFIX = "builder1:pending-image:"
# Align with Builder1 zip export cap; raw image bytes before base64 expansion.
PENDING_IMAGE_MAX_BYTES = 12 * 1024 * 1024

_memory_lock = threading.Lock()
_memory_pending: Dict[str, Dict[str, Any]] = {}


class PendingImageStoreError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message or code
        super().__init__(self.message)


@dataclass
class PendingBuilder1Image:
    reference: str
    campaign_id: str
    ad_index: int
    plan_revision: int
    image_bytes: bytes
    visual_prompt: str
    created_at: float


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def pending_image_max_bytes() -> int:
    raw = (os.environ.get("BUILDER1_PENDING_IMAGE_MAX_BYTES") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return PENDING_IMAGE_MAX_BYTES


def pending_image_reference(*, campaign_id: str, ad_index: int, plan_revision: int) -> str:
    return f"{campaign_id.strip()}:{int(ad_index)}:r{int(plan_revision)}"


def save_pending_image(
    *,
    campaign_id: str,
    ad_index: int,
    plan_revision: int,
    image_bytes: bytes,
    visual_prompt: str,
) -> str:
    max_bytes = pending_image_max_bytes()
    if len(image_bytes) > max_bytes:
        logger.error(
            "BUILDER1_PENDING_IMAGE_TOO_LARGE campaignId=%s adIndex=%s planRevision=%s bytes=%s maxBytes=%s",
            campaign_id,
            ad_index,
            plan_revision,
            len(image_bytes),
            max_bytes,
        )
        raise PendingImageStoreError(
            "pending_image_too_large",
            f"pending image exceeds max bytes ({max_bytes})",
        )

    reference = pending_image_reference(
        campaign_id=campaign_id,
        ad_index=ad_index,
        plan_revision=plan_revision,
    )
    payload = {
        "reference": reference,
        "campaignId": campaign_id.strip(),
        "adIndex": int(ad_index),
        "planRevision": int(plan_revision),
        "visualPrompt": str(visual_prompt or ""),
        "imageBase64": base64.b64encode(image_bytes).decode("ascii"),
        "createdAt": time.time(),
    }
    redis_key = f"{PENDING_IMAGE_KEY_PREFIX}{reference}"
    if _redis_configured():
        try:
            _get_redis().set(
                redis_key,
                json.dumps(payload),
                ex=PENDING_IMAGE_TTL_SECONDS,
            )
        except Exception as exc:
            logger.error(
                "BUILDER1_PENDING_IMAGE_SAVE_ERR campaignId=%s adIndex=%s err=%s",
                campaign_id,
                ad_index,
                exc,
            )
            raise PendingImageStoreError("pending_image_store_error", str(exc)) from exc
    else:
        with _memory_lock:
            _memory_pending[reference] = payload
    logger.info(
        "BUILDER1_PENDING_IMAGE_SAVED campaignId=%s adIndex=%s planRevision=%s reference=%s bytes=%s redisKey=%s ttlSeconds=%s",
        campaign_id,
        ad_index,
        plan_revision,
        reference,
        len(image_bytes),
        redis_key,
        PENDING_IMAGE_TTL_SECONDS,
    )
    return reference


def load_pending_image(reference: str) -> PendingBuilder1Image:
    ref = (reference or "").strip()
    if not ref:
        raise PendingImageStoreError("pending_image_missing")
    payload: Optional[Dict[str, Any]] = None
    if _redis_configured():
        try:
            raw = _get_redis().get(f"{PENDING_IMAGE_KEY_PREFIX}{ref}")
            if raw:
                payload = json.loads(raw)
        except Exception as exc:
            logger.error("BUILDER1_PENDING_IMAGE_LOAD_ERR reference=%s err=%s", ref, exc)
            raise PendingImageStoreError("pending_image_store_error", str(exc)) from exc
    else:
        with _memory_lock:
            payload = dict(_memory_pending.get(ref) or {})
    if not payload:
        raise PendingImageStoreError("pending_image_missing")
    try:
        image_bytes = base64.b64decode(str(payload.get("imageBase64") or ""), validate=True)
    except Exception as exc:
        raise PendingImageStoreError("pending_image_corrupt", str(exc)) from exc
    if not image_bytes:
        raise PendingImageStoreError("pending_image_corrupt")
    return PendingBuilder1Image(
        reference=ref,
        campaign_id=str(payload.get("campaignId") or ""),
        ad_index=int(payload.get("adIndex") or 0),
        plan_revision=int(payload.get("planRevision") or 1),
        image_bytes=image_bytes,
        visual_prompt=str(payload.get("visualPrompt") or ""),
        created_at=float(payload.get("createdAt") or time.time()),
    )


def delete_pending_image(reference: str) -> None:
    ref = (reference or "").strip()
    if not ref:
        return
    if _redis_configured():
        try:
            _get_redis().delete(f"{PENDING_IMAGE_KEY_PREFIX}{ref}")
        except Exception as exc:
            logger.error("BUILDER1_PENDING_IMAGE_DELETE_ERR reference=%s err=%s", ref, exc)
    else:
        with _memory_lock:
            _memory_pending.pop(ref, None)
    logger.info("BUILDER1_PENDING_IMAGE_DELETED reference=%s", ref)


def clear_memory_pending_for_tests() -> None:
    with _memory_lock:
        _memory_pending.clear()
