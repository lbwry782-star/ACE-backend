"""
Builder2 video purchase allowance service — create, generate-next, derived status.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

from engine.builder2_job_ownership import (
    extract_owner_context_from_request,
    ownership_fields_for_job_create,
    verify_owner_context,
)
from engine.builder2_video_allowance_store import (
    Builder2VideoAllowanceStoreError,
    create_video_allowance,
    get_video_allowance,
    reserve_video_two_slot,
)
from engine.video_jobs_redis import video_job_create, video_job_get_raw

logger = logging.getLogger(__name__)


def parse_target_video_count(raw: Any) -> Tuple[int, Optional[str]]:
    """Return (count, error_code). Default 1 when absent."""
    if raw is None:
        return 1, None
    if isinstance(raw, bool):
        return 1, "invalid_target_video_count"
    if isinstance(raw, (list, dict)):
        return 1, "invalid_target_video_count"
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return 1, None
        if not token.isdigit():
            return 1, "invalid_target_video_count"
        value = int(token)
    elif isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            return 1, "invalid_target_video_count"
        value = int(raw)
    else:
        return 1, "invalid_target_video_count"
    if value not in {1, 2}:
        return 1, "invalid_target_video_count"
    return value, None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def job_has_final_delivery(job_hash: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(job_hash, Mapping):
        return False
    status = _clean(job_hash.get("status"))
    video_url = _clean(job_hash.get("video_url") or job_hash.get("videoUrl"))
    return status == "done" and bool(video_url)


def job_is_in_progress_or_incomplete(job_hash: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(job_hash, Mapping):
        return True
    status = _clean(job_hash.get("status"))
    if status in {"queued", "running", "interrupted"}:
        return True
    if status == "done":
        return not job_has_final_delivery(job_hash)
    return status in {"error", "cancelled"} or status != "done"


def allowance_job_fields(*, video_allowance_id: str, video_index: int) -> Dict[str, str]:
    return {
        "videoAllowanceId": _clean(video_allowance_id),
        "videoIndex": str(int(video_index)),
    }


def build_allowance_public_fields(
    video_allowance_id: str,
    *,
    request: Any = None,
) -> Optional[Dict[str, Any]]:
    allowance = get_video_allowance(video_allowance_id)
    if not allowance:
        return None
    if request is not None:
        owner_ref = extract_owner_context_from_request(request).get("ownerContextRef") or ""
        if owner_ref and allowance.get("ownerContextRef") != owner_ref:
            return None
    return _derive_allowance_payload(allowance)


def _video_entries(allowance: Mapping[str, Any]) -> list[Dict[str, Any]]:
    entries = allowance.get("videos") or []
    if not isinstance(entries, list):
        return []
    normalized: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("videoIndex") or 0)
        except (TypeError, ValueError):
            continue
        jid = _clean(entry.get("jobId"))
        if idx in {1, 2} and jid:
            normalized.append({"videoIndex": idx, "jobId": jid, "createdAt": _clean(entry.get("createdAt"))})
    normalized.sort(key=lambda item: item["videoIndex"])
    return normalized


def _derive_allowance_payload(allowance: Mapping[str, Any]) -> Dict[str, Any]:
    target = int(allowance.get("targetVideoCount") or 1)
    videos_out: list[Dict[str, Any]] = []
    generated = 0
    for entry in _video_entries(allowance):
        raw = video_job_get_raw(entry["jobId"]) or {}
        status = _clean(raw.get("status")) or "unknown"
        final_available = job_has_final_delivery(raw)
        if final_available:
            generated += 1
        video_payload: Dict[str, Any] = {
            "videoIndex": entry["videoIndex"],
            "jobId": entry["jobId"],
            "status": status,
            "finalVideoAvailable": final_available,
        }
        if final_available:
            video_payload["videoUrl"] = _clean(raw.get("video_url"))
            video_payload["marketingText"] = _clean(raw.get("marketing_text"))
        videos_out.append(video_payload)

    remaining = max(0, target - generated)
    consumed = generated >= target
    has_video_two_slot = any(v["videoIndex"] == 2 for v in _video_entries(allowance))
    video_one = next((v for v in videos_out if v["videoIndex"] == 1), None)
    video_one_done = bool(video_one and video_one.get("finalVideoAvailable"))
    can_generate_next = (
        target == 2
        and not consumed
        and video_one_done
        and not has_video_two_slot
        and generated == 1
    )

    return {
        "videoAllowanceId": _clean(allowance.get("videoAllowanceId")),
        "targetVideoCount": target,
        "generatedVideoCount": generated,
        "remainingVideoCount": remaining,
        "canGenerateNext": can_generate_next,
        "consumed": consumed,
        "videos": videos_out,
        "productName": _clean(allowance.get("productName")),
        "productDescription": _clean(allowance.get("productDescription")),
    }


def enrich_status_with_allowance(
    job_id: str,
    job_hash: Optional[Mapping[str, Any]],
    *,
    request: Any = None,
) -> Dict[str, Any]:
    if not isinstance(job_hash, Mapping):
        return {}
    allowance_id = _clean(job_hash.get("videoAllowanceId"))
    if not allowance_id:
        return {}
    allowance = get_video_allowance(allowance_id)
    if not allowance:
        return {"videoAllowanceId": allowance_id}
    if request is not None:
        owner_ref = extract_owner_context_from_request(request).get("ownerContextRef") or ""
        if owner_ref and allowance.get("ownerContextRef") != owner_ref:
            return {"videoAllowanceId": allowance_id, "allowanceAccessDenied": True}
    payload = _derive_allowance_payload(allowance)
    try:
        payload["videoIndex"] = int(job_hash.get("videoIndex") or 0) or None
    except (TypeError, ValueError):
        payload["videoIndex"] = None
    payload["jobId"] = _clean(job_id)
    return payload


def create_initial_allowance_and_job_fields(
    *,
    request: Any,
    payload: Mapping[str, Any],
    target_video_count: int,
    job_id: str,
    product_name: str,
    product_description: str,
) -> Tuple[Dict[str, str], str]:
    owner_fields = ownership_fields_for_job_create(request, payload)
    owner_ref = _clean(owner_fields.get("ownerContextRef"))
    allowance_id = str(uuid.uuid4())
    create_video_allowance(
        video_allowance_id=allowance_id,
        owner_context_ref=owner_ref,
        target_video_count=target_video_count,
        product_name=product_name,
        product_description=product_description,
        first_job_id=job_id,
    )
    owner_fields.update(allowance_job_fields(video_allowance_id=allowance_id, video_index=1))
    return owner_fields, allowance_id


def request_generate_video_next(
    *,
    video_allowance_id: str,
    request: Any,
    public_base_url: str,
) -> Dict[str, Any]:
    aid = _clean(video_allowance_id)
    if not aid:
        return {"ok": False, "error": "missing_param", "message": "videoAllowanceId is required"}

    allowance = get_video_allowance(aid)
    if not allowance:
        return {"ok": False, "error": "allowance_not_found", "videoAllowanceId": aid}

    owner_ref = extract_owner_context_from_request(request).get("ownerContextRef") or ""
    if not owner_ref or allowance.get("ownerContextRef") != owner_ref:
        return {"ok": False, "error": "ownership_mismatch", "videoAllowanceId": aid}

    target = int(allowance.get("targetVideoCount") or 1)
    if target != 2:
        return {"ok": False, "error": "target_video_count_not_two", "videoAllowanceId": aid}

    derived = _derive_allowance_payload(allowance)
    if derived.get("consumed"):
        return {"ok": False, "error": "allowance_consumed", "videoAllowanceId": aid, **derived}

    entries = _video_entries(allowance)
    video_one = next((e for e in entries if e["videoIndex"] == 1), None)
    if not video_one:
        return {"ok": False, "error": "video_one_missing", "videoAllowanceId": aid}

    video_one_hash = video_job_get_raw(video_one["jobId"]) or {}
    if job_is_in_progress_or_incomplete(video_one_hash) or not job_has_final_delivery(video_one_hash):
        return {
            "ok": False,
            "error": "video_one_not_complete",
            "videoAllowanceId": aid,
            **derived,
        }

    existing_two = next((e for e in entries if e["videoIndex"] == 2), None)
    if existing_two:
        return {
            "ok": True,
            "videoAllowanceId": aid,
            "jobId": existing_two["jobId"],
            "videoIndex": 2,
            "status": "queued",
            "idempotent": True,
            **derived,
        }

    proposed_job_id = str(uuid.uuid4())
    reserve = reserve_video_two_slot(
        aid,
        owner_context_ref=owner_ref,
        job_id=proposed_job_id,
    )
    if not reserve.ok:
        return {"ok": False, "error": reserve.code or "reserve_failed", "videoAllowanceId": aid}

    job_id = reserve.job_id
    if reserve.idempotent and job_id != proposed_job_id:
        return {
            "ok": True,
            "videoAllowanceId": aid,
            "jobId": job_id,
            "videoIndex": 2,
            "status": "queued",
            "idempotent": True,
            **_derive_allowance_payload(get_video_allowance(aid) or allowance),
        }

    product_name = _clean(allowance.get("productName"))
    product_description = _clean(allowance.get("productDescription"))
    extra_fields = ownership_fields_for_job_create(
        request,
        {"productDescription": product_description, "productName": product_name},
    )
    extra_fields.update(allowance_job_fields(video_allowance_id=aid, video_index=2))

    try:
        video_job_create(job_id, product_name, product_description, public_base_url, extra_fields=extra_fields)
    except Exception as exc:
        logger.error(
            "BUILDER2_VIDEO_TWO_ENQUEUE_FAILED videoAllowanceId=%s jobId=%s err=%s",
            aid,
            job_id,
            exc,
            exc_info=True,
        )
        return {"ok": False, "error": "video_generation_failed", "videoAllowanceId": aid, "jobId": job_id}

    logger.info(
        "BUILDER2_VIDEO_TWO_JOB_CREATED videoAllowanceId=%s jobId=%s videoIndex=2",
        aid,
        job_id,
    )
    refreshed = get_video_allowance(aid) or allowance
    return {
        "ok": True,
        "videoAllowanceId": aid,
        "jobId": job_id,
        "videoIndex": 2,
        "status": "queued",
        "idempotent": False,
        **_derive_allowance_payload(refreshed),
    }


def resolve_zip_payload_from_job(
    job_id: str,
    *,
    request: Any,
    supplied_video_url: str = "",
    supplied_marketing_text: str = "",
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Resolve owned job video URL + marketing text for ZIP download.
    Returns (video_url, marketing_text, error_code).
    """
    jid = _clean(job_id)
    if not jid:
        return None, "", "missing_job_id"
    raw = video_job_get_raw(jid)
    if not raw:
        return None, "", "not_found"
    ok, reason = verify_owner_context(raw, request)
    if not ok:
        return None, "", reason or "ownership_required"
    if _clean(raw.get("builder")) != "builder2" and not _clean(raw.get("builder2ResumeContractVersion")):
        return None, "", "not_builder2_job"
    video_url = _clean(raw.get("video_url"))
    marketing_text = _clean(raw.get("marketing_text"))
    if not video_url:
        return None, "", "video_not_ready"
    if supplied_video_url and supplied_video_url != video_url:
        return None, "", "video_url_mismatch"
    if supplied_marketing_text and supplied_marketing_text != marketing_text:
        return None, "", "marketing_text_mismatch"
    return video_url, marketing_text, None
