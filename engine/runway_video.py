"""
ACE Runway video — first MVP path (isolated from the image engine).

Currently: one video output only, simple prompting, Runway POST /v1/image_to_video; gen4_turbo needs a minimal promptImage (neutral data URI), gen4.5 can omit for text-only.
Future ACE video engine may produce two outputs and richer concept prompting; keep this module minimal until then.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from engine.video_planning import (
    build_runway_interaction_prompt_from_plan,
    build_runway_prompt_from_plan,
    fetch_video_plan_o3,
)
from engine.video_headline_postprocess import postprocess_video_headline
from engine.video_start_image import generate_video_start_image_data_uri

logger = logging.getLogger(__name__)

# Official Runway API (see https://docs.dev.runwayml.com/guides/using-the-api)
RUNWAY_API_VERSION_HEADER = "2024-11-06"
DEFAULT_RUNWAY_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_VIDEO_MODEL = "gen4_turbo"

# Polling: docs recommend ≥5s between polls for a given task
_POLL_INTERVAL_SECONDS = 5.0
# Overall wall-clock limit for create + poll (video generation can be slow)
_MAX_WAIT_SECONDS = 600
# Task create + non-poll calls
_HTTP_TIMEOUT_SECONDS = 60
# Per poll GET — must never block indefinitely (network stalls)
_POLL_HTTP_TIMEOUT_SECONDS = float((os.environ.get("RUNWAY_POLL_HTTP_TIMEOUT_SECONDS") or "25").strip() or "25")

# Runway task status (normalized upper); extend as API evolves
_RUNNING_STATUSES = frozenset(
    {"PENDING", "THROTTLED", "RUNNING", "QUEUED", "IN_PROGRESS", "STARTING", "PREPARING"}
)
_SUCCESS_STATUSES = frozenset({"SUCCEEDED", "COMPLETED", "SUCCESS"})
_FAILED_STATUSES = frozenset(
    {"FAILED", "CANCELLED", "CANCELED", "EXPIRED", "REJECTED", "ERROR", "ERRORED"}
)

# gen4_turbo image_to_video requires promptImage (API returns 400 if omitted). Opaque light neutral frame
# (soft gray-beige, no transparency) so Runway does not substitute a default color (e.g. red) for alpha.
# 8x8 PNG, no subject/text; motion/scene still come from promptText.
_NEUTRAL_PROMPT_IMAGE_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFUlEQVR4nGN8/folAzbAhFV00EoAACbiAs8zJy7JAAAAAElFTkSuQmCC"
)


class RunwayVideoMVPError(Exception):
    """Internal failure for MVP path; callers map to generic client error."""


def _env_api_key() -> str:
    return (os.environ.get("RUNWAY_API_KEY", "") or "").strip()


def _env_model() -> str:
    raw = (os.environ.get("RUNWAY_VIDEO_MODEL", "") or "").strip()
    return raw or DEFAULT_VIDEO_MODEL


def _env_base_url() -> str:
    raw = (os.environ.get("RUNWAY_API_BASE_URL", "") or "").strip()
    return raw.rstrip("/") if raw else DEFAULT_RUNWAY_BASE_URL


def log_config_warning_if_missing_key() -> None:
    """Call at import or startup; does not raise."""
    if not _env_api_key():
        logger.warning(
            "RUNWAY_API_KEY is missing or empty; POST /api/generate-video will return ok=false until set"
        )


def is_configured() -> bool:
    return bool(_env_api_key())


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_env_api_key()}",
        "Content-Type": "application/json",
        "X-Runway-Version": RUNWAY_API_VERSION_HEADER,
    }


def build_simple_prompt(product_name: str, product_description: str) -> str:
    """Minimal robust prompt from product fields (no ACE A/B or advanced ad logic)."""
    name = (product_name or "").strip() or "the product"
    desc = (product_description or "").strip()
    text = (
        "Clean commercial video, modern advertising style, cinematic lighting, "
        "smooth camera movement, product-focused scene. "
        f"Feature {name}. "
        f"{desc}"
    ).strip()
    # API: up to 1000 UTF-16 code units; slice by Python len is safe enough for MVP
    if len(text) > 1000:
        text = text[:1000]
    return text


def _create_text_to_video_task(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt_text: str,
    prompt_image_data_uri: Optional[str] = None,
) -> str:
    url = f"{base_url}/v1/image_to_video"
    body: Dict[str, Any] = {
        "model": model,
        "promptText": prompt_text,
        "ratio": "1280:720",
        "duration": 5,
    }
    # gen4.5: text-to-video — omit promptImage per Runway docs. gen4_turbo (default): promptImage required.
    if model == "gen4.5":
        logger.info("RUNWAY_MVP task_create model=%s mode=text_only promptImage=omitted", model)
    elif prompt_image_data_uri:
        body["promptImage"] = prompt_image_data_uri
        logger.info("RUNWAY_MVP task_create model=%s promptImage=generated", model)
    else:
        body["promptImage"] = _NEUTRAL_PROMPT_IMAGE_DATA_URI
        logger.info("RUNWAY_MVP task_create model=%s promptImage=neutral", model)
    resp = session.post(url, json=body, headers=_headers(), timeout=_HTTP_TIMEOUT_SECONDS)
    if resp.status_code >= 400:
        logger.error(
            "RUNWAY_MVP task_create_http_failed status=%s body_len=%s",
            resp.status_code,
            len(resp.content or b""),
        )
        raise RunwayVideoMVPError("create_failed")
    try:
        data = resp.json()
    except ValueError:
        logger.error("RUNWAY_MVP task_create_invalid_json")
        raise RunwayVideoMVPError("create_failed")
    task_id = (data.get("id") or "").strip()
    if not task_id:
        logger.error("RUNWAY_MVP task_create_missing_id")
        raise RunwayVideoMVPError("create_failed")
    logger.info("RUNWAY_MVP task_created task_id=%s", task_id)
    return task_id


def _poll_get_task_once(
    session: requests.Session,
    base_url: str,
    task_id: str,
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """
    Single poll GET with hard timeout. Returns None on timeout/transport/HTTP/JSON parse errors
    so the outer loop can retry until monotonic deadline (never hang forever on one socket).
    """
    url = f"{base_url}/v1/tasks/{task_id}"
    try:
        resp = session.get(url, headers=_headers(), timeout=timeout)
    except requests.exceptions.Timeout:
        logger.warning(
            "RUNWAY_MVP poll_timeout task_id=%s timeout_s=%s",
            task_id,
            timeout,
        )
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(
            "RUNWAY_MVP poll_http_error task_id=%s err_type=%s err=%s",
            task_id,
            type(e).__name__,
            e,
        )
        return None
    if resp.status_code >= 400:
        logger.error(
            "RUNWAY_MVP poll_http_error task_id=%s http_status=%s body_len=%s",
            task_id,
            resp.status_code,
            len(resp.content or b""),
        )
        return None
    try:
        data = resp.json()
    except ValueError:
        logger.error("RUNWAY_MVP poll_invalid_json task_id=%s", task_id)
        return None
    if not isinstance(data, dict):
        logger.error("RUNWAY_MVP poll_invalid_payload task_id=%s type=%s", task_id, type(data).__name__)
        return None
    return data


def _normalize_task_status(task: Dict[str, Any]) -> str:
    raw = task.get("status")
    return (str(raw).strip() if raw is not None else "").upper()


def _extract_video_url(task: Dict[str, Any]) -> Optional[str]:
    status = _normalize_task_status(task)
    if status in _SUCCESS_STATUSES:
        out: List[Any] = task.get("output") or []
        if out and isinstance(out[0], str):
            return out[0].strip() or None
        logger.error("RUNWAY_MVP succeeded_but_no_output_url task_id=%s", task.get("id"))
        return None
    if status in _FAILED_STATUSES:
        code = task.get("failureCode")
        logger.error(
            "RUNWAY_MVP task_failed task_id=%s failure_code=%s status=%s",
            task.get("id"),
            code,
            status,
        )
        return None
    return None


def _sleep_poll_interval(deadline: float) -> None:
    """Sleep up to _POLL_INTERVAL_SECONDS without exceeding deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    sleep_s = min(_POLL_INTERVAL_SECONDS, remaining)
    logger.info("RUNWAY_MVP poll_sleep seconds=%s", round(sleep_s, 2))
    time.sleep(sleep_s)


def generate_one_video_mvp(
    product_name: str,
    product_description: str,
    public_base_url: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Create one Runway video task, poll until done or timeout.
    Returns (video_url, headline_for_ui): second value is the planned headline text if any (else ""),
    for the existing marketingText API field — not 50-word body copy.
    Raises RunwayVideoMVPError on any failure.
    """
    if not _env_api_key():
        logger.error("RUNWAY_MVP aborted missing_api_key")
        raise RunwayVideoMVPError("not_configured")

    base = _env_base_url()
    model = _env_model()
    plan = fetch_video_plan_o3(product_name, product_description)
    prompt_image_data_uri: Optional[str] = None
    headline_decision: Optional[str] = None
    if plan:
        marketing = (plan.get("headlineText") or "").strip()
        headline_decision = (plan.get("headlineDecision") or "").strip() or None
        # gen4.5 omits promptImage; skip start-image generation (not used by Runway).
        if model != "gen4.5":
            prompt_image_data_uri = generate_video_start_image_data_uri(plan)
            if prompt_image_data_uri:
                prompt = build_runway_interaction_prompt_from_plan(plan)
            else:
                prompt = build_runway_prompt_from_plan(plan)
        else:
            prompt = build_runway_prompt_from_plan(plan)
    else:
        prompt = build_simple_prompt(product_name, product_description)
        marketing = ""
        headline_decision = None
        logger.info(
            "RUNWAY_MVP ACE_video_planning_fallback simple_prompt=true "
            "(planning failed; see VIDEO_PLAN_FAIL_* log lines above for this request)"
        )

    session = requests.Session()
    task_id = _create_text_to_video_task(
        session, base, model, prompt, prompt_image_data_uri=prompt_image_data_uri
    )

    poll_start = time.monotonic()
    deadline = poll_start + _MAX_WAIT_SECONDS
    logger.info(
        "RUNWAY_MVP polling_started task_id=%s max_wait_s=%s poll_http_timeout_s=%s",
        task_id,
        _MAX_WAIT_SECONDS,
        _POLL_HTTP_TIMEOUT_SECONDS,
    )

    poll_attempt = 0
    while time.monotonic() < deadline:
        poll_attempt += 1
        elapsed_s = round(time.monotonic() - poll_start, 1)
        logger.info(
            "RUNWAY_MVP poll_attempt=%s elapsed_s=%s task_id=%s deadline_remaining_s=%s",
            poll_attempt,
            elapsed_s,
            task_id,
            round(max(0.0, deadline - time.monotonic()), 1),
        )

        task = _poll_get_task_once(session, base, task_id, _POLL_HTTP_TIMEOUT_SECONDS)
        if task is None:
            if time.monotonic() >= deadline:
                break
            _sleep_poll_interval(deadline)
            continue

        status = _normalize_task_status(task)
        logger.info("RUNWAY_MVP poll_status status=%s task_id=%s", status or "(empty)", task_id)

        if status in _RUNNING_STATUSES:
            _sleep_poll_interval(deadline)
            continue

        if status in _SUCCESS_STATUSES:
            url = _extract_video_url(task)
            if url:
                logger.info("RUNWAY_MVP polling_done task_id=%s status=%s", task_id, status)
                final_url = postprocess_video_headline(
                    url,
                    marketing,
                    public_base_url or "",
                    headline_decision=headline_decision,
                )
                return final_url, marketing
            raise RunwayVideoMVPError("generation_failed")

        if status in _FAILED_STATUSES:
            _extract_video_url(task)
            raise RunwayVideoMVPError("generation_failed")

        logger.warning(
            "RUNWAY_MVP poll_unknown_status status=%s task_id=%s (will_retry_until_deadline)",
            status,
            task_id,
        )
        _sleep_poll_interval(deadline)

    logger.error("RUNWAY_MVP timeout task_id=%s max_wait_s=%s", task_id, _MAX_WAIT_SECONDS)
    raise RunwayVideoMVPError("timeout")


log_config_warning_if_missing_key()
