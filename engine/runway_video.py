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

from engine.video_planning import build_runway_prompt_from_plan, fetch_video_plan_o3

logger = logging.getLogger(__name__)

# Official Runway API (see https://docs.dev.runwayml.com/guides/using-the-api)
RUNWAY_API_VERSION_HEADER = "2024-11-06"
DEFAULT_RUNWAY_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_VIDEO_MODEL = "gen4_turbo"

# Polling: docs recommend ≥5s between polls for a given task
_POLL_INTERVAL_SECONDS = 5.0
# Overall wall-clock limit for create + poll (video generation can be slow)
_MAX_WAIT_SECONDS = 600
_HTTP_TIMEOUT_SECONDS = 60

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


def _get_task(session: requests.Session, base_url: str, task_id: str) -> Dict[str, Any]:
    url = f"{base_url}/v1/tasks/{task_id}"
    resp = session.get(url, headers=_headers(), timeout=_HTTP_TIMEOUT_SECONDS)
    if resp.status_code >= 400:
        logger.error(
            "RUNWAY_MVP task_poll_http_failed task_id=%s status=%s",
            task_id,
            resp.status_code,
        )
        raise RunwayVideoMVPError("poll_failed")
    try:
        return resp.json()
    except ValueError:
        logger.error("RUNWAY_MVP task_poll_invalid_json task_id=%s", task_id)
        raise RunwayVideoMVPError("poll_failed")


def _extract_video_url(task: Dict[str, Any]) -> Optional[str]:
    status = (task.get("status") or "").strip()
    if status == "SUCCEEDED":
        out: List[Any] = task.get("output") or []
        if out and isinstance(out[0], str):
            return out[0].strip() or None
        logger.error("RUNWAY_MVP succeeded_but_no_output_url")
        return None
    if status == "FAILED":
        code = task.get("failureCode")
        logger.error("RUNWAY_MVP task_failed task_id=%s failure_code=%s", task.get("id"), code)
        return None
    if status == "CANCELLED":
        logger.error("RUNWAY_MVP task_cancelled task_id=%s", task.get("id"))
        return None
    return None


def generate_one_video_mvp(
    product_name: str,
    product_description: str,
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
    if plan:
        prompt = build_runway_prompt_from_plan(plan)
        marketing = (plan.get("headlineText") or "").strip()
    else:
        prompt = build_simple_prompt(product_name, product_description)
        marketing = ""
        logger.info(
            "RUNWAY_MVP ACE_video_planning_fallback simple_prompt=true "
            "(planning failed; see VIDEO_PLAN_FAIL_* log lines above for this request)"
        )

    session = requests.Session()
    task_id = _create_text_to_video_task(session, base, model, prompt)

    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    logger.info("RUNWAY_MVP polling_started task_id=%s max_wait_s=%s", task_id, _MAX_WAIT_SECONDS)

    while time.monotonic() < deadline:
        task = _get_task(session, base, task_id)
        status = (task.get("status") or "").strip()
        if status in ("PENDING", "THROTTLED", "RUNNING"):
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue
        url = _extract_video_url(task)
        if url:
            logger.info("RUNWAY_MVP polling_done task_id=%s status=SUCCEEDED", task_id)
            return url, marketing
        raise RunwayVideoMVPError("generation_failed")

    logger.error("RUNWAY_MVP timeout task_id=%s", task_id)
    raise RunwayVideoMVPError("timeout")


log_config_warning_if_missing_key()
