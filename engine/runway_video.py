"""
ACE Runway video — first MVP path (isolated from the image engine).

Currently: one video output only, pure text-to-video using Runway gen4.5.
Future ACE video engine may produce two outputs and richer concept prompting; keep this module minimal until then.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from engine.ad_promise_memory import record_ad_promise_generation_success
from engine.video_planning import (
    VideoPlanningTimeoutError,
    build_runway_prompt_from_plan,
    fetch_video_plan_o3,
    log_video_job_plan_integrity,
    sanitize_runway_prompt_for_video_text_policy,
    video_plan_struct_ok_for_runway,
)
from engine.video_bidi import (
    finalize_hebrew_mixed_bidi_for_display,
    format_bidi_segments_for_log,
    prepare_ffmpeg_overlay_headline,
)
from engine.video_headline_postprocess import postprocess_video_headline
from engine.video_job_context import video_job_set_phase
from engine.video_jobs_redis import video_job_set_resolved_product_name
from engine.video_language import (
    detect_text_language,
    evaluate_headline_overlay_language,
    is_english_only_product_name_script,
    log_video_language_decision,
    normalize_video_content_language,
    text_predominantly_matches_language,
)
from engine.video_product_name import (
    VideoProductNameError,
    apply_canonical_product_name_to_video_plan,
    product_name_reused_in_copy,
    product_name_reused_in_headline,
    resolve_video_product_name,
)

logger = logging.getLogger(__name__)


def _maybe_log_ad_promise_skip_after_failed_generation(
    plan: Optional[Dict[str, Any]], promise_saved: bool
) -> None:
    """Product-level Redis memory only; log when a plan existed but post-success save never ran."""
    if plan is not None and not promise_saved:
        logger.info("AD_PROMISE_MEMORY_SKIP_SAVE reason=generation_failed")


# Official Runway API (see https://docs.dev.runwayml.com/guides/using-the-api)
RUNWAY_API_VERSION_HEADER = "2024-11-06"
DEFAULT_RUNWAY_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_VIDEO_MODEL = "gen4.5"

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

class RunwayVideoMVPError(Exception):
    """Internal failure for MVP path; callers map to generic client error."""


def _env_api_key() -> str:
    return (os.environ.get("RUNWAY_API_KEY", "") or "").strip()


def _env_model() -> str:
    # Locked pipeline: gen4.5 text-only
    model = DEFAULT_VIDEO_MODEL
    logger.info("RUNWAY_VIDEO_MODEL_SELECTED model=%s source=forced", model)
    return model


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


def _enforce_headline_overlay_language(
    required_lang: str, headline: str, canonical_name: str = ""
) -> None:
    """
    Headline must match required content language, with an exception for Hebrew jobs where the
    only Latin is the resolved English product name and the rest is Hebrew.
    """
    req = normalize_video_content_language(required_lang)
    h = (headline or "").strip()
    if not h:
        return
    logger.info("VIDEO_HEADLINE_LANGUAGE_REQUIRED=%s", req)
    cn = (canonical_name or "").strip()
    product_name_only = (
        req == "he"
        and bool(cn)
        and bool(h)
        and h.casefold() == cn.casefold()
        and is_english_only_product_name_script(h)
        and is_english_only_product_name_script(cn)
    )
    logger.info("VIDEO_HEADLINE_PRODUCT_NAME_ONLY=%s", str(product_name_only).lower())
    ok, actual, allowed_latin_product = evaluate_headline_overlay_language(
        h, req, canonical_name
    )
    logger.info("VIDEO_HEADLINE_LANGUAGE_ACTUAL=%s", actual)
    logger.info(
        "VIDEO_HEADLINE_ALLOWED_LATIN_PRODUCT_NAME=%s",
        str(allowed_latin_product).lower(),
    )
    latin_allowed_he = bool(allowed_latin_product) or (
        req == "he" and actual == "hebrew_job_latin_product_name_headline_only"
    )
    logger.info(
        "VIDEO_HEADLINE_LATIN_PRODUCT_NAME_ALLOWED_IN_HEBREW=%s",
        str(latin_allowed_he).lower(),
    )
    if ok and actual == "hebrew_job_latin_product_name_headline_only":
        logger.info("VIDEO_HEADLINE_LANGUAGE_OVERRIDE_REASON=product_name_only_allowed")
    if not ok:
        logger.warning(
            "VIDEO_HEADLINE_LANGUAGE_ADVISORY_MISMATCH required=%s (planner headline retained)",
            req,
        )


def _enforce_marketing_copy_language(required_lang: str, marketing_text: str) -> None:
    """Marketing copy must be predominantly in the classified language; loanwords/abbreviations allowed."""
    req = normalize_video_content_language(required_lang)
    if not (marketing_text or "").strip():
        logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=marketing_copy_empty")
        raise RunwayVideoMVPError("marketing_copy_empty")
    if not text_predominantly_matches_language(marketing_text, req):
        logger.warning(
            "VIDEO_MARKETING_LANGUAGE_ADVISORY_MISMATCH required=%s (copy retained)",
            req,
        )


def _create_text_to_video_task(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt_text: str,
    prompt_image_data_uri: Optional[str] = None,
) -> str:
    if (prompt_image_data_uri or "").strip():
        raise RunwayVideoMVPError("promptImage is not supported in gen4.5 pipeline")
    if model != "gen4.5":
        logger.error("RUNWAY_MODEL_UNSUPPORTED model=%s expected=gen4.5", model)
        raise RunwayVideoMVPError("unsupported_runway_model")
    url = f"{base_url}/v1/text_to_video"
    body: Dict[str, Any] = {
        "model": model,
        "promptText": prompt_text,
        "ratio": "1280:720",
        "duration": 5,
    }
    logger.info("RUNWAY_MODE=gen4.5_text_only")
    logger.info("PROMPT_IMAGE_DISABLED=true")
    logger.info(
        "RUNWAY_MVP task_create model=%s mode=text_to_video",
        model,
    )
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


def _generate_one_video_mvp_body(
    product_name: str,
    product_description: str,
    public_base_url: Optional[str] = None,
    job_id: str = "",
) -> Tuple[str, str, str]:
    plan: Optional[Dict[str, Any]] = None
    promise_saved = False
    t_job0 = time.monotonic()
    plan_ms = runway_ms = post_ms = 0.0

    if not _env_api_key():
        logger.error("RUNWAY_MVP aborted missing_api_key")
        raise RunwayVideoMVPError("not_configured")

    base = _env_base_url()
    model = _env_model()
    marketing_lang = detect_text_language(product_description)
    video_lang, _, _ = log_video_language_decision(product_description)
    try:
        pn_source, canonical_name = resolve_video_product_name(
            product_name,
            product_description,
            video_lang,
            marketing_language=marketing_lang,
        )
    except VideoProductNameError as e:
        reason = (e.args[0] if getattr(e, "args", None) else None) or "error"
        logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=product_name_%s", reason)
        raise RunwayVideoMVPError("product_name_generation_failed")
    logger.info("VIDEO_PRODUCT_NAME_SOURCE=%s", pn_source)
    logger.info(
        "VIDEO_PRODUCT_NAME_RESOLVED_CREATED value=%s",
        json.dumps(canonical_name, ensure_ascii=False),
    )
    logger.info(
        "VIDEO_PRODUCT_NAME_RESOLVED=%s", json.dumps(canonical_name, ensure_ascii=False)
    )
    if job_id:
        try:
            video_job_set_resolved_product_name(job_id, canonical_name, pn_source)
        except Exception as e:
            logger.warning(
                "VIDEO_JOB_RESOLVED_NAME_REDIS_WRITE_FAIL jobId=%s err=%s",
                job_id,
                e,
            )

    logger.info("VIDEO_JOB_STEP step=plan_video start")
    video_job_set_phase("planning")
    plan = None
    plan_fail_reason = ""
    t_plan0 = time.monotonic()
    try:
        plan, plan_fail_reason = fetch_video_plan_o3(
            canonical_name,
            product_description,
            content_language=video_lang,
            session_id=job_id or "",
        )
    except VideoPlanningTimeoutError:
        plan_ms = (time.monotonic() - t_plan0) * 1000.0
        logger.info("VIDEO_TIMING_PLAN_MS=%.1f", plan_ms)
        logger.info("VIDEO_TIMING_RUNWAY_MS=0.0")
        logger.info("VIDEO_TIMING_POSTPROCESS_MS=0.0")
        logger.info(
            "VIDEO_TIMING_TOTAL_MS=%.1f", (time.monotonic() - t_job0) * 1000.0
        )
        logger.info("VIDEO_TIMING_DOMINANT_PHASE=plan")
        logger.info("VIDEO_PLAN_ABORTED reason=planning_timeout")
        logger.info("VIDEO_PLAN_REQUIRED_FIELDS_OK=false")
        logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=planning_timeout")
        _maybe_log_ad_promise_skip_after_failed_generation(None, promise_saved)
        raise RunwayVideoMVPError("planning_timeout")
    plan_ms = (time.monotonic() - t_plan0) * 1000.0
    logger.info("VIDEO_TIMING_PLAN_MS=%.1f", plan_ms)
    logger.info("VIDEO_JOB_STEP step=plan_video done has_plan=%s", bool(plan))

    if not plan:
        fail = (plan_fail_reason or "").strip() or "planning_failed"
        logger.info("VIDEO_TIMING_RUNWAY_MS=0.0")
        logger.info("VIDEO_TIMING_POSTPROCESS_MS=0.0")
        logger.info(
            "VIDEO_TIMING_TOTAL_MS=%.1f", (time.monotonic() - t_job0) * 1000.0
        )
        logger.info("VIDEO_TIMING_DOMINANT_PHASE=plan")
        logger.info("VIDEO_PLAN_ABORTED reason=%s", fail)
        logger.info("VIDEO_PLAN_REQUIRED_FIELDS_OK=false")
        logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=%s", fail)
        logger.info("VIDEO_PLAN_INTEGRITY skipped reason=%s", fail)
        _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
        raise RunwayVideoMVPError(fail)

    apply_canonical_product_name_to_video_plan(plan, canonical_name)
    plan["marketingLanguage"] = marketing_lang

    struct_ok, struct_reason = video_plan_struct_ok_for_runway(plan)
    if not struct_ok:
        logger.info("VIDEO_PLAN_REQUIRED_FIELDS_OK=false reason=%s", struct_reason)
        logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=%s", struct_reason)
        logger.info("VIDEO_TIMING_RUNWAY_MS=0.0")
        logger.info("VIDEO_TIMING_POSTPROCESS_MS=0.0")
        total_ms = (time.monotonic() - t_job0) * 1000.0
        logger.info("VIDEO_TIMING_TOTAL_MS=%.1f", total_ms)
        logger.info("VIDEO_TIMING_DOMINANT_PHASE=plan")
        _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
        raise RunwayVideoMVPError("planning_failed_incomplete_plan")

    logger.info("VIDEO_PLAN_REQUIRED_FIELDS_OK=true")
    log_video_job_plan_integrity(plan)

    video_job_set_phase("runway")
    prompt = build_runway_prompt_from_plan(plan)

    prompt, text_policy_sanitized = sanitize_runway_prompt_for_video_text_policy(prompt)
    logger.info("VIDEO_TEXT_POLICY_SANITIZED=%s", text_policy_sanitized)

    session = requests.Session()
    t_runway0 = time.monotonic()
    logger.info("VIDEO_JOB_STEP step=runway_create_task start")
    task_id = _create_text_to_video_task(
        session, base, model, prompt, prompt_image_data_uri=None
    )
    logger.info("VIDEO_JOB_STEP step=runway_create_task done task_id=%s", task_id)

    poll_start = time.monotonic()
    deadline = poll_start + _MAX_WAIT_SECONDS
    logger.info(
        "RUNWAY_MVP polling_started task_id=%s max_wait_s=%s poll_http_timeout_s=%s",
        task_id,
        _MAX_WAIT_SECONDS,
        _POLL_HTTP_TIMEOUT_SECONDS,
    )
    logger.info("VIDEO_JOB_STEP step=runway_poll_loop start")

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
                runway_ms = (time.monotonic() - t_runway0) * 1000.0
                logger.info("VIDEO_TIMING_RUNWAY_MS=%.1f", runway_ms)
                logger.info("RUNWAY_MVP polling_done task_id=%s status=%s", task_id, status)
                logger.info("VIDEO_JOB_STEP step=runway_poll_loop done outcome=success")
                video_job_set_phase("postprocess")
                t_post0 = time.monotonic()
                logger.info("VIDEO_JOB_STEP step=packaging_result start")
                headline_for_overlay = (plan.get("headlineText") or "").strip()
                headline_decision = (plan.get("headlineDecision") or "").strip()
                try:
                    from engine.side_by_side_v1 import generate_marketing_copy

                    ad_goal = (plan.get("advertisingPromise") or "").strip()
                    logger.info(
                        "VIDEO_COPY_INPUT_PRODUCT_NAME=%s",
                        json.dumps(canonical_name, ensure_ascii=False),
                    )
                    logger.info("MARKETING_TEXT_LANGUAGE_APPLIED lang=%s", marketing_lang)
                    marketing_text_for_api = generate_marketing_copy(
                        canonical_name,
                        (product_description or "").strip(),
                        ad_goal,
                        output_language=marketing_lang,
                        require_verbatim_product_name=True,
                    )
                except Exception as e:
                    logger.error("VIDEO_JOB_MARKETING_COPY_FAIL err=%s", e, exc_info=True)
                    logger.info("VIDEO_PLAN_ABORTED reason=marketing_copy_exception")
                    logger.error("VIDEO_JOB_FAILED_INTEGRITY reason=marketing_copy_failed")
                    _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
                    raise RunwayVideoMVPError("marketing_copy_failed") from e
                reuse_h = product_name_reused_in_headline(
                    canonical_name, headline_for_overlay, headline_decision
                )
                reuse_c = product_name_reused_in_copy(canonical_name, marketing_text_for_api)
                logger.info("VIDEO_HEADLINE_USED_PRODUCT_NAME=%s", str(reuse_h).lower())
                logger.info("VIDEO_COPY_USED_PRODUCT_NAME=%s", str(reuse_c).lower())
                name_mismatch = not reuse_h or not reuse_c
                logger.info("VIDEO_PRODUCT_NAME_MISMATCH=%s", str(name_mismatch).lower())
                if not reuse_h:
                    logger.warning(
                        "VIDEO_PLAN_SERVER_CREATIVE_CHECK_SKIPPED field=headline product_name_overlap=false advisory_only=1"
                    )
                if not reuse_c:
                    logger.warning(
                        "VIDEO_PLAN_SERVER_CREATIVE_CHECK_SKIPPED field=marketing_copy product_name_overlap=false advisory_only=1"
                    )
                try:
                    _enforce_marketing_copy_language(marketing_lang, marketing_text_for_api)
                    _enforce_headline_overlay_language(
                        video_lang, headline_for_overlay, canonical_name
                    )
                    _bidi_prot = (
                        (canonical_name.strip(),)
                        if (canonical_name or "").strip()
                        else ()
                    )
                    marketing_text_for_api, bidi_copy, segs_copy = (
                        finalize_hebrew_mixed_bidi_for_display(
                            marketing_text_for_api,
                            content_language=marketing_lang,
                            protected_phrases=_bidi_prot,
                        )
                    )
                    if marketing_lang == "he":
                        logger.info(
                            "MARKETING_TEXT_BIDI_NORMALIZED applied=%s lang=he",
                            "true" if bidi_copy else "false",
                        )
                    else:
                        logger.info(
                            "MARKETING_TEXT_BIDI_NORMALIZED applied=false lang=en",
                        )
                    overlay_prep = prepare_ffmpeg_overlay_headline(
                        headline_for_overlay,
                        content_language=video_lang,
                        canonical_name=canonical_name,
                    )
                    headline_for_overlay = overlay_prep.text_plain
                    overlay_bidi_strategy = overlay_prep.strategy
                    logger.info(
                        "VIDEO_BIDI_FIX_APPLIED_COPY=%s",
                        str(bidi_copy).lower(),
                    )
                    logger.info(
                        "VIDEO_BIDI_LATIN_SEGMENTS_COPY=%s",
                        format_bidi_segments_for_log(segs_copy),
                    )
                    logger.info(
                        "VIDEO_HEADLINE_BIDI_OVERLAY_STRATEGY=%s",
                        overlay_bidi_strategy,
                    )
                    logger.info(
                        "VIDEO_HEADLINE_OVERLAY_RENDER_MODE=%s",
                        overlay_prep.render_mode,
                    )
                    logger.info("VIDEO_HEADLINE_OVERLAY_USED_ISOLATES=false")
                    logger.info("VIDEO_JOB_STEP step=packaging_result done")
                    final_url = postprocess_video_headline(
                        url,
                        public_base_url or "",
                        headline=headline_for_overlay,
                        job_id=job_id,
                        overlay_language=video_lang,
                        overlay_render_mode=overlay_prep.render_mode,
                        overlay_dual_latin=overlay_prep.dual_latin,
                        overlay_dual_hebrew=overlay_prep.dual_hebrew,
                        overlay_canonical_name=canonical_name,
                    )
                    post_ms = (time.monotonic() - t_post0) * 1000.0
                    logger.info("VIDEO_TIMING_POSTPROCESS_MS=%.1f", post_ms)
                    if final_url.rstrip("/") == url.rstrip("/"):
                        logger.info(
                            "VIDEO_JOB_CHOSEN_URL source=runway_fallback jobId=%s",
                            job_id,
                        )
                    elif "/api/video-headline/" in final_url and "/api/video-headline-artifact" not in final_url:
                        logger.info(
                            "VIDEO_JOB_CHOSEN_URL source=processed_uploaded jobId=%s",
                            job_id,
                        )
                    else:
                        logger.info(
                            "VIDEO_JOB_CHOSEN_URL source=processed jobId=%s",
                            job_id,
                        )
                    logger.info(
                        "VIDEO_PRODUCT_NAME_RESOLVED_PACKAGED value=%s",
                        json.dumps(canonical_name, ensure_ascii=False),
                    )
                    ap_mem = (plan.get("advertisingPromise") or "").strip()
                    if ap_mem:
                        record_ad_promise_generation_success(
                            canonical_name,
                            product_description,
                            ap_mem,
                            job_or_trace_id=job_id or "",
                            source_type="video",
                        )
                    promise_saved = True
                    logger.info("VIDEO_PROMISE_ACCEPTED_NEW=true")
                    total_ms = (time.monotonic() - t_job0) * 1000.0
                    logger.info("VIDEO_TIMING_TOTAL_MS=%.1f", total_ms)
                    phases = (("plan", plan_ms), ("runway", runway_ms), ("postprocess", post_ms))
                    dom = (
                        max(phases, key=lambda x: x[1])[0]
                        if max(p[1] for p in phases) > 0.5
                        else "unknown"
                    )
                    logger.info("VIDEO_TIMING_DOMINANT_PHASE=%s", dom)
                    return final_url, marketing_text_for_api, headline_for_overlay
                except Exception:
                    _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
                    raise
            _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
            raise RunwayVideoMVPError("runway_failed")

        if status in _FAILED_STATUSES:
            _extract_video_url(task)
            _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
            raise RunwayVideoMVPError("runway_failed")

        logger.warning(
            "RUNWAY_MVP poll_unknown_status status=%s task_id=%s (will_retry_until_deadline)",
            status,
            task_id,
        )
        _sleep_poll_interval(deadline)

    logger.error("RUNWAY_MVP timeout task_id=%s max_wait_s=%s", task_id, _MAX_WAIT_SECONDS)
    logger.info("VIDEO_JOB_STEP step=runway_poll_loop done outcome=timeout")
    _maybe_log_ad_promise_skip_after_failed_generation(plan, promise_saved)
    raise RunwayVideoMVPError("runway_failed")


def generate_one_video_mvp(
    product_name: str,
    product_description: str,
    public_base_url: Optional[str] = None,
    job_id: str = "",
) -> Tuple[str, str, str]:
    """
    Create one Runway video task, poll until done or timeout.
    Returns (final_video_url, marketing_text_api, overlay_headline):
    - final_video_url: postprocess_video_headline output when overlay succeeds; else Runway URL on failure/skip.
    - marketing_text_api: 45–55 word copy for Redis/API when a plan exists (unchanged; not overlaid on video).
    - overlay_headline: planner headlineText for Redis (non-empty on success; planning must pass gate).
    Raises RunwayVideoMVPError on any failure (no generic prompt fallback).
    """
    from engine.video_job_context import video_job_clear_phase

    try:
        return _generate_one_video_mvp_body(
            product_name,
            product_description,
            public_base_url=public_base_url,
            job_id=job_id,
        )
    finally:
        video_job_clear_phase()


log_config_warning_if_missing_key()
