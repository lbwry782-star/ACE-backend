"""
Standalone ACE video worker: consume Redis queue and run generate_one_video_mvp.

Deploy on Render as a Background Worker service:
  start command: python worker_video.py
  env: same REDIS_URL, RUNWAY_*, OPENAI_*, ACE_PUBLIC_BASE_URL, etc. as web service.

Does not import Flask.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("worker_video")

_active_lock = threading.Lock()
_active_job_id: str | None = None
_heartbeat_stop = threading.Event()


def _get_active_job_id() -> str | None:
    with _active_lock:
        return _active_job_id


def _set_active_job_id(job_id: str | None) -> None:
    global _active_job_id
    with _active_lock:
        _active_job_id = job_id


def _heartbeat_loop() -> None:
    from engine.video_jobs_redis import video_job_touch_progress

    while not _heartbeat_stop.wait(30.0):
        jid = _get_active_job_id()
        if not jid:
            continue
        try:
            video_job_touch_progress(jid)
        except Exception:
            logger.debug("VIDEO_JOB_HEARTBEAT_FAIL jobId=%s", jid, exc_info=True)


def _install_shutdown_signals() -> None:
    def _handle(signum: int, frame: object | None) -> None:
        jid = _get_active_job_id()
        logger.info(
            "VIDEO_JOB_INTERRUPT_CAUGHT jobId=%s reason=worker_shutdown_during_job",
            jid or "(none)",
        )
        if jid:
            try:
                from engine.video_job_context import video_job_get_phase
                from engine.video_jobs_redis import video_job_mark_interrupted

                phase = video_job_get_phase()
                logger.info("VIDEO_JOB_PHASE_AT_INTERRUPT=%s", phase)
                logger.info("VIDEO_JOB_INFRA_FAILURE=true")
                video_job_mark_interrupted(jid)
                logger.info(
                    "VIDEO_JOB_INTERRUPT_FINALIZED jobId=%s interrupt_code=interrupted_worker_shutdown error=worker_shutdown_during_job",
                    jid,
                )
            except Exception as e:
                logger.error(
                    "VIDEO_JOB_ERROR_FINALIZE_FAILED jobId=%s err=%s",
                    jid,
                    e,
                    exc_info=True,
                )
        _heartbeat_stop.set()
        sys.exit(128 + signum if signum > 0 else 0)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main() -> None:
    if not (os.environ.get("REDIS_URL") or "").strip():
        logger.error("REDIS_URL is required for worker_video")
        sys.exit(1)

    # Import after env check so engine can read other env vars
    from engine.runway_video import RunwayVideoMVPError, generate_one_video_mvp
    from engine.video_headline_postprocess import log_video_headline_delivery_startup
    from engine.video_jobs_redis import (
        get_redis,
        job_key,
        video_job_brpop,
        video_job_mark_done,
        video_job_mark_error,
        video_job_touch_progress,
        QUEUE_KEY,
    )

    try:
        log_video_headline_delivery_startup("worker")
    except Exception as e:
        logger.warning("VIDEO_HEADLINE_UPLOAD_CONFIG worker startup failed err=%s", e)

    get_redis()  # connect once
    _install_shutdown_signals()
    hb_thread = threading.Thread(target=_heartbeat_loop, name="video_job_heartbeat", daemon=True)
    hb_thread.start()
    logger.info("VIDEO_WORKER_START queue=%s", QUEUE_KEY)

    while True:
        logger.info("VIDEO_WORKER_WAITING queue=%s", QUEUE_KEY)
        try:
            job_id = video_job_brpop(timeout_seconds=30)
        except Exception as e:
            logger.error("VIDEO_WORKER_REDIS_ERROR err=%s", e, exc_info=True)
            time.sleep(1.0)
            continue
        if not job_id:
            logger.info("VIDEO_WORKER_EMPTY_QUEUE")
            continue
        logger.info("VIDEO_WORKER_DEQUEUED jobId=%s", job_id)

        logger.info("VIDEO_JOB_STARTED jobId=%s", job_id)
        _set_active_job_id(job_id)
        try:
            try:
                video_job_touch_progress(job_id)
            except Exception as e:
                logger.warning("VIDEO_JOB_TOUCH_PROGRESS_FAIL jobId=%s err=%s", job_id, e)
            logger.info("VIDEO_JOB_STEP step=redis_get_client start jobId=%s", job_id)
            r = get_redis()
            logger.info("VIDEO_JOB_STEP step=redis_get_client done jobId=%s", job_id)

            logger.info("VIDEO_JOB_STEP step=load_job_data start jobId=%s", job_id)
            data = r.hgetall(job_key(job_id))
            logger.info(
                "VIDEO_JOB_STEP step=load_job_data done jobId=%s fields=%s",
                job_id,
                len(data),
            )
            if not data:
                logger.warning("VIDEO_JOB_MISSING jobId=%s (no hash)", job_id)
                continue

            product_name = data.get("product_name") or ""
            product_description = data.get("product_description") or ""
            public_base_url = data.get("public_base_url") or ""

            logger.info("VIDEO_JOB_STEP step=generate_one_video_mvp start jobId=%s", job_id)
            video_url, marketing_text, overlay_headline = generate_one_video_mvp(
                product_name,
                product_description,
                public_base_url=public_base_url,
                job_id=job_id,
            )
            logger.info("VIDEO_JOB_STEP step=generate_one_video_mvp done jobId=%s", job_id)

            logger.info("VIDEO_JOB_CHOSEN_URL video_url=%s jobId=%s", video_url, job_id)
            logger.info(
                "VIDEO_JOB_CHOSEN_URL jobId=%s video_url=%s before_redis=1",
                job_id,
                video_url,
            )
            logger.info("VIDEO_JOB_STEP step=redis_mark_done start jobId=%s", job_id)
            video_job_mark_done(job_id, video_url, marketing_text or "", overlay_headline or "")
            logger.info("VIDEO_JOB_STEP step=redis_mark_done done jobId=%s", job_id)
            logger.info("VIDEO_JOB_RESULT video_url=%s jobId=%s", video_url, job_id)
            logger.info(
                "VIDEO_JOB_RESULT jobId=%s video_url=%s redis_written=1",
                job_id,
                video_url,
            )
            logger.info("VIDEO_JOB_DONE jobId=%s outcome=success", job_id)
        except RunwayVideoMVPError as e:
            _reason = e.args[0] if getattr(e, "args", None) else "runway_mvp"
            logger.warning("VIDEO_JOB_FAILED jobId=%s reason=%s", job_id, _reason)
            logger.warning("VIDEO_JOB_ERROR jobId=%s err=RunwayVideoMVPError", job_id)
            try:
                video_job_mark_error(job_id, (_reason or "").strip() or "video_generation_failed")
            except Exception as mark_err:
                logger.error(
                    "VIDEO_JOB_ERROR mark_failed jobId=%s err=%s",
                    job_id,
                    mark_err,
                    exc_info=True,
                )
            logger.info("VIDEO_JOB_DONE jobId=%s outcome=error", job_id)
        except Exception as e:
            logger.error("VIDEO_JOB_FATAL jobId=%s error=%s", job_id, e, exc_info=True)
            try:
                video_job_mark_error(job_id, "video_generation_failed")
            except Exception as mark_err:
                logger.error(
                    "VIDEO_JOB_FATAL mark_error_failed jobId=%s err=%s",
                    job_id,
                    mark_err,
                    exc_info=True,
                )
            logger.info("VIDEO_JOB_DONE jobId=%s outcome=error", job_id)
        finally:
            _set_active_job_id(None)


if __name__ == "__main__":
    main()
