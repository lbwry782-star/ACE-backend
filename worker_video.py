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
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("worker_video")


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
        QUEUE_KEY,
    )

    try:
        log_video_headline_delivery_startup("worker")
    except Exception as e:
        logger.warning("VIDEO_HEADLINE_UPLOAD_CONFIG worker startup failed err=%s", e)

    get_redis()  # connect once
    logger.info("VIDEO_WORKER_START queue=%s", QUEUE_KEY)

    while True:
        job_id = video_job_brpop(timeout_seconds=30)
        if not job_id:
            continue

        logger.info("VIDEO_JOB_STARTED jobId=%s", job_id)
        try:
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
                video_job_mark_error(job_id, "video_generation_failed")
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


if __name__ == "__main__":
    main()
