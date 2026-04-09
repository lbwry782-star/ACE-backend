"""
Run headline overlay postprocess on the web service when polling /api/video-status (split deploy).

Worker stores the Runway source URL + overlay_headline in Redis; ffmpeg runs here so /api/test-video/<jobId> reads local disk.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from engine.video_bidi import finalize_hebrew_mixed_bidi_for_display, format_bidi_segments_for_log
from engine.video_language import detect_product_description_language

logger = logging.getLogger(__name__)


def ensure_video_postprocessed_for_poll(job_id: str, job: Dict[str, Any]) -> None:
    """
    If job is done and postprocess has not run yet, download Runway MP4, overlay headline, save under /tmp, update Redis.
    Idempotent: postprocessRan=1 skips. On failure, keeps Runway URL and still marks ran.
    """
    if (job.get("postprocessRan") or "") == "1":
        return
    source_url = (job.get("videoUrl") or "").strip()
    # Worker already uploaded artifact; web serves GET /api/video-headline/<token> — do not re-postprocess.
    if "/api/video-headline/" in source_url and "/api/video-headline-artifact" not in source_url:
        from engine.video_jobs_redis import video_job_set_postprocess_ran_only

        video_job_set_postprocess_ran_only(job_id)
        return
    # Older Redis rows: /api/test-video/<jobId> only on same host as file.
    if "/api/test-video/" in source_url:
        from engine.video_jobs_redis import video_job_set_postprocess_ran_only

        video_job_set_postprocess_ran_only(job_id)
        return
    base = (job.get("publicBaseUrl") or "").strip()
    headline = (job.get("overlayHeadline") or "").strip()
    _, content_lang, _, _ = detect_product_description_language(
        job.get("productDescription") or ""
    )
    _rp = (job.get("resolvedProductName") or "").strip()
    _prot = (_rp,) if _rp else ()
    headline, bidi_hl, segs_hl = finalize_hebrew_mixed_bidi_for_display(
        headline,
        content_language=content_lang,
        protected_phrases=_prot,
    )
    logger.info("VIDEO_BIDI_FIX_APPLIED_HEADLINE=%s", str(bidi_hl).lower())
    logger.info(
        "VIDEO_BIDI_LATIN_SEGMENTS_HEADLINE=%s",
        format_bidi_segments_for_log(segs_hl),
    )
    logger.info("VIDEO_BIDI_FIX_APPLIED_COPY=false")
    logger.info("VIDEO_BIDI_LATIN_SEGMENTS_COPY=%s", json.dumps([], ensure_ascii=False))
    if not source_url:
        from engine.video_jobs_redis import video_job_set_postprocess_ran_only

        video_job_set_postprocess_ran_only(job_id)
        return

    from engine.video_headline_postprocess import postprocess_video_headline
    from engine.video_jobs_redis import video_job_set_postprocess_result

    try:
        final_url = postprocess_video_headline(
            source_url,
            base,
            headline=headline,
            job_id=job_id,
        )
    except Exception as e:
        logger.warning(
            "VIDEO_WEB_POSTPROCESS_FAIL jobId=%s err=%s fallback_source=1",
            job_id,
            e,
            exc_info=True,
        )
        final_url = source_url
    video_job_set_postprocess_result(job_id, final_url)
