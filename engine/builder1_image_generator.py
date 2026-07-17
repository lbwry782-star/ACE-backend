"""
Builder1 campaign-series image generation (active production).
"""
from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_visual_prompt import build_visual_prompt

logger = logging.getLogger(__name__)

ImageCaller = Callable[[str, str], bytes]


class ProgressCallback(Protocol):
    def __call__(self, completed_ads: int, total_ads: int) -> None: ...


@dataclass
class Builder1SeriesImageResult:
    index: int
    visual_prompt: str
    image_bytes: bytes


@dataclass
class Builder1SeriesImagesResult:
    images: List[Builder1SeriesImageResult]


def _generate_one_with_retry(
    *,
    series_plan: Builder1SeriesPlan,
    ad_index: int,
    image_caller: ImageCaller,
) -> Builder1SeriesImageResult:
    ad = next(a for a in series_plan.ads if a.index == ad_index)
    prompt = build_visual_prompt(series_plan, ad)
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            logger.info(
                "BUILDER1_SERIES_IMAGE_START adIndex=%s attempt=%s",
                ad_index,
                attempt,
            )
            image_bytes = image_caller(prompt, series_plan.format)
            logger.info("BUILDER1_SERIES_IMAGE_OK adIndex=%s", ad_index)
            return Builder1SeriesImageResult(
                index=ad_index,
                visual_prompt=prompt,
                image_bytes=image_bytes,
            )
        except Exception as exc:
            last_exc = exc
            logger.error(
                "BUILDER1_SERIES_IMAGE_FAILED adIndex=%s attempt=%s err=%s",
                ad_index,
                attempt,
                exc,
            )
    raise RuntimeError(f"image_generation_failed_ad_{ad_index}") from last_exc


def generate_builder1_series_images(
    series_plan: Builder1SeriesPlan,
    image_caller: ImageCaller,
    *,
    on_progress: Optional[ProgressCallback] = None,
) -> Builder1SeriesImagesResult:
    """
    Generate one image per ad with bounded concurrency min(adCount, 2).
    Atomic: any failure raises and caller must fail the whole campaign.
    """
    ad_count = series_plan.ad_count
    indexes = [ad.index for ad in sorted(series_plan.ads, key=lambda a: a.index)]
    max_workers = min(ad_count, 2)
    results_by_index: dict[int, Builder1SeriesImageResult] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _generate_one_with_retry,
                series_plan=series_plan,
                ad_index=idx,
                image_caller=image_caller,
            ): idx
            for idx in indexes
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results_by_index[result.index] = result
            completed += 1
            if on_progress:
                on_progress(completed, ad_count)

    ordered = [results_by_index[i] for i in indexes]
    return Builder1SeriesImagesResult(images=ordered)


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


# Legacy single-image API removed from active production path.
def generate_builder1_image(*_args, **_kwargs):
    raise NotImplementedError("generate_builder1_image removed; use generate_builder1_series_images")
