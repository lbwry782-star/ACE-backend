"""Shared mocks for Builder2 fresh-production generate-video regression tests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List

from unittest.mock import MagicMock, patch

from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL
from tests.test_builder2_media_resume import _mock_start_image_data_uri

FRESH_GENERATE_ENV = {
    "RUNWAY_API_KEY": "rk-test",
    "OPENAI_API_KEY": "sk-test",
    "BUILDER2_TOURNAMENT_ENABLED": "true",
    "ACE_PUBLIC_BASE_URL": "https://example.com",
    "ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret",
}

RAW_RUNWAY_URL = "https://runway.example.com/raw.mp4"


@contextmanager
def patch_fresh_generate_media_mocks(*, final_url: str = CLOSURE_URL) -> Iterator[List[str]]:
    """Patch durable finalization and default media-pipeline paid steps for generate tests."""
    runway_submission_calls: List[str] = []

    def _submit(**_kwargs: Any) -> Any:
        runway_submission_calls.append("submit")
        return Builder2RunwaySubmissionResult(
            task_id="task-mock-1",
            task_created=True,
            request_submitted=True,
        )

    from engine.builder2_runway_submission import Builder2RunwaySubmissionResult

    def _poll(**_kwargs: Any) -> tuple[str, str]:
        return ("SUCCEEDED", RAW_RUNWAY_URL)

    capability_patch, closure_patch, publish_patch, _publish_mock = patch_media_pipeline_durable_finalization(final_url)
    start_image_uri = _mock_start_image_data_uri()
    start_counters = MagicMock(
        startImageNormalCalls=1,
        startImageRepairCalls=0,
        startImageRetryCalls=0,
        startImageGeneratedCount=1,
    )
    start_result = MagicMock(
        data_uri=start_image_uri,
        counters=start_counters,
        metadata={
            "startImageGenerationSize": "1536x1024",
            "startImageOutputWidth": 1280,
            "startImageOutputHeight": 720,
            "startImageMimeType": "image/png",
        },
    )
    with capability_patch, closure_patch, publish_patch, patch(
        "engine.builder2_runway_submission.submit_builder2_runway_task",
        side_effect=_submit,
    ), patch(
        "engine.builder2_media_pipeline._default_submit_runway_task",
        side_effect=lambda **_kwargs: "task-mock-1",
    ), patch(
        "engine.builder2_media_pipeline._default_poll_runway_task",
        side_effect=_poll,
    ), patch(
        "engine.builder2_media_pipeline._default_generate_start_image",
        return_value=start_image_uri,
    ), patch(
        "engine.builder2_start_image_pipeline.generate_builder2_start_image",
        return_value=start_result,
    ), patch(
        "engine.builder2_media_pipeline._default_postprocess_video",
        return_value=f"{final_url.rsplit('/', 1)[0]}/headline.mp4",
    ):
        yield runway_submission_calls
