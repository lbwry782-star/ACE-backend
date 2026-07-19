"""Builder1 planning job metrics — model-call counts and stage durations."""
from __future__ import annotations

import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_metrics_ctx: ContextVar[Optional["Builder1PlanningMetrics"]] = ContextVar(
    "builder1_planning_metrics",
    default=None,
)

NORMAL_PLANNING_CALLS_WITH_NAME = 6
NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME = 7
PLANNING_LATENCY_PREFERRED_MS = 150_000
PLANNING_LATENCY_WARN_MS = 180_000
PLANNING_LATENCY_ALERT_MS = 240_000


@dataclass
class Builder1PlanningMetrics:
    campaign_id: str = ""
    job_id: str = ""
    product_name_call_used: bool = False
    product_name_stage_calls: int = 0
    strategy_stage_calls: int = 0
    slogan_stage_calls: int = 0
    conceptual_stage_calls: int = 0
    physical_stage_calls: int = 0
    graphic_stage_calls: int = 0
    series_stage_calls: int = 0
    focused_repair_calls: int = 0
    stage_repair_calls: int = 0
    stage_retry_calls: int = 0
    stage_model_fallback_calls: int = 0
    total_planning_model_calls: int = 0
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_planning_duration_ms: int = 0
    _stage_starts: Dict[str, float] = field(default_factory=dict, repr=False)
    _pipeline_start: float = field(default_factory=time.perf_counter, repr=False)

    def begin_pipeline_pass(self, pass_name: str) -> None:
        self._pipeline_start = time.perf_counter()

    def end_pipeline_pass(self) -> None:
        self.total_planning_duration_ms = int((time.perf_counter() - self._pipeline_start) * 1000)
        logger.info(
            "BUILDER1_PIPELINE_DURATION durationMs=%s",
            self.total_planning_duration_ms,
        )

    def begin_stage(self, stage: str, *, attempt: int = 1) -> None:
        self._stage_starts[f"{stage}:{attempt}"] = time.perf_counter()

    def end_stage(self, stage: str, *, attempt: int = 1) -> None:
        self._stage_starts.pop(f"{stage}:{attempt}", None)

    def record_model_call(self, stage: Optional[str]) -> None:
        self.total_planning_model_calls += 1
        if not stage:
            return
        if stage == "product_name_resolution":
            self.product_name_call_used = True
            self.product_name_stage_calls += 1
        elif stage == "strategy_stage":
            self.strategy_stage_calls += 1
        elif stage == "slogan_stage":
            self.slogan_stage_calls += 1
        elif stage == "conceptual_stage":
            self.conceptual_stage_calls += 1
        elif stage == "brand_physical":
            self.physical_stage_calls += 1
        elif stage == "graphic_system":
            self.graphic_stage_calls += 1
        elif stage == "series_ads":
            self.series_stage_calls += 1
        elif stage in {
            "strategy_candidate_repair",
            "slogan_candidate_repair",
            "conceptual_candidate_repair",
            "marketing_text_repair",
        }:
            self.focused_repair_calls += 1

    def record_stage_repair(self, stage: str) -> None:
        self.stage_repair_calls += 1

    def record_stage_retry(self, stage: str) -> None:
        self.stage_retry_calls += 1

    def record_stage_model_fallback(self, stage: str) -> None:
        self.stage_model_fallback_calls += 1

    def record_token_usage(self, *, prompt_tokens: Optional[int], output_tokens: Optional[int]) -> None:
        if prompt_tokens is None and output_tokens is None:
            return
        if prompt_tokens is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + int(prompt_tokens)
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + int(output_tokens)
        if self.prompt_tokens is not None or self.output_tokens is not None:
            self.total_tokens = (self.prompt_tokens or 0) + (self.output_tokens or 0)

    def log_summary(self) -> None:
        expected = (
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME
            if self.product_name_call_used
            else NORMAL_PLANNING_CALLS_WITH_NAME
        )
        warn_ms = int(os.environ.get("BUILDER1_PLANNING_LATENCY_WARN_MS", str(PLANNING_LATENCY_WARN_MS)))
        alert_ms = int(os.environ.get("BUILDER1_PLANNING_LATENCY_ALERT_MS", str(PLANNING_LATENCY_ALERT_MS)))
        preferred_ms = int(
            os.environ.get("BUILDER1_PLANNING_LATENCY_PREFERRED_MS", str(PLANNING_LATENCY_PREFERRED_MS))
        )
        actual_planning_calls = self.total_planning_model_calls
        if self.total_planning_duration_ms > alert_ms:
            logger.error(
                "BUILDER1_PLANNING_LATENCY_ALERT campaignId=%s jobId=%s durationMs=%s thresholdMs=%s",
                self.campaign_id or "",
                self.job_id or "",
                self.total_planning_duration_ms,
                alert_ms,
            )
        elif self.total_planning_duration_ms > warn_ms:
            logger.warning(
                "BUILDER1_PLANNING_LATENCY_HIGH campaignId=%s jobId=%s durationMs=%s thresholdMs=%s",
                self.campaign_id or "",
                self.job_id or "",
                self.total_planning_duration_ms,
                warn_ms,
            )
        elif self.total_planning_duration_ms > preferred_ms:
            logger.info(
                "BUILDER1_PLANNING_LATENCY_ABOVE_PREFERRED campaignId=%s jobId=%s durationMs=%s preferredMs=%s",
                self.campaign_id or "",
                self.job_id or "",
                self.total_planning_duration_ms,
                preferred_ms,
            )
        if actual_planning_calls > expected:
            logger.warning(
                "BUILDER1_PLANNING_CALL_COUNT_HIGH campaignId=%s jobId=%s total=%s expected=%s "
                "marketingTextRepairs=%s stageRepairs=%s stageRetries=%s stageModelFallbacks=%s",
                self.campaign_id or "",
                self.job_id or "",
                actual_planning_calls,
                expected,
                self.focused_repair_calls,
                self.stage_repair_calls,
                self.stage_retry_calls,
                self.stage_model_fallback_calls,
            )

        logger.info(
            "BUILDER1_PLANNING_CALL_SUMMARY campaignId=%s jobId=%s "
            "productNameCallUsed=%s productNameStageCalls=%s strategyStageCalls=%s sloganStageCalls=%s "
            "conceptualStageCalls=%s physicalStageCalls=%s graphicStageCalls=%s "
            "seriesStageCalls=%s marketingTextRepairCalls=%s stageRepairCalls=%s stageRetryCalls=%s "
            "stageModelFallbackCalls=%s normalExpectedCalls=%s actualPlanningCalls=%s "
            "totalPlanningModelCalls=%s "
            "promptTokens=%s outputTokens=%s totalTokens=%s totalPlanningDurationMs=%s",
            self.campaign_id or "",
            self.job_id or "",
            str(self.product_name_call_used).lower(),
            self.product_name_stage_calls,
            self.strategy_stage_calls,
            self.slogan_stage_calls,
            self.conceptual_stage_calls,
            self.physical_stage_calls,
            self.graphic_stage_calls,
            self.series_stage_calls,
            self.focused_repair_calls,
            self.stage_repair_calls,
            self.stage_retry_calls,
            self.stage_model_fallback_calls,
            expected,
            actual_planning_calls,
            self.total_planning_model_calls,
            self.prompt_tokens if self.prompt_tokens is not None else "",
            self.output_tokens if self.output_tokens is not None else "",
            self.total_tokens if self.total_tokens is not None else "",
            self.total_planning_duration_ms,
        )


def log_builder1_initial_ad_timing(
    *,
    campaign_id: str,
    job_id: str,
    planning_duration_ms: int,
    campaign_persistence_duration_ms: int,
    reservation_duration_ms: int,
    image_generation_duration_ms: int,
    compliance_review_duration_ms: int,
    compliance_regeneration_count: int,
    total_initial_request_duration_ms: int,
) -> None:
    logger.info(
        "BUILDER1_INITIAL_AD_TIMING campaignId=%s jobId=%s planningDurationMs=%s "
        "campaignPersistenceDurationMs=%s reservationDurationMs=%s "
        "imageGenerationDurationMs=%s complianceReviewDurationMs=%s "
        "complianceRegenerationCount=%s totalInitialRequestDurationMs=%s",
        campaign_id,
        job_id,
        planning_duration_ms,
        campaign_persistence_duration_ms,
        reservation_duration_ms,
        image_generation_duration_ms,
        compliance_review_duration_ms,
        compliance_regeneration_count,
        total_initial_request_duration_ms,
    )


def log_builder1_next_ad_timing(
    *,
    campaign_id: str,
    job_id: str,
    ad_index: int,
    image_generation_duration_ms: int,
    compliance_review_duration_ms: int,
    compliance_regeneration_count: int,
    total_next_ad_duration_ms: int,
) -> None:
    logger.info(
        "BUILDER1_NEXT_AD_TIMING campaignId=%s jobId=%s adIndex=%s "
        "imageGenerationDurationMs=%s complianceReviewDurationMs=%s "
        "complianceRegenerationCount=%s totalNextAdDurationMs=%s",
        campaign_id,
        job_id,
        ad_index,
        image_generation_duration_ms,
        compliance_review_duration_ms,
        compliance_regeneration_count,
        total_next_ad_duration_ms,
    )


def get_planning_metrics() -> Optional[Builder1PlanningMetrics]:
    return _metrics_ctx.get()


def set_planning_metrics(metrics: Builder1PlanningMetrics):
    return _metrics_ctx.set(metrics)


def reset_planning_metrics(token) -> None:
    _metrics_ctx.reset(token)
