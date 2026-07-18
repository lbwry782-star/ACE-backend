"""Builder1 planning job metrics — model-call counts and stage durations."""
from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_metrics_ctx: ContextVar[Optional["Builder1PlanningMetrics"]] = ContextVar(
    "builder1_planning_metrics",
    default=None,
)


@dataclass
class Builder1PlanningMetrics:
    campaign_id: str = ""
    job_id: str = ""
    strategic_restart_used: bool = False
    strategy_scan_calls: int = 0
    strategy_repair_calls: int = 0
    strategy_selection_calls: int = 0
    slogan_scan_calls: int = 0
    slogan_review_calls: int = 0
    slogan_repair_calls: int = 0
    final_judge_calls: int = 0
    total_planning_model_calls: int = 0
    _stage_starts: Dict[str, float] = field(default_factory=dict, repr=False)
    _pipeline_start: float = field(default_factory=time.perf_counter, repr=False)
    _pipeline_pass: str = "initial"

    def begin_pipeline_pass(self, pass_name: str) -> None:
        self._pipeline_pass = pass_name
        self._pipeline_start = time.perf_counter()

    def end_pipeline_pass(self) -> None:
        duration_ms = int((time.perf_counter() - self._pipeline_start) * 1000)
        logger.info(
            "BUILDER1_PIPELINE_DURATION pass=%s durationMs=%s",
            self._pipeline_pass,
            duration_ms,
        )

    def begin_stage(self, stage: str, *, attempt: int = 1) -> None:
        self._stage_starts[f"{stage}:{attempt}"] = time.perf_counter()

    def end_stage(self, stage: str, *, attempt: int = 1) -> None:
        self._stage_starts.pop(f"{stage}:{attempt}", None)

    def record_model_call(self, stage: Optional[str]) -> None:
        self.total_planning_model_calls += 1
        if not stage:
            return
        if stage == "strategy_scan":
            self.strategy_scan_calls += 1
        elif stage == "strategy_candidate_repair":
            self.strategy_repair_calls += 1
        elif stage == "strategy_selection":
            self.strategy_selection_calls += 1
        elif stage == "slogan_scan":
            self.slogan_scan_calls += 1
        elif stage == "slogan_quality_review":
            self.slogan_review_calls += 1
        elif stage == "slogan_candidate_repair":
            self.slogan_repair_calls += 1
        elif stage == "strategy_judge":
            self.final_judge_calls += 1

    def log_summary(self) -> None:
        logger.info(
            "BUILDER1_PLANNING_CALL_SUMMARY campaignId=%s jobId=%s "
            "strategicRestartUsed=%s strategyScanCalls=%s strategyRepairCalls=%s "
            "strategySelectionCalls=%s sloganScanCalls=%s sloganReviewCalls=%s "
            "sloganRepairCalls=%s finalJudgeCalls=%s totalPlanningModelCalls=%s",
            self.campaign_id or "",
            self.job_id or "",
            str(self.strategic_restart_used).lower(),
            self.strategy_scan_calls,
            self.strategy_repair_calls,
            self.strategy_selection_calls,
            self.slogan_scan_calls,
            self.slogan_review_calls,
            self.slogan_repair_calls,
            self.final_judge_calls,
            self.total_planning_model_calls,
        )


def get_planning_metrics() -> Optional[Builder1PlanningMetrics]:
    return _metrics_ctx.get()


def set_planning_metrics(metrics: Builder1PlanningMetrics):
    return _metrics_ctx.set(metrics)


def reset_planning_metrics(token) -> None:
    _metrics_ctx.reset(token)
