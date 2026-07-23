"""
Builder2 tournament metrics — call counts and elapsed time tracking.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_METRICS: Dict[str, Any] = {
    "strategyCalls": 0,
    "strategyRepairCalls": 0,
    "creatorCalls": 0,
    "creatorRepairCalls": 0,
    "creatorRetryCalls": 0,
    "creatorRejectedAttempts": 0,
    "creatorEligibleCandidates": 0,
    "judgeCalls": 0,
    "judgeRepairCalls": 0,
    "judgeRetryCalls": 0,
    "judgeRejectedResponses": 0,
    "judgeValidJudgments": 0,
    "judgeEligibleCandidates": 0,
    "judgeIneligibleCandidates": 0,
    "judgeUnavailableCandidates": 0,
    "winnerDevelopmentCalls": 0,
    "totalReasoningCalls": 0,
    "strategyElapsedMs": 0,
    "creatorElapsedMs": 0,
    "judgeElapsedMs": 0,
    "winnerDevelopmentElapsedMs": 0,
    "tournamentElapsedMs": 0,
    "tokenUsage": {
        "strategy": {},
        "creator": {},
        "judge": {},
        "winner": {},
    },
}


def ensure_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        metrics = dict(DEFAULT_METRICS)
        metrics["tokenUsage"] = dict(DEFAULT_METRICS["tokenUsage"])
        state["metrics"] = metrics
    for key, default in DEFAULT_METRICS.items():
        if key == "tokenUsage":
            continue
        if key not in metrics:
            metrics[key] = default
    return metrics


def _recalc_total_calls(metrics: Dict[str, Any]) -> None:
    metrics["totalReasoningCalls"] = (
        int(metrics.get("strategyCalls") or 0)
        + int(metrics.get("strategyRepairCalls") or 0)
        + int(metrics.get("creatorCalls") or 0)
        + int(metrics.get("creatorRepairCalls") or 0)
        + int(metrics.get("creatorRetryCalls") or 0)
        + int(metrics.get("judgeCalls") or 0)
        + int(metrics.get("judgeRepairCalls") or 0)
        + int(metrics.get("judgeRetryCalls") or 0)
        + int(metrics.get("winnerDevelopmentCalls") or 0)
    )


def record_model_call(
    state: Dict[str, Any],
    *,
    role: str,
    elapsed_ms: float,
    repair: bool = False,
    retry: bool = False,
    token_usage: Optional[Dict[str, Any]] = None,
) -> None:
    metrics = ensure_metrics(state)
    if role == "builder2_strategy":
        key = "strategyRepairCalls" if repair else "strategyCalls"
        metrics[key] = int(metrics.get(key) or 0) + 1
        metrics["strategyElapsedMs"] = float(metrics.get("strategyElapsedMs") or 0) + elapsed_ms
        bucket = "strategy"
    elif role == "builder2_creator":
        if repair:
            key = "creatorRepairCalls"
        elif retry:
            key = "creatorRetryCalls"
        else:
            key = "creatorCalls"
        metrics[key] = int(metrics.get(key) or 0) + 1
        metrics["creatorElapsedMs"] = float(metrics.get("creatorElapsedMs") or 0) + elapsed_ms
        bucket = "creator"
    elif role == "builder2_judge":
        if repair:
            key = "judgeRepairCalls"
        elif retry:
            key = "judgeRetryCalls"
        else:
            key = "judgeCalls"
        metrics[key] = int(metrics.get(key) or 0) + 1
        metrics["judgeElapsedMs"] = float(metrics.get("judgeElapsedMs") or 0) + elapsed_ms
        bucket = "judge"
    elif role == "builder2_winner":
        metrics["winnerDevelopmentCalls"] = int(metrics.get("winnerDevelopmentCalls") or 0) + 1
        metrics["winnerDevelopmentElapsedMs"] = float(metrics.get("winnerDevelopmentElapsedMs") or 0) + elapsed_ms
        bucket = "winner"
    else:
        bucket = None
    if bucket and token_usage:
        usage_bucket = metrics.setdefault("tokenUsage", {})
        usage_bucket[bucket] = _merge_token_usage(usage_bucket.get(bucket) or {}, token_usage)
    _recalc_total_calls(metrics)


def record_creator_rejected(state: Dict[str, Any]) -> None:
    metrics = ensure_metrics(state)
    metrics["creatorRejectedAttempts"] = int(metrics.get("creatorRejectedAttempts") or 0) + 1


def record_creator_eligible(state: Dict[str, Any]) -> None:
    metrics = ensure_metrics(state)
    metrics["creatorEligibleCandidates"] = int(metrics.get("creatorEligibleCandidates") or 0) + 1


def record_judge_unavailable(state: Dict[str, Any]) -> None:
    metrics = ensure_metrics(state)
    metrics["judgeUnavailableCandidates"] = int(metrics.get("judgeUnavailableCandidates") or 0) + 1
    metrics["judgeRejectedResponses"] = int(metrics.get("judgeRejectedResponses") or 0) + 1


def record_judge_valid(state: Dict[str, Any], *, eligible: bool) -> None:
    metrics = ensure_metrics(state)
    metrics["judgeValidJudgments"] = int(metrics.get("judgeValidJudgments") or 0) + 1
    if eligible:
        metrics["judgeEligibleCandidates"] = int(metrics.get("judgeEligibleCandidates") or 0) + 1
    else:
        metrics["judgeIneligibleCandidates"] = int(metrics.get("judgeIneligibleCandidates") or 0) + 1


def _merge_token_usage(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in new.items():
        if isinstance(value, (int, float)):
            merged[key] = int(merged.get(key) or 0) + int(value)
    return merged


def finalize_tournament_metrics(state: Dict[str, Any], *, elapsed_ms: float) -> None:
    metrics = ensure_metrics(state)
    metrics["tournamentElapsedMs"] = elapsed_ms
    judged = sum(1 for cand in state.get("candidates", {}).values() if cand.get("judgmentId"))
    eligible = sum(
        1
        for cand in state.get("candidates", {}).values()
        if cand.get("eligible") and cand.get("validationStatus") == "accepted"
    )
    rejected_creator = sum(
        1
        for cand in state.get("candidates", {}).values()
        if cand.get("validationStatus") == "creator_rejected"
    )
    unavailable = sum(
        1
        for cand in state.get("candidates", {}).values()
        if cand.get("validationStatus") == "judge_unavailable"
    )
    ineligible_judged = sum(
        1
        for cand in state.get("candidates", {}).values()
        if cand.get("judgmentId") and not cand.get("eligible")
    )
    metrics["creatorRejectedAttempts"] = max(int(metrics.get("creatorRejectedAttempts") or 0), rejected_creator)
    metrics["creatorEligibleCandidates"] = eligible
    metrics["judgeUnavailableCandidates"] = max(int(metrics.get("judgeUnavailableCandidates") or 0), unavailable)
    metrics["judgeValidJudgments"] = max(int(metrics.get("judgeValidJudgments") or 0), judged)
    metrics["judgeEligibleCandidates"] = eligible
    metrics["judgeIneligibleCandidates"] = max(int(metrics.get("judgeIneligibleCandidates") or 0), ineligible_judged)
    _recalc_total_calls(metrics)
    logger.info(
        "BUILDER2_TOURNAMENT_METRICS jobId=%s activePrototypes=%s candidates=%s judgedCandidates=%s "
        "eligibleCandidates=%s rejectedCreatorAttempts=%s judgeUnavailable=%s judgeValid=%s judgeIneligible=%s "
        "strategyCalls=%s strategyRepairCalls=%s creatorCalls=%s creatorRepairCalls=%s creatorRetryCalls=%s "
        "judgeCalls=%s judgeRepairCalls=%s judgeRetryCalls=%s winnerDevelopmentCalls=%s totalReasoningCalls=%s "
        "tournamentElapsedMs=%.1f",
        state.get("jobId"),
        len(state.get("initialActivePrototypeIds") or state.get("activePrototypeIds") or []),
        len(state.get("candidates") or {}),
        judged,
        eligible,
        metrics.get("creatorRejectedAttempts"),
        metrics.get("judgeUnavailableCandidates"),
        metrics.get("judgeValidJudgments"),
        metrics.get("judgeIneligibleCandidates"),
        metrics.get("strategyCalls"),
        metrics.get("strategyRepairCalls"),
        metrics.get("creatorCalls"),
        metrics.get("creatorRepairCalls"),
        metrics.get("creatorRetryCalls"),
        metrics.get("judgeCalls"),
        metrics.get("judgeRepairCalls"),
        metrics.get("judgeRetryCalls"),
        metrics.get("winnerDevelopmentCalls"),
        metrics.get("totalReasoningCalls"),
        elapsed_ms,
    )


class MetricsTimer:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0
