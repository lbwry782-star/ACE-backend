"""
Builder1 paid provider dispatch — outcome classification and safe execution.
"""
from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

OutcomeKind = Literal["failed_known", "outcome_unknown"]


class PaidStageOutcomeUnknownError(Exception):
    """Provider outcome ambiguous after submission — no automatic paid retry."""

    def __init__(self, stage: str, message: str = "paid_stage_outcome_unknown"):
        self.stage = stage
        super().__init__(message)


class PaidStageRetryBlockedError(Exception):
    """Prior paid stage left job in submitted/outcome_unknown — no blind retry."""

    def __init__(self, stage: str, prior_status: str):
        self.stage = stage
        self.prior_status = prior_status
        super().__init__(f"paid_stage_retry_blocked:{stage}:{prior_status}")


_BLOCKING_PAID_STATUSES = frozenset({"submitted", "in_flight", "outcome_unknown"})


def classify_provider_exception(exc: Exception, *, after_submit: bool) -> OutcomeKind:
    """
    Classify provider/transport errors after a paid dispatch was recorded as submitted.
    Timeout / connection loss without HTTP response → outcome_unknown.
    """
    if not after_submit:
        return "failed_known"

    name = type(exc).__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "BrokenPipeError",
        "RemoteDisconnected",
    }:
        return "outcome_unknown"

    resp = getattr(exc, "response", None)
    if resp is not None:
        return "failed_known"

    text = str(exc).lower()
    unknown_tokens = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection error",
        "connection refused",
        "broken pipe",
        "remote disconnected",
        "unexpected eof",
        "ssl:",
        "reset by peer",
    )
    if any(token in text for token in unknown_tokens):
        return "outcome_unknown"
    return "failed_known"


def assert_paid_stage_may_start(job_id: str) -> None:
    """Block automatic paid retry when a prior dispatch outcome is unknown or in-flight."""
    jid = (job_id or "").strip()
    if not jid:
        return
    from engine.builder1_jobs_store import get_builder1_job

    job = get_builder1_job(jid)
    if not job:
        return
    status = str(job.get("lastPaidStageStatus") or "").strip()
    if status in _BLOCKING_PAID_STATUSES:
        stage = str(job.get("lastPaidStage") or "")
        if status == "outcome_unknown":
            raise PaidStageOutcomeUnknownError(stage or "unknown")
        raise PaidStageRetryBlockedError(stage or "unknown", status)


def run_paid_provider_call(
    stage: str,
    dispatch: Callable[[], T],
    *,
    ad_index: Optional[int] = None,
    pre_submit: Optional[Callable[[], None]] = None,
) -> T:
    """
    Execute one paid provider dispatch with durable submitted/succeeded/outcome tracking.
    pre_submit runs before submission (config validation — failures are failed_known).
    """
    from engine.builder1_job_cancellation import Builder1JobCancelledError, checkpoint_builder1_cancellation
    from engine.builder1_paid_stage_guard import (
        current_builder1_campaign_id,
        current_builder1_job_id,
        record_paid_stage_failed_known,
        record_paid_stage_outcome_unknown,
        record_paid_stage_submitted,
        record_paid_stage_succeeded,
    )

    stage_name = (stage or "").strip() or "paid_provider"
    jid = current_builder1_job_id()
    cid = current_builder1_campaign_id()
    checkpoint_builder1_cancellation(jid, campaign_id=cid, stage=stage_name)
    assert_paid_stage_may_start(jid)

    if pre_submit is not None:
        try:
            pre_submit()
        except Builder1JobCancelledError:
            raise
        except Exception as exc:
            if jid:
                record_paid_stage_failed_known(
                    job_id=jid,
                    stage=stage_name,
                    campaign_id=cid,
                    ad_index=ad_index,
                    error=str(exc)[:200],
                )
            raise

    submitted = False
    if jid:
        record_paid_stage_submitted(job_id=jid, stage=stage_name, campaign_id=cid, ad_index=ad_index)
        submitted = True

    try:
        result = dispatch()
    except Builder1JobCancelledError:
        raise
    except PaidStageOutcomeUnknownError:
        raise
    except Exception as exc:
        if jid and submitted:
            kind = classify_provider_exception(exc, after_submit=True)
            if kind == "outcome_unknown":
                record_paid_stage_outcome_unknown(
                    job_id=jid,
                    stage=stage_name,
                    campaign_id=cid,
                    ad_index=ad_index,
                )
                logger.error(
                    "BUILDER1_PAID_STAGE_OUTCOME_UNKNOWN jobId=%s stage=%s adIndex=%s err=%s",
                    jid,
                    stage_name,
                    ad_index,
                    exc,
                )
                raise PaidStageOutcomeUnknownError(stage_name) from exc
            record_paid_stage_failed_known(
                job_id=jid,
                stage=stage_name,
                campaign_id=cid,
                ad_index=ad_index,
                error=str(exc)[:200],
            )
        raise

    if jid:
        record_paid_stage_succeeded(job_id=jid, stage=stage_name, campaign_id=cid, ad_index=ad_index)
    return result
