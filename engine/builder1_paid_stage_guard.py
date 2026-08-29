"""
Builder1 paid-stage guard — checkpoint context + outcome tracking helpers.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_builder1_job_id: ContextVar[str] = ContextVar("builder1_job_id", default="")
_builder1_campaign_id: ContextVar[str] = ContextVar("builder1_campaign_id", default="")


@contextmanager
def builder1_paid_stage_context(*, job_id: str, campaign_id: str = "") -> Iterator[None]:
    t_job = _builder1_job_id.set((job_id or "").strip())
    t_campaign = _builder1_campaign_id.set((campaign_id or "").strip())
    try:
        yield
    finally:
        _builder1_job_id.reset(t_job)
        _builder1_campaign_id.reset(t_campaign)


def current_builder1_job_id() -> str:
    return _builder1_job_id.get()


def current_builder1_campaign_id() -> str:
    return _builder1_campaign_id.get()


def checkpoint_before_paid_call(stage: str) -> None:
    from engine.builder1_job_cancellation import checkpoint_builder1_cancellation

    checkpoint_builder1_cancellation(
        current_builder1_job_id(),
        campaign_id=current_builder1_campaign_id(),
        stage=stage,
    )


def record_paid_stage_submitted(
    *,
    job_id: str,
    stage: str,
    campaign_id: str = "",
    ad_index: Optional[int] = None,
) -> None:
    from engine.builder1_jobs_store import update_builder1_job

    jid = (job_id or "").strip()
    if not jid:
        return
    fields: dict = {
        "lastPaidStage": stage,
        "lastPaidStageStatus": "submitted",
    }
    if ad_index is not None:
        fields["lastPaidAdIndex"] = int(ad_index)
    update_builder1_job(jid, **fields)
    if campaign_id:
        from engine.builder1_campaign_store import record_campaign_paid_stage

        record_campaign_paid_stage(
            campaign_id,
            stage=stage,
            status="submitted",
            ad_index=ad_index,
            owner_job_id=jid,
        )


def record_paid_stage_succeeded(
    *,
    job_id: str,
    stage: str,
    campaign_id: str = "",
    ad_index: Optional[int] = None,
) -> None:
    from engine.builder1_jobs_store import update_builder1_job

    jid = (job_id or "").strip()
    if not jid:
        return
    update_builder1_job(
        jid,
        lastPaidStage=stage,
        lastPaidStageStatus="succeeded",
    )
    if campaign_id:
        from engine.builder1_campaign_store import record_campaign_paid_stage

        record_campaign_paid_stage(
            campaign_id,
            stage=stage,
            status="succeeded",
            ad_index=ad_index,
            owner_job_id=jid,
        )


def record_paid_stage_failed_known(
    *,
    job_id: str,
    stage: str,
    campaign_id: str = "",
    ad_index: Optional[int] = None,
    error: str = "",
) -> None:
    from engine.builder1_jobs_store import update_builder1_job

    jid = (job_id or "").strip()
    if not jid:
        return
    fields: dict = {
        "lastPaidStage": stage,
        "lastPaidStageStatus": "failed_known",
    }
    if error:
        fields["lastPaidStageError"] = error[:200]
    if ad_index is not None:
        fields["lastPaidAdIndex"] = int(ad_index)
    update_builder1_job(jid, **fields)
    if campaign_id:
        from engine.builder1_campaign_store import record_campaign_paid_stage

        record_campaign_paid_stage(
            campaign_id,
            stage=stage,
            status="failed_known",
            ad_index=ad_index,
            owner_job_id=jid,
        )


def record_paid_stage_outcome_unknown(
    *,
    job_id: str,
    stage: str,
    campaign_id: str = "",
    ad_index: Optional[int] = None,
) -> None:
    from engine.builder1_jobs_store import update_builder1_job

    jid = (job_id or "").strip()
    if not jid:
        return
    update_builder1_job(
        jid,
        status="error",
        error="paid_stage_outcome_unknown",
        lastPaidStage=stage,
        lastPaidStageStatus="outcome_unknown",
        retryable=False,
    )
    if campaign_id:
        from engine.builder1_campaign_store import record_campaign_paid_stage

        record_campaign_paid_stage(
            campaign_id,
            stage=stage,
            status="outcome_unknown",
            ad_index=ad_index,
            owner_job_id=jid,
        )
