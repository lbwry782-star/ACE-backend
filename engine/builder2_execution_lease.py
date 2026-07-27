"""
Builder2 distributed execution lease — heartbeat renewal and safe status reporting.
"""
from __future__ import annotations

from typing import Any, Dict

from engine.builder2_tournament_recovery import (
    acquire_job_lease,
    expire_job_lease,
    has_active_lease,
    release_job_lease,
    renew_job_lease,
)


def get_execution_lease_status(job_id: str) -> str:
    if has_active_lease(job_id):
        return "active"
    return "none"


def execution_lease_public_fields(job_id: str) -> Dict[str, Any]:
    return {
        "executionLeaseStatus": get_execution_lease_status(job_id),
    }


__all__ = [
    "acquire_job_lease",
    "release_job_lease",
    "expire_job_lease",
    "has_active_lease",
    "renew_job_lease",
    "get_execution_lease_status",
    "execution_lease_public_fields",
]
