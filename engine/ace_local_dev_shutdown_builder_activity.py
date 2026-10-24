"""
Builder1 / Builder2 / Redis queue activity observation for LOCAL shutdown gate.

Read-only: does not cancel jobs, dequeue Redis, or stop workers.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BUILDER1_JOB_PREFIX = "builder1:job:"
BUILDER1_CAMPAIGN_PREFIX = "builder1:campaign:"
VIDEO_JOB_PREFIX = "ace:video:job:"
VIDEO_QUEUE_KEY = "ace:video:queue"
BUILDER2_LEASE_PREFIX = "ace:builder2:lease:"

BUILDER1_TERMINAL_STATUSES = frozenset({"done", "error", "cancelled"})
BUILDER2_VIDEO_TERMINAL_STATUSES = frozenset(
    {"done", "error", "failed", "cancelled", "recovery_exhausted", "interrupted"}
)


@dataclass
class BuilderActivityObservation:
    kind: str
    status: str = "UNKNOWN"
    job_id: Optional[str] = None
    campaign_id: Optional[str] = None
    builder: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _obs_to_dict(obs: BuilderActivityObservation) -> Dict[str, Any]:
    return {
        "kind": obs.kind,
        "status": obs.status,
        "jobId": obs.job_id,
        "campaignId": obs.campaign_id,
        "builder": obs.builder,
        "notes": list(obs.notes),
    }


def _classify_builder1_job(job_id: str, job: Dict[str, Any]) -> BuilderActivityObservation:
    from engine.builder1_jobs_store import is_builder1_job_stale

    status = str(job.get("status") or "").strip().lower()
    campaign_id = str(job.get("campaignId") or "").strip() or None
    obs = BuilderActivityObservation(
        kind="BUILDER1",
        job_id=job_id,
        campaign_id=campaign_id,
        builder="builder1",
    )
    if not status:
        obs.status = "UNKNOWN"
        obs.notes.append("builder1_missing_status")
        return obs
    if status in BUILDER1_TERMINAL_STATUSES:
        obs.status = "IDLE"
        return obs
    if status == "running":
        if is_builder1_job_stale(job):
            obs.status = "UNKNOWN"
            obs.notes.append("builder1_stale_running")
            return obs
        obs.status = "ACTIVE"
        obs.notes.append("builder1_running")
        stage = str(job.get("stage") or "").strip()
        if stage:
            obs.notes.append(f"stage:{stage}")
        return obs
    obs.status = "UNKNOWN"
    obs.notes.append(f"builder1_status:{status}")
    return obs


def _classify_builder2_video_job(
    job_id: str,
    raw: Dict[str, str],
    *,
    has_lease: bool,
) -> BuilderActivityObservation:
    status = str(raw.get("status") or "").strip().lower()
    builder = str(raw.get("builder") or "builder2").strip() or "builder2"
    obs = BuilderActivityObservation(
        kind="BUILDER2",
        job_id=job_id,
        builder=builder,
    )
    if not status:
        obs.status = "UNKNOWN"
        obs.notes.append("builder2_missing_status")
        return obs
    if status in BUILDER2_VIDEO_TERMINAL_STATUSES:
        obs.status = "IDLE"
        return obs
    if status == "queued":
        obs.status = "ACTIVE"
        obs.notes.append("builder2_queued")
        return obs
    if status == "running":
        obs.status = "ACTIVE"
        obs.notes.append("builder2_running")
        if has_lease:
            obs.notes.append("execution_lease_active")
        stage = str(raw.get("progressStage") or raw.get("progress_stage") or "").strip()
        if stage:
            obs.notes.append(f"progressStage:{stage}")
        return obs
    obs.status = "UNKNOWN"
    obs.notes.append(f"builder2_status:{status}")
    return obs


def _classify_builder1_campaign(campaign_id: str, raw: Dict[str, Any]) -> Optional[BuilderActivityObservation]:
    from engine.builder1_campaign_store import _lock_is_stale  # read-only stale helper

    if raw.get("generatingIndex") is None:
        return None
    if _lock_is_stale(raw):
        return BuilderActivityObservation(
            kind="BUILDER1",
            status="UNKNOWN",
            campaign_id=campaign_id,
            builder="builder1",
            notes=["builder1_campaign_lock_stale"],
        )
    obs = BuilderActivityObservation(
        kind="BUILDER1",
        status="ACTIVE",
        campaign_id=campaign_id,
        builder="builder1",
        notes=["builder1_campaign_generation_lock"],
    )
    owner = str(raw.get("generatingLockOwnerJobId") or "").strip()
    if owner:
        obs.job_id = owner
        obs.notes.append(f"lockOwnerJobId:{owner}")
    return obs


def _pid_alive_windows(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def _worker_registered_alive(ace_root: Optional[Path]) -> Tuple[Optional[bool], List[str]]:
    if ace_root is None:
        return None, []
    reg_path = ace_root / "_backups" / "local-dev" / "managed-processes.json"
    if not reg_path.is_file():
        return None, ["worker_registry_missing"]
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False, ["worker_registry_unreadable"]
    if not isinstance(data, dict):
        return False, ["worker_registry_invalid"]
    components = data.get("components")
    if not isinstance(components, dict):
        return None, ["worker_component_unregistered"]
    worker = components.get("worker")
    if not isinstance(worker, dict):
        return None, ["worker_component_unregistered"]
    pid_raw = worker.get("pid")
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        return False, ["worker_pid_missing"]
    if not _pid_alive_windows(pid):
        return False, ["worker_pid_not_alive"]
    return True, []


def _scan_memory_builder_activity(
    *,
    ace_root: Optional[Path],
    worker_reachable: Optional[bool],
) -> Dict[str, Any]:
    from engine import builder1_jobs_store as b1_jobs
    from engine import builder1_campaign_store as b1_campaign
    from engine import video_jobs_redis as video
    from engine.builder2_tournament_recovery import has_active_lease, _memory_leases, _use_memory_recovery

    observations: List[BuilderActivityObservation] = []
    notes: List[str] = ["backend:memory"]

    with b1_jobs._memory_lock:
        for jid, job in list(b1_jobs._memory_jobs.items()):
            obs = _classify_builder1_job(jid, job if isinstance(job, dict) else {})
            if obs.status != "IDLE":
                observations.append(obs)

    with b1_campaign._memory_lock:
        for cid, raw in list(b1_campaign._memory_campaigns.items()):
            if not isinstance(raw, dict):
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER1",
                        status="UNKNOWN",
                        campaign_id=cid,
                        builder="builder1",
                        notes=["builder1_campaign_corrupt"],
                    )
                )
                continue
            lock_obs = _classify_builder1_campaign(cid, raw)
            if lock_obs and lock_obs.status != "IDLE":
                observations.append(lock_obs)

    if video._use_memory_jobs:
        for jid, raw in list(video._memory_job_hashes.items()):
            if not isinstance(raw, dict):
                continue
            lease = has_active_lease(jid) if _use_memory_recovery else False
            obs = _classify_builder2_video_job(jid, raw, has_lease=lease)
            if obs.status != "IDLE":
                observations.append(obs)
    elif (os.environ.get("REDIS_URL") or "").strip():
        return _scan_redis_builder_activity(ace_root=ace_root, worker_reachable=worker_reachable)

    if _use_memory_recovery:
        now = int(time.time())
        for jid, lease in list(_memory_leases.items()):
            if int(lease.get("expiresAt") or 0) > now:
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER2_WORKER",
                        status="ACTIVE",
                        job_id=jid,
                        builder="builder2",
                        notes=["execution_lease_active"],
                    )
                )

    worker_state, worker_notes = _resolve_worker_reachability(
        ace_root=ace_root,
        worker_reachable=worker_reachable,
        observations=observations,
    )
    notes.extend(worker_notes)
    return {
        "observations": [_obs_to_dict(o) for o in observations],
        "redisReachable": True,
        "workerReachable": worker_state,
        "notes": notes,
    }


def _scan_redis_builder_activity(
    *,
    ace_root: Optional[Path],
    worker_reachable: Optional[bool],
) -> Dict[str, Any]:
    from engine.builder1_jobs_store import JOB_KEY_PREFIX as B1_JOB_PREFIX
    from engine.builder2_tournament_recovery import LEASE_KEY_PREFIX, has_active_lease
    from engine.video_jobs_redis import JOB_KEY_PREFIX, QUEUE_KEY, get_redis, redis_configured

    observations: List[BuilderActivityObservation] = []
    notes: List[str] = []

    if not redis_configured():
        return _scan_memory_builder_activity(ace_root=ace_root, worker_reachable=worker_reachable)

    try:
        r = get_redis()
        r.ping()
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER_REDIS_UNREACHABLE err=%s", exc)
        return {
            "observations": [
                _obs_to_dict(
                    BuilderActivityObservation(
                        kind="REDIS",
                        status="UNKNOWN",
                        notes=["redis_unreachable"],
                    )
                )
            ],
            "redisReachable": False,
            "workerReachable": None,
            "notes": ["redis_ping_failed"],
        }

    try:
        queue_depth = int(r.llen(QUEUE_KEY) or 0)
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER_REDIS_QUEUE_ERR err=%s", exc)
        return {
            "observations": [
                _obs_to_dict(
                    BuilderActivityObservation(kind="REDIS", status="UNKNOWN", notes=["redis_queue_unreadable"])
                )
            ],
            "redisReachable": False,
            "workerReachable": None,
            "notes": ["redis_queue_failed"],
        }

    if queue_depth > 0:
        observations.append(
            BuilderActivityObservation(
                kind="BUILDER2_QUEUE",
                status="ACTIVE",
                builder="builder2",
                notes=[f"queue_depth:{queue_depth}"],
            )
        )

    try:
        for key in r.scan_iter(match=f"{B1_JOB_PREFIX}*", count=64):
            job_id = str(key)[len(B1_JOB_PREFIX) :]
            raw = r.get(key)
            if not raw:
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER1",
                        status="UNKNOWN",
                        job_id=job_id,
                        builder="builder1",
                        notes=["builder1_job_key_missing_value"],
                    )
                )
                continue
            try:
                job = json.loads(raw)
            except json.JSONDecodeError:
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER1",
                        status="UNKNOWN",
                        job_id=job_id,
                        builder="builder1",
                        notes=["builder1_job_json_invalid"],
                    )
                )
                continue
            if not isinstance(job, dict):
                continue
            obs = _classify_builder1_job(job_id, job)
            if obs.status != "IDLE":
                observations.append(obs)
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER1_SCAN_ERR err=%s", exc)
        observations.append(
            BuilderActivityObservation(
                kind="REDIS",
                status="UNKNOWN",
                notes=["builder1_scan_failed"],
            )
        )

    try:
        for key in r.scan_iter(match=f"{BUILDER1_CAMPAIGN_PREFIX}*", count=64):
            campaign_id = str(key)[len(BUILDER1_CAMPAIGN_PREFIX) :]
            raw = r.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER1",
                        status="UNKNOWN",
                        campaign_id=campaign_id,
                        builder="builder1",
                        notes=["builder1_campaign_json_invalid"],
                    )
                )
                continue
            if isinstance(data, dict):
                lock_obs = _classify_builder1_campaign(campaign_id, data)
                if lock_obs and lock_obs.status != "IDLE":
                    observations.append(lock_obs)
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER1_CAMPAIGN_SCAN_ERR err=%s", exc)
        observations.append(
            BuilderActivityObservation(
                kind="BUILDER1",
                status="UNKNOWN",
                notes=["builder1_campaign_scan_failed"],
            )
        )

    active_job_ids: set[str] = set()
    try:
        for key in r.scan_iter(match=f"{JOB_KEY_PREFIX}*", count=64):
            job_id = str(key)[len(JOB_KEY_PREFIX) :]
            data = r.hgetall(key)
            if not data:
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER2",
                        status="UNKNOWN",
                        job_id=job_id,
                        builder="builder2",
                        notes=["builder2_job_hash_missing"],
                    )
                )
                continue
            lease = has_active_lease(job_id)
            obs = _classify_builder2_video_job(job_id, data, has_lease=lease)
            if obs.status != "IDLE":
                observations.append(obs)
                active_job_ids.add(job_id)
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER2_SCAN_ERR err=%s", exc)
        observations.append(
            BuilderActivityObservation(
                kind="REDIS",
                status="UNKNOWN",
                notes=["builder2_scan_failed"],
            )
        )

    try:
        for key in r.scan_iter(match=f"{LEASE_KEY_PREFIX}*", count=64):
            job_id = str(key)[len(LEASE_KEY_PREFIX) :]
            if job_id in active_job_ids:
                continue
            if has_active_lease(job_id):
                observations.append(
                    BuilderActivityObservation(
                        kind="BUILDER2_WORKER",
                        status="ACTIVE",
                        job_id=job_id,
                        builder="builder2",
                        notes=["execution_lease_active"],
                    )
                )
    except Exception as exc:
        logger.warning("SHUTDOWN_BUILDER2_LEASE_SCAN_ERR err=%s", exc)
        observations.append(
            BuilderActivityObservation(
                kind="BUILDER2_WORKER",
                status="UNKNOWN",
                notes=["builder2_lease_scan_failed"],
            )
        )

    worker_state, worker_notes = _resolve_worker_reachability(
        ace_root=ace_root,
        worker_reachable=worker_reachable,
        observations=observations,
    )
    notes.extend(worker_notes)

    return {
        "observations": [_obs_to_dict(o) for o in observations],
        "redisReachable": True,
        "workerReachable": worker_state,
        "notes": notes,
    }


def _builder_work_pending(observations: List[BuilderActivityObservation]) -> bool:
    for obs in observations:
        if obs.status == "ACTIVE" and obs.kind in (
            "BUILDER1",
            "BUILDER2",
            "BUILDER2_QUEUE",
            "BUILDER2_WORKER",
        ):
            return True
    return False


def _resolve_worker_reachability(
    *,
    ace_root: Optional[Path],
    worker_reachable: Optional[bool],
    observations: List[BuilderActivityObservation],
) -> Tuple[Optional[bool], List[str]]:
    notes: List[str] = []
    if not _builder_work_pending(observations):
        return worker_reachable, notes

    if worker_reachable is not None:
        if worker_reachable is False:
            notes.append("worker_unreachable_while_builder_work_pending")
        return worker_reachable, notes

    alive, reg_notes = _worker_registered_alive(ace_root)
    notes.extend(reg_notes)
    if alive is False:
        notes.append("worker_unreachable_while_builder_work_pending")
    return alive, notes


def observe_builder_shutdown_activity(
    *,
    ace_root: Optional[Path] = None,
    redis_reachable: Optional[bool] = None,
    worker_reachable: Optional[bool] = None,
    builder_activity_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns builderActivity block for shutdown gate scan.
    builder_activity_override (isolated tests): {observations, redisReachable, workerReachable, notes}
    """
    if builder_activity_override is not None:
        return {
            "observations": list(builder_activity_override.get("observations") or []),
            "redisReachable": builder_activity_override.get("redisReachable"),
            "workerReachable": builder_activity_override.get("workerReachable"),
            "notes": list(builder_activity_override.get("notes") or []),
        }

    if redis_reachable is False:
        return {
            "observations": [
                _obs_to_dict(
                    BuilderActivityObservation(kind="REDIS", status="UNKNOWN", notes=["redis_unreachable_injected"])
                )
            ],
            "redisReachable": False,
            "workerReachable": worker_reachable,
            "notes": ["redis_unreachable_injected"],
        }

    result = _scan_redis_builder_activity(ace_root=ace_root, worker_reachable=worker_reachable)
    if redis_reachable is True and result.get("redisReachable") is False:
        result["observations"] = [
            _obs_to_dict(
                BuilderActivityObservation(kind="REDIS", status="UNKNOWN", notes=["redis_expected_reachable"])
            )
        ]
        result["redisReachable"] = False
    return result


def merge_builder_observations_into_shutdown_scan(
    scan: Dict[str, Any],
    builder_scan: Dict[str, Any],
) -> None:
    base = list(scan.get("observations") or [])
    for row in builder_scan.get("observations") or []:
        if not isinstance(row, dict):
            continue
        base.append(
            {
                "kind": row.get("kind"),
                "sessionId": None,
                "operationId": row.get("jobId"),
                "status": row.get("status"),
                "currentStep": None,
                "cancelCritical": False,
                "publishExternalUnknown": False,
                "publishExternalRunning": False,
                "notes": list(row.get("notes") or []),
                "jobId": row.get("jobId"),
                "campaignId": row.get("campaignId"),
                "builder": row.get("builder"),
            }
        )
    scan["observations"] = base
    scan["builderActivity"] = builder_scan
    if builder_scan.get("redisReachable") is False:
        scan["redisReachable"] = False
    elif "redisReachable" not in scan:
        scan["redisReachable"] = builder_scan.get("redisReachable")
    scan["workerReachable"] = builder_scan.get("workerReachable")


__all__ = [
    "observe_builder_shutdown_activity",
    "merge_builder_observations_into_shutdown_scan",
    "BUILDER1_TERMINAL_STATUSES",
    "BUILDER2_VIDEO_TERMINAL_STATUSES",
]
