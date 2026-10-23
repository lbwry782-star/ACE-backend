"""Render Public API domain types for local publish infrastructure."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

RENDER_API_BASE_URL = "https://api.render.com/v1"

DEPLOY_STATUS_CREATED = "created"
DEPLOY_STATUS_QUEUED = "queued"
DEPLOY_STATUS_BUILD_IN_PROGRESS = "build_in_progress"
DEPLOY_STATUS_PRE_DEPLOY_IN_PROGRESS = "pre_deploy_in_progress"
DEPLOY_STATUS_UPDATE_IN_PROGRESS = "update_in_progress"
DEPLOY_STATUS_LIVE = "live"
DEPLOY_STATUS_DEACTIVATED = "deactivated"
DEPLOY_STATUS_BUILD_FAILED = "build_failed"
DEPLOY_STATUS_UPDATE_FAILED = "update_failed"
DEPLOY_STATUS_CANCELED = "canceled"
DEPLOY_STATUS_PRE_DEPLOY_FAILED = "pre_deploy_failed"

DEPLOY_IN_PROGRESS_STATUSES = frozenset(
    {
        DEPLOY_STATUS_CREATED,
        DEPLOY_STATUS_QUEUED,
        DEPLOY_STATUS_BUILD_IN_PROGRESS,
        DEPLOY_STATUS_PRE_DEPLOY_IN_PROGRESS,
        DEPLOY_STATUS_UPDATE_IN_PROGRESS,
    }
)

DEPLOY_TERMINAL_FAILURE_STATUSES = frozenset(
    {
        DEPLOY_STATUS_BUILD_FAILED,
        DEPLOY_STATUS_UPDATE_FAILED,
        DEPLOY_STATUS_PRE_DEPLOY_FAILED,
        DEPLOY_STATUS_CANCELED,
        DEPLOY_STATUS_DEACTIVATED,
    }
)

TRIGGER_NEW_COMMIT = "new_commit"
TRIGGER_API = "api"

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def normalize_commit_sha(value: str) -> str:
    """Require a full 40-char Git SHA; reject prefix-only matches."""
    token = (value or "").strip().lower()
    if not _FULL_SHA_RE.fullmatch(token):
        raise ValueError("ace_render_commit_sha_invalid")
    return token


def commit_sha_matches(actual: Optional[str], expected_full: str) -> bool:
    if not actual:
        return False
    try:
        return normalize_commit_sha(actual) == normalize_commit_sha(expected_full)
    except ValueError:
        return False


@dataclass(frozen=True)
class RenderCommit:
    id: str
    message: str = ""
    created_at: Optional[str] = None


@dataclass(frozen=True)
class RenderDeploy:
    id: str
    status: str
    commit: Optional[RenderCommit] = None
    trigger: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass(frozen=True)
class RenderService:
    id: str
    name: str = ""
    repo: str = ""
    branch: str = ""
    auto_deploy: str = ""
    auto_deploy_trigger: str = ""
    suspended: str = "not_suspended"
    service_type: str = ""


@dataclass
class RenderDeployWaitResult:
    deploy: RenderDeploy
    timed_out: bool = False


@dataclass
class RenderWebDeployPlan:
    action: str
    deploy: Optional[RenderDeploy] = None
    trigger_new: bool = False


@dataclass
class RenderWorkerDeployPlan:
    action: str
    deploy: Optional[RenderDeploy] = None
    observe_only: bool = True


@dataclass
class RenderPairVerification:
    ok: bool
    expected_sha: str
    web_deploy: Optional[RenderDeploy] = None
    worker_deploy: Optional[RenderDeploy] = None
    code: str = ""
    message: str = ""


def parse_commit(raw: Any) -> Optional[RenderCommit]:
    if not isinstance(raw, dict):
        return None
    commit_id = str(raw.get("id") or "").strip()
    if not commit_id:
        return None
    return RenderCommit(
        id=commit_id,
        message=str(raw.get("message") or ""),
        created_at=raw.get("createdAt"),
    )


def parse_deploy(raw: Any) -> RenderDeploy:
    if not isinstance(raw, dict):
        raise ValueError("ace_render_deploy_payload_invalid")
    deploy_id = str(raw.get("id") or "").strip()
    if not deploy_id:
        raise ValueError("ace_render_deploy_id_missing")
    commit = parse_commit(raw.get("commit"))
    return RenderDeploy(
        id=deploy_id,
        status=str(raw.get("status") or ""),
        commit=commit,
        trigger=str(raw.get("trigger") or ""),
        started_at=raw.get("startedAt"),
        finished_at=raw.get("finishedAt"),
        created_at=raw.get("createdAt"),
        updated_at=raw.get("updatedAt"),
    )


def parse_deploy_list(raw: Any) -> List[RenderDeploy]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("ace_render_deploy_list_invalid")
    out: List[RenderDeploy] = []
    for row in raw:
        if isinstance(row, dict) and "deploy" in row:
            out.append(parse_deploy(row.get("deploy")))
        elif isinstance(row, dict):
            out.append(parse_deploy(row))
        else:
            raise ValueError("ace_render_deploy_list_invalid")
    return out


def parse_service(raw: Any) -> RenderService:
    if not isinstance(raw, dict):
        raise ValueError("ace_render_service_payload_invalid")
    service_id = str(raw.get("id") or "").strip()
    if not service_id:
        raise ValueError("ace_render_service_id_missing")
    auto_trigger = str(raw.get("autoDeployTrigger") or "")
    auto_deploy = str(raw.get("autoDeploy") or "")
    service_details = raw.get("serviceDetails")
    service_type = ""
    if isinstance(service_details, dict):
        service_type = str(service_details.get("type") or "")
    return RenderService(
        id=service_id,
        name=str(raw.get("name") or ""),
        repo=str(raw.get("repo") or ""),
        branch=str(raw.get("branch") or ""),
        auto_deploy=auto_deploy,
        auto_deploy_trigger=auto_trigger,
        suspended=str(raw.get("suspended") or "not_suspended"),
        service_type=service_type,
    )


def deploy_commit_sha(deploy: RenderDeploy) -> Optional[str]:
    if deploy.commit is None:
        return None
    return deploy.commit.id
