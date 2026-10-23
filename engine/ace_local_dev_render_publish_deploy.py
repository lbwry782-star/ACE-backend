"""Render deploy orchestration for publish — Web trigger vs Worker auto-deploy observe."""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from engine.ace_local_dev_render_api_client import (
    RenderApiClient,
    find_in_progress_for_commit,
    find_live_for_commit,
)
from engine.ace_local_dev_render_api_errors import (
    RenderDeployFailedError,
    RenderDeployTimeoutError,
    RenderDeployVerifyError,
)
from engine.ace_local_dev_render_api_types import (
    DEPLOY_IN_PROGRESS_STATUSES,
    DEPLOY_STATUS_LIVE,
    DEPLOY_TERMINAL_FAILURE_STATUSES,
    TRIGGER_NEW_COMMIT,
    RenderDeploy,
    RenderPairVerification,
    RenderWebDeployPlan,
    RenderWorkerDeployPlan,
    RenderDeployWaitResult,
    commit_sha_matches,
    deploy_commit_sha,
    normalize_commit_sha,
)

ACE_RENDER_DEFAULT_WEB_SERVICE_ID = "srv-d4ev8fc9c44c73ckgmm0"
ACE_RENDER_DEFAULT_WORKER_SERVICE_ID = "srv-d78ornkjhs738i63eg"

ACTION_ALREADY_LIVE = "already_live"
ACTION_WAIT_IN_PROGRESS = "wait_in_progress"
ACTION_TRIGGER_DEPLOY = "trigger_deploy"
ACTION_OBSERVE_AUTO = "observe_auto_deploy"


def plan_web_deploy(client: RenderApiClient, service_id: str, commit_sha: str) -> RenderWebDeployPlan:
    expected = normalize_commit_sha(commit_sha)
    deploys = client.list_deploys(service_id, limit=30)
    live = find_live_for_commit(deploys, expected)
    if live is not None:
        return RenderWebDeployPlan(action=ACTION_ALREADY_LIVE, deploy=live, trigger_new=False)
    in_progress = find_in_progress_for_commit(deploys, expected)
    if in_progress is not None:
        return RenderWebDeployPlan(action=ACTION_WAIT_IN_PROGRESS, deploy=in_progress, trigger_new=False)
    return RenderWebDeployPlan(action=ACTION_TRIGGER_DEPLOY, deploy=None, trigger_new=True)


def execute_web_deploy(
    client: RenderApiClient,
    service_id: str,
    commit_sha: str,
    *,
    wait_deadline_seconds: float = 60.0,
    poll_interval_seconds: float = 0.01,
    clock: Optional[Callable[[], float]] = None,
    sleep_fn=time.sleep,
) -> RenderDeploy:
    plan = plan_web_deploy(client, service_id, commit_sha)
    deploy: RenderDeploy
    if plan.action == ACTION_ALREADY_LIVE:
        assert plan.deploy is not None
        return plan.deploy
    if plan.action == ACTION_WAIT_IN_PROGRESS:
        assert plan.deploy is not None
        deploy = plan.deploy
    else:
        deploy = client.trigger_deploy(service_id, commit_id=commit_sha)
    wait = client.wait_for_deploy(
        service_id,
        deploy.id,
        deadline_seconds=wait_deadline_seconds,
        poll_interval_seconds=poll_interval_seconds,
        clock=clock,
        sleep_fn=sleep_fn,
    )
    if wait.timed_out:
        raise RenderDeployTimeoutError(wait.deploy.id)
    if wait.deploy.status in DEPLOY_TERMINAL_FAILURE_STATUSES:
        raise RenderDeployFailedError(wait.deploy.id, wait.deploy.status)
    if wait.deploy.status != DEPLOY_STATUS_LIVE:
        raise RenderDeployFailedError(wait.deploy.id, wait.deploy.status)
    return wait.deploy


def plan_worker_deploy(
    client: RenderApiClient,
    service_id: str,
    commit_sha: str,
    *,
    auto_deploy_on_commit: bool = True,
) -> RenderWorkerDeployPlan:
    if not auto_deploy_on_commit:
        raise RenderDeployVerifyError(
            "ace_render_worker_manual_only_not_supported",
            message="Worker auto-deploy observe mode required for publish.",
        )
    expected = normalize_commit_sha(commit_sha)
    deploys = client.list_deploys(service_id, limit=30)
    live = find_live_for_commit(deploys, expected)
    if live is not None:
        return RenderWorkerDeployPlan(action=ACTION_ALREADY_LIVE, deploy=live, observe_only=True)
    in_progress = find_in_progress_for_commit(deploys, expected)
    if in_progress is not None:
        return RenderWorkerDeployPlan(action=ACTION_WAIT_IN_PROGRESS, deploy=in_progress, observe_only=True)
    auto = _find_auto_deploy_for_commit(deploys, expected)
    if auto is not None:
        return RenderWorkerDeployPlan(action=ACTION_OBSERVE_AUTO, deploy=auto, observe_only=True)
    return RenderWorkerDeployPlan(action=ACTION_WAIT_IN_PROGRESS, deploy=None, observe_only=True)


def _find_auto_deploy_for_commit(deploys: List[RenderDeploy], expected: str) -> Optional[RenderDeploy]:
    for deploy in deploys:
        sha = deploy_commit_sha(deploy)
        if not sha or not commit_sha_matches(sha, expected):
            continue
        if deploy.trigger == TRIGGER_NEW_COMMIT:
            return deploy
    return None


def execute_worker_auto_deploy(
    client: RenderApiClient,
    service_id: str,
    commit_sha: str,
    *,
    auto_deploy_on_commit: bool = True,
    wait_deadline_seconds: float = 60.0,
    poll_interval_seconds: float = 0.01,
    clock: Optional[Callable[[], float]] = None,
    sleep_fn=time.sleep,
) -> RenderDeploy:
    expected = normalize_commit_sha(commit_sha)
    if clock is None:
        clock = time.monotonic
    start = clock()
    deploy: Optional[RenderDeploy] = None
    while True:
        plan = plan_worker_deploy(
            client,
            service_id,
            expected,
            auto_deploy_on_commit=auto_deploy_on_commit,
        )
        if plan.action == ACTION_ALREADY_LIVE and plan.deploy is not None:
            return plan.deploy
        if plan.deploy is not None:
            deploy = plan.deploy
            wait = client.wait_for_deploy(
                service_id,
                deploy.id,
                deadline_seconds=max(0.0, wait_deadline_seconds - (clock() - start)),
                poll_interval_seconds=poll_interval_seconds,
                clock=clock,
                sleep_fn=sleep_fn,
            )
            if wait.timed_out:
                raise RenderDeployTimeoutError(wait.deploy.id)
            if wait.deploy.status in DEPLOY_TERMINAL_FAILURE_STATUSES:
                raise RenderDeployFailedError(wait.deploy.id, wait.deploy.status)
            if wait.deploy.status == DEPLOY_STATUS_LIVE:
                return wait.deploy
            deploy = wait.deploy
        if clock() - start >= wait_deadline_seconds:
            did = deploy.id if deploy else "unknown"
            raise RenderDeployTimeoutError(did)
        sleep_fn(poll_interval_seconds)


def verify_service_active_commit(
    deploys: List[RenderDeploy],
    expected_sha: str,
) -> RenderDeploy:
    expected = normalize_commit_sha(expected_sha)
    if not deploys:
        raise RenderDeployVerifyError(
            "ace_render_verify_no_deploys",
            message="No deploy history available for verification.",
        )
    live_idx = None
    live_deploy: Optional[RenderDeploy] = None
    for idx, deploy in enumerate(deploys):
        if deploy.status == DEPLOY_STATUS_LIVE:
            live_idx = idx
            live_deploy = deploy
            break
    if live_deploy is None or live_idx is None:
        raise RenderDeployVerifyError(
            "ace_render_verify_not_live",
            message="No live deploy found for service.",
        )
    sha = deploy_commit_sha(live_deploy)
    if not sha or not commit_sha_matches(sha, expected):
        raise RenderDeployVerifyError(
            "ace_render_verify_wrong_sha",
            message="Live deploy commit does not match expected product SHA.",
        )
    for newer in deploys[:live_idx]:
        if newer.status in DEPLOY_IN_PROGRESS_STATUSES:
            raise RenderDeployVerifyError(
                "ace_render_verify_competing_deploy",
                message="A newer deploy is still in progress.",
            )
        if newer.status == DEPLOY_STATUS_LIVE:
            newer_sha = deploy_commit_sha(newer)
            if newer_sha and not commit_sha_matches(newer_sha, expected):
                raise RenderDeployVerifyError(
                    "ace_render_verify_competing_live",
                    message="A newer live deploy exists with a different commit.",
                )
    return live_deploy


def verify_web_and_worker_pair(
    client: RenderApiClient,
    *,
    web_service_id: str,
    worker_service_id: str,
    expected_sha: str,
) -> RenderPairVerification:
    expected = normalize_commit_sha(expected_sha)
    web_deploys = client.list_deploys(web_service_id, limit=30)
    worker_deploys = client.list_deploys(worker_service_id, limit=30)
    try:
        web = verify_service_active_commit(web_deploys, expected)
        worker = verify_service_active_commit(worker_deploys, expected)
    except RenderDeployVerifyError as exc:
        return RenderPairVerification(
            ok=False,
            expected_sha=expected,
            code=exc.code,
            message=exc.message,
        )
    web_sha = deploy_commit_sha(web)
    worker_sha = deploy_commit_sha(worker)
    if not web_sha or not worker_sha:
        return RenderPairVerification(
            ok=False,
            expected_sha=expected,
            code="ace_render_verify_missing_commit",
            message="Deploy is live but commit metadata is missing.",
        )
    if normalize_commit_sha(web_sha) != normalize_commit_sha(worker_sha):
        return RenderPairVerification(
            ok=False,
            expected_sha=expected,
            web_deploy=web,
            worker_deploy=worker,
            code="ace_render_verify_pair_mismatch",
            message="Web and Worker live commits differ.",
        )
    return RenderPairVerification(
        ok=True,
        expected_sha=expected,
        web_deploy=web,
        worker_deploy=worker,
        code="ace_render_verify_ok",
        message="Web and Worker are live on the expected commit.",
    )
