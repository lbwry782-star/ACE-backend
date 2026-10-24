"""RenderWebDeployPort and RenderWorkerObservePort — mock transport only (isolated tests)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from engine.ace_local_dev_publish_candidate_registry import normalize_full_sha
from engine.ace_local_dev_publish_real_execute_ports import RenderDeployOutcome
from engine.ace_local_dev_publish_render_sim_execute_adapters import ACE_PUBLISH_RENDER_SIM_MOCK_API_KEY
from engine.ace_local_dev_publish_render_sim_transport import (
    PublishRenderSimState,
    build_publish_sim_mock_transport,
    deploy_row,
)
from engine.ace_local_dev_render_api_client import RenderApiClient
from engine.ace_local_dev_render_api_errors import (
    RenderApiError,
    RenderDeployFailedError,
    RenderDeployTimeoutError,
)
from engine.ace_local_dev_render_api_transport import MockRenderApiTransport, UrllibRenderApiTransport
from engine.ace_local_dev_render_api_types import DEPLOY_STATUS_LIVE, deploy_commit_sha
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
    execute_web_deploy,
    execute_worker_auto_deploy,
    find_live_for_commit,
)

_RENDER_TEST_SERVICE_TOKEN = "ace-publish-render-test"
_FORBIDDEN_PRODUCTION_SERVICE_IDS = frozenset(
    {
        ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
        ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
    }
)


class RenderMockPortError(Exception):
    def __init__(self, code: str, *, message: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


@dataclass(frozen=True)
class RenderMockDeployPortConfig:
    web_service_id: str
    worker_service_id: str
    mock_api_key: str = ACE_PUBLISH_RENDER_SIM_MOCK_API_KEY
    wait_deadline_seconds: float = 30.0
    poll_interval_seconds: float = 0.01


def assert_render_mock_service_id(service_id: str) -> None:
    sid = (service_id or "").strip()
    if not sid:
        raise RenderMockPortError("ace_publish_real_render_service_id_missing")
    if sid in _FORBIDDEN_PRODUCTION_SERVICE_IDS:
        raise RenderMockPortError(
            "ace_publish_real_render_production_service_id_forbidden",
            message="Production Render service IDs are not allowed in mock ports.",
        )
    if _RENDER_TEST_SERVICE_TOKEN not in sid.lower():
        raise RenderMockPortError(
            "ace_publish_real_render_service_id_not_test",
            message="Service ID must include ace-publish-render-test.",
        )


def assert_render_mock_port_config(config: RenderMockDeployPortConfig) -> None:
    assert_render_mock_service_id(config.web_service_id)
    assert_render_mock_service_id(config.worker_service_id)
    key = (config.mock_api_key or "").strip()
    if not key or key != ACE_PUBLISH_RENDER_SIM_MOCK_API_KEY:
        raise RenderMockPortError(
            "ace_publish_real_render_api_key_not_mock",
            message="Only the publish simulation mock API key is allowed.",
        )


def assert_mock_render_transport(transport) -> None:
    if isinstance(transport, UrllibRenderApiTransport):
        raise RenderMockPortError(
            "ace_publish_real_render_urllib_transport_forbidden",
            message="HTTP Render transport is not allowed.",
        )
    if not isinstance(transport, MockRenderApiTransport):
        raise RenderMockPortError(
            "ace_publish_real_render_transport_not_mock",
            message="Render transport must be MockRenderApiTransport.",
        )


def _normalize_candidate(candidate_sha: str) -> str:
    normalized = normalize_full_sha(candidate_sha)
    if not normalized:
        raise RenderMockPortError(
            "ace_publish_real_render_candidate_invalid",
            message="Candidate SHA must be a full 40-character commit hash.",
        )
    return normalized


@dataclass
class RenderMockPortBundle:
    config: RenderMockDeployPortConfig
    state: PublishRenderSimState
    transport: MockRenderApiTransport
    client: RenderApiClient
    web_port: "MockRenderWebDeployPort"
    worker_port: "MockRenderWorkerObservePort"


def build_render_mock_port_bundle(
    config: RenderMockDeployPortConfig,
) -> RenderMockPortBundle:
    assert_render_mock_port_config(config)
    state = PublishRenderSimState(
        web_service_id=config.web_service_id,
        worker_service_id=config.worker_service_id,
    )
    transport = build_publish_sim_mock_transport(
        state,
        web_service_id=config.web_service_id,
        worker_service_id=config.worker_service_id,
    )
    assert_mock_render_transport(transport)
    client = RenderApiClient(
        transport,
        api_key=config.mock_api_key,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    web_port = MockRenderWebDeployPort(client, config, state, transport=transport)
    worker_port = MockRenderWorkerObservePort(client, config, state, transport=transport)
    return RenderMockPortBundle(
        config=config,
        state=state,
        transport=transport,
        client=client,
        web_port=web_port,
        worker_port=worker_port,
    )


class MockRenderWebDeployPort:
    """RenderWebDeployPort using execute_web_deploy + mock transport only."""

    def __init__(
        self,
        client: RenderApiClient,
        config: RenderMockDeployPortConfig,
        state: PublishRenderSimState,
        *,
        transport: MockRenderApiTransport,
    ) -> None:
        assert_render_mock_port_config(config)
        assert_mock_render_transport(transport)
        self._client = client
        self._config = config
        self._state = state

    def wait_for_web_live(self, *, candidate_sha: str) -> RenderDeployOutcome:
        try:
            candidate = _normalize_candidate(candidate_sha)
        except RenderMockPortError as exc:
            return RenderDeployOutcome(ok=False, code=exc.code, message=exc.message)

        if self._state.web_outcome_unknown:
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_web_unknown_no_retry",
                message="Previous web deploy outcome unknown; retry blocked.",
                commit_sha=candidate,
            )

        deploys = self._client.list_deploys(self._config.web_service_id, limit=30)
        live = find_live_for_commit(deploys, candidate)
        if live is not None:
            return RenderDeployOutcome(
                ok=True,
                code="ace_publish_real_render_web_already_live",
                message="Web service already live on candidate SHA.",
                deploy_id=live.id,
                commit_sha=deploy_commit_sha(live) or candidate,
                already_live=True,
            )

        posts_before = int(self._state.post_count.get(self._config.web_service_id, 0))
        try:
            deploy = execute_web_deploy(
                self._client,
                self._config.web_service_id,
                candidate,
                wait_deadline_seconds=self._config.wait_deadline_seconds,
                poll_interval_seconds=self._config.poll_interval_seconds,
            )
        except RenderDeployTimeoutError as exc:
            self._state.web_outcome_unknown = True
            deploy_id = str((getattr(exc, "details", None) or {}).get("deployId") or "")
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_web_timeout",
                message="Web deploy wait timed out.",
                deploy_id=deploy_id,
                commit_sha=candidate,
            )
        except RenderDeployFailedError as exc:
            deploy_id = str((getattr(exc, "details", None) or {}).get("deployId") or "")
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_web_failed",
                message=getattr(exc, "message", str(exc)),
                deploy_id=deploy_id,
                commit_sha=candidate,
            )
        except RenderApiError as exc:
            status = deploy_commit_sha(getattr(exc, "deploy", None)) if hasattr(exc, "deploy") else ""
            if getattr(self._state, "web_unknown_status", False):
                self._state.web_outcome_unknown = True
                return RenderDeployOutcome(
                    ok=False,
                    code="ace_publish_real_render_web_unknown",
                    message="Web deploy ended in unknown state.",
                    commit_sha=candidate or status,
                )
            return RenderDeployOutcome(
                ok=False,
                code=getattr(exc, "code", "ace_publish_real_render_web_error"),
                message=getattr(exc, "message", str(exc)),
                commit_sha=candidate,
            )

        posts_after = int(self._state.post_count.get(self._config.web_service_id, 0))
        triggered = posts_after > posts_before
        commit = deploy_commit_sha(deploy) or candidate
        return RenderDeployOutcome(
            ok=True,
            code="ace_publish_real_render_web_ok",
            message="Web deploy live on candidate SHA.",
            deploy_id=deploy.id,
            commit_sha=commit,
            already_live=not triggered and posts_before > 0,
        )


class MockRenderWorkerObservePort:
    """RenderWorkerObservePort — observe auto-deploy only (no manual trigger)."""

    def __init__(
        self,
        client: RenderApiClient,
        config: RenderMockDeployPortConfig,
        state: PublishRenderSimState,
        *,
        transport: MockRenderApiTransport,
    ) -> None:
        assert_render_mock_port_config(config)
        assert_mock_render_transport(transport)
        self._client = client
        self._config = config
        self._state = state

    def wait_for_worker_live(self, *, candidate_sha: str) -> RenderDeployOutcome:
        try:
            candidate = _normalize_candidate(candidate_sha)
        except RenderMockPortError as exc:
            return RenderDeployOutcome(ok=False, code=exc.code, message=exc.message)

        if self._state.pushed_sha and self._state.pushed_sha != candidate:
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_worker_sha_mismatch",
                message="Worker observe SHA does not match last simulated git push.",
                commit_sha=candidate,
            )

        worker_posts = int(self._state.post_count.get(self._config.worker_service_id, 0))
        if worker_posts > 0:
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_worker_manual_deploy_forbidden",
                message="Worker manual deploy is not allowed.",
                commit_sha=candidate,
            )

        deploys = self._client.list_deploys(self._config.worker_service_id, limit=30)
        live = find_live_for_commit(deploys, candidate)
        if live is not None:
            return RenderDeployOutcome(
                ok=True,
                code="ace_publish_real_render_worker_already_live",
                message="Worker already live on candidate SHA.",
                deploy_id=live.id,
                commit_sha=deploy_commit_sha(live) or candidate,
                already_live=True,
                observed_only=True,
            )

        if not self._state.pushed_sha:
            self._state.on_git_push(candidate)

        try:
            deploy = execute_worker_auto_deploy(
                self._client,
                self._config.worker_service_id,
                candidate,
                auto_deploy_on_commit=True,
                wait_deadline_seconds=self._config.wait_deadline_seconds,
                poll_interval_seconds=self._config.poll_interval_seconds,
            )
        except RenderDeployTimeoutError as exc:
            deploy_id = str((getattr(exc, "details", None) or {}).get("deployId") or "")
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_worker_timeout",
                message="Worker auto-deploy observe timed out.",
                deploy_id=deploy_id,
                commit_sha=candidate,
            )
        except RenderDeployFailedError as exc:
            deploy_id = str((getattr(exc, "details", None) or {}).get("deployId") or "")
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_worker_failed",
                message=getattr(exc, "message", str(exc)),
                deploy_id=deploy_id,
                commit_sha=candidate,
            )
        except RenderApiError as exc:
            return RenderDeployOutcome(
                ok=False,
                code=getattr(exc, "code", "ace_publish_real_render_worker_error"),
                message=getattr(exc, "message", str(exc)),
                commit_sha=candidate,
            )

        commit = deploy_commit_sha(deploy) or candidate
        if not commit.startswith(candidate[:7]) and candidate not in commit:
            return RenderDeployOutcome(
                ok=False,
                code="ace_publish_real_render_worker_wrong_sha",
                message="Worker live deploy commit does not match candidate.",
                deploy_id=deploy.id,
                commit_sha=commit,
            )
        return RenderDeployOutcome(
            ok=True,
            code="ace_publish_real_render_worker_ok",
            message="Worker auto-deploy observed successfully.",
            deploy_id=deploy.id,
            commit_sha=commit,
            observed_only=True,
        )
