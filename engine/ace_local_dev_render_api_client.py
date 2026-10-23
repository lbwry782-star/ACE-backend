"""Render Public API client — logic separated from transport."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_render_api_errors import (
    RenderApiConflictError,
    RenderApiError,
    RenderApiForbiddenError,
    RenderApiMissingApiKeyError,
    RenderApiNotFoundError,
    RenderApiParseError,
    RenderApiRateLimitError,
    RenderApiUnauthorizedError,
    RenderApiUnavailableError,
)
from engine.ace_local_dev_render_api_transport import (
    RenderApiResponse,
    RenderApiTransport,
    build_render_api_url,
    redact_sensitive_text,
)
from engine.ace_local_dev_render_api_types import (
    DEPLOY_IN_PROGRESS_STATUSES,
    DEPLOY_STATUS_LIVE,
    DEPLOY_TERMINAL_FAILURE_STATUSES,
    RenderDeploy,
    RenderDeployWaitResult,
    RenderService,
    commit_sha_matches,
    normalize_commit_sha,
    parse_deploy,
    parse_deploy_list,
    parse_service,
)

ENV_RENDER_API_KEY = "ACE_RENDER_API_KEY"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.05


class RenderApiClient:
    def __init__(
        self,
        transport: RenderApiTransport,
        *,
        api_key: Optional[str] = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._transport = transport
        self._api_key = (api_key if api_key is not None else os.environ.get(ENV_RENDER_API_KEY) or "").strip()
        self._request_timeout_seconds = request_timeout_seconds
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def require_api_key(self) -> str:
        if not self._api_key:
            raise RenderApiMissingApiKeyError()
        return self._api_key

    def get_service(self, service_id: str) -> RenderService:
        body = self._request_json("GET", f"/services/{service_id}")
        return parse_service(body)

    def list_deploys(
        self,
        service_id: str,
        *,
        limit: int = 20,
        status: Optional[List[str]] = None,
    ) -> List[RenderDeploy]:
        query: Dict[str, str] = {"limit": str(limit)}
        if status:
            for idx, st in enumerate(status):
                query[f"status[{idx}]"] = st
        body = self._request_json("GET", f"/services/{service_id}/deploys", query=query)
        return parse_deploy_list(body)

    def get_deploy(self, service_id: str, deploy_id: str) -> RenderDeploy:
        body = self._request_json("GET", f"/services/{service_id}/deploys/{deploy_id}")
        return parse_deploy(body)

    def trigger_deploy(
        self,
        service_id: str,
        *,
        commit_id: Optional[str] = None,
        clear_cache: bool = False,
    ) -> RenderDeploy:
        payload: Dict[str, Any] = {}
        if commit_id is not None:
            payload["commitId"] = normalize_commit_sha(commit_id)
        if clear_cache:
            payload["clearCache"] = "clear"
        body = self._request_json("POST", f"/services/{service_id}/deploys", json_body=payload or {})
        return parse_deploy(body)

    def wait_for_deploy(
        self,
        service_id: str,
        deploy_id: str,
        *,
        deadline_seconds: float,
        poll_interval_seconds: float = 0.01,
        clock: Optional[Any] = None,
        sleep_fn=time.sleep,
    ) -> RenderDeployWaitResult:
        if clock is None:
            clock = time.monotonic
        start = clock()
        last: Optional[RenderDeploy] = None
        while True:
            last = self.get_deploy(service_id, deploy_id)
            if last.status == DEPLOY_STATUS_LIVE:
                return RenderDeployWaitResult(deploy=last, timed_out=False)
            if last.status in DEPLOY_TERMINAL_FAILURE_STATUSES:
                return RenderDeployWaitResult(deploy=last, timed_out=False)
            if clock() - start >= deadline_seconds:
                return RenderDeployWaitResult(deploy=last, timed_out=True)
            sleep_fn(poll_interval_seconds)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = build_render_api_url(path, query)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.require_api_key()}",
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._transport.send(
                    method,
                    url,
                    headers=headers,
                    json_body=json_body,
                    timeout_seconds=self._request_timeout_seconds,
                )
            except TimeoutError as exc:
                if attempt > self._max_retries:
                    raise RenderApiError(
                        "ace_render_api_timeout",
                        message="Render API request timed out.",
                    ) from exc
                self._backoff(attempt)
                continue

            if response.status_code in (429, 503):
                if attempt > self._max_retries:
                    self._raise_for_status(response)
                retry_after = self._parse_retry_after(response.headers)
                if retry_after is not None:
                    time.sleep(min(retry_after, 1.0))
                else:
                    self._backoff(attempt)
                continue

            if response.status_code >= 400:
                self._raise_for_status(response)

            return self._parse_json_body(response)

    def _backoff(self, attempt: int) -> None:
        time.sleep(self._retry_backoff_seconds * attempt)

    @staticmethod
    def _parse_retry_after(headers: Any) -> Optional[int]:
        if not headers:
            return None
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except ValueError:
            return None

    @staticmethod
    def _parse_json_body(response: RenderApiResponse) -> Any:
        if response.body is not None and not isinstance(response.body, (str, bytes)):
            return response.body
        text = response.raw_text or ""
        if isinstance(response.body, str):
            text = response.body
        text = text.strip()
        if not text:
            raise RenderApiParseError("ace_render_api_empty_body")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RenderApiParseError("ace_render_api_json_invalid") from exc

    @staticmethod
    def _raise_for_status(response: RenderApiResponse) -> None:
        message = RenderApiClient._extract_error_message(response.body)
        message = redact_sensitive_text(message)
        status = response.status_code
        if status == 401:
            raise RenderApiUnauthorizedError(message or "ace_render_api_unauthorized")
        if status == 403:
            raise RenderApiForbiddenError(message or "ace_render_api_forbidden")
        if status == 404:
            raise RenderApiNotFoundError(message or "ace_render_api_not_found")
        if status == 409:
            raise RenderApiConflictError(message or "ace_render_api_conflict")
        if status == 429:
            raise RenderApiRateLimitError(
                retry_after_seconds=RenderApiClient._parse_retry_after(response.headers),
                message=message or "ace_render_api_rate_limited",
            )
        if status == 503:
            raise RenderApiUnavailableError(message or "ace_render_api_unavailable")
        raise RenderApiError(
            "ace_render_api_http_error",
            message=message or f"HTTP {status}",
            http_status=status,
        )

    @staticmethod
    def _extract_error_message(body: Any) -> str:
        if isinstance(body, dict):
            return str(body.get("message") or body.get("code") or "")
        return ""


def find_deploy_for_commit(deploys: List[RenderDeploy], commit_sha: str) -> List[RenderDeploy]:
    expected = normalize_commit_sha(commit_sha)
    matches: List[RenderDeploy] = []
    for deploy in deploys:
        commit = deploy.commit.id if deploy.commit else ""
        if commit and commit_sha_matches(commit, expected):
            matches.append(deploy)
    return matches


def select_newest(deploys: List[RenderDeploy]) -> Optional[RenderDeploy]:
    if not deploys:
        return None
    return deploys[0]


def find_in_progress_for_commit(deploys: List[RenderDeploy], commit_sha: str) -> Optional[RenderDeploy]:
    for deploy in find_deploy_for_commit(deploys, commit_sha):
        if deploy.status in DEPLOY_IN_PROGRESS_STATUSES:
            return deploy
    return None


def find_live_for_commit(deploys: List[RenderDeploy], commit_sha: str) -> Optional[RenderDeploy]:
    for deploy in find_deploy_for_commit(deploys, commit_sha):
        if deploy.status == DEPLOY_STATUS_LIVE:
            return deploy
    return None
