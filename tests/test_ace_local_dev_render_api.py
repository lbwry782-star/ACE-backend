"""
Render API local publish infrastructure — mock transport only (no network).

Run: python -m unittest tests.test_ace_local_dev_render_api -v
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from engine.ace_local_dev_render_api_client import RenderApiClient
from engine.ace_local_dev_render_api_errors import (
    RenderApiMissingApiKeyError,
    RenderApiParseError,
    RenderApiRateLimitError,
    RenderApiUnauthorizedError,
    RenderDeployFailedError,
    RenderDeployTimeoutError,
    RenderDeployVerifyError,
)
from engine.ace_local_dev_render_api_transport import MockRenderApiTransport, RenderApiResponse
from engine.ace_local_dev_render_api_types import (
    DEPLOY_STATUS_BUILD_FAILED,
    DEPLOY_STATUS_BUILD_IN_PROGRESS,
    DEPLOY_STATUS_LIVE,
    DEPLOY_STATUS_QUEUED,
    TRIGGER_API,
    TRIGGER_NEW_COMMIT,
    normalize_commit_sha,
)
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
    ACTION_ALREADY_LIVE,
    ACTION_TRIGGER_DEPLOY,
    execute_web_deploy,
    execute_worker_auto_deploy,
    plan_web_deploy,
    verify_service_active_commit,
    verify_web_and_worker_pair,
)

TARGET = "258337ca9fbc01553b11adbb49f528e1ed059a58"
OTHER = "fadbddd4abb672fd2f4c436f2afddeeff5ea2f67"
WEB = ACE_RENDER_DEFAULT_WEB_SERVICE_ID
WORKER = ACE_RENDER_DEFAULT_WORKER_SERVICE_ID


def _deploy(
    deploy_id: str,
    status: str,
    *,
    commit: str = TARGET,
    trigger: str = TRIGGER_API,
) -> Dict[str, Any]:
    return {
        "id": deploy_id,
        "status": status,
        "commit": {"id": commit, "message": "test"},
        "trigger": trigger,
        "createdAt": "2026-10-24T00:00:00Z",
    }


def _wrap_list(*deploys: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"deploy": d, "cursor": f"c-{d['id']}"} for d in deploys]


class RenderMockState:
    def __init__(self) -> None:
        self.deploys: Dict[str, Dict[str, Dict[str, Any]]] = {WEB: {}, WORKER: {}}
        self.post_count: Dict[str, int] = {WEB: 0, WORKER: 0}
        self.rate_limit_hits = 0

    def seed(self, service_id: str, *rows: Dict[str, Any]) -> None:
        for row in rows:
            self.deploys[service_id][row["id"]] = dict(row)

    def list_payload(self, service_id: str) -> List[Dict[str, Any]]:
        rows = list(self.deploys[service_id].values())
        rows.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
        return _wrap_list(*rows)

    def set_status(self, service_id: str, deploy_id: str, status: str) -> None:
        self.deploys[service_id][deploy_id]["status"] = status


def _service_id_from_url(url: str) -> str:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    idx = parts.index("services")
    return parts[idx + 1]


def _route(state: RenderMockState):
    def handler(method: str, url: str, *, headers, json_body=None) -> Optional[RenderApiResponse]:
        auth = headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return RenderApiResponse(status_code=401, headers={}, body={"message": "missing auth"})

        path = urlparse(url).path
        if method == "GET" and "/deploys/" in path:
            service_id = _service_id_from_url(url)
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            row = state.deploys.get(service_id, {}).get(deploy_id)
            if not row:
                return RenderApiResponse(status_code=404, headers={}, body={"message": "not found"})
            return RenderApiResponse(status_code=200, headers={}, body=row)

        if method == "GET" and path.endswith("/deploys"):
            service_id = _service_id_from_url(url)
            return RenderApiResponse(status_code=200, headers={}, body=state.list_payload(service_id))

        if method == "POST" and path.endswith("/deploys"):
            service_id = _service_id_from_url(url)
            state.post_count[service_id] = state.post_count.get(service_id, 0) + 1
            commit = (json_body or {}).get("commitId", TARGET)
            deploy_id = f"dep-{service_id}-{state.post_count[service_id]}"
            row = _deploy(deploy_id, DEPLOY_STATUS_QUEUED, commit=commit, trigger=TRIGGER_API)
            state.deploys.setdefault(service_id, {})[deploy_id] = row
            return RenderApiResponse(status_code=201, headers={}, body=row)

        if method == "GET" and "/services/" in path and not path.endswith("/deploys"):
            service_id = _service_id_from_url(url)
            return RenderApiResponse(
                status_code=200,
                headers={},
                body={
                    "id": service_id,
                    "name": "ace-backend",
                    "branch": "main",
                    "autoDeployTrigger": "off" if service_id == WEB else "commit",
                },
            )
        return None

    return handler


class RenderApiTests(unittest.TestCase):
    def _client(self, state: RenderMockState, *, api_key: str = "test-key-mock") -> RenderApiClient:
        transport = MockRenderApiTransport()
        transport.set_default_handler(_route(state))
        return RenderApiClient(
            transport,
            api_key=api_key,
            max_retries=2,
            retry_backoff_seconds=0.0,
        )

    def test_web_deploy_success(self) -> None:
        state = RenderMockState()
        client = self._client(state)

        def advance(method, url, *, headers, json_body=None):
            path = urlparse(url).path
            if method == "GET" and "/deploys/dep-" in path:
                deploy_id = path.rstrip("/").split("/deploys/")[-1]
                current = state.deploys[WEB][deploy_id]["status"]
                if current == DEPLOY_STATUS_QUEUED:
                    state.set_status(WEB, deploy_id, DEPLOY_STATUS_BUILD_IN_PROGRESS)
                else:
                    state.set_status(WEB, deploy_id, DEPLOY_STATUS_LIVE)
            return None

        client._transport.add_handler(advance)  # type: ignore[attr-defined]

        result = execute_web_deploy(
            client,
            WEB,
            TARGET,
            wait_deadline_seconds=5.0,
            poll_interval_seconds=0.0,
        )
        self.assertEqual(result.status, DEPLOY_STATUS_LIVE)
        self.assertEqual(state.post_count[WEB], 1)
        self.assertTrue(result.commit)
        self.assertEqual(normalize_commit_sha(result.commit.id), TARGET)

    def test_worker_auto_deploy_success(self) -> None:
        state = RenderMockState()
        state.seed(
            WORKER,
            _deploy("w-auto-1", DEPLOY_STATUS_BUILD_IN_PROGRESS, trigger=TRIGGER_NEW_COMMIT),
        )
        client = self._client(state)
        clock = {"t": 0.0}

        def fake_clock() -> float:
            return clock["t"]

        def fake_sleep(_: float) -> None:
            clock["t"] += 0.5
            state.set_status(WORKER, "w-auto-1", DEPLOY_STATUS_LIVE)

        result = execute_worker_auto_deploy(
            client,
            WORKER,
            TARGET,
            wait_deadline_seconds=5.0,
            clock=fake_clock,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(result.status, DEPLOY_STATUS_LIVE)
        self.assertEqual(state.post_count[WORKER], 0)

    def test_deploy_already_in_progress_web_waits(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("web-ip-1", DEPLOY_STATUS_BUILD_IN_PROGRESS))
        client = self._client(state)
        plan = plan_web_deploy(client, WEB, TARGET)
        self.assertEqual(plan.action, "wait_in_progress")

        def on_get(_m, url, *, headers, json_body=None):
            if "/deploys/web-ip-1" in url:
                state.set_status(WEB, "web-ip-1", DEPLOY_STATUS_LIVE)
            return None

        client._transport.add_handler(on_get)  # type: ignore[attr-defined]
        done = execute_web_deploy(client, WEB, TARGET, wait_deadline_seconds=3.0)
        self.assertEqual(done.id, "web-ip-1")
        self.assertEqual(state.post_count[WEB], 0)

    def test_deploy_already_live_skips_trigger(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("web-live-1", DEPLOY_STATUS_LIVE))
        client = self._client(state)
        plan = plan_web_deploy(client, WEB, TARGET)
        self.assertEqual(plan.action, ACTION_ALREADY_LIVE)
        done = execute_web_deploy(client, WEB, TARGET)
        self.assertEqual(done.id, "web-live-1")
        self.assertEqual(state.post_count[WEB], 0)

    def test_verify_wrong_sha(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("web-live-bad", DEPLOY_STATUS_LIVE, commit=OTHER))
        client = self._client(state)
        with self.assertRaises(RenderDeployVerifyError) as ctx:
            verify_service_active_commit(client.list_deploys(WEB), TARGET)
        self.assertEqual(ctx.exception.code, "ace_render_verify_wrong_sha")

    def test_deploy_failed(self) -> None:
        state = RenderMockState()
        client = self._client(state)

        def fail_deploy(_m, url, *, headers, json_body=None):
            if url.endswith("/deploys") and _m == "POST":
                return None
            if "/deploys/dep-" in url and _m == "GET":
                deploy_id = url.split("/deploys/")[1]
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_BUILD_FAILED)
            return None

        client._transport.add_handler(fail_deploy)  # type: ignore[attr-defined]
        with self.assertRaises(RenderDeployFailedError):
            execute_web_deploy(client, WEB, TARGET, wait_deadline_seconds=2.0, poll_interval_seconds=0.0)

    def test_deploy_timeout(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("web-stuck", DEPLOY_STATUS_BUILD_IN_PROGRESS))
        client = self._client(state)
        clock = {"t": 0.0}

        def fake_clock() -> float:
            return clock["t"]

        def fake_sleep(_: float) -> None:
            clock["t"] += 2.0

        with self.assertRaises(RenderDeployTimeoutError):
            execute_web_deploy(
                client,
                WEB,
                TARGET,
                wait_deadline_seconds=1.0,
                poll_interval_seconds=0.0,
                clock=fake_clock,
                sleep_fn=fake_sleep,
            )

    def test_unauthorized(self) -> None:
        transport = MockRenderApiTransport()
        transport.set_default_handler(
            lambda *_a, **_k: RenderApiResponse(status_code=401, headers={}, body={"message": "bad key"})
        )
        client = RenderApiClient(transport, api_key="x")
        with self.assertRaises(RenderApiUnauthorizedError):
            client.list_deploys(WEB)

    def test_rate_limit_then_success(self) -> None:
        state = RenderMockState()
        transport = MockRenderApiTransport()
        hits = {"n": 0}

        def rl(method, url, *, headers, json_body=None):
            if method == "GET" and urlparse(url).path.endswith("/deploys"):
                hits["n"] += 1
                if hits["n"] == 1:
                    return RenderApiResponse(
                        status_code=429,
                        headers={"Retry-After": "0"},
                        body={"message": "rate limit exceeded"},
                    )
                return RenderApiResponse(status_code=200, headers={}, body=state.list_payload(WEB))
            return _route(state)(method, url, headers=headers, json_body=json_body)

        transport.set_default_handler(rl)
        client = RenderApiClient(transport, api_key="k", max_retries=3, retry_backoff_seconds=0.0)
        rows = client.list_deploys(WEB)
        self.assertEqual(rows, [])
        self.assertGreaterEqual(hits["n"], 2)

    def test_competing_newer_deploy_blocks_verify(self) -> None:
        deploys = [
            parse_deploy_row(_deploy("newer-ip", DEPLOY_STATUS_BUILD_IN_PROGRESS, commit=TARGET)),
            parse_deploy_row(_deploy("current-live", DEPLOY_STATUS_LIVE, commit=TARGET)),
        ]
        with self.assertRaises(RenderDeployVerifyError) as ctx:
            verify_service_active_commit(deploys, TARGET)
        self.assertEqual(ctx.exception.code, "ace_render_verify_competing_deploy")

    def test_duplicate_trigger_prevented_when_live(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("live-1", DEPLOY_STATUS_LIVE))
        client = self._client(state)
        plan = plan_web_deploy(client, WEB, TARGET)
        self.assertNotEqual(plan.action, ACTION_TRIGGER_DEPLOY)
        execute_web_deploy(client, WEB, TARGET)
        self.assertEqual(state.post_count[WEB], 0)

    def test_missing_api_key(self) -> None:
        transport = MockRenderApiTransport()
        client = RenderApiClient(transport, api_key="")
        with self.assertRaises(RenderApiMissingApiKeyError):
            client.get_service(WEB)

    def test_malformed_api_response(self) -> None:
        transport = MockRenderApiTransport()
        transport.set_default_handler(
            lambda *_a, **_k: RenderApiResponse(status_code=200, headers={}, body="not-json")
        )
        client = RenderApiClient(transport, api_key="k")
        with self.assertRaises(RenderApiParseError):
            client.list_deploys(WEB)

    def test_pair_verification_ok(self) -> None:
        state = RenderMockState()
        state.seed(WEB, _deploy("w1", DEPLOY_STATUS_LIVE))
        state.seed(WORKER, _deploy("k1", DEPLOY_STATUS_LIVE, trigger=TRIGGER_NEW_COMMIT))
        client = self._client(state)
        result = verify_web_and_worker_pair(
            client,
            web_service_id=WEB,
            worker_service_id=WORKER,
            expected_sha=TARGET,
        )
        self.assertTrue(result.ok)

    def test_prefix_sha_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_commit_sha("258337ca")

    def test_no_authorization_in_mock_records(self) -> None:
        state = RenderMockState()
        transport = MockRenderApiTransport()
        transport.set_default_handler(_route(state))
        client = RenderApiClient(transport, api_key="secret-token-value")
        client.list_deploys(WEB)
        auth_headers = [r.headers.get("Authorization", "") for r in transport.records]
        self.assertTrue(all(h == "[redacted]" for h in auth_headers))
        joined = json.dumps([r.headers for r in transport.records])
        self.assertNotIn("secret-token-value", joined)


def parse_deploy_row(raw: Dict[str, Any]):
    from engine.ace_local_dev_render_api_types import parse_deploy

    return parse_deploy(raw)


def method_is_post_deploy(method: str, url: str) -> bool:
    return method == "POST" and url.endswith("/deploys")


if __name__ == "__main__":
    unittest.main()
