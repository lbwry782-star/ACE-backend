"""Shared mock Render transport helpers for publish simulation (no network)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from engine.ace_local_dev_render_api_transport import MockRenderApiTransport, RenderApiResponse
from engine.ace_local_dev_render_api_types import (
    DEPLOY_STATUS_BUILD_IN_PROGRESS,
    DEPLOY_STATUS_LIVE,
    DEPLOY_STATUS_QUEUED,
    TRIGGER_API,
    TRIGGER_NEW_COMMIT,
)
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
)

WEB = ACE_RENDER_DEFAULT_WEB_SERVICE_ID
WORKER = ACE_RENDER_DEFAULT_WORKER_SERVICE_ID


def deploy_row(
    deploy_id: str,
    status: str,
    *,
    commit: str,
    trigger: str = TRIGGER_API,
) -> Dict[str, Any]:
    return {
        "id": deploy_id,
        "status": status,
        "commit": {"id": commit, "message": "sim"},
        "trigger": trigger,
        "createdAt": "2026-10-24T03:00:00Z",
    }


class PublishRenderSimState:
    def __init__(self) -> None:
        self.deploys: Dict[str, Dict[str, Dict[str, Any]]] = {WEB: {}, WORKER: {}}
        self.post_count: Dict[str, int] = {WEB: 0, WORKER: 0}
        self.pushed_sha: Optional[str] = None
        self.fail_web: bool = False
        self.fail_worker: bool = False

    def seed(self, service_id: str, *rows: Dict[str, Any]) -> None:
        for row in rows:
            self.deploys[service_id][row["id"]] = dict(row)

    def list_payload(self, service_id: str) -> List[Dict[str, Any]]:
        rows = list(self.deploys[service_id].values())
        rows.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
        return [{"deploy": r, "cursor": f"c-{r['id']}"} for r in rows]

    def set_status(self, service_id: str, deploy_id: str, status: str) -> None:
        self.deploys[service_id][deploy_id]["status"] = status

    def on_git_push(self, sha: str) -> None:
        self.pushed_sha = sha
        if "w-auto" not in self.deploys[WORKER]:
            self.seed(
                WORKER,
                deploy_row("w-auto", DEPLOY_STATUS_BUILD_IN_PROGRESS, commit=sha, trigger=TRIGGER_NEW_COMMIT),
            )


def _service_id_from_url(url: str) -> str:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return parts[parts.index("services") + 1]


def build_publish_sim_mock_transport(
    state: PublishRenderSimState,
    *,
    web_service_id: str = WEB,
    worker_service_id: str = WORKER,
) -> MockRenderApiTransport:
    transport = MockRenderApiTransport()

    def route(method: str, url: str, *, headers, json_body=None) -> Optional[RenderApiResponse]:
        if not (headers.get("Authorization") or "").startswith("Bearer "):
            return RenderApiResponse(status_code=401, headers={}, body={"message": "auth"})
        path = urlparse(url).path
        sid = _service_id_from_url(url)

        if method == "GET" and "/deploys/" in path:
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            row = state.deploys.get(sid, {}).get(deploy_id)
            if not row:
                return RenderApiResponse(status_code=404, headers={}, body={"message": "missing"})
            return RenderApiResponse(status_code=200, headers={}, body=row)

        if method == "GET" and path.endswith("/deploys"):
            return RenderApiResponse(status_code=200, headers={}, body=state.list_payload(sid))

        if method == "POST" and path.endswith("/deploys"):
            state.post_count[sid] = state.post_count.get(sid, 0) + 1
            commit = (json_body or {}).get("commitId") or state.pushed_sha or ""
            deploy_id = f"dep-{sid}-{state.post_count[sid]}"
            row = deploy_row(deploy_id, DEPLOY_STATUS_QUEUED, commit=commit)
            state.deploys.setdefault(sid, {})[deploy_id] = row
            return RenderApiResponse(status_code=201, headers={}, body=row)

        return RenderApiResponse(status_code=404, headers={}, body={"message": "unhandled"})

    def advance(method: str, url: str, *, headers, json_body=None) -> Optional[RenderApiResponse]:
        path = urlparse(url).path
        if state.fail_web and method == "GET" and "/deploys/dep-" in path and WEB in path:
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            state.set_status(WEB, deploy_id, "build_failed")
            return None
        if state.fail_worker and "/deploys/w-auto" in path:
            state.set_status(WORKER, "w-auto", "build_failed")
            return None
        if method == "GET" and "/deploys/dep-" in path and WEB in path:
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            cur = state.deploys[WEB][deploy_id]["status"]
            if cur == DEPLOY_STATUS_QUEUED:
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_BUILD_IN_PROGRESS)
            else:
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_LIVE)
        if method == "GET" and "/deploys/w-auto" in path:
            state.set_status(WORKER, "w-auto", DEPLOY_STATUS_LIVE)
        return None

    transport.set_default_handler(route)
    transport.add_handler(advance)
    return transport
