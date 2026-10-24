"""
Render mock ports for RealPublishExecuteAdapters — mock transport only.

Run: python -m unittest tests.test_ace_local_dev_publish_real_render_ports -v
"""
from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from engine.ace_local_dev_publish_execute_adapters import PublishExecuteContext
from engine.ace_local_dev_publish_real_execute_adapters import RealPublishExecuteAdapters
from engine.ace_local_dev_publish_real_execute_ports import RealPublishExecutePorts
from engine.ace_local_dev_publish_real_render_ports import (
    RenderMockDeployPortConfig,
    RenderMockPortError,
    assert_mock_render_transport,
    assert_render_mock_port_config,
    build_render_mock_port_bundle,
)
from engine.ace_local_dev_publish_render_sim_transport import deploy_row
from engine.ace_local_dev_render_api_transport import UrllibRenderApiTransport
from engine.ace_local_dev_render_api_types import DEPLOY_STATUS_LIVE, TRIGGER_NEW_COMMIT
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
)
from tests.test_ace_local_dev_publish_real_execute_adapters import MockPorts, _bundle

CAND = "c" * 40
WEB_TEST = "ace-publish-render-test-web-srv"
WORKER_TEST = "ace-publish-render-test-worker-srv"


def _config(**kwargs) -> RenderMockDeployPortConfig:
    base = {
        "web_service_id": WEB_TEST,
        "worker_service_id": WORKER_TEST,
        "wait_deadline_seconds": 5.0,
        "poll_interval_seconds": 0.01,
    }
    base.update(kwargs)
    return RenderMockDeployPortConfig(**base)


class RenderMockPortTests(unittest.TestCase):
    def test_web_deploy_success(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.on_git_push(CAND)
        out = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertTrue(out.ok)
        self.assertTrue(out.deploy_id)
        self.assertEqual(out.commit_sha, CAND)

    def test_web_already_live(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.seed(
            WEB_TEST,
            deploy_row("dep-live", DEPLOY_STATUS_LIVE, commit=CAND),
        )
        out = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertTrue(out.ok)
        self.assertTrue(out.already_live)
        self.assertEqual(bundle.state.post_count[WEB_TEST], 0)

    def test_web_deploy_failed(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.fail_web = True
        out = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_render_web_failed")

    def test_web_timeout(self) -> None:
        bundle = build_render_mock_port_bundle(_config(wait_deadline_seconds=0.05))
        bundle.state.web_stall_live = True
        out = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_render_web_timeout")
        retry = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertEqual(retry.code, "ace_publish_real_render_web_unknown_no_retry")

    def test_web_unknown_status(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.web_unknown_status = True
        out = bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertIn(
            out.code,
            (
                "ace_publish_real_render_web_unknown",
                "ace_publish_real_render_web_failed",
                "ace_publish_real_render_web_timeout",
            ),
        )

    def test_worker_auto_deploy_success(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.on_git_push(CAND)
        out = bundle.worker_port.wait_for_worker_live(candidate_sha=CAND)
        self.assertTrue(out.ok)
        self.assertTrue(out.observed_only)
        self.assertEqual(bundle.state.post_count[WORKER_TEST], 0)

    def test_worker_already_live(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.seed(
            WORKER_TEST,
            deploy_row("w-live", DEPLOY_STATUS_LIVE, commit=CAND, trigger=TRIGGER_NEW_COMMIT),
        )
        out = bundle.worker_port.wait_for_worker_live(candidate_sha=CAND)
        self.assertTrue(out.ok)
        self.assertTrue(out.already_live)

    def test_worker_failed(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.on_git_push(CAND)
        bundle.state.fail_worker = True
        out = bundle.worker_port.wait_for_worker_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_render_worker_failed")

    def test_worker_timeout(self) -> None:
        bundle = build_render_mock_port_bundle(_config(wait_deadline_seconds=0.05))
        bundle.state.on_git_push(CAND)
        bundle.state.worker_stall_live = True
        out = bundle.worker_port.wait_for_worker_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_render_worker_timeout")

    def test_worker_wrong_sha(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.pushed_sha = "d" * 40
        out = bundle.worker_port.wait_for_worker_live(candidate_sha=CAND)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_render_worker_sha_mismatch")

    def test_production_service_id_forbidden(self) -> None:
        with self.assertRaises(RenderMockPortError) as ctx:
            assert_render_mock_port_config(
                _config(web_service_id=ACE_RENDER_DEFAULT_WEB_SERVICE_ID)
            )
        self.assertEqual(ctx.exception.code, "ace_publish_real_render_production_service_id_forbidden")

    def test_urllib_transport_forbidden(self) -> None:
        with self.assertRaises(RenderMockPortError):
            assert_mock_render_transport(UrllibRenderApiTransport())

    def test_git_push_failed_blocks_render_in_adapters(self) -> None:
        mocks = MockPorts(git_fail=True)
        built = mocks.build()
        render = build_render_mock_port_bundle(_config())
        ports = RealPublishExecutePorts(
            git_backend_push=built.git_backend_push,
            render_web=render.web_port,
            render_worker=render.worker_port,
            github_pages=built.github_pages,
            production_verify=built.production_verify,
            registry_update=built.registry_update,
        )
        adapters = RealPublishExecuteAdapters(ports)
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="render-block-git")
        adapters.prepare_isolated_release(ctx)
        self.assertFalse(adapters.git_push_backend(ctx).ok)
        web = adapters.wait_render_web(ctx)
        self.assertFalse(web.ok)
        self.assertEqual(web.code, "ace_publish_real_render_web_prerequisite")
        self.assertEqual(render.state.post_count[WEB_TEST], 0)

    def test_registry_not_updated_when_web_fails(self) -> None:
        mocks = MockPorts()
        built = mocks.build()
        render = build_render_mock_port_bundle(_config())
        render.state.fail_web = True
        ports = RealPublishExecutePorts(
            git_backend_push=built.git_backend_push,
            render_web=render.web_port,
            render_worker=render.worker_port,
            github_pages=built.github_pages,
            production_verify=built.production_verify,
            registry_update=built.registry_update,
        )
        adapters = RealPublishExecuteAdapters(ports)
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="render-no-reg")
        adapters.prepare_isolated_release(ctx)
        adapters.git_push_backend(ctx)
        self.assertFalse(adapters.wait_render_web(ctx).ok)
        reg = adapters.update_production_registry(ctx)
        self.assertFalse(reg.ok)
        self.assertEqual(mocks.registry_calls, 0)

    def test_no_real_render_module_imports_urllib(self) -> None:
        text = Path(
            importlib.import_module("engine.ace_local_dev_publish_real_render_ports").__file__
        ).read_text(encoding="utf-8")
        self.assertNotIn("UrllibRenderApiTransport()", text)
        self.assertIn("UrllibRenderApiTransport", text)

    def test_routes_not_wired(self) -> None:
        routes = Path(importlib.import_module("engine.ace_local_dev_routes").__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("ace_local_dev_publish_real_render_ports", routes)
        self.assertNotIn("MockRenderWebDeployPort", routes)

    def test_transport_records_only_mock_calls(self) -> None:
        bundle = build_render_mock_port_bundle(_config())
        bundle.state.on_git_push(CAND)
        bundle.web_port.wait_for_web_live(candidate_sha=CAND)
        self.assertTrue(all("api.render.com" in r.url for r in bundle.transport.records))
        self.assertGreater(len(bundle.transport.records), 0)


if __name__ == "__main__":
    unittest.main()
