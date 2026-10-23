"""
Publish Uri Preview API and response shaping — mocked PowerShell; no deploy.

Run: python -m unittest tests.test_ace_local_dev_publish_uri_preview -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_preview_fingerprint
from engine.ace_local_dev_publish_uri_preview import build_public_publish_uri_preview_response
from tests.test_ace_local_dev_cycle import _session_fixture


def _base_plan(session_id: str, **overrides) -> dict:
    plan = {
        "sessionId": session_id,
        "production": {
            "backendWeb": "a" * 40,
            "backendWorker": "a" * 40,
            "backendBranch": "main",
            "frontend": "b" * 40,
            "frontendBranch": "main",
        },
        "proposedChanges": {
            "backend": {
                "deployTargetBranch": "main",
                "sourceRef": "ori/builder1-wip",
                "proposedHeadSha": "c" * 40,
                "commits": [],
                "files": [],
            },
            "frontend": {
                "deployTargetBranch": "main",
                "sourceRef": "main",
                "proposedHeadSha": "b" * 40,
                "commits": [],
                "files": [],
            },
        },
        "workerImpact": {
            "deployWithBackendWeb": True,
            "proposedCommitSha": "a" * 40,
            "note": "test",
        },
        "commits": {
            "productCandidate": {"backend": [], "frontend": []},
            "excludedLocal": {"backend": [], "frontend": []},
            "mixedOrBlocked": [],
        },
        "registryInfraDrift": [],
        "registryWipDrift": [],
        "cancelCycle": {"status": "OPEN"},
        "blockers": [],
        "mixedCommits": [],
        "readiness": {
            "hasProductChanges": False,
            "canPrepareLocalPublish": True,
            "verifiedSafeForLiveProduction": False,
        },
    }
    plan.update(overrides)
    return plan


def _raw_from_plan(plan: dict) -> dict:
    return {"ok": True, "mode": "PREVIEW", "plan": plan}


class PublishUriPreviewResponseTests(unittest.TestCase):
    def test_fingerprint_stable(self) -> None:
        raw = _raw_from_plan(_base_plan("ori-20261024-120000-aabbccdd"))
        fp1 = compute_publish_uri_preview_fingerprint(raw)
        fp2 = compute_publish_uri_preview_fingerprint(raw)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_no_product_changes(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        pub = build_public_publish_uri_preview_response(_raw_from_plan(_base_plan(sid)))
        self.assertTrue(pub["ok"])
        self.assertFalse(pub["readiness"]["hasProductChanges"])
        self.assertFalse(pub["readiness"]["verifiedSafeForLiveProduction"])

    def test_frontend_only(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        plan = _base_plan(sid)
        plan["proposedChanges"]["frontend"]["commits"] = [
            {
                "shortSha": "abc1234",
                "subject": "fe product",
                "classification": "productCandidate",
                "paths": [{"path": "src/App.jsx", "classification": "product"}],
            }
        ]
        plan["readiness"]["hasProductChanges"] = True
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertEqual(len(pub["proposedChanges"]["frontend"]["commits"]), 1)
        self.assertEqual(pub["proposedChanges"]["backend"]["commits"], [])

    def test_backend_only(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        plan["proposedChanges"]["backend"]["commits"] = [
            {
                "shortSha": "def5678",
                "subject": "builder1",
                "classification": "productCandidate",
                "paths": [{"path": "engine/builder1_foo.py", "classification": "product"}],
            }
        ]
        plan["readiness"]["hasProductChanges"] = True
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertEqual(len(pub["proposedChanges"]["backend"]["commits"]), 1)

    def test_both_repos(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        be = {"shortSha": "1111111", "subject": "be"}
        fe = {"shortSha": "2222222", "subject": "fe"}
        plan["proposedChanges"]["backend"]["commits"] = [be]
        plan["proposedChanges"]["frontend"]["commits"] = [fe]
        plan["commits"]["productCandidate"]["backend"] = [be]
        plan["commits"]["productCandidate"]["frontend"] = [fe]
        plan["readiness"]["hasProductChanges"] = True
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertEqual(len(pub["commits"]["productCandidate"]["backend"]), 1)
        self.assertEqual(len(pub["commits"]["productCandidate"]["frontend"]), 1)

    def test_mixed_commit_blockers(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        mixed = {
            "shortSha": "mix0001",
            "subject": "mixed",
            "classification": "mixedProductAndLocal",
            "paths": [
                {"path": "engine/builder1_x.py", "classification": "product"},
                {"path": "engine/ace_local_dev_routes.py", "classification": "forbiddenLocal"},
            ],
        }
        plan["mixedCommits"] = [mixed]
        plan["blockers"] = ["BACKEND_PUBLISH_MIXED_COMMIT:mix0001"]
        plan["readiness"]["verifiedSafeForLiveProduction"] = False
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertIn("BACKEND_PUBLISH_MIXED_COMMIT:mix0001", pub["blockers"])
        self.assertEqual(len(pub["commits"]["mixedOrBlocked"]), 1)

    def test_security_blocker(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        plan["blockers"] = ["BACKEND_PUBLISH_SECURITY:abc1234:ACE_SECURITY_WEAKENED"]
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertTrue(any("SECURITY" in b for b in pub["blockers"]))

    def test_env_secret_path_blocker(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        plan["blockers"] = ["BACKEND_PUBLISH_UNCLEAR_PATHS:abc1234"]
        plan["proposedChanges"]["backend"]["commits"] = [
            {
                "shortSha": "abc1234",
                "subject": "env",
                "paths": [{"path": ".env.production", "classification": "forbiddenSecret"}],
            }
        ]
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertTrue(pub["blockers"])

    def test_registry_mismatch(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        plan["registryInfraDrift"] = ["REGISTRY_INFRA_HEAD_MISMATCH:backend|branchHead=x|registry=y"]
        plan["readiness"]["canPrepareLocalPublish"] = False
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertEqual(len(pub["registryInfraDrift"]), 1)
        self.assertFalse(pub["readiness"]["canPrepareLocalPublish"])

    def test_cycle_not_open(self) -> None:
        plan = _base_plan("ori-20261024-120000-aabbccdd")
        plan["cancelCycle"] = {"status": "CANCELLED"}
        plan["blockers"] = ["CYCLE_ALREADY_CANCELLED"]
        pub = build_public_publish_uri_preview_response(_raw_from_plan(plan))
        self.assertEqual(pub["activeCycle"]["status"], "CANCELLED")


class PublishUriPreviewApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-publish-uri-api-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("engine.ace_local_dev_routes.run_publish_uri_preview_powershell")
    def test_api_success(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_ps.return_value = _raw_from_plan(_base_plan(sid))

        resp = self.client.get("/api/local-dev/publish-uri/preview")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["ok"])
        self.assertIn("publishUriPreviewFingerprint", body)
        mock_ps.assert_called_once_with(sid)

    def test_non_loopback_blocked(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.get(
            "/api/local-dev/publish-uri/preview",
            environ_overrides={"REMOTE_ADDR": "203.0.113.50"},
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
