"""
Local cancel-development Preview API — offline tests.

Run: python -m unittest tests.test_ace_local_dev_cancel_preview -v
"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import app as app_module
from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_local_dev_cancel_preview import build_public_preview_response


def _sample_raw_plan() -> dict:
    return {
        "ok": True,
        "mode": "PREVIEW",
        "plan": {
            "sessionId": "ori-20261023-010342-2d952e88",
            "canExecute": True,
            "blockers": [],
            "cancelCycle": {"status": "OPEN"},
            "commitsToCancel": {
                "backend": [
                    {
                        "shortSha": "abc1234",
                        "subject": "wip-only",
                        "classification": "sessionWip",
                        "branches": ["ori/builder1-wip"],
                        "preservedInWipRef": True,
                        "preservedInBackup": True,
                    }
                ],
                "frontend": [],
            },
            "git": {
                "backend": {"PathsToRevert": ["product-temp.txt"], "CurrentDirtyCount": 1},
                "frontend": {"PathsToRevert": [], "CurrentDirtyCount": 0},
            },
            "infraAdvancementPaths": {"backend": ["infra-permanent.txt"], "frontend": []},
            "infraCommitsPreserved": {"backend": [], "frontend": []},
            "env": {
                "backend": {
                    "AddedKeys": ["SESSION_ADDED"],
                    "RemovedKeys": [],
                    "ChangedKeys": ["SESSION_TEMP"],
                    "PreservedKeys": ["PORT"],
                    "CanApply": True,
                },
                "frontend": {"AddedKeys": [], "RemovedKeys": [], "ChangedKeys": [], "PreservedKeys": [], "CanApply": True},
            },
            "commitGraph": {
                "backend": {
                    "PreBaselineCommits": [
                        {
                            "shortSha": "deadbeef",
                            "subject": "legacy",
                            "explicitlyIncluded": False,
                        }
                    ]
                }
            },
            "registryInfraDrift": [],
            "willCancel": {"git": ["reset to infra"]},
            "infraTargets": {"backendCommit": "5ffe7b9", "frontendCommit": "62e0951"},
        },
    }


class LocalDevCancelPreviewApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def test_blocked_when_local_dev_disabled(self) -> None:
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json().get("error"), "ace_local_dev_not_enabled")

    def test_blocked_when_client_not_loopback(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get(
            "/api/local-dev/cancel-development/preview",
            environ_overrides={"REMOTE_ADDR": "203.0.113.50"},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json().get("error"), "ace_local_dev_loopback_client_required")

    def test_rejects_query_parameters(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get("/api/local-dev/cancel-development/preview?sessionId=evil")
        self.assertEqual(resp.status_code, 400)

    def test_no_session_configured_returns_clear_status(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body["activeCycle"]["status"], "NONE")

    @patch("engine.ace_local_dev_routes.run_cancel_dev_preview_powershell")
    def test_local_preview_success(self, mock_run) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        os.environ["ACE_ORI_DEV_SESSION_ID"] = "ori-20261023-010342-2d952e88"
        mock_run.return_value = _sample_raw_plan()

        resp = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertFalse(body.get("executeEndpointAvailable"))
        self.assertEqual(len(body["commitsToCancel"]["backend"]), 1)
        self.assertIn("product-temp.txt", body["pathsToRevert"]["backend"])
        self.assertEqual(len(body["preBaselineWarnings"]), 1)
        text = json.dumps(body)
        self.assertNotIn("sk-test-secret", text)
        self.assertNotIn(".dpapi", text.lower())

    def test_sanitize_strips_sensitive_keys(self) -> None:
        raw = _sample_raw_plan()
        raw["plan"]["secretToken"] = "sk-test-secret"
        raw["plan"]["env"]["backend"]["DestDpapiPath"] = "C:\\secret\\x.dpapi"
        public = build_public_preview_response(raw)
        dumped = json.dumps(public)
        self.assertNotIn("sk-test-secret", dumped)
        self.assertNotIn("DestDpapiPath", dumped)

    def test_invalid_session_env_pattern(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        os.environ["ACE_ORI_DEV_SESSION_ID"] = "../../../etc/passwd"
        resp = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(resp.status_code, 400)

    def test_post_method_not_allowed(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.post("/api/local-dev/cancel-development/preview")
        self.assertIn(resp.status_code, (404, 405))


if __name__ == "__main__":
    unittest.main()
