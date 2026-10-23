"""
Publish candidate local registry — register and discovery.

Run: python -m unittest tests.test_ace_local_dev_publish_candidate_registry -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
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
from engine.ace_local_dev_publish_candidate_registry import (
    build_registry_record,
    evaluate_registered_candidate,
    git_commit_exists,
    git_is_ancestor,
    load_publish_candidate_registry,
    publish_candidate_registry_path,
    save_active_publish_candidate,
)
from engine.ace_local_dev_publish_candidate_register import (
    CONFIRM_REGISTER_TOKEN,
    discover_registered_publish_candidate,
    register_publish_candidate,
)
from tests.test_ace_local_dev_cycle import _session_fixture

PROD = "fadbddd4abb672fd2f4c436f2afddeeff5ea2f67"
CAND = "258337ca9fbc01553b11adbb49f528e1ed059a58"
WIP = "6fdf09649017ccf787ac3c896af81a7330bf52ea"


def _write_production_registry(path: Path, *, backend_prod: str = PROD, wip: str = WIP) -> None:
    payload = {
        "schemaVersion": 1,
        "production": {
            "backendWeb": {"deployedCommitSha": backend_prod},
            "backendWorker": {"deployedCommitSha": backend_prod},
            "frontend": {"deployedCommitSha": "2" * 40},
        },
        "wip": {"builder1": {"commitSha": wip}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_preview_raw(sid: str, candidate: str = CAND) -> dict:
    return {
        "ok": True,
        "mode": "CANDIDATE_PREVIEW",
        "plan": {
            "sessionId": sid,
            "mode": "PUBLISH_URI_CANDIDATE_PREVIEW",
            "candidateBackendSha": candidate,
            "production": {"backendWeb": PROD, "backendWorker": PROD, "frontend": "2" * 40},
            "proposedChanges": {
                "backend": {
                    "proposedHeadSha": candidate,
                    "files": ["engine/builder1_literal_embodiment.py"],
                    "commits": [{"fullSha": candidate}],
                },
                "frontend": {"proposedHeadSha": "2" * 40, "files": [], "commits": []},
            },
            "candidateBlockers": [],
            "blockers": [],
            "warnings": [],
            "registryInfraDrift": [],
            "registryWipDrift": [],
            "cancelCycle": {"status": "OPEN"},
            "readiness": {
                "candidateVerifiedSafeForPublish": True,
                "canPrepareLocalPublish": True,
                "hasProductChanges": True,
            },
            "candidateCommitShas": [candidate],
        },
    }


class RegistryEvaluateTests(unittest.TestCase):
    def test_stale_session_mismatch(self) -> None:
        record = build_registry_record(
            session_id="ori-20261024-120000-aaaaaaaa",
            candidate_backend_sha=CAND,
            production_backend_baseline_sha=PROD,
        )
        status, reasons = evaluate_registered_candidate(
            record=record,
            active_session_id="ori-20261024-120000-bbbbbbbb",
            cycle_status="OPEN",
            current_production_baseline_sha=PROD,
        )
        self.assertEqual(status, "STALE")
        self.assertIn("SESSION_MISMATCH", reasons)

    def test_stale_baseline_changed(self) -> None:
        record = build_registry_record(
            session_id="ori-20261024-120000-aaaaaaaa",
            candidate_backend_sha=CAND,
            production_backend_baseline_sha=PROD,
        )
        status, _ = evaluate_registered_candidate(
            record=record,
            active_session_id="ori-20261024-120000-aaaaaaaa",
            cycle_status="OPEN",
            current_production_baseline_sha="a" * 40,
        )
        self.assertEqual(status, "STALE")

    def test_stale_cancelled_cycle(self) -> None:
        record = build_registry_record(
            session_id="ori-20261024-120000-aaaaaaaa",
            candidate_backend_sha=CAND,
            production_backend_baseline_sha=PROD,
        )
        status, reasons = evaluate_registered_candidate(
            record=record,
            active_session_id="ori-20261024-120000-aaaaaaaa",
            cycle_status="CANCELLED",
            current_production_baseline_sha=PROD,
        )
        self.assertEqual(status, "STALE")
        self.assertTrue(any("CYCLE" in r for r in reasons))


class RegisterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-pub-cand-reg-")
        self.sessions_root = Path(self._tmp) / "sessions"
        self.sessions_root.mkdir(parents=True)
        self.reg_dir = Path(self._tmp) / "registry"
        self.reg_dir.mkdir()
        self.prod_reg = self.reg_dir / "production-registry.json"
        _write_production_registry(self.prod_reg)
        self.cand_reg = self.reg_dir / "publish-candidate-registry.json"

        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ["ACE_ORI_PS_REGISTRY_PATH"] = str(self.prod_reg)
        os.environ["ACE_PUBLISH_CANDIDATE_REGISTRY_PATH"] = str(self.cand_reg)
        os.environ["ACE_ORI_PS_BACKEND_ROOT"] = str(
            Path(__file__).resolve().parent.parent
        )
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

        self.sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, self.sid)
        write_active_dev_cycle_pointer(session_id=self.sid, sessions_root=self.sessions_root)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        for key in (
            "ACE_SESSIONS_ROOT",
            "ACE_ORI_PS_REGISTRY_PATH",
            "ACE_PUBLISH_CANDIDATE_REGISTRY_PATH",
            "ACE_ORI_PS_BACKEND_ROOT",
        ):
            os.environ.pop(key, None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("engine.ace_local_dev_publish_candidate_register.run_publish_uri_candidate_preview_powershell")
    def test_register_success(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        mock_ps.return_value = _candidate_preview_raw(self.sid)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/register",
            json={"candidateBackendSha": CAND, "confirmRegister": CONFIRM_REGISTER_TOKEN},
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["ok"])
        self.assertEqual(body["candidateBackendSha"], CAND)
        stored = load_publish_candidate_registry()
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["sessionId"], self.sid)

    @patch("engine.ace_local_dev_publish_candidate_register.run_publish_uri_candidate_preview_powershell")
    def test_register_rejects_wip_head(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        mock_ps.return_value = _candidate_preview_raw(self.sid, WIP)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/register",
            json={"candidateBackendSha": WIP, "confirmRegister": CONFIRM_REGISTER_TOKEN},
        )
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertEqual(body["error"], "ace_publish_candidate_is_wip_head")

    def test_register_invalid_sha(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/register",
            json={"candidateBackendSha": "bad", "confirmRegister": CONFIRM_REGISTER_TOKEN},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("engine.ace_local_dev_publish_candidate_register.run_publish_uri_candidate_preview_powershell")
    def test_discover_active(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        mock_ps.return_value = _candidate_preview_raw(self.sid)
        register_publish_candidate(
            candidate_backend_sha=CAND,
            confirm_register=CONFIRM_REGISTER_TOKEN,
        )

        resp = self.client.get("/api/local-dev/publish-uri/candidate/registered")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["hasRegisteredCandidate"])
        self.assertEqual(body["candidateBackendSha"], CAND)

    def test_discover_none(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")

        resp = self.client.get("/api/local-dev/publish-uri/candidate/registered")
        body = json.loads(resp.data)
        self.assertFalse(body["hasRegisteredCandidate"])
        self.assertEqual(body["status"], "NONE")

    @patch("engine.ace_local_dev_publish_candidate_register.run_publish_uri_candidate_preview_powershell")
    def test_discover_stale_after_baseline_change(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        mock_ps.return_value = _candidate_preview_raw(self.sid)
        register_publish_candidate(
            candidate_backend_sha=CAND,
            confirm_register=CONFIRM_REGISTER_TOKEN,
        )
        _write_production_registry(self.prod_reg, backend_prod="a" * 40)

        resp = self.client.get("/api/local-dev/publish-uri/candidate/registered")
        body = json.loads(resp.data)
        self.assertFalse(body["hasRegisteredCandidate"])
        self.assertEqual(body["status"], "STALE")

    def test_non_loopback_blocked(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get(
            "/api/local-dev/publish-uri/candidate/registered",
            environ_overrides={"REMOTE_ADDR": "203.0.113.8"},
        )
        self.assertEqual(resp.status_code, 403)


class GitValidationIntegrationTests(unittest.TestCase):
    def test_real_candidate_exists_and_ancestry(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.assertTrue(git_commit_exists(root, CAND))
        self.assertTrue(git_is_ancestor(root, PROD, CAND))
        self.assertFalse(CAND == WIP)


if __name__ == "__main__":
    unittest.main()
