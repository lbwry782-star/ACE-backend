"""
Publish Uri Candidate prepare-simulate — unit and API tests.

Run: python -m unittest tests.test_ace_local_dev_publish_uri_candidate_prepare -v
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
from engine.ace_local_dev_publish_candidate_registry import build_registry_record, save_active_publish_candidate
from engine.ace_local_dev_publish_uri_candidate_prepare import (
    FORBIDDEN_MIXED_COMMIT_SHA,
    PublishCandidatePrepareRejected,
    build_explicit_execution_plan,
    execute_publish_uri_candidate_prepare_simulate,
)
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_candidate_preview_fingerprint
from tests.test_ace_local_dev_cycle import _session_fixture
from tests.test_ace_local_dev_publish_uri_candidate import CAND, PROD, _candidate_plan, _raw

OP_ID = "a" * 32


def _ready_plan(session_id: str) -> dict:
    plan = _candidate_plan(session_id)
    plan["warnings"] = []
    plan["registryInfraDrift"] = []
    plan["registryWipDrift"] = []
    plan["readiness"]["canPrepareLocalPublish"] = True
    return plan


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-publish-prepare-plan-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_plan_uses_explicit_candidate_not_local_head(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        raw = _raw(_ready_plan(sid))
        plan = raw["plan"]
        execution = build_explicit_execution_plan(plan=plan, raw_preview=raw)
        self.assertTrue(execution["simulationOnly"])
        self.assertFalse(execution["usesLocalHead"])
        self.assertEqual(execution["verifiedSources"]["candidateBackendSha"], CAND)
        self.assertEqual(execution["verifiedSources"]["backendSourceRef"], "explicit-candidate-sha")
        self.assertEqual(len(execution["allowedProductCommits"]["backend"]), 1)

    def test_forbidden_e41f418_detected(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        plan = _ready_plan(sid)
        plan["candidateCommitShas"] = [FORBIDDEN_MIXED_COMMIT_SHA]
        with patch(
            "engine.ace_local_dev_publish_uri_candidate_prepare._resolve_active_registered_candidate",
            return_value=(sid, CAND, {}),
        ):
            with patch(
                "engine.ace_local_dev_publish_uri_candidate_prepare.run_publish_uri_candidate_preview_powershell",
                return_value=_raw(plan),
            ):
                fp = compute_publish_uri_candidate_preview_fingerprint(_raw(plan))
                with self.assertRaises(PublishCandidatePrepareRejected) as ctx:
                    execute_publish_uri_candidate_prepare_simulate(
                        publish_uri_preview_fingerprint=fp,
                        operation_id=OP_ID,
                    )
                self.assertEqual(ctx.exception.code, "ace_publish_prepare_blocked")
        progress_path = self.sessions_root / sid / f"publish-prepare-progress-{OP_ID}.json"
        self.assertTrue(progress_path.is_file())
        real_sessions = Path(__file__).resolve().parent.parent.parent / "_backups" / "sessions"
        self.assertFalse(
            (real_sessions / sid).is_dir(),
            "unit test must not create session folders under production sessions root",
        )


class PrepareSimulateFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-publish-prepare-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)
        self._reg_tmp = tempfile.mkdtemp(prefix="ace-publish-reg-")
        os.environ["ACE_PUBLISH_CANDIDATE_REGISTRY_PATH"] = str(
            Path(self._reg_tmp) / "publish-candidate-registry.json"
        )

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_PUBLISH_CANDIDATE_REGISTRY_PATH", None)
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._reg_tmp, ignore_errors=True)

    def _bootstrap(self, sid: str) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        save_active_publish_candidate(
            build_registry_record(
                session_id=sid,
                candidate_backend_sha=CAND,
                production_backend_baseline_sha=PROD,
            )
        )

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.run_publish_uri_candidate_preview_powershell"
    )
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.load_production_backend_baseline_sha",
        return_value=PROD,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_commit_exists",
        return_value=True,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_is_ancestor",
        return_value=True,
    )
    def test_prepare_simulate_success(self, *_mocks) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        self._bootstrap(sid)
        raw = _raw(_ready_plan(sid))
        _mocks[3].return_value = raw
        fp = compute_publish_uri_candidate_preview_fingerprint(raw)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": fp, "operationId": OP_ID},
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["simulationOnly"])
        self.assertFalse(body["executePublishAvailable"])
        self.assertTrue(body["realExecuteBlocked"])
        self.assertEqual(body["operationId"], OP_ID)
        self.assertIn("executionPlan", body)

        prog = self.client.get(
            f"/api/local-dev/publish-uri/candidate/prepare-progress?operationId={OP_ID}&sessionId={sid}"
        )
        self.assertEqual(prog.status_code, 200)
        prog_body = json.loads(prog.data)
        self.assertEqual(prog_body["operationStatus"], "COMPLETED")

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.run_publish_uri_candidate_preview_powershell"
    )
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.load_production_backend_baseline_sha",
        return_value=PROD,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_commit_exists",
        return_value=True,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_is_ancestor",
        return_value=True,
    )
    def test_stale_fingerprint(self, *_mocks) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        self._bootstrap(sid)
        _mocks[3].return_value = _raw(_ready_plan(sid))

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": "0" * 64, "operationId": "b" * 32},
        )
        self.assertEqual(resp.status_code, 409)
        body = json.loads(resp.data)
        self.assertEqual(body["error"], "ace_publish_uri_candidate_fingerprint_stale")

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.run_publish_uri_candidate_preview_powershell"
    )
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.load_production_backend_baseline_sha",
        return_value=PROD,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_commit_exists",
        return_value=True,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_is_ancestor",
        return_value=True,
    )
    def test_duplicate_operation_id(self, *_mocks) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        self._bootstrap(sid)
        raw = _raw(_ready_plan(sid))
        _mocks[3].return_value = raw
        fp = compute_publish_uri_candidate_preview_fingerprint(raw)

        first = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": fp, "operationId": OP_ID},
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": fp, "operationId": OP_ID},
        )
        self.assertEqual(second.status_code, 409)
        body = json.loads(second.data)
        self.assertEqual(body["error"], "ace_publish_prepare_operation_id_already_used")

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.run_publish_uri_candidate_preview_powershell"
    )
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_prepare.load_production_backend_baseline_sha",
        return_value=PROD,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_commit_exists",
        return_value=True,
    )
    @patch(
        "engine.ace_local_dev_publish_candidate_registry.git_is_ancestor",
        return_value=True,
    )
    def test_infra_drift_blocks_prepare(self, *_mocks) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        self._bootstrap(sid)
        plan = _ready_plan(sid)
        plan["registryInfraDrift"] = ["REGISTRY_INFRA_HEAD_MISMATCH:backend"]
        plan["readiness"]["canPrepareLocalPublish"] = False
        raw = _raw(plan)
        _mocks[3].return_value = raw
        fp = compute_publish_uri_candidate_preview_fingerprint(raw)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": fp, "operationId": "c" * 32},
        )
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertEqual(body["error"], "ace_publish_prepare_not_ready")

    def test_no_registered_candidate(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/prepare-simulate",
            json={"publishUriPreviewFingerprint": "d" * 64, "operationId": "e" * 32},
        )
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertEqual(body["error"], "ace_publish_candidate_not_registered")


if __name__ == "__main__":
    unittest.main()
