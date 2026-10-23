"""
Publish Uri Candidate Preview + Dry-run — mocked PowerShell where noted.

Run: python -m unittest tests.test_ace_local_dev_publish_uri_candidate -v
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
from engine.ace_local_dev_publish_uri_candidate_preview import (
    build_public_publish_uri_candidate_preview_response,
    normalize_candidate_backend_sha,
)
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)
from tests.test_ace_local_dev_cycle import _session_fixture

PROD = "f" * 40
CAND = "a" * 40
CAND2 = "b" * 40


def _candidate_plan(session_id: str, **overrides) -> dict:
    plan = {
        "sessionId": session_id,
        "mode": "PUBLISH_URI_CANDIDATE_PREVIEW",
        "candidateBackendSha": CAND,
        "production": {
            "backendWeb": PROD,
            "backendWorker": PROD,
            "backendBranch": "main",
            "frontend": "c" * 40,
            "frontendBranch": "main",
        },
        "proposedChanges": {
            "backend": {
                "deployTargetBranch": "main",
                "sourceRef": "explicit-candidate-sha",
                "proposedHeadSha": CAND,
                "commits": [
                    {
                        "fullSha": CAND,
                        "shortSha": CAND[:7],
                        "subject": "product",
                        "classification": "productCandidate",
                        "paths": [{"path": "engine/builder1_literal_embodiment.py"}],
                        "securityFlags": [],
                    }
                ],
                "files": ["engine/builder1_literal_embodiment.py"],
            },
            "frontend": {
                "deployTargetBranch": "main",
                "sourceRef": "main",
                "proposedHeadSha": "c" * 40,
                "commits": [],
                "files": [],
            },
        },
        "workerImpact": {"deployWithBackendWeb": True, "proposedCommitSha": CAND},
        "commits": {
            "productCandidate": {"backend": [], "frontend": []},
            "excludedLocal": {"backend": [], "frontend": []},
            "mixedOrBlocked": [],
        },
        "candidateCommitShas": [CAND],
        "candidateBlockers": [],
        "warnings": ["REGISTRY_INFRA_HEAD_MISMATCH:backend|branchHead=x|registry=y"],
        "registryInfraDrift": ["REGISTRY_INFRA_HEAD_MISMATCH:backend|branchHead=x|registry=y"],
        "registryWipDrift": [],
        "cancelCycle": {"status": "OPEN"},
        "blockers": [],
        "mixedCommits": [],
        "readiness": {
            "hasProductChanges": True,
            "hasBackendProductChanges": True,
            "candidateVerifiedSafeForPublish": True,
            "verifiedSafeForLiveProduction": True,
            "canPrepareLocalPublish": False,
            "candidateDryRunAvailable": True,
        },
    }
    plan.update(overrides)
    return plan


def _raw(plan: dict) -> dict:
    return {"ok": True, "mode": "CANDIDATE_PREVIEW", "plan": plan}


class CandidateShaValidationTests(unittest.TestCase):
    def test_normalize_full_sha(self) -> None:
        self.assertEqual(normalize_candidate_backend_sha(CAND.upper()), CAND.lower())

    def test_reject_short_sha(self) -> None:
        self.assertIsNone(normalize_candidate_backend_sha("abc"))


class CandidateFingerprintTests(unittest.TestCase):
    def test_stable_and_changes_with_blocker(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        raw = _raw(_candidate_plan(sid))
        fp1 = compute_publish_uri_candidate_preview_fingerprint(raw)
        fp2 = compute_publish_uri_candidate_preview_fingerprint(raw)
        self.assertEqual(fp1, fp2)

        plan = _candidate_plan(sid)
        plan["candidateBlockers"] = ["BACKEND_PUBLISH_MIXED_COMMIT:deadbeef"]
        plan["readiness"]["candidateVerifiedSafeForPublish"] = False
        fp3 = compute_publish_uri_candidate_preview_fingerprint(_raw(plan))
        self.assertNotEqual(fp1, fp3)

    def test_stale_when_baseline_changes(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        raw = _raw(_candidate_plan(sid))
        fp1 = compute_publish_uri_candidate_preview_fingerprint(raw)
        plan = _candidate_plan(sid)
        plan["production"]["backendWeb"] = CAND2
        fp2 = compute_publish_uri_candidate_preview_fingerprint(_raw(plan))
        self.assertNotEqual(fp1, fp2)


class CandidateResponseTests(unittest.TestCase):
    def test_separates_warnings_from_blockers(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        pub = build_public_publish_uri_candidate_preview_response(_raw(_candidate_plan(sid)))
        self.assertEqual(pub["previewKind"], "CANDIDATE_BACKEND")
        self.assertEqual(len(pub["candidateBlockers"]), 0)
        self.assertEqual(len(pub["warnings"]), 1)
        self.assertFalse(pub["readiness"]["canPrepareLocalPublish"])
        self.assertTrue(pub["readiness"]["candidateVerifiedSafeForPublish"])

    def test_mixed_commit_blocker(self) -> None:
        sid = "ori-20261024-120000-aabbccdd"
        plan = _candidate_plan(sid)
        plan["candidateBlockers"] = ["BACKEND_PUBLISH_MIXED_COMMIT:abc1234"]
        plan["mixedCommits"] = [{"shortSha": "abc1234", "classification": "mixedProductAndLocal"}]
        plan["readiness"]["candidateVerifiedSafeForPublish"] = False
        pub = build_public_publish_uri_candidate_preview_response(_raw(plan))
        self.assertIn("BACKEND_PUBLISH_MIXED_COMMIT", pub["candidateBlockers"][0])


class CandidateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-publish-candidate-api-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("engine.ace_local_dev_routes.run_publish_uri_candidate_preview_powershell")
    def test_candidate_preview_api(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_ps.return_value = _raw(_candidate_plan(sid))

        resp = self.client.get(
            f"/api/local-dev/publish-uri/candidate/preview?candidateBackendSha={CAND}"
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["ok"])
        self.assertEqual(body["previewKind"], "CANDIDATE_BACKEND")
        mock_ps.assert_called_once_with(sid, CAND)

    def test_invalid_sha_400(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.get(
            "/api/local-dev/publish-uri/candidate/preview?candidateBackendSha=not-a-sha"
        )
        self.assertEqual(resp.status_code, 400)

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_dry_run.run_publish_uri_candidate_preview_powershell"
    )
    def test_dry_run_success(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _raw(_candidate_plan(sid))
        mock_ps.return_value = raw
        fp = compute_publish_uri_candidate_preview_fingerprint(raw)

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/dry-run",
            json={"candidateBackendSha": CAND, "publishUriPreviewFingerprint": fp},
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertTrue(body["ok"])
        self.assertTrue(body["fingerprintMatched"])

    @patch(
        "engine.ace_local_dev_publish_uri_candidate_dry_run.run_publish_uri_candidate_preview_powershell"
    )
    def test_dry_run_stale_fingerprint(self, mock_ps) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_ps.return_value = _raw(_candidate_plan(sid))

        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/dry-run",
            json={
                "candidateBackendSha": CAND,
                "publishUriPreviewFingerprint": "0" * 64,
            },
        )
        self.assertEqual(resp.status_code, 409)
        body = json.loads(resp.data)
        self.assertEqual(body["error"], "ace_publish_uri_candidate_fingerprint_stale")


if __name__ == "__main__":
    unittest.main()
