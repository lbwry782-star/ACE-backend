"""
Full E2E: Start → dev changes → Preview → Execute → Start (TEMP only, real API + PowerShell).

Run: set ACE_RUN_E2E_TEMP=1 && python -m unittest tests.test_ace_local_dev_e2e_temp -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import app as app_module
from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_local_dev_cycle_store import (
    ACTIVE_CYCLE_FILENAME,
    resolve_active_dev_cycle,
)
from engine.ace_local_dev_execute_guard import end_local_dev_execute, try_begin_local_dev_execute


def _run_setup_layout() -> dict:
    script = Path(__file__).resolve().parent.parent / "scripts" / "Setup-OriLocalDevApiE2ETempLayout.ps1"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "setup_failed")[:3000])
    layout_path = (completed.stdout or "").strip().splitlines()[-1].strip()
    return json.loads(Path(layout_path).read_text(encoding="utf-8-sig"))


def _apply_isolated_env(layout: dict) -> None:
    os.environ["ACE_SESSIONS_ROOT"] = layout["sessionsRoot"]
    os.environ["ACE_ORI_PS_BACKEND_ROOT"] = layout["backendRoot"]
    os.environ["ACE_ORI_PS_FRONTEND_ROOT"] = layout["frontendRoot"]
    os.environ["ACE_ORI_PS_REGISTRY_PATH"] = layout["registryPath"]
    os.environ["ACE_ORI_PS_SESSIONS_ROOT"] = layout["sessionsRoot"]


def _clear_isolated_env() -> None:
    for key in (
        "ACE_SESSIONS_ROOT",
        "ACE_ORI_PS_BACKEND_ROOT",
        "ACE_ORI_PS_FRONTEND_ROOT",
        "ACE_ORI_PS_REGISTRY_PATH",
        "ACE_ORI_PS_SESSIONS_ROOT",
        "ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE",
    ):
        os.environ.pop(key, None)


def _git(args: list[str], repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return (completed.stdout or "").strip()


@unittest.skipUnless(os.environ.get("ACE_RUN_E2E_TEMP") == "1", "set ACE_RUN_E2E_TEMP=1")
class LocalDevApiE2ETemp(unittest.TestCase):
    layout: dict
    temp_root: Path
    client: object

    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = _run_setup_layout()
        cls.temp_root = Path(cls.layout["tempRoot"])
        _apply_isolated_env(cls.layout)
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        cls.client = app_module.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        _clear_isolated_env()
        reset_local_dev_config_for_tests()
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def test_full_cycle_start_cancel_start_via_api(self) -> None:
        backend = Path(self.layout["backendRoot"])
        frontend = Path(self.layout["frontendRoot"])
        sessions = Path(self.layout["sessionsRoot"])
        infra_head = self.layout["backendInfraHead"]

        resp = self.client.post("/api/local-dev/dev-cycle/start", json={"confirmStart": "START"})
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        sid = body["sessionId"]
        self.assertTrue((sessions / sid / "session.json").is_file())
        self.assertTrue((sessions / ACTIVE_CYCLE_FILENAME).is_file())
        res = resolve_active_dev_cycle()
        self.assertEqual(res.cycle_status, "OPEN")
        self.assertEqual(res.session_id, sid)

        env_before = (backend / ".env.local").read_text(encoding="utf-8")
        _git(["checkout", "-q", "ori/builder1-wip"], backend)
        (backend / "api-e2e-wip.txt").write_text("dev", encoding="utf-8")
        (backend / ".env.local").write_text(
            env_before.replace("SESSION_TEMP=pre-start", "SESSION_TEMP=during-cycle"),
            encoding="utf-8",
        )
        _git(["add", "api-e2e-wip.txt"], backend)
        _git(["commit", "-q", "-m", "api-e2e-wip-only"], backend)
        wip_head = _git(["rev-parse", "HEAD"], backend)

        prev = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(prev.status_code, 200)
        plan = prev.get_json()
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["activeCycle"]["sessionId"], sid)
        self.assertEqual(len(plan.get("blockers") or []), 0)
        fp = plan.get("previewFingerprint")
        self.assertTrue(fp)
        commits_be = plan["commitsToCancel"]["backend"]
        self.assertGreaterEqual(len(commits_be), 1)
        for c in commits_be:
            self.assertNotEqual(c.get("classification"), "infra")
        targets = plan.get("infraTargets") or {}
        self.assertEqual(str(targets.get("backendCommit") or ""), infra_head)

        exec_resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertEqual(exec_resp.status_code, 200, exec_resp.get_data(as_text=True))
        self.assertTrue(exec_resp.get_json().get("ok"))

        self.assertFalse((backend / "api-e2e-wip.txt").exists())
        self.assertIn("SESSION_TEMP=pre-start", (backend / ".env.local").read_text(encoding="utf-8"))
        after_head = _git(["rev-parse", "HEAD"], backend)
        self.assertNotEqual(after_head, wip_head)
        _git(["rev-parse", "-q", "--verify", infra_head], backend)

        cancelled = resolve_active_dev_cycle()
        self.assertEqual(cancelled.cycle_status, "CANCELLED")
        self.assertEqual(cancelled.session_id, sid)

        resp2 = self.client.post("/api/local-dev/dev-cycle/start", json={"confirmStart": "START"})
        self.assertEqual(resp2.status_code, 200)
        sid2 = resp2.get_json()["sessionId"]
        self.assertNotEqual(sid2, sid)
        res2 = resolve_active_dev_cycle()
        self.assertEqual(res2.cycle_status, "OPEN")
        self.assertEqual(res2.session_id, sid2)
        self.assertTrue((sessions / sid / "cancel-dev-cycle.state.json").is_file())

    def test_stale_fingerprint_rejected(self) -> None:
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": "0" * 64},
        )
        self.assertIn(resp.status_code, (400, 409))

    def test_execute_without_confirm(self) -> None:
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "NO", "previewFingerprint": "a" * 64},
        )
        self.assertEqual(resp.status_code, 400)

    def test_parallel_execute_rejected(self) -> None:
        try_begin_local_dev_execute()
        try:
            resp = self.client.post(
                "/api/local-dev/cancel-development/execute",
                json={"confirmExecute": "CANCEL", "previewFingerprint": "b" * 64},
            )
            self.assertEqual(resp.status_code, 409)
        finally:
            end_local_dev_execute()

    def test_non_loopback_blocked(self) -> None:
        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
            environ_overrides={"REMOTE_ADDR": "203.0.113.50"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_client_session_id_rejected(self) -> None:
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={
                "confirmExecute": "CANCEL",
                "previewFingerprint": "c" * 64,
                "sessionId": "evil",
            },
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
