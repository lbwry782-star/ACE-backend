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
from engine.ace_local_dev_cancel_progress import load_public_progress
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


def _repo_scripts_src() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def _ensure_e2e_backend_wip_markers(backend: Path) -> None:
    scripts_dir = backend / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    src = _repo_scripts_src()
    for name in (
        "OriDevSessionBaselineLib.ps1",
        "OriDevCancelDevLib.ps1",
        "New-OriDevSessionBaseline.ps1",
        "Invoke-OriCancelDev.ps1",
    ):
        shutil.copy2(src / name, scripts_dir / name)
    (backend / ".env.local.example").write_text(
        "ACE_LOCAL_DEV_ENABLED=true\nPORT=5000\nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000\n",
        encoding="utf-8",
    )


def _restore_e2e_pre_start_wip(layout: dict) -> None:
    backend = Path(layout["backendRoot"])
    frontend = Path(layout["frontendRoot"])
    _git(["clean", "-fd"], backend)
    _git(["checkout", "-f", "-q", "ori/local-dev-infrastructure"], backend)
    _git(["reset", "--hard", layout["backendInfraHead"]], backend)
    _git(["checkout", "-f", "-q", "ori/builder1-wip"], backend)
    _ensure_e2e_backend_wip_markers(backend)
    (backend / ".env.local").write_text(
        "ACE_LOCAL_DEV_ENABLED=true\nPORT=5000\nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000\nSESSION_TEMP=pre-start\n",
        encoding="utf-8",
    )
    (frontend / ".env.development.local").write_text(
        "VITE_BACKEND_URL=http://127.0.0.1:5000",
        encoding="utf-8",
    )
    wip_file = backend / "wip-cycle.txt"
    if not wip_file.is_file():
        wip_file.write_text("pre-start-wip", encoding="utf-8")


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

    def setUp(self) -> None:
        end_local_dev_execute()
        _restore_e2e_pre_start_wip(self.layout)
        sessions = Path(self.layout["sessionsRoot"])
        for entry in sessions.iterdir():
            if entry.is_dir() and entry.name.startswith("ori-"):
                shutil.rmtree(entry, ignore_errors=True)
            elif entry.name in (
                ACTIVE_CYCLE_FILENAME,
                ".ori-cancel-dev.lock",
                ".post-cancel-auto-start.lock",
            ):
                entry.unlink(missing_ok=True)

    def _ensure_open_dev_cycle(self) -> None:
        res = resolve_active_dev_cycle()
        if res.cycle_status == "OPEN":
            return
        resp = self.client.post("/api/local-dev/dev-cycle/start", json={"confirmStart": "START"})
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))

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
        _ensure_e2e_backend_wip_markers(backend)
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

        import uuid

        client_op = uuid.uuid4().hex
        exec_resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={
                "confirmExecute": "CANCEL",
                "previewFingerprint": fp,
                "operationId": client_op,
            },
        )
        self.assertEqual(exec_resp.status_code, 200, exec_resp.get_data(as_text=True))
        self.assertTrue(exec_resp.get_json().get("ok"))
        self.assertEqual(exec_resp.get_json().get("operationId"), client_op)
        prog = load_public_progress(client_op, session_id_hint=sid)
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.get("cancelStatus"), "COMPLETED")
        self.assertEqual(prog.get("operationStatus"), "COMPLETED")
        step_ids = {s.get("id") for s in prog.get("completedSteps") or []}
        for expected in (
            "pre_checks",
            "backup_creation",
            "backend_restore",
            "frontend_restore",
            "env_restore",
            "final_verify",
            "close_cancel_cycle",
            "post_cancel_auto_start",
        ):
            self.assertIn(expected, step_ids, msg=f"missing progress step {expected}")

        self.assertFalse((backend / "api-e2e-wip.txt").exists())
        self.assertIn("SESSION_TEMP=pre-start", (backend / ".env.local").read_text(encoding="utf-8"))
        after_head = _git(["rev-parse", "HEAD"], backend)
        self.assertNotEqual(after_head, wip_head)
        _git(["rev-parse", "-q", "--verify", infra_head], backend)

        exec_body = exec_resp.get_json()
        auto = exec_body.get("postCancelAutoStart") or {}
        self.assertTrue(auto.get("ok"), exec_body)
        sid2 = auto.get("newSessionId")
        self.assertTrue(sid2)
        self.assertNotEqual(sid2, sid)

        res2 = resolve_active_dev_cycle()
        self.assertEqual(res2.cycle_status, "OPEN")
        self.assertEqual(res2.session_id, sid2)

        old_state = json.loads(
            (sessions / sid / "cancel-dev-cycle.state.json").read_text(encoding="utf-8-sig")
        )
        self.assertEqual(old_state.get("status"), "CANCELLED")
        self.assertTrue((sessions / sid / "cancel-dev-cycle.state.json").is_file())

    def test_full_cycle_no_product_changes_via_api(self) -> None:
        """Cancel Execute when preview has empty paths/commits (ENV-only drift since Start)."""
        backend = Path(self.layout["backendRoot"])
        frontend = Path(self.layout["frontendRoot"])
        sessions = Path(self.layout["sessionsRoot"])
        infra_head = self.layout["backendInfraHead"]

        resp = self.client.post("/api/local-dev/dev-cycle/start", json={"confirmStart": "START"})
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        sid = resp.get_json()["sessionId"]

        env_before = (backend / ".env.local").read_text(encoding="utf-8")
        _git(["checkout", "-q", "ori/builder1-wip"], backend)
        _ensure_e2e_backend_wip_markers(backend)
        (backend / ".env.local").write_text(
            env_before.replace("SESSION_TEMP=pre-start", "SESSION_TEMP=during-cycle-no-wip"),
            encoding="utf-8",
        )

        prev = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(prev.status_code, 200)
        plan = prev.get_json()
        self.assertTrue(plan.get("ok"))
        self.assertEqual(len(plan.get("commitsToCancel", {}).get("backend") or []), 0)
        paths_be = (plan.get("git") or {}).get("backend") or {}
        paths_list = paths_be.get("pathsToRevert") or paths_be.get("PathsToRevert") or []
        self.assertEqual(len(paths_list), 0)
        fp = plan.get("previewFingerprint")
        self.assertTrue(fp)

        exec_resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertEqual(exec_resp.status_code, 200, exec_resp.get_data(as_text=True))
        self.assertTrue(exec_resp.get_json().get("ok"))

        self.assertIn("SESSION_TEMP=pre-start", (backend / ".env.local").read_text(encoding="utf-8"))
        self.assertEqual(_git(["rev-parse", "HEAD"], backend), infra_head)
        _git(["rev-parse", "-q", "--verify", infra_head], backend)

        exec_body = exec_resp.get_json()
        auto = exec_body.get("postCancelAutoStart") or {}
        self.assertTrue(auto.get("ok"), exec_body)
        sid2 = auto.get("newSessionId")
        self.assertNotEqual(sid2, sid)

        res2 = resolve_active_dev_cycle()
        self.assertEqual(res2.cycle_status, "OPEN")
        self.assertEqual(res2.session_id, sid2)

        old_state = json.loads(
            (sessions / sid / "cancel-dev-cycle.state.json").read_text(encoding="utf-8-sig")
        )
        self.assertEqual(old_state.get("status"), "CANCELLED")

    def test_stale_fingerprint_rejected(self) -> None:
        self._ensure_open_dev_cycle()
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
        self._ensure_open_dev_cycle()
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
