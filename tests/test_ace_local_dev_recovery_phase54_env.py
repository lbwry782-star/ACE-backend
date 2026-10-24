"""Phase 5.4 — ENV DPAPI auto restore (temp only)."""
from __future__ import annotations

import json
import platform
import subprocess
import unittest
from pathlib import Path

from engine.ace_local_dev_recovery_auto_restore_engine import run_conditional_auto_recovery
from engine.ace_local_dev_recovery_env_dpapi import (
    restore_missing_env_from_checkpoint,
    validate_env_dpapi_blob,
)
from engine.ace_local_dev_recovery_startup_coordinator import COORDINATOR_AUTO_RECOVERY_ALLOWED, COORDINATOR_START_ALLOWED
from tests.recovery_checkpoint_fixtures import write_full_checkpoint
from tests.test_ace_local_dev_recovery_auto_restore_engine import _decision, _init_git_repo
from tests.test_ace_local_dev_recovery_startup_coordinator import CoordinatorTempLayout


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def _create_windows_dpapi_blob(dest: Path, plain: str) -> str:
    script = _scripts_dir() / "AceLocalRecoveryEnvRestoreLib.ps1"
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-CreateTestBlobPath",
            str(dest),
            "-CreateTestPlainText",
            plain,
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"dpapi_test_blob_failed:{err}")
    return (proc.stdout or b"").decode("utf-8", errors="replace").strip().lower()


@unittest.skipUnless(platform.system().lower() == "windows", "DPAPI tests require Windows")
class TestRecoveryPhase54EnvWindows(unittest.TestCase):
    def _layout(self) -> CoordinatorTempLayout:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        _init_git_repo(layout.ace_root / "ACE-Backend")
        _init_git_repo(layout.ace_root / "ACE-Frontend")
        return layout

    def test_env_exists_unchanged(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp54-exists"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "x.txt", "x\n")])
            blob_path = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "ENV" / "backend.env.local.dpapi"
            digest = _create_windows_dpapi_blob(blob_path, "ACE_TEST=1\n")
            manifest = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "MANIFEST.csv"
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if "backend.env.local" not in ln]
            lines.append(f"ENV,backend.env.local,env-dpapi,{digest}")
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            (be / ".env.local").write_bytes(b"KEEP=1\n")
            before = (be / ".env.local").read_bytes()
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual((be / ".env.local").read_bytes(), before)
            self.assertEqual(engine.env_restored, [])
        finally:
            layout.close()

    def test_env_missing_restored_via_dpapi(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp54-restore"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "x.txt", "x\n")])
            blob_path = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "ENV" / "backend.env.local.dpapi"
            digest = _create_windows_dpapi_blob(blob_path, "ACE_TEST=restored\n")
            manifest = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "MANIFEST.csv"
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if "backend.env.local" not in ln]
            lines.append(f"ENV,backend.env.local,env-dpapi,{digest}")
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            meta_path = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "checkpoint-metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["aceRoot"] = str(layout.ace_root)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            self.assertFalse((be / ".env.local").exists())
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertIn("backend.env.local", engine.env_restored)
            self.assertTrue((be / ".env.local").is_file())
            content = (be / ".env.local").read_text(encoding="utf-8")
            self.assertIn("ACE_TEST=restored", content)
            blob = json.dumps(engine.to_dict())
            self.assertNotIn("ACE_TEST=", blob)
        finally:
            layout.close()

    def test_corrupt_dpapi_hash_mismatch(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp54-corrupt"
            write_full_checkpoint(
                layout.ace_root,
                cp,
                [],
                env_blobs={"backend.env.local": b"not-valid-dpapi-bytes-xx"},
            )
            manifest = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp / "MANIFEST.csv"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "ENV,backend.env.local,env-dpapi,",
                    "ENV,backend.env.local,env-dpapi," + "a" * 64,
                ),
                encoding="utf-8",
            )
            cp_root = layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp
            blob = cp_root / "ENV" / "backend.env.local.dpapi"
            val = validate_env_dpapi_blob(blob, expected_blob_sha256="a" * 64)
            self.assertFalse(val["ok"])
            attempt = restore_missing_env_from_checkpoint(
                ace_root=layout.ace_root,
                backend_root=be,
                frontend_root=fe,
                logical_name="backend.env.local",
                dpapi_path=blob,
                expected_blob_sha256="a" * 64,
                checkpoint_root=cp_root,
            )
            self.assertFalse(attempt.ok)
            self.assertFalse((be / ".env.local").exists())
        finally:
            layout.close()

    def test_decrypt_failure_partial(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp54-fail"
            write_full_checkpoint(
                layout.ace_root,
                cp,
                [],
                env_blobs={"backend.env.local": b"\x01\x02\x03\x04" * 8},
            )
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
                decrypt_and_write=lambda **_: (False, "env_dpapi_decrypt_or_write_failed"),
            )
            self.assertTrue(engine.partial_recovery)
            self.assertFalse((be / ".env.local").exists())
        finally:
            layout.close()

    def test_start_allowed_no_env_restore(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_START_ALLOWED),
                backend_root=be,
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(engine.skipped)
        finally:
            layout.close()


class TestRecoveryPhase54EnvCrossPlatform(unittest.TestCase):
    def test_validate_blob_sha256(self) -> None:
        layout = CoordinatorTempLayout()
        try:
            cp = "cp54-x"
            root = write_full_checkpoint(
                layout.ace_root,
                cp,
                [],
                env_blobs={"backend.env.local": b"abc12345"},
            )
            blob = root / "ENV" / "backend.env.local.dpapi"
            ok = validate_env_dpapi_blob(blob, expected_blob_sha256="wrong")
            self.assertFalse(ok["ok"])
        finally:
            layout.close()


if __name__ == "__main__":
    unittest.main()
