"""
UTF-8 decoding for Ori local-dev PowerShell subprocess wrappers.

Run: python -m unittest tests.test_ace_local_dev_subprocess_utf8 -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from engine.ace_local_dev_cancel_preview import run_cancel_dev_preview_powershell
from engine.ace_local_dev_powershell import (
    _run_ps1,
    decode_powershell_captured_bytes,
    decode_powershell_stderr_bytes,
    decode_powershell_stdout_bytes,
    run_powershell_script_file,
)


class DecodePowerShellCapturedBytesTests(unittest.TestCase):
    def test_decodes_utf8_hebrew_path(self) -> None:
        payload = {"sessionRoot": "D:\\אס2\\ACE\\_backups\\sessions\\ori-test"}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        text = decode_powershell_captured_bytes(raw)
        self.assertIn("אס2", text)
        self.assertEqual(json.loads(text)["sessionRoot"], payload["sessionRoot"])

    def test_empty_and_none_bytes(self) -> None:
        self.assertEqual(decode_powershell_captured_bytes(None), "")
        self.assertEqual(decode_powershell_captured_bytes(b""), "")

    def test_invalid_utf8_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            decode_powershell_stdout_bytes(b"\xff\xfe")
        self.assertIn("ace_local_dev_powershell_stdout_not_utf8", str(ctx.exception))

    def test_stderr_cp1255_fallback(self) -> None:
        # Console diagnostics may be cp1255 while JSON stdout stays UTF-8.
        text = decode_powershell_stderr_bytes("שגיאה".encode("cp1255"))
        self.assertIn("ש", text)

    def test_stdout_cp1255_json_fallback(self) -> None:
        payload = {"ok": True, "sessionRoot": "D:\\אס2\\ACE"}
        raw = json.dumps(payload, ensure_ascii=False).encode("cp1255")
        text = decode_powershell_stdout_bytes(raw)
        self.assertEqual(json.loads(text)["sessionRoot"], payload["sessionRoot"])


class RunPs1EncodingTests(unittest.TestCase):
    @patch("engine.ace_local_dev_powershell.run_powershell_script_file")
    def test_run_ps1_parses_utf8_json_from_bytes_path(self, mock_capture: MagicMock) -> None:
        hebrew_json = json.dumps(
            {"ok": True, "result": {"status": "SUCCESS"}},
            ensure_ascii=False,
        ).encode("utf-8")
        # run_powershell_script_file returns decoded str; simulate Hebrew in stdout string
        mock_capture.return_value = (
            0,
            hebrew_json.decode("utf-8"),
            "",
        )
        data = _run_ps1("Export-OriCancelDevExecuteJson.ps1", ["-SessionId", "ori-20261023-010342-2d952e88"])
        self.assertTrue(data.get("ok"))

    @patch("engine.ace_local_dev_powershell.run_powershell_script_file")
    def test_run_ps1_invalid_json_raises(self, mock_capture: MagicMock) -> None:
        mock_capture.return_value = (0, "not-json", "")
        with self.assertRaises(RuntimeError) as ctx:
            _run_ps1("Export-OriCancelDevExecuteJson.ps1", [])
        self.assertIn("ace_local_dev_powershell_invalid_json", str(ctx.exception))

    @patch("engine.ace_local_dev_powershell.run_powershell_script_file")
    def test_run_ps1_empty_stdout_raises(self, mock_capture: MagicMock) -> None:
        mock_capture.return_value = (0, "", "")
        with self.assertRaises(RuntimeError) as ctx:
            _run_ps1("Export-OriCancelDevExecuteJson.ps1", [])
        self.assertIn("ace_local_dev_powershell_empty_stdout", str(ctx.exception))

    @patch("engine.ace_local_dev_powershell.subprocess.run")
    def test_subprocess_capture_decodes_utf8_bytes_not_locale(self, mock_run: MagicMock) -> None:
        payload = {"ok": True, "plan": {"sessionRoot": "D:\\אס2\\ACE"}}
        out_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        mock_run.return_value = MagicMock(returncode=0, stdout=out_bytes, stderr=b"")

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake.ps1"
            script.write_text("", encoding="utf-8")
            _rc, stdout, _stderr = run_powershell_script_file(script, ["-SessionId", "x"], timeout=10)

        self.assertIn("אס2", stdout)
        self.assertTrue(json.loads(stdout).get("ok"))


class CancelPreviewSubprocessTests(unittest.TestCase):
    @patch("engine.ace_local_dev_cancel_preview.run_powershell_script_file")
    def test_preview_hebrew_stdout_parses(self, mock_capture: MagicMock) -> None:
        plan = {
            "ok": True,
            "plan": {
                "sessionId": "ori-20261023-094907-078ee0d3",
                "canExecute": False,
                "blockers": ["REGISTRY_INFRA_HEAD_MISMATCH:frontend"],
                "cancelCycle": {"status": "OPEN"},
                "commitsToCancel": {"backend": [], "frontend": []},
                "git": {"backend": {}, "frontend": {}},
                "registryInfraDrift": [],
            },
        }
        body = json.dumps(plan, ensure_ascii=False)
        mock_capture.return_value = (0, body, "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Export-OriCancelDevPreviewJson.ps1").write_text("", encoding="utf-8")
            with patch("engine.ace_local_dev_cancel_preview._backend_scripts_dir", return_value=root):
                raw = run_cancel_dev_preview_powershell("ori-20261023-094907-078ee0d3")
        self.assertTrue(raw.get("ok"))

    @patch("engine.ace_local_dev_cancel_preview.run_powershell_script_file")
    def test_preview_non_utf8_raises_runtime_error(self, mock_capture: MagicMock) -> None:
        mock_capture.side_effect = RuntimeError("ace_local_dev_powershell_stdout_not_utf8")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Export-OriCancelDevPreviewJson.ps1").write_text("", encoding="utf-8")
            with patch("engine.ace_local_dev_cancel_preview._backend_scripts_dir", return_value=root):
                with self.assertRaises(RuntimeError):
                    run_cancel_dev_preview_powershell("ori-20261023-094907-078ee0d3")


if __name__ == "__main__":
    unittest.main()
