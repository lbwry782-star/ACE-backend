"""Invoke fixed Ori local-dev PowerShell export scripts (no arbitrary commands)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def _run_ps1(script_name: str, args: List[str], *, timeout: int = 600) -> Dict[str, Any]:
    script = _scripts_dir() / script_name
    if not script.is_file():
        raise FileNotFoundError(f"ace_local_dev_script_missing:{script_name}")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        *args,
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if not stdout:
        raise RuntimeError(stderr or "ace_local_dev_powershell_empty_stdout")

    json_text = stdout
    if not json_text.startswith("{"):
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{"):
                json_text = stripped
                break

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        tail = (stdout + "\n" + stderr)[:800]
        raise RuntimeError(f"ace_local_dev_powershell_invalid_json:{tail}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ace_local_dev_powershell_invalid_payload")
    if completed.returncode != 0:
        if not data.get("ok"):
            return data
        raise RuntimeError(stderr or stdout or "ace_local_dev_powershell_failed")
    return data


def run_baseline_validate_json() -> Dict[str, Any]:
    return _run_ps1("Export-OriDevSessionBaselineJson.ps1", ["-ValidateOnly"], timeout=120)


def run_new_session_baseline_json(session_id: Optional[str] = None) -> Dict[str, Any]:
    args: List[str] = []
    if session_id:
        args.extend(["-SessionId", session_id])
    return _run_ps1("Export-OriDevSessionBaselineJson.ps1", args, timeout=600)


def run_cancel_execute_json(
    session_id: str,
    *,
    allow_real_execute: bool = False,
) -> Dict[str, Any]:
    args = ["-SessionId", session_id, "-ConfirmExecute", "CANCEL"]
    if allow_real_execute:
        args.append("-AllowRealExecute")
    return _run_ps1("Export-OriCancelDevExecuteJson.ps1", args, timeout=900)
