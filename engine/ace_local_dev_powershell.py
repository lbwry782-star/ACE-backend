"""Invoke fixed Ori local-dev PowerShell export scripts (no arbitrary commands)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PS_JSON_UTF8 = "utf-8"


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def decode_powershell_stdout_bytes(raw: bytes | None) -> str:
    """Decode PowerShell stdout; Ori export scripts emit UTF-8 JSON."""
    if not raw:
        return ""
    for encoding in (_PS_JSON_UTF8, "utf-8-sig"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    try:
        text = raw.decode("cp1255")
    except UnicodeDecodeError as exc:
        raise RuntimeError("ace_local_dev_powershell_stdout_not_utf8") from exc
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ace_local_dev_powershell_stdout_not_utf8") from exc
    return text


def decode_powershell_stderr_bytes(raw: bytes | None) -> str:
    """Decode PowerShell stderr (diagnostics; may use Windows console code page)."""
    if not raw:
        return ""
    try:
        return raw.decode(_PS_JSON_UTF8)
    except UnicodeDecodeError:
        return raw.decode("cp1255")


def decode_powershell_captured_bytes(raw: bytes | None) -> str:
    """Backward-compatible alias for stdout decoding (tests)."""
    return decode_powershell_stdout_bytes(raw)


def run_powershell_script_file(
    script: Path,
    ps_args: List[str],
    *,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    if not script.is_file():
        raise FileNotFoundError(f"ace_local_dev_script_missing:{script.name}")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        *ps_args,
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env if env is not None else os.environ.copy(),
    )
    stdout = decode_powershell_stdout_bytes(completed.stdout)
    stderr = decode_powershell_stderr_bytes(completed.stderr)
    return completed.returncode, stdout, stderr


def _run_ps1(
    script_name: str,
    args: List[str],
    *,
    timeout: int = 600,
    env_overlay: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    script = _scripts_dir() / script_name
    env = os.environ.copy()
    if env_overlay:
        env.update(env_overlay)
    returncode, stdout, stderr = run_powershell_script_file(
        script,
        args,
        timeout=timeout,
        env=env,
    )
    stdout = stdout.strip()
    stderr = stderr.strip()
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
    if returncode != 0:
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
    progress_path: Optional[str] = None,
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    args = ["-SessionId", session_id, "-ConfirmExecute", "CANCEL"]
    if allow_real_execute:
        args.append("-AllowRealExecute")
    if progress_path:
        args.extend(["-ProgressPath", progress_path])
    if operation_id:
        args.extend(["-ProgressOperationId", operation_id])
    env_overlay: Dict[str, str] = {}
    if progress_path:
        env_overlay["ACE_CANCEL_EXECUTE_PROGRESS_PATH"] = progress_path
    if operation_id:
        env_overlay["ACE_CANCEL_EXECUTE_OPERATION_ID"] = operation_id
    return _run_ps1(
        "Export-OriCancelDevExecuteJson.ps1",
        args,
        timeout=900,
        env_overlay=env_overlay or None,
    )
