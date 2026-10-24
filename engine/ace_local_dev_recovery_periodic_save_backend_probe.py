"""
Local backend health probe for recovery periodic save (phase 5.8).

GET /health on loopback only — read-only, short timeout, no side effects.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from engine.ace_local_dev_publish_candidate_registry import resolve_ace_root
from engine.ace_local_dev_startup_recovery_check import read_managed_registry

logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"
DEFAULT_BACKEND_PORT = 5000
DEFAULT_PROBE_TIMEOUT_SECONDS = 2.0
RUNTIME_CONFIG_FILENAME = "recovery-periodic-save-runtime.json"
PROBE_FIXTURE_ENV = "ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"
PROBE_URL_OVERRIDE_ENV = "ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_loopback_host(host: str) -> bool:
    token = (host or "").strip().lower()
    if token in _LOOPBACK_HOSTS:
        return True
    if token.startswith("127."):
        return True
    return False


def validate_probe_url(url: str) -> Tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "probe_url_invalid"
    if parsed.scheme not in ("http", "https"):
        return False, "probe_url_scheme_blocked"
    if not _is_loopback_host(parsed.hostname or ""):
        return False, "probe_url_non_localhost_blocked"
    if parsed.path != HEALTH_PATH:
        return False, "probe_url_path_not_health"
    return True, "ok"


def read_periodic_save_runtime_config(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = resolve_local_dev_state_dir(ace_root) / RUNTIME_CONFIG_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_local_backend_port(ace_root: Optional[Path] = None) -> int:
    env_port = (os.environ.get("PORT") or "").strip()
    if env_port.isdigit():
        return int(env_port)
    root = ace_root if ace_root is not None else resolve_ace_root()
    runtime = read_periodic_save_runtime_config(root)
    if runtime:
        try:
            port = int(runtime.get("backendPort") or 0)
        except (TypeError, ValueError):
            port = 0
        if 1 <= port <= 65535:
            return port
    return DEFAULT_BACKEND_PORT


def resolve_local_backend_health_url(ace_root: Optional[Path] = None) -> str:
    override = (os.environ.get(PROBE_URL_OVERRIDE_ENV) or "").strip()
    if override:
        ok, reason = validate_probe_url(override)
        if not ok:
            raise ValueError(reason)
        return override
    root = ace_root if ace_root is not None else resolve_ace_root()
    port = resolve_local_backend_port(root)
    _, corrupt = read_managed_registry(root)
    if corrupt:
        logger.warning("RECOVERY_BACKEND_PROBE_MANAGED_REGISTRY_CORRUPT ace_root=%s", root)
    return f"http://127.0.0.1:{port}{HEALTH_PATH}"


def _probe_fixture(name: str) -> Dict[str, Any]:
    endpoint = f"http://127.0.0.1:{DEFAULT_BACKEND_PORT}{HEALTH_PATH}"
    base: Dict[str, Any] = {
        "checkedAt": _utc_now_iso(),
        "endpoint": endpoint,
        "source": "probe_fixture",
        "fixture": name,
    }
    if name == "ok":
        return {**base, "ok": True, "reachable": True, "reason": "", "httpStatus": 200}
    if name == "unreachable":
        return {**base, "ok": False, "reachable": False, "reason": "backend_probe_connection_refused"}
    if name == "timeout":
        return {**base, "ok": False, "reachable": False, "reason": "backend_probe_timeout"}
    if name == "http500":
        return {
            **base,
            "ok": False,
            "reachable": False,
            "reason": "backend_probe_http_500",
            "httpStatus": 500,
        }
    if name == "bad_body":
        return {
            **base,
            "ok": False,
            "reachable": False,
            "reason": "backend_probe_bad_body",
            "httpStatus": 200,
        }
    if name == "wrong_endpoint":
        return {
            **base,
            "ok": False,
            "reachable": False,
            "reason": "backend_probe_http_404",
            "httpStatus": 404,
            "endpoint": f"http://127.0.0.1:{DEFAULT_BACKEND_PORT}/not-health",
        }
    if name == "non_localhost":
        return {
            **base,
            "ok": False,
            "reachable": False,
            "reason": "probe_url_non_localhost_blocked",
            "endpoint": "http://203.0.113.1/health",
        }
    return {**base, "ok": False, "reachable": False, "reason": f"probe_fixture_unknown:{name}"}


def probe_local_backend_health(
    *,
    ace_root: Optional[Path] = None,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Probe loopback /health. When probe was not performed successfully, ok=False.
    """
    url_override = (os.environ.get(PROBE_URL_OVERRIDE_ENV) or "").strip()
    fixture = (os.environ.get(PROBE_FIXTURE_ENV) or "").strip()
    if fixture and not url_override:
        return _probe_fixture(fixture)

    try:
        url = resolve_local_backend_health_url(ace_root)
    except ValueError as exc:
        payload = {
            "ok": False,
            "reachable": False,
            "reason": str(exc),
            "endpoint": "",
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=%s", payload["reason"])
        return payload

    ok_url, url_reason = validate_probe_url(url)
    if not ok_url:
        payload = {
            "ok": False,
            "reachable": False,
            "reason": url_reason,
            "endpoint": url,
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=%s", url_reason)
        return payload

    req = urllib.request.Request(url, method="GET", headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            body = resp.read(128).decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        status = int(exc.code or 0)
        reason = f"backend_probe_http_{status}" if status else "backend_probe_http_error"
        payload = {
            "ok": False,
            "reachable": False,
            "reason": reason,
            "endpoint": url,
            "httpStatus": status,
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=%s endpoint=%s", reason, url)
        return payload
    except urllib.error.URLError as exc:
        reason = "backend_probe_timeout" if isinstance(exc.reason, socket.timeout) else "backend_probe_connection_refused"
        payload = {
            "ok": False,
            "reachable": False,
            "reason": reason,
            "endpoint": url,
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
            "detail": str(exc.reason)[:120],
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=%s endpoint=%s", reason, url)
        return payload
    except Exception as exc:
        payload = {
            "ok": False,
            "reachable": False,
            "reason": "backend_probe_error",
            "endpoint": url,
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
            "detail": str(exc)[:120],
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=backend_probe_error endpoint=%s", url)
        return payload

    if status != 200 or body != "ok":
        reason = "backend_probe_bad_body" if status == 200 else f"backend_probe_http_{status}"
        payload = {
            "ok": False,
            "reachable": False,
            "reason": reason,
            "endpoint": url,
            "httpStatus": status,
            "bodyPreview": body[:32],
            "source": "backend_probe",
            "checkedAt": _utc_now_iso(),
        }
        logger.info("RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_SKIP reason=%s endpoint=%s", reason, url)
        return payload

    return {
        "ok": True,
        "reachable": True,
        "reason": "",
        "endpoint": url,
        "httpStatus": status,
        "source": "backend_probe",
        "checkedAt": _utc_now_iso(),
    }


def _cli_main(argv: Optional[list] = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    ace_text = ""
    for idx, token in enumerate(args):
        if token == "--ace-root" and idx + 1 < len(args):
            ace_text = args[idx + 1]
    root = Path(ace_text).resolve() if ace_text else resolve_ace_root()
    payload = probe_local_backend_health(ace_root=root)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
