"""
Local development mode — explicit ACE_LOCAL_DEV_ENABLED with loopback bind and safe Redis.

Production (Render) must not set ACE_LOCAL_DEV_ENABLED. Security bypass (ACE_SECURITY_ENABLED=false)
is honored only when local dev is active and the process bind host is loopback.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOCAL_DEV_TRUE = frozenset({"true", "1", "yes", "on"})

_loopback_bind_confirmed: bool = False
_confirmed_bind_host: str = ""

# Host suffixes treated as production ACE / Render (block when local dev is on).
_PRODUCTION_PUBLIC_HOST_SUFFIXES = (
    ".onrender.com",
    "onrender.com",
    ".render.com",
    "render.com",
    "ace-advertising.agency",
    "www.ace-advertising.agency",
)


def _env_flag(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in _LOCAL_DEV_TRUE


def is_local_dev_enabled() -> bool:
    return _env_flag("ACE_LOCAL_DEV_ENABLED")


def confirm_local_dev_loopback_bind(host: str) -> None:
    """Call once at process startup with the Flask/gunicorn bind address."""
    global _loopback_bind_confirmed, _confirmed_bind_host
    normalized = (host or "").strip().lower()
    _confirmed_bind_host = normalized
    _loopback_bind_confirmed = is_loopback_bind_host(normalized)


def reset_local_dev_bind_confirmation_for_tests() -> None:
    global _loopback_bind_confirmed, _confirmed_bind_host
    _loopback_bind_confirmed = False
    _confirmed_bind_host = ""


def is_loopback_bind_confirmed() -> bool:
    return _loopback_bind_confirmed


def is_loopback_bind_host(host: str) -> bool:
    token = (host or "").strip().lower()
    if not token:
        return False
    if token in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        addr = ipaddress.ip_address(token)
        return bool(addr.is_loopback)
    except ValueError:
        return False


def is_loopback_redis_host(host: Optional[str]) -> bool:
    if host is None:
        return False
    token = host.strip().lower()
    if not token:
        return False
    if token in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        addr = ipaddress.ip_address(token)
        return bool(addr.is_loopback)
    except ValueError:
        pass
    # Docker Desktop / some local Redis images advertise service names — still resolve to loopback.
    try:
        for info in socket.getaddrinfo(token, None, type=socket.SOCK_STREAM):
            sockaddr = info[4]
            if not sockaddr:
                continue
            resolved = sockaddr[0]
            if is_loopback_redis_host(resolved):
                return True
    except OSError:
        return False
    return False


def redis_url_host(redis_url: str) -> str:
    parsed = urlparse((redis_url or "").strip())
    return (parsed.hostname or "").strip()


def classify_redis_url_for_local_dev(redis_url: str) -> Optional[str]:
    """
    Return an error code if redis_url must not be used under local dev, else None.
    Empty URL is allowed (Builder1 in-memory / no Redis).
    """
    url = (redis_url or "").strip()
    if not url:
        return None
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in {"redis", "rediss", "unix"}:
        return "ace_local_dev_redis_unsupported_scheme"
    if scheme == "unix":
        return "ace_local_dev_redis_unix_not_allowed"
    host = redis_url_host(url)
    if not host:
        return "ace_local_dev_redis_host_missing"
    if not is_loopback_redis_host(host):
        return "ace_local_dev_redis_not_loopback"
    return None


def classify_public_base_url_for_local_dev(public_base_url: str) -> Optional[str]:
    url = (public_base_url or "").strip()
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None
    for suffix in _PRODUCTION_PUBLIC_HOST_SUFFIXES:
        if host == suffix.lstrip(".") or host.endswith(suffix):
            return "ace_local_dev_public_base_url_production"
    return None


_SECURITY_DISABLED_VALUES = frozenset({"false", "0", "no", "off"})


def _security_env_explicitly_disabled() -> bool:
    raw = os.environ.get("ACE_SECURITY_ENABLED")
    if raw is None:
        return False
    cleaned = raw.strip().lower()
    return cleaned in _SECURITY_DISABLED_VALUES


def is_local_dev_security_bypass_allowed() -> bool:
    if not is_local_dev_enabled():
        return False
    if not is_loopback_bind_confirmed():
        return False
    return _security_env_explicitly_disabled()


def resolve_local_dev_bind_host() -> str:
    """Bind address for `python app.py` when local dev is enabled."""
    if is_local_dev_enabled():
        return "127.0.0.1"
    return (os.environ.get("ACE_BIND_HOST") or "0.0.0.0").strip() or "0.0.0.0"


def assert_local_dev_startup_safe() -> None:
    """
    Fail fast before serving traffic or opening Redis when local dev misconfigured.
    No-op when ACE_LOCAL_DEV_ENABLED is not set.
    """
    if not is_local_dev_enabled():
        return

    if not is_loopback_bind_confirmed():
        raise RuntimeError("ace_local_dev_loopback_bind_required")

    redis_err = classify_redis_url_for_local_dev(os.environ.get("REDIS_URL") or "")
    if redis_err:
        raise RuntimeError(redis_err)

    pub_err = classify_public_base_url_for_local_dev(os.environ.get("ACE_PUBLIC_BASE_URL") or "")
    if pub_err:
        raise RuntimeError(pub_err)

    if _env_flag("BUILDER1_PRODUCTION_MODE"):
        raise RuntimeError("ace_local_dev_builder1_production_mode_forbidden")


def assert_local_dev_redis_url_allowed(redis_url: str) -> None:
    if not is_local_dev_enabled():
        return
    code = classify_redis_url_for_local_dev(redis_url)
    if code:
        raise RuntimeError(code)


def log_local_dev_config_on_startup() -> None:
    if not is_local_dev_enabled():
        return
    logger.info(
        "ACE_LOCAL_DEV enabled=true bindConfirmed=%s bindHost=%s securityBypassAllowed=%s",
        is_loopback_bind_confirmed(),
        _confirmed_bind_host or "(unset)",
        is_local_dev_security_bypass_allowed(),
    )


def enable_local_dev_security_bypass_for_tests() -> None:
    """Unit tests: simulate verified local dev + loopback bind."""
    os.environ["ACE_LOCAL_DEV_ENABLED"] = "true"
    confirm_local_dev_loopback_bind("127.0.0.1")


def reset_local_dev_config_for_tests() -> None:
    os.environ.pop("ACE_LOCAL_DEV_ENABLED", None)
    reset_local_dev_bind_confirmation_for_tests()
