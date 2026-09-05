"""
Cryptographic token helpers for ACE payment security — never log raw secrets.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional


def sha256_hex(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8", errors="replace")).hexdigest()


def generate_browser_token() -> str:
    """256-bit browser authorization token (hex). Returned once; store hash only."""
    return secrets.token_hex(32)


def generate_payment_reference() -> str:
    """Separate high-entropy iCount correlation reference."""
    return secrets.token_urlsafe(32)


def constant_time_hex_equal(left: str, right: str) -> bool:
    left_clean = (left or "").strip().lower()
    right_clean = (right or "").strip().lower()
    if not left_clean or not right_clean:
        return False
    if len(left_clean) != len(right_clean):
        return False
    return hmac.compare_digest(left_clean, right_clean)


def verify_browser_token(raw_token: str, stored_hash: str) -> bool:
    if not raw_token or not stored_hash:
        return False
    return constant_time_hex_equal(sha256_hex(raw_token), stored_hash)


def verify_icount_ipn_secret(raw_header: str, expected_sha256: str) -> bool:
    if not raw_header or not expected_sha256:
        return False
    computed = sha256_hex(raw_header)
    return constant_time_hex_equal(computed, expected_sha256.strip().lower())


def load_icount_ipn_secret_sha256_from_env() -> Optional[str]:
    import os

    raw = (os.environ.get("ACE_ICOUNT_IPN_SECRET_SHA256") or "").strip().lower()
    if not raw:
        return None
    if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
        return None
    return raw
