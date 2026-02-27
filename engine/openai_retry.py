"""
Shared helper for OpenAI API calls with 429 rate-limit retry and backoff.

Use for both Responses API (o3-pro) and Image generations (gpt-image-1.5).
- On HTTP 429: exponential backoff (2s, 4s, 8s, 16s) + jitter, respect Retry-After.
- Max 4 retries; then raise OpenAIRateLimitError for a clean API response.
"""

import logging
import random
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Backoff base delays in seconds (attempt 0 -> 2s, 1 -> 4s, 2 -> 8s, 3 -> 16s)
BACKOFF_BASE = [2, 4, 8, 16]
MAX_RETRIES = 4


class OpenAIRateLimitError(Exception):
    """Raised when 429 retries are exhausted. API should return 503 with ok:false, error:rate_limited."""

    def __init__(self, message: str = "Temporarily rate limited. Please retry."):
        self.message = message
        super().__init__(message)


def _is_429(exc: BaseException) -> bool:
    err_str = str(exc).lower()
    if "429" in str(exc) or "rate_limit" in err_str or "rate limit" in err_str or "quota" in err_str:
        return True
    code = getattr(exc, "status_code", None)
    if code is not None and int(code) == 429:
        return True
    return False


def _get_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Return Retry-After in seconds if present on the exception's response."""
    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None) or {}
        if isinstance(headers, dict):
            ra = headers.get("Retry-After") or headers.get("retry-after")
        else:
            ra = getattr(headers, "get", lambda k: None)("Retry-After") or getattr(headers, "get", lambda k: None)("retry-after")
        if ra is None:
            return None
        ra = str(ra).strip()
        if ra.isdigit():
            return float(ra)
        # Could parse HTTP-date here; for simplicity treat as 60s
        return 60.0
    except Exception:
        return None


T = TypeVar("T")


def openai_call_with_retry(
    fn: Callable[[], T],
    endpoint: str = "responses",
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Execute an OpenAI call with 429 retry and exponential backoff.

    Args:
        fn: Callable that performs the API call (e.g. lambda: client.responses.create(...)).
        endpoint: "responses" or "images" for logging.
        max_retries: Number of retries after initial attempt (default 4).

    Returns:
        Whatever fn() returns on success.

    Raises:
        OpenAIRateLimitError: If 429 persists after all retries.
        Other exceptions from fn() are re-raised (no retry for non-429).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_429(e) or attempt >= max_retries:
                if _is_429(e) and attempt >= max_retries:
                    logger.error(f"RATE_LIMIT_RETRY exhausted after {max_retries + 1} attempts endpoint={endpoint}")
                    raise OpenAIRateLimitError("Temporarily rate limited. Please retry.") from e
                raise

            # 429 and we have retries left
            retry_after = _get_retry_after_seconds(e)
            if retry_after is not None and retry_after > 0:
                wait_s = min(retry_after, 60.0)  # cap at 60s
            else:
                base = BACKOFF_BASE[attempt] if attempt < len(BACKOFF_BASE) else BACKOFF_BASE[-1]
                jitter = random.uniform(0, 1)
                wait_s = base + jitter

            logger.info(
                f"RATE_LIMIT_RETRY attempt={attempt + 1} wait_s={wait_s:.2f} endpoint={endpoint}"
            )
            time.sleep(wait_s)

    if last_exc is not None and _is_429(last_exc):
        raise OpenAIRateLimitError("Temporarily rate limited. Please retry.") from last_exc
    raise last_exc  # type: ignore[misc]
