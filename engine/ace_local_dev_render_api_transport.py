"""Transport layer for Render API — injectable for tests (no network in unit tests)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

from engine.ace_local_dev_render_api_types import RENDER_API_BASE_URL

_AUTH_RE = re.compile(r"(?i)(authorization|bearer)\s*[:=]?\s*[^\s,]+")


@dataclass(frozen=True)
class RenderApiResponse:
    status_code: int
    headers: Mapping[str, str]
    body: Any
    raw_text: str = ""


@dataclass
class RenderApiRequestRecord:
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    json_body: Optional[Dict[str, Any]] = None


class RenderApiTransport(Protocol):
    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Optional[Dict[str, Any]] = None,
        timeout_seconds: float,
    ) -> RenderApiResponse:
        ...


def redact_sensitive_text(text: str) -> str:
    if not text:
        return text
    return _AUTH_RE.sub("[redacted]", text)


class UrllibRenderApiTransport:
    """Live HTTP transport — not used in unit tests."""

    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Optional[Dict[str, Any]] = None,
        timeout_seconds: float,
    ) -> RenderApiResponse:
        data: Optional[bytes] = None
        req_headers = dict(headers)
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(
            url,
            data=data,
            headers=req_headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                parsed: Any = None
                if raw.strip():
                    parsed = json.loads(raw)
                return RenderApiResponse(
                    status_code=int(resp.status),
                    headers={k: v for k, v in resp.headers.items()},
                    body=parsed,
                    raw_text=raw,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed: Any = None
            if raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
            return RenderApiResponse(
                status_code=int(exc.code),
                headers={k: v for k, v in exc.headers.items()},
                body=parsed,
                raw_text=raw,
            )
        except TimeoutError as exc:
            raise TimeoutError("ace_render_api_transport_timeout") from exc


class MockRenderApiTransport:
    """In-memory transport for deterministic tests."""

    def __init__(self) -> None:
        self.records: list[RenderApiRequestRecord] = []
        self.handlers: list[Any] = []
        self._default_handler: Optional[Any] = None

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    def set_default_handler(self, handler) -> None:
        self._default_handler = handler

    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Optional[Dict[str, Any]] = None,
        timeout_seconds: float,
    ) -> RenderApiResponse:
        safe_headers: Dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() == "authorization":
                safe_headers[key] = "[redacted]"
            else:
                safe_headers[key] = redact_sensitive_text(value)
        self.records.append(
            RenderApiRequestRecord(
                method=method.upper(),
                url=url,
                headers=safe_headers,
                json_body=json_body,
            )
        )
        for handler in self.handlers:
            result = handler(method.upper(), url, headers=headers, json_body=json_body)
            if result is not None:
                return result
        if self._default_handler is not None:
            result = self._default_handler(method.upper(), url, headers=headers, json_body=json_body)
            if result is not None:
                return result
        return RenderApiResponse(status_code=500, headers={}, body={"message": "no mock handler"})


def build_render_api_url(path: str, query: Optional[Mapping[str, str]] = None) -> str:
    base = RENDER_API_BASE_URL.rstrip("/")
    token = (path or "").strip()
    if not token.startswith("/"):
        token = f"/{token}"
    url = f"{base}{token}"
    if query:
        params = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        if params:
            url = f"{url}?{params}"
    return url
