"""Render API errors — no secrets or Authorization values in messages."""
from __future__ import annotations

from typing import Any, Optional


class RenderApiError(Exception):
    def __init__(
        self,
        code: str,
        *,
        message: Optional[str] = None,
        http_status: Optional[int] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code
        self.http_status = http_status
        self.details = details


class RenderApiMissingApiKeyError(RenderApiError):
    def __init__(self) -> None:
        super().__init__("ace_render_api_key_missing", message="Render API key is not configured.")


class RenderApiParseError(RenderApiError):
    def __init__(self, message: str = "ace_render_api_response_invalid") -> None:
        super().__init__("ace_render_api_response_invalid", message=message)


class RenderApiHttpError(RenderApiError):
    pass


class RenderApiUnauthorizedError(RenderApiHttpError):
    def __init__(self, message: str = "ace_render_api_unauthorized") -> None:
        super().__init__(message, http_status=401)


class RenderApiForbiddenError(RenderApiHttpError):
    def __init__(self, message: str = "ace_render_api_forbidden") -> None:
        super().__init__(message, http_status=403)


class RenderApiNotFoundError(RenderApiHttpError):
    def __init__(self, message: str = "ace_render_api_not_found") -> None:
        super().__init__(message, http_status=404)


class RenderApiConflictError(RenderApiHttpError):
    def __init__(self, message: str = "ace_render_api_conflict") -> None:
        super().__init__(message, http_status=409)


class RenderApiRateLimitError(RenderApiHttpError):
    def __init__(
        self,
        *,
        retry_after_seconds: Optional[int] = None,
        message: str = "ace_render_api_rate_limited",
    ) -> None:
        super().__init__(message, http_status=429, details={"retryAfter": retry_after_seconds})
        self.retry_after_seconds = retry_after_seconds


class RenderApiUnavailableError(RenderApiHttpError):
    def __init__(self, message: str = "ace_render_api_unavailable") -> None:
        super().__init__(message, http_status=503)


class RenderPublishDeployError(RenderApiError):
    pass


class RenderDeployFailedError(RenderPublishDeployError):
    def __init__(self, deploy_id: str, status: str) -> None:
        super().__init__(
            "ace_render_deploy_failed",
            message=f"Deploy ended with status {status}.",
            details={"deployId": deploy_id, "status": status},
        )


class RenderDeployTimeoutError(RenderPublishDeployError):
    def __init__(self, deploy_id: str) -> None:
        super().__init__(
            "ace_render_deploy_timeout",
            message="Timed out waiting for deploy to finish.",
            details={"deployId": deploy_id},
        )


class RenderDeployVerifyError(RenderPublishDeployError):
    pass
