"""
Helpers for tests that require actual openai==2.30.0 SDK model instances.

These are NOT used by production code. Import failure here is a hard test-suite
failure: production-contract tests must not silently skip when the SDK is missing
or incompatible with the active interpreter.
"""
from __future__ import annotations

import json
from typing import Any, Dict

try:
    from openai.types.responses.response import Response
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
except Exception as exc:  # pragma: no cover - fail fast at import time
    raise ImportError(
        "openai==2.30.0 SDK response types are required for production-contract tests. "
        "Install openai==2.30.0 in a supported Python runtime (see .python-version)."
    ) from exc


def build_actual_openai_sdk_response(inner: Dict[str, Any]) -> Response:
    """Build a real openai.types.responses.Response via model_validate."""
    text = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    payload = {
        "id": "resp_actual_sdk",
        "object": "response",
        "created_at": 1710000000.0,
        "model": "gpt-4o",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "output": [
            {
                "id": "msg_actual_sdk",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
    }
    return Response.model_validate(payload)
