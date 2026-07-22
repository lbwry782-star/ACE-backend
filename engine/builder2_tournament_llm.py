"""
Builder2 tournament model-call helper — JSON responses only.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, Optional

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty_response")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError("no_json_object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("json_not_object")
    return parsed


def call_builder2_role_json(
    *,
    role: str,
    model: str,
    prompt: str,
    call_type: str = "normal",
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    log_builder2_model_selected(role=role, call_type=call_type)
    if llm_client is not None:
        raw = llm_client(role=role, model=model, prompt=prompt)
        if isinstance(raw, dict):
            return raw
        return _extract_json_object(str(raw))

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    timeout = float((os.environ.get("BUILDER2_TOURNAMENT_TIMEOUT_SECONDS") or "150").strip() or "150")
    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(timeout), max_retries=0)
    reasoning = build_builder2_reasoning_payload()
    response = openai_retry.openai_call_with_retry(
        lambda: client.responses.create(model=model, input=prompt, reasoning=reasoning),
        endpoint="responses",
    )
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return _extract_json_object("\n".join(chunks))
