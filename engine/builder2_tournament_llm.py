"""
Builder2 tournament model-call helper — JSON responses only.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_responses_output_text(response: Any) -> str:
    """
    Extract assistant text from an OpenAI Responses API result.
    Compatible with reasoning models that expose output_text parts.
    """
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    for block in getattr(response, "output", None) or []:
        contents = getattr(block, "content", None)
        if contents is None and isinstance(block, dict):
            contents = block.get("content")
        if not contents:
            continue
        for content in contents:
            content_type = getattr(content, "type", None) if not isinstance(content, dict) else content.get("type")
            if content_type != "output_text":
                continue
            text = getattr(content, "text", None) if not isinstance(content, dict) else content.get("text")
            if text:
                chunks.append(str(text))
                continue
            parsed = getattr(content, "parsed", None) if not isinstance(content, dict) else content.get("parsed")
            if parsed is None:
                continue
            if isinstance(parsed, str):
                chunks.append(parsed)
            else:
                chunks.append(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))

    combined = "".join(chunks).strip()
    if combined:
        return combined

    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        if isinstance(output_parsed, str):
            return output_parsed.strip()
        return json.dumps(output_parsed, ensure_ascii=False, separators=(",", ":"))
    return ""


def parse_json_object(text: str) -> Dict[str, Any]:
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


def _extract_json_object(text: str) -> Dict[str, Any]:
    return parse_json_object(text)


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
    text = extract_responses_output_text(response)
    return _extract_json_object(text)
