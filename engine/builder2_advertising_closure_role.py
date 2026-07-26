"""
Builder2 advertising_closure role — isolated one-shot proposal generation.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import httpx
from openai import OpenAI

from engine.builder2_advertising_closure_proposal import parse_advertising_closure_response
from engine.builder2_reasoning_config import build_builder2_reasoning_payload, log_builder2_model_selected, resolve_builder2_reasoning_model
from engine import openai_retry


def call_advertising_closure_role(
    *,
    prompt: str,
    on_reasoning_submitted: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard

    AdvertisingClosureResumeGuard.assert_reasoning_call_allowed("advertising_closure")
    log_builder2_model_selected(role="advertising_closure", call_type="normal", attempt=1)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    timeout = float((os.environ.get("BUILDER2_TOURNAMENT_TIMEOUT_SECONDS") or "150").strip() or "150")
    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(timeout), max_retries=0)
    reasoning = build_builder2_reasoning_payload()
    AdvertisingClosureResumeGuard.record_reasoning_call_submitted("advertising_closure")
    if on_reasoning_submitted is not None:
        on_reasoning_submitted()
    response = openai_retry.openai_call_with_retry(
        lambda: client.responses.create(model=resolve_builder2_reasoning_model(), input=prompt, reasoning=reasoning),
        endpoint="responses",
    )
    text = getattr(response, "output_text", None) or ""
    return parse_advertising_closure_response(text)
