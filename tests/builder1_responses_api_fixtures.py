"""
OpenAI Responses API envelope fixtures for Builder1 image-compliance tests.

These are repository-defined SDK-SHAPED TEST FIXTURES that mirror the nesting used
by openai==2.30.0. They are NOT instances of openai.types.responses.Response.

For actual OpenAI SDK model instances, use tests.openai_sdk_test_helpers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def canonical_compliance_inner(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "reviewStatus": "completed",
        "hardViolations": [],
        "advisories": [],
        "evidence": [],
        "overallConfidence": "high",
    }
    payload.update(overrides)
    return payload


# Reconstructed from the pre-fix contract failure (missing legacy pass/reviewStatus).
# Not derived from a logged production envelope — only the inner JSON contract was known.
PRODUCTION_FAILURE_INNER: Dict[str, Any] = {
    "hardViolations": [],
    "advisories": [],
    "evidence": [],
    "overallConfidence": "high",
}


def legacy_compliance_inner(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "pass": True,
        "violations": [],
        "confidence": "high",
    }
    payload.update(overrides)
    return payload


@dataclass
class SdkShapedFixtureOutputText:
    """SDK-shaped test fixture only — not an OpenAI SDK class."""

    text: str = ""
    parsed: Any = None
    type: str = "output_text"
    annotations: List[Any] = field(default_factory=list)


@dataclass
class SdkShapedFixtureOutputMessage:
    """SDK-shaped test fixture only — not an OpenAI SDK class."""

    id: str
    content: List[SdkShapedFixtureOutputText]
    role: str = "assistant"
    status: str = "completed"
    type: str = "message"


@dataclass
class SdkShapedFixtureResponse:
    """SDK-shaped test fixture mirroring Response field layout — not an OpenAI SDK class."""

    id: str
    output: List[SdkShapedFixtureOutputMessage]
    object: str = "response"
    model: str = "gpt-4o"
    created_at: float = 1710000000.0
    parallel_tool_calls: bool = False
    status: str = "completed"

    @property
    def output_text(self) -> str:
        texts: List[str] = []
        for block in self.output:
            if block.type == "message":
                for content in block.content:
                    if content.type == "output_text" and content.text:
                        texts.append(content.text)
        return "".join(texts)


class EmptyOutputTextNestedFixture:
    """SDK-shaped test fixture where output_text is empty but nested content has text."""

    def __init__(self, *, output: List[SdkShapedFixtureOutputMessage]):
        self.id = "resp_empty_output_text"
        self.object = "response"
        self.model = "gpt-4o"
        self.created_at = 1710000000.0
        self.parallel_tool_calls = False
        self.status = "completed"
        self.output = output
        self.output_text = ""


# Deprecated aliases retained for existing tests.
SdkResponseOutputText = SdkShapedFixtureOutputText
SdkResponseOutputMessage = SdkShapedFixtureOutputMessage
SdkShapedResponse = SdkShapedFixtureResponse


def _message_with_text(text: str, *, parsed: Any = None) -> SdkShapedFixtureOutputMessage:
    return SdkShapedFixtureOutputMessage(
        id="msg_test",
        content=[SdkShapedFixtureOutputText(text=text, parsed=parsed)],
    )


def build_envelope_output_text_property(inner: Dict[str, Any]) -> SdkShapedFixtureResponse:
    text = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    return SdkShapedFixtureResponse(
        id="resp_output_text",
        output=[_message_with_text(text)],
    )


def build_envelope_nested_content(inner: Dict[str, Any]) -> SdkShapedFixtureResponse:
    text = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    return SdkShapedFixtureResponse(
        id="resp_nested",
        output=[_message_with_text(text)],
    )


def build_envelope_empty_output_text_nested(inner: Dict[str, Any]) -> EmptyOutputTextNestedFixture:
    text = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    return EmptyOutputTextNestedFixture(output=[_message_with_text(text)])


def build_envelope_strict_parsed_content(inner: Dict[str, Any]) -> SdkShapedFixtureResponse:
    return SdkShapedFixtureResponse(
        id="resp_strict_parsed",
        output=[_message_with_text(text="", parsed=inner)],
    )


def build_envelope_fenced_json_text(inner: Dict[str, Any]) -> SdkShapedFixtureResponse:
    text = "```json\n" + json.dumps(inner, ensure_ascii=False, indent=2) + "\n```"
    return SdkShapedFixtureResponse(
        id="resp_fenced",
        output=[_message_with_text(text)],
    )


def build_envelope_production_failure_inner() -> SdkShapedFixtureResponse:
    return build_envelope_nested_content(PRODUCTION_FAILURE_INNER)


def legacy_pass_gate_would_reject(data: dict) -> bool:
    return "pass" not in data or not isinstance(data.get("pass"), bool)


def image_bytes_from_create_kwargs(kwargs: Dict[str, Any]) -> Optional[str]:
    try:
        content = kwargs["input"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    for item in content:
        if isinstance(item, dict) and item.get("type") == "input_image":
            url = str(item.get("image_url") or "")
            if "base64," in url:
                return url.split("base64,", 1)[1]
    return None
