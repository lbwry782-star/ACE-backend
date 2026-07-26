"""
Builder2 Advertising Closure proposal — isolated advertising_closure role.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from engine.builder2_advertising_closure_contract import (
    DEFAULT_CLOSURE_DURATION_SECONDS,
    DEFAULT_CLOSURE_PRESENTATION_MODE,
    normalize_advertising_closure,
    validate_advertising_closure_object,
)
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_tournament_contracts import Builder2TournamentError


def build_advertising_closure_authoritative_input(plan: Dict[str, Any]) -> Dict[str, Any]:
    relative_advantage = plan.get("relativeAdvantage")
    problem = plan.get("problemPerception")
    return {
        "productNameResolved": str(plan.get("productNameResolved") or "").strip(),
        "problemPerception": problem if isinstance(problem, dict) else str(problem or "").strip(),
        "relativeAdvantage": relative_advantage if isinstance(relative_advantage, dict) else str(relative_advantage or "").strip(),
        "prototypeId": str(plan.get("prototypeId") or "").strip(),
        "coreCreativeMechanism": str(plan.get("coreCreativeMechanism") or "").strip(),
        "coreVisualIdea": str(plan.get("coreVisualIdea") or "").strip(),
        "visualAnchor": plan.get("visualAnchor"),
        "language": str(plan.get("language") or "en").strip(),
        "advertisingPromise": str(plan.get("advertisingPromise") or "").strip(),
        "headlineDecision": get_normalized_headline_decision(plan),
    }


def build_advertising_closure_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Generate ONLY a JSON object for Builder2 Advertising Closure.\n"
        "Do not redesign the video, change the winner, invent a new strategic problem, "
        "change the relative advantage, call Runway, create images, or write marketing paragraphs.\n"
        "Return exactly these keys: productNameText, sloganText, language, presentationMode, durationSeconds.\n"
        "The slogan must derive from the relative advantage, reinforce the visual mechanism, stay memorable, "
        "contain no more than seven words excluding the product name, avoid unsupported superiority claims, "
        "not merely describe the visible action, and not introduce a new strategic promise.\n"
        "presentationMode must be end_card. durationSeconds must be 1.5.\n"
        "Do not include logos or invented brand marks.\n\n"
        f"Authoritative input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_advertising_closure_response(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise Builder2TournamentError("builder2_advertising_closure_invalid:response")
        data = json.loads(text[start : end + 1])
    else:
        raise Builder2TournamentError("builder2_advertising_closure_invalid:response")
    if not isinstance(data, dict):
        raise Builder2TournamentError("builder2_advertising_closure_invalid:response")
    return normalize_advertising_closure(
        {
            "required": True,
            "productNameText": data.get("productNameText"),
            "sloganText": data.get("sloganText"),
            "language": data.get("language"),
            "presentationMode": data.get("presentationMode") or DEFAULT_CLOSURE_PRESENTATION_MODE,
            "durationSeconds": data.get("durationSeconds") or DEFAULT_CLOSURE_DURATION_SECONDS,
            "headlineSource": "advertising_closure_role",
            "noLogo": True,
        }
    )


def generate_advertising_closure_proposal(
    plan: Dict[str, Any],
    *,
    llm_client: Optional[Callable[..., Any]] = None,
    on_reasoning_submitted: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    existing = plan.get("advertisingClosure")
    if isinstance(existing, dict) and str(existing.get("sloganText") or "").strip():
        return normalize_advertising_closure({**existing, "headlineSource": existing.get("headlineSource") or "persisted"})
    if headline_decision_requires_headline(get_normalized_headline_decision(plan)):
        built = normalize_advertising_closure(
            {
                "required": True,
                "productNameText": plan.get("productNameResolved"),
                "sloganText": plan.get("headlineTextRemainder") or plan.get("headline") or plan.get("headlineText"),
                "language": plan.get("language") or "en",
                "presentationMode": DEFAULT_CLOSURE_PRESENTATION_MODE,
                "durationSeconds": DEFAULT_CLOSURE_DURATION_SECONDS,
                "headlineSource": "winner_development",
                "noLogo": True,
            }
        )
        validate_advertising_closure_object(built, plan=plan)
        return built

    payload = build_advertising_closure_authoritative_input(plan)
    prompt = build_advertising_closure_prompt(payload)
    if llm_client is not None:
        from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard

        AdvertisingClosureResumeGuard.assert_reasoning_call_allowed("advertising_closure")
        AdvertisingClosureResumeGuard.record_reasoning_call_submitted("advertising_closure")
        if on_reasoning_submitted is not None:
            on_reasoning_submitted()
        raw = llm_client(role="advertising_closure", prompt=prompt)
        proposal = parse_advertising_closure_response(raw)
    else:
        from engine.builder2_advertising_closure_role import call_advertising_closure_role

        proposal = call_advertising_closure_role(prompt=prompt, on_reasoning_submitted=on_reasoning_submitted)
    validate_advertising_closure_object(proposal, plan=plan)
    return proposal
