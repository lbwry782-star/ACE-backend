"""
Builder1 strategy judge — validates campaign plan before image generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeAlias

logger = logging.getLogger(__name__)

JudgeModelCaller: TypeAlias = Callable[[str, str], object]

BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT = """
You are a strict advertising strategy auditor for Builder1 campaigns.

Return JSON only:
{
  "pass": true,
  "rejectionReasonCodes": [],
  "unsupportedClaimDetected": false,
  "strategicProblemReal": true,
  "relativeAdvantageSupported": true,
  "relativeAdvantageDistinctive": true,
  "conceptualCandidateCountSufficient": true,
  "conceptualGeneratorIsAction": true,
  "conceptualGeneratorIsNotObject": true,
  "conceptualGeneratorDirectlyExpressesAdvantage": true,
  "conceptualGeneratorSupportsSeries": true,
  "conceptualGeneratorExpressesAdvantage": true,
  "physicalGeneratorEmbodiesConcept": true,
  "physicalGeneratorWasDerivedFromConcept": true,
  "physicalGeneratorDidNotReplaceConcept": true,
  "graphicGeneratorConcrete": true,
  "seriesCoherent": true
}

Fail if:
- unsupported product capabilities are presented as facts (dashboards, guaranteed results, live reporting, sales attribution, transparency systems, automation, personal service) without brief support
- relative advantage is generic transparency/quality/trust/results
- conceptual generator is a theme, emotion, object, or equals physicalGenerator
- physical generator appears chosen before the conceptual action
- ads merely swap objects without performing the same conceptual action
- graphic generator lacks concrete renderable fields and recurring visible device
- conceptualGeneratorScan has fewer than 6 action-based candidates

Return structured JSON only.
""".strip()


@dataclass
class StrategyJudgeResult:
    passed: bool
    rejection_reason_codes: List[str]
    unsupported_claim_detected: bool = False
    raw: Dict[str, Any] | None = None


def _coerce_judge_dict(raw_payload: object) -> Dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no_json_object")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("judge_output_not_object")
        return obj
    raise ValueError("judge_output_not_object")


def build_strategy_judge_user_prompt(
    *,
    product_description: str,
    plan_dict: Dict[str, Any],
) -> str:
    strip_keys = {
        "strategyCandidateScan",
        "conceptualGeneratorScan",
        "campaignSelfCheck",
        "strategyJudgeResult",
    }
    public_plan = {k: v for k, v in plan_dict.items() if k not in strip_keys}
    return (
        f"Brief:\n{product_description.strip()}\n\n"
        f"Proposed campaign plan:\n{json.dumps(public_plan, ensure_ascii=False, indent=2)}\n\n"
        "Audit this plan. Return JSON only."
    )


def judge_builder1_strategy(
    *,
    product_description: str,
    plan_dict: Dict[str, Any],
    model_caller: JudgeModelCaller,
) -> StrategyJudgeResult:
    user_prompt = build_strategy_judge_user_prompt(
        product_description=product_description,
        plan_dict=plan_dict,
    )
    try:
        raw_payload = model_caller(BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT, user_prompt)
        data = _coerce_judge_dict(raw_payload)
    except Exception as exc:
        logger.error("BUILDER1_STRATEGY_JUDGE_FAIL stage=call err=%s", exc)
        return StrategyJudgeResult(
            passed=False,
            rejection_reason_codes=["judge_call_failed"],
        )

    passed = bool(data.get("pass"))
    codes = [str(c) for c in (data.get("rejectionReasonCodes") or []) if str(c).strip()]
    unsupported = bool(data.get("unsupportedClaimDetected"))
    if not passed and not codes:
        codes = ["strategy_judge_failed"]
    if passed:
        logger.info("BUILDER1_STRATEGY_JUDGE_PASS")
    else:
        logger.error("BUILDER1_STRATEGY_JUDGE_FAIL codes=%s unsupported=%s", codes, unsupported)
    return StrategyJudgeResult(
        passed=passed,
        rejection_reason_codes=codes,
        unsupported_claim_detected=unsupported,
        raw=data,
    )
