"""
Builder2 Judge role — scoring, validation, and purity checks.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import resolve_builder2_judge_model
from engine.builder2_tournament_contracts import (
    JUDGE_PURITY_FORBIDDEN_PATTERNS,
    JUDGE_SCORE_MAX_TOTAL,
    JUDGE_SCORE_RANGES,
    JUDGMENT_SCHEMA_VERSION,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import call_builder2_role_json
from engine.builder2_tournament_prompts import build_judge_prompt

logger = logging.getLogger(__name__)


def calculate_judge_total(scores: Dict[str, int]) -> int:
    total = 0
    for name, (low, high) in JUDGE_SCORE_RANGES.items():
        value = scores.get(name)
        if not isinstance(value, int):
            raise Builder2TournamentError("builder2_judge_invalid_response")
        if value < low or value > high:
            raise Builder2TournamentError("builder2_judge_invalid_response")
        total += value
    if total > JUDGE_SCORE_MAX_TOTAL:
        raise Builder2TournamentError("builder2_judge_invalid_response")
    return total


def validate_judge_purity(judgment: Dict[str, Any]) -> None:
    blob = json.dumps(judgment, ensure_ascii=False).lower()
    for pattern in JUDGE_PURITY_FORBIDDEN_PATTERNS:
        if pattern in blob:
            raise Builder2TournamentError("builder2_judge_purity_violation")


def validate_judge_response(
    judgment: Dict[str, Any],
    *,
    candidate_id: str,
) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
    if judgment.get("schemaVersion") != JUDGMENT_SCHEMA_VERSION:
        raise Builder2TournamentError("builder2_judge_invalid_response")
    if require_non_empty_str(judgment.get("candidateId"), field="candidateId") != candidate_id:
        raise Builder2TournamentError("builder2_judge_invalid_response")
    eligible = judgment.get("eligible")
    if not isinstance(eligible, bool):
        raise Builder2TournamentError("builder2_judge_invalid_response")
    disqualifiers = judgment.get("disqualifiers")
    if not isinstance(disqualifiers, list):
        raise Builder2TournamentError("builder2_judge_invalid_response")
    scores_raw = require_dict(judgment.get("scores"), field="scores")
    if "total" in scores_raw or "totalScore" in judgment:
        raise Builder2TournamentError("builder2_judge_invalid_response")
    scores: Dict[str, int] = {}
    for name in JUDGE_SCORE_RANGES:
        value = scores_raw.get(name)
        if not isinstance(value, int):
            raise Builder2TournamentError("builder2_judge_invalid_response")
        scores[name] = value
    total = calculate_judge_total(scores)
    require_non_empty_str(judgment.get("verdict"), field="verdict")
    strengths = judgment.get("strengths")
    weaknesses = judgment.get("weaknesses")
    if not isinstance(strengths, list) or not isinstance(weaknesses, list):
        raise Builder2TournamentError("builder2_judge_invalid_response")
    require_non_empty_str(judgment.get("prototypeQualityComparison"), field="prototypeQualityComparison")
    confidence = judgment.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise Builder2TournamentError("builder2_judge_invalid_response")
    validate_judge_purity(judgment)
    if not eligible and not disqualifiers:
        disqualifiers = ["ineligible_without_reason"]
        judgment = dict(judgment)
        judgment["disqualifiers"] = disqualifiers
    return judgment, total, scores


def judge_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype_id: str,
    candidate_id: str,
    candidate: Dict[str, Any],
    llm_client: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
    prototype = require_prototype(prototype_id)
    prompt = build_judge_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        prototype=prototype,
        candidate=candidate,
        candidate_id=candidate_id,
    )
    judgment_id = f"judge-{candidate_id}-{uuid.uuid4().hex[:8]}"
    logger.info(
        "BUILDER2_JUDGE_START candidateId=%s judgmentId=%s prototypeId=%s",
        candidate_id,
        judgment_id,
        prototype_id,
    )
    raw = call_builder2_role_json(
        role="builder2_judge",
        model=resolve_builder2_judge_model(),
        prompt=prompt,
        llm_client=llm_client,
    )
    judgment, total, scores = validate_judge_response(raw, candidate_id=candidate_id)
    logger.info(
        "BUILDER2_JUDGE_OK candidateId=%s judgmentId=%s eligible=%s total=%s",
        candidate_id,
        judgment_id,
        str(judgment.get("eligible")).lower(),
        total,
    )
    return judgment_id, judgment, total, scores
