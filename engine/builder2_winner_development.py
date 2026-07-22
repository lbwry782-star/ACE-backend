"""
Builder2 winner development — convert winning candidate to Runway-compatible plan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import resolve_builder2_winner_model
from engine.builder2_tournament_contracts import (
    WINNER_PLAN_SCHEMA_VERSION,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import call_builder2_role_json
from engine.builder2_tournament_prompts import build_winner_development_prompt
from engine.builder2_winner_plan import validate_and_normalize_builder2_winner_plan, validate_builder2_winner_plan

logger = logging.getLogger(__name__)


def validate_winner_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    return validate_builder2_winner_plan(raw)


def normalize_winner_plan_for_runway(
    winner_plan: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Dict[str, Any]:
    return validate_and_normalize_builder2_winner_plan(
        winner_plan,
        product_name=product_name,
        product_description=product_description,
        content_language=content_language,
    )


def develop_builder2_winning_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    prototype_id: str,
    runway_mode: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    prototype = require_prototype(prototype_id)
    logger.info("BUILDER2_WINNER_DEVELOPMENT_START prototypeId=%s", prototype_id)
    prompt = build_winner_development_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        winning_candidate=winning_candidate,
        prototype=prototype,
        runway_mode=runway_mode,
    )
    raw = call_builder2_role_json(
        role="builder2_winner",
        model=resolve_builder2_winner_model(),
        prompt=prompt,
        llm_client=llm_client,
    )
    winner_plan = validate_winner_plan(raw)
    logger.info("BUILDER2_WINNER_DEVELOPMENT_OK prototypeId=%s", prototype_id)
    return winner_plan
