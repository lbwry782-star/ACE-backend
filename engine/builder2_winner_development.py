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
from engine.video_planning import validate_and_normalize_plan

logger = logging.getLogger(__name__)


def validate_winner_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    if raw.get("planningFailure"):
        raise Builder2TournamentError(str(raw.get("planningFailure")))
    if raw.get("schemaVersion") != WINNER_PLAN_SCHEMA_VERSION:
        raise Builder2TournamentError("builder2_winner_development_failed")
    require_non_empty_str(raw.get("productNameResolved"), field="productNameResolved")
    require_non_empty_str(raw.get("language"), field="language")
    require_non_empty_str(raw.get("problemPerception"), field="problemPerception")
    require_non_empty_str(raw.get("relativeAdvantage"), field="relativeAdvantage")
    require_non_empty_str(raw.get("prototypeId"), field="prototypeId")
    require_non_empty_str(raw.get("coreCreativeMechanism"), field="coreCreativeMechanism")
    require_non_empty_str(raw.get("visualParallelType"), field="visualParallelType")
    require_non_empty_str(raw.get("visualFamily"), field="visualFamily")
    structure = require_non_empty_str(raw.get("structureType"), field="structureType")
    require_non_empty_str(raw.get("headline"), field="headline")
    require_non_empty_str(raw.get("headlineCoreKeyword"), field="headlineCoreKeyword")
    require_non_empty_str(raw.get("coreVisualIdea"), field="coreVisualIdea")
    sequence = require_dict(raw.get("sequence"), field="sequence")
    for key in ("beginning", "development", "resolution"):
        require_non_empty_str(sequence.get(key), field=f"sequence.{key}")
    require_non_empty_str(raw.get("visualAnchor"), field="visualAnchor")
    require_non_empty_str(raw.get("openingFrameDescription"), field="openingFrameDescription")
    require_non_empty_str(raw.get("videoPrompt"), field="videoPrompt")
    variations = raw.get("sceneVariations")
    if not isinstance(variations, list):
        variations = []
    if structure == "variation_montage":
        if len(variations) < 2 or len(variations) > 4:
            raise Builder2TournamentError("builder2_winner_development_failed")
    if structure == "continuous_event" and not variations:
        variations = [sequence["beginning"], sequence["development"], sequence["resolution"]]
    raw = dict(raw)
    raw["sceneVariations"] = [str(v).strip() for v in variations if str(v).strip()]
    return raw


def normalize_winner_plan_for_runway(
    winner_plan: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Dict[str, Any]:
    """Map winner plan into legacy Builder2 runway plan shape."""
    headline_rem = (winner_plan.get("headline") or "").strip()
    pn = (winner_plan.get("productNameResolved") or product_name or "").strip()
    variations = winner_plan.get("sceneVariations") or []
    scene = " | ".join(variations) if variations else (
        f"{winner_plan['sequence']['beginning']} | "
        f"{winner_plan['sequence']['development']} | "
        f"{winner_plan['sequence']['resolution']}"
    )
    legacy = {
        "productNameResolved": pn,
        "headline": headline_rem,
        "headlineCoreKeyword": winner_plan.get("headlineCoreKeyword"),
        "coreVisualIdea": winner_plan.get("coreVisualIdea"),
        "sceneVariations": variations or [
            winner_plan["sequence"]["beginning"],
            winner_plan["sequence"]["development"],
            winner_plan["sequence"]["resolution"],
        ],
        "videoPrompt": winner_plan.get("videoPrompt"),
        "language": winner_plan.get("language") or content_language,
        "planInferenceMode": "builder2_tournament_winner_v1",
        "openingFrameDescription": winner_plan.get("openingFrameDescription"),
        "structureType": winner_plan.get("structureType"),
        "prototypeId": winner_plan.get("prototypeId"),
        "coreCreativeMechanism": winner_plan.get("coreCreativeMechanism"),
        "visualFamily": winner_plan.get("visualFamily"),
        "visualAnchor": winner_plan.get("visualAnchor"),
        "sequence": winner_plan.get("sequence"),
    }
    normalized, reason = validate_and_normalize_plan(
        legacy,
        product_name=product_name,
        product_description=product_description,
        content_language=content_language,
    )
    if not normalized:
        raise Builder2TournamentError(reason or "builder2_winner_development_failed")
    normalized["planInferenceMode"] = "builder2_tournament_winner_v1"
    normalized["structureType"] = winner_plan.get("structureType")
    normalized["prototypeId"] = winner_plan.get("prototypeId")
    normalized["coreCreativeMechanism"] = winner_plan.get("coreCreativeMechanism")
    normalized["visualFamily"] = winner_plan.get("visualFamily")
    normalized["visualAnchor"] = winner_plan.get("visualAnchor")
    normalized["sequence"] = winner_plan.get("sequence")
    normalized["openingFrameDescription"] = winner_plan.get("openingFrameDescription")
    return normalized


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
