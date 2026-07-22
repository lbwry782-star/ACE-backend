"""
Builder2 Creator role — candidate generation and validation.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_prototypes import Builder2Prototype, require_prototype
from engine.builder2_tournament_config import resolve_builder2_creator_model
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CREATOR_PURITY_FORBIDDEN_PATTERNS,
    VALID_CONTINUITY_RISK,
    VALID_STRUCTURE_TYPES,
    VALID_VISUAL_PARALLEL_TYPES,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_llm import call_builder2_role_json
from engine.builder2_tournament_prompts import build_creator_prompt

logger = logging.getLogger(__name__)


def _contains_forbidden_purity(text: str, patterns: Tuple[str, ...]) -> Optional[str]:
    lowered = (text or "").lower()
    for pattern in patterns:
        if pattern in lowered:
            return pattern
    return None


def validate_creator_purity(candidate: Dict[str, Any]) -> None:
    blob = json.dumps(candidate, ensure_ascii=False).lower()
    hit = _contains_forbidden_purity(blob, CREATOR_PURITY_FORBIDDEN_PATTERNS)
    if hit:
        raise Builder2TournamentError("builder2_creator_purity_violation")


def validate_creator_candidate(
    candidate: Dict[str, Any],
    *,
    assigned_prototype_id: str,
) -> Dict[str, Any]:
    if candidate.get("planningFailure"):
        raise Builder2TournamentError(str(candidate.get("planningFailure")))
    if candidate.get("schemaVersion") != CANDIDATE_SCHEMA_VERSION:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")
    if require_non_empty_str(candidate.get("prototypeId"), field="prototypeId") != assigned_prototype_id:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")
    require_non_empty_str(candidate.get("prototypeMethodApplied"), field="prototypeMethodApplied")
    require_non_empty_str(candidate.get("coreCreativeMechanism"), field="coreCreativeMechanism")
    require_non_empty_str(candidate.get("conceptSummary"), field="conceptSummary")
    vpt = require_non_empty_str(candidate.get("visualParallelType"), field="visualParallelType")
    if vpt not in VALID_VISUAL_PARALLEL_TYPES:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")
    require_non_empty_str(candidate.get("visualFamily"), field="visualFamily")
    structure = require_non_empty_str(candidate.get("structureType"), field="structureType")
    if structure not in VALID_STRUCTURE_TYPES:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")

    seven = require_dict(candidate.get("sevenSecondStructure"), field="sevenSecondStructure")
    for key in ("beginning", "development", "resolution"):
        require_non_empty_str(seven.get(key), field=f"sevenSecondStructure.{key}")

    anchor = require_dict(candidate.get("visualAnchor"), field="visualAnchor")
    require_non_empty_str(anchor.get("description"), field="visualAnchor.description")
    require_non_empty_str(anchor.get("whyEssential"), field="visualAnchor.whyEssential")

    silent = require_dict(candidate.get("silentVerification"), field="silentVerification")
    if silent.get("understandableWithoutAudio") is not True:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")
    require_non_empty_str(silent.get("explanation"), field="silentVerification.explanation")

    runway = require_dict(candidate.get("runwayFeasibility"), field="runwayFeasibility")
    require_non_empty_str(runway.get("mainSubject"), field="runwayFeasibility.mainSubject")
    require_non_empty_str(runway.get("mainAction"), field="runwayFeasibility.mainAction")
    require_non_empty_str(runway.get("location"), field="runwayFeasibility.location")
    require_non_empty_str(runway.get("openingFrame"), field="runwayFeasibility.openingFrame")
    risk = require_non_empty_str(runway.get("continuityRisk"), field="runwayFeasibility.continuityRisk")
    if risk not in VALID_CONTINUITY_RISK:
        raise Builder2TournamentError("builder2_creator_invalid_candidate")

    editing = require_dict(candidate.get("editingPlan"), field="editingPlan")
    for key in ("purpose", "reveal", "pacing"):
        require_non_empty_str(editing.get(key), field=f"editingPlan.{key}")

    report = require_dict(candidate.get("creatorReport"), field="creatorReport")
    for key in (
        "problemPerception",
        "relativeAdvantage",
        "mechanismScanSummary",
        "goldPrototypeUsed",
        "visualParallelType",
        "whyParallelExpressesAdvantage",
        "whyRunwayShouldUnderstand",
    ):
        require_non_empty_str(report.get(key), field=f"creatorReport.{key}")

    forbidden_headline_keys = ("headline", "headlineText", "headlineCoreKeyword", "videoPrompt")
    for key in forbidden_headline_keys:
        if str(candidate.get(key) or "").strip():
            raise Builder2TournamentError("builder2_creator_invalid_candidate")

    if assigned_prototype_id == "think_small":
        blob = json.dumps(candidate, ensure_ascii=False).lower()
        if "weakness" not in blob and "weak" not in blob:
            raise Builder2TournamentError("builder2_creator_invalid_candidate")

    if assigned_prototype_id == "greenpeace_essential_pairing":
        blob = json.dumps(report, ensure_ascii=False).lower()
        if "appearance" in blob and "only" in blob:
            raise Builder2TournamentError("builder2_creator_invalid_candidate")

    if vpt == "context_collision":
        bridge_blob = json.dumps(report, ensure_ascii=False).lower()
        if "bridge" not in bridge_blob and "connect" not in bridge_blob:
            raise Builder2TournamentError("builder2_creator_invalid_candidate")

    validate_creator_purity(candidate)
    return candidate


def generate_creator_candidate(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    runway_mode: str,
    llm_client: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    prototype = require_prototype(prototype_id)
    candidate_id = f"cand-{round_index}-{prototype_id}-{attempt_number}-{uuid.uuid4().hex[:8]}"
    prompt = build_creator_prompt(
        product_name=product_name,
        product_description=product_description,
        language=language,
        strategy_foundation=strategy_foundation,
        prototype=prototype,
        candidate_id=candidate_id,
        attempt_number=attempt_number,
        runway_mode=runway_mode,
    )
    logger.info(
        "BUILDER2_CREATOR_START candidateId=%s prototypeId=%s roundIndex=%s attempt=%s",
        candidate_id,
        prototype_id,
        round_index,
        attempt_number,
    )
    raw = call_builder2_role_json(
        role="builder2_creator",
        model=resolve_builder2_creator_model(),
        prompt=prompt,
        llm_client=llm_client,
    )
    candidate = validate_creator_candidate(raw, assigned_prototype_id=prototype_id)
    logger.info("BUILDER2_CREATOR_OK candidateId=%s prototypeId=%s", candidate_id, prototype_id)
    return candidate_id, candidate
