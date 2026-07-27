"""
Builder2 Winner continuous-event sceneVariations normalization — deterministic pre-validation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

CONTINUOUS_EVENT_STRUCTURE = "continuous_event"
NORMALIZATION_REASON = "continuous_event_sequence_is_authoritative"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def describe_scene_variations_metadata(plan: Dict[str, Any]) -> Dict[str, Any]:
    key_existed = "sceneVariations" in plan
    original = plan.get("sceneVariations")
    if original is None:
        value_type = "null" if key_existed else "missing"
    elif isinstance(original, list):
        value_type = "list"
    else:
        value_type = type(original).__name__
    return {
        "keyExisted": key_existed,
        "originalValueType": value_type,
        "originalListCount": len(original) if isinstance(original, list) else None,
    }


def normalize_continuous_event_scene_variations_for_execution(
    plan: Dict[str, Any],
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> bool:
    """
    For continuous_event Winner plans, clear sceneVariations before base validation.

    Returns True when normalization was applied.
    """
    structure = _clean(plan.get("structureType"))
    if structure != CONTINUOUS_EVENT_STRUCTURE:
        return False

    metadata = describe_scene_variations_metadata(plan)
    plan["sceneVariations"] = []

    logger.info(
        "BUILDER2_CONTINUOUS_EVENT_SCENE_VARIATIONS_NORMALIZED jobId=%s tournamentId=%s candidateId=%s "
        "prototypeId=%s structureType=%s keyExisted=%s originalValueType=%s originalListCount=%s "
        "normalizedListCount=%s normalizationReason=%s",
        job_id or "(none)",
        tournament_id or "(none)",
        candidate_id or "(none)",
        prototype_id or "(none)",
        structure,
        str(metadata["keyExisted"]).lower(),
        metadata["originalValueType"],
        metadata["originalListCount"] if metadata["originalListCount"] is not None else "(none)",
        0,
        NORMALIZATION_REASON,
    )
    return True
