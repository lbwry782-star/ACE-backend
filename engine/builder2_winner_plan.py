"""
Builder2 winner-plan validation and normalization — isolated from legacy variation_montage planner.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from engine.builder2_tournament_contracts import (
    WINNER_PLAN_SCHEMA_VERSION,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.video_planning import (
    _assemble_headline_full,
    _headline_contains_core_keyword,
    _headline_depends_on_fixed_phrase,
    _headline_remainder_word_count,
    _is_weak_industry_keyword,
    _keyword_depends_on_headline_phrase,
    _MAX_HEADLINE_REMAINDER_WORDS,
    _plan_scene_variations_list,
    validate_and_normalize_plan,
)

logger = logging.getLogger(__name__)

_MONTAGE_LANGUAGE = re.compile(
    r"\b(montage|multiple clips|quick cuts|variation moments|cut between)\b",
    re.IGNORECASE,
)


def validate_builder2_winner_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
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

    out = dict(raw)
    variations = out.get("sceneVariations")
    if not isinstance(variations, list):
        variations = []

    if structure == "continuous_event":
        if not variations:
            out["sceneVariations"] = []
            out["sceneSequenceSemantics"] = "temporal_beats"
        else:
            cleaned = [str(v).strip() for v in variations if str(v).strip()]
            if len(cleaned) not in {0, 3}:
                cleaned = [sequence["beginning"], sequence["development"], sequence["resolution"]]
            out["sceneVariations"] = cleaned
            out["sceneSequenceSemantics"] = "temporal_beats"
        vp = out["videoPrompt"]
        if _MONTAGE_LANGUAGE.search(vp):
            raise Builder2TournamentError("builder2_winner_development_failed")
        logger.info("BUILDER2_CONTINUOUS_EVENT_PLAN_VALIDATED prototypeId=%s", out.get("prototypeId"))
    elif structure == "variation_montage":
        cleaned = [str(v).strip() for v in variations if str(v).strip()]
        if len(cleaned) < 2 or len(cleaned) > 4:
            raise Builder2TournamentError("builder2_winner_development_failed")
        families = {v.split()[0].lower() for v in cleaned if v}
        out["sceneVariations"] = cleaned
        out["sceneSequenceSemantics"] = "montage_variations"
    else:
        raise Builder2TournamentError("builder2_winner_development_failed")
    return out


def validate_and_normalize_builder2_winner_plan(
    winner_plan: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Dict[str, Any]:
    validated = validate_builder2_winner_plan(winner_plan)
    headline_rem = (validated.get("headline") or "").strip()
    pn = (validated.get("productNameResolved") or product_name or "").strip()
    structure = validated.get("structureType")
    sequence = validated.get("sequence") or {}
    variations = validated.get("sceneVariations") or []

    if structure == "continuous_event":
        if variations:
            scene = " → ".join(variations)
        else:
            scene = " → ".join([sequence["beginning"], sequence["development"], sequence["resolution"]])
        legacy_variations = variations or [
            sequence["beginning"],
            sequence["development"],
            sequence["resolution"],
        ]
    else:
        scene = " | ".join(variations)
        legacy_variations = variations

    legacy = {
        "productNameResolved": pn,
        "headline": headline_rem,
        "headlineCoreKeyword": validated.get("headlineCoreKeyword"),
        "coreVisualIdea": validated.get("coreVisualIdea"),
        "sceneVariations": legacy_variations,
        "videoPrompt": validated.get("videoPrompt"),
        "language": validated.get("language") or content_language,
        "planInferenceMode": "builder2_tournament_winner_v1",
        "openingFrameDescription": validated.get("openingFrameDescription"),
        "structureType": structure,
        "sceneSequenceSemantics": validated.get("sceneSequenceSemantics"),
        "prototypeId": validated.get("prototypeId"),
        "coreCreativeMechanism": validated.get("coreCreativeMechanism"),
        "visualFamily": validated.get("visualFamily"),
        "visualAnchor": validated.get("visualAnchor"),
        "sequence": sequence,
        "problemPerception": validated.get("problemPerception"),
        "relativeAdvantage": validated.get("relativeAdvantage"),
    }

    if structure == "variation_montage":
        normalized, reason = validate_and_normalize_plan(
            legacy,
            product_name=product_name,
            product_description=product_description,
            content_language=content_language,
        )
        if not normalized:
            raise Builder2TournamentError(reason or "builder2_winner_development_failed")
    else:
        normalized = _normalize_continuous_event_plan(
            legacy,
            product_name=product_name,
            product_description=product_description,
            content_language=content_language,
        )

    for key in (
        "structureType",
        "sceneSequenceSemantics",
        "prototypeId",
        "coreCreativeMechanism",
        "visualFamily",
        "visualAnchor",
        "sequence",
        "problemPerception",
        "relativeAdvantage",
        "openingFrameDescription",
    ):
        normalized[key] = validated.get(key)
    normalized["planInferenceMode"] = "builder2_tournament_winner_v1"
    return normalized


def _normalize_continuous_event_plan(
    legacy: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    content_language: str,
) -> Dict[str, Any]:
    data = dict(legacy)
    pn = (data.get("productNameResolved") or product_name or "").strip()
    headline_rem = (data.get("headline") or "").strip()
    core_kw = (data.get("headlineCoreKeyword") or "").strip()
    core_visual = (data.get("coreVisualIdea") or "").strip()
    video_prompt = (data.get("videoPrompt") or "").strip()
    sequence = data.get("sequence") or {}
    variations = data.get("sceneVariations") or []

    if not pn or not headline_rem or not core_kw or not core_visual or not video_prompt:
        raise Builder2TournamentError("builder2_winner_development_failed")
    if _headline_remainder_word_count(headline_rem) > _MAX_HEADLINE_REMAINDER_WORDS:
        raise Builder2TournamentError("planning_failed_headline_too_long")
    if _headline_depends_on_fixed_phrase(headline_rem):
        raise Builder2TournamentError("planning_failed_phrase_dependent_headline")
    if len(core_kw.split()) != 1 or _is_weak_industry_keyword(core_kw):
        raise Builder2TournamentError("planning_failed_invalid_keyword")
    if _keyword_depends_on_headline_phrase(headline_rem, core_kw):
        raise Builder2TournamentError("planning_failed_phrase_dependent_keyword")
    if not _headline_contains_core_keyword(headline_rem, core_kw):
        raise Builder2TournamentError("planning_failed_invalid_keyword")

    scene = data.get("sceneConcept") or " → ".join(
        variations
        or [sequence.get("beginning", ""), sequence.get("development", ""), sequence.get("resolution", "")]
    )
    pn_for_headline = (product_name or "").strip() or pn
    headline_full = _assemble_headline_full(pn_for_headline, headline_rem)
    opening_fd = (data.get("openingFrameDescription") or sequence.get("beginning") or core_visual)[:400]

    return {
        "productNameResolved": pn,
        "advertisingPromise": headline_rem,
        "headline": headline_rem,
        "headlineText": headline_full,
        "headlineTextRemainder": headline_rem,
        "headlineCoreKeyword": core_kw,
        "coreVisualIdea": core_visual,
        "sceneVariations": variations,
        "sceneConcept": scene,
        "videoPrompt": video_prompt,
        "videoPromptCore": video_prompt.strip(),
        "language": data.get("language") or content_language,
        "headlineDecision": "include_product_name",
        "planInferenceMode": "builder2_tournament_winner_v1",
        "openingFrameDescription": opening_fd,
        "structureType": "continuous_event",
        "sceneSequenceSemantics": data.get("sceneSequenceSemantics") or "temporal_beats",
        "sequence": sequence,
        "visualAnchor": data.get("visualAnchor"),
        "visualFamily": data.get("visualFamily"),
        "coreCreativeMechanism": data.get("coreCreativeMechanism"),
        "prototypeId": data.get("prototypeId"),
        "problemPerception": data.get("problemPerception"),
        "relativeAdvantage": data.get("relativeAdvantage"),
    }


def builder2_video_plan_struct_ok_for_runway(plan: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if not plan:
        return False, "no_plan"
    required = (
        "productNameResolved",
        "advertisingPromise",
        "headlineText",
        "headlineCoreKeyword",
        "coreVisualIdea",
        "sceneConcept",
        "videoPromptCore",
        "openingFrameDescription",
    )
    for key in required:
        if not str(plan.get(key) or "").strip():
            return False, f"missing_{key}"
    structure = (plan.get("structureType") or "").strip()
    variations = _plan_scene_variations_list(plan)
    if structure == "continuous_event":
        semantics = (plan.get("sceneSequenceSemantics") or "").strip()
        if semantics == "temporal_beats":
            if variations and len(variations) not in {0, 3}:
                return False, "invalid_continuous_sceneVariations"
        return True, ""
    if len(variations) < 2 or len(variations) > 4:
        return False, "invalid_sceneVariations"
    return True, ""
