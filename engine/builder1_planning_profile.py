"""
Builder1 planning profile — stage-specific model and reasoning routing.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from engine.openai_reasoning import (
    DEFAULT_OPENAI_REASONING_MODEL,
    log_openai_reasoning_config,
    normalize_legacy_text_model,
    resolve_default_reasoning_effort,
)

logger = logging.getLogger(__name__)

PLANNING_STAGES = (
    "strategy_slogan_stage",
    "strategy_stage",
    "slogan_stage",
    "conceptual_stage",
    "brand_physical",
    "graphic_system",
    "series_ads",
    "product_name_resolution",
)

STAGE_MODEL_ENV_KEYS: Dict[str, str] = {
    "strategy_slogan_stage": "BUILDER1_STRATEGY_SLOGAN_STAGE_MODEL",
    "strategy_stage": "BUILDER1_STRATEGY_STAGE_MODEL",
    "slogan_stage": "BUILDER1_SLOGAN_STAGE_MODEL",
    "conceptual_stage": "BUILDER1_CONCEPTUAL_STAGE_MODEL",
    "brand_physical": "BUILDER1_PHYSICAL_STAGE_MODEL",
    "graphic_system": "BUILDER1_GRAPHIC_STAGE_MODEL",
    "series_ads": "BUILDER1_SERIES_STAGE_MODEL",
    "product_name_resolution": "BUILDER1_PRODUCT_NAME_MODEL",
}

STAGE_REASONING_ENV_KEYS: Dict[str, str] = {
    "strategy_slogan_stage": "BUILDER1_STRATEGY_SLOGAN_STAGE_REASONING_EFFORT",
    "strategy_stage": "BUILDER1_STRATEGY_STAGE_REASONING_EFFORT",
    "slogan_stage": "BUILDER1_SLOGAN_STAGE_REASONING_EFFORT",
    "conceptual_stage": "BUILDER1_CONCEPTUAL_STAGE_REASONING_EFFORT",
    "brand_physical": "BUILDER1_PHYSICAL_STAGE_REASONING_EFFORT",
    "graphic_system": "BUILDER1_GRAPHIC_STAGE_REASONING_EFFORT",
    "series_ads": "BUILDER1_SERIES_STAGE_REASONING_EFFORT",
}

STRATEGY_SLOGAN_FAMILY_STAGES = frozenset(
    {
        "strategy_slogan_stage",
        "strategy_slogan_repair",
        "slogan_only_repair",
        "strategy_stage",
        "slogan_stage",
    }
)

CORE_QUALITY_STAGES = frozenset(
    {
        "strategy_slogan_stage",
        "strategy_stage",
        "slogan_stage",
        "conceptual_stage",
        "brand_physical",
    }
)

BALANCED_EXECUTION_STAGES = frozenset(
    {
        "product_name_resolution",
        "graphic_system",
        "series_ads",
    }
)

FAST_EXECUTION_STAGES = frozenset(
    {
        "product_name_resolution",
        "slogan_stage",
        "conceptual_stage",
        "brand_physical",
        "graphic_system",
        "series_ads",
    }
)

VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high"})

_config_logged = False
_execution_model_warning_logged = False
_reasoning_effort_omit_logged: set[tuple[str, str, str]] = set()
_stage_model_overrides: ContextVar[Dict[str, str]] = ContextVar(
    "builder1_stage_model_overrides",
    default={},
)
_available_models_cache: Optional[frozenset[str]] = None


class PlanningProfile(str, Enum):
    QUALITY = "QUALITY"
    BALANCED = "BALANCED"
    FAST = "FAST"


@dataclass(frozen=True)
class StageRoutingDecision:
    model: str
    reasoning_effort: Optional[str]
    execution_optimization_active: bool


def quality_model() -> str:
    configured = (os.environ.get("BUILDER1_QUALITY_MODEL") or "").strip()
    if configured:
        return normalize_legacy_text_model(configured)
    legacy = (os.environ.get("BUILDER1_PLANNING_MODEL") or "").strip()
    if legacy:
        return normalize_legacy_text_model(legacy)
    return DEFAULT_OPENAI_REASONING_MODEL


def configured_execution_model() -> Optional[str]:
    raw = (os.environ.get("BUILDER1_EXECUTION_MODEL") or "").strip()
    return raw or None


def execution_model_resolved() -> str:
    configured = configured_execution_model()
    if configured:
        return configured
    return quality_model()


def execution_optimization_active() -> bool:
    profile = resolve_planning_profile()
    if profile not in {PlanningProfile.BALANCED, PlanningProfile.FAST}:
        return False
    configured = configured_execution_model()
    if not configured:
        return False
    return configured != quality_model()


def resolve_planning_profile() -> PlanningProfile:
    raw = (os.environ.get("BUILDER1_PLANNING_PROFILE") or "BALANCED").strip().upper()
    try:
        return PlanningProfile(raw)
    except ValueError:
        logger.warning(
            "BUILDER1_PLANNING_PROFILE_INVALID profile=%s fallback=BALANCED",
            raw,
        )
        return PlanningProfile.BALANCED


def model_supports_reasoning_effort(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith(("o1", "o3", "gpt-5"))


def _routing_stage_key(stage: Optional[str]) -> Optional[str]:
    if not stage:
        return stage
    if stage in STRATEGY_SLOGAN_FAMILY_STAGES:
        return "strategy_slogan_stage"
    return stage


def _env_stage_model(stage: str) -> str:
    env_key = STAGE_MODEL_ENV_KEYS.get(stage, "")
    if env_key:
        configured = (os.environ.get(env_key) or "").strip()
        if configured:
            return configured
    return ""


def _profile_default_model(stage: str, profile: PlanningProfile) -> str:
    q_model = quality_model()
    exec_configured = configured_execution_model()
    exec_model = execution_model_resolved()
    if profile == PlanningProfile.QUALITY:
        return q_model
    if stage == "strategy_slogan_stage":
        return q_model
    if profile == PlanningProfile.BALANCED:
        if stage in BALANCED_EXECUTION_STAGES and exec_configured:
            return exec_model
        return q_model
    if stage == "strategy_stage":
        return q_model
    if stage in FAST_EXECUTION_STAGES and exec_configured:
        return exec_model
    return q_model


def _profile_default_reasoning(stage: str, profile: PlanningProfile) -> str:
    del stage, profile  # All Builder1 stages inherit OPENAI_REASONING_EFFORT (default medium).
    return resolve_default_reasoning_effort()


def _log_missing_execution_model_warning(profile: PlanningProfile) -> None:
    global _execution_model_warning_logged
    if _execution_model_warning_logged:
        return
    if profile not in {PlanningProfile.BALANCED, PlanningProfile.FAST}:
        return
    if configured_execution_model():
        return
    _execution_model_warning_logged = True
    logger.warning(
        "BUILDER1_EXECUTION_MODEL_MISSING profile=%s qualityModel=%s "
        "executionOptimizationActive=false allStagesFallbackToQualityModel=true",
        profile.value,
        quality_model(),
    )


def resolve_stage_model(stage: Optional[str]) -> str:
    overrides = _stage_model_overrides.get()
    if stage and stage in overrides:
        return overrides[stage]
    profile = resolve_planning_profile()
    _log_missing_execution_model_warning(profile)
    if not stage:
        return quality_model()

    routing_stage = _routing_stage_key(stage) or stage
    exec_configured = configured_execution_model()
    profile_model = _profile_default_model(routing_stage, profile)
    explicit = _env_stage_model(routing_stage)

    if (
        profile in {PlanningProfile.BALANCED, PlanningProfile.FAST}
        and routing_stage in (BALANCED_EXECUTION_STAGES if profile == PlanningProfile.BALANCED else FAST_EXECUTION_STAGES)
        and exec_configured
    ):
        if explicit and explicit == quality_model() and profile_model != quality_model():
            logger.warning(
                "BUILDER1_STAGE_MODEL_IGNORED profile=%s stage=%s configuredStageModel=%s "
                "routingToExecutionModel=%s",
                profile.value,
                stage,
                explicit,
                profile_model,
            )
            return profile_model
        if not explicit:
            return profile_model

    if explicit:
        return normalize_legacy_text_model(explicit)
    return profile_model


def resolve_stage_reasoning_effort(stage: Optional[str], model: str) -> Optional[str]:
    routing_stage = _routing_stage_key(stage) if stage else stage
    if not routing_stage:
        effort = resolve_default_reasoning_effort()
    else:
        env_key = STAGE_REASONING_ENV_KEYS.get(routing_stage, "")
        configured = (os.environ.get(env_key) or "").strip().lower() if env_key else ""
        if configured:
            if configured not in VALID_REASONING_EFFORTS:
                logger.warning(
                    "BUILDER1_REASONING_EFFORT_UNSUPPORTED stage=%s effort=%s fallback=profile_default",
                    stage,
                    configured,
                )
                configured = _profile_default_reasoning(routing_stage, resolve_planning_profile())
            effort = configured
        else:
            effort = _profile_default_reasoning(routing_stage, resolve_planning_profile())

    if not model_supports_reasoning_effort(model):
        if effort:
            warn_key = (stage or "", model, effort)
            if warn_key not in _reasoning_effort_omit_logged:
                _reasoning_effort_omit_logged.add(warn_key)
                logger.warning(
                    "BUILDER1_REASONING_EFFORT_UNSUPPORTED stage=%s model=%s effort=%s action=omit",
                    stage or "",
                    model,
                    effort,
                )
        return None
    return effort


@contextmanager
def stage_model_override(overrides: Dict[str, str]):
    token = _stage_model_overrides.set(dict(overrides))
    try:
        yield
    finally:
        _stage_model_overrides.reset(token)


def resolve_stage_routing(stage: Optional[str]) -> StageRoutingDecision:
    model = resolve_stage_model(stage)
    effort = resolve_stage_reasoning_effort(stage, model)
    return StageRoutingDecision(
        model=model,
        reasoning_effort=effort,
        execution_optimization_active=execution_optimization_active(),
    )


def log_builder1_planning_profile_config() -> None:
    global _config_logged
    if _config_logged:
        return
    profile = resolve_planning_profile()
    _log_missing_execution_model_warning(profile)
    q_model = quality_model()
    exec_model = configured_execution_model() or "(unset)"
    models = {stage: resolve_stage_model(stage) for stage in PLANNING_STAGES}
    shared_effort = resolve_default_reasoning_effort()
    logger.info(
        "BUILDER1_PLANNING_PROFILE_CONFIG profile=%s qualityModel=%s executionModel=%s "
        "reasoningEffort=%s productNameModel=%s strategyModel=%s sloganModel=%s "
        "conceptualModel=%s physicalModel=%s graphicModel=%s seriesModel=%s "
        "executionOptimizationActive=%s",
        profile.value,
        q_model,
        exec_model,
        shared_effort,
        models["product_name_resolution"],
        models["strategy_stage"],
        models["slogan_stage"],
        models["conceptual_stage"],
        models["brand_physical"],
        models["graphic_system"],
        models["series_ads"],
        str(execution_optimization_active()).lower(),
    )
    log_openai_reasoning_config(model=q_model)
    _config_logged = True


def list_available_openai_model_ids(*, force_refresh: bool = False) -> frozenset[str]:
    global _available_models_cache
    if _available_models_cache is not None and not force_refresh:
        return _available_models_cache
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return frozenset()
    try:
        from openai import OpenAI
        import httpx

        client = OpenAI(api_key=api_key, timeout=httpx.Timeout(30.0), max_retries=0)
        response = client.models.list()
        ids = frozenset(getattr(item, "id", "") for item in response.data if getattr(item, "id", ""))
        _available_models_cache = ids
        return ids
    except Exception as exc:
        logger.warning("BUILDER1_MODEL_LIST_UNAVAILABLE err=%s", exc)
        return frozenset()


def run_builder1_model_list_diagnostic() -> int:
    models = sorted(list_available_openai_model_ids(force_refresh=True))
    if not models:
        print("No models returned. Check OPENAI_API_KEY and network access.")
        return 1
    for model_id in models:
        print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_builder1_model_list_diagnostic())
