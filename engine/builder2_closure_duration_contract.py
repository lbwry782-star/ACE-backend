"""
Builder2 closure duration contract — canonical v3 timing and inspector semantics.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS = 3.5
BUILDER2_CLOSURE_V2_SEGMENT_DURATION_SECONDS = 2.0
BUILDER2_CLOSURE_V2_TYPICAL_FINAL_DURATION_SECONDS = 12.0

ENV_BUILDER2_END_CARD_DURATION_SECONDS = "BUILDER2_END_CARD_DURATION_SECONDS"
ENV_BUILDER2_FINAL_VIDEO_DURATION_SECONDS = "BUILDER2_FINAL_VIDEO_DURATION_SECONDS"


def _parse_duration_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_environment_end_card_duration_seconds() -> Optional[float]:
    return _parse_duration_value(os.environ.get(ENV_BUILDER2_END_CARD_DURATION_SECONDS))


def read_environment_final_video_duration_seconds() -> Optional[float]:
    return _parse_duration_value(os.environ.get(ENV_BUILDER2_FINAL_VIDEO_DURATION_SECONDS))


def enforce_v3_closure_duration_contract() -> None:
    env_value = read_environment_end_card_duration_seconds()
    if env_value is None:
        return
    if abs(env_value - BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS) > 0.01:
        raise Builder2TournamentError("builder2_closure_duration_contract_mismatch")


def is_v3_closure_duration_contract_satisfied() -> bool:
    try:
        enforce_v3_closure_duration_contract()
        return True
    except Builder2TournamentError:
        return False


def resolve_configured_closure_segment_duration_seconds(
    *,
    typography_contract_version: str,
    requested_duration_seconds: float | None = None,
) -> float:
    from engine.builder2_closure_typography import BUILDER2_CLOSURE_TYPOGRAPHY_VERSION

    if typography_contract_version == BUILDER2_CLOSURE_TYPOGRAPHY_VERSION:
        enforce_v3_closure_duration_contract()
        if requested_duration_seconds is not None:
            requested = float(requested_duration_seconds)
            if abs(requested - BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS) > 0.01:
                logger.info(
                    "BUILDER2_CLOSURE_SEGMENT_DURATION_COERCED requested=%.3f effective=%.3f typography=%s",
                    requested,
                    BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS,
                    typography_contract_version,
                )
        return BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS

    from engine.builder2_new_format_config import resolve_builder2_end_card_duration_seconds

    effective = float(resolve_builder2_end_card_duration_seconds())
    if requested_duration_seconds is not None:
        requested = float(requested_duration_seconds)
        if abs(requested - effective) > 0.01:
            logger.info(
                "BUILDER2_CLOSURE_SEGMENT_DURATION_COERCED requested=%.3f effective=%.3f typography=%s",
                requested,
                effective,
                typography_contract_version,
            )
    return effective


def resolve_expected_final_video_duration_seconds(
    *,
    raw_video_duration_seconds: float | None = None,
    typography_contract_version: str | None = None,
) -> float:
    from engine.builder2_closure_typography import BUILDER2_CLOSURE_TYPOGRAPHY_VERSION

    version = typography_contract_version or BUILDER2_CLOSURE_TYPOGRAPHY_VERSION
    closure = resolve_configured_closure_segment_duration_seconds(
        typography_contract_version=version,
    )
    if raw_video_duration_seconds is not None:
        return float(raw_video_duration_seconds) + closure
    from engine.builder2_runway_config import resolve_builder2_video_duration_seconds

    return float(resolve_builder2_video_duration_seconds()) + closure


def extract_current_artifact_closure_duration_seconds(state: Dict[str, Any]) -> Optional[float]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
    for bucket, key in (
        (media, "measuredClosureDurationSeconds"),
        (media, "endCardDurationSeconds"),
        (closure, "durationSeconds"),
    ):
        value = _parse_duration_value(bucket.get(key))
        if value is not None:
            return value
    return None


def extract_previous_final_duration_seconds(state: Dict[str, Any]) -> Optional[float]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    for key in (
        "actualFinalVideoDurationSeconds",
        "measuredFinalDurationSeconds",
        "finalVideoDurationSeconds",
    ):
        value = _parse_duration_value(media.get(key))
        if value is not None:
            return value
    return None


def extract_measured_raw_video_duration_seconds(state: Dict[str, Any]) -> Optional[float]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    for key in (
        "measuredRawRunwayDurationSeconds",
        "measuredRawVideoDurationSeconds",
        "actualVisualDurationSeconds",
        "headlineReconstructionDurationSeconds",
    ):
        value = _parse_duration_value(media.get(key))
        if value is not None:
            return value
    return None


def resolve_inspector_raw_video_duration_seconds(state: Dict[str, Any]) -> float:
    measured = extract_measured_raw_video_duration_seconds(state)
    if measured is not None:
        return measured
    from engine.builder2_runway_config import resolve_builder2_video_duration_seconds

    return float(resolve_builder2_video_duration_seconds())


def build_closure_duration_inspector_fields(
    state: Dict[str, Any],
    *,
    requested_typography_version: str,
) -> Dict[str, Any]:
    from engine.builder2_closure_typography import (
        BUILDER2_CLOSURE_TYPOGRAPHY_V2,
        BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        current_closure_typography_version,
    )

    current_version = current_closure_typography_version(
        state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    )
    current_artifact_closure = extract_current_artifact_closure_duration_seconds(state)
    if current_artifact_closure is None and current_version == BUILDER2_CLOSURE_TYPOGRAPHY_V2:
        current_artifact_closure = BUILDER2_CLOSURE_V2_SEGMENT_DURATION_SECONDS

    if requested_typography_version == BUILDER2_CLOSURE_TYPOGRAPHY_VERSION:
        requested_closure = BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS
        contract_satisfied = is_v3_closure_duration_contract_satisfied()
    else:
        requested_closure = resolve_configured_closure_segment_duration_seconds(
            typography_contract_version=requested_typography_version,
        )
        contract_satisfied = True
    raw_duration = resolve_inspector_raw_video_duration_seconds(state)
    requested_expected_final = raw_duration + requested_closure
    duration_upgrade_needed = False
    if requested_typography_version and current_artifact_closure is not None:
        duration_upgrade_needed = abs(current_artifact_closure - requested_closure) > 0.01
    elif requested_typography_version:
        duration_upgrade_needed = True

    env_end_card = read_environment_end_card_duration_seconds()
    env_final = read_environment_final_video_duration_seconds()

    return {
        "currentArtifactClosureDurationSeconds": current_artifact_closure,
        "previousClosureDurationSeconds": current_artifact_closure,
        "previousFinalDurationSeconds": extract_previous_final_duration_seconds(state),
        "requestedClosureDurationSeconds": requested_closure,
        "measuredRawVideoDurationSeconds": raw_duration,
        "requestedExpectedFinalDurationSeconds": requested_expected_final,
        "configuredClosureSegmentDurationSeconds": requested_closure,
        "configuredFinalVideoDurationSeconds": requested_expected_final,
        "expectedFinalVideoDurationFromComponents": requested_expected_final,
        "environmentEndCardDurationSeconds": env_end_card,
        "environmentFinalVideoDurationSeconds": env_final,
        "durationUpgradeNeeded": duration_upgrade_needed,
        "closureDurationContractSatisfied": contract_satisfied,
    }


__all__ = [
    "BUILDER2_CLOSURE_V2_SEGMENT_DURATION_SECONDS",
    "BUILDER2_CLOSURE_V2_TYPICAL_FINAL_DURATION_SECONDS",
    "BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS",
    "ENV_BUILDER2_END_CARD_DURATION_SECONDS",
    "ENV_BUILDER2_FINAL_VIDEO_DURATION_SECONDS",
    "build_closure_duration_inspector_fields",
    "enforce_v3_closure_duration_contract",
    "extract_current_artifact_closure_duration_seconds",
    "extract_measured_raw_video_duration_seconds",
    "extract_previous_final_duration_seconds",
    "is_v3_closure_duration_contract_satisfied",
    "read_environment_end_card_duration_seconds",
    "read_environment_final_video_duration_seconds",
    "resolve_configured_closure_segment_duration_seconds",
    "resolve_expected_final_video_duration_seconds",
    "resolve_inspector_raw_video_duration_seconds",
]
