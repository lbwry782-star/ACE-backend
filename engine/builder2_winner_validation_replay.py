"""
Builder2 Winner validation replay — stage-by-stage offline instrumentation.
"""
from __future__ import annotations

import re
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError, require_dict, require_non_empty_str
from engine.builder2_winner_downstream import Builder2WinnerDownstreamError, extract_builder2_video_prompt_text
from engine.builder2_winner_plan import (
    _MONTAGE_LANGUAGE,
    _clean_scene_variations,
    _headline_decision_value,
    _validate_visual_anchor,
    validate_builder2_winner_plan,
)
from engine.builder2_winner_preservation_contract import (
    apply_server_owned_preservation,
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    detect_winner_immutable_identity_violations,
    normalize_winner_response_compatibility_fields,
    prepare_and_validate_persisted_winner_offline,
    validate_winner_source_identity,
)
from engine.builder2_winner_scene_variations_normalization import (
    normalize_continuous_event_scene_variations_for_execution,
)

_REGEX_OPERATIONS: Tuple[Tuple[str, str], ...] = (
    ("validate_builder2_winner_plan", "_MONTAGE_LANGUAGE.search", "videoPrompt"),
)


def _preview(value: Any, *, limit: int = 120) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        keys = ",".join(sorted(str(key) for key in value.keys())[:8])
        text = f"dict keys={keys}"
    elif isinstance(value, list):
        text = f"list len={len(value)}"
    else:
        text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _stage_result(
    *,
    stage: str,
    function: str,
    field_path: Optional[str],
    expected_type: str,
    actual_type: str,
    value_preview: str,
    accepted: bool,
    exception_class: Optional[str] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "function": function,
        "fieldPath": field_path,
        "expectedType": expected_type,
        "actualType": actual_type,
        "valuePreview": value_preview,
        "accepted": accepted,
        "exceptionClass": exception_class,
        "message": message,
    }


def replay_winner_validation_stages(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []
    plan = deepcopy(raw)

    def run(stage: str, function: str, fn: Callable[[], Any], *, field_path: Optional[str] = None) -> bool:
        try:
            fn()
            stages.append(
                _stage_result(
                    stage=stage,
                    function=function,
                    field_path=field_path,
                    expected_type="valid",
                    actual_type="valid",
                    value_preview="",
                    accepted=True,
                )
            )
            return True
        except Builder2TournamentError as exc:
            message = str(exc.args[0] if exc.args else exc)
            field = message.split(":", 1)[-1] if ":" in message else field_path
            stages.append(
                _stage_result(
                    stage=stage,
                    function=function,
                    field_path=field,
                    expected_type="contract",
                    actual_type="Builder2TournamentError",
                    value_preview=message,
                    accepted=False,
                    exception_class=exc.__class__.__name__,
                    message=message,
                )
            )
            return False
        except TypeError as exc:
            stages.append(
                _stage_result(
                    stage=stage,
                    function=function,
                    field_path=field_path,
                    expected_type="str",
                    actual_type=type(getattr(exc, "object", plan.get(field_path or ""))).__name__,
                    value_preview=_preview(plan.get(field_path or "")),
                    accepted=False,
                    exception_class=exc.__class__.__name__,
                    message=str(exc),
                )
            )
            return False
        except Exception as exc:
            stages.append(
                _stage_result(
                    stage=stage,
                    function=function,
                    field_path=field_path,
                    expected_type="valid",
                    actual_type=exc.__class__.__name__,
                    value_preview=str(exc),
                    accepted=False,
                    exception_class=exc.__class__.__name__,
                    message=str(exc),
                )
            )
            return False

    normalized = normalize_winner_response_compatibility_fields(plan)
    stages.append(
        _stage_result(
            stage="normalize_winner_response_compatibility_fields",
            function="normalize_winner_response_compatibility_fields",
            field_path=None,
            expected_type="dict",
            actual_type=type(normalized).__name__,
            value_preview=f"keys={len(normalized)}",
            accepted=True,
        )
    )
    plan = normalized

    if not run(
        "detect_winner_immutable_identity_violations",
        "detect_winner_immutable_identity_violations",
        lambda: detect_winner_immutable_identity_violations(plan, source_reference=source_reference),
    ):
        return stages

    merged = apply_server_owned_preservation(plan, source_reference=source_reference)
    plan = merged
    stages.append(
        _stage_result(
            stage="apply_server_owned_preservation",
            function="apply_server_owned_preservation",
            field_path=None,
            expected_type="dict",
            actual_type="dict",
            value_preview="preservation applied",
            accepted=True,
        )
    )

    if not compatibility_mode:
        from engine.builder2_single_slogan_contract import apply_persisted_winner_copy_contract_normalization

        if not run(
            "apply_persisted_winner_copy_contract_normalization",
            "apply_persisted_winner_copy_contract_normalization",
            lambda: apply_persisted_winner_copy_contract_normalization(
                plan,
                winning_candidate=winning_candidate,
                winning_judgment=winning_judgment,
                tournament_state=tournament_state,
            ),
        ):
            return stages

    if not run(
        "validate_winner_source_identity",
        "validate_winner_source_identity",
        lambda: validate_winner_source_identity(plan, source_reference=source_reference),
    ):
        return stages

    if not compatibility_mode:
        from engine.builder2_complete_ad_contract import validate_winner_slogan_preservation

        if not run(
            "validate_winner_slogan_preservation",
            "validate_winner_slogan_preservation",
            lambda: validate_winner_slogan_preservation(plan, winning_candidate=winning_candidate),
        ):
            return stages

    if not run(
        "normalize_continuous_event_scene_variations_for_execution",
        "normalize_continuous_event_scene_variations_for_execution",
        lambda: normalize_continuous_event_scene_variations_for_execution(
            plan,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=str(source_reference.get("sourceCandidateId") or ""),
            prototype_id=str(source_reference.get("sourcePrototypeId") or ""),
        ),
    ):
        return stages

    structure = str(plan.get("structureType") or "")
    sequence = require_dict(plan.get("sequence"), field="sequence")
    if not run(
        "videoPrompt.extract",
        "extract_builder2_video_prompt_text",
        lambda: extract_builder2_video_prompt_text(plan.get("videoPrompt"), "videoPrompt"),
        field_path="videoPrompt",
    ):
        return stages

    if structure == "continuous_event":
        video_prompt_text = extract_builder2_video_prompt_text(plan.get("videoPrompt"), "videoPrompt")
        if not run(
            "continuous_event_videoPrompt_montage_language",
            "validate_builder2_winner_plan._MONTAGE_LANGUAGE.search",
            lambda: (
                (_ for _ in ()).throw(Builder2TournamentError("builder2_winner_development_failed"))
                if _MONTAGE_LANGUAGE.search(video_prompt_text)
                else None
            ),
            field_path="videoPrompt",
        ):
            return stages

    snapshot = preservation_snapshot or build_winning_candidate_preservation_snapshot(
        strategy_foundation={"strategyFoundationId": source_reference.get("strategyFoundationId")},
        winning_candidate=winning_candidate,
        candidate_id=str(source_reference.get("sourceCandidateId") or ""),
    )
    if not run(
        "validate_builder2_winner_plan",
        "validate_builder2_winner_plan",
        lambda: validate_builder2_winner_plan(
            plan,
            winning_candidate=winning_candidate,
            preservation_snapshot=snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
            tournament_state=tournament_state,
        ),
    ):
        return stages

    return stages


def first_failed_validation_stage(stages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((stage for stage in stages if not stage.get("accepted")), None)


def infer_typeerror_failure(
    exc: BaseException,
    *,
    plan: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    message = str(exc)
    if exc.__class__.__name__ != "TypeError" or "expected string or bytes-like object" not in message:
        return None, None, None
    for _function, operation, field_path in _REGEX_OPERATIONS:
        value = plan.get(field_path)
        if isinstance(value, dict):
            return field_path, "validate_builder2_winner_plan", operation
    if isinstance(plan.get("videoPrompt"), dict):
        return "videoPrompt", "validate_builder2_winner_plan", "_MONTAGE_LANGUAGE.search"
    return None, None, message


def replay_prepare_and_validate(
    raw: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    preservation_snapshot: Optional[Dict[str, Any]] = None,
    winning_judgment: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    job_id: str = "",
    tournament_id: str = "",
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stages = replay_winner_validation_stages(
        raw,
        source_reference=source_reference,
        winning_candidate=winning_candidate,
        preservation_snapshot=preservation_snapshot,
        winning_judgment=winning_judgment,
        compatibility_mode=compatibility_mode,
        job_id=job_id,
        tournament_id=tournament_id,
        tournament_state=tournament_state,
    )
    first_fail = first_failed_validation_stage(stages)
    accepted = first_fail is None
    validated: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None
    if accepted:
        try:
            validated = prepare_and_validate_persisted_winner_offline(
                raw,
                source_reference=source_reference,
                winning_candidate=winning_candidate,
                preservation_snapshot=preservation_snapshot,
                winning_judgment=winning_judgment,
                compatibility_mode=compatibility_mode,
                job_id=job_id,
                tournament_id=tournament_id,
                tournament_state=tournament_state,
            )
        except Exception as exc:
            accepted = False
            error = exc
            first_fail = {
                "stage": "prepare_and_validate_persisted_winner_offline",
                "function": "prepare_and_validate_persisted_winner_offline",
                "fieldPath": str(exc.args[0]).split(":", 1)[-1] if getattr(exc, "args", None) else None,
                "accepted": False,
                "exceptionClass": exc.__class__.__name__,
                "message": str(exc),
            }
            stages.append(first_fail)
    return {
        "accepted": accepted,
        "stages": stages,
        "firstFailure": first_fail,
        "validatedPlan": validated,
        "error": error,
        "traceback": traceback.format_exc() if error else "",
    }
