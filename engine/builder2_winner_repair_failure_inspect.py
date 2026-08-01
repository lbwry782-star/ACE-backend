"""
Builder2 Winner headline repair failure inspector — read-only offline validation replay.

Run:
  BUILDER2_WINNER_REPAIR_FAILURE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_winner_repair_failure_inspect
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_methodology_contract import VALID_HEADLINE_FORMS, VALID_STRUCTURE_TYPES
from engine.builder2_methodology_validation import validate_winner_methodology
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_contracts import (
    WINNER_PLAN_SCHEMA_VERSION,
    Builder2TournamentError,
    require_dict,
    require_non_empty_str,
)
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_development_diagnostics import _failure_code, _failure_field
from engine.builder2_winner_downstream import (
    Builder2WinnerDownstreamError,
    validate_builder2_winner_headline_composition_pure,
)
from engine.builder2_winner_headline_repair import validate_and_finalize_repaired_winner_plan
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_plan import (
    _MONTAGE_LANGUAGE,
    _clean_scene_variations,
    _headline_decision_value,
    _validate_visual_anchor,
    validate_builder2_winner_plan,
)
from engine.builder2_winner_scene_variations_normalization import (
    CONTINUOUS_EVENT_STRUCTURE,
    describe_scene_variations_metadata,
    normalize_continuous_event_scene_variations_for_execution,
)
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    SERVER_OWNED_WINNER_SOURCE_KEY,
    SERVER_PRESERVATION_CHECK_KEY,
    apply_server_owned_preservation,
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    detect_winner_immutable_identity_violations,
    normalize_winner_response_compatibility_fields,
    validate_winner_source_identity,
)
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

_WINNER_METRIC_KEYS = (
    "winnerDevelopmentCalls",
    "winnerNormalCalls",
    "winnerRepairCalls",
    "winnerRetryCalls",
    "totalReasoningCalls",
)

_HEADLINE_FIELD_KEYS = frozenset(
    {
        "headline",
        "headlineText",
        "headlineTextRemainder",
        "headlineCoreKeyword",
        "advertisingPromise",
    }
)

_MAX_HEADLINE_FORM_CHARS = 32
_MAX_HEADLINE_DECISION_REASON_CHARS = 80

_GENERIC_WINNER_DEVELOPMENT_FAILED = "builder2_winner_development_failed"

_GENERIC_WRAPPER_ORIGINS: Dict[str, Dict[str, Any]] = {
    "sceneVariations_invalid_entry": {
        "file": "engine/builder2_winner_plan.py",
        "function": "_clean_scene_variations",
        "approxLine": 87,
        "inferredField": "sceneVariations",
        "chainedWithFrom": False,
    },
    "continuous_event_videoPrompt_montage_language": {
        "file": "engine/builder2_winner_plan.py",
        "function": "validate_builder2_winner_plan",
        "approxLine": 162,
        "inferredField": "videoPrompt",
        "chainedWithFrom": False,
    },
    "variation_montage_sceneVariations_count": {
        "file": "engine/builder2_winner_plan.py",
        "function": "validate_builder2_winner_plan",
        "approxLine": 167,
        "inferredField": "sceneVariations",
        "chainedWithFrom": False,
    },
    "structureType_unrecognized": {
        "file": "engine/builder2_winner_plan.py",
        "function": "validate_builder2_winner_plan",
        "approxLine": 171,
        "inferredField": "structureType",
        "chainedWithFrom": False,
    },
}

_SERVER_OWNED_REQUIRED_CHILDREN = (
    "sourceCandidateId",
    "sourcePrototypeId",
    "strategyFoundationId",
    "methodologyVersion",
    "structureType",
    "visualParallelType",
    "coreCreativeMechanism",
    "coreVisualIdea",
    "visualFamily",
)

_SERVER_PRESERVATION_CHECK_CHILDREN = (
    "problemPreserved",
    "relativeAdvantagePreserved",
    "prototypeMethodPreserved",
    "coreMechanismPreserved",
    "source",
)

_ADVERTISING_CLOSURE_STRUCTURAL_CHILDREN = (
    "required",
    "productNameText",
    "sloganText",
)

_HEADLINE_RELATED_PREFIXES = (
    "headline",
    "builder2_headline",
    "builder2_tournament_invalid_field:headline",
    "planning_failed",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _word_count(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    return len([part for part in value.split() if part.strip()])


def _safe_text_field(value: Any, *, allow_short_enum: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "keyExists": value is not None,
        "valuePresent": False,
        "valueType": type(value).__name__ if value is not None else "missing",
    }
    if value is None:
        report["value"] = None
        report["characterCount"] = 0
        report["wordCount"] = 0
        return report
    if isinstance(value, bool):
        report["valuePresent"] = True
        report["value"] = value
        return report
    if isinstance(value, str):
        report["characterCount"] = len(value)
        report["wordCount"] = _word_count(value)
        report["valuePresent"] = bool(value.strip())
        if not value.strip():
            report["value"] = value
        elif allow_short_enum and len(value.strip()) <= _MAX_HEADLINE_FORM_CHARS:
            report["value"] = value.strip()
        return report
    report["valuePresent"] = True
    return report


def _safe_headline_decision(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"keyExists": False, "valuePresent": False, "valueType": "missing"}
    if isinstance(value, str):
        decision = _clean(value)
        return {
            "keyExists": True,
            "valuePresent": bool(decision),
            "valueType": "str",
            "decision": decision or None,
        }
    if not isinstance(value, dict):
        return {"keyExists": True, "valuePresent": True, "valueType": type(value).__name__}
    reason = _clean(value.get("reason")) if isinstance(value.get("reason"), str) else ""
    report: Dict[str, Any] = {
        "keyExists": True,
        "valuePresent": True,
        "valueType": "dict",
        "decision": _clean(value.get("decision")) or None,
        "reasonSource": _clean(value.get("reasonSource")) or None,
        "reasonPresent": bool(reason),
    }
    if reason and len(reason) <= _MAX_HEADLINE_DECISION_REASON_CHARS:
        report["reason"] = reason
    elif reason:
        report["reasonRedacted"] = True
        report["reasonCharacterCount"] = len(reason)
    return report


def _safe_error_string(exc: BaseException) -> str:
    message = str(exc.args[0] if exc.args else exc)
    if len(message) > 240:
        return message[:240]
    return message


def _safe_failure_code(exc: BaseException) -> str:
    if isinstance(exc, Builder2WinnerDownstreamError):
        return exc.code
    if isinstance(exc, Builder2TournamentError):
        return _failure_code(exc)
    return type(exc).__name__


def _safe_failure_field(exc: BaseException) -> Optional[str]:
    if isinstance(exc, Builder2WinnerDownstreamError):
        return exc.code.split(":", 1)[-1] if ":" in exc.code else exc.code
    if isinstance(exc, Builder2TournamentError):
        return _failure_field(exc)
    return None


def _exception_metadata(exc: BaseException, *, wrapped: bool = False) -> Dict[str, Any]:
    code = _safe_failure_code(exc)
    field = _safe_failure_field(exc)
    return {
        "exceptionClass": type(exc).__name__,
        "safeErrorCode": code,
        "safeErrorString": _safe_error_string(exc),
        "validationField": field,
        "causeClass": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
        "contextClass": type(exc.__context__).__name__ if exc.__context__ is not None else None,
        "wrapped": wrapped,
    }


def _build_exception_chain(exc: BaseException) -> List[Dict[str, Any]]:
    chain: List[Dict[str, Any]] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(_exception_metadata(current, wrapped=len(chain) > 0))
        nxt = current.__cause__ or current.__context__
        if nxt is current:
            break
        current = nxt
    return chain


def _stage_ok() -> Dict[str, Any]:
    return {"attempted": True, "accepted": True, "failureCode": None, "failureField": None, "notRunReason": None}


def _stage_fail(exc: BaseException) -> Dict[str, Any]:
    return {
        "attempted": True,
        "accepted": False,
        "failureCode": _safe_failure_code(exc),
        "failureField": _safe_failure_field(exc),
        "notRunReason": None,
    }


def _stage_not_run(reason: str) -> Dict[str, Any]:
    return {
        "attempted": False,
        "accepted": False,
        "failureCode": None,
        "failureField": None,
        "notRunReason": reason,
    }


def _run_stage(name: str, fn: Callable[[], Any]) -> Tuple[Dict[str, Any], Any]:
    try:
        result = fn()
        return _stage_ok(), result
    except (Builder2TournamentError, Builder2WinnerDownstreamError) as exc:
        return _stage_fail(exc), None
    except Exception as exc:
        wrapped = Builder2TournamentError(f"builder2_winner_repair_failure_inspect_stage_error:{name}")
        wrapped.__cause__ = exc
        return _stage_fail(wrapped), None


def _is_headline_related(code: Optional[str], field: Optional[str]) -> bool:
    blob = f"{code or ''}:{field or ''}".lower()
    return any(token in blob for token in _HEADLINE_RELATED_PREFIXES)


def _get_nested_value(obj: Any, path: str) -> Tuple[bool, Any]:
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return False, None
        if part not in current:
            return False, None
        current = current[part]
    return True, current


def _headline_decision_context(
    plan: Dict[str, Any],
    *,
    winning_judgment: Optional[Dict[str, Any]],
    winning_candidate: Optional[Dict[str, Any]],
) -> str:
    from engine.builder2_headline_decision_contract import (
        headline_decision_is_omit,
        normalize_headline_decision_object,
    )

    complete_ad_winner = isinstance((winning_candidate or {}).get("advertisingClosure"), dict)
    decision = _headline_decision_value(plan)
    if winning_candidate is not None:
        normalized = normalize_headline_decision_object(
            plan.get("headlineDecision"),
            winning_judgment=winning_judgment,
        )
        decision = str(normalized.get("decision") or "")
        if complete_ad_winner and headline_decision_is_omit(decision):
            decision = "omit"
    return decision


def _audit_scalar_field(
    plan: Dict[str, Any],
    *,
    field_path: str,
    expected_type: str,
    required: bool,
    enum_values: Optional[frozenset[str]] = None,
    word_count_relevant: bool = False,
) -> Dict[str, Any]:
    key_exists, value = _get_nested_value(plan, field_path)
    top_key = field_path.split(".")[0]
    if not key_exists and "." not in field_path:
        key_exists = top_key in plan
        value = plan.get(top_key)
    elif not key_exists:
        parent_path, leaf = field_path.rsplit(".", 1)
        parent_exists, parent = _get_nested_value(plan, parent_path)
        key_exists = parent_exists and isinstance(parent, dict) and leaf in parent
        value = parent.get(leaf) if key_exists and isinstance(parent, dict) else None

    entry: Dict[str, Any] = {
        "fieldPath": field_path,
        "keyExists": key_exists,
        "expectedType": expected_type,
        "requiredUnderCurrentDecision": required,
    }

    if value is None:
        entry["valueType"] = "null" if key_exists else "missing"
        entry["valuePresent"] = False
        entry["emptyString"] = False
        if not required:
            entry["structuralStatus"] = "not_required"
        elif not key_exists:
            entry["structuralStatus"] = "missing"
        else:
            entry["structuralStatus"] = "null"
        if expected_type == "str":
            entry["characterCount"] = 0
            if word_count_relevant:
                entry["wordCount"] = 0
        return entry

    value_type = type(value).__name__
    entry["valueType"] = value_type

    if expected_type == "str":
        if not isinstance(value, str):
            entry["valuePresent"] = True
            entry["emptyString"] = False
            entry["structuralStatus"] = "wrong_type"
            return entry
        stripped = value.strip()
        entry["valuePresent"] = bool(stripped)
        entry["emptyString"] = value != "" and not stripped
        entry["characterCount"] = len(value)
        if word_count_relevant:
            entry["wordCount"] = _word_count(value)
        if not required:
            entry["structuralStatus"] = "not_required"
        elif not stripped:
            entry["structuralStatus"] = "empty"
        elif enum_values is not None and stripped not in enum_values:
            entry["structuralStatus"] = "invalid_enum"
        else:
            entry["structuralStatus"] = "valid_shape"
        return entry

    if expected_type == "dict":
        if not isinstance(value, dict):
            entry["valuePresent"] = True
            entry["emptyString"] = False
            entry["structuralStatus"] = "wrong_type" if required else "not_required"
            return entry
        entry["valuePresent"] = bool(value)
        entry["emptyString"] = False
        entry["objectKeyCount"] = len(value)
        if not required:
            entry["structuralStatus"] = "not_required"
        elif not value:
            entry["structuralStatus"] = "empty"
        else:
            entry["structuralStatus"] = "valid_shape"
        return entry

    if expected_type == "list":
        if not isinstance(value, list):
            entry["valuePresent"] = True
            entry["emptyString"] = False
            entry["structuralStatus"] = "wrong_type" if required else "not_required"
            return entry
        entry["valuePresent"] = True
        entry["emptyString"] = False
        entry["listLength"] = len(value)
        entry["structuralStatus"] = "valid_shape" if not required or value else "empty"
        return entry

    if expected_type == "bool":
        if not isinstance(value, bool):
            entry["valuePresent"] = True
            entry["emptyString"] = False
            entry["structuralStatus"] = "wrong_type" if required else "not_required"
            return entry
        entry["valuePresent"] = True
        entry["emptyString"] = False
        entry["structuralStatus"] = "valid_shape"
        return entry

    entry["valuePresent"] = True
    entry["emptyString"] = False
    entry["structuralStatus"] = "unknown_without_semantic_validation"
    return entry


def _audit_visual_anchor(plan: Dict[str, Any], *, required: bool) -> List[Dict[str, Any]]:
    anchor = plan.get("visualAnchor")
    entries: List[Dict[str, Any]] = []
    if anchor is None:
        entries.append(
            {
                "fieldPath": "visualAnchor",
                "keyExists": "visualAnchor" in plan,
                "valueType": "missing",
                "valuePresent": False,
                "emptyString": False,
                "expectedType": "str|dict",
                "requiredUnderCurrentDecision": required,
                "structuralStatus": "missing" if required else "not_required",
            }
        )
        return entries
    if isinstance(anchor, str):
        entries.append(
            _audit_scalar_field(plan, field_path="visualAnchor", expected_type="str", required=required)
        )
        return entries
    if isinstance(anchor, dict):
        entries.append(
            {
                "fieldPath": "visualAnchor",
                "keyExists": True,
                "valueType": "dict",
                "valuePresent": bool(anchor),
                "emptyString": False,
                "objectKeyCount": len(anchor),
                "expectedType": "dict",
                "requiredUnderCurrentDecision": required,
                "structuralStatus": "valid_shape" if anchor else "empty",
            }
        )
        entries.append(
            _audit_scalar_field(
                plan,
                field_path="visualAnchor.description",
                expected_type="str",
                required=required,
            )
        )
        why_exists = isinstance(anchor, dict) and "whyEssential" in anchor
        why_required = why_exists and anchor.get("whyEssential") is not None
        why_entry = _audit_scalar_field(
            plan,
            field_path="visualAnchor.whyEssential",
            expected_type="str",
            required=why_required,
        )
        if not why_required:
            why_entry["requiredUnderCurrentDecision"] = False
            why_entry["structuralStatus"] = "not_required"
        entries.append(why_entry)
        return entries
    entries.append(
        {
            "fieldPath": "visualAnchor",
            "keyExists": True,
            "valueType": type(anchor).__name__,
            "valuePresent": True,
            "emptyString": False,
            "expectedType": "str|dict",
            "requiredUnderCurrentDecision": required,
            "structuralStatus": "wrong_type",
        }
    )
    return entries


def _build_required_winner_field_audit(
    plan: Dict[str, Any],
    *,
    winning_candidate: Optional[Dict[str, Any]],
    winning_judgment: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from engine.builder2_headline_decision_contract import headline_decision_requires_headline

    decision = _headline_decision_context(
        plan,
        winning_judgment=winning_judgment,
        winning_candidate=winning_candidate,
    )
    headline_required = headline_decision_requires_headline(decision)
    complete_ad = isinstance((winning_candidate or {}).get("advertisingClosure"), dict)
    closure_required = complete_ad or bool(plan.get("advertisingClosure"))
    structure = _clean(plan.get("structureType"))
    scene_variations_required = structure == "variation_montage"

    audit: List[Dict[str, Any]] = []

    audit.append(
        _audit_scalar_field(
            plan,
            field_path="schemaVersion",
            expected_type="str",
            required=True,
            enum_values=frozenset({WINNER_PLAN_SCHEMA_VERSION}),
        )
    )
    for field_path in (
        "productNameResolved",
        "language",
        "problemPerception",
        "relativeAdvantage",
        "prototypeId",
        "coreCreativeMechanism",
        "visualParallelType",
        "visualFamily",
        "structureType",
        "coreVisualIdea",
        "openingFrameDescription",
        "videoPrompt",
    ):
        enum_values = VALID_STRUCTURE_TYPES if field_path == "structureType" else None
        audit.append(
            _audit_scalar_field(
                plan,
                field_path=field_path,
                expected_type="str",
                required=True,
                enum_values=enum_values,
            )
        )

    headline_decision_entry = _audit_scalar_field(
        plan,
        field_path="headlineDecision",
        expected_type="dict",
        required=winning_candidate is not None,
    )
    audit.append(headline_decision_entry)
    decision_entry = _audit_scalar_field(
        plan,
        field_path="headlineDecision.decision",
        expected_type="str",
        required=winning_candidate is not None,
        enum_values=frozenset({"use", "omit"}),
    )
    audit.append(decision_entry)

    audit.append(
        _audit_scalar_field(
            plan,
            field_path="headlineForm",
            expected_type="str",
            required=plan.get("headlineForm") is not None,
            enum_values=VALID_HEADLINE_FORMS,
        )
    )
    audit.append(
        _audit_scalar_field(
            plan,
            field_path="headline",
            expected_type="str",
            required=headline_required,
            word_count_relevant=True,
        )
    )
    audit.append(
        _audit_scalar_field(
            plan,
            field_path="headlineCoreKeyword",
            expected_type="str",
            required=headline_required,
            word_count_relevant=True,
        )
    )
    audit.append(
        _audit_scalar_field(
            plan,
            field_path="headlineText",
            expected_type="str",
            required=False,
            word_count_relevant=True,
        )
    )

    sequence_entry = _audit_scalar_field(plan, field_path="sequence", expected_type="dict", required=True)
    audit.append(sequence_entry)
    for child in ("beginning", "development", "resolution"):
        audit.append(
            _audit_scalar_field(
                plan,
                field_path=f"sequence.{child}",
                expected_type="str",
                required=True,
            )
        )

    audit.extend(_audit_visual_anchor(plan, required=True))

    scene_entry = _audit_scalar_field(
        plan,
        field_path="sceneVariations",
        expected_type="list",
        required=scene_variations_required,
    )
    audit.append(scene_entry)

    closure_entry = _audit_scalar_field(
        plan,
        field_path="advertisingClosure",
        expected_type="dict",
        required=closure_required,
    )
    audit.append(closure_entry)
    if isinstance(plan.get("advertisingClosure"), dict):
        for child in _ADVERTISING_CLOSURE_STRUCTURAL_CHILDREN:
            child_type = "bool" if child == "required" else "str"
            child_required = closure_required
            child_entry = _audit_scalar_field(
                plan,
                field_path=f"advertisingClosure.{child}",
                expected_type=child_type,
                required=child_required,
            )
            if child == "sloganText":
                child_entry["structuralStatus"] = (
                    child_entry["structuralStatus"]
                    if child_entry["structuralStatus"] != "valid_shape"
                    else "unknown_without_semantic_validation"
                )
            audit.append(child_entry)

    owned_entry = _audit_scalar_field(
        plan,
        field_path=SERVER_OWNED_WINNER_SOURCE_KEY,
        expected_type="dict",
        required=True,
    )
    audit.append(owned_entry)
    owned = plan.get(SERVER_OWNED_WINNER_SOURCE_KEY)
    if isinstance(owned, dict):
        for child in _SERVER_OWNED_REQUIRED_CHILDREN:
            audit.append(
                _audit_scalar_field(
                    plan,
                    field_path=f"{SERVER_OWNED_WINNER_SOURCE_KEY}.{child}",
                    expected_type="str",
                    required=True,
                )
            )

    check_entry = _audit_scalar_field(
        plan,
        field_path=SERVER_PRESERVATION_CHECK_KEY,
        expected_type="dict",
        required=True,
    )
    audit.append(check_entry)
    check = plan.get(SERVER_PRESERVATION_CHECK_KEY)
    if isinstance(check, dict):
        for child in _SERVER_PRESERVATION_CHECK_CHILDREN:
            child_type = "bool" if child.endswith("Preserved") else "str"
            audit.append(
                _audit_scalar_field(
                    plan,
                    field_path=f"{SERVER_PRESERVATION_CHECK_KEY}.{child}",
                    expected_type=child_type,
                    required=True,
                    enum_values=frozenset({"server_owned_contract"}) if child == "source" else None,
                )
            )

    return audit


def _exact_error_code(exc: BaseException) -> str:
    message = str(exc.args[0] if exc.args else exc)
    if len(message) > 240:
        return message[:240]
    return message


def _low_level_stage_result(
    *,
    stage_name: str,
    accepted: bool,
    exc: Optional[BaseException] = None,
    inferred_field: Optional[str] = None,
    first_failure: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "stageName": stage_name,
        "attempted": True,
        "accepted": accepted,
        "exactSafeErrorCode": None,
        "exactFieldPath": None,
        "exceptionClass": None,
        "firstFailure": first_failure,
    }
    if exc is not None:
        result["exactSafeErrorCode"] = _exact_error_code(exc)
        result["exactFieldPath"] = _safe_failure_field(exc) or inferred_field
        result["exceptionClass"] = type(exc).__name__
    return result


def _replay_low_level_winner_plan_validation(
    raw: Dict[str, Any],
    *,
    winning_candidate: Optional[Dict[str, Any]],
    preservation_snapshot: Optional[Dict[str, Any]],
    winning_judgment: Optional[Dict[str, Any]],
    compatibility_mode: bool,
) -> List[Dict[str, Any]]:
    from engine.builder2_headline_decision_contract import (
        CANONICAL_HEADLINE_DECISIONS,
        headline_decision_is_omit,
        headline_decision_requires_headline,
        normalize_headline_decision_object,
    )

    plan = deepcopy(raw)
    stages: List[Dict[str, Any]] = []
    first_failure_marked = False

    def run(stage_name: str, fn: Callable[[], Any], *, inferred_field: Optional[str] = None) -> bool:
        nonlocal first_failure_marked
        try:
            fn()
            stages.append(_low_level_stage_result(stage_name=stage_name, accepted=True))
            return True
        except Builder2TournamentError as exc:
            stages.append(
                _low_level_stage_result(
                    stage_name=stage_name,
                    accepted=False,
                    exc=exc,
                    inferred_field=inferred_field,
                    first_failure=not first_failure_marked,
                )
            )
            if not first_failure_marked:
                first_failure_marked = True
            return False

    if not run(
        "planningFailure_absent",
        lambda: (
            (_ for _ in ()).throw(Builder2TournamentError(str(plan.get("planningFailure"))))
            if plan.get("planningFailure")
            else None
        ),
    ):
        return stages

    def _check_schema_version() -> None:
        if plan.get("schemaVersion") != WINNER_PLAN_SCHEMA_VERSION:
            raise Builder2TournamentError("builder2_winner_schema_invalid:schemaVersion")

    if not run("schemaVersion", _check_schema_version):
        return stages

    scalar_fields = (
        "productNameResolved",
        "language",
        "problemPerception",
        "relativeAdvantage",
        "prototypeId",
        "coreCreativeMechanism",
        "visualParallelType",
        "visualFamily",
    )
    for field in scalar_fields:
        if not run(field, lambda f=field: require_non_empty_str(plan.get(f), field=f)):
            return stages

    structure_holder: Dict[str, str] = {}
    if not run(
        "structureType",
        lambda: structure_holder.update(
            {"value": require_non_empty_str(plan.get("structureType"), field="structureType")}
        ),
    ):
        return stages
    structure = structure_holder["value"]

    headline_decision = _headline_decision_value(plan)
    complete_ad_winner = isinstance((winning_candidate or {}).get("advertisingClosure"), dict)
    if winning_candidate is not None:
        normalized_holder: Dict[str, Any] = {}

        def _normalize_headline() -> None:
            normalized = normalize_headline_decision_object(
                plan.get("headlineDecision"),
                winning_judgment=winning_judgment,
            )
            normalized_holder["value"] = normalized
            if normalized.get("decision") not in CANONICAL_HEADLINE_DECISIONS:
                raise Builder2TournamentError("builder2_winner_validation_failed:headlineDecision.decision")

        if not run("headlineDecision.normalize", _normalize_headline):
            return stages
        headline_decision = str(normalized_holder["value"].get("decision") or "")
        if complete_ad_winner and headline_decision_is_omit(headline_decision):
            headline_decision = "omit"

    if headline_decision_requires_headline(headline_decision) and not (
        complete_ad_winner and headline_decision_is_omit(headline_decision)
    ):
        for field in ("headline", "headlineCoreKeyword"):
            if not run(field, lambda f=field: require_non_empty_str(plan.get(f), field=f)):
                return stages

    if not run("coreVisualIdea", lambda: require_non_empty_str(plan.get("coreVisualIdea"), field="coreVisualIdea")):
        return stages

    sequence_holder: Dict[str, Any] = {}
    if not run(
        "sequence",
        lambda: sequence_holder.update(
            {"value": require_dict(plan.get("sequence"), field="sequence")}
        ),
    ):
        return stages
    sequence = sequence_holder["value"]
    for key in ("beginning", "development", "resolution"):
        if not run(f"sequence.{key}", lambda k=key: require_non_empty_str(sequence.get(k), field=f"sequence.{k}")):
            return stages

    if not run("visualAnchor", lambda: _validate_visual_anchor(plan)):
        return stages
    if not run(
        "openingFrameDescription",
        lambda: require_non_empty_str(plan.get("openingFrameDescription"), field="openingFrameDescription"),
    ):
        return stages
    if not run("videoPrompt", lambda: require_non_empty_str(plan.get("videoPrompt"), field="videoPrompt")):
        return stages

    variations = plan.get("sceneVariations")
    if structure == "continuous_event":
        cleaned_holder: Dict[str, Any] = {}

        def _clean_continuous() -> None:
            cleaned = _clean_scene_variations(variations, structure=structure, sequence=sequence)
            cleaned_holder["value"] = cleaned

        if not run("sceneVariations.clean_continuous_event", _clean_continuous, inferred_field="sceneVariations"):
            return stages
        if not run(
            "continuous_event_videoPrompt_montage_language",
            lambda: (
                (_ for _ in ()).throw(Builder2TournamentError(_GENERIC_WINNER_DEVELOPMENT_FAILED))
                if _MONTAGE_LANGUAGE.search(str(plan.get("videoPrompt") or ""))
                else None
            ),
            inferred_field="videoPrompt",
        ):
            return stages
    elif structure == "variation_montage":
        cleaned_holder = {}

        def _clean_montage() -> None:
            cleaned = _clean_scene_variations(variations, structure=structure, sequence=sequence)
            cleaned_holder["value"] = cleaned
            if len(cleaned) < 2 or len(cleaned) > 4:
                raise Builder2TournamentError(_GENERIC_WINNER_DEVELOPMENT_FAILED)

        if not run(
            "variation_montage_sceneVariations_count",
            _clean_montage,
            inferred_field="sceneVariations",
        ):
            return stages
    else:
        if not run(
            "structureType_unrecognized",
            lambda: (_ for _ in ()).throw(Builder2TournamentError(_GENERIC_WINNER_DEVELOPMENT_FAILED)),
            inferred_field="structureType",
        ):
            return stages

    if winning_candidate is not None:
        if not run(
            "validate_winner_methodology",
            lambda: validate_winner_methodology(
                deepcopy(plan),
                winning_candidate=winning_candidate,
                preservation_snapshot=preservation_snapshot,
                winning_judgment=winning_judgment,
                compatibility_mode=compatibility_mode,
            ),
        ):
            return stages

    return stages


def _first_invalid_audit_field(audit: List[Dict[str, Any]]) -> Optional[str]:
    invalid_statuses = {"missing", "null", "empty", "wrong_type", "invalid_enum"}
    for entry in audit:
        if entry.get("requiredUnderCurrentDecision") and entry.get("structuralStatus") in invalid_statuses:
            return str(entry.get("fieldPath"))
    return None


def _build_headline_text_timing_assessment() -> Dict[str, Any]:
    return {
        "canonicalHelper": "compose_builder2_headline_text",
        "canonicalHelperModule": "engine.builder2_winner_downstream",
        "normalCreationStage": "validate_builder2_winner_headline_composition_pure",
        "normalCreationVia": "apply_builder2_headline_composition",
        "requiredByBaseWinnerPlanValidation": False,
        "requiredByHeadlineCompositionValidation": True,
        "repairPathInvokesHelperBeforeBaseValidation": False,
        "missingHeadlineTextCanExplainBaseWinnerPlanFailure": False,
        "assessment": (
            "validate_builder2_winner_plan requires headline and headlineCoreKeyword when decision=use "
            "but does not require headlineText; headlineText is derived by compose_builder2_headline_text "
            "inside apply_builder2_headline_composition, which runs only in "
            "validate_builder2_winner_headline_composition_pure after base validation succeeds."
        ),
    }


def _build_generic_wrapper_metadata(
    *,
    low_level_stages: List[Dict[str, Any]],
    base_stage: Dict[str, Any],
    reproduce_exc: Optional[BaseException],
) -> Dict[str, Any]:
    first_fail = next((stage for stage in low_level_stages if not stage.get("accepted")), None)
    origin_key = str((first_fail or {}).get("stageName") or "")
    origin = dict(_GENERIC_WRAPPER_ORIGINS.get(origin_key, {}))
    if not origin and first_fail:
        origin = {
            "file": "engine/builder2_winner_plan.py",
            "function": "validate_builder2_winner_plan",
            "inferredField": first_fail.get("exactFieldPath"),
            "chainedWithFrom": False,
        }
    wrapped_code = _safe_failure_code(reproduce_exc) if reproduce_exc else base_stage.get("failureCode")
    inner_preserved = bool(
        reproduce_exc is not None
        and (
            reproduce_exc.__cause__ is not None
            or (
                _safe_failure_code(reproduce_exc) != _GENERIC_WINNER_DEVELOPMENT_FAILED
                and ":" in _safe_failure_code(reproduce_exc)
            )
        )
    )
    lost_inner = (
        wrapped_code == _GENERIC_WINNER_DEVELOPMENT_FAILED
        and not inner_preserved
        and first_fail is not None
        and first_fail.get("exactSafeErrorCode") == _GENERIC_WINNER_DEVELOPMENT_FAILED
    )
    return {
        "genericWrapperOrigin": origin or None,
        "genericWrapperLostInnerError": lost_inner,
    }


def _derive_concrete_failure_summary(
    *,
    low_level_stages: List[Dict[str, Any]],
    field_audit: List[Dict[str, Any]],
    headline_timing: Dict[str, Any],
) -> Dict[str, Any]:
    first_fail = next((stage for stage in low_level_stages if not stage.get("accepted")), None)
    first_field = _first_invalid_audit_field(field_audit)
    offline_sufficient = first_fail is not None and bool(first_fail.get("exactFieldPath"))
    return {
        "firstConcreteFailingStage": (first_fail or {}).get("stageName"),
        "firstConcreteFailureCode": (first_fail or {}).get("exactSafeErrorCode"),
        "firstConcreteFailureField": (first_fail or {}).get("exactFieldPath"),
        "firstStructurallyInvalidField": first_field,
        "headlineTextTimingAssessment": headline_timing,
        "additionalPaidCallRequiredToDiagnose": False,
        "offlineDataSufficientForDiagnosis": offline_sufficient,
    }


def _non_scene_variations_fingerprint(plan: Dict[str, Any]) -> str:
    filtered = {
        key: plan[key]
        for key in sorted(plan.keys())
        if key not in _HEADLINE_FIELD_KEYS
        and key != "headlineDecision"
        and key != "headlineForm"
        and key != "sceneVariations"
        and key != "continuousEventSceneVariationsNormalization"
    }
    payload = json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _inspect_continuous_event_scene_variations_normalization(
    preserved_plan: Dict[str, Any],
    *,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    prototype_id: str,
) -> Dict[str, Any]:
    structure = _clean(preserved_plan.get("structureType"))
    applicable = structure == CONTINUOUS_EVENT_STRUCTURE
    if not applicable:
        return {
            "applicable": False,
            "keyExisted": False,
            "originalValueType": None,
            "originalListCount": None,
            "normalizedListCount": None,
            "sequenceAuthoritative": False,
            "downstreamUsesOriginalSceneVariations": None,
            "normalizationChangedOtherFields": False,
            "nonSceneVariationsFingerprintBefore": None,
            "nonSceneVariationsFingerprintAfter": None,
        }

    metadata = describe_scene_variations_metadata(preserved_plan)
    before_fp = _non_scene_variations_fingerprint(preserved_plan)
    normalized_copy = deepcopy(preserved_plan)
    normalize_continuous_event_scene_variations_for_execution(
        normalized_copy,
        job_id=job_id,
        tournament_id=tournament_id,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
    )
    after_fp = _non_scene_variations_fingerprint(normalized_copy)
    return {
        "applicable": True,
        "keyExisted": metadata["keyExisted"],
        "originalValueType": metadata["originalValueType"],
        "originalListCount": metadata["originalListCount"],
        "normalizedListCount": 0,
        "sequenceAuthoritative": True,
        "downstreamUsesOriginalSceneVariations": False,
        "normalizationChangedOtherFields": before_fp != after_fp,
        "nonSceneVariationsFingerprintBefore": before_fp,
        "nonSceneVariationsFingerprintAfter": after_fp,
    }


def _replay_offline_after_normalization(
    *,
    parsed_plan: Dict[str, Any],
    preserved_plan: Dict[str, Any],
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    preservation_snapshot: Dict[str, Any],
    compatibility_mode: bool,
    job_id: str,
    tournament_id: str,
    candidate_id: str,
    prototype_id: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "offlineWinnerRevalidationAfterNormalizationAttempted": False,
        "offlineWinnerRevalidationAfterNormalizationAccepted": False,
        "firstFailureAfterNormalizationStage": None,
        "firstFailureAfterNormalizationCode": None,
        "firstFailureAfterNormalizationField": None,
        "headlineCompositionAttempted": False,
        "headlineCompositionAccepted": False,
        "headlineTextDerived": False,
        "finalWinnerPlanValidOffline": False,
        "additionalPaidCallRequired": False,
        "inspectionOpenAICalls": 0,
        "inspectionRedisMutations": 0,
    }
    if _clean(preserved_plan.get("structureType")) != CONTINUOUS_EVENT_STRUCTURE:
        return result

    result["offlineWinnerRevalidationAfterNormalizationAttempted"] = True
    try:
        validated = validate_and_finalize_repaired_winner_plan(
            deepcopy(parsed_plan),
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
            job_id=job_id,
            tournament_id=tournament_id,
        )
        result["offlineWinnerRevalidationAfterNormalizationAccepted"] = True
        result["headlineCompositionAttempted"] = True
        result["headlineCompositionAccepted"] = True
        result["headlineTextDerived"] = bool(_clean(validated.get("headlineText")))
        result["finalWinnerPlanValidOffline"] = True
        return result
    except Builder2TournamentError as exc:
        normalized_preserved = deepcopy(preserved_plan)
        normalize_continuous_event_scene_variations_for_execution(
            normalized_preserved,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
        )
        low_stages = _replay_low_level_winner_plan_validation(
            normalized_preserved,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
        )
        first_fail = next((stage for stage in low_stages if not stage.get("accepted")), None)
        if first_fail is not None:
            result["firstFailureAfterNormalizationStage"] = first_fail.get("stageName")
            result["firstFailureAfterNormalizationCode"] = first_fail.get("exactSafeErrorCode")
            result["firstFailureAfterNormalizationField"] = first_fail.get("exactFieldPath")
            return result

        result["headlineCompositionAttempted"] = True
        result["firstFailureAfterNormalizationStage"] = "headlineCompositionValidation"
        result["firstFailureAfterNormalizationCode"] = _exact_error_code(exc)
        result["firstFailureAfterNormalizationField"] = _safe_failure_field(exc)
        return result
    except Builder2WinnerDownstreamError as exc:
        normalized_preserved = deepcopy(preserved_plan)
        normalize_continuous_event_scene_variations_for_execution(
            normalized_preserved,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
        )
        low_stages = _replay_low_level_winner_plan_validation(
            normalized_preserved,
            winning_candidate=winning_candidate,
            preservation_snapshot=preservation_snapshot,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
        )
        if all(stage.get("accepted") for stage in low_stages):
            result["headlineCompositionAttempted"] = True
            result["firstFailureAfterNormalizationStage"] = "headlineCompositionValidation"
            result["firstFailureAfterNormalizationCode"] = exc.code
            result["firstFailureAfterNormalizationField"] = (
                exc.code.split(":", 1)[-1] if ":" in exc.code else exc.code
            )
        else:
            first_fail = next((stage for stage in low_stages if not stage.get("accepted")), None)
            if first_fail is not None:
                result["firstFailureAfterNormalizationStage"] = first_fail.get("stageName")
                result["firstFailureAfterNormalizationCode"] = first_fail.get("exactSafeErrorCode")
                result["firstFailureAfterNormalizationField"] = first_fail.get("exactFieldPath")
        return result


def _structural_fingerprint(plan: Dict[str, Any]) -> str:
    filtered = {
        key: plan[key]
        for key in sorted(plan.keys())
        if key not in _HEADLINE_FIELD_KEYS and key != "headlineDecision" and key != "headlineForm"
    }
    payload = json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _prepare_preserved_plan(
    parsed_plan: Dict[str, Any],
    *,
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    compatibility_mode: bool,
) -> Dict[str, Any]:
    normalized = normalize_winner_response_compatibility_fields(deepcopy(parsed_plan))
    detect_winner_immutable_identity_violations(normalized, source_reference=source_reference)
    merged = apply_server_owned_preservation(normalized, source_reference=source_reference)
    creator_closure = (winning_candidate or {}).get("advertisingClosure")
    if isinstance(creator_closure, dict):
        from engine.builder2_advertising_closure_contract import normalize_advertising_closure
        from engine.builder2_complete_ad_contract import apply_complete_ad_winner_plan_normalization

        merged["advertisingClosure"] = normalize_advertising_closure(
            {**creator_closure, "headlineSource": creator_closure.get("headlineSource") or "creator_candidate"}
        )
        if not compatibility_mode:
            apply_complete_ad_winner_plan_normalization(
                merged,
                winning_candidate=winning_candidate,
                winning_judgment=winning_judgment,
            )
    validate_winner_source_identity(merged, source_reference=source_reference)
    return merged


def _reproduce_validation_stages(
    *,
    parsed_plan: Dict[str, Any],
    source_reference: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    preservation_snapshot: Dict[str, Any],
    compatibility_mode: bool,
    job_id: str,
    tournament_id: str,
) -> Tuple[Dict[str, Any], Optional[BaseException], Optional[str], Optional[str], bool]:
    stages: Dict[str, Any] = {}
    preserved_plan: Optional[Dict[str, Any]] = None
    validated_plan: Optional[Dict[str, Any]] = None
    reproduce_exc: Optional[BaseException] = None

    stages["preservationApplied"], preserved_plan = _run_stage(
        "preservationApplied",
        lambda: _prepare_preserved_plan(
            parsed_plan,
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            compatibility_mode=compatibility_mode,
        ),
    )

    if preserved_plan is None:
        stages["advertisingClosureValidation"] = _stage_not_run("preservationApplied_failed")
        stages["methodologyValidation"] = _stage_not_run("preservationApplied_failed")
        stages["baseWinnerPlanValidation"] = _stage_not_run("preservationApplied_failed")
        stages["headlineCompositionValidation"] = _stage_not_run("preservationApplied_failed")
        stages["finalOfflineProcessing"] = _stage_not_run("preservationApplied_failed")
    else:
        def _advertising_closure() -> None:
            from engine.builder2_advertising_closure_contract import validate_advertising_closure_methodology
            from engine.builder2_complete_ad_contract import validate_winner_slogan_preservation

            if not compatibility_mode:
                validate_winner_slogan_preservation(preserved_plan, winning_candidate=winning_candidate)
            validate_advertising_closure_methodology(preserved_plan, require_present=False)

        stages["advertisingClosureValidation"], _ = _run_stage("advertisingClosureValidation", _advertising_closure)

        if not stages["advertisingClosureValidation"]["accepted"]:
            stages["methodologyValidation"] = _stage_not_run("advertisingClosureValidation_failed")
            stages["baseWinnerPlanValidation"] = _stage_not_run("advertisingClosureValidation_failed")
            stages["headlineCompositionValidation"] = _stage_not_run("advertisingClosureValidation_failed")
            stages["finalOfflineProcessing"] = _stage_not_run("advertisingClosureValidation_failed")
        else:

            def _methodology() -> None:
                validate_winner_methodology(
                    deepcopy(preserved_plan),
                    winning_candidate=winning_candidate,
                    preservation_snapshot=preservation_snapshot,
                    winning_judgment=winning_judgment,
                    compatibility_mode=compatibility_mode,
                )

            stages["methodologyValidation"], _ = _run_stage("methodologyValidation", _methodology)

            def _base_plan() -> Dict[str, Any]:
                return validate_builder2_winner_plan(
                    deepcopy(preserved_plan),
                    winning_candidate=winning_candidate,
                    preservation_snapshot=preservation_snapshot,
                    winning_judgment=winning_judgment,
                    compatibility_mode=compatibility_mode,
                )

            stages["baseWinnerPlanValidation"], validated_plan = _run_stage("baseWinnerPlanValidation", _base_plan)

            if validated_plan is None:
                stages["headlineCompositionValidation"] = _stage_not_run("baseWinnerPlanValidation_failed")
                stages["finalOfflineProcessing"] = _stage_not_run("baseWinnerPlanValidation_failed")
            else:

                def _headline_composition() -> None:
                    validate_builder2_winner_headline_composition_pure(deepcopy(validated_plan))

                stages["headlineCompositionValidation"], _ = _run_stage(
                    "headlineCompositionValidation",
                    _headline_composition,
                )

                def _final_offline() -> Dict[str, Any]:
                    return validate_and_finalize_repaired_winner_plan(
                        deepcopy(parsed_plan),
                        source_reference=source_reference,
                        winning_candidate=winning_candidate,
                        winning_judgment=winning_judgment,
                        preservation_snapshot=preservation_snapshot,
                        compatibility_mode=compatibility_mode,
                        job_id=job_id,
                        tournament_id=tournament_id,
                    )

                stages["finalOfflineProcessing"], _ = _run_stage("finalOfflineProcessing", _final_offline)

    first_failing_stage: Optional[str] = None
    first_failing_field: Optional[str] = None
    for stage_name, stage_report in stages.items():
        if stage_report.get("accepted") is False and stage_report.get("attempted"):
            first_failing_stage = stage_name
            first_failing_field = stage_report.get("failureField")
            break

    try:
        validate_and_finalize_repaired_winner_plan(
            deepcopy(parsed_plan),
            source_reference=source_reference,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
            preservation_snapshot=preservation_snapshot,
            compatibility_mode=compatibility_mode,
            job_id=job_id,
            tournament_id=tournament_id,
        )
        offline_repairable = True
    except Builder2TournamentError as exc:
        reproduce_exc = exc
        offline_repairable = False
    except Builder2WinnerDownstreamError as exc:
        reproduce_exc = Builder2TournamentError(exc.code)
        reproduce_exc.__cause__ = exc
        offline_repairable = False

    return stages, reproduce_exc, first_failing_stage, first_failing_field, offline_repairable


def inspect_builder2_winner_repair_failure(job_id: str = "") -> Dict[str, Any]:
    jid = _clean(job_id)
    if not jid:
        return {"ok": False, "error": "builder2_winner_repair_failure_inspect_job_id_missing", "jobId": None}
    if not redis_configured():
        return {"ok": False, "error": "builder2_winner_repair_failure_inspect_redis_unconfigured", "jobId": jid}

    with read_only_builder2_inspection() as mutation_counter:
        raw = _read_raw(jid)
        if raw is None:
            return {
                "ok": False,
                "error": "builder2_winner_repair_failure_inspect_tournament_not_found",
                "jobId": jid,
                "redisMutations": mutation_counter.redis_mutations,
            }
        state = deepcopy(raw)
        original_parsed_bucket = deepcopy(state.get(PARSED_WINNER_RESPONSE_KEY))

        winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId")) or None
        winner_rec = (state.get("candidates") or {}).get(winner_id or "") or {}
        judgment_id = _clean(winner_rec.get("judgmentId")) or None
        judgment_rec = (state.get("judgments") or {}).get(judgment_id or "") if judgment_id else None
        winning_judgment = (
            (judgment_rec or {}).get("judgment") if isinstance((judgment_rec or {}).get("judgment"), dict) else {}
        )
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}

        parsed_bucket = state.get(PARSED_WINNER_RESPONSE_KEY)
        parsed_exists = (
            isinstance(parsed_bucket, dict)
            and isinstance(parsed_bucket.get("parsed"), dict)
            and bool(parsed_bucket.get("parsed"))
        )
        parsed_plan = dict(parsed_bucket.get("parsed") or {}) if parsed_exists else {}
        parsed_candidate_id = _clean(parsed_bucket.get("candidateId")) if isinstance(parsed_bucket, dict) else None

        metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
        compatibility_mode = bool(state.get("methodologyCompatibilityMode"))

        source_reference = (
            build_server_owned_winner_source_reference(
                strategy_foundation=strategy,
                winning_candidate=winning_candidate,
                candidate_id=winner_id or "",
            )
            if winner_id and winning_candidate
            else {}
        )
        preservation_snapshot = (
            build_winning_candidate_preservation_snapshot(
                strategy_foundation=strategy,
                winning_candidate=winning_candidate,
                candidate_id=winner_id or "",
            )
            if winner_id and winning_candidate
            else {}
        )

        stages: Dict[str, Any] = {}
        reproduce_exc: Optional[BaseException] = None
        first_failing_stage: Optional[str] = None
        first_failing_field: Optional[str] = None
        offline_repairable = False
        required_winner_field_audit: List[Dict[str, Any]] = []
        low_level_validation_stages: List[Dict[str, Any]] = []
        preserved_plan_for_audit: Optional[Dict[str, Any]] = None
        continuous_event_scene_variations_normalization: Dict[str, Any] = {
            "applicable": False,
        }
        offline_after_normalization: Dict[str, Any] = {
            "offlineWinnerRevalidationAfterNormalizationAttempted": False,
            "offlineWinnerRevalidationAfterNormalizationAccepted": False,
            "additionalPaidCallRequired": False,
            "inspectionOpenAICalls": 0,
            "inspectionRedisMutations": 0,
        }
        if parsed_exists and winner_id and source_reference:
            stages, reproduce_exc, first_failing_stage, first_failing_field, offline_repairable = _reproduce_validation_stages(
                parsed_plan=parsed_plan,
                source_reference=source_reference,
                winning_candidate=winning_candidate,
                winning_judgment=winning_judgment,
                preservation_snapshot=preservation_snapshot,
                compatibility_mode=compatibility_mode,
                job_id=jid,
                tournament_id=_clean(state.get("tournamentId")),
            )
            try:
                preserved_plan_for_audit = _prepare_preserved_plan(
                    parsed_plan,
                    source_reference=source_reference,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    compatibility_mode=compatibility_mode,
                )
            except (Builder2TournamentError, Builder2WinnerDownstreamError):
                preserved_plan_for_audit = None
            if preserved_plan_for_audit is not None:
                required_winner_field_audit = _build_required_winner_field_audit(
                    preserved_plan_for_audit,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                )
                low_level_validation_stages = _replay_low_level_winner_plan_validation(
                    preserved_plan_for_audit,
                    winning_candidate=winning_candidate,
                    preservation_snapshot=preservation_snapshot,
                    winning_judgment=winning_judgment,
                    compatibility_mode=compatibility_mode,
                )
                continuous_event_scene_variations_normalization = _inspect_continuous_event_scene_variations_normalization(
                    preserved_plan_for_audit,
                    job_id=jid,
                    tournament_id=_clean(state.get("tournamentId")),
                    candidate_id=winner_id or "",
                    prototype_id=_clean(winner_rec.get("prototypeId")) or "",
                )
                offline_after_normalization = _replay_offline_after_normalization(
                    parsed_plan=parsed_plan,
                    preserved_plan=preserved_plan_for_audit,
                    source_reference=source_reference,
                    winning_candidate=winning_candidate,
                    winning_judgment=winning_judgment,
                    preservation_snapshot=preservation_snapshot,
                    compatibility_mode=compatibility_mode,
                    job_id=jid,
                    tournament_id=_clean(state.get("tournamentId")),
                    candidate_id=winner_id or "",
                    prototype_id=_clean(winner_rec.get("prototypeId")) or "",
                )
        else:
            reason = "parsed_winner_or_winner_identity_unavailable"
            for stage_name in (
                "preservationApplied",
                "advertisingClosureValidation",
                "methodologyValidation",
                "baseWinnerPlanValidation",
                "headlineCompositionValidation",
                "finalOfflineProcessing",
            ):
                stages[stage_name] = _stage_not_run(reason)

        chain = _build_exception_chain(reproduce_exc) if reproduce_exc is not None else []
        outer_code = chain[0]["safeErrorCode"] if chain else _clean(state.get("failureReason")) or None
        inner_code = chain[-1]["safeErrorCode"] if chain else None
        inner_field = chain[-1].get("validationField") if chain else None

        winner_score = winner_rec.get("totalScore")
        if winner_score is None and isinstance(judgment_rec, dict):
            winner_score = judgment_rec.get("totalScore")

        parsed_unchanged = original_parsed_bucket == state.get(PARSED_WINNER_RESPONSE_KEY)

        headline_timing = _build_headline_text_timing_assessment()
        wrapper_metadata = _build_generic_wrapper_metadata(
            low_level_stages=low_level_validation_stages,
            base_stage=stages.get("baseWinnerPlanValidation") or {},
            reproduce_exc=reproduce_exc,
        )
        concrete_summary = _derive_concrete_failure_summary(
            low_level_stages=low_level_validation_stages,
            field_audit=required_winner_field_audit,
            headline_timing=headline_timing,
        )

        report: Dict[str, Any] = {
            "ok": True,
            "inspectionCompleted": True,
            "jobId": jid,
            "redisMutations": mutation_counter.redis_mutations,
            "currentState": {
                "winnerCandidateId": winner_id,
                "winnerPrototypeId": _clean(winner_rec.get("prototypeId") or state.get("winnerDevelopmentPrototypeId"))
                or None,
                "winnerScore": winner_score,
                "acceptedCreatorCount": accepted_creator_count(state, read_only=True),
                "acceptedJudgmentCount": accepted_judgment_count(state, read_only=True),
                "parsedWinnerCandidateId": parsed_candidate_id,
                "parsedWinnerExists": parsed_exists,
                "failureStage": state.get("failureStage"),
                "failureReason": state.get("failureReason"),
                "reasoningComplete": bool(state.get("reasoningComplete")),
                "mediaStarted": bool(state.get("mediaStarted")),
                "winnerDevelopmentPlanExists": isinstance(state.get("winnerDevelopmentPlan"), dict)
                and bool(state.get("winnerDevelopmentPlan")),
                "winnerDevelopmentAccepted": is_valid_persisted_winner_development(state),
            },
            "repairedFieldMetadata": {
                "headline": _safe_text_field(parsed_plan.get("headline")),
                "headlineCoreKeyword": _safe_text_field(parsed_plan.get("headlineCoreKeyword")),
                "headlineText": _safe_text_field(parsed_plan.get("headlineText")),
                "headlineForm": _safe_text_field(parsed_plan.get("headlineForm"), allow_short_enum=True),
                "headlineDecision": _safe_headline_decision(parsed_plan.get("headlineDecision")),
            },
            "repairMetrics": {key: int(metrics.get(key) or 0) for key in _WINNER_METRIC_KEYS},
            "winnerDevelopmentPaidCallRecorded": {
                "keyExists": "winnerDevelopmentPaidCallRecorded" in state,
                "valueType": type(state.get("winnerDevelopmentPaidCallRecorded")).__name__
                if "winnerDevelopmentPaidCallRecorded" in state
                else "missing",
                "value": state.get("winnerDevelopmentPaidCallRecorded")
                if isinstance(state.get("winnerDevelopmentPaidCallRecorded"), bool)
                else None,
            },
            "repairPersistence": {
                "parsedWinnerStillPresent": parsed_exists,
                "parsedWinnerUnchangedDuringInspection": parsed_unchanged,
                "topLevelKeyCount": int((parsed_bucket or {}).get("topLevelKeyCount") or len(parsed_plan)),
                "responseCharCount": int((parsed_bucket or {}).get("responseCharCount") or 0),
                "nonHeadlineStructuralFingerprint": _structural_fingerprint(parsed_plan) if parsed_plan else None,
                "headlineDecisionFingerprint": _safe_headline_decision(parsed_plan.get("headlineDecision")),
                "headlineFormFingerprint": _safe_text_field(parsed_plan.get("headlineForm"), allow_short_enum=True),
            },
            "validationStages": stages,
            "requiredWinnerFieldAudit": required_winner_field_audit,
            "lowLevelValidationStages": low_level_validation_stages,
            "continuousEventSceneVariationsNormalization": continuous_event_scene_variations_normalization,
            **offline_after_normalization,
            "outerException": outer_code,
            "innermostException": inner_code,
            "exceptionChain": chain,
            "firstFailingStage": first_failing_stage,
            "firstFailingField": first_failing_field or inner_field,
            "failureIsHeadlineRelated": _is_headline_related(
                inner_code or outer_code,
                first_failing_field or inner_field,
            ),
            "offlineRepairableWithoutModelCall": offline_repairable,
            **wrapper_metadata,
            **concrete_summary,
            "inspectionCallCounts": {
                "openAICalls": 0,
                "runwayCalls": 0,
                "imageCalls": 0,
                "ffmpegCalls": 0,
                "winnerRepairCalls": 0,
                "persistMutations": 0,
            },
        }
        return report


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_WINNER_REPAIR_FAILURE_INSPECT_JOB_ID"))
    try:
        report = inspect_builder2_winner_repair_failure(job_id)
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return 0 if report.get("ok") else 1
    except Exception:
        logger.exception(
            "BUILDER2_WINNER_REPAIR_FAILURE_INSPECT_FAILED jobId=%s",
            job_id or "(none)",
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "builder2_winner_repair_failure_inspect_unhandled_exception",
                    "jobId": job_id or None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
