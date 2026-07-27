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

from engine.builder2_methodology_validation import validate_winner_methodology
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import _read_raw
from engine.builder2_winner_development_diagnostics import _failure_code, _failure_field
from engine.builder2_winner_downstream import (
    Builder2WinnerDownstreamError,
    validate_builder2_winner_headline_composition_pure,
)
from engine.builder2_winner_headline_repair import validate_and_finalize_repaired_winner_plan
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_plan import validate_builder2_winner_plan
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
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
