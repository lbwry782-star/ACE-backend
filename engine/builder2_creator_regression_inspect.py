"""
Builder2 Creator metaphor regression inspector — read-only, zero side effects.

Run:
  BUILDER2_CREATOR_REGRESSION_INSPECT_JOB_ID=<jobId> python -m engine.builder2_creator_regression_inspect
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    can_offline_revalidate_rejected_creator,
    load_rejected_creator_parsed_response,
)
from engine.builder2_creator import validate_creator_candidate
from engine.builder2_metaphorical_embodiment_contract import inspect_literal_symbol_disposition
from engine.builder2_tournament_completion_gate import (
    accepted_creator_count,
    accepted_judgment_count,
    assigned_prototype_ids,
)
from engine.builder2_tournament_store import load_tournament_state


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _response_fingerprint(parsed: Dict[str, Any]) -> str:
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strategy_fingerprint(state: Dict[str, Any]) -> str:
    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict):
        return ""
    payload = json.dumps(strategy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inspect_rejected_creator_entry(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    parsed = deepcopy(payload.get("parsed") or {})
    prototype_id = _clean(payload.get("prototypeId") or parsed.get("prototypeId"))
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    product_name = _clean(strategy.get("productNameResolved"))
    failure_reason = _clean(payload.get("failureReason"))
    metaphor = parsed.get("metaphoricalEmbodiment") if isinstance(parsed.get("metaphoricalEmbodiment"), dict) else {}
    disposition_report = inspect_literal_symbol_disposition(parsed, tournament_state=state)
    validation_errors_before: List[str] = []
    if failure_reason:
        validation_errors_before.append(failure_reason)
    remaining_errors: List[str] = []
    would_pass = False
    offline_possible = False
    offline_blocked = ""
    try:
        validate_creator_candidate(
            parsed,
            assigned_prototype_id=prototype_id,
            prototype_display_name=prototype_id,
            strategy_foundation=strategy,
            compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
            job_id=_clean(state.get("jobId")),
            tournament_id=_clean(state.get("tournamentId")),
            candidate_id=candidate_id,
            tournament_state=state,
        )
        would_pass = True
    except Exception as exc:
        remaining_errors.append(str(exc.args[0] if getattr(exc, "args", None) else exc))
    ok, blocked = can_offline_revalidate_rejected_creator(
        state,
        candidate_id=candidate_id,
        product_name=product_name,
        compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
    )
    offline_possible = ok
    offline_blocked = blocked or ""
    return {
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "roundIndex": payload.get("roundIndex"),
        "attemptNumber": payload.get("attemptNumber"),
        "responseTextPresent": bool(payload.get("responseText") or parsed),
        "responseTextChars": len(_clean(payload.get("responseText"))),
        "parsedResponsePresent": bool(parsed),
        "responseFingerprint": _response_fingerprint(parsed) if parsed else "",
        "metaphoricalEmbodiment": metaphor,
        "literalSymbolDispositionPresent": bool(_clean(metaphor.get("literalSymbolDisposition"))),
        "literalSymbolDisposition": _clean(metaphor.get("literalSymbolDisposition")) or None,
        "literalSymbolsRejectedOrTransformedPresent": "literalSymbolsRejectedOrTransformed" in metaphor,
        "literalSymbolsRejectedOrTransformedRaw": metaphor.get("literalSymbolsRejectedOrTransformed"),
        "literalSymbolsRejectedOrTransformedType": type(metaphor.get("literalSymbolsRejectedOrTransformed")).__name__
        if "literalSymbolsRejectedOrTransformed" in metaphor
        else None,
        "executionLiteralHits": disposition_report.get("executionLiteralHits"),
        "declaredLiteralHits": disposition_report.get("declaredLiteralHits"),
        "transformationEvidencePresent": disposition_report.get("transformationEvidencePresent"),
        "validationErrorsBeforeCorrectedContract": validation_errors_before,
        "remainingValidationErrorsAfterCorrectedContract": remaining_errors,
        "wouldPassCorrectedContract": would_pass,
        "offlineRecoveryPossible": offline_possible,
        "offlineRecoveryBlockedReason": offline_blocked or None,
        "repairAttempted": bool(payload.get("repairAttempted")),
        "cleanRetryAttempted": bool(payload.get("cleanRetryAttempted")),
    }


def inspect_builder2_creator_regression(state: Dict[str, Any]) -> Dict[str, Any]:
    job_id = _clean(state.get("jobId"))
    report: Dict[str, Any] = {
        "jobId": job_id,
        "tournamentId": _clean(state.get("tournamentId")),
        "paidCalls": 0,
        "stateMutated": False,
        "strategyPresent": isinstance(state.get("strategyFoundation"), dict),
        "strategyFingerprint": _strategy_fingerprint(state),
        "acceptedCreatorsCount": accepted_creator_count(state, read_only=True),
        "rejectedCreatorsCount": 0,
        "acceptedJudgmentsCount": accepted_judgment_count(state, read_only=True),
        "assignedPrototypeIds": assigned_prototype_ids(state),
        "advertisingSloganQualityContractVersion": _clean(state.get("advertisingSloganQualityContractVersion")),
        "metaphoricalEmbodimentContractVersion": _clean(state.get("metaphoricalEmbodimentContractVersion")),
        "creators": [],
        "missingResponses": [],
    }
    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if not isinstance(index, dict):
        index = {}
    report["rejectedCreatorsCount"] = len(index)
    seen_prototypes = set()
    for candidate_id, payload in sorted(index.items()):
        if not isinstance(payload, dict):
            continue
        entry = inspect_rejected_creator_entry(state, candidate_id=candidate_id, payload=payload)
        report["creators"].append(entry)
        seen_prototypes.add(_clean(entry.get("prototypeId")))
    for prototype_id in assigned_prototype_ids(state):
        if prototype_id not in seen_prototypes:
            report["missingResponses"].append(prototype_id)
    report["readyForJudges"] = report["acceptedCreatorsCount"] >= len(report["assignedPrototypeIds"])
    report["reasoningResumePossible"] = report["readyForJudges"] and report["acceptedJudgmentsCount"] == 0
    return report


def main(argv: Optional[List[str]] = None) -> int:
    job_id = (os.environ.get("BUILDER2_CREATOR_REGRESSION_INSPECT_JOB_ID") or "").strip()
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "missing_job_id"}, ensure_ascii=False, indent=2))
        return 1
    state = load_tournament_state(job_id)
    if state is None:
        print(json.dumps({"ok": False, "failureReason": "job_not_found"}, ensure_ascii=False, indent=2))
        return 1
    report = inspect_builder2_creator_regression(state)
    report["ok"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
