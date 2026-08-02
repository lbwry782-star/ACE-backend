"""
Builder2 Creator factual-grounding failure inspector — read-only production audit.

Run:
  BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_JOB_ID=<jobId> \\
  BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_CANDIDATE_ID=<candidateId> \\
  python -m engine.builder2_creator_grounding_failure_inspect
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    can_offline_revalidate_rejected_creator,
    find_rejected_creator_for_prototype,
    load_rejected_creator_parsed_response,
)
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_creator import collect_creator_structural_errors, validate_creator_candidate
from engine.builder2_strategy_evidence_grounding_contract import (
    BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
    CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION,
    build_product_input_audit,
    collect_creator_text_fields,
    compare_creator_relative_advantage_to_strategy,
    contract_version,
    detect_capabilities_in_text,
    requires_strategy_evidence_grounding,
    scan_capability_occurrences,
    scan_texts_for_unsupported_capabilities,
    stamp_creator_evidence_inheritance,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get

logger = logging.getLogger(__name__)

_GROUNDED_IDENTITY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"סוכן\s+פרסום\s+דיגיטלי", re.I),
    re.compile(r"digital[-\s]?advertis", re.I),
    re.compile(r"professional\s+address", re.I),
    re.compile(r"identifiable\s+(?:personal\s+)?name", re.I),
    re.compile(r"named\s+professional", re.I),
    re.compile(r"כתובת\s+מקצועית", re.I),
    re.compile(r"שם\s+(?:אישי|מזוהה)", re.I),
)

_VISUAL_FIELD_PREFIXES: Tuple[str, ...] = (
    "sevenSecondStructure.",
    "visualAnchor.",
    "runwayFeasibility.",
)

_CONCLUSION_VALID_REJECTION = "valid_rejection_actual_unsupported_claim"
_CONCLUSION_VISUAL_METAPHOR = "false_positive_visual_or_strategic_metaphor"
_CONCLUSION_GROUNDED_RA = "false_positive_grounded_relative_advantage"
_CONCLUSION_SELF_REPORT = "creator_self_report_contradiction"
_CONCLUSION_UNAVAILABLE = "response_unavailable"
_CONCLUSION_INSUFFICIENT = "insufficient_persisted_evidence"
_CONCLUSION_NEGATED_CAPABILITY = "false_positive_negated_capability_mention"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _response_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _parsed_fingerprint(parsed: Dict[str, Any]) -> str:
    return _response_fingerprint(json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _failure_field(reason: str) -> Optional[str]:
    msg = _clean(reason)
    if ":" in msg:
        return msg.split(":", 1)[1]
    return None


def _load_product_input(state: Dict[str, Any], job_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    product_name = _clean(state.get("productName") or state.get("product_name"))
    product_description = _clean(state.get("productDescription") or state.get("product_description"))
    target_audience = _clean(state.get("targetAudience") or state.get("target_audience"))
    if isinstance(job_record, dict):
        if not product_name:
            product_name = _clean(job_record.get("productName") or job_record.get("product_name"))
        if not product_description:
            product_description = _clean(job_record.get("productDescription") or job_record.get("product_description"))
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    if not product_name:
        product_name = _clean(strategy.get("productNameResolved"))
    return build_product_input_audit(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _diagnostics_entry(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    by_candidate = state.get("creatorDiagnosticsByCandidate") or {}
    entry = by_candidate.get(candidate_id) or _candidate_record(state, candidate_id).get("creatorDiagnostics") or {}
    return entry if isinstance(entry, dict) else {}


def _field_category(field_path: str) -> str:
    path = _clean(field_path)
    if path.startswith("creatorReport.relativeAdvantage") or path == "creatorReport.relativeAdvantage":
        return "relativeAdvantage"
    if path.startswith("creatorReport.problemPerception"):
        return "problemPerception"
    if path == "conceptSummary":
        return "conceptSummary"
    if path == "coreCreativeMechanism":
        return "coreCreativeMechanism"
    if path.startswith("sevenSecondStructure"):
        return "scene_or_sequence"
    if path.startswith("visualAnchor"):
        return "visualAnchor"
    if path.startswith("runwayFeasibility") or "videoPrompt" in path:
        return "videoPrompt"
    if path.startswith("advertisingClosure"):
        return "advertisingClosure"
    if path.startswith("advertisingSloganFormulation"):
        return "advertisingSlogan"
    return "other"


def _capability_patterns_for(capability: str) -> List[str]:
    return [capability.replace("_", " ")]


def _matches_grounded_relative_advantage(text: str, *, strategy: Dict[str, Any], product_input: Dict[str, Any]) -> bool:
    blob = _clean(text)
    if not blob:
        return False
    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    strategy_ra = _clean(ra.get("statement"))
    product_name = _clean(product_input.get("productName"))
    if product_name and product_name in blob:
        return True
    if strategy_ra and strategy_ra.lower() in blob.lower():
        return True
    return any(pattern.search(blob) for pattern in _GROUNDED_IDENTITY_PATTERNS)


def _looks_like_visual_metaphor(field_path: str, text: str) -> bool:
    category = _field_category(field_path)
    if category in {"scene_or_sequence", "visualAnchor", "videoPrompt"}:
        return True
    if any(field_path.startswith(prefix) for prefix in _VISUAL_FIELD_PREFIXES):
        return True
    blob = _clean(text).lower()
    visual_markers = ("visible", "camera", "frame", "card", "playing card", "נראה", "קלף", "מצלמה")
    return any(marker in blob for marker in visual_markers) and category != "advertisingClosure"


def _is_category_convention(capability: str, *, strategy: Dict[str, Any]) -> bool:
    block = strategy.get("strategyEvidenceGrounding") if isinstance(strategy.get("strategyEvidenceGrounding"), dict) else {}
    conventions = [str(item).lower() for item in (block.get("categoryConventions") or [])]
    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    deps = [str(item).lower() for item in (ra.get("categoryConventionDependencies") or [])]
    needle = capability.replace("_", " ").lower()
    return any(needle in item for item in conventions + deps)


def _analyze_claim(
    capability: str,
    *,
    hits: List[Dict[str, Any]],
    strategy: Dict[str, Any],
    product_input: Dict[str, Any],
    allowed_capabilities: List[str],
) -> Dict[str, Any]:
    matching = [hit for hit in hits if _clean(hit.get("capability")) == capability]
    primary = matching[0] if matching else {}
    field_path = _clean(primary.get("fieldPath"))
    matched_text = _clean(primary.get("matchedText"))
    normalized_claim = capability.replace("_", " ")
    explicit_facts = list((strategy.get("strategyEvidenceGrounding") or {}).get("explicitProductFacts") or [])
    matches_explicit_fact = any(normalized_claim in _clean(fact).lower() or _clean(fact).lower() in matched_text.lower() for fact in explicit_facts)
    matches_allowed = capability in (allowed_capabilities or [])
    category_convention = _is_category_convention(capability, strategy=strategy)
    visual_metaphor = _looks_like_visual_metaphor(field_path, matched_text) if field_path else False
    grounded_ra = _matches_grounded_relative_advantage(matched_text, strategy=strategy, product_input=product_input)
    violates_boundary = not matches_allowed and not grounded_ra and not category_convention and not visual_metaphor
    if violates_boundary:
        classification = _CONCLUSION_VALID_REJECTION
    elif grounded_ra:
        classification = _CONCLUSION_GROUNDED_RA
    elif visual_metaphor:
        classification = _CONCLUSION_VISUAL_METAPHOR
    elif category_convention:
        classification = "category_convention_not_product_fact"
    else:
        classification = "indeterminate_requires_operator_review"
    return {
        "claimText": capability,
        "normalizedClaimText": normalized_claim,
        "fieldPath": field_path or None,
        "sourceCategory": _field_category(field_path) if field_path else None,
        "supportingSubstring": matched_text or None,
        "scannerRule": "scan_texts_for_unsupported_capabilities",
        "scannerPatterns": _capability_patterns_for(capability),
        "whyConsideredNew": "Capability token matched in Creator text and is not in allowedCapabilities.",
        "matchesExplicitProductFact": matches_explicit_fact,
        "matchesAllowedCapability": matches_allowed,
        "isCategoryConventionOnly": category_convention,
        "isVisualMetaphorNotProductAssertion": visual_metaphor,
        "matchesGroundedRelativeAdvantage": grounded_ra,
        "violatesProductionTruthBoundary": violates_boundary,
        "classification": classification,
        "allMatchingFieldPaths": sorted({_clean(hit.get("fieldPath")) for hit in matching if _clean(hit.get("fieldPath"))}),
    }


def discover_rejected_creator_sources(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str = "",
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    index = state.get(REJECTED_CREATOR_PARSED_INDEX_KEY)
    if isinstance(index, dict):
        payload = index.get(candidate_id)
        if isinstance(payload, dict):
            sources.append({"location": REJECTED_CREATOR_PARSED_INDEX_KEY, "payload": dict(payload)})
        if prototype_id:
            for cid, item in index.items():
                if isinstance(item, dict) and _clean(item.get("prototypeId")) == prototype_id:
                    sources.append({"location": f"{REJECTED_CREATOR_PARSED_INDEX_KEY}[prototype={prototype_id}]", "payload": dict(item), "candidateId": cid})
    by_proto = find_rejected_creator_for_prototype(state, prototype_id) if prototype_id else None
    if isinstance(by_proto, dict):
        sources.append({"location": "find_rejected_creator_for_prototype", "payload": dict(by_proto)})
    record = _candidate_record(state, candidate_id)
    if record:
        sources.append(
            {
                "location": "candidates",
                "payload": {
                    "failureReason": record.get("failureReason"),
                    "validationStatus": record.get("validationStatus"),
                    "status": record.get("status"),
                    "parsed": record.get("creatorOutput") or record.get("creatorSnapshot"),
                    "responseLength": (_diagnostics_entry(state, candidate_id) or {}).get("responseLength"),
                },
            }
        )
    diagnostics = _diagnostics_entry(state, candidate_id)
    if diagnostics:
        sources.append({"location": "creatorDiagnosticsByCandidate", "payload": dict(diagnostics)})
    recovery = state.get("offlineCreatorRecoveryAt") or state.get("offlineCreatorRecoveryVersion")
    if recovery:
        sources.append({"location": "offlineCreatorRecoveryMetadata", "payload": {"offlineCreatorRecoveryAt": recovery}})
    return sources


def _resolve_primary_payload(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload = load_rejected_creator_parsed_response(state, candidate_id)
    if payload is not None:
        return payload, REJECTED_CREATOR_PARSED_INDEX_KEY
    if prototype_id:
        payload = find_rejected_creator_for_prototype(state, prototype_id)
        if isinstance(payload, dict) and _clean(payload.get("candidateId")) == candidate_id:
            return payload, "find_rejected_creator_for_prototype"
    record = _candidate_record(state, candidate_id)
    embedded = record.get("creatorOutput") or record.get("creatorSnapshot")
    if isinstance(embedded, dict) and embedded and record.get("validationStatus") in {"creator_rejected", "rejected"}:
        return {
            "candidateId": candidate_id,
            "prototypeId": prototype_id or record.get("prototypeId"),
            "parsed": embedded,
            "failureReason": record.get("failureReason"),
        }, "candidates.creatorOutput"
    return None, None


def _infer_original_rejection_component(
    *,
    original_failure_reason: str,
    creator_self_report: List[str],
) -> str:
    original_field = _failure_field(original_failure_reason)
    if original_field != "newProductClaimsIntroduced":
        return "other_validation_rule" if original_field else "none"
    if creator_self_report:
        return "creator_self_report_only"
    return "server_scanner"


def replay_creator_grounding_validation(
    state: Dict[str, Any],
    *,
    parsed: Dict[str, Any],
    candidate_id: str,
    prototype_id: str,
    original_failure_reason: str = "",
    compatibility_mode: bool = False,
) -> Dict[str, Any]:
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    product_name = _clean(strategy.get("productNameResolved"))
    structural_errors = collect_creator_structural_errors(
        parsed,
        assigned_prototype_id=prototype_id,
        prototype_display_name=prototype_id,
        strategy_foundation=strategy,
        compatibility_mode=compatibility_mode,
        job_id=_clean(state.get("jobId")),
        tournament_id=_clean(state.get("tournamentId")),
        candidate_id=candidate_id,
        prototype_id=prototype_id,
    )
    trial = copy.deepcopy(parsed)
    creator_self_report = list(trial.get("newProductClaimsIntroduced") or []) if isinstance(trial.get("newProductClaimsIntroduced"), list) else []
    stamp_creator_evidence_inheritance(trial, strategy_foundation=strategy)
    scanner_introduced = list(trial.get("newProductClaimsIntroduced") or [])
    contradiction = bool(creator_self_report) and sorted(creator_self_report) != sorted(scanner_introduced)
    structural_accepted = not structural_errors
    factual_accepted = not scanner_introduced
    current_failure_field: Optional[str] = None
    current_failure_reason: Optional[str] = None
    rejection_component = "none"
    try:
        validate_creator_candidate(
            trial,
            assigned_prototype_id=prototype_id,
            prototype_display_name=prototype_id,
            strategy_foundation=strategy,
            compatibility_mode=compatibility_mode,
            job_id=_clean(state.get("jobId")),
            tournament_id=_clean(state.get("tournamentId")),
            candidate_id=candidate_id,
            tournament_state=state,
        )
    except Builder2TournamentError as exc:
        current_failure_reason = str(exc.args[0] if exc.args else "validation_failed")
        current_failure_field = _failure_field(current_failure_reason)
        if current_failure_field == "newProductClaimsIntroduced":
            if creator_self_report and not scanner_introduced:
                rejection_component = "creator_self_report_only"
            elif scanner_introduced:
                rejection_component = "server_scanner"
            else:
                rejection_component = "non_empty_creator_newProductClaimsIntroduced"
        elif structural_errors:
            rejection_component = "structural_validation"
        else:
            rejection_component = "other_validation_rule"
    else:
        rejection_component = "accepted"
    original_rejection_component = _infer_original_rejection_component(
        original_failure_reason=original_failure_reason,
        creator_self_report=creator_self_report,
    )
    return {
        "structuralValidationAccepted": structural_accepted,
        "factualGroundingValidationAccepted": factual_accepted,
        "creatorSelfReportAccepted": not bool(creator_self_report),
        "deterministicScannerAccepted": not bool(scanner_introduced),
        "contradictionDetected": contradiction,
        "currentValidationFailureField": current_failure_field,
        "currentValidationFailureReason": current_failure_reason,
        "rejectionStillValidUnderCurrentContract": bool(current_failure_reason),
        "rejectionReasonChangedSinceOriginalRun": bool(original_failure_reason) and not bool(current_failure_reason),
        "rejectionComponent": rejection_component,
        "originalRejectionComponent": original_rejection_component,
        "creatorProvidedNewProductClaimsIntroduced": creator_self_report,
        "scannerDerivedNewProductClaimsIntroduced": scanner_introduced,
        "structuralErrors": structural_errors,
        "structuralErrorCount": len(structural_errors),
    }


def inspect_creator_grounding_failure(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str = "",
    job_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    sources = discover_rejected_creator_sources(state, candidate_id=candidate_id, prototype_id=prototype_id)
    payload, response_location = _resolve_primary_payload(state, candidate_id=candidate_id, prototype_id=prototype_id)
    prototype_id = (
        prototype_id
        or _clean(record.get("prototypeId"))
        or _clean((payload or {}).get("prototypeId"))
        or _clean(((payload or {}).get("parsed") or {}).get("prototypeId") if isinstance((payload or {}).get("parsed"), dict) else "")
    )
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    product_input = _load_product_input(state, job_record)
    compatibility_mode = not requires_strategy_evidence_grounding(state=state, strategy=strategy)
    diagnostics = _diagnostics_entry(state, candidate_id)
    parsed: Optional[Dict[str, Any]] = None
    raw_text = ""
    if isinstance(payload, dict):
        parsed_candidate = payload.get("parsed")
        if isinstance(parsed_candidate, dict) and parsed_candidate:
            parsed = copy.deepcopy(parsed_candidate)
        raw_text = _clean(payload.get("responseText"))
    response_available = bool(payload)
    raw_response_available = bool(raw_text)
    parsed_response_available = isinstance(parsed, dict) and bool(parsed)
    rejection_reason = _clean((payload or {}).get("failureReason") or record.get("failureReason") or diagnostics.get("failureReason"))
    rejection_field = _failure_field(rejection_reason) or _clean((diagnostics.get("failureFieldPaths") or [None])[0])
    replay = replay_creator_grounding_validation(
        state,
        parsed=parsed or {},
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        original_failure_reason=rejection_reason,
        compatibility_mode=compatibility_mode,
    ) if parsed_response_available else {
        "structuralValidationAccepted": None,
        "factualGroundingValidationAccepted": None,
        "creatorSelfReportAccepted": None,
        "deterministicScannerAccepted": None,
        "contradictionDetected": None,
        "currentValidationFailureField": None,
        "currentValidationFailureReason": None,
        "rejectionStillValidUnderCurrentContract": None,
        "rejectionReasonChangedSinceOriginalRun": None,
        "rejectionComponent": "response_unavailable",
        "creatorProvidedNewProductClaimsIntroduced": [],
        "scannerDerivedNewProductClaimsIntroduced": [],
        "structuralErrors": [],
        "structuralErrorCount": None,
    }
    block = strategy.get("strategyEvidenceGrounding") if isinstance(strategy.get("strategyEvidenceGrounding"), dict) else {}
    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    allowed_capabilities = list(block.get("allowedCapabilities") or product_input.get("explicitCapabilitiesSupplied") or [])
    unsupported_hits: List[Dict[str, Any]] = []
    capability_occurrences: List[Dict[str, Any]] = []
    lexical_tokens: List[str] = []
    introduced: List[str] = []
    if parsed_response_available and parsed is not None:
        creator_fields = collect_creator_text_fields(parsed)
        for field_path, text in creator_fields:
            lexical_tokens.extend(detect_capabilities_in_text(text))
            capability_occurrences.extend(
                scan_capability_occurrences(
                    text,
                    allowed_capabilities=allowed_capabilities,
                    field_path=field_path,
                )
            )
        lexical_tokens = sorted(set(lexical_tokens))
        unsupported_hits = scan_texts_for_unsupported_capabilities(
            creator_fields,
            allowed_capabilities=allowed_capabilities,
        )
        trial = copy.deepcopy(parsed)
        stamp_creator_evidence_inheritance(trial, strategy_foundation=strategy)
        introduced = list(trial.get("newProductClaimsIntroduced") or [])
    claim_analyses = [
        _analyze_claim(
            capability,
            hits=unsupported_hits,
            strategy=strategy,
            product_input=product_input,
            allowed_capabilities=allowed_capabilities,
        )
        for capability in introduced
    ]
    if not claim_analyses and rejection_field == "newProductClaimsIntroduced" and parsed_response_available:
        negated_only = [
            item
            for item in capability_occurrences
            if item.get("occurrenceClassification") == CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION
        ]
        if negated_only and not introduced:
            claim_analyses = [
                {
                    "claimText": item.get("capability"),
                    "normalizedClaimText": str(item.get("capability") or "").replace("_", " "),
                    "fieldPath": item.get("fieldPath"),
                    "sourceCategory": _field_category(_clean(item.get("fieldPath"))),
                    "supportingSubstring": item.get("matchedSpan"),
                    "scannerRule": "scan_texts_for_unsupported_capabilities",
                    "scannerPatterns": _capability_patterns_for(str(item.get("capability") or "")),
                    "whyConsideredNew": "Lexical capability token matched, but occurrence is an explicit negation or truth-boundary denial.",
                    "matchesExplicitProductFact": False,
                    "matchesAllowedCapability": False,
                    "isCategoryConventionOnly": False,
                    "isVisualMetaphorNotProductAssertion": False,
                    "matchesGroundedRelativeAdvantage": False,
                    "violatesProductionTruthBoundary": False,
                    "occurrenceClassification": CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION,
                    "productClaimEmitted": False,
                    "classification": _CONCLUSION_NEGATED_CAPABILITY,
                    "allMatchingFieldPaths": [_clean(item.get("fieldPath"))] if _clean(item.get("fieldPath")) else [],
                }
                for item in negated_only
            ]
        elif unsupported_hits:
            claim_analyses = [
                _analyze_claim(
                    capability,
                    hits=unsupported_hits,
                    strategy=strategy,
                    product_input=product_input,
                    allowed_capabilities=allowed_capabilities,
                )
                for capability in sorted({hit["capability"] for hit in unsupported_hits})
            ]
    offline_ok, offline_blocked = (
        can_offline_revalidate_rejected_creator(
            state,
            candidate_id=candidate_id,
            product_name=_clean(product_input.get("productName")),
            compatibility_mode=compatibility_mode,
        )
        if parsed_response_available
        else (False, "parsed_response_missing")
    )
    plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
    prototype_plan = (plan.get("resumePlanByPrototype") or {}).get(prototype_id) or {}
    replacement_planned = _clean(prototype_plan.get("creatorAction")) == "dispatch"
    classifications = [item.get("classification") for item in claim_analyses]
    negated_lexical_only = bool(capability_occurrences) and not introduced and not unsupported_hits and any(
        item.get("occurrenceClassification") == CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION for item in capability_occurrences
    )
    if not parsed_response_available:
        inspection_conclusion = _CONCLUSION_UNAVAILABLE
    elif not payload:
        inspection_conclusion = _CONCLUSION_INSUFFICIENT
    elif negated_lexical_only and rejection_field == "newProductClaimsIntroduced" and not replay.get("rejectionStillValidUnderCurrentContract"):
        inspection_conclusion = _CONCLUSION_NEGATED_CAPABILITY
    elif all(item.get("classification") == _CONCLUSION_GROUNDED_RA for item in claim_analyses) and claim_analyses:
        inspection_conclusion = _CONCLUSION_GROUNDED_RA
    elif any(item.get("classification") == _CONCLUSION_VISUAL_METAPHOR for item in claim_analyses) and not any(
        item.get("violatesProductionTruthBoundary") for item in claim_analyses
    ):
        inspection_conclusion = _CONCLUSION_VISUAL_METAPHOR
    elif replay.get("contradictionDetected"):
        inspection_conclusion = _CONCLUSION_SELF_REPORT
    elif replay.get("rejectionStillValidUnderCurrentContract"):
        inspection_conclusion = _CONCLUSION_VALID_REJECTION
    elif claim_analyses:
        inspection_conclusion = classifications[0]
    else:
        inspection_conclusion = "rejection_field_without_scanner_hits"
    response_chars = len(raw_text) if raw_text else int(diagnostics.get("responseLength") or payload.get("responseCharacterCount") or 0) if payload else 0
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "callType": _clean((payload or {}).get("callType")) or "normal",
        "attempt": (payload or {}).get("attemptNumber") or record.get("attemptNumber"),
        "responseAvailable": response_available,
        "rawResponseAvailable": raw_response_available,
        "parsedResponseAvailable": parsed_response_available,
        "responseLocation": response_location,
        "responseFingerprint": _response_fingerprint(raw_text) if raw_text else None,
        "parsedResponseFingerprint": _parsed_fingerprint(parsed) if parsed_response_available and parsed else None,
        "responseCharacterCount": response_chars if response_chars else None,
        "rejectionField": rejection_field or None,
        "rejectionReason": rejection_reason or None,
        "originalRejectionReason": rejection_reason or None,
        "originalRejectionComponent": replay.get("originalRejectionComponent"),
        "structuralErrors": replay.get("structuralErrors") or [],
        "structuralErrorCount": replay.get("structuralErrorCount"),
        "responseStructurallyValidUnderCurrentContract": replay.get("structuralValidationAccepted"),
        "productCapabilityClaimsDetected": sorted({hit["capability"] for hit in unsupported_hits}),
        "lexicalCapabilityTokensDetected": lexical_tokens if parsed_response_available else None,
        "capabilityOccurrences": capability_occurrences if parsed_response_available else None,
        "unsupportedProductClaims": unsupported_hits,
        "newProductClaimsIntroduced": introduced if parsed_response_available else None,
        "inheritedProductFacts": list((copy.deepcopy(parsed) if parsed else {}).get("inheritedProductFacts") or []) if parsed_response_available else None,
        "relativeAdvantageRelationshipToStrategy": compare_creator_relative_advantage_to_strategy(parsed or {}, strategy_foundation=strategy)
        if parsed_response_available
        else None,
        "creatorFactuallyGrounded": (not bool(introduced)) if parsed_response_available else None,
        "strategyEvidenceGroundingContractVersion": contract_version(state=state, strategy=strategy) or "legacy_unknown",
        "strategyEvidencePaths": list(block.get("relativeAdvantageEvidenceSourcePaths") or ra.get("relativeAdvantageEvidenceSourcePaths") or []),
        "allowedCapabilities": allowed_capabilities,
        "categoryConventionDependencies": list(ra.get("categoryConventionDependencies") or block.get("categoryConventions") or []),
        "unsupportedAssumptions": list(block.get("unsupportedAssumptions") or ra.get("unsupportedAssumptions") or []),
        "selfPurityEquivalent": {
            "creatorFactuallyGrounded": (not bool(introduced)) if parsed_response_available else None,
            "creatorProvidedNewProductClaimsIntroduced": replay.get("creatorProvidedNewProductClaimsIntroduced"),
            "scannerDerivedNewProductClaimsIntroduced": replay.get("scannerDerivedNewProductClaimsIntroduced"),
        },
        "claimAnalyses": claim_analyses,
        "originalProductInput": product_input,
        "sharedStrategyRelativeAdvantage": _clean(ra.get("statement")) or None,
        "sharedStrategyTruthBoundary": _clean(ra.get("truthBoundary")) or None,
        "structuralValidationAccepted": replay.get("structuralValidationAccepted"),
        "factualGroundingValidationAccepted": replay.get("factualGroundingValidationAccepted"),
        "creatorSelfReportAccepted": replay.get("creatorSelfReportAccepted"),
        "deterministicScannerAccepted": replay.get("deterministicScannerAccepted"),
        "contradictionDetected": replay.get("contradictionDetected"),
        "currentValidationFailureField": replay.get("currentValidationFailureField"),
        "currentValidationFailureReason": replay.get("currentValidationFailureReason"),
        "rejectionStillValidUnderCurrentContract": replay.get("rejectionStillValidUnderCurrentContract"),
        "rejectionReasonChangedSinceOriginalRun": replay.get("rejectionReasonChangedSinceOriginalRun"),
        "rejectionComponent": replay.get("rejectionComponent"),
        "creatorProvidedNewProductClaimsIntroduced": replay.get("creatorProvidedNewProductClaimsIntroduced"),
        "scannerDerivedNewProductClaimsIntroduced": replay.get("scannerDerivedNewProductClaimsIntroduced"),
        "offlineRevalidationPossible": bool(offline_ok),
        "offlineRevalidationBlockedReason": offline_blocked or None,
        "offlinePersistencePossible": parsed_response_available and not _candidate_record(state, candidate_id).get("validationStatus") == "accepted",
        "offlineSalvagePossible": bool(offline_ok),
        "responseCompleteEnoughForSalvage": parsed_response_available and bool(replay.get("structuralValidationAccepted")),
        "replacementCreatorCallCurrentlyPlanned": replacement_planned,
        "replacementCreatorCallCanBeAvoided": bool(offline_ok),
        "replacementCreatorCallAvoidanceBlockedReason": offline_blocked if not offline_ok else None,
        "inspectionConclusion": inspection_conclusion,
        "discoveredSources": [{"location": item.get("location"), "candidateId": item.get("candidateId") or candidate_id} for item in sources],
        "paidCalls": 0,
        "openAICalls": 0,
        "stateMutated": False,
    }


def inspect_creator_grounding_failure_for_job(
    job_id: str,
    *,
    candidate_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if tournament_state is None:
        state = load_tournament_state(job_id)
        if not isinstance(state, dict) or not state:
            if not redis_configured():
                return {
                    "ok": False,
                    "failureReason": "builder2_creator_grounding_failure_inspect_redis_unconfigured",
                    "jobId": job_id,
                    "candidateId": candidate_id,
                }
            return {
                "ok": False,
                "failureReason": "builder2_creator_grounding_failure_inspect_job_not_found",
                "jobId": job_id,
                "candidateId": candidate_id,
            }
        job_record = video_job_get(job_id) if redis_configured() else None
    else:
        state = tournament_state
        job_record = None
    record = _candidate_record(state, candidate_id)
    report = inspect_creator_grounding_failure(
        state,
        candidate_id=candidate_id,
        prototype_id=_clean(record.get("prototypeId")),
        job_record=job_record if isinstance(job_record, dict) else None,
    )
    report["ok"] = True
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_JOB_ID"))
    candidate_id = _clean(os.environ.get("BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_CANDIDATE_ID"))
    if not job_id or not candidate_id:
        print(
            "BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_JOB_ID and "
            "BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_CANDIDATE_ID are required",
            file=sys.stderr,
        )
        return 2
    logger.info(
        "BUILDER2_CREATOR_GROUNDING_FAILURE_INSPECT_START jobId=%s candidateId=%s",
        job_id,
        candidate_id,
    )
    report = inspect_creator_grounding_failure_for_job(job_id, candidate_id=candidate_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
