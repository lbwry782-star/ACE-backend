"""
Builder2 Strategy grounding inspector — read-only production audit.

Run:
  BUILDER2_STRATEGY_GROUNDING_INSPECT_JOB_ID=<jobId> python -m engine.builder2_strategy_grounding_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from engine.builder2_strategy_evidence_grounding_contract import (
    BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
    build_product_input_audit,
    collect_creator_text_fields,
    compare_creator_relative_advantage_to_strategy,
    contract_version,
    detect_capabilities_in_text,
    inspect_disputed_capability_introduction,
    requires_strategy_evidence_grounding,
    scan_texts_for_unsupported_capabilities,
    strategy_fingerprint,
)
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    return str(value or "").strip()


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


def _creator_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    creator = record.get("creatorOutput") or record.get("creatorSnapshot") or {}
    return creator if isinstance(creator, dict) else {}


def inspect_strategy_grounding(
    state: Dict[str, Any],
    *,
    job_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    product_input = _load_product_input(state, job_record)
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    grounding = strategy.get("strategyEvidenceGrounding") if isinstance(strategy.get("strategyEvidenceGrounding"), dict) else {}
    pp = strategy.get("problemPerception") if isinstance(strategy.get("problemPerception"), dict) else {}
    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    ms = strategy.get("mechanismScan") if isinstance(strategy.get("mechanismScan"), dict) else {}

    creators: List[Dict[str, Any]] = []
    creator_payloads: List[Dict[str, Any]] = []
    for candidate_id, record in sorted((state.get("candidates") or {}).items()):
        if not isinstance(record, dict):
            continue
        creator = _creator_record(state, candidate_id)
        if not creator:
            continue
        creator_payloads.append(creator)
        report = creator.get("creatorReport") if isinstance(creator.get("creatorReport"), dict) else {}
        unsupported = scan_texts_for_unsupported_capabilities(
            collect_creator_text_fields(creator),
            allowed_capabilities=grounding.get("allowedCapabilities") or product_input.get("explicitCapabilitiesSupplied") or [],
        )
        creators.append(
            {
                "candidateId": candidate_id,
                "prototypeId": _clean(record.get("prototypeId") or creator.get("prototypeId")),
                "problemPerception": _clean(report.get("problemPerception")),
                "relativeAdvantage": _clean(report.get("relativeAdvantage")),
                "coreCreativeMechanism": _clean(creator.get("coreCreativeMechanism")),
                "conceptSummary": _clean(creator.get("conceptSummary")),
                "advertisingSlogan": _clean((creator.get("advertisingClosure") or {}).get("sloganText")),
                "productCapabilityClaimsDetected": sorted({hit["capability"] for hit in unsupported}),
                "relativeAdvantageRelationshipToStrategy": compare_creator_relative_advantage_to_strategy(
                    creator,
                    strategy_foundation=strategy,
                ),
                "unsupportedProductClaims": unsupported,
                "inheritedProductFacts": list(creator.get("inheritedProductFacts") or []),
                "newProductClaimsIntroduced": list(creator.get("newProductClaimsIntroduced") or []),
                "creatorFactuallyGrounded": creator.get("creatorFactuallyGrounded"),
            }
        )

    judgments: List[Dict[str, Any]] = []
    for judgment_id, record in sorted((state.get("judgments") or {}).items()):
        if not isinstance(record, dict):
            continue
        judgment = record.get("judgment") if isinstance(record.get("judgment"), dict) else {}
        candidate_id = _clean(judgment.get("candidateId") or record.get("candidateId"))
        creator = _creator_record(state, candidate_id)
        factual = judgment.get("factualGroundingAssessment") if isinstance(judgment.get("factualGroundingAssessment"), dict) else {}
        unsupported = scan_texts_for_unsupported_capabilities(
            collect_creator_text_fields(creator),
            allowed_capabilities=grounding.get("allowedCapabilities") or product_input.get("explicitCapabilitiesSupplied") or [],
        )
        judgments.append(
            {
                "judgmentId": judgment_id,
                "candidateId": candidate_id,
                "prototypeId": _clean((state.get("candidates") or {}).get(candidate_id, {}).get("prototypeId")),
                "score": int((judgment.get("scores") or {}).get("total") or record.get("totalScore") or 0),
                "eligible": judgment.get("eligible"),
                "factualGroundingAssessment": factual,
                "unsupportedFeatureClaimsDetected": sorted({hit["capability"] for hit in unsupported}),
                "comparedAgainstOriginalProductInput": factual.get("comparedAgainstOriginalProductInput"),
                "wouldRemainEligibleWithoutDisputedCapability": not bool(unsupported) or judgment.get("eligible") is not True,
            }
        )

    winner_id = _clean(state.get("winnerCandidateId") or state.get("winnerDevelopmentCandidateId"))
    winner_rec = (state.get("candidates") or {}).get(winner_id) or {}
    winner_plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    winner_creator = _creator_record(state, winner_id)
    disputed = inspect_disputed_capability_introduction(
        product_input=product_input,
        strategy=strategy,
        candidates=creator_payloads,
    )
    shared_strategy = bool(strategy) and all(
        _clean(_creator_record(state, cid).get("strategyFoundationId")) == _clean(strategy.get("strategyFoundationId"))
        for cid in (state.get("candidates") or {})
        if _creator_record(state, cid)
    )
    divergence = sorted(
        {
            item["relativeAdvantageRelationshipToStrategy"]
            for item in creators
            if item.get("relativeAdvantageRelationshipToStrategy") not in {"identical_to_strategy", "semantically_inherited"}
        }
    )
    unsupported_all = sorted(
        {
            hit["capability"]
            for hit in scan_texts_for_unsupported_capabilities(
                [
                    *[
                        (field, _clean(value))
                        for field, value in (
                            ("strategy.problemPerception.statement", pp.get("statement")),
                            ("strategy.relativeAdvantage.statement", ra.get("statement")),
                        )
                    ],
                    *collect_creator_text_fields(winner_creator),
                ],
                allowed_capabilities=grounding.get("allowedCapabilities") or product_input.get("explicitCapabilitiesSupplied") or [],
            )
        }
    )
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "strategyEvidenceGroundingContractVersion": contract_version(state=state, strategy=strategy) or "legacy_unknown",
        "originalProductInput": product_input,
        "sharedStrategy": {
            "strategyFoundationId": _clean(strategy.get("strategyFoundationId")),
            "strategyFingerprint": strategy_fingerprint(strategy) if strategy else "",
            "problemPerception": pp,
            "relativeAdvantage": ra,
            "mechanismScan": ms,
            "strategyEvidenceGrounding": grounding or {"contractVersion": "legacy_unknown"},
            "unsupportedAssumptionsDetected": list(grounding.get("unsupportedAssumptions") or []),
        },
        "creators": creators,
        "judgments": judgments,
        "winner": {
            "winnerCandidateId": winner_id,
            "winnerPrototypeId": _clean(winner_rec.get("prototypeId")),
            "preservedProblemPerception": _clean(winner_plan.get("problemPerception") or (strategy.get("problemPerception") or {}).get("statement")),
            "preservedRelativeAdvantage": _clean(winner_plan.get("relativeAdvantage") or (strategy.get("relativeAdvantage") or {}).get("statement")),
            "productCapabilityClaimsDetected": detect_capabilities_in_text(
                " ".join(
                    part
                    for part in (
                        _clean(winner_plan.get("coreCreativeMechanism")),
                        _clean((winner_plan.get("advertisingClosure") or {}).get("sloganText")),
                        _clean(winner_creator.get("coreCreativeMechanism")),
                    )
                    if part
                )
            ),
            "relationshipToSharedStrategy": "server_owned_preservation",
        },
        "sharedStrategyUsedByAllCreators": shared_strategy,
        "canonicalProblemPerceptionShared": _clean(pp.get("statement")),
        "canonicalRelativeAdvantageShared": _clean(ra.get("statement")),
        "creatorRelativeAdvantageDivergence": divergence,
        "strategySupportedByInput": not bool(grounding.get("unsupportedAssumptions")) if grounding else None,
        "relativeAdvantageSupportedByInput": ra.get("relativeAdvantageFactuallyGrounded"),
        "unsupportedProductClaims": unsupported_all,
        "allCreatorsAffected": len(creators) >= 6,
        **disputed,
        "paidCalls": 0,
        "openAICalls": 0,
        "stateMutated": False,
    }


def inspect_strategy_grounding_for_job(
    job_id: str,
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if tournament_state is None:
        if not redis_configured():
            return {"ok": False, "failureReason": "builder2_strategy_grounding_inspect_redis_unconfigured", "jobId": job_id}
        state = load_tournament_state(job_id)
        job_record = video_job_get(job_id)
    else:
        state = tournament_state
        job_record = None
    if not isinstance(state, dict) or not state:
        return {"ok": False, "failureReason": "builder2_strategy_grounding_inspect_job_not_found", "jobId": job_id}
    report = inspect_strategy_grounding(state, job_record=job_record if isinstance(job_record, dict) else None)
    report["ok"] = True
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_STRATEGY_GROUNDING_INSPECT_JOB_ID"))
    if not job_id:
        print("BUILDER2_STRATEGY_GROUNDING_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    logger.info("BUILDER2_STRATEGY_GROUNDING_INSPECT_START jobId=%s", job_id)
    report = inspect_strategy_grounding_for_job(job_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
