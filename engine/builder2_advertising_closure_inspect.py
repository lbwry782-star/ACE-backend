"""
Builder2 Advertising Closure foundation inspector — read-only strategic review.

Run:
  BUILDER2_ADVERTISING_CLOSURE_INSPECT_JOB_ID=<jobId> python -m engine.builder2_advertising_closure_inspect
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_accepted_creator_store import load_accepted_creator_candidate
from engine.builder2_accepted_judgment_store import _judgment_record_for_candidate
from engine.builder2_advertising_closure_contract import (
    NEW_PROMISE_PATTERNS,
    _word_count_excluding_product,
    get_advertising_closure_status,
)
from engine.builder2_headline_decision_contract import get_normalized_headline_decision
from engine.builder2_tournament_store import load_tournament_state
from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY
from engine.video_jobs_redis import redis_configured

logger = logging.getLogger(__name__)

DEFAULT_INSPECT_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"

_GENERIC_JOURNEY_PATTERNS = (
    re.compile(r"part\s+of\s+(the|your|my|our)\s+journey", re.I),
    re.compile(r"part\s+of\s+the\s+way", re.I),
    re.compile(r"along\s+the\s+way", re.I),
    re.compile(r"on\s+the\s+journey", re.I),
    re.compile(r"חלק\s+מהדרך"),
    re.compile(r"בדרך\s+שלך"),
    re.compile(r"חלק\s+מ(?:ה)?(?:מ)?סע"),
)

_GENERIC_QUALITY_PATTERNS = (
    re.compile(r"quality\s+you\s+can\s+trust", re.I),
    re.compile(r"the\s+best\s+choice", re.I),
    re.compile(r"better\s+every\s+day", re.I),
    re.compile(r"experience\s+the\s+difference", re.I),
    re.compile(r"איכות\s+ש(?:אפ)?שר\s+(?:ל)?(?:ס)?(?:מ)?(?:וך)?(?: על)?"),
    re.compile(r"הבחירה\s+ה(?:טוב)?(?:ב)?(?:י)?(?:ת)?"),
)

_SENSITIVE_OUTPUT_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "REDIS_URL",
        "startImageArtifact",
        "startImageDataUri",
        "runwayTaskId",
        "runwayVideoUrl",
        "user_id",
        "userId",
        "session_id",
        "sessionId",
        "videoPrompt",
        "videoPromptCore",
    }
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _field_report(*, value: Any, source_path: str, authoritative: bool) -> Dict[str, Any]:
    return {
        "value": value if _present(value) else None,
        "sourcePath": source_path,
        "authoritative": authoritative,
        "present": _present(value),
    }


def _first_present_mapping(
    candidates: List[Tuple[str, Any, bool]],
) -> Tuple[Any, str, bool]:
    for source_path, value, authoritative in candidates:
        if _present(value):
            return value, source_path, authoritative
    return None, candidates[-1][0] if candidates else "missing", candidates[-1][2] if candidates else False


def _resolve_winner_identity(state: Dict[str, Any]) -> Tuple[str, str, Optional[int]]:
    candidate_id = str(
        state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or ""
    ).strip()
    prototype_id = str(state.get("winnerDevelopmentPrototypeId") or "").strip()
    winner_rec = (state.get("candidates") or {}).get(candidate_id) or {}
    if not prototype_id:
        prototype_id = str(winner_rec.get("prototypeId") or "").strip()
    score = winner_rec.get("totalScore")
    if score is None:
        judgment_id = str(winner_rec.get("judgmentId") or "").strip()
        judgment_rec = (state.get("judgments") or {}).get(judgment_id) or {}
        score = judgment_rec.get("totalScore")
    try:
        winner_score = int(score) if score is not None else None
    except (TypeError, ValueError):
        winner_score = None
    return candidate_id or "", prototype_id or "", winner_score


def _load_selected_creator(
    *,
    job_id: str,
    candidate_id: str,
    state: Dict[str, Any],
    winner_rec: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    try:
        snapshot = load_accepted_creator_candidate(
            job_id=job_id,
            candidate_id=candidate_id,
            tournament_state=state,
        )
        return snapshot, "acceptedCreatorCandidates"
    except Exception:
        fallback = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        if isinstance(fallback, dict) and fallback:
            return fallback, "candidates.creatorSnapshot"
        return {}, "missing"


def _resolve_foundation_fields(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    owned: Dict[str, Any],
    strategy: Dict[str, Any],
    creator: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    creator_output = creator.get("creatorOutput") if isinstance(creator.get("creatorOutput"), dict) else creator

    def report_field(name: str, candidates: List[Tuple[str, Any, bool]]) -> Dict[str, Any]:
        value, source_path, authoritative = _first_present_mapping(candidates)
        return _field_report(value=value, source_path=source_path, authoritative=authoritative)

    return {
        "productNameResolved": report_field(
            "productNameResolved",
            [
                ("winnerDevelopmentPlan.productNameResolved", plan.get("productNameResolved"), True),
                ("state.productNameResolved", state.get("productNameResolved"), True),
                ("state.productName", state.get("productName"), True),
                ("strategyFoundation.productNameResolved", strategy.get("productNameResolved"), True),
            ],
        ),
        "language": report_field(
            "language",
            [
                ("winnerDevelopmentPlan.language", plan.get("language"), True),
                ("state.contentLanguage", state.get("contentLanguage"), True),
                ("state.language", state.get("language"), True),
                ("strategyFoundation.language", strategy.get("language"), True),
            ],
        ),
        "problemPerception": report_field(
            "problemPerception",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.problemPerception", owned.get("problemPerception"), True),
                ("winnerDevelopmentPlan.problemPerception", plan.get("problemPerception"), True),
                ("strategyFoundation.problemPerception", strategy.get("problemPerception"), True),
            ],
        ),
        "relativeAdvantage": report_field(
            "relativeAdvantage",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.relativeAdvantage", owned.get("relativeAdvantage"), True),
                ("winnerDevelopmentPlan.relativeAdvantage", plan.get("relativeAdvantage"), True),
                ("strategyFoundation.relativeAdvantage", strategy.get("relativeAdvantage"), True),
            ],
        ),
        "advertisingPromise": report_field(
            "advertisingPromise",
            [
                ("winnerDevelopmentPlan.advertisingPromise", plan.get("advertisingPromise"), True),
                ("winnerDevelopmentPlan.headlineTextRemainder", plan.get("headlineTextRemainder"), True),
            ],
        ),
        "coreCreativeMechanism": report_field(
            "coreCreativeMechanism",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.coreCreativeMechanism", owned.get("coreCreativeMechanism"), True),
                ("winnerDevelopmentPlan.coreCreativeMechanism", plan.get("coreCreativeMechanism"), True),
                ("acceptedCreatorCandidate.coreCreativeMechanism", creator_output.get("coreCreativeMechanism"), True),
            ],
        ),
        "coreVisualIdea": report_field(
            "coreVisualIdea",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.coreVisualIdea", owned.get("coreVisualIdea"), True),
                ("winnerDevelopmentPlan.coreVisualIdea", plan.get("coreVisualIdea"), True),
                ("acceptedCreatorCandidate.coreVisualIdea", creator_output.get("coreVisualIdea") or creator_output.get("conceptSummary"), True),
            ],
        ),
        "visualAnchor": report_field(
            "visualAnchor",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.visualAnchor", owned.get("visualAnchor"), True),
                ("winnerDevelopmentPlan.visualAnchor", plan.get("visualAnchor"), True),
                ("acceptedCreatorCandidate.visualAnchor", creator_output.get("visualAnchor"), True),
            ],
        ),
        "prototypeMethodContract": report_field(
            "prototypeMethodContract",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.prototypeMethodContract", owned.get("prototypeMethodContract"), True),
                ("acceptedCreatorCandidate.prototypeMethodContract", creator.get("prototypeMethodContract") or creator_output.get("prototypeMethodContract"), True),
                ("winnerDevelopmentPlan.prototypeMethodApplication", plan.get("prototypeMethodApplication"), True),
            ],
        ),
        "visualParallelType": report_field(
            "visualParallelType",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.visualParallelType", owned.get("visualParallelType"), True),
                ("winnerDevelopmentPlan.visualParallelType", plan.get("visualParallelType"), True),
                ("acceptedCreatorCandidate.visualParallelType", creator_output.get("visualParallelType"), True),
            ],
        ),
        "participationMechanism": report_field(
            "participationMechanism",
            [
                (f"winnerDevelopmentPlan.{SERVER_OWNED_WINNER_SOURCE_KEY}.participationMechanism", owned.get("participationMechanism"), True),
                ("acceptedCreatorCandidate.participationMechanism", creator_output.get("participationMechanism"), True),
                ("winnerDevelopmentPlan.participationMechanism", plan.get("participationMechanism"), True),
            ],
        ),
        "headline": report_field(
            "headline",
            [
                ("winnerDevelopmentPlan.headline", plan.get("headline"), True),
            ],
        ),
        "headlineText": report_field(
            "headlineText",
            [
                ("winnerDevelopmentPlan.headlineText", plan.get("headlineText"), True),
            ],
        ),
        "headlineTextRemainder": report_field(
            "headlineTextRemainder",
            [
                ("winnerDevelopmentPlan.headlineTextRemainder", plan.get("headlineTextRemainder"), True),
            ],
        ),
        "advertisingClosureSloganText": report_field(
            "advertisingClosureSloganText",
            [
                ("advertisingClosure.sloganText", (state.get("advertisingClosure") or plan.get("advertisingClosure") or {}).get("sloganText"), False),
            ],
        ),
    }


def _resolve_current_proposal(state: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    proposal = state.get("advertisingClosure")
    source_path = "state.advertisingClosure"
    if not isinstance(proposal, dict):
        proposal = plan.get("advertisingClosure")
        source_path = "winnerDevelopmentPlan.advertisingClosure"
    if not isinstance(proposal, dict):
        media = state.get("mediaResume")
        if isinstance(media, dict) and isinstance(media.get("advertisingClosureArtifact"), dict):
            proposal = media.get("advertisingClosureArtifact")
            source_path = "mediaResume.advertisingClosureArtifact"
    if not isinstance(proposal, dict):
        proposal = {}
    status = get_advertising_closure_status(state)
    return {
        "status": status,
        "sourcePath": source_path if _present(proposal) else "missing",
        "productNameText": str(proposal.get("productNameText") or "").strip() or None,
        "sloganText": str(proposal.get("sloganText") or "").strip() or None,
        "authoritative": False,
        "present": _present(proposal),
    }


def _slogan_word_count(slogan: str) -> int:
    return len(re.findall(r"\S+", str(slogan or "").strip()))


def build_proposal_diagnostics(
    *,
    product_name: str,
    slogan: str,
    status: str,
    media: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    product = str(product_name or "").strip()
    text = str(slogan or "").strip()
    lowered_product = product.lower()
    lowered_slogan = text.lower()
    diagnostics = {
        "diagnosticNote": "Deterministic pattern checks only; not an automatic creative verdict.",
        "sloganWordCount": _slogan_word_count(text),
        "productNameRepeatedInsideSlogan": bool(product and lowered_product in lowered_slogan),
        "sloganEqualsProductName": bool(product and text == product),
        "sloganIsEmpty": not bool(text),
        "sloganExceedsSevenWords": _word_count_excluding_product(text, product) > 7 if text else False,
        "sloganIntroducesUnsupportedSuperiorityLanguage": any(pattern.search(text) for pattern in NEW_PROMISE_PATTERNS),
        "sloganContainsGenericJourneyPhrase": any(pattern.search(text) for pattern in _GENERIC_JOURNEY_PATTERNS),
        "sloganContainsGenericQualityPhrase": any(pattern.search(text) for pattern in _GENERIC_QUALITY_PATTERNS),
        "proposalApproved": status in {"approved", "completed"},
        "proposalRendered": status == "completed" and bool((media or {}).get("finalVideoWithClosureUrl")),
    }
    return diagnostics


def inspect_advertising_closure_foundation(
    job_id: str,
    *,
    tournament_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    jid = str(job_id or "").strip()
    load_tournament = tournament_loader or load_tournament_state
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "tournamentExists": False,
        "winnerLoaded": False,
        "winnerCandidateId": None,
        "winnerPrototypeId": None,
        "winnerScore": None,
        "foundation": {},
        "headlineDecision": None,
        "selectedJudgment": {},
        "currentProposal": {},
        "proposalDiagnostics": {},
        "redisMutations": 0,
        "openAICalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "ffmpegCalls": 0,
        "ok": False,
    }
    if not jid:
        report["failureReason"] = "builder2_advertising_closure_inspect_job_id_missing"
        return report
    if not redis_configured() and tournament_loader is None:
        report["failureReason"] = "builder2_advertising_closure_inspect_redis_unconfigured"
        return report

    state = load_tournament(jid)
    if state is None:
        report["failureReason"] = "builder2_advertising_closure_inspect_job_not_found"
        return report

    report["tournamentExists"] = True
    candidate_id, prototype_id, winner_score = _resolve_winner_identity(state)
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict):
        report["failureReason"] = "builder2_advertising_closure_inspect_missing_winner_plan"
        return report

    report["winnerLoaded"] = True
    report["winnerCandidateId"] = candidate_id or None
    report["winnerPrototypeId"] = prototype_id or None
    report["winnerScore"] = winner_score

    owned = plan.get(SERVER_OWNED_WINNER_SOURCE_KEY)
    if not isinstance(owned, dict):
        owned = state.get(SERVER_OWNED_WINNER_SOURCE_KEY) if isinstance(state.get(SERVER_OWNED_WINNER_SOURCE_KEY), dict) else {}
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    winner_rec = (state.get("candidates") or {}).get(candidate_id) or {}
    creator, creator_source = _load_selected_creator(job_id=jid, candidate_id=candidate_id, state=state, winner_rec=winner_rec)

    report["foundation"] = _resolve_foundation_fields(
        state=state,
        plan=plan,
        owned=owned if isinstance(owned, dict) else {},
        strategy=strategy,
        creator={"creatorOutput": creator, "prototypeMethodContract": creator.get("prototypeMethodContract")},
    )
    report["headlineDecision"] = get_normalized_headline_decision(plan)

    judgment_record = _judgment_record_for_candidate(state, candidate_id)
    judgment = (judgment_record or {}).get("judgment") if isinstance(judgment_record, dict) else {}
    if not isinstance(judgment, dict):
        judgment = {}
    report["selectedJudgment"] = {
        "candidateId": _field_report(
            value=judgment.get("candidateId") or candidate_id,
            source_path="judgments.judgment.candidateId",
            authoritative=True,
        ),
        "headlineNecessityAssessment": _field_report(
            value=judgment.get("headlineNecessityAssessment"),
            source_path="judgments.judgment.headlineNecessityAssessment",
            authoritative=True,
        ),
        "advertisingCompletionAssessment": _field_report(
            value=judgment.get("advertisingCompletionAssessment"),
            source_path="judgments.judgment.advertisingCompletionAssessment",
            authoritative=True,
        ),
        "creatorSourcePath": _field_report(
            value=creator_source,
            source_path="acceptedCreatorCandidates|fallback",
            authoritative=True,
        ),
    }

    current_proposal = _resolve_current_proposal(state, plan)
    report["currentProposal"] = current_proposal
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    report["proposalDiagnostics"] = build_proposal_diagnostics(
        product_name=str(current_proposal.get("productNameText") or ""),
        slogan=str(current_proposal.get("sloganText") or ""),
        status=str(current_proposal.get("status") or get_advertising_closure_status(state)),
        media=media if isinstance(media, dict) else {},
    )
    report["ok"] = True
    return report


def print_advertising_closure_inspect_report(report: Dict[str, Any]) -> None:
    safe = {key: value for key, value in report.items() if key not in _SENSITIVE_OUTPUT_KEYS}
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_ADVERTISING_CLOSURE_INSPECT_JOB_ID", DEFAULT_INSPECT_JOB_ID)
    logger.info("BUILDER2_ADVERTISING_CLOSURE_INSPECT_START jobId=%s", job_id)
    report = inspect_advertising_closure_foundation(job_id)
    print_advertising_closure_inspect_report(report)
    logger.info(
        "BUILDER2_ADVERTISING_CLOSURE_INSPECT_DONE jobId=%s ok=%s status=%s",
        job_id,
        report.get("ok"),
        (report.get("currentProposal") or {}).get("status"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
