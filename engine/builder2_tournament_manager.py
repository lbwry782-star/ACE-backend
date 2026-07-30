"""
Builder2 tournament manager — deterministic orchestration (never a model role).
"""
from __future__ import annotations

import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_accepted_creator_store import (
    persist_accepted_creator_candidate,
    update_candidate_judge_state,
)
from engine.builder2_creator import generate_creator_candidate
from engine.builder2_creator_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE,
    assert_creator_contract_available,
    is_creator_contract_circuit_breaker_tripped,
    record_process_contract_failure,
)
from engine.builder2_judge import judge_candidate
from engine.builder2_judge_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE as JUDGE_SYSTEMIC_FAILURE_CODE,
    assert_judge_contract_available,
    is_judge_contract_circuit_breaker_tripped,
    record_judge_process_contract_failure,
)
from engine.builder2_prototypes import require_prototype
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_tournament_config import (
    resolve_builder2_active_prototype_ids,
    resolve_builder2_creator_model,
    resolve_builder2_tournament_attempts_per_prototype_per_round,
    resolve_builder2_tournament_eliminations_per_round,
    resolve_builder2_tournament_max_rounds,
)
from engine.builder2_strategy import generate_strategy_foundation
from engine.builder2_tournament_contracts import Builder2TournamentError, compare_candidate_rankings
from engine.builder2_tournament_metrics import (
    MetricsTimer,
    ensure_metrics,
    finalize_tournament_metrics,
    record_creator_eligible,
    record_judge_valid,
    record_model_call,
)
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import assign_strategy_foundation_identity
from engine.builder2_tournament_store import (
    ensure_methodology_compatibility_decided,
    load_tournament_state,
    mutate_tournament_state,
    new_tournament_state,
    record_process_failure_tag,
    register_candidate,
    register_judgment,
    save_tournament_state,
    update_best_candidate_if_stronger,
)
from engine.builder2_winner_development import (
    develop_builder2_winning_candidate,
    normalize_winner_plan_for_runway,
)
from engine.builder2_winner_persistence import (
    WINNER_DEVELOPMENT_SOURCE_NORMAL,
    persist_accepted_winner_development_for_media,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shuffle_prototypes(active_ids: List[str], seed: str) -> List[str]:
    rng = random.Random(seed)
    deck = list(active_ids)
    rng.shuffle(deck)
    return deck


def _candidate_rank_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    cand = state["candidates"][candidate_id]
    return {
        "candidateId": candidate_id,
        "totalScore": cand.get("totalScore", -1),
        "tieScores": cand.get("tieScores") or {},
        "completedAt": cand.get("completedAt") or "",
        "eligible": bool(cand.get("eligible")),
    }


def _creator_was_accepted(cand: Dict[str, Any], *, state: Optional[Dict[str, Any]] = None) -> bool:
    if cand.get("creatorAcceptanceStatus") == "accepted":
        return True
    candidate_id = str(cand.get("candidateId") or "")
    if state is not None and candidate_id:
        index = state.get("acceptedCreatorCandidates") or {}
        if candidate_id in index:
            return True
    if cand.get("validationStatus") == "accepted":
        if isinstance(cand.get("creatorOutput"), dict):
            return True
        if cand.get("eligible") or cand.get("totalScore") is not None or cand.get("judgmentId"):
            return True
    return False


def _has_valid_judgment(cand: Dict[str, Any]) -> bool:
    if cand.get("judgmentId"):
        return True
    if cand.get("judgeStatus") == "accepted":
        return True
    if cand.get("judgeStatus") in (None, "pending") and cand.get("validationStatus") == "accepted":
        return cand.get("eligible") is True or cand.get("totalScore") is not None
    return False


def _resolve_judgment_for_candidate(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    cand = (state.get("candidates") or {}).get(candidate_id) or {}
    judgment_id = cand.get("judgmentId")
    if not judgment_id:
        snapshot = cand.get("judgmentSnapshot")
        return snapshot if isinstance(snapshot, dict) else {}
    record = (state.get("judgments") or {}).get(judgment_id) or {}
    judgment = record.get("judgment")
    return judgment if isinstance(judgment, dict) else {}


def select_global_winner(state: Dict[str, Any]) -> str:
    from engine.builder2_metaphorical_embodiment_contract import judgment_rejects_literal_execution
    from engine.builder2_no_logo_contract import judgment_rejects_logo_policy

    def _literal_winner_blocked(candidate_id: str) -> bool:
        judgment = _resolve_judgment_for_candidate(state, candidate_id)
        return judgment_rejects_literal_execution(judgment) if isinstance(judgment, dict) else False

    def _logo_winner_blocked(candidate_id: str) -> bool:
        judgment = _resolve_judgment_for_candidate(state, candidate_id)
        return judgment_rejects_logo_policy(judgment) if isinstance(judgment, dict) else False

    eligible_ids = [
        cid
        for cid, cand in state["candidates"].items()
        if cand.get("eligible")
        and _creator_was_accepted(cand, state=state)
        and _has_valid_judgment(cand)
        and not _literal_winner_blocked(cid)
        and not _logo_winner_blocked(cid)
    ]
    if eligible_ids:
        best_id = eligible_ids[0]
        best_record = _candidate_rank_record(state, best_id)
        for cid in eligible_ids[1:]:
            record = _candidate_rank_record(state, cid)
            if compare_candidate_rankings(record, best_record) > 0:
                best_id = cid
                best_record = record
        return best_id

    creator_accepted = [
        cand
        for cand in state["candidates"].values()
        if _creator_was_accepted(cand, state=state)
    ]
    judged = [cand for cand in creator_accepted if _has_valid_judgment(cand)]
    if judged and len(judged) == len(creator_accepted):
        raise Builder2TournamentError("builder2_tournament_no_eligible_candidate")
    raise Builder2TournamentError("builder2_tournament_no_valid_candidate")


def _round_record(state: Dict[str, Any], round_index: int) -> Dict[str, Any]:
    for rnd in state["rounds"]:
        if rnd.get("roundIndex") == round_index:
            return rnd
    raise Builder2TournamentError("builder2_tournament_state_error")


def _ensure_round(state: Dict[str, Any], round_index: int, deck: List[str]) -> None:
    for rnd in state["rounds"]:
        if rnd.get("roundIndex") == round_index:
            return
    state["rounds"].append(
        {
            "roundIndex": round_index,
            "shuffledPrototypeOrder": list(deck),
            "attemptsRequested": resolve_builder2_tournament_attempts_per_prototype_per_round(),
            "attemptsCompleted": 0,
            "judgmentsCompleted": 0,
            "bestCandidateByPrototype": {},
            "eliminatedPrototypeId": None,
            "eliminationReason": None,
        }
    )


def _count_completed_rounds(state: Dict[str, Any]) -> int:
    return len([rnd for rnd in state.get("rounds", []) if rnd.get("roundComplete")])


def _should_eliminate_after_round(*, max_rounds: int, completed_rounds: int, state: Dict[str, Any]) -> bool:
    if len(state.get("activePrototypeIds") or []) <= 1:
        return False
    if max_rounds > 0 and completed_rounds >= max_rounds:
        return False
    if max_rounds == 0:
        return len(state.get("activePrototypeIds") or []) > 1
    return completed_rounds < max_rounds and len(state.get("activePrototypeIds") or []) > 1


def _prototypes_for_round(state: Dict[str, Any], round_index: int) -> List[str]:
    initial = list(state.get("initialActivePrototypeIds") or state.get("activePrototypeIds") or [])
    if round_index == 1:
        source = initial
    else:
        source = list(state.get("activePrototypeIds") or [])
    return _current_round_deck(state, round_index, source)


def _next_step_name(state: Dict[str, Any]) -> str:
    if not state.get("strategyFoundation"):
        return "strategy"
    completed = _count_completed_rounds(state)
    max_rounds = resolve_builder2_tournament_max_rounds()
    round_index = state.get("currentRound") or 1
    if max_rounds > 0 and completed >= max_rounds:
        if not state.get("winnerCandidateId"):
            return "select_winner"
        if not state.get("winnerDevelopmentPlan"):
            return "winner_development"
        return "normalize_plan"
    if _round_is_complete(state, round_index) and _should_eliminate_after_round(
        max_rounds=max_rounds,
        completed_rounds=completed,
        state=state,
    ):
        return f"round_{round_index + 1}"
    if not _round_is_complete(state, round_index):
        return f"round_{round_index}"
    return "select_winner"


def _generate_strategy(
    *,
    product_name: str,
    product_description: str,
    language: str,
    llm_client: Optional[Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    return generate_strategy_foundation(
        product_name=product_name,
        product_description=product_description,
        language=language,
        llm_client=llm_client,
        state=state,
    )


def _register_rejected_creator(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    failure_reason: str,
    creator_diagnostics: Optional[Dict[str, Any]] = None,
    parsed_creator: Optional[Dict[str, Any]] = None,
) -> None:
    if isinstance(parsed_creator, dict) and parsed_creator:
        from engine.builder2_complete_ad_creator_recovery import persist_rejected_creator_parsed_response

        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            round_index=round_index,
            attempt_number=attempt_number,
            parsed=parsed_creator,
            failure_reason=failure_reason,
            top_level_keys=sorted(parsed_creator.keys()),
        )
    register_candidate(
        state,
        {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "roundIndex": round_index,
            "attemptNumber": attempt_number,
            "creatorOutput": None,
            "creatorDiagnostics": dict(creator_diagnostics or {}),
            "validationStatus": "creator_rejected",
            "status": "creator_rejected",
            "judgmentId": None,
            "eligible": False,
            "totalScore": None,
            "tieScores": {},
            "failureReason": failure_reason,
            "completedAt": _utc_now_iso(),
        },
    )


def _run_creator_and_judge_for_assignment(
    *,
    state: Dict[str, Any],
    product_name: str,
    product_description: str,
    language: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    runway_mode: str,
    llm_client: Optional[Any],
    compatibility_mode: bool = False,
) -> None:
    strategy = state["strategyFoundation"]
    existing_judged = [
        c
        for c in state["candidates"].values()
        if c.get("prototypeId") == prototype_id
        and c.get("roundIndex") == round_index
        and c.get("attemptNumber") == attempt_number
        and c.get("validationStatus") == "accepted"
        and c.get("judgmentId")
    ]
    if existing_judged:
        return

    existing_rejected = [
        c
        for c in state["candidates"].values()
        if c.get("prototypeId") == prototype_id
        and c.get("roundIndex") == round_index
        and c.get("attemptNumber") == attempt_number
        and c.get("validationStatus") == "creator_rejected"
    ]
    if existing_rejected:
        return

    existing_judge_unavailable = [
        c
        for c in state["candidates"].values()
        if c.get("prototypeId") == prototype_id
        and c.get("roundIndex") == round_index
        and c.get("attemptNumber") == attempt_number
        and c.get("judgeStatus") == "unavailable"
    ]
    if existing_judge_unavailable:
        return

    pending = [
        c
        for c in state["candidates"].values()
        if c.get("prototypeId") == prototype_id
        and c.get("roundIndex") == round_index
        and c.get("attemptNumber") == attempt_number
        and _creator_was_accepted(c, state=state)
        and c.get("judgeStatus") in (None, "pending")
        and not c.get("judgmentId")
    ]
    if pending:
        candidate_id = pending[0]["candidateId"]
        candidate = pending[0].get("creatorSnapshot") or pending[0]["creatorOutput"]
        if candidate_id not in (state.get("acceptedCreatorCandidates") or {}):
            persist_accepted_creator_candidate(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                round_index=round_index,
                attempt_number=attempt_number,
                creator_output=candidate,
                strategy_foundation=strategy,
            )
            save_tournament_state(str(state.get("jobId") or ""), state)
    else:
        logger.info(
            "BUILDER2_PROTOTYPE_ASSIGNED prototypeId=%s roundIndex=%s attempt=%s",
            prototype_id,
            round_index,
            attempt_number,
        )
        assert_creator_contract_available(state)
        candidate_id = f"cand-{round_index}-{prototype_id}-{attempt_number}-{uuid.uuid4().hex[:8]}"
        try:
            candidate_id, candidate = generate_creator_candidate(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=strategy,
                prototype_id=prototype_id,
                round_index=round_index,
                attempt_number=attempt_number,
                runway_mode=runway_mode,
                llm_client=llm_client,
                state=state,
                candidate_id=candidate_id,
                compatibility_mode=compatibility_mode,
            )
        except Builder2TournamentError as exc:
            reason = str(exc.args[0] if exc.args else "builder2_creator_invalid_candidate")
            if reason.startswith(SYSTEMIC_FAILURE_CODE) or is_creator_contract_circuit_breaker_tripped(state):
                record_process_contract_failure(state, exc)
                save_tournament_state(str(state.get("jobId") or ""), state)
                raise
            record_process_failure_tag(state, reason)
            diagnostics = (state.get("creatorDiagnosticsByCandidate") or {}).get(candidate_id, {})
            logger.info(
                "BUILDER2_CREATOR_REJECTED prototypeId=%s candidateId=%s reason=%s",
                prototype_id,
                candidate_id,
                reason,
            )
            _register_rejected_creator(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                round_index=round_index,
                attempt_number=attempt_number,
                failure_reason=reason,
                creator_diagnostics=diagnostics,
            )
            return
        register_candidate(
            state,
            {
                "candidateId": candidate_id,
                "prototypeId": prototype_id,
                "roundIndex": round_index,
                "attemptNumber": attempt_number,
                "creatorOutput": candidate,
                "creatorDiagnostics": dict((state.get("creatorDiagnosticsByCandidate") or {}).get(candidate_id, {})),
                "validationStatus": "accepted",
                "creatorAcceptanceStatus": "accepted",
                "status": "accepted",
                "judgeStatus": "pending",
                "judgmentSnapshot": None,
                "judgeFailure": None,
                "judgmentId": None,
                "eligible": False,
                "totalScore": None,
                "tieScores": {},
                "failureReason": None,
                "completedAt": _utc_now_iso(),
            },
        )
        persist_accepted_creator_candidate(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            round_index=round_index,
            attempt_number=attempt_number,
            creator_output=candidate,
            strategy_foundation=strategy,
        )
        save_tournament_state(str(state.get("jobId") or ""), state)

    cand_rec = state["candidates"][candidate_id]
    candidate = cand_rec.get("creatorSnapshot") or cand_rec.get("creatorOutput") or candidate
    if cand_rec.get("judgmentId"):
        return

    try:
        assert_judge_contract_available(state)
        judgment_id = f"judge-{candidate_id}-{uuid.uuid4().hex[:8]}"
        judgment_id, judgment, total, scores = judge_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype_id=prototype_id,
            candidate_id=candidate_id,
            candidate=candidate,
            llm_client=llm_client,
            state=state,
            judgment_id=judgment_id,
            compatibility_mode=compatibility_mode,
        )
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_judge_invalid_response")
        if reason.startswith(JUDGE_SYSTEMIC_FAILURE_CODE) or is_judge_contract_circuit_breaker_tripped(state):
            if not reason.startswith(JUDGE_SYSTEMIC_FAILURE_CODE):
                breaker = state.get("judgeContractCircuitBreaker") or {}
                paths = breaker.get("repeatedFieldPaths") or []
                trip_reason = breaker.get("trippedReason") or "contract_failure"
                exc = Builder2TournamentError(
                    f"{JUDGE_SYSTEMIC_FAILURE_CODE}:{trip_reason}:{','.join(paths[:8])}"
                )
            record_judge_process_contract_failure(state, exc)
            save_tournament_state(str(state.get("jobId") or ""), state)
            raise exc
        record_process_failure_tag(state, reason)
        diagnostics = (state.get("judgeDiagnosticsByCandidate") or {}).get(candidate_id, {})
        logger.info(
            "BUILDER2_JUDGE_REJECTED candidateId=%s judgmentId=%s reason=%s",
            candidate_id,
            judgment_id,
            reason,
        )
        update_candidate_judge_state(
            state,
            candidate_id=candidate_id,
            judge_status="unavailable",
            failure_reason=reason,
        )
        cand_rec = state["candidates"][candidate_id]
        cand_rec["judgeDiagnostics"] = dict(diagnostics)
        cand_rec["eligible"] = False
        cand_rec["totalScore"] = None
        cand_rec["tieScores"] = {}
        cand_rec["completedAt"] = _utc_now_iso()
        return

    register_judgment(
        state,
        {
            "judgmentId": judgment_id,
            "candidateId": candidate_id,
            "judgment": judgment,
            "totalScore": total,
            "scores": scores,
            "eligible": judgment.get("eligible"),
            "completedAt": _utc_now_iso(),
        },
    )
    cand_rec["judgmentId"] = judgment_id
    cand_rec["eligible"] = bool(judgment.get("eligible"))
    cand_rec["totalScore"] = total
    cand_rec["tieScores"] = scores
    cand_rec["judgeDiagnostics"] = dict((state.get("judgeDiagnosticsByCandidate") or {}).get(candidate_id, {}))
    cand_rec["completedAt"] = _utc_now_iso()
    update_candidate_judge_state(
        state,
        candidate_id=candidate_id,
        judge_status="accepted",
        judgment_id=judgment_id,
        judgment_snapshot=judgment,
    )
    record_judge_valid(state, eligible=bool(judgment.get("eligible")))

    if cand_rec["eligible"]:
        record_creator_eligible(state)
        updated = update_best_candidate_if_stronger(
            state,
            prototype_id=prototype_id,
            candidate_id=candidate_id,
            total_score=total,
            tie_scores=scores,
            completed_at=cand_rec["completedAt"],
        )
        if updated:
            logger.info(
                "BUILDER2_PROTOTYPE_BEST_UPDATED prototypeId=%s candidateId=%s total=%s",
                prototype_id,
                candidate_id,
                total,
            )


def _eliminate_lowest_prototypes(state: Dict[str, Any], round_index: int) -> None:
    active = list(state["activePrototypeIds"])
    if len(active) <= 1:
        return
    elim_count = min(resolve_builder2_tournament_eliminations_per_round(), len(active) - 1)
    ranked: List[Tuple[str, Dict[str, Any]]] = []
    for pid in active:
        best_id = state["bestCandidateByPrototype"].get(pid)
        if best_id and state["candidates"].get(best_id, {}).get("eligible"):
            ranked.append((pid, _candidate_rank_record(state, best_id)))
        else:
            ranked.append((pid, {"candidateId": "", "totalScore": -1, "tieScores": {}, "completedAt": "", "eligible": False}))
    ranked.sort(key=lambda item: (
        item[1]["totalScore"],
        item[1]["tieScores"].get("silentVisualClarity", -1),
        item[1]["tieScores"].get("problemAdvantageIntegrity", -1),
        item[1]["tieScores"].get("runwayFeasibility", -1),
    ))
    to_eliminate = [pid for pid, _ in ranked[:elim_count]]
    for pid in to_eliminate:
        if pid in state["activePrototypeIds"]:
            state["activePrototypeIds"].remove(pid)
            state["eliminatedPrototypeIds"].append(pid)
            logger.info(
                "BUILDER2_PROTOTYPE_ELIMINATED prototypeId=%s roundIndex=%s",
                pid,
                round_index,
            )
    rnd = _round_record(state, round_index)
    if to_eliminate:
        rnd["eliminatedPrototypeId"] = to_eliminate[0]
        rnd["eliminationReason"] = "lowest_best_candidate_rank"


def run_builder2_tournament(
    *,
    job_id: str,
    product_name: str,
    product_description: str,
    content_language: str,
    llm_client: Optional[Any] = None,
    rng_seed: Optional[str] = None,
) -> Dict[str, Any]:
    t_tournament0 = time.monotonic()
    from engine.builder2_normal_production_guard import NormalProductionGuard

    NormalProductionGuard.begin()
    try:
        return _run_builder2_tournament_body(
            job_id=job_id,
            product_name=product_name,
            product_description=product_description,
            content_language=content_language,
            llm_client=llm_client,
            rng_seed=rng_seed,
            t_tournament0=t_tournament0,
        )
    finally:
        NormalProductionGuard.end()


def _run_builder2_tournament_body(
    *,
    job_id: str,
    product_name: str,
    product_description: str,
    content_language: str,
    llm_client: Optional[Any],
    rng_seed: Optional[str],
    t_tournament0: float,
) -> Dict[str, Any]:
    language = content_language
    runway_model = resolve_builder2_runway_video_model()
    runway_mode = builder2_runway_generation_mode(runway_model)
    active_ids = resolve_builder2_active_prototype_ids()
    attempts_per = resolve_builder2_tournament_attempts_per_prototype_per_round()
    max_rounds = resolve_builder2_tournament_max_rounds()

    state = load_tournament_state(job_id)
    is_new_job = state is None
    if state:
        ensure_methodology_compatibility_decided(state, is_new_job=False)
        next_step = _next_step_name(state)
        logger.info(
            "BUILDER2_TOURNAMENT_RESUMED jobId=%s tournamentId=%s lastCompletedStep=%s nextStep=%s roundIndex=%s",
            job_id,
            state.get("tournamentId"),
            state.get("lastCompletedStep"),
            next_step,
            state.get("currentRound") or 1,
        )
    else:
        seed = rng_seed or f"{job_id}-{uuid.uuid4().hex}"
        state = new_tournament_state(
            job_id=job_id,
            language=language,
            active_prototype_ids=active_ids,
            random_seed=seed,
        )
        state["methodologyVersion"] = METHODOLOGY_VERSION
        state["methodologyCompatibilityMode"] = False
        save_tournament_state(job_id, state)
        logger.info(
            "BUILDER2_TOURNAMENT_START jobId=%s tournamentId=%s prototypes=%s maxRounds=%s",
            job_id,
            state["tournamentId"],
            len(active_ids),
            max_rounds,
        )

    ensure_metrics(state)
    compatibility_mode = bool(state.get("methodologyCompatibilityMode"))

    if not state.get("strategyFoundation"):
        state["status"] = "strategy_generating"
        state["lastCompletedStep"] = "strategy_generating"
        save_tournament_state(job_id, state)
        try:
            state["strategyFoundation"] = _generate_strategy(
                product_name=product_name,
                product_description=product_description,
                language=language,
                llm_client=llm_client,
                state=state,
            )
        except Builder2TournamentError as exc:
            state["status"] = "failed"
            state["error"] = str(exc.args[0] if exc.args else "builder2_strategy_validation_failed")
            record_process_failure_tag(state, state["error"])
            save_tournament_state(job_id, state)
            raise
        state["status"] = "strategy_complete"
        state["lastCompletedStep"] = "strategy_complete"
        state["methodologyVersion"] = METHODOLOGY_VERSION
        state["methodologyCompatibilityMode"] = False
        save_tournament_state(job_id, state)
    elif not state["strategyFoundation"].get("strategyFoundationId"):
        state["strategyFoundation"] = assign_strategy_foundation_identity(
            state["strategyFoundation"],
            tournament_id=state.get("tournamentId") or "",
        )
        save_tournament_state(job_id, state)

    from engine.builder2_creator_preflight import creator_preflight_only_enabled, run_one_isolated_creator_preflight

    if creator_preflight_only_enabled():
        return run_one_isolated_creator_preflight(
            product_name=product_name,
            product_description=product_description,
            content_language=language,
            llm_client=llm_client,
        )

    round_index = max(int(state.get("currentRound") or 0), 1)
    state["currentRound"] = round_index

    while True:
        completed_rounds = _count_completed_rounds(state)
        if max_rounds > 0 and completed_rounds >= max_rounds:
            break
        if max_rounds == 0 and len(state.get("activePrototypeIds") or []) <= 1 and completed_rounds > 0:
            break

        if _round_is_complete(state, round_index):
            round_index += 1
            state["currentRound"] = round_index
            save_tournament_state(job_id, state)
            if max_rounds > 0 and _count_completed_rounds(state) >= max_rounds:
                break
            if max_rounds == 0 and len(state.get("activePrototypeIds") or []) <= 1:
                break
            continue

        deck = _prototypes_for_round(state, round_index)
        _ensure_round(state, round_index, deck)
        state["status"] = "round_generating"
        state["lastCompletedStep"] = f"round_{round_index}_generating"
        logger.info("BUILDER2_ROUND_START roundIndex=%s prototypes=%s", round_index, deck)
        save_tournament_state(job_id, state)

        for prototype_id in deck:
            if is_creator_contract_circuit_breaker_tripped(state):
                logger.error("BUILDER2_CREATOR_CONTRACT_CIRCUIT_BREAKER stoppingRemainingCreators=true")
                break
            if is_judge_contract_circuit_breaker_tripped(state):
                logger.error("BUILDER2_JUDGE_CONTRACT_CIRCUIT_BREAKER stoppingRemainingJudges=true")
                break
            for attempt in range(1, attempts_per + 1):
                _run_creator_and_judge_for_assignment(
                    state=state,
                    product_name=product_name,
                    product_description=product_description,
                    language=language,
                    prototype_id=prototype_id,
                    round_index=round_index,
                    attempt_number=attempt,
                    runway_mode=runway_mode,
                    llm_client=llm_client,
                    compatibility_mode=compatibility_mode,
                )
                save_tournament_state(job_id, state)

        rnd = _round_record(state, round_index)
        rnd["attemptsCompleted"] = attempts_per * len(deck)
        rnd["judgmentsCompleted"] = len(
            [
                c
                for c in state["candidates"].values()
                if c.get("roundIndex") == round_index and c.get("judgmentId")
            ]
        )
        rnd["roundComplete"] = True
        state["status"] = "round_complete"
        state["lastCompletedStep"] = f"round_{round_index}_complete"
        save_tournament_state(job_id, state)

        completed_rounds = _count_completed_rounds(state)
        if _should_eliminate_after_round(
            max_rounds=max_rounds,
            completed_rounds=completed_rounds,
            state=state,
        ):
            _eliminate_lowest_prototypes(state, round_index)
            state["status"] = "eliminating"
            state["lastCompletedStep"] = f"round_{round_index}_eliminated"
            save_tournament_state(job_id, state)
            round_index += 1
            state["currentRound"] = round_index
            save_tournament_state(job_id, state)
            continue

        if max_rounds > 0 and completed_rounds >= max_rounds:
            state["completionReason"] = "max_rounds_reached"
            logger.info(
                "BUILDER2_ONE_ROUND_COMPLETE jobId=%s tournamentId=%s roundIndex=%s prototypes=%s",
                job_id,
                state.get("tournamentId"),
                round_index,
                len(state.get("initialActivePrototypeIds") or []),
            )
        break

    state["status"] = "tournament_complete"
    state["lastCompletedStep"] = "tournament_complete"
    if is_creator_contract_circuit_breaker_tripped(state):
        exc = Builder2TournamentError(
            f"{SYSTEMIC_FAILURE_CODE}:{(state.get('creatorContractCircuitBreaker') or {}).get('trippedReason', 'contract_failure')}"
        )
        record_process_contract_failure(state, exc)
        save_tournament_state(job_id, state)
        raise exc
    if is_judge_contract_circuit_breaker_tripped(state):
        exc = Builder2TournamentError(
            f"{JUDGE_SYSTEMIC_FAILURE_CODE}:{(state.get('judgeContractCircuitBreaker') or {}).get('trippedReason', 'contract_failure')}"
        )
        record_judge_process_contract_failure(state, exc)
        save_tournament_state(job_id, state)
        raise exc
    from engine.builder2_tournament_completion_gate import (
        assert_tournament_ready_for_winner_selection,
        invalidate_provisional_winner_if_incomplete,
        mark_authoritative_winner_selection,
    )

    invalidate_provisional_winner_if_incomplete(state)
    try:
        assert_tournament_ready_for_winner_selection(state)
    except Builder2TournamentError as exc:
        state["status"] = "tournament_incomplete"
        state["lastCompletedStep"] = "awaiting_creator_or_judge_completion"
        state["tournamentBlockingReason"] = str(exc.args[0] if exc.args else exc)
        state["canResume"] = True
        save_tournament_state(job_id, state)
        raise

    winner_id = state.get("winnerCandidateId")
    if not winner_id:
        winner_id = select_global_winner(state)
        mark_authoritative_winner_selection(state, winner_id=winner_id)
        winner_rec = state["candidates"][winner_id]
        judgment_rec = state["judgments"].get(winner_rec.get("judgmentId") or "")
        from engine.builder2_complete_ad_contract import copy_winner_advertising_closure_from_candidate

        if not compatibility_mode:
            copy_winner_advertising_closure_from_candidate(
                state,
                candidate_id=winner_id,
                winning_candidate=winner_rec.get("creatorOutput") or {},
                winning_judgment=(judgment_rec or {}).get("judgment"),
            )
        logger.info(
            "BUILDER2_TOURNAMENT_WINNER_SELECTED jobId=%s candidateId=%s",
            job_id,
            winner_id,
        )
    save_tournament_state(job_id, state)

    if not state.get("winnerDevelopmentPlan"):
        state["status"] = "winner_developing"
        state["lastCompletedStep"] = "winner_developing"
        save_tournament_state(job_id, state)
        winner_rec = state["candidates"][winner_id]
        judgment_rec = state["judgments"].get(winner_rec.get("judgmentId") or "")
        winning_judgment = (judgment_rec or {}).get("judgment") or {}
        try:
            winner_plan = develop_builder2_winning_candidate(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winner_rec["creatorOutput"],
                winning_judgment=winning_judgment,
                prototype_id=winner_rec["prototypeId"],
                runway_mode=runway_mode,
                llm_client=llm_client,
                compatibility_mode=compatibility_mode,
                state=state,
                candidate_id=winner_id,
            )
            persist_accepted_winner_development_for_media(
                state,
                candidate_id=winner_id,
                prototype_id=str(winner_rec.get("prototypeId") or ""),
                winner_plan=winner_plan,
                winning_candidate=winner_rec.get("creatorOutput") or {},
                winning_judgment=winning_judgment,
                preservation_snapshot=winner_plan.get("winningCandidatePreservationSnapshot"),
                compatibility_mode=compatibility_mode,
                source=WINNER_DEVELOPMENT_SOURCE_NORMAL,
                job_id=job_id,
                tournament_id=str(state.get("tournamentId") or ""),
                save=False,
            )
        except Builder2TournamentError as exc:
            record_process_failure_tag(state, str(exc.args[0] if exc.args else "builder2_winner_development_failed"))
            logger.error("BUILDER2_WINNER_DEVELOPMENT_FAILED candidateId=%s", winner_id)
            state["status"] = "failed"
            save_tournament_state(job_id, state)
            raise
        state["status"] = "winner_plan_complete"
        state["lastCompletedStep"] = "winner_plan_complete"
        state["mediaContinuationRequired"] = True
        from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION

        state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
        save_tournament_state(job_id, state)
        logger.info("BUILDER2_WINNER_DEVELOPMENT_OK candidateId=%s", winner_id)
    elif state.get("winnerDevelopmentPlan"):
        logger.info(
            "BUILDER2_PERSISTED_WINNER_RESUME jobId=%s tournamentId=%s winnerCandidateId=%s",
            job_id,
            state.get("tournamentId"),
            state.get("winnerCandidateId"),
        )

    finalize_tournament_metrics(state, elapsed_ms=(time.monotonic() - t_tournament0) * 1000.0)
    save_tournament_state(job_id, state)

    normalized = normalize_winner_plan_for_runway(
        state["winnerDevelopmentPlan"],
        product_name=product_name,
        product_description=product_description,
        content_language=language,
    )
    normalized["tournamentId"] = state.get("tournamentId")
    normalized["winnerCandidateId"] = winner_id
    normalized["completionReason"] = state.get("completionReason")
    return normalized


def _round_is_complete(state: Dict[str, Any], round_index: int) -> bool:
    try:
        rnd = _round_record(state, round_index)
    except Builder2TournamentError:
        return False
    return bool(rnd.get("roundComplete"))


def _current_round_deck(state: Dict[str, Any], round_index: int, source_ids: List[str]) -> List[str]:
    for rnd in state["rounds"]:
        if rnd.get("roundIndex") == round_index and rnd.get("shuffledPrototypeOrder"):
            return list(rnd["shuffledPrototypeOrder"])
    seed = f"{state['randomSeed']}-round-{round_index}"
    return _shuffle_prototypes(source_ids, seed)
