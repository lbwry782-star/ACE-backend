"""
Builder2 tournament manager — deterministic orchestration (never a model role).
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_creator import generate_creator_candidate
from engine.builder2_judge import judge_candidate
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
from engine.builder2_tournament_metrics import MetricsTimer, ensure_metrics, finalize_tournament_metrics, record_creator_eligible, record_model_call
from engine.builder2_tournament_store import (
    load_tournament_state,
    mutate_tournament_state,
    new_tournament_state,
    register_candidate,
    register_judgment,
    save_tournament_state,
    update_best_candidate_if_stronger,
)
from engine.builder2_winner_development import (
    develop_builder2_winning_candidate,
    normalize_winner_plan_for_runway,
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


def select_global_winner(state: Dict[str, Any]) -> str:
    eligible_ids = [
        cid
        for cid, cand in state["candidates"].items()
        if cand.get("eligible") and cand.get("validationStatus") == "accepted"
    ]
    if not eligible_ids:
        raise Builder2TournamentError("builder2_tournament_no_valid_candidate")
    best_id = eligible_ids[0]
    best_record = _candidate_rank_record(state, best_id)
    for cid in eligible_ids[1:]:
        record = _candidate_rank_record(state, cid)
        if compare_candidate_rankings(record, best_record) > 0:
            best_id = cid
            best_record = record
    return best_id


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
) -> None:
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

    pending = [
        c
        for c in state["candidates"].values()
        if c.get("prototypeId") == prototype_id
        and c.get("roundIndex") == round_index
        and c.get("attemptNumber") == attempt_number
        and c.get("validationStatus") == "accepted"
        and not c.get("judgmentId")
    ]
    if pending:
        candidate_id = pending[0]["candidateId"]
        candidate = pending[0]["creatorOutput"]
    else:
        logger.info(
            "BUILDER2_PROTOTYPE_ASSIGNED prototypeId=%s roundIndex=%s attempt=%s",
            prototype_id,
            round_index,
            attempt_number,
        )
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
            )
        except Builder2TournamentError as exc:
            reason = str(exc.args[0] if exc.args else "builder2_creator_invalid_candidate")
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
                "status": "accepted",
                "judgmentId": None,
                "eligible": False,
                "totalScore": None,
                "tieScores": {},
                "failureReason": None,
                "completedAt": _utc_now_iso(),
            },
        )

    cand_rec = state["candidates"][candidate_id]
    if cand_rec.get("judgmentId"):
        return

    try:
        timer = MetricsTimer()
        judgment_id, judgment, total, scores = judge_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype_id=prototype_id,
            candidate_id=candidate_id,
            candidate=candidate,
            llm_client=llm_client,
        )
        record_model_call(state, role="builder2_judge", elapsed_ms=timer.elapsed_ms())
    except Builder2TournamentError as exc:
        logger.info(
            "BUILDER2_JUDGE_REJECTED candidateId=%s reason=%s",
            candidate_id,
            exc.args[0],
        )
        cand_rec["validationStatus"] = "rejected"
        raise

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
    cand_rec["completedAt"] = _utc_now_iso()

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
    language = content_language
    runway_model = resolve_builder2_runway_video_model()
    runway_mode = builder2_runway_generation_mode(runway_model)
    active_ids = resolve_builder2_active_prototype_ids()
    attempts_per = resolve_builder2_tournament_attempts_per_prototype_per_round()
    max_rounds = resolve_builder2_tournament_max_rounds()
    t_tournament0 = time.monotonic()

    state = load_tournament_state(job_id)
    if state:
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
        save_tournament_state(job_id, state)
        logger.info(
            "BUILDER2_TOURNAMENT_START jobId=%s tournamentId=%s prototypes=%s maxRounds=%s",
            job_id,
            state["tournamentId"],
            len(active_ids),
            max_rounds,
        )

    ensure_metrics(state)

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
            save_tournament_state(job_id, state)
            raise
        state["status"] = "strategy_complete"
        state["lastCompletedStep"] = "strategy_complete"
        save_tournament_state(job_id, state)

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
    winner_id = state.get("winnerCandidateId")
    if not winner_id:
        winner_id = select_global_winner(state)
        state["winnerCandidateId"] = winner_id
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
        try:
            timer = MetricsTimer()
            winner_plan = develop_builder2_winning_candidate(
                product_name=product_name,
                product_description=product_description,
                language=language,
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winner_rec["creatorOutput"],
                prototype_id=winner_rec["prototypeId"],
                runway_mode=runway_mode,
                llm_client=llm_client,
            )
            record_model_call(state, role="builder2_winner", elapsed_ms=timer.elapsed_ms())
        except Builder2TournamentError:
            logger.error("BUILDER2_WINNER_DEVELOPMENT_FAILED candidateId=%s", winner_id)
            state["status"] = "failed"
            save_tournament_state(job_id, state)
            raise
        state["winnerDevelopmentPlan"] = winner_plan
        state["status"] = "winner_plan_complete"
        state["lastCompletedStep"] = "winner_plan_complete"
        save_tournament_state(job_id, state)
        logger.info("BUILDER2_WINNER_DEVELOPMENT_OK candidateId=%s", winner_id)

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
