"""
Builder2 Strategy→Creator→Judge pipeline preflight — isolated three-call reasoning verification.

Run: python -m engine.builder2_reasoning_pipeline_preflight

Environment:
  BUILDER2_PIPELINE_PREFLIGHT_PRODUCT_NAME
  BUILDER2_PIPELINE_PREFLIGHT_PRODUCT_DESCRIPTION
  BUILDER2_PIPELINE_PREFLIGHT_LANGUAGE (default he)
  BUILDER2_PIPELINE_PREFLIGHT_PROTOTYPE_ID (default think_small)

Does not use Winner Development, Runway, FFmpeg, recovery, queue, or tournament Manager.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from engine.builder2_accepted_creator_store import build_accepted_creator_snapshot
from engine.builder2_creator import generate_creator_candidate
from engine.builder2_judge import judge_candidate
from engine.builder2_prototypes import require_prototype
from engine.builder2_reasoning_pipeline_preflight_guard import (
    PREFLIGHT_ISOLATION_ERROR,
    PipelinePreflightIsolationGuard,
)
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_strategy import generate_strategy_foundation
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import ensure_metrics

logger = logging.getLogger(__name__)


def _preflight_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _new_preflight_id() -> str:
    return f"pipeline-preflight-{uuid.uuid4().hex[:12]}"


def _empty_metrics() -> Dict[str, Any]:
    return {
        "strategyCalls": 0,
        "creatorCalls": 0,
        "creatorRepairCalls": 0,
        "creatorRetryCalls": 0,
        "judgeCalls": 0,
        "judgeRepairCalls": 0,
        "judgeRetryCalls": 0,
        "winnerDevelopmentCalls": 0,
    }


def _assert_single_call(metrics: Dict[str, Any], role: str, *, after_repair_key: str = "") -> None:
    if role == "builder2_strategy" and int(metrics.get("strategyCalls") or 0) > 1:
        raise Builder2TournamentError("builder2_reasoning_pipeline_preflight_cost_breaker:strategy")
    if role == "builder2_creator":
        normal = int(metrics.get("creatorCalls") or 0)
        repair = int(metrics.get("creatorRepairCalls") or 0)
        retry = int(metrics.get("creatorRetryCalls") or 0)
        if normal > 1 or repair > 1 or retry > 1:
            raise Builder2TournamentError("builder2_reasoning_pipeline_preflight_cost_breaker:creator")
    if role == "builder2_judge":
        normal = int(metrics.get("judgeCalls") or 0)
        repair = int(metrics.get("judgeRepairCalls") or 0)
        retry = int(metrics.get("judgeRetryCalls") or 0)
        if normal > 1 or repair > 1 or retry > 1:
            raise Builder2TournamentError("builder2_reasoning_pipeline_preflight_cost_breaker:judge")


def run_one_isolated_reasoning_pipeline_preflight(
    *,
    product_name: str,
    product_description: str,
    content_language: str,
    prototype_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    preflight_id: Optional[str] = None,
) -> Dict[str, Any]:
    PipelinePreflightIsolationGuard.begin()
    preflight_id = preflight_id or _new_preflight_id()
    language = content_language or _preflight_env("BUILDER2_PIPELINE_PREFLIGHT_LANGUAGE", "he")
    active_ids = resolve_builder2_active_prototype_ids()
    assigned = (prototype_id or _preflight_env("BUILDER2_PIPELINE_PREFLIGHT_PROTOTYPE_ID", "think_small")).strip()
    if assigned not in active_ids:
        assigned = active_ids[0]

    state: Dict[str, Any] = {
        "jobId": preflight_id,
        "tournamentId": f"pipeline-preflight-{preflight_id}",
        "preflightMode": True,
        "preflightLocalOnly": True,
        "acceptedCreatorCandidates": {},
        "metrics": _empty_metrics(),
    }
    ensure_metrics(state)

    report: Dict[str, Any] = {
        "preflightId": preflight_id,
        "prototypeId": assigned,
        "strategyAccepted": False,
        "creatorAccepted": False,
        "creatorPersisted": False,
        "judgeAccepted": False,
        "judgeEligible": None,
        "strategyCalls": 0,
        "creatorNormalCalls": 0,
        "creatorRepairCalls": 0,
        "creatorRetryCalls": 0,
        "judgeNormalCalls": 0,
        "judgeRepairCalls": 0,
        "judgeRetryCalls": 0,
        "winnerCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "validationFailurePaths": [],
        "ok": False,
    }

    runway_mode = builder2_runway_generation_mode(resolve_builder2_runway_video_model())
    candidate_id = f"cand-pipeline-{assigned}-1"
    require_prototype(assigned)

    try:
        PipelinePreflightIsolationGuard.assert_safe_before_paid_call()
        strategy = generate_strategy_foundation(
            product_name=product_name,
            product_description=product_description,
            language=language,
            llm_client=llm_client,
            state=state,
        )
        _assert_single_call(state.get("metrics") or {}, "builder2_strategy")
        report["strategyAccepted"] = True

        PipelinePreflightIsolationGuard.assert_safe_before_paid_call()
        candidate_id, candidate = generate_creator_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype_id=assigned,
            round_index=1,
            attempt_number=1,
            runway_mode=runway_mode,
            llm_client=llm_client,
            state=state,
            candidate_id=candidate_id,
        )
        _assert_single_call(state.get("metrics") or {}, "builder2_creator")
        report["creatorAccepted"] = True

        snapshot = build_accepted_creator_snapshot(
            candidate_id=candidate_id,
            prototype_id=assigned,
            round_index=1,
            attempt_number=1,
            creator_output=candidate,
            strategy_foundation=strategy,
        )
        state.setdefault("acceptedCreatorCandidates", {})[candidate_id] = snapshot
        report["creatorPersisted"] = True

        PipelinePreflightIsolationGuard.assert_safe_before_paid_call()
        _, judgment, _, _ = judge_candidate(
            product_name=product_name,
            product_description=product_description,
            language=language,
            strategy_foundation=strategy,
            prototype_id=assigned,
            candidate_id=candidate_id,
            candidate=candidate,
            llm_client=llm_client,
            state=state,
            judgment_id=f"judge-pipeline-{candidate_id}",
        )
        _assert_single_call(state.get("metrics") or {}, "builder2_judge")
        report["judgeAccepted"] = True
        report["judgeEligible"] = bool(judgment.get("eligible"))
        report["ok"] = True
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_reasoning_pipeline_preflight_failed")
        if ":" in reason:
            report["validationFailurePaths"] = [reason.split(":", 1)[-1]]
        report["failureReason"] = reason
    finally:
        metrics = state.get("metrics") or {}
        report["strategyCalls"] = int(metrics.get("strategyCalls") or 0)
        report["creatorNormalCalls"] = int(metrics.get("creatorCalls") or 0)
        report["creatorRepairCalls"] = int(metrics.get("creatorRepairCalls") or 0)
        report["creatorRetryCalls"] = int(metrics.get("creatorRetryCalls") or 0)
        report["judgeNormalCalls"] = int(metrics.get("judgeCalls") or 0)
        report["judgeRepairCalls"] = int(metrics.get("judgeRepairCalls") or 0)
        report["judgeRetryCalls"] = int(metrics.get("judgeRetryCalls") or 0)
        report["winnerCalls"] = int(metrics.get("winnerDevelopmentCalls") or 0)
        report["runwayCalls"] = 0
        report["ffmpegCalls"] = 0
        PipelinePreflightIsolationGuard.end()

    return report


def print_preflight_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "preflightId",
            "prototypeId",
            "strategyAccepted",
            "creatorAccepted",
            "creatorPersisted",
            "judgeAccepted",
            "judgeEligible",
            "strategyCalls",
            "creatorNormalCalls",
            "creatorRepairCalls",
            "creatorRetryCalls",
            "judgeNormalCalls",
            "judgeRepairCalls",
            "judgeRetryCalls",
            "winnerCalls",
            "runwayCalls",
            "ffmpegCalls",
            "validationFailurePaths",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    product_name = _preflight_env("BUILDER2_PIPELINE_PREFLIGHT_PRODUCT_NAME", "Preflight Product")
    product_description = _preflight_env(
        "BUILDER2_PIPELINE_PREFLIGHT_PRODUCT_DESCRIPTION",
        "A product used to verify the Builder2 Strategy→Creator→Judge pipeline before a paid tournament.",
    )
    language = _preflight_env("BUILDER2_PIPELINE_PREFLIGHT_LANGUAGE", "he")
    prototype_id = _preflight_env("BUILDER2_PIPELINE_PREFLIGHT_PROTOTYPE_ID", "think_small") or None

    logger.info(
        "BUILDER2_REASONING_PIPELINE_PREFLIGHT_START prototypeId=%s language=%s",
        prototype_id,
        language,
    )
    report = run_one_isolated_reasoning_pipeline_preflight(
        product_name=product_name,
        product_description=product_description,
        content_language=language,
        prototype_id=prototype_id,
    )
    print_preflight_report(report)
    logger.info(
        "BUILDER2_REASONING_PIPELINE_PREFLIGHT_DONE preflightId=%s ok=%s strategyAccepted=%s creatorAccepted=%s judgeAccepted=%s",
        report.get("preflightId"),
        report.get("ok"),
        report.get("strategyAccepted"),
        report.get("creatorAccepted"),
        report.get("judgeAccepted"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
