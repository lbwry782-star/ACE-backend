"""
Builder2 Creator preflight — isolated one-shot Strategy + Creator contract verification.

Run: python -m engine.builder2_creator_preflight

Does not use the video queue, recovery registry, Judge, Winner Development, Runway, or FFmpeg.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from engine.builder2_creator import generate_creator_candidate
from engine.builder2_prototypes import require_prototype
from engine.builder2_runway_config import builder2_runway_generation_mode, resolve_builder2_runway_video_model
from engine.builder2_strategy import generate_strategy_foundation
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import ensure_metrics

logger = logging.getLogger(__name__)

PREFLIGHT_ISOLATION_ERROR = "builder2_creator_preflight_isolation_failed"


class PreflightIsolationGuard:
    """Process-local guard ensuring preflight never touches worker/recovery/media paths."""

    recovery_scan_performed: bool = False
    ordinary_job_dequeued: bool = False
    judge_enabled: bool = False
    winner_development_enabled: bool = False
    runway_enabled: bool = False
    ffmpeg_enabled: bool = False
    active: bool = False

    @classmethod
    def begin(cls) -> None:
        cls.recovery_scan_performed = False
        cls.ordinary_job_dequeued = False
        cls.judge_enabled = False
        cls.winner_development_enabled = False
        cls.runway_enabled = False
        cls.ffmpeg_enabled = False
        cls.active = True

    @classmethod
    def end(cls) -> None:
        cls.active = False

    @classmethod
    def assert_safe_before_paid_call(cls) -> None:
        if not cls.active:
            return
        checks = {
            "recoveryScanPerformed": cls.recovery_scan_performed,
            "ordinaryJobDequeued": cls.ordinary_job_dequeued,
            "judgeEnabled": cls.judge_enabled,
            "winnerDevelopmentEnabled": cls.winner_development_enabled,
            "runwayEnabled": cls.runway_enabled,
            "ffmpegEnabled": cls.ffmpeg_enabled,
        }
        violations = [name for name, happened in checks.items() if happened]
        if violations:
            raise Builder2TournamentError(f"{PREFLIGHT_ISOLATION_ERROR}:{','.join(violations)}")


def creator_preflight_only_enabled() -> bool:
    raw = (os.environ.get("BUILDER2_CREATOR_PREFLIGHT_ONLY") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _preflight_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _new_preflight_id() -> str:
    return f"preflight-{uuid.uuid4().hex[:12]}"


def _empty_metrics() -> Dict[str, Any]:
    return {
        "strategyCalls": 0,
        "creatorCalls": 0,
        "creatorRepairCalls": 0,
        "creatorRetryCalls": 0,
        "judgeCalls": 0,
        "winnerDevelopmentCalls": 0,
    }


def run_one_isolated_creator_preflight(
    *,
    product_name: str,
    product_description: str,
    content_language: str,
    prototype_id: Optional[str] = None,
    llm_client: Optional[Any] = None,
    preflight_id: Optional[str] = None,
) -> Dict[str, Any]:
    PreflightIsolationGuard.begin()
    preflight_id = preflight_id or _new_preflight_id()
    language = content_language or _preflight_env("BUILDER2_PREFLIGHT_LANGUAGE", "he")
    active_ids = resolve_builder2_active_prototype_ids()
    assigned = (prototype_id or _preflight_env("BUILDER2_PREFLIGHT_PROTOTYPE_ID", "think_small")).strip()
    if assigned not in active_ids:
        assigned = active_ids[0]

    state: Dict[str, Any] = {
        "jobId": preflight_id,
        "tournamentId": f"preflight-{preflight_id}",
        "preflightMode": True,
        "preflightLocalOnly": True,
        "metrics": _empty_metrics(),
    }
    ensure_metrics(state)

    report: Dict[str, Any] = {
        "preflightId": preflight_id,
        "prototypeId": assigned,
        "strategyAccepted": False,
        "creatorAccepted": False,
        "creatorNormalCalls": 0,
        "creatorRepairCalls": 0,
        "creatorRetryCalls": 0,
        "validationFailurePaths": [],
        "serverDerivedFieldPaths": [],
        "judgeCalls": 0,
        "winnerCalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
    }

    runway_mode = builder2_runway_generation_mode(resolve_builder2_runway_video_model())

    try:
        PreflightIsolationGuard.assert_safe_before_paid_call()
        strategy = generate_strategy_foundation(
            product_name=product_name,
            product_description=product_description,
            language=language,
            llm_client=llm_client,
            state=state,
        )
        report["strategyAccepted"] = True

        PreflightIsolationGuard.assert_safe_before_paid_call()
        candidate_id, _candidate = generate_creator_candidate(
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
            candidate_id=f"cand-preflight-{assigned}-1",
        )
        report["creatorAccepted"] = True
        report["candidateId"] = candidate_id
    except Builder2TournamentError as exc:
        reason = str(exc.args[0] if exc.args else "builder2_creator_preflight_failed")
        if ":" in reason:
            report["validationFailurePaths"] = [reason.split(":", 1)[-1]]
        report["failureReason"] = reason
    finally:
        metrics = state.get("metrics") or {}
        report["creatorNormalCalls"] = int(metrics.get("creatorCalls") or 0)
        report["creatorRepairCalls"] = int(metrics.get("creatorRepairCalls") or 0)
        report["creatorRetryCalls"] = int(metrics.get("creatorRetryCalls") or 0)
        report["judgeCalls"] = int(metrics.get("judgeCalls") or 0)
        report["winnerCalls"] = int(metrics.get("winnerDevelopmentCalls") or 0)
        report["runwayCalls"] = 0
        report["ffmpegCalls"] = 0
        diagnostics = (state.get("creatorDiagnosticsByCandidate") or {}).get(report.get("candidateId") or "", {})
        resolved = diagnostics.get("normalizationResolvedFields") or []
        if isinstance(resolved, list):
            report["serverDerivedFieldPaths"] = [str(p) for p in resolved[:20]]
        PreflightIsolationGuard.end()

    report["ok"] = bool(report.get("strategyAccepted") and report.get("creatorAccepted"))
    return report


def print_preflight_report(report: Dict[str, Any]) -> None:
    safe = {
        key: report.get(key)
        for key in (
            "preflightId",
            "prototypeId",
            "strategyAccepted",
            "creatorAccepted",
            "creatorNormalCalls",
            "creatorRepairCalls",
            "creatorRetryCalls",
            "validationFailurePaths",
            "serverDerivedFieldPaths",
            "judgeCalls",
            "winnerCalls",
            "runwayCalls",
            "ffmpegCalls",
            "failureReason",
            "ok",
        )
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    product_name = _preflight_env("BUILDER2_PREFLIGHT_PRODUCT_NAME", "Preflight Product")
    product_description = _preflight_env(
        "BUILDER2_PREFLIGHT_PRODUCT_DESCRIPTION",
        "A product used to verify the Builder2 Creator contract before a paid tournament.",
    )
    language = _preflight_env("BUILDER2_PREFLIGHT_LANGUAGE", "he")
    prototype_id = _preflight_env("BUILDER2_PREFLIGHT_PROTOTYPE_ID", "think_small") or None

    logger.info(
        "BUILDER2_CREATOR_PREFLIGHT_START prototypeId=%s language=%s",
        prototype_id,
        language,
    )
    report = run_one_isolated_creator_preflight(
        product_name=product_name,
        product_description=product_description,
        content_language=language,
        prototype_id=prototype_id,
    )
    print_preflight_report(report)
    logger.info(
        "BUILDER2_CREATOR_PREFLIGHT_DONE preflightId=%s ok=%s strategyAccepted=%s creatorAccepted=%s",
        report.get("preflightId"),
        report.get("ok"),
        report.get("strategyAccepted"),
        report.get("creatorAccepted"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
