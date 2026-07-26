"""
Isolation guard for Builder2 Judge preflight.
"""
from __future__ import annotations

from engine.builder2_tournament_contracts import Builder2TournamentError

PREFLIGHT_ISOLATION_ERROR = "builder2_judge_preflight_isolation_failed"


class JudgePreflightIsolationGuard:
    recovery_scan_performed: bool = False
    ordinary_job_dequeued: bool = False
    strategy_enabled: bool = False
    creator_enabled: bool = False
    winner_development_enabled: bool = False
    runway_enabled: bool = False
    ffmpeg_enabled: bool = False
    active: bool = False

    @classmethod
    def begin(cls) -> None:
        cls.recovery_scan_performed = False
        cls.ordinary_job_dequeued = False
        cls.strategy_enabled = False
        cls.creator_enabled = False
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
            "strategyEnabled": cls.strategy_enabled,
            "creatorEnabled": cls.creator_enabled,
            "winnerDevelopmentEnabled": cls.winner_development_enabled,
            "runwayEnabled": cls.runway_enabled,
            "ffmpegEnabled": cls.ffmpeg_enabled,
        }
        violations = [name for name, happened in checks.items() if happened]
        if violations:
            raise Builder2TournamentError(f"{PREFLIGHT_ISOLATION_ERROR}:{','.join(violations)}")
