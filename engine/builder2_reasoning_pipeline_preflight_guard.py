"""
Isolation guard for Builder2 Strategy→Creator→Judge pipeline preflight.
"""
from __future__ import annotations

from engine.builder2_tournament_contracts import Builder2TournamentError

PREFLIGHT_ISOLATION_ERROR = "builder2_reasoning_pipeline_preflight_isolation_failed"


class PipelinePreflightIsolationGuard:
    ordinary_queue_enabled: bool = False
    recovery_scan_enabled: bool = False
    tournament_loop_enabled: bool = False
    winner_development_enabled: bool = False
    start_image_enabled: bool = False
    runway_enabled: bool = False
    ffmpeg_enabled: bool = False
    active: bool = False

    @classmethod
    def begin(cls) -> None:
        cls.ordinary_queue_enabled = False
        cls.recovery_scan_enabled = False
        cls.tournament_loop_enabled = False
        cls.winner_development_enabled = False
        cls.start_image_enabled = False
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
            "ordinaryQueueEnabled": cls.ordinary_queue_enabled,
            "recoveryScanEnabled": cls.recovery_scan_enabled,
            "tournamentLoopEnabled": cls.tournament_loop_enabled,
            "winnerDevelopmentEnabled": cls.winner_development_enabled,
            "startImageEnabled": cls.start_image_enabled,
            "runwayEnabled": cls.runway_enabled,
            "ffmpegEnabled": cls.ffmpeg_enabled,
        }
        violations = [name for name, enabled in checks.items() if enabled]
        if violations:
            raise Builder2TournamentError(f"{PREFLIGHT_ISOLATION_ERROR}:{','.join(violations)}")
