"""
Isolation guard for Builder2 media-only resume.
"""
from __future__ import annotations

from engine.builder2_tournament_contracts import Builder2TournamentError

MEDIA_RESUME_ISOLATION_ERROR = "builder2_media_resume_isolation_failed"


class MediaResumeIsolationGuard:
    ordinary_queue_enabled: bool = False
    recovery_scan_enabled: bool = False
    strategy_generation_enabled: bool = False
    creator_generation_enabled: bool = False
    judge_generation_enabled: bool = False
    winner_development_enabled: bool = False
    tournament_loop_enabled: bool = False
    start_image_enabled: bool = False
    runway_enabled: bool = False
    ffmpeg_enabled: bool = False
    zip_generation_enabled: bool = False
    active: bool = False

    @classmethod
    def begin(cls) -> None:
        cls.ordinary_queue_enabled = False
        cls.recovery_scan_enabled = False
        cls.strategy_generation_enabled = False
        cls.creator_generation_enabled = False
        cls.judge_generation_enabled = False
        cls.winner_development_enabled = False
        cls.tournament_loop_enabled = False
        cls.start_image_enabled = False
        cls.runway_enabled = False
        cls.ffmpeg_enabled = False
        cls.zip_generation_enabled = False
        cls.active = True

    @classmethod
    def end(cls) -> None:
        cls.active = False

    @classmethod
    def enable_start_image(cls) -> None:
        cls.start_image_enabled = True

    @classmethod
    def enable_runway(cls) -> None:
        cls.runway_enabled = True

    @classmethod
    def enable_ffmpeg(cls) -> None:
        cls.ffmpeg_enabled = True

    @classmethod
    def assert_reasoning_isolated(cls) -> None:
        if not cls.active:
            return
        reasoning_checks = {
            "strategyGenerationEnabled": cls.strategy_generation_enabled,
            "creatorGenerationEnabled": cls.creator_generation_enabled,
            "judgeGenerationEnabled": cls.judge_generation_enabled,
            "winnerDevelopmentEnabled": cls.winner_development_enabled,
            "tournamentLoopEnabled": cls.tournament_loop_enabled,
            "ordinaryQueueEnabled": cls.ordinary_queue_enabled,
            "recoveryScanEnabled": cls.recovery_scan_enabled,
        }
        violations = [name for name, enabled in reasoning_checks.items() if enabled]
        if violations:
            raise Builder2TournamentError(f"{MEDIA_RESUME_ISOLATION_ERROR}:{','.join(violations)}")

    @classmethod
    def assert_safe_before_start_image(cls) -> None:
        cls.assert_reasoning_isolated()
        if not cls.start_image_enabled:
            raise Builder2TournamentError(f"{MEDIA_RESUME_ISOLATION_ERROR}:startImageDisabled")

    @classmethod
    def assert_safe_before_runway(cls) -> None:
        cls.assert_reasoning_isolated()
        if not cls.runway_enabled:
            raise Builder2TournamentError(f"{MEDIA_RESUME_ISOLATION_ERROR}:runwayDisabled")

    @classmethod
    def assert_safe_before_ffmpeg(cls) -> None:
        cls.assert_reasoning_isolated()
        if not cls.ffmpeg_enabled:
            raise Builder2TournamentError(f"{MEDIA_RESUME_ISOLATION_ERROR}:ffmpegDisabled")
