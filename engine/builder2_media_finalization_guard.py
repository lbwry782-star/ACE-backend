"""
Isolation guard for Builder2 media finalization recovery.
"""
from __future__ import annotations

from engine.builder2_media_reasoning_guard import (
    MEDIA_RESUME_REASONING_BLOCKED,
    MediaResumeReasoningCounters,
    assert_media_resume_reasoning_call_allowed,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

MEDIA_FINALIZATION_ISOLATION_ERROR = "builder2_media_finalization_isolation_failed"


class MediaFinalizationIsolationGuard:
    active: bool = False
    closure_enabled: bool = False
    publication_enabled: bool = False
    reasoning_counters: MediaResumeReasoningCounters = MediaResumeReasoningCounters()
    closure_ffmpeg_calls: int = 0
    publication_calls: int = 0

    @classmethod
    def begin(cls) -> None:
        cls.active = True
        cls.closure_enabled = False
        cls.publication_enabled = False
        cls.reasoning_counters = MediaResumeReasoningCounters()
        cls.closure_ffmpeg_calls = 0
        cls.publication_calls = 0

    @classmethod
    def end(cls) -> None:
        cls.active = False

    @classmethod
    def enable_closure(cls) -> None:
        cls.closure_enabled = True

    @classmethod
    def enable_publication(cls) -> None:
        cls.publication_enabled = True

    @classmethod
    def assert_reasoning_isolated(cls) -> None:
        if not cls.active:
            return
        if cls.reasoning_counters.totalReasoningCalls > 0:
            raise Builder2TournamentError(MEDIA_FINALIZATION_ISOLATION_ERROR)

    @classmethod
    def assert_reasoning_call_allowed(cls, role: str) -> None:
        assert_media_resume_reasoning_call_allowed(role=role, active=cls.active)

    @classmethod
    def record_closure_ffmpeg(cls) -> None:
        if cls.active:
            cls.closure_ffmpeg_calls += 1

    @classmethod
    def record_publication(cls) -> None:
        if cls.active:
            cls.publication_calls += 1

    @classmethod
    def assert_safe_before_closure(cls) -> None:
        cls.assert_reasoning_isolated()
        if not cls.closure_enabled:
            raise Builder2TournamentError(f"{MEDIA_FINALIZATION_ISOLATION_ERROR}:closureDisabled")

    @classmethod
    def assert_safe_before_publication(cls) -> None:
        cls.assert_reasoning_isolated()
        if not cls.publication_enabled:
            raise Builder2TournamentError(f"{MEDIA_FINALIZATION_ISOLATION_ERROR}:publicationDisabled")

    @classmethod
    def reasoning_report(cls) -> dict[str, int]:
        report = cls.reasoning_counters.to_report_dict()
        report["closureFfmpegCalls"] = cls.closure_ffmpeg_calls
        report["publicationCalls"] = cls.publication_calls
        return report
