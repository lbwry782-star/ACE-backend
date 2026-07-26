"""
Isolation guard for Builder2 advertising-closure resume.
"""
from __future__ import annotations

from typing import Dict

from engine.builder2_media_reasoning_guard import normalize_reasoning_role
from engine.builder2_tournament_contracts import Builder2TournamentError

ADVERTISING_CLOSURE_REASONING_BLOCKED = "builder2_advertising_closure_reasoning_call_blocked"


class AdvertisingClosureReasoningCounters:
    advertisingClosureCalls: int = 0
    advertisingClosureRepairCalls: int = 0
    advertisingClosureRetryCalls: int = 0
    otherReasoningCalls: int = 0

    def increment(self, role: str, *, call_type: str = "normal") -> None:
        bucket = normalize_reasoning_role(role)
        if bucket == "other" and role == "advertising_closure":
            bucket = "advertising_closure"
        if bucket == "advertising_closure":
            if call_type == "repair":
                self.advertisingClosureRepairCalls += 1
            elif call_type == "retry":
                self.advertisingClosureRetryCalls += 1
            else:
                self.advertisingClosureCalls += 1
        else:
            self.otherReasoningCalls += 1

    @property
    def totalReasoningCalls(self) -> int:
        return (
            self.advertisingClosureCalls
            + self.advertisingClosureRepairCalls
            + self.advertisingClosureRetryCalls
            + self.otherReasoningCalls
        )

    def to_report_dict(self) -> Dict[str, int]:
        return {
            "advertisingClosureCalls": self.advertisingClosureCalls,
            "advertisingClosureRepairCalls": self.advertisingClosureRepairCalls,
            "advertisingClosureRetryCalls": self.advertisingClosureRetryCalls,
            "otherReasoningCalls": self.otherReasoningCalls,
            "totalReasoningCalls": self.totalReasoningCalls,
        }


class AdvertisingClosureResumeGuard:
    active: bool = False
    proposal_mode: bool = False
    render_mode: bool = False
    reasoning_counters: AdvertisingClosureReasoningCounters = AdvertisingClosureReasoningCounters()
    closure_ffmpeg_enabled: bool = False

    @classmethod
    def begin(cls, *, proposal_mode: bool = False, render_mode: bool = False) -> None:
        cls.active = True
        cls.proposal_mode = proposal_mode
        cls.render_mode = render_mode
        cls.closure_ffmpeg_enabled = False
        cls.reasoning_counters = AdvertisingClosureReasoningCounters()

    @classmethod
    def end(cls) -> None:
        cls.active = False
        cls.proposal_mode = False
        cls.render_mode = False
        cls.closure_ffmpeg_enabled = False

    @classmethod
    def enable_closure_ffmpeg(cls) -> None:
        cls.closure_ffmpeg_enabled = True

    @classmethod
    def assert_reasoning_call_allowed(cls, role: str) -> None:
        if not cls.active:
            return
        if cls.render_mode:
            raise Builder2TournamentError(f"{ADVERTISING_CLOSURE_REASONING_BLOCKED}:{normalize_reasoning_role(role)}")
        if cls.proposal_mode and normalize_reasoning_role(role) != "advertising_closure" and role != "advertising_closure":
            raise Builder2TournamentError(f"{ADVERTISING_CLOSURE_REASONING_BLOCKED}:{normalize_reasoning_role(role)}")

    @classmethod
    def record_reasoning_call_submitted(cls, role: str, *, call_type: str = "normal") -> None:
        if not cls.active:
            return
        cls.reasoning_counters.increment(role, call_type=call_type)

    @classmethod
    def assert_safe_before_closure_ffmpeg(cls) -> None:
        if not cls.active or not cls.render_mode:
            return
        if not cls.closure_ffmpeg_enabled:
            raise Builder2TournamentError("builder2_advertising_closure_ffmpeg_disabled")

    @classmethod
    def reasoning_report(cls) -> Dict[str, int]:
        return cls.reasoning_counters.to_report_dict()
