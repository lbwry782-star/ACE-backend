"""
Builder2 media-only reasoning isolation — block every Responses API role.
"""
from __future__ import annotations

from typing import Dict

from engine.builder2_tournament_contracts import Builder2TournamentError

MEDIA_RESUME_REASONING_BLOCKED = "builder2_media_resume_reasoning_call_blocked"
MEDIA_RESUME_MODEL_DEPENDENT_DELIVERY = "builder2_media_resume_model_dependent_delivery"

_KNOWN_ROLE_BUCKETS: Dict[str, str] = {
    "builder2_strategy": "strategy",
    "strategy": "strategy",
    "builder2_creator": "creator",
    "creator": "creator",
    "builder2_judge": "judge",
    "judge": "judge",
    "builder2_winner": "winner",
    "winner": "winner",
    "winner_development": "winner",
    "marketing_copy": "marketing_copy",
    "video_headline": "headline",
    "headline": "headline",
    "keyword": "keyword",
    "plan_repair": "other",
    "copy_repair": "other",
    "copy_retry": "other",
    "generic_text_fallback": "other",
}


def normalize_reasoning_role(role: str) -> str:
    token = (role or "").strip().lower()
    return _KNOWN_ROLE_BUCKETS.get(token, "other")


class MediaResumeReasoningCounters:
    strategyCalls: int = 0
    creatorCalls: int = 0
    judgeCalls: int = 0
    winnerCalls: int = 0
    marketingCopyCalls: int = 0
    headlineCalls: int = 0
    keywordCalls: int = 0
    otherReasoningCalls: int = 0

    def increment(self, role: str) -> None:
        bucket = normalize_reasoning_role(role)
        if bucket == "strategy":
            self.strategyCalls += 1
        elif bucket == "creator":
            self.creatorCalls += 1
        elif bucket == "judge":
            self.judgeCalls += 1
        elif bucket == "winner":
            self.winnerCalls += 1
        elif bucket == "marketing_copy":
            self.marketingCopyCalls += 1
        elif bucket == "headline":
            self.headlineCalls += 1
        elif bucket == "keyword":
            self.keywordCalls += 1
        else:
            self.otherReasoningCalls += 1

    @property
    def totalReasoningCalls(self) -> int:
        return (
            self.strategyCalls
            + self.creatorCalls
            + self.judgeCalls
            + self.winnerCalls
            + self.marketingCopyCalls
            + self.headlineCalls
            + self.keywordCalls
            + self.otherReasoningCalls
        )

    def to_report_dict(self) -> Dict[str, int]:
        return {
            "strategyCalls": self.strategyCalls,
            "creatorCalls": self.creatorCalls,
            "judgeCalls": self.judgeCalls,
            "winnerCalls": self.winnerCalls,
            "marketingCopyCalls": self.marketingCopyCalls,
            "headlineCalls": self.headlineCalls,
            "keywordCalls": self.keywordCalls,
            "otherReasoningCalls": self.otherReasoningCalls,
            "totalReasoningCalls": self.totalReasoningCalls,
        }


def assert_media_resume_reasoning_call_allowed(*, role: str, active: bool) -> None:
    if not active:
        return
    blocked_role = normalize_reasoning_role(role)
    raise Builder2TournamentError(f"{MEDIA_RESUME_REASONING_BLOCKED}:{blocked_role}")
