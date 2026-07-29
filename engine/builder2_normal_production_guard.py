"""
Builder2 normal production guard — enforce 14-call reasoning budget and block post-winner copy roles.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from engine.builder2_new_format_config import NORMAL_REASONING_CALL_BUDGET
from engine.builder2_tournament_contracts import Builder2TournamentError

NORMAL_PRODUCTION_BLOCKED_ROLES = frozenset(
    {
        "advertising_closure",
        "advertising_closure_judge",
        "marketing_copy",
    }
)

NORMAL_PRODUCTION_ALLOWED_BUCKETS = frozenset(
    {
        "strategy",
        "creator",
        "judge",
        "winner",
    }
)


class NormalProductionGuard:
    active: bool = False
    _calls: Dict[str, int] = {}

    @classmethod
    def begin(cls) -> None:
        cls.active = True
        cls._calls = {}

    @classmethod
    def end(cls) -> None:
        cls.active = False

    @classmethod
    def record_call(cls, role: str) -> None:
        from engine.builder2_media_reasoning_guard import normalize_reasoning_role

        bucket = normalize_reasoning_role(role)
        cls._calls[bucket] = int(cls._calls.get(bucket) or 0) + 1

    @classmethod
    def assert_reasoning_call_allowed(cls, role: str) -> None:
        if not cls.active:
            return
        from engine.builder2_media_reasoning_guard import normalize_reasoning_role

        bucket = normalize_reasoning_role(role)
        if bucket in NORMAL_PRODUCTION_BLOCKED_ROLES or role in NORMAL_PRODUCTION_BLOCKED_ROLES:
            raise Builder2TournamentError("builder2_normal_production_blocked_role")
        if bucket not in NORMAL_PRODUCTION_ALLOWED_BUCKETS and role not in {
            "builder2_strategy",
            "builder2_creator",
            "builder2_creator_semantic_bridge_repair",
            "builder2_judge",
            "builder2_winner",
            "winner_development",
        }:
            raise Builder2TournamentError("builder2_normal_production_blocked_role")

    @classmethod
    def snapshot(cls) -> Dict[str, int]:
        return dict(cls._calls)

    @classmethod
    def total_reasoning_calls(cls) -> int:
        return sum(cls._calls.values())

    @classmethod
    def assert_budget_not_exceeded(cls, *, max_calls: Optional[int] = None) -> None:
        limit = max_calls if max_calls is not None else NORMAL_REASONING_CALL_BUDGET
        total = cls.total_reasoning_calls()
        if total > limit:
            raise Builder2TournamentError("builder2_normal_production_reasoning_budget_exceeded")
