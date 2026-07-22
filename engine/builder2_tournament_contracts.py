"""
Builder2 tournament contracts — schemas, errors, score ranges.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Tuple

STRATEGY_SCHEMA_VERSION = "builder2_strategy_v1"
CANDIDATE_SCHEMA_VERSION = "builder2_candidate_v1"
JUDGMENT_SCHEMA_VERSION = "builder2_judgment_v1"
TOURNAMENT_STATE_SCHEMA_VERSION = "builder2_tournament_state_v1"
WINNER_PLAN_SCHEMA_VERSION = "builder2_winner_video_plan_v1"

VALID_GROUNDING_TYPES: FrozenSet[str] = frozenset(
    {
        "user_provided_fact",
        "observable_practice",
        "physical_reality",
        "common_market_behavior",
        "professional_knowledge",
    }
)

VALID_VISUAL_PARALLEL_TYPES: FrozenSet[str] = frozenset(
    {
        "replacement",
        "side_by_side",
        "motion_similarity",
        "physical_behavior",
        "graphic_similarity",
        "context_collision",
        "context_replacement",
        "media_replacement",
        "medium_as_object",
        "essential_pairing",
        "spatial_proximity",
        "consequence_embodiment",
        "other",
    }
)

VALID_STRUCTURE_TYPES: FrozenSet[str] = frozenset({"continuous_event", "variation_montage"})
VALID_CONTINUITY_RISK: FrozenSet[str] = frozenset({"low", "medium", "high"})

JUDGE_SCORE_RANGES: Dict[str, Tuple[int, int]] = {
    "problemAdvantageIntegrity": (0, 20),
    "mechanismQuality": (0, 15),
    "prototypeMethodApplication": (0, 10),
    "silentVisualClarity": (0, 15),
    "originalityFreshness": (0, 15),
    "eleganceSimplicity": (0, 10),
    "runwayFeasibility": (0, 10),
    "editingContribution": (0, 5),
}

JUDGE_SCORE_MAX_TOTAL = sum(high for _, high in JUDGE_SCORE_RANGES.values())

CREATOR_PURITY_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "previous candidate",
    "earlier candidate",
    "other prototype",
    "judge score",
    "judge feedback",
    "tournament standing",
    "current ranking",
    "eliminated prototype",
    "best candidate",
    "outperform",
    "higher score",
    "leading prototype",
)

CREATOR_PURITY_RULES: Tuple[Tuple[str, str], ...] = (
    ("previous candidate", "mentions_other_candidate"),
    ("earlier candidate", "mentions_other_candidate"),
    ("other prototype", "mentions_other_candidate"),
    ("judge score", "mentions_judge_score"),
    ("judge feedback", "mentions_judge_score"),
    ("tournament standing", "mentions_tournament_ranking"),
    ("current ranking", "mentions_tournament_ranking"),
    ("eliminated prototype", "mentions_tournament_ranking"),
    ("best candidate", "mentions_other_candidate"),
    ("outperform", "mentions_other_candidate"),
    ("higher score", "mentions_judge_score"),
    ("leading prototype", "mentions_tournament_ranking"),
)

JUDGE_PURITY_FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "other candidate",
    "other candidates",
    "tournament standing",
    "compared to candidate",
    "ranked against",
    "replace the idea",
    "redesign the candidate",
    "new advertisement",
    "replacement creator report",
    "creator probably",
    "creator meant",
    "missing creator report",
)


class Builder2TournamentError(Exception):
    """Tournament failure mapped to RunwayVideoMVPError by caller."""


def compare_candidate_rankings(new_record: Dict[str, Any], old_record: Dict[str, Any]) -> int:
    """
    Return positive if new_record ranks higher than old_record.
    Comparison order:
    1 total score
    2 silentVisualClarity
    3 problemAdvantageIntegrity
    4 runwayFeasibility
    5 earlier completedAt
    6 lexicographically smaller candidateId
    """
    if not old_record.get("eligible"):
        return 1 if new_record.get("eligible") else 0
    if not new_record.get("eligible"):
        return -1

    def _key(rec: Dict[str, Any]) -> tuple:
        tie = rec.get("tieScores") or {}
        return (
            int(rec.get("totalScore") or -1),
            int(tie.get("silentVisualClarity") or -1),
            int(tie.get("problemAdvantageIntegrity") or -1),
            int(tie.get("runwayFeasibility") or -1),
            -int(_timestamp_key(rec.get("completedAt") or "")),
            tuple(-ord(c) for c in str(rec.get("candidateId") or "")),
        )

    new_key = _key(new_record)
    old_key = _key(old_record)
    if new_key == old_key:
        return 0
    return 1 if new_key > old_key else -1


def _timestamp_key(value: str) -> int:
    # Earlier ISO timestamps sort higher via negation above; use ordinal proxy.
    return sum(ord(ch) for ch in value)


def require_dict(value: Any, *, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise Builder2TournamentError(f"builder2_tournament_invalid_field:{field}")
    return value


def require_non_empty_str(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise Builder2TournamentError(f"builder2_tournament_invalid_field:{field}")
    return text
