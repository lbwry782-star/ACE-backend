"""
Builder2 tournament configuration — isolated from Builder1.
"""
from __future__ import annotations

import logging
import os
from typing import FrozenSet, List, Tuple

from engine.builder2_reasoning_config import (
    DEFAULT_BUILDER2_REASONING_MODEL,
    resolve_builder2_reasoning_model,
)

logger = logging.getLogger(__name__)

DEFAULT_BUILDER2_TOURNAMENT_ENABLED = True
DEFAULT_BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND = 1
DEFAULT_BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND = 1
DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS = 1
DEFAULT_BUILDER2_CREATOR_BATCH_SIZE = 1
DEFAULT_BUILDER2_JUDGE_BATCH_SIZE = 1

DEFAULT_ACTIVE_PROTOTYPE_IDS: Tuple[str, ...] = (
    "winning_card",
    "summer_fan",
    "greenpeace_essential_pairing",
    "forgot",
    "closest",
    "think_small",
)

REFERENCE_ONLY_PROTOTYPE_IDS: FrozenSet[str] = frozenset(
    {
        "shared_word_line_mechanism",
        "old_commercial_code_inversion",
        "context_collision",
    }
)

ALL_KNOWN_PROTOTYPE_IDS: FrozenSet[str] = frozenset(
    set(DEFAULT_ACTIVE_PROTOTYPE_IDS) | REFERENCE_ONLY_PROTOTYPE_IDS
)


class Builder2TournamentConfigError(ValueError):
    """Invalid Builder2 tournament configuration."""


def _parse_bool(raw: str, *, default: bool) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise Builder2TournamentConfigError(f"builder2_tournament_invalid_bool:{raw}")


def _parse_positive_int(raw: str, *, default: int, name: str, min_value: int = 0) -> int:
    text = (raw or "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        logger.error("BUILDER2_TOURNAMENT_CONFIG_INVALID name=%s value=%s reason=not_integer", name, raw)
        raise Builder2TournamentConfigError(f"builder2_tournament_invalid_int:{name}:{raw}")
    if value < min_value:
        logger.error(
            "BUILDER2_TOURNAMENT_CONFIG_INVALID name=%s value=%s min=%s",
            name,
            value,
            min_value,
        )
        raise Builder2TournamentConfigError(f"builder2_tournament_invalid_int:{name}:{value}")
    return value


def resolve_builder2_tournament_enabled() -> bool:
    raw = os.environ.get("BUILDER2_TOURNAMENT_ENABLED")
    if raw is None or not str(raw).strip():
        return DEFAULT_BUILDER2_TOURNAMENT_ENABLED
    return _parse_bool(str(raw), default=DEFAULT_BUILDER2_TOURNAMENT_ENABLED)


def resolve_builder2_tournament_attempts_per_prototype_per_round() -> int:
    return _parse_positive_int(
        os.environ.get("BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND") or "",
        default=DEFAULT_BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND,
        name="BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND",
        min_value=1,
    )


def resolve_builder2_tournament_eliminations_per_round() -> int:
    return _parse_positive_int(
        os.environ.get("BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND") or "",
        default=DEFAULT_BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND,
        name="BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND",
        min_value=1,
    )


def resolve_builder2_tournament_max_rounds() -> int:
    return _parse_positive_int(
        os.environ.get("BUILDER2_TOURNAMENT_MAX_ROUNDS") or "",
        default=DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS,
        name="BUILDER2_TOURNAMENT_MAX_ROUNDS",
        min_value=0,
    )


def resolve_builder2_creator_batch_size() -> int:
    return _parse_positive_int(
        os.environ.get("BUILDER2_CREATOR_BATCH_SIZE") or "",
        default=DEFAULT_BUILDER2_CREATOR_BATCH_SIZE,
        name="BUILDER2_CREATOR_BATCH_SIZE",
        min_value=1,
    )


def resolve_builder2_judge_batch_size() -> int:
    return _parse_positive_int(
        os.environ.get("BUILDER2_JUDGE_BATCH_SIZE") or "",
        default=DEFAULT_BUILDER2_JUDGE_BATCH_SIZE,
        name="BUILDER2_JUDGE_BATCH_SIZE",
        min_value=1,
    )


def resolve_builder2_creator_model() -> str:
    raw = (os.environ.get("BUILDER2_CREATOR_MODEL") or "").strip()
    if raw:
        return raw
    return resolve_builder2_reasoning_model() or DEFAULT_BUILDER2_REASONING_MODEL


def resolve_builder2_judge_model() -> str:
    raw = (os.environ.get("BUILDER2_JUDGE_MODEL") or "").strip()
    if raw:
        return raw
    return resolve_builder2_reasoning_model() or DEFAULT_BUILDER2_REASONING_MODEL


def resolve_builder2_winner_model() -> str:
    raw = (os.environ.get("BUILDER2_WINNER_MODEL") or "").strip()
    if raw:
        return raw
    return resolve_builder2_reasoning_model() or DEFAULT_BUILDER2_REASONING_MODEL


def resolve_builder2_active_prototype_ids() -> List[str]:
    raw = (os.environ.get("BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES") or "").strip()
    if not raw:
        return list(DEFAULT_ACTIVE_PROTOTYPE_IDS)
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [pid for pid in ids if pid not in ALL_KNOWN_PROTOTYPE_IDS]
    if unknown:
        logger.error("BUILDER2_TOURNAMENT_UNKNOWN_PROTOTYPES ids=%s", unknown)
        raise Builder2TournamentConfigError(f"builder2_tournament_unknown_prototype:{','.join(unknown)}")
    reference = [pid for pid in ids if pid in REFERENCE_ONLY_PROTOTYPE_IDS]
    if reference:
        logger.error("BUILDER2_TOURNAMENT_REFERENCE_ONLY_PROTOTYPES ids=%s", reference)
        raise Builder2TournamentConfigError(
            f"builder2_tournament_reference_only_prototype:{','.join(reference)}"
        )
    if not ids:
        raise Builder2TournamentConfigError("builder2_tournament_empty_prototype_pool")
    return ids
