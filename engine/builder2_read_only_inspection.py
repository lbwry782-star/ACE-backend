"""
Builder2 read-only inspection guard — derive indexes without persisting mutations.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_read_only_depth = 0


@dataclass
class Builder2ReadOnlyInspectionCounter:
    redis_mutations: int = field(default=0)

    def record_redis_mutation(self, method: str = "") -> None:
        self.redis_mutations += 1
        if _read_only_depth > 0:
            logger.debug("BUILDER2_READ_ONLY_INSPECTION_BLOCKED_MUTATION method=%s", method or "unknown")


def read_only_inspection_active() -> bool:
    return _read_only_depth > 0


@contextmanager
def read_only_builder2_inspection() -> Iterator[Builder2ReadOnlyInspectionCounter]:
    global _read_only_depth
    counter = Builder2ReadOnlyInspectionCounter()
    _read_only_depth += 1
    try:
        yield counter
    finally:
        _read_only_depth -= 1


def record_read_only_redis_mutation(counter: Optional[Builder2ReadOnlyInspectionCounter], method: str = "") -> None:
    if counter is not None:
        counter.record_redis_mutation(method)
