"""
Clean Builder1 rebuild scaffold.
Not wired into production yet.
"""

SIMILARITY_THRESHOLD_REPLACEMENT = 85
MODE_SIDE_BY_SIDE = "SIDE_BY_SIDE"
MODE_REPLACEMENT = "REPLACEMENT"


def decide_mode(similarity: float) -> str:
    if similarity >= SIMILARITY_THRESHOLD_REPLACEMENT:
        return MODE_REPLACEMENT
    return MODE_SIDE_BY_SIDE
