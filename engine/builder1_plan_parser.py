"""
Disconnected Builder1 planning response parser and validator.
"""
from __future__ import annotations

from engine.builder1_plan_spec import (
    Builder1Plan,
    MODE_REPLACEMENT,
    MODE_SIDE_BY_SIDE,
    REPLACEMENT_THRESHOLD,
)


class Builder1PlanParseError(ValueError):
    pass


def _require_dict(value: object) -> dict:
    if not isinstance(value, dict):
        raise Builder1PlanParseError("payload_must_be_dict")
    return value


def _require_string(value: object, field_name: str) -> str:
    s = (value if isinstance(value, str) else str(value or "")).strip()
    if not s:
        raise Builder1PlanParseError(f"invalid_{field_name}")
    return s


def _require_language(value: object) -> str:
    language = _require_string(value, "detectedLanguage")
    if language not in {"he", "en"}:
        raise Builder1PlanParseError("invalid_detectedLanguage")
    return language


def _require_mode(value: object) -> str:
    mode = _require_string(value, "modeDecision")
    if mode not in {MODE_REPLACEMENT, MODE_SIDE_BY_SIDE}:
        raise Builder1PlanParseError("invalid_modeDecision")
    return mode


def _require_similarity_score(value: object) -> int:
    if isinstance(value, bool):
        raise Builder1PlanParseError("invalid_visualSimilarityScore")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str):
        v = value.strip()
        if not v:
            raise Builder1PlanParseError("invalid_visualSimilarityScore")
        if v.startswith(("+", "-")):
            sign = v[0]
            num = v[1:]
            if not num.isdigit():
                raise Builder1PlanParseError("invalid_visualSimilarityScore")
            score = int(sign + num)
        elif v.isdigit():
            score = int(v)
        else:
            raise Builder1PlanParseError("invalid_visualSimilarityScore")
    else:
        raise Builder1PlanParseError("invalid_visualSimilarityScore")
    if score < 0 or score > 100:
        raise Builder1PlanParseError("invalid_visualSimilarityScore_range")
    return score


def _check_mode_matches_threshold(score: int, mode: str) -> None:
    expected = MODE_REPLACEMENT if score >= REPLACEMENT_THRESHOLD else MODE_SIDE_BY_SIDE
    if mode != expected:
        raise Builder1PlanParseError("mode_threshold_mismatch")


def parse_builder1_plan(payload: object) -> Builder1Plan:
    data = _require_dict(payload)
    required_keys = {
        "productNameResolved",
        "detectedLanguage",
        "advertisingPromise",
        "objectA",
        "objectASecondary",
        "objectB",
        "visualSimilarityScore",
        "modeDecision",
        "visualDescription",
    }
    keys = set(data.keys())
    missing = required_keys - keys
    extra = keys - required_keys
    if missing:
        raise Builder1PlanParseError(f"missing_keys:{','.join(sorted(missing))}")
    if extra:
        raise Builder1PlanParseError(f"extra_keys:{','.join(sorted(extra))}")

    score = _require_similarity_score(data["visualSimilarityScore"])
    mode = _require_mode(data["modeDecision"])
    _check_mode_matches_threshold(score, mode)

    return Builder1Plan(
        product_name="",
        product_description="",
        format="",
        product_name_resolved=_require_string(data["productNameResolved"], "productNameResolved"),
        detected_language=_require_language(data["detectedLanguage"]),
        advertising_promise=_require_string(data["advertisingPromise"], "advertisingPromise"),
        object_a=_require_string(data["objectA"], "objectA"),
        object_a_secondary=_require_string(data["objectASecondary"], "objectASecondary"),
        object_b=_require_string(data["objectB"], "objectB"),
        visual_similarity_score=score,
        mode_decision=mode,
        visual_description=_require_string(data["visualDescription"], "visualDescription"),
    )
