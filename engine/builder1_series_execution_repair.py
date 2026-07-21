"""
Targeted Builder1 series execution repair for confirmed duplicate advertisements.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence

from engine.builder1_final_stages import SeriesAdsOutput, coerce_json_dict
from engine.builder1_planning_contract import (
    STAGE_SERIES_EXECUTION_REPAIR_SYSTEM,
    build_series_execution_repair_user_prompt,
)
from engine.builder1_series_distinctness import (
    DuplicateExecutionFinding,
    duplicated_ad_indexes,
    validate_ad_execution_distinctness,
)
from engine.builder1_staged_parsers import StageParseError

logger = logging.getLogger(__name__)


def _ad_by_index(ads: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for ad in ads:
        if not isinstance(ad, Mapping):
            continue
        try:
            idx = int(ad.get("index"))
        except (TypeError, ValueError):
            continue
        indexed[idx] = dict(ad)
    return indexed


def merge_repaired_ads(
    *,
    original_ads: Sequence[Mapping[str, Any]],
    repaired_ads: Sequence[Mapping[str, Any]],
    repair_indexes: Sequence[int],
) -> List[Dict[str, Any]]:
    original_by_index = _ad_by_index(original_ads)
    repaired_by_index = _ad_by_index(repaired_ads)
    merged: List[Dict[str, Any]] = []
    for idx in sorted(original_by_index):
        if idx in repair_indexes and idx in repaired_by_index:
            merged.append(repaired_by_index[idx])
        else:
            merged.append(original_by_index[idx])
    return merged


def parse_series_execution_repair_output(
    raw_payload: object,
    *,
    original_ads: Sequence[Mapping[str, Any]],
    expected_indexes: Sequence[int],
) -> List[Dict[str, Any]]:
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("series_execution_repair", ["series_execution_repair_not_object"]) from exc

    ads_raw = obj.get("ads")
    if not isinstance(ads_raw, list) or not ads_raw:
        raise StageParseError("series_execution_repair", ["series_execution_repair_missing_ads"])

    repaired_by_index = _ad_by_index(ads_raw)
    missing = [str(idx) for idx in expected_indexes if idx not in repaired_by_index]
    if missing:
        raise StageParseError(
            "series_execution_repair",
            [f"series_execution_repair_missing_index:{idx}" for idx in missing],
        )

    merged_ads = merge_repaired_ads(
        original_ads=original_ads,
        repaired_ads=list(repaired_by_index.values()),
        repair_indexes=list(expected_indexes),
    )
    duplicate_reasons, _findings = validate_ad_execution_distinctness(merged_ads)
    if duplicate_reasons:
        raise StageParseError("series_execution_repair", duplicate_reasons)

    return [repaired_by_index[idx] for idx in expected_indexes]


def attempt_series_execution_repair(
    *,
    series_ads: SeriesAdsOutput,
    duplicate_reasons: Sequence[str],
    findings: Sequence[DuplicateExecutionFinding],
    brand_slogan: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    graphic_generator: Dict[str, Any],
    detected_language: str,
    model_caller: Callable[..., Any],
    run_stage: Callable[..., Any],
) -> SeriesAdsOutput:
    repair_indexes = duplicated_ad_indexes(findings)
    if not repair_indexes:
        raise StageParseError("series_execution_repair", ["series_execution_repair_no_target_ads"])

    ads_by_index = _ad_by_index(series_ads.ads)
    valid_ads = [ads_by_index[idx] for idx in sorted(ads_by_index) if idx not in repair_indexes]
    duplicated_ads = [ads_by_index[idx] for idx in repair_indexes if idx in ads_by_index]

    user_prompt = build_series_execution_repair_user_prompt(
        duplicate_reasons=list(duplicate_reasons),
        duplicate_ad_indexes=repair_indexes,
        valid_ads=valid_ads,
        duplicated_ads=duplicated_ads,
        brand_slogan=brand_slogan,
        conceptual=conceptual,
        brand_physical=brand_physical,
        graphic_generator=graphic_generator,
        detected_language=detected_language,
    )

    def _parse(payload: object) -> List[Dict[str, Any]]:
        return parse_series_execution_repair_output(
            payload,
            original_ads=series_ads.ads,
            expected_indexes=repair_indexes,
        )

    raw = run_stage(
        "series_execution_repair",
        model_caller,
        STAGE_SERIES_EXECUTION_REPAIR_SYSTEM,
        user_prompt,
        _parse,
        repair_builder=lambda broken, reasons: user_prompt,
    )
    if not isinstance(raw, list):
        raise StageParseError("series_execution_repair", ["series_execution_repair_invalid_result"])

    repaired_by_index = {int(ad["index"]): ad for ad in raw if isinstance(ad, dict)}
    merged_ads = merge_repaired_ads(
        original_ads=series_ads.ads,
        repaired_ads=list(repaired_by_index.values()),
        repair_indexes=repair_indexes,
    )
    logger.info(
        "BUILDER1_SERIES_EXECUTION_REPAIR_OK repairedIndexes=%s duplicateReasons=%s targetedRepairUsed=true",
        repair_indexes,
        list(duplicate_reasons),
    )
    return SeriesAdsOutput(series_generator=dict(series_ads.series_generator), ads=merged_ads)
