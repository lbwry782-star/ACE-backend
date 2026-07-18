"""Deterministic Builder1 campaign integrity validation after series_ads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.builder1_consolidated_stages import Builder1UpstreamSnapshot, build_conceptual_lineage
from engine.builder1_creative_methodology import deterministic_builder1_integrity_checks
from engine.builder1_plan_parser import _norm_text, _word_count
from engine.builder1_plan_spec import MARKETING_TEXT_WORD_COUNT, Builder1SeriesPlan, series_plan_to_store_dict


@dataclass
class Builder1IntegrityResult:
    ok: bool
    reasons: List[str]
    needs_marketing_text_repair: bool = False
    needs_series_stage_retry: bool = False
    upstream_mutation: bool = False


def _graphic_identity(graphic: Any) -> tuple[str, str]:
    if graphic is None:
        return "", ""
    return (
        _norm_text(getattr(graphic, "layout_template", "")),
        _norm_text(getattr(graphic, "recurring_graphic_device", "")),
    )


def make_upstream_snapshot(
    *,
    product_name_resolved: str,
    selected_strategy: Any,
    selected_slogan: Any,
    selected_conceptual: Any,
    brand_physical: Any,
    graphic: Any,
) -> Builder1UpstreamSnapshot:
    layout, device = _graphic_identity(graphic)
    return Builder1UpstreamSnapshot(
        product_name_resolved=_norm_text(product_name_resolved),
        strategic_problem=_norm_text(selected_strategy.strategic_problem),
        relative_advantage=_norm_text(selected_strategy.relative_advantage),
        brand_slogan=_norm_text(selected_slogan.brand_slogan),
        implied_action=_norm_text(selected_slogan.implied_action),
        selected_slogan_id=_norm_text(getattr(selected_slogan, "id", "")).upper(),
        conceptual_generator=_norm_text(selected_conceptual.generator),
        selected_conceptual_id=_norm_text(getattr(selected_conceptual, "id", "")).upper(),
        physical_generator=_norm_text(brand_physical.physical_generator),
        graphic_layout_template=layout,
        graphic_recurring_device=device,
    )


def _validate_conceptual_lineage(
    plan: Builder1SeriesPlan,
    *,
    upstream: Builder1UpstreamSnapshot,
    reasons: List[str],
) -> None:
    lineage = plan.planning_internals.get("conceptualLineage")
    if not isinstance(lineage, dict) or not lineage:
        reasons.append("conceptual_lineage_missing")
        return
    if not plan.conceptual_generator.strip():
        reasons.append("conceptual_generator_missing")
    selected_concept_id = str(lineage.get("selectedConceptCandidateId") or "").strip().upper()
    source_slogan_id = str(lineage.get("sourceSloganCandidateId") or "").strip().upper()
    if upstream.selected_conceptual_id and selected_concept_id != upstream.selected_conceptual_id:
        reasons.append("conceptual_candidate_id_mismatch")
    if upstream.selected_slogan_id and source_slogan_id != upstream.selected_slogan_id:
        reasons.append("conceptual_source_slogan_mismatch")
    if lineage.get("fixedBrandSlogan") != plan.brand_slogan:
        reasons.append("conceptual_source_slogan_mismatch")
    if lineage.get("fixedImpliedAction") != plan.slogan_action:
        reasons.append("conceptual_source_action_mismatch")


def validate_builder1_campaign_integrity(
    plan: Builder1SeriesPlan,
    *,
    upstream: Builder1UpstreamSnapshot,
    detected_language: str,
) -> Builder1IntegrityResult:
    """Structural and invariant checks only — no creative rescoring or strategic restart."""
    reasons: List[str] = []
    needs_marketing_text_repair = False
    needs_series_stage_retry = False
    upstream_mutation = False

    if _norm_text(plan.product_name_resolved) != upstream.product_name_resolved:
        reasons.append("upstream_product_name_mutated")
        upstream_mutation = True
    if _norm_text(plan.strategic_problem) != upstream.strategic_problem:
        reasons.append("upstream_strategic_problem_mutated")
        upstream_mutation = True
    if _norm_text(plan.relative_advantage) != upstream.relative_advantage:
        reasons.append("upstream_relative_advantage_mutated")
        upstream_mutation = True
    if _norm_text(plan.brand_slogan) != upstream.brand_slogan:
        reasons.append("upstream_brand_slogan_mutated")
        upstream_mutation = True
    if _norm_text(plan.slogan_action) != upstream.implied_action:
        reasons.append("upstream_implied_action_mutated")
        upstream_mutation = True
    if _norm_text(plan.conceptual_generator) != upstream.conceptual_generator:
        reasons.append("conceptual_generator_mutated")
        upstream_mutation = True
    if _norm_text(plan.physical_generator) != upstream.physical_generator:
        reasons.append("upstream_physical_generator_mutated")
        upstream_mutation = True

    layout, device = _graphic_identity(plan.graphic_generator)
    if layout and upstream.graphic_layout_template and layout != upstream.graphic_layout_template:
        reasons.append("upstream_graphic_layout_mutated")
        upstream_mutation = True
    if device and upstream.graphic_recurring_device and device != upstream.graphic_recurring_device:
        reasons.append("upstream_graphic_device_mutated")
        upstream_mutation = True

    if len(plan.ads) != plan.ad_count:
        reasons.append("ad_count_mismatch")
        needs_series_stage_retry = True

    indices = [ad.index for ad in plan.ads]
    expected = list(range(1, plan.ad_count + 1))
    if sorted(indices) != expected:
        reasons.append("ad_index_sequence_invalid")
        needs_series_stage_retry = True
    if len(set(indices)) != len(indices):
        reasons.append("duplicate_ad_indices")
        needs_series_stage_retry = True

    _validate_conceptual_lineage(plan, upstream=upstream, reasons=reasons)

    fixed_slogan = plan.brand_slogan
    for ad in plan.ads:
        ad_dict = plan.planning_internals.get("adInternals", {}).get(ad.index, {})
        ad_slogan = ad_dict.get("brandSlogan")
        if ad_slogan is not None and ad_slogan != fixed_slogan:
            reasons.append(f"ad_{ad.index}_slogan_inconsistent")

        headline_required = ad_dict.get("headlineRequired")
        if headline_required is False and ad.headline:
            reasons.append(f"ad_{ad.index}_headline_forbidden")
        if headline_required is True and not _norm_text(ad.headline):
            reasons.append(f"ad_{ad.index}_headline_required_missing")
        if headline_required is True and not _norm_text(ad_dict.get("headlineReason")):
            reasons.append(f"ad_{ad.index}_headline_reason_missing")

        word_count = _word_count(ad.marketing_text)
        if word_count != MARKETING_TEXT_WORD_COUNT:
            reasons.append(f"ad_{ad.index}_marketing_text_word_count:{word_count}")
            needs_marketing_text_repair = True

    plan_dict = series_plan_to_store_dict(plan)
    plan_dict["detectedLanguage"] = detected_language
    reasons.extend(deterministic_builder1_integrity_checks(plan_dict))

    structural_series_codes = {
        "ad_count_mismatch",
        "ad_index_sequence_invalid",
        "duplicate_ad_indices",
    }
    if any(code in structural_series_codes for code in reasons):
        needs_series_stage_retry = True

    return Builder1IntegrityResult(
        ok=len(reasons) == 0,
        reasons=list(dict.fromkeys(reasons)),
        needs_marketing_text_repair=needs_marketing_text_repair and not upstream_mutation,
        needs_series_stage_retry=needs_series_stage_retry and not upstream_mutation,
        upstream_mutation=upstream_mutation,
    )
