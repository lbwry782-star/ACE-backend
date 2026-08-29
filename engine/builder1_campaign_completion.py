"""
Builder1 campaign completion gate — authoritative readiness checks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.builder1_marketing_copy import MARKETING_TEXT_WORD_COUNT, count_marketing_words
from engine.builder1_marketing_placeholders import has_builder1_marketing_placeholder_residue
from engine.builder1_image_artifact_store import get_builder1_image_artifact_path


def _ad_marketing_text(plan, ad_index: int) -> str:
    for ad in plan.ads:
        if ad.index == ad_index:
            return str(ad.marketing_text or "").strip()
    return ""


def evaluate_campaign_completion(session) -> Dict[str, Any]:
    target = int(session.target_ad_count)
    generated = sorted(int(x) for x in (session.generated_indexes or []))
    expected = list(range(1, target + 1))
    missing_indexes = [idx for idx in expected if idx not in generated]
    artifacts = dict(getattr(session, "ad_artifacts", None) or {})
    missing_artifacts: List[int] = []
    missing_marketing: List[int] = []
    invalid_marketing: List[int] = []
    placeholder_marketing: List[int] = []

    for idx in expected:
        art = artifacts.get(str(idx)) or artifacts.get(idx)
        token = str((art or {}).get("token") or "").strip()
        status = str((art or {}).get("status") or "").strip()
        if not token or status != "succeeded" or get_builder1_image_artifact_path(token) is None:
            missing_artifacts.append(idx)
        text = _ad_marketing_text(session.plan, idx)
        if not text:
            missing_marketing.append(idx)
        elif count_marketing_words(text) != MARKETING_TEXT_WORD_COUNT:
            invalid_marketing.append(idx)
        elif has_builder1_marketing_placeholder_residue(text):
            placeholder_marketing.append(idx)

    count_ok = len(generated) == target and not missing_indexes
    artifacts_ok = not missing_artifacts and count_ok
    marketing_ok = not missing_marketing and not invalid_marketing and not placeholder_marketing
    campaign_complete = bool(session.complete) and count_ok
    campaign_ready = campaign_complete and artifacts_ok and marketing_ok

    return {
        "campaignComplete": campaign_complete,
        "campaignReady": campaign_ready,
        "generatedCount": len(generated),
        "targetAdCount": target,
        "missingIndexes": missing_indexes,
        "missingArtifacts": missing_artifacts,
        "missingMarketingText": missing_marketing,
        "invalidMarketingWordCount": invalid_marketing,
        "marketingPlaceholderResidue": placeholder_marketing,
        "deliveryReconstructible": campaign_ready,
    }


def campaign_completion_public_fields(session) -> Dict[str, Any]:
    report = evaluate_campaign_completion(session)
    return {
        "campaignComplete": report["campaignComplete"],
        "campaignReady": report["campaignReady"],
        "deliveryReconstructible": report["deliveryReconstructible"],
        "generatedCount": report["generatedCount"],
        "targetAdCount": report["targetAdCount"],
    }
