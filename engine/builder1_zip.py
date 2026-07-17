"""
Builder1 campaign-series ZIP builder.
"""
from __future__ import annotations

import base64
import io
import re
import zipfile
from typing import Any, Dict, List, Tuple

from engine.builder1_marketing_copy import (
    MARKETING_TEXT_WORD_COUNT,
    count_marketing_words,
    normalize_marketing_text,
    validate_marketing_text_50_words,
)
from engine.builder1_plan_spec import AD_COUNT_MAX, AD_COUNT_MIN

MAX_ZIP_IMAGE_BYTES = 12 * 1024 * 1024
MAX_ZIP_PAYLOAD_BYTES = 16 * 1024 * 1024
SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _decode_image_bytes(image_base64: str) -> bytes:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        comma = raw.find(",")
        if comma >= 0:
            raw = raw[comma + 1 :]
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("invalid_image_base64") from exc
    if not image_bytes:
        raise ValueError("invalid_image_base64")
    if len(image_bytes) > MAX_ZIP_IMAGE_BYTES:
        raise ValueError("image_too_large")
    return image_bytes


def _public_ad_text_file(
    *,
    product_name: str,
    brand_slogan: str,
    headline: str | None,
    marketing_text: str,
) -> str:
    lines: List[str] = []
    if product_name:
        lines.append(f"Product: {product_name}")
    if brand_slogan:
        lines.append(f"Slogan: {brand_slogan}")
    if headline:
        lines.append(f"Headline: {headline}")
    if lines:
        lines.append("")
    lines.append(marketing_text)
    return "\n".join(lines) + "\n"


def _validate_marketing_text_for_zip(marketing_text: object) -> str:
    text = normalize_marketing_text(marketing_text)
    if not text:
        raise ValueError("missing_marketing_text")
    validate_marketing_text_50_words(text)
    return text


def _normalize_single_ad(ad: object) -> Dict[str, Any]:
    if not isinstance(ad, dict):
        raise ValueError("invalid_ad_entry")
    try:
        idx = int(ad.get("index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_ad_index") from exc
    if idx < 1 or idx > AD_COUNT_MAX:
        raise ValueError("invalid_ad_index")
    image_b64 = ad.get("imageBase64") or ""
    if not str(image_b64).strip():
        raise ValueError("missing_image")
    headline = ad.get("headline")
    if headline is not None and not isinstance(headline, str):
        headline = str(headline)
    if isinstance(headline, str) and not headline.strip():
        headline = None
    marketing = _validate_marketing_text_for_zip(ad.get("marketingText"))
    return {
        "index": idx,
        "imageBase64": str(image_b64),
        "headline": headline,
        "marketingText": marketing,
    }


def _normalize_ads(ads: object) -> List[Dict[str, Any]]:
    if not isinstance(ads, list):
        raise ValueError("invalid_ads")
    if len(ads) < AD_COUNT_MIN or len(ads) > AD_COUNT_MAX:
        raise ValueError("invalid_ad_count")
    parsed: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for item in ads:
        ad = _normalize_single_ad(item)
        idx = ad["index"]
        if idx in seen:
            raise ValueError("duplicate_ad_index")
        seen.add(idx)
        parsed.append(ad)
    expected = set(range(1, len(parsed) + 1))
    if seen != expected:
        raise ValueError("ad_indexes_not_sequential")
    parsed.sort(key=lambda a: a["index"])
    return parsed


def build_builder1_single_ad_zip_bytes(payload: Dict[str, Any]) -> Tuple[bytes, str]:
    """Build a single-ad ZIP and return (bytes, download_filename)."""
    scope = str(payload.get("scope") or "").strip().lower()
    if scope != "single_ad":
        raise ValueError("invalid_zip_scope")

    campaign = payload.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("invalid_campaign")

    product_name = str(campaign.get("productNameResolved") or campaign.get("productName") or "").strip()
    brand_slogan = str(campaign.get("brandSlogan") or "").strip()
    ad = _normalize_single_ad(payload.get("ad"))

    label = f"{ad['index']:02d}"
    download_name = f"ad-{label}.zip"
    if not SAFE_FILENAME_RE.match(download_name):
        raise ValueError("unsafe_filename")

    image_bytes = _decode_image_bytes(ad["imageBase64"])
    txt = _public_ad_text_file(
        product_name=product_name,
        brand_slogan=brand_slogan,
        headline=ad.get("headline"),
        marketing_text=ad["marketingText"],
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"ad-{label}.jpg", image_bytes)
        zf.writestr(f"ad-{label}.txt", txt)
    zip_bytes = buf.getvalue()
    if len(zip_bytes) > MAX_ZIP_PAYLOAD_BYTES:
        raise ValueError("zip_payload_too_large")
    return zip_bytes, download_name


def build_builder1_zip_bytes(payload: Dict[str, Any]) -> bytes:
    """
    Build ZIP with ad-XX.jpg, ad-XX.txt, and campaign.txt for a full campaign series.
    Backward-compatible full-campaign ZIP support.
    """
    scope = str(payload.get("scope") or "campaign").strip().lower()
    if scope == "single_ad":
        zip_bytes, _ = build_builder1_single_ad_zip_bytes(payload)
        return zip_bytes

    product_name = str(payload.get("productName") or payload.get("productNameResolved") or "").strip()
    brand_slogan = str(payload.get("brandSlogan") or "").strip()
    ads = _normalize_ads(payload.get("ads"))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        campaign_lines = []
        if product_name:
            campaign_lines.append(f"Product: {product_name}")
        if brand_slogan:
            campaign_lines.append(f"Slogan: {brand_slogan}")
        zf.writestr("campaign.txt", "\n".join(campaign_lines) + ("\n" if campaign_lines else ""))

        for ad in ads:
            idx = ad["index"]
            label = f"{idx:02d}"
            image_bytes = _decode_image_bytes(ad["imageBase64"])
            txt = _public_ad_text_file(
                product_name=product_name,
                brand_slogan=brand_slogan,
                headline=ad.get("headline"),
                marketing_text=ad["marketingText"],
            )
            zf.writestr(f"ad-{label}.jpg", image_bytes)
            zf.writestr(f"ad-{label}.txt", txt)

    zip_bytes = buf.getvalue()
    if len(zip_bytes) > MAX_ZIP_PAYLOAD_BYTES:
        raise ValueError("zip_payload_too_large")
    return zip_bytes


def build_builder1_zip_from_request(payload: Dict[str, Any]) -> Tuple[bytes, str]:
    """Route Builder1 ZIP requests by scope and return bytes plus download filename."""
    scope = str(payload.get("scope") or "campaign").strip().lower()
    if scope == "single_ad":
        return build_builder1_single_ad_zip_bytes(payload)
    return build_builder1_zip_bytes(payload), "builder1-campaign.zip"
