"""
Builder1 campaign-series ZIP builder.
"""
from __future__ import annotations

import base64
import io
import zipfile
from typing import Any, Dict, List

from engine.builder1_plan_spec import AD_COUNT_MAX, AD_COUNT_MIN


def _decode_image_bytes(image_base64: str) -> bytes:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        comma = raw.find(",")
        if comma >= 0:
            raw = raw[comma + 1 :]
    try:
        return base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("invalid_image_base64") from exc


def _normalize_ads(ads: object) -> List[Dict[str, Any]]:
    if not isinstance(ads, list):
        raise ValueError("invalid_ads")
    if len(ads) < AD_COUNT_MIN or len(ads) > AD_COUNT_MAX:
        raise ValueError("invalid_ad_count")
    parsed: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for item in ads:
        if not isinstance(item, dict):
            raise ValueError("invalid_ad_entry")
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_ad_index") from exc
        if idx in seen:
            raise ValueError("duplicate_ad_index")
        seen.add(idx)
        image_b64 = item.get("imageBase64") or ""
        if not str(image_b64).strip():
            raise ValueError("missing_image")
        headline = item.get("headline")
        if headline is not None and not isinstance(headline, str):
            headline = str(headline)
        if isinstance(headline, str) and not headline.strip():
            headline = None
        marketing = str(item.get("marketingText") or "").strip()
        parsed.append(
            {
                "index": idx,
                "imageBase64": str(image_b64),
                "headline": headline,
                "marketingText": marketing,
            }
        )
    expected = set(range(1, len(parsed) + 1))
    if seen != expected:
        raise ValueError("ad_indexes_not_sequential")
    parsed.sort(key=lambda a: a["index"])
    return parsed


def build_builder1_zip_bytes(payload: Dict[str, Any]) -> bytes:
    """
    Build ZIP with ad-XX.jpg, ad-XX.txt, and campaign.txt for a full campaign series.
    """
    product_name = str(payload.get("productName") or "").strip()
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
            zf.writestr(f"ad-{label}.jpg", image_bytes)
            txt_lines: List[str] = []
            if ad.get("headline"):
                txt_lines.append(f"Headline: {ad['headline']}")
            if ad.get("marketingText"):
                txt_lines.append(f"Marketing: {ad['marketingText']}")
            zf.writestr(f"ad-{label}.txt", "\n".join(txt_lines) + ("\n" if txt_lines else ""))

    buf.seek(0)
    return buf.read()
