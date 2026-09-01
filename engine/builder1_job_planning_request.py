"""
Server-authoritative Builder1 planning request snapshot stored on the job record.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from engine.builder1_job_ownership import _hash_token
from engine.builder1_request_idempotency import fingerprint_initial_generate


def _clean(value: object) -> str:
    return str(value or "").strip()


def _canonical_brand_guidelines(raw: object) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): raw[k] for k in sorted(raw.keys(), key=str)}
    return None


def build_planning_request_snapshot(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    brand_guidelines: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    guidelines = _canonical_brand_guidelines(brand_guidelines)
    fingerprint = fingerprint_initial_generate(
        {
            "productName": _clean(product_name),
            "productDescription": _clean(product_description),
            "format": _clean(format_value) or "portrait",
            "brandGuidelines": guidelines,
        },
        ad_count=int(ad_count),
    )
    return {
        "productName": _clean(product_name),
        "productDescription": _clean(product_description),
        "format": _clean(format_value) or "portrait",
        "adCount": int(ad_count),
        "brandGuidelines": guidelines,
        "requestFingerprint": fingerprint,
    }


def planning_request_snapshot_from_job(job: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = job.get("planningRequestSnapshot")
    if isinstance(raw, dict) and _clean(raw.get("productDescription")):
        return dict(raw)
    return None


def snapshot_request_fingerprint(snapshot: Mapping[str, Any]) -> str:
    stored = _clean(snapshot.get("requestFingerprint"))
    if stored:
        return stored
    return fingerprint_initial_generate(
        {
            "productName": _clean(snapshot.get("productName")),
            "productDescription": _clean(snapshot.get("productDescription")),
            "format": _clean(snapshot.get("format")) or "portrait",
            "brandGuidelines": snapshot.get("brandGuidelines"),
        },
        ad_count=int(snapshot.get("adCount") or 2),
    )


def snapshot_identity_hash(snapshot: Mapping[str, Any]) -> str:
    canonical = {
        "productName": _clean(snapshot.get("productName")),
        "productDescription": _clean(snapshot.get("productDescription")),
        "format": _clean(snapshot.get("format")) or "portrait",
        "adCount": int(snapshot.get("adCount") or 2),
        "brandGuidelines": snapshot.get("brandGuidelines"),
        "requestFingerprint": snapshot_request_fingerprint(snapshot),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))
