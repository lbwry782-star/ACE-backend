"""
Builder2 strategy foundation identity — server-owned ID and deterministic digest.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def compute_strategy_foundation_digest(strategy: Dict[str, Any]) -> str:
    """Stable digest over canonical strategic content (excludes identity fields)."""
    canonical = {
        "productNameResolved": strategy.get("productNameResolved"),
        "language": strategy.get("language"),
        "problemPerception": strategy.get("problemPerception"),
        "relativeAdvantage": strategy.get("relativeAdvantage"),
        "mechanismScan": strategy.get("mechanismScan"),
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def assign_strategy_foundation_identity(
    strategy: Dict[str, Any],
    *,
    tournament_id: str,
    existing_id: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(strategy)
    out["strategyFoundationId"] = existing_id or f"{tournament_id}-strategy"
    out["strategyFoundationDigest"] = compute_strategy_foundation_digest(out)
    return out


def expected_strategy_foundation_id(strategy: Dict[str, Any]) -> str:
    return str(strategy.get("strategyFoundationId") or "").strip()
