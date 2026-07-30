"""
Builder2 closure-only re-render preflight inspector — read-only.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from engine.builder2_closure_copy import closure_copy_fields_present
from engine.builder2_closure_typography import (
    BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    CLOSURE_BACKGROUND_STYLE_VERSION,
    CLOSURE_TEXT_REVEAL_VERSION,
    closure_typography_upgrade_needed,
    current_closure_typography_version,
    resolve_builder2_closure_product_font_path,
    resolve_builder2_closure_slogan_font_path,
)
from engine.builder2_closure_duration_contract import build_closure_duration_inspector_fields
from engine.builder2_final_output_diagnostics import (
    durable_final_url_present,
    is_builder2_media_diagnostically_completed,
)
from engine.builder2_media_finalization_contract import resolve_raw_runway_artifact_url
from engine.builder2_tournament_store import load_tournament_state


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def inspect_builder2_closure_rerender(
    state: Dict[str, Any],
    *,
    requested_typography_version: str = BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
) -> Dict[str, Any]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    current_version = current_closure_typography_version(media)
    upgrade_needed = closure_typography_upgrade_needed(media, requested_version=requested_typography_version)
    duration_fields = build_closure_duration_inspector_fields(
        state,
        requested_typography_version=requested_typography_version,
    )
    missing: List[str] = []
    raw_present = bool(resolve_raw_runway_artifact_url(state))
    media_completed = is_builder2_media_diagnostically_completed(state)
    closure_present = bool(_clean(media.get("finalVideoWithClosureUrl")))
    durable_present = durable_final_url_present(state)
    if not media_completed:
        missing.append("mediaCompleted")
    if not raw_present:
        missing.append("rawRunwayVideo")
    product_font_present = False
    slogan_font_present = False
    product_font_path = ""
    slogan_font_path = ""
    try:
        product_font_path = str(resolve_builder2_closure_product_font_path())
        product_font_present = True
    except Exception:
        missing.append("productFont")
    try:
        slogan_font_path = str(resolve_builder2_closure_slogan_font_path())
        slogan_font_present = True
    except Exception:
        missing.append("sloganFont")
    product_present, slogan_present = closure_copy_fields_present(state)
    if not product_present:
        missing.append("canonicalProductName")
    if not slogan_present:
        missing.append("canonicalSlogan")
    if not upgrade_needed:
        missing.append("typographyAlreadyCurrent")
    eligible = not missing
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "mediaCompleted": media_completed,
        "rawRunwayVideoPresent": raw_present,
        "currentClosureVideoPresent": closure_present,
        "durableFinalUrlPresent": durable_present,
        "canonicalProductNamePresent": product_present,
        "canonicalSloganPresent": slogan_present,
        "productFontPresent": product_font_present,
        "sloganFontPresent": slogan_font_present,
        "productFontPathResolved": product_font_path,
        "sloganFontPathResolved": slogan_font_path,
        "currentTypographyContractVersion": current_version or None,
        "requestedTypographyContractVersion": requested_typography_version,
        "closureBackgroundStyleVersion": CLOSURE_BACKGROUND_STYLE_VERSION,
        "closureTextRevealVersion": CLOSURE_TEXT_REVEAL_VERSION,
        "typographyUpgradeNeeded": upgrade_needed,
        "closureOnlyRerenderEligible": eligible,
        "closureOnlyRerenderMissingFields": missing,
        "runwaySubmissionRequired": False,
        "imageGenerationRequired": False,
        "reasoningCallRequired": False,
        "stateMutated": False,
        "paidCalls": 0,
        **duration_fields,
    }


def main() -> int:
    job_id = _env("BUILDER2_CLOSURE_RERENDER_INSPECT_JOB_ID")
    if not job_id:
        print("BUILDER2_CLOSURE_RERENDER_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    requested = _env(
        "BUILDER2_CLOSURE_ONLY_RERENDER_EXPECTED_VERSION",
        BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
    )
    state = load_tournament_state(job_id, read_only=True)
    if not isinstance(state, dict) or not state:
        print(json.dumps({"jobId": job_id, "error": "tournament_state_missing"}))
        return 1
    report = inspect_builder2_closure_rerender(state, requested_typography_version=requested)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
