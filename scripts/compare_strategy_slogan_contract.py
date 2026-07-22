#!/usr/bin/env python3
"""
Deterministic comparison of final-path strategy/slogan fixtures.

Uses saved mock payloads only — no paid model calls.

Run: python scripts/compare_strategy_slogan_contract.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.builder1_consolidated_stages import (
    process_strategy_stage_response,
    process_strategy_slogan_stage_response,
)
from engine.builder1_strategy_slogan_final import FINAL_SLOGAN_ID, FINAL_STRATEGY_ID
from tests.test_builder1_staged_planning import (
    _strategy_slogan_stage_payload,
    _strategy_stage_payload,
)

BRIEF = "Reinforced shell product for daily carry"


def _single_stage_path(model_caller):
    return process_strategy_stage_response(
        _strategy_stage_payload(),
        product_name="TestBrand",
        product_description=BRIEF,
        model_caller=model_caller,
    )


def _combined_path(model_caller, run_stage):
    return process_strategy_slogan_stage_response(
        _strategy_slogan_stage_payload(),
        product_name="TestBrand",
        product_name_resolved="TestBrand",
        product_description=BRIEF,
        detected_language="en",
        model_caller=model_caller,
        run_stage=run_stage,
    )


def main() -> int:
    from engine.builder1_planner import _run_stage

    single = _single_stage_path(lambda *_a, **_k: {})
    combined = _combined_path(lambda *_a, **_k: {}, _run_stage)

    single_strategy = single[1]
    combined_strategy = combined[1]
    combined_slogan = combined[5]

    checks = {
        "singleFinalPath": single_strategy.id == FINAL_STRATEGY_ID,
        "combinedFinalPath": combined_strategy.id == FINAL_STRATEGY_ID,
        "problemPreserved": single_strategy.strategic_problem == combined_strategy.strategic_problem,
        "relativeAdvantagePreserved": single_strategy.relative_advantage == combined_strategy.relative_advantage,
        "sloganPresent": bool(combined_slogan.brand_slogan),
        "derivationFieldPreserved": bool(combined_slogan.derivation_from_advantage),
        "combinedHasNestedSections": set(_strategy_slogan_stage_payload().keys()) == {"strategy", "slogan"},
        "noCandidateArraysInFixture": "candidates" not in _strategy_slogan_stage_payload()["strategy"],
        "finalSloganId": combined_slogan.id == FINAL_SLOGAN_ID,
    }

    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
