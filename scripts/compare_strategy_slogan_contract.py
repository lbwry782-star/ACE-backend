#!/usr/bin/env python3
"""
Deterministic comparison of legacy separate-stage fixtures vs combined strategy_slogan_stage.

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
    process_slogan_stage_response,
    process_strategy_stage_response,
    process_strategy_slogan_stage_response,
)
from tests.test_builder1_staged_planning import (
    _strategy_slogan_stage_payload,
    _strategy_stage_payload,
    _slogan_stage_payload,
)

BRIEF = "Reinforced shell product for daily carry"


def _legacy_path(model_caller):
    strategy = process_strategy_stage_response(
        _strategy_stage_payload(),
        product_name="TestBrand",
        product_description=BRIEF,
        model_caller=model_caller,
    )
    slogan = process_slogan_stage_response(_slogan_stage_payload())
    return strategy, slogan


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

    legacy = _legacy_path(lambda *_a, **_k: {})
    combined = _combined_path(lambda *_a, **_k: {}, _run_stage)

    legacy_strategy = legacy[0][1]
    legacy_slogan = legacy[1][1]
    combined_strategy = combined[1]
    combined_slogan = combined[5]

    checks = {
        "strategyCandidateCount": len(legacy[0][2]) == len(combined[2]) == 12,
        "sloganCandidateCount": len(legacy[1][2]) == len(combined[6]) == 6,
        "problemPreserved": legacy_strategy.strategic_problem == combined_strategy.strategic_problem,
        "relativeAdvantagePreserved": legacy_strategy.relative_advantage == combined_strategy.relative_advantage,
        "selectedSloganPreserved": legacy_slogan.brand_slogan == combined_slogan.brand_slogan,
        "derivationFieldPreserved": bool(combined_slogan.derivation_from_advantage),
        "combinedHasNestedSections": set(_strategy_slogan_stage_payload().keys()) == {"strategy", "slogan"},
    }

    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
