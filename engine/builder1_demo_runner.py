"""
Tiny runner for the disconnected Builder1 demo flow.
"""
from __future__ import annotations

from engine.builder1_generate_demo import build_demo_ad
from engine.builder1_generate_flow import Builder1GenerateResult


def run_demo() -> Builder1GenerateResult:
    return build_demo_ad()
