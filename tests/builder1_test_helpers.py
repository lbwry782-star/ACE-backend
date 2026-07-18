"""Shared Builder1 test helpers."""

from __future__ import annotations

from typing import Any, Dict

from engine.builder1_image_compliance import ImageComplianceResult


def pass_compliance_reviewer(**_kwargs: Any) -> ImageComplianceResult:
    """Explicit test-only injected reviewer — approves without a real multimodal call."""
    return ImageComplianceResult(passed=True, violations=[], confidence="high")


def marketing_text_words(count: int = 50, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(1, count + 1))


def marketing_text_hebrew(count: int = 50) -> str:
    return " ".join(f"מילה{i}" for i in range(1, count + 1))


def marketing_text_hebrew_with_brand(count: int = 50, brand: str = "TestBrand") -> str:
    words: list[str] = []
    brand_at = max(1, count // 2)
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"מילה{word_num}")
            word_num += 1
    return " ".join(words)


def marketing_text_english_with_hebrew_brand(count: int = 50, brand: str = "מותג") -> str:
    words: list[str] = []
    brand_at = 5
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"word{word_num}")
            word_num += 1
    return " ".join(words)


DEFAULT_STRATEGY_BOUNDARY_FIELDS: Dict[str, Any] = {
    "campaignExecutableNow": True,
    "requiresClientConsultation": False,
    "clientActionLevel": "none",
    "implementationCostLevel": "none",
    "simpleStrategicAction": None,
}


def strategy_scan_candidate(
    *,
    index: int,
    lens: str,
    problem: str | None = None,
    advantage: str | None = None,
    brief_support: str = "Follows from brief reinforced shell mention",
    **boundary_overrides: Any,
) -> Dict[str, Any]:
    candidate = {
        "id": f"S{index:02d}",
        "lens": lens,
        "strategicProblem": problem or f"Distinct buyer problem {index}",
        "relativeAdvantage": advantage or f"Distinct advantage {index}",
        "briefSupport": brief_support,
        "advantageSource": "observable_product_mechanism",
        "claimRisk": "low",
        **DEFAULT_STRATEGY_BOUNDARY_FIELDS,
    }
    candidate.update(boundary_overrides)
    return candidate


def marketing_text_with_punctuation() -> str:
    words = [f"word{i}" for i in range(1, 50)]
    words[0] = "Hello,"
    words.append("finished.")
    return " ".join(words)
