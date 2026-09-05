"""
Builder1 post-selection creative brief isolation.

After strategy commits selectedCreativeBrief, downstream creative generation must
operate from that structured brief — not the unrestricted raw productDescription.
Raw description remains available for validation, compliance, and audit only.
"""
from __future__ import annotations

from typing import Optional

from engine.builder1_selected_creative_brief import (
    SelectedCreativeBrief,
    format_selected_creative_brief_block,
)

POST_SELECTION_CREATIVE_STAGES = frozenset(
    {
        "conceptual_stage",
        "brand_physical",
        "graphic_system",
        "series_ads",
    }
)

POST_SELECTION_BRIEF_ISOLATION = """
SELECTION BOUNDARY — POST-SELECTION CREATIVE ISOLATION

Strategy selection commits selectedCreativeBrief. That commitment is a boundary.

Before selection:
- Full raw product information may be used to diagnose, compare, select, and reason.

After selection (this stage and all downstream creative stages):
- Creative generation operates ONLY from selectedCreativeBrief, fixed strategy fields,
  and upstream creative outputs already committed in the pipeline.
- Do NOT reopen, re-read, or infer from the full unrestricted raw productDescription.
- Facts intentionally omitted from selectedCreativeBrief must not re-enter creative reasoning.
- discardedFacts are stored for audit only and must never appear in creative prompt input.

Raw productDescription remains available elsewhere only for validation, compliance,
factual grounding checks, unsupported-claim detection, repair/audit where required,
and deterministic product identity verification — never as an unrestricted creative brief.
""".strip()

POST_SELECTION_USER_PROMPT_NOTICE = (
    "Post-selection creative input (selection boundary — do not use unrestricted raw productDescription):"
)


def format_post_selection_creative_input_block(brief: SelectedCreativeBrief) -> str:
    """Structured creative input after the selection boundary."""
    body = format_selected_creative_brief_block(brief)
    return f"{POST_SELECTION_USER_PROMPT_NOTICE}\n{body}"


def build_creative_stage_information_block(
    *,
    selected_creative_brief: Optional[SelectedCreativeBrief],
    product_description: str,
    legacy_label: str = "Brief",
) -> str:
    """
    Return post-selection brief block when selection is committed; otherwise legacy raw fallback.
    """
    if selected_creative_brief is not None:
        return format_post_selection_creative_input_block(selected_creative_brief)
    return f"{legacy_label}: {product_description}\n"


def raw_product_description_visible_in_prompt(
    prompt: str,
    raw_product_description: str,
) -> bool:
    """
    True when unrestricted raw productDescription appears to have leaked into a
    post-selection creative prompt.
    """
    raw = raw_product_description.strip()
    if not raw:
        return False
    if raw in prompt:
        return True
    markers = (
        f"Brief: {raw}",
        f"Description: {raw}",
        f"Product description:\n{raw}",
        "Product description (raw information — read completely)",
    )
    return any(marker in prompt for marker in markers)
