"""
Builder2 closure copy resolution — trusted persisted product/slogan for re-render.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError


def _clean(value: Any) -> str:
    return str(value or "").strip()


def closure_only_rerender_force_requested() -> bool:
    return _clean(os.environ.get("BUILDER2_CLOSURE_ONLY_RERENDER_FORCE")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_closure_only_rerender_slogan_override(*, state: Dict[str, Any]) -> str:
    """
    Optional operator-provided slogan for closure-only re-render.
    Env wins over persisted mediaResume.closureSloganOverride for one-off corrections.
    """
    env_override = _clean(os.environ.get("BUILDER2_CLOSURE_ONLY_RERENDER_SLOGAN_TEXT"))
    if env_override:
        return env_override
    media = state.get("mediaResume")
    if isinstance(media, dict):
        return _clean(media.get("closureSloganOverride"))
    return ""


def resolve_trusted_closure_copy(
    state: Dict[str, Any],
    *,
    slogan_override: str = "",
) -> Tuple[str, str, str]:
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict):
        raise Builder2TournamentError("builder2_closure_rerender_missing:winnerDevelopmentPlan")
    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict):
        closure = plan.get("advertisingClosure")
    if not isinstance(closure, dict):
        raise Builder2TournamentError("builder2_closure_rerender_missing:advertisingClosure")

    from engine.builder2_single_slogan_contract import resolve_canonical_slogan_text, sync_closure_slogan_from_canonical

    sync_closure_slogan_from_canonical(plan=plan, state=state)
    product_name = _clean(closure.get("productNameText") or plan.get("productNameResolved"))
    override = _clean(slogan_override) or resolve_closure_only_rerender_slogan_override(state=state)
    if override:
        slogan = override
    else:
        slogan = _clean(resolve_canonical_slogan_text(plan=plan, state=state) or closure.get("sloganText"))
    language = _clean(closure.get("language") or state.get("contentLanguage") or plan.get("language") or "he")
    if not product_name:
        raise Builder2TournamentError("builder2_closure_rerender_missing:productNameText")
    if not slogan:
        raise Builder2TournamentError("builder2_closure_rerender_missing:sloganText")
    return product_name, slogan, language


def apply_closure_only_rerender_copy_override(
    state: Dict[str, Any],
    *,
    product_name: str,
    slogan: str,
    language: str,
    override_applied: bool,
) -> None:
    if not override_applied:
        return
    plan = state.get("winnerDevelopmentPlan")
    closure = state.get("advertisingClosure")
    if not isinstance(closure, dict):
        closure = {}
        state["advertisingClosure"] = closure
    closure["productNameText"] = product_name
    closure["sloganText"] = slogan
    closure["language"] = language
    if isinstance(plan, dict):
        plan["productNameResolved"] = product_name
        plan["sloganText"] = slogan
        if isinstance(plan.get("advertisingClosure"), dict):
            plan["advertisingClosure"]["productNameText"] = product_name
            plan["advertisingClosure"]["sloganText"] = slogan
            plan["advertisingClosure"]["language"] = language
    media = state.setdefault("mediaResume", {})
    if isinstance(media, dict):
        media["closureSloganOverride"] = slogan
        from datetime import datetime, timezone

        media["closureCopyCorrectedAt"] = datetime.now(timezone.utc).isoformat()


def closure_copy_fields_present(state: Dict[str, Any]) -> Tuple[bool, bool]:
    try:
        product_name, slogan, _language = resolve_trusted_closure_copy(state)
        return bool(product_name), bool(slogan)
    except Builder2TournamentError:
        return False, False
