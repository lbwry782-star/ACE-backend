"""
Builder2 closure copy resolution — trusted persisted product/slogan for re-render.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError


def _clean(value: Any) -> str:
    return str(value or "").strip()


def resolve_trusted_closure_copy(state: Dict[str, Any]) -> Tuple[str, str, str]:
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
    slogan = _clean(resolve_canonical_slogan_text(plan=plan, state=state) or closure.get("sloganText"))
    language = _clean(closure.get("language") or state.get("contentLanguage") or plan.get("language") or "he")
    if not product_name:
        raise Builder2TournamentError("builder2_closure_rerender_missing:productNameText")
    if not slogan:
        raise Builder2TournamentError("builder2_closure_rerender_missing:sloganText")
    return product_name, slogan, language


def closure_copy_fields_present(state: Dict[str, Any]) -> Tuple[bool, bool]:
    try:
        product_name, slogan, _language = resolve_trusted_closure_copy(state)
        return bool(product_name), bool(slogan)
    except Builder2TournamentError:
        return False, False
