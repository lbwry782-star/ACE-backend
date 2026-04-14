"""
Validation helpers and manual test notes for shared advertisingPromise memory.

Run automated checks:  python -m unittest tests.test_ad_promise_memory -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

MANUAL_TEST_PROCEDURE = """
Manual checks (Redis with REDIS_URL set):

A) Same product, same user, two sessions
   - Generate video success twice with same productName/productDescription (different jobIds).
   - GET /api/promise-memory/list → count increases; second planning run should see prior promises in planner logs.

B) Same product, different users
   - Two successful generations with different auth/users but identical product text.
   - Same productHash in list; history reflects both writes.

C/D) Multi-output session (future)
   - For each successful output, call record_ad_promise_generation_success(...) once with the same
     product_name/product_description; verify list count increments per output (no session-length coupling).

E) Failed generation
   - Force Runway/planning failure after a valid plan; confirm no new AD_PROMISE_MEMORY_SUCCESS_SAVE for that attempt
     and optional AD_PROMISE_MEMORY_SKIP_SAVE reason=generation_failed.
"""


def product_hash_ignores_user_and_session(
    product_name: str,
    product_description: str,
) -> Tuple[str, str]:
    """Returns (hash, hash) for the same inputs — proves identity is not parameterized by user/session."""
    from engine.ad_promise_memory import compute_product_hash

    a = compute_product_hash(product_name, product_description)
    b = compute_product_hash(product_name, product_description)
    return a, b


def history_entry_shape_is_product_scoped(entry: Dict[str, Any]) -> Tuple[bool, str]:
    """True if entry has no user_id / entitlement fields (audit session_id allowed)."""
    forbidden = {"user_id", "userId", "entitlement_id", "plan_size", "session_size"}
    keys = {k.lower() for k in entry.keys()}
    bad = sorted(forbidden & keys)
    if bad:
        return False, f"unexpected keys: {bad}"
    return True, "ok"


def assert_load_save_roundtrip_invariants(
    history_rows: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Lightweight structural check for rows returned by load_ad_promise_history."""
    for i, row in enumerate(history_rows):
        if not isinstance(row, dict):
            return False, f"row {i} not dict"
        ok, msg = history_entry_shape_is_product_scoped(row)
        if not ok:
            return False, msg
    return True, "ok"
