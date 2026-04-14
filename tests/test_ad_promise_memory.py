"""
Unit tests for shared advertisingPromise memory identity and helpers.

Run: python -m unittest tests.test_ad_promise_memory -v
"""

from __future__ import annotations

import unittest

from engine.ad_promise_memory import compute_product_hash, normalize_product_component
from engine.ad_promise_memory_validation import (
    assert_load_save_roundtrip_invariants,
    history_entry_shape_is_product_scoped,
    product_hash_ignores_user_and_session,
)


class TestAdPromiseMemoryIdentity(unittest.TestCase):
    def test_same_product_same_hash_regardless_of_notional_user(self) -> None:
        """B: identity is product text only — same hash for identical product fields."""
        h1 = compute_product_hash("Acme CRM", "Cloud sales tool")
        h2 = compute_product_hash("Acme CRM", "Cloud sales tool")
        self.assertEqual(h1, h2)
        a, b = product_hash_ignores_user_and_session("Acme CRM", "Cloud sales tool")
        self.assertEqual(a, b)

    def test_normalize_stable_for_whitespace(self) -> None:
        n1 = normalize_product_component("  Foo  Bar  ")
        n2 = normalize_product_component("foo bar")
        self.assertEqual(n1, n2)

    def test_history_entry_shape_allows_session_audit_only(self) -> None:
        ok, _ = history_entry_shape_is_product_scoped(
            {
                "promise": "x",
                "created_at": "2026-01-01T00:00:00+00:00",
                "session_id": "job-123",
                "source_type": "video",
            }
        )
        self.assertTrue(ok)

    def test_history_entry_shape_rejects_user_scoping_fields(self) -> None:
        ok, msg = history_entry_shape_is_product_scoped(
            {"promise": "x", "user_id": "u1"},
        )
        self.assertFalse(ok)
        self.assertIn("user_id", msg.lower())

    def test_roundtrip_invariants_helper(self) -> None:
        ok, _ = assert_load_save_roundtrip_invariants(
            [{"promise": "a", "session_id": "s", "source_type": "video"}]
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
