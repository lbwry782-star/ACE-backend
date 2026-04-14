"""
Backward-compatible import path for advertisingPromise memory.

Implementation: engine.ad_promise_memory (shared, media-agnostic). Prefer importing that module in new code.
"""

from __future__ import annotations

from engine.ad_promise_memory import (  # noqa: F401
    ad_promise_history_redis_key,
    angle_seed_for_attempt,
    build_promise_diversity_addon,
    clear_ad_promise_history,
    clear_all_ad_promise_history,
    clear_all_ad_promise_memory,
    compute_product_hash,
    delete_product_memory_by_hash,
    delete_product_memory_by_text,
    forbidden_promises_for_prompt,
    get_all_products_with_memory,
    increment_promise_stat,
    is_promise_too_similar,
    load_ad_promise_history,
    load_promise_stats,
    maybe_soft_reset_promise_memory,
    max_promise_similarity_vs_list,
    normalize_product_component,
    product_history_hash,
    promise_similarity,
    record_ad_promise_generation_success,
    reset_promise_stats,
    save_ad_promise_entry,
    save_promise_stats,
)
