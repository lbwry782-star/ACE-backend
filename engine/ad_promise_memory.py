"""
Shared persistent advertisingPromise memory (media-agnostic).

Used by the video pipeline today; image (Builder1) integration stays disabled until explicitly wired.

Redis keys:
  ACE:AD_PROMISE_HISTORY:<product_hash> — JSON list of entries (max 40, FIFO), each includes source_type.
  ACE:AD_PROMISE_INDEX                  — JSON map product_hash → listing metadata
  ACE:AD_PROMISE_STATS:<product_hash>   — JSON counters for retention / soft-reset

product_hash = sha256_hex(normalize(name) + "||" + normalize(description))  (64-char hex, deterministic)

Memory identity never uses user id, session id, entitlement, plan size, or output count — only the
normalized product name and description, so all users and all future multi-output sessions share one pool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_HISTORY_ENTRIES = int((os.environ.get("VIDEO_AD_PROMISE_HISTORY_MAX") or "40").strip() or "40")
_MAX_ENTRY_AGE_DAYS = int((os.environ.get("VIDEO_PROMISE_HISTORY_MAX_AGE_DAYS") or "30").strip() or "30")
_SOFT_RESET_KEEP_ENTRIES = int((os.environ.get("VIDEO_PROMISE_SOFT_RESET_KEEP_ENTRIES") or "10").strip() or "10")
_SOFT_RESET_DUP_THRESHOLD = int(
    (os.environ.get("VIDEO_PROMISE_SOFT_RESET_DUPLICATE_THRESHOLD") or "5").strip() or "5"
)
_SOFT_RESET_CONCEPT_THRESHOLD = int(
    (os.environ.get("VIDEO_PROMISE_SOFT_RESET_CONCEPT_THRESHOLD") or "5").strip() or "5"
)
_SOFT_RESET_FALLBACK_THRESHOLD = int(
    (os.environ.get("VIDEO_PROMISE_SOFT_RESET_FALLBACK_THRESHOLD") or "2").strip() or "2"
)
_SOFT_RESET_PLANNING_FAILED_THRESHOLD = int(
    (os.environ.get("VIDEO_PROMISE_SOFT_RESET_PLANNING_FAILED_THRESHOLD") or "2").strip() or "2"
)

_FORBIDDEN_K = int((os.environ.get("VIDEO_AD_PROMISE_FORBIDDEN_K") or "10").strip() or "10")
_SIM_THRESHOLD = float((os.environ.get("VIDEO_AD_PROMISE_SIMILARITY_THRESHOLD") or "0.75").strip() or "0.75")

_HISTORY_KEY_PREFIX = "ACE:AD_PROMISE_HISTORY:"
_INDEX_KEY = "ACE:AD_PROMISE_INDEX"
_STATS_KEY_PREFIX = "ACE:AD_PROMISE_STATS:"

_STAT_FIELDS = (
    "duplicate_rejections",
    "conceptual_match_rejections",
    "fallback_used_count",
    "planning_failed_count",
    "recent_generations_count",
)


def normalize_product_component(raw: str) -> str:
    """
    Per-field normalization: lowercase, trim, collapse spaces, strip punctuation noise,
    keep Hebrew + Latin letters + digits as word chars.
    """
    t = (raw or "").strip().lower()
    t = re.sub(r"[^\w\s\u0590-\u05FF]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_product_hash(product_name: str, product_description: str) -> str:
    """
    Deterministic product_hash from normalized_product_name + '||' + normalized_product_description.
    Returns full sha256 hex (64 chars).
    """
    a = normalize_product_component(product_name)
    b = normalize_product_component(product_description)
    payload = f"{a}||{b}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ad_promise_history_redis_key(product_hash: str) -> str:
    """Exact Redis key for the history list for one product_hash."""
    return f"{_HISTORY_KEY_PREFIX}{(product_hash or '').strip()}"


def product_history_hash(product_name: str, product_description: str) -> str:
    """Backward-compatible alias for compute_product_hash."""
    return compute_product_hash(product_name, product_description)


def _redis_key(product_name: str, product_description: str) -> str:
    return ad_promise_history_redis_key(compute_product_hash(product_name, product_description))


def _redis_get_json(key: str) -> Optional[Any]:
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            return None
        raw = get_redis().get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning("AD_PROMISE_MEMORY_REDIS_FAIL op=get err=%s", e)
        return None


def _redis_set_json(key: str, value: Any) -> bool:
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            return False
        get_redis().set(key, json.dumps(value, ensure_ascii=False))
        return True
    except Exception as e:
        logger.warning("AD_PROMISE_MEMORY_REDIS_FAIL op=set err=%s", e)
        return False


def _stats_redis_key(product_hash: str) -> str:
    return f"{_STATS_KEY_PREFIX}{(product_hash or '').strip()}"


def _default_stats() -> Dict[str, int]:
    return {f: 0 for f in _STAT_FIELDS}


def _normalize_stats_dict(raw: Any) -> Dict[str, int]:
    base = _default_stats()
    if not isinstance(raw, dict):
        return base
    for k in _STAT_FIELDS:
        try:
            v = int(raw.get(k) or 0)
        except Exception:
            v = 0
        base[k] = max(0, v)
    return base


def _entry_created_at_dt(entry: Dict[str, Any]) -> Optional[datetime]:
    s = (entry.get("created_at") or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _cleanup_history_entries(
    hist: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Drop entries older than _MAX_ENTRY_AGE_DAYS (when created_at parses), then trim to _MAX_HISTORY_ENTRIES (FIFO).
    Returns (cleaned oldest→newest, aged_removed, trimmed_removed).
    """
    if not hist:
        return [], 0, 0
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_MAX_ENTRY_AGE_DAYS)
    kept: List[Dict[str, Any]] = []
    aged_removed = 0
    for row in hist:
        if not isinstance(row, dict) or not (row.get("promise") or "").strip():
            continue
        dt = _entry_created_at_dt(row)
        if dt is not None and dt < cutoff:
            aged_removed += 1
            continue
        kept.append(row)
    trimmed_removed = 0
    if len(kept) > _MAX_HISTORY_ENTRIES:
        trimmed_removed = len(kept) - _MAX_HISTORY_ENTRIES
        kept = kept[-_MAX_HISTORY_ENTRIES :]
    if aged_removed or trimmed_removed:
        logger.info(
            "AD_PROMISE_MEMORY_CLEANUP aged_removed=%s trimmed_removed=%s",
            aged_removed,
            trimmed_removed,
        )
    return kept, aged_removed, trimmed_removed


_SOFT_RESET_STAT_FIELDS = frozenset(
    {
        "duplicate_rejections",
        "conceptual_match_rejections",
        "fallback_used_count",
        "planning_failed_count",
    }
)


def _stats_trigger_reason(stats: Dict[str, int]) -> Optional[str]:
    if stats["duplicate_rejections"] >= _SOFT_RESET_DUP_THRESHOLD:
        return f"duplicate_rejections>={_SOFT_RESET_DUP_THRESHOLD}"
    if stats["conceptual_match_rejections"] >= _SOFT_RESET_CONCEPT_THRESHOLD:
        return f"conceptual_match_rejections>={_SOFT_RESET_CONCEPT_THRESHOLD}"
    if stats["fallback_used_count"] >= _SOFT_RESET_FALLBACK_THRESHOLD:
        return f"fallback_used_count>={_SOFT_RESET_FALLBACK_THRESHOLD}"
    if stats["planning_failed_count"] >= _SOFT_RESET_PLANNING_FAILED_THRESHOLD:
        return f"planning_failed_count>={_SOFT_RESET_PLANNING_FAILED_THRESHOLD}"
    return None


def load_promise_stats(product_hash: str) -> Dict[str, int]:
    ph = (product_hash or "").strip()
    if not ph:
        return _default_stats()
    raw = _redis_get_json(_stats_redis_key(ph))
    return _normalize_stats_dict(raw)


def save_promise_stats(product_hash: str, stats: Dict[str, int]) -> bool:
    ph = (product_hash or "").strip()
    if not ph:
        return False
    return _redis_set_json(_stats_redis_key(ph), _normalize_stats_dict(stats))


def reset_promise_stats(product_hash: str) -> None:
    ph = (product_hash or "").strip()
    if not ph:
        return
    save_promise_stats(ph, _default_stats())


def increment_promise_stat(
    product_hash: str,
    field: str,
    delta: int = 1,
    *,
    product_name: str = "",
    product_description: str = "",
) -> None:
    ph = (product_hash or "").strip()
    if not ph or field not in _STAT_FIELDS:
        return
    stats = load_promise_stats(ph)
    stats[field] = max(0, stats[field] + int(delta))
    if not save_promise_stats(ph, stats):
        return
    logger.info(
        "VIDEO_PROMISE_STATS_UPDATED duplicate_rejections=%s conceptual_match_rejections=%s "
        "fallback_used_count=%s planning_failed_count=%s recent_generations_count=%s",
        stats["duplicate_rejections"],
        stats["conceptual_match_rejections"],
        stats["fallback_used_count"],
        stats["planning_failed_count"],
        stats["recent_generations_count"],
    )
    if field in _SOFT_RESET_STAT_FIELDS and _stats_trigger_reason(stats):
        maybe_soft_reset_promise_memory(
            ph, product_name=product_name, product_description=product_description
        )


def maybe_soft_reset_promise_memory(
    product_hash: str,
    *,
    product_name: str = "",
    product_description: str = "",
) -> bool:
    """
    If quality counters exceed thresholds, keep only the last _SOFT_RESET_KEEP_ENTRIES promises
    and reset stats. Returns True if a reset was applied.
    """
    ph = (product_hash or "").strip()
    if not ph:
        return False
    stats = load_promise_stats(ph)
    reason = _stats_trigger_reason(stats)
    if not reason:
        return False
    logger.info("VIDEO_PROMISE_SOFT_RESET_TRIGGERED product_hash=%s reason=%s", ph, reason)
    key = ad_promise_history_redis_key(ph)
    cur = _redis_get_json(key)
    hist: List[Dict[str, Any]] = cur if isinstance(cur, list) else []
    hist = [x for x in hist if isinstance(x, dict)]
    hist, _, _ = _cleanup_history_entries(hist)
    if len(hist) > _SOFT_RESET_KEEP_ENTRIES:
        hist = hist[-_SOFT_RESET_KEEP_ENTRIES :]
    _redis_set_json(key, hist)
    reset_promise_stats(ph)
    idx = _load_promise_index()
    meta = idx.get(ph) if isinstance(idx.get(ph), dict) else {}
    pn = (product_name or str(meta.get("productName") or "")).strip()
    pd = (product_description or str(meta.get("productDescription") or "")).strip()
    _update_promise_index_entry(ph, pn, pd, len(hist))
    logger.info("VIDEO_PROMISE_SOFT_RESET_DONE kept=%s", len(hist))
    return True


def load_ad_promise_history(product_name: str, product_description: str) -> List[Dict[str, Any]]:
    """Return prior entries oldest→newest. Applies age filter + max-size trim; persists if cleanup ran."""
    key = _redis_key(product_name, product_description)
    ph = compute_product_hash(product_name, product_description)
    data = _redis_get_json(key)
    if not isinstance(data, list):
        logger.info("VIDEO_PROMISE_HISTORY_LOADED count=0")
        return []
    raw_list = [x for x in data if isinstance(x, dict)]
    cleaned, aged_removed, trimmed_removed = _cleanup_history_entries(raw_list)
    if aged_removed or trimmed_removed:
        if _redis_set_json(key, cleaned):
            _update_promise_index_entry(ph, product_name, product_description, len(cleaned))
    logger.info("VIDEO_PROMISE_HISTORY_LOADED count=%s", len(cleaned))
    return cleaned


def _tokenize_universal(s: str) -> set:
    return {
        w.lower()
        for w in re.findall(r"[^\W\d_]+", (s or ""), flags=re.UNICODE)
        if len(w) >= 2
    }


def _normalize_promise_compare(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"[^\w\s\u0590-\u05FF]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _char_ngrams(s: str, n: int = 3) -> set:
    t = re.sub(r"\s+", "", _normalize_promise_compare(s))
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


# Lightweight concept buckets (EN + HE) — same bucket overlap raises conceptual similarity.
_CONCEPT_BUCKETS: Tuple[Tuple[str, frozenset], ...] = (
    (
        "speed",
        frozenset(
            {
                "speed",
                "fast",
                "faster",
                "quick",
                "rapid",
                "instant",
                "velocity",
                "accelerate",
                "מהירות",
                "מהיר",
                "מיידי",
            }
        ),
    ),
    (
        "accuracy",
        frozenset(
            {
                "accurate",
                "accuracy",
                "precision",
                "exact",
                "perfect",
                "pixel",
                "בדיוק",
                "דיוק",
                "מדויק",
            }
        ),
    ),
    (
        "security",
        frozenset(
            {
                "secure",
                "security",
                "safe",
                "safety",
                "protect",
                "protection",
                "trust",
                "בטוח",
                "אבטחה",
                "הגנה",
            }
        ),
    ),
    (
        "ease",
        frozenset(
            {
                "easy",
                "easier",
                "simple",
                "simply",
                "effortless",
                "smooth",
                "intuitive",
                "קל",
                "פשוט",
                "נוח",
            }
        ),
    ),
    (
        "growth",
        frozenset(
            {
                "grow",
                "growth",
                "scale",
                "scaling",
                "more",
                "boost",
                "gain",
                "success",
                "צמיחה",
                "הצלחה",
                "רווח",
            }
        ),
    ),
    (
        "cost",
        frozenset(
            {
                "save",
                "saving",
                "cheap",
                "cheaper",
                "affordable",
                "cost",
                "money",
                "value",
                "חיסכון",
                "מחיר",
                "עלות",
            }
        ),
    ),
)


def _concept_bucket_overlap(a: str, b: str) -> Tuple[float, str]:
    ta, tb = _tokenize_universal(a), _tokenize_universal(b)
    bonus = 0.0
    names: List[str] = []
    for name, words in _CONCEPT_BUCKETS:
        if (ta & words) and (tb & words):
            bonus += 0.16
            names.append(name)
    return min(0.48, bonus), ",".join(names) if names else ""


def promise_similarity(candidate: str, previous: str) -> float:
    """
    Heuristic 0..1 similarity (no embeddings): token Jaccard + char n-gram Jaccard + concept buckets.
    """
    ca, pb = _normalize_promise_compare(candidate), _normalize_promise_compare(previous)
    if not ca or not pb:
        return 0.0 if ca != pb else 1.0
    if ca == pb:
        return 1.0
    if len(ca) >= 12 and len(pb) >= 12 and (ca in pb or pb in ca):
        return 0.92

    ta, tb = _tokenize_universal(candidate), _tokenize_universal(previous)
    if not ta and not tb:
        return 1.0 if ca == pb else 0.0
    union = ta | tb
    jw = (len(ta & tb) / len(union)) if union else 0.0

    ga, gb = _char_ngrams(candidate), _char_ngrams(previous)
    union_g = ga | gb
    jc = (len(ga & gb) / len(union_g)) if union_g else 0.0

    cboost, _ = _concept_bucket_overlap(candidate, previous)
    raw = 0.52 * jw + 0.33 * jc + cboost
    return float(max(0.0, min(1.0, raw)))


def max_promise_similarity_vs_list(
    candidate: str, entries: List[Dict[str, Any]]
) -> Tuple[float, Optional[str]]:
    """Max heuristic similarity vs any prior promise (0..1)."""
    best = 0.0
    best_p: Optional[str] = None
    nc = _normalize_promise_compare(candidate)
    for row in entries:
        p = (row.get("promise") or "").strip()
        if not p:
            continue
        if _normalize_promise_compare(p) == nc:
            return 1.0, p
        s = promise_similarity(candidate, p)
        if s > best:
            best = s
            best_p = p
    return best, best_p


def is_promise_too_similar(
    candidate: str, history: List[Dict[str, Any]], session_rejected: List[str]
) -> Tuple[bool, float, str, str]:
    """
    True if candidate should be rejected vs history + in-session rejected list.
    Returns (too_similar, similarity, kind, detail) where detail is e.g. overlapping concept bucket names.
    """
    extra = [{"promise": x} for x in session_rejected if x.strip()]
    combined = list(history) + extra
    sim, prev = max_promise_similarity_vs_list(candidate, combined)
    if sim <= _SIM_THRESHOLD:
        return False, sim, "", ""
    if sim >= 0.98 or (
        prev and _normalize_promise_compare(prev) == _normalize_promise_compare(candidate)
    ):
        return True, sim, "duplicate", ""
    ta = _tokenize_universal(candidate)
    tb = _tokenize_universal(prev or "")
    union = ta | tb
    jw = (len(ta & tb) / len(union)) if union else 0.0
    _, bucket_names = _concept_bucket_overlap(candidate, prev or "")
    if bucket_names and jw < 0.5:
        return True, sim, "concept_match", bucket_names
    return True, sim, "paraphrase", ""


def forbidden_promises_for_prompt(history: List[Dict[str, Any]], k: int) -> List[str]:
    """Last k non-empty promises from history (most recent)."""
    texts: List[str] = []
    for row in history[-k:]:
        p = (row.get("promise") or "").strip()
        if p:
            texts.append(p)
    return texts


def build_promise_diversity_addon(
    forbidden: List[str],
    angle_instruction: str = "",
) -> str:
    """Appended to planner input: hard rule + forbidden list + optional angle seed."""
    lines: List[str] = []
    if forbidden:
        lines.append("ADVERTISING PROMISE — CLIENT MEMORY (hard constraints):")
        lines.append(
            "The advertisingPromise MUST be materially different from every line below: "
            "new core idea, not a synonym, not the same benefit reframed, not the same metaphor family."
        )
        lines.append("Previously used promises for this product (do NOT restate or lightly paraphrase):")
        for i, p in enumerate(forbidden[:_FORBIDDEN_K], 1):
            short = (p or "").replace("\n", " ").strip()
            if len(short) > 240:
                short = short[:237] + "…"
            lines.append(f"  {i}. «{short}»")
        lines.append(
            "Do not recycle speed, accuracy, security, ease, savings, or growth as the headline story "
            "if that theme already appears above — pick a genuinely different angle."
        )
    if angle_instruction.strip():
        lines.append(angle_instruction.strip())
    if not lines:
        return ""
    return "\n\n" + "\n".join(lines) + "\n"


_ANGLE_SEEDS = (
    "MANDATORY_ANGLE_SHIFT: Lead with an emotional benefit or relief the buyer feels (not specs or adjectives alone).",
    "MANDATORY_ANGLE_SHIFT: Lead with a concrete business outcome a team can see or measure after adopting the product.",
    "MANDATORY_ANGLE_SHIFT: Lead with user transformation — who the customer becomes, in plain human terms, without repeating earlier themes.",
    "MANDATORY_ANGLE_SHIFT: Use one fresh metaphor domain (nature, craft, travel, sport, food) that none of the forbidden lines already used.",
)


def angle_seed_for_attempt(attempt_index: int, promise_reject_count: int) -> str:
    """Retry planner calls get a programmatic angle shift; stronger mix after promise rejects."""
    if attempt_index <= 0:
        return ""
    base = _ANGLE_SEEDS[(attempt_index - 1) % len(_ANGLE_SEEDS)]
    if promise_reject_count >= 1:
        extra = _ANGLE_SEEDS[promise_reject_count % len(_ANGLE_SEEDS)]
        return base + "\n" + extra
    return base


def _normalize_source_type(source_type: str) -> str:
    s = (source_type or "").strip().lower()
    return s if s in ("video", "image") else "video"


def record_ad_promise_generation_success(
    product_name: str,
    product_description: str,
    advertising_promise: str,
    *,
    job_or_trace_id: str = "",
    source_type: str = "video",
) -> None:
    """
    Persist one successful output's promise (call once per successful generation, not once per checkout).

    Product identity uses only name + description; job_or_trace_id is audit metadata only.
    """
    save_ad_promise_entry(
        product_name,
        product_description,
        advertising_promise,
        job_or_trace_id,
        source_type=source_type,
    )


def save_ad_promise_entry(
    product_name: str,
    product_description: str,
    promise: str,
    session_id: str,
    *,
    source_type: str = "video",
) -> None:
    """Append one entry; age-cleanup + trim to _MAX_HISTORY_ENTRIES (FIFO); no-op if Redis unavailable."""
    p = (promise or "").strip()
    if not p:
        return
    st = _normalize_source_type(source_type)
    key = _redis_key(product_name, product_description)
    ph = compute_product_hash(product_name, product_description)
    cur = _redis_get_json(key)
    hist: List[Dict[str, Any]] = cur if isinstance(cur, list) else []
    hist = [x for x in hist if isinstance(x, dict)]
    hist, _, _ = _cleanup_history_entries(hist)
    entry = {
        "promise": p,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": (session_id or "").strip() or "unknown",
        "source_type": st,
        "embedding": None,
    }
    hist.append(entry)
    if len(hist) > _MAX_HISTORY_ENTRIES:
        removed = len(hist) - _MAX_HISTORY_ENTRIES
        hist = hist[-_MAX_HISTORY_ENTRIES :]
        logger.info(
            "AD_PROMISE_MEMORY_CLEANUP aged_removed=%s trimmed_removed=%s",
            0,
            removed,
        )
    if _redis_set_json(key, hist):
        logger.info("AD_PROMISE_MEMORY_SUCCESS_SAVE hash=%s source_type=%s", ph, st)
        logger.info("VIDEO_PROMISE_HISTORY_SAVED size=%s", len(hist))
        _update_promise_index_entry(
            ph,
            product_name,
            product_description,
            len(hist),
        )
    else:
        logger.info("VIDEO_PROMISE_HISTORY_SAVED size=skipped")


def _load_promise_index() -> Dict[str, Any]:
    data = _redis_get_json(_INDEX_KEY)
    return data if isinstance(data, dict) else {}


def _save_promise_index(idx: Dict[str, Any]) -> bool:
    ok = _redis_set_json(_INDEX_KEY, idx)
    if ok:
        logger.info("VIDEO_PROMISE_INDEX_UPDATED size=%s", len(idx))
    return ok


def _update_promise_index_entry(
    product_hash: str,
    product_name: str,
    product_description: str,
    history_len: int,
) -> None:
    ph = (product_hash or "").strip()
    if not ph:
        return
    idx = _load_promise_index()
    idx[ph] = {
        "productName": product_name or "",
        "productDescription": product_description or "",
        "last_used": datetime.now(timezone.utc).isoformat(),
        "count": int(history_len),
    }
    _save_promise_index(idx)


def _distinct_sources_from_history(hist: List[Dict[str, Any]]) -> List[str]:
    """Derive media sources present in stored history (legacy rows without source_type count as video)."""
    found: set[str] = set()
    for row in hist:
        if not isinstance(row, dict):
            continue
        st_raw = (row.get("source_type") or "").strip().lower()
        if st_raw == "image":
            found.add("image")
        else:
            found.add("video")
    return sorted(found)


def get_all_products_with_memory() -> List[Dict[str, Any]]:
    """
    List products that appear in ACE:AD_PROMISE_INDEX and still have a history key.
    Drops index entries whose history key is missing (self-heal).
    """
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            logger.info("AD_PROMISE_MEMORY_INDEX_LIST count=0")
            return []
        r = get_redis()
        idx = _load_promise_index()
        if not idx:
            logger.info("AD_PROMISE_MEMORY_INDEX_LIST count=0")
            return []
        out: List[Dict[str, Any]] = []
        dirty = False
        for h, meta in list(idx.items()):
            if not isinstance(meta, dict):
                dirty = True
                del idx[h]
                continue
            hkey = ad_promise_history_redis_key(h)
            if not r.exists(hkey):
                dirty = True
                del idx[h]
                continue
            data = _redis_get_json(hkey)
            hist_list = data if isinstance(data, list) else []
            cnt = len(hist_list) if hist_list else int(meta.get("count") or 0)
            sources = _distinct_sources_from_history(
                [x for x in hist_list if isinstance(x, dict)]
            )
            if not sources:
                sources = ["video"]
            out.append(
                {
                    "productHash": h,
                    "productName": str(meta.get("productName") or ""),
                    "productDescription": str(meta.get("productDescription") or ""),
                    "count": cnt,
                    "sources": sources,
                    "lastUsed": str(meta.get("last_used") or ""),
                }
            )
        if dirty:
            _save_promise_index(idx)
        out.sort(key=lambda row: row.get("productHash") or "")
        logger.info("AD_PROMISE_MEMORY_INDEX_LIST count=%s", len(out))
        return out
    except Exception as e:
        logger.warning("VIDEO_PROMISE_INDEX_LOAD_FAIL err=%s", e)
        logger.info("AD_PROMISE_MEMORY_INDEX_LIST count=0")
        return []


def delete_product_memory_by_hash(product_hash: str) -> bool:
    """
    Delete ACE:AD_PROMISE_HISTORY:<product_hash> and remove that hash from ACE:AD_PROMISE_INDEX.
    """
    ph = (product_hash or "").strip().lower()
    if not ph:
        return False
    logger.info("AD_PROMISE_MEMORY_DELETE_BY_HASH hash=%s", ph)
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            return False
        r = get_redis()
        r.delete(ad_promise_history_redis_key(ph))
        r.delete(_stats_redis_key(ph))
        idx = _load_promise_index()
        if ph in idx:
            del idx[ph]
            _save_promise_index(idx)
        return True
    except Exception as e:
        logger.error("VIDEO_PROMISE_MEMORY_DELETE_BY_HASH_FAIL hash=%s err=%s", ph, e)
        return False


def delete_product_memory_by_text(product_name: str, product_description: str) -> Tuple[bool, str]:
    """Recompute product_hash and delete history + index entry."""
    ph = compute_product_hash(product_name, product_description)
    logger.info("AD_PROMISE_MEMORY_DELETE_BY_TEXT hash=%s", ph)
    ok = delete_product_memory_by_hash(ph)
    return ok, ph


def clear_ad_promise_history(product_name: str, product_description: str) -> Tuple[bool, str]:
    """
    Admin helper: same as delete_product_memory_by_text, plus legacy CLEAR_* logs.
    Returns (ok, product_hash_hex).
    """
    ph = compute_product_hash(product_name, product_description)
    logger.info("VIDEO_PROMISE_HISTORY_CLEAR_REQUEST product_hash=%s", ph)
    ok, ph2 = delete_product_memory_by_text(product_name, product_description)
    if ok:
        logger.info("VIDEO_PROMISE_HISTORY_CLEAR_SUCCESS product_hash=%s", ph2)
        logger.info("VIDEO_PROMISE_HISTORY_CLEARED product_hash=%s", ph2)
    else:
        logger.warning(
            "VIDEO_PROMISE_HISTORY_CLEAR_SKIP redis_unconfigured product_hash=%s",
            ph2,
        )
    return ok, ph2


def clear_all_ad_promise_history() -> int:
    """
    Delete every key matching ACE:AD_PROMISE_HISTORY:* using SCAN (non-blocking),
    then reset ACE:AD_PROMISE_INDEX to {}.
    Returns count of history keys removed, or -1 on fatal error.
    """
    logger.info("VIDEO_PROMISE_HISTORY_GLOBAL_RESET_REQUEST")
    try:
        from engine.video_jobs_redis import get_redis, redis_configured

        if not redis_configured():
            logger.warning("VIDEO_PROMISE_HISTORY_GLOBAL_RESET_SKIP redis_unconfigured")
            return 0
        r = get_redis()
        pattern = f"{_HISTORY_KEY_PREFIX}*"
        cursor = 0
        total_deleted = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                total_deleted += int(r.delete(*keys))
            if cursor == 0:
                break
        stats_pattern = f"{_STATS_KEY_PREFIX}*"
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=stats_pattern, count=200)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break
        _redis_set_json(_INDEX_KEY, {})
        logger.info("VIDEO_PROMISE_INDEX_UPDATED size=0")
        logger.info("VIDEO_PROMISE_HISTORY_GLOBAL_RESET total_deleted=%s", total_deleted)
        logger.info("VIDEO_PROMISE_HISTORY_GLOBAL_RESET_SUCCESS total_deleted=%s", total_deleted)
        return total_deleted
    except Exception as e:
        logger.error("VIDEO_PROMISE_HISTORY_GLOBAL_RESET_FAIL err=%s", e, exc_info=True)
        return -1


def clear_all_ad_promise_memory() -> int:
    """Delete all shared ACE:AD_PROMISE_HISTORY:* keys, stats keys, and reset the index."""
    return clear_all_ad_promise_history()


logger.info("AD_PROMISE_MEMORY_SHARED_MODULE_LOADED=true")
logger.info("AD_PROMISE_IMAGE_INTEGRATION_ENABLED=false")
