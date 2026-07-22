"""
Builder1 global creative-idea memory — one FIFO Redis store for all users/products.

Retains up to BUILDER1_IDEA_MEMORY_MAX_ADS ad-level creative records. Not image bytes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from engine.builder1_plan_spec import Builder1SeriesPlan, graphic_generator_to_dict
from engine.builder1_series_distinctness import (
    CORE_EXECUTION_FIELDS,
    count_distinct_dimensions,
    execution_dimension_values,
    execution_fingerprint,
    fingerprint_hash,
    normalize_execution_text,
)

logger = logging.getLogger(__name__)

BUILDER1_IDEA_MEMORY_MAX_ADS = 200
SCHEMA_VERSION = 1
MEMORY_SCOPE_GLOBAL_IDEA = "global_idea"
MEMORY_SCOPE_GLOBAL_PRODUCT = MEMORY_SCOPE_GLOBAL_IDEA  # legacy alias

_KEY_PREFIX = "builder1:idea-memory:v3:global"
_LEGACY_KEY_PREFIX = "builder1:idea-memory"
_V2_KEY_PREFIX = "builder1:idea-memory:v2:global"
_RECORDS_KEY = f"{_KEY_PREFIX}:records"
_RECORD_IDS_KEY = f"{_KEY_PREFIX}:record-ids"
_BACKFILL_FLAG_KEY = f"{_KEY_PREFIX}:backfill-v1"
_MIGRATE_V1_FLAG_KEY = f"{_KEY_PREFIX}:migrate-v1"
_CAMPAIGN_INDEX_KEY = f"{_KEY_PREFIX}:campaign-index"

_memory_lock = threading.Lock()
_memory_fallback: Dict[str, Any] = {}

_CAMPAIGN_IDEA_FIELDS: Tuple[str, ...] = (
    "strategicProblem",
    "relativeAdvantage",
    "conceptualGenerator",
    "physicalGenerator",
    "transferredObject",
    "transferredObjectAction",
)

_EXECUTION_FP_FIELDS: Tuple[str, ...] = (
    "conceptualExecution",
    "physicalExecution",
    "visualExecution",
    *CORE_EXECUTION_FIELDS,
)

_GENERIC_BOILERPLATE = frozenset(
    {
        "show product",
        "highlight benefit",
        "brand awareness",
        "quality service",
        "innovative solution",
    }
)

GLOBAL_IDEA_MEMORY_SCOPE = None  # initialized after IdeaMemoryScope definition


@dataclass(frozen=True)
class IdeaMemoryScope:
    memory_scope: str = MEMORY_SCOPE_GLOBAL_IDEA


GLOBAL_IDEA_MEMORY_SCOPE = IdeaMemoryScope()


@dataclass
class IdeaMemoryRecord:
    schema_version: int
    record_id: str
    campaign_id: str
    ad_index: int
    created_at: str
    strategic_problem: str
    relative_advantage: str
    slogan: str
    conceptual_generator: str
    physical_generator: str
    transferred_object: str
    transferred_object_action: str
    graphic_summary: str
    conceptual_execution: str
    physical_execution: str
    visual_execution: str
    scene_description: str
    execution_subject: str
    execution_action: str
    execution_object_state: str
    execution_scene: str
    execution_punchline: str
    campaign_idea_fingerprint: str
    ad_execution_fingerprint: str
    product_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schemaVersion": self.schema_version,
            "recordId": self.record_id,
            "campaignId": self.campaign_id,
            "adIndex": self.ad_index,
            "createdAt": self.created_at,
            "strategicProblem": self.strategic_problem,
            "relativeAdvantage": self.relative_advantage,
            "slogan": self.slogan,
            "conceptualGenerator": self.conceptual_generator,
            "physicalGenerator": self.physical_generator,
            "transferredObject": self.transferred_object,
            "transferredObjectAction": self.transferred_object_action,
            "graphicSummary": self.graphic_summary,
            "conceptualExecution": self.conceptual_execution,
            "physicalExecution": self.physical_execution,
            "visualExecution": self.visual_execution,
            "sceneDescription": self.scene_description,
            "executionSubject": self.execution_subject,
            "executionAction": self.execution_action,
            "executionObjectState": self.execution_object_state,
            "executionScene": self.execution_scene,
            "executionPunchline": self.execution_punchline,
            "campaignIdeaFingerprint": self.campaign_idea_fingerprint,
            "adExecutionFingerprint": self.ad_execution_fingerprint,
        }
        if self.product_name:
            payload["productName"] = self.product_name
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Optional["IdeaMemoryRecord"]:
        record_id = str(raw.get("recordId") or "").strip()
        campaign_id = str(raw.get("campaignId") or "").strip()
        if not record_id or not campaign_id:
            logger.warning("BUILDER1_IDEA_MEMORY_MIGRATE_SKIP reason=malformed_record")
            return None
        try:
            ad_index = int(raw.get("adIndex"))
        except (TypeError, ValueError):
            logger.warning("BUILDER1_IDEA_MEMORY_MIGRATE_SKIP recordId=%s reason=malformed_ad_index", record_id)
            return None
        return cls(
            schema_version=int(raw.get("schemaVersion") or SCHEMA_VERSION),
            record_id=record_id,
            campaign_id=campaign_id,
            ad_index=ad_index,
            created_at=str(raw.get("createdAt") or ""),
            strategic_problem=str(raw.get("strategicProblem") or ""),
            relative_advantage=str(raw.get("relativeAdvantage") or ""),
            slogan=str(raw.get("slogan") or ""),
            conceptual_generator=str(raw.get("conceptualGenerator") or ""),
            physical_generator=str(raw.get("physicalGenerator") or ""),
            transferred_object=str(raw.get("transferredObject") or ""),
            transferred_object_action=str(raw.get("transferredObjectAction") or ""),
            graphic_summary=str(raw.get("graphicSummary") or ""),
            conceptual_execution=str(raw.get("conceptualExecution") or ""),
            physical_execution=str(raw.get("physicalExecution") or ""),
            visual_execution=str(raw.get("visualExecution") or ""),
            scene_description=str(raw.get("sceneDescription") or ""),
            execution_subject=str(raw.get("executionSubject") or ""),
            execution_action=str(raw.get("executionAction") or ""),
            execution_object_state=str(raw.get("executionObjectState") or ""),
            execution_scene=str(raw.get("executionScene") or ""),
            execution_punchline=str(raw.get("executionPunchline") or ""),
            campaign_idea_fingerprint=str(raw.get("campaignIdeaFingerprint") or ""),
            ad_execution_fingerprint=str(raw.get("adExecutionFingerprint") or ""),
            product_name=str(raw.get("productName") or ""),
        )


@dataclass
class IdeaMemorySnapshot:
    scope: IdeaMemoryScope
    records: List[IdeaMemoryRecord] = field(default_factory=list)
    backfill_status: str = "none"

    def historical_records(self, *, exclude_campaign_id: str = "") -> List[IdeaMemoryRecord]:
        excluded = (exclude_campaign_id or "").strip()
        if not excluded:
            return list(self.records)
        return [r for r in self.records if r.campaign_id != excluded]


@dataclass(frozen=True)
class HistoricalDuplicateFinding:
    stage: str
    duplicate_type: str
    matching_record_id: str
    matching_campaign_id: str
    fingerprint: str
    ad_index: Optional[int] = None


@dataclass(frozen=True)
class IdeaMemoryWriteResult:
    added_count: int
    skipped_idempotent_count: int
    count_after: int
    evicted_count: int


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def idea_memory_active() -> bool:
    if (os.environ.get("BUILDER1_IDEA_MEMORY_DISABLED") or "").strip() == "1":
        return False
    if _redis_configured():
        return True
    return (os.environ.get("BUILDER1_IDEA_MEMORY_FORCE") or "").strip() == "1"


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def _records_key(scope: Optional[IdeaMemoryScope] = None) -> str:
    del scope
    return _RECORDS_KEY


def _record_ids_key(scope: Optional[IdeaMemoryScope] = None) -> str:
    del scope
    return _RECORD_IDS_KEY


def _campaign_meta_key(campaign_id: str) -> str:
    return f"{_KEY_PREFIX}:campaign-meta:{campaign_id.strip()}"


def normalize_product_scope_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\.,;:!?\u05BE\u05C0\u05C3\"'`()\[\]{}«»„”“‘’\-–—/\\|]+", " ", text)
    text = " ".join(text.split())
    if text.isascii():
        text = text.casefold()
    return text


def resolve_idea_memory_scope(
    *,
    user_product_name: str = "",
    user_product_description: str = "",
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> IdeaMemoryScope:
    del user_product_name, user_product_description, brand_guidelines
    return GLOBAL_IDEA_MEMORY_SCOPE


def _graphic_summary(plan: Builder1SeriesPlan) -> str:
    graphic = graphic_generator_to_dict(plan.graphic_generator)
    device = str(graphic.get("recurringGraphicDevice") or "").strip()
    layout = str(graphic.get("layoutTemplate") or "").strip()
    if device and layout:
        return f"{device}; layout={layout}"
    return device or layout or ""


def _campaign_idea_payload(
    *,
    strategic_problem: str,
    relative_advantage: str,
    conceptual_generator: str,
    physical_generator: str,
    transferred_object: str,
    transferred_object_action: str,
    slogan: str = "",
) -> Dict[str, str]:
    del slogan
    return {
        "strategicProblem": normalize_execution_text(strategic_problem),
        "relativeAdvantage": normalize_execution_text(relative_advantage),
        "conceptualGenerator": normalize_execution_text(conceptual_generator),
        "physicalGenerator": normalize_execution_text(physical_generator),
        "transferredObject": normalize_execution_text(transferred_object),
        "transferredObjectAction": normalize_execution_text(transferred_object_action),
    }


def compute_campaign_idea_fingerprint(payload: Mapping[str, str]) -> str:
    filtered = {
        key: value
        for key, value in payload.items()
        if value and value not in _GENERIC_BOILERPLATE
    }
    if len(filtered) < 3:
        return ""
    fp = tuple(sorted(filtered.items()))
    return fingerprint_hash(fp)


def compute_ad_execution_fingerprint(ad: Mapping[str, Any]) -> str:
    fp = execution_fingerprint(ad, _EXECUTION_FP_FIELDS)
    if len(fp) < 2:
        return ""
    return fingerprint_hash(fp)


def _record_execution_ad(record: IdeaMemoryRecord) -> Dict[str, str]:
    return {
        "conceptualExecution": record.conceptual_execution,
        "physicalExecution": record.physical_execution,
        "visualExecution": record.visual_execution,
        "executionSubject": record.execution_subject,
        "executionAction": record.execution_action,
        "executionObjectState": record.execution_object_state,
        "executionScene": record.execution_scene,
        "executionPunchline": record.execution_punchline,
    }


def _campaign_mechanism_overlap(
    *,
    record: IdeaMemoryRecord,
    conceptual_generator: str = "",
    physical_generator: str = "",
    transferred_object: str = "",
    transferred_object_action: str = "",
) -> bool:
    proposed = {
        "conceptualGenerator": normalize_execution_text(conceptual_generator),
        "physicalGenerator": normalize_execution_text(physical_generator),
        "transferredObject": normalize_execution_text(transferred_object),
        "transferredObjectAction": normalize_execution_text(transferred_object_action),
    }
    historical = {
        "conceptualGenerator": normalize_execution_text(record.conceptual_generator),
        "physicalGenerator": normalize_execution_text(record.physical_generator),
        "transferredObject": normalize_execution_text(record.transferred_object),
        "transferredObjectAction": normalize_execution_text(record.transferred_object_action),
    }
    matches = 0
    compared = 0
    for field in ("conceptualGenerator", "physicalGenerator", "transferredObject", "transferredObjectAction"):
        left = proposed.get(field, "")
        right = historical.get(field, "")
        if not left or not right:
            continue
        compared += 1
        if left == right:
            matches += 1
    return compared >= 2 and matches >= max(2, compared - 1)


def _execution_semantically_duplicate(proposed: Mapping[str, Any], record: IdeaMemoryRecord) -> bool:
    if (
        record.ad_execution_fingerprint
        and proposed.get("_executionFingerprint") == record.ad_execution_fingerprint
    ):
        return True
    record_ad = _record_execution_ad(record)
    diff_count = count_distinct_dimensions(proposed, record_ad, _EXECUTION_FP_FIELDS)
    if diff_count == 0:
        return True
    if diff_count == 1:
        values_a = execution_dimension_values(proposed, _EXECUTION_FP_FIELDS)
        values_b = execution_dimension_values(record_ad, _EXECUTION_FP_FIELDS)
        differing = [
            field
            for field in sorted(set(values_a) | set(values_b))
            if values_a.get(field, "") != values_b.get(field, "")
        ]
        if differing == ["executionSubject"]:
            return True
    return False


def build_records_from_plan(
    plan: Builder1SeriesPlan,
    *,
    scope: Optional[IdeaMemoryScope] = None,
    campaign_id: str,
) -> List[IdeaMemoryRecord]:
    del scope
    created_at = datetime.now(timezone.utc).isoformat()
    idea_payload = _campaign_idea_payload(
        strategic_problem=plan.strategic_problem,
        relative_advantage=plan.relative_advantage,
        slogan=plan.brand_slogan,
        conceptual_generator=plan.conceptual_generator,
        physical_generator=plan.physical_generator,
        transferred_object=plan.transferred_object,
        transferred_object_action=plan.transferred_object_action,
    )
    campaign_fp = compute_campaign_idea_fingerprint(idea_payload)
    graphic_summary = _graphic_summary(plan)
    product_name = normalize_product_scope_text(plan.product_name or plan.product_name_resolved)
    internals = (plan.planning_internals or {}).get("adInternals") or {}
    records: List[IdeaMemoryRecord] = []
    for ad in plan.ads:
        ad_dict: Dict[str, Any] = {
            "index": ad.index,
            "conceptualExecution": ad.conceptual_execution,
            "physicalExecution": ad.physical_execution,
            "visualExecution": ad.visual_execution,
            "sceneDescription": ad.scene_description,
        }
        internal = internals.get(ad.index) or internals.get(str(ad.index)) or {}
        if isinstance(internal, dict):
            for key in _EXECUTION_FP_FIELDS:
                if internal.get(key):
                    ad_dict[key] = internal.get(key)
        execution_fp = compute_ad_execution_fingerprint(ad_dict)
        record_id = f"{campaign_id.strip()}:{ad.index}"
        records.append(
            IdeaMemoryRecord(
                schema_version=SCHEMA_VERSION,
                record_id=record_id,
                campaign_id=campaign_id.strip(),
                ad_index=ad.index,
                created_at=created_at,
                strategic_problem=plan.strategic_problem,
                relative_advantage=plan.relative_advantage,
                slogan=plan.brand_slogan,
                conceptual_generator=plan.conceptual_generator,
                physical_generator=plan.physical_generator,
                transferred_object=plan.transferred_object,
                transferred_object_action=plan.transferred_object_action,
                graphic_summary=graphic_summary,
                conceptual_execution=ad.conceptual_execution,
                physical_execution=ad.physical_execution,
                visual_execution=ad.visual_execution,
                scene_description=ad.scene_description,
                execution_subject=str(ad_dict.get("executionSubject") or ""),
                execution_action=str(ad_dict.get("executionAction") or ""),
                execution_object_state=str(ad_dict.get("executionObjectState") or ""),
                execution_scene=str(ad_dict.get("executionScene") or ""),
                execution_punchline=str(ad_dict.get("executionPunchline") or ""),
                campaign_idea_fingerprint=campaign_fp,
                ad_execution_fingerprint=execution_fp,
                product_name=product_name,
            )
        )
    return records


def _fallback_bucket() -> Dict[str, Any]:
    with _memory_lock:
        bucket = _memory_fallback.setdefault(
            "_global",
            {
                "records": [],
                "record_ids": {},
                "backfill_done": False,
                "migrate_v1_done": False,
                "campaign_index": {},
            },
        )
        return bucket


def _load_records_fallback() -> List[IdeaMemoryRecord]:
    bucket = _fallback_bucket()
    parsed: List[IdeaMemoryRecord] = []
    for raw in bucket.get("records") or []:
        if isinstance(raw, dict):
            record = IdeaMemoryRecord.from_dict(raw)
            if record:
                parsed.append(record)
    return parsed


_APPEND_RECORDS_LUA = """
local records_key = KEYS[1]
local ids_key = KEYS[2]
local max_records = tonumber(ARGV[1])
local payload = cjson.decode(ARGV[2])
local added = 0
local skipped = 0
local evicted = 0
for _, item in ipairs(payload.records or {}) do
  local record_id = item.recordId or ''
  if record_id ~= '' then
    if redis.call('HEXISTS', ids_key, record_id) == 1 then
      skipped = skipped + 1
    else
      redis.call('RPUSH', records_key, cjson.encode(item))
      redis.call('HSET', ids_key, record_id, '1')
      added = added + 1
    end
  end
end
while redis.call('LLEN', records_key) > max_records do
  local old = redis.call('LPOP', records_key)
  if old then
    local ok, parsed = pcall(cjson.decode, old)
    if ok and parsed and parsed.recordId then
      redis.call('HDEL', ids_key, parsed.recordId)
    end
    evicted = evicted + 1
  else
    break
  end
end
return cjson.encode({added=added, skipped=skipped, evicted=evicted, count_after=redis.call('LLEN', records_key)})
"""


def persist_idea_memory_records(
    records: Sequence[IdeaMemoryRecord],
    *,
    scope: Optional[IdeaMemoryScope] = None,
) -> IdeaMemoryWriteResult:
    del scope
    if not records:
        return IdeaMemoryWriteResult(0, 0, 0, 0)
    payload_records = [r.to_dict() for r in records]
    if _redis_configured():
        try:
            redis = _get_redis()
            result_raw = redis.eval(
                _APPEND_RECORDS_LUA,
                2,
                _RECORDS_KEY,
                _RECORD_IDS_KEY,
                str(BUILDER1_IDEA_MEMORY_MAX_ADS),
                json.dumps({"records": payload_records}, ensure_ascii=False),
            )
            parsed = json.loads(result_raw)
            write_result = IdeaMemoryWriteResult(
                added_count=int(parsed.get("added") or 0),
                skipped_idempotent_count=int(parsed.get("skipped") or 0),
                count_after=int(parsed.get("count_after") or 0),
                evicted_count=int(parsed.get("evicted") or 0),
            )
            logger.info(
                "BUILDER1_IDEA_MEMORY_WRITE memoryScope=%s campaignId=%s addedCount=%s "
                "skippedIdempotentCount=%s countAfter=%s evictedCount=%s backend=redis",
                MEMORY_SCOPE_GLOBAL_IDEA,
                records[0].campaign_id,
                write_result.added_count,
                write_result.skipped_idempotent_count,
                write_result.count_after,
                write_result.evicted_count,
            )
            return write_result
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_WRITE_REDIS_FAIL err=%s", exc)

    bucket = _fallback_bucket()
    added = 0
    skipped = 0
    evicted = 0
    for item in payload_records:
        record_id = item["recordId"]
        if record_id in bucket["record_ids"]:
            skipped += 1
            continue
        bucket["records"].append(item)
        bucket["record_ids"][record_id] = True
        added += 1
    while len(bucket["records"]) > BUILDER1_IDEA_MEMORY_MAX_ADS:
        old = bucket["records"].pop(0)
        old_id = old.get("recordId")
        if old_id:
            bucket["record_ids"].pop(old_id, None)
        evicted += 1
    write_result = IdeaMemoryWriteResult(
        added_count=added,
        skipped_idempotent_count=skipped,
        count_after=len(bucket["records"]),
        evicted_count=evicted,
    )
    logger.info(
        "BUILDER1_IDEA_MEMORY_WRITE memoryScope=%s campaignId=%s addedCount=%s "
        "skippedIdempotentCount=%s countAfter=%s evictedCount=%s backend=memory",
        MEMORY_SCOPE_GLOBAL_IDEA,
        records[0].campaign_id,
        write_result.added_count,
        write_result.skipped_idempotent_count,
        write_result.count_after,
        write_result.evicted_count,
    )
    return write_result


def register_completed_campaign_for_backfill(
    *,
    campaign_id: str,
    scope: Optional[IdeaMemoryScope] = None,
    ad_count: int,
) -> None:
    del scope
    cid = (campaign_id or "").strip()
    if not cid:
        return
    meta = {
        "campaignId": cid,
        "memoryScope": MEMORY_SCOPE_GLOBAL_IDEA,
        "adCount": int(ad_count),
        "createdAt": time.time(),
    }
    if _redis_configured():
        try:
            redis = _get_redis()
            redis.sadd(_CAMPAIGN_INDEX_KEY, cid)
            redis.set(_campaign_meta_key(cid), json.dumps(meta, ensure_ascii=False))
            return
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_INDEX_FAIL err=%s", exc)
    bucket = _fallback_bucket()
    index = bucket.setdefault("campaign_index", {})
    index[cid] = meta


def _discover_campaign_ids(redis) -> List[str]:
    from engine.builder1_campaign_store import CAMPAIGN_KEY_PREFIX

    discovered: List[str] = []
    seen: set[str] = set()
    for raw_cid in redis.smembers(_CAMPAIGN_INDEX_KEY) or []:
        cid = raw_cid.decode() if isinstance(raw_cid, bytes) else str(raw_cid)
        if cid and cid not in seen:
            seen.add(cid)
            discovered.append(cid)
    cursor = 0
    while True:
        cursor, keys = redis.scan(cursor, match=f"{CAMPAIGN_KEY_PREFIX}*", count=200)
        for raw_key in keys:
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            if not key.startswith(CAMPAIGN_KEY_PREFIX):
                continue
            cid = key[len(CAMPAIGN_KEY_PREFIX) :]
            if cid and cid not in seen:
                seen.add(cid)
                discovered.append(cid)
        if cursor == 0:
            break
    return discovered


def _dedupe_migration_records(records: Sequence[IdeaMemoryRecord]) -> List[IdeaMemoryRecord]:
    sorted_records = sorted(records, key=lambda item: (item.created_at or "", item.record_id))
    seen_ids: set[str] = set()
    seen_execution: set[Tuple[str, str, int]] = set()
    unique: List[IdeaMemoryRecord] = []
    for record in sorted_records:
        if record.record_id in seen_ids:
            continue
        execution_key = (
            record.campaign_idea_fingerprint,
            record.ad_execution_fingerprint,
            record.ad_index,
        )
        if record.ad_execution_fingerprint and execution_key in seen_execution:
            continue
        seen_ids.add(record.record_id)
        if record.ad_execution_fingerprint:
            seen_execution.add(execution_key)
        unique.append(record)
    return unique


def _scan_legacy_record_lists(redis) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    patterns = (
        f"{_LEGACY_KEY_PREFIX}:*:*:records",
        f"{_V2_KEY_PREFIX}:*:records",
    )
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, found = redis.scan(cursor, match=pattern, count=200)
            for raw_key in found:
                key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
                if key.endswith(":records") and key not in seen and key != _RECORDS_KEY:
                    seen.add(key)
                    keys.append(key)
            if cursor == 0:
                break
    return keys


def _collect_legacy_list_records() -> Tuple[List[IdeaMemoryRecord], int]:
    records: List[IdeaMemoryRecord] = []
    skipped = 0
    if _redis_configured():
        try:
            redis = _get_redis()
            for key in _scan_legacy_record_lists(redis):
                for raw in redis.lrange(key, 0, -1) or []:
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        skipped += 1
                        logger.warning("BUILDER1_IDEA_MEMORY_MIGRATE_SKIP reason=malformed_json")
                        continue
                    record = IdeaMemoryRecord.from_dict(item if isinstance(item, dict) else {})
                    if not record:
                        skipped += 1
                        continue
                    records.append(record)
            return records, skipped
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_MIGRATE_LIST_FAIL err=%s", exc)
            return [], 0

    legacy_records: List[IdeaMemoryRecord] = []
    with _memory_lock:
        for key, legacy_bucket in list(_memory_fallback.items()):
            if key == "_global":
                continue
            for raw in legacy_bucket.get("records") or []:
                if not isinstance(raw, dict):
                    skipped += 1
                    continue
                record = IdeaMemoryRecord.from_dict(raw)
                if not record:
                    skipped += 1
                    continue
                legacy_records.append(record)
    return legacy_records, skipped


def _migrate_legacy_scoped_memory() -> Tuple[int, int]:
    if _redis_configured():
        try:
            redis = _get_redis()
            if redis.get(_MIGRATE_V1_FLAG_KEY):
                return 0, 0
        except Exception:
            pass
    else:
        bucket = _fallback_bucket()
        if bucket.get("migrate_v1_done"):
            return 0, 0

    collected, skipped = _collect_legacy_list_records()
    deduped = _dedupe_migration_records(collected)
    if deduped:
        persist_idea_memory_records(deduped)
    if deduped or skipped:
        logger.info(
            "BUILDER1_IDEA_MEMORY_MIGRATE_V1 migratedAdCount=%s skippedCount=%s memoryScope=%s",
            len(deduped),
            skipped,
            MEMORY_SCOPE_GLOBAL_IDEA,
        )
    if _redis_configured():
        try:
            redis = _get_redis()
            redis.set(_MIGRATE_V1_FLAG_KEY, "1")
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_MIGRATE_V1_FLAG_FAIL err=%s", exc)
    else:
        bucket = _fallback_bucket()
        bucket["migrate_v1_done"] = True
    return len(deduped), skipped


def _lazy_backfill() -> Tuple[int, int, int]:
    if _redis_configured():
        try:
            redis = _get_redis()
            if redis.get(_BACKFILL_FLAG_KEY):
                return 0, 0, 0
            migrated = 0
            skipped = 0
            scanned = 0
            batch: List[IdeaMemoryRecord] = []
            from engine.builder1_campaign_store import get_campaign_session

            campaign_entries: List[Tuple[float, str, List[IdeaMemoryRecord]]] = []
            for cid in _discover_campaign_ids(redis):
                scanned += 1
                try:
                    session = get_campaign_session(cid)
                    plan = session.plan
                except Exception:
                    skipped += 1
                    continue
                built = build_records_from_plan(plan, campaign_id=cid)
                if not built:
                    skipped += 1
                    continue
                created_at = float(session.created_at or 0.0)
                campaign_entries.append((created_at, cid, built))
            campaign_entries.sort(key=lambda item: (item[0], item[1]))
            for _created_at, cid, built in campaign_entries:
                batch.extend(built)
                migrated += len(built)
                register_completed_campaign_for_backfill(campaign_id=cid, ad_count=len(built))
            if batch:
                persist_idea_memory_records(batch)
            redis.set(_BACKFILL_FLAG_KEY, "1")
            if scanned:
                logger.info(
                    "BUILDER1_IDEA_MEMORY_BACKFILL scannedCampaignCount=%s migratedAdCount=%s "
                    "skippedCount=%s memoryScope=%s",
                    scanned,
                    migrated,
                    skipped,
                    MEMORY_SCOPE_GLOBAL_IDEA,
                )
            return scanned, migrated, skipped
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_BACKFILL_FAIL err=%s", exc)
            return 0, 0, 0

    bucket = _fallback_bucket()
    if bucket.get("backfill_done"):
        return 0, 0, 0
    bucket["backfill_done"] = True
    return 0, 0, 0


def load_builder1_idea_memory(
    *,
    scope: Optional[IdeaMemoryScope] = None,
    exclude_campaign_id: str = "",
) -> IdeaMemorySnapshot:
    resolved_scope = scope or GLOBAL_IDEA_MEMORY_SCOPE
    _migrate_legacy_scoped_memory()
    backfill_status = "none"
    scanned, migrated, skipped = _lazy_backfill()
    if scanned:
        backfill_status = f"lazy:{migrated}:{skipped}"

    records: List[IdeaMemoryRecord] = []
    if _redis_configured():
        try:
            redis = _get_redis()
            raw_items = redis.lrange(_RECORDS_KEY, 0, -1) or []
            for raw in raw_items:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                record = IdeaMemoryRecord.from_dict(item if isinstance(item, dict) else {})
                if record:
                    records.append(record)
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_READ_REDIS_FAIL err=%s", exc)
            records = _load_records_fallback()
    else:
        records = _load_records_fallback()

    snapshot = IdeaMemorySnapshot(scope=resolved_scope, records=records, backfill_status=backfill_status)
    historical = snapshot.historical_records(exclude_campaign_id=exclude_campaign_id)
    logger.info(
        "BUILDER1_IDEA_MEMORY_READ backend=%s memoryScope=%s recordCount=%s schemaVersion=%s backfillStatus=%s",
        "redis" if _redis_configured() else "memory",
        MEMORY_SCOPE_GLOBAL_IDEA,
        len(historical),
        SCHEMA_VERSION,
        backfill_status,
    )
    return snapshot


def _compact_line(record: IdeaMemoryRecord, *, fields: Sequence[str]) -> str:
    parts: List[str] = []
    mapping = record.to_dict()
    for name in fields:
        value = str(mapping.get(name) or "").strip()
        if value:
            parts.append(f"{name}={value[:120]}")
    return "; ".join(parts)


def build_stage_memory_block(stage: str, snapshot: IdeaMemorySnapshot, *, exclude_campaign_id: str = "") -> str:
    records = snapshot.historical_records(exclude_campaign_id=exclude_campaign_id)
    if not records:
        return ""
    stage_fields: Dict[str, Tuple[str, ...]] = {
        "strategy_slogan_stage": (
            "strategicProblem",
            "relativeAdvantage",
            "slogan",
            "conceptualGenerator",
            "physicalGenerator",
        ),
        "conceptual_stage": ("conceptualGenerator", "campaignIdeaFingerprint"),
        "brand_physical": ("physicalGenerator", "transferredObject", "transferredObjectAction"),
        "graphic_system": ("graphicSummary",),
        "series_ads": (
            "conceptualExecution",
            "physicalExecution",
            "visualExecution",
            "executionSubject",
            "executionAction",
            "executionObjectState",
            "executionScene",
            "executionPunchline",
        ),
    }
    fields = stage_fields.get(stage, ())
    lines: List[str] = []
    seen: set[str] = set()
    for record in records[-40:]:
        line = _compact_line(record, fields=fields)
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(f"- prior ad {record.ad_index} ({record.record_id}): {line}")
    if not lines:
        return ""
    header = (
        "Previous Builder1 advertisements whose underlying creative ideas must NOT be repeated for any product. "
        "Similar underlying idea/mechanism is forbidden even with new wording, product, scene, crop, palette, or language. "
        "Current campaign sibling ads may share series-level generators; compare only against prior campaigns.\n"
    )
    logger.info(
        "BUILDER1_IDEA_MEMORY_INJECTED memoryScope=%s stage=%s recordCount=%s summaryCount=%s",
        MEMORY_SCOPE_GLOBAL_IDEA,
        stage,
        len(records),
        len(lines),
    )
    return header + "\n".join(lines)


def find_historical_duplicate(
    *,
    stage: str,
    snapshot: IdeaMemorySnapshot,
    exclude_campaign_id: str,
    campaign_idea_fingerprint: str = "",
    ad_execution_fingerprint: str = "",
    proposed_execution: Optional[Mapping[str, Any]] = None,
    strategic_problem: str = "",
    relative_advantage: str = "",
    brand_slogan: str = "",
    conceptual_generator: str = "",
    physical_generator: str = "",
    transferred_object: str = "",
    transferred_object_action: str = "",
) -> Optional[HistoricalDuplicateFinding]:
    del brand_slogan
    historical = snapshot.historical_records(exclude_campaign_id=exclude_campaign_id)
    if not historical:
        return None

    if stage == "strategy_slogan_stage":
        sp = normalize_execution_text(strategic_problem)
        ra = normalize_execution_text(relative_advantage)
        for record in historical:
            if (
                sp
                and ra
                and normalize_execution_text(record.strategic_problem) == sp
                and normalize_execution_text(record.relative_advantage) == ra
            ):
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="strategy_campaign_idea",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=f"{sp}|{ra}"[:32],
                )
        if campaign_idea_fingerprint:
            for record in historical:
                if record.campaign_idea_fingerprint == campaign_idea_fingerprint:
                    return HistoricalDuplicateFinding(
                        stage=stage,
                        duplicate_type="strategy_campaign_idea",
                        matching_record_id=record.record_id,
                        matching_campaign_id=record.campaign_id,
                        fingerprint=campaign_idea_fingerprint,
                    )
        cg = normalize_execution_text(conceptual_generator)
        pg = normalize_execution_text(physical_generator)
        ta = normalize_execution_text(transferred_object_action)
        to = normalize_execution_text(transferred_object)
        if cg or pg or ta:
            for record in historical:
                if _campaign_mechanism_overlap(
                    record=record,
                    conceptual_generator=cg,
                    physical_generator=pg,
                    transferred_object=to,
                    transferred_object_action=ta,
                ):
                    return HistoricalDuplicateFinding(
                        stage=stage,
                        duplicate_type="strategy_campaign_mechanism",
                        matching_record_id=record.record_id,
                        matching_campaign_id=record.campaign_id,
                        fingerprint=record.campaign_idea_fingerprint or cg[:16],
                    )
        return None

    if stage == "conceptual_stage":
        normalized = normalize_execution_text(conceptual_generator)
        for record in historical:
            if normalized and normalize_execution_text(record.conceptual_generator) == normalized:
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="conceptual_generator",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=normalized[:16],
                )
            if normalized and record.campaign_idea_fingerprint:
                proposed_fp = compute_campaign_idea_fingerprint(
                    _campaign_idea_payload(
                        strategic_problem=record.strategic_problem,
                        relative_advantage=record.relative_advantage,
                        conceptual_generator=normalized,
                        physical_generator=record.physical_generator,
                        transferred_object=record.transferred_object,
                        transferred_object_action=record.transferred_object_action,
                    )
                )
                if proposed_fp and proposed_fp == record.campaign_idea_fingerprint:
                    return HistoricalDuplicateFinding(
                        stage=stage,
                        duplicate_type="conceptual_campaign_mechanism",
                        matching_record_id=record.record_id,
                        matching_campaign_id=record.campaign_id,
                        fingerprint=proposed_fp,
                    )

    if stage == "brand_physical":
        pg = normalize_execution_text(physical_generator)
        ta = normalize_execution_text(transferred_object_action)
        to = normalize_execution_text(transferred_object)
        for record in historical:
            if (
                pg
                and ta
                and normalize_execution_text(record.physical_generator) == pg
                and normalize_execution_text(record.transferred_object_action) == ta
            ):
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="physical_mechanism",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=f"{pg}|{ta}"[:32],
                )
            if (
                pg
                and ta
                and to
                and normalize_execution_text(record.physical_generator) == pg
                and normalize_execution_text(record.transferred_object_action) == ta
                and normalize_execution_text(record.transferred_object) == to
            ):
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="physical_mechanism_exact",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=f"{pg}|{to}|{ta}"[:32],
                )

    if stage == "series_ads":
        proposed = dict(proposed_execution or {})
        if ad_execution_fingerprint:
            proposed["_executionFingerprint"] = ad_execution_fingerprint
        for record in historical:
            if ad_execution_fingerprint and record.ad_execution_fingerprint == ad_execution_fingerprint:
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="execution_fingerprint",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=ad_execution_fingerprint,
                    ad_index=record.ad_index,
                )
            if proposed and _execution_semantically_duplicate(proposed, record):
                fp = ad_execution_fingerprint or record.ad_execution_fingerprint or "semantic"
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="execution_semantic",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=fp,
                    ad_index=record.ad_index,
                )
    return None


def log_historical_duplicate(finding: HistoricalDuplicateFinding, *, campaign_id: str, repair_attempted: bool) -> None:
    logger.info(
        "BUILDER1_GLOBAL_DUPLICATE_DETECTED stage=%s campaignId=%s adIndex=%s duplicateType=%s "
        "matchingRecordId=%s matchingCampaignId=%s fingerprintHash=%s repairAttempted=%s",
        finding.stage,
        campaign_id,
        finding.ad_index if finding.ad_index is not None else "",
        finding.duplicate_type,
        finding.matching_record_id,
        finding.matching_campaign_id,
        finding.fingerprint,
        str(repair_attempted).lower(),
    )


# Backward-compatible alias used in tests / migration helpers.
_migrate_legacy_tenant_scoped_memory = _migrate_legacy_scoped_memory


def clear_idea_memory_for_tests() -> None:
    with _memory_lock:
        _memory_fallback.clear()
