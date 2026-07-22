"""
Builder1 cross-campaign idea memory — FIFO Redis store per tenant/product scope.

Retains up to BUILDER1_IDEA_MEMORY_MAX_ADS ad-level creative records. Not image bytes.
"""
from __future__ import annotations

import hashlib
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
    execution_dimension_values,
    execution_fingerprint,
    fingerprint_hash,
    normalize_execution_text,
)

logger = logging.getLogger(__name__)

BUILDER1_IDEA_MEMORY_MAX_ADS = 200
SCHEMA_VERSION = 1

_KEY_PREFIX = "builder1:idea-memory"
_CAMPAIGN_INDEX_KEY = f"{_KEY_PREFIX}:campaign-index"
_BACKFILL_FLAG_SUFFIX = ":backfill-v1"

_memory_lock = threading.Lock()
_memory_fallback: Dict[str, Dict[str, Any]] = {}

_CAMPAIGN_IDEA_FIELDS: Tuple[str, ...] = (
    "strategicProblem",
    "relativeAdvantage",
    "slogan",
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


@dataclass(frozen=True)
class IdeaMemoryScope:
    tenant_scope_hash: str
    product_scope_hash: str
    normalized_product_name: str
    normalized_product_description: str
    tenant_limitation: str = ""


@dataclass
class IdeaMemoryRecord:
    schema_version: int
    record_id: str
    tenant_scope_hash: str
    product_scope_hash: str
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "recordId": self.record_id,
            "tenantScopeHash": self.tenant_scope_hash,
            "productScopeHash": self.product_scope_hash,
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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Optional["IdeaMemoryRecord"]:
        record_id = str(raw.get("recordId") or "").strip()
        campaign_id = str(raw.get("campaignId") or "").strip()
        if not record_id or not campaign_id:
            return None
        try:
            ad_index = int(raw.get("adIndex"))
        except (TypeError, ValueError):
            return None
        return cls(
            schema_version=int(raw.get("schemaVersion") or SCHEMA_VERSION),
            record_id=record_id,
            tenant_scope_hash=str(raw.get("tenantScopeHash") or ""),
            product_scope_hash=str(raw.get("productScopeHash") or ""),
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


def _scope_key(scope: IdeaMemoryScope) -> str:
    return f"{scope.tenant_scope_hash}:{scope.product_scope_hash}"


def _records_key(scope: IdeaMemoryScope) -> str:
    return f"{_KEY_PREFIX}:{_scope_key(scope)}:records"


def _record_ids_key(scope: IdeaMemoryScope) -> str:
    return f"{_KEY_PREFIX}:{_scope_key(scope)}:record-ids"


def _backfill_flag_key(scope: IdeaMemoryScope) -> str:
    return f"{_KEY_PREFIX}:{_scope_key(scope)}{_BACKFILL_FLAG_SUFFIX}"


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
    user_product_name: str,
    user_product_description: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> IdeaMemoryScope:
    normalized_name = normalize_product_scope_text(user_product_name)
    normalized_description = normalize_product_scope_text(user_product_description)
    product_payload = f"{normalized_name}||{normalized_description}"
    product_scope_hash = hashlib.sha256(product_payload.encode("utf-8")).hexdigest()[:32]

    tenant_source = ""
    limitation = ""
    env_tenant = (os.environ.get("BUILDER1_MEMORY_TENANT_SCOPE") or "").strip()
    if env_tenant:
        tenant_source = env_tenant
    elif isinstance(brand_guidelines, dict):
        for key in ("tenantId", "clientId", "accountId", "organizationId"):
            candidate = str(brand_guidelines.get(key) or "").strip()
            if candidate:
                tenant_source = candidate
                break
    if not tenant_source:
        tenant_source = "default-shared"
        limitation = "no_stable_tenant_identifier"

    tenant_scope_hash = hashlib.sha256(tenant_source.encode("utf-8")).hexdigest()[:32]
    return IdeaMemoryScope(
        tenant_scope_hash=tenant_scope_hash,
        product_scope_hash=product_scope_hash,
        normalized_product_name=normalized_name,
        normalized_product_description=normalized_description,
        tenant_limitation=limitation,
    )


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
    slogan: str,
    conceptual_generator: str,
    physical_generator: str,
    transferred_object: str,
    transferred_object_action: str,
) -> Dict[str, str]:
    return {
        "strategicProblem": normalize_execution_text(strategic_problem),
        "relativeAdvantage": normalize_execution_text(relative_advantage),
        "slogan": normalize_execution_text(slogan),
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


def build_records_from_plan(
    plan: Builder1SeriesPlan,
    *,
    scope: IdeaMemoryScope,
    campaign_id: str,
) -> List[IdeaMemoryRecord]:
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
                tenant_scope_hash=scope.tenant_scope_hash,
                product_scope_hash=scope.product_scope_hash,
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
            )
        )
    return records


def _fallback_bucket(scope: IdeaMemoryScope) -> Dict[str, Any]:
    key = _scope_key(scope)
    with _memory_lock:
        bucket = _memory_fallback.setdefault(
            key,
            {"records": [], "record_ids": {}, "backfill_done": False},
        )
        return bucket


def _load_records_fallback(scope: IdeaMemoryScope) -> List[IdeaMemoryRecord]:
    bucket = _fallback_bucket(scope)
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
    scope: IdeaMemoryScope,
) -> IdeaMemoryWriteResult:
    if not records:
        return IdeaMemoryWriteResult(0, 0, 0, 0)
    payload_records = [r.to_dict() for r in records]
    if _redis_configured():
        try:
            redis = _get_redis()
            result_raw = redis.eval(
                _APPEND_RECORDS_LUA,
                2,
                _records_key(scope),
                _record_ids_key(scope),
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
                "BUILDER1_IDEA_MEMORY_WRITE campaignId=%s addedCount=%s skippedIdempotentCount=%s "
                "countAfter=%s evictedCount=%s tenantScopeHash=%s productScopeHash=%s",
                records[0].campaign_id,
                write_result.added_count,
                write_result.skipped_idempotent_count,
                write_result.count_after,
                write_result.evicted_count,
                scope.tenant_scope_hash,
                scope.product_scope_hash,
            )
            return write_result
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_WRITE_REDIS_FAIL err=%s", exc)

    bucket = _fallback_bucket(scope)
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
        "BUILDER1_IDEA_MEMORY_WRITE campaignId=%s addedCount=%s skippedIdempotentCount=%s "
        "countAfter=%s evictedCount=%s tenantScopeHash=%s productScopeHash=%s backend=memory",
        records[0].campaign_id,
        write_result.added_count,
        write_result.skipped_idempotent_count,
        write_result.count_after,
        write_result.evicted_count,
        scope.tenant_scope_hash,
        scope.product_scope_hash,
    )
    return write_result


def register_completed_campaign_for_backfill(
    *,
    campaign_id: str,
    scope: IdeaMemoryScope,
    ad_count: int,
) -> None:
    cid = (campaign_id or "").strip()
    if not cid:
        return
    meta = {
        "campaignId": cid,
        "tenantScopeHash": scope.tenant_scope_hash,
        "productScopeHash": scope.product_scope_hash,
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
    bucket = _fallback_bucket(scope)
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


def _scope_matches_plan(scope: IdeaMemoryScope, plan: Builder1SeriesPlan) -> bool:
    plan_scope = resolve_idea_memory_scope(
        user_product_name=plan.product_name or plan.product_name_resolved,
        user_product_description=plan.product_description,
    )
    return (
        plan_scope.tenant_scope_hash == scope.tenant_scope_hash
        and plan_scope.product_scope_hash == scope.product_scope_hash
    )


def _lazy_backfill(scope: IdeaMemoryScope) -> Tuple[int, int, int]:
    if _redis_configured():
        try:
            redis = _get_redis()
            if redis.get(_backfill_flag_key(scope)):
                return 0, 0, 0
            migrated = 0
            skipped = 0
            scanned = 0
            batch: List[IdeaMemoryRecord] = []
            from engine.builder1_campaign_store import get_campaign_session

            campaign_entries: List[Tuple[float, str, List[IdeaMemoryRecord]]] = []
            for cid in _discover_campaign_ids(redis):
                scanned += 1
                meta_raw = redis.get(_campaign_meta_key(cid))
                if meta_raw:
                    try:
                        meta = json.loads(meta_raw)
                    except json.JSONDecodeError:
                        meta = {}
                    if meta.get("productScopeHash") != scope.product_scope_hash:
                        continue
                    if meta.get("tenantScopeHash") != scope.tenant_scope_hash:
                        continue
                try:
                    session = get_campaign_session(cid)
                    plan = session.plan
                except Exception:
                    skipped += 1
                    continue
                if not _scope_matches_plan(scope, plan):
                    continue
                built = build_records_from_plan(plan, scope=scope, campaign_id=cid)
                if not built:
                    skipped += 1
                    continue
                created_at = float(session.created_at or 0.0)
                campaign_entries.append((created_at, cid, built))
            campaign_entries.sort(key=lambda item: (item[0], item[1]))
            for _created_at, cid, built in campaign_entries:
                batch.extend(built)
                migrated += len(built)
                register_completed_campaign_for_backfill(
                    campaign_id=cid,
                    scope=scope,
                    ad_count=len(built),
                )
            if batch:
                persist_idea_memory_records(batch, scope=scope)
            redis.set(_backfill_flag_key(scope), "1")
            if scanned:
                logger.info(
                    "BUILDER1_IDEA_MEMORY_BACKFILL scannedCampaignCount=%s migratedAdCount=%s "
                    "skippedCount=%s productScopeHash=%s tenantScopeHash=%s",
                    scanned,
                    migrated,
                    skipped,
                    scope.product_scope_hash,
                    scope.tenant_scope_hash,
                )
            return scanned, migrated, skipped
        except Exception as exc:
            logger.warning("BUILDER1_IDEA_MEMORY_BACKFILL_FAIL err=%s", exc)
            return 0, 0, 0

    bucket = _fallback_bucket(scope)
    if bucket.get("backfill_done"):
        return 0, 0, 0
    bucket["backfill_done"] = True
    return 0, 0, 0


def load_builder1_idea_memory(
    *,
    scope: IdeaMemoryScope,
    exclude_campaign_id: str = "",
) -> IdeaMemorySnapshot:
    backfill_status = "none"
    scanned, migrated, skipped = _lazy_backfill(scope)
    if scanned:
        backfill_status = f"lazy:{migrated}:{skipped}"

    records: List[IdeaMemoryRecord] = []
    if _redis_configured():
        try:
            redis = _get_redis()
            raw_items = redis.lrange(_records_key(scope), 0, -1) or []
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
            records = _load_records_fallback(scope)
    else:
        records = _load_records_fallback(scope)

    snapshot = IdeaMemorySnapshot(scope=scope, records=records, backfill_status=backfill_status)
    historical = snapshot.historical_records(exclude_campaign_id=exclude_campaign_id)
    logger.info(
        "BUILDER1_IDEA_MEMORY_READ backend=%s tenantScopeHash=%s productScopeHash=%s "
        "recordCount=%s schemaVersion=%s backfillStatus=%s tenantLimitation=%s",
        "redis" if _redis_configured() else "memory",
        scope.tenant_scope_hash,
        scope.product_scope_hash,
        len(historical),
        SCHEMA_VERSION,
        backfill_status,
        scope.tenant_limitation or "none",
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
        "Previous advertisements for this product that must NOT be repeated. "
        "Similar underlying idea/mechanism is forbidden even with new wording, scene, crop, or palette. "
        "Current campaign sibling ads may share series-level generators; compare only against prior campaigns.\n"
    )
    logger.info(
        "BUILDER1_IDEA_MEMORY_INJECTED stage=%s recordCount=%s summaryCount=%s tenantScopeHash=%s productScopeHash=%s",
        stage,
        len(records),
        len(lines),
        snapshot.scope.tenant_scope_hash,
        snapshot.scope.product_scope_hash,
    )
    return header + "\n".join(lines)


def find_historical_duplicate(
    *,
    stage: str,
    snapshot: IdeaMemorySnapshot,
    exclude_campaign_id: str,
    campaign_idea_fingerprint: str = "",
    ad_execution_fingerprint: str = "",
    strategic_problem: str = "",
    relative_advantage: str = "",
    brand_slogan: str = "",
    conceptual_generator: str = "",
    physical_generator: str = "",
    transferred_object: str = "",
    transferred_object_action: str = "",
) -> Optional[HistoricalDuplicateFinding]:
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

    if stage == "brand_physical":
        pg = normalize_execution_text(physical_generator)
        to = normalize_execution_text(transferred_object)
        ta = normalize_execution_text(transferred_object_action)
        for record in historical:
            if (
                pg
                and normalize_execution_text(record.physical_generator) == pg
                and normalize_execution_text(record.transferred_object_action) == ta
                and normalize_execution_text(record.transferred_object) == to
            ):
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="physical_mechanism",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=f"{pg}|{ta}"[:32],
                )

    if stage == "series_ads" and ad_execution_fingerprint:
        for record in historical:
            if record.ad_execution_fingerprint == ad_execution_fingerprint:
                return HistoricalDuplicateFinding(
                    stage=stage,
                    duplicate_type="execution_fingerprint",
                    matching_record_id=record.record_id,
                    matching_campaign_id=record.campaign_id,
                    fingerprint=ad_execution_fingerprint,
                    ad_index=record.ad_index,
                )
    return None


def log_historical_duplicate(finding: HistoricalDuplicateFinding, *, campaign_id: str, repair_attempted: bool) -> None:
    logger.info(
        "BUILDER1_HISTORICAL_DUPLICATE_DETECTED stage=%s campaignId=%s adIndex=%s duplicateType=%s "
        "matchingRecordId=%s matchingCampaignId=%s fingerprint=%s repairAttempted=%s",
        finding.stage,
        campaign_id,
        finding.ad_index if finding.ad_index is not None else "",
        finding.duplicate_type,
        finding.matching_record_id,
        finding.matching_campaign_id,
        finding.fingerprint,
        str(repair_attempted).lower(),
    )


def clear_idea_memory_for_tests() -> None:
    with _memory_lock:
        _memory_fallback.clear()
