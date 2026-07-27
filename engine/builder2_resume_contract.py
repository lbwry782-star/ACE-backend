"""
Builder2 durable resume contract — canonical stage machine and checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

BUILDER2_RESUME_CONTRACT_VERSION = "builder2_resume_v1"

CANONICAL_BUILDER2_STAGES: Tuple[str, ...] = (
    "queued",
    "strategy",
    "creator_generation",
    "creator_complete",
    "judge_generation",
    "judge_complete",
    "winner_selection",
    "winner_development",
    "advertising_closure",
    "media_prerequisite_validation",
    "start_image_generation",
    "start_image_complete",
    "runway_submission",
    "runway_waiting",
    "runway_complete",
    "video_download",
    "postprocessing",
    "rendering_advertising_closure",
    "publishing_final_video",
    "completed",
)

_STAGE_ALIASES: Dict[str, str] = {
    "created": "queued",
    "strategy_generating": "strategy",
    "strategy_complete": "creator_generation",
    "round_1_generating": "creator_generation",
    "round_1_complete": "judge_generation",
    "tournament_complete": "winner_selection",
    "winner_developing": "winner_development",
    "winner_plan_complete": "advertising_closure",
    "preparing_start_image": "start_image_generation",
    "start_image": "start_image_generation",
    "runway": "runway_submission",
    "runway_polling": "runway_waiting",
    "done": "completed",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_builder2_stage(stage: Any) -> str:
    token = _clean(stage).lower()
    if not token:
        return "queued"
    if token in CANONICAL_BUILDER2_STAGES:
        return token
    return _STAGE_ALIASES.get(token, token)


def _media_bucket(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _runway_bucket(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    runway = state.get("runway")
    return runway if isinstance(runway, dict) else {}


def _final_video_url(
    job_state: Optional[Mapping[str, Any]],
    tournament_state: Optional[Mapping[str, Any]],
) -> str:
    for mapping in (job_state, tournament_state):
        if not isinstance(mapping, Mapping):
            continue
        for key in ("videoUrl", "video_url", "finalVideoUrl", "final_video_url"):
            value = _clean(mapping.get(key))
            if value.startswith("http://") or value.startswith("https://"):
                return value
    media = _media_bucket(tournament_state)
    for key in ("finalPublicUrl", "finalVideoWithClosureUrl", "finalVideoUrl"):
        value = _clean(media.get(key))
        if value.startswith("http://") or value.startswith("https://"):
            return value
    delivery = tournament_state.get("completedDelivery") if isinstance(tournament_state, Mapping) else None
    if isinstance(delivery, dict):
        for key in ("publicUrl", "videoUrl", "finalPublicUrl"):
            value = _clean(delivery.get(key))
            if value.startswith("http://") or value.startswith("https://"):
                return value
    return ""


def identity_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_stage_checkpoints(state: Dict[str, Any]) -> Dict[str, Any]:
    checkpoints = state.get("stageCheckpoints")
    if not isinstance(checkpoints, dict):
        checkpoints = {}
        state["stageCheckpoints"] = checkpoints
    return checkpoints


def get_stage_checkpoint(state: Mapping[str, Any], stage: str) -> Optional[Dict[str, Any]]:
    checkpoints = state.get("stageCheckpoints")
    if not isinstance(checkpoints, dict):
        return None
    entry = checkpoints.get(stage)
    return entry if isinstance(entry, dict) else None


def upsert_stage_checkpoint(
    state: Dict[str, Any],
    stage: str,
    *,
    status: str,
    artifact_ref: str = "",
    identity: Any = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    attempt: int = 1,
) -> Dict[str, Any]:
    checkpoints = ensure_stage_checkpoints(state)
    existing = checkpoints.get(stage)
    if not isinstance(existing, dict):
        existing = {}
    entry = dict(existing)
    entry["status"] = status
    if started_at:
        entry["startedAt"] = started_at
    elif not entry.get("startedAt"):
        entry["startedAt"] = _utc_now_iso()
    if completed_at:
        entry["completedAt"] = completed_at
    if status == "completed" and not entry.get("completedAt"):
        entry["completedAt"] = _utc_now_iso()
    entry["attempt"] = max(int(entry.get("attempt") or 0), int(attempt or 1))
    if artifact_ref:
        entry["artifactRef"] = artifact_ref
    if identity is not None:
        entry["identityHash"] = identity_hash(identity)
    checkpoints[stage] = entry
    state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
    return entry


def completed_stage_names(state: Mapping[str, Any]) -> List[str]:
    checkpoints = state.get("stageCheckpoints")
    if not isinstance(checkpoints, dict):
        return []
    return sorted(
        stage
        for stage, entry in checkpoints.items()
        if isinstance(entry, dict) and _clean(entry.get("status")) == "completed"
    )


def sync_builder2_stage_checkpoints_from_state(
    *,
    job_state: Optional[Mapping[str, Any]],
    tournament_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Infer durable checkpoints from authoritative persisted artifacts (read-only inference).
    Does not mutate Redis unless tournament_state dict is passed for in-memory enrichment.
    """
    working: Dict[str, Any] = dict(tournament_state) if isinstance(tournament_state, dict) else {}
    checkpoints = ensure_stage_checkpoints(working)

    job_status = _clean((job_state or {}).get("status")).lower()
    if job_status in {"running", "queued", "processing", "interrupted", "error", "failed"} or working:
        upsert_stage_checkpoint(working, "queued", status="completed")

    strategy = working.get("strategyFoundation")
    if isinstance(strategy, dict) and strategy:
        upsert_stage_checkpoint(
            working,
            "strategy",
            status="completed",
            artifact_ref="strategyFoundation",
            identity=strategy.get("strategyFoundationId") or strategy,
        )

    from engine.builder2_accepted_creator_store import ACCEPTED_CREATOR_INDEX_KEY, backfill_accepted_creator_index
    from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids

    backfill_accepted_creator_index(working)
    creator_index = working.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    active_prototypes = list(
        working.get("initialActivePrototypeIds")
        or working.get("activePrototypeIds")
        or resolve_builder2_active_prototype_ids()
    )
    if active_prototypes and len(creator_index) >= len(active_prototypes):
        upsert_stage_checkpoint(
            working,
            "creator_generation",
            status="completed",
            artifact_ref=ACCEPTED_CREATOR_INDEX_KEY,
            identity=sorted(str(k) for k in creator_index.keys()),
        )
        upsert_stage_checkpoint(
            working,
            "creator_complete",
            status="completed",
            artifact_ref=ACCEPTED_CREATOR_INDEX_KEY,
            identity=len(creator_index),
        )

    from engine.builder2_accepted_judgment_store import ACCEPTED_JUDGMENT_INDEX_KEY, backfill_accepted_judgment_index

    backfill_accepted_judgment_index(working)
    judgment_index = working.get(ACCEPTED_JUDGMENT_INDEX_KEY) or {}
    if active_prototypes and len(judgment_index) >= len(active_prototypes):
        upsert_stage_checkpoint(
            working,
            "judge_generation",
            status="completed",
            artifact_ref=ACCEPTED_JUDGMENT_INDEX_KEY,
            identity=sorted(str(k) for k in judgment_index.keys()),
        )
        upsert_stage_checkpoint(
            working,
            "judge_complete",
            status="completed",
            artifact_ref=ACCEPTED_JUDGMENT_INDEX_KEY,
            identity=len(judgment_index),
        )

    winner_id = _clean(working.get("winnerCandidateId") or working.get("winnerDevelopmentCandidateId"))
    if winner_id:
        upsert_stage_checkpoint(
            working,
            "winner_selection",
            status="completed",
            artifact_ref="winnerCandidateId",
            identity=winner_id,
        )

    from engine.builder2_winner_persistence import is_valid_persisted_winner_development

    if is_valid_persisted_winner_development(working):
        upsert_stage_checkpoint(
            working,
            "winner_development",
            status="completed",
            artifact_ref="winnerDevelopmentPlan",
            identity=working.get("winnerDevelopmentCandidateId"),
        )

    from engine.builder2_advertising_closure_contract import (
        advertising_closure_is_required,
        get_advertising_closure_status,
    )

    plan = working.get("winnerDevelopmentPlan")
    closure_status = get_advertising_closure_status(working)
    if isinstance(plan, dict) and (
        not advertising_closure_is_required(plan)
        or closure_status in {"approved", "completed", "proposed"}
    ):
        upsert_stage_checkpoint(
            working,
            "advertising_closure",
            status="completed" if closure_status in {"approved", "completed"} else "attempting",
            artifact_ref="advertisingClosureStatus",
            identity=closure_status,
        )

    if working.get("mediaContinuationRequired"):
        upsert_stage_checkpoint(working, "media_prerequisite_validation", status="completed")

    media = _media_bucket(working)
    runway = _runway_bucket(working)
    start_image = _clean(media.get("startImageArtifact") or runway.get("startImageDataUri"))
    if start_image:
        upsert_stage_checkpoint(
            working,
            "start_image_generation",
            status="completed",
            artifact_ref="startImageArtifact",
            identity=media.get("startImageArtifactHash") or start_image[:64],
        )
        upsert_stage_checkpoint(
            working,
            "start_image_complete",
            status="completed",
            artifact_ref="startImageArtifact",
            identity=media.get("startImageStatus") or "completed",
        )

    runway_task_id = _clean(media.get("runwayTaskId") or runway.get("taskId"))
    if runway_task_id:
        upsert_stage_checkpoint(
            working,
            "runway_submission",
            status="completed",
            artifact_ref="runwayTaskId",
            identity=runway_task_id,
        )
        runway_status = _clean(media.get("runwayStatus") or runway.get("status")).lower()
        runway_output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl") or runway.get("outputUrl"))
        if runway_status in {"succeeded", "completed", "complete"} or runway_output:
            upsert_stage_checkpoint(
                working,
                "runway_waiting",
                status="completed",
                artifact_ref="runwayTaskId",
                identity=runway_task_id,
            )
            upsert_stage_checkpoint(
                working,
                "runway_complete",
                status="completed",
                artifact_ref="runwayOutputUrl",
                identity=runway_output or runway_task_id,
            )
        else:
            upsert_stage_checkpoint(
                working,
                "runway_waiting",
                status="attempting",
                artifact_ref="runwayTaskId",
                identity=runway_task_id,
            )

    downloaded = _clean(media.get("downloadedVideoPath") or media.get("localVideoPath"))
    if downloaded:
        upsert_stage_checkpoint(working, "video_download", status="completed", artifact_ref="downloadedVideoPath")

    if _clean(media.get("postprocessStatus")) == "completed" or _clean(job_state.get("postprocessRan")) == "1":
        upsert_stage_checkpoint(working, "postprocessing", status="completed")

    closure_url = _clean(media.get("finalVideoWithClosureUrl"))
    if closure_url:
        upsert_stage_checkpoint(
            working,
            "rendering_advertising_closure",
            status="completed",
            artifact_ref="finalVideoWithClosureUrl",
            identity=closure_url,
        )

    final_url = _final_video_url(job_state, working)
    if final_url:
        upsert_stage_checkpoint(
            working,
            "publishing_final_video",
            status="completed",
            artifact_ref="finalPublicUrl",
            identity=final_url,
        )
        upsert_stage_checkpoint(working, "completed", status="completed", artifact_ref="finalPublicUrl", identity=final_url)

    if isinstance(tournament_state, dict):
        tournament_state.clear()
        tournament_state.update(working)
        tournament_state["stageCheckpoints"] = checkpoints

    return working
