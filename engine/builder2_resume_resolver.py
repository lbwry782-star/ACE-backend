"""
Builder2 durable resume resolver — first incomplete stage from persisted state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

from engine.builder2_accepted_creator_store import ACCEPTED_CREATOR_INDEX_KEY, backfill_accepted_creator_index
from engine.builder2_accepted_judgment_store import ACCEPTED_JUDGMENT_INDEX_KEY, backfill_accepted_judgment_index
from engine.builder2_advertising_closure_contract import (
    advertising_closure_is_required,
    get_advertising_closure_status,
)
from engine.builder2_resume_contract import (
    CANONICAL_BUILDER2_STAGES,
    _final_video_url,
    _media_bucket,
    _runway_bucket,
    completed_stage_names,
    get_stage_checkpoint,
    normalize_builder2_stage,
    sync_builder2_stage_checkpoints_from_state,
)
from engine.builder2_tournament_completion_gate import (
    is_tournament_ready_for_winner_selection,
    missing_creator_prototype_ids,
    missing_judge_prototype_ids,
)
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_resume_stage
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids
from engine.builder2_winner_persistence import is_valid_persisted_winner_development


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _stage_index(stage: str) -> int:
    normalized = normalize_builder2_stage(stage)
    try:
        return CANONICAL_BUILDER2_STAGES.index(normalized)
    except ValueError:
        return 0


def _checkpoint_completed(state: Mapping[str, Any], stage: str) -> bool:
    entry = get_stage_checkpoint(state, stage)
    return isinstance(entry, dict) and _clean(entry.get("status")) == "completed"


def _validate_checkpoint_artifact(state: Mapping[str, Any], stage: str) -> Optional[str]:
    entry = get_stage_checkpoint(state, stage)
    if not isinstance(entry, dict) or _clean(entry.get("status")) != "completed":
        return None
    ref = _clean(entry.get("artifactRef"))
    if stage == "strategy":
        strategy = state.get("strategyFoundation")
        if not isinstance(strategy, dict) or not strategy:
            return f"consistency_failure:strategy_missing despite checkpoint"
    if stage in {"creator_generation", "creator_complete"}:
        backfill_accepted_creator_index(dict(state))
        index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
        active = list(
            state.get("initialActivePrototypeIds")
            or state.get("activePrototypeIds")
            or resolve_builder2_active_prototype_ids()
        )
        if len(index) < len(active):
            return "consistency_failure:creator_checkpoint_without_artifacts"
    if stage in {"judge_generation", "judge_complete"}:
        backfill_accepted_judgment_index(dict(state))
        index = state.get(ACCEPTED_JUDGMENT_INDEX_KEY) or {}
        active = list(
            state.get("initialActivePrototypeIds")
            or state.get("activePrototypeIds")
            or resolve_builder2_active_prototype_ids()
        )
        if len(index) < len(active):
            return "consistency_failure:judge_checkpoint_without_artifacts"
    if stage == "winner_selection" and not _clean(state.get("winnerCandidateId")):
        return "consistency_failure:winner_selection_checkpoint_without_winner"
    if stage == "winner_development" and not is_valid_persisted_winner_development(dict(state)):
        return "consistency_failure:winner_development_checkpoint_without_plan"
    if stage == "runway_submission":
        media = _media_bucket(state)
        runway = _runway_bucket(state)
        if not _clean(media.get("runwayTaskId") or runway.get("taskId")):
            return "consistency_failure:runway_submission_checkpoint_without_task_id"
    if stage == "runway_complete":
        media = _media_bucket(state)
        runway = _runway_bucket(state)
        output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl") or runway.get("outputUrl"))
        if not output:
            return "consistency_failure:runway_complete_checkpoint_without_output"
    if stage == "publishing_final_video" and not _final_video_url(None, state):
        return "consistency_failure:final_video_checkpoint_without_url"
    if ref and ref not in state and ref not in _media_bucket(state):
        if stage not in {
            "strategy",
            "creator_generation",
            "creator_complete",
            "judge_generation",
            "judge_complete",
            "winner_selection",
            "winner_development",
            "runway_submission",
            "runway_complete",
            "publishing_final_video",
            "completed",
        }:
            return f"consistency_failure:missing_artifact_ref:{ref}"
    return None


def _infer_resume_stage(state: Mapping[str, Any], job_state: Optional[Mapping[str, Any]], *, read_only: bool = False) -> str:
    final_url = _final_video_url(job_state, state)
    if final_url:
        return "completed"

    strategy = state.get("strategyFoundation")
    if not isinstance(strategy, dict) or not strategy:
        return "strategy"

    state_dict = dict(state)
    if not read_only:
        backfill_accepted_creator_index(state_dict)
    else:
        backfill_accepted_creator_index(state_dict, persist=False)

    complete_ad_stage = resolve_complete_ad_resume_stage(state_dict, read_only=read_only)
    assigned_count = len(
        list(
            state.get("initialActivePrototypeIds")
            or state.get("activePrototypeIds")
            or resolve_builder2_active_prototype_ids()
        )
    )
    if assigned_count >= 6 and complete_ad_stage in {
        "creator_generation",
        "judge_generation",
        "winner_selection",
        "winner_development",
    }:
        if complete_ad_stage == "winner_development" and not _clean(state.get("winnerCandidateId")):
            return "winner_selection"
        return complete_ad_stage

    if missing_creator_prototype_ids(state_dict, read_only=read_only):
        return "creator_generation"

    if not read_only:
        backfill_accepted_judgment_index(state_dict)
    else:
        backfill_accepted_judgment_index(state_dict, persist=False)

    if missing_judge_prototype_ids(state_dict, read_only=read_only):
        return "judge_generation"

    if not is_tournament_ready_for_winner_selection(state_dict, read_only=read_only):
        if missing_creator_prototype_ids(state_dict, read_only=read_only):
            return "creator_generation"
        return "judge_generation"

    if not _clean(state.get("winnerCandidateId")):
        return "winner_selection"

    if not is_valid_persisted_winner_development(dict(state)):
        return "winner_development"

    plan = state.get("winnerDevelopmentPlan")
    if isinstance(plan, dict) and advertising_closure_is_required(plan):
        closure_status = get_advertising_closure_status(dict(state))
        if closure_status not in {"approved", "completed"}:
            return "advertising_closure"

    if not state.get("mediaContinuationRequired"):
        return "media_prerequisite_validation"

    media = _media_bucket(state)
    runway = _runway_bucket(state)
    start_image = _clean(media.get("startImageArtifact") or runway.get("startImageDataUri"))
    if not start_image or _clean(media.get("startImageStatus")) != "completed":
        return "start_image_generation"

    runway_task_id = _clean(media.get("runwayTaskId") or runway.get("taskId"))
    if not runway_task_id:
        return "runway_submission"

    runway_status = _clean(media.get("runwayStatus") or runway.get("status")).lower()
    runway_output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl") or runway.get("outputUrl"))
    if runway_status not in {"succeeded", "completed", "complete"} and not runway_output:
        return "runway_waiting"

    if not _clean(media.get("downloadedVideoPath") or media.get("localVideoPath")) and not runway_output.startswith("http"):
        return "video_download"

    if _clean(media.get("postprocessStatus")) != "completed" and _clean((job_state or {}).get("postprocessRan")) != "1":
        headline_done = _clean(media.get("headlinePostprocessStatus")) in {
            "completed",
            "skipped_single_slogan_contract",
            "reused",
        } or bool(_clean(media.get("headlineArtifactUrl")))
        if not headline_done:
            return "postprocessing"

    from engine.builder2_lyria_config import resolve_builder2_lyria_enabled

    if resolve_builder2_lyria_enabled():
        headline_ready = (
            _clean(media.get("postprocessStatus")) == "completed"
            or _clean((job_state or {}).get("postprocessRan")) == "1"
            or _clean(media.get("headlinePostprocessStatus")) in {
                "completed",
                "skipped_single_slogan_contract",
                "reused",
            }
            or bool(_clean(media.get("headlineArtifactUrl")))
        )
        if headline_ready:
            music_status = _clean(media.get("musicGenerationStatus")).lower()
            music_durable = _clean(media.get("musicArtifactUrl"))
            from engine.builder2_lyria import music_artifact_is_valid

            music_local_ok = music_artifact_is_valid(_clean(media.get("musicArtifactPath")))
            music_complete = music_status == "succeeded" and (music_durable or music_local_ok)
            if not music_complete:
                return "generating_music"

    closure_url = _clean(media.get("finalVideoWithClosureUrl"))
    plan_dict = plan if isinstance(plan, dict) else {}
    if advertising_closure_is_required(plan_dict) and not closure_url and get_advertising_closure_status(dict(state)) == "approved":
        return "rendering_advertising_closure"

    if not final_url:
        return "publishing_final_video"
    return "completed"


def _reusable_artifacts(state: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []
    if isinstance(state.get("strategyFoundation"), dict):
        refs.append("strategyFoundation")
    if state.get(ACCEPTED_CREATOR_INDEX_KEY):
        refs.append(ACCEPTED_CREATOR_INDEX_KEY)
    if state.get(ACCEPTED_JUDGMENT_INDEX_KEY):
        refs.append(ACCEPTED_JUDGMENT_INDEX_KEY)
    if _clean(state.get("winnerCandidateId")):
        refs.append("winnerCandidateId")
    if is_valid_persisted_winner_development(dict(state)):
        refs.append("winnerDevelopmentPlan")
    media = _media_bucket(state)
    runway = _runway_bucket(state)
    if _clean(media.get("startImageArtifact") or runway.get("startImageDataUri")):
        refs.append("startImageArtifact")
    task_id = _clean(media.get("runwayTaskId") or runway.get("taskId"))
    if task_id:
        refs.append("runwayTaskId")
    output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl"))
    if output:
        refs.append("runwayOutputUrl")
    music_path = _clean(media.get("musicArtifactPath"))
    music_url = _clean(media.get("musicArtifactUrl"))
    if str(media.get("musicGenerationStatus") or "").lower() == "succeeded" and (music_url or music_path):
        refs.append("musicArtifactUrl" if music_url else "musicArtifactPath")
    final_url = _final_video_url(None, state)
    if final_url:
        refs.append("finalPublicUrl")
    return refs


def resolve_builder2_resume_stage(
    job_state: Optional[Mapping[str, Any]],
    tournament_state: Optional[Mapping[str, Any]],
    *,
    read_only: bool = False,
) -> Dict[str, Any]:
    enriched: Dict[str, Any] = dict(tournament_state) if isinstance(tournament_state, dict) else {}
    if read_only:
        from copy import deepcopy

        enriched = deepcopy(enriched)
    sync_builder2_stage_checkpoints_from_state(job_state=job_state, tournament_state=enriched)

    completed: Set[str] = set(completed_stage_names(enriched))
    consistency_failures: List[str] = []
    for stage in completed:
        failure = _validate_checkpoint_artifact(enriched, stage)
        if failure:
            consistency_failures.append(failure)

    final_url = _final_video_url(job_state, enriched)
    job_status = _clean((job_state or {}).get("status")).lower()
    if final_url and job_status in {"done", "completed"}:
        return {
            "resumeRequired": False,
            "resumeFromStage": "completed",
            "completedStages": sorted(completed),
            "reusableArtifacts": _reusable_artifacts(enriched),
            "blockedReason": None,
            "jobAlreadyCompleted": True,
            "canResume": False,
            "consistencyFailures": consistency_failures,
        }

    if consistency_failures:
        return {
            "resumeRequired": False,
            "resumeFromStage": None,
            "completedStages": sorted(completed),
            "reusableArtifacts": _reusable_artifacts(enriched),
            "blockedReason": consistency_failures[0],
            "jobAlreadyCompleted": False,
            "canResume": False,
            "consistencyFailures": consistency_failures,
        }

    resume_from = _infer_resume_stage(enriched, job_state, read_only=read_only)
    resume_from = normalize_builder2_stage(resume_from)

    failure_info = enriched.get("resumeFailure") if isinstance(enriched.get("resumeFailure"), dict) else {}
    can_resume = resume_from != "completed" and not consistency_failures
    if job_status in {"recovery_exhausted", "cancelled"}:
        can_resume = False

    return {
        "resumeRequired": can_resume,
        "resumeFromStage": resume_from,
        "completedStages": sorted(completed),
        "reusableArtifacts": _reusable_artifacts(enriched),
        "blockedReason": None if can_resume else _clean(failure_info.get("failureReason")) or None,
        "jobAlreadyCompleted": False,
        "canResume": can_resume,
        "consistencyFailures": consistency_failures,
    }
