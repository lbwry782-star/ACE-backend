"""
Builder2 completed-job inspector — read-only Redis diagnostics.

Run:
  BUILDER2_JOB_INSPECT_ID=<jobId> python -m engine.builder2_job_inspect

Reads only:
  ace:video:job:{jobId}
  ace:builder2:tournament:{jobId}

Does not mutate Redis, rerun jobs, or call OpenAI/Runway/FFmpeg.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import job_key, redis_configured, video_job_get

logger = logging.getLogger(__name__)

DEFAULT_JOB_INSPECT_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"

_JOB_VIDEO_URL_KEYS = ("videoUrl", "video_url", "finalVideoUrl", "final_video_url")
_TOURNAMENT_DELIVERY_URL_KEYS = ("finalPublicUrl", "finalVideoUrl", "final_video_url", "deliveryVideoUrl")
_OWNERSHIP_FIELD_NAMES = (
    "user_id",
    "userId",
    "session_id",
    "sessionId",
    "owner_id",
    "ownerId",
    "account_id",
    "accountId",
)
_SENSITIVE_OUTPUT_KEYS = frozenset(
    {
        "marketingText",
        "marketing_text",
        "productDescription",
        "product_description",
        "prompt",
        "startImageArtifact",
        "startImageDataUri",
        "runwayTaskId",
        "runwayVideoUrl",
        "OPENAI_API_KEY",
        "REDIS_URL",
    }
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _timestamp_from_raw(raw_value: Any) -> Optional[str]:
    token = _clean(raw_value)
    if not token:
        return None
    if token.isdigit():
        try:
            return datetime.fromtimestamp(int(token), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return token
    return token


def _read_job_hash_raw(job_id: str) -> Optional[Dict[str, str]]:
    if not redis_configured():
        return None
    from engine.video_jobs_redis import get_redis

    data = get_redis().hgetall(job_key(job_id))
    return data if data else None


def _first_present(mapping: Optional[Dict[str, Any]], keys: Tuple[str, ...]) -> str:
    if not isinstance(mapping, dict):
        return ""
    for key in keys:
        value = _clean(mapping.get(key))
        if value:
            return value
    return ""


def _ownership_fields_present(*mappings: Optional[Dict[str, Any]]) -> bool:
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in _OWNERSHIP_FIELD_NAMES:
            if _clean(mapping.get(key)):
                return True
    return False


def _is_public_http_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _media_bucket(tournament_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    media = (tournament_state or {}).get("mediaResume")
    return media if isinstance(media, dict) else {}


def _resolve_video_url(
    *,
    job_record: Optional[Dict[str, Any]],
    job_raw: Optional[Dict[str, str]],
    tournament_state: Optional[Dict[str, Any]],
) -> Tuple[str, str, List[str]]:
    missing_paths: List[str] = []
    media = _media_bucket(tournament_state)

    job_url = _first_present(job_record, _JOB_VIDEO_URL_KEYS)
    if not job_url and isinstance(job_raw, dict):
        job_url = _first_present(job_raw, _JOB_VIDEO_URL_KEYS)
    if not job_url:
        missing_paths.append("job.video_url")

    media_url = _clean(media.get("finalPublicUrl"))
    if not media_url:
        missing_paths.append("mediaResume.finalPublicUrl")

    tournament_url = _first_present(tournament_state, _TOURNAMENT_DELIVERY_URL_KEYS)
    if not tournament_url:
        missing_paths.append("tournament.finalPublicUrl")

    delivery_url = ""
    delivery = tournament_state.get("completedDelivery") if isinstance(tournament_state, dict) else None
    if isinstance(delivery, dict):
        delivery_url = _first_present(delivery, ("publicUrl", "videoUrl", "finalPublicUrl", "finalVideoUrl"))
    if not delivery_url:
        missing_paths.append("tournament.completedDelivery.publicUrl")

    for candidate, source in (
        (job_url, "job_video_url"),
        (media_url, "media_final_public_url"),
        (tournament_url, "tournament_final_public_url"),
        (delivery_url, "completed_delivery_public_url"),
    ):
        if candidate and _is_public_http_url(candidate):
            return candidate, source, missing_paths
    return "", "missing", missing_paths


def inspect_builder2_completed_job(
    job_id: str,
    *,
    job_getter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    tournament_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    raw_job_reader: Optional[Callable[[str], Optional[Dict[str, str]]]] = None,
) -> Dict[str, Any]:
    read_job = job_getter or video_job_get
    load_tournament = tournament_loader or load_tournament_state
    read_raw_job = raw_job_reader or _read_job_hash_raw
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "jobExists": False,
        "jobStatus": None,
        "jobCompleted": False,
        "videoUrl": None,
        "videoUrlPresent": False,
        "marketingTextPresent": False,
        "publicBaseUrl": None,
        "ownershipFieldsPresent": False,
        "createdAt": None,
        "completedAt": None,
        "errorPresent": False,
        "tournamentExists": False,
        "tournamentStatus": None,
        "winnerCandidateId": None,
        "winnerPrototypeId": None,
        "mediaContinuationRequired": None,
        "mediaResumeStatus": None,
        "progressStage": None,
        "finalVideoAvailable": False,
        "finalPublicUrl": None,
        "finalPublicUrlPresent": False,
        "finalVideoPathPresent": False,
        "runwayTaskIdPresent": False,
        "runwayVideoUrlPresent": False,
        "mediaCompletedAt": None,
        "mediaFailurePresent": False,
        "resolvedVideoUrl": None,
        "resolvedVideoUrlSource": None,
        "downloadableCandidate": False,
        "frontendRecoveryLikely": False,
        "backendDeliveryIncomplete": False,
        "missingVideoUrlPaths": [],
        "redisMutations": 0,
        "openAICalls": 0,
        "runwayCalls": 0,
        "ffmpegCalls": 0,
        "ok": False,
    }

    if not jid:
        report["failureReason"] = "builder2_job_inspect_job_id_missing"
        return report

    if not redis_configured():
        report["failureReason"] = "builder2_job_inspect_redis_unconfigured"
        return report

    job_raw = read_raw_job(jid)
    job_record = read_job(jid)
    report["jobExists"] = bool(job_raw or job_record)

    if isinstance(job_record, dict):
        status = _clean(job_record.get("status")) or None
        report["jobStatus"] = status
        report["jobCompleted"] = status in {"done", "completed"}
        report["videoUrl"] = _clean(job_record.get("videoUrl")) or None
        report["videoUrlPresent"] = bool(report["videoUrl"])
        report["marketingTextPresent"] = bool(_clean(job_record.get("marketingText")))
        report["publicBaseUrl"] = _clean(job_record.get("publicBaseUrl")) or None
        report["errorPresent"] = bool(_clean(job_record.get("error")))

    if isinstance(job_raw, dict):
        if not report["videoUrl"]:
            report["videoUrl"] = _first_present(job_raw, _JOB_VIDEO_URL_KEYS) or None
            report["videoUrlPresent"] = bool(report["videoUrl"])
        if not report["publicBaseUrl"]:
            report["publicBaseUrl"] = _clean(job_raw.get("public_base_url")) or None
        if not report["marketingTextPresent"]:
            report["marketingTextPresent"] = bool(_clean(job_raw.get("marketing_text")))
        if not report["errorPresent"]:
            report["errorPresent"] = bool(_clean(job_raw.get("error")))
        report["createdAt"] = _timestamp_from_raw(job_raw.get("enqueued_ts"))
        if report["jobCompleted"]:
            report["completedAt"] = _timestamp_from_raw(job_raw.get("completed_ts") or job_raw.get("last_progress_ts"))

    tournament_state = load_tournament(jid)
    report["tournamentExists"] = isinstance(tournament_state, dict) and bool(tournament_state)
    media = _media_bucket(tournament_state)

    if isinstance(tournament_state, dict):
        report["tournamentStatus"] = _clean(tournament_state.get("status")) or None
        report["winnerCandidateId"] = _clean(
            tournament_state.get("winnerDevelopmentCandidateId") or tournament_state.get("winnerCandidateId")
        ) or None
        report["winnerPrototypeId"] = _clean(tournament_state.get("winnerDevelopmentPrototypeId")) or None
        report["mediaContinuationRequired"] = bool(tournament_state.get("mediaContinuationRequired"))
        report["ownershipFieldsPresent"] = _ownership_fields_present(job_raw, tournament_state)

    if media:
        report["mediaResumeStatus"] = _clean(media.get("mediaResumeStatus")) or None
        report["progressStage"] = _clean(media.get("progressStage")) or None
        final_public_url = _clean(media.get("finalPublicUrl"))
        report["finalPublicUrl"] = final_public_url or None
        report["finalPublicUrlPresent"] = bool(final_public_url)
        report["finalVideoPathPresent"] = bool(_clean(media.get("finalVideoPath")))
        report["runwayTaskIdPresent"] = bool(_clean(media.get("runwayTaskId")))
        report["runwayVideoUrlPresent"] = bool(_clean(media.get("runwayVideoUrl")))
        report["mediaCompletedAt"] = _clean(media.get("mediaCompletedAt")) or None
        report["mediaFailurePresent"] = isinstance(media.get("mediaFailure"), dict) and bool(media.get("mediaFailure"))

    if not report["ownershipFieldsPresent"]:
        report["ownershipFieldsPresent"] = _ownership_fields_present(job_raw)

    report["finalVideoAvailable"] = bool(
        report["videoUrlPresent"]
        or report["finalPublicUrlPresent"]
        or (report["jobCompleted"] and report["mediaResumeStatus"] == "completed")
    )

    resolved_url, resolved_source, missing_paths = _resolve_video_url(
        job_record=job_record,
        job_raw=job_raw,
        tournament_state=tournament_state,
    )
    report["resolvedVideoUrl"] = resolved_url or None
    report["resolvedVideoUrlSource"] = resolved_source if resolved_url else None
    report["downloadableCandidate"] = bool(resolved_url and _is_public_http_url(resolved_url))
    report["missingVideoUrlPaths"] = missing_paths if not resolved_url else []

    report["frontendRecoveryLikely"] = bool(
        report["jobExists"] and report["jobCompleted"] and bool(resolved_url)
    )
    report["backendDeliveryIncomplete"] = bool(report["jobCompleted"] and not resolved_url)
    report["ok"] = True
    return report


def print_builder2_job_inspect_report(report: Dict[str, Any]) -> None:
    safe = {key: value for key, value in report.items() if key not in _SENSITIVE_OUTPUT_KEYS}
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_JOB_INSPECT_ID", DEFAULT_JOB_INSPECT_ID)
    logger.info("BUILDER2_JOB_INSPECT_START jobId=%s", job_id)
    report = inspect_builder2_completed_job(job_id)
    print_builder2_job_inspect_report(report)
    logger.info(
        "BUILDER2_JOB_INSPECT_DONE jobId=%s ok=%s resolved=%s source=%s",
        job_id,
        report.get("ok"),
        bool(report.get("resolvedVideoUrl")),
        report.get("resolvedVideoUrlSource"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
