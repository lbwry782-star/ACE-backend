"""Publish Uri Candidate Dry-run — re-preview + fingerprint match; no Git or deploy."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.ace_local_dev_execute_guard import (
    LocalDevExecuteBusyError,
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_publish_uri_candidate_preview import (
    build_public_publish_uri_candidate_preview_response,
    normalize_candidate_backend_sha,
    run_publish_uri_candidate_preview_powershell,
)
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)

logger = logging.getLogger(__name__)


class PublishCandidateDryRunRejected(Exception):
    def __init__(self, code: str, *, message: Optional[str] = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


def _validate_fingerprint_token(value: str) -> str:
    fp = (value or "").strip().lower()
    if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
        raise PublishCandidateDryRunRejected("ace_publish_uri_candidate_fingerprint_invalid")
    return fp


def execute_publish_uri_candidate_dry_run(
    *,
    session_id: str,
    candidate_backend_sha: str,
    publish_uri_preview_fingerprint: str,
) -> Dict[str, Any]:
    candidate = normalize_candidate_backend_sha(candidate_backend_sha)
    if not candidate:
        raise PublishCandidateDryRunRejected("ace_publish_uri_candidate_sha_invalid")

    expected_fp = _validate_fingerprint_token(publish_uri_preview_fingerprint)

    try:
        try_begin_local_dev_execute()
    except LocalDevExecuteBusyError as exc:
        raise PublishCandidateDryRunRejected("ace_local_dev_execute_in_progress") from exc

    try:
        raw = run_publish_uri_candidate_preview_powershell(session_id, candidate)
        if not raw.get("ok"):
            raise PublishCandidateDryRunRejected(
                "ace_publish_uri_candidate_preview_failed",
                message=str(raw.get("error") or "unknown"),
            )

        fresh_fp = compute_publish_uri_candidate_preview_fingerprint(raw)
        if fresh_fp.lower() != expected_fp:
            raise PublishCandidateDryRunRejected("ace_publish_uri_candidate_fingerprint_stale")

        preview = build_public_publish_uri_candidate_preview_response(raw)
        return {
            "ok": True,
            "dryRun": True,
            "dryRunMode": "CANDIDATE_PUBLISH_NO_SIDE_EFFECTS",
            "fingerprintMatched": True,
            "publishUriPreviewFingerprint": fresh_fp,
            "candidateBackendSha": candidate,
            "sessionId": session_id,
            "preview": preview,
            "note": "Dry-run only; no push, deploy, registry, or git mutations.",
        }
    finally:
        end_local_dev_execute()
