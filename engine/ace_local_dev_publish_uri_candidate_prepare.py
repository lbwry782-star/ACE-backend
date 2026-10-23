"""Publish Uri Candidate prepare — simulation only; explicit execution plan, no side effects."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_cancel_progress import new_operation_id, resolve_operation_id
from engine.ace_local_dev_cycle_store import resolve_active_dev_session_id_for_api
from engine.ace_local_dev_execute_guard import (
    LocalDevExecuteBusyError,
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_publish_candidate_registry import (
    evaluate_registered_candidate,
    load_production_backend_baseline_sha,
    load_publish_candidate_registry,
    normalize_full_sha,
)
from engine.ace_local_dev_publish_prepare_progress import (
    STEP_EXECUTION_PLAN,
    STEP_FINGERPRINT,
    STEP_FRESH_PREVIEW,
    STEP_PRODUCT_SCOPE,
    STEP_REGISTERED_CANDIDATE,
    STEP_SIMULATE_GITHUB_PAGES,
    STEP_SIMULATE_GIT_BACKEND,
    STEP_SIMULATE_GIT_FRONTEND,
    STEP_SIMULATE_REGISTRY,
    STEP_SIMULATE_RENDER,
    STEP_WORKER_COUPLING,
    STATUS_SIMULATED,
    PublishPrepareProgressWriter,
    assert_publish_prepare_operation_id_available,
)
from engine.ace_local_dev_publish_uri_candidate_preview import (
    build_public_publish_uri_candidate_preview_response,
    run_publish_uri_candidate_preview_powershell,
)
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)

logger = logging.getLogger(__name__)

# Known mixed/local-infra commit — must never appear in allowed product publish set.
FORBIDDEN_MIXED_COMMIT_SHA = "e41f41834392858f7b5ab20c8acfc5ee162b7957"

ALLOWED_PRODUCT_CLASSIFICATION = "productCandidate"


class PublishCandidatePrepareRejected(Exception):
    def __init__(
        self,
        code: str,
        *,
        message: Optional[str] = None,
        operation_id: Optional[str] = None,
        blockers: Optional[List[str]] = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code
        self.operation_id = operation_id
        self.blockers = blockers or []


def _validate_fingerprint_token(value: str) -> str:
    fp = (value or "").strip().lower()
    if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
        raise PublishCandidatePrepareRejected("ace_publish_uri_candidate_fingerprint_invalid")
    return fp


def _commit_rows(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


def _collect_commit_shas(plan: Dict[str, Any]) -> List[str]:
    shas: List[str] = []
    for token in plan.get("candidateCommitShas") or []:
        norm = normalize_full_sha(str(token))
        if norm:
            shas.append(norm)
    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    for side in ("backend", "frontend"):
        block = proposed.get(side) if isinstance(proposed.get(side), dict) else {}
        for row in _commit_rows(block.get("commits")):
            norm = normalize_full_sha(str(row.get("fullSha") or ""))
            if norm:
                shas.append(norm)
    commits = plan.get("commits") if isinstance(plan.get("commits"), dict) else {}
    for bucket in ("productCandidate", "excludedLocal"):
        bucket_obj = commits.get(bucket) if isinstance(commits.get(bucket), dict) else {}
        for side in ("backend", "frontend"):
            for row in _commit_rows(bucket_obj.get(side)):
                norm = normalize_full_sha(str(row.get("fullSha") or ""))
                if norm:
                    shas.append(norm)
    return sorted(set(shas))


def _validate_no_forbidden_commits(plan: Dict[str, Any]) -> List[str]:
    blockers: List[str] = []
    forbidden = FORBIDDEN_MIXED_COMMIT_SHA.lower()
    for sha in _collect_commit_shas(plan):
        if sha == forbidden or sha.startswith("e41f418"):
            blockers.append(f"PREPARE_FORBIDDEN_COMMIT:{sha[:12]}")

    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    be = proposed.get("backend") if isinstance(proposed.get("backend"), dict) else {}
    for row in _commit_rows(be.get("commits")):
        classification = str(row.get("classification") or "")
        if classification != ALLOWED_PRODUCT_CLASSIFICATION:
            short = str(row.get("shortSha") or row.get("fullSha") or "?")[:12]
            blockers.append(f"PREPARE_BACKEND_COMMIT_NOT_PRODUCT:{short}")
        for flag in row.get("securityFlags") or []:
            blockers.append(f"PREPARE_SECURITY_FLAG:{short}:{flag}")
        for path_row in row.get("paths") or []:
            if not isinstance(path_row, dict):
                continue
            kind = str(path_row.get("classification") or "")
            if kind in ("forbiddenLocal", "forbiddenSecret"):
                blockers.append(f"PREPARE_FORBIDDEN_PATH:{short}:{path_row.get('path')}")

    fe = proposed.get("frontend") if isinstance(proposed.get("frontend"), dict) else {}
    for row in _commit_rows(fe.get("commits")):
        classification = str(row.get("classification") or "")
        if classification != ALLOWED_PRODUCT_CLASSIFICATION:
            short = str(row.get("shortSha") or row.get("fullSha") or "?")[:12]
            blockers.append(f"PREPARE_FRONTEND_COMMIT_NOT_PRODUCT:{short}")
        for flag in row.get("securityFlags") or []:
            blockers.append(f"PREPARE_SECURITY_FLAG:{short}:{flag}")

    for mixed in _commit_rows(plan.get("mixedCommits")):
        short = str(mixed.get("shortSha") or mixed.get("fullSha") or "?")
        blockers.append(f"PREPARE_MIXED_COMMIT:{short}")

    return sorted(set(blockers))


def _validate_worker_coupling(plan: Dict[str, Any]) -> List[str]:
    blockers: List[str] = []
    production = plan.get("production") if isinstance(plan.get("production"), dict) else {}
    prod_web = normalize_full_sha(str(production.get("backendWeb") or ""))
    prod_worker = normalize_full_sha(str(production.get("backendWorker") or ""))
    if not prod_web or not prod_worker:
        blockers.append("PREPARE_PRODUCTION_BASELINE_INCOMPLETE")
    elif prod_web != prod_worker:
        blockers.append("PRODUCTION_BACKEND_WORKER_SHA_MISMATCH")

    worker = plan.get("workerImpact") if isinstance(plan.get("workerImpact"), dict) else {}
    if not worker.get("deployWithBackendWeb"):
        blockers.append("PREPARE_WORKER_NOT_COUPLED_TO_WEB")
    candidate = normalize_full_sha(str(plan.get("candidateBackendSha") or ""))
    proposed_worker = normalize_full_sha(str(worker.get("proposedCommitSha") or ""))
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    if readiness.get("hasBackendProductChanges") and candidate and proposed_worker != candidate:
        blockers.append("PREPARE_WORKER_SHA_MISMATCH_CANDIDATE")
    return blockers


def build_explicit_execution_plan(*, plan: Dict[str, Any], raw_preview: Dict[str, Any]) -> Dict[str, Any]:
    """Execution plan from registry-verified preview — never from LOCAL branch HEAD."""
    production = plan.get("production") if isinstance(plan.get("production"), dict) else {}
    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    be = proposed.get("backend") if isinstance(proposed.get("backend"), dict) else {}
    fe = proposed.get("frontend") if isinstance(proposed.get("frontend"), dict) else {}
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    worker = plan.get("workerImpact") if isinstance(plan.get("workerImpact"), dict) else {}

    def _slim_commits(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            if str(row.get("classification") or "") != ALLOWED_PRODUCT_CLASSIFICATION:
                continue
            out.append(
                {
                    "fullSha": str(row.get("fullSha") or ""),
                    "shortSha": str(row.get("shortSha") or ""),
                    "subject": str(row.get("subject") or ""),
                    "classification": ALLOWED_PRODUCT_CLASSIFICATION,
                    "paths": [
                        str(p.get("path"))
                        for p in (row.get("paths") or [])
                        if isinstance(p, dict) and p.get("path")
                    ],
                }
            )
        return out

    backend_product = _slim_commits(_commit_rows(be.get("commits")))
    frontend_product = _slim_commits(_commit_rows(fe.get("commits")))

    candidate = normalize_full_sha(str(plan.get("candidateBackendSha") or "")) or ""
    prod_web = str(production.get("backendWeb") or "")
    prod_worker = str(production.get("backendWorker") or "")
    prod_fe = str(production.get("frontend") or "")

    simulated_stages: List[Dict[str, Any]] = []
    if readiness.get("hasBackendProductChanges"):
        simulated_stages.append(
            {
                "id": "git_push_backend_main",
                "targetBranch": str(be.get("deployTargetBranch") or "main"),
                "fromSha": prod_web,
                "toSha": candidate,
                "commitShas": [c["fullSha"] for c in backend_product if c.get("fullSha")],
                "mode": STATUS_SIMULATED,
                "note": "Not executed — prepare simulation only.",
            }
        )
        simulated_stages.append(
            {
                "id": "render_deploy_web_and_worker",
                "platform": str(worker.get("platform") or "render"),
                "commitSha": candidate,
                "webAndWorkerSameSha": True,
                "mode": STATUS_SIMULATED,
            }
        )
    if readiness.get("hasFrontendProductChanges"):
        simulated_stages.append(
            {
                "id": "git_push_frontend_main",
                "targetBranch": str(fe.get("deployTargetBranch") or "main"),
                "fromSha": prod_fe,
                "toSha": str(fe.get("proposedHeadSha") or ""),
                "commitShas": [c["fullSha"] for c in frontend_product if c.get("fullSha")],
                "mode": STATUS_SIMULATED,
            }
        )
        simulated_stages.append(
            {
                "id": "github_pages_deploy",
                "mode": STATUS_SIMULATED,
            }
        )

    simulated_stages.append(
        {
            "id": "production_registry_update",
            "mode": STATUS_SIMULATED,
            "note": "Would run only after live verification in a future execute phase.",
        }
    )

    return {
        "executionPlanVersion": 1,
        "simulationOnly": True,
        "realExecuteBlocked": True,
        "planSource": "CANDIDATE_PREVIEW_VERIFIED",
        "usesLocalHead": False,
        "verifiedSources": {
            "candidateBackendSha": candidate,
            "productionBackendWeb": prod_web,
            "productionBackendWorker": prod_worker,
            "productionFrontend": prod_fe,
            "backendSourceRef": str(be.get("sourceRef") or "explicit-candidate-sha"),
            "frontendSourceRef": str(fe.get("sourceRef") or ""),
        },
        "allowedProductCommits": {
            "backend": backend_product,
            "frontend": frontend_product,
        },
        "allowedProductFiles": {
            "backend": sorted(set(str(p) for p in (be.get("files") or []))),
            "frontend": sorted(set(str(p) for p in (fe.get("files") or []))),
        },
        "workerCoupling": {
            "required": bool(readiness.get("hasBackendProductChanges")),
            "deployWithBackendWeb": bool(worker.get("deployWithBackendWeb")),
            "targetSha": candidate if readiness.get("hasBackendProductChanges") else prod_web,
            "productionWorkerBaseline": prod_worker,
        },
        "simulatedStages": simulated_stages,
        "publishUriPreviewFingerprint": compute_publish_uri_candidate_preview_fingerprint(raw_preview),
        "forbiddenLocalDevInfraBranchPublish": True,
    }


def _resolve_active_registered_candidate() -> Tuple[str, str, Dict[str, Any]]:
    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        raise PublishCandidatePrepareRejected("ace_active_dev_cycle_invalid")
    session_id = resolution.session_id
    if not session_id or resolution.cycle_status == "NONE":
        raise PublishCandidatePrepareRejected("ace_dev_cycle_none")
    if resolution.cycle_status != "OPEN" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        raise PublishCandidatePrepareRejected("ace_dev_cycle_not_open")

    record = load_publish_candidate_registry()
    if not record:
        raise PublishCandidatePrepareRejected("ace_publish_candidate_not_registered")

    baseline = load_production_backend_baseline_sha() or ""
    effective, reasons = evaluate_registered_candidate(
        record=record,
        active_session_id=session_id,
        cycle_status=resolution.cycle_status,
        current_production_baseline_sha=baseline,
    )
    if effective != "ACTIVE":
        raise PublishCandidatePrepareRejected(
            "ace_publish_candidate_registry_stale",
            blockers=reasons,
        )

    candidate = normalize_full_sha(str(record.get("candidateBackendSha") or ""))
    if not candidate:
        raise PublishCandidatePrepareRejected("ace_publish_uri_candidate_sha_invalid")
    return session_id, candidate, record


def execute_publish_uri_candidate_prepare_simulate(
    *,
    publish_uri_preview_fingerprint: str,
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build explicit publish execution plan and journal simulated stages.
    No git push, deploy, production registry update, or paid API calls.
    """
    expected_fp = _validate_fingerprint_token(publish_uri_preview_fingerprint)

    session_id, candidate_sha, _record = _resolve_active_registered_candidate()

    try:
        op = resolve_operation_id(operation_id)
    except ValueError as exc:
        if str(exc) == "ace_cancel_progress_operation_id_invalid":
            raise PublishCandidatePrepareRejected("ace_publish_prepare_operation_id_invalid") from None
        raise
    try:
        assert_publish_prepare_operation_id_available(op, session_id=session_id)
    except ValueError as exc:
        code = str(exc)
        if code in (
            "ace_publish_prepare_operation_id_invalid",
            "ace_publish_prepare_operation_id_already_used",
        ):
            raise PublishCandidatePrepareRejected(code) from None
        raise

    progress = PublishPrepareProgressWriter(session_id=session_id, operation_id=op)

    try:
        try_begin_local_dev_execute()
    except LocalDevExecuteBusyError as exc:
        progress.fail_step(STEP_REGISTERED_CANDIDATE, error="ace_local_dev_execute_in_progress")
        raise PublishCandidatePrepareRejected(
            "ace_local_dev_execute_in_progress",
            operation_id=op,
        ) from exc

    try:
        progress.begin_step(STEP_REGISTERED_CANDIDATE)
        progress.complete_step(STEP_REGISTERED_CANDIDATE)

        progress.begin_step(STEP_FRESH_PREVIEW)
        raw = run_publish_uri_candidate_preview_powershell(session_id, candidate_sha)
        if not raw.get("ok"):
            progress.fail_step(STEP_FRESH_PREVIEW, error="ace_publish_uri_candidate_preview_failed")
            raise PublishCandidatePrepareRejected(
                "ace_publish_uri_candidate_preview_failed",
                operation_id=op,
            )
        plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else {}
        if normalize_full_sha(str(plan.get("candidateBackendSha") or "")) != candidate_sha:
            progress.fail_step(STEP_FRESH_PREVIEW, error="ace_publish_prepare_candidate_mismatch")
            raise PublishCandidatePrepareRejected(
                "ace_publish_prepare_candidate_mismatch",
                operation_id=op,
            )
        progress.complete_step(STEP_FRESH_PREVIEW)

        progress.begin_step(STEP_FINGERPRINT)
        fresh_fp = compute_publish_uri_candidate_preview_fingerprint(raw)
        if fresh_fp.lower() != expected_fp:
            progress.fail_step(STEP_FINGERPRINT, error="ace_publish_uri_candidate_fingerprint_stale")
            raise PublishCandidatePrepareRejected(
                "ace_publish_uri_candidate_fingerprint_stale",
                operation_id=op,
            )
        progress.complete_step(STEP_FINGERPRINT)

        preview = build_public_publish_uri_candidate_preview_response(raw)
        blockers = list(preview.get("candidateBlockers") or preview.get("blockers") or [])
        readiness = preview.get("readiness") if isinstance(preview.get("readiness"), dict) else {}

        progress.begin_step(STEP_PRODUCT_SCOPE)
        extra_blockers = _validate_no_forbidden_commits(plan)
        if extra_blockers:
            blockers = sorted(set(blockers + extra_blockers))
        if blockers:
            progress.fail_step(STEP_PRODUCT_SCOPE, error=blockers[0])
            raise PublishCandidatePrepareRejected(
                "ace_publish_prepare_blocked",
                operation_id=op,
                blockers=blockers,
            )
        if not readiness.get("candidateVerifiedSafeForPublish"):
            progress.fail_step(STEP_PRODUCT_SCOPE, error="ace_publish_candidate_not_verified_safe")
            raise PublishCandidatePrepareRejected(
                "ace_publish_candidate_not_verified_safe",
                operation_id=op,
            )
        if not readiness.get("canPrepareLocalPublish"):
            drift = list(preview.get("registryInfraDrift") or []) + list(
                preview.get("registryWipDrift") or []
            )
            progress.fail_step(STEP_PRODUCT_SCOPE, error="ace_publish_prepare_not_ready")
            raise PublishCandidatePrepareRejected(
                "ace_publish_prepare_not_ready",
                operation_id=op,
                blockers=drift or ["canPrepareLocalPublish=false"],
            )
        progress.complete_step(STEP_PRODUCT_SCOPE)

        progress.begin_step(STEP_WORKER_COUPLING)
        worker_blockers = _validate_worker_coupling(plan)
        if worker_blockers:
            progress.fail_step(STEP_WORKER_COUPLING, error=worker_blockers[0])
            raise PublishCandidatePrepareRejected(
                "ace_publish_prepare_worker_blocked",
                operation_id=op,
                blockers=worker_blockers,
            )
        progress.complete_step(STEP_WORKER_COUPLING)

        progress.begin_step(STEP_EXECUTION_PLAN)
        execution_plan = build_explicit_execution_plan(plan=plan, raw_preview=raw)
        progress.set_execution_plan(execution_plan)
        progress.complete_step(STEP_EXECUTION_PLAN)

        for step_id, needed in (
            (STEP_SIMULATE_GIT_BACKEND, readiness.get("hasBackendProductChanges")),
            (STEP_SIMULATE_RENDER, readiness.get("hasBackendProductChanges")),
            (STEP_SIMULATE_GIT_FRONTEND, readiness.get("hasFrontendProductChanges")),
            (STEP_SIMULATE_GITHUB_PAGES, readiness.get("hasFrontendProductChanges")),
        ):
            if not needed:
                continue
            progress.begin_step(step_id)
            progress.complete_step(step_id, status=STATUS_SIMULATED)

        progress.begin_step(STEP_SIMULATE_REGISTRY)
        progress.complete_step(STEP_SIMULATE_REGISTRY, status=STATUS_SIMULATED)

        progress.finish_success()

        return {
            "ok": True,
            "prepareMode": "SIMULATION_ONLY",
            "simulationOnly": True,
            "realExecuteBlocked": True,
            "executePublishAvailable": False,
            "operationId": op,
            "sessionId": session_id,
            "candidateBackendSha": candidate_sha,
            "publishUriPreviewFingerprint": fresh_fp,
            "executionPlan": execution_plan,
            "progressPath": str(progress.path),
            "preview": preview,
            "note": "Prepare simulation only; no push, deploy, or production registry update.",
        }
    except PublishCandidatePrepareRejected:
        raise
    except Exception as exc:
        logger.exception(
            "LOCAL_DEV_PUBLISH_PREPARE_SIMULATE_FAILED sessionId=%s operationId=%s",
            session_id,
            op,
        )
        progress.fail_operation(error=str(exc))
        raise PublishCandidatePrepareRejected(
            "ace_publish_prepare_simulate_failed",
            operation_id=op,
            message=str(exc)[:240],
        ) from exc
    finally:
        end_local_dev_execute()


def new_publish_prepare_operation_id() -> str:
    return new_operation_id()
