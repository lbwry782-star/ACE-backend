"""
Post-restore completeness verification (phase 5.3 — read-only checks, no secrets in output).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_recovery_auto_restore_plan import AutoRestorePlan
from engine.ace_local_dev_recovery_asset_coverage import asset_coverage_to_dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path, *, glob: str = "**/*") -> str:
    if not root.is_dir():
        return ""
    h = hashlib.sha256()
    for path in sorted(root.glob(glob)):
        if path.is_file():
            h.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
            h.update(_sha256_file(path).encode("ascii"))
    return h.hexdigest()


def read_git_snapshot(repo_root: Path) -> Dict[str, str]:
    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        return {"available": "false"}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {
            "available": "true",
            "head": head,
            "branch": branch,
            "porcelain": status,
        }
    except (subprocess.CalledProcessError, OSError):
        return {"available": "false", "error": "git_read_failed"}


def collect_protected_state_fingerprints(ace_root: Path) -> Dict[str, str]:
    ace_root = Path(ace_root)
    out: Dict[str, str] = {}
    registry = ace_root / "_registry"
    if registry.is_dir():
        out["registry_tree"] = _hash_tree(registry)
    sessions = ace_root / "_backups" / "sessions"
    if sessions.is_dir():
        out["sessions_tree"] = _hash_tree(sessions)
    local_dev = ace_root / "_backups" / "local-dev"
    ledger = local_dev / "recovery-ledger.json"
    if ledger.is_file():
        out["recovery_ledger"] = _sha256_file(ledger)
    return out


@dataclass
class RestoreCompletenessReport:
    ok: bool
    git_backend_unchanged: bool = True
    git_frontend_unchanged: bool = True
    protected_unchanged: bool = True
    env_auto_restore_attempted: bool = False
    env_staging_present: bool = False
    dev_files_ready: bool = True
    notes: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "gitBackendUnchanged": self.git_backend_unchanged,
            "gitFrontendUnchanged": self.git_frontend_unchanged,
            "protectedUnchanged": self.protected_unchanged,
            "envAutoRestoreAttempted": self.env_auto_restore_attempted,
            "envStagingPresent": self.env_staging_present,
            "devFilesReady": self.dev_files_ready,
            "notes": list(self.notes),
            "checks": self.checks,
            "assetCoverageRef": "engine/ace_local_dev_recovery_asset_coverage.py",
        }


def assess_env_restore_from_plan(plan: AutoRestorePlan) -> Dict[str, Any]:
    env_entries = [e for e in plan.entries if str(e.area).upper() == "ENV"]
    auto_candidates = [e for e in env_entries if e.action == "RESTORE_FROM_CHECKPOINT"]
    blocked = [e for e in env_entries if e.action in ("PRESERVE_BOTH", "PRESERVE_WORKSPACE_FILE", "BLOCK_UNCERTAIN")]
    return {
        "envEntryCount": len(env_entries),
        "autoRestoreCandidates": len(auto_candidates),
        "blockedCount": len(blocked),
        "autoRestoreAllowed": len(auto_candidates) > 0 and len(blocked) == 0,
        "relPaths": [e.rel_path for e in env_entries],
    }


def verify_restore_completeness(
    *,
    ace_root: Path,
    backend_root: Path,
    frontend_root: Path,
    git_before: Dict[str, Dict[str, str]],
    protected_before: Dict[str, str],
    plan: Optional[AutoRestorePlan] = None,
    engine_result: Optional[Dict[str, Any]] = None,
) -> RestoreCompletenessReport:
    ace_root = Path(ace_root)
    report = RestoreCompletenessReport(ok=True)

    git_be_after = read_git_snapshot(backend_root)
    git_fe_after = read_git_snapshot(frontend_root)
    report.checks["gitBackendAfter"] = {k: v for k, v in git_be_after.items() if k != "porcelain"}
    report.checks["gitFrontendAfter"] = {k: v for k, v in git_fe_after.items() if k != "porcelain"}

    for label, before, after in (
        ("backend", git_before.get("backend", {}), git_be_after),
        ("frontend", git_before.get("frontend", {}), git_fe_after),
    ):
        if before.get("available") != "true" or after.get("available") != "true":
            continue
        if before.get("head") != after.get("head"):
            report.ok = False
            report.notes.append(f"git_head_changed:{label}")
            if label == "backend":
                report.git_backend_unchanged = False
            else:
                report.git_frontend_unchanged = False
        if before.get("branch") != after.get("branch"):
            report.ok = False
            report.notes.append(f"git_branch_changed:{label}")

    protected_after = collect_protected_state_fingerprints(ace_root)
    report.checks["protectedAfter"] = list(protected_after.keys())
    for key, digest in protected_before.items():
        if protected_after.get(key) != digest:
            report.ok = False
            report.protected_unchanged = False
            report.notes.append(f"protected_changed:{key}")

    if plan:
        env_assess = assess_env_restore_from_plan(plan)
        report.checks["envAssessment"] = env_assess
        report.env_auto_restore_attempted = bool(
            engine_result and (engine_result.get("envRestored") or engine_result.get("envFailed"))
        )
        if engine_result and engine_result.get("envRestored"):
            report.env_staging_present = True

    if engine_result and engine_result.get("skipped") and engine_result.get("skipReason") == "start_allowed_no_recovery":
        report.notes.append("start_allowed_no_restore_expected")
        report.checks["assetCoverage"] = asset_coverage_to_dict()

    if plan and plan.checkpoint_valid:
        pending = [e for e in plan.entries if e.action in ("BLOCK_UNCERTAIN", "PRESERVE_BOTH")]
        if pending and not (engine_result and engine_result.get("conflicts")):
            report.dev_files_ready = len(pending) == 0
        report.checks["planSafeAutoApply"] = plan.safe_auto_apply

    report.ok = (
        report.git_backend_unchanged
        and report.git_frontend_unchanged
        and report.protected_unchanged
        and report.dev_files_ready
    )
    return report


def snapshot_pre_restore_state(
    ace_root: Path,
    backend_root: Path,
    frontend_root: Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    git_before = {
        "backend": read_git_snapshot(backend_root),
        "frontend": read_git_snapshot(frontend_root),
    }
    protected = collect_protected_state_fingerprints(ace_root)
    return git_before, protected
